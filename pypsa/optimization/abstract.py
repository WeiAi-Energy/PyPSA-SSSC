#!/usr/bin/env python3
"""
Build abstracted, extended optimisation problems from PyPSA networks with
Linopy.
"""

from __future__ import annotations

import copy
import gc
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING, Any
import numpy as np
import pandas as pd
import xarray as xr
from linopy import LinearExpression, QuadraticExpression, merge

from pypsa.descriptors import nominal_attrs
from pypsa.optimization.constraints import (
    capacity_reference,
    kirchhoff_voltage_cycles,
)
from pypsa.utils import as_index

if TYPE_CHECKING:
    from pypsa import Network
logger = logging.getLogger(__name__)

# Weight of the proximal term. It is fixed for the whole run: the weight is a
# property of the problem, not a state of the iteration, and the schedule that
# used to adapt it never fired on the cases it was meant for - its triggers
# needed a KVL residual outside a pair of tolerances, while a run spends its
# iterations between them. Only the trust region radius is adapted now, and off
# the angle between consecutive steps rather than that residual.
#
# The weight is the share of the capital cost of a branch the term charges for
# moving it by its own size, so it is dimensionless.
#
# The term has a dead zone: at the anchor its subgradient is ``delta * c_l``
# times ``[-1, 1]``, so no move worth less than that per MW is taken at all,
# and the converged point is only optimal to within it. That is what makes it
# selective - a reallocation between two plans of equal cost gains nothing and
# is refused, while a move worth more than the hurdle is taken at full size -
# and it is also how it goes wrong. Measured on the meshed brownfield system
# over five decades of the weight, the bias of the converged cost stays below
# 1e-4 relative up to ``1e-1`` and then jumps to between two and eleven per cent
# at ``1``, while the iteration count *falls*: the term freezes the iterate and
# the convergence test reads the freeze as convergence. ``1e-1`` is therefore
# the last decade in which the term perturbs the objective rather than deciding
# the plan. The term cannot be relied on as the only step control: the weight it
# would need to control a large step is past the point where it distorts the
# answer.
PROXIMAL_WEIGHT = 1e-2

@dataclass
class TransmissionIterationResult:
    """Minimal solution data required by one transmission-expansion step."""

    capacities: pd.Series
    flows: dict[str, pd.DataFrame]
    q_sssc: pd.DataFrame | None
    objective: float
    tracked_capacities: dict[str, pd.Series] | None = None
    sssc_nom: pd.Series | None = None
    link_caps: pd.Series | None = None


def optimize_transmission_expansion_iteratively(
    n: Network,
    snapshots: Sequence | None = None,
    msq_threshold: float | None = None,
    min_iterations: int = 1,
    max_iterations: int = 100,
    track_iterations: bool = False,
    scheme: str = "slp",
    proximal_target: str = "branches",
    cost_threshold: float = 1e-5,
    cost_window: int = 1,
    proximal_weight: float | None = None,
    trust_region_initial: float = 1.0,
    trust_region_bounds: tuple[float, float] = (1e-2, 1.0),
    trust_region_alignment: tuple[float, float] = (-0.5, 0.0),
    trust_region_factors: tuple[float, float] = (0.3, 3.0),
    sensitivity_tolerance: float = 1e-6,
    **kwargs: Any,
) -> tuple[str, str]:
    """
    Iterative linear optimization updating the line parameters for passive AC
    and DC lines. This is helpful when line expansion is enabled. After each
    successful solving, line impedances and line resistance are recalculated
    based on the optimization result. If warmstart is possible, it uses the
    result from the previous iteration to fasten the optimization.

    Expanding an AC line lowers its impedance, which enters the Kirchhoff
    voltage law (KVL) and - on ``LineX`` branches - the series compensation
    term of the SSSC. Both couplings are nonlinear in the branch capacity and
    are resolved by an outer iteration, whose scheme and step control are set
    by ``scheme`` and the step controls below.

    Parameters
    ----------
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        network.snapshots, defaults to network.snapshots
    msq_threshold: float, optional
        Deprecated and ignored. The relative change of the transmission
        capacities is a step size, not a measure of convergence: on meshed
        networks degenerate alternative optima keep it bouncing long after the
        solution has settled, while a step can be small by accident. It is
        still reported as ``step`` in ``n.iteration_log`` as a diagnostic, but
        convergence is decided by ``cost_threshold``.
    min_iterations : integer, default 1
        Minimal number of iterations to run regardless whether the convergence
        criterion is already met
    max_iterations : integer, default 100
        Maximal number of iterations to run regardless whether the convergence
        criterion is already met
    track_iterations: bool, default False
        If True, the intermediate branch capacities and values of the
        objective function are recorded for each iteration. The values of
        iteration 0 represent the initial state.
        Intermediate solves retain only the transmission
        capacities, KVL branch flows and SSSC compensation required by the
        outer loop. If tracking is enabled, the requested nominal-capacity
        columns and objective scalars are retained as well. Complete primal
        results, duals and derived network time series always come from the
        report solve that closes the run at the capacities of the last
        iterate, whether the loop converged or exhausted ``max_iterations``.
    scheme : {'slp', 'fixed_point'}, default 'slp'
        How the capacity dependence of the voltage law is resolved.

        ``'fixed_point'``
            The impedance and the SSSC compensation term are frozen at the
            capacity of the previous iterate, so the KVL constraint stays
            zeroth-order in the capacity and every inner problem is the plain
            linear expansion problem.
        ``'slp'``
            Sequential linear programming: the KVL constraint carries the
            first-order sensitivity of its branch terms with respect to the
            branch capacity, evaluated at the previous iterate, so the inner
            problem accounts for the change of the voltage law its own step
            causes. The first iteration is a fixed-point step, since no
            linearisation point exists yet. Once converged, the linearisation
            is exact at that point, so the report solve that closes the run
            reproduces the converged iterate rather than moving away from it.

        A linearisation is only valid over a limited step, which is what
        the step controls are for. Every solved iterate is
        accepted however large the residual it leaves; the controls only narrow
        the *next* step. With both off, ``'slp'`` is a bare Gauss-Newton
        iteration, and on a meshed network it oscillates instead of
        converging.
    cost_threshold : float, default 1e-5
        Convergence criterion: the iteration stops once the system cost has
        changed by less than this fraction of itself in each of the last
        ``cost_window`` accepted iterations. The cost is invariant under the
        exchange of degenerate alternative optima, which the change of the
        capacities is not, and it is the quantity the results are reported in.
    cost_window : int, default 1
        Number of consecutive relative cost changes that have to undercut
        ``cost_threshold``.
    proximal_weight : float, optional
        Weight ``delta`` of the proximal term, fixed for the whole run, default
        ``PROXIMAL_WEIGHT``. The term penalises moving the capacity of branch
        ``l`` away from the previous iterate ``F_l'`` in units of its capital
        cost ``c_l``,

        ``delta * sum_l c_l |F_l - F_l'|``,

        modelled with a non-negative deviation variable per branch, so the
        inner problem stays a linear program. ``delta`` is the share of the
        capital cost of a branch the term charges for moving it by its own
        size, hence dimensionless.

        It doubles as a hurdle rate on capital reallocation: the subgradient at
        the anchor is ``delta * c_l`` times ``[-1, 1]``, so a branch is moved
        only where the move returns more than ``delta`` of the capital it
        shifts. That dead zone is what makes the term selective - a reallocation
        between two plans of equal cost gains nothing and is refused, while a
        move worth more than the hurdle is taken at full size - and it is also
        how the term can go wrong: past about ``1e-1`` on the systems this was
        measured on, the dead zone decides the plan rather than damping the
        iteration, and a frozen iterate reports itself as a converged one.

        ``0`` switches the term off.
    proximal_target : {'branches', 'sssc', 'both'}, default 'branches'
        What the term holds: the branch capacities, the series compensation
        ``sssc_nom``, or both under the one weight.

        ``'sssc'`` exists because the compensation is the one degree of freedom
        the iteration otherwise leaves uncontrolled - the trust region and the
        term itself both act on the capacities - and because on a network that
        prices compensation per MVAr at one rate and caps only its total, moving
        it from one branch to another is free to the objective. The allocation
        is then decided entirely by a flow pattern that the previous allocation
        moved, and it answers bang-bang: measured, one set of branches carries
        its full rating on odd iterates and nothing on even ones, trading with
        an antiphase set, and it reverses direction before the capacities do.

        A dead zone is the right instrument for that, and the objection that
        makes a uniform weight wrong for the capacities - it charges a cheap
        branch a larger share of its own cost than an expensive one - does not
        apply here, since the underlying rate really is uniform.

        ``'both'`` shares one weight between two terms whose scales need not
        agree, so it is only useful where they happen to.
    trust_region_initial : float, default 1.0
        Initial trust region radius. The
        capacity of branch ``l`` is restricted to

        ``F_def_l - radius / (1 + radius) * S_l <= F_l <= F_def_l + radius * S_l``

        with ``S_l = max(F_def_l, F0_l)``, i.e. the radius is relative to the
        capacity of the linearisation point, floored by the initial capacity.
        The region is wider above the linearisation point than below it, which
        for ``S_l = F_def_l`` makes it the geometrically symmetric interval
        ``F_def_l / (1 + radius) <= F_l <= F_def_l * (1 + radius)``. That is the
        natural symmetry of the branch terms of the voltage law, which scale
        with the inverse capacity, and it leaves the same linearisation error at
        both ends; a symmetric region would tolerate a much larger error on the
        side where the capacity shrinks.
    trust_region_bounds : tuple of float, default (1e-2, 1.0)
        Smallest and largest admissible trust region radius. The radius
        saturates at the lower bound; the iteration then continues at that
        radius until it converges or ``max_iterations`` runs out.
    trust_region_alignment : tuple of float, default (-0.5, 0.0)
        Thresholds ``(tighten, relax)`` on the angle between the fixed-point
        residuals ``G(z) - z`` of two consecutive steps, weighted with the cost
        of the capacity they move. At a cosine of ``tighten`` or below the step
        undid the one before it and the radius is shrunk; at ``relax`` or above
        the iteration is walking one way and the radius is expanded if it was
        binding. In between the radius is left alone. The step itself is kept
        either way - a solved iterate is never discarded.

        The angle is what the radius has to be steered by here. An expanding
        branch lowers its own impedance and attracts flow, which the next
        model answers by shrinking it again, and where that feedback is
        reflecting rather than contracting the iteration settles into an orbit
        of period two. Neither of the quantities the iteration used to be
        steered by sees it: the residual of the voltage law is small on an
        orbit, and the size of the step is flat rather than growing. The
        radius is nonetheless the control that resolves it. The region is
        geometrically symmetric, so a branch that jumps to one wall finds the
        other wall exactly back at its previous capacity - the orbit lives on
        the walls, reproduces itself for as long as the radius is held, and
        shrinks in proportion to it.
    trust_region_factors : tuple of float, default (0.5, 2.0)
        Factors ``(shrink, expand)`` the trust region radius is multiplied
        with. The weight of the proximal
        term is not adapted and these do not apply to it.
    sensitivity_tolerance : float, default 1e-6
        Relative size below which the linearisation of ``scheme='slp'`` drops
        a branch sensitivity, measured against the voltage drop that
        branch causes at its rated capacity. The sensitivity is proportional to
        the flow the branch carries, so a branch that is idle at a snapshot
        contributes a coefficient many orders of magnitude below the rest of its
        constraint - too small to move the solution, large enough to widen the
        range of the constraint matrix and to stall the solver. Set to ``0`` to
        keep every sensitivity.
    **kwargs
        Keyword arguments of the `n.optimize` function which runs at each iteration

    Returns
    -------
    status, condition : str, str
        Status and termination condition of the final optimization. The
        convergence history is written to ``n.iteration_log``.
    """
    schemes = ("slp", "fixed_point")
    if scheme not in schemes:
        raise ValueError(f"scheme must be one of {schemes}, got {scheme!r}.")
    if msq_threshold is not None:
        logger.warning(
            "'msq_threshold' is ignored, convergence is decided by the relative "
            "change of the system cost ('cost_threshold'). The relative change "
            "of the capacities remains available as 'step' in n.iteration_log."
        )
    cost_threshold = float(cost_threshold)
    cost_window = int(cost_window)
    # the weight doubles as the switch: zero leaves the term out of the model
    proximal_on = proximal_weight is None or float(proximal_weight) > 0.0
    proximal_targets = ("branches", "sssc", "both")
    if proximal_target not in proximal_targets:
        raise ValueError(
            f"proximal_target must be one of {proximal_targets}, "
            f"got {proximal_target!r}."
        )
    if cost_threshold < 0.0:
        raise ValueError("cost_threshold must be >= 0.")
    if cost_window < 1:
        raise ValueError("cost_window must be >= 1.")
    if proximal_weight is None:
        proximal_weight = PROXIMAL_WEIGHT
    proximal_weight = float(proximal_weight)
    if proximal_weight < 0.0:
        raise ValueError("proximal_weight must be >= 0; zero switches the term off.")
    sensitivity_tolerance = float(sensitivity_tolerance)
    if not 0.0 <= sensitivity_tolerance < 1.0:
        raise ValueError("sensitivity_tolerance must be in [0, 1).")

    if snapshots is None:
        snapshots = n.snapshots
    snapshots = as_index(n, snapshots, "snapshots", "snapshot")

    def solve_once(
        solve_snapshots: Sequence,
        lightweight: bool = False,
        **extra_kwargs: Any,
    ) -> tuple[str, str]:
        """
        One solve of the network with the settings of this run.

        ``lightweight`` reads back only the variable groups the outer iteration
        needs and skips assigning the user-facing solution, which is what an
        iterate wants; the solve that reports a run needs the full assignment.
        """
        solve_kwargs = {**kwargs, **extra_kwargs}
        if lightweight:
            return n.optimize(
                solve_snapshots,
                _solution_variables=iteration_solution_variables(),
                _assign_solution=False,
                _assign_duals=False,
                _post_processing=False,
                **solve_kwargs,
            )
        return n.optimize(solve_snapshots, **solve_kwargs)

    branch_components = [
        c for c in ("Line", "LineX") if c in n.components and not n.df(c).empty
    ]
    # DC links carry no voltage law, so the iteration neither linearises nor
    # boxes them - but they are part of the plan, and the solve that closes a
    # run has to hold them as well as the AC capacities, see
    # ``pin_capacities``
    link_ext_i = pd.Index([], name="Link")
    if "Link" in n.components and not n.links.empty:
        extendable = n.links.get("p_nom_extendable")
        if extendable is not None:
            link_ext_i = n.links.index[extendable.fillna(False).astype(bool)]
    if not branch_components:
        return solve_once(snapshots)

    # the proximal term is injected through the extra functionality hook, so a
    # callback of the caller has to be chained rather than replaced
    user_extra_functionality = kwargs.pop("extra_functionality", None)

    for c in branch_components:
        n.df(c)["carrier"] = n.df(c).bus0.map(n.buses.carrier)

    branch_data: dict[str, dict[str, pd.Series | pd.Index]] = {}
    for c in branch_components:
        df = n.df(c)
        ext_i = n.get_extendable_i(c).copy()
        typed_i = df.query('type != ""').index
        ext_untyped_i = ext_i.difference(typed_i)
        ext_typed_i = ext_i.intersection(typed_i)
        base_s_nom = np.sqrt(3) * df["type"].map(n.line_types.i_nom) * df.bus0.map(
            n.buses.v_nom
        )
        if not ext_typed_i.empty:
            df.loc[ext_typed_i, "num_parallel"] = (df.s_nom / base_s_nom)[ext_typed_i]
        # branches whose impedance is rescaled with the capacity below, i.e. the
        # branches the KVL constraint depends on nonlinearly
        scale_untyped_i = df.query("carrier == 'AC'").index.intersection(ext_untyped_i)
        branch_data[c] = {
            "ext_i": ext_i,
            "typed_i": typed_i,
            "ext_untyped_i": ext_untyped_i,
            "ext_typed_i": ext_typed_i,
            "scale_untyped_i": scale_untyped_i,
            "scaling_i": scale_untyped_i.union(ext_typed_i),
            "base_s_nom": base_s_nom,
            "x_0": df.x.copy(),
            "r_0": df.r.copy(),
        }

    # passive branches contributing to the voltage law, including the
    # non-extendable ones needed to evaluate the KVL residual
    passive_components = [
        c for c in n.passive_branch_components if not n.df(c).empty
    ]
    tracked_branch_components = list(
        dict.fromkeys([*n.branch_components, *branch_components])
    )

    def collect_branch_caps(attr: str) -> pd.Series:
        return pd.concat(
            {c: n.df(c)[attr] for c in branch_components},
            names=["component", "name"],
        )

    def iteration_solution_variables() -> list[str]:
        """Variable groups needed to advance one outer iteration."""
        names = [
            f"{c}-{nominal_attrs[c]}"
            for c in branch_components
            if not branch_data[c]["ext_i"].empty
        ]
        names.extend(f"{c}-s" for c in passive_components)
        if "LineX" in branch_components:
            names.append("LineX-q_sssc")
        if track_iterations:
            names.extend(
                f"{c}-{nominal_attrs[c]}" for c in tracked_branch_components
            )
        # the SSSC rating is the anchor of its own proximal term, so it has to
        # be read back for the next iteration even where nothing is tracked
        if (track_iterations or sssc_on) and "LineX" in branch_components:
            names.append("LineX-sssc_nom")
        if not link_ext_i.empty:
            names.append("Link-p_nom")
        return list(dict.fromkeys(names))

    def collect_iteration_result(network: Network) -> TransmissionIterationResult:
        """Read the minimal outer-iteration state from selected model groups."""
        m = network.model
        capacities: dict[str, pd.Series] = {}
        for c in branch_components:
            attr = nominal_attrs[c]
            values = network.df(c)[attr].astype(float).copy()
            name = f"{c}-{attr}"
            if name in m.variables:
                solution = m[name].solution.to_pandas()
                values.loc[solution.index] = solution
            capacities[c] = values

        flows = {
            c: m[f"{c}-s"].solution.to_pandas() for c in passive_components
        }
        q_sssc = (
            m["LineX-q_sssc"].solution.to_pandas()
            if "LineX-q_sssc" in m.variables
            else None
        )
        tracked_capacities = None
        if track_iterations:
            tracked_capacities = {}
            for c in tracked_branch_components:
                attr = nominal_attrs[c]
                values = network.df(c)[attr].astype(float).copy()
                name = f"{c}-{attr}"
                if name in m.variables:
                    solution = m[name].solution.to_pandas()
                    values.loc[solution.index] = solution
                tracked_capacities[c] = values

        sssc_nom = (
            m["LineX-sssc_nom"].solution.to_pandas()
            if (track_iterations or sssc_on) and "LineX-sssc_nom" in m.variables
            else None
        )
        link_caps = (
            m["Link-p_nom"].solution.to_pandas()
            if not link_ext_i.empty and "Link-p_nom" in m.variables
            else None
        )
        objective = float(m.objective.value)
        if getattr(network, "_objective_constant_missing_from_expression", False):
            objective -= float(getattr(network, "objective_constant", 0.0))
        return TransmissionIterationResult(
            capacities=pd.concat(capacities, names=["component", "name"]),
            flows=flows,
            q_sssc=q_sssc,
            objective=objective,
            tracked_capacities=tracked_capacities,
            sssc_nom=sssc_nom,
            link_caps=link_caps,
        )

    def as_branch_index(key: str) -> pd.MultiIndex:
        pairs = [(c, i) for c in branch_components for i in branch_data[c][key]]
        if not pairs:
            return pd.MultiIndex.from_arrays(
                [[], []], names=["component", "name"]
            )
        return pd.MultiIndex.from_tuples(pairs, names=["component", "name"])

    ext_branches = as_branch_index("ext_i")
    scaling_branches = as_branch_index("scaling_i")

    # capacities the iteration starts from; they define both the floor of the
    # impedance-defining capacity and the scale of the trust region
    initial_caps = collect_branch_caps("s_nom").astype(float)
    capacity_floor = initial_caps.abs() * 1e-6
    radius_scale = initial_caps.abs()
    positive_scale = radius_scale[radius_scale > 0]
    radius_scale = radius_scale.mask(
        radius_scale <= 0,
        float(positive_scale.mean()) if not positive_scale.empty else 1.0,
    )
    def update_line_params(network: Network, s_nom_define: pd.Series) -> None:
        """
        Update branch impedance-defining parameters from the current F_def vector.
        """
        for c in branch_components:
            df = network.df(c)
            data = branch_data[c]
            target = (
                s_nom_define.xs(c, level="component")
                .reindex(df.index)
                .fillna(df["s_nom"])
            )
            # the impedance scales with the inverse capacity, so a vanishing
            # linearisation capacity has to be kept away from zero
            floor = capacity_floor.xs(c, level="component").reindex(df.index)
            if (target < floor).any():
                logger.warning(
                    "Capacities of %s %s fell below the impedance floor and are "
                    "raised to it for the parameter update.",
                    c,
                    list(df.index[target < floor]),
                )
            target = target.clip(lower=floor)
            df["_s_nom_def"] = target
            factor = target / df["s_nom"].replace(0, np.nan)

            ac_i = data["scale_untyped_i"]
            if not ac_i.empty:
                df.loc[ac_i, "x"] = data["x_0"][ac_i] / factor[ac_i]
                df.loc[ac_i, "r"] = data["r_0"][ac_i] / factor[ac_i]

            typed_i = data["ext_typed_i"]
            if not typed_i.empty:
                df.loc[typed_i, "num_parallel"] = target[typed_i] / data["base_s_nom"][typed_i]

    def collect_branch_bounds(network: Network) -> tuple[pd.Series, pd.Series]:
        lower = {}
        upper = {}
        for c in branch_components:
            df = network.df(c)
            lower_bounds = pd.Series(0.0, index=df.index, dtype=float)
            upper_bounds = pd.Series(np.inf, index=df.index, dtype=float)
            ext_i = branch_data[c]["ext_i"]
            if f"{nominal_attrs[c]}_min" in df:
                lower_bounds.loc[ext_i] = (
                    df[f"{nominal_attrs[c]}_min"].reindex(ext_i).fillna(0.0)
                )
            if f"{nominal_attrs[c]}_max" in df:
                upper_bounds.loc[ext_i] = (
                    df[f"{nominal_attrs[c]}_max"].reindex(ext_i).fillna(np.inf)
                )
            fixed_i = df.index.difference(ext_i)
            if not fixed_i.empty:
                nominal = df[nominal_attrs[c]].reindex(fixed_i)
                lower_bounds.loc[fixed_i] = nominal
                upper_bounds.loc[fixed_i] = nominal
            lower[c] = lower_bounds
            upper[c] = upper_bounds
        return (
            pd.concat(lower, names=["component", "name"]),
            pd.concat(upper, names=["component", "name"]),
        )

    def violates_bounds(caps: pd.Series, lower: pd.Series) -> bool:
        """
        Detect a solve that reports success but returns capacities below the
        lower bounds of its own model. Solvers do report an optimal status for a
        numerically failed solve, in which case the interface reads back a
        trivial solution which would otherwise silently be clipped and used as
        the next iterate.
        """
        if ext_branches.empty:
            return False
        bound = lower.reindex(caps.index)
        tolerance = 1e-6 * bound.abs().clip(lower=1.0)
        return bool((caps < bound - tolerance).any())

    def clip_branch_caps(
        caps: pd.Series, lower: pd.Series, upper: pd.Series
    ) -> pd.Series:
        clipped = caps.copy()
        lower_aligned = lower.reindex(clipped.index).fillna(-np.inf)
        upper_aligned = upper.reindex(clipped.index).fillna(np.inf)
        return clipped.clip(lower=lower_aligned, upper=upper_aligned)

    def save_optimal_capacities(
        network: Network,
        iteration: int,
        status: str,
        capacities: dict[str, pd.Series] | None = None,
        sssc_nom: pd.Series | None = None,
        objective: float | None = None,
    ) -> None:
        if capacities is None:
            for c, attr in pd.Series(nominal_attrs)[
                list(network.branch_components)
            ].items():
                network.df(c)[f"{attr}_opt_{iteration}"] = network.df(c)[
                    f"{attr}_opt"
                ]
        else:
            for c, values in capacities.items():
                attr = nominal_attrs[c]
                network.df(c)[f"{attr}_opt_{iteration}"] = values.reindex(
                    network.df(c).index
                )
        if "LineX" in network.components and not network.line_xs.empty:
            values = (
                sssc_nom
                if sssc_nom is not None
                else network.line_xs["sssc_nom"]
            )
            network.line_xs[f"sssc_nom_opt_{iteration}"] = values.reindex(
                network.line_xs.index
            )
        setattr(network, f"status_{iteration}", status)
        setattr(
            network,
            f"objective_{iteration}",
            network.objective if objective is None else objective,
        )
        network.iteration = iteration
        if "mu" in network.global_constraints:
            network.global_constraints = network.global_constraints.rename(
                columns={"mu": f"mu_{iteration}"}
            )

    def relative_capacity_change(
        current: pd.Series, previous: pd.Series, initial: pd.Series
    ) -> float:
        if ext_branches.empty:
            return 0.0
        denom = np.linalg.norm(initial.loc[ext_branches].to_numpy())
        denom = max(denom, 1e-12)
        return float(
            np.linalg.norm(
                (current - previous).loc[ext_branches].to_numpy()
            )
            / denom
        )

    # weights of the proximal term: moving a branch capacity is penalised in
    # units of the capital cost of that branch
    proximal_costs: dict[str, pd.Series] = {}
    for c in branch_components:
        ext_i = branch_data[c]["ext_i"]
        weights = n.df(c)["capital_cost"].reindex(ext_i).astype(float)
        weights = weights.where(np.isfinite(weights) & (weights > 0.0))
        # a branch whose capital cost is missing or non-positive is charged the
        # mean of the ones that have it, so that it is damped like the rest
        # rather than being free to absorb the whole degeneracy
        fallback = float(weights.mean()) if weights.notna().any() else 1.0
        proximal_costs[c] = weights.fillna(fallback)

    # ---------------------------------------------------------------- SSSC
    # The same term on the series compensation. It needs no scale of its own:
    # the term charges ``delta * c_l`` per MVAr moved and the anchor enters only
    # the right-hand side of the deviation constraints, so the small positive
    # allocations a crossover-free barrier leaves on the branches that carry
    # nothing - measured 2e-7 MVAr against a smallest real allocation of about
    # 1 MVAr - cost nothing and need no threshold to be told apart.
    sssc_ext_i = pd.Index([], name="LineX")
    if "LineX" in n.components and not n.line_xs.empty:
        extendable = n.line_xs.get("sssc_nom_extendable")
        if extendable is not None:
            sssc_ext_i = n.line_xs.index[extendable.fillna(False).astype(bool)]
    sssc_on = proximal_on and proximal_target in ("sssc", "both")
    if sssc_on and sssc_ext_i.empty:
        logger.warning(
            "proximal_target=%r asks for a proximal term on the series "
            "compensation, but no LineX has 'sssc_nom_extendable' set. The "
            "term is skipped.",
            proximal_target,
        )
        sssc_on = False
    branches_on = proximal_on and proximal_target in ("branches", "both")

    if sssc_on:
        sssc_weights = (
            n.line_xs["capital_cost_sssc"].reindex(sssc_ext_i).astype(float)
        )
        sssc_weights = sssc_weights.where(
            np.isfinite(sssc_weights) & (sssc_weights > 0.0)
        )
        sssc_costs = sssc_weights.fillna(
            float(sssc_weights.mean()) if sssc_weights.notna().any() else 1.0
        )
    else:
        sssc_costs = pd.Series(dtype=float)
    # cost of the capacity that is already installed, which the objective of
    # every iteration carries along. It is frozen at the value of the first
    # iteration, where the capacities still are the initial ones.
    installed_cost = [0.0]

    move_costs = (
        pd.concat(proximal_costs, names=["component", "name"]).reindex(ext_branches)
        if branch_components
        else pd.Series(dtype=float)
    )

    def sssc_move(sssc: pd.Series | None, center: pd.Series | None) -> float:
        """Size of the SSSC part of a step, in the metric the term charges."""
        if not sssc_on or sssc is None or center is None:
            return 0.0
        distance = (
            sssc.reindex(sssc_ext_i).astype(float)
            - center.reindex(sssc_ext_i).astype(float)
        ).abs()
        return float((sssc_costs * distance).sum())

    def proximal_move(caps: pd.Series, center: pd.Series) -> float:
        """
        Size of a step in the metric the proximal term charges for, i.e. the
        capital cost of the capacity it moved.
        """
        moved = 0.0
        for c in branch_components if branches_on else ():
            ext_i = branch_data[c]["ext_i"]
            if ext_i.empty:
                continue
            distance = (
                caps.xs(c, level="component").reindex(ext_i)
                - center.xs(c, level="component").reindex(ext_i)
            ).abs()
            moved += float((proximal_costs[c] * distance).sum())
        return moved

    def add_proximal_penalty(
        network: Network,
        center: pd.Series,
        weight: float,
        sssc_center: pd.Series | None = None,
    ) -> None:
        """
        Penalise moving the branch capacities away from the previous iterate.

        The absolute value is modelled with a non-negative deviation variable
        ``d_l >= |F_l - F_l'|``, which the penalty drives onto the bound, so the
        model stays a linear program.
        """
        m = network.model
        if sssc_on and sssc_center is not None and "LineX-sssc_nom" in m.variables:
            rating = m["LineX-sssc_nom"]
            # ``intersection`` drops the index name, which linopy needs as the
            # dimension name of the deviation variable
            index = (
                pd.Index(rating.indexes[rating.dims[0]])
                .intersection(sssc_ext_i)
                .rename(rating.dims[0])
            )
            if not index.empty:
                cost = sssc_costs.reindex(index)
                anchor = sssc_center.reindex(index).astype(float)
                deviation = m.add_variables(
                    lower=0, coords=[index], name="LineX-sssc_nom_deviation"
                )
                m.add_constraints(
                    deviation - rating >= -anchor,
                    name="LineX-sssc_nom_deviation-upper",
                )
                m.add_constraints(
                    deviation + rating >= anchor,
                    name="LineX-sssc_nom_deviation-lower",
                )
                m.objective = m.objective + (deviation * (weight * cost)).sum()
        if center is None or not branches_on:
            return
        for c in branch_components:
            ext_i = branch_data[c]["ext_i"]
            attr = nominal_attrs[c]
            if ext_i.empty or f"{c}-{attr}" not in m.variables:
                continue
            capacity = m[f"{c}-{attr}"]
            anchor = center.xs(c, level="component").reindex(ext_i)
            deviation = m.add_variables(
                lower=0, coords=[ext_i], name=f"{c}-{attr}_deviation"
            )
            m.add_constraints(
                deviation - capacity >= -anchor, name=f"{c}-{attr}_deviation-upper"
            )
            m.add_constraints(
                deviation + capacity >= anchor, name=f"{c}-{attr}_deviation-lower"
            )
            m.objective = m.objective + (
                deviation * (weight * proximal_costs[c])
            ).sum()

    def extra_functionality_for(
        center: pd.Series | None,
        weight: float,
        sssc_center: pd.Series | None = None,
    ) -> Any:
        def extra_functionality(network: Network, sns: pd.Index) -> None:
            anchored = center is not None or sssc_center is not None
            if weight > 0.0 and anchored:
                add_proximal_penalty(network, center, weight, sssc_center)
            if user_extra_functionality is not None:
                user_extra_functionality(network, sns)

        return extra_functionality

    def solved_cost(
        network: Network,
        caps: pd.Series,
        center: pd.Series | None,
        weight: float,
        sssc: pd.Series | None = None,
        sssc_center: pd.Series | None = None,
    ) -> float:
        """
        System cost of the solution just obtained, i.e. the cost of the
        expansion and of the operation.

        The value of the linear program is used rather than ``n.objective``:
        for branches carrying a standard type PyPSA derives ``s_nom`` from
        ``num_parallel``, so the constant of the already installed capacity,
        which ``n.objective`` subtracts, drifts with the linearisation capacity
        and is not comparable between iterations. The cost of the already
        installed capacity is therefore subtracted as it was determined in the
        first iteration, and the proximal penalty, which is not part of the
        system cost, is removed again.
        """
        model = getattr(network, "model", None)
        objective = getattr(model, "objective", None) if model is not None else None
        value = getattr(objective, "value", None)
        if value is None:
            # without a model, fall back to the objective reported on the
            # network and add the constant of the installed capacity back in
            value = float(getattr(network, "objective", np.nan))
            if getattr(network, "_objective_constant_missing_from_expression", False):
                value += float(getattr(network, "objective_constant", 0.0))
        value = float(value)
        if weight > 0.0 and center is not None and branches_on:
            value -= weight * proximal_move(caps, center)
        if weight > 0.0 and sssc_center is not None:
            value -= weight * sssc_move(sssc, sssc_center)
        return value - installed_cost[0]

    def cost_converged(history: list[float], cost: float) -> bool:
        """
        The system cost has to be stationary over ``cost_window`` iterations.
        """
        if len(history) < cost_window:
            return False
        window = [*history[-cost_window:], cost]
        reference = max(abs(window[-1]), 1e-12)
        return all(
            abs(window[i + 1] - window[i]) / reference <= cost_threshold
            for i in range(cost_window)
        )

    def branch_flow(network: Network, c: str, index: pd.Index) -> pd.DataFrame:
        """
        Flow of the branches of ``c`` as it enters the voltage law.

        The voltage law is written in the flow variable ``s``, while the
        reported ``p0`` carries half of the branch loss on top of it once
        ``transmission_losses`` is enabled (see ``assign_solution``). That half
        loss is not part of the voltage law and does not cancel around a cycle,
        so leaving it in would add a systematic offset to the residual instead
        of the linearisation error the residual is meant to measure.
        """
        pnl = network.pnl(c)
        flow = pnl["p0"].reindex(index=index)
        loss = pnl.get("loss")
        if loss is not None and not loss.empty:
            flow = flow - loss.reindex(
                index=index, columns=flow.columns
            ).fillna(0.0) / 2
        return flow

    def evaluate_kvl(
        network: Network,
        solved_flows: Mapping[str, pd.DataFrame] | None = None,
        solved_q_sssc: pd.DataFrame | None = None,
    ) -> tuple[pd.DataFrame, float, float]:
        """
        Evaluate the voltage law for an outer-iteration solution.

        The branch parameters have to be consistent with the capacities of that
        solution, i.e. ``update_line_params`` has to be called with them
        beforehand. Returns the branch terms ``x_pu_eff * f - q_sssc / F`` of
        all branches participating in a cycle, the residual of the exact
        voltage law in the 1-norm and the same residual relative to the
        voltage drop the branches of the cycle produce at their rated
        capacity. The latter scale is invariant under the capacity iteration,
        since the impedance scales with the inverse capacity.
        """
        network.calculate_dependent_values()
        names = ["component", "name"]
        multi_invest = getattr(network, "_multi_invest", False)
        periods = snapshots.unique("period") if multi_invest else [None]

        q_sssc = solved_q_sssc
        if (
            q_sssc is None
            and "LineX" in passive_components
            and "q_sssc" in network.line_xs_t
        ):
            q_sssc = network.line_xs_t["q_sssc"]

        s_nom_def = pd.concat(
            {c: capacity_reference(network, c) for c in passive_components},
            names=names,
        )

        period_frames = []
        residual_norm = 0.0
        term_norm = 0.0
        for period in periods:
            period_sns = (
                snapshots if period is None else snapshots[snapshots.get_loc(period)]
            )
            flows = pd.concat(
                {
                    c: (
                        solved_flows[c].reindex(index=period_sns)
                        if solved_flows is not None
                        else branch_flow(network, c, period_sns)
                    )
                    for c in passive_components
                },
                axis=1,
                names=names,
            )
            sub_frames = []
            for branches_i, C, weightings, carrier in kirchhoff_voltage_cycles(
                network, period
            ):
                terms = (
                    flows.reindex(columns=branches_i).to_numpy() * weightings[None, :]
                )
                if q_sssc is not None and carrier == "AC":
                    is_x = np.asarray(branches_i.get_level_values(0) == "LineX")
                    if is_x.any():
                        f_ref = s_nom_def.reindex(branches_i[is_x]).to_numpy()
                        terms[:, is_x] -= (
                            q_sssc.reindex(
                                index=period_sns,
                                columns=branches_i.get_level_values(1)[is_x],
                            ).to_numpy()
                            / f_ref[None, :]
                        )
                # Keep the cycle matrix sparse. Densifying it on every outer
                # iteration is especially costly for large meshed networks.
                residual = (C.T @ terms.T).T
                residual_norm += float(np.abs(residual).sum())
                rated = np.abs(weightings * s_nom_def.reindex(branches_i).to_numpy())
                term_norm += len(period_sns) * float((abs(C).T @ rated).sum())
                sub_frames.append(
                    pd.DataFrame(terms, index=period_sns, columns=branches_i)
                )
            if sub_frames:
                period_frames.append(pd.concat(sub_frames, axis=1))

        if not period_frames:
            empty = pd.DataFrame(
                index=snapshots,
                columns=pd.MultiIndex.from_arrays([[], []], names=names),
                dtype=float,
            )
            return empty, 0.0, 0.0

        cycle_terms = pd.concat(period_frames)
        relative = residual_norm / term_norm if term_norm > 0 else 0.0
        return cycle_terms, residual_norm, relative

    def capacity_sensitivity(
        cycle_terms: pd.DataFrame, center: pd.Series
    ) -> pd.DataFrame | None:
        """
        First-order sensitivity of the KVL branch terms to the branch capacity.

        Both the natural reactance and the SSSC compensation term scale with the
        inverse capacity, hence the term of branch ``l`` has the derivative
        ``- term_l / F_l`` at the linearisation capacity ``F_l``.

        Sensitivities that are negligible against the voltage drop their own
        branch causes at its rated capacity are set to zero, see
        ``sensitivity_tolerance``.
        """
        columns = cycle_terms.columns.intersection(scaling_branches)
        if columns.empty:
            return None
        reference = center.reindex(columns).clip(
            lower=capacity_floor.reindex(columns)
        )
        terms = cycle_terms[columns]
        sensitivity = terms.div(reference, axis=1)

        if sensitivity_tolerance > 0.0:
            # The term of a branch that is idle at a snapshot vanishes, and so
            # does its sensitivity. Such a term cannot move the solution, but it
            # enters the constraint matrix as a coefficient orders of magnitude
            # below the rest of its row: the flows come from a barrier solve,
            # which leaves the idle branches at a small but non-zero flow.
            n.calculate_dependent_values()
            weights = pd.concat(
                {
                    c: n.df(c)["x_pu_eff"]
                    .where(
                        n.df(c).bus0.map(n.buses.carrier) == "AC",
                        n.df(c)["r_pu_eff"],
                    )
                    .astype(float)
                    for c in branch_components
                },
                names=["component", "name"],
            )
            # the drop of the branch at its rated capacity, which the impedance
            # scaling leaves invariant under the iteration
            rated = (weights.reindex(columns) * reference).abs()
            negligible = terms.abs().le(sensitivity_tolerance * rated, axis=1)
            sensitivity = sensitivity.mask(negligible, 0.0)
            logger.info(
                "Dropped %d of %d branch sensitivities below %.1e of the rated "
                "voltage drop of their branch.",
                int(negligible.to_numpy().sum()),
                negligible.size,
                sensitivity_tolerance,
            )

        return sensitivity

    def trust_region_widths(
        center: pd.Series, radius: float
    ) -> tuple[pd.Series, pd.Series]:
        """
        Half widths ``(down, up)`` of the trust region per branch.

        The radius is relative to the capacity of the linearisation point, so
        that the region stays meaningful once the capacities have grown well
        beyond their initial value, and is floored by the initial capacity, so
        that it does not collapse for branches shrinking towards zero.

        The region is not symmetric. Both parts of a branch term of the voltage
        law scale with the inverse capacity, so in the relative deviation
        ``u = F / F_def - 1`` of ``define_relative_capacity_deviation`` the
        exact term is ``T / (1 + u)`` while the linearisation carried by the
        constraint is ``T (1 - u)``, leaving the error

        .. math::
            e(u) = T \\, \\frac{u^2}{1 + u}.

        That error is even in neither direction: it stays second order while
        the capacity grows but diverges as the capacity collapses, so a
        symmetric box tolerates a much larger error below the linearisation
        point than above it - three times as much at ``radius = 0.5`` and an
        unbounded amount from ``radius = 1`` on, where the symmetric region
        reaches a vanishing capacity.
        Widening upwards by ``radius`` and downwards by ``radius / (1 + radius)``
        instead makes the region geometrically symmetric,

        .. math::
            \\bar{F} / (1 + \\rho) \\;\\le\\; F \\;\\le\\; \\bar{F} (1 + \\rho),

        which is the natural symmetry of a term proportional to ``1 / F`` and
        equalises the error exactly: both ends leave ``e = T \\rho^2 / (1 + \\rho)``.
        """
        scale = np.maximum(center.abs(), radius_scale)
        return radius / (1.0 + radius) * scale, radius * scale

    def trust_region_binding(
        current: pd.Series, previous: pd.Series, radius: float
    ) -> bool:
        """
        Whether the trust region restricts the step.

        Measured as the share of the moved capital cost that sits at the
        boundary of the region. In the maximum norm a single small branch
        swinging between degenerate optima already fills its own width and
        would report a restriction that does not exist, so the branches at the
        boundary are weighted with the cost of the capacity they move.
        """
        if ext_branches.empty:
            return False
        delta = (current - previous).loc[ext_branches]
        down, up = trust_region_widths(previous, radius)
        # the region is asymmetric, so a branch is measured against the half
        # width on the side it actually moved to
        widths = up.loc[ext_branches].where(delta >= 0, down.loc[ext_branches])
        change = delta.abs()
        at_boundary = (change / widths.clip(lower=1e-12)) >= 0.9
        moved = move_costs * change
        total = float(moved.sum())
        if total <= 0.0:
            return False
        return float(moved[at_boundary].sum()) / total >= 0.1

    def step_alignment(current: pd.Series, previous: pd.Series) -> float:
        """
        Cosine between the fixed-point residuals of two consecutive steps.

        The residual ``r = G(z) - z`` is the move the linear model asks for
        from the point it was linearised at. A value near ``+1`` means the
        iteration keeps walking in the same direction and is making progress; a
        value near ``-1`` means the step undoes the one before it, which is the
        signature of an oscillation: the branch capacity and the impedance it
        sets move against each other, so the underlying fixed-point map is
        reflecting rather than contracting. The box does not damp that on its
        own - it is geometrically symmetric, so a branch that jumps to one wall
        finds the other wall exactly back at its previous capacity, and the
        orbit reproduces itself for as long as the radius is held. Only a
        smaller radius shrinks it.

        The directions are weighted with the cost of the capacity they move,
        for the same reason ``trust_region_binding`` is: a swarm of tiny cheap
        branches swinging between degenerate optima would otherwise decide the
        angle.

        """
        if ext_branches.empty:
            return 0.0
        a = (move_costs * current.loc[ext_branches]).to_numpy()
        b = (move_costs * previous.loc[ext_branches]).to_numpy()
        na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
        if na <= 0.0 or nb <= 0.0:
            return 0.0
        return float(a @ b / (na * nb))

    def set_trust_region(network: Network, center: pd.Series, radius: float) -> None:
        down, up = trust_region_widths(center, radius)
        for c in branch_components:
            ext_i = branch_data[c]["ext_i"]
            if ext_i.empty:
                continue
            attr = nominal_attrs[c]
            lower, upper = original_nominal_bounds[c]
            middle = center.xs(c, level="component").reindex(ext_i)
            delta_down = down.xs(c, level="component").reindex(ext_i)
            delta_up = up.xs(c, level="component").reindex(ext_i)
            network.df(c).loc[ext_i, f"{attr}_min"] = np.maximum(
                lower.reindex(ext_i), middle - delta_down
            )
            network.df(c).loc[ext_i, f"{attr}_max"] = np.minimum(
                upper.reindex(ext_i), middle + delta_up
            )

    link_bounds: dict[str, pd.Series] = {}

    def unpin_links(network: Network) -> None:
        """Restore the link bounds ``pin_capacities`` replaced."""
        for column, values in link_bounds.items():
            network.links[column] = values
        link_bounds.clear()

    def pin_capacities(
        network: Network, center: pd.Series, links: pd.Series | None = None
    ) -> None:
        """
        Fix the extendable branch capacities at ``center``.

        Used for the report solve that closes every run. That solve exists to
        cost the last accepted plan, not to look for a better one, and the two
        are not the same thing: its impedances are those of ``center``, so
        leaving the capacities free lets it move away from the point they were
        computed at and buy an objective the network cannot deliver. On a
        converged run the difference is small, since the iteration has stopped
        moving, but it is the same distinction, and pinning is what makes the
        two exits report the same quantity. Measured on
        ``test_tr_4_3GVAsssc``, on a run that did not converge, an
        unrestricted report solve returned a plan 4.1 % (capital cost weighted)
        away from its linearisation point, with a residual of the exact voltage
        law of 1.0e-2 against the 2.7e-4 of the iterate it replaced, and an
        objective 0.33 % below every iterate of the run.

        The DC links are held too, at the capacities of the same solve. They
        carry no voltage law and so never enter the iteration's own controls,
        but they are transmission capacity all the same and on the network
        above they expand from 21 to 553 GW: re-optimised at fixed AC
        capacities they make the report a different plan rather than the price
        of this one. Everything that is not capacity - the series compensation,
        the dispatch, the storage - is left free, which is what makes this the
        cost *of* the plan.
        """
        if links is not None and not link_ext_i.empty:
            link_bounds["p_nom_min"] = network.links["p_nom_min"].copy()
            link_bounds["p_nom_max"] = network.links["p_nom_max"].copy()
            fixed = links.reindex(link_ext_i).astype(float).clip(
                lower=link_bounds["p_nom_min"].reindex(link_ext_i),
                upper=link_bounds["p_nom_max"].reindex(link_ext_i),
            )
            network.links.loc[link_ext_i, "p_nom_min"] = fixed
            network.links.loc[link_ext_i, "p_nom_max"] = fixed
        for c in branch_components:
            ext_i = branch_data[c]["ext_i"]
            if ext_i.empty:
                continue
            attr = nominal_attrs[c]
            lower, upper = original_nominal_bounds[c]
            fixed = (
                center.xs(c, level="component")
                .reindex(ext_i)
                .clip(lower=lower.reindex(ext_i), upper=upper.reindex(ext_i))
            )
            network.df(c).loc[ext_i, f"{attr}_min"] = fixed
            network.df(c).loc[ext_i, f"{attr}_max"] = fixed

    def reset_trust_region(network: Network) -> None:
        for c in branch_components:
            attr = nominal_attrs[c]
            lower, upper = original_nominal_bounds[c]
            network.df(c)[f"{attr}_min"] = lower
            network.df(c)[f"{attr}_max"] = upper

    if track_iterations:
        for c, attr in pd.Series(nominal_attrs)[list(n.branch_components)].items():
            n.df(c)[f"{attr}_opt_0"] = n.df(c)[f"{attr}"]
        if "LineX" in n.components and not n.line_xs.empty:
            n.line_xs["sssc_nom_opt_0"] = n.line_xs["sssc_nom"]

    branch_cap_min, branch_cap_max = collect_branch_bounds(n)
    original_nominal_bounds = {
        c: (
            n.df(c)[f"{nominal_attrs[c]}_min"].copy(),
            n.df(c)[f"{nominal_attrs[c]}_max"].copy(),
        )
        for c in branch_components
    }
    radius_min, radius_max = (float(b) for b in trust_region_bounds)
    shrink, expand = (float(f) for f in trust_region_factors)
    tighten_cosine, relax_cosine = (float(a) for a in trust_region_alignment)
    if not 0.0 < radius_min <= radius_max:
        raise ValueError("trust_region_bounds must satisfy 0 < lower <= upper.")
    if not -1.0 <= tighten_cosine <= relax_cosine <= 1.0:
        raise ValueError(
            "trust_region_alignment must satisfy -1 <= tighten <= relax <= 1."
        )
    if not 0.0 < shrink <= 1.0 <= expand:
        raise ValueError(
            "trust_region_factors must satisfy 0 < shrink <= 1 <= expand."
        )
    radius = float(np.clip(trust_region_initial, radius_min, radius_max))
    linearised = scheme == "slp"

    current_def = initial_caps.copy()
    cost_history: list[float] = []
    proximal_delta = proximal_weight if proximal_on else 0.0

    def tighten() -> None:
        """
        Shrink the trust region, i.e. restrict the step the linear model is
        trusted over. The weight of the proximal term is not touched: it is
        fixed for the whole run, see ``PROXIMAL_WEIGHT``.
        """
        nonlocal radius
        radius = max(radius * shrink, radius_min)

    def relax() -> None:
        """Widen the trust region again."""
        nonlocal radius
        radius = min(radius * expand, radius_max)

    reference_terms: pd.DataFrame | None = None
    # anchor of the SSSC proximal term, carried like ``current_def`` is for the
    # capacities. It is the solved rating, not a linearisation point, so it is
    # never damped or clipped.
    current_sssc: pd.Series | None = None
    # last solved DC link capacities, held by the report solve alongside the AC
    current_links: pd.Series | None = None
    previous_residual: pd.Series | None = None
    plain_step = False
    has_converged = False
    iteration = 1
    status = "ok"
    condition = "optimal"
    records: list[dict[str, Any]] = []
    # Intermediate iterations only need capacities, KVL flows and (where
    # present) SSSC compensation. Tracking adds only nominal-capacity columns
    # and objective scalars, not full primal/dual network results. Every
    # iterate is superseded by the report solve that closes the run, so every
    # iterate's model is released here.

    def discard_model(network: Network) -> None:
        pending = getattr(network, "_pending_full_solve", None)
        if pending is not None:
            pending.discard()
            network._pending_full_solve = None
        if hasattr(network, "model"):
            del network.model

    try:
        while True:
            if iteration > max_iterations:
                logger.info(
                    "Iteration %d beyond max_iterations %d. Stopping ...",
                    iteration,
                    max_iterations,
                )
                break

            update_line_params(n, current_def)

            # linearise the KVL constraint in the branch capacities and restrict
            # them to the trust region around the linearisation point
            sensitivity = None
            if linearised and reference_terms is not None and not plain_step:
                sensitivity = capacity_sensitivity(reference_terms, current_def)
            # a fallback step onto the plain fixed point is taken on its own,
            # with neither the linearisation nor the box that made the previous
            # attempt infeasible
            plain_step_now = plain_step
            plain_step = False
            n._kvl_capacity_sensitivity = sensitivity
            # the box needs a previous iterate to be centred on, not a
            # linearisation, so it applies under either scheme
            boxed = (
                not plain_step_now
                and (sensitivity is not None or iteration > 1)
            )
            if boxed:
                set_trust_region(n, current_def, radius)
            else:
                reset_trust_region(n)

            # anchor of the proximal term, absent in the first iteration where
            # no previous iterate exists
            anchor = current_def if iteration > 1 else None
            sssc_anchor = current_sssc if iteration > 1 else None
            weight = proximal_delta if anchor is not None else 0.0
            status, condition = solve_once(
                snapshots,
                lightweight=True,
                extra_functionality=extra_functionality_for(
                    anchor, weight, sssc_anchor
                ),
            )
            if status != "ok":
                if sensitivity is not None:
                    # the linearised problem can be infeasible where the frozen
                    # impedances require more capacity than the trust region
                    # admits. Fall back to an unrestricted fixed-point step,
                    # which is always feasible, and be more careful afterwards.
                    tighten()
                    plain_step = True
                    logger.warning(
                        "Iteration %d failed with status %s/%s, falling back to a "
                        "fixed-point step and restricting the next one "
                        "(radius %.3e).",
                        iteration,
                        status,
                        condition,
                        radius,
                    )
                    records.append(
                        {
                            "iteration": iteration,
                            "status": status,
                            "cost": np.nan,
                            "cost_change": np.nan,
                            "step": np.nan,
                            "violation": np.nan,
                            "violation_rel": np.nan,
                            "radius": radius,
                            "binding": False,
                            "proximal": np.nan,
                            "accepted": False,
                        }
                    )
                    iteration += 1
                    continue
                raise RuntimeError(
                    f"Optimization failed with status {status} and "
                    f"termination {condition}"
                )

            iteration_result = (
                collect_iteration_result(n)
                if hasattr(n, "model")
                else None
            )
            caps = (
                iteration_result.capacities
                if iteration_result is not None
                else collect_branch_caps("s_nom_opt")
            )
            if violates_bounds(caps, branch_cap_min):
                logger.warning(
                    "Iteration %d reports status %s/%s but returns transmission "
                    "capacities below the lower bounds of the problem, which "
                    "indicates a numerically failed solve.",
                    iteration,
                    status,
                    condition,
                )
                if sensitivity is not None:
                    # such a solution must not become the linearisation point
                    tighten()
                    plain_step = True
                    records.append(
                        {
                            "iteration": iteration,
                            "status": "failed solution",
                            "cost": np.nan,
                            "cost_change": np.nan,
                            "step": np.nan,
                            "violation": np.nan,
                            "violation_rel": np.nan,
                            "radius": radius,
                            "binding": False,
                            "proximal": np.nan,
                            "accepted": False,
                        }
                    )
                    discard_model(n)
                    iteration += 1
                    continue

            if iteration == 1:
                installed_cost[0] = float(getattr(n, "objective_constant", 0.0))
            solved_sssc = (
                None if iteration_result is None else iteration_result.sssc_nom
            )
            if iteration_result is not None and iteration_result.link_caps is not None:
                current_links = iteration_result.link_caps
            cost = solved_cost(n, caps, anchor, weight, solved_sssc, sssc_anchor)
            diff = relative_capacity_change(caps, current_def, initial_caps)

            if track_iterations:
                save_optimal_capacities(
                    n,
                    iteration,
                    status,
                    None
                    if iteration_result is None
                    else iteration_result.tracked_capacities,
                    None if iteration_result is None else iteration_result.sssc_nom,
                    None if iteration_result is None else iteration_result.objective,
                )

            # residual of the exact voltage law at the capacities of the new
            # iterate, which is the error of the linear model solved above
            update_line_params(n, caps)
            cycle_terms, violation, violation_rel = evaluate_kvl(
                n,
                None if iteration_result is None else iteration_result.flows,
                None if iteration_result is None else iteration_result.q_sssc,
            )
            # ``cycle_terms`` and the scalar records above are detached from
            # Linopy, so the model itself is not needed for the decisions
            # below. Whether it is completed in place or discarded is decided
            # once that decision (converged / superseded) is known.

            # the linear model predicted a vanishing residual, so the residual
            # left over is the error the trust region has to control. It only
            # steers the width of the next step: a solved iterate is never
            # discarded, and at a converged point the linearisation reproduces
            # its own point, where the constraint coincides with the exact one.
            binding = boxed and trust_region_binding(caps, current_def, radius)
            # A step held by the box is not a converged one, however little the
            # cost moved: the size of the step is proportional to the radius, so
            # a narrow enough region satisfies any tolerance on the cost while
            # the iterate is still being dragged. Measured on a case where the
            # region bound on every iteration, the four settings of
            # ``trust_region_alignment`` all stopped on the boundary and, at
            # matched residual, the ones that had shrunk the radius fastest
            # returned the most expensive plan - up to 0.05 % over the slowest -
            # because the plan is a prisoner of the path the box allowed. The
            # region has to let go before the point counts as one the iteration
            # chose rather than one it was held at. It does let go on its own:
            # ``binding`` is the share of the moved capital sitting at the
            # boundary, which falls as the step shrinks against a fixed radius,
            # and the relax branch below widens the region while the steps stay
            # aligned.
            converged = (
                cost_converged(cost_history, cost)
                and iteration >= min_iterations
                and not binding
            )
            residual = caps - current_def
            alignment = (
                np.nan
                if previous_residual is None
                else step_alignment(residual, previous_residual)
            )
            # every solved iterate is kept; the controls only set how far the
            # model is trusted over the *next* step
            # only the radius is adapted, so there is nothing to decide where
            # the box is off
            # the first step has no previous direction to be compared with, so
            # there is no evidence to move the radius on
            controlled = not converged and alignment == alignment
            if controlled and alignment <= tighten_cosine:
                # the step undid the one before it. Nothing else the iteration
                # measures sees this: the residual of the voltage law is small
                # on an orbit, and the size of the step is flat rather than
                # growing, so a criterion on either only fires by accident.
                # The radius is the one control that shrinks the orbit, since
                # the orbit lives on the walls of the box.
                tighten()
            elif controlled and alignment >= relax_cosine and binding:
                # the iteration keeps walking the same way and the box is what
                # holds it back
                relax()

            cost_change = (
                abs(cost - cost_history[-1]) / max(abs(cost), 1e-12)
                if cost_history
                else np.nan
            )
            # signed for readability in the log only; convergence and the
            # records use the unsigned magnitude above
            cost_change_signed = (
                (cost - cost_history[-1]) / max(abs(cost), 1e-12)
                if cost_history
                else np.nan
            )
            logger.info(
                "Iteration %d: relative cost change = %+.3e, relative capacity "
                "change = %.3e, relative KVL residual = %.3e%s",
                iteration,
                cost_change_signed,
                diff,
                violation_rel,
                ""
                if not (boxed or proximal_on)
                else (
                    (f", radius = {radius:.3e}" if boxed else "")
                    + (
                        f", alignment = {alignment:+.3f}"
                        if alignment == alignment
                        else ""
                    )
                    + (f", proximal = {proximal_delta:.3e}" if proximal_on else "")
                    + (", binding" if binding else "")
                ),
            )
            records.append(
                {
                    "iteration": iteration,
                    "status": status,
                    "cost": cost,
                    "cost_change": cost_change,
                    "step": diff,
                    "violation": violation,
                    "violation_rel": violation_rel,
                    "radius": radius if boxed else np.nan,
                    "alignment": alignment,
                    "binding": binding,
                    "proximal": weight,
                    "accepted": True,
                }
            )

            if converged:
                current_def = clip_branch_caps(
                    caps.copy(), branch_cap_min, branch_cap_max
                )
                # linearise the final solve at the converged point, where the
                # linearisation is exact
                reference_terms = cycle_terms
                cost_history.append(cost)
                has_converged = True
                # The report solve below closes every run, so this iterate's
                # model is released like any other rather than completed in
                # place: it is solved once more at the same capacities, with
                # them pinned there.
                discard_model(n)
                break

            previous_residual = residual
            if solved_sssc is not None:
                current_sssc = solved_sssc.clip(lower=0.0)
            cost_history.append(cost)
            if linearised:
                current_def = clip_branch_caps(
                    caps.copy(), branch_cap_min, branch_cap_max
                )
                reference_terms = cycle_terms
            elif iteration == 1:
                current_def = caps.copy()
            else:
                current_def = clip_branch_caps(
                    caps.copy(), branch_cap_min, branch_cap_max
                )

            discard_model(n)
            iteration += 1
    finally:
        # never leave the linearisation or the trust region on the network, not
        # even when the loop is left through an exception
        n._kvl_capacity_sensitivity = None
        reset_trust_region(n)
        n.iteration_log = pd.DataFrame(
            records,
            columns=[
                "iteration",
                "status",
                "cost",
                "cost_change",
                "step",
                "violation",
                "violation_rel",
                "radius",
                "alignment",
                "binding",
                "proximal",
                "accepted",
            ],
        ).set_index("iteration")

    if hasattr(n, "model"):
        logger.info(
            "Deleting model instance `n.model` from previous run to reclaim memory."
        )
        del n.model
        gc.collect()

    # Every run is closed by this solve, converged or not: the iterates are
    # solved lightweight and their models discarded, so no solved iterate is
    # left to read a full result from. Solve once more at the capacities of the
    # last iterate to obtain a fully populated network.
    logger.info(
        "Preparing final solve with updated transmission parameters at the "
        "capacities of the %s iterate.",
        "converged" if has_converged else "last",
    )
    last_success_status, last_success_condition = status, condition

    update_line_params(n, current_def)

    try:
        n._kvl_capacity_sensitivity = None
        # the capacities are held at the last iterate rather than released:
        # this solve reports that plan, it does not look for another one, see
        # ``pin_capacities``
        pin_capacities(n, current_def, current_links)
        n.calculate_dependent_values()
        status, condition = solve_once(
            snapshots,
            # No penalty: this solve is not a step of the iteration but the
            # report of one, so it has to return the true system cost at the
            # last accepted capacities. The capacities are pinned, so the
            # proximal term would only add a constant, but the trust region
            # and the penalty are both step controls and neither belongs in a
            # solve that takes no step.
            extra_functionality=extra_functionality_for(None, 0.0),
        )
    finally:
        n._kvl_capacity_sensitivity = None
        reset_trust_region(n)
        unpin_links(n)

    if status == "ok":
        return status, condition
    else:
        logger.warning(
            "Final solve at the capacities of the last iterate failed with "
            "status %s/%s. The status of the last successful iterate is "
            "returned, but its model was released, so the network carries no "
            "solution from it - only 'n.iteration_log' and, where tracking is "
            "enabled, the per-iteration capacity columns.",
            status,
            condition,
        )
        status, condition = last_success_status, last_success_condition

    return status, condition


def optimize_security_constrained(
    n: Network,
    snapshots: Sequence | None = None,
    branch_outages: Sequence | pd.Index | pd.MultiIndex | None = None,
    multi_investment_periods: bool = False,
    model_kwargs: dict = {},
    **kwargs: Any,
) -> tuple[str, str]:
    """
    Computes Security-Constrained Linear Optimal Power Flow (SCLOPF).

    This ensures that no branch is overloaded even given the branch outages.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : list-like, optional
        Set of snapshots to consider in the optimization. The default is None.
    branch_outages : list-like/pandas.Index/pandas.MultiIndex, optional
        Subset of passive branches to consider as possible outages. If a list
        or a pandas.Index is passed, it is assumed to identify lines. If a
        multiindex is passed, its first level has to contain the component names,
        the second the assets. The default None results in all passive branches
        to be considered.
    multi_investment_periods : bool, default False
        Whether to optimise as a single investment period or to optimise in multiple
        investment periods. Then, snapshots should be a ``pd.MultiIndex``.
    model_kwargs: dict
        Keyword arguments used by `linopy.Model`, such as `solver_dir` or `chunk`.
    **kwargs:
        Keyword argument used by `linopy.Model.solve`, such as `solver_name`,
        `problem_fn` or solver options directly passed to the solver.

    Returns
    -------
    None
    """
    all_passive_branches = n.passive_branches().index

    if branch_outages is None:
        branch_outages = all_passive_branches
    elif isinstance(branch_outages, (list, pd.Index)):
        branch_outages = pd.MultiIndex.from_product([("Line",), branch_outages])

        if diff := set(branch_outages) - set(all_passive_branches):
            raise ValueError(
                f"The following passive branches are not in the network: {diff}"
            )

    if not len(all_passive_branches):
        return n.optimize(
            snapshots,  # type: ignore
            multi_investment_periods=multi_investment_periods,
            model_kwargs=model_kwargs,
            **kwargs,
        )

    m = n.optimize.create_model(
        snapshots=snapshots,
        multi_investment_periods=multi_investment_periods,
        **model_kwargs,
    )

    for sn in n.sub_networks.obj:
        branches_i = sn.branches_i()
        outages = branches_i.intersection(branch_outages)

        if outages.empty:
            continue

        sn.calculate_BODF()
        BODF = pd.DataFrame(sn.BODF, index=branches_i, columns=branches_i)[outages]

        for c_outage, c_affected in product(outages.unique(0), branches_i.unique(0)):
            c_outage_ = c_outage + "-outage"
            c_outages = outages.get_loc_level(c_outage)[1]
            flow_outage = m.variables[c_outage + "-s"].loc[:, c_outages]
            flow_outage = flow_outage.rename({c_outage: c_outage_})

            bodf = BODF.loc[c_affected, c_outage]
            bodf = xr.DataArray(bodf, dims=[c_affected, c_outage_])
            additional_flow = flow_outage * bodf
            for bound, kind in product(("lower", "upper"), ("fix", "ext")):
                coord = c_affected + "-" + kind
                constraint = coord + "-s-" + bound
                if constraint not in m.constraints:
                    continue
                rename = {c_affected: coord}
                added_flow = additional_flow.rename(rename)
                con = m.constraints[constraint]  # use this as a template
                # idx now contains fixed/extendable for the subnetwork
                idx = con.lhs.indexes[coord].intersection(added_flow.indexes[coord])
                sel = {coord: idx}
                lhs = con.lhs.sel(sel) + added_flow.sel(sel)
                name = constraint + f"-security-for-{c_outage_}-in-{sn}"
                m.add_constraints(lhs, con.sign.sel(sel), con.rhs.sel(sel), name=name)

    return n.optimize.solve_model(**kwargs)


def optimize_with_rolling_horizon(
    n: Network,
    snapshots: Sequence | None = None,
    horizon: int = 100,
    overlap: int = 0,
    **kwargs: Any,
) -> Network:
    """
    Optimizes the network in a rolling horizon fashion.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : list-like
        Set of snapshots to consider in the optimization. The default is None.
    horizon : int
        Number of snapshots to consider in each iteration. Defaults to 100.
    overlap : int
        Number of snapshots to overlap between two iterations. Defaults to 0.
    **kwargs:
        Keyword argument used by `linopy.Model.solve`, such as `solver_name`,

    Returns
    -------
    None
    """
    if snapshots is None:
        snapshots = n.snapshots

    if horizon <= overlap:
        raise ValueError("overlap must be smaller than horizon")

    starting_points = range(0, len(snapshots), horizon - overlap)
    for i, start in enumerate(starting_points):
        end = min(len(snapshots), start + horizon)
        sns = snapshots[start:end]
        logger.info(
            f"Optimizing network for snapshot horizon [{sns[0]}:{sns[-1]}] ({i+1}/{len(starting_points)})."
        )

        if i:
            if not n.stores.empty:
                n.stores.e_initial = n.stores_t.e.loc[snapshots[start - 1]]
            if not n.storage_units.empty:
                n.storage_units.state_of_charge_initial = (
                    n.storage_units_t.state_of_charge.loc[snapshots[start - 1]]
                )

        status, condition = n.optimize(sns, **kwargs)  # type: ignore
        if status != "ok":
            logger.warning(
                f"Optimization failed with status {status} and condition {condition}"
            )
    return n


def optimize_mga(
    n: Network,
    snapshots: Sequence | None = None,
    multi_investment_periods: bool = False,
    weights: dict | None = None,
    sense: str | int = "min",
    slack: float = 0.05,
    model_kwargs: dict = {},
    **kwargs: Any,
) -> tuple[str, str]:
    """
    Run modelling-to-generate-alternatives (MGA) on network to find near-
    optimal solutions.

    Parameters
    ----------
    n : pypsa.Network snapshots : list-like
        Set of snapshots to consider in the optimization. The default is None.
    multi_investment_periods : bool, default False
        Whether to optimise as a single investment period or to optimize in
        multiple investment periods. Then, snapshots should be a
        ``pd.MultiIndex``.
    weights : dict-like
        Weights for alternate objective function. The default is None, which
        minimizes generation capacity. The weights dictionary should be keyed
        with the component and variable (see ``pypsa/variables.csv``), followed
        by a float, dict, pd.Series or pd.DataFrame for the coefficients of the
        objective function. Examples:

        >>> {"Generator": {"p_nom": 1}}
        >>> {"Generator": {"p_nom": pd.Series(1, index=n.generators.index)}}
        >>> {"Generator": {"p_nom": {"gas": 1, "coal": 2}}}
        >>> {"Generator": {"p": pd.Series(1, index=n.generators.index)}
        >>> {"Generator": {"p": pd.DataFrame(1, columns=n.generators.index, index=n.snapshots)}

        Weights for non-extendable components are ignored. The dictionary does
        not need to provide weights for all extendable components.
    sense : str|int
        Optimization sense of alternate objective function. Defaults to 'min'.
        Can also be 'max'.
    slack : float
        Cost slack for budget constraint. Defaults to 0.05.
    model_kwargs: dict
        Keyword arguments used by `linopy.Model`, such as `solver_dir` or
        `chunk`.
    **kwargs:
        Keyword argument used by `linopy.Model.solve`, such as `solver_name`,

    Returns
    -------
    status : str
        The status of the optimization, either "ok" or one of the codes listed
        in https://linopy.readthedocs.io/en/latest/generated/linopy.constants.SolverStatus.html
    condition : str
        The termination condition of the optimization, either
        "optimal" or one of the codes listed in
        https://linopy.readthedocs.io/en/latest/generated/linopy.constants.TerminationCondition.html
    """
    if snapshots is None:
        snapshots = n.snapshots

    if weights is None:
        weights = dict(Generator=dict(p_nom=pd.Series(1, index=n.generators.index)))

    # check that network has been solved
    if not hasattr(n, "objective"):
        msg = "Network needs to be solved with `n.optimize()` before running MGA."
        raise ValueError(msg)

    # create basic model
    m = n.optimize.create_model(
        snapshots=snapshots,
        multi_investment_periods=multi_investment_periods,
        **model_kwargs,
    )

    # build budget constraint
    if not multi_investment_periods:
        optimal_cost = n.statistics.capex().sum() + n.statistics.opex().sum()
        fixed_cost = n.statistics.installed_capex().sum()
    else:
        w = n.investment_period_weightings.objective
        optimal_cost = (
            n.statistics.capex().sum() * w + n.statistics.opex().sum() * w
        ).sum()
        fixed_cost = (n.statistics.installed_capex().sum() * w).sum()

    objective = m.objective
    if not isinstance(objective, (LinearExpression, QuadraticExpression)):
        objective = objective.expression

    if getattr(n, "_objective_constant_missing_from_expression", False):
        fixed_cost = 0.0

    m.add_constraints(
        objective + fixed_cost <= (1 + slack) * optimal_cost, name="budget"
    )

    # parse optimization sense
    if (
        isinstance(sense, str)
        and sense.startswith("min")
        or isinstance(sense, int)
        and sense > 0
    ):
        sense = 1
    elif (
        isinstance(sense, str)
        and sense.startswith("max")
        or isinstance(sense, int)
        and sense < 0
    ):
        sense = -1
    else:
        raise ValueError(f"Could not parse optimization sense {sense}")

    # build alternate objective
    objective = []
    for c, attrs in weights.items():
        for attr, coeffs in attrs.items():
            if isinstance(coeffs, dict):
                coeffs = pd.Series(coeffs)
            if attr == nominal_attrs[c] and isinstance(coeffs, pd.Series):
                coeffs = coeffs.reindex(n.get_extendable_i(c))
                coeffs.index.name = ""
            elif isinstance(coeffs, pd.Series):
                coeffs = coeffs.reindex(columns=n.df(c).index)
            elif isinstance(coeffs, pd.DataFrame):
                coeffs = coeffs.reindex(columns=n.df(c).index, index=snapshots)
            objective.append(m[f"{c}-{attr}"] * coeffs * sense)

    m.objective = merge(objective)

    status, condition = n.optimize.solve_model(**kwargs)

    # write MGA coefficients into metadata
    n.meta["slack"] = slack
    n.meta["sense"] = sense

    def convert_to_dict(obj: Any) -> Any:
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="list")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        else:
            return obj

    n.meta["weights"] = convert_to_dict(weights)

    return status, condition
