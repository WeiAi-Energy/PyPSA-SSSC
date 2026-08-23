#!/usr/bin/env python3
"""
Build abstracted, extended optimisation problems from PyPSA networks with
Linopy.
"""

from __future__ import annotations

import copy
import gc
import logging
from collections.abc import Sequence
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

# Share of the convergence tolerance a capacity move of the size of the previous
# one is charged by the proximal term, which bounds the bias the term can
# introduce. The weight is recalibrated in every iteration and hence grows as
# the moves get smaller, so a value below one still anneals into the damping
# that separates the degenerate alternative optima, only a few iterations later,
# while keeping the deviation of the reported cost around 1e-5 relative. Ten
# times smaller the annealing no longer reaches that level within a typical
# iteration budget and the term stays without effect, ten times larger it moves
# the cost by several times the convergence tolerance.
PROXIMAL_CALIBRATION = 0.1


def optimize_transmission_expansion_iteratively(
    n: Network,
    snapshots: Sequence | None = None,
    msq_threshold: float | None = None,
    min_iterations: int = 1,
    max_iterations: int = 100,
    track_iterations: bool = False,
    method: str = "trust_region",
    cost_threshold: float = 1e-5,
    cost_window: int = 2,
    proximal: bool = False,
    trust_region_initial: float = 0.5,
    trust_region_bounds: tuple[float, float] = (1e-3, 4.0),
    trust_region_tolerances: tuple[float, float] = (1e-3, 1e-1),
    trust_region_factors: tuple[float, float] = (0.5, 2.0),
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
    are resolved by an outer iteration, for which two schemes are available
    (see ``method``).

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
    method : {'fixed_point', 'trust_region'}, default 'trust_region'
        Scheme resolving the capacity dependence of the KVL constraint.

        ``'fixed_point'``
            Plain fixed-point iteration: the impedance and the SSSC
            compensation term are frozen at the capacity of the previous
            iterate, so the KVL constraint stays zeroth-order in the
            capacity.
        ``'trust_region'``
            Sequential linear programming: the KVL constraint carries the
            first-order sensitivity of its branch terms with respect to the
            branch capacity, evaluated at the previous iterate, and the
            capacities are restricted to a trust region around it. The
            radius is adapted from the exact KVL residual left by the new
            iterate, which the linear model predicted to vanish, and from the
            progress of the fixed-point residual; steps with too large a
            residual are rejected. The first iteration is a plain fixed-point
            step, since no linearisation point exists yet. Once converged, the
            linearisation is exact at that point and is kept for the final
            solve, so that the reported capacities and flows are consistent.
    cost_threshold : float, default 1e-5
        Convergence criterion: the iteration stops once the system cost has
        changed by less than this fraction of itself in each of the last
        ``cost_window`` accepted iterations. The cost is invariant under the
        exchange of degenerate alternative optima, which the change of the
        capacities is not, and it is the quantity the results are reported in.
    cost_window : int, default 2
        Number of consecutive relative cost changes that have to undercut
        ``cost_threshold``.
    proximal : bool, default False
        Add a proximal term ``delta * sum_l c_l |F_l - F_l'|`` to the objective,
        which penalises moving the capacity of branch ``l`` away from the
        previous iterate ``F_l'`` in units of its capital cost. It damps the
        iteration and resolves the degeneracy that lets equal-cost solutions
        exchange capacity between branches. The term vanishes at the fixed
        point, is excluded from the reported system cost, applies to both
        methods and is active from the second iteration onward, where a
        previous iterate exists.

        The weight is calibrated to the convergence criterion rather than
        given: ``delta`` is chosen such that a capacity move of the size of the
        previous one is charged ``PROXIMAL_CALIBRATION`` times the cost
        difference the iteration considers converged. Since it is recalibrated
        in every iteration, it grows as the moves get smaller, which anneals
        the iteration; it is bounded to keep the induced bias below a tenth of
        the moved capital cost. The value used in each iteration is reported in
        ``n.iteration_log``.
    trust_region_initial : float, default 0.5
        Initial trust region radius of ``method='trust_region'``. The capacity
        of branch ``l`` is restricted to
        ``F_l = F_def_l +- radius * max(F_def_l, F0_l)``, i.e. the radius is
        relative to the capacity of the linearisation point, floored by the
        initial capacity.
    trust_region_bounds : tuple of float, default (1e-3, 4.0)
        Smallest and largest admissible trust region radius. The iteration
        stops if the radius has to be shrunk below the lower bound.
    trust_region_tolerances : tuple of float, default (1e-3, 1e-1)
        Tolerances ``(target, maximum)`` on the linearisation error, measured
        as the residual of the exact voltage law of the new iterate relative to
        the voltage drop the branches of the cycle cause at their rated
        capacity. Below ``target`` the linear model described the step well and
        the radius is expanded if it was binding, above ``maximum`` the step is
        rejected and the radius is shrunk. The radius is also shrunk if a step
        did not reduce the fixed-point residual.
    trust_region_factors : tuple of float, default (0.5, 2.0)
        Factors ``(shrink, expand)`` the trust region radius is multiplied
        with.
    sensitivity_tolerance : float, default 1e-6
        Relative size below which the linearisation of ``method='trust_region'``
        drops a branch sensitivity, measured against the voltage drop that
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
    methods = ("fixed_point", "trust_region")
    if method not in methods:
        raise ValueError(f"method must be one of {methods}, got {method!r}.")
    if msq_threshold is not None:
        logger.warning(
            "'msq_threshold' is ignored, convergence is decided by the relative "
            "change of the system cost ('cost_threshold'). The relative change "
            "of the capacities remains available as 'step' in n.iteration_log."
        )
    cost_threshold = float(cost_threshold)
    cost_window = int(cost_window)
    proximal = bool(proximal)
    if cost_threshold < 0.0:
        raise ValueError("cost_threshold must be >= 0.")
    if cost_window < 1:
        raise ValueError("cost_window must be >= 1.")
    sensitivity_tolerance = float(sensitivity_tolerance)
    if not 0.0 <= sensitivity_tolerance < 1.0:
        raise ValueError("sensitivity_tolerance must be in [0, 1).")

    if snapshots is None:
        snapshots = n.snapshots
    snapshots = as_index(n, snapshots, "snapshots", "snapshot")

    branch_components = [
        c for c in ("Line", "LineX") if c in n.components and not n.df(c).empty
    ]
    if not branch_components:
        return n.optimize(snapshots, **kwargs)

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

    def collect_branch_caps(attr: str) -> pd.Series:
        return pd.concat(
            {c: n.df(c)[attr] for c in branch_components},
            names=["component", "name"],
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

    def save_optimal_capacities(network: Network, iteration: int, status: str) -> None:
        for c, attr in pd.Series(nominal_attrs)[list(network.branch_components)].items():
            network.df(c)[f"{attr}_opt_{iteration}"] = network.df(c)[f"{attr}_opt"]
        if "LineX" in network.components and not network.line_xs.empty:
            network.line_xs[f"sssc_nom_opt_{iteration}"] = network.line_xs["sssc_nom_opt"]
        setattr(network, f"status_{iteration}", status)
        setattr(network, f"objective_{iteration}", network.objective)
        network.iteration = iteration
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
        fallback = float(weights.mean()) if weights.notna().any() else 1.0
        proximal_costs[c] = weights.fillna(fallback)

    # cost of the capacity that is already installed, which the objective of
    # every iteration carries along. It is frozen at the value of the first
    # iteration, where the capacities still are the initial ones.
    installed_cost = [0.0]

    move_costs = (
        pd.concat(proximal_costs, names=["component", "name"]).reindex(ext_branches)
        if branch_components
        else pd.Series(dtype=float)
    )

    def capex_weighted_move(caps: pd.Series, center: pd.Series) -> float:
        """
        Capital cost of the capacity that a step moves, which is the quantity
        the proximal term charges for.
        """
        moved = 0.0
        for c in branch_components:
            ext_i = branch_data[c]["ext_i"]
            if ext_i.empty:
                continue
            distance = (
                caps.xs(c, level="component").reindex(ext_i)
                - center.xs(c, level="component").reindex(ext_i)
            ).abs()
            moved += float((proximal_costs[c] * distance).sum())
        return moved

    def calibrate_proximal(move: float, cost: float) -> float:
        """
        Weight for which a move of the size of the previous one costs as much
        as the cost difference the iteration considers converged, so that only
        moves worth more than the convergence tolerance are taken. Bounded to
        keep the induced bias below a tenth of the moved capital cost.
        """
        if not proximal:
            return 0.0
        lower, upper = 1e-6, 1e-1
        if move <= 0.0 or not np.isfinite(cost) or cost == 0.0:
            return upper
        target = PROXIMAL_CALIBRATION * cost_threshold * abs(cost)
        return float(np.clip(target / move, lower, upper))

    def add_proximal_penalty(
        network: Network, center: pd.Series, weight: float
    ) -> None:
        """
        Penalise moving the branch capacities away from the previous iterate.

        The absolute value is modelled with a non-negative deviation variable
        ``d_l >= |F_l - F_l'|``, which the penalty drives onto the bound.
        """
        m = network.model
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

    def extra_functionality_for(center: pd.Series | None, weight: float) -> Any:
        def extra_functionality(network: Network, sns: pd.Index) -> None:
            if weight > 0.0 and center is not None:
                add_proximal_penalty(network, center, weight)
            if user_extra_functionality is not None:
                user_extra_functionality(network, sns)

        return extra_functionality

    def solved_cost(
        network: Network, caps: pd.Series, center: pd.Series | None, weight: float
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
        if weight > 0.0 and center is not None:
            value -= weight * capex_weighted_move(caps, center)
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

    def evaluate_kvl(network: Network) -> tuple[pd.DataFrame, float, float]:
        """
        Evaluate the voltage law for the solution stored in the network.

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

        q_sssc = None
        if "LineX" in passive_components and "q_sssc" in network.line_xs_t:
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
                    c: network.pnl(c)["p0"].reindex(index=period_sns)
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
                C_dense = C.toarray()
                residual_norm += float(np.abs(terms @ C_dense).sum())
                rated = np.abs(weightings * s_nom_def.reindex(branches_i).to_numpy())
                term_norm += len(period_sns) * float(
                    (rated @ np.abs(C_dense)).sum()
                )
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

    def trust_region_widths(center: pd.Series, radius: float) -> pd.Series:
        """
        Half width of the trust region per branch. The radius is relative to
        the capacity of the linearisation point, so that the region stays
        meaningful once the capacities have grown well beyond their initial
        value, and is floored by the initial capacity, so that it does not
        collapse for branches shrinking towards zero.
        """
        scale = np.maximum(center.abs(), radius_scale)
        return radius * scale

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
        change = (current - previous).loc[ext_branches].abs()
        widths = trust_region_widths(previous, radius).loc[ext_branches]
        at_boundary = (change / widths.clip(lower=1e-12)) >= 0.9
        moved = move_costs * change
        total = float(moved.sum())
        if total <= 0.0:
            return False
        return float(moved[at_boundary].sum()) / total >= 0.1

    def set_trust_region(network: Network, center: pd.Series, radius: float) -> None:
        widths = trust_region_widths(center, radius)
        for c in branch_components:
            ext_i = branch_data[c]["ext_i"]
            if ext_i.empty:
                continue
            attr = nominal_attrs[c]
            lower, upper = original_nominal_bounds[c]
            middle = center.xs(c, level="component").reindex(ext_i)
            delta = widths.xs(c, level="component").reindex(ext_i)
            network.df(c).loc[ext_i, f"{attr}_min"] = np.maximum(
                lower.reindex(ext_i), middle - delta
            )
            network.df(c).loc[ext_i, f"{attr}_max"] = np.minimum(
                upper.reindex(ext_i), middle + delta
            )

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
    error_target, error_max = (float(t) for t in trust_region_tolerances)
    shrink, expand = (float(f) for f in trust_region_factors)
    if not 0.0 < radius_min <= radius_max:
        raise ValueError("trust_region_bounds must satisfy 0 < lower <= upper.")
    if not 0.0 <= error_target <= error_max:
        raise ValueError(
            "trust_region_tolerances must satisfy 0 <= target <= maximum."
        )
    if not 0.0 < shrink <= 1.0 <= expand:
        raise ValueError(
            "trust_region_factors must satisfy 0 < shrink <= 1 <= expand."
        )
    radius = float(np.clip(trust_region_initial, radius_min, radius_max))

    current_def = initial_caps.copy()
    cost_history: list[float] = []
    proximal_delta = 0.0
    reference_terms: pd.DataFrame | None = None
    previous_step: float | None = None
    plain_step = False
    has_converged = False
    iteration = 1
    status = "ok"
    condition = "optimal"
    records: list[dict[str, Any]] = []

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
            if (
                method == "trust_region"
                and reference_terms is not None
                and not plain_step
            ):
                sensitivity = capacity_sensitivity(reference_terms, current_def)
            plain_step = False
            n._kvl_capacity_sensitivity = sensitivity
            if sensitivity is None:
                reset_trust_region(n)
            else:
                set_trust_region(n, current_def, radius)

            # anchor of the proximal term, absent in the first iteration where
            # no previous iterate exists
            anchor = current_def if iteration > 1 else None
            weight = proximal_delta if anchor is not None else 0.0
            status, condition = n.optimize(
                snapshots,
                extra_functionality=extra_functionality_for(anchor, weight),
                **kwargs,
            )
            if status != "ok":
                if sensitivity is not None:
                    # the linearised problem can be infeasible where the frozen
                    # impedances require more capacity than the trust region
                    # admits. Fall back to an unrestricted fixed-point step,
                    # which is always feasible, and be more careful afterwards.
                    radius = max(radius * shrink, radius_min)
                    plain_step = True
                    logger.warning(
                        "Iteration %d failed with status %s/%s, falling back to a "
                        "fixed-point step and shrinking the trust region radius "
                        "to %.3e.",
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

            caps = collect_branch_caps("s_nom_opt")
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
                    radius = max(radius * shrink, radius_min)
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
                    iteration += 1
                    continue

            if iteration == 1:
                installed_cost[0] = float(getattr(n, "objective_constant", 0.0))
            cost = solved_cost(n, caps, anchor, weight)
            move = capex_weighted_move(caps, current_def)
            diff = relative_capacity_change(caps, current_def, initial_caps)

            if track_iterations:
                save_optimal_capacities(n, iteration, status)

            # residual of the exact voltage law at the capacities of the new
            # iterate, which is the error of the linear model solved above
            update_line_params(n, caps)
            cycle_terms, violation, violation_rel = evaluate_kvl(n)

            # the linear model predicted a vanishing residual, so the residual
            # left over is the error the trust region has to control. A step
            # small enough to converge is always accepted: the linearisation
            # then reproduces its own point, where the constraint coincides
            # with the exact one.
            converged = (
                cost_converged(cost_history, cost) and iteration >= min_iterations
            )
            accepted = True
            binding = trust_region_binding(caps, current_def, radius)
            if sensitivity is not None and not converged:
                if violation_rel > error_max:
                    # the linear model did not describe the step
                    accepted = False
                    radius = max(radius * shrink, radius_min)
                elif previous_step is not None and diff >= previous_step:
                    # the step did not reduce the fixed-point residual: the
                    # linear model is exploited over too wide a range, which a
                    # small residual of the voltage law alone does not reveal
                    radius = max(radius * shrink, radius_min)
                elif violation_rel <= error_target and binding:
                    radius = min(radius * expand, radius_max)

            cost_change = (
                abs(cost - cost_history[-1]) / max(abs(cost), 1e-12)
                if cost_history
                else np.nan
            )
            logger.info(
                "Iteration %d: relative cost change = %.3e, relative capacity "
                "change = %.3e, relative KVL residual = %.3e%s",
                iteration,
                cost_change,
                diff,
                violation_rel,
                ""
                if sensitivity is None
                else f", radius = {radius:.3e}, "
                f"{'binding' if binding else 'not binding'}, step "
                f"{'accepted' if accepted else 'rejected'}",
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
                    "radius": radius if sensitivity is not None else np.nan,
                    "binding": binding if sensitivity is not None else False,
                    "proximal": weight,
                    "accepted": accepted,
                }
            )

            if not accepted:
                if radius <= radius_min:
                    logger.warning(
                        "Trust region radius reached its lower bound %.3e without "
                        "resolving the linearisation error. Stopping ...",
                        radius_min,
                    )
                    break
                iteration += 1
                continue

            if converged:
                current_def = clip_branch_caps(
                    caps.copy(), branch_cap_min, branch_cap_max
                )
                if sensitivity is not None and binding:
                    logger.warning(
                        "The capacities converged against the boundary of the "
                        "trust region of radius %.3e. The iterate is consistent "
                        "but may not be a local optimum, consider a larger "
                        "'trust_region_bounds'.",
                        radius,
                    )
                # linearise the final solve at the converged point, where the
                # linearisation is exact
                reference_terms = cycle_terms
                cost_history.append(cost)
                has_converged = True
                break

            previous_step = diff
            cost_history.append(cost)
            proximal_delta = calibrate_proximal(move, cost)
            if method == "trust_region":
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

    logger.info(
        "Preparing final iteration with updated transmission parameters and extendable transmission capacities."
    )
    last_success_status, last_success_condition = status, condition

    update_line_params(n, current_def)

    # The final solve of the fixed-point iteration is the relaxation with frozen
    # impedances. Its optimum can undercut the converged one by expanding
    # capacity without paying for the flow redistribution the lower impedance
    # causes. For a converged trust region iteration the linearisation is exact
    # at that point, so keeping it reports the solution actually converged to.
    final_sensitivity = None
    if method == "trust_region" and has_converged and reference_terms is not None:
        final_sensitivity = capacity_sensitivity(reference_terms, current_def)

    try:
        n._kvl_capacity_sensitivity = final_sensitivity
        if final_sensitivity is None:
            reset_trust_region(n)
        else:
            set_trust_region(n, current_def, radius)
        n.calculate_dependent_values()
        status, condition = n.optimize(
            snapshots,
            extra_functionality=extra_functionality_for(
                current_def, proximal_delta
            ),
            **kwargs,
        )
    finally:
        n._kvl_capacity_sensitivity = None
        reset_trust_region(n)

    if status == "ok":
        return status, condition
    else:
        logger.warning(
            "Final rerun with updated transmission parameters failed with status %s/%s. "
            "Keeping the last successful loop solution.",
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
