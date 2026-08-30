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

# Range the adapted proximal weight is clipped to, per norm. The weight is the
# share of the capital cost of a branch the term charges for moving it by its
# own size, so it is dimensionless and comparable between the two norms - but
# what a given share does to the solution is not, and the upper bounds differ
# by three decades because of it.
#
# ``l1`` has a dead zone: at the anchor its subgradient is ``delta * c_l``
# times ``[-1, 1]``, so no move worth less than that per MW is taken at all,
# and the converged point is only optimal to within it. Measured on the meshed
# brownfield system over five decades of the weight, the bias of the converged
# cost stays below 1e-4 relative up to ``1e-1`` and then jumps to between two
# and eleven per cent at ``1``, while the iteration count *falls* - the term
# freezes the iterate and the convergence test reads the freeze as
# convergence. ``1e-1`` is therefore the last decade in which the term is a
# perturbation of the objective rather than what decides the plan, and it is a
# hard ceiling, not a tuning range: an ``l1`` term cannot be relied on as the
# only step control, since the weight it would need to control a large step is
# past the point where it distorts the answer.
#
# ``l2`` has no dead zone - its gradient vanishes at the anchor, so a converged
# point satisfies the unpenalised optimality conditions exactly.  Its default
# is the calibrated unit weight; callers that want the former adaptive policy
# can supply wider ``proximal_bounds`` explicitly.
PROXIMAL_BOUNDS: dict[str, tuple[float, float]] = {
    "l1": (1e-6, 1e-1),
    "l2": (0.5, 0.5),
}

# ``l2`` is deliberately kept at its calibrated unit weight by default.  A
# caller can still request adaptive behaviour explicitly with wider
# ``proximal_bounds``.  The ``l1`` term retains its historical adaptive range.
PROXIMAL_INITIAL: dict[str, float] = {"l1": 1e-3, "l2": 0.5}

# At a proximal weight of 0.5, the L2 cross term cancels the ordinary linear
# capacity cost exactly. Floating-point division by and multiplication with
# the anchor can leave a coefficient around 1e-13. Such a residual has no
# useful effect on a QP solution but needlessly widens the objective range.
PROXIMAL_L2_MIN_LINEAR_COEFFICIENT = 1e-6


@dataclass
class TransmissionIterationResult:
    """Minimal solution data required by one transmission-expansion step."""

    capacities: pd.Series
    flows: dict[str, pd.DataFrame]
    q_sssc: pd.DataFrame | None
    objective: float
    tracked_capacities: dict[str, pd.Series] | None = None
    sssc_nom: pd.Series | None = None


def optimize_transmission_expansion_iteratively(
    n: Network,
    snapshots: Sequence | None = None,
    msq_threshold: float | None = None,
    min_iterations: int = 1,
    max_iterations: int = 100,
    track_iterations: bool = False,
    scheme: str = "slp",
    trust_region: bool = True,
    proximal: str = "l2",
    cost_threshold: float = 1e-5,
    cost_window: int = 1,
    proximal_metric: str = "capex",
    proximal_initial: float | None = None,
    proximal_bounds: tuple[float, float] | None = None,
    trust_region_initial: float = 1.0,
    trust_region_bounds: tuple[float, float] = (1e-2, 1.0),
    trust_region_tolerances: tuple[float, float] = (1e-5, 1e-2),
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
    are resolved by an outer iteration, whose scheme and step control are set
    by ``scheme``, ``trust_region`` and ``proximal``.

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
        results, duals and derived network time series are assigned from the
        iterate at which the loop converges; if it stops without converging
        (``max_iterations`` exhausted), they come from one further solve at the
        last capacities of the iteration instead, since no solved iterate is
        left to reuse in that case.
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
            is exact at that point, and the converged iterate's own solve is
            completed and returned directly, with no further re-solve.

        A linearisation is only valid over a limited step, which is what
        ``trust_region`` and ``proximal`` are for. Every solved iterate is
        accepted however large the residual it leaves; the controls only narrow
        the *next* step. With both off, ``'slp'`` is a bare Gauss-Newton
        iteration, and on a meshed network it oscillates instead of
        converging.
    trust_region : bool, default True
        Restrict the capacities of each inner problem to a box around the
        previous iterate, of the relative width given by ``trust_region_*``.
        The radius is adapted from the exact KVL residual the new iterate
        leaves, which the linear model predicted to vanish, and from the
        progress of the fixed-point residual; after a step whose residual
        exceeds ``trust_region_tolerances[1]`` the radius is shrunk, but the
        step itself is kept and the next one is taken from it.

        This is the only control that bounds the step *hard*: it is the
        capacities themselves that are restricted, so the inner problem cannot
        return a longer step whatever its objective would gain from one. The
        response to a large residual is driven by the residual, not by this
        control: a proximal term alone narrows the next step just as well with
        a heavier weight.

        Its weakness is that a box cannot distinguish a step that is too long
        from a step in a bad direction: where the optimal plan is many times the
        brownfield capacity, it answers the degeneracy between equal-cost plans
        by shrinking the radius until the iteration crawls, and can be left far
        from the optimum when the iteration budget runs out. That is a failure
        mode of the box *alone*; a proximal term removes it, since it prices the
        direction rather than only the length.

        Under ``scheme='fixed_point'`` the box is applied as well, from the
        second iteration onward, where it damps the fixed-point step; there is
        no linearisation error to adapt on there, so the radius is only adapted
        from the progress of the residual.

        On by default, but on the measured grid below it is *inert* next to the
        default ``proximal='l2'``: with the term active, on and off agree to
        2.4e-6 in cost on 17 of the 18 instances and agree exactly in
        iterations and convergence. It is kept on because the bound is the only
        hard one and because it costs nothing measurable; switching it off is a
        defensible choice.
    proximal : {'off', 'l1', 'l2'}, default 'l2'
        Add a proximal term to the objective which penalises moving the
        capacity of branch ``l`` away from the previous iterate ``F_l'``, in
        units of its capital cost ``c_l``:

        ``'l1'``
            ``delta * sum_l c_l |F_l - F_l'|``, modelled with a non-negative
            deviation variable per extendable branch, which keeps the inner
            problem a linear program. Its subgradient does not vanish at the
            anchor, so it has a dead zone: a move worth less than ``delta *
            c_l`` per MW is not taken at all. That is what resolves the
            degeneracy between equal-cost plans, but it also means the
            converged point is only optimal to within the dead zone, and that a
            weight large enough to control a long step is already large enough
            to decide the plan - see ``PROXIMAL_BOUNDS``. It also fabricates
            convergence evidence: a frozen iterate reproduces its own
            capacities, so the fixed-point residual and the KVL residual both
            read zero at a point that is not the optimum.
        ``'l2'``
            ``delta * sum_l c_l (F_l - F_l')**2 / F_l'``, added to the objective
            directly, which needs neither variable nor constraint but makes the
            inner problem a quadratic program - with a diagonal, positive
            semi-definite Hessian on the capacities alone, so it stays convex
            and cheap. The division by the anchor makes the term ``delta *
            sum_l (c_l F_l') * u_l**2`` in the *relative* deviation ``u_l``,
            which is the metric the linearisation error lives in and the one
            the trust region is symmetric in, and it gives ``delta`` the same
            meaning and scale as under ``'l1'``: the share of the capital cost
            of a branch charged for moving it by its own size. Its gradient
            vanishes at the anchor, so there is no dead zone - the term cannot
            freeze the iteration, and a converged point satisfies the
            unpenalised optimality conditions exactly - at the price of damping
            less per unit of bias, which is why its weight is allowed three
            decades further.

        The term vanishes at the fixed point, is excluded from the reported
        system cost, applies to both schemes, and is active from the second
        iteration onward, where a previous iterate exists. Its weight is
        adapted like a trust region radius and reported per iteration in
        ``n.iteration_log``: a quadratic penalty of weight ``delta`` admits a
        relative step of order ``1 / delta``, so raising the weight is the same
        move as shrinking the radius, and the two controls can be used together
        or on their own.

        Whichever norm is used, the suboptimality the term can introduce is
        bounded by its own value at the unpenalised optimum: with ``F*`` that
        optimum and ``F@`` the penalised one, ``J(F@) <= J(F@) + P(F@) <=
        J(F*) + P(F*)``, so ``J(F@) - J(F*) <= P(F*)``. That bound is what
        ``PROXIMAL_BOUNDS`` keeps small.

        ``'l2'`` by default, which is the switch the defaults turn on. Measured
        over the full grid of the three switches on the meshed test system of
        ``test/test_lopf_iteratively.py`` - 18 instances, SSSC on/off times
        three brownfield capacities times three capital costs, 40 iterations -
        as the system cost in excess of the best plan any of the twelve
        configurations found on the same instance::

            scheme  trust_region  proximal   converged  mean exc.  worst exc.
            slp     off           off            16/18     1.19 %     21.37 %
            slp     off           l1              6/18     5.65 %     24.57 %
            slp     off           l2             16/18     0.12 %      1.64 %
            slp     on            off            15/18     1.71 %     21.01 %
            slp     on            l1             16/18     0.55 %      5.52 %
            slp     on            l2             16/18     0.12 %      1.64 %
            fixed_point, any control             13/18  4.8-5.0 %     26.26 %

        The term in its ``'l2'`` norm is the control that carries that result.
        It removes the failures in which the iteration walks away from the
        optimum and stops there: on the hardest instance the box alone returns a
        plan certified 21.0 % above the global optimum, with a KVL residual of
        1e-8 and so with no indication in any diagnostic, where ``'l2'``
        converges to within 3e-4 %. What it costs is about five iterations on
        average, and on two instances a small regression - on one of them bare
        ``'slp'`` converges in 7 iterations onto the certified optimum where
        ``'l2'`` exhausts the budget 1.64 % above it. ``'l1'`` alone is worse
        than no control at all, since its dead zone freezes the iterate and the
        freeze reports itself as convergence; it only becomes usable behind the
        box.
    cost_threshold : float, default 1e-5
        Convergence criterion: the iteration stops once the system cost has
        changed by less than this fraction of itself in each of the last
        ``cost_window`` accepted iterations. The cost is invariant under the
        exchange of degenerate alternative optima, which the change of the
        capacities is not, and it is the quantity the results are reported in.
    cost_window : int, default 2
        Number of consecutive relative cost changes that have to undercut
        ``cost_threshold``.
    proximal_metric : {'capex', 'uniform'}, default 'capex'
        Weight ``c_l`` the move of branch ``l`` is charged with: its capital
        cost, or the mean capital cost for every branch, which charges the
        capacity moved rather than its cost. ``'capex'`` perturbs every branch
        by the same *share* of its own cost and keeps ``delta`` dimensionless;
        ``'uniform'`` perturbs branch ``l`` by ``delta / c_l`` of its cost, so
        the distortion falls on the cheapest branches - the ones the expansion
        wants to use - and grows without bound as ``c_l`` falls. It is provided
        for comparison, not as a recommendation.
    proximal_initial : float, optional
        Initial weight of the proximal term. Defaults to
        ``PROXIMAL_INITIAL`` of the chosen norm and is clipped into the
        admissible range. The default L2 bounds fix this value at 0.5; pass
        wider ``proximal_bounds`` to make it adaptive.
    proximal_bounds : tuple of float, optional
        Smallest and largest admissible weight of the proximal term. Defaults
        to ``PROXIMAL_BOUNDS`` of the chosen norm. The default L2 bounds are
        ``(0.5, 0.5)``, so its weight is fixed. An adaptive weight saturates
        at its upper bound; the iteration then continues at that weight until
        it converges or ``max_iterations`` runs out.
    trust_region_initial : float, default 0.5
        Initial trust region radius. The capacity of branch ``l`` is restricted
        to

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
    trust_region_tolerances : tuple of float, default (1e-4, 1e-2)
        Tolerances ``(target, maximum)`` on the linearisation error, measured
        as the residual of the exact voltage law of the new iterate relative to
        the voltage drop the branches of the cycle cause at their rated
        capacity. Below ``target`` the linear model described the step well and
        the radius is expanded if it was binding, above ``maximum`` the radius
        is shrunk. The step itself is kept either way - a solved iterate is
        never discarded. The radius is also shrunk if a step did not reduce the
        fixed-point residual.
    trust_region_factors : tuple of float, default (0.5, 2.0)
        Factors ``(shrink, expand)`` the trust region radius is multiplied
        with.
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
    # ``**kwargs`` is forwarded to ``n.optimize``, so a removed argument would
    # otherwise be swallowed there and the run would silently use the defaults
    # instead of what the caller asked for. These are not accepted in any form;
    # they are named only to say what replaced them.
    removed = {
        "method": (
            "pass the scheme and the step controls separately: "
            "method='fixed_point' is scheme='fixed_point' with "
            "trust_region=False, proximal='off'; method='trust_region' is "
            "scheme='slp', trust_region=True, proximal='off'; "
            "method='proximal' is scheme='slp', trust_region=False, "
            "proximal='l1'"
        ),
        "proximal_norm": "pass the norm as proximal='l1' or proximal='l2'",
    }
    for name, replacement in removed.items():
        if name in kwargs:
            raise TypeError(
                f"{name!r} has been removed from "
                f"optimize_transmission_expansion_iteratively: {replacement}."
            )

    trust_region = bool(trust_region)

    schemes = ("slp", "fixed_point")
    if scheme not in schemes:
        raise ValueError(f"scheme must be one of {schemes}, got {scheme!r}.")
    proximal_options = ("off", "l1", "l2")
    if proximal not in proximal_options:
        raise ValueError(
            f"proximal must be one of {proximal_options}, got {proximal!r}."
        )
    if msq_threshold is not None:
        logger.warning(
            "'msq_threshold' is ignored, convergence is decided by the relative "
            "change of the system cost ('cost_threshold'). The relative change "
            "of the capacities remains available as 'step' in n.iteration_log."
        )
    cost_threshold = float(cost_threshold)
    cost_window = int(cost_window)
    # the norm doubles as the switch; ``proximal_norm`` is what the penalty
    # builders read, and is meaningless when the term is off
    proximal_norm = proximal if proximal != "off" else "l1"
    proximal_on = proximal != "off"
    if scheme == "slp" and not (trust_region or proximal_on):
        logger.warning(
            "scheme='slp' without a step control: the linearisation is trusted "
            "over an unbounded step, and every step is taken however large its "
            "KVL residual. Expect the "
            "iteration to oscillate rather than converge on a meshed network."
        )
    if proximal == "l1" and not trust_region:
        logger.warning(
            "proximal='l1' is the only step control: its weight is capped at "
            "%.1e, above which the dead zone of the term decides the plan "
            "instead of damping the iteration, so it may not be able to "
            "control the step. Consider proximal='l2' or trust_region=True.",
            PROXIMAL_BOUNDS["l1"][1],
        )
    metrics = ("capex", "uniform")
    if proximal_metric not in metrics:
        raise ValueError(
            f"proximal_metric must be one of {metrics}, got {proximal_metric!r}."
        )
    if cost_threshold < 0.0:
        raise ValueError("cost_threshold must be >= 0.")
    if cost_window < 1:
        raise ValueError("cost_window must be >= 1.")
    if proximal_bounds is None:
        proximal_bounds = PROXIMAL_BOUNDS[proximal_norm]
    delta_min, delta_max = (float(b) for b in proximal_bounds)
    if not 0.0 < delta_min <= delta_max:
        raise ValueError("proximal_bounds must satisfy 0 < lower <= upper.")
    if proximal_initial is None:
        proximal_initial = PROXIMAL_INITIAL[proximal_norm]
    sensitivity_tolerance = float(sensitivity_tolerance)
    if not 0.0 <= sensitivity_tolerance < 1.0:
        raise ValueError("sensitivity_tolerance must be in [0, 1).")

    if snapshots is None:
        snapshots = n.snapshots
    snapshots = as_index(n, snapshots, "snapshots", "snapshot")

    # imported here rather than at module level to avoid a circular import
    # with pypsa.optimization.optimize, which imports this module
    from pypsa.optimization.optimize import assign_duals, assign_solution, post_processing

    assign_all_duals = bool(kwargs.get("assign_all_duals", False))

    def solve_inner(
        inner_snapshots: Sequence,
        lightweight: bool = False,
        **inner_kwargs: Any,
    ) -> tuple[str, str]:
        solve_kwargs = {**kwargs, **inner_kwargs}
        if lightweight:
            return n.optimize(
                inner_snapshots,
                _solution_variables=iteration_solution_variables(),
                _assign_solution=False,
                _assign_duals=False,
                _post_processing=False,
                **solve_kwargs,
            )
        return n.optimize(inner_snapshots, **solve_kwargs)

    branch_components = [
        c for c in ("Line", "LineX") if c in n.components and not n.df(c).empty
    ]
    if not branch_components:
        return solve_inner(snapshots)

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
            if "LineX" in branch_components:
                names.append("LineX-sssc_nom")
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
            if track_iterations and "LineX-sssc_nom" in m.variables
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
        fallback = float(weights.mean()) if weights.notna().any() else 1.0
        weights = weights.fillna(fallback)
        if proximal_metric == "uniform":
            # penalise the capacity itself rather than its cost; the mean
            # capital cost is kept as the unit so that ``delta`` stays
            # dimensionless and its calibration remains comparable
            weights = pd.Series(fallback, index=ext_i, dtype=float)
        proximal_costs[c] = weights

    # cost of the capacity that is already installed, which the objective of
    # every iteration carries along. It is frozen at the value of the first
    # iteration, where the capacities still are the initial ones.
    installed_cost = [0.0]

    move_costs = (
        pd.concat(proximal_costs, names=["component", "name"]).reindex(ext_branches)
        if branch_components
        else pd.Series(dtype=float)
    )

    def proximal_scales(center: pd.Series) -> dict[str, pd.Series]:
        """
        Capacity scale the quadratic term is measured against, which is the
        anchor itself, floored to stay away from a vanishing capacity.

        With that scale the term reads ``delta * sum_l c_l F_l' * u_l**2`` in
        the relative deviation ``u_l = (F_l - F_l') / F_l'``, i.e. it charges
        ``delta`` times the capital cost of branch ``l`` for a move of the size
        of the branch itself, which is exactly what the ``l1`` term charges for
        the same move. Both norms are therefore calibrated on the same scale
        and ``delta`` is dimensionless in both.
        """
        values = center.reindex(ext_branches).to_numpy(dtype=float)
        reference = float(np.nanmax(np.abs(values))) if values.size else 0.0
        floor = max(1e-3 * reference, 1e-6)
        return {
            c: center.xs(c, level="component")
            .reindex(branch_data[c]["ext_i"])
            .astype(float)
            .clip(lower=floor)
            for c in branch_components
        }

    def proximal_move(caps: pd.Series, center: pd.Series) -> float:
        """
        Size of a step in the metric the proximal term charges for: the capital
        cost of the capacity moved for ``l1``, and the same quantity weighted
        with the relative size of the move for ``l2``.
        """
        scales = proximal_scales(center) if proximal_norm == "l2" else {}
        moved = 0.0
        for c in branch_components:
            ext_i = branch_data[c]["ext_i"]
            if ext_i.empty:
                continue
            distance = (
                caps.xs(c, level="component").reindex(ext_i)
                - center.xs(c, level="component").reindex(ext_i)
            ).abs()
            if proximal_norm == "l2":
                moved += float((proximal_costs[c] * distance**2 / scales[c]).sum())
            else:
                moved += float((proximal_costs[c] * distance).sum())
        return moved

    def l2_anchor(anchor: pd.Series, quadratic: pd.Series) -> pd.Series:
        """Replace a numerically-zero L2 anchor by its exact zero value."""
        linear = 2.0 * quadratic * anchor
        return anchor.where(
            linear.abs() >= PROXIMAL_L2_MIN_LINEAR_COEFFICIENT,
            0.0,
        )

    def proximal_objective_value(
        caps: pd.Series, center: pd.Series, penalty_weight: float
    ) -> float:
        """
        Value the proximal term contributes to the objective at ``caps``, which
        is what has to be removed again from the reported system cost.

        For ``l2`` this is not the penalty itself: the constant of the expanded
        square is left out of the objective, since linopy rejects constants
        there. It cancels in the reported cost because it is left out here as
        well.
        """
        if proximal_norm != "l2":
            return proximal_move(caps, center)
        scales = proximal_scales(center)
        value = 0.0
        for c in branch_components:
            ext_i = branch_data[c]["ext_i"]
            if ext_i.empty:
                continue
            anchor = center.xs(c, level="component").reindex(ext_i)
            capacity = caps.xs(c, level="component").reindex(ext_i)
            weight = proximal_costs[c] / scales[c]
            anchor = l2_anchor(anchor, penalty_weight * weight)
            value += float((weight * (capacity**2 - 2.0 * anchor * capacity)).sum())
        return value

    def add_proximal_penalty(
        network: Network, center: pd.Series, weight: float
    ) -> None:
        """
        Penalise moving the branch capacities away from the previous iterate.

        For ``proximal_norm='l1'`` the absolute value is modelled with a
        non-negative deviation variable ``d_l >= |F_l - F_l'|``, which the
        penalty drives onto the bound, and the model stays a linear program.
        For ``'l2'`` the square enters the objective directly, which needs
        neither variable nor constraint but makes the model a quadratic program
        with a diagonal, positive semi-definite Hessian on the capacities. Its
        constant is dropped, see ``proximal_objective_value``.
        """
        m = network.model
        scales = proximal_scales(center) if proximal_norm == "l2" else {}
        for c in branch_components:
            ext_i = branch_data[c]["ext_i"]
            attr = nominal_attrs[c]
            if ext_i.empty or f"{c}-{attr}" not in m.variables:
                continue
            capacity = m[f"{c}-{attr}"]
            anchor = center.xs(c, level="component").reindex(ext_i)
            if proximal_norm == "l2":
                quadratic = weight * proximal_costs[c] / scales[c]
                anchor = l2_anchor(anchor, quadratic)
                m.objective = (
                    m.objective
                    + (capacity * capacity * quadratic).sum()
                    - (capacity * (2.0 * quadratic * anchor)).sum()
                )
                continue
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

    def drop_tiny_l2_linear_objective_terms(network: Network) -> None:
        """Remove numerically irrelevant net linear terms left in an L2 QP.

        With the normal scale ``F'``, a proximal weight of 0.5 makes the L2
        cross term exactly cancel the ordinary capacity cost. Linopy aggregates
        the two summands only while exporting, where roundoff can leave a
        coefficient around 1e-13. Aggregate by variable here and zero a tiny
        *net* coefficient before export. The diagonal L2 terms remain intact
        and continue to provide the proximal curvature.
        """
        objective = network.model.objective
        labels = objective.vars.data
        coefficients = objective.coeffs.data
        linear = (labels[0] >= 0) & (labels[1] == -1)
        if not linear.any():
            return
        linear_labels = labels[0, linear]
        net = np.bincount(
            linear_labels,
            weights=coefficients[linear],
            minlength=int(linear_labels.max()) + 1,
        )
        tiny = linear & (
            np.abs(net[labels[0]]) < PROXIMAL_L2_MIN_LINEAR_COEFFICIENT
        )
        if tiny.any():
            coefficients[tiny] = 0.0

    def extra_functionality_for(center: pd.Series | None, weight: float) -> Any:
        def extra_functionality(network: Network, sns: pd.Index) -> None:
            if weight > 0.0 and center is not None:
                add_proximal_penalty(network, center, weight)
            if user_extra_functionality is not None:
                user_extra_functionality(network, sns)
            if proximal_norm == "l2" and weight > 0.0 and center is not None:
                drop_tiny_l2_linear_objective_terms(network)

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
            value -= weight * proximal_objective_value(caps, center, weight)
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
    linearised = scheme == "slp"

    current_def = initial_caps.copy()
    cost_history: list[float] = []
    proximal_delta = (
        float(np.clip(proximal_initial, delta_min, delta_max)) if proximal_on else 0.0
    )

    def tighten() -> None:
        """
        Restrict the step the model is trusted over, on every control that is
        switched on: shrink the trust region and raise the weight of the
        proximal term. The two are the same move - the relative step a
        quadratic penalty of weight ``delta`` admits goes as ``1 / delta`` - so
        they shrink by the same factor and can be used together or alone.
        """
        nonlocal radius, proximal_delta
        if trust_region:
            radius = max(radius * shrink, radius_min)
        if proximal_on:
            proximal_delta = float(min(proximal_delta / shrink, delta_max))

    def relax() -> None:
        """Widen the step the model is trusted over, on every active control."""
        nonlocal radius, proximal_delta
        if trust_region:
            radius = min(radius * expand, radius_max)
        if proximal_on:
            proximal_delta = float(max(proximal_delta / expand, delta_min))

    reference_terms: pd.DataFrame | None = None
    previous_step: float | None = None
    plain_step = False
    has_converged = False
    iteration = 1
    status = "ok"
    condition = "optimal"
    records: list[dict[str, Any]] = []
    # Intermediate iterations only need capacities, KVL flows and (where
    # present) SSSC compensation. Tracking adds only nominal-capacity columns
    # and objective scalars, not full primal/dual network results. The
    # converged iterate is completed and kept in place (see below), so only a
    # superseded iterate's model needs releasing here.

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
                trust_region
                and not plain_step_now
                and (sensitivity is not None or iteration > 1)
            )
            if boxed:
                set_trust_region(n, current_def, radius)
            else:
                reset_trust_region(n)

            # anchor of the proximal term, absent in the first iteration where
            # no previous iterate exists
            anchor = current_def if iteration > 1 else None
            weight = proximal_delta if anchor is not None else 0.0
            status, condition = solve_inner(
                snapshots,
                lightweight=True,
                extra_functionality=extra_functionality_for(anchor, weight),
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
                        "(radius %.3e, proximal weight %.3e).",
                        iteration,
                        status,
                        condition,
                        radius,
                        proximal_delta,
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
            cost = solved_cost(n, caps, anchor, weight)
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
            converged = (
                cost_converged(cost_history, cost) and iteration >= min_iterations
            )
            binding = boxed and trust_region_binding(caps, current_def, radius)
            # every solved iterate is kept; the controls only set how far the
            # model is trusted over the *next* step
            controlled = (trust_region or proximal_on) and not converged
            if controlled and sensitivity is not None and violation_rel > error_max:
                # the linear model did not describe the step
                tighten()
            elif controlled and previous_step is not None and diff >= previous_step:
                # the step did not reduce the fixed-point residual: the model is
                # exploited over too wide a range, which a small residual of the
                # voltage law alone does not reveal. This is the only signal
                # under the fixed-point scheme, which leaves no such residual.
                tighten()
            elif (
                controlled
                # The first step has no linearisation error estimate.  Keep
                # the configured initial proximal weight for the first
                # anchored solve rather than relaxing it on missing evidence.
                and sensitivity is not None
                and violation_rel <= error_target
                and (binding or proximal_on)
            ):
                # the proximal term is never at a boundary, so there is no
                # binding to wait for before widening
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
                if not (trust_region or proximal_on)
                else (
                    (f", radius = {radius:.3e}" if boxed else "")
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
                    "binding": binding,
                    "proximal": weight,
                    "accepted": True,
                }
            )

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
                # This iterate's own solve is the result: complete it in
                # place (for the Gurobi fast path, from the already-solved
                # model, with no second call to the solver) and assign it,
                # rather than discarding it and solving again at the same
                # point below.
                pending = getattr(n, "_pending_full_solve", None)
                if pending is not None:
                    pending.finish()
                    n._pending_full_solve = None
                assign_solution(n)
                assign_duals(n, assign_all_duals)
                post_processing(n)
                break

            previous_step = diff
            cost_history.append(cost)
            # the weight is a step control in its own right and has already been
            # updated by ``tighten`` / ``relax``
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
                "binding",
                "proximal",
                "accepted",
            ],
        ).set_index("iteration")

    if has_converged:
        # The converged iterate was already completed and assigned in place
        # (see the ``if converged:`` branch above), so its own solve is the
        # result: no further rerun is needed or run.
        return status, condition

    if hasattr(n, "model"):
        logger.info(
            "Deleting model instance `n.model` from previous run to reclaim memory."
        )
        del n.model
        gc.collect()

    # Reached only if the loop stopped without convergence (max_iterations
    # exhausted): there is no solved iterate left to reuse, since the last
    # attempted solve was itself discarded. Solve once more at the last
    # capacities of the iteration to obtain a fully populated network.
    logger.info(
        "Preparing final iteration with updated transmission parameters and extendable transmission capacities."
    )
    last_success_status, last_success_condition = status, condition

    update_line_params(n, current_def)

    try:
        n._kvl_capacity_sensitivity = None
        reset_trust_region(n)
        n.calculate_dependent_values()
        status, condition = solve_inner(
            snapshots,
            # No penalty: this solve is not a step of the iteration but the
            # report of one, so it has to return the true system cost at the
            # last accepted capacities. At a point the iteration did not
            # converge to the penalty does not vanish, so leaving it in would
            # both bias the reported plan and, where the norm is quadratic,
            # hand a QP to a solver that was only ever asked for an LP.
            extra_functionality=extra_functionality_for(None, 0.0),
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
