#!/usr/bin/env python3
"""
Build abstracted, extended optimisation problems from PyPSA networks with
Linopy.
"""

from __future__ import annotations

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

# Factor the proximal weight is multiplied by when ``proximal_adaptive`` sees
# the iteration turn. Doubling is the coarsest response that recovers a step
# which is too long by any margin, in a bounded number of iterations.
PROXIMAL_ADAPTIVE_FACTOR = 2.0

# Bar on the displacement measure below which ``proximal_adaptive`` reads the
# iteration as turning rather than walking.
#
# Measured on a 1250 bus case against the share of each step that actually
# closed the distance to the converged plan: while the iteration walks, that
# share runs 0.42 to 0.70 and the measure reads 0.77 to 0.94; once it stalls,
# the share collapses to 0.05 to 0.31 - individual steps ending *further* from
# the answer than they started - and the measure reads 0.46 to 0.68. The two
# bands meet rather than leaving a gap, so the bar is a boundary drawn inside a
# continuum, not a value picked out of an empty interval.
#
# What holds it down here rather than up against the stalling band is the
# iterate that immediately follows a doubling. Its step was solved under a
# different penalty from the one before it, so the measure compares two steps
# from two regimes and reads low for that reason alone: 0.48 to 0.65 on runs
# that were not turning at all, one of them closing 69 % of its own remaining
# distance. A bar above those readings would raise the weight again on the
# strength of an artefact, on every doubling.
PROXIMAL_TURN_BAR = 0.65

# A window of steps that moves less than this share of the capital of the plan
# itself is numerical noise rather than a direction. Below it the iterate is
# read as a fixed point, which is what keeps the rule from ratcheting the
# weight up on a run that has already stopped.
PROXIMAL_STEP_FLOOR = 1e-9


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
    min_iterations: int = 1,
    max_iterations: int = 100,
    track_iterations: bool = False,
    scheme: str = "slp",
    proximal: bool = True,
    cost_threshold: float = 1e-5,
    step_threshold: float = 0.01,
    proximal_weight: float = 0.5,
    proximal_adaptive: bool = True,
    proximal_ceiling: float = 4.0,
    sensitivity_tolerance: float = 1e-5,
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
            linear expansion problem. There is no linearisation error for a
            step restriction to control, so the step control is off and the
            ``proximal_*`` arguments are ignored: the step is the bare fixed
            point.
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
        the proximal term below is for, and it therefore applies to
        ``'slp'`` alone. Every solved iterate is accepted however large the
        residual it leaves; the term only narrows the *next* step. With it
        off, ``'slp'`` is a bare Gauss-Newton iteration, and on a meshed
        network it oscillates instead of converging - which is what
        ``'fixed_point'`` is, one order lower.
    cost_threshold : float, default 1e-5
        Convergence criterion on the system cost: the relative change of the
        cost from the previous accepted iterate has to be at or below this.
        The cost is invariant under the exchange of degenerate alternative
        optima, which the change of the capacities is not, and it is the
        quantity the results are reported in.

        It is not sufficient on its own, which is why ``step_threshold`` is
        tested with it: the cost is *stationary along the directions an orbit
        turns in*, so a run that has settled into a limit cycle rather than
        onto a fixed point passes this test while its plan is still moving.
    step_threshold : float, default 0.01
        Convergence criterion on the size of the step, tested together with
        ``cost_threshold`` and on the same iterate: the relative change of the
        transmission capacities has to be at or below this. ``0`` switches the
        test off, leaving the cost test alone.

        The step is the gap between the plan a solve returns and the
        capacities its own impedances were taken at, measured in the
        capital-cost weighted Euclidean norm and referred to the initial
        capacities,

        ``sqrt(sum_l c_l (s_nom_opt - _s_nom_def)**2 / sum_l c_l s_nom_0**2)``,

        so this bounds how much of the plan's *capital* is still undetermined
        when the run stops. It is weighted with the same capital cost the
        proximal term charges in, so the quantity that is damped and the
        quantity that is tested are the same one. It is what separates a fixed point
        from an orbit, and it is also the only signal that catches a run held
        together by its own damping: a heavy proximal weight shortens the step,
        the change of the cost and the KVL residual alike, so the cost test can
        be met by damping rather than by stationarity. A damped iterate is
        still moving; a converged one is not.

        Where the alternative optima are degenerate the step can floor above
        any useful bar - the cost is then fully determined while the plan is
        not - so loosen it, rather than waiting, on a system whose plan is
        inherently indeterminate.
    proximal : bool, default True
        Whether ``scheme='slp'`` damps its step with the proximal term on the
        extendable ``Line`` and ``LineX`` capacities, which are its only
        target. ``False`` switches the term off, as ``proximal_weight=0`` does.

        The term is

        ``delta * sum_l c_l (F_l - F_l')**2 / F_l'``,

        scaled by the previous capacity ``F_l'``. A scaled free deviation
        variable gives the QP identical diagonal quadratic coefficients and
        puts the branch-specific scaling in linear defining equalities.

        Its gradient vanishes at the anchor, which is the property the rest of
        the iteration is built on: a fixed point of the penalised step is a
        fixed point of the unpenalised problem, so the converged plan does not
        depend on the weight the run needed to reach it.
    proximal_weight : float, default 0.5
        Weight ``delta`` of the proximal term, used by ``scheme='slp'`` only,
        and fixed for the whole run. The term penalises moving the capacity of
        branch ``l`` away from the previous iterate ``F_l'`` in units of its
        capital cost ``c_l``: it charges ``delta`` times the branch's capital
        cost for moving it by its own size, so ``delta`` is dimensionless.

        Each system has a stability boundary below which the iteration turns
        in a limit cycle instead of contracting; it is a property of the system
        rather than of this default, so a run that oscillates wants a larger
        weight.

        Raising the weight is safe in a way that bounding the step is not,
        because a *fixed point* of the penalised step is a fixed point of the
        unpenalised problem - the gradient of the term vanishes at the anchor -
        so the converged plan does not depend on the weight the run needed to
        reach it. A point the iteration merely *stopped at* is a different
        matter: a heavier weight shortens every step like a damping factor, and
        with the step it shortens the change of the cost and the KVL residual
        too, so the convergence signals can be produced by the damping rather
        than by stationarity. Read ``step`` in ``n.iteration_log`` to tell the
        two apart - a damped iterate is still moving, a converged one is not.

        It doubles as a hurdle rate on capital reallocation: the subgradient at
        the anchor is ``delta * c_l`` times ``[-1, 1]``, so a branch is moved
        only where the move returns more than ``delta`` of the capital it
        shifts. That dead zone is what makes the term selective - a
        reallocation between two plans of equal cost gains nothing and is
        refused, while a move worth more than the hurdle is taken at full size.
        It is also how the term can go wrong: a large enough weight lets the
        dead zone decide the plan rather than damp the iteration, and a frozen
        iterate reports itself as a converged one.

        ``0`` switches the term off.

        Since ``proximal_adaptive`` is on by default, this is the *initial*
        weight rather than the weight of the whole run.
    proximal_adaptive : bool, default True
        Whether the weight is raised during the run when the step outruns the
        linearisation it was taken on.

        With ``False`` the weight is a constant. With ``True`` it is doubled
        once *two consecutive* iterates, both solved under the same weight,
        report that most of the capital their steps moved was moved back
        again:

        ``progress < 0.5 twice in a row at one delta  =>  delta <- 2 * delta``

        ``progress`` is the share of the moved capital that is net
        displacement over those two steps - one if every branch walked in one
        direction, zero if every branch came back - and it is reported per
        accepted iterate in ``n.iteration_log``. It reads the direction of the
        movement and its size together; see ``displacement`` for why neither
        alone is enough, and ``PROXIMAL_TURN_BAR`` for where the bar comes
        from and what holds it down.

        The count restarts whenever a reading clears the bar and whenever the
        weight changes, so the first reading after a doubling - which compares
        two steps solved under different penalties and reads low for that
        reason alone - can arm the gate but can never trip it on its own. What
        the gate buys is that a single pair of steps that happen to disagree no
        longer damps a run that is walking; what it costs is one extra iterate
        per doubling on a run that really is turning, an orbit reading low on
        every iterate it runs for.

        It is on by default because the systems it is aimed at cannot be
        recognised in advance and fail silently when it is off: a weight below
        the system's stability boundary produces a *converged* run whose plan
        is still moving. Measured on a 1250 bus case (2435 extendable
        branches, 36 snapshots), the default weight of 1.0 held fixed stops on
        a period-2 orbit at iteration 16 with a relative KVL residual of
        1.6e-4, while the rule from the same start converges at iteration 18
        with 8.7e-7 - and beats the best fixed weight on that system, 2.5, by
        five iterations. That run predates the confirmation gate described
        above, which delays each doubling by one iterate; what it establishes
        is the failure the rule prevents, not the iteration count it reaches
        now.

        It costs nothing on a system that never needed it, which earlier
        signals did not manage: on the small meshed test system, where a fixed
        weight of 1.0 reaches an exact fixed point in 16 iterations, the rule
        reproduces that run iterate for iterate and never raises the weight,
        ``progress`` staying between 0.87 and 1.00. Triggering on a rise in the
        step took 29 iterations there and triggering on a rise in the KVL
        residual 30, both by damping a run that was already converging.

        The weight is never released again, and it is bounded above by
        ``proximal_ceiling``. A fixed point of the penalised step is a fixed
        point of the unpenalised problem, so a weight that ends up too high
        costs iterations but does not move the plan it converges to, whereas
        releasing it re-opens the oscillation it was raised to close.

        The rule needs the ceiling because a cosine is meaningless once the
        steps are numerical noise: the sign is then random, so half the
        iterates of a run that has already stopped would be read as turns and
        ratchet the weight up. ``PROXIMAL_STEP_FLOOR`` catches the clearly dead
        case - a window moving less than 1e-9 of the plan's own capital is read
        as a fixed point rather than as a direction - and the ceiling bounds
        whatever gets through.

        The rule is aimed at the false convergence a fixed weight below the
        system's stability boundary produces: the iteration settles into an
        orbit in which the cost is stationary - the cost is flat along the
        directions an orbit turns in - while the plan is still moving. Such a
        run passes the cost test of ``cost_threshold`` and reports itself
        converged at an iterate whose ``step`` is still large. Reading ``step``
        on the last row of ``n.iteration_log`` is what detects it.
    proximal_ceiling : float, default 4.0
        Upper bound on the weight ``proximal_adaptive`` raises, in the same
        dimensionless units as ``proximal_weight``. Ignored when the rule is
        off, since a constant weight needs no bound. It has to be at least
        ``proximal_weight``.

        The default is what the rule needed on the systems this was measured
        on rather than a safety margin above them: on the 1250 bus case the
        weight settled at 3 to 4 and never approached a higher bound, and on
        the small meshed system a bound of 4 was the difference between
        reaching the fixed point in 30 iterations and still moving after 40.
        Raise it for a system whose stability boundary is known to be higher,
        at the price of the over-damping described above.

        A ceiling is safe in a way that it would not be for a step bound,
        because the weight does not decide the plan the iteration converges to,
        only how fast it gets there; what the ceiling costs is the ability to
        damp a system whose stability boundary lies above it, which shows up as
        a run that keeps oscillating at the ceiling rather than one that
        reports a wrong answer.
    sensitivity_tolerance : float, default 1e-5
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
        Status and termination condition of the report solve that closes the
        run at the capacities of its last iterate. The convergence history is
        written to ``n.iteration_log``.
    """
    schemes = ("slp", "fixed_point")
    if scheme not in schemes:
        raise ValueError(f"scheme must be one of {schemes}, got {scheme!r}.")
    cost_threshold = float(cost_threshold)
    step_threshold = float(step_threshold)
    # the weight doubles as the switch: zero leaves the term out of the model
    proximal_on = proximal and float(proximal_weight) > 0.0
    # The scheme decides both what the inner problem is and what restricts its
    # step. It is resolved into the switches the iteration reads here, once,
    # rather than tested again at each of them.
    if scheme == "slp":
        # the voltage law is linearised in the capacities, and the proximal
        # term is there to keep that linearisation valid over the step
        linearised = True
    else:
        # ``'fixed_point'``: the voltage law is frozen rather than linearised,
        # so there is no linearisation error for a control to keep small. The
        # step control is off and the inner problem is the plain linear
        # expansion problem, taken at the bare fixed-point step.
        linearised = False
        if proximal_on:
            logger.info(
                "The proximal term is inactive under scheme='fixed_point', "
                "which takes the plain fixed-point step. Use scheme='slp' for "
                "a damped iteration, or proximal=False to silence this.",
            )
        proximal_on = False
    if cost_threshold < 0.0:
        raise ValueError("cost_threshold must be >= 0.")
    if step_threshold < 0.0:
        raise ValueError("step_threshold must be >= 0; zero switches the test off.")
    proximal_weight = float(proximal_weight)
    if proximal_weight < 0.0:
        raise ValueError("proximal_weight must be >= 0; zero switches the term off.")
    proximal_adaptive = bool(proximal_adaptive)
    proximal_ceiling = float(proximal_ceiling)
    if proximal_adaptive and proximal_ceiling < proximal_weight:
        raise ValueError(
            "proximal_ceiling must be >= proximal_weight; it bounds the weight "
            f"the adaptive rule raises from it, got {proximal_ceiling} < "
            f"{proximal_weight}."
        )
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
    # DC links carry no voltage law, so the iteration does not linearise them
    # - but they are part of the plan, and the solve that closes a run has to
    # hold them as well as the AC capacities, see ``pin_capacities``
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
        if track_iterations and "LineX" in branch_components:
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
            if track_iterations and "LineX-sssc_nom" in m.variables
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

    # capacities the iteration starts from; they define the floor of the
    # impedance-defining capacity
    initial_caps = collect_branch_caps("s_nom").astype(float)
    capacity_floor = initial_caps.abs() * 1e-6

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

    def displacement(current: np.ndarray, previous: np.ndarray, plan: pd.Series) -> float:
        r"""
        Share of the moved capital that is net displacement over the last two
        steps.

        .. math::
            \eta_n = \frac{\sum_l c_l \,\bigl|\Delta^n_l + \Delta^{n-1}_l\bigr|}
                           {\sum_l c_l \bigl(|\Delta^n_l| + |\Delta^{n-1}_l|\bigr)}

        with :math:`\Delta^k = F^{k,*} - F^{k,\mathrm{fix}}` the movement the
        plan actually made at iteration :math:`k`. One means every branch
        walked in one direction, zero that every branch came back to where it
        started: the iteration turned instead of moving.

        **It reads direction and size together**, which is what neither of the
        two obvious alternatives does. The size of the step alone cannot tell a
        plan that is walking steadily from one that is stuck: measured on a
        1250 bus case a healthy run holds ``step[n]/step[n-1]`` between 0.88
        and 0.94 for twenty iterations while an orbit holds 0.99 to 1.02, so a
        bar on the ratio has to thread a 1.5 % gap - and on the small meshed
        system a perfectly healthy run twice takes a *longer* step than the one
        before it while walking straight. The direction alone is no better,
        because it ignores how far the plan went: a long step followed by a
        short one in the opposite direction reverses the cosine to -0.11 while
        the plan has plainly still moved, and at a fixed point, where the
        remaining motion is numerical noise, the cosine is a random sign.
        :math:`\eta` reads 0.88 and 1.00 on those same two iterates, because
        the net displacement is dominated by the step that was long.

        Two properties of the form matter. The modulus is taken **per branch**
        before the sum, so a branch growing while another shrinks does not
        cancel into a false reading of progress. And the capital cost weights
        it, as it weights the step and the proximal term, so the many small
        branches that swap between degenerate optima cannot outvote the plan.
        """
        weights = move_costs.to_numpy()
        gross = float((weights * (np.abs(current) + np.abs(previous))).sum())
        capital = float((move_costs * plan.loc[ext_branches].abs()).sum())
        if gross <= PROXIMAL_STEP_FLOOR * max(capital, 1.0):
            # nothing moved at all, which is the fixed point, not a turn
            return 1.0
        return float((weights * np.abs(current + previous)).sum() / gross)

    def relative_capacity_change(
        current: pd.Series, previous: pd.Series, initial: pd.Series
    ) -> float:
        r"""
        Size of the step, in the capital-cost weighted Euclidean norm.

        .. math::
            \mathrm{step} = \sqrt{\frac{\sum_l c_l (F_l - F'_l)^2}
                                        {\sum_l c_l (F^0_l)^2}}

        over the extendable branches. Weighting by the capital cost is what
        makes the measure about the *plan* rather than about megawatts: an
        expensive branch that keeps moving says the plan is not settled, while
        the same movement on a cheap one barely changes it, and an unweighted
        norm cannot tell the two apart. It is the same weight the proximal
        term charges in, so the quantity the step control prices and the
        quantity the convergence test reads are the same quantity.

        The reference is the *initial* capacity vector, fixed for the run, so
        the measure is a displacement relative to the system the run started
        from rather than to whatever the current iterate happens to be.
        """
        if ext_branches.empty:
            return 0.0
        weights = move_costs.to_numpy()
        reference = initial.loc[ext_branches].to_numpy()
        denom = float(np.sqrt((weights * reference**2).sum()))
        if denom <= 0.0:
            denom = 1e-12
        delta = (current - previous).loc[ext_branches].to_numpy()
        return float(np.sqrt((weights * delta**2).sum()) / denom)

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

    # capital cost of each extendable branch, aligned on ``ext_branches``.
    # Both the proximal term and the step measure are weighted with it: what
    # matters about a step is the capital it moves, not the megawatts.
    move_costs = (
        pd.concat(proximal_costs, names=["component", "name"]).reindex(ext_branches)
        if branch_components
        else pd.Series(dtype=float)
    )

    # cost of the capacity that is already installed, which the objective of
    # every iteration carries along. It is frozen at the value of the first
    # iteration, where the capacities still are the initial ones.
    installed_cost = [0.0]

    def proximal_term_value(caps: pd.Series, center: pd.Series) -> float:
        """
        Value of the proximal expression, without its outer weight.
        """
        value = 0.0
        for c in branch_components:
            ext_i = branch_data[c]["ext_i"]
            if ext_i.empty:
                continue
            anchor = center.xs(c, level="component").reindex(ext_i)
            current = caps.xs(c, level="component").reindex(ext_i)
            value += float(
                (proximal_costs[c] * (current - anchor) ** 2 / anchor).sum()
            )
        return value

    def add_proximal_penalty(
        network: Network,
        center: pd.Series,
        weight: float,
    ) -> None:
        """
        Penalise moving the branch capacities away from the previous iterate.

        A scaled free deviation variable per branch keeps every diagonal
        quadratic coefficient identical and leaves the branch-specific scaling
        in linear defining equalities, which is kinder to the solver than
        writing the branch weights into the quadratic objective directly.
        """
        m = network.model
        for c in branch_components:
            ext_i = branch_data[c]["ext_i"]
            attr = nominal_attrs[c]
            if ext_i.empty or f"{c}-{attr}" not in m.variables:
                continue
            capacity = m[f"{c}-{attr}"]
            anchor = center.xs(c, level="component").reindex(ext_i)
            if not np.isfinite(anchor).all() or (anchor <= 0.0).any():
                raise ValueError(
                    "The proximal term requires strictly positive finite "
                    "previous branch capacities, since it scales by them."
                )
            scale = np.sqrt(weight * proximal_costs[c] / anchor)
            deviation = m.add_variables(
                lower=-np.inf,
                upper=np.inf,
                coords=[ext_i],
                name=f"{c}-{attr}_deviation",
            )
            m.add_constraints(
                deviation - capacity * scale == -anchor * scale,
                name=f"{c}-{attr}_deviation-definition",
            )
            m.objective = m.objective + (deviation * deviation).sum()

    def extra_functionality_for(
        center: pd.Series | None,
        weight: float,
    ) -> Any:
        def extra_functionality(network: Network, sns: pd.Index) -> None:
            if weight > 0.0 and center is not None:
                add_proximal_penalty(network, center, weight)
            if user_extra_functionality is not None:
                user_extra_functionality(network, sns)

        return extra_functionality

    def solved_cost(
        network: Network,
        caps: pd.Series,
        center: pd.Series | None,
        weight: float,
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
            value -= weight * proximal_term_value(caps, center)
        return value - installed_cost[0]

    def has_settled(history: list[float], cost: float, step: float) -> bool:
        """
        Both convergence tests, on the same iterate.

        The cost has to have stopped changing *and* the plan has to have
        stopped moving. Either alone is met by a run that has not converged:
        the cost is stationary along the directions a limit cycle turns in,
        and a small step can be produced by damping rather than by
        stationarity.
        """
        if not history:
            return False
        if abs(cost - history[-1]) / max(abs(cost), 1e-12) > cost_threshold:
            return False
        return step_threshold <= 0.0 or step <= step_threshold

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
        two exits report the same quantity.

        The DC links are held too, at the capacities of the same solve. They
        carry no voltage law and so never enter the iteration's own controls,
        but they are transmission capacity all the same: re-optimised at fixed
        AC capacities they make the report a different plan rather than the
        price of this one. Everything that is not capacity - the series
        compensation, the dispatch, the storage - is left free, which is what
        makes this the cost *of* the plan.
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

    def restore_nominal_bounds(network: Network) -> None:
        """Restore the capacity bounds ``pin_capacities`` replaced."""
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
    current_def = initial_caps.copy()
    cost_history: list[float] = []
    proximal_delta = proximal_weight if proximal_on else 0.0
    # Step vector of the previous accepted iterate, which is what
    # ``proximal_adaptive`` measures the displacement against.
    previous_step_vector: np.ndarray | None = None
    # A doubling changes the penalty regime. Keep the following iterate as a
    # cooldown step, so its low progress cannot immediately trigger another
    # doubling.
    proximal_raised_previous_iteration = False

    reference_terms: pd.DataFrame | None = None
    # last solved DC link capacities, held by the report solve alongside the AC
    current_links: pd.Series | None = None
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

            # linearise the KVL constraint in the branch capacities around the
            # linearisation point
            sensitivity = None
            if linearised and reference_terms is not None and not plain_step:
                sensitivity = capacity_sensitivity(reference_terms, current_def)
            # a fallback step onto the plain fixed point is taken on its own,
            # without the linearisation that made the previous attempt
            # infeasible
            plain_step = False
            n._kvl_capacity_sensitivity = sensitivity

            # anchor of the proximal term, absent in the first iteration where
            # no previous iterate exists
            anchor = current_def if iteration > 1 else None
            weight = proximal_delta if anchor is not None else 0.0
            status, condition = solve_once(
                snapshots,
                lightweight=True,
                extra_functionality=extra_functionality_for(anchor, weight),
            )
            if status != "ok":
                if sensitivity is not None:
                    # the linearised problem can be infeasible where the frozen
                    # impedances require more capacity than the linearised
                    # voltage law admits. Fall back to a plain fixed-point
                    # step, which is always feasible.
                    plain_step = True
                    logger.warning(
                        "Iteration %d failed with status %s/%s, falling back to a "
                        "fixed-point step.",
                        iteration,
                        status,
                        condition,
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
                            "progress": np.nan,
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
                            "progress": np.nan,
                            "proximal": np.nan,
                            "accepted": False,
                        }
                    )
                    discard_model(n)
                    iteration += 1
                    continue

            if iteration == 1:
                installed_cost[0] = float(getattr(n, "objective_constant", 0.0))
            if iteration_result is not None and iteration_result.link_caps is not None:
                current_links = iteration_result.link_caps
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
            # left over is the error of the model that was solved. Every solved
            # iterate is accepted however large that residual is: at a converged
            # point the linearisation reproduces its own point, where the
            # constraint coincides with the exact one.

            converged = (
                has_settled(cost_history, cost, diff)
                and iteration >= min_iterations
            )
            step_vector = (caps - current_def).loc[ext_branches].to_numpy()
            progress = (
                displacement(step_vector, previous_step_vector, caps)
                if previous_step_vector is not None
                else float("nan")
            )
            # A window whose capital was mostly moved back is a turning step.
            # Shorten the *next* step immediately, unless the preceding
            # iteration already raised the weight: that one-iteration cooldown
            # lets the new penalty take effect before it is judged again.
            can_raise_proximal = (
                proximal_on
                and proximal_adaptive
                and not converged
                and not proximal_raised_previous_iteration
                and np.isfinite(progress)
                and progress < PROXIMAL_TURN_BAR
            )
            if can_raise_proximal:
                previous_proximal_delta = proximal_delta
                proximal_delta = min(
                    proximal_delta * PROXIMAL_ADAPTIVE_FACTOR, proximal_ceiling
                )
                proximal_raised_previous_iteration = (
                    proximal_delta > previous_proximal_delta
                )
            else:
                proximal_raised_previous_iteration = False
            previous_step_vector = step_vector

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
                if not proximal_on
                else (
                    f", progress = {progress:.3f}, proximal = {weight:.3e}"
                    + (f" -> {proximal_delta:.3e}" if proximal_adaptive else "")
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
                    "progress": progress,
                    "proximal": weight,
                    "accepted": True,
                }
            )

            if converged:
                has_converged = True
                current_def = clip_branch_caps(
                    caps.copy(), branch_cap_min, branch_cap_max
                )
                cost_history.append(cost)
                # the report solve below closes this run as it closes any
                # other, so this iterate's model is released like the rest
                discard_model(n)
                break

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
        # never leave the linearisation on the network, not even when the loop
        # is left through an exception
        n._kvl_capacity_sensitivity = None
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
                "progress",
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

    # Every run is closed by this solve, converged or not. The iterates are
    # solved lightweight and their models discarded, so no solved iterate is
    # left to read a full result from; and an iterate's dispatch comes from the
    # *linearised* voltage law with the proximal penalty still in its
    # objective, which is a step of the iteration rather than a plan priced at
    # its own impedances. Solve once more with the transmission capacities
    # pinned at the last iterate and the impedances put on them, leaving
    # everything that is not transmission capacity free.
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
            # proximal term would only add a constant, and a step control does
            # not belong in a solve that takes no step.
            extra_functionality=extra_functionality_for(None, 0.0),
        )
    finally:
        n._kvl_capacity_sensitivity = None
        restore_nominal_bounds(n)
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
