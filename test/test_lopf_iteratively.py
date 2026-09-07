import importlib
import inspect
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for module_name in list(sys.modules):
    if module_name == "pypsa" or module_name.startswith("pypsa."):
        del sys.modules[module_name]

pypsa = importlib.import_module("pypsa")
optimize_module = importlib.import_module("pypsa.optimization.optimize")
constraints_module = importlib.import_module("pypsa.optimization.constraints")
abstract_module = importlib.import_module("pypsa.optimization.abstract")

SOLVER = dict(
    solver_name="gurobi",
    solver_options={"OutputFlag": 0},
    # the ohmic losses are part of the model the iteration has to reproduce
    transmission_losses=2,
)


def _build_simple_network() -> pypsa.Network:
    n = pypsa.Network()
    n.madd("Bus", ["a", "b"], v_nom=220.0, carrier="AC")
    n.add("Generator", "gen", bus="a", p_nom=200.0, marginal_cost=10.0)
    n.add("Load", "load", bus="b", p_set=100.0)
    n.add(
        "Line",
        "ab",
        bus0="a",
        bus1="b",
        x=0.1,
        r=0.01,
        s_nom=100.0,
        s_nom_extendable=True,
        capital_cost=1.0,
    )
    return n


def test_iterative_updates_next_def_from_outer_optimum(monkeypatch):
    n = _build_simple_network()
    outer_caps_seen = []
    outer_targets = [150.0, 180.0]
    outer_call_count = 0

    def fake_optimize(network, snapshots=None, *args, **kwargs):
        nonlocal outer_call_count
        network.objective = 0.0
        outer_caps_seen.append(
            float(
                network.lines.at["ab", "_s_nom_def"]
                if "_s_nom_def" in network.lines
                else network.lines.at["ab", "s_nom"]
            )
        )
        target = outer_targets[min(outer_call_count, len(outer_targets) - 1)]
        network.lines["s_nom_opt"] = target
        outer_call_count += 1
        return "ok", "optimal"

    monkeypatch.setattr(optimize_module, "optimize", fake_optimize)

    status, condition = n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=2,
        min_iterations=2,
    )

    assert status == "ok"
    assert condition == "optimal"
    assert outer_caps_seen[:2] == [100.0, 150.0]


def _plan_sequence(monkeypatch, network, values):
    """Stub the solve so the plan walks through ``values``, cycling them."""
    seq = iter(list(values) * 50)

    def fake_optimize(net, snapshots=None, *args, **kwargs):
        net.objective = 0.0
        net.lines["s_nom_opt"] = next(seq)
        return "ok", "optimal"

    monkeypatch.setattr(optimize_module, "optimize", fake_optimize)
    for name in ("assign_solution", "assign_duals", "post_processing"):
        monkeypatch.setattr(optimize_module, name, lambda *a, **kw: None)
    return network


def _orbit(monkeypatch, network, values):
    """
    Stub the solve so the plan walks back and forth between two capacities.

    Consecutive steps are then exactly opposite, so the displacement measure
    reads zero and the run is a period-2 orbit by construction. The small
    meshed system converges from every weight it was tried at, so an orbit has
    to be built rather than found.
    """
    return _plan_sequence(monkeypatch, network, values)


def test_the_weight_is_doubled_while_the_plan_turns(monkeypatch):
    """
    ``proximal_adaptive`` reads ``progress``, the share of the moved capital
    that is net displacement over the last two steps. On an orbit every branch
    comes back to where it started, so it reads zero on every iterate and the
    weight doubles after a low-progress reading, with one cooldown iteration
    between doublings, until the ceiling stops it.
    """
    n = _orbit(monkeypatch, _build_simple_network(), [150.0, 100.0])

    n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=6, proximal_weight=1.0, proximal_adaptive=True,
        proximal_ceiling=1024.0,
    )

    log = n.iteration_log
    accepted = log[log.accepted]
    # iteration 1 has no previous step to compare to
    measured = accepted.progress.iloc[1:]
    assert (measured < 0.5).all(), accepted.progress.tolist()
    weights = accepted.proximal.tolist()
    assert weights[1] == 1.0, weights
    # Each low-progress iterate doubles the following weight, then the next
    # iterate is a cooldown step under that new weight.
    for i in range(2, len(weights)):
        if i % 2 == 0:
            assert weights[i] == pytest.approx(2.0 * weights[i - 1]), weights
        else:
            assert weights[i] == weights[i - 1], weights


def test_a_single_turn_raises_the_weight(monkeypatch):
    """
    Here the plan overshoots once and then walks straight down. One low
    progress reading is enough to increase the proximal weight for the next
    iteration, even though subsequent steps clear the bar.
    """
    n = _plan_sequence(
        monkeypatch,
        _build_simple_network(),
        [200.0, 150.0, 100.0, 60.0, 30.0, 15.0],
    )

    n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=6, proximal_weight=1.0, proximal_adaptive=True,
        proximal_ceiling=1024.0,
    )

    accepted = n.iteration_log[n.iteration_log.accepted]
    # only the step back from 200 to 150 reverses; every later step continues
    progress = accepted.progress.dropna()
    assert (progress < 0.5).sum() == 1, progress.tolist()
    assert accepted.proximal.iloc[1] == 1.0
    assert accepted.proximal.iloc[2] == 2.0
    assert (accepted.proximal.iloc[3:] == 2.0).all(), accepted.proximal.tolist()


def test_the_adaptive_weight_stops_at_the_ceiling(monkeypatch):
    """The doubling is bounded, which is what makes an orbit safe to chase."""
    n = _orbit(monkeypatch, _build_simple_network(), [150.0, 100.0])

    n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=10, proximal_weight=1.0, proximal_adaptive=True,
        proximal_ceiling=4.0,
    )

    weights = n.iteration_log[n.iteration_log.accepted].proximal
    assert weights.max() == 4.0
    # and it does get there, i.e. the ceiling is what stopped it
    assert (weights == 4.0).any()


def test_the_adaptive_rule_is_on_by_default(monkeypatch):
    """
    The rule is the default because a weight below a system's stability
    boundary fails silently: it reports a converged run whose plan is still
    moving. On an orbit the default has to raise the weight and switching the
    rule off has to leave it alone.
    """
    kwargs = dict(max_iterations=6, proximal_weight=1.0, proximal_ceiling=64.0)

    default = _orbit(monkeypatch, _build_simple_network(), [150.0, 100.0])
    default.optimize.optimize_transmission_expansion_iteratively(**kwargs)

    off = _orbit(monkeypatch, _build_simple_network(), [150.0, 100.0])
    off.optimize.optimize_transmission_expansion_iteratively(
        proximal_adaptive=False, **kwargs
    )

    assert default.iteration_log.proximal.max() > 1.0
    assert off.iteration_log.proximal.max() == 1.0


def test_the_rule_is_free_on_a_system_that_does_not_turn():
    """
    On a system that walks straight the rule must cost nothing at all. Earlier
    signals did not manage this: triggering on a rise in the step, or in the
    KVL residual, damped a run that was already converging.
    """
    fixed = _build_meshed_network()
    fixed.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=40, proximal_weight=1.0, proximal_adaptive=False, **SOLVER
    )
    adaptive = _build_meshed_network()
    adaptive.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=40, proximal_weight=1.0, proximal_adaptive=True, **SOLVER
    )

    assert (adaptive.iteration_log.progress.dropna() >= 0.5).all()
    assert adaptive.iteration_log.proximal.max() == 1.0
    assert len(adaptive.iteration_log) == len(fixed.iteration_log)
    pd.testing.assert_series_equal(
        adaptive.iteration_log.step, fixed.iteration_log.step, rtol=1e-9
    )


def test_proximal_weight_is_fixed_for_the_whole_run(monkeypatch):
    """The proximal weight never moves off its default: it is not adapted."""
    n = _build_simple_network()

    def fake_optimize(network, snapshots=None, *args, **kwargs):
        network.objective = 0.0
        network.lines["s_nom_opt"] = 150.0
        return "ok", "optimal"

    monkeypatch.setattr(optimize_module, "optimize", fake_optimize)
    # The fake solve returns the same capacities every time, so the system cost
    # does not move and the run converges on the third iteration. Converging
    # materialises the user-facing solution, which a solve that never built a
    # model cannot provide - and this test is about the weights in the log, not
    # about the solution. Stub the three materialisation steps out.
    for name in ("assign_solution", "assign_duals", "post_processing"):
        monkeypatch.setattr(optimize_module, name, lambda *a, **kw: None)

    n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=3,
        min_iterations=3,
        proximal_adaptive=False,
    )

    # Iteration 1 has no anchor. Every anchored solve uses the same weight:
    # it is a constant of the run rather than something the run steers.
    default = (
        inspect.signature(
            abstract_module.optimize_transmission_expansion_iteratively
        )
        .parameters["proximal_weight"]
        .default
    )
    assert n.iteration_log.proximal.tolist() == [0.0, default, default]


def test_iterative_invalid_switches_raise():
    n = _build_simple_network()

    with pytest.raises(ValueError, match="scheme must be one of"):
        n.optimize.optimize_transmission_expansion_iteratively(scheme="newton")
@pytest.mark.parametrize("tolerance", [-1e-9, 1.0])
def test_iterative_invalid_sensitivity_tolerance_raises(tolerance):
    n = _build_simple_network()

    with pytest.raises(ValueError, match="sensitivity_tolerance"):
        n.optimize.optimize_transmission_expansion_iteratively(
            sensitivity_tolerance=tolerance
        )


def _build_meshed_network(sssc: bool = False, s_nom: float = 50.0) -> pypsa.Network:
    """
    Four buses in a ring with a chord, hence two independent cycles, supplied by
    a cheap generator far from the load. Brownfield expansion (s_nom_min =
    s_nom) of all lines is needed, so the impedance feedback is strong.
    """
    n = pypsa.Network()
    n.set_snapshots(pd.Index(range(3), name="snapshot"))
    n.madd("Bus", list("abcd"), v_nom=220.0, carrier="AC")
    n.add("Generator", "cheap", bus="a", p_nom=500.0, marginal_cost=5.0)
    n.add("Generator", "mid", bus="b", p_nom=500.0, marginal_cost=40.0)
    n.add("Generator", "peak", bus="c", p_nom=500.0, marginal_cost=200.0)
    n.add("Load", "load_c", bus="c", p_set=[200.0, 160.0, 240.0])
    n.add("Load", "load_d", bus="d", p_set=[100.0, 140.0, 80.0])
    for name, bus0, bus1, x in [
        ("ab", "a", "b", 0.20),
        ("bc", "b", "c", 0.20),
        ("cd", "c", "d", 0.15),
        ("da", "d", "a", 0.30),
        ("ac", "a", "c", 0.45),
    ]:
        n.add(
            "Line",
            name,
            bus0=bus0,
            bus1=bus1,
            x=x,
            r=0.1 * x,
            s_nom=s_nom,
            s_nom_min=s_nom,
            s_nom_extendable=True,
            capital_cost=200.0,
        )
    if sssc:
        n.calculate_dependent_values()
        # an SSSC of rating Q can shift the reactance of a line loaded to F by
        # Q / F**2, so this is a compensation degree of 30%
        cap = 0.3 * n.lines.x_pu_eff * n.lines.s_nom**2
        n.convert_lines_to_line_x(
            n.lines.index,
            sssc_nom_extendable=True,
            sssc_nom_max=cap,
            capital_cost_sssc=0.05 * n.lines.capital_cost * n.lines.s_nom / cap,
        )
    return n


def _branch_capacities(n: pypsa.Network) -> pd.Series:
    return pd.concat(
        {
            c: n.df(c).s_nom_opt
            for c in ("Line", "LineX")
            if c in n.components and not n.df(c).empty
        }
    )


@pytest.mark.parametrize("sssc", [False, True])
def test_slp_converges_faster_and_consistently(sssc):
    """
    On a meshed brownfield expansion the plain fixed-point iteration converges
    slowly or not at all, while the linearised iteration converges quickly and
    leaves a much smaller residual of the exact voltage law.
    """
    results = {}
    for scheme in ("fixed_point", "slp"):
        n = _build_meshed_network(sssc)
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            scheme=scheme, proximal_weight=0.0,
            max_iterations=40, **SOLVER
        )
        assert status == "ok"
        assert condition == "optimal"
        log = n.iteration_log
        residual = log.violation_rel.dropna()
        reached = residual[residual <= 1e-3]
        results[scheme] = {
            "iterations": len(log),
            "cost": float(log.cost.dropna().iloc[-1]),
            "capacities": _branch_capacities(n),
            "kvl": float(residual.iloc[-1]),
            # iteration at which the solution first becomes consistent with the
            # impedances it assumes, independent of the stopping rule
            "consistent_at": int(reached.index[0]) if len(reached) else 10**6,
        }

    fixed, slp = results["fixed_point"], results["slp"]

    # the linearisation reaches the tolerance in far fewer iterations
    assert slp["iterations"] < fixed["iterations"]
    # and becomes consistent with its own impedances earlier
    assert slp["consistent_at"] <= fixed["consistent_at"]
    assert slp["kvl"] < 1e-4
    # the fixed point ends up in a limit cycle, so it also reports the more
    # expensive plan of the two
    assert slp["cost"] < fixed["cost"]
    # brownfield: existing capacity is never removed
    assert (slp["capacities"] >= 50.0 - 1e-6).all()


def test_iteration_log_records_and_restores_bounds():
    n = _build_meshed_network()
    bounds = n.lines[["s_nom_min", "s_nom_max"]].copy()

    n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=20, **SOLVER
    )

    log = n.iteration_log
    assert list(log.columns) == [
        "status",
        "cost",
        "cost_change",
        "step",
        "violation",
        "violation_rel",
        "progress",
        "proximal",
        "accepted",
    ]
    assert log.index.name == "iteration"
    assert not log.empty
    # the converged iterate is consistent with the impedances it assumes
    assert log.violation_rel.dropna().iloc[-1] < 1e-4
    # the iteration must not leave modified capacity bounds behind
    pd.testing.assert_frame_equal(n.lines[["s_nom_min", "s_nom_max"]], bounds)
    assert n._kvl_capacity_sensitivity is None


@pytest.mark.parametrize("sssc", [False, True])
def test_unconverged_run_reports_its_last_iterate(sssc):
    """
    A run that exhausts its budget closes with a solve that costs the last
    iterate. That solve is linearised at the last iterate, so it has to be
    taken *at* it: released, it walks away from the point its own impedances
    came from and buys an objective the network cannot deliver.
    """
    n = _build_meshed_network(sssc)
    bounds = {c: n.df(c)[["s_nom_min", "s_nom_max"]].copy() for c in ("Line", "LineX")
              if not n.df(c).empty}

    n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=2, track_iterations=True, **SOLVER
    )

    assert len(n.iteration_log) == 2, "the run has to be the unconverged one"
    for c, before in bounds.items():
        df = n.df(c)
        ext = df.s_nom_extendable
        if ext.any():
            # the plan returned is the last iterate, at the impedances that
            # iterate defines, so the two agree and the voltage law is exact
            pd.testing.assert_series_equal(
                df.s_nom_opt[ext], df._s_nom_def[ext],
                rtol=1e-9, check_names=False,
            )
            pd.testing.assert_series_equal(
                df.s_nom_opt[ext], df.s_nom_opt_2[ext],
                rtol=1e-9, check_names=False,
            )
        # and the bounds it was held at do not leak out of the call
        pd.testing.assert_frame_equal(df[["s_nom_min", "s_nom_max"]], before)



def test_converged_run_is_closed_by_the_report_solve(monkeypatch):
    """
    Every run is closed by the pinned report solve, converged or not. At a
    converged point the linearisation is exact, so that solve reproduces the
    iterate rather than moving away from it - but it is what prices the plan at
    its own impedances, carries the duals and the derived time series, and
    carries no proximal penalty.
    """
    n = _build_meshed_network()
    bounds = n.lines[["s_nom_min", "s_nom_max"]].copy()
    solves = []
    original = optimize_module.optimize

    def counted(network, *args, **kwargs):
        solves.append(kwargs.get("_assign_solution", True))
        return original(network, *args, **kwargs)

    monkeypatch.setattr(optimize_module, "optimize", counted)

    n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=40, track_iterations=True, **SOLVER
    )

    log = n.iteration_log
    assert len(log) < 40, "the run has to be the converged one"
    # one lightweight solve per iterate, then exactly one full solve to close
    assert len(solves) == len(log) + 1
    assert not any(solves[:-1])
    assert solves[-1] is True

    ext = n.lines.s_nom_extendable
    # the report solve is held at the converged capacities, and the impedances
    # are those of the same capacities
    pd.testing.assert_series_equal(
        n.lines.s_nom_opt[ext], n.lines._s_nom_def[ext],
        rtol=1e-9, check_names=False,
    )
    pd.testing.assert_series_equal(
        n.lines.s_nom_opt[ext], n.lines[f"s_nom_opt_{len(log)}"][ext],
        rtol=1e-9, check_names=False,
    )
    # it is the plain formulation: no step control survives into it
    assert "Line-s_nom_deviation" not in n.model.variables
    assert "Line-s_nom_relative" not in n.model.variables
    assert not bool(n.model.objective.is_quadratic)
    assert n._kvl_capacity_sensitivity is None
    pd.testing.assert_frame_equal(n.lines[["s_nom_min", "s_nom_max"]], bounds)
    # and the export is that of a full solve: capacities, dispatch and duals
    assert n.generators.p_nom_opt.notna().all()
    assert np.isfinite(n.generators_t.p.to_numpy()).all()
    assert np.isfinite(n.lines_t.p0.to_numpy()).all()
    assert np.isfinite(n.buses_t.v_ang.to_numpy()).all()
    assert not n.buses_t.marginal_price.empty
    assert np.isfinite(n.buses_t.marginal_price.to_numpy()).all()
    assert all("dual" in con for _, con in n.model.constraints.items())


def test_the_report_solve_reoptimises_everything_but_transmission():
    """
    The transmission capacities are pinned; the generation, storage and
    dispatch are not, which is what makes the result the cost *of* that plan.
    """
    n = _build_meshed_network()
    n.generators["p_nom_extendable"] = True
    n.generators["capital_cost"] = 100.0
    n.generators["p_nom_max"] = 1000.0
    bounds = n.generators[["p_nom_min", "p_nom_max"]].copy()

    n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=40, **SOLVER
    )

    # the branch capacities the report solve was held at
    ext = n.lines.s_nom_extendable
    pd.testing.assert_series_equal(
        n.lines.s_nom_opt[ext], n.lines._s_nom_def[ext],
        rtol=1e-9, check_names=False,
    )
    # the generators were free in it, and their bounds are unchanged
    assert "Generator-p_nom" in n.model.variables
    pd.testing.assert_frame_equal(
        n.generators[["p_nom_min", "p_nom_max"]], bounds
    )


def test_the_proximal_weight_is_a_constant_of_the_run():
    """The weight is whatever it was configured with, on every anchored step."""
    n = _build_meshed_network()
    weight = 1.5
    n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=6, cost_threshold=0.0, proximal_weight=weight,
        proximal_adaptive=False, **SOLVER
    )

    accepted = n.iteration_log[n.iteration_log.accepted]
    # the first step has no anchor to be penalised against
    assert accepted.proximal.iloc[0] == 0.0
    assert (accepted.proximal.iloc[1:] == weight).all()


def test_the_adaptive_weight_only_ever_doubles():
    """
    The rule is one-directional: the weight is never released, so the sequence
    is non-decreasing and every change is a power of two of the initial weight.
    """
    n = _build_meshed_network()
    n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=10, cost_threshold=0.0, proximal_weight=1.5,
        proximal_adaptive=True, **SOLVER
    )

    weights = n.iteration_log[n.iteration_log.accepted].proximal.iloc[1:]
    assert (weights.diff().dropna() >= 0).all()
    ratios = (weights / 1.5).tolist()
    assert all(
        r == pytest.approx(2 ** round(np.log2(r))) for r in ratios
    ), ratios


def test_the_ceiling_has_to_admit_the_initial_weight():
    n = _build_simple_network()

    with pytest.raises(ValueError, match="proximal_ceiling"):
        n.optimize.optimize_transmission_expansion_iteratively(
            proximal_weight=2.0, proximal_adaptive=True, proximal_ceiling=1.0
        )
    # inert while the rule is off, so it must not reject anything there
    n.optimize.optimize_transmission_expansion_iteratively(
        proximal_weight=2.0, proximal_ceiling=1.0, proximal_adaptive=False,
        max_iterations=1,
        solver_name="highs", solver_options={"log_to_console": False},
    )


def test_proximal_targets_branch_capacities_only():
    """The term adds no penalty variable or curvature to the SSSC ratings."""
    n = _build_meshed_network(sssc=True)
    models = []

    def observe(network, snapshots):
        models.append(
            (
                set(network.model.variables),
                bool(network.model.objective.is_quadratic),
            )
        )

    n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=2,
        extra_functionality=observe,
        **SOLVER,
    )

    anchored = models[1]
    names, quadratic = anchored
    assert "LineX-sssc_nom_deviation" not in names
    assert "LineX-s_nom_deviation" in names
    assert quadratic


@pytest.mark.parametrize("track_iterations", [False, True])
def test_iterations_only_assign_the_final_network_solution(
    monkeypatch, track_iterations
):
    """Tracking capacities does not require full intermediate network results."""
    n = _build_meshed_network()
    calls = {"solution": 0, "duals": 0, "post_processing": 0}

    for key, function in [
        ("solution", optimize_module.assign_solution),
        ("duals", optimize_module.assign_duals),
        ("post_processing", optimize_module.post_processing),
    ]:
        def wrapped(network, *args, _function=function, _key=key, **kwargs):
            calls[_key] += 1
            return _function(network, *args, **kwargs)

        monkeypatch.setattr(optimize_module, function.__name__, wrapped)

    status, condition = n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=1,
        track_iterations=track_iterations,
        solver_name="highs",
        solver_options={"log_to_console": False},
        transmission_losses=2,
    )

    assert (status, condition) == ("ok", "optimal")
    # The outer solves are lightweight. This run does not converge, so it is
    # closed by the report solve, which materialises the complete user-facing
    # network once.
    assert calls == {"solution": 1, "duals": 1, "post_processing": 1}
    if track_iterations:
        assert "s_nom_opt_1" in n.lines
        assert hasattr(n, "objective_1")


@pytest.mark.parametrize("scheme", ["fixed_point", "slp"])
def test_convergence_needs_both_the_cost_and_the_step(scheme):
    """
    The run stops on one iterate that satisfies both tests: the cost has
    stopped changing *and* the plan has stopped moving.
    """
    n = _build_meshed_network()
    n.optimize.optimize_transmission_expansion_iteratively(
        scheme=scheme, cost_threshold=1e-4, step_threshold=0.02,
        max_iterations=60, **SOLVER,
    )
    log = n.iteration_log
    accepted = log[log.accepted]

    if len(log) < 60:  # converged rather than exhausted
        assert accepted.cost_change.iloc[-1] <= 1e-4
        assert accepted.step.iloc[-1] <= 0.02
    assert log.cost.notna().any()
    assert "step" in log


def test_the_step_test_refuses_a_stationary_cost_on_a_moving_plan():
    """
    The cost is flat along the directions a limit cycle turns in, so the cost
    test alone stops on an iterate whose plan is still moving. The step test is
    what refuses it.
    """
    n = _build_meshed_network()
    # loose enough that the cost test fires on the second iterate either way
    loose = dict(cost_threshold=1.0, min_iterations=1, max_iterations=6)

    n.optimize.optimize_transmission_expansion_iteratively(
        **loose, step_threshold=0.0, **SOLVER
    )
    without_step_test = len(n.iteration_log)

    n = _build_meshed_network()
    n.optimize.optimize_transmission_expansion_iteratively(
        **loose, step_threshold=1e-12, **SOLVER
    )
    with_step_test = len(n.iteration_log)

    assert without_step_test == 2
    assert with_step_test == 6, "an unreachable step bar has to run to the cap"


def test_step_threshold_is_validated():
    n = _build_simple_network()

    with pytest.raises(ValueError, match="step_threshold"):
        n.optimize.optimize_transmission_expansion_iteratively(step_threshold=-1.0)


def test_proximal_term_does_not_distort_the_optimum():
    """
    The proximal term damps the iteration but neither enters the reported cost
    nor moves the optimum noticeably. Only the converging scheme is compared:
    two points of the limit cycle the fixed point ends up in are not optima and
    carry no such statement.
    """
    costs = {}
    for proximal in (0.0, 1.0):
        n = _build_meshed_network()
        inner = []

        def observe(network, sns, _seen=inner):
            # the penalty is added before the caller's hook runs, so this sees
            # the inner problem of an iteration as it is solved
            _seen.append(
                (
                    "Line-s_nom_deviation" in network.model.variables,
                    bool(network.model.objective.is_quadratic),
                )
            )

        n.optimize.optimize_transmission_expansion_iteratively(
            proximal_weight=proximal,
            max_iterations=40,
            extra_functionality=observe,
            **SOLVER,
        )
        log = n.iteration_log
        costs[proximal] = float(log.cost.dropna().iloc[-1])
        on = proximal > 0.0
        # Iteration 1 has no anchor. From the second onward the term adds a
        # scaled free-deviation variable, so the inner problem becomes a QP.
        assert any(present for present, _ in inner[:-1]) == on
        assert any(quadratic for _, quadratic in inner[:-1]) == on
        # the report solve that closes the run carries no penalty, so the
        # model left on the network never has the deviation variable
        assert inner[-1][0] is False
        assert "Line-s_nom_deviation" not in n.model.variables
        assert not bool(n.model.objective.is_quadratic)
        # and the weight it was solved with is reported per iteration
        assert (log.proximal.fillna(0.0) > 0.0).any() == on

    off, on = costs[0.0], costs[1.0]
    assert abs(on - off) / abs(off) < 1e-3


def test_kvl_capacity_sensitivity_enters_constraint():
    """
    The linearised voltage law carries the capacity sensitivity as a term on the
    relative capacity deviation, which leaves its right-hand side at zero.
    """
    n = _build_meshed_network()
    n.lines["s_nom_extendable"] = False
    n.lines.loc["ac", "s_nom_extendable"] = True

    n.optimize.create_model()
    # This network is small enough that every cycle lands in the same term-count
    # bucket (see define_kirchhoff_voltage_constraints), so there is exactly one
    # "Kirchhoff-Voltage-Law-*" block and its suffix is the bucket id, "0".
    plain = n.model.constraints["Kirchhoff-Voltage-Law-0"]
    assert np.allclose(plain.rhs.values, 0.0)
    plain_terms = plain.vars.shape[-1]

    alpha = pd.DataFrame(
        0.25, index=n.snapshots, columns=pd.MultiIndex.from_tuples([("Line", "ac")])
    )
    n._kvl_capacity_sensitivity = alpha
    try:
        n.optimize.create_model()
        constraint = n.model.constraints["Kirchhoff-Voltage-Law-0"]
        label = n.model["Line-s_nom_relative"].labels.to_pandas().at["ac"]
        definition = n.model.constraints["Line-s_nom_relative-definition"]
        cycles = constraints_module.kirchhoff_voltage_cycles(n)
    finally:
        n._kvl_capacity_sensitivity = None

    assert constraint.vars.shape[-1] == plain_terms + 1

    # orientation of line 'ac' in each cycle it belongs to
    orientation = {}
    for branches_i, C, _, _ in cycles:
        position = branches_i.get_loc(("Line", "ac"))
        for j in range(C.shape[1]):
            rows = C.indices[C.indptr[j] : C.indptr[j + 1]]
            if position in rows:
                orientation[j] = float(C.data[C.indptr[j] : C.indptr[j + 1]][
                    list(rows).index(position)
                ])

    vars_ = constraint.vars.values
    coeffs = constraint.coeffs.values
    s_nom_def = n.lines.at["ac", "s_nom"]
    for cycle, sign in orientation.items():
        # the deviation is relative, so the sensitivity is scaled by the
        # linearisation capacity rather than divided out of the right-hand side
        expected = -1e4 * sign * 0.25 * s_nom_def
        found = coeffs[:, cycle, :][vars_[:, cycle, :] == label]
        assert np.allclose(found, expected)

    # the constant part of the linearisation is absorbed by the deviation and
    # no longer enters the voltage law, whose right-hand side stays zero
    assert np.allclose(constraint.rhs.values, 0.0)

    # u = F / F_def - 1
    assert np.allclose(definition.rhs.values, 1.0)
    assert np.allclose(
        np.sort(definition.coeffs.values, axis=-1),
        np.sort(np.array([1.0 / s_nom_def, -1.0]), axis=-1),
    )
