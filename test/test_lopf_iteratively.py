import importlib
from pathlib import Path
import sys

import linopy
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
    )

    # Iteration 1 has no anchor. Every anchored solve uses the same default
    # weight; the relax/tighten schedule only moves the radius.
    default = abstract_module.PROXIMAL_WEIGHT
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
def test_trust_region_converges_faster_and_consistently(sssc):
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

    fixed, trust = results["fixed_point"], results["slp"]

    # the linearisation reaches the tolerance in far fewer iterations
    assert trust["iterations"] < fixed["iterations"]
    # and becomes consistent with its own impedances earlier
    assert trust["consistent_at"] <= fixed["consistent_at"]
    assert trust["kvl"] < 1e-4
    # the fixed point ends up in a limit cycle, so it also reports the more
    # expensive plan of the two
    assert trust["cost"] < fixed["cost"]
    # brownfield: existing capacity is never removed
    assert (trust["capacities"] >= 50.0 - 1e-6).all()


def test_trust_region_records_iteration_log_and_restores_bounds():
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
        "radius",
        "alignment",
        "binding",
        "proximal",
        "accepted",
    ]
    assert log.index.name == "iteration"
    assert not log.empty
    # the converged iterate is consistent with the impedances it assumes
    assert log.violation_rel.dropna().iloc[-1] < 1e-4
    # the trust region must not leak into the network
    pd.testing.assert_frame_equal(n.lines[["s_nom_min", "s_nom_max"]], bounds)
    assert n._kvl_capacity_sensitivity is None



def test_radius_follows_the_angle_between_consecutive_steps():
    """
    The radius is steered by the direction of the step and by nothing else: a
    step that reverses its predecessor shrinks it, a run of aligned steps
    against the boundary widens it, and neither happens on the first step,
    which has no predecessor to be compared with.
    """
    n = _build_meshed_network()
    n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=20, **SOLVER
    )

    log = n.iteration_log
    assert np.isnan(log.alignment.iloc[0])
    assert log.alignment.dropna().between(-1.0, 1.0).all()

    shrink, expand = 0.5, 2.0
    for (_, before), (_, after) in zip(log.iloc[:-1].iterrows(), log.iloc[1:].iterrows()):
        if np.isnan(before.alignment) or np.isnan(after.radius):
            continue
        if before.alignment <= -0.8:
            expected = max(before.radius * shrink, 1e-2)
        elif before.alignment >= 0.0 and before.binding:
            expected = min(before.radius * expand, 1.0)
        else:
            expected = before.radius
        assert after.radius == pytest.approx(expected)



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



def test_converged_run_is_also_closed_by_the_report_solve():
    """
    The run that converges is closed by the same pinned report solve as the one
    that runs out of budget. At a converged point the linearisation is exact, so
    that solve reproduces the iterate rather than moving away from it - but it
    is what carries the duals and the derived time series, and it carries
    neither the trust region nor the penalty.
    """
    n = _build_meshed_network()
    bounds = n.lines[["s_nom_min", "s_nom_max"]].copy()

    n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=40, track_iterations=True, **SOLVER
    )

    log = n.iteration_log
    assert len(log) < 40, "the run has to be the converged one"
    ext = n.lines.s_nom_extendable
    # the report solve is held at the converged capacities
    pd.testing.assert_series_equal(
        n.lines.s_nom_opt[ext], n.lines._s_nom_def[ext],
        rtol=1e-9, check_names=False,
    )
    pd.testing.assert_series_equal(
        n.lines.s_nom_opt[ext], n.lines[f"s_nom_opt_{len(log)}"][ext],
        rtol=1e-9, check_names=False,
    )
    # and reproduces the cost the iteration converged to
    assert n.objective == pytest.approx(float(log.cost.dropna().iloc[-1]), rel=1e-9)
    # it is the plain formulation: no step control of either kind survives it
    assert "Line-s_nom_deviation" not in n.model.variables
    assert n._kvl_capacity_sensitivity is None
    pd.testing.assert_frame_equal(n.lines[["s_nom_min", "s_nom_max"]], bounds)
    # and it is a full solve, so the user-facing results are there
    assert not n.buses_t.marginal_price.empty


def test_proximal_target_is_validated():
    n = _build_simple_network()

    with pytest.raises(ValueError, match="proximal_target must be one of"):
        n.optimize.optimize_transmission_expansion_iteratively(
            proximal_target="lines"
        )


def test_proximal_on_the_compensation_warns_where_there_is_none(caplog):
    """A network without an extendable SSSC has nothing for the term to hold."""
    n = _build_meshed_network(sssc=False)

    with caplog.at_level("WARNING"):
        n.optimize.optimize_transmission_expansion_iteratively(
            proximal_target="sssc", max_iterations=20, **SOLVER
        )

    assert any("sssc_nom_extendable" in r.message for r in caplog.records)


def test_sssc_proximal_does_not_move_the_converged_plan():
    """
    On the compensation the ``l2`` term is a step control, not a preference: it
    vanishes with its own gradient where the iterate reproduces its anchor, so
    the plan the iteration settles on must not depend on how hard it charged
    for getting there - as long as the weight stays below the point at which
    its dead zone starts deciding the plan instead.
    """
    plans = {}
    for weight in (1e-4, 1e-3):
        n = _build_meshed_network(sssc=True)
        n.optimize.optimize_transmission_expansion_iteratively(
            proximal_target="sssc",
            proximal_weight=weight,
            max_iterations=40,
            **SOLVER,
        )
        log = n.iteration_log
        plans[weight] = (
            _branch_capacities(n),
            n.line_xs.sssc_nom_opt.astype(float),
            float(log.cost.dropna().iloc[-1]),
            float(log.violation_rel.dropna().iloc[-1]),
        )

    (caps_a, q_a, cost_a, res_a), (caps_b, q_b, cost_b, res_b) = (
        plans[1e-4],
        plans[1e-3],
    )
    # both have to have converged for the comparison to mean anything
    assert res_a < 1e-4 and res_b < 1e-4
    assert cost_b == pytest.approx(cost_a, rel=1e-4)
    pd.testing.assert_series_equal(caps_b, caps_a, rtol=1e-3, check_names=False)
    # the compensation is the quantity the term acts on, so it is the one that
    # would show a bias
    assert float((q_b - q_a).abs().sum()) <= 1e-2 * max(float(q_a.sum()), 1.0)


def test_alignment_thresholds_are_validated():
    n = _build_simple_network()

    for bad in ((-2.0, 0.0), (0.5, -0.5), (-0.8, 1.5)):
        with pytest.raises(ValueError, match="trust_region_alignment"):
            n.optimize.optimize_transmission_expansion_iteratively(
                trust_region_alignment=bad
            )


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
    # The outer solve is lightweight; the mandatory final rerun materialises
    # the complete user-facing network once.
    assert calls == {"solution": 1, "duals": 1, "post_processing": 1}
    if track_iterations:
        assert "s_nom_opt_1" in n.lines
        assert hasattr(n, "objective_1")


def test_trust_region_is_geometrically_symmetric():
    """
    Both parts of a branch term of the voltage law scale with the inverse
    capacity, so in the relative deviation ``u`` the linearisation leaves the
    error ``T u**2 / (1 + u)``, which grows much faster below the linearisation
    point than above it and diverges as the capacity collapses. The trust
    region is therefore symmetric in ``log F`` rather than in ``F``: it widens
    by ``rho`` upwards and by ``rho / (1 + rho)`` downwards, which puts the
    linearisation point at the geometric mean of the region and leaves the same
    error at both of its ends.
    """
    initial = 50.0
    n = _build_meshed_network(s_nom=initial)
    # let the region extend downwards instead of clipping at the brownfield
    # capacity, so that its lower half is visible at all
    n.lines["s_nom_min"] = 0.0

    regions = []

    def record(network: pypsa.Network, snapshots: pd.Index) -> None:
        regions.append(
            network.lines[["_s_nom_def", "s_nom_min", "s_nom_max"]]
            .astype(float)
            .copy()
        )

    n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=20,
        extra_functionality=record,
        **SOLVER,
    )

    checked = 0
    for region in regions:
        centre, lower, upper = (
            region["_s_nom_def"],
            region["s_nom_min"],
            region["s_nom_max"],
        )
        # iterations without a linearisation carry the original bounds, the
        # width is floored by the initial capacity, which breaks the geometric
        # symmetry on purpose, and the solve that closes an unconverged run
        # holds the capacities at the last iterate rather than boxing them.
        # None of the three is what this test is about.
        interior = np.isfinite(upper) & (centre >= initial) & (upper > lower)
        if not interior.any():
            continue
        centre, lower, upper = centre[interior], lower[interior], upper[interior]

        # the linearisation point is the geometric, not the arithmetic, mean
        assert np.allclose(lower * upper, centre**2)
        # hence the region is wider above it and never reaches a zero capacity,
        # where the linearisation error would diverge
        assert ((upper - centre) > (centre - lower)).all()
        assert (lower > 0).all()
        checked += len(centre)

    assert checked > 0


@pytest.mark.parametrize("scheme", ["fixed_point", "slp"])
def test_convergence_is_decided_by_the_system_cost(scheme):
    """
    The iteration stops on a stationary system cost, not on the capacity step,
    which is only reported as a diagnostic.
    """
    n = _build_meshed_network()
    n.optimize.optimize_transmission_expansion_iteratively(
        scheme=scheme, cost_threshold=1e-4, cost_window=3, max_iterations=60,
        **SOLVER,
    )
    log = n.iteration_log
    changes = log.cost_change[log.accepted].dropna()

    if len(log) < 60:  # converged rather than exhausted
        assert (changes.iloc[-3:] <= 1e-4).all()
    assert log.cost.notna().any()
    assert "step" in log


def test_msq_threshold_is_ignored_with_a_warning(caplog):
    n = _build_meshed_network()
    with caplog.at_level("WARNING"):
        n.optimize.optimize_transmission_expansion_iteratively(
            msq_threshold=0.5, max_iterations=6, **SOLVER
        )
    assert any("msq_threshold" in record.message for record in caplog.records)


def test_proximal_term_does_not_distort_the_optimum():
    """
    The proximal term damps the iteration but neither enters the reported cost
    nor moves the optimum noticeably. Only the converging scheme is compared:
    two points of the limit cycle the fixed point ends up in are not optima and
    carry no such statement.
    """
    costs = {}
    for proximal in (0.0, abstract_module.PROXIMAL_WEIGHT):
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
        # the term is modelled with a deviation variable, so the inner problem
        # stays linear whether it is on or off. Iteration 1 has no anchor yet
        # and so carries no penalty either way.
        assert any(present for present, _ in inner[:-1]) == on
        assert not any(quadratic for _, quadratic in inner)
        # the report solve that closes the run carries no penalty either way,
        # so the model left on the network never has the deviation variable
        assert inner[-1][0] is False
        assert "Line-s_nom_deviation" not in n.model.variables
        assert not bool(n.model.objective.is_quadratic)
        # and the weight it was solved with is reported per iteration
        assert (log.proximal.fillna(0.0) > 0.0).any() == on

    off, on = costs[0.0], costs[abstract_module.PROXIMAL_WEIGHT]
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
