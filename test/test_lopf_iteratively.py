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


def test_proximal_initial_weight_is_not_relaxed_before_first_linearisation(monkeypatch):
    """The default L2 proximal weight stays at its default after the first solve."""
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
        proximal="l2",
    )

    # Iteration 1 has no anchor. Every anchored solve uses the fixed default
    # L2 weight, despite the normal relax/tighten control flow.
    default = abstract_module.PROXIMAL_INITIAL["l2"]
    assert n.iteration_log.proximal.tolist() == [0.0, default, default]


def test_iterative_invalid_switches_raise():
    n = _build_simple_network()

    with pytest.raises(ValueError, match="scheme must be one of"):
        n.optimize.optimize_transmission_expansion_iteratively(scheme="newton")
    with pytest.raises(ValueError, match="proximal must be one of"):
        n.optimize.optimize_transmission_expansion_iteratively(proximal="l3")
    # the three switches replaced a single combined 'method' setting, and the
    # norm used to be its own argument; neither is accepted any more
    for removed in (dict(method="trust_region"), dict(proximal_norm="l1")):
        with pytest.raises(TypeError, match="has been removed"):
            n.optimize.optimize_transmission_expansion_iteratively(**removed)


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
            scheme=scheme, trust_region=True, proximal="off",
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
        trust_region=True, max_iterations=20, **SOLVER
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
        trust_region=True,
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
        trust_region=True,
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
        # iterations without a linearisation carry the original bounds, and the
        # width is floored by the initial capacity, which breaks the geometric
        # symmetry on purpose. Neither is what this test is about.
        interior = np.isfinite(upper) & (centre >= initial)
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
    for proximal in ("off", "l1", "l2"):
        n = _build_meshed_network()
        n.optimize.optimize_transmission_expansion_iteratively(
            trust_region=True, proximal=proximal, max_iterations=40, **SOLVER
        )
        log = n.iteration_log
        costs[proximal] = float(log.cost.dropna().iloc[-1])
        # only the l1 term needs a deviation variable; l2 goes straight into the
        # objective and makes it quadratic
        assert ("Line-s_nom_deviation" in n.model.variables) == (proximal == "l1")
        assert bool(n.model.objective.is_quadratic) == (proximal == "l2")
        # the weight is adapted, not given, and reported per iteration
        assert (log.proximal.fillna(0.0) > 0.0).any() == (proximal != "off")

    for norm in ("l1", "l2"):
        assert abs(costs[norm] - costs["off"]) / abs(costs["off"]) < 1e-3


def test_l2_drops_numerically_zero_anchor_cross_terms(monkeypatch):
    """A zero anchor or residual tiny L2 summand must not reach the solver."""
    n = _build_meshed_network()
    # This represents a candidate whose capacity is zero up to solver
    # feasibility tolerance. It stays in the voltage-law model but cannot be
    # selected by the first (LP) iteration.
    n.lines.loc["ab", ["s_nom", "s_nom_min", "s_nom_max"]] = 1e-16

    original_solve = linopy.Model.solve
    calls = 0
    captured = {}

    def capture_second_model(model, *args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            captured["data"] = model.objective.data
            raise RuntimeError("captured second iteration")
        return original_solve(model, *args, **kwargs)

    def add_tiny_objective_summand(network, snapshots):
        # Linopy preserves separate summands until model export. This mimics
        # the tiny linear residue that can remain after a real L2 expansion.
        network.model.objective += 2e-13 * network.model["Line-s_nom"].sel(
            **{"Line-ext": "ac"}
        )

    monkeypatch.setattr(linopy.Model, "solve", capture_second_model)
    with pytest.raises(RuntimeError, match="captured second iteration"):
        n.optimize.optimize_transmission_expansion_iteratively(
            max_iterations=3,
            min_iterations=2,
            solver_name="highs",
            solver_options={"log_to_console": False},
            transmission_losses=2,
            extra_functionality=add_tiny_objective_summand,
        )

    labels = captured["data"].vars.values
    coeffs = captured["data"].coeffs.values
    linear = labels[1] == -1
    net = np.bincount(
        labels[0, linear],
        weights=coeffs[linear],
        minlength=int(labels[0, linear].max()) + 1,
    )
    nonzero = np.abs(net)
    nonzero = nonzero[nonzero > 0.0]
    assert nonzero.min() >= 1e-6


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
