import importlib
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


def test_iterative_invalid_method_raises():
    n = _build_simple_network()

    with pytest.raises(ValueError, match="method must be one of"):
        n.optimize.optimize_transmission_expansion_iteratively(method="newton")


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
    for method in ("fixed_point", "trust_region"):
        n = _build_meshed_network(sssc)
        status, condition = n.optimize.optimize_transmission_expansion_iteratively(
            method=method, max_iterations=40, **SOLVER
        )
        assert status == "ok"
        assert condition == "optimal"
        log = n.iteration_log
        residual = log.violation_rel.dropna()
        reached = residual[residual <= 1e-3]
        results[method] = {
            "iterations": len(log),
            "cost": float(log.cost.dropna().iloc[-1]),
            "capacities": _branch_capacities(n),
            "kvl": float(residual.iloc[-1]),
            # iteration at which the solution first becomes consistent with the
            # impedances it assumes, independent of the stopping rule
            "consistent_at": int(reached.index[0]) if len(reached) else 10**6,
        }

    fixed, trust = results["fixed_point"], results["trust_region"]

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
        method="trust_region", max_iterations=20, **SOLVER
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


@pytest.mark.parametrize("method", ["fixed_point", "trust_region"])
def test_convergence_is_decided_by_the_system_cost(method):
    """
    The iteration stops on a stationary system cost, not on the capacity step,
    which is only reported as a diagnostic.
    """
    n = _build_meshed_network()
    n.optimize.optimize_transmission_expansion_iteratively(
        method=method, cost_threshold=1e-4, cost_window=3, max_iterations=60,
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
    for proximal in (False, True):
        n = _build_meshed_network()
        n.optimize.optimize_transmission_expansion_iteratively(
            method="trust_region", proximal=proximal, max_iterations=40, **SOLVER
        )
        log = n.iteration_log
        costs[proximal] = float(log.cost.dropna().iloc[-1])
        assert ("Line-s_nom_deviation" in n.model.variables) == proximal
        # the weight is calibrated, not given, and reported per iteration
        assert (log.proximal.fillna(0.0) > 0.0).any() == proximal

    assert abs(costs[True] - costs[False]) / abs(costs[False]) < 1e-3


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
