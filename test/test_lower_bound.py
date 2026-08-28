"""
Lower bounds for capacity-dependent transmission expansion.

Checks that the lifted relaxation of ``pypsa.optimization.lower_bound`` really
is a relaxation (it never cuts off an attainable plan), that it collapses onto
the exact problem where it has to, and that it certifies the trust-region
solution of the meshed brownfield expansion of ``test_lopf_iteratively``.
"""

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for module_name in list(sys.modules):
    if module_name == "pypsa" or module_name.startswith("pypsa."):
        del sys.modules[module_name]

pypsa = importlib.import_module("pypsa")
lower_bound = importlib.import_module("pypsa.optimization.lower_bound")

SOLVER = dict(solver_name="gurobi", OutputFlag=0)
NETWORK_SOLVER = dict(solver_name="gurobi", solver_options={"OutputFlag": 0})


def _build_meshed_network(sssc: bool = False, s_nom: float = 50.0) -> pypsa.Network:
    """
    The network of ``test_lopf_iteratively``: four buses in a ring with a
    chord, hence two independent cycles, supplied by a cheap generator far from
    the load, with brownfield expansion of every line.
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
        cap = 0.3 * n.lines.x_pu_eff * n.lines.s_nom**2
        n.convert_lines_to_line_x(
            n.lines.index,
            sssc_nom_extendable=True,
            sssc_nom_max=cap,
            capital_cost_sssc=0.05 * n.lines.capital_cost * n.lines.s_nom / cap,
        )
    return n


def _trust_region_plan(n: pypsa.Network) -> pd.Series:
    n = n.copy()
    status, condition = n.optimize.optimize_transmission_expansion_iteratively(
        trust_region=True, max_iterations=40, **NETWORK_SOLVER
    )
    assert (status, condition) == ("ok", "optimal")
    return lower_bound.branch_series(n, "s_nom_opt").astype(float)


# --------------------------------------------------------------------------- #
# the constants of the voltage law
# --------------------------------------------------------------------------- #
def test_branch_constants_pick_out_the_nonlinear_branches():
    n = _build_meshed_network()
    constants = lower_bound.branch_constants(n)

    assert set(constants.index.get_level_values("name")) == {
        "ab", "bc", "cd", "da", "ac"
    }
    # every line is extendable, AC and in a cycle, so every term is lifted
    assert constants["lift"].all()
    # a = x_pu_eff * F is the invariant of the capacity scaling
    n.calculate_dependent_values()
    expected = n.lines.x_pu_eff * n.lines.s_nom
    assert np.allclose(constants["a"].droplevel("component"), expected)


def test_branch_constants_follow_the_linearisation_point():
    """
    ``a`` has to be read off the capacity the impedance belongs to, not off
    ``s_nom``, so that a network that has already been through the iteration
    gives the same constants as the one that went into it.
    """
    n = _build_meshed_network()
    entry = lower_bound.branch_constants(n)["a"]

    iterated = n.copy()
    caps = lower_bound.branch_series(n, "s_nom") * 3.0
    lower_bound.set_capacity_dependent_parameters(iterated, caps)

    assert np.allclose(iterated.lines.x_pu_eff, n.lines.x_pu_eff / 3.0)
    assert np.allclose(lower_bound.branch_constants(iterated)["a"], entry)


def test_branch_constants_reject_multi_period():
    n = _build_meshed_network()
    n._multi_invest = True
    with pytest.raises(NotImplementedError):
        lower_bound.branch_constants(n)


# --------------------------------------------------------------------------- #
# the capacity box
# --------------------------------------------------------------------------- #
def test_capacity_box_contains_every_plan_that_attains_the_cutoff():
    n = _build_meshed_network()
    plan = _trust_region_plan(n)
    cost, _ = lower_bound.evaluate_plan(n, plan, **SOLVER)

    lower, upper = lower_bound.capacity_box(n, cost)

    assert (lower <= plan + 1e-9).all()
    assert (plan <= upper + 1e-9).all()
    # spending the whole budget on one branch is what bounds it
    capital = lower_bound.branch_series(n, "capital_cost")
    slack = cost - float((capital * lower).sum())
    assert np.allclose(upper, lower + slack / capital)


def test_capacity_box_rejects_an_unattainable_cutoff():
    n = _build_meshed_network()
    with pytest.raises(ValueError, match="below the capital cost"):
        lower_bound.capacity_box(n, 1.0)


def test_capacity_box_needs_a_finite_bound():
    n = _build_meshed_network()
    n.lines["capital_cost"] = 0.0
    with pytest.raises(ValueError, match="no finite capacity upper bound"):
        lower_bound.capacity_box(n, 1e6)


# --------------------------------------------------------------------------- #
# the relaxation
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sssc", [False, True])
def test_relaxation_is_exact_on_a_pinned_box(sssc):
    """
    With ``F_low == F_up`` the two pairs of McCormick rows collapse onto the
    bilinear equality itself, so the relaxation is no longer a relaxation: its
    optimum has to be the cost of that very plan, evaluated at the impedances
    the plan implies. This is what ties the lifted formulation to the exact
    voltage law of :mod:`pypsa.optimization.constraints`.
    """
    n = _build_meshed_network(sssc)
    plan = _trust_region_plan(n)

    exact, _ = lower_bound.evaluate_plan(n, plan, **SOLVER)
    relaxed, _, _ = lower_bound.relaxation_bound(n, plan, plan, **SOLVER)

    assert relaxed == pytest.approx(exact, rel=1e-7)


@pytest.mark.parametrize("sssc", [False, True])
def test_relaxation_never_cuts_off_an_attainable_plan(sssc):
    n = _build_meshed_network(sssc)
    plan = _trust_region_plan(n)
    cost, _ = lower_bound.evaluate_plan(n, plan, **SOLVER)
    lower, upper = lower_bound.capacity_box(n, cost)

    bound, _, _ = lower_bound.relaxation_bound(n, lower, upper, cutoff=cost, **SOLVER)
    assert bound <= cost + 1e-6

    # and it stays a lower bound for plans other than the one it was built from
    for factor in (1.1, 1.4):
        other = (plan * factor).clip(upper=upper)
        assert lower_bound.evaluate_plan(n, other, **SOLVER)[0] >= bound - 1e-6


def test_relaxation_replaces_the_voltage_law_rather_than_adding_to_it():
    n = _build_meshed_network()
    lower, upper = lower_bound.capacity_box(n, 1e6)
    _, model, lifted = lower_bound.build_relaxation(n, lower, upper)

    assert not [name for name in model.constraints if name.startswith("Kirchhoff")]
    assert "kvl-lifted" in model.constraints
    assert lifted.n_lifted == len(n.lines)
    for tag in ("lo1", "lo2", "up1", "up2"):
        assert f"kvl-mccormick-{tag}" in model.constraints


def test_relaxation_of_a_radial_network_has_nothing_to_lift():
    n = pypsa.Network()
    n.madd("Bus", ["a", "b"], v_nom=220.0, carrier="AC")
    n.add("Generator", "gen", bus="a", p_nom=200.0, marginal_cost=10.0)
    n.add("Load", "load", bus="b", p_set=100.0)
    n.add("Line", "ab", bus0="a", bus1="b", x=0.1, r=0.01, s_nom=100.0,
          s_nom_extendable=True, capital_cost=1.0)

    lower, upper = lower_bound.capacity_box(n, 1e4)
    bound, _, lifted = lower_bound.relaxation_bound(n, lower, upper, **SOLVER)

    # no cycles, hence nothing to lift, no nonconvexity and no gap: the bound
    # is the optimum of the ordinary problem over the same box
    assert lifted.n_lifted == 0
    reference = n.copy()
    reference.lines["s_nom_min"] = float(lower[("Line", "ab")])
    reference.lines["s_nom_max"] = float(upper[("Line", "ab")])
    reference.optimize(**NETWORK_SOLVER)
    assert bound == pytest.approx(float(reference.model.objective.value), rel=1e-7)


# --------------------------------------------------------------------------- #
# bound tightening
# --------------------------------------------------------------------------- #
def test_tightening_shrinks_the_box_and_lifts_the_bound():
    n = _build_meshed_network()
    plan = _trust_region_plan(n)
    cost, _ = lower_bound.evaluate_plan(n, plan, **SOLVER)
    lower, upper = lower_bound.capacity_box(n, cost)

    plain, _, _ = lower_bound.relaxation_bound(n, lower, upper, cutoff=cost, **SOLVER)
    tight_lower, tight_upper, history = lower_bound.tighten_capacity_box(
        n, lower, upper, cost, rounds=3, **SOLVER
    )

    assert history
    # the box only ever shrinks, and still holds the plan that attains the cutoff
    assert (tight_lower >= lower - 1e-6).all()
    assert (tight_upper <= upper + 1e-6).all()
    assert (tight_lower <= plan + 1e-6).all()
    assert (plan <= tight_upper + 1e-6).all()
    # the bound improves monotonically and never passes the plan
    bounds = [plain] + [z for _, z, _ in history]
    assert all(b <= nxt + 1e-6 for b, nxt in zip(bounds, bounds[1:]))
    assert bounds[-1] <= cost + 1e-6
    assert bounds[-1] > plain
    # the widest interval narrows, which is what the bound is bought with
    widths = [w for _, _, w in history]
    assert widths[-1] < float((upper - lower).max())


# --------------------------------------------------------------------------- #
# the plan itself
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sssc", [False, True])
def test_evaluate_plan_returns_a_consistent_point(sssc):
    """
    The re-costed network has to satisfy the invariant ``x_pu_eff * F = a`` on
    every branch, which is exactly what a non-converged iteration does not.
    """
    n = _build_meshed_network(sssc)
    plan = _trust_region_plan(n)
    entry = lower_bound.branch_constants(n)["a"]

    _, pinned = lower_bound.evaluate_plan(n, plan, **SOLVER)
    caps = lower_bound.branch_series(pinned, "s_nom_opt")
    weight = lower_bound.branch_constants(pinned)["w"]

    assert np.allclose(caps.reindex(entry.index), plan.reindex(entry.index), atol=1e-6)
    assert np.allclose(weight * caps.reindex(weight.index), entry.reindex(weight.index))


# --------------------------------------------------------------------------- #
# the certificate
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("sssc", [False, True])
def test_trust_region_solution_is_globally_optimal_on_the_meshed_network(sssc):
    """
    The converged trust-region plan is not merely a KKT point here: adding the
    bilinear equalities back and letting Gurobi branch on the capacities
    reproduces it, so its gap to the certified global optimum closes.
    """
    pytest.importorskip("gurobipy")
    n = _build_meshed_network(sssc)
    plan = _trust_region_plan(n)

    certificate = n.optimize.certify_expansion(
        plan,
        obbt_rounds=2,
        spatial=True,
        spatial_kwargs=dict(time_limit=600.0, mip_gap=1e-7),
        **SOLVER,
    )

    assert certificate.mccormick_bound <= certificate.lower_bound
    assert certificate.lower_bound <= certificate.upper_bound + 1e-6
    assert certificate.spatial["closed"]
    assert certificate.gap < 1e-5
    # and the certified optimum is the plan itself
    optimum = certificate.spatial["capacities"]
    assert np.allclose(optimum, plan.reindex(optimum.index), atol=1e-4)
    assert str(certificate).startswith("candidate plan")


def test_certificate_defaults_to_the_plan_in_the_network():
    n = _build_meshed_network()
    solved = n.copy()
    status, condition = solved.optimize.optimize_transmission_expansion_iteratively(
        trust_region=True, max_iterations=40, **NETWORK_SOLVER
    )
    assert (status, condition) == ("ok", "optimal")

    certificate = lower_bound.certify_expansion(solved, obbt_rounds=0, **SOLVER)
    plan = lower_bound.branch_series(solved, "s_nom_opt")

    assert np.allclose(certificate.capacities, plan)
    assert certificate.lower_bound == certificate.mccormick_bound
    assert certificate.lower_bound <= certificate.upper_bound + 1e-6
    assert certificate.spatial is None
