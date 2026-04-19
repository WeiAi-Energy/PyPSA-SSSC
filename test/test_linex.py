import numpy as np
import pandas as pd

import pypsa


def make_triangle_network() -> pypsa.Network:
    n = pypsa.Network()
    n.set_snapshots(pd.Index(["now"], name="snapshot"))
    for bus in ["a", "b", "c"]:
        n.add("Bus", bus, v_nom=220.0, carrier="AC")
    n.add("Generator", "gen", bus="a", p_nom=200.0, marginal_cost=10.0)
    n.add("Load", "load", bus="c", p_set=100.0)
    n.add("Line", "ab", bus0="a", bus1="b", x=0.1, r=0.01, s_nom=80.0)
    n.add("Line", "bc", bus0="b", bus1="c", x=0.1, r=0.01, s_nom=80.0)
    n.add(
        "Line",
        "ac",
        bus0="a",
        bus1="c",
        x=0.2,
        r=0.02,
        s_nom=80.0,
        s_nom_extendable=True,
        capital_cost=1.0,
    )
    return n


def test_convert_lines_to_line_x():
    n = pypsa.Network()
    n.set_snapshots(pd.Index(["now"], name="snapshot"))
    for bus in ["a", "b"]:
        n.add("Bus", bus, v_nom=220.0, carrier="AC")
    n.add(
        "Line",
        "ab",
        bus0="a",
        bus1="b",
        x=0.1,
        r=0.01,
        s_nom=100.0,
        capital_cost=2.0,
    )

    converted = n.convert_lines_to_line_x("ab", capital_cost_sssc=7.0)

    assert list(converted) == ["ab"]
    assert "ab" not in n.lines.index
    assert "ab" in n.line_xs.index
    assert n.line_xs.at["ab", "capital_cost"] == 2.0
    assert n.line_xs.at["ab", "capital_cost_sssc"] == 7.0
    assert bool(n.line_xs.at["ab", "sssc_nom_extendable"])


def test_line_x_optimize_records_sssc_results():
    n = make_triangle_network()
    n.convert_lines_to_line_x(
        "ab",
        capital_cost_sssc=0.5,
        sssc_nom_extendable=True,
        sssc_nom_max=50.0,
    )

    status, condition = n.optimize()

    assert status == "ok"
    assert condition == "optimal"
    assert "q_sssc" in n.line_xs_t
    assert "ab" in n.line_xs_t.q_sssc.columns
    assert "sssc_nom_opt" in n.line_xs.columns
    assert np.isfinite(n.line_xs.at["ab", "sssc_nom_opt"])


def test_line_x_statistics_include_sssc_costs():
    n = pypsa.Network()
    n.set_snapshots(pd.Index(["now"], name="snapshot"))
    for bus in ["a", "b"]:
        n.add("Bus", bus, v_nom=220.0, carrier="AC")
    n.add(
        "LineX",
        "ab",
        bus0="a",
        bus1="b",
        x=0.1,
        r=0.01,
        s_nom=6.0,
        s_nom_opt=10.0,
        capital_cost=2.0,
        sssc_nom=1.0,
        sssc_nom_opt=3.0,
        capital_cost_sssc=5.0,
    )

    capex = float(n.statistics.capex(comps=["LineX"]).to_numpy().sum())
    installed = float(n.statistics.installed_capex(comps=["LineX"]).to_numpy().sum())

    assert capex == 35.0
    assert installed == 17.0


def test_line_x_iterative_optimization_runs():
    n = make_triangle_network()
    n.convert_lines_to_line_x(
        "ab",
        capital_cost_sssc=0.5,
        s_nom_extendable=True,
        capital_cost=1.0,
        sssc_nom_extendable=True,
        sssc_nom_max=50.0,
    )

    status, condition = n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=1,
        min_iterations=1,
    )

    assert status == "ok"
    assert condition == "optimal"
    assert "ab" in n.line_xs_t.q_sssc.columns
    assert np.isfinite(n.line_xs.at["ab", "sssc_nom_opt"])


def test_lpf_sssc_matches_opf():
    """
    Verify that LPF with q_sssc from OPF reproduces OPF branch flows.

    After OPF, p0 values for each branch are set by the optimizer.
    Running LPF with the same q_sssc input (and same generator dispatch)
    should reproduce those flows up to DC linearization tolerance.
    """
    n = make_triangle_network()
    n.convert_lines_to_line_x(
        "ab",
        capital_cost_sssc=0.5,
        sssc_nom_extendable=True,
        sssc_nom_max=50.0,
    )

    status, _ = n.optimize()
    assert status == "ok"

    # Record OPF branch flows
    opf_line_xs_p0 = n.line_xs_t.p0.copy()   # LineX (ab)
    opf_lines_p0 = n.lines_t.p0.copy()        # Lines (bc, ac)

    # Feed OPF q_sssc and generator dispatch into LPF
    # q_sssc is already in n.line_xs_t.q_sssc from optimize()
    # Generator p is already in n.generators_t.p from optimize()
    # Set p_set from optimized dispatch so LPF uses the same injections
    for gen in n.generators.index:
        n.generators_t["p_set"] = n.generators_t.p.copy()

    n.lpf()

    lpf_line_xs_p0 = n.line_xs_t.p0
    lpf_lines_p0 = n.lines_t.p0

    print("\n--- OPF vs LPF flow comparison ---")
    print(f"q_sssc (ab): {n.line_xs_t.q_sssc['ab'].values}")
    print(f"LineX ab  OPF p0={opf_line_xs_p0['ab'].values}  LPF p0={lpf_line_xs_p0['ab'].values}")
    for line in n.lines.index:
        print(f"Line  {line}  OPF p0={opf_lines_p0[line].values}  LPF p0={lpf_lines_p0[line].values}")

    np.testing.assert_allclose(
        lpf_line_xs_p0.values, opf_line_xs_p0.values, atol=1e-3,
        err_msg="LineX p0 mismatch between LPF and OPF"
    )
    np.testing.assert_allclose(
        lpf_lines_p0.values, opf_lines_p0.values, atol=1e-3,
        err_msg="Lines p0 mismatch between LPF and OPF"
    )

