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
