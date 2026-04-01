import pandas as pd
import pytest

import pypsa
import pypsa.optimization.optimize as optimize_module

from pypsa.optimization.abstract import (
    RTEP_INNER_EQUIVALENT_CARRIER,
    RTEP_INNER_LOAD_SUFFIX,
)


def test_optimize_post_discretization():
    n = pypsa.Network()

    n.madd("Bus", ["a", "b", "c"], v_nom=380.0)
    n.add("Generator", "generator", bus="a", p_nom=900.0, marginal_cost=10.0)
    n.add("Load", "load", bus="c", p_set=900.0)
    n.add(
        "Line",
        "ab",
        bus0="a",
        bus1="b",
        x=0.0001,
        s_nom_extendable=True,
        capital_cost=1000,
    )
    n.add(
        "Link",
        "bc",
        bus0="b",
        bus1="c",
        p_nom_extendable=True,
        capital_cost=1000,
        carrier="HVDC",
    )

    line_unit_size = 500
    link_unit_size = dict(HVDC=600)

    status, _ = n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=1,
        line_unit_size=line_unit_size,
        link_unit_size=link_unit_size,
        link_threshold=dict(HVDC=0.4),
    )

    assert status == "ok"
    assert all(n.lines.query("s_nom_extendable").s_nom_opt % line_unit_size == 0.0)
    assert all(
        n.links.query("p_nom_extendable and carrier == 'HVDC'").p_nom_opt
        % link_unit_size["HVDC"]
        == 0.0
    )


def test_iterative_optimization_preserves_extra_functionality_attrs():
    n = pypsa.Network()

    n.madd("Bus", ["a", "b", "c"], v_nom=380.0)
    n.add("Generator", "generator", bus="a", p_nom=900.0, marginal_cost=10.0)
    n.add("Load", "load", bus="c", p_set=900.0)
    n.add(
        "Line",
        "ab",
        bus0="a",
        bus1="b",
        x=0.0001,
        s_nom_extendable=True,
        capital_cost=1000,
    )
    n.add(
        "Line",
        "bc",
        bus0="b",
        bus1="c",
        x=0.0001,
        s_nom=900.0,
    )

    n.opts = ["unit-test"]
    n.config = {"section": {"enabled": True}}
    seen_network_ids = []

    def extra_functionality(network, snapshots):
        assert network.opts == n.opts
        assert network.config == n.config
        seen_network_ids.append(id(network))

    status, condition = n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=1,
        min_iterations=1,
        extra_functionality=extra_functionality,
    )

    assert status == "ok"
    assert condition == "optimal"
    assert len(seen_network_ids) >= 2
    assert all(network_id == id(n) for network_id in seen_network_ids)


def test_iterative_inner_optimization_sets_gurobi_numeric_focus(monkeypatch):
    n = pypsa.Network()

    n.madd("Bus", ["a", "b"], v_nom=380.0, carrier="AC")
    n.add("Generator", "generator", bus="a", p_nom=200.0, marginal_cost=10.0)
    n.add("Load", "load", bus="b", p_set=100.0)
    n.add(
        "Line",
        "ab",
        bus0="a",
        bus1="b",
        x=0.0001,
        r=0.0001,
        s_nom=100.0,
        s_nom_extendable=True,
        capital_cost=1.0,
    )

    captured: dict[str, object] = {}

    def fake_optimize(network, snapshots=None, *args, **kwargs):
        network.objective = 0.0

        if network is n:
            network.lines["s_nom_opt"] = network.lines["s_nom"] * 2
            return "ok", "optimal"

        captured["solver_name"] = kwargs.get("solver_name", "highs")
        captured["solver_options"] = dict(kwargs.get("solver_options", {}))
        network.lines["s_nom_opt"] = network.lines["s_nom"]
        return "ok", "optimal"

    monkeypatch.setattr(optimize_module, "optimize", fake_optimize)

    status, condition = n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=1,
        min_iterations=1,
        solver_name="gurobi",
        solver_options={
            "BarConvTol": 1e-8,
            "NumericFocus": 1,
            "AggFill": 0,
            "PreDual": 0,
        },
    )

    assert status == "ok"
    assert condition == "optimal"
    assert captured["solver_name"] == "gurobi"
    assert captured["solver_options"]["NumericFocus"] == 3
    assert "BarConvTol" not in captured["solver_options"]
    assert "AggFill" not in captured["solver_options"]
    assert "PreDual" not in captured["solver_options"]


def test_iterative_inner_failure_reuses_outer_capacities(monkeypatch):
    n = pypsa.Network()

    n.madd("Bus", ["a", "b"], v_nom=380.0, carrier="AC")
    n.add("Generator", "generator", bus="a", p_nom=200.0, marginal_cost=10.0)
    n.add("Load", "load", bus="b", p_set=100.0)
    n.add(
        "Line",
        "ab",
        bus0="a",
        bus1="b",
        x=0.0001,
        r=0.0001,
        s_nom=100.0,
        s_nom_extendable=True,
        capital_cost=1.0,
    )

    outer_caps_seen = []
    outer_targets = [150.0, 180.0]
    outer_call_count = 0

    def fake_optimize(network, snapshots=None, *args, **kwargs):
        nonlocal outer_call_count
        network.objective = 0.0

        if network is n and network.lines.s_nom_extendable.any():
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

        if network is not n:
            return "warning", "infeasible"

        network.lines["s_nom_opt"] = network.lines["s_nom"]
        return "ok", "optimal"

    monkeypatch.setattr(optimize_module, "optimize", fake_optimize)

    status, condition = n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=2,
        min_iterations=2,
    )

    assert status == "ok"
    assert condition == "optimal"
    assert outer_caps_seen[:2] == [100.0, 150.0]


def test_iterative_inner_network_uses_ac_equivalent_subnetwork(monkeypatch):
    n = pypsa.Network()
    n.set_snapshots(pd.Index(["s1", "s2"], name="snapshot"))

    n.madd("Bus", ["a", "b", "c"], v_nom=220.0, carrier="AC")
    n.add("Bus", "d", v_nom=220.0, carrier="DC")
    n.add("Generator", "gen_a", bus="a", p_nom=200.0, marginal_cost=10.0)
    n.add("Load", "load_c", bus="c", p_set=[80.0, 60.0])
    n.add("Load", "load_d", bus="d", p_set=[30.0, 20.0])
    n.add(
        "StorageUnit",
        "su_b",
        bus="b",
        p_nom=20.0,
        max_hours=2.0,
        efficiency_store=1.0,
        efficiency_dispatch=1.0,
        state_of_charge_initial=20.0,
        cyclic_state_of_charge=False,
        marginal_cost=0.0,
    )
    n.add("Link", "bd", bus0="b", bus1="d", p_nom=60.0, efficiency=1.0)
    n.add(
        "Line",
        "ab",
        bus0="a",
        bus1="b",
        x=0.1,
        r=0.01,
        s_nom=120.0,
        s_nom_extendable=True,
        capital_cost=1.0,
    )
    n.add("Transformer", "bc", bus0="b", bus1="c", x=0.1, r=0.01, s_nom=120.0)

    captured: dict[str, object] = {}

    original_optimize = optimize_module.optimize

    def recording_optimize(network, snapshots=None, *args, **kwargs):
        status, condition = original_optimize(network, snapshots, *args, **kwargs)
        if network is n or captured:
            return status, condition

        captured["buses"] = network.buses.index.copy()
        captured["lines"] = network.lines.index.copy()
        captured["transformers"] = network.transformers.index.copy()
        captured["links"] = network.links.index.copy()
        captured["storage_units"] = network.storage_units.index.copy()
        captured["global_constraints"] = network.global_constraints.copy()
        captured["loads"] = network.loads.copy()
        captured["load_p_set"] = network.loads_t.p_set.copy()
        expected_net = n.buses_t.p.reindex(
            index=snapshots, columns=network.buses.index, fill_value=0.0
        ).copy()
        trafo_i = n.transformers.index[
            n.transformers.bus0.isin(network.buses.index)
            | n.transformers.bus1.isin(network.buses.index)
        ]
        if not trafo_i.empty:
            trafo_p0 = n.transformers_t.p0.reindex(
                index=snapshots, columns=trafo_i, fill_value=0.0
            )
            trafo_p1 = n.transformers_t.p1.reindex(
                index=snapshots, columns=trafo_i, fill_value=0.0
            )
            bus0 = n.transformers.bus0.reindex(trafo_i)
            bus1 = n.transformers.bus1.reindex(trafo_i)
            expected_net = expected_net.add(
                (-trafo_p0).rename(columns=bus0).T.groupby(level=0).sum().T,
                fill_value=0.0,
            )
            expected_net = expected_net.add(
                (-trafo_p1).rename(columns=bus1).T.groupby(level=0).sum().T,
                fill_value=0.0,
            )
        captured["expected_net"] = expected_net.reindex(
            columns=network.buses.index, fill_value=0.0
        )
        return status, condition

    monkeypatch.setattr(optimize_module, "optimize", recording_optimize)

    status, condition = n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=1,
        min_iterations=1,
    )

    assert status == "ok"
    assert condition == "optimal"
    assert list(captured["buses"]) == ["a", "b", "c"]
    assert list(captured["lines"]) == ["ab"]
    assert captured["transformers"].empty
    assert captured["links"].empty
    assert captured["storage_units"].empty
    assert captured["global_constraints"].empty

    loads = captured["loads"]
    expected_load_names = pd.Index(
        [f"{bus}{RTEP_INNER_LOAD_SUFFIX}" for bus in captured["buses"]]
    )

    assert n.generators.index.isin(loads.index).sum() == 0
    assert loads.index.equals(expected_load_names)
    assert captured["loads"].index.equals(expected_load_names)
    assert captured["loads"].index.intersection(n.generators.index).empty
    assert loads.index.equals(expected_load_names)
    assert (loads.carrier == RTEP_INNER_EQUIVALENT_CARRIER).all()

    expected_net = captured["expected_net"]
    expected_load_p_set = (-expected_net).rename(
        columns=lambda bus: f"{bus}{RTEP_INNER_LOAD_SUFFIX}"
    )

    pd.testing.assert_frame_equal(
        captured["load_p_set"].reindex(columns=expected_load_names),
        expected_load_p_set.reindex(columns=expected_load_names),
    )


def test_iterative_inner_network_distributes_bus_injection_residual(
    monkeypatch,
):
    n = pypsa.Network()
    n.set_snapshots(pd.Index(["now"], name="snapshot"))

    n.madd("Bus", ["a", "b"], v_nom=220.0, carrier="AC")
    n.add("Generator", "gen_a", bus="a", p_nom=200.0, marginal_cost=10.0)
    n.add("Load", "load_b", bus="b", p_set=100.0)
    n.add(
        "Line",
        "ab",
        bus0="a",
        bus1="b",
        x=0.1,
        r=0.01,
        s_nom=120.0,
        s_nom_extendable=True,
        capital_cost=1.0,
    )

    captured: dict[str, object] = {}

    def fake_optimize(network, snapshots=None, *args, **kwargs):
        network.objective = 0.0

        if network is n:
            network.lines["s_nom_opt"] = network.lines["s_nom"] + 10.0
            network.buses_t.p = pd.DataFrame(
                [[50.0, -49.9]],
                index=snapshots,
                columns=["a", "b"],
            )
            return "ok", "optimal"

        if not captured:
            captured["loads"] = network.loads.copy()
            captured["load_p_set"] = network.loads_t.p_set.copy()
            captured["balancing"] = network._rtep_inner_injection_balancing.copy()
            captured["generators"] = network.generators.copy()

        network.lines["s_nom_opt"] = network.lines["s_nom"]
        return "ok", "optimal"

    monkeypatch.setattr(optimize_module, "optimize", fake_optimize)

    status, condition = n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=1,
        min_iterations=1,
    )

    assert status == "ok"
    assert condition == "optimal"
    assert captured["generators"].empty

    expected_load_names = pd.Index(
        [f"{bus}{RTEP_INNER_LOAD_SUFFIX}" for bus in ["a", "b"]]
    )
    expected_load_p_set = pd.DataFrame(
        [[-49.95, 49.95]],
        index=n.snapshots,
        columns=expected_load_names,
    )

    pd.testing.assert_frame_equal(
        captured["load_p_set"].reindex(columns=expected_load_names),
        expected_load_p_set,
    )

    balancing = captured["balancing"]
    assert list(balancing["sub_network"]) == ["0"]
    assert list(balancing["bus_count"]) == [2]
    assert balancing["max_abs_share"].iloc[0] == pytest.approx(0.05)


def test_iterative_inner_network_keeps_ac_losses(monkeypatch):
    zero_tol = 1e-2
    n = pypsa.Network()
    n.set_snapshots(pd.Index(["now"], name="snapshot"))
    n.madd("Bus", ["a", "b", "c"], v_nom=220.0, carrier="AC")
    n.add("Generator", "gen", bus="a", p_nom=200.0, marginal_cost=10.0)
    n.add("Load", "load", bus="c", p_set=100.0)
    n.add(
        "Line",
        "ab",
        bus0="a",
        bus1="b",
        x=0.1,
        r=0.01,
        s_nom=120.0,
        capital_cost=1.0,
    )
    n.add(
        "Line",
        "bc",
        bus0="b",
        bus1="c",
        x=0.1,
        r=0.01,
        s_nom=80.0,
        s_nom_extendable=True,
        capital_cost=1.0,
    )
    n.convert_lines_to_line_x("ab", capital_cost_sssc=0.5, sssc_nom_extendable=True)

    captured: dict[str, object] = {}

    original_optimize = optimize_module.optimize

    def recording_optimize(network, snapshots=None, *args, **kwargs):
        status, condition = original_optimize(network, snapshots, *args, **kwargs)
        if network is n or captured:
            return status, condition

        captured["has_line_loss"] = "Line-loss" in network.model.variables
        captured["has_linex_loss"] = "LineX-loss" in network.model.variables
        captured["buses"] = network.buses.index.copy()
        captured["load_p_set"] = network.loads_t.p_set.copy()
        captured["loads"] = network.loads.copy()
        expected_net = n.buses_t.p.reindex(
            index=snapshots, columns=network.buses.index, fill_value=0.0
        )
        allocated_losses = pd.DataFrame(
            0.0, index=snapshots, columns=network.buses.index
        )
        for c in ("Line", "LineX"):
            if f"{c}-loss" not in n.model.variables or n.df(c).empty:
                continue
            loss = n.model[f"{c}-loss"].solution.to_pandas().reindex(
                index=snapshots, columns=n.df(c).index, fill_value=0.0
            )
            loss = loss.where(loss.abs() >= zero_tol, 0.0)
            bus0 = n.df(c).bus0.reindex(loss.columns)
            bus1 = n.df(c).bus1.reindex(loss.columns)
            allocated_losses = allocated_losses.add(
                (0.5 * loss).rename(columns=bus0).T.groupby(level=0).sum().T,
                fill_value=0.0,
            )
            allocated_losses = allocated_losses.add(
                (0.5 * loss).rename(columns=bus1).T.groupby(level=0).sum().T,
                fill_value=0.0,
            )
        expected_net = expected_net.sub(allocated_losses, fill_value=0.0)
        captured["expected_net"] = expected_net.where(
            expected_net.abs() >= zero_tol, 0.0
        )
        return status, condition

    monkeypatch.setattr(optimize_module, "optimize", recording_optimize)

    status, condition = n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=1,
        min_iterations=1,
        transmission_losses=2,
    )

    assert status == "ok"
    assert condition == "optimal"
    assert captured["has_line_loss"]
    assert captured["has_linex_loss"]
    assert captured["loads"].index.equals(
        pd.Index([f"{bus}{RTEP_INNER_LOAD_SUFFIX}" for bus in captured["buses"]])
    )
    assert (captured["loads"].carrier == RTEP_INNER_EQUIVALENT_CARRIER).all()

    expected_net = captured["expected_net"]
    expected_load_names = pd.Index(
        [f"{bus}{RTEP_INNER_LOAD_SUFFIX}" for bus in captured["buses"]]
    )
    expected_load_p_set = (-expected_net).rename(
        columns=lambda bus: f"{bus}{RTEP_INNER_LOAD_SUFFIX}"
    )

    pd.testing.assert_frame_equal(
        captured["load_p_set"].reindex(columns=expected_load_names),
        expected_load_p_set.reindex(columns=expected_load_names),
    )


def test_iterative_inner_network_zeroes_small_losses_and_bus_injection(monkeypatch):
    n = pypsa.Network()
    n.set_snapshots(pd.Index(["now"], name="snapshot"))
    n.madd("Bus", ["a", "b"], v_nom=220.0, carrier="AC")
    n.add("Generator", "gen", bus="a", p_nom=1.0, marginal_cost=10.0)
    n.add("Load", "load", bus="b", p_set=1e-3)
    n.add(
        "Line",
        "ab",
        bus0="a",
        bus1="b",
        x=0.1,
        r=0.01,
        s_nom=1.0,
        s_nom_extendable=True,
        capital_cost=1.0,
    )

    captured: dict[str, object] = {}
    original_optimize = optimize_module.optimize

    def recording_optimize(network, snapshots=None, *args, **kwargs):
        status, condition = original_optimize(network, snapshots, *args, **kwargs)
        if network is n or captured:
            return status, condition

        captured["load_p_set"] = network.loads_t.p_set.copy()
        captured["fixed_losses"] = {
            component: losses.copy()
            for component, losses in getattr(
                network, "_rtep_inner_fixed_branch_losses", {}
            ).items()
        }
        return status, condition

    monkeypatch.setattr(optimize_module, "optimize", recording_optimize)

    status, condition = n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=1,
        min_iterations=1,
        transmission_losses=1,
    )

    assert status == "ok"
    assert condition == "optimal"
    assert (captured["load_p_set"].abs() < 1e-12).all().all()
    assert "Line" in captured["fixed_losses"]
    assert (captured["fixed_losses"]["Line"].abs() < 1e-12).all().all()


def test_iterative_inner_network_has_no_slack_generators(monkeypatch):
    n = pypsa.Network()
    n.set_snapshots(pd.Index(["now"], name="snapshot"))

    buses = [f"b{i:02d}" for i in range(41)]
    n.madd("Bus", buses, v_nom=220.0, carrier="AC")
    n.add("Generator", "gen", bus=buses[0], p_nom=5000.0, marginal_cost=10.0)
    n.add("Load", "load", bus=buses[-1], p_set=100.0)

    for i, (bus0, bus1) in enumerate(zip(buses[:-1], buses[1:])):
        n.add(
            "Line",
            f"l{i:02d}",
            bus0=bus0,
            bus1=bus1,
            x=0.1,
            r=0.01,
            s_nom=120.0,
            s_nom_extendable=True,
            capital_cost=1.0,
        )

    captured: dict[str, object] = {}

    def fake_optimize(network, snapshots=None, *args, **kwargs):
        network.objective = 0.0

        if network is n:
            network.lines["s_nom_opt"] = network.lines["s_nom"] * 2.0
            return "ok", "optimal"

        if "generators" not in captured:
            captured["generators"] = network.generators.copy()
            captured["loads"] = network.loads.copy()

        network.lines["s_nom_opt"] = network.lines["s_nom"]
        return "ok", "optimal"

    monkeypatch.setattr(optimize_module, "optimize", fake_optimize)

    status, condition = n.optimize.optimize_transmission_expansion_iteratively(
        max_iterations=1,
        min_iterations=1,
    )

    assert status == "ok"
    assert condition == "optimal"

    assert captured["generators"].empty
    assert len(captured["loads"]) == len(buses)
