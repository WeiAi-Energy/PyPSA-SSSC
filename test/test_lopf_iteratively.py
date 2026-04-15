import importlib
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
for module_name in list(sys.modules):
    if module_name == "pypsa" or module_name.startswith("pypsa."):
        del sys.modules[module_name]

pypsa = importlib.import_module("pypsa")
optimize_module = importlib.import_module("pypsa.optimization.optimize")


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


def test_iterative_invalid_relaxation_factor_raises():
    n = _build_simple_network()

    with pytest.raises(ValueError, match="relaxation_factor"):
        n.optimize.optimize_transmission_expansion_iteratively(
            relaxation_factor=-0.1
        )
