#!/usr/bin/env python3

import pytest

import pypsa


def make_network():
    n = pypsa.Network(snapshots=range(1))
    n.add("Bus", "bus")
    n.add(
        "Generator",
        "gen",
        bus="bus",
        p_nom=10,
        p_nom_extendable=True,
        capital_cost=5,
        marginal_cost=0,
    )
    n.add("Load", "load", bus="bus", p_set=12)
    return n


def test_include_objective_constant_false_preserves_solution_and_objective():
    n_with_constant = make_network()
    n_without_constant = make_network()

    status_with, condition_with = n_with_constant.optimize(
        include_objective_constant=True
    )
    status_without, condition_without = n_without_constant.optimize(
        include_objective_constant=False
    )

    assert (status_with, condition_with) == ("ok", "optimal")
    assert (status_without, condition_without) == ("ok", "optimal")

    assert "objective_constant" in n_with_constant.model.variables
    assert "objective_constant" not in n_without_constant.model.variables

    assert n_with_constant.objective_constant == pytest.approx(50.0)
    assert n_without_constant.objective_constant == pytest.approx(50.0)

    assert n_with_constant.generators.at["gen", "p_nom_opt"] == pytest.approx(12.0)
    assert n_without_constant.generators.at["gen", "p_nom_opt"] == pytest.approx(12.0)

    assert n_with_constant.objective == pytest.approx(10.0)
    assert n_without_constant.objective == pytest.approx(10.0)


def test_include_objective_constant_defaults_to_false():
    n = make_network()

    status, condition = n.optimize()

    assert (status, condition) == ("ok", "optimal")
    assert "objective_constant" not in n.model.variables
    assert n.objective_constant == pytest.approx(50.0)
    assert n.objective == pytest.approx(10.0)
    assert n.model.objective.value == pytest.approx(60.0)
