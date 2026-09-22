#!/usr/bin/env python3

import pytest


def test_operational_limit_ac_dc_meshed(ac_dc_network):
    n = ac_dc_network.copy()

    limit = 30_000

    n.global_constraints.drop(n.global_constraints.index, inplace=True)

    n.add(
        "GlobalConstraint",
        "gas_limit",
        type="operational_limit",
        carrier_attribute="gas",
        sense="<=",
        constant=limit,
    )

    n.optimize()
    assert n.statistics.energy_balance().loc[:, "gas"].sum().round(3) == limit


def test_operational_limit_storage_hvdc(storage_hvdc_network):
    n = storage_hvdc_network.copy()

    limit = 5_000

    n.global_constraints.drop(n.global_constraints.index, inplace=True)

    n.add(
        "GlobalConstraint",
        "battery_limit",
        type="operational_limit",
        carrier_attribute="battery",
        sense="<=",
        constant=limit,
    )

    n.storage_units["state_of_charge_initial"] = 1_000
    n.storage_units.p_nom_extendable = True
    n.storage_units.cyclic_state_of_charge = False

    n.optimize()

    soc_diff = (
        n.storage_units.state_of_charge_initial.sum()
        - n.storage_units_t.state_of_charge.sum(1).iloc[-1]
    )
    assert soc_diff.round(3) == limit


@pytest.mark.parametrize("assign", [True, False])
def test_assign_all_duals(ac_dc_network, assign):
    n = ac_dc_network.copy()

    limit = 30_000

    m = n.optimize.create_model()

    transmission = m.variables["Link-p"]
    m.add_constraints(
        transmission.sum() <= limit, name="GlobalConstraint-generation_limit"
    )
    m.add_constraints(
        transmission.sum(dim="Link") <= limit,
        name="GlobalConstraint-generation_limit_dynamic",
    )

    n.optimize.solve_model(assign_all_duals=assign)

    assert ("generation_limit" in n.global_constraints.index) == assign
    assert ("mu_generation_limit_dynamic" in n.global_constraints_t) == assign


def _volume_limited_network(limit):
    """Two buses whose only line must be expanded, under a transmission volume cap.

    Relaxing the cap by one MW km buys 1/length MW of line, which displaces the
    expensive generator: the shadow price is known in closed form.
    """
    import pandas as pd

    import pypsa

    n = pypsa.Network()
    n.set_snapshots(pd.date_range("2030-01-01", periods=2, freq="h"))
    n.add("Carrier", "AC")
    n.add("Bus", "b0", carrier="AC", v_nom=380)
    n.add("Bus", "b1", carrier="AC", v_nom=380)
    n.add("Generator", "cheap", bus="b0", p_nom=1000, marginal_cost=1.0)
    n.add("Generator", "dear", bus="b1", p_nom=1000, marginal_cost=50.0)
    n.add("Load", "l", bus="b1", p_set=500.0)
    n.add(
        "Line", "ln", bus0="b0", bus1="b1", carrier="AC", x=0.1, r=0.01,
        length=100.0, s_nom=0.0, s_nom_extendable=True, capital_cost=1.0,
    )
    n.add(
        "GlobalConstraint", "lv_limit", type="transmission_volume_expansion_limit",
        carrier_attribute="AC", sense="<=", constant=limit,
    )
    return n


def test_assign_scaled_global_constraint_dual(monkeypatch):
    """``mu`` is the physical shadow price, with the row scale divided back out."""
    from pypsa.optimization import global_constraints as gc

    def mu_at(scale):
        monkeypatch.setattr(gc, "TRANSMISSION_GLOBAL_CONSTRAINT_SCALE", scale)
        n = _volume_limited_network(25_000)
        n.optimize(assign_all_duals=True)
        raw_dual = float(n.model.constraints["GlobalConstraint-lv_limit"].dual)
        # Without a binding row the assertions below hold for any scaling.
        assert raw_dual != 0
        assert n.global_constraints.at["lv_limit", "mu"] == pytest.approx(
            raw_dual / scale
        )
        return n.global_constraints.at["lv_limit", "mu"]

    # 2 h of displacing a 50/MWh generator by a 1/MWh one, per 1/100 MW of line,
    # less that line's 1/MW capital cost: 2 * 49 / 100 - 1 / 100.
    assert mu_at(1.0) == pytest.approx(-0.97)
    # The scaling is a modelling device; it must not reach the reported price.
    assert mu_at(1e3) == pytest.approx(-0.97)
