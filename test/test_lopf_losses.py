import pytest

import pypsa

R_REF, S_REF = 0.01, 100.0  # rho = r * s_nom, the conductor-scaling invariant


def _two_bus(extendable, s_nom=S_REF, r=R_REF):
    n = pypsa.Network()
    n.set_snapshots([0])
    n.madd("Bus", ["a", "b"], v_nom=220.0, carrier="AC")
    n.add("Generator", "cheap", bus="a", p_nom=500.0, marginal_cost=10.0)
    n.add("Generator", "expensive", bus="b", p_nom=500.0, marginal_cost=200.0)
    n.add("Load", "load", bus="b", p_set=150.0)
    n.add(
        "Line",
        "ab",
        bus0="a",
        bus1="b",
        x=0.1,
        r=r,
        s_nom=s_nom,
        s_nom_extendable=extendable,
        s_nom_min=50.0,
        capital_cost=5.0,
    )
    return n


@pytest.mark.parametrize("transmission_losses", [1, 2, 4])
def test_losses_are_exact_in_capacity(transmission_losses):
    """
    The loss linearisation is exact in s_nom: because r scales as rho / s_nom
    and the fitting range as s_max_pu * s_nom, the piecewise slopes are
    capacity-independent and the offsets linear in s_nom. Optimising with a
    free capacity must therefore give the same losses as re-optimising with the
    capacity frozen at the optimum and the conductor rescaled accordingly - no
    outer iteration on r needed.
    """
    n = _two_bus(extendable=True)
    n.optimize(transmission_losses=transmission_losses)
    s_nom_opt = float(n.lines.s_nom_opt.iloc[0])
    free_flow = float(n.lines_t.p0.iloc[0, 0])
    free_loss = free_flow + float(n.lines_t.p1.iloc[0, 0])

    frozen = _two_bus(
        extendable=False, s_nom=s_nom_opt, r=R_REF * S_REF / s_nom_opt
    )
    frozen.optimize(transmission_losses=transmission_losses)
    frozen_flow = float(frozen.lines_t.p0.iloc[0, 0])
    frozen_loss = frozen_flow + float(frozen.lines_t.p1.iloc[0, 0])

    assert free_flow == pytest.approx(frozen_flow, rel=1e-6)
    assert free_loss == pytest.approx(frozen_loss, rel=1e-6)

    # and it approaches the quadratic r_pu_eff * p**2 from below
    quadratic = (R_REF * S_REF / s_nom_opt) / 220.0**2 * free_flow**2
    assert 0 < free_loss <= quadratic * (1 + 1e-9)


def test_losses_reject_undefined_reference_capacity():
    n = _two_bus(extendable=True, s_nom=0.0)
    n.lines["s_nom_min"] = 0.0

    with pytest.raises(ValueError, match="reference capacity"):
        n.optimize(transmission_losses=1)


@pytest.mark.parametrize("transmission_losses", [1, 2])
def test_optimize_losses(scipy_network, transmission_losses):
    n = scipy_network
    n.lines.s_max_pu = 0.7
    n.lines.loc[["316", "527", "602"], "s_nom"] = 1715

    n.optimize(
        snapshots=n.snapshots[0],
        transmission_losses=transmission_losses,
    )

    gen = n.generators_t.p.iloc[0].sum() + n.storage_units_t.p.iloc[0].sum()
    dem = n.loads_t.p_set.iloc[0].sum()

    assert gen > 1.01 * dem, "For this example, losses should be greater than 1%"
    assert gen < 1.05 * dem, "For this example, losses should be lower than 5%"
