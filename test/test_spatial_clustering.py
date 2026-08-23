#!/usr/bin/env python3
"""
Created on Mon Jan 31 18:11:09 2022.

@author: fabian
"""

import numpy as np
import pandas as pd
import pytest

import pypsa
from pypsa.clustering.spatial import (
    aggregatelines,
    aggregateoneport,
    busmap_by_hac,
    busmap_by_kmeans,
    get_clustering_from_busmap,
    normed_or_uniform,
)


def test_aggregate_generators(ac_dc_network):
    n = ac_dc_network
    busmap = pd.Series("all", n.buses.index)
    df, pnl = aggregateoneport(n, busmap, "Generator")

    assert (
        df.loc["all gas", "p_nom"] == n.generators.query("carrier == 'gas'").p_nom.sum()
    )
    assert (
        df.loc["all wind", "p_nom"]
        == n.generators.query("carrier == 'wind'").p_nom.sum()
    )

    capacity_norm = normed_or_uniform(n.generators.query("carrier == 'wind'").p_nom)
    assert np.allclose(
        pnl["p_max_pu"]["all wind"],
        (n.generators_t.p_max_pu * capacity_norm).sum(axis=1),
    )
    assert np.allclose(
        df.loc["all wind", "marginal_cost"],
        (n.generators.marginal_cost * capacity_norm).sum(),
    )


def test_aggregate_generators_custom_strategies(ac_dc_network):
    n = ac_dc_network
    n.generators.loc["Frankfurt Wind", "p_nom_max"] = 100

    busmap = pd.Series("all", n.buses.index)

    strategies = {"p_max_pu": "max", "p_nom_max": "weighted_min"}
    df, pnl = aggregateoneport(n, busmap, "Generator", custom_strategies=strategies)

    assert (
        df.loc["all gas", "p_nom"] == n.generators.query("carrier == 'gas'").p_nom.sum()
    )
    assert (
        df.loc["all wind", "p_nom"]
        == n.generators.query("carrier == 'wind'").p_nom.sum()
    )
    assert (
        df["p_nom_max"]["all wind"]
        == n.generators.loc["Frankfurt Wind", "p_nom_max"] * 3
    )
    assert np.allclose(pnl["p_max_pu"]["all wind"], n.generators_t.p_max_pu.max(axis=1))


def test_aggregate_generators_consent_error(ac_dc_network):
    n = ac_dc_network
    n.add(
        "Generator",
        "Manchester Wind 2",
        bus="Manchester",
        carrier="wind",
        p_nom_extendable=False,
    )

    busmap = pd.Series("all", n.buses.index)

    with pytest.raises(AssertionError):
        df, pnl = aggregateoneport(n, busmap, "Generator")


def test_aggregate_storage_units(ac_dc_network):
    n = ac_dc_network

    n.add(
        "StorageUnit",
        "Frankfurt Storage",
        bus="Frankfurt",
        p_nom_extendable=True,
        p_nom_max=100,
        p_nom=100,
        marginal_cost=10,
        capital_cost=100,
    )
    n.add(
        "StorageUnit",
        "Manchester Storage",
        bus="Manchester",
        p_nom_extendable=True,
        p_nom_max=200,
        p_nom=200,
        marginal_cost=30,
        capital_cost=50,
    )

    busmap = pd.Series("all", n.buses.index)
    df, pnl = aggregateoneport(n, busmap, "StorageUnit")
    capacity_norm = normed_or_uniform(n.storage_units.p_nom)

    assert df.loc["all", "p_nom"] == n.storage_units.p_nom.sum()
    assert df.loc["all", "p_nom_extendable"] == n.storage_units.p_nom_extendable.all()
    assert df.loc["all", "p_nom_min"] == n.storage_units.p_nom_min.sum()
    assert df.loc["all", "p_nom_max"] == n.storage_units.p_nom_max.sum()
    assert (
        df.loc["all", "marginal_cost"]
        == (n.storage_units.marginal_cost * capacity_norm).sum()
    )
    assert (
        df.loc["all", "capital_cost"]
        == (n.storage_units.capital_cost * capacity_norm).sum()
    )


def test_aggregate_storage_units_consent_error(ac_dc_network):
    n = ac_dc_network
    n.add("StorageUnit", "Bremen Storage", bus="Bremen", p_nom_extendable=False)

    busmap = pd.Series("all", n.buses.index)
    with pytest.raises(AssertionError):
        df, pnl = aggregateoneport(n, busmap, "StorageUnit")


def test_aggregate_loads_dynamic_sum_float32_tolerance():
    n = pypsa.Network()
    n.set_snapshots(range(3))
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Load", "load0", bus="bus0", p_set=1000.0)
    n.add("Load", "load1", bus="bus1", p_set=2000.0)

    n.loads_t.p_set = pd.DataFrame(
        {
            "load0": [1000.125, 1000.25, 1000.375],
            "load1": [2000.5, 2000.625, 2000.75],
        },
        index=n.snapshots,
    )

    busmap = pd.Series("all", n.buses.index)
    df, pnl = aggregateoneport(n, busmap, "Load")

    expected = n.loads_t.p_set.sum(axis=1)

    assert df.loc["all", "p_set"] == 3000.0
    np.testing.assert_allclose(
        pnl["p_set"]["all"], expected, rtol=1e-6, atol=1e-4
    )


def test_aggregate_loads_static_series_filtered_after_float32_aggregation():
    n = pypsa.Network()
    n.set_snapshots(range(3))
    n.add("Bus", "bus0")
    n.add("Bus", "bus1")
    n.add("Load", "load0", bus="bus0", p_set=1000.125)
    n.add("Load", "load1", bus="bus1", p_set=2000.25)

    n.loads_t.p_set = pd.DataFrame(
        {
            "load0": [1000.125, 1000.125, 1000.125],
            "load1": [2000.25, 2000.25, 2000.25],
        },
        index=n.snapshots,
    )

    busmap = pd.Series("all", n.buses.index)
    df, pnl = aggregateoneport(n, busmap, "Load")

    assert df.loc["all", "p_set"] == 3000.375
    assert pnl["p_set"].empty


def prepare_network_for_aggregation(n):
    n.lines = n.lines.reindex(columns=n.components["Line"]["attrs"].index[1:])
    n.lines["type"] = np.nan
    n.buses = n.buses.reindex(columns=n.components["Bus"]["attrs"].index[1:])
    n.buses["frequency"] = 50


def test_default_clustering_k_means(scipy_network):
    n = scipy_network
    prepare_network_for_aggregation(n)
    weighting = pd.Series(1, n.buses.index)
    busmap = busmap_by_kmeans(n, bus_weightings=weighting, n_clusters=50)
    C = get_clustering_from_busmap(n, busmap)
    nc = C.network
    assert len(nc.buses) == 50


def test_default_clustering_hac(scipy_network):
    n = scipy_network
    prepare_network_for_aggregation(n)
    busmap = busmap_by_hac(n, n_clusters=50)
    C = get_clustering_from_busmap(n, busmap)
    nc = C.network
    assert len(nc.buses) == 50


def test_cluster_accessor(scipy_network):
    n = scipy_network
    prepare_network_for_aggregation(n)

    weighting = pd.Series(1, n.buses.index)
    busmap = n.cluster.busmap_by_kmeans(
        bus_weightings=weighting, n_clusters=50, random_state=42
    )
    buses = n.cluster.cluster_by_busmap(busmap).buses

    buses_direct = n.cluster.cluster_spatially_by_kmeans(
        bus_weightings=weighting, n_clusters=50, random_state=42
    ).buses
    assert buses.equals(buses_direct)


def test_custom_line_groupers(scipy_network):
    n = scipy_network
    random_build_years = [1900, 2000]
    rng = np.random.default_rng()
    n.lines.loc[:, "build_year"] = rng.choice(random_build_years, size=len(n.lines))
    prepare_network_for_aggregation(n)
    weighting = pd.Series(1, n.buses.index)
    busmap = busmap_by_kmeans(n, bus_weightings=weighting, n_clusters=20)
    C = get_clustering_from_busmap(n, busmap, custom_line_groupers=["build_year"])
    linemap = C.linemap
    nc = C.network
    assert len(nc.buses) == 20
    assert (n.lines.groupby(linemap).build_year.nunique() == 1).all()


def _two_parallel_circuits(x1, x2, s1, s2, length1=111.0, length2=111.0):
    """Two circuits that a busmap turns into one parallel corridor A-B."""
    n = pypsa.Network()
    for bus, x in (("A1", 0.0), ("A2", 0.0), ("B1", 1.0), ("B2", 1.0)):
        n.add("Bus", bus, x=x, y=0.0, v_nom=230.0)
    n.add("Line", "l1", bus0="A1", bus1="B1", x=x1, r=0.0, s_nom=s1, length=length1)
    n.add("Line", "l2", bus0="A2", bus1="B2", x=x2, r=0.0, s_nom=s2, length=length2)
    busmap = pd.Series({"A1": "A", "A2": "A", "B1": "B", "B2": "B"})
    return n, busmap


def test_aggregate_lines_rating_is_set_by_the_bottleneck_circuit():
    # Flow divides 3:1, so the stiffer circuit saturates while the other is at
    # a third of its rating: the corridor is worth 400/3 MW, not 200 MW.
    n, busmap = _two_parallel_circuits(1.0, 3.0, 100.0, 100.0)
    lines, _, _ = aggregatelines(n, busmap, with_time=False)

    assert len(lines) == 1
    assert np.isclose(lines.s_nom.iloc[0], 400 / 3)
    assert lines.s_nom.iloc[0] < n.lines.s_nom.sum()


def test_aggregate_lines_rating_matches_where_the_power_flow_saturates():
    n, busmap = _two_parallel_circuits(1.0, 3.0, 100.0, 100.0)
    s_nom = aggregatelines(n, busmap, with_time=False)[0].s_nom.iloc[0]

    # Tie the terminals together so the two circuits are genuinely parallel,
    # then push exactly the aggregate rating through and read the loadings.
    n.add("Line", "tieA", bus0="A1", bus1="A2", x=1e-6, r=0.0, s_nom=1e6, length=1.0)
    n.add("Line", "tieB", bus0="B1", bus1="B2", x=1e-6, r=0.0, s_nom=1e6, length=1.0)
    n.add("Generator", "g", bus="A1", p_nom=1e6, p_set=s_nom)
    n.add("Load", "d", bus="B1", p_set=s_nom)
    n.lpf()

    loading = n.lines_t.p0.iloc[0][["l1", "l2"]].abs() / 100.0
    assert np.isclose(loading.max(), 1.0, atol=1e-4)  # first circuit exactly full
    assert loading.min() < 1.0  # the other still has headroom


def test_aggregate_lines_identical_circuits_still_pool():
    n, busmap = _two_parallel_circuits(2.0, 2.0, 100.0, 100.0)
    lines, _, _ = aggregatelines(n, busmap, with_time=False)
    assert np.isclose(lines.s_nom.iloc[0], 200.0)


def test_aggregate_lines_rating_never_exceeds_the_pooled_rating():
    # A circuit rated zero reads as missing data, not as a corridor rated zero,
    # but it must not inflate the corridor past what the ratings add up to.
    n, busmap = _two_parallel_circuits(1.0, 1.0, 100.0, 0.0)
    lines, _, _ = aggregatelines(n, busmap, with_time=False)
    assert np.isclose(lines.s_nom.iloc[0], 100.0)


def test_aggregate_lines_infinite_expansion_limit_survives():
    n, busmap = _two_parallel_circuits(1.0, 3.0, 100.0, 100.0)
    n.lines["s_nom_max"] = np.inf
    lines, _, _ = aggregatelines(n, busmap, with_time=False)
    assert np.isinf(lines.s_nom_max.iloc[0])


def test_aggregate_lines_single_circuit_is_untouched():
    n, busmap = _two_parallel_circuits(1.0, 3.0, 100.0, 100.0)
    n.remove("Line", "l2")
    lines, _, _ = aggregatelines(n, busmap, with_time=False)
    assert lines.s_nom.iloc[0] == 100.0


def test_aggregate_lines_rating_ignores_the_clustered_length():
    # Circuits of very different length, with reactance running proportional to
    # it as real conductor data does.  Flow splits 10:1 towards the short one,
    # which saturates at 110 MW for the corridor.  Rescaling the susceptances to
    # the clustered buses' great-circle distance first would divide out the
    # length, leave both circuits looking equally stiff, and hand back the plain
    # 200 MW sum -- the rating has to turn on the reactances the pre-aggregation
    # network actually splits flow by.
    n, busmap = _two_parallel_circuits(0.3, 3.0, 100.0, 100.0, 30.0, 300.0)
    lines, _, _ = aggregatelines(n, busmap, with_time=False)
    assert np.isclose(lines.s_nom.iloc[0], 110.0)

    n.add("Line", "tieA", bus0="A1", bus1="A2", x=1e-7, r=0.0, s_nom=1e7, length=1.0)
    n.add("Line", "tieB", bus0="B1", bus1="B2", x=1e-7, r=0.0, s_nom=1e7, length=1.0)
    n.add("Generator", "g", bus="A1", p_nom=1e7, p_set=110.0)
    n.add("Load", "d", bus="B1", p_set=110.0)
    n.lpf()
    loading = n.lines_t.p0.iloc[0][["l1", "l2"]].abs() / 100.0
    assert np.isclose(loading.max(), 1.0, atol=1e-4)
    assert loading.min() < 1.0
