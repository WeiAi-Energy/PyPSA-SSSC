"""
Tests for the cycle basis construction in :mod:`pypsa.cycle_basis`.

The fill search is a heuristic, so nothing here asserts that it finds a
particular basis. What it does assert is everything the rest of PyPSA relies
on being true whatever the search decides: the result is a basis of the same
cycle space, made of connected simple cycles whose columns are circulations,
it never regresses against the basis it started from, it is reproducible run
to run and process to process, and it falls back to that starting basis
whenever any of this cannot be established.

``unrefined()`` below is how the starting basis is obtained for comparison:
there is only one construction, so "before the search" means "with no budget
for it".
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import networkx as nx
import numpy as np
import pytest

import pypsa
from pypsa import cycle_basis, pf

SOLVER = dict(solver_name="gurobi", solver_options={"OutputFlag": 0})


def unrefined(**kwargs) -> cycle_basis.CycleBasisConfig:
    """
    The basis ``bfs-refined`` starts from, i.e. before the fill search.

    Every stage checks the deadline before it does anything, so a zero
    budget returns the breadth-first basis untouched. That is what the
    no-regression tests compare against.
    """
    return cycle_basis.CycleBasisConfig(time_budget=0.0, **kwargs)


# ---------------------------------------------------------------------------
# graph fixtures
# ---------------------------------------------------------------------------


def single_cycle(k: int = 6) -> list[tuple[str, str]]:
    """One ring: cycle rank 1, and the only basis there is."""
    return [(f"b{i}", f"b{(i + 1) % k}") for i in range(k)]


def square_with_diagonal() -> list[tuple[str, str]]:
    """The smallest graph where a basis choice exists: rank 2, three faces."""
    return [("a", "b"), ("b", "c"), ("c", "d"), ("d", "a"), ("a", "c")]


def grid(rows: int, cols: int) -> list[tuple[str, str]]:
    """Planar rows x cols lattice."""
    edges = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((f"n{r}_{c}", f"n{r}_{c + 1}"))
            if r + 1 < rows:
                edges.append((f"n{r}_{c}", f"n{r + 1}_{c}"))
    return edges


def complete_graph(k: int = 5) -> list[tuple[str, str]]:
    """K5: non-planar, every edge in many short cycles."""
    return [(f"k{i}", f"k{j}") for i in range(k) for j in range(i + 1, k)]


def with_bridges() -> list[tuple[str, str]]:
    """Two meshed blobs joined by a bridge, plus a dangling tree."""
    edges = complete_graph(4)
    edges += [(f"m{i}", f"m{j}") for i in range(4) for j in range(i + 1, 4)]
    edges += [("k0", "m0")]  # bridge: in no cycle at all
    edges += [("m3", "leaf1"), ("leaf1", "leaf2")]
    return edges


def two_components() -> list[tuple[str, str]]:
    """A K5 and a 3x4 grid that share no bus."""
    return complete_graph(5) + grid(3, 4)


def irregular() -> list[tuple[str, str]]:
    """
    A deliberately lopsided mesh: a long ring with a hub wired to every
    fourth node, which is the shape that makes a naive fundamental basis
    pile many cycles onto the hub's spokes.
    """
    ring = 24
    edges = [(f"r{i}", f"r{(i + 1) % ring}") for i in range(ring)]
    edges += [("hub", f"r{i}") for i in range(0, ring, 4)]
    edges += [("hub2", f"r{i}") for i in range(2, ring, 6)]
    edges += [("hub", "hub2")]
    return edges


def meshed(nodes: int = 40, degree: int = 4, rewire: float = 0.2, seed: int = 7):
    """
    A small-world mesh, as a stand-in for a transmission grid: mostly local
    ring connectivity with a few long ties. This is the shape on which a
    fundamental cycle basis is genuinely unbalanced -- some branches end up
    in ten or more basis cycles -- so it is where the search has something
    to do. ``seed`` is fixed, so the graph is the same on every run.
    """
    graph = nx.watts_strogatz_graph(nodes, degree, rewire, seed=seed)
    return [(f"w{u}", f"w{v}") for u, v in sorted(tuple(sorted(e)) for e in graph.edges())]


def build_network(
    edges,
    snapshots: int = 1,
    extendable=True,
    parallel: int = 0,
) -> pypsa.Network:
    """
    A minimal network carrying ``edges`` as AC lines.

    ``extendable`` is either a bool for all lines or a callable taking the
    line's position. ``parallel`` duplicates the first ``parallel`` edges, so
    that the sub-network gets fixed two-edge cycles.
    """
    n = pypsa.Network()
    n.set_snapshots(range(snapshots))
    buses = sorted({b for edge in edges for b in edge})
    n.madd("Bus", buses, v_nom=220.0, carrier="AC")

    def is_ext(i: int) -> bool:
        return bool(extendable(i)) if callable(extendable) else bool(extendable)

    for i, (u, v) in enumerate(edges):
        n.add(
            "Line",
            f"l{i}",
            bus0=u,
            bus1=v,
            x=0.1,
            r=0.01,
            s_nom=100.0,
            s_nom_extendable=is_ext(i),
            capital_cost=1.0,
        )
    for i in range(parallel):
        u, v = edges[i]
        n.add(
            "Line",
            f"p{i}",
            bus0=u,
            bus1=v,
            x=0.2,
            r=0.02,
            s_nom=50.0,
            s_nom_extendable=is_ext(i),
            capital_cost=1.0,
        )
    return n


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def fill_config(snapshots: int = 4, **kwargs) -> cycle_basis.CycleBasisConfig:
    """The default search; ``snapshots`` is kept so the callers read the same."""
    return cycle_basis.CycleBasisConfig(**{"time_budget": 60.0, **kwargs})


def bases(n: pypsa.Network, config: cycle_basis.CycleBasisConfig) -> dict:
    """Rebuild the topology under ``config`` and read every sub-network's basis."""
    n.__dict__.pop("_cycle_basis_cache", None)
    cycle_basis.configure_network(n, config)
    n.determine_network_topology()
    out = {}
    for sub in n.sub_networks.obj:
        if not sub.C.size:
            continue
        bg, cycles = cycle_basis.sub_network_basis(sub)
        out[str(sub.name)] = (sub, bg, cycles)
    return out


def metrics(n: pypsa.Network, config) -> dict:
    """Score every sub-network's basis under ``config``."""
    return {
        name: cycle_basis.basis_metrics(bg, cycles)
        for name, (_, bg, cycles) in bases(n, config).items()
    }


def compare(edges, snapshots=4, **network_kwargs):
    """``(starting basis, refined basis)`` metrics for the same network."""
    n = build_network(edges, snapshots=snapshots, **network_kwargs)
    return metrics(n, unrefined()), metrics(n, fill_config(snapshots))


ALL_GRAPHS = {
    "single-cycle": single_cycle(),
    "square-diagonal": square_with_diagonal(),
    "grid": grid(5, 6),
    "k5": complete_graph(5),
    "k6": complete_graph(6),
    "bridges": with_bridges(),
    "two-components": two_components(),
    "irregular": irregular(),
    "meshed": meshed(),
    "meshed-sparse": meshed(nodes=60, degree=4, rewire=0.1, seed=3),
}

# ---------------------------------------------------------------------------
# correctness of the produced basis
# ---------------------------------------------------------------------------


def _multigraph(n: pypsa.Network):
    n.determine_network_topology()
    return n.sub_networks.obj.iat[0].graph(weight="x_pu", inf_weight=False)


@pytest.mark.parametrize("name", sorted(ALL_GRAPHS))
def test_basis_is_a_valid_basis_of_the_same_space(name):
    """
    The heart of the contract: whatever the search does, the result is a
    basis of the *same* cycle space, every element is a connected simple
    cycle, and every column of C is a circulation.
    """
    edges = ALL_GRAPHS[name]
    n = build_network(edges, snapshots=4)

    base = bases(n, unrefined())
    new = bases(n, fill_config(4))
    assert set(base) == set(new)

    for key in base:
        _, bg, base_cycles = base[key]
        _, bg_new, new_cycles = new[key]
        assert bg.edges == bg_new.edges

        report = cycle_basis.verify_cycle_basis(
            bg_new,
            new_cycles,
            expected_rank=len(base_cycles),
            reference=base_cycles,
        )
        assert report["simple_cycles"], report
        assert report["count"], report
        assert report["rank"], report
        assert report["spans_reference"], report
        assert report["circulation"], report
        assert report["ok"], report


@pytest.mark.parametrize("name", sorted(ALL_GRAPHS))
def test_incidence_times_cycle_matrix_is_zero(name):
    """
    ``K C == 0`` on the sub-network's own matrices, i.e. the columns of the
    installed ``C`` are circulations of the *branch* graph, parallel
    branches and their two-edge cycles included.
    """
    n = build_network(ALL_GRAPHS[name], snapshots=2, parallel=2)
    for _, (sub, _, _) in bases(n, fill_config(2)).items():
        C = np.asarray(sub.C.todense())
        assert np.all(np.abs(C) <= 1)

        branches = sub.branches()
        buses = list(sub.buses_i())
        index = {bus: i for i, bus in enumerate(buses)}
        K = np.zeros((len(buses), len(branches)))
        for j, (bus0, bus1) in enumerate(zip(branches.bus0, branches.bus1)):
            K[index[bus0], j] += 1
            K[index[bus1], j] -= 1
        assert np.allclose(K @ C, 0.0)


def test_single_cycle_graph_has_the_only_possible_basis():
    """Rank one leaves no choice, and the search must not invent one."""
    base, new = compare(single_cycle(8))
    assert new["0"]["n_cycles"] == 1
    assert new["0"]["sum_length"] == 8
    assert new == base


def test_graph_with_bridges_keeps_bridges_out_of_every_cycle():
    """A bridge is in no cycle, so its participation stays zero."""
    n = build_network(with_bridges(), snapshots=2)
    for _, (sub, bg, cycles) in bases(n, fill_config(2)).items():
        counts = [0] * bg.n_edges
        for cycle in cycles:
            for e in cycle:
                counts[e] += 1
        graph = nx.Graph()
        graph.add_edges_from(bg.edges)
        for u, v in nx.bridges(graph):
            pair = (u, v) if u < v else (v, u)
            assert counts[bg.edge_lookup()[pair]] == 0


def test_multiple_components_are_each_covered():
    """Two disjoint meshes give one sub-network each, both with a full basis."""
    n = build_network(two_components(), snapshots=2)
    found = bases(n, fill_config(2))
    assert len(found) == 2
    for _, bg, cycles in found.values():
        assert len(cycles) == bg.simple_cycle_rank


def test_parallel_branches_give_fixed_two_edge_cycles():
    """
    Parallel branches contribute one fixed two-edge cycle each; they are part
    of the cycle rank and of the reported participation, and the optimized
    part of the basis has the remaining dimension.
    """
    n = build_network(grid(4, 4), snapshots=3, parallel=3)
    for _, (sub, bg, cycles) in bases(n, fill_config(3)).items():
        assert sum(bg.offset) == 3
        assert len(cycles) == bg.simple_cycle_rank
        assert bg.cycle_rank == bg.simple_cycle_rank + 3
        m = cycle_basis.basis_metrics(bg, cycles)
        assert m["n_cycles"] == bg.cycle_rank
        assert m["n_fixed_two_edge_cycles"] == 3
        assert sub.C.shape[1] == bg.cycle_rank


def test_parallel_branch_participation_enters_the_score():
    """
    The fixed two-edge cycles must be counted *before* the search starts,
    otherwise the representative branch of a parallel edge looks emptier
    than it is and the refinement piles more cycles onto it.
    """
    edges = grid(4, 4)
    bg_plain = cycle_basis.BranchGraph.from_multigraph(
        _multigraph(build_network(edges, snapshots=2))
    )
    bg_parallel = cycle_basis.BranchGraph.from_multigraph(
        _multigraph(build_network(edges, snapshots=2, parallel=4))
    )
    assert sum(bg_plain.offset) == 0
    assert sum(bg_parallel.offset) == 4

    # The offset shifts the cost curve: the *first* cycle put on a parallel
    # edge already costs what the *second* would cost on a plain one, which
    # is what stops the search concentrating cycles there. (The totals at
    # zero participation agree, because choose(1, 2) == 0 -- it is the
    # marginal that carries the information.)
    counts = [0] * bg_plain.n_edges
    busy = bg_parallel.offset.index(1)
    one = list(counts)
    one[busy] = 1
    assert cycle_basis._fill_delta(bg_parallel.offset, counts, [], [busy]) == 1
    assert cycle_basis._fill_delta(bg_plain.offset, counts, [], [busy]) == 0
    assert cycle_basis._fill_delta(bg_plain.offset, one, [], [busy]) == 1


# ---------------------------------------------------------------------------
# no regression against the basis the search started from
# ---------------------------------------------------------------------------


NO_REGRESSION = ("phi", "sum_length", "sum_length_sq", "max_length")


@pytest.mark.parametrize("name", sorted(ALL_GRAPHS))
@pytest.mark.parametrize("snapshots", [1, 4, 32])
def test_never_worse_than_the_starting_basis(name, snapshots):
    """The hard constraints of the search, checked end to end."""
    base, new = compare(ALL_GRAPHS[name], snapshots=snapshots)
    assert set(base) == set(new)
    for key in base:
        for metric in NO_REGRESSION:
            assert new[key][metric] <= base[key][metric], (key, metric, snapshots)
        assert new[key]["n_cycles"] == base[key]["n_cycles"]
        assert new[key]["cycle_rank"] == base[key]["cycle_rank"]


def test_mixed_extendable_and_fixed_branches():
    """
    A mixed sub-network must satisfy the no-regression constraints, and the
    purely topological proxy must not depend on the extendable flags at all.
    """
    edges = grid(5, 5)
    base, new = compare(edges, snapshots=6, extendable=lambda i: i % 3 == 0)
    for key in base:
        for metric in NO_REGRESSION:
            assert new[key][metric] <= base[key][metric], (key, metric)

    fixed = metrics(build_network(edges, snapshots=6, extendable=False), fill_config(6))
    ext = metrics(build_network(edges, snapshots=6, extendable=True), fill_config(6))
    assert {k: v["phi"] for k, v in fixed.items()} == {
        k: v["phi"] for k, v in ext.items()
    }


@pytest.mark.parametrize("name", ["meshed", "meshed-sparse"])
def test_search_improves_a_lopsided_mesh(name):
    """
    On a graph whose fundamental bases are genuinely unbalanced, the search
    has to actually find something -- otherwise the no-regression tests
    above would be satisfied by a search that never does anything.
    """
    base, new = compare(ALL_GRAPHS[name], snapshots=8)
    for key in base:
        assert new[key]["phi"] < base[key]["phi"]
        assert new[key]["max_participation"] <= base[key]["max_participation"]


# ---------------------------------------------------------------------------
# determinism
# ---------------------------------------------------------------------------


def _basis_signature(n: pypsa.Network, config) -> str:
    out = []
    for name, (_, bg, cycles) in sorted(bases(n, config).items()):
        out.append(
            [name, sorted(sorted(bg.edges[e] for e in cycle) for cycle in cycles)]
        )
    return json.dumps(out, sort_keys=True)


@pytest.mark.parametrize("name", sorted(ALL_GRAPHS))
def test_repeated_runs_are_identical(name):
    edges = ALL_GRAPHS[name]
    config = fill_config(4)
    first = _basis_signature(build_network(edges, snapshots=4), config)
    second = _basis_signature(build_network(edges, snapshots=4), config)
    third = _basis_signature(build_network(edges, snapshots=4), config)
    assert first == second == third


DETERMINISM_SCRIPT = """
import json, sys
sys.path.insert(0, %r)
import pypsa
from pypsa import cycle_basis

edges = json.loads(sys.argv[1])
n = pypsa.Network()
n.set_snapshots(range(4))
buses = sorted({b for e in edges for b in e})
n.madd("Bus", buses, v_nom=220.0, carrier="AC")
for i, (u, v) in enumerate(edges):
    n.add("Line", "l%%d" %% i, bus0=u, bus1=v, x=0.1, r=0.01, s_nom=100.0,
          s_nom_extendable=True, capital_cost=1.0)

n.determine_network_topology()
out = []
for sub in n.sub_networks.obj:
    if not sub.C.size:
        continue
    bg, cycles = cycle_basis.sub_network_basis(sub)
    out.append([str(sub.name),
                sorted(sorted(bg.edges[e] for e in c) for c in cycles)])
print(json.dumps(sorted(out), sort_keys=True))
""" % str(Path(__file__).resolve().parents[1])


def test_result_does_not_depend_on_pythonhashseed(tmp_path):
    """
    Nothing in the search may depend on the iteration order of a set or a
    dict of bus labels, so the answer must be identical under hash seeds
    that reorder every string-keyed container in the process.
    """
    script = tmp_path / "run.py"
    script.write_text(DETERMINISM_SCRIPT)
    edges = json.dumps(meshed())

    outputs = []
    for hash_seed in ("0", "1", "12345"):
        completed = subprocess.run(
            [sys.executable, str(script), edges],
            capture_output=True,
            text=True,
            env=dict(os.environ, PYTHONHASHSEED=hash_seed),
            check=True,
        )
        outputs.append(completed.stdout.strip().splitlines()[-1])
    assert outputs[0] == outputs[1] == outputs[2]
    assert json.loads(outputs[0])  # and it actually found a basis


# ---------------------------------------------------------------------------
# caching
# ---------------------------------------------------------------------------


def test_cache_hits_on_an_unchanged_topology_and_config():
    n = build_network(grid(4, 5), snapshots=4)
    cycle_basis.configure_network(n, fill_config(4))
    n.determine_network_topology()
    cache = n.__dict__["_cycle_basis_cache"]
    assert len(cache) == 1

    calls = []
    original = cycle_basis._refine_cycles_for_fill

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    cycle_basis._refine_cycles_for_fill = counting
    try:
        n.determine_network_topology()
        n.determine_network_topology()
        assert not calls, "an unchanged topology must reuse the cached basis"
        assert len(cache) == 1
    finally:
        cycle_basis._refine_cycles_for_fill = original


def test_cache_misses_when_the_budget_changes():
    """
    The budget is part of the cache key: a different budget can produce a
    different basis, so handing back the old one would silently substitute a
    basis the caller did not ask for.
    """
    n = build_network(grid(4, 5), snapshots=4)
    first = fill_config(4)
    second = fill_config(4, time_budget=5.0)
    assert first.fingerprint() != second.fingerprint()

    cycle_basis.configure_network(n, first)
    n.determine_network_topology()
    cycle_basis.configure_network(n, second)
    n.determine_network_topology()
    assert len(n.__dict__["_cycle_basis_cache"]) == 2


def test_cache_returns_a_copy():
    """A caller mutating ``sub.C`` must not corrupt the cached basis."""
    n = build_network(grid(4, 5), snapshots=4)
    cycle_basis.configure_network(n, fill_config(4))
    n.determine_network_topology()
    sub = n.sub_networks.obj.iat[0]
    before = np.asarray(sub.C.todense()).copy()
    sub.C[0, 0] = 99.0
    n.determine_network_topology()
    sub = n.sub_networks.obj.iat[0]
    assert np.allclose(np.asarray(sub.C.todense()), before)


# ---------------------------------------------------------------------------
# invariants of a single exchange
# ---------------------------------------------------------------------------


def test_every_admissible_pairwise_exchange_preserves_the_rank():
    """
    ``A <- A xor B`` is a Gaussian elimination pivot, so it cannot change the
    rank or the span. Rather than trusting that argument, enumerate every
    exchange the search would consider on a small mesh and check both.
    """
    n = build_network(complete_graph(5), snapshots=2)
    (_, bg, cycles), = list(bases(n, cycle_basis.CycleBasisConfig()).values())
    rank = cycle_basis.gf2_rank(bg, cycles)
    assert rank == bg.simple_cycle_rank

    checked = 0
    for a in range(len(cycles)):
        for b in range(len(cycles)):
            if a == b:
                continue
            new = cycles[a] ^ cycles[b]
            if cycle_basis.cycle_node_order(bg, new) is None:
                continue
            for replace in (a, b):
                exchanged = [set(c) for c in cycles]
                exchanged[replace] = new
                assert cycle_basis.gf2_rank(bg, exchanged) == rank
                report = cycle_basis.verify_cycle_basis(
                    bg, exchanged, expected_rank=rank, reference=cycles
                )
                assert report["ok"], report
                checked += 1
    assert checked > 0


def test_incremental_score_matches_a_full_recomputation():
    """
    The search updates its four score components in place; a drift would
    silently corrupt every acceptance decision, so it is an exact integer
    equality that is asserted at every stage boundary.
    """
    n = build_network(meshed(), snapshots=5)
    (_, bg, cycles), = list(bases(n, cycle_basis.CycleBasisConfig()).values())
    state = cycle_basis._BasisState(bg, cycles)
    assert state.score() == state.recomputed_score()

    moves = 0
    for a in range(len(cycles)):
        for b in range(a + 1, len(cycles)):
            new = state.cycles[a] ^ state.cycles[b]
            if cycle_basis.cycle_node_order(bg, new) is None:
                continue
            score, removed, added = state.evaluate(a, new)
            state.apply(a, new, score, removed, added)
            assert state.score() == state.recomputed_score()
            moves += 1
            break
    assert moves > 0


# ---------------------------------------------------------------------------
# fallback
# ---------------------------------------------------------------------------


def test_fallback_when_the_budget_is_exhausted():
    """
    A zero time budget must leave the baseline in place rather than a
    half-searched basis.
    """
    n = build_network(meshed(), snapshots=4)
    base = metrics(n, unrefined())
    starved = metrics(n, fill_config(4, time_budget=0.0))
    assert starved == base


def test_fallback_when_verification_fails(monkeypatch):
    """
    If the final verification ever rejects a basis, the baseline is returned
    unchanged -- not the rejected basis, and not an exception.
    """
    edges = meshed()
    n = build_network(edges, snapshots=4)
    base = metrics(n, unrefined())

    def rejecting(*args, **kwargs):
        return {"ok": False, "why": "forced"}

    monkeypatch.setattr(cycle_basis, "verify_cycle_basis", rejecting)
    n2 = build_network(edges, snapshots=4)
    forced = metrics(n2, fill_config(4))
    assert forced == base


def test_fallback_when_a_stage_raises(monkeypatch):
    edges = meshed()
    base = metrics(build_network(edges, snapshots=4), unrefined())

    def exploding(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(cycle_basis, "_pairwise_pass", exploding)
    forced = metrics(build_network(edges, snapshots=4), fill_config(4))
    assert forced == base



# ---------------------------------------------------------------------------
# the optimization model is unchanged
# ---------------------------------------------------------------------------


def optimisable(edges, snapshots=3, seed=0):
    """A small meshed network with enough generation and load to solve."""
    n = build_network(edges, snapshots=snapshots)
    buses = sorted(n.buses.index)
    rng = np.random.default_rng(seed)
    for i, bus in enumerate(buses):
        n.add(
            "Generator",
            f"g{i}",
            bus=bus,
            p_nom=200.0,
            marginal_cost=float(10 + 5 * (i % 7)),
        )
        n.add(
            "Load",
            f"d{i}",
            bus=bus,
            p_set=float(20 + 30 * rng.random()),
        )
    n.lines["s_nom"] = 60.0
    # the iterative expansion scales its proximal term by the previous
    # capacity, so no branch may be allowed to collapse to zero
    n.lines["s_nom_min"] = 30.0
    n.lines["capital_cost"] = 50.0
    return n




@pytest.mark.parametrize("name", ["k5", "grid", "meshed"])
def test_lp_optimum_is_unchanged(name):
    """
    A different basis is a different set of KVL rows for the same voltage
    law, so the pure LP optimum must be identical, not merely close.
    """
    pytest.importorskip("gurobipy")
    edges = ALL_GRAPHS[name]
    objectives = []
    for config in (unrefined(), fill_config(3)):
        n = optimisable(edges)
        n.lines["s_nom_extendable"] = False
        cycle_basis.configure_network(n, config)
        n.optimize(**SOLVER)
        objectives.append(n.objective)
    assert objectives[0] == pytest.approx(objectives[1], rel=1e-9)


@pytest.mark.parametrize("name", ["k5", "meshed"])
def test_slp_optimum_is_unchanged(name):
    """
    The same, through the iterative expansion, i.e. with the capacity
    sensitivity columns the later iterations add.
    """
    pytest.importorskip("gurobipy")
    edges = ALL_GRAPHS[name]
    objectives = []
    capacities = []
    for config in (unrefined(), fill_config(3)):
        n = optimisable(edges)
        cycle_basis.configure_network(n, config)
        n.optimize.optimize_transmission_expansion_iteratively(
            max_iterations=6, **SOLVER
        )
        objectives.append(n.objective)
        capacities.append(n.lines.s_nom_opt.sort_index().to_numpy())
    assert objectives[0] == pytest.approx(objectives[1], rel=1e-6)
    np.testing.assert_allclose(capacities[0], capacities[1], rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("name", ["k5", "grid", "meshed"])
def test_linear_power_flow_is_unchanged(name):
    """The basis also defines the KVL of ``sub_network_lpf``."""
    edges = ALL_GRAPHS[name]
    flows = []
    angles = []
    for config in (unrefined(), fill_config(3)):
        n = optimisable(edges)
        n.generators["control"] = "PQ"
        n.generators.iloc[0, n.generators.columns.get_loc("control")] = "Slack"
        n.generators["p_set"] = n.loads.set_index("bus").p_set.reindex(
            n.generators.bus.to_numpy()
        ).to_numpy() * np.linspace(0.5, 1.5, len(n.generators))
        cycle_basis.configure_network(n, config)
        n.lpf()
        flows.append(n.lines_t.p0.sort_index(axis=1).to_numpy())
        angles.append(n.buses_t.v_ang.sort_index(axis=1).to_numpy())
    np.testing.assert_allclose(flows[0], flows[1], rtol=1e-8, atol=1e-8)
    np.testing.assert_allclose(angles[0], angles[1], rtol=1e-8, atol=1e-8)


# ---------------------------------------------------------------------------
# the default path
# ---------------------------------------------------------------------------


def test_the_fill_search_runs_without_being_asked_for():
    """
    There is one construction and the fill search is part of it, so an
    untouched network gets the refined basis -- no configuration needed and
    nothing required from the optimization model.
    """
    plain = build_network(meshed(), snapshots=4)
    plain.determine_network_topology()

    explicit = build_network(meshed(), snapshots=4)
    cycle_basis.configure_network(explicit, cycle_basis.CycleBasisConfig())
    explicit.determine_network_topology()
    np.testing.assert_array_equal(
        np.asarray(plain.sub_networks.obj.iat[0].C.todense()),
        np.asarray(explicit.sub_networks.obj.iat[0].C.todense()),
    )

    started_from = metrics(build_network(meshed(), snapshots=4), unrefined())
    default = metrics(
        build_network(meshed(), snapshots=4), cycle_basis.CycleBasisConfig()
    )
    for key in started_from:
        assert default[key]["phi"] < started_from[key]["phi"]


def test_configure_network_can_be_cleared():
    n = build_network(grid(3, 3), snapshots=2)
    cycle_basis.configure_network(n, fill_config(2))
    assert cycle_basis.network_config(n).time_budget == 60.0
    cycle_basis.configure_network(n, None)
    assert cycle_basis.network_config(n) is cycle_basis.get_default_config()
    assert cycle_basis.network_config(n).time_budget == 100.0



def test_fill_delta_matches_a_recomputation():
    """
    The affine marginal the search uses has to agree exactly with the
    difference of two totals -- that is the whole basis of the incremental
    bookkeeping.
    """
    n = build_network(grid(5, 5), snapshots=2, parallel=3)
    bg = cycle_basis.BranchGraph.from_multigraph(_multigraph(n))
    rng = np.random.default_rng(3)
    counts = [int(x) for x in rng.integers(1, 5, bg.n_edges)]
    checked = 0
    for _ in range(50):
        removed = sorted(rng.choice(bg.n_edges, 3, replace=False).tolist())
        added = sorted(
            set(rng.choice(bg.n_edges, 3, replace=False).tolist()) - set(removed)
        )
        if not added:
            continue
        moved = list(counts)
        for e in removed:
            moved[e] -= 1
        for e in added:
            moved[e] += 1
        expected = cycle_basis._fill_cost(bg.offset, moved) - cycle_basis._fill_cost(
            bg.offset, counts
        )
        assert cycle_basis._fill_delta(bg.offset, counts, removed, added) == expected
        checked += 1
    assert checked > 0
