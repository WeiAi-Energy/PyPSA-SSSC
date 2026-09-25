r"""
Cycle basis construction for the Kirchhoff voltage law.

A cycle basis only has to span a sub-network's cycle space; *which* basis is
used is a free choice that leaves the voltage law -- and therefore the
feasible region and the optimum -- untouched. It does change the sparsity
pattern of the constraint matrix, and with it how much fill-in an interior
point method's Cholesky factorization produces.

This module is the second half of the ``"bfs-refined"`` cycle basis
construction. :mod:`pypsa.pf` builds a starting basis -- the best of several
breadth-first spanning trees, greedily shortened by local exchange -- and the
search here then rewrites it to reduce a structural *fill* proxy, against the
sparsity pattern itself rather than against any solver-internal threshold.

Length was the previous objective, and it is the wrong one: it counts
nonzeros, while fill is driven by how many rows each *column* touches. On a
10,000-bus US case the starting basis puts one branch in 21 different cycles;
the search cuts the worst case to 7, and with it the first-round fill proxy by
15 %, the SLP proxy by 25 % and the total cycle length by 12 % -- so the
sparsity gain does not come at the price of a longer basis.

The fill proxy
--------------

Let ``p_e(B)`` be the number of basis cycles a branch ``e`` takes part in.
Every one of those cycles puts one nonzero into the branch's flow column, and
a column with ``d`` nonzeros couples ``choose(d, 2)`` row pairs in ``A A'``
-- the matrix whose Cholesky factor the barrier method forms. So

.. math::
    \Phi(B) = \sum_e \binom{p_e(B)}{2}

is a structural proxy for the fill the branch columns produce. It is
minimised by a basis that is short *and* spreads its cycles evenly over the
branches, because ``choose(d, 2)`` is convex: moving participation from a
heavily used branch to a lightly used one lowers the sum even at constant
total length.

The proxy mentions no solver, no dense-column threshold and no component,
and the search never tunes one. It also needs nothing from the optimization
model -- which matters, because :func:`pypsa.pf.find_cycles` is driven by
the lazy :attr:`pypsa.SubNetwork.C`, whose readers include the power flow
and the diagnostics, not only the optimization. Weighting the
proxy by the snapshot count and by the capacity-sensitivity columns of an
iterative expansion was tried and measured: on a 10,000-bus US case it moved
the result by 0.35 %, which does not justify threading model state into
topology construction.

What is guaranteed and what is not
----------------------------------

The search is a deterministic heuristic that walks the binary matroid of
cycle bases by elementary exchanges. It is *not* a global optimizer.

It also does not control how many iterations a barrier method takes: measured
on a production model, the *work per iteration* fell in every solve (7 % to
23 % in Gurobi work units, tracking a 23--41 % drop in factorization
operations), while the iteration count moved in both directions. Sparsity
decides the cost of a step, not the number of steps.

What it does guarantee is that the returned basis

* spans the same cycle space as the starting basis (same rank, same span),
* consists of connected simple cycles (plus the fixed two-edge cycles that
  parallel branches force), and
* is no worse than the starting basis on ``Phi_LP``, total length, sum of
  squared lengths and maximum length -- otherwise the starting basis is
  returned unchanged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from operator import itemgetter
from typing import Any

import networkx as nx
import numpy as np
from scipy.sparse import dok_matrix

logger = logging.getLogger(__name__)

#: Name of the cycle basis construction, for log messages.
BFS_REFINED = "bfs-refined"

#: How far the search may temporarily exceed the bounds it must satisfy at
#: the end (total length, sum of squared lengths, ``Phi_LP``). Leaving a
#: local optimum needs room to move through worse states; the final answer is
#: re-checked against the starting basis with no slack at all. Not a knob:
#: it has never needed changing, and a search that needs a different value
#: here needs a different escape mechanism, not a different number.
SEARCH_SLACK = 0.02

#: Size of the perturbation used to leave a local optimum: how many
#: deliberately uphill exchanges are taken when a pass finds no improving
#: move. This is a *move size*, not a budget on how long the search runs --
#: unbounded it would stop being a nudge and become a random restart. Worth
#: 0.013 % on a 10,000-bus case.
KICK_BUDGET = 4

#: How far the search may temporarily exceed the bounds it must satisfy at
#: the end (total length and sum of squared lengths). Leaving a local optimum
#: needs room to move through worse states; the final answer is re-checked
#: against the starting basis with no slack at all. Also an algorithmic
#: parameter rather than a budget.
SEARCH_SLACK = 0.02

#: Seed for the deterministic stride that orders candidate-exchange roots.
#: Fixed, so the search is reproducible.
SEED = 0

#: *Cumulative* fractions of the time budget at which each stage hands over
#: to the next, so stage one gets ``[0, S0)``, stage two ``[S0, S1)`` and
#: stage three the rest.
#:
#: A stage that converges hands the remainder on, because the shares are
#: absolute deadlines rather than durations. The shares therefore only bind
#: on a stage that does *not* converge inside the budget, and on a
#: 10,000-bus case that is the first one: left to run it keeps finding
#: genuine improvements and leaves nothing for the other two, which costs
#: the maximum cycle length (46 against 34).
#:
#: The split was measured by holding the total budget fixed and moving the
#: stage-one/stage-two boundary, at two budgets:
#:
#: ===========  ==========  ==========
#: stage 1       Phi @20s    Phi @60s
#: ===========  ==========  ==========
#: 0.10             29777       28872
#: 0.20             29887       28952
#: 0.40             30231       29508
#: 0.80             31956       31488
#: ===========  ==========  ==========
#:
#: Monotone at both: time is worth more to the candidate exchange, which had
#: not saturated even at 32 s per sub-network, than to the first stage, which
#: saturates at about 10 s and past that only wanders.
#:
#: Note the mismatch this leaves: what stage one needs is an *absolute*
#: duration (~10 s on that network), not a fraction, so the larger the
#: budget the more a fixed share over-feeds it. 0.10 is the smallest value
#: measured, not a proven optimum, and it is a property of one network
#: family rather than a general constant.
STAGE_SHARES = (0.1, 0.9, 1.0)

def _comb2(d: int) -> int:
    """``choose(d, 2)``, clamped to zero for ``d < 2``."""
    return d * (d - 1) // 2 if d > 1 else 0


# ---------------------------------------------------------------------------
# edge indexed topology
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BranchGraph:
    """
    Edge indexed view of a sub-network's passive branch topology.

    The search works on integer edge indices rather than on bus labels: a
    cycle is a ``set[int]`` of edge indices, which makes the symmetric
    difference that drives every basis exchange a single set operation and
    makes participation a plain list lookup.

    Parallel branches are collapsed into one *simple* edge here, exactly as
    ``pypsa.pf.find_cycles`` collapses the branch multigraph into an
    ``nx.Graph`` before looking for cycles. ``branches[e]`` keeps every
    branch on that edge in multigraph order; ``branches[e][0]`` is the
    representative, i.e. the branch that carries the edge in every cycle of
    the basis *and* in each of the ``len(branches[e]) - 1`` fixed two-edge
    cycles the parallel branches force. That is why ``offset[e]`` below is
    not zero for a parallel edge: those fixed cycles are part of the
    participation the fill proxy has to account for, and ignoring them would
    let the search pile further cycles onto a branch that is already the
    busiest one in the sub-network.

    Attributes
    ----------
    nodes : tuple
        Bus labels, in multigraph order. Node indices refer to this tuple.
    edges : tuple of (int, int)
        One entry per simple edge, as a sorted pair of node indices.
    branches : tuple of tuple
        Branch keys on each simple edge, representative first.
    offset : tuple of int
        Participation of each edge's representative branch in the fixed
        two-edge cycles, i.e. ``len(branches[e]) - 1``.
    adj : tuple of tuple of (int, int)
        Per node, the sorted ``(neighbour, edge)`` pairs.
    component : tuple of int
        Connected component id per node.
    n_components : int
        Number of connected components (isolated nodes included).
    """

    nodes: tuple[Any, ...]
    edges: tuple[tuple[int, int], ...]
    branches: tuple[tuple[Any, ...], ...]
    offset: tuple[int, ...]
    adj: tuple[tuple[tuple[int, int], ...], ...]
    component: tuple[int, ...]
    n_components: int

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_edges(self) -> int:
        """Number of *simple* edges, i.e. the rows the search can move."""
        return len(self.edges)

    @property
    def n_branches(self) -> int:
        """Number of branches, counting parallel ones separately."""
        return sum(len(b) for b in self.branches)

    @property
    def cycle_rank(self) -> int:
        """
        Dimension of the full cycle space, ``m - n + c`` over the multigraph.

        Equals :attr:`simple_cycle_rank` plus one fixed two-edge cycle per
        extra parallel branch.
        """
        return self.n_branches - self.n_nodes + self.n_components

    @property
    def simple_cycle_rank(self) -> int:
        """Dimension of the cycle space of the collapsed simple graph."""
        return self.n_edges - self.n_nodes + self.n_components

    @classmethod
    def from_multigraph(cls, mgraph: nx.MultiGraph) -> BranchGraph:
        """
        Build from the branch multigraph ``pypsa.SubNetwork.graph`` returns.

        Node order is taken from the multigraph (insertion ordered, hence
        reproducible); edges are then sorted by node index pair, so the edge
        indexing depends only on the branch set and not on the iteration
        order of any hash based container.
        """
        nodes = tuple(mgraph.nodes())
        node_index = {node: i for i, node in enumerate(nodes)}

        pair_branches: dict[tuple[int, int], list[Any]] = {}
        for u, v, key in mgraph.edges(keys=True):
            iu, iv = node_index[u], node_index[v]
            if iu == iv:
                # A self loop is already a cycle on its own and can never be
                # part of a basis exchange; leave it to the caller.
                continue
            pair = (iu, iv) if iu < iv else (iv, iu)
            pair_branches.setdefault(pair, []).append(key)

        edges = tuple(sorted(pair_branches))
        branches = tuple(tuple(pair_branches[pair]) for pair in edges)
        offset = tuple(len(b) - 1 for b in branches)

        adj_lists: list[list[tuple[int, int]]] = [[] for _ in nodes]
        for e, (u, v) in enumerate(edges):
            adj_lists[u].append((v, e))
            adj_lists[v].append((u, e))
        adj = tuple(tuple(sorted(a)) for a in adj_lists)

        component = [-1] * len(nodes)
        n_components = 0
        for start in range(len(nodes)):
            if component[start] >= 0:
                continue
            component[start] = n_components
            stack = [start]
            while stack:
                u = stack.pop()
                for v, _ in adj[u]:
                    if component[v] < 0:
                        component[v] = n_components
                        stack.append(v)
            n_components += 1

        return cls(
            nodes=nodes,
            edges=edges,
            branches=branches,
            offset=offset,
            adj=adj,
            component=tuple(component),
            n_components=n_components,
        )

    def edge_lookup(self) -> dict[tuple[int, int], int]:
        """Map a sorted node index pair to its edge index."""
        cached = self.__dict__.get("_edge_lookup")
        if cached is None:
            cached = {pair: e for e, pair in enumerate(self.edges)}
            object.__setattr__(self, "_edge_lookup", cached)
        return cached

    def node_lookup(self) -> dict[Any, int]:
        """Map a bus label to its node index."""
        cached = self.__dict__.get("_node_lookup")
        if cached is None:
            cached = {node: i for i, node in enumerate(self.nodes)}
            object.__setattr__(self, "_node_lookup", cached)
        return cached

    def component_edges(self) -> list[list[int]]:
        """Edge indices grouped by connected component."""
        groups: list[list[int]] = [[] for _ in range(self.n_components)]
        for e, (u, _) in enumerate(self.edges):
            groups[self.component[u]].append(e)
        return groups

    def component_nodes(self) -> list[list[int]]:
        """Node indices grouped by connected component."""
        groups: list[list[int]] = [[] for _ in range(self.n_components)]
        for node, comp in enumerate(self.component):
            groups[comp].append(node)
        return groups

    def node_cycle_to_edges(self, cycle: Sequence[Any]) -> set[int] | None:
        """
        Convert a cycle given as bus labels in cyclic order (the format
        ``nx.cycle_basis`` and :mod:`pypsa.pf` use) into an edge index set,
        or ``None`` if some step is not an edge of this graph.
        """
        length = len(cycle)
        if length < 3:
            return None
        lookup = self.edge_lookup()
        nodes = self.node_lookup()
        out: set[int] = set()
        for i, u in enumerate(cycle):
            v = cycle[(i + 1) % length]
            iu, iv = nodes.get(u), nodes.get(v)
            if iu is None or iv is None:
                return None
            e = lookup.get((iu, iv) if iu < iv else (iv, iu))
            if e is None:
                return None
            out.add(e)
        if len(out) != length:
            return None
        return out

    def edges_to_node_cycle(self, edge_ids: Iterable[int]) -> list[Any] | None:
        """
        Convert an edge index set that forms one simple cycle into the bus
        label list in cyclic order, or ``None`` if it is not a single simple
        cycle.
        """
        order = cycle_node_order(self, edge_ids)
        if order is None:
            return None
        return [self.nodes[i] for i in order]


def cycle_node_order(bg: BranchGraph, edge_ids: Iterable[int]) -> list[int] | None:
    """
    Walk an edge index set and return its nodes in cyclic order, or ``None``
    if the set is not one connected simple cycle.

    A single simple cycle is exactly an edge set in which every touched node
    has degree two *and* one walk consumes all of it; requiring both rejects
    the disjoint unions of loops that a symmetric difference of two basis
    cycles can produce, which are legitimate cycle space elements but cannot
    be written as one cyclic node list.
    """
    edge_ids = list(edge_ids)
    if len(edge_ids) < 3:
        return None

    adjacency: dict[int, list[int]] = {}
    for e in edge_ids:
        u, v = bg.edges[e]
        au = adjacency.setdefault(u, [])
        av = adjacency.setdefault(v, [])
        au.append(v)
        av.append(u)
        if len(au) > 2 or len(av) > 2:
            return None
    if any(len(nb) != 2 for nb in adjacency.values()):
        return None

    start = min(adjacency)
    order = [start]
    previous = -1
    current = start
    while True:
        a, b = adjacency[current]
        nxt = b if a == previous else a
        if nxt == start:
            return order if len(order) == len(edge_ids) else None
        if len(order) >= len(edge_ids):
            return None
        order.append(nxt)
        previous, current = current, nxt


# ---------------------------------------------------------------------------
# the fill proxy
# ---------------------------------------------------------------------------


def _fill_cost(offset: Sequence[int], counts: Sequence[int]) -> int:
    """
    ``Phi(B) = sum_e choose(p_e(B), 2)`` over the simple edges.

    ``p_e`` is the total participation of the edge's representative branch:
    ``offset[e]`` fixed two-edge cycles plus ``counts[e]`` basis cycles. The
    non-representative parallel branches sit in exactly one two-edge cycle
    whatever the basis does, and ``choose(1, 2) = 0``, so they contribute
    nothing and are left out.
    """
    return sum(_comb2(offset[e] + c) for e, c in enumerate(counts))


def _fill_delta(
    offset: Sequence[int],
    counts: Sequence[int],
    removed: Iterable[int],
    added: Iterable[int],
) -> int:
    """
    Change in ``Phi`` when ``removed`` lose one cycle and ``added`` gain one.

    ``choose(p, 2)`` has the affine marginal ``choose(p + 1, 2) -
    choose(p, 2) = p``, so an exchange costs one add per touched edge instead
    of a rescoring of the whole basis. Integer arithmetic throughout, so the
    incremental value is exact and cannot drift from a recomputation.
    """
    delta = 0
    for e in removed:
        delta -= offset[e] + counts[e] - 1
    for e in added:
        delta += offset[e] + counts[e]
    return delta


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CycleBasisConfig:
    """
    The one thing the caller decides: how long the search may run.

    Everything else is fixed (see the module constants above). Each of the
    other parameters this class used to expose was measured on a 10,000-bus
    case and changed the result by less than 0.07 %, so exposing them would
    only have invited tuning that cannot pay.

    ``time_budget`` enters the cycle basis cache key together with the
    topology fingerprint: a different budget can produce a different basis,
    so reusing one for the other would silently substitute a basis the
    caller did not ask for.

    Parameters
    ----------
    time_budget : float
        Seconds the whole search may take, per sub-network. When it runs out
        the best basis found so far that satisfies every no-regression bound
        is returned; a budget of zero returns the starting basis untouched.
        The default is enough for a 10,000-bus sub-network, where the search
        converges in about 35 seconds.
    """

    time_budget: float = 300.0

    def fingerprint(self) -> str:
        """Content hash of everything that can change the result."""
        payload = json.dumps(
            [round(float(self.time_budget), 6)], separators=(",", ":")
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:32]


#: Module level default, used when a network carries no configuration.
_DEFAULT_CONFIG = CycleBasisConfig()


def get_default_config() -> CycleBasisConfig:
    """The configuration ``find_cycles`` falls back to."""
    return _DEFAULT_CONFIG


def configure_network(n: Any, config: CycleBasisConfig | None) -> None:
    """
    Attach a cycle basis configuration to one network.

    ``determine_network_topology`` calls ``find_cycles`` without arguments,
    so this is how a caller selects the basis construction for a particular
    network. Changing it invalidates that network's cached bases, because
    the configuration is part of the cache key.
    """
    if config is None:
        n.__dict__.pop("_cycle_basis_config", None)
    else:
        n.__dict__["_cycle_basis_config"] = config


def network_config(n: Any) -> CycleBasisConfig:
    """The configuration in force for ``n``."""
    return n.__dict__.get("_cycle_basis_config") or _DEFAULT_CONFIG


# ---------------------------------------------------------------------------
# mutable search state
# ---------------------------------------------------------------------------


class _BasisState:
    """
    A cycle basis together with everything the search scores it by, kept up
    to date incrementally.

    An exchange touches only the edges in ``old_cycle`` symmetric difference
    ``new_cycle``, so recomputing the whole basis after every move -- the
    obvious implementation -- would turn an O(cycle length) step into an
    O(total length) one. All four score components are therefore updated in
    place: the fill proxy from its affine marginal, the length sums
    arithmetically, and the maximum length from a histogram of lengths.
    Integer arithmetic throughout, so the incremental values are exact and
    :meth:`recomputed_score` must reproduce them bit for bit -- which is
    asserted at the end of every stage.
    """

    __slots__ = (
        "bg",
        "offset",
        "cycles",
        "counts",
        "edge_to_cycles",
        "phi",
        "sum_len",
        "sum_len_sq",
        "len_hist",
        "max_len",
    )

    def __init__(self, bg: BranchGraph, cycles: Iterable[Iterable[int]]) -> None:
        self.bg = bg
        self.offset = bg.offset
        self.cycles = [set(c) for c in cycles]
        self.counts = [0] * bg.n_edges
        self.edge_to_cycles: list[set[int]] = [set() for _ in range(bg.n_edges)]
        for i, cycle in enumerate(self.cycles):
            for e in cycle:
                self.counts[e] += 1
                self.edge_to_cycles[e].add(i)
        self.phi = _fill_cost(self.offset, self.counts)
        self.sum_len = sum(len(c) for c in self.cycles)
        self.sum_len_sq = sum(len(c) * len(c) for c in self.cycles)
        self.len_hist: dict[int, int] = {}
        for cycle in self.cycles:
            self.len_hist[len(cycle)] = self.len_hist.get(len(cycle), 0) + 1
        self.max_len = max(self.len_hist, default=0)

    def score(self) -> tuple[int, int, int, int]:
        """
        The lexicographic objective ``(Phi, sum l^2, sum l, max l)``.

        ``Phi`` leads because fill is what the barrier pays for; the length
        terms break its ties towards a smaller and flatter KVL block.
        """
        return (self.phi, self.sum_len_sq, self.sum_len, self.max_len)

    def recomputed_score(self) -> tuple[int, int, int, int]:
        """The same score recomputed from scratch, to validate the
        incremental bookkeeping at the end of a stage."""
        counts = [0] * self.bg.n_edges
        for cycle in self.cycles:
            for e in cycle:
                counts[e] += 1
        lengths = [len(c) for c in self.cycles]
        return (
            _fill_cost(self.offset, counts),
            sum(x * x for x in lengths),
            sum(lengths),
            max(lengths, default=0),
        )

    def evaluate(
        self, index: int, new_edges: set[int]
    ) -> tuple[tuple[int, int, int, int], set[int], set[int]]:
        """
        Score the basis that results from replacing cycle ``index`` with
        ``new_edges``, without changing anything.
        """
        old = self.cycles[index]
        removed = old - new_edges
        added = new_edges - old
        old_len = len(old)
        new_len = len(new_edges)
        score = (
            self.phi + _fill_delta(self.offset, self.counts, removed, added),
            self.sum_len_sq - old_len * old_len + new_len * new_len,
            self.sum_len - old_len + new_len,
            self._max_len_after(old_len, new_len),
        )
        return score, removed, added

    def _max_len_after(self, old_len: int, new_len: int) -> int:
        """Maximum cycle length once ``old_len`` is replaced by ``new_len``."""
        max_len = self.max_len
        if new_len >= max_len:
            return new_len
        if old_len < max_len or self.len_hist.get(max_len, 0) > 1:
            return max_len
        # the unique longest cycle is shrinking: rescan the (few) distinct
        # lengths for the runner-up
        best = new_len
        for length, n in self.len_hist.items():
            if n > 0 and (length != old_len or n > 1) and length > best:
                best = length
        return best

    def apply(
        self,
        index: int,
        new_edges: set[int],
        score: tuple[int, int, int, int],
        removed: set[int],
        added: set[int],
    ) -> None:
        """Commit an exchange evaluated by :meth:`evaluate`."""
        counts = self.counts
        edge_to_cycles = self.edge_to_cycles
        for e in removed:
            counts[e] -= 1
            edge_to_cycles[e].discard(index)
        for e in added:
            counts[e] += 1
            edge_to_cycles[e].add(index)

        old_len = len(self.cycles[index])
        new_len = len(new_edges)
        hist = self.len_hist
        hist[old_len] -= 1
        if hist[old_len] == 0:
            del hist[old_len]
        hist[new_len] = hist.get(new_len, 0) + 1

        self.cycles[index] = set(new_edges)
        self.phi, self.sum_len_sq, self.sum_len, self.max_len = score

    def snapshot(self) -> list[frozenset[int]]:
        return [frozenset(c) for c in self.cycles]


# ---------------------------------------------------------------------------
# acceptance rules
# ---------------------------------------------------------------------------

_FILL = "fill"
_LENGTH = "length"


def _is_better(objective: str, candidate: tuple, current: tuple) -> bool:
    """
    Whether a move is an improvement under the stage's objective.

    ``fill`` uses the full lexicographic score, so a move that lowers the
    fill proxy is taken even if it makes a cycle longer. ``length`` is the
    final refinement: it shortens the basis but may not give back anything
    on the fill proxy.
    """
    if objective == _FILL:
        return candidate < current
    return candidate[0] <= current[0] and candidate[1:] < current[1:]


def _no_regression(score: tuple, baseline: tuple) -> bool:
    """
    The hard constraints the returned basis must satisfy against the basis
    the search started from: the sum of squared lengths, the total length
    and the maximum length may none of them grow.

    ``Phi`` is absent on purpose -- it is the objective, and the search only
    ever returns a basis whose full lexicographic score is at most the
    starting one's, so it is bounded as well.
    """
    return (
        score[1] <= baseline[1]
        and score[2] <= baseline[2]
        and score[3] <= baseline[3]
    )


def _make_guard(baseline: tuple, slack: float) -> Callable[[tuple], bool]:
    """
    The bound the search itself respects.

    Escaping a local optimum needs room to move through worse states, so the
    length limits are relaxed by ``slack`` *during* the search; the maximum
    cycle length is not relaxed, because it sets the width of the padded KVL
    coefficient arrays and there is nothing to be gained by letting it grow.
    The result is always re-checked against :func:`_no_regression` with no
    slack at all.
    """
    limit_sq = int(baseline[1] * (1.0 + slack))
    limit_len = int(baseline[2] * (1.0 + slack))
    limit_max = baseline[3]

    def guard(score: tuple) -> bool:
        return (
            score[1] <= limit_sq
            and score[2] <= limit_len
            and score[3] <= limit_max
        )

    return guard


# ---------------------------------------------------------------------------
# breadth-first cycle generation
# ---------------------------------------------------------------------------


def _bfs_fundamental_cycle_edges(
    bg: BranchGraph, root: int, edges: Sequence[int]
) -> list[set[int]]:
    """
    Fundamental cycles of the breadth-first tree of ``root``'s component,
    as edge index sets.

    A breadth-first tree keeps tree paths close to graph shortest paths, so
    most fundamental cycles come out small; the chord's cycle is found by
    climbing both endpoints to their lowest common ancestor, which costs one
    step per edge of the cycle and needs no separate LCA structure. The
    adjacency is pre-sorted in :class:`BranchGraph`, so the tree -- and
    hence the basis -- depends only on the branch set.
    """
    adj = bg.adj
    parent_edge = {root: -1}
    parent = {root: -1}
    depth = {root: 0}
    order = [root]
    head = 0
    while head < len(order):
        u = order[head]
        head += 1
        du = depth[u]
        for v, e in adj[u]:
            if v not in depth:
                depth[v] = du + 1
                parent[v] = u
                parent_edge[v] = e
                order.append(v)

    tree_edges = {e for e in parent_edge.values() if e >= 0}
    cycles: list[set[int]] = []
    for e in edges:
        if e in tree_edges:
            continue
        u, v = bg.edges[e]
        cycle = {e}
        a, b = u, v
        while depth[a] > depth[b]:
            cycle.add(parent_edge[a])
            a = parent[a]
        while depth[b] > depth[a]:
            cycle.add(parent_edge[b])
            b = parent[b]
        while a != b:
            cycle.add(parent_edge[a])
            a = parent[a]
            cycle.add(parent_edge[b])
            b = parent[b]
        cycles.append(cycle)
    return cycles


def _ranked_roots(bg: BranchGraph, component_nodes: Sequence[int]) -> list[int]:
    """
    Nodes of one component ranked by descending degree, ties broken by node
    index so the ranking never depends on iteration order.
    """
    return sorted(component_nodes, key=lambda u: (-len(bg.adj[u]), u))


# ---------------------------------------------------------------------------
# pairwise basis exchange
# ---------------------------------------------------------------------------


def _move_order(state: _BasisState, objective: str) -> list[int]:
    """
    Which basis cycle to try to improve first.

    For the fill objective that is the cycle sitting on the busiest branch,
    because that is where the convex cost has the steepest marginal; for the
    length refinement it is the longest cycle, as in the baseline algorithm.
    Both orders are total (the cycle index breaks every tie), so the pass is
    reproducible.
    """
    cycles = state.cycles
    if objective == _LENGTH:
        return sorted(range(len(cycles)), key=lambda i: (-len(cycles[i]), i))
    counts = state.counts
    offset = state.bg.offset
    keyed = []
    for i, cycle in enumerate(cycles):
        busiest = max((offset[e] + counts[e] for e in cycle), default=0)
        keyed.append((-busiest, -len(cycle), i))
    keyed.sort()
    return [k[2] for k in keyed]


def _pairwise_pass(
    state: _BasisState,
    objective: str,
    guard: Callable[[tuple], bool],
    deadline: float,
    tabu: list[set[frozenset[int]]] | None = None,
) -> int:
    """
    One sweep of elementary pairwise exchanges ``A <- A xor B``.

    Replacing ``A`` by ``A xor B`` (or ``B`` by it) is the cycle-space
    equivalent of a Gaussian elimination pivot: the other basis vector is
    untouched, so independence and the spanned space are preserved by
    construction and no rank computation is needed inside the loop. The
    symmetric difference is only admissible when it is itself one connected
    simple cycle, which :func:`cycle_node_order` checks -- a disjoint union
    of loops is a perfectly good cycle-space element but cannot be written
    as the single cyclic node list the KVL matrix builder expects.

    Only cycles that share an edge with ``A`` are considered, via the
    edge -> cycle index, and each candidate is scored by the incremental
    delta rather than by rescoring the basis.
    """
    cycles = state.cycles
    swaps = 0
    checked = 0
    for a in _move_order(state, objective):
        checked += 1
        if not checked % 256 and time.monotonic() >= deadline:
            break
        current = state.score()
        neighbours: set[int] = set()
        for e in cycles[a]:
            neighbours |= state.edge_to_cycles[e]
        neighbours.discard(a)

        best: tuple | None = None
        for b in sorted(neighbours):
            new = cycles[a] ^ cycles[b]
            if len(new) < 3:
                continue
            if cycle_node_order(state.bg, new) is None:
                continue
            frozen = frozenset(new)
            for replace in (a, b):
                if tabu is not None and frozen in tabu[replace]:
                    continue
                evaluated = state.evaluate(replace, new)
                if evaluated is None:
                    continue
                score, removed, added = evaluated
                if not guard(score) or not _is_better(objective, score, current):
                    continue
                if best is None or (score, replace) < best[0]:
                    best = ((score, replace), replace, new, score, removed, added)
        if best is not None:
            _, replace, new, score, removed, added = best
            state.apply(replace, new, score, removed, added)
            if tabu is not None:
                tabu[replace].add(frozenset(new))
            swaps += 1
    return swaps


def _kick(
    state: _BasisState,
    guard: Callable[[tuple], bool],
    deadline: float,
    tabu: list[set[frozenset[int]]],
    budget: int,
) -> int:
    """
    Leave a local optimum with a bounded number of deliberately uphill
    exchanges.

    The move is chosen deterministically as the least bad admissible one on
    the busiest branch, it must still satisfy the search guard, and the
    tabu list stops the very next pass from undoing it. The caller has
    already stored the best feasible basis, so a kick can never lose it.
    """
    offset = state.bg.offset
    counts = state.counts
    kicks = 0
    busiest = sorted(
        range(state.bg.n_edges),
        key=lambda e: (-(offset[e] + counts[e]), e),
    )
    for edge in busiest:
        if kicks >= budget or time.monotonic() >= deadline:
            break
        incident = sorted(state.edge_to_cycles[edge])
        current = state.score()
        best: tuple | None = None
        for i, a in enumerate(incident):
            for b in incident[i + 1 :]:
                new = state.cycles[a] ^ state.cycles[b]
                if len(new) < 3 or cycle_node_order(state.bg, new) is None:
                    continue
                frozen = frozenset(new)
                for replace in (a, b):
                    if frozen in tabu[replace]:
                        continue
                    evaluated = state.evaluate(replace, new)
                    if evaluated is None:
                        continue
                    score, removed, added = evaluated
                    if not guard(score) or score <= current:
                        continue
                    if best is None or (score, replace) < best[0]:
                        best = ((score, replace), replace, new, score, removed, added)
        if best is None:
            continue
        _, replace, new, score, removed, added = best
        state.apply(replace, new, score, removed, added)
        tabu[replace].add(frozenset(new))
        kicks += 1
    return kicks


# ---------------------------------------------------------------------------
# stage three: exchange against cycles from outside the basis
# ---------------------------------------------------------------------------


def _spanning_forest_chords(bg: BranchGraph) -> tuple[list[int], dict[int, int]]:
    """
    A deterministic spanning forest and the chord coordinates it induces.

    Every element of the cycle space is determined by which chords it uses,
    so projecting a cycle onto the chords is an isomorphism onto
    ``GF(2)^mu``: independence and span can be decided there, on ``mu`` bit
    integers, instead of on the full edge sets.
    """
    seen = [False] * bg.n_nodes
    tree_edges: set[int] = set()
    for start in range(bg.n_nodes):
        if seen[start]:
            continue
        seen[start] = True
        queue = [start]
        head = 0
        while head < len(queue):
            u = queue[head]
            head += 1
            for v, e in bg.adj[u]:
                if not seen[v]:
                    seen[v] = True
                    tree_edges.add(e)
                    queue.append(v)
    chords = [e for e in range(bg.n_edges) if e not in tree_edges]
    return chords, {e: pos for pos, e in enumerate(chords)}


def _coordinate_system(
    bg: BranchGraph, cycles: Sequence[Iterable[int]], chord_position: dict[int, int]
) -> tuple[list[int], list[int]]:
    """
    Triangularise the basis in chord coordinates.

    ``pivot_vector[p]`` is a chord-coordinate vector whose highest set bit is
    ``p``, and ``pivot_combination[p]`` records which basis cycles were
    XORed to build it. Together they solve "write this cycle as a
    combination of the current basis", which is what stage three needs to
    know which basis cycles a candidate may legally replace.

    Raises ``ValueError`` if ``cycles`` is not a basis.
    """
    dimension = len(cycles)
    pivot_vector = [0] * dimension
    pivot_combination = [0] * dimension
    for index, cycle in enumerate(cycles):
        vector = 0
        for e in cycle:
            position = chord_position.get(e)
            if position is not None:
                vector |= 1 << position
        combination = 1 << index
        while vector:
            pivot = vector.bit_length() - 1
            if pivot_vector[pivot]:
                vector ^= pivot_vector[pivot]
                combination ^= pivot_combination[pivot]
            else:
                pivot_vector[pivot] = vector
                pivot_combination[pivot] = combination
                break
        if not vector:
            msg = "cycles are linearly dependent, not a basis"
            raise ValueError(msg)
    if any(v == 0 for v in pivot_vector):
        msg = "cycles do not span the cycle space"
        raise ValueError(msg)
    return pivot_vector, pivot_combination


def _stage_three(
    state: _BasisState,
    config: CycleBasisConfig,
    guard: Callable[[tuple], bool],
    deadline: float,
) -> int:
    """
    Exchange basis cycles against simple cycles from *outside* the basis.

    Pairwise XOR can only reach bases that differ from the current one by a
    chain of two-cycle combinations, and it stalls early. A candidate cycle
    ``q`` taken from another BFS tree is written as a GF(2) combination of
    the current basis; by the binary matroid exchange property *any* basis
    cycle whose coefficient in that combination is one may be replaced by
    ``q``, and the result is again a basis. Since ``q`` came from a BFS tree
    it is a connected simple cycle, so the replacement is representable.

    The coordinate bookkeeping is updated after each exchange, and the
    caller re-derives the rank from scratch afterwards rather than trusting
    it.
    """
    bg = state.bg
    dimension = len(state.cycles)
    if dimension == 0:
        return 0

    chords, chord_position = _spanning_forest_chords(bg)
    if len(chords) != dimension:
        logger.debug(
            "cycle basis: skipping candidate exchange, rank %d != chords %d",
            dimension,
            len(chords),
        )
        return 0
    pivot_vector, pivot_combination = _coordinate_system(
        bg, state.cycles, chord_position
    )

    comp_nodes = bg.component_nodes()
    comp_edges = bg.component_edges()
    roots: list[tuple[int, list[int]]] = []
    for comp, nodes in enumerate(comp_nodes):
        if not comp_edges[comp]:
            continue
        # Every node is a usable root; the order is what matters, since the
        # deadline decides how far down the list the search gets. High
        # degree roots first -- they generate the shortest fundamental
        # cycles -- then a deterministic stride through the rest so that the
        # periphery of a large component is reached early rather than last.
        ranked = _ranked_roots(bg, nodes)
        split = max(1, len(ranked) // 2)
        head, tail_pool = ranked[:split], ranked[split:]
        stride = max(1, len(tail_pool) // max(1, split))
        tail = tail_pool[SEED % stride :: stride]
        seen = set(head) | set(tail)
        rest = [node for node in tail_pool if node not in seen]
        for root in head + tail + rest:
            roots.append((comp, root))

    swaps = 0
    while True:
        round_swaps = 0
        for comp, root in roots:
            if time.monotonic() >= deadline:
                return swaps
            candidates = _bfs_fundamental_cycle_edges(bg, root, comp_edges[comp])
            candidates.sort(key=lambda c: (len(c), sorted(c)))
            for candidate in candidates:
                # a BFS fundamental cycle is simple by construction, but the
                # basis is only allowed to hold cycles the KVL builder can
                # write as one cyclic node list, so check rather than assume
                if cycle_node_order(bg, candidate) is None:
                    continue
                coordinate = 0
                for e in candidate:
                    position = chord_position.get(e)
                    if position is not None:
                        coordinate |= 1 << position
                if not coordinate:
                    continue

                representation = 0
                vector = coordinate
                while vector:
                    pivot = vector.bit_length() - 1
                    vector ^= pivot_vector[pivot]
                    representation ^= pivot_combination[pivot]

                current = state.score()
                best: tuple | None = None
                choices = representation
                while choices:
                    bit = choices & -choices
                    replace = bit.bit_length() - 1
                    choices ^= bit
                    if state.cycles[replace] == candidate:
                        best = None
                        break
                    evaluated = state.evaluate(replace, candidate)
                    if evaluated is None:
                        continue
                    score, removed, added = evaluated
                    if not guard(score) or not _is_better(_FILL, score, current):
                        continue
                    if best is None or (score, replace) < best[0]:
                        best = (
                            (score, replace),
                            replace,
                            score,
                            removed,
                            added,
                        )
                if best is None:
                    continue

                _, replace, score, removed, added = best
                state.apply(replace, candidate, score, removed, added)
                # candidate == XOR of the basis cycles in `representation`,
                # so after the exchange the old row `replace` equals
                # candidate XOR (the rest of the representation). Every
                # stored combination that used `replace` has to absorb that.
                adjustment = representation ^ (1 << replace)
                replace_bit = 1 << replace
                for pivot in range(dimension):
                    if pivot_combination[pivot] & replace_bit:
                        pivot_combination[pivot] ^= adjustment
                swaps += 1
                round_swaps += 1
        if round_swaps == 0 or time.monotonic() >= deadline:
            break
    return swaps


# ---------------------------------------------------------------------------
# verification
# ---------------------------------------------------------------------------


def gf2_rank(bg: BranchGraph, cycles: Sequence[Iterable[int]]) -> int:
    """
    Rank over GF(2) of a set of cycle-space elements.

    Computed in chord coordinates, which is an isomorphism of the cycle
    space onto ``GF(2)^mu``: a tree edge's coefficient is determined by the
    chords, so dropping it loses no information.
    """
    _, chord_position = _spanning_forest_chords(bg)
    pivots: dict[int, int] = {}
    rank = 0
    for cycle in cycles:
        vector = 0
        for e in cycle:
            position = chord_position.get(e)
            if position is not None:
                vector |= 1 << position
        while vector:
            pivot = vector.bit_length() - 1
            if pivot not in pivots:
                pivots[pivot] = vector
                rank += 1
                break
            vector ^= pivots[pivot]
    return rank


def verify_cycle_basis(
    bg: BranchGraph,
    cycles: Sequence[Iterable[int]],
    expected_rank: int | None = None,
    reference: Sequence[Iterable[int]] | None = None,
) -> dict[str, Any]:
    """
    Check a basis against everything the KVL constraints rely on.

    Returns a report with one boolean per property rather than raising, so
    a benchmark can print it and a caller can decide what to do:

    ``simple_cycles``
        every element is one connected simple cycle,
    ``count``/``rank``
        the number of cycles and the GF(2) rank both equal the cycle rank,
    ``spans_reference``
        every cycle of ``reference`` is a GF(2) combination of ``cycles``,
        i.e. the two span the same space (given equal rank),
    ``circulation``
        the signed incidence matrix times the cycle matrix is zero, i.e.
        every column of ``C`` is a circulation and the coefficients are all
        in ``{-1, 0, +1}``.
    """
    rank_expected = bg.simple_cycle_rank if expected_rank is None else expected_rank
    report: dict[str, Any] = {
        "count": len(cycles) == rank_expected,
        "n_cycles": len(cycles),
        "expected_rank": rank_expected,
    }
    report["simple_cycles"] = all(
        cycle_node_order(bg, cycle) is not None for cycle in cycles
    )
    rank = gf2_rank(bg, cycles)
    report["gf2_rank"] = rank
    report["rank"] = rank == rank_expected

    if reference is not None:
        _, chord_position = _spanning_forest_chords(bg)
        pivots: dict[int, int] = {}
        for cycle in cycles:
            vector = 0
            for e in cycle:
                position = chord_position.get(e)
                if position is not None:
                    vector |= 1 << position
            while vector:
                pivot = vector.bit_length() - 1
                if pivot not in pivots:
                    pivots[pivot] = vector
                    break
                vector ^= pivots[pivot]
        spans = True
        for cycle in reference:
            vector = 0
            for e in cycle:
                position = chord_position.get(e)
                if position is not None:
                    vector |= 1 << position
            while vector:
                pivot = vector.bit_length() - 1
                if pivot not in pivots:
                    spans = False
                    break
                vector ^= pivots[pivot]
        report["spans_reference"] = spans

    report["circulation"] = _check_circulation(bg, cycles)
    report["ok"] = all(
        bool(report[key])
        for key in ("count", "simple_cycles", "rank", "circulation")
        if key in report
    ) and report.get("spans_reference", True)
    return report


def _check_circulation(bg: BranchGraph, cycles: Sequence[Iterable[int]]) -> bool:
    """
    Every cycle, oriented along its own walk, must balance at every node.

    This is the property the voltage law needs: the node-branch incidence
    matrix times the cycle column is zero, with all coefficients in
    ``{-1, 0, +1}``.
    """
    lookup = bg.edge_lookup()
    for cycle in cycles:
        cycle = set(cycle)
        order = cycle_node_order(bg, cycle)
        if order is None:
            return False
        balance: dict[int, int] = {}
        walked: set[int] = set()
        length = len(order)
        for i, u in enumerate(order):
            v = order[(i + 1) % length]
            edge = lookup.get((u, v) if u < v else (v, u))
            if edge is None:
                return False
            walked.add(edge)
            # +1 leaving u, -1 entering v: one unit of flow around the walk
            balance[u] = balance.get(u, 0) + 1
            balance[v] = balance.get(v, 0) - 1
        # the walk has to use exactly the given edges once each, so the
        # column is the cycle itself with coefficients in {-1, 0, +1}
        if walked != cycle or len(walked) != length:
            return False
        if any(value != 0 for value in balance.values()):
            return False
    return True


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def basis_metrics(
    bg: BranchGraph, cycles: Sequence[Iterable[int]]
) -> dict[str, Any]:
    """
    Every diagnostic worth reporting for one basis of one sub-network.

    Participation is reported per *branch*, so the fixed two-edge cycles of
    parallel branches are included: the representative branch of a parallel
    edge carries them on top of its basis cycles, and each further parallel
    branch carries exactly one.
    """
    counts = [0] * bg.n_edges
    for cycle in cycles:
        for e in cycle:
            counts[e] += 1

    branch_participation: list[int] = []
    for e in range(bg.n_edges):
        branch_participation.append(bg.offset[e] + counts[e])
        branch_participation.extend([1] * bg.offset[e])

    lengths = [len(c) for c in cycles]
    participation = np.asarray(branch_participation, dtype=np.int64)
    histogram = np.bincount(participation) if participation.size else np.zeros(1, int)
    return {
        "n_cycles": len(cycles) + sum(bg.offset),
        "n_fixed_two_edge_cycles": sum(bg.offset),
        "cycle_rank": bg.cycle_rank,
        "simple_cycle_rank": bg.simple_cycle_rank,
        "nodes": bg.n_nodes,
        "branches": bg.n_branches,
        "max_participation": int(participation.max()) if participation.size else 0,
        "participation_histogram": {
            int(p): int(n) for p, n in enumerate(histogram) if n
        },
        "sum_participation_sq": int((participation**2).sum()),
        "phi": _fill_cost(bg.offset, counts),
        "sum_length": sum(lengths) + 2 * sum(bg.offset),
        "sum_length_sq": sum(x * x for x in lengths) + 4 * sum(bg.offset),
        # a fixed two-edge cycle is a cycle of length two, so it sets the
        # floor of the maximum once there is one
        "max_length": max(lengths + ([2] if sum(bg.offset) else []), default=0),
    }


# ---------------------------------------------------------------------------
# the search
# ---------------------------------------------------------------------------


@dataclass
class CycleBasisResult:
    """What :func:`optimize_cycle_basis` returns."""

    cycles: list[set[int]]
    score: tuple[int, int, int, int]
    baseline_score: tuple[int, int, int, int]
    stages: list[dict[str, Any]] = field(default_factory=list)
    fell_back: bool = False
    reason: str = ""

    @property
    def improved(self) -> bool:
        return self.score < self.baseline_score


def optimize_cycle_basis(
    bg: BranchGraph,
    baseline: Sequence[Iterable[int]],
    config: CycleBasisConfig,
) -> CycleBasisResult:
    """
    Search for a cycle basis with a lower structural fill proxy than
    ``baseline``, never returning a worse one.

    Three stages, in order:

    1. pairwise exchange ``A <- A xor B`` against the fill objective, with
       bounded uphill kicks to leave local optima;
    2. exchange against simple cycles from outside the basis, using GF(2)
       chord coordinates to find which basis cycles a candidate may legally
       replace;
    3. a final length refinement that may not worsen either fill proxy.

    Between stages the incremental score is checked against a full
    recomputation and the best basis seen that satisfies the no-regression
    constraints is kept. Whatever happens -- budget exhausted, validation
    failure, an exception in a stage -- the returned basis is either an
    improvement on ``baseline`` or ``baseline`` itself.

    Complexity: a pairwise pass visits every basis cycle once and, per cycle,
    only the cycles sharing one of its edges, so a pass is
    ``O(sum_c l_c * max_e p_e)`` set operations, each exchange being scored by
    an ``O(l)`` delta rather than an ``O(sum_c l_c)`` rescoring. The candidate
    exchange adds ``O(mu)`` big-integer XORs per accepted exchange for the
    coordinate update. Nothing in the search is a rank computation; the
    ``O(mu * nnz)`` GF(2) elimination runs once, at the end, as verification.
    """
    baseline_cycles = [set(c) for c in baseline]
    state = _BasisState(bg, baseline_cycles)
    baseline_score = state.score()
    result = CycleBasisResult(
        cycles=[set(c) for c in baseline_cycles],
        score=baseline_score,
        baseline_score=baseline_score,
    )
    if not baseline_cycles:
        result.reason = "no cycles"
        return result

    start = time.monotonic()
    # ``>=`` everywhere below, so that a zero budget really means no time:
    # ``time.monotonic()`` has a ~15 ms resolution on some platforms and a
    # strict ``>`` would let a whole stage run inside one tick.
    budget = max(0.0, config.time_budget)
    deadline = start + budget
    fill_deadline = start + budget * STAGE_SHARES[0]
    candidate_deadline = start + budget * STAGE_SHARES[1]
    guard = _make_guard(baseline_score, SEARCH_SLACK)
    best = state.snapshot()
    best_score = baseline_score

    def record(name: str, t0: float, **extra: Any) -> None:
        entry = {
            "stage": name,
            "seconds": round(time.monotonic() - t0, 3),
            "score": state.score(),
            **extra,
        }
        result.stages.append(entry)
        logger.debug("cycle basis stage %s", entry)

    def keep_best() -> None:
        nonlocal best, best_score
        score = state.score()
        if _no_regression(score, baseline_score) and score < best_score:
            best = state.snapshot()
            best_score = score

    try:
        # -- fill exchange ----------------------------------------------
        t0 = time.monotonic()
        tabu: list[set[frozenset[int]]] = [set() for _ in state.cycles]
        swaps = 0
        kicks = 0
        # A pass that finds nothing gets one perturbation to work with. If
        # the pass after that still finds nothing, the stage is done and the
        # rest of the clock belongs to the next one -- kicking again would
        # only wander. Measured on a 10,000-bus case, the old "kick until the
        # deadline" loop spent 8 seconds on 604 kicks in the smallest
        # sub-network and made the *final* basis worse, because the later
        # stages start from wherever the wandering left it.
        stalled = False
        while time.monotonic() < fill_deadline:
            moved = _pairwise_pass(state, _FILL, guard, fill_deadline, tabu)
            swaps += moved
            keep_best()
            if moved:
                stalled = False
                continue
            if stalled:
                break
            kicked = _kick(state, guard, fill_deadline, tabu, KICK_BUDGET)
            kicks += kicked
            if kicked == 0:
                break
            stalled = True
        _validate_incremental(state, "fill exchange")
        record("fill-exchange", t0, swaps=swaps, kicks=kicks)

        # -- candidate exchange -----------------------------------------
        t0 = time.monotonic()
        candidate_swaps = 0
        try:
            candidate_swaps = _stage_three(
                state, config, guard, max(candidate_deadline, time.monotonic())
            )
        except ValueError as error:  # pragma: no cover - defensive
            logger.debug("cycle basis: candidate exchange skipped (%s)", error)
        keep_best()
        _validate_incremental(state, "candidate exchange")
        record("candidate-exchange", t0, swaps=candidate_swaps)

        # -- length refinement ------------------------------------------
        t0 = time.monotonic()
        swaps = 0
        while time.monotonic() < deadline:
            moved = _pairwise_pass(state, _LENGTH, guard, deadline)
            swaps += moved
            keep_best()
            if moved == 0:
                break
        _validate_incremental(state, "length refinement")
        record("length-refinement", t0, swaps=swaps)

    except Exception as error:  # pragma: no cover - defensive
        logger.warning(
            "cycle basis: %s search failed (%s), falling back to the "
            "unrefined basis",
            BFS_REFINED,
            error,
        )
        result.fell_back = True
        result.reason = f"exception: {error}"
        return result

    keep_best()
    if best_score >= baseline_score:
        result.reason = "no improvement over the baseline"
        return result

    report = verify_cycle_basis(
        bg, best, expected_rank=len(baseline_cycles), reference=baseline_cycles
    )
    if not report["ok"]:
        logger.warning(
            "cycle basis: %s produced an invalid basis (%s), falling back to "
            "the unrefined basis",
            BFS_REFINED,
            report,
        )
        result.fell_back = True
        result.reason = f"verification failed: {report}"
        return result

    result.cycles = [set(c) for c in best]
    result.score = best_score
    result.reason = "improved"
    logger.info(
        "cycle basis: %s improved Phi %d -> %d, length %d -> %d in %.1fs",
        BFS_REFINED,
        baseline_score[0],
        best_score[0],
        baseline_score[2],
        best_score[2],
        time.monotonic() - start,
    )
    return result


def _validate_incremental(state: _BasisState, stage: str) -> None:
    """
    Confirm at a stage boundary that the incremental score still matches a
    full recomputation.

    Everything is integer arithmetic, so this is an exact equality; a
    mismatch is a bug in the bookkeeping, not a rounding artefact, and
    raising here sends the caller down the fallback path.
    """
    recomputed = state.recomputed_score()
    if recomputed != state.score():
        msg = (
            f"incremental score drifted during {stage}: "
            f"{state.score()} != {recomputed}"
        )
        raise AssertionError(msg)


def _bfs_fundamental_cycles_from_root(graph: nx.Graph, root: Any) -> list[list[Any]]:
    """
    Fundamental cycle basis from a breadth-first spanning tree of ``graph``
    rooted at ``root``.

    ``nx.cycle_basis`` builds its spanning tree by an arbitrary depth-first
    traversal, which gives no control over how long the resulting
    fundamental cycles are: a chord that happens to connect two nodes far
    apart on that DFS tree drags in every branch on the tree path between
    them. A breadth-first tree keeps tree-paths close to shortest paths in
    the graph, which keeps most fundamental cycles small.

    Returns a list of cycles in the same format as ``nx.cycle_basis``: each
    cycle is a list of nodes in cyclic order, i.e. consecutive entries
    (with wraparound) are connected by a graph edge.
    """
    tree = nx.bfs_tree(graph, root)

    parent = dict.fromkeys([root])
    for u, v in tree.edges():
        parent[v] = u

    tree_edges = {frozenset(e) for e in tree.edges()}
    chords = [e for e in graph.edges() if frozenset(e) not in tree_edges]
    if not chords:
        return []

    lca = dict(
        nx.tree_all_pairs_lowest_common_ancestor(tree, root=root, pairs=chords)
    )

    cycles: list[list[Any]] = []
    for u, v in chords:
        ancestor = lca.get((u, v), lca.get((v, u)))

        up = [u]
        while up[-1] != ancestor:
            up.append(parent[up[-1]])

        down = [v]
        while down[-1] != ancestor:
            down.append(parent[down[-1]])

        cycles.append(up + down[-2::-1])

    return cycles


def _multi_root_bfs_cycles(graph: nx.Graph, num_roots: int = 5) -> list[list[Any]]:
    """
    Fundamental cycle basis picked from the best of several breadth-first
    spanning trees per connected component.

    A single BFS tree already keeps most fundamental cycles small (see
    ``_bfs_fundamental_cycles_from_root``), but which node is the root still
    matters: a chord that lands far from the root on the tree still drags in
    a long tree path. Trying a handful of high-degree roots and keeping
    whichever tree gives the smallest worst-case cycle catches cases the
    single highest-degree root misses, for a small constant-factor cost
    (``num_roots`` BFS traversals instead of one). Like the single-root
    version, this is purely topological.
    """
    cycles: list[list[Any]] = []

    for component in nx.connected_components(graph):
        if len(component) < 2:
            continue

        sub = graph.subgraph(component)
        roots = [
            node
            for node, _ in sorted(sub.degree(), key=itemgetter(1), reverse=True)[
                :num_roots
            ]
        ]

        best: list[list[Any]] | None = None
        best_score: tuple[int, int] | None = None
        for root in roots:
            candidate = _bfs_fundamental_cycles_from_root(sub, root)
            score = (
                max((len(c) for c in candidate), default=0),
                sum(len(c) for c in candidate),
            )
            if best_score is None or score < best_score:
                best, best_score = candidate, score

        cycles.extend(best or [])

    return cycles


def _edges_to_cycle_order(edges: list[tuple[Any, Any]]) -> list[Any]:
    """
    Turn an unordered edge set that is known to form one simple cycle into a
    node list in cyclic order, i.e. the format ``nx.cycle_basis`` and
    ``_bfs_fundamental_cycles_from_root`` use.

    Every node touched by a simple cycle's edges has degree exactly 2 within
    that edge set, so a plain walk (never stepping back the way we came)
    closes the loop.
    """
    adjacency: dict[Any, list[Any]] = {}
    for u, v in edges:
        adjacency.setdefault(u, []).append(v)
        adjacency.setdefault(v, []).append(u)

    start = next(iter(adjacency))
    order = [start]
    previous = None
    current = start
    while True:
        a, b = adjacency[current]
        nxt = b if a == previous else a
        if nxt == start:
            return order
        order.append(nxt)
        previous, current = current, nxt


def _pairwise_exchange_refine(
    cycles: list[list[Any]], max_passes: int = 50
) -> list[list[Any]]:
    """
    Shrink a fundamental cycle basis by repeated pairwise exchange.

    A cycle basis only has to span the graph's cycle space; which specific
    basis gets used is a free choice that does not change the KVL
    constraints it defines (any basis represents the same voltage-law
    equations, just organised as a different set of rows). It does change
    how large those rows are, which matters because the optimization model
    pads every cycle's coefficient/variable arrays to the size of the
    largest cycle in the basis (see
    ``pypsa.optimization.constraints.define_kirchhoff_voltage_constraints``),
    so a handful of very large cycles can bloat memory and solver fill-in
    even though most cycles are tiny.

    For any two basis cycles whose symmetric difference (XOR of their edge
    sets) is itself a single simple cycle shorter than the longer of the
    two, replacing that longer cycle with the symmetric difference keeps the
    basis independent (it is the same elementary row operation as a
    Gaussian-elimination pivot: cycle B is untouched, cycle A becomes
    A XOR B) while strictly shrinking it. Repeating this greedily, longest
    cycles first, until no swap helps converges to a local optimum -- not
    the true minimum cycle basis (that needs a global search, e.g. an
    integer program per cycle), but it captures most of the achievable
    reduction at a tiny fraction of the cost: milliseconds to a few seconds
    even for a sub-network with thousands of independent cycles, versus
    minutes or more for an exact minimum cycle basis solve.

    Only candidate pairs that actually share an edge are ever considered
    (via an edge -> cycle-indices index), and every candidate swap is
    verified -- by checking every touched node has degree exactly 2 in the
    symmetric difference, and that a single walk consumes every edge in it
    -- to reject the cases where two cycles overlap but their symmetric
    difference splits into more than one disjoint loop (which would not be
    representable as one cyclic node list).
    """
    cycles = [list(c) for c in cycles]
    cyc_edges: list[set[frozenset]] = [
        {frozenset((c[i], c[(i + 1) % len(c)])) for i in range(len(c))} for c in cycles
    ]

    for _ in range(max_passes):
        edge_to_cycles: dict[frozenset, list[int]] = {}
        for idx, edges in enumerate(cyc_edges):
            for e in edges:
                edge_to_cycles.setdefault(e, []).append(idx)

        n_swaps = 0
        for a in sorted(range(len(cycles)), key=lambda i: -len(cycles[i])):
            candidates: set[int] = set()
            for e in cyc_edges[a]:
                candidates.update(edge_to_cycles[e])
            candidates.discard(a)

            # The first admissible partner wins (the loop breaks below), so
            # the order this is walked in decides which basis comes out.
            # ``cyc_edges`` holds frozensets of bus labels, whose set
            # iteration order depends on the string hash seed, and that
            # order leaks into ``candidates``; sorting the cycle indices
            # makes the choice -- and the whole basis -- reproducible from
            # one process to the next.
            for b in sorted(candidates):
                if len(cycles[b]) >= len(cycles[a]):
                    continue

                sym_diff = cyc_edges[a] ^ cyc_edges[b]
                if not sym_diff or len(sym_diff) >= len(cycles[a]):
                    continue

                degree: dict[Any, int] = {}
                touches_only_degree_2 = True
                for e in sym_diff:
                    u, v = tuple(e)
                    degree[u] = degree.get(u, 0) + 1
                    degree[v] = degree.get(v, 0) + 1
                    if degree[u] > 2 or degree[v] > 2:
                        touches_only_degree_2 = False
                        break
                if not touches_only_degree_2 or any(d != 2 for d in degree.values()):
                    continue

                new_cycle = _edges_to_cycle_order(list(sym_diff))
                if len(new_cycle) != len(sym_diff):
                    continue  # sym_diff is several disjoint loops, not one

                for e in cyc_edges[a]:
                    edge_to_cycles[e].remove(a)
                cycles[a] = new_cycle
                cyc_edges[a] = sym_diff
                for e in sym_diff:
                    edge_to_cycles.setdefault(e, []).append(a)
                n_swaps += 1
                break  # cycle `a` changed; re-evaluate it on the next pass

        if n_swaps == 0:
            break

    return cycles


def _topology_fingerprint(sub_network: Any) -> str:
    """
    Content hash of a sub-network's active branch topology (component, name,
    bus0, bus1), independent of branch ordering.

    One half of the cycle basis cache key: it is purely a function of which
    buses are connected by which branches, not of impedance or capacity
    values, so it is safe to reuse across repeated
    ``determine_network_topology`` calls (e.g. every outer iteration of
    ``optimize_transmission_expansion_iteratively``, which rebuilds the
    sub-networks and their KVL cycle basis on every solve via
    ``pypsa.optimization.constraints.kirchhoff_voltage_cycles``) as long as
    the branch set hasn't actually changed.

    The other half is the fingerprint of the
    :class:`pypsa.cycle_basis.CycleBasisConfig` in force, because the basis
    is only a function of the topology once the search budget is fixed: a
    topology-only key would happily hand back a basis built under a budget
    the caller did not ask for.
    """
    branches = sub_network.branches()
    if branches.empty:
        return "empty"
    key = "|".join(
        f"{idx}:{bus0}:{bus1}"
        for idx, bus0, bus1 in sorted(
            zip(map(str, branches.index), branches["bus0"], branches["bus1"]),
        )
    )
    return hashlib.sha256(key.encode()).hexdigest()


def _refine_cycles_for_fill(
    sub_network: Any,
    mgraph: nx.MultiGraph,
    baseline: list[list[Any]],
    config: CycleBasisConfig,
) -> list[list[Any]]:
    """
    Second half of ``bfs-refined``: rewrite the basis for lower fill.

    ``baseline`` is what the breadth-first construction above produced, a
    basis already shortened by local exchange. :mod:`pypsa.cycle_basis`
    searches from there for one whose branch columns couple fewer row pairs
    in ``A A'``, which is what an interior point method's Cholesky factor
    grows on; length alone does not measure that.

    Every step that could fail -- a cycle that is not expressible on the
    collapsed simple graph, a search that ends no better than where it
    started, a result that fails verification -- returns ``baseline``
    unchanged, so this stage can only improve the basis, never invalidate
    it.
    """
    bg = BranchGraph.from_multigraph(mgraph)

    edge_cycles: list[set[int]] = []
    for cycle in baseline:
        edges = bg.node_cycle_to_edges(cycle)
        if edges is None:
            logger.warning(
                "cycle basis: sub-network %s has a baseline cycle that is not a "
                "simple cycle of the collapsed graph; keeping the baseline basis",
                sub_network.name,
            )
            return baseline
        edge_cycles.append(edges)

    result = optimize_cycle_basis(bg, edge_cycles, config)
    if result.fell_back or not result.improved:
        logger.debug(
            "cycle basis: sub-network %s keeps the baseline basis (%s)",
            sub_network.name,
            result.reason,
        )
        return baseline

    out: list[list[Any]] = []
    for edges in result.cycles:
        order = bg.edges_to_node_cycle(edges)
        if order is None:  # pragma: no cover - verification rules this out
            logger.warning(
                "cycle basis: sub-network %s produced a cycle that is not a "
                "simple cycle; keeping the baseline basis",
                sub_network.name,
            )
            return baseline
        out.append(order)
    return out


def build_cycle_matrix(
    sub_network: Any,
    weight: str = "x_pu",
    config: CycleBasisConfig | None = None,
) -> Any:
    """
    Build the cycle incidence matrix ``C`` of one sub-network.

    networkx collects the cycles with more than 2 edges; then the 2-edge
    cycles from the MultiGraph must be collected separately (for cases
    where there are multiple lines between the same pairs of buses).

    Cycles with infinite impedance are skipped.

    The cycle basis depends only on topology (which buses are connected by
    which branches) and on the search configuration, never on
    impedance/capacity values, so it is cached on the parent network keyed
    by ``(topology fingerprint, config fingerprint)``: repeated calls with
    an unchanged branch set and an unchanged configuration (e.g. successive
    outer iterations of ``optimize_transmission_expansion_iteratively``,
    which only update impedances/capacities) reuse the cached basis instead
    of recomputing it, while changing the search budget is a cache miss.

    The basis is built in three purely topological, impedance-independent
    steps ("bfs-refined"):

    1. ``_multi_root_bfs_cycles`` picks the best of a handful of
       breadth-first spanning trees;
    2. ``_pairwise_exchange_refine`` greedily shrinks that basis by local
       exchange;
    3. :mod:`pypsa.cycle_basis` searches from there for a basis with a lower
       structural *fill* proxy -- one whose branch columns couple fewer row
       pairs in ``A A'``, which is what an interior point method's Cholesky
       factor grows on. Steps 1 and 2 optimise length, which counts nonzeros
       rather than fill; step 3 optimises fill directly, and on a 10,000-bus
       US case it cuts the worst branch's cycle count from 21 to 7 and the
       total cycle length by a further 12 %.

    Every step only ever replaces basis cycles with verified-independent,
    verified-simple-cycle equivalents, so none of them changes the cycle
    space or the KVL constraints it defines. Step 3 is additionally bounded
    by a time budget and falls back to the step-2 basis on any failure; see
    :mod:`pypsa.cycle_basis` for what "fill" means and
    :class:`pypsa.cycle_basis.CycleBasisConfig` for the budget.
    """
    branches_bus0 = sub_network.branches()["bus0"]
    branches_i = branches_bus0.index

    n = sub_network.network
    if config is None:
        config = network_config(n)
    cache = n.__dict__.setdefault("_cycle_basis_cache", {})
    key = (_topology_fingerprint(sub_network), config.fingerprint())

    if key in cache:
        return cache[key].copy()

    # reduce to a non-multi-graph for cycles with > 2 edges
    mgraph = sub_network.graph(weight=weight, inf_weight=False)
    graph = nx.Graph(mgraph)

    cycles = _multi_root_bfs_cycles(graph)
    cycles = _pairwise_exchange_refine(cycles)
    cycles = _refine_cycles_for_fill(sub_network, mgraph, cycles, config)

    # number of 2-edge cycles
    num_multi = len(mgraph.edges()) - len(graph.edges())

    C = dok_matrix((len(branches_bus0), len(cycles) + num_multi))

    for j, cycle in enumerate(cycles):
        for i in range(len(cycle)):
            branch = next(iter(mgraph[cycle[i]][cycle[(i + 1) % len(cycle)]].keys()))
            branch_i = branches_i.get_loc(branch)
            sign = +1 if branches_bus0.iat[branch_i] == cycle[i] else -1
            C[branch_i, j] += sign

    # counter for multis
    c = len(cycles)

    # add multi-graph 2-edge cycles for multiple branches between same pairs of buses
    for u, v in graph.edges():
        bs = list(mgraph[u][v].keys())
        if len(bs) > 1:
            first = bs[0]
            first_i = branches_i.get_loc(first)
            for b in bs[1:]:
                b_i = branches_i.get_loc(b)
                sign = (
                    -1 if branches_bus0.iat[b_i] == branches_bus0.iat[first_i] else +1
                )
                C[first_i, c] = 1
                C[b_i, c] = sign
                c += 1

    cache[key] = C.copy()
    return C



def sub_network_basis(
    sub_network: Any, weight: str = "x_pu"
) -> tuple[BranchGraph, list[set[int]]]:
    """
    Read back the basis a sub-network's ``C`` matrix encodes.

    Returns the edge-indexed graph and the basis cycles as edge index sets,
    with the fixed two-edge cycles of parallel branches left out -- they are
    not on the simple graph the search works on, and
    :func:`basis_metrics` re-adds them from :attr:`BranchGraph.offset`.

    Used by the diagnostics and the tests to score and verify a basis that
    ``find_cycles`` has already installed, whichever method built it.
    """
    mgraph = sub_network.graph(weight=weight, inf_weight=False)
    bg = BranchGraph.from_multigraph(mgraph)
    edge_of_branch: dict[Any, int] = {}
    for e, keys in enumerate(bg.branches):
        for key in keys:
            edge_of_branch[key] = e

    matrix = sub_network.C.tocsc()
    branches_i = sub_network.branches().index
    cycles: list[set[int]] = []
    for j in range(matrix.shape[1]):
        rows = matrix.indices[matrix.indptr[j] : matrix.indptr[j + 1]]
        if len(rows) <= 2:
            continue  # a fixed two-edge cycle between parallel branches
        cycles.append({edge_of_branch[branches_i[i]] for i in rows})
    return bg, cycles
