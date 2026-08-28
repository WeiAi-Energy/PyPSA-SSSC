#!/usr/bin/env python3
"""
Lower bounds for transmission expansion with capacity-dependent impedance.

``optimize_transmission_expansion_iteratively`` solves a nonconvex problem: the
per-unit impedance of an expandable branch scales with the inverse of its
capacity, so the Kirchhoff voltage law couples the capacity and the flow
bilinearly. Both the fixed point and the trust region converge to a *local*
solution, and neither of them produces a bound that would say how good it is.

This module builds such a bound. Write the voltage law of cycle ``c`` as

.. math::
    \\sum_\\ell C_{\\ell c}\\, g_{\\ell t} = 0, \\qquad
    g_{\\ell t} = \\hat x_\\ell(F_\\ell) f_{\\ell t}
                  - \\tilde q_{\\ell t} / F_\\ell ,

with the capacity-dependent reactance :math:`\\hat x_\\ell(F_\\ell) =
\\hat x^0_\\ell F^0_\\ell / F_\\ell` and the series compensation
:math:`\\tilde q` of an SSSC on a ``LineX``. Lifting the branch term
:math:`g_{\\ell t}` to a variable of its own makes the voltage law linear and
concentrates the whole nonconvexity in one bilinear equality per branch and
snapshot,

.. math::
    F_\\ell\\, g_{\\ell t} = a_\\ell f_{\\ell t} - \\tilde q_{\\ell t},
    \\qquad a_\\ell := \\hat x^0_\\ell F^0_\\ell = \\hat x_\\ell(F_\\ell) F_\\ell ,

whose coefficient :math:`a_\\ell` is invariant under the capacity iteration.
Relaxing it by its McCormick envelope over a box :math:`F_\\ell \\in [F^L,F^U]`,
:math:`g_{\\ell t} \\in [g^L,g^U]` gives a linear program whose optimum is a
valid lower bound on every capacity plan inside the box. The envelope of a
*single* bilinear term is its convex hull, so no tighter convex relaxation
exists over the given box, and the remaining slack is a matter of box width
alone: the per-term gap is at most :math:`\\tfrac14 (g^U-g^L)(F^U-F^L)`.
Tightening the box, not the envelope, is therefore what buys accuracy, which is
what :func:`tighten_capacity_box` does.

The lifted variable has a box that does not depend on the capacity at all. With
the loading :math:`\\lambda_{\\ell t} = f_{\\ell t}/F_\\ell \\in
[-\\bar u_\\ell, \\bar u_\\ell]`,

.. math::
    |g_{\\ell t}| = |a_\\ell \\lambda_{\\ell t} - \\tilde q_{\\ell t}/F_\\ell|
                  \\le a_\\ell \\bar u_\\ell + Q^{\\max}_\\ell / F^L_\\ell ,

the first term being the voltage drop of the branch at its rated flow.

Typical use::

    cert = certify_expansion(n, obbt_rounds=3, solver_name="gurobi")
    print(cert)

which re-costs the plan in ``n`` exactly, bounds it from below and reports the
gap. With ``spatial=True`` and Gurobi available the bilinear equalities are
added back and solved by spatial branch and bound, which closes the gap and
certifies the global optimum on problems small enough to afford it.

Caveats
-------
The bound covers the nonconvexity of the *voltage law*. If
``transmission_losses`` is positive, the loss tangents are held at the
resistances of the reference capacities, exactly as every inner problem of
``optimize_transmission_expansion_iteratively`` holds them; the bound is then
rigorous for that model rather than for one in which the resistance follows the
capacity as well. Keep ``transmission_losses=0`` for a bound with no such
qualification.

Multi-period expansion is not supported.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from linopy.expressions import LinearExpression
from xarray import DataArray, Dataset

from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.descriptors import nominal_attrs
from pypsa.optimization.constraints import capacity_reference, kirchhoff_voltage_cycles

if TYPE_CHECKING:
    from linopy import Model

    from pypsa import Network

logger = logging.getLogger(__name__)

#: The voltage law rows of :mod:`pypsa.optimization.constraints` are scaled by
#: this factor to put their coefficients on the scale of the other rows. The
#: lifted variable is stored with the same scaling, so that the McCormick rows
#: and the cycle rows are conditioned like the constraints they replace.
KVL_SCALE = 1e4

BRANCH_COMPONENTS = ("Line", "LineX")

NAMES = ["component", "name"]


# --------------------------------------------------------------------------- #
# bookkeeping
# --------------------------------------------------------------------------- #
@dataclass
class LiftedKVL:
    """
    Where the lifted voltage law ended up in the model.

    Carries the linopy variable labels of the terms of every bilinear equality,
    which is what :func:`spatial_bound` needs to add the equalities back on top
    of their relaxation.
    """

    index: pd.MultiIndex
    """``(component, name)`` of the branches whose term was lifted."""
    a: np.ndarray
    """The invariant ``x_pu_eff * F``, one per lifted branch."""
    g_max: np.ndarray
    """Bound of the *scaled* lifted variable, i.e. ``scale * |g|``."""
    lower: np.ndarray
    """Capacity lower bounds the envelope was built over."""
    upper: np.ndarray
    """Capacity upper bounds the envelope was built over."""
    label_g: np.ndarray
    """``(snapshot, branch)`` labels of the lifted variable."""
    label_f: np.ndarray
    """``(snapshot, branch)`` labels of the branch flow."""
    label_q: np.ndarray
    """``(snapshot, branch)`` labels of the SSSC compensation, ``-1`` if none."""
    label_capacity: np.ndarray
    """``(branch,)`` labels of the branch capacity."""
    scale: float
    """The scaling of the lifted variable, see :data:`KVL_SCALE`."""
    constants: pd.DataFrame
    """Per-branch constants of the voltage law, see :func:`branch_constants`."""

    @property
    def n_lifted(self) -> int:
        return len(self.index)


@dataclass
class Certificate:
    """Result of :func:`certify_expansion`."""

    upper_bound: float
    """Cost of the candidate plan, evaluated at the impedances it implies."""
    lower_bound: float
    """Best lower bound obtained."""
    mccormick_bound: float
    """Lower bound before any bound tightening."""
    capacities: pd.Series
    """The candidate plan that was certified."""
    lower: pd.Series
    """Capacity box the final bound was proved over."""
    upper: pd.Series
    obbt_history: list[tuple[int, float, float]] = field(default_factory=list)
    """``(round, bound, widest box)`` of every tightening round."""
    spatial: dict[str, Any] | None = None
    """Outcome of the spatial branch and bound, if it was run."""

    @property
    def gap(self) -> float:
        """Relative gap of the candidate plan to the lower bound."""
        if not np.isfinite(self.upper_bound) or self.upper_bound == 0:
            return np.nan
        return (self.upper_bound - self.lower_bound) / abs(self.upper_bound)

    def _relative(self, bound: float) -> float:
        return (self.upper_bound - bound) / abs(self.upper_bound) * 100

    def __str__(self) -> str:
        lines = [
            f"candidate plan          {self.upper_bound:16.4f}",
            f"McCormick bound         {self.mccormick_bound:16.4f}"
            f"   ({self._relative(self.mccormick_bound):8.4f} %)",
        ]
        for r, z, width in self.obbt_history:
            lines.append(
                f"  + tightening round {r}  {z:16.4f}"
                f"   ({self._relative(z):8.4f} %)   widest box {width:10.3f}"
            )
        if self.spatial is not None:
            lines.append(
                f"spatial branch & bound  {self.spatial['bound']:16.4f}"
                f"   ({self._relative(self.spatial['bound']):8.4f} %)"
                f"   {'closed' if self.spatial['closed'] else 'not closed'}"
            )
        lines.append(f"gap of the plan         {self.gap * 100:16.6f} %")
        return "\n".join(lines)


# --------------------------------------------------------------------------- #
# per-branch constants
# --------------------------------------------------------------------------- #
def _detached_copy(n: Network) -> Network:
    """
    Deep copy of a network without whatever optimisation model is attached.

    ``Network.copy`` deep-copies everything it finds on the network, and a
    linopy model does not survive that: its ``Variables`` recurse in
    ``__getattr__`` while being copied, and its ``solver_model`` is a handle
    into the solver. A network that has just been optimised therefore cannot be
    copied at all, which is precisely the network the bound is asked about. The
    model of the *input* is of no use here in any case, so it is put aside for
    the duration of the copy and the copy is handed back without one.
    """
    stashed = {
        key: n.__dict__.pop(key)
        for key in ("model", "_kvl_capacity_sensitivity")
        if key in n.__dict__
    }
    try:
        out = n.copy()
    finally:
        n.__dict__.update(stashed)
    out.__dict__.pop("model", None)
    out._kvl_capacity_sensitivity = None
    return out


def branch_series(n: Network, attr: str) -> pd.Series:
    """Concatenate a passive branch attribute over the branch components."""
    out = {
        c: n.df(c)[attr]
        for c in BRANCH_COMPONENTS
        if c in n.components and not n.df(c).empty
    }
    if not out:
        empty = pd.MultiIndex.from_arrays([[], []], names=NAMES)
        return pd.Series(dtype=float, index=empty)
    return pd.concat(out, names=NAMES)


def branch_constants(n: Network, snapshots: pd.Index | None = None) -> pd.DataFrame:
    """
    Constants of the voltage law, per passive branch of a meshed sub-network.

    Returns a frame indexed by ``(component, name)`` with

    ``w``
        the impedance entering the voltage law, ``x_pu_eff`` on an AC and
        ``r_pu_eff`` on a DC sub-network,
    ``f_ref``
        the capacity ``w`` belongs to, i.e. ``capacity_reference``,
    ``a``
        the invariant ``w * f_ref``,
    ``in_cycle``
        whether the branch appears in any independent cycle,
    ``scaling``
        whether its impedance follows the capacity, which is what makes the
        voltage law nonlinear,
    ``lift``
        ``in_cycle & scaling``, i.e. the branches whose term is lifted,
    ``u_bar``
        the largest ``s_max_pu`` over the snapshots,
    ``q_max``
        the largest series compensation an SSSC can provide.
    """
    if getattr(n, "_multi_invest", False):
        raise NotImplementedError(
            "the lower bound does not support multi-period expansion"
        )
    n.calculate_dependent_values()
    if snapshots is None:
        snapshots = n.snapshots

    rows = []
    for branches_i, C, weightings, carrier in kirchhoff_voltage_cycles(n):
        active = np.asarray(abs(C).sum(axis=1)).ravel() > 0
        for k, (comp, name) in enumerate(branches_i):
            rows.append(
                dict(
                    component=comp,
                    name=name,
                    w=float(weightings[k]),
                    carrier=carrier,
                    in_cycle=bool(active[k]),
                )
            )
    if not rows:
        empty = pd.MultiIndex.from_arrays([[], []], names=NAMES)
        columns = ["w", "carrier", "in_cycle", "f_ref", "a", "ext", "scaling",
                   "lift", "u_bar", "q_max"]
        return pd.DataFrame(columns=columns, index=empty)

    info = pd.DataFrame(rows).set_index(NAMES)

    f_ref, u_bar, q_max, ext, scaling = {}, {}, {}, {}, {}
    for comp, name in info.index:
        df = n.df(comp)
        key = (comp, name)
        f_ref[key] = float(capacity_reference(n, comp).at[name])
        u_bar[key] = float(
            get_as_dense(n, comp, "s_max_pu").loc[snapshots, name].abs().max()
        )
        is_ext = bool(df.at[name, f"{nominal_attrs[comp]}_extendable"])
        ext[key] = is_ext
        typed = str(df.at[name, "type"]) != ""
        # the branches abstract.py rescales with the capacity: untyped AC lines
        # through their impedance, typed ones through num_parallel
        is_ac = str(n.buses.carrier.get(df.at[name, "bus0"], "AC")) == "AC"
        scaling[key] = is_ext and (typed or is_ac)
        if comp == "LineX":
            q_extendable = bool(df.at[name, "sssc_nom_extendable"])
            q_max[key] = float(
                df.at[name, "sssc_nom_max"] if q_extendable else df.at[name, "sssc_nom"]
            )
        else:
            q_max[key] = 0.0

    info["f_ref"] = pd.Series(f_ref)
    info["u_bar"] = pd.Series(u_bar)
    info["q_max"] = pd.Series(q_max)
    info["ext"] = pd.Series(ext)
    info["scaling"] = pd.Series(scaling)
    info["a"] = info["w"] * info["f_ref"]
    info["lift"] = info["scaling"] & info["in_cycle"]
    return info


# --------------------------------------------------------------------------- #
# the capacity box
# --------------------------------------------------------------------------- #
def capacity_box(n: Network, cutoff: float) -> tuple[pd.Series, pd.Series]:
    """
    Finite capacity box implied by an objective cutoff.

    Every capacity plan that is at least as good as ``cutoff`` spends at most
    ``cutoff`` minus the capital cost of the lower bounds on any single branch,
    because the operating cost and every other capital cost term are bounded
    below by their value at the lower bounds. Hence

    .. math::
        F^U_\\ell = F^L_\\ell
                  + \\bigl(z^{UB} - \\sum_j c_j F^L_j\\bigr) / c_\\ell .

    A branch with no capital cost keeps whatever ``s_nom_max`` it was given,
    which then has to be finite.

    Parameters
    ----------
    n : pypsa.Network
    cutoff : float
        An attainable objective value, e.g. the cost of a known plan. Passing
        anything smaller can cut off the global optimum.

    Returns
    -------
    (lower, upper) : pd.Series
        Indexed by ``(component, name)`` over the passive branches.
    """
    ext = branch_series(n, "s_nom_extendable").astype(bool)
    s_nom = branch_series(n, "s_nom").astype(float)
    lower = branch_series(n, "s_nom_min").astype(float).where(ext, s_nom)
    cost = branch_series(n, "capital_cost").astype(float)

    slack = cutoff - float((cost * lower).sum())
    if slack < 0:
        raise ValueError(
            "the cutoff is below the capital cost of the capacity lower bounds, "
            "so no plan can attain it"
        )

    room = pd.Series(
        np.where(cost > 0, slack / cost.where(cost > 0, 1.0), np.inf),
        index=cost.index,
    )
    upper = s_nom.copy()
    upper[ext] = (lower + room)[ext]
    given = branch_series(n, "s_nom_max").astype(float).fillna(np.inf)
    upper = pd.Series(np.minimum(upper, given), index=lower.index)

    unbounded = ext & ~np.isfinite(upper)
    if unbounded.any():
        raise ValueError(
            "no finite capacity upper bound for "
            f"{list(upper.index[unbounded])}: give them a capital_cost or an "
            "s_nom_max"
        )
    return lower, upper.clip(lower=lower)


# --------------------------------------------------------------------------- #
# the relaxation
# --------------------------------------------------------------------------- #
def _labels_flow(m: Model, index: pd.MultiIndex, n_sns: int) -> np.ndarray:
    out = np.empty((n_sns, len(index)), dtype=int)
    for k, (comp, name) in enumerate(index):
        out[:, k] = m[f"{comp}-s"].labels.sel({comp: name}).values
    return out


def _labels_q(m: Model, index: pd.MultiIndex, n_sns: int) -> np.ndarray:
    out = np.full((n_sns, len(index)), -1, dtype=int)
    if "LineX-q_sssc" not in m.variables:
        return out
    for k, (comp, name) in enumerate(index):
        if comp == "LineX":
            out[:, k] = m["LineX-q_sssc"].labels.sel({"LineX": name}).values
    return out


def _labels_capacity(m: Model, index: pd.MultiIndex) -> np.ndarray:
    out = np.full(len(index), -1, dtype=int)
    for k, (comp, name) in enumerate(index):
        variable = f"{comp}-{nominal_attrs[comp]}"
        if variable in m.variables and name in m[variable].indexes[m[variable].dims[0]]:
            out[k] = int(m[variable].labels.sel({m[variable].dims[0]: name}).values)
    return out


def add_lifted_kvl(
    n: Network,
    m: Model,
    lower: pd.Series,
    upper: pd.Series,
    scale: float = KVL_SCALE,
) -> LiftedKVL:
    """
    Add the lifted voltage law and its McCormick envelope to a KVL-free model.

    The model has to have been built without the voltage law, which
    :func:`build_relaxation` takes care of. Adds

    * the lifted variable ``kvl-G`` = ``scale * g``, bounded by the a-priori
      bound of the module docstring,
    * ``kvl-lifted``, the linear voltage law in the lifted variable, in which
      the branches that do not scale with their capacity keep their ordinary
      term,
    * ``kvl-mccormick-{lo1,lo2,up1,up2}``, the four envelope rows of the
      bilinear equality.
    """
    snapshots = m.parameters.snapshots.to_index()
    n_sns = len(snapshots)
    constants = branch_constants(n, snapshots)

    lifted_i = constants.index[constants["lift"]]
    labels = pd.Index([f"{c}::{k}" for c, k in lifted_i], name="lift")
    n_lift = len(labels)

    f_low = lower.reindex(lifted_i).to_numpy(float)
    f_up = upper.reindex(lifted_i).to_numpy(float)
    if n_lift and not np.all(np.isfinite(f_up)):
        raise ValueError("the McCormick envelope needs finite capacity upper bounds")
    if n_lift and (f_low < 0).any():
        raise ValueError("negative capacity lower bound")

    a = constants.loc[lifted_i, "a"].to_numpy(float)
    u_bar = constants.loc[lifted_i, "u_bar"].to_numpy(float)
    q_max = constants.loc[lifted_i, "q_max"].to_numpy(float)
    if np.any((q_max > 0) & (f_low <= 0)):
        raise ValueError(
            "a compensated branch needs a positive capacity lower bound, "
            "otherwise its voltage law term is unbounded"
        )
    # |g| <= a * u_bar + q_max / F, the drop at rated flow plus the compensation
    q_term = np.where(f_low > 0, q_max / np.where(f_low > 0, f_low, 1.0), 0.0)
    g_max = scale * (a * u_bar + q_term)

    if n_lift:
        bound = DataArray(
            np.tile(g_max, (n_sns, 1)), coords={"snapshot": snapshots, "lift": labels}
        )
        lifted_variable = m.add_variables(lower=-bound, upper=bound, name="kvl-G")
        label_g = lifted_variable.labels.transpose("snapshot", "lift").values
    else:
        label_g = np.zeros((n_sns, 0), dtype=int)

    label_f = _labels_flow(m, lifted_i, n_sns)
    label_q = _labels_q(m, lifted_i, n_sns)
    label_capacity = _labels_capacity(m, lifted_i)
    if n_lift and (label_capacity < 0).any():
        raise ValueError(
            "a lifted branch has no capacity variable: "
            f"{list(lifted_i[label_capacity < 0])}"
        )

    if n_lift:
        _add_envelope(m, snapshots, labels, a, g_max, f_low, f_up,
                      label_g, label_f, label_q, label_capacity, scale)
    _add_cycle_rows(n, m, constants, labels, label_g, scale)

    return LiftedKVL(
        index=lifted_i,
        a=a,
        g_max=g_max,
        lower=f_low,
        upper=f_up,
        label_g=label_g,
        label_f=label_f,
        label_q=label_q,
        label_capacity=label_capacity,
        scale=scale,
        constants=constants,
    )


def _add_envelope(
    m, snapshots, labels, a, g_max, f_low, f_up,
    label_g, label_f, label_q, label_capacity, scale,
) -> None:
    """
    The four McCormick rows of ``F * g = a f - q`` over ``[f_low, f_up]`` and
    ``[-g_max, g_max]``, written in the scaled lifted variable ``G``.

    Each row reads ``scale * (a f - q) - c_g F - c_F G  {sign}  -c_g c_F``.
    """
    n_sns, n_lift = len(snapshots), len(labels)
    coords = {"snapshot": snapshots, "lift": labels}
    a_scaled = np.broadcast_to(scale * a, (n_sns, n_lift))
    has_q = (label_q >= 0).astype(float)
    stacked_capacity = np.broadcast_to(label_capacity, (n_sns, n_lift))

    def row(c_g, c_F, sign, tag):
        coeffs = np.stack(
            [
                a_scaled,
                -scale * has_q,
                np.broadcast_to(-c_g, (n_sns, n_lift)),
                np.broadcast_to(-c_F, (n_sns, n_lift)),
            ],
            axis=-1,
        )
        variables = np.stack([label_f, label_q, stacked_capacity, label_g], axis=-1)
        data = Dataset(
            {
                "coeffs": DataArray(
                    coeffs, dims=("snapshot", "lift", "_term"), coords=coords
                ),
                "vars": DataArray(
                    variables, dims=("snapshot", "lift", "_term"), coords=coords
                ),
            }
        )
        m.add_constraints(
            LinearExpression(data, m),
            sign,
            DataArray(-c_g * c_F, coords={"lift": labels}),
            name=f"kvl-mccormick-{tag}",
        )

    g_low, g_up = -g_max, g_max
    row(g_low, f_low, ">=", "lo1")
    row(g_up, f_up, ">=", "lo2")
    row(g_up, f_low, "<=", "up1")
    row(g_low, f_up, "<=", "up2")


def _add_cycle_rows(n, m, constants, labels, label_g, scale) -> None:
    """
    ``sum_l C_lc * term_l = 0`` with the lifted variable on the branches whose
    impedance follows the capacity and the ordinary term on the rest.
    """
    snapshots = m.parameters.snapshots.to_index()
    n_sns = len(snapshots)
    position = {label: j for j, label in enumerate(labels)}

    rows = []
    for branches_i, C, weightings, carrier in kirchhoff_voltage_cycles(n):
        f_ref = constants["f_ref"].reindex(branches_i).to_numpy(float)
        for j in range(C.shape[1]):
            sl = slice(C.indptr[j], C.indptr[j + 1])
            coeffs, variables = [], []
            for k, orientation in zip(C.indices[sl], C.data[sl]):
                comp, name = branches_i[k]
                label = f"{comp}::{name}"
                if label in position:
                    coeffs.append(np.full(n_sns, float(orientation)))
                    variables.append(label_g[:, position[label]])
                    continue
                coeffs.append(np.full(n_sns, scale * orientation * weightings[k]))
                variables.append(m[f"{comp}-s"].labels.sel({comp: name}).values)
                if comp == "LineX" and "LineX-q_sssc" in m.variables:
                    coeffs.append(np.full(n_sns, -scale * orientation / f_ref[k]))
                    variables.append(
                        m["LineX-q_sssc"].labels.sel({"LineX": name}).values
                    )
            rows.append((np.stack(coeffs, axis=-1), np.stack(variables, axis=-1)))

    if not rows:
        return

    width = max(c.shape[-1] for c, _ in rows)
    coeffs = np.zeros((n_sns, len(rows), width))
    variables = np.full((n_sns, len(rows), width), -1, dtype=int)
    for j, (c, v) in enumerate(rows):
        coeffs[:, j, : c.shape[-1]] = c
        variables[:, j, : v.shape[-1]] = v
    coords = {"snapshot": snapshots, "cycle": np.arange(len(rows))}
    data = Dataset(
        {
            "coeffs": DataArray(
                coeffs, dims=("snapshot", "cycle", "_term"), coords=coords
            ),
            "vars": DataArray(
                variables, dims=("snapshot", "cycle", "_term"), coords=coords
            ),
        }
    )
    m.add_constraints(LinearExpression(data, m), "=", 0, name="kvl-lifted")


def build_relaxation(
    n: Network,
    lower: pd.Series,
    upper: pd.Series,
    snapshots: pd.Index | None = None,
    scale: float = KVL_SCALE,
    **kwargs: Any,
) -> tuple[Network, Model, LiftedKVL]:
    """
    Build the McCormick relaxation over a capacity box.

    The network is copied, the box is written into ``s_nom_min``/``s_nom_max``,
    the ordinary optimisation model is created and its voltage law rows are
    replaced by :func:`add_lifted_kvl`. The branch impedances are left where
    they are; they enter the relaxation only through the invariant ``a``, which
    does not depend on the capacity.

    Parameters
    ----------
    n : pypsa.Network
    lower, upper : pd.Series
        The capacity box, indexed by ``(component, name)``, e.g. from
        :func:`capacity_box`.
    snapshots : pd.Index, optional
    scale : float
        Scaling of the lifted variable, see :data:`KVL_SCALE`.
    **kwargs
        Passed to ``create_model``, in particular ``transmission_losses``.

    Returns
    -------
    (network, model, lifted) : (pypsa.Network, linopy.Model, LiftedKVL)
    """
    from pypsa.optimization.optimize import create_model

    # a linearisation the caller may have left behind would enter the voltage
    # law rows that are about to be discarded and drag its own variables in
    n = _detached_copy(n)
    for comp in BRANCH_COMPONENTS:
        if comp not in n.components or n.df(comp).empty:
            continue
        df = n.df(comp)
        ext = df.s_nom_extendable.to_numpy()
        df.loc[ext, "s_nom_min"] = lower.xs(comp, level="component").reindex(df.index)[ext]
        df.loc[ext, "s_nom_max"] = upper.xs(comp, level="component").reindex(df.index)[ext]

    m = create_model(n, snapshots=snapshots, **kwargs)

    discarded = [
        name for name in m.constraints if name.startswith("Kirchhoff-Voltage-Law")
    ]
    for name in discarded:
        m.remove_constraints(name)
    if not discarded and kirchhoff_voltage_cycles(n):
        raise RuntimeError(
            "no Kirchhoff voltage law rows were found to replace; the naming of "
            "define_kirchhoff_voltage_constraints must have changed"
        )

    lifted = add_lifted_kvl(n, m, lower, upper, scale=scale)
    return n, m, lifted


def relaxation_bound(
    n: Network,
    lower: pd.Series,
    upper: pd.Series,
    cutoff: float | None = None,
    snapshots: pd.Index | None = None,
    solver_name: str = "gurobi",
    build_kwargs: dict | None = None,
    **solver_kwargs: Any,
) -> tuple[float, Model, LiftedKVL]:
    """
    Solve the McCormick relaxation and return its optimum.

    Parameters
    ----------
    cutoff : float, optional
        Adds ``objective <= cutoff``. This does not change the bound, but it is
        what makes the bound tightening of :func:`tighten_capacity_box` work,
        which optimises over the same feasible set.
    **solver_kwargs
        Passed to ``linopy.Model.solve``, e.g. ``OutputFlag=0``.

    Returns
    -------
    (bound, model, lifted)
    """
    _, m, lifted = build_relaxation(
        n, lower, upper, snapshots=snapshots, **(build_kwargs or {})
    )
    if cutoff is not None:
        m.add_constraints(m.objective.expression <= cutoff, name="objective-cutoff")
    status, condition = m.solve(solver_name=solver_name, **solver_kwargs)
    if condition != "optimal":
        raise RuntimeError(f"the relaxation did not solve: {status}, {condition}")
    return float(m.objective.value), m, lifted


def tighten_capacity_box(
    n: Network,
    lower: pd.Series,
    upper: pd.Series,
    cutoff: float,
    rounds: int = 3,
    tol: float = 1e-4,
    snapshots: pd.Index | None = None,
    solver_name: str = "gurobi",
    build_kwargs: dict | None = None,
    **solver_kwargs: Any,
) -> tuple[pd.Series, pd.Series, list[tuple[int, float, float]]]:
    """
    Optimality-based bound tightening of the capacity box.

    Minimises and maximises every extendable capacity over the relaxation
    *under the objective cutoff*. Any plan attaining ``cutoff`` therefore lies
    in the tightened box, which is what makes it valid to keep bounding over
    it. Because the McCormick envelope is already the convex hull of each
    bilinear term separately, this is where the accuracy comes from: the
    per-term gap is proportional to the box width.

    Parameters
    ----------
    cutoff : float
        An attainable objective value; see :func:`capacity_box`.
    rounds : int
        Maximum number of passes. Each pass solves two linear programs per
        extendable branch, so the effort grows quickly with the network.
    tol : float
        Stop once a pass shrinks the widest interval by less than this,
        relative to the widest interval the pass started from.

    Returns
    -------
    (lower, upper, history)
        ``history`` holds ``(round, bound, widest box)`` per pass.
    """
    ext = branch_series(n, "s_nom_extendable").astype(bool)
    history: list[tuple[int, float, float]] = []
    build_kwargs = build_kwargs or {}

    for r in range(rounds):
        _, m, _ = build_relaxation(n, lower, upper, snapshots=snapshots, **build_kwargs)
        m.add_constraints(m.objective.expression <= cutoff, name="objective-cutoff")

        new_lower, new_upper = lower.copy(), upper.copy()
        for comp, name in lower.index[ext]:
            variable = f"{comp}-{nominal_attrs[comp]}"
            if variable not in m.variables:
                continue
            capacity = m[variable].sel({m[variable].dims[0]: name})
            for sense, target in (("min", new_lower), ("max", new_upper)):
                # linopy 0.3 wants a LinearExpression, not a bare variable
                m.objective = (1.0 * capacity) if sense == "min" else (-1.0 * capacity)
                _, condition = m.solve(solver_name=solver_name, **solver_kwargs)
                if condition != "optimal":
                    # an infeasible sub-problem would mean the cutoff is not
                    # attainable; leave the bound alone rather than guess
                    logger.warning(
                        "bound tightening of %s %s (%s) ended %s",
                        comp, name, sense, condition,
                    )
                    continue
                value = float(m.objective.value)
                value = value if sense == "min" else -value
                if sense == "min":
                    target[(comp, name)] = max(lower[(comp, name)], value - 1e-6)
                else:
                    target[(comp, name)] = min(upper[(comp, name)], value + 1e-6)

        widest_before = float((upper - lower).max())
        shrink = float(((upper - lower) - (new_upper - new_lower)).abs().max())
        lower, upper = new_lower, new_upper

        bound, _, _ = relaxation_bound(
            n, lower, upper, cutoff=cutoff, snapshots=snapshots,
            solver_name=solver_name, build_kwargs=build_kwargs, **solver_kwargs,
        )
        history.append((r + 1, bound, float((upper - lower)[ext].max())))
        if shrink < tol * max(widest_before, 1e-9):
            break

    return lower, upper, history


# --------------------------------------------------------------------------- #
# the upper bound
# --------------------------------------------------------------------------- #
def set_capacity_dependent_parameters(n: Network, capacities: pd.Series) -> None:
    """
    Put the branch parameters on the capacities they belong to, in place.

    The untyped AC branches have their impedance scaled by the inverse capacity
    ratio, the typed ones their ``num_parallel``; ``_s_nom_def`` records the
    capacities the parameters now belong to. This is the same update the outer
    iteration performs between two inner problems, extracted so that a plan can
    be evaluated without running one.
    """
    n.calculate_dependent_values()
    for comp in BRANCH_COMPONENTS:
        if comp not in n.components or n.df(comp).empty:
            continue
        df = n.df(comp)
        target = (
            capacities.xs(comp, level="component").reindex(df.index).fillna(df.s_nom)
        ).astype(float)
        # the impedance scales with the inverse capacity, so the capacity the
        # parameters are defined at has to be kept away from zero
        target = target.clip(lower=df.s_nom.abs() * 1e-6)
        df["_s_nom_def"] = target
        factor = target / df.s_nom.replace(0, np.nan)

        ext_i = df.index[df.s_nom_extendable]
        typed_i = df.index[df.type.astype(str) != ""]
        carrier = df.bus0.map(n.buses.carrier)
        untyped_ac_i = ext_i.difference(typed_i).intersection(df.index[carrier == "AC"])
        if not untyped_ac_i.empty:
            df.loc[untyped_ac_i, "x"] = df.loc[untyped_ac_i, "x"] / factor[untyped_ac_i]
            df.loc[untyped_ac_i, "r"] = df.loc[untyped_ac_i, "r"] / factor[untyped_ac_i]

        ext_typed_i = ext_i.intersection(typed_i)
        if not ext_typed_i.empty:
            base = (
                np.sqrt(3)
                * df.type.map(n.line_types.i_nom)
                * df.bus0.map(n.buses.v_nom)
            )
            df.loc[ext_typed_i, "num_parallel"] = (target / base)[ext_typed_i]
    n.calculate_dependent_values()


def evaluate_plan(
    n: Network,
    capacities: pd.Series,
    snapshots: pd.Index | None = None,
    solver_name: str = "gurobi",
    build_kwargs: dict | None = None,
    **solver_kwargs: Any,
) -> tuple[float, Network]:
    """
    Cost of a capacity plan, at the impedances the plan itself implies.

    Pins the capacities, puts the branch parameters on them and re-solves the
    dispatch. The result is the objective value of a point that is exactly
    feasible for the nonlinear problem, hence a rigorous upper bound on its
    optimum -- unlike the objective of the last inner problem of the outer
    iteration, which is evaluated at the impedances of the *previous* iterate
    and can fall below the true optimum when the iteration has not converged.

    Returns
    -------
    (cost, network)
        The pinned network is returned as well, so the dispatch behind the cost
        can be inspected.
    """
    n = _detached_copy(n)
    set_capacity_dependent_parameters(n, capacities)
    for comp in BRANCH_COMPONENTS:
        if comp not in n.components or n.df(comp).empty:
            continue
        df = n.df(comp)
        ext = df.s_nom_extendable.to_numpy()
        target = df["_s_nom_def"]
        df.loc[ext, "s_nom_min"] = target[ext]
        df.loc[ext, "s_nom_max"] = target[ext]

    status, condition = n.optimize(
        snapshots=snapshots,
        solver_name=solver_name,
        solver_options=solver_kwargs,
        **(build_kwargs or {}),
    )
    if condition != "optimal":
        raise RuntimeError(f"the pinned plan did not solve: {status}, {condition}")
    return float(n.model.objective.value), n


# --------------------------------------------------------------------------- #
# the global bound
# --------------------------------------------------------------------------- #
def spatial_bound(
    n: Network,
    lower: pd.Series,
    upper: pd.Series,
    cutoff: float,
    snapshots: pd.Index | None = None,
    time_limit: float = 600.0,
    mip_gap: float = 1e-6,
    build_kwargs: dict | None = None,
    output_flag: int = 0,
) -> dict[str, Any]:
    """
    Close the relaxation gap by spatial branch and bound.

    Writes the relaxation to an LP file, reads it back with ``gurobipy`` and
    adds the bilinear equalities ``F * G = scale * (a f - q)`` on top of their
    envelope, which turns the relaxation into the exact model. Gurobi's
    ``NonConvex=2`` then branches on the capacities, so the bound it reports is
    valid at any point and equals the global optimum once it terminates.

    Requires ``gurobipy``; linopy 0.3 cannot express quadratic constraints,
    which is why the model takes the detour through the file.

    Parameters
    ----------
    cutoff : float
        An attainable objective value; the branch and bound is told not to look
        above it. Pass a loose multiple of a known cost to get a certificate
        that does not depend on the candidate plan at all.

    Returns
    -------
    dict
        ``bound``, ``incumbent``, ``capacities``, ``closed``, ``nodes``,
        ``runtime`` and ``status``.
    """
    try:
        import gurobipy as gp
    except ImportError as error:  # pragma: no cover - optional dependency
        raise ImportError(
            "the spatial branch and bound needs gurobipy, which is not installed"
        ) from error

    import tempfile
    from pathlib import Path

    _, m, lifted = build_relaxation(
        n, lower, upper, snapshots=snapshots, **(build_kwargs or {})
    )
    m.add_constraints(m.objective.expression <= cutoff + 1e-6, name="objective-cutoff")

    with tempfile.TemporaryDirectory() as folder:
        path = Path(folder) / "bilinear.lp"
        m.to_file(path)
        model = gp.read(str(path))

    model.Params.OutputFlag = output_flag
    model.Params.NonConvex = 2
    model.Params.TimeLimit = time_limit
    model.Params.MIPGap = mip_gap
    model.update()

    def variable(label):
        # linopy names the columns it writes after the label it assigned them
        return model.getVarByName(f"x{int(label)}")

    n_sns, n_lift = lifted.label_g.shape
    for j in range(n_lift):
        capacity = variable(lifted.label_capacity[j])
        coefficient = lifted.scale * lifted.a[j]
        for t in range(n_sns):
            expression = (
                capacity * variable(lifted.label_g[t, j])
                - coefficient * variable(lifted.label_f[t, j])
            )
            if lifted.label_q[t, j] >= 0:
                expression = expression + lifted.scale * variable(lifted.label_q[t, j])
            model.addQConstr(expression == 0.0, name=f"kvl-bilinear[{j},{t}]")
    model.optimize()

    capacities = pd.Series(dtype=float)
    if model.SolCount:
        capacities = pd.Series(
            {
                key: variable(label).X
                for key, label in zip(lifted.index, lifted.label_capacity)
            }
        )
        capacities.index = pd.MultiIndex.from_tuples(capacities.index, names=NAMES)

    return dict(
        bound=float(model.ObjBound),
        incumbent=float(model.ObjVal) if model.SolCount else np.nan,
        capacities=capacities,
        closed=model.Status == gp.GRB.OPTIMAL,
        nodes=float(model.NodeCount),
        runtime=float(model.Runtime),
        status=int(model.Status),
    )


# --------------------------------------------------------------------------- #
# driver
# --------------------------------------------------------------------------- #
def certify_expansion(
    n: Network,
    capacities: pd.Series | None = None,
    snapshots: pd.Index | None = None,
    obbt_rounds: int = 3,
    spatial: bool = False,
    transmission_losses: int = 0,
    solver_name: str = "gurobi",
    spatial_kwargs: dict | None = None,
    **solver_kwargs: Any,
) -> Certificate:
    """
    How far a capacity expansion plan is from the global optimum.

    Re-costs the plan at the impedances it implies (:func:`evaluate_plan`),
    bounds the problem from below over the box that cost implies
    (:func:`capacity_box`, :func:`relaxation_bound`), tightens the box
    (:func:`tighten_capacity_box`) and optionally closes the remaining gap by
    spatial branch and bound (:func:`spatial_bound`).

    Parameters
    ----------
    n : pypsa.Network
        The network the plan was computed for. Its ``s_nom``, ``x`` and ``r``
        are read as the reference the capacity dependence is written around, so
        pass the network *as it went into* the expansion, or one whose
        ``_s_nom_def`` is consistent with its impedances.
    capacities : pd.Series, optional
        The plan, indexed by ``(component, name)``. Defaults to ``s_nom_opt``.
    obbt_rounds : int
        Rounds of optimality-based bound tightening; ``0`` skips it.
    spatial : bool
        Run the spatial branch and bound as well. Needs ``gurobipy`` and is
        only affordable on small problems.
    transmission_losses : int
        Passed on to every model built. See the caveat in the module docstring:
        with losses the bound is rigorous for the loss linearisation taken at
        the reference resistances.
    **solver_kwargs
        Solver options, e.g. ``OutputFlag=0`` for Gurobi.

    Examples
    --------
    >>> n.optimize.optimize_transmission_expansion_iteratively(  # doctest: +SKIP
    ...     trust_region=True, solver_name="gurobi"
    ... )
    >>> print(certify_expansion(n0, branch_series(n, "s_nom_opt")))  # doctest: +SKIP
    """
    if capacities is None:
        capacities = branch_series(n, "s_nom_opt").astype(float)
    if transmission_losses:
        logger.warning(
            "with transmission_losses=%s the loss tangents of the relaxation "
            "are held at the reference resistances, so the bound is rigorous "
            "for that model rather than for one whose resistance follows the "
            "capacity",
            transmission_losses,
        )

    build_kwargs = dict(transmission_losses=transmission_losses)
    common = dict(
        snapshots=snapshots,
        solver_name=solver_name,
        build_kwargs=build_kwargs,
        **solver_kwargs,
    )

    upper_bound, _ = evaluate_plan(n, capacities, **common)
    lower, upper = capacity_box(n, upper_bound)

    mccormick, _, _ = relaxation_bound(n, lower, upper, cutoff=upper_bound, **common)
    bound = mccormick

    history: list[tuple[int, float, float]] = []
    if obbt_rounds:
        lower, upper, history = tighten_capacity_box(
            n, lower, upper, upper_bound, rounds=obbt_rounds, **common
        )
        if history:
            bound = history[-1][1]

    result = None
    if spatial:
        result = spatial_bound(
            n, lower, upper, upper_bound, snapshots=snapshots,
            build_kwargs=build_kwargs, **(spatial_kwargs or {}),
        )
        bound = max(bound, result["bound"])

    return Certificate(
        upper_bound=upper_bound,
        lower_bound=bound,
        mccormick_bound=mccormick,
        capacities=capacities,
        lower=lower,
        upper=upper,
        obbt_history=history,
        spatial=result,
    )
