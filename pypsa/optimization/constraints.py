#!/usr/bin/env python3
"""
Define optimisation constraints from PyPSA networks with Linopy.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import TYPE_CHECKING

import linopy
import numpy as np
import pandas as pd
from linopy import LinearExpression, merge
from numpy import inf
from scipy import sparse
from xarray import DataArray, Dataset, concat

from pypsa.descriptors import (
    additional_linkports,
    expand_series,
    get_activity_mask,
    get_bounds_pu,
    nominal_attrs,
)
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.optimization.common import reindex

if TYPE_CHECKING:
    from xarray import DataArray

    from pypsa import Network

logger = logging.getLogger(__name__)


def define_operational_constraints_for_non_extendables(
    n: Network, sns: pd.Index, c: str, attr: str, transmission_losses: int
) -> None:
    """
    Sets power dispatch constraints for non-extendable and non-commitable
    assets for a given component and a given attribute.

    Parameters
    ----------
    n : pypsa.Network
    sns : pd.Index
        Snapshots of the constraint.
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'
    """
    dispatch_lower: DataArray | tuple
    dispatch_upper: DataArray | tuple

    fix_i = n.get_non_extendable_i(c)
    fix_i = fix_i.difference(n.get_committable_i(c)).rename(fix_i.name)

    if fix_i.empty:
        return

    nominal_fix = n.df(c)[nominal_attrs[c]].reindex(fix_i)
    min_pu, max_pu = get_bounds_pu(n, c, sns, fix_i, attr)
    lower = min_pu.mul(nominal_fix)
    upper = max_pu.mul(nominal_fix)

    active = get_activity_mask(n, c, sns, fix_i) if n._multi_invest else None

    dispatch_lower = reindex(n.model[f"{c}-{attr}"], c, fix_i)
    dispatch_upper = reindex(n.model[f"{c}-{attr}"], c, fix_i)
    if c in n.passive_branch_components and transmission_losses:
        loss = reindex(n.model[f"{c}-loss"], c, fix_i)
        dispatch_lower = (1, dispatch_lower), (-1, loss)
        dispatch_upper = (1, dispatch_upper), (1, loss)
    n.model.add_constraints(
        dispatch_lower, ">=", lower, name=f"{c}-fix-{attr}-lower", mask=active
    )
    n.model.add_constraints(
        dispatch_upper, "<=", upper, name=f"{c}-fix-{attr}-upper", mask=active
    )


def define_operational_constraints_for_extendables(
    n: Network, sns: pd.Index, c: str, attr: str, transmission_losses: int
) -> None:
    """
    Sets power dispatch constraints for extendable devices for a given
    component and a given attribute.

    Parameters
    ----------
    n : pypsa.Network
    sns : pd.Index
        Snapshots of the constraint.
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'
    """
    lhs_lower: DataArray | tuple
    lhs_upper: DataArray | tuple

    ext_i = n.get_extendable_i(c)

    if ext_i.empty:
        return

    min_pu, max_pu = map(DataArray, get_bounds_pu(n, c, sns, ext_i, attr))
    dispatch = reindex(n.model[f"{c}-{attr}"], c, ext_i)
    capacity = n.model[f"{c}-{nominal_attrs[c]}"]

    active = get_activity_mask(n, c, sns, ext_i) if n._multi_invest else None

    lhs_lower = (1, dispatch), (-min_pu, capacity)
    lhs_upper = (1, dispatch), (-max_pu, capacity)
    if c in n.passive_branch_components and transmission_losses:
        loss = reindex(n.model[f"{c}-loss"], c, ext_i)
        lhs_lower += ((-1, loss),)
        lhs_upper += ((1, loss),)

    n.model.add_constraints(
        lhs_lower, ">=", 0, name=f"{c}-ext-{attr}-lower", mask=active
    )
    n.model.add_constraints(
        lhs_upper, "<=", 0, name=f"{c}-ext-{attr}-upper", mask=active
    )


def define_operational_constraints_for_committables(
    n: Network, sns: pd.Index, c: str
) -> None:
    """
    Sets power dispatch constraints for committable devices for a given
    component and a given attribute. The linearized approximation of the unit
    commitment problem is inspired by Hua et al. (2017) DOI:
    10.1109/TPWRS.2017.2735026.

    Parameters
    ----------
    n : pypsa.Network
    sns : pd.Index
        Snapshots of the constraint.
    c : str
        name of the network component
    """
    com_i = n.get_committable_i(c)

    if com_i.empty:
        return

    # variables
    status = n.model[f"{c}-status"]
    start_up = n.model[f"{c}-start_up"]
    shut_down = n.model[f"{c}-shut_down"]
    status_diff = status - status.shift(snapshot=1)
    p = reindex(n.model[f"{c}-p"], c, com_i)
    active = get_activity_mask(n, c, sns, com_i) if n._multi_invest else None

    # parameters
    nominal = DataArray(n.df(c)[nominal_attrs[c]].reindex(com_i))
    min_pu, max_pu = map(DataArray, get_bounds_pu(n, c, sns, com_i, "p"))
    lower_p = min_pu * nominal
    upper_p = max_pu * nominal
    min_up_time_set = n.df(c).min_up_time[com_i]
    min_down_time_set = n.df(c).min_down_time[com_i]
    ramp_up_limit = nominal * n.df(c).ramp_limit_up[com_i].fillna(1)
    ramp_down_limit = nominal * n.df(c).ramp_limit_down[com_i].fillna(1)
    ramp_start_up = nominal * n.df(c).ramp_limit_start_up[com_i]
    ramp_shut_down = nominal * n.df(c).ramp_limit_shut_down[com_i]
    up_time_before_set = n.df(c)["up_time_before"].reindex(com_i)
    down_time_before_set = n.df(c)["down_time_before"].reindex(com_i)
    initially_up = up_time_before_set.astype(bool)
    initially_down = down_time_before_set.astype(bool)

    # check if there are status calculated/fixed before given sns interval
    if sns[0] != n.snapshots[0]:
        start_i = n.snapshots.get_loc(sns[0])
        # get generators which are online until the first regarded snapshot
        until_start_up = n.pnl(c).status.iloc[:start_i][::-1].reindex(columns=com_i)
        ref = range(1, len(until_start_up) + 1)
        up_time_before = until_start_up[until_start_up.cumsum().eq(ref, axis=0)].sum()
        up_time_before_set = up_time_before.clip(upper=min_up_time_set)
        initially_up = up_time_before_set.astype(bool)
        # get number of snapshots for generators which are offline before the first regarded snapshot
        until_start_down = ~until_start_up.astype(bool)
        ref = range(1, len(until_start_down) + 1)
        down_time_before = until_start_down[
            until_start_down.cumsum().eq(ref, axis=0)
        ].sum()
        down_time_before_set = down_time_before.clip(upper=min_down_time_set)
        initially_down = down_time_before_set.astype(bool)

    # lower dispatch level limit
    lhs_tuple = (1, p), (-lower_p, status)
    n.model.add_constraints(lhs_tuple, ">=", 0, name=f"{c}-com-p-lower", mask=active)

    # upper dispatch level limit
    lhs_tuple = (1, p), (-upper_p, status)
    n.model.add_constraints(lhs_tuple, "<=", 0, name=f"{c}-com-p-upper", mask=active)

    # state-transition constraint
    rhs = pd.DataFrame(0, sns, com_i)
    rhs.loc[sns[0], initially_up] = -1
    lhs = start_up - status_diff
    n.model.add_constraints(
        lhs, ">=", rhs, name=f"{c}-com-transition-start-up", mask=active
    )

    rhs = pd.DataFrame(0, sns, com_i)
    rhs.loc[sns[0], initially_up] = 1
    lhs = shut_down + status_diff
    n.model.add_constraints(
        lhs, ">=", rhs, name=f"{c}-com-transition-shut-down", mask=active
    )

    # min up time
    mask = get_activity_mask(n, c, sns[1:], com_i)
    expr = []
    min_up_time_i = com_i[min_up_time_set.astype(bool)]
    if not min_up_time_i.empty:
        for g in min_up_time_i:
            su = start_up.loc[:, g]
            expr.append(su.rolling(snapshot=min_up_time_set[g]).sum())
        lhs = -status.loc[:, min_up_time_i] + merge(expr, dim=com_i.name)
        lhs = lhs.sel(snapshot=sns[1:])
        n.model.add_constraints(
            lhs, "<=", 0, name=f"{c}-com-up-time", mask=mask[min_up_time_i]
        )

    # min down time
    expr = []
    min_down_time_i = com_i[min_down_time_set.astype(bool)]
    if not min_down_time_i.empty:
        for g in min_down_time_i:
            su = shut_down.loc[:, g]
            expr.append(su.rolling(snapshot=min_down_time_set[g]).sum())
        lhs = status.loc[:, min_down_time_i] + merge(expr, dim=com_i.name)
        lhs = lhs.sel(snapshot=sns[1:])
        n.model.add_constraints(
            lhs, "<=", 1, name=f"{c}-com-down-time", mask=mask[min_down_time_i]
        )

    # up time before
    timesteps = pd.DataFrame([range(1, len(sns) + 1)] * len(com_i), com_i, sns).T
    if initially_up.any():
        must_stay_up = (min_up_time_set - up_time_before_set).clip(lower=0)
        mask = (must_stay_up >= timesteps) & initially_up
        name = f"{c}-com-status-min_up_time_must_stay_up"
        mask = mask & active if active is not None else mask
        n.model.add_constraints(status, "=", 1, name=name, mask=mask)

    # down time before
    if initially_down.any():
        must_stay_down = (min_down_time_set - down_time_before_set).clip(lower=0)
        mask = (must_stay_down >= timesteps) & initially_down
        name = f"{c}-com-status-min_down_time_must_stay_up"
        mask = mask & active if active is not None else mask
        n.model.add_constraints(status, "=", 0, name=name, mask=mask)

    # linearized approximation because committable can partly start up and shut down
    cost_equal = all(
        n.df(c).loc[com_i, "start_up_cost"] == n.df(c).loc[com_i, "shut_down_cost"]
    )
    # only valid additional constraints if start up costs equal to shut down costs
    if n._linearized_uc and not cost_equal:
        logger.warning(
            "The linear relaxation of the unit commitment cannot be "
            "tightened since the start up costs are not equal to the "
            "shut down costs. Proceed with the linear relaxation "
            "without the tightening by additional constraints. "
            "This might result in a longer solving time."
        )
    if n._linearized_uc and cost_equal:
        # dispatch limit for partly start up/shut down for t-1
        lhs = (
            p.shift(snapshot=1)
            - ramp_shut_down * status.shift(snapshot=1)
            - (upper_p - ramp_shut_down) * (status - start_up)
        )
        lhs = lhs.sel(snapshot=sns[1:])
        n.model.add_constraints(lhs, "<=", 0, name=f"{c}-com-p-before", mask=active)

        # dispatch limit for partly start up/shut down for t
        lhs = p - upper_p * status + (upper_p - ramp_start_up) * start_up
        lhs = lhs.sel(snapshot=sns[1:])
        n.model.add_constraints(lhs, "<=", 0, name=f"{c}-com-p-current", mask=active)

        # ramp up if committable is only partly active and some capacity is starting up
        lhs = (
            p
            - p.shift(snapshot=1)
            - (lower_p + ramp_up_limit) * status
            + lower_p * status.shift(snapshot=1)
            + (lower_p + ramp_up_limit - ramp_start_up) * start_up
        )
        lhs = lhs.sel(snapshot=sns[1:])
        n.model.add_constraints(
            lhs, "<=", 0, name=f"{c}-com-partly-start-up", mask=active
        )

        # ramp down if committable is only partly active and some capacity is shutting up
        lhs = (
            p.shift(snapshot=1)
            - p
            - ramp_shut_down * status.shift(snapshot=1)
            + (ramp_shut_down - ramp_down_limit) * status
            - (lower_p + ramp_down_limit - ramp_shut_down) * start_up
        )
        lhs = lhs.sel(snapshot=sns[1:])
        n.model.add_constraints(
            lhs, "<=", 0, name=f"{c}-com-partly-shut-down", mask=active
        )


def define_nominal_constraints_for_extendables(n: Network, c: str, attr: str) -> None:
    """
    Sets capacity expansion constraints for extendable assets for a given
    component and a given attribute.

    Note: As GLPK does not like inf values on the right-hand-side we as masking these out.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'
    """
    ext_i = n.get_extendable_i(c)

    if ext_i.empty:
        return

    capacity = n.model[f"{c}-{attr}"]
    lower = n.df(c)[attr + "_min"].reindex(ext_i)
    upper = n.df(c)[attr + "_max"].reindex(ext_i)
    mask = upper != inf
    n.model.add_constraints(capacity, ">=", lower, name=f"{c}-ext-{attr}-lower")
    n.model.add_constraints(
        capacity, "<=", upper, name=f"{c}-ext-{attr}-upper", mask=mask
    )


def define_ramp_limit_constraints(n: Network, sns: pd.Index, c: str, attr: str) -> None:
    """
    Defines ramp limits for assets with valid ramplimit.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        name of the network component
    """
    m = n.model

    if {"ramp_limit_up", "ramp_limit_down"}.isdisjoint(n.df(c)):
        return

    ramp_limit_up = get_as_dense(n, c, "ramp_limit_up", sns)
    ramp_limit_down = get_as_dense(n, c, "ramp_limit_down", sns)

    if (ramp_limit_up.isnull().all() & ramp_limit_down.isnull().all()).all():
        return
    if (ramp_limit_up.eq(1).all() & ramp_limit_down.eq(1).all()).all():
        return

    # ---------------- Check if ramping is at start of n.snapshots --------------- #

    pnl = n.pnl(c)
    attr = {"p", "p0"}.intersection(pnl).pop()  # dispatch for either one or two ports
    start_i = n.snapshots.get_loc(sns[0]) - 1
    p_start = pnl[attr].iloc[start_i]

    is_rolling_horizon = (sns[0] != n.snapshots[0]) and not p_start.empty
    p = m[f"{c}-p"]

    if is_rolling_horizon:
        active = get_activity_mask(n, c, sns)
        rhs_start = pd.DataFrame(0.0, index=sns, columns=n.df(c).index)
        rhs_start.loc[sns[0]] = p_start

        def p_actual(idx: pd.Index) -> DataArray:
            return reindex(p, c, idx)

        def p_previous(idx: pd.Index) -> DataArray:
            return reindex(p, c, idx).shift(snapshot=1)

    else:
        active = get_activity_mask(n, c, sns[1:])
        rhs_start = pd.DataFrame(0, index=sns[1:], columns=n.df(c).index)
        rhs_start.index.name = "snapshot"

        def p_actual(idx: pd.Index) -> DataArray:
            return reindex(p, c, idx).sel(snapshot=sns[1:])

        def p_previous(idx: pd.Index) -> DataArray:
            return reindex(p, c, idx).shift(snapshot=1).sel(snapshot=sns[1:])

    com_i = n.get_committable_i(c)
    fix_i = n.get_non_extendable_i(c)
    fix_i = fix_i.difference(com_i).rename(fix_i.name)
    ext_i = n.get_extendable_i(c)

    # ----------------------------- Fixed Generators ----------------------------- #

    assets = n.df(c).reindex(fix_i)

    p_nom = n.df(c)[nominal_attrs[c]].reindex(fix_i)

    # fix up
    if not ramp_limit_up[fix_i].isnull().all().all():
        lhs = p_actual(fix_i) - p_previous(fix_i)
        rhs = (ramp_limit_up * p_nom).reindex(
            active.index, columns=fix_i
        ) + rhs_start.reindex(columns=fix_i)
        mask = active.reindex(columns=fix_i) & ~ramp_limit_up.isnull().reindex(
            active.index, columns=fix_i
        )
        m.add_constraints(
            lhs, "<=", rhs, name=f"{c}-fix-{attr}-ramp_limit_up", mask=mask
        )

    # fix down
    if not ramp_limit_down[fix_i].isnull().all().all():
        lhs = p_actual(fix_i) - p_previous(fix_i)
        rhs = (-ramp_limit_down * p_nom).reindex(
            active.index, columns=fix_i
        ) + rhs_start.reindex(columns=fix_i)
        mask = active.reindex(columns=fix_i) & ~ramp_limit_down.isnull().reindex(
            active.index, columns=fix_i
        )
        m.add_constraints(
            lhs, ">=", rhs, name=f"{c}-fix-{attr}-ramp_limit_down", mask=mask
        )

    # ----------------------------- Extendable Generators ----------------------------- #

    assets = n.df(c).reindex(ext_i)

    # ext up
    if not ramp_limit_up[ext_i].isnull().all().all():
        p_nom = m[f"{c}-p_nom"]
        limit_pu = DataArray(ramp_limit_up.reindex(active.index, columns=ext_i))
        lhs = p_actual(ext_i) - p_previous(ext_i) - limit_pu * p_nom
        rhs = rhs_start.reindex(columns=ext_i)
        mask = active.reindex(columns=ext_i) & ~ramp_limit_up.isnull().reindex(
            active.index, columns=ext_i
        )
        m.add_constraints(
            lhs, "<=", rhs, name=f"{c}-ext-{attr}-ramp_limit_up", mask=mask
        )

    # ext down
    if not ramp_limit_down[ext_i].isnull().all().all():
        p_nom = m[f"{c}-p_nom"]
        limit_pu = DataArray(ramp_limit_down.reindex(active.index, columns=ext_i))
        lhs = p_actual(ext_i) - p_previous(ext_i) + limit_pu * p_nom
        rhs = rhs_start.reindex(columns=ext_i)
        mask = active.reindex(columns=ext_i) & ~ramp_limit_down.isnull().reindex(
            active.index, columns=ext_i
        )
        m.add_constraints(
            lhs, ">=", rhs, name=f"{c}-ext-{attr}-ramp_limit_down", mask=mask
        )

    # ----------------------------- Committable Generators ----------------------------- #

    assets = n.df(c).reindex(com_i)

    # com up
    if not assets.ramp_limit_up.isnull().all():
        limit_start = assets.eval("ramp_limit_start_up * p_nom").to_xarray()
        limit_up = assets.eval("ramp_limit_up * p_nom").to_xarray()

        status = m[f"{c}-status"].sel(snapshot=active.index)
        status_prev = m[f"{c}-status"].shift(snapshot=1).sel(snapshot=active.index)

        lhs_tuple = (
            (1, p_actual(com_i)),
            (-1, p_previous(com_i)),
            (limit_start - limit_up, status_prev),
            (-limit_start, status),
        )

        rhs = rhs_start.reindex(columns=com_i)
        if is_rolling_horizon:
            status_start = n.pnl(c)["status"][com_i].iloc[start_i]
            rhs.loc[sns[0]] += (limit_up - limit_start) * status_start

        mask = active.reindex(columns=com_i) & assets.ramp_limit_up.notnull()
        m.add_constraints(
            lhs_tuple, "<=", rhs, name=f"{c}-com-{attr}-ramp_limit_up", mask=mask
        )

    # com down
    if not assets.ramp_limit_down.isnull().all():
        limit_shut = assets.eval("ramp_limit_shut_down * p_nom").to_xarray()
        limit_down = assets.eval("ramp_limit_down * p_nom").to_xarray()

        status = m[f"{c}-status"].sel(snapshot=active.index)
        status_prev = m[f"{c}-status"].shift(snapshot=1).sel(snapshot=active.index)

        lhs_tuple = (
            (1, p_actual(com_i)),
            (-1, p_previous(com_i)),
            (limit_down - limit_shut, status),
            (limit_shut, status_prev),
        )

        rhs = rhs_start.reindex(columns=com_i)
        if is_rolling_horizon:
            status_start = n.pnl(c)["status"][com_i].iloc[start_i]
            rhs.loc[sns[0]] += -limit_shut * status_start

        mask = active.reindex(columns=com_i) & assets.ramp_limit_down.notnull()

        m.add_constraints(
            lhs_tuple, ">=", rhs, name=f"{c}-com-{attr}-ramp_limit_down", mask=mask
        )


def define_nodal_balance_constraints(
    n: Network,
    sns: pd.Index,
    transmission_losses: int = 0,
    buses: Sequence | None = None,
    suffix: str = "",
) -> None:
    """
    Defines nodal balance constraints.
    """
    m = n.model
    if buses is None:
        buses = n.buses.index

    args = [
        ["Generator", "p", "bus", 1],
        ["Store", "p", "bus", 1],
        ["StorageUnit", "p_dispatch", "bus", 1],
        ["StorageUnit", "p_store", "bus", -1],
        ["Line", "s", "bus0", -1],
        ["Line", "s", "bus1", 1],
        ["LineX", "s", "bus0", -1],
        ["LineX", "s", "bus1", 1],
        ["Transformer", "s", "bus0", -1],
        ["Transformer", "s", "bus1", 1],
        ["Link", "p", "bus0", -1],
        ["Link", "p", "bus1", get_as_dense(n, "Link", "efficiency", sns)],
    ]

    if not n.links.empty:
        for i in additional_linkports(n):
            eff = get_as_dense(n, "Link", f"efficiency{i}", sns)
            args.append(["Link", "p", f"bus{i}", eff])

    if transmission_losses:
        args.extend(
            [
                ["Line", "loss", "bus0", -0.5],
                ["Line", "loss", "bus1", -0.5],
                ["LineX", "loss", "bus0", -0.5],
                ["LineX", "loss", "bus1", -0.5],
                ["Transformer", "loss", "bus0", -0.5],
                ["Transformer", "loss", "bus1", -0.5],
            ]
        )

    exprs = []

    for arg in args:
        c, attr, column, sign = arg

        if n.df(c).empty:
            continue

        if "sign" in n.df(c):
            # additional sign necessary for branches in reverse direction
            sign = sign * n.df(c).sign

        expr = DataArray(sign) * m[f"{c}-{attr}"]
        cbuses = n.df(c)[column][lambda ds: ds.isin(buses)].rename("Bus")

        #  drop non-existent multiport buses which are ''
        if column in ["bus" + i for i in additional_linkports(n)]:
            cbuses = cbuses[cbuses != ""]

        expr = expr.sel({c: cbuses.index})

        if expr.size:
            exprs.append(expr.groupby(cbuses).sum())

    lhs = merge(exprs, join="outer").reindex(Bus=buses)
    rhs = (
        (-get_as_dense(n, "Load", "p_set", sns) * n.loads.sign)
        .T.groupby(n.loads.bus)
        .sum()
        .T.reindex(columns=buses, fill_value=0)
    )
    # the name for multi-index is getting lost by groupby before pandas 1.4.0
    # TODO remove once we bump the required pandas version to >= 1.4.0
    rhs.index.name = "snapshot"

    empty_nodal_balance = (lhs.vars == -1).all("_term")
    rhs = DataArray(rhs)
    if empty_nodal_balance.any():
        if (empty_nodal_balance & (rhs != 0)).any().item():
            raise ValueError("Empty LHS with non-zero RHS in nodal balance constraint.")

        mask = ~empty_nodal_balance
    else:
        mask = None

    if suffix:
        lhs = lhs.rename(Bus=f"Bus{suffix}")
        rhs = rhs.rename(Bus=f"Bus{suffix}")
        if mask is not None:
            mask = mask.rename(Bus=f"Bus{suffix}")
    n.model.add_constraints(lhs, "=", rhs, name=f"Bus{suffix}-nodal_balance", mask=mask)


def capacity_reference(n: Network, c: str) -> pd.Series:
    """
    Reference capacity the capacity-dependent branch parameters are evaluated
    at.

    Iterative transmission expansion (see
    ``optimize_transmission_expansion_iteratively``) linearises the
    capacity-dependent branch impedance around a capacity vector which it
    stores in the private column ``_s_nom_def``. Without such a linearisation
    point the nominal capacity of the branch is used.
    """
    df = n.df(c)
    nominal = df[nominal_attrs[c]].astype(float)
    if "_s_nom_def" not in df:
        return nominal
    return df["_s_nom_def"].astype(float).fillna(nominal)


def kirchhoff_voltage_cycles(
    n: Network, period: int | None = None
) -> list[tuple[pd.MultiIndex, sparse.csc_matrix, np.ndarray, str]]:
    """
    Independent cycles of the passive branch network per sub-network.

    The network topology is (re-)determined, so the returned cycle bases are
    consistent with ``n.sub_networks``.

    Parameters
    ----------
    n : pypsa.Network
    period : int, optional
        Investment period to restrict the active assets to.

    Returns
    -------
    list of tuple
        One entry per sub-network which contains at least one cycle, given as
        ``(branches, C, weightings, carrier)``. ``branches`` is a
        ``pd.MultiIndex`` of ``(component, name)`` pairs of all branches of the
        sub-network, ``C`` the cycle incidence matrix with the branches of
        ``branches`` as rows and the independent cycles as columns,
        ``weightings`` the impedance entering the voltage law (``x_pu_eff`` for
        AC and ``r_pu_eff`` for DC sub-networks) and ``carrier`` the carrier of
        the sub-network.
    """
    n.determine_network_topology(investment_period=period, skip_isolated_buses=True)

    cycles = []
    for sub in n.sub_networks.obj:
        if not sub.C.size:
            continue

        branches = sub.branches()
        branches_i = branches.index
        if not isinstance(branches_i, pd.MultiIndex):
            branches_i = pd.MultiIndex.from_arrays(
                [pd.Index(["Line"] * len(branches_i)), branches_i]
            )
        branches_i = branches_i.set_names(["component", "name"])

        carrier = n.sub_networks.carrier[sub.name]
        weightings = branches.x_pu_eff if carrier == "AC" else branches.r_pu_eff
        cycles.append(
            (
                branches_i,
                sparse.csc_matrix(sub.C),
                weightings.to_numpy(dtype=float),
                carrier,
            )
        )
    return cycles


def define_relative_capacity_deviation(
    n: Network, c: str, reference: pd.Series
) -> pd.Series:
    """
    Auxiliary variable carrying the capacity linearisation of the voltage law.

    The sensitivity of a branch term of the voltage law to the branch capacity
    is of the order ``term / F``, so with the capacity measured in MW the
    linearised constraint carries coefficients that are smaller than its own
    flow coefficients by the capacity itself, i.e. by three to four orders of
    magnitude. Rescaling the row cannot repair that, since the spread sits
    *within* the row, and rescaling the capacity column is not available
    either: the same column carries the capital cost and the flow limits, where
    the MW scale is the right one.

    The linearisation is therefore expressed in the dimensionless deviation

    .. math::
        u_l = F_l / \\bar{F}_l - 1

    of the capacity from the linearisation point, which is defined here by one
    equation per branch. Its coefficient in the voltage law is the branch term
    itself, which sits on the scale of the flow coefficients of the same row,
    and the right hand side of the voltage law stays identically zero instead
    of becoming the residual of the linearisation point, i.e. a sum of terms
    that cancel down to round-off.

    Returns
    -------
    labels : pandas.Series
        Variable labels of ``u`` indexed by the branch names.
    """
    m = n.model
    attr = nominal_attrs[c]
    capacity = m[f"{c}-{attr}"]
    dim = capacity.dims[0]
    index = capacity.indexes[dim]

    scale = reference.reindex(index).astype(float)
    # the caller floors the linearisation capacity away from zero, this only
    # guards against a division by zero
    scale = scale.where(scale.abs() > 0.0, 1.0)
    inverse = DataArray(1.0 / scale.to_numpy(), coords=[index], dims=[dim])

    # F_l >= 0 makes u_l >= -1 an exact bound, independent of the step size
    deviation = m.add_variables(
        lower=-1.0, coords=[index], name=f"{c}-{attr}_relative"
    )
    m.add_constraints(
        capacity * inverse - deviation,
        "=",
        1.0,
        name=f"{c}-{attr}_relative-definition",
    )
    return deviation.labels.to_pandas()


def define_kirchhoff_voltage_constraints(n: Network, sns: pd.Index) -> None:
    """
    Defines Kirchhoff voltage constraints.

    Per independent cycle the sum of the branch terms
    ``x_pu_eff * s - q_sssc / s_nom_def`` has to vanish, where the second term
    is the series compensation of an SSSC on a ``LineX`` branch.

    If ``n._kvl_capacity_sensitivity`` is set, the constraint additionally
    carries the first-order sensitivity of these terms with respect to the
    branch capacity, i.e. the term of branch ``l`` is replaced by its
    linearisation

    .. math::
        x_l s_l - q_l / \\bar{F}_l - \\alpha_{l,t} \\bar{F}_l u_l

    around the capacity ``s_nom_def`` = :math:`\\bar{F}`, written in the
    relative capacity deviation :math:`u_l = F_l / \\bar{F}_l - 1` of
    ``define_relative_capacity_deviation`` rather than in the capacity itself,
    which keeps the coefficients of the linearisation on the scale of the flow
    coefficients of their row and the right hand side at zero. The
    sensitivities :math:`\\alpha` are provided as a DataFrame with the
    snapshots as index and ``(component, name)`` pairs as columns by
    ``optimize_transmission_expansion_iteratively`` with
    ``scheme='slp'``. Without them, the capacity dependence of the
    impedance is only resolved by the outer fixed-point iteration.
    """
    m = n.model
    n.calculate_dependent_values()

    comps = [c for c in n.passive_branch_components if not n.df(c).empty]

    if not comps:
        return

    names = ["component", "name"]
    s = pd.concat({c: m[f"{c}-s"].to_pandas() for c in comps}, axis=1, names=names)
    line_x_q = None
    if "LineX-q_sssc" in m.variables:
        line_x_q = m["LineX-q_sssc"].to_pandas()

    s_nom_def = pd.concat({c: capacity_reference(n, c) for c in comps}, names=names)

    sensitivity = getattr(n, "_kvl_capacity_sensitivity", None)
    deviation_labels: dict[str, pd.Series] = {}
    if sensitivity is not None and not sensitivity.empty:
        extendable: dict[str, pd.Index] = {}
        for c in comps:
            var = f"{c}-{nominal_attrs[c]}"
            if var in m.variables:
                extendable[c] = m[var].indexes[m[var].dims[0]]
        keep = [
            col
            for col in sensitivity.columns
            if col[0] in extendable and col[1] in extendable[col[0]]
        ]
        sensitivity = sensitivity[keep] if keep else None
        if sensitivity is not None:
            # the linearisation enters through the relative capacity deviation
            # rather than through the capacity itself
            for c in sorted({col[0] for col in keep}):
                deviation_labels[c] = define_relative_capacity_deviation(
                    n, c, s_nom_def.xs(c, level="component")
                )
    else:
        sensitivity = None

    # Cycle bases built from a spanning tree ("fundamental cycles") are not size
    # balanced: most chords close a short local loop, but a chord connecting two
    # points that are far apart on the tree drags in every branch on the tree path
    # between them, so a handful of cycles can be orders of magnitude larger than
    # the rest. linopy's merge pads every cycle's term axis to the size of the
    # largest one being merged, so merging the whole cycle basis in one call turns
    # a mostly-small, right-skewed distribution into one dense (snapshot x cycles x
    # max_terms) array sized by the few outliers. Bucketing by term count before
    # merging keeps the padding local to cycles of similar size instead.
    kvl_size_buckets = np.array([32, 128, 512, 2048, inf])

    added = 0
    periods = sns.unique("period") if n._multi_invest else [None]

    for period in periods:
        snapshots = sns if period is None else sns[sns.get_loc(period)]
        nsns = len(snapshots)

        exprs_list = []
        for branches_i, C, weightings, carrier in kirchhoff_voltage_cycles(n, period):
            ssub = s.loc[snapshots, branches_i].to_numpy()
            f_ref = s_nom_def.loc[branches_i].to_numpy()

            # series compensation of the SSSCs, active on LineX branches only
            has_q = np.zeros(len(branches_i), dtype=bool)
            q_sub = None
            if carrier == "AC" and line_x_q is not None:
                has_q = np.asarray(branches_i.get_level_values(0) == "LineX")
                if has_q.any():
                    q_sub = np.zeros((nsns, len(branches_i)))
                    q_sub[:, has_q] = line_x_q.loc[
                        snapshots, branches_i.get_level_values(1)[has_q]
                    ].to_numpy()

            # first-order sensitivity of the branch terms to the capacity
            has_alpha = np.zeros(len(branches_i), dtype=bool)
            alpha_sub = None
            alpha_labels = None
            if sensitivity is not None:
                has_alpha = np.asarray(branches_i.isin(sensitivity.columns))
                if has_alpha.any():
                    alpha_sub = np.zeros((nsns, len(branches_i)))
                    alpha_sub[:, has_alpha] = sensitivity.reindex(
                        index=snapshots, columns=branches_i[has_alpha]
                    ).to_numpy()
                    alpha_labels = np.full(len(branches_i), -1, dtype=int)
                    alpha_labels[has_alpha] = [
                        deviation_labels[c].at[i] for c, i in branches_i[has_alpha]
                    ]

            for j in range(C.shape[1]):
                sl = slice(C.indptr[j], C.indptr[j + 1])
                rows = C.indices[sl]
                orientation = C.data[sl]

                coeffs_parts = [1e4 * orientation * weightings[rows]]
                vars_parts = [ssub[:, rows]]

                if q_sub is not None and has_q[rows].any():
                    sel = has_q[rows]
                    q_rows = rows[sel]
                    coeffs_parts.append(-1e4 * orientation[sel] / f_ref[q_rows])
                    vars_parts.append(q_sub[:, q_rows])

                if alpha_sub is not None and has_alpha[rows].any():
                    sel = has_alpha[rows]
                    a_rows = rows[sel]
                    # the deviation is relative, so the coefficient of branch l
                    # is alpha_l * F_def_l, i.e. its voltage law term itself
                    alpha_coeffs = -1e4 * (
                        orientation[sel] * alpha_sub[:, a_rows] * f_ref[a_rows]
                    )
                    # sensitivities the caller truncated away leave a vanishing
                    # coefficient. Masking the variable as well keeps the term
                    # out of the problem for either of linopy's writers, only
                    # one of which filters on the coefficient.
                    alpha_vars = np.where(
                        alpha_coeffs == 0.0, -1, alpha_labels[a_rows]
                    )
                    coeffs_parts.append(alpha_coeffs)
                    vars_parts.append(alpha_vars)

                if any(part.ndim == 2 for part in coeffs_parts):
                    coeffs_parts = [
                        part
                        if part.ndim == 2
                        else np.broadcast_to(part, (nsns, part.size))
                        for part in coeffs_parts
                    ]
                    coeffs = DataArray(
                        np.concatenate(coeffs_parts, axis=1),
                        dims=("snapshot", "_term"),
                        coords={"snapshot": snapshots},
                    )
                else:
                    coeffs = DataArray(np.concatenate(coeffs_parts), dims="_term")
                vars = DataArray(
                    np.concatenate(vars_parts, axis=1),
                    dims=("snapshot", "_term"),
                    coords={"snapshot": snapshots},
                )
                ds = Dataset({"coeffs": coeffs, "vars": vars})
                exprs_list.append(LinearExpression(ds, m))

        if not len(exprs_list):
            continue

        nterms = np.fromiter(
            (e.data.sizes["_term"] for e in exprs_list), dtype=int, count=len(exprs_list)
        )
        bucket_ids = np.searchsorted(kvl_size_buckets, nterms, side="left")

        for bucket_id in np.unique(bucket_ids):
            idx = np.flatnonzero(bucket_ids == bucket_id)

            bucket_exprs = merge([exprs_list[i] for i in idx], dim="cycles")
            bucket_exprs = bucket_exprs.assign_coords(cycles=range(len(idx)))

            suffix = f"{bucket_id}" if period is None else f"{period}-{bucket_id}"
            m.add_constraints(
                bucket_exprs, "=", 0, name=f"Kirchhoff-Voltage-Law-{suffix}"
            )
            added += 1

    if not added:
        return


def define_line_x_sssc_constraints(n: Network, sns: pd.Index) -> None:
    """
    Define LineX SSSC capacity and operation constraints.
    """
    c = "LineX"
    if c not in n.components or n.df(c).empty:
        return

    m = n.model
    q = m[f"{c}-q_sssc"]
    ext_i = n.df(c).index[n.df(c).sssc_nom_extendable].rename(f"{c}-sssc-ext")
    fix_i = n.df(c).index.difference(ext_i).rename(c)

    if not fix_i.empty:
        active = get_activity_mask(n, c, sns, fix_i) if n._multi_invest else None
        q_fix = reindex(q, c, fix_i)
        sssc_fix = n.df(c).sssc_nom.reindex(fix_i)
        m.add_constraints(q_fix <= sssc_fix, name=f"{c}-fix-q_sssc-upper", mask=active)
        m.add_constraints(
            q_fix >= -sssc_fix, name=f"{c}-fix-q_sssc-lower", mask=active
        )

    if ext_i.empty:
        return

    active = get_activity_mask(n, c, sns, ext_i) if n._multi_invest else None
    q_ext = reindex(q, c, ext_i)
    sssc = m[f"{c}-sssc_nom"]

    m.add_constraints(q_ext - sssc <= 0, name=f"{c}-ext-q_sssc-upper", mask=active)
    m.add_constraints(q_ext + sssc >= 0, name=f"{c}-ext-q_sssc-lower", mask=active)

def define_fixed_nominal_constraints(n: Network, c: str, attr: str) -> None:
    """
    Sets constraints for fixing static variables of a given component and
    attribute to the corresponding values in `n.df(c)[attr + '_set']`.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'
    """
    if attr + "_set" not in n.df(c):
        return

    dim = f"{c}-{attr}_set_i"
    fix = n.df(c)[attr + "_set"].dropna().rename_axis(dim)

    if fix.empty:
        return

    var = n.model[f"{c}-{attr}"]
    var = reindex(var, var.dims[0], fix.index)
    n.model.add_constraints(var, "=", fix, name=f"{c}-{attr}_set")


def define_modular_constraints(n: Network, c: str, attr: str) -> None:
    """
    Sets constraints for fixing modular variables of a given component. It
    allows to define optimal capacity of a component as multiple of the nominal
    capacity of the single module.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        name of the network component
    attr : str
        name of the variable, e.g. 'n_opt'
    """
    m = n.model
    mod_i = n.df(c).query(f"{attr}_extendable and ({attr}_mod>0)").index

    if (mod_i).empty:
        return

    modularity = m.variables[f"{c}-n_mod"]
    modular_capacity = n.df(c)[f"{attr}_mod"].loc[mod_i]
    capacity = m.variables[f"{c}-{attr}"].loc[mod_i]

    con = capacity - modularity * modular_capacity.values == 0
    n.model.add_constraints(con, name=f"{c}-{attr}_modularity", mask=None)


def define_fixed_operation_constraints(
    n: Network, sns: pd.Index, c: str, attr: str
) -> None:
    """
    Sets constraints for fixing time-dependent variables of a given component
    and attribute to the corresponding values in `n.pnl(c)[attr + '_set']`.

    Parameters
    ----------
    n : pypsa.Network
    c : str
        name of the network component
    attr : str
        name of the attribute, e.g. 'p'
    """
    if attr + "_set" not in n.pnl(c):
        return

    dim = f"{c}-{attr}_set_i"
    fix = n.pnl(c)[attr + "_set"].reindex(index=sns).rename_axis(columns=dim)
    fix.index.name = "snapshot"  # still necessary: reindex loses the index name

    if fix.empty:
        return

    if n._multi_invest:
        active = get_activity_mask(n, c, sns, index=fix.columns)
        mask = fix.notna() & active
    else:
        active = None
        mask = fix.notna()

    var = reindex(n.model[f"{c}-{attr}"], c, fix.columns)
    n.model.add_constraints(var, "=", fix, name=f"{c}-{attr}_set", mask=mask)


def define_storage_unit_constraints(n: Network, sns: pd.Index) -> None:
    """
    Defines energy balance constraints for storage units. In principal the
    constraints states:

    previous_soc + p_store - p_dispatch + inflow - spill == soc
    """
    m = n.model
    c = "StorageUnit"
    dim = "snapshot"
    assets = n.df(c)
    active = DataArray(get_activity_mask(n, c, sns))

    if assets.empty:
        return

    # elapsed hours
    eh = expand_series(n.snapshot_weightings.stores[sns], assets.index)
    # efficiencies
    eff_stand = (1 - get_as_dense(n, c, "standing_loss", sns)).pow(eh)
    eff_dispatch = get_as_dense(n, c, "efficiency_dispatch", sns)
    eff_store = get_as_dense(n, c, "efficiency_store", sns)

    soc = m[f"{c}-state_of_charge"]

    lhs = [
        (-1, soc),
        (-1 / eff_dispatch * eh, m[f"{c}-p_dispatch"]),
        (eff_store * eh, m[f"{c}-p_store"]),
    ]

    if f"{c}-spill" in m.variables:
        lhs += [(-eh, m[f"{c}-spill"])]

    # We create a mask `include_previous_soc` which excludes the first snapshot
    # for non-cyclic assets.
    noncyclic_b = ~assets.cyclic_state_of_charge.to_xarray()
    include_previous_soc = (active.cumsum(dim) != 1).where(noncyclic_b, True)

    previous_soc = (
        soc.where(active)
        .ffill(dim)
        .roll(snapshot=1)
        .ffill(dim)
        .where(include_previous_soc)
    )

    # We add inflow and initial soc for noncyclic assets to rhs
    soc_init = assets.state_of_charge_initial.to_xarray()
    rhs = DataArray(-get_as_dense(n, c, "inflow", sns).mul(eh))

    if isinstance(sns, pd.MultiIndex):
        # If multi-horizon optimizing, we update the previous_soc and the rhs
        # for all assets which are cyclid/non-cyclid per period.
        periods = soc.coords["period"]
        per_period = (
            assets.cyclic_state_of_charge_per_period.to_xarray()
            | assets.state_of_charge_initial_per_period.to_xarray()
        )

        # We calculate the previous soc per period while cycling within a period
        # Normally, we should use groupby, but is broken for multi-index
        # see https://github.com/pydata/xarray/issues/6836
        ps = sns.unique("period")
        sl = slice(None)
        previous_soc_pp_list = [
            soc.data.sel(snapshot=(p, sl)).roll(snapshot=1) for p in ps
        ]
        previous_soc_pp = concat(previous_soc_pp_list, dim="snapshot")

        # We create a mask `include_previous_soc_pp` which excludes the first
        # snapshot of each period for non-cyclic assets.
        include_previous_soc_pp = active & (periods == periods.shift(snapshot=1))
        include_previous_soc_pp = include_previous_soc_pp.where(noncyclic_b, True)
        # We take values still to handle internal xarray multi-index difficulties
        previous_soc_pp = previous_soc_pp.where(
            include_previous_soc_pp.values, linopy.variables.FILL_VALUE
        )

        # update the previous_soc variables and right hand side
        previous_soc = previous_soc.where(~per_period, previous_soc_pp)
        include_previous_soc = include_previous_soc_pp.where(
            per_period, include_previous_soc
        )
    lhs += [(eff_stand, previous_soc)]
    rhs = rhs.where(include_previous_soc, rhs - soc_init)
    m.add_constraints(lhs, "=", rhs, name=f"{c}-energy_balance", mask=active)


def define_store_constraints(n: Network, sns: pd.Index) -> None:
    """
    Defines energy balance constraints for stores. In principal the constraints
    states:

    previous_e - p == e
    """
    m = n.model
    c = "Store"
    dim = "snapshot"
    assets = n.df(c)
    active = DataArray(get_activity_mask(n, c, sns))

    if assets.empty:
        return

    # elapsed hours
    eh = expand_series(n.snapshot_weightings.stores[sns], assets.index)
    # efficiencies
    eff_stand = (1 - get_as_dense(n, c, "standing_loss", sns)).pow(eh)

    e = m[f"{c}-e"]
    p = m[f"{c}-p"]

    lhs = [(-1, e), (-eh, p)]

    # We create a mask `include_previous_e` which excludes the first snapshot
    # for non-cyclic assets.
    noncyclic_b = ~assets.e_cyclic.to_xarray()
    include_previous_e = (active.cumsum(dim) != 1).where(noncyclic_b, True)

    previous_e = (
        e.where(active).ffill(dim).roll(snapshot=1).ffill(dim).where(include_previous_e)
    )

    # We add inflow and initial e for for noncyclic assets to rhs
    e_init = assets.e_initial.to_xarray()

    if isinstance(sns, pd.MultiIndex):
        # If multi-horizon optimizing, we update the previous_e and the rhs
        # for all assets which are cyclid/non-cyclid per period.
        periods = e.coords["period"]
        per_period = (
            assets.e_cyclic_per_period.to_xarray()
            | assets.e_initial_per_period.to_xarray()
        )

        # We calculate the previous e per period while cycling within a period
        # Normally, we should use groupby, but is broken for multi-index
        # see https://github.com/pydata/xarray/issues/6836
        ps = sns.unique("period")
        sl = slice(None)
        previous_e_pp_list = [e.data.sel(snapshot=(p, sl)).roll(snapshot=1) for p in ps]
        previous_e_pp = concat(previous_e_pp_list, dim="snapshot")

        # We create a mask `include_previous_e_pp` which excludes the first
        # snapshot of each period for non-cyclic assets.
        include_previous_e_pp = active & (periods == periods.shift(snapshot=1))
        include_previous_e_pp = include_previous_e_pp.where(noncyclic_b, True)
        # We take values still to handle internal xarray multi-index difficulties
        previous_e_pp = previous_e_pp.where(
            include_previous_e_pp.values, linopy.variables.FILL_VALUE
        )

        # update the previous_e variables and right hand side
        previous_e = previous_e.where(~per_period, previous_e_pp)
        include_previous_e = include_previous_e_pp.where(per_period, include_previous_e)

    lhs += [(eff_stand, previous_e)]
    rhs = -e_init.where(~include_previous_e, 0)

    m.add_constraints(lhs, "=", rhs, name=f"{c}-energy_balance", mask=active)


def define_loss_constraints(
    n: Network, sns: pd.Index, c: str, transmission_losses: int
) -> None:
    """
    Sets the piecewise-linear approximation of the quadratic branch losses.

    The approximation is exact in the nominal capacity: since the conductor is
    scaled as r = rho / s_nom and the flow range as p_max = s_max_pu * s_nom,
    the segment slopes are capacity-independent and the segment offsets are
    linear in s_nom, so no outer iteration on r is required for the losses.

    Parameters
    ----------
    n : pypsa.Network
    sns : pd.Index
        Snapshots of the constraint.
    c : str
        name of the network component
    transmission_losses : int
        Number of least-squares segments per half-axis; the loss parabola is
        approximated on [-p_max, p_max] by 2 * transmission_losses segments.
    """
    if n.df(c).empty or c not in n.passive_branch_components:
        return

    n_side = max(1, int(transmission_losses))

    # Piecewise-linear approximation of the quadratic loss r * p**2 on
    # [-p_max, p_max]. Each half-axis is split into n_side segments of width
    # delta = p_max / n_side; segment k = 1 ... n_side covers
    # [(k-1) delta, k delta] and is fitted by least squares. The innermost
    # segment is instead fitted under the constraint that it passes through the
    # origin, so that the envelope does too and the losses stay non-negative:
    #     min_a int_0^delta (a p - r p**2)**2 dp  =>  a = 3 r delta / 4.
    #
    # The least-squares fit of r * p**2 on [x0, x1] has the closed form
    #     slope  =  r * (x0 + x1)
    #     offset = -r * (x0**2 + 4 * x0 * x1 + x1**2) / 6
    # which is homogeneous: scaling the interval by alpha scales the slope by
    # alpha and the offset by alpha**2. Evaluating it on the normalised
    # interval (r = 1, p_max = 1) therefore yields branch-independent
    # coefficients that only have to be rescaled by r and p_max:
    #     slope_k  =  r * p_max    * slope_hat[k]
    #     offset_k = -r * p_max**2 * offset_hat[k]
    k = np.arange(1, n_side + 1, dtype=float)
    slope_hat = (2.0 * k - 1.0) / n_side
    offset_hat = (6.0 * k**2 - 6.0 * k + 1.0) / (6.0 * n_side**2)
    slope_hat[0] = 0.75 / n_side
    offset_hat[0] = 0.0

    # Both factors are tied to the branch capacity: the conductor is scaled as
    # r = rho / s_nom (rho = r * s_nom invariant, see
    # optimize_transmission_expansion_iteratively) and the flow is bounded by
    # p_max = u * s_nom with u = max_t s_max_pu. Substituting both,
    #     slope_k  = rho * u    * slope_hat[k]                (s_nom cancels)
    #     offset_k = rho * u**2 * offset_hat[k] * s_nom       (linear in s_nom)
    # so for extendable branches the offset can be carried as a term in the
    # s_nom variable instead of being frozen at an assumed capacity. The loss
    # constraints are then exact in s_nom and need no outer iteration.
    df = n.df(c)
    s_nom_def = capacity_reference(n, c)
    rho = df["r_pu_eff"].astype(float) * s_nom_def
    u = get_as_dense(n, c, "s_max_pu").loc[sns].max(axis=0).reindex(df.index)

    ext_i = n.get_extendable_i(c)
    undefined = ext_i[
        ~(np.isfinite(rho.reindex(ext_i)) & (s_nom_def.reindex(ext_i) > 0))
    ]
    if not undefined.empty:
        msg = (
            "The loss approximation scales the resistance with the branch "
            "capacity and hence requires a strictly positive reference "
            f"capacity with a finite 'r_pu_eff' for every extendable {c}. Set "
            "'s_nom' to the capacity the given 'r' refers to for:\n"
            f"{list(undefined)}"
        )
        raise ValueError(msg)

    rho = rho.where(np.isfinite(rho), 0.0)
    u = u.where(np.isfinite(u), 0.0)

    loss = n.model[f"{c}-loss"]
    flow = n.model[f"{c}-s"]

    fix_i = df.index.difference(ext_i).rename(c)
    if not fix_i.empty:
        active = get_activity_mask(n, c, sns, fix_i) if n._multi_invest else None
        loss_fix = reindex(loss, c, fix_i)
        flow_fix = reindex(flow, c, fix_i)
        # s_nom is a parameter here, so rho * u**2 * s_nom collapses back to
        # the plain constant r * p_max**2.
        base = rho.reindex(fix_i) * u.reindex(fix_i)
        offset = base * u.reindex(fix_i) * s_nom_def.reindex(fix_i)
        for j in range(n_side):
            slope = base * slope_hat[j]
            rhs = -offset * offset_hat[j]
            n.model.add_constraints(
                n.model.linexpr((1, loss_fix), (-slope, flow_fix)),
                ">=",
                rhs,
                name=f"{c}-fix-loss_tangents-{j + 1}-1",
                mask=active,
            )
            n.model.add_constraints(
                n.model.linexpr((1, loss_fix), (slope, flow_fix)),
                ">=",
                rhs,
                name=f"{c}-fix-loss_tangents-{j + 1}--1",
                mask=active,
            )

    if not ext_i.empty:
        active = get_activity_mask(n, c, sns, ext_i) if n._multi_invest else None
        loss_ext = reindex(loss, c, ext_i)
        flow_ext = reindex(flow, c, ext_i)
        capacity = n.model[f"{c}-{nominal_attrs[c]}"]
        base = rho.reindex(ext_i) * u.reindex(ext_i)
        for j in range(n_side):
            slope = base * slope_hat[j]
            gamma = base * u.reindex(ext_i) * offset_hat[j]
            n.model.add_constraints(
                n.model.linexpr((1, loss_ext), (-slope, flow_ext), (gamma, capacity)),
                ">=",
                0,
                name=f"{c}-ext-loss_tangents-{j + 1}-1",
                mask=active,
            )
            n.model.add_constraints(
                n.model.linexpr((1, loss_ext), (slope, flow_ext), (gamma, capacity)),
                ">=",
                0,
                name=f"{c}-ext-loss_tangents-{j + 1}--1",
                mask=active,
            )
