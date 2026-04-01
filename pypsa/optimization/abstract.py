#!/usr/bin/env python3
"""
Build abstracted, extended optimisation problems from PyPSA networks with
Linopy.
"""

from __future__ import annotations

import copy
import gc
import logging
import re
from collections.abc import Sequence
from itertools import product
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from linopy import LinearExpression, QuadraticExpression, merge

from pypsa.descriptors import nominal_attrs
from pypsa.utils import as_index

if TYPE_CHECKING:
    from pypsa import Network
logger = logging.getLogger(__name__)


RTEP_INNER_EQUIVALENT_CARRIER = "__rtep_inner_equivalent__"
RTEP_INNER_LOAD_SUFFIX = "__rtep_inner_load"
RTEP_INNER_MAX_ITERATIONS = 5
RTEP_INNER_INJECTION_ZERO_THRESHOLD = 1e-2
RTEP_INNER_FIXED_LOSS_ZERO_THRESHOLD = 1e-2


def optimize_transmission_expansion_iteratively(
    n: Network,
    snapshots: Sequence | None = None,
    outer_msq_threshold: float = 0.01,
    min_iterations: int = 1,
    max_iterations: int = 100,
    track_iterations: bool = False,
    line_unit_size: float | None = None,
    link_unit_size: dict | None = None,
    line_threshold: float | None = None,
    link_threshold: dict | None = None,
    **kwargs: Any,
) -> tuple[str, str]:
    """
    Iterative linear optimization updating the line parameters for passive AC
    and DC lines. This is helpful when line expansion is enabled. After each
    successful solving, line impedances and line resistance are recalculated
    based on the optimization result. If warmstart is possible, it uses the
    result from the previous iteration to fasten the optimization.

    Parameters
    ----------
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        network.snapshots, defaults to network.snapshots
    outer_msq_threshold: float, default 0.03
        Maximal mean square difference between optimized line capacity of
        the current and the previous outer iteration. As soon as this threshold is
        undercut, and the number of iterations is bigger than 'min_iterations'
        the outer iterative optimization stops
    min_iterations : integer, default 1
        Minimal number of iteration to run regardless whether the outer_msq_threshold
        is already undercut
    max_iterations : integer, default 100
        Maximal number of iterations to run regardless whether outer_msq_threshold
        is already undercut
    track_iterations: bool, default False
        If True, the intermediate branch capacities and values of the
        objective function are recorded for each iteration. The values of
        iteration 0 represent the initial state.
    line_unit_size: float, default None
        The unit size for line components.
        Use None if no discretization is desired.
    link_unit_size: dict-like, default None
        A dictionary containing the unit sizes for link components,
        with carrier names as keys. Use None if no discretization is desired.
    line_threshold: float, default 0.3
        The threshold relative to the unit size for discretizing line components.
    link_threshold: dict-like, default 0.3 per carrier
        The threshold relative to the unit size for discretizing link components.
    **kwargs
        Keyword arguments of the `n.optimize` function which runs at each iteration
    """
    if snapshots is None:
        snapshots = n.snapshots
    snapshots = as_index(n, snapshots, "snapshots", "snapshot")

    branch_components = [c for c in ("Line", "LineX") if c in n.components and not n.df(c).empty]
    if not branch_components:
        return n.optimize(snapshots, **kwargs)

    for c in branch_components:
        n.df(c)["carrier"] = n.df(c).bus0.map(n.buses.carrier)

    branch_data: dict[str, dict[str, pd.Series | pd.Index]] = {}
    for c in branch_components:
        df = n.df(c)
        ext_i = n.get_extendable_i(c).copy()
        typed_i = df.query('type != ""').index
        ext_untyped_i = ext_i.difference(typed_i)
        ext_typed_i = ext_i.intersection(typed_i)
        base_s_nom = np.sqrt(3) * df["type"].map(n.line_types.i_nom) * df.bus0.map(
            n.buses.v_nom
        )
        if not ext_typed_i.empty:
            df.loc[ext_typed_i, "num_parallel"] = (df.s_nom / base_s_nom)[ext_typed_i]
        branch_data[c] = {
            "ext_i": ext_i,
            "typed_i": typed_i,
            "ext_untyped_i": ext_untyped_i,
            "ext_typed_i": ext_typed_i,
            "base_s_nom": base_s_nom,
            "x_0": df.x.copy(),
            "r_0": df.r.copy(),
        }

    def collect_branch_caps(attr: str) -> pd.Series:
        return pd.concat(
            {c: n.df(c)[attr] for c in branch_components},
            names=["component", "name"],
        )

    def parse_constraint_carriers(value: Any) -> list[str]:
        if pd.isna(value):
            return []
        return [
            re.sub(r"[\[\]\(\)]", "", carrier.strip())
            for carrier in str(value).split(",")
            if carrier.strip()
        ]

    def get_branch_caps_for_reporting(
        network: Network, component: str, branch_caps: pd.Series
    ) -> pd.Series:
        df = network.df(component)
        try:
            caps = branch_caps.xs(component, level="component")
        except (KeyError, ValueError):
            return pd.Series(index=df.index, dtype=float)
        return caps.reindex(df.index)

    def report_inner_transmission_budget_usage(
        base_network: Network,
        inner_branch_caps: pd.Series,
        outer_iteration: int,
        inner_iteration: int,
    ) -> None:
        if base_network.global_constraints.empty:
            return

        period_weighting = None
        if base_network._multi_invest and isinstance(snapshots, pd.MultiIndex):
            periods = snapshots.unique("period")
            period_weighting = base_network.investment_period_weightings.objective[
                periods
            ]

        for name, glc in base_network.global_constraints.iterrows():
            if glc.type not in {
                "transmission_expansion_cost_limit",
                "transmission_volume_expansion_limit",
            }:
                continue

            carriers = parse_constraint_carriers(glc.carrier_attribute)
            period = glc.investment_period
            implied = 0.0

            def filter_active_assets(
                component: str, index: pd.Index
            ) -> tuple[pd.Index, int | pd.Series]:
                if index.empty:
                    return index, 1

                if not pd.isna(period):
                    index = index[base_network.get_active_assets(component, period)[index]]
                    return index, 1

                if isinstance(snapshots, pd.MultiIndex):
                    index = index[
                        base_network.get_active_assets(component, snapshots.unique("period"))[
                            index
                        ]
                    ]
                    if (
                        glc.type == "transmission_expansion_cost_limit"
                        and period_weighting is not None
                    ):
                        active = pd.concat(
                            {
                                current_period: base_network.get_active_assets(
                                    component, current_period
                                )[index]
                                for current_period in snapshots.unique("period")
                            },
                            axis=1,
                        )
                        return index, active @ period_weighting

                return index, 1

            for c in ["Line", "LineX", "Link"]:
                if c not in nominal_attrs or c not in base_network.components:
                    continue

                df = base_network.df(c)
                if df.empty:
                    continue

                ext_i = base_network.get_extendable_i(c)
                if ext_i.empty or "carrier" not in df.columns:
                    continue

                ext_i = ext_i.intersection(df.query("carrier in @carriers").index)
                ext_i, weights = filter_active_assets(c, ext_i)

                if ext_i.empty:
                    continue

                if c == "Link":
                    attr = nominal_attrs[c]
                    caps = (
                        df.get(f"{attr}_opt", df[attr])
                        .reindex(ext_i)
                        .fillna(df[attr].reindex(ext_i))
                    )
                else:
                    caps = get_branch_caps_for_reporting(base_network, c, inner_branch_caps)
                    caps = caps.reindex(ext_i).fillna(df[nominal_attrs[c]].reindex(ext_i))

                if glc.type == "transmission_expansion_cost_limit":
                    coeff = df.capital_cost.reindex(ext_i)
                    implied += float((coeff * caps * weights).sum())
                else:
                    coeff = df.length.reindex(ext_i)
                    implied += float((coeff * caps).sum())

            budget = float(glc.constant)
            ratio = np.nan if abs(budget) < 1e-12 else implied / budget
            logger.info(
                "RTEP inner iteration %s.%s: %s (%s) implied=%#.6g budget=%#.6g ratio=%s",
                outer_iteration,
                inner_iteration,
                name,
                glc.type,
                implied,
                budget,
                "nan" if np.isnan(ratio) else f"{ratio:.3e}",
            )

    ext_branches = pd.MultiIndex.from_tuples(
        [(c, i) for c in branch_components for i in branch_data[c]["ext_i"]],
        names=["component", "name"],
    )

    def update_line_params(network: Network, s_nom_define: pd.Series) -> None:
        """
        Update branch impedance-defining parameters from the current F_def vector.
        """
        for c in branch_components:
            df = network.df(c)
            data = branch_data[c]
            target = (
                s_nom_define.xs(c, level="component")
                .reindex(df.index)
                .fillna(df["s_nom"])
            )
            df["_s_nom_def"] = target
            factor = target / df["s_nom"].replace(0, np.nan)

            ac_i = df.query("carrier == 'AC'").index.intersection(data["ext_untyped_i"])
            if not ac_i.empty:
                df.loc[ac_i, "x"] = data["x_0"][ac_i] / factor[ac_i]
                df.loc[ac_i, "r"] = data["r_0"][ac_i] / factor[ac_i]

            typed_i = data["ext_typed_i"]
            if not typed_i.empty:
                df.loc[typed_i, "num_parallel"] = target[typed_i] / data["base_s_nom"][typed_i]

    def save_optimal_capacities(network: Network, iteration: int, status: str) -> None:
        for c, attr in pd.Series(nominal_attrs)[list(network.branch_components)].items():
            network.df(c)[f"{attr}_opt_{iteration}"] = network.df(c)[f"{attr}_opt"]
        if "LineX" in network.components and not network.line_xs.empty:
            network.line_xs[f"sssc_nom_opt_{iteration}"] = network.line_xs["sssc_nom_opt"]
        setattr(network, f"status_{iteration}", status)
        setattr(network, f"objective_{iteration}", network.objective)
        network.iteration = iteration
        network.global_constraints = network.global_constraints.rename(
            columns={"mu": f"mu_{iteration}"}
        )

    def discretized_capacity(nom_opt, unit_size, threshold, min_units=0):
        units = nom_opt // unit_size + (nom_opt % unit_size >= threshold * unit_size)
        return max(min_units, units) * unit_size

    def discretize_branch_components(
        network: Network,
        line_unit_size: float | None,
        link_unit_size: dict | None,
        line_threshold: float | None,
        link_threshold: dict | None,
    ) -> None:
        line_threshold = line_threshold or 0.3
        link_threshold = link_threshold or {}

        if line_unit_size:
            min_units = 1
            for c in branch_components:
                network.df(c)["s_nom"] = network.df(c)["s_nom_opt"].apply(
                    discretized_capacity,
                    args=(line_unit_size, line_threshold, min_units),
                )

        if link_unit_size:
            for carrier in link_unit_size.keys() & set(network.links.carrier.unique()):
                sel = network.links.carrier == carrier
                network.links.loc[sel, "p_nom"] = network.links.loc[sel, "p_nom_opt"].apply(
                    discretized_capacity,
                    args=(link_unit_size[carrier], link_threshold.get(carrier, 0.3)),
                )

    def relative_capacity_change(current: pd.Series, previous: pd.Series) -> float:
        if ext_branches.empty:
            return 0.0
        denom = np.linalg.norm(previous.loc[ext_branches].to_numpy())
        denom = max(denom, 1e-12)
        return float(
            np.linalg.norm(
                (current - previous).loc[ext_branches].to_numpy()
            )
            / denom
        )

    def sanitize_solver_kwargs(solve_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Copy solver kwargs without altering user-provided solver tolerances.
        """
        clean_kwargs = dict(solve_kwargs)
        solver_options = clean_kwargs.get("solver_options")
        if isinstance(solver_options, dict):
            clean_kwargs["solver_options"] = dict(solver_options)

        return clean_kwargs

    def sanitize_inner_solver_kwargs(solve_kwargs: dict[str, Any]) -> dict[str, Any]:
        """
        Remove workflow-specific solver tweaks that should not be enforced in the
        simplified inner RTEP solves.
        """
        clean_kwargs = sanitize_solver_kwargs(solve_kwargs)

        solver_options = clean_kwargs.get("solver_options")
        if isinstance(solver_options, dict):
            solver_options = dict(solver_options)
        else:
            solver_options = {}

        solver_name = str(clean_kwargs.get("solver_name", "highs")).lower()
        if solver_name == "gurobi":
            solver_options["crossover"] = 1
            # solver_options["BarHomogeneous"] = 1

        clean_kwargs["solver_options"] = solver_options

        return clean_kwargs

    def inherit_missing_network_attrs(source: Network, target: Network) -> None:
        """
        Preserve user-defined runtime attributes on selectively copied networks.
        """
        for attr, value in source.__dict__.items():
            if attr == "model" or attr in target.__dict__:
                continue
            try:
                setattr(target, attr, copy.deepcopy(value))
            except Exception:
                setattr(target, attr, value)

    def disable_inner_only_runtime_features(network: Network) -> None:
        """
        Disable workflow-specific features that should not be active in the
        simplified inner RTEP subproblem.
        """
        config = getattr(network, "config", None)
        if not isinstance(config, dict):
            return

        rep_cfg = (
            config.get("clustering", {})
            .get("temporal", {})
            .get("representative_periods", {})
        )
        if isinstance(rep_cfg, dict) and rep_cfg.get("enable", False):
            rep_cfg["enable"] = False
            logger.info(
                "Disabled representative periods for the inner RTEP optimization."
            )

    def normalize_empty_component_indices(network: Network) -> None:
        """
        Keep empty component indices string-typed so external workflow hooks
        using `.index.str` on empty tables do not fail.
        """
        for c in network.all_components:
            df = network.df(c)
            if not df.empty:
                continue
            df.index = pd.Index([], dtype=object, name=df.index.name or c)

    def collect_outer_branch_losses(base_network: Network) -> dict[str, pd.DataFrame]:
        """
        Read the outer-loop passive branch losses that should be kept fixed in
        the inner RTEP subproblem.
        """
        if not hasattr(base_network, "model"):
            return {}

        losses: dict[str, pd.DataFrame] = {}
        for c in branch_components:
            name = f"{c}-loss"
            if name not in base_network.model.variables:
                continue

            loss = (
                base_network.model[name]
                .solution.to_pandas()
                .reindex(index=snapshots, columns=base_network.df(c).index, fill_value=0.0)
                .clip(lower=0.0)
            )
            losses[c] = loss

        return losses

    def prepare_rtep_network(
        base_network: Network,
        s_nom_define: pd.Series,
        outer_branch_losses: dict[str, pd.DataFrame],
    ) -> Network:
        override_components, override_component_attrs = (
            base_network._retrieve_overridden_components()
        )
        rtep_n = base_network.__class__(
            override_components=override_components,
            override_component_attrs=override_component_attrs,
        )
        rtep_n.set_snapshots(base_network.snapshots)
        rtep_n._snapshot_weightings = base_network.snapshot_weightings.copy()
        rtep_n._investment_periods = base_network.investment_periods.copy()
        rtep_n._investment_period_weightings = (
            base_network.investment_period_weightings.copy()
        )
        inherit_missing_network_attrs(base_network, rtep_n)
        disable_inner_only_runtime_features(rtep_n)
        normalize_empty_component_indices(rtep_n)

        if not base_network.carriers.empty:
            rtep_n.import_components_from_dataframe(
                pd.DataFrame(base_network.carriers), "Carrier"
            )

        def unique_temp_names(
            buses: pd.Index, existing: pd.Index, suffix: str
        ) -> pd.Index:
            names = []
            used = set(existing.astype(str))
            for bus in buses.astype(str):
                candidate = f"{bus}{suffix}"
                counter = 1
                while candidate in used:
                    candidate = f"{bus}{suffix}_{counter}"
                    counter += 1
                used.add(candidate)
                names.append(candidate)
            return pd.Index(names)

        keep_components = {"Line", "LineX"}
        explicit_ac_buses = base_network.buses.index[
            base_network.buses.carrier.fillna("") == "AC"
        ]
        ac_buses = explicit_ac_buses.copy()
        if ac_buses.empty:
            ac_buses = pd.Index([], dtype=object)
            for c in keep_components:
                if c not in base_network.components or base_network.df(c).empty:
                    continue
                branch_i = base_network.df(c).index
                if "carrier" in base_network.df(c):
                    carrier = base_network.df(c).carrier.fillna("")
                    branch_i = branch_i.intersection(carrier[carrier != "DC"].index)
                if branch_i.empty:
                    continue
                ac_buses = ac_buses.union(base_network.df(c).loc[branch_i, "bus0"])
                ac_buses = ac_buses.union(base_network.df(c).loc[branch_i, "bus1"])

        if ac_buses.empty:
            return rtep_n

        bus_df = pd.DataFrame(base_network.buses.loc[ac_buses]).assign(sub_network="")
        rtep_n.import_components_from_dataframe(bus_df, "Bus")
        allocated_bus_losses = pd.DataFrame(0.0, index=snapshots, columns=ac_buses)
        fixed_branch_losses: dict[str, pd.DataFrame] = {}

        if "Line" in keep_components and not base_network.lines.empty:
            line_type_i = (
                base_network.lines.loc[lambda df: df.bus0.isin(ac_buses) & df.bus1.isin(ac_buses), "type"]
                .mask(lambda s: s.eq(""))
                .dropna()
                .unique()
            )
            if len(line_type_i):
                rtep_n.import_components_from_dataframe(
                    pd.DataFrame(base_network.line_types.loc[line_type_i]), "LineType"
                )

        for c in keep_components:
            if c not in base_network.components or base_network.df(c).empty:
                continue

            keep_i = base_network.df(c).index
            if "carrier" in base_network.df(c):
                carrier = base_network.df(c).carrier.fillna("")
                keep_i = keep_i.intersection(carrier[carrier != "DC"].index)

            buses0 = base_network.df(c).bus0.reindex(keep_i).isin(ac_buses)
            buses1 = base_network.df(c).bus1.reindex(keep_i).isin(ac_buses)
            keep_i = keep_i[buses0 & buses1]
            if keep_i.empty:
                continue

            rtep_n.import_components_from_dataframe(
                pd.DataFrame(base_network.df(c).loc[keep_i]), c
            )

            outer_loss = outer_branch_losses.get(c)
            if outer_loss is None:
                continue

            fixed_loss = outer_loss.reindex(
                index=snapshots, columns=keep_i, fill_value=0.0
            ).clip(lower=0.0)
            fixed_loss = fixed_loss.where(
                fixed_loss >= RTEP_INNER_FIXED_LOSS_ZERO_THRESHOLD, 0.0
            )
            fixed_branch_losses[c] = fixed_loss

            branch_bus0 = base_network.df(c).bus0.reindex(keep_i)
            branch_bus1 = base_network.df(c).bus1.reindex(keep_i)
            allocated_bus_losses = allocated_bus_losses.add(
                (0.5 * fixed_loss).rename(columns=branch_bus0).T.groupby(level=0).sum().T,
                fill_value=0.0,
            )
            allocated_bus_losses = allocated_bus_losses.add(
                (0.5 * fixed_loss).rename(columns=branch_bus1).T.groupby(level=0).sum().T,
                fill_value=0.0,
            )

        update_line_params(rtep_n, s_nom_define)
        rtep_n.determine_network_topology(skip_isolated_buses=False)

        try:
            net_ac_injection = base_network.buses_t.p.reindex(
                index=snapshots, columns=ac_buses, fill_value=0.0
            ).copy()
        except AttributeError:
            net_ac_injection = pd.DataFrame(0.0, index=snapshots, columns=ac_buses)

        if not base_network.transformers.empty:
            trafo_i = base_network.transformers.index[
                base_network.transformers.bus0.isin(ac_buses)
                | base_network.transformers.bus1.isin(ac_buses)
            ]
            if not trafo_i.empty:
                trafo_p0 = base_network.transformers_t.p0.reindex(
                    index=snapshots, columns=trafo_i, fill_value=0.0
                )
                trafo_p1 = base_network.transformers_t.p1.reindex(
                    index=snapshots, columns=trafo_i, fill_value=0.0
                )
                if not trafo_p0.empty:
                    bus0 = base_network.transformers.bus0.reindex(trafo_i)
                    bus1 = base_network.transformers.bus1.reindex(trafo_i)
                    net_ac_injection = net_ac_injection.add(
                        (-trafo_p0).rename(columns=bus0).T.groupby(level=0).sum().T,
                        fill_value=0.0,
                    )
                    net_ac_injection = net_ac_injection.add(
                        (-trafo_p1).rename(columns=bus1).T.groupby(level=0).sum().T,
                        fill_value=0.0,
                    )
                net_ac_injection = net_ac_injection.reindex(
                    columns=ac_buses, fill_value=0.0
                )

        if fixed_branch_losses:
            net_ac_injection = net_ac_injection.sub(
                allocated_bus_losses.reindex(
                    index=snapshots, columns=ac_buses, fill_value=0.0
                ),
                fill_value=0.0,
            )

        injection_balance_records: list[dict[str, Any]] = []
        if not rtep_n.sub_networks.empty:
            for sub_network in rtep_n.sub_networks.index:
                sn_buses = rtep_n.buses.index[rtep_n.buses.sub_network == sub_network]
                if sn_buses.empty:
                    continue

                sn_injection = net_ac_injection.reindex(
                    columns=sn_buses, fill_value=0.0
                ).copy()
                sn_injection = sn_injection.where(
                    sn_injection.abs() >= RTEP_INNER_INJECTION_ZERO_THRESHOLD, 0.0
                )

                nonzero_mask = sn_injection.ne(0.0)
                active_bus_count = nonzero_mask.sum(axis=1)
                residual = sn_injection.sum(axis=1)
                balancing_share = pd.Series(0.0, index=sn_injection.index, dtype=float)
                has_active_buses = active_bus_count > 0
                balancing_share.loc[has_active_buses] = (
                    residual.loc[has_active_buses]
                    / active_bus_count.loc[has_active_buses]
                )

                sn_injection = sn_injection.where(
                    ~nonzero_mask, sn_injection.sub(balancing_share, axis=0)
                )
                net_ac_injection.loc[:, sn_buses] = sn_injection

                injection_balance_records.append(
                    {
                        "sub_network": sub_network,
                        "bus_count": len(sn_buses),
                        "avg_active_bus_count": float(active_bus_count.mean()),
                        "avg_share": float(balancing_share.mean()),
                        "max_abs_share": float(balancing_share.abs().max()),
                    }
                )
                logger.info(
                    "RTEP inner balanced bus injection for sub-network %s after zeroing |injection| < %.3e (bus_count=%s, avg active buses=%#.6g, avg share=%#.6g, max abs share=%#.6g).",
                    sub_network,
                    RTEP_INNER_INJECTION_ZERO_THRESHOLD,
                    len(sn_buses),
                    float(active_bus_count.mean()),
                    float(balancing_share.mean()),
                    float(balancing_share.abs().max()),
                )

        if RTEP_INNER_EQUIVALENT_CARRIER not in rtep_n.carriers.index:
            rtep_n.add("Carrier", RTEP_INNER_EQUIVALENT_CARRIER)

        load_names = unique_temp_names(
            ac_buses, rtep_n.loads.index, RTEP_INNER_LOAD_SUFFIX
        )

        load_df = pd.DataFrame(
            {
                "bus": ac_buses.to_numpy(),
                "carrier": RTEP_INNER_EQUIVALENT_CARRIER,
            },
            index=load_names,
        )
        load_p_set = (-net_ac_injection).copy()
        load_p_set.columns = load_names
        rtep_n.import_components_from_dataframe(load_df, "Load")
        rtep_n.import_series_from_dataframe(load_p_set, "Load", "p_set")

        rtep_n._rtep_inner_injection_balancing = pd.DataFrame(
            injection_balance_records
        )
        rtep_n._rtep_inner_fixed_branch_losses = fixed_branch_losses
        rtep_n._rtep_inner_losses_accounted_in_bus_injection = bool(
            fixed_branch_losses
        )

        return rtep_n

    def log_inner_injection_balancing(
        network: Network, outer_iteration: int, inner_iteration: int
    ) -> None:
        balancing = getattr(network, "_rtep_inner_injection_balancing", None)
        if not isinstance(balancing, pd.DataFrame) or balancing.empty:
            return

        for record in balancing.itertuples(index=False):
            logger.info(
                "RTEP inner injection balancing %s.%s: sub-network=%s bus_count=%s avg active buses=%#.6g avg share=%#.6g max abs share=%#.6g",
                outer_iteration,
                inner_iteration,
                record.sub_network,
                record.bus_count,
                record.avg_active_bus_count,
                record.avg_share,
                record.max_abs_share,
            )

    def run_rtep_inner(
        base_network: Network,
        outer_caps: pd.Series,
        outer_sssc: pd.Series | None,
        outer_iteration: int,
        outer_msq: float,
    ) -> tuple[pd.Series, pd.Series | None]:
        if ext_branches.empty:
            return outer_caps, outer_sssc

        s_nom_hat = outer_caps.copy()
        sssc_hat = outer_sssc.copy() if outer_sssc is not None else None
        last_status = ("ok", "optimal")
        adaptive_inner_threshold = outer_msq / 5
        logger.info(
            "RTEP inner loop threshold for outer iteration %s set to %.3e.",
            outer_iteration,
            adaptive_inner_threshold,
        )
        outer_branch_losses = collect_outer_branch_losses(base_network)

        for k in range(min(max_iterations, RTEP_INNER_MAX_ITERATIONS)):
            rtep_n = prepare_rtep_network(
                base_network, s_nom_hat, outer_branch_losses
            )
            inner_kwargs = sanitize_inner_solver_kwargs(solve_kwargs)
            inner_kwargs.pop("extra_functionality", None)
            status, condition = rtep_n.optimize(snapshots, **inner_kwargs)
            last_status = (status, condition)
            if status != "ok":
                logger.warning(
                    "RTEP inner optimization failed with status %s/%s in outer iteration %s. "
                    "Using the last outer-loop capacities to start the next outer iteration.",
                    status,
                    condition,
                    outer_iteration,
                )
                return outer_caps.copy(), (
                    outer_sssc.copy() if outer_sssc is not None else None
                )
            log_inner_injection_balancing(rtep_n, outer_iteration, k + 1)

            s_nom_tilde = pd.concat(
                {c: rtep_n.df(c)["s_nom_opt"] for c in branch_components},
                names=["component", "name"],
            )
            if "LineX" in branch_components:
                sssc_hat = rtep_n.line_xs["sssc_nom_opt"].copy()
            err = relative_capacity_change(s_nom_tilde, s_nom_hat)
            logger.info(
                f"RTEP inner iteration {k + 1}: relative capacity change = {err:.3e}"
            )
            report_inner_transmission_budget_usage(
                base_network, s_nom_tilde, outer_iteration, k + 1
            )
            s_nom_hat = s_nom_tilde
            if err <= adaptive_inner_threshold:
                break

        logger.info(
            "RTEP inner loop finished with status %s/%s",
            last_status[0],
            last_status[1],
        )
        return s_nom_hat, sssc_hat

    if link_threshold is None:
        link_threshold = {}

    solve_kwargs = sanitize_solver_kwargs(kwargs)
    if track_iterations:
        for c, attr in pd.Series(nominal_attrs)[list(n.branch_components)].items():
            n.df(c)[f"{attr}_opt_0"] = n.df(c)[f"{attr}"]
        if "LineX" in n.components and not n.line_xs.empty:
            n.line_xs["sssc_nom_opt_0"] = n.line_xs["sssc_nom"]

    current_def = collect_branch_caps("s_nom")
    prev_outer_caps = current_def.copy()
    iteration = 1
    status = "ok"
    condition = "optimal"

    while True:
        if iteration > max_iterations:
            logger.info(
                f"Iteration {iteration} beyond max_iterations {max_iterations}. Stopping ..."
            )
            break

        update_line_params(n, current_def)
        status, condition = n.optimize(snapshots, **solve_kwargs)
        if status != "ok":
            raise RuntimeError(
                f"Optimization failed with status {status} and termination {condition}"
            )

        outer_caps = collect_branch_caps("s_nom_opt")
        outer_sssc = (
            n.line_xs["sssc_nom_opt"].copy() if "LineX" in branch_components else None
        )
        diff = relative_capacity_change(outer_caps, prev_outer_caps)
        logger.info(
            f"Outer iteration {iteration}: relative capacity change = {diff:.3e}"
        )

        if track_iterations:
            save_optimal_capacities(n, iteration, status)

        if diff < outer_msq_threshold and iteration >= min_iterations:
            current_def = outer_caps.copy()
            break

        current_def, _ = run_rtep_inner(
            n, outer_caps, outer_sssc, iteration, diff
        )

        prev_outer_caps = outer_caps.copy()
        iteration += 1

    if hasattr(n, "model"):
        logger.info(
            "Deleting model instance `n.model` from previous run to reclaim memory."
        )
        del n.model
        gc.collect()

    logger.info(
        "Preparing final iteration with updated transmission parameters and extendable transmission capacities."
    )
    last_success_status, last_success_condition = status, condition

    update_line_params(n, current_def)

    n.calculate_dependent_values()
    status, condition = n.optimize(snapshots, **solve_kwargs)

    if status == "ok":
        return status, condition
    else:
        logger.warning(
            "Final rerun with updated transmission parameters failed with status %s/%s. "
            "Keeping the last successful outer-loop solution.",
            status,
            condition,
        )
        status, condition = last_success_status, last_success_condition

    return status, condition


def optimize_security_constrained(
    n: Network,
    snapshots: Sequence | None = None,
    branch_outages: Sequence | pd.Index | pd.MultiIndex | None = None,
    multi_investment_periods: bool = False,
    model_kwargs: dict = {},
    **kwargs: Any,
) -> tuple[str, str]:
    """
    Computes Security-Constrained Linear Optimal Power Flow (SCLOPF).

    This ensures that no branch is overloaded even given the branch outages.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : list-like, optional
        Set of snapshots to consider in the optimization. The default is None.
    branch_outages : list-like/pandas.Index/pandas.MultiIndex, optional
        Subset of passive branches to consider as possible outages. If a list
        or a pandas.Index is passed, it is assumed to identify lines. If a
        multiindex is passed, its first level has to contain the component names,
        the second the assets. The default None results in all passive branches
        to be considered.
    multi_investment_periods : bool, default False
        Whether to optimise as a single investment period or to optimise in multiple
        investment periods. Then, snapshots should be a ``pd.MultiIndex``.
    model_kwargs: dict
        Keyword arguments used by `linopy.Model`, such as `solver_dir` or `chunk`.
    **kwargs:
        Keyword argument used by `linopy.Model.solve`, such as `solver_name`,
        `problem_fn` or solver options directly passed to the solver.

    Returns
    -------
    None
    """
    all_passive_branches = n.passive_branches().index

    if branch_outages is None:
        branch_outages = all_passive_branches
    elif isinstance(branch_outages, (list, pd.Index)):
        branch_outages = pd.MultiIndex.from_product([("Line",), branch_outages])

        if diff := set(branch_outages) - set(all_passive_branches):
            raise ValueError(
                f"The following passive branches are not in the network: {diff}"
            )

    if not len(all_passive_branches):
        return n.optimize(
            snapshots,  # type: ignore
            multi_investment_periods=multi_investment_periods,
            model_kwargs=model_kwargs,
            **kwargs,
        )

    m = n.optimize.create_model(
        snapshots=snapshots,
        multi_investment_periods=multi_investment_periods,
        **model_kwargs,
    )

    for sn in n.sub_networks.obj:
        branches_i = sn.branches_i()
        outages = branches_i.intersection(branch_outages)

        if outages.empty:
            continue

        sn.calculate_BODF()
        BODF = pd.DataFrame(sn.BODF, index=branches_i, columns=branches_i)[outages]

        for c_outage, c_affected in product(outages.unique(0), branches_i.unique(0)):
            c_outage_ = c_outage + "-outage"
            c_outages = outages.get_loc_level(c_outage)[1]
            flow_outage = m.variables[c_outage + "-s"].loc[:, c_outages]
            flow_outage = flow_outage.rename({c_outage: c_outage_})

            bodf = BODF.loc[c_affected, c_outage]
            bodf = xr.DataArray(bodf, dims=[c_affected, c_outage_])
            additional_flow = flow_outage * bodf
            for bound, kind in product(("lower", "upper"), ("fix", "ext")):
                coord = c_affected + "-" + kind
                constraint = coord + "-s-" + bound
                if constraint not in m.constraints:
                    continue
                rename = {c_affected: coord}
                added_flow = additional_flow.rename(rename)
                con = m.constraints[constraint]  # use this as a template
                # idx now contains fixed/extendable for the subnetwork
                idx = con.lhs.indexes[coord].intersection(added_flow.indexes[coord])
                sel = {coord: idx}
                lhs = con.lhs.sel(sel) + added_flow.sel(sel)
                name = constraint + f"-security-for-{c_outage_}-in-{sn}"
                m.add_constraints(lhs, con.sign.sel(sel), con.rhs.sel(sel), name=name)

    return n.optimize.solve_model(**kwargs)


def optimize_with_rolling_horizon(
    n: Network,
    snapshots: Sequence | None = None,
    horizon: int = 100,
    overlap: int = 0,
    **kwargs: Any,
) -> Network:
    """
    Optimizes the network in a rolling horizon fashion.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : list-like
        Set of snapshots to consider in the optimization. The default is None.
    horizon : int
        Number of snapshots to consider in each iteration. Defaults to 100.
    overlap : int
        Number of snapshots to overlap between two iterations. Defaults to 0.
    **kwargs:
        Keyword argument used by `linopy.Model.solve`, such as `solver_name`,

    Returns
    -------
    None
    """
    if snapshots is None:
        snapshots = n.snapshots

    if horizon <= overlap:
        raise ValueError("overlap must be smaller than horizon")

    starting_points = range(0, len(snapshots), horizon - overlap)
    for i, start in enumerate(starting_points):
        end = min(len(snapshots), start + horizon)
        sns = snapshots[start:end]
        logger.info(
            f"Optimizing network for snapshot horizon [{sns[0]}:{sns[-1]}] ({i+1}/{len(starting_points)})."
        )

        if i:
            if not n.stores.empty:
                n.stores.e_initial = n.stores_t.e.loc[snapshots[start - 1]]
            if not n.storage_units.empty:
                n.storage_units.state_of_charge_initial = (
                    n.storage_units_t.state_of_charge.loc[snapshots[start - 1]]
                )

        status, condition = n.optimize(sns, **kwargs)  # type: ignore
        if status != "ok":
            logger.warning(
                f"Optimization failed with status {status} and condition {condition}"
            )
    return n


def optimize_mga(
    n: Network,
    snapshots: Sequence | None = None,
    multi_investment_periods: bool = False,
    weights: dict | None = None,
    sense: str | int = "min",
    slack: float = 0.05,
    model_kwargs: dict = {},
    **kwargs: Any,
) -> tuple[str, str]:
    """
    Run modelling-to-generate-alternatives (MGA) on network to find near-
    optimal solutions.

    Parameters
    ----------
    n : pypsa.Network snapshots : list-like
        Set of snapshots to consider in the optimization. The default is None.
    multi_investment_periods : bool, default False
        Whether to optimise as a single investment period or to optimize in
        multiple investment periods. Then, snapshots should be a
        ``pd.MultiIndex``.
    weights : dict-like
        Weights for alternate objective function. The default is None, which
        minimizes generation capacity. The weights dictionary should be keyed
        with the component and variable (see ``pypsa/variables.csv``), followed
        by a float, dict, pd.Series or pd.DataFrame for the coefficients of the
        objective function. Examples:

        >>> {"Generator": {"p_nom": 1}}
        >>> {"Generator": {"p_nom": pd.Series(1, index=n.generators.index)}}
        >>> {"Generator": {"p_nom": {"gas": 1, "coal": 2}}}
        >>> {"Generator": {"p": pd.Series(1, index=n.generators.index)}
        >>> {"Generator": {"p": pd.DataFrame(1, columns=n.generators.index, index=n.snapshots)}

        Weights for non-extendable components are ignored. The dictionary does
        not need to provide weights for all extendable components.
    sense : str|int
        Optimization sense of alternate objective function. Defaults to 'min'.
        Can also be 'max'.
    slack : float
        Cost slack for budget constraint. Defaults to 0.05.
    model_kwargs: dict
        Keyword arguments used by `linopy.Model`, such as `solver_dir` or
        `chunk`.
    **kwargs:
        Keyword argument used by `linopy.Model.solve`, such as `solver_name`,

    Returns
    -------
    status : str
        The status of the optimization, either "ok" or one of the codes listed
        in https://linopy.readthedocs.io/en/latest/generated/linopy.constants.SolverStatus.html
    condition : str
        The termination condition of the optimization, either
        "optimal" or one of the codes listed in
        https://linopy.readthedocs.io/en/latest/generated/linopy.constants.TerminationCondition.html
    """
    if snapshots is None:
        snapshots = n.snapshots

    if weights is None:
        weights = dict(Generator=dict(p_nom=pd.Series(1, index=n.generators.index)))

    # check that network has been solved
    if not hasattr(n, "objective"):
        msg = "Network needs to be solved with `n.optimize()` before running MGA."
        raise ValueError(msg)

    # create basic model
    m = n.optimize.create_model(
        snapshots=snapshots,
        multi_investment_periods=multi_investment_periods,
        **model_kwargs,
    )

    # build budget constraint
    if not multi_investment_periods:
        optimal_cost = n.statistics.capex().sum() + n.statistics.opex().sum()
        fixed_cost = n.statistics.installed_capex().sum()
    else:
        w = n.investment_period_weightings.objective
        optimal_cost = (
            n.statistics.capex().sum() * w + n.statistics.opex().sum() * w
        ).sum()
        fixed_cost = (n.statistics.installed_capex().sum() * w).sum()

    objective = m.objective
    if not isinstance(objective, (LinearExpression, QuadraticExpression)):
        objective = objective.expression

    if getattr(n, "_objective_constant_missing_from_expression", False):
        fixed_cost = 0.0

    m.add_constraints(
        objective + fixed_cost <= (1 + slack) * optimal_cost, name="budget"
    )

    # parse optimization sense
    if (
        isinstance(sense, str)
        and sense.startswith("min")
        or isinstance(sense, int)
        and sense > 0
    ):
        sense = 1
    elif (
        isinstance(sense, str)
        and sense.startswith("max")
        or isinstance(sense, int)
        and sense < 0
    ):
        sense = -1
    else:
        raise ValueError(f"Could not parse optimization sense {sense}")

    # build alternate objective
    objective = []
    for c, attrs in weights.items():
        for attr, coeffs in attrs.items():
            if isinstance(coeffs, dict):
                coeffs = pd.Series(coeffs)
            if attr == nominal_attrs[c] and isinstance(coeffs, pd.Series):
                coeffs = coeffs.reindex(n.get_extendable_i(c))
                coeffs.index.name = ""
            elif isinstance(coeffs, pd.Series):
                coeffs = coeffs.reindex(columns=n.df(c).index)
            elif isinstance(coeffs, pd.DataFrame):
                coeffs = coeffs.reindex(columns=n.df(c).index, index=snapshots)
            objective.append(m[f"{c}-{attr}"] * coeffs * sense)

    m.objective = merge(objective)

    status, condition = n.optimize.solve_model(**kwargs)

    # write MGA coefficients into metadata
    n.meta["slack"] = slack
    n.meta["sense"] = sense

    def convert_to_dict(obj: Any) -> Any:
        if isinstance(obj, pd.DataFrame):
            return obj.to_dict(orient="list")
        elif isinstance(obj, pd.Series):
            return obj.to_dict()
        elif isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items()}
        else:
            return obj

    n.meta["weights"] = convert_to_dict(weights)

    return status, condition
