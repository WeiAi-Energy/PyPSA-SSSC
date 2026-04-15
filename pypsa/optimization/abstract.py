#!/usr/bin/env python3
"""
Build abstracted, extended optimisation problems from PyPSA networks with
Linopy.
"""

from __future__ import annotations

import copy
import gc
import logging
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


def optimize_transmission_expansion_iteratively(
    n: Network,
    snapshots: Sequence | None = None,
    msq_threshold: float = 0.001,
    min_iterations: int = 1,
    max_iterations: int = 100,
    track_iterations: bool = False,
    relaxation_factor: float = 1.0,
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
    msq_threshold: float, default 0.03
        Maximal relative fixed-point residual between the current defining
        transmission capacities given to the network and the optimized
        transmission capacities returned by the current iteration. As
        soon as this threshold is undercut, and the number of iterations is
        bigger than 'min_iterations', the iterative optimization stops.
    min_iterations : integer, default 1
        Minimal number of iteration to run regardless whether the msq_threshold
        is already undercut
    max_iterations : integer, default 100
        Maximal number of iterations to run regardless whether msq_threshold
        is already undercut
    track_iterations: bool, default False
        If True, the intermediate branch capacities and values of the
        objective function are recorded for each iteration. The values of
        iteration 0 represent the initial state.
    relaxation_factor : float, default 1.0
        Convex relaxation factor applied to the next iterate. A value of
        ``1.0`` keeps the current behavior, values in ``[0, 1)`` blend the
        new iterate with the previous one, and values greater than ``1.0``
        apply over-relaxation. Relaxation is applied from the second
        update onward.
    **kwargs
        Keyword arguments of the `n.optimize` function which runs at each iteration
    """
    if snapshots is None:
        snapshots = n.snapshots
    snapshots = as_index(n, snapshots, "snapshots", "snapshot")

    branch_components = [c for c in ("Line", "LineX") if c in n.components and not n.df(c).empty]
    if not branch_components:
        status, condition = n.optimize(snapshots, **kwargs)
        return status, condition

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

    ext_branches = pd.MultiIndex.from_tuples(
        [(c, i) for c in branch_components for i in branch_data[c]["ext_i"]],
        names=["component", "name"],
    )
    relaxation_factor = float(relaxation_factor)
    if not np.isfinite(relaxation_factor) or relaxation_factor < 0.0:
        raise ValueError("relaxation_factor must be finite and >= 0.")

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

    def collect_branch_bounds(network: Network) -> tuple[pd.Series, pd.Series]:
        lower = {}
        upper = {}
        for c in branch_components:
            df = network.df(c)
            lower_bounds = pd.Series(0.0, index=df.index, dtype=float)
            upper_bounds = pd.Series(np.inf, index=df.index, dtype=float)
            ext_i = branch_data[c]["ext_i"]
            if f"{nominal_attrs[c]}_min" in df:
                lower_bounds.loc[ext_i] = (
                    df[f"{nominal_attrs[c]}_min"].reindex(ext_i).fillna(0.0)
                )
            if f"{nominal_attrs[c]}_max" in df:
                upper_bounds.loc[ext_i] = (
                    df[f"{nominal_attrs[c]}_max"].reindex(ext_i).fillna(np.inf)
                )
            fixed_i = df.index.difference(ext_i)
            if not fixed_i.empty:
                nominal = df[nominal_attrs[c]].reindex(fixed_i)
                lower_bounds.loc[fixed_i] = nominal
                upper_bounds.loc[fixed_i] = nominal
            lower[c] = lower_bounds
            upper[c] = upper_bounds
        return (
            pd.concat(lower, names=["component", "name"]),
            pd.concat(upper, names=["component", "name"]),
        )

    def clip_branch_caps(
        caps: pd.Series, lower: pd.Series, upper: pd.Series
    ) -> pd.Series:
        clipped = caps.copy()
        lower_aligned = lower.reindex(clipped.index).fillna(-np.inf)
        upper_aligned = upper.reindex(clipped.index).fillna(np.inf)
        return clipped.clip(lower=lower_aligned, upper=upper_aligned)

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

    def relative_capacity_change(
        current: pd.Series, previous: pd.Series, initial: pd.Series
    ) -> float:
        if ext_branches.empty:
            return 0.0
        denom = np.linalg.norm(initial.loc[ext_branches].to_numpy())
        denom = max(denom, 1e-12)
        return float(
            np.linalg.norm(
                (current - previous).loc[ext_branches].to_numpy()
            )
            / denom
        )

    def relax_iterate(current: pd.Series, target: pd.Series) -> pd.Series:
        if relaxation_factor == 1.0:
            return target.copy()
        aligned_target = target.reindex(current.index)
        return current + relaxation_factor * (aligned_target - current)

    if relaxation_factor != 1.0:
        logger.info(
            "Applying fixed-point relaxation factor %.3f.",
            relaxation_factor,
        )

    if track_iterations:
        for c, attr in pd.Series(nominal_attrs)[list(n.branch_components)].items():
            n.df(c)[f"{attr}_opt_0"] = n.df(c)[f"{attr}"]
        if "LineX" in n.components and not n.line_xs.empty:
            n.line_xs["sssc_nom_opt_0"] = n.line_xs["sssc_nom"]

    branch_cap_min, branch_cap_max = collect_branch_bounds(n)
    current_def = collect_branch_caps("s_nom")
    initial_caps = current_def.copy()
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
        status, condition = n.optimize(snapshots, **kwargs)
        if status != "ok":
            raise RuntimeError(
                f"Optimization failed with status {status} and termination {condition}"
            )

        optimized_caps = collect_branch_caps("s_nom_opt")
        diff = relative_capacity_change(optimized_caps, current_def, initial_caps)
        logger.info("Iteration %s: relative fixed-point residual = %.3e", iteration, diff)

        if track_iterations:
            save_optimal_capacities(n, iteration, status)

        if diff < msq_threshold and iteration >= min_iterations:
            current_def = clip_branch_caps(
                optimized_caps.copy(), branch_cap_min, branch_cap_max
            )
            break

        next_def = optimized_caps.copy()
        if iteration == 1:
            current_def = next_def
        else:
            current_def = relax_iterate(current_def, next_def)
            current_def = clip_branch_caps(
                current_def, branch_cap_min, branch_cap_max
            )
        logger.info("Iteration %s: using relaxation step.", iteration)

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
    status, condition = n.optimize(snapshots, **kwargs)

    if status == "ok":
        return status, condition
    else:
        logger.warning(
            "Final rerun with updated transmission parameters failed with status %s/%s. "
            "Keeping the last successful iterative solution.",
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
