#!/usr/bin/env python3
"""
Build optimisation problems from PyPSA networks with Linopy.
"""

from __future__ import annotations

import contextlib
import logging
import os
from collections.abc import Callable, Sequence
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
import xarray as xr
from linopy import Model, merge
from linopy.constants import Status
from linopy.solvers import available_solvers
from scipy.sparse.linalg import spsolve

from pypsa.descriptors import additional_linkports, get_committable_i, nominal_attrs
from pypsa.descriptors import get_switchable_as_dense as get_as_dense
from pypsa.optimization.abstract import (
    optimize_mga,
    optimize_security_constrained,
    optimize_transmission_expansion_iteratively,
    optimize_with_rolling_horizon,
)
from pypsa.optimization.common import (
    get_strongly_meshed_buses,
    set_from_frame,
)
from pypsa.optimization.constraints import (
    define_fixed_nominal_constraints,
    define_fixed_operation_constraints,
    define_kirchhoff_voltage_constraints,
    define_line_x_sssc_constraints,
    define_loss_constraints,
    define_modular_constraints,
    define_nodal_balance_constraints,
    define_nominal_constraints_for_extendables,
    define_operational_constraints_for_committables,
    define_operational_constraints_for_extendables,
    define_operational_constraints_for_non_extendables,
    define_ramp_limit_constraints,
    define_storage_unit_constraints,
    define_store_constraints,
)
from pypsa.optimization.global_constraints import (
    define_growth_limit,
    define_nominal_constraints_per_bus_carrier,
    define_operational_limit,
    define_primary_energy_limit,
    define_tech_capacity_expansion_limit,
    define_transmission_expansion_cost_limit,
    define_transmission_volume_expansion_limit,
)
from pypsa.optimization.variables import (
    define_line_x_variables,
    define_loss_variables,
    define_modular_variables,
    define_nominal_variables,
    define_operational_variables,
    define_shut_down_variables,
    define_spillage_variables,
    define_start_up_variables,
    define_status_variables,
)
from pypsa.utils import as_index

if TYPE_CHECKING:
    from pypsa import Network, SubNetwork
logger = logging.getLogger(__name__)


lookup = pd.read_csv(
    os.path.join(os.path.dirname(__file__), "..", "variables.csv"),
    index_col=["component", "variable"],
)


def define_objective(
    n: Network, sns: pd.Index, include_objective_constant: bool = False
) -> None:
    """
    Defines and writes out the objective function.
    """
    m = n.model
    objective = []
    is_quadratic = False

    if n._multi_invest:
        periods = sns.unique("period")
        period_weighting = n.investment_period_weightings.objective[periods]

    # constant for already done investment
    nom_attr = nominal_attrs.items()
    constant = 0
    for c, attr in nom_attr:
        ext_i = n.get_extendable_i(c)
        cost = n.df(c)["capital_cost"][ext_i]
        if cost.empty:
            continue

        if n._multi_invest:
            active = pd.concat(
                {
                    period: n.get_active_assets(c, period)[ext_i]
                    for period in sns.unique("period")
                },
                axis=1,
            )
            cost = active @ period_weighting * cost

        constant += (cost * n.df(c)[attr][ext_i]).sum()

    if "LineX" in n.components and not n.df("LineX").empty:
        ext_i = n.df("LineX").index[n.df("LineX").sssc_nom_extendable].rename(
            "LineX-sssc-ext"
        )
        cost = n.df("LineX")["capital_cost_sssc"].reindex(ext_i)
        if not cost.empty:
            if n._multi_invest:
                active = pd.concat(
                    {
                        period: n.get_active_assets("LineX", period)[ext_i]
                        for period in sns.unique("period")
                    },
                    axis=1,
                )
                cost = active @ period_weighting * cost
            constant += (cost * n.df("LineX")["sssc_nom"].reindex(ext_i)).sum()

    n.objective_constant = constant
    n._objective_constant_missing_from_expression = False

    if constant != 0 and include_objective_constant:
        object_const = m.add_variables(constant, constant, name="objective_constant")
        objective.append(-1 * object_const)

    # Weightings
    weighting = n.snapshot_weightings.objective
    if n._multi_invest:
        weighting = weighting.mul(period_weighting, level=0).loc[sns]
    else:
        weighting = weighting.loc[sns]

    # marginal costs, marginal storage cost, and spill cost
    for cost_type in ["marginal_cost", "marginal_cost_storage", "spill_cost"]:
        for c, attr in lookup.query(cost_type).index:
            cost = (
                get_as_dense(n, c, cost_type, sns)
                .loc[:, lambda ds: (ds != 0).any()]
                .mul(weighting, axis=0)
            )
            if cost.empty:
                continue
            operation = m[f"{c}-{attr}"].sel({"snapshot": sns, c: cost.columns})
            objective.append((operation * cost).sum())

    # marginal cost quadratic
    for c, attr in lookup.query("marginal_cost").index:
        if "marginal_cost_quadratic" in n.df(c):
            cost = (
                get_as_dense(n, c, "marginal_cost_quadratic", sns)
                .loc[:, lambda ds: (ds != 0).any()]
                .mul(weighting, axis=0)
            )
            if cost.empty:
                continue
            operation = m[f"{c}-{attr}"].sel({"snapshot": sns, c: cost.columns})
            objective.append((operation * operation * cost).sum())
            is_quadratic = True

    # stand-by cost
    comps = {"Generator", "Link"}
    for c in comps:
        com_i = get_committable_i(n, c)

        if com_i.empty:
            continue

        stand_by_cost = (
            get_as_dense(n, c, "stand_by_cost", sns, com_i)
            .loc[:, lambda ds: (ds != 0).any()]
            .mul(weighting, axis=0)
        )
        stand_by_cost.columns.name = f"{c}-com"
        status = n.model.variables[f"{c}-status"].loc[:, stand_by_cost.columns]
        objective.append((status * stand_by_cost).sum())

    # investment
    for c, attr in nominal_attrs.items():
        ext_i = n.get_extendable_i(c)
        cost = n.df(c)["capital_cost"][ext_i]
        if cost.empty:
            continue

        if n._multi_invest:
            active = pd.concat(
                {
                    period: n.get_active_assets(c, period)[ext_i]
                    for period in sns.unique("period")
                },
                axis=1,
            )
            cost = active @ period_weighting * cost

        caps = m[f"{c}-{attr}"]
        objective.append((caps * cost).sum())

    if (
        "LineX" in n.components
        and not n.df("LineX").empty
        and "LineX-sssc_nom" in m.variables
    ):
        ext_i = n.df("LineX").index[n.df("LineX").sssc_nom_extendable].rename(
            "LineX-sssc-ext"
        )
        cost = n.df("LineX")["capital_cost_sssc"].reindex(ext_i)
        if n._multi_invest:
            active = pd.concat(
                {
                    period: n.get_active_assets("LineX", period)[ext_i]
                    for period in sns.unique("period")
                },
                axis=1,
            )
            cost = active @ period_weighting * cost

        objective.append((m["LineX-sssc_nom"] * cost).sum())

    # unit commitment
    keys = ["start_up", "shut_down"]  # noqa: F841
    for c, attr in lookup.query("variable in @keys").index:
        com_i = n.get_committable_i(c)
        cost = n.df(c)[attr + "_cost"].reindex(com_i)

        if cost.sum():
            var = m[f"{c}-{attr}"]
            objective.append((var * cost).sum())

    if not len(objective):
        raise ValueError(
            "Objective function could not be created. "
            "Please make sure the components have assigned costs."
        )

    objective_expression = sum(objective) if is_quadratic else merge(objective)

    if constant != 0 and not include_objective_constant:
        n._objective_constant_missing_from_expression = True

    m.objective = objective_expression


def create_model(
    n: Network,
    snapshots: Sequence | None = None,
    multi_investment_periods: bool = False,
    transmission_losses: int = 0,
    linearized_unit_commitment: bool = False,
    include_objective_constant: bool = False,
    **kwargs: Any,
) -> Model:
    """
    Create a linopy.Model instance from a pypsa network.

    The model is stored at `n.model`.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        network.snapshots, defaults to network.snapshots
    multi_investment_periods : bool, default False
        Whether to optimise as a single investment period or to optimize in multiple
        investment periods. Then, snapshots should be a ``pd.MultiIndex``.
    transmission_losses : int, default 0
        Number of least-squares segments per half-axis used for the piecewise
        linear approximation of the branch loss parabola. Defaults to 0, which
        ignores losses.
    linearized_unit_commitment : bool, default False
        Whether to optimise using the linearised unit commitment formulation or not.
    include_objective_constant : bool, default False
        Whether to include the objective constant for existing extendable assets
        via a helper variable in the model objective.
    **kwargs:
        Keyword arguments used by `linopy.Model()`, such as `solver_dir` or `chunk`.

    Returns
    -------
    linopy.model
    """
    sns = as_index(n, snapshots, "snapshots", "snapshot")
    n._linearized_uc = int(linearized_unit_commitment)
    n._multi_invest = int(multi_investment_periods)
    n.consistency_check()

    kwargs.setdefault("force_dim_names", True)
    n.model = Model(**kwargs)
    n.model.parameters = n.model.parameters.assign(snapshots=sns)
    n._global_constraint_scales = {}

    # Define variables
    for c, attr in lookup.query("nominal").index:
        define_nominal_variables(n, c, attr)
        define_modular_variables(n, c, attr)

    for c, attr in lookup.query("not nominal and not handle_separately").index:
        define_operational_variables(n, sns, c, attr)
        define_status_variables(n, sns, c)
        define_start_up_variables(n, sns, c)
        define_shut_down_variables(n, sns, c)

    define_spillage_variables(n, sns)
    define_operational_variables(n, sns, "Store", "p")
    define_line_x_variables(n, sns)

    if transmission_losses:
        for c in n.passive_branch_components:
            define_loss_variables(n, sns, c)

    # Define constraints
    for c, attr in lookup.query("nominal").index:
        define_nominal_constraints_for_extendables(n, c, attr)
        define_fixed_nominal_constraints(n, c, attr)
        define_modular_constraints(n, c, attr)

    for c, attr in lookup.query("not nominal and not handle_separately").index:
        define_operational_constraints_for_non_extendables(
            n, sns, c, attr, transmission_losses
        )
        define_operational_constraints_for_extendables(
            n, sns, c, attr, transmission_losses
        )
        define_operational_constraints_for_committables(n, sns, c)
        define_ramp_limit_constraints(n, sns, c, attr)
        define_fixed_operation_constraints(n, sns, c, attr)

    meshed_buses = get_strongly_meshed_buses(n)
    weakly_meshed_buses = n.buses.index.difference(meshed_buses)
    if not meshed_buses.empty and not weakly_meshed_buses.empty:
        # Write constraint for buses many terms and for buses with a few terms
        # separately. This reduces memory usage for large networks.
        define_nodal_balance_constraints(
            n, sns, transmission_losses=transmission_losses, buses=weakly_meshed_buses
        )
        define_nodal_balance_constraints(
            n,
            sns,
            transmission_losses=transmission_losses,
            buses=meshed_buses,
            suffix="-meshed",
        )
    else:
        define_nodal_balance_constraints(
            n, sns, transmission_losses=transmission_losses
        )

    define_kirchhoff_voltage_constraints(n, sns)
    define_line_x_sssc_constraints(n, sns)
    define_storage_unit_constraints(n, sns)
    define_store_constraints(n, sns)

    if transmission_losses:
        for c in n.passive_branch_components:
            define_loss_constraints(n, sns, c, transmission_losses)

    # Define global constraints
    define_primary_energy_limit(n, sns)
    define_transmission_expansion_cost_limit(n, sns)
    define_transmission_volume_expansion_limit(n, sns)
    define_tech_capacity_expansion_limit(n, sns)
    define_operational_limit(n, sns)
    define_nominal_constraints_per_bus_carrier(n, sns)
    define_growth_limit(n, sns)

    define_objective(n, sns, include_objective_constant)

    return n.model


def assign_solution(n: Network) -> None:
    """
    Map solution to network components.
    """
    m = n.model
    sns = n.model.parameters.snapshots.to_index()

    for name, variable in m.variables.items():
        sol = variable.solution
        if name == "objective_constant":
            continue

        try:
            c, attr = name.split("-", 1)
        except ValueError:
            continue
        # Extra-functionality callbacks may add auxiliary variables whose
        # names are not component attributes. They belong to the optimization
        # model, but there is no network component table to which their
        # solution should be assigned.
        if c not in n.components:
            continue
        df = sol.to_pandas()

        if "snapshot" in sol.dims:
            if c in n.passive_branch_components and attr == "s":
                set_from_frame(n, c, "p0", df)
                set_from_frame(n, c, "p1", -df)

            elif c == "Link" and attr == "p":
                set_from_frame(n, c, "p0", df)

                for i in ["1"] + additional_linkports(n):
                    i_eff = "" if i == "1" else i
                    eff = get_as_dense(n, "Link", f"efficiency{i_eff}", sns)
                    set_from_frame(n, c, f"p{i}", -df * eff)
                    n.pnl(c)[f"p{i}"].loc[
                        sns, n.links.index[n.links[f"bus{i}"] == ""]
                    ] = float(n.components["Link"]["attrs"].loc[f"p{i}", "default"])

            else:
                set_from_frame(n, c, attr, df)
        elif attr != "n_mod" and hasattr(df, "index"):
            idx = df.index.intersection(n.df(c).index)
            n.df(c).loc[idx, attr + "_opt"] = df.loc[idx]

    # if nominal capacity was no variable set optimal value to nominal
    for c, attr in lookup.query("nominal").index:
        fix_i = n.get_non_extendable_i(c)
        if not fix_i.empty:
            n.df(c).loc[fix_i, f"{attr}_opt"] = n.df(c).loc[fix_i, attr]

    if "LineX" in n.components and not n.df("LineX").empty:
        fix_i = n.df("LineX").index[~n.df("LineX").sssc_nom_extendable]
        if not fix_i.empty:
            n.df("LineX").loc[fix_i, "sssc_nom_opt"] = n.df("LineX").loc[
                fix_i, "sssc_nom"
            ]

    # recalculate storageunit net dispatch
    if not n.df("StorageUnit").empty:
        c = "StorageUnit"
        n.pnl(c)["p"] = n.pnl(c)["p_dispatch"] - n.pnl(c)["p_store"]

    n.objective = m.objective.value
    if getattr(n, "_objective_constant_missing_from_expression", False):
        n.objective -= getattr(n, "objective_constant", 0.0)


def assign_duals(n: Network, assign_all_duals: bool = False) -> None:
    """
    Map dual values i.e. shadow prices to network components.

    Parameters
    ----------
    n : pypsa.Network
    assign_all_duals : bool, default False
        Whether to assign all dual values or only those that already
        have a designated place in the network.
    """
    m = n.model
    unassigned = []
    if all("dual" not in constraint for _, constraint in m.constraints.items()):
        logger.info("No shadow prices were assigned to the network.")
        return

    for name, constraint in m.constraints.items():
        dual = constraint.dual
        try:
            c, attr = name.split("-", 1)
        except ValueError:
            unassigned.append(name)
            continue

        if "snapshot" in dual.dims:
            try:
                df = dual.transpose("snapshot", ...).to_pandas()

                try:
                    spec = attr.rsplit("-", 1)[-1]
                except ValueError:
                    spec = attr

                if attr.endswith("nodal_balance"):
                    set_from_frame(n, c, "marginal_price", df)
                elif assign_all_duals or f"mu_{spec}" in n.df(c):
                    set_from_frame(n, c, "mu_" + spec, df)
                else:
                    unassigned.append(name)

            except:  # noqa: E722 # TODO: specify exception
                unassigned.append(name)

        elif (c == "GlobalConstraint") and (assign_all_duals or attr in n.df(c).index):
            scale = getattr(n, "_global_constraint_scales", {}).get(attr, 1.0)
            n.df(c).loc[attr, "mu"] = dual * scale

    # if unassigned:
    #     logger.info(
    #         f"The shadow-prices of the constraints {', '.join(unassigned)} were "
    #         "not assigned to the network."
    #     )


def post_processing(n: Network) -> None:
    """
    Post-process the optimized network.

    This calculates quantities derived from the optimized values such as
    power injection per bus and snapshot, voltage angle.
    """
    sns = n.model.parameters.snapshots.to_index()

    # correct prices with objective weightings
    if n._multi_invest:
        period_weighting = n.investment_period_weightings.objective
        weightings = n.snapshot_weightings.objective.mul(
            period_weighting, level=0, axis=0
        ).loc[sns]
    else:
        weightings = n.snapshot_weightings.objective.loc[sns]

    n.buses_t.marginal_price.loc[sns] = n.buses_t.marginal_price.loc[sns].divide(
        weightings, axis=0
    )

    # load
    if len(n.loads):
        set_from_frame(n, "Load", "p", get_as_dense(n, "Load", "p_set", sns))

    # passive branch losses
    for c in n.passive_branch_components:
        name = f"{c}-loss"
        if name not in n.model.variables:
            continue
        losses = n.model[name].solution.to_pandas()
        n.pnl(c).p0 += losses / 2
        n.pnl(c).p1 += losses / 2

    # recalculate injection
    ca = [
        ("Generator", "p", "bus"),
        ("Store", "p", "bus"),
        ("Load", "p", "bus"),
        ("StorageUnit", "p", "bus"),
        ("Link", "p0", "bus0"),
        ("Link", "p1", "bus1"),
    ]
    for i in additional_linkports(n):
        ca.append(("Link", f"p{i}", f"bus{i}"))

    def sign(c: str) -> int:
        return n.df(c).sign if "sign" in n.df(c) else -1  # sign for 'Link'

    n.buses_t.p = (
        pd.concat(
            [
                n.pnl(c)[attr].mul(sign(c)).rename(columns=n.df(c)[group])
                for c, attr, group in ca
            ],
            axis=1,
        )
        .T.groupby(level=0)
        .sum()
        .T.reindex(columns=n.buses.index, fill_value=0.0)
    )

    def v_ang_for_(sub: SubNetwork) -> pd.DataFrame:
        buses_i = sub.buses_o
        if len(buses_i) == 1:
            return pd.DataFrame(0, index=sns, columns=buses_i)
        sub.calculate_B_H(skip_pre=True)
        # buses_i[0] is always the slack bus (see `calculate_control_shift`), so
        # solving the slack-reduced sparse system avoids ever densifying `sub.B`
        # (a dense pinv of an (n_buses, n_buses) matrix is infeasible for large
        # networks, e.g. 9+ GiB and an O(n^3) SVD for 35k buses).
        p = n.buses_t.p.reindex(columns=buses_i).to_numpy()
        v_ang = np.zeros((len(sns), len(buses_i)))
        v_ang[:, 1:] = spsolve(sub.B[1:, 1:], p[:, 1:].T).T
        return pd.DataFrame(v_ang, sns, buses_i)

    # TODO: if multi investment optimization, the network topology is not the necessarily the same,
    # i.e. one has to iterate over the periods in order to get the correct angles.
    # Determine_network_topology is not necessarily called (only if KVL was assigned)
    if "obj" in n.sub_networks:
        n.buses_t.v_ang = pd.concat(
            [v_ang_for_(sub) for sub in n.sub_networks.obj], axis=1
        ).reindex(columns=n.buses.index, fill_value=0.0)


_GUROBI_TERMINATION_CONDITIONS = {
    1: "unknown",
    2: "optimal",
    3: "infeasible",
    4: "infeasible_or_unbounded",
    5: "unbounded",
    6: "other",
    7: "iteration_limit",
    8: "terminated_by_limit",
    9: "time_limit",
    10: "optimal",
    11: "user_interrupt",
    12: "other",
    13: "suboptimal",
    14: "unknown",
    15: "terminated_by_limit",
    16: "internal_solver_error",
    17: "internal_solver_error",
}


def _int_index_from_names(named_values: dict[str, float]) -> pd.Series:
    """
    Map Gurobi variable/constraint names (``x0``, ``x1``, ... or ``c0``, ...)
    back to the integer Linopy label they were generated from.

    Mirrors ``linopy.common.set_int_index`` without importing it directly:
    that helper is internal and its module layout has changed across Linopy
    releases, whereas the plain (name, value) pairs read from Gurobi via
    ``getVars``/``getConstrs`` are stable.
    """
    series = pd.Series(named_values, dtype=float)
    if series.empty or pd.api.types.is_integer_dtype(series.index):
        return series
    cutoff = sum(1 for ch in str(series.index[0]) if ch.isalpha())
    try:
        series.index = series.index.str[cutoff:].astype(int)
    except ValueError:
        series.index = series.index.str.replace(".*#", "", regex=True).astype(int)
    return series


def _solve_model_with_selected_variables(
    n: Network,
    m: Model,
    selected_variables: Sequence[str],
    solver_name: str,
    solver_options: dict[str, Any],
    **kwargs: Any,
) -> tuple[str, str]:
    """
    Solve a Gurobi model and read back only selected primal variable groups.

    This is an internal fast path for outer algorithms which need a small
    subset of the primal solution but neither duals nor a fully populated
    :class:`linopy.Model`. Other solvers retain Linopy's standard behaviour.

    On a successful Gurobi solve, ``n._pending_full_solve`` is set to an
    object exposing ``finish()`` and ``discard()``. A caller who later decides
    the partial solution is not enough can call ``finish()`` to complete the
    solve from the *same* already-solved Gurobi model - every remaining
    primal variable and every dual, mapped exactly as a full ``m.solve()``
    would - without a second call to Gurobi's ``optimize``. ``discard()``
    instead releases the retained Gurobi handles when the partial solution
    turned out to be enough. Exactly one of the two must be called before the
    model is solved again. On any other path (solve failure, non-Gurobi
    solver, missing bindings) ``n._pending_full_solve`` is ``None`` and there
    is nothing to release.
    """
    n._pending_full_solve = None

    if solver_name != "gurobi" or kwargs.get("remote") is not None:
        return m.solve(solver_name=solver_name, **solver_options, **kwargs)

    try:
        import gurobipy
    except ImportError:
        logger.warning(
            "Gurobi Python bindings are unavailable; falling back to the full "
            "Linopy solution import."
        )
        return m.solve(solver_name=solver_name, **solver_options, **kwargs)

    m.matrices.clean_cached_properties()
    m.reset_solution()

    io_api = kwargs.pop("io_api", None)
    problem_fn = kwargs.pop("problem_fn", None)
    solution_fn = kwargs.pop("solution_fn", None)
    log_fn = kwargs.pop("log_fn", None)
    basis_fn = kwargs.pop("basis_fn", None)
    warmstart_fn = kwargs.pop("warmstart_fn", None)
    keep_files = kwargs.pop("keep_files", False)
    env = kwargs.pop("env", None)
    sanitize_zeros = kwargs.pop("sanitize_zeros", True)
    kwargs.pop("remote", None)

    if io_api not in (None, "lp", "lp-polars", "mps", "direct"):
        raise ValueError(
            "Keyword argument `io_api` has to be one of "
            "'lp', 'lp-polars', 'mps', 'direct' or None"
        )

    if problem_fn is None:
        problem_fn = m.get_problem_file(io_api=io_api)
    problem_path = Path(problem_fn)

    if sanitize_zeros:
        m.constraints.sanitize_zeros()

    # The Gurobi handles (env, in-memory model) are kept open past this
    # function on success, so that a caller who ends up needing the complete
    # solution can read it from the model already solved here instead of
    # solving it a second time. They are released by whichever of
    # ``finish``/``discard`` the caller calls, or immediately below on
    # failure.
    stack = contextlib.ExitStack()

    def release() -> None:
        stack.close()
        if problem_path.exists() and not keep_files:
            problem_path.unlink()

    try:
        if env is None:
            env = stack.enter_context(gurobipy.Env())

        if io_api is None or io_api in ("lp", "lp-polars", "mps"):
            problem_path = m.to_file(problem_path, io_api=io_api)
            solver_model = gurobipy.read(str(problem_path), env=env)
        else:
            solver_model = m.to_gurobipy(env=env)

        for key, value in {**solver_options, **kwargs}.items():
            solver_model.setParam(key, value)
        if log_fn is not None:
            solver_model.setParam("logfile", str(log_fn))
        if warmstart_fn is not None:
            solver_model.read(str(warmstart_fn))

        solver_model.optimize()

        if basis_fn is not None:
            try:
                solver_model.write(str(basis_fn))
            except gurobipy.GurobiError as err:
                logger.info("No model basis stored. Raised error: %s", err)
        if solution_fn is not None and Path(solution_fn).suffix == ".sol":
            try:
                solver_model.write(str(solution_fn))
            except gurobipy.GurobiError as err:
                logger.info("Unable to save solution file. Raised error: %s", err)

        condition = _GUROBI_TERMINATION_CONDITIONS.get(
            solver_model.status, str(solver_model.status)
        )
        status = Status.from_termination_condition(condition)

        m.status = status.status.value
        m.termination_condition = status.termination_condition.value
        m.solver_name = solver_name
        m.solver_model = None

        if not status.is_ok:
            release()
            return m.status, m.termination_condition

        m.objective._value = float(solver_model.ObjVal)
        for name in selected_variables:
            if name not in m.variables:
                continue
            variable = m.variables[name]
            labels = np.asarray(variable.labels.values)
            values = np.full(labels.size, np.nan)
            valid = labels.ravel() >= 0
            if valid.any():
                solver_variables = [
                    solver_model.getVarByName(f"x{label}")
                    for label in labels.ravel()[valid]
                ]
                if any(v is None for v in solver_variables):
                    raise RuntimeError(
                        f"Could not retrieve Gurobi variables for Linopy group {name!r}."
                    )
                values[valid] = solver_model.getAttr("X", solver_variables)
            variable.solution = xr.DataArray(
                values.reshape(labels.shape), variable.coords
            )
    except BaseException:
        release()
        raise

    called = False

    def finish() -> None:
        nonlocal called
        if called:
            return
        called = True
        sol = _int_index_from_names(
            {v.VarName: v.X for v in solver_model.getVars()}
        )
        sol.loc[-1] = np.nan
        for _, var in m.variables.items():
            idx = np.ravel(var.labels)
            try:
                vals = sol[idx].values.reshape(var.labels.shape)
            except KeyError:
                vals = sol.reindex(idx).values.reshape(var.labels.shape)
            var.solution = xr.DataArray(vals, var.coords)

        try:
            dual = _int_index_from_names(
                {c.ConstrName: c.Pi for c in solver_model.getConstrs()}
            )
        except AttributeError:
            logger.warning("Dual values of MILP couldn't be parsed")
            dual = pd.Series(dtype=float)
        if not dual.empty:
            dual.loc[-1] = np.nan
            for _, con in m.constraints.items():
                idx = np.ravel(con.labels)
                try:
                    vals = dual[idx].values.reshape(con.labels.shape)
                except KeyError:
                    vals = dual.reindex(idx).values.reshape(con.labels.shape)
                con.dual = xr.DataArray(vals, con.labels.coords)

        m.solver_model = solver_model
        release()

    def discard() -> None:
        nonlocal called
        if called:
            return
        called = True
        release()

    n._pending_full_solve = SimpleNamespace(finish=finish, discard=discard)

    logger.info(
        "Optimization successful: imported %d selected primal variable group(s) "
        "without duals.",
        sum(name in m.variables for name in selected_variables),
    )
    return m.status, m.termination_condition


def optimize(
    n: Network,
    snapshots: Sequence | None = None,
    multi_investment_periods: bool = False,
    transmission_losses: int = 0,
    linearized_unit_commitment: bool = False,
    model_kwargs: dict = {},
    extra_functionality: Callable | None = None,
    assign_all_duals: bool = False,
    solver_name: str = "highs",
    solver_options: dict = {},
    compute_infeasibilities: bool = False,
    include_objective_constant: bool = False,
    _solution_variables: Sequence[str] | None = None,
    _assign_solution: bool = True,
    _assign_duals: bool = True,
    _post_processing: bool = True,
    **kwargs: Any,
) -> tuple[str, str]:
    """
    Optimize the pypsa network using linopy.

    Parameters
    ----------
    n : pypsa.Network
    snapshots : list or index slice
        A list of snapshots to optimise, must be a subset of
        n.snapshots, defaults to n.snapshots
    multi_investment_periods : bool, default False
        Whether to optimise as a single investment period or to optimise in multiple
        investment periods. Then, snapshots should be a ``pd.MultiIndex``.
    transmission_losses : int, default 0
        Whether an approximation of transmission losses should be included
        in the linearised power flow formulation. A passed number denotes the
        number of least-squares segments per half-axis used for the piecewise
        linear approximation of the loss parabola, which is thus approximated
        by ``2 * transmission_losses`` segments.
        Defaults to 0, which ignores losses.
    linearized_unit_commitment : bool, default False
        Whether to optimise using the linearised unit commitment formulation or not.
    model_kwargs: dict
        Keyword arguments used by `linopy.Model`, such as `solver_dir` or `chunk`.
    extra_functionality : callable
        This function must take two arguments
        `extra_functionality(network, snapshots)` and is called after
        the model building is complete, but before it is sent to the
        solver. It allows the user to
        add/change constraints and add/change the objective function.
    assign_all_duals : bool, default False
        Whether to assign all dual values or only those that already
        have a designated place in the network.
    solver_name : str
        Name of the solver to use.
    solver_options : dict
        Keyword arguments used by the solver. Can also be passed via `**kwargs`.
    compute_infeasibilities : bool, default False
        Whether to compute and print Irreducible Inconsistent Subsystem (IIS) in case
        of an infeasible solution. Requires Gurobi.
    include_objective_constant : bool, default False
        Whether to include the objective constant for existing extendable assets
        via a helper variable in the model objective.
    **kwargs:
        Keyword argument used by `linopy.Model.solve`, such as `solver_name`,
        `problem_fn` or solver options directly passed to the solver.

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

    sns = as_index(n, snapshots, "snapshots", "snapshot")
    n._multi_invest = int(multi_investment_periods)
    n._linearized_uc = linearized_unit_commitment

    n.consistency_check()
    m = create_model(
        n,
        sns,
        multi_investment_periods,
        transmission_losses,
        linearized_unit_commitment,
        include_objective_constant,
        **model_kwargs,
    )
    if extra_functionality:
        extra_functionality(n, sns)
    if _solution_variables is None:
        n._pending_full_solve = None
        status, condition = m.solve(
            solver_name=solver_name, **solver_options, **kwargs
        )
    else:
        status, condition = _solve_model_with_selected_variables(
            n,
            m,
            _solution_variables,
            solver_name,
            solver_options,
            **kwargs,
        )

    if status == "ok":
        if _assign_solution:
            assign_solution(n)
        if _assign_duals:
            assign_duals(n, assign_all_duals)
        if _post_processing:
            post_processing(n)

    if (
        condition == "infeasible"
        and compute_infeasibilities
        and "gurobi" in available_solvers
    ):
        n.model.print_infeasibilities()

    return status, condition


class OptimizationAccessor:
    """
    Optimization accessor for building and solving models using linopy.
    """

    def __init__(self, network: Network) -> None:
        self._parent = network

    @wraps(optimize)
    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return optimize(self._parent, *args, **kwargs)

    @wraps(create_model)
    def create_model(self, *args: Any, **kwargs: Any) -> Any:
        return create_model(self._parent, *args, **kwargs)

    def solve_model(
        self,
        extra_functionality: Callable | None = None,
        solver_name: str = "highs",
        solver_options: dict = {},
        assign_all_duals: bool = False,
        **kwargs: Any,
    ) -> tuple[str, str]:
        """
        Solve an already created model and assign its solution to the network.

        Parameters
        ----------
        solver_name : str
            Name of the solver to use.
        solver_options : dict
            Keyword arguments used by the solver. Can also be passed via `**kwargs`.
        assign_all_duals : bool, default False
            Whether to assign all dual values or only those that already
            have a designated place in the network.
        **kwargs:
            Keyword argument used by `linopy.Model.solve`, such as `solver_name`,
            `problem_fn` or solver options directly passed to the solver.

        Returns
        -------
        status : str
            The status of the optimization, either "ok" or one of the
            codes listed in
            https://linopy.readthedocs.io/en/latest/generated/linopy.constants.SolverStatus.html
        condition : str
            The termination condition of the optimization, either
            "optimal" or one of the codes listed in
            https://linopy.readthedocs.io/en/latest/generated/linopy.constants.TerminationCondition.html
        """
        n = self._parent
        if extra_functionality:
            extra_functionality(n, n.snapshots)
        m = n.model
        status, condition = m.solve(solver_name=solver_name, **solver_options, **kwargs)

        if status == "ok":
            assign_solution(n)
            assign_duals(n, assign_all_duals)
            post_processing(n)

        return status, condition

    @wraps(assign_solution)
    def assign_solution(self, *args: Any, **kwargs: Any) -> Any:
        return assign_solution(self._parent, **kwargs)

    @wraps(assign_duals)
    def assign_duals(self, *args: Any, **kwargs: Any) -> Any:
        return assign_duals(self._parent, **kwargs)

    @wraps(post_processing)
    def post_processing(self, *args: Any, **kwargs: Any) -> Any:
        return post_processing(self._parent, **kwargs)

    @wraps(optimize_transmission_expansion_iteratively)
    def optimize_transmission_expansion_iteratively(
        self, *args: Any, **kwargs: Any
    ) -> Any:
        return optimize_transmission_expansion_iteratively(
            self._parent, *args, **kwargs
        )

    @wraps(optimize_security_constrained)
    def optimize_security_constrained(self, *args: Any, **kwargs: Any) -> Any:
        return optimize_security_constrained(self._parent, *args, **kwargs)

    @wraps(optimize_with_rolling_horizon)
    def optimize_with_rolling_horizon(self, *args: Any, **kwargs: Any) -> Any:
        return optimize_with_rolling_horizon(self._parent, *args, **kwargs)

    @wraps(optimize_mga)
    def optimize_mga(self, *args: Any, **kwargs: Any) -> Any:
        return optimize_mga(self._parent, *args, **kwargs)

    def certify_expansion(self, *args: Any, **kwargs: Any) -> Any:
        """
        Bound the transmission expansion problem this network poses from below.

        See :func:`pypsa.optimization.lower_bound.certify_expansion`. The
        capacity dependence of the impedance is written around the reference
        data of this network, so call it on the network as it went into
        ``optimize_transmission_expansion_iteratively`` and pass the plan to
        certify, or on one whose ``_s_nom_def`` matches its impedances.
        """
        from pypsa.optimization.lower_bound import certify_expansion

        return certify_expansion(self._parent, *args, **kwargs)

    def fix_optimal_capacities(self) -> None:
        """
        Fix capacities of extendable assets to optimized capacities.

        Use this function when a capacity expansion optimization was
        already performed and a operational optimization should be done
        afterwards.
        """
        n = self._parent
        for c, attr in nominal_attrs.items():
            ext_i = n.get_extendable_i(c)
            n.df(c).loc[ext_i, attr] = n.df(c).loc[ext_i, attr + "_opt"]
            n.df(c)[attr + "_extendable"] = False

    def fix_optimal_dispatch(self) -> None:
        """
        Fix dispatch of all assets to optimized values.

        Use this function when the optimal dispatch should be used as an
        starting point for power flow calculation (`Network.pf`).
        """
        n = self._parent
        for c in n.one_port_components:
            n.pnl(c).p_set = n.pnl(c).p
        for c in n.controllable_branch_components:
            n.pnl(c).p_set = n.pnl(c).p0

    def add_load_shedding(
        self,
        suffix: str = " load shedding",
        buses: pd.Index | None = None,
        sign: float | pd.Series = 1e-3,
        marginal_cost: float | pd.Series = 1e2,
        p_nom: float | pd.Series = 1e9,
    ) -> pd.Index:
        """
        Add load shedding in form of generators to all or a subset of buses.

        For more information on load shedding see
        http://journal.frontiersin.org/article/10.3389/fenrg.2015.00055/full

        Parameters
        ----------
        buses : pandas.Index, optional
            Subset of buses where load shedding should be available.
            Defaults to all buses.
        sign : float/Series, optional
            Scaling of the load shedding. This is used to scale the price of the
            load shedding. The default is 1e-3 which translates to a measure in kW instead
            of MW.
        marginal_cost : float/Series, optional
            Price of the load shedding. The default is 1e2.
        p_nom : float/Series, optional
            Maximal load shedding. The default is 1e9 (kW).
        """
        n = self._parent
        if "Load" not in n.carriers.index:
            n.add("Carrier", "Load")
        if buses is None:
            buses = n.buses.index

        return n.madd(
            "Generator",
            buses,
            suffix,
            bus=buses,
            carrier="load",
            sign=sign,
            marginal_cost=marginal_cost,
            p_nom=p_nom,
        )
