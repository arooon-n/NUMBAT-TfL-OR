from __future__ import annotations
import pandas as pd
from pyomo.environ import ConcreteModel, Var, NonNegativeReals, NonNegativeIntegers, Objective, Constraint, minimize, value
from src.models.optimization import _inputs, _solver


def model3(frame, capacity, constraints, weights: dict[str, float], target_utilization: float = 0.80, multiplier: float = 1.0):
    work = _inputs(frame, capacity, constraints, multiplier)
    if "line_boarders" not in work:
        raise ValueError("Model 3 requires NUMBAT line_boarders.")
    work["boarders"] = work.line_boarders.fillna(0).astype(float)
    grouped = work.groupby(["line", "service_group", "time_period"], as_index=False).agg(demand=("boarders", "max"), capacity=("capacity_per_train", "max"), minimum=("min_frequency_per_15min", "max"), maximum=("max_frequency_per_15min", "min"), target=("target_frequency_per_15min", "max"))
    keys=list(grouped.itertuples(index=False, name=None)); model=ConcreteModel(); model.K=range(len(keys)); model.x=Var(model.K,domain=NonNegativeIntegers); model.y=Var(model.K,domain=NonNegativeReals); model.unmet=Var(model.K,domain=NonNegativeReals); model.freq_minus=Var(model.K,domain=NonNegativeReals); model.freq_plus=Var(model.K,domain=NonNegativeReals); model.crowd=Var(model.K,domain=NonNegativeReals)
    for k,row in enumerate(keys):
        model.add_component(f"bounds_{k}",Constraint(expr=(row[5],model.x[k],row[6]))); model.add_component(f"demand_goal_{k}",Constraint(expr=model.y[k]+model.unmet[k]==row[3])); model.add_component(f"frequency_goal_{k}",Constraint(expr=model.x[k]+model.freq_minus[k]-model.freq_plus[k]==row[7])); model.add_component(f"served_cap_{k}",Constraint(expr=model.y[k]<=row[4]*model.x[k])); model.add_component(f"crowding_{k}",Constraint(expr=row[3]-target_utilization*row[4]*model.x[k]<=model.crowd[k]))
    model.obj=Objective(expr=weights.get("unmet_demand",1)*sum(model.unmet[k] for k in model.K)+weights.get("frequency_deviation",1)*sum(model.freq_minus[k]+model.freq_plus[k] for k in model.K)+weights.get("crowding",1)*sum(model.crowd[k] for k in model.K)+weights.get("service_usage",0.01)*sum(model.x[k] for k in model.K),sense=minimize)
    result=_solver().solve(model,tee=False); status=str(result.solver.status); termination=str(result.solver.termination_condition); rows=[]
    for k,row in enumerate(keys):
        rows.append({"line":row[0],"service_group":row[1],"time_period":row[2],"optimized_frequency":int(round(value(model.x[k]))) if termination in ("optimal","feasible") else None,"demand":row[3],"demand_served":value(model.y[k]),"unmet_demand_deviation":value(model.unmet[k]),"frequency_minus_deviation":value(model.freq_minus[k]),"frequency_plus_deviation":value(model.freq_plus[k]),"crowding_deviation":value(model.crowd[k]),"solver_status":status,"termination":termination})
    return pd.DataFrame(rows), status, termination
