"""
Export simulation traces for the Unity 3D playback scene.

The script re-runs evaluation day 1 (seed 1) for each traffic controller with
trace logging switched on, then writes compact CSV files that the Unity
project reads from Assets/StreamingAssets/SmartCityData:

    hops_<key>.csv      one row per vehicle per intersection visit
                        (vid, iid, approach heading, next heading,
                         arrival time at stop line, departure time)
    signals_<key>.csv   every signal change (time, iid, phase, state)
    vehicles_<key>.csv  completed trips (vid, t_enter, t_exit, delay)
    energy_dispatch.csv 15-minute energy dispatch for both energy policies
    summary.csv         5-day mean metrics used by the Unity dashboard

Trace logging only records events; it does not change the random-number
stream, so the replayed day is identical to evaluation day 1 in the report.

Usage:  python export_unity_data.py [--unity-dir ../SmartCityUnity]
"""
import argparse
import os

import numpy as np
import pandas as pd
import simpy

import smart_city_sim as sc

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "outputs", "data")


def traced_day(controller, seed=1, hours=24.0):
    env = simpy.Environment()
    city = sc.City(env, controller, seed=seed)
    city.trace = []
    env.run(until=hours * 3600.0)
    return city


def hops_table(trace):
    """Pair each arrival (A) with the departure (D) from the same intersection."""
    open_hops = {}
    rows = []
    for ev, vid, iid, h, t in trace:
        if ev == "A":
            open_hops[vid] = [vid, iid, h, "", round(t, 2), -1.0]
            rows.append(open_hops[vid])
        else:
            hop = open_hops.pop(vid)
            hop[3] = h
            hop[5] = round(t, 2)
    df = pd.DataFrame(rows, columns=["vid", "iid", "appr", "next", "t_arr", "t_dep"])
    df["next"] = df["next"].replace("", "-")
    return df.sort_values(["vid", "t_arr"]).reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--unity-dir", default=os.path.join(HERE, "..", "SmartCityUnity"))
    args = ap.parse_args()
    out = os.path.join(os.path.abspath(args.unity_dir), "Assets", "StreamingAssets", "SmartCityData")
    os.makedirs(out, exist_ok=True)

    Q = np.load(os.path.join(DATA, "q_table.npy"))
    controllers = {
        "fixed": sc.FixedTimeController(),
        "rule": sc.ActuatedController(),
        "rl": sc.QLearningController(Q.copy(), alpha=0.0, epsilon=0.0),
    }
    check = pd.read_csv(os.path.join(DATA, "traffic_metrics_by_seed.csv"))
    for key, ctrl in controllers.items():
        city = traced_day(ctrl)
        veh = pd.DataFrame(city.completed)
        hops = hops_table(city.trace)
        sig = pd.DataFrame(city.phase_log, columns=["t", "iid", "phase", "state"])
        hops.to_csv(os.path.join(out, f"hops_{key}.csv"), index=False)
        sig.to_csv(os.path.join(out, f"signals_{key}.csv"), index=False)
        veh[["vid", "t_enter", "t_exit", "delay"]].round(2).to_csv(
            os.path.join(out, f"vehicles_{key}.csv"), index=False)
        ref = check[(check.controller == ctrl.name) & (check.seed == 1)]["avg_delay_s"].iloc[0]
        print(f"{ctrl.name:22s} vehicles={len(veh):6d} hops={len(hops):7d} signal events={len(sig):6d} "
              f"avg delay={veh['delay'].mean():.2f} s (report seed 1: {ref:.2f} s)")

    edf = pd.read_csv(os.path.join(DATA, "energy_dispatch_seed1.csv"))
    edf.round(3).to_csv(os.path.join(out, "energy_dispatch.csv"), index=False)

    tm = pd.read_csv(os.path.join(DATA, "traffic_metrics_by_seed.csv"))
    em = pd.read_csv(os.path.join(DATA, "energy_metrics_by_seed.csv"))
    t = tm.groupby("controller")[["avg_delay_s", "p95_delay_s", "throughput_vph",
                                   "total_wait_veh_h"]].mean().round(2)
    e = em.groupby("policy")[["grid_import_kwh", "peak_grid_kw", "energy_cost_usd",
                               "energy_waste_kwh", "lighting_kwh"]].mean().round(1)
    rows = [("traffic", k, c, v) for k, r in t.iterrows() for c, v in r.items()]
    rows += [("energy", k, c, v) for k, r in e.iterrows() for c, v in r.items()]
    pd.DataFrame(rows, columns=["group", "name", "metric", "value"]).to_csv(
        os.path.join(out, "summary.csv"), index=False)
    print("Unity data written to", out)


if __name__ == "__main__":
    main()
