"""
Validation tests for the smart-city simulation (run with:  python -m pytest -v tests).
Each test checks one property that must hold for the results to be trustworthy.
"""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import smart_city_sim as sc  # noqa: E402


@pytest.fixture(scope="module")
def short_runs():
    out = {}
    for ctrl in [sc.FixedTimeController(), sc.ActuatedController(),
                 sc.QLearningController(alpha=0.0, epsilon=0.0)]:
        out[ctrl.name] = sc.run_day(ctrl, seed=3, hours=3.0, start_hour=6.5)
    return out


def test_vehicle_conservation(short_runs):
    """Every generated vehicle has either exited or is still inside the network."""
    for name, (city, _, veh, _) in short_runs.items():
        assert city.generated == len(veh) + city.vehicles_in_system(), name


def test_no_conflicting_greens(short_runs):
    """N-S and E-W movements are never green at the same time."""
    for name, (city, _, _, samp) in short_runs.items():
        assert city.violations == 0, name
        assert set(samp["phase"].unique()) <= {0, 1}


def test_yellow_interval_always_applied(short_runs):
    """Every switch to a new green is preceded by a yellow/all-red interval."""
    city = short_runs["Rule-based actuated"][0]
    by_int = {}
    for t, iid, phase, state in city.phase_log:
        by_int.setdefault(iid, []).append((t, state))
    for iid, events in by_int.items():
        for (t0, s0), (t1, s1) in zip(events, events[1:]):
            if s1 == "G":
                assert s0 == "Y" and t1 - t0 >= sc.YELLOW_TICKS * sc.TICK - 1e-9


def test_fixed_time_cycle_length(short_runs):
    """Fixed-time control produces a constant 30 s green phase."""
    city = short_runs["Fixed-time"][0]
    greens = [t for t, iid, p, s in city.phase_log if iid == 4 and s == "G"]
    yellows = [t for t, iid, p, s in city.phase_log if iid == 4 and s == "Y"]
    durations = [y - g for g, y in zip(greens, yellows[1:])]
    assert np.allclose(durations, 30.0)


def test_delays_non_negative(short_runs):
    for name, (_, _, veh, _) in short_runs.items():
        assert (veh["delay"] >= 0).all() and (veh["wait"] >= 0).all(), name


def test_saturation_flow_respected(short_runs):
    """No approach discharges faster than 1 vehicle per 2 s (1,800 veh/h)."""
    _, _, veh, _ = short_runs["Rule-based actuated"]
    exits = np.sort(veh["t_exit"].values)
    per_min = np.histogram(exits, bins=np.arange(0, exits.max() + 60, 60))[0]
    # 12 exit approaches x 30 veh/min upper bound
    assert per_min.max() <= 12 * 30


def test_link_capacity_respected(short_runs):
    for name, (city, _, _, _) in short_runs.items():
        for it in city.intersections:
            for h in sc.HEADINGS:
                assert len(it.transit[h]) <= sc.LINK_CAPACITY, name


def test_battery_state_of_charge_bounds(short_runs):
    _, grid, _, _ = short_runs["Q-learning (RL)"]
    df = grid.frames()
    for z, zc in sc.ZONES.items():
        soc = df[df.zone == z]["soc"]
        assert soc.min() >= -1e-6 and soc.max() <= zc["bat"] + 1e-6


def test_energy_balance(short_runs):
    """Supply equals demand every 15 minutes for both energy policies."""
    _, grid, _, _ = short_runs["Q-learning (RL)"]
    df = grid.frames()
    supply = df["grid"] + df["pv"] + df["bat_dis"] + df["share_in"]
    use = df["load"] + df["bat_ch"] + df["share_out"] + df["curtail"]
    assert np.allclose(supply, use, atol=1e-6)


def test_adaptive_lighting_never_below_safety_level(short_runs):
    _, grid, _, _ = short_runs["Q-learning (RL)"]
    df = grid.frames()
    dark = df[df.time_h.apply(sc.is_dark) & (df.policy == "AI-optimized")]
    for z, zc in sc.ZONES.items():
        lk = dark[dark.zone == z]["light_kw"]
        if len(lk):
            assert lk.min() >= 0.3 * zc["lights"] * sc.LIGHT_KW - 1e-9


def test_forecaster_accuracy():
    model, m, _ = sc.train_forecaster()
    assert m["r2"] > 0.95
    assert m["mae_kw"] < m["naive_mae_kw"]


def test_q_learning_updates_values():
    Q, hist = sc.train_q_agent(episodes=4, episode_hours=0.5)
    assert np.abs(Q).sum() > 0
    assert hist["q_states_visited"].iloc[-1] > 10


def test_trained_agent_beats_fixed_time():
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "outputs", "data", "q_table.npy")
    if not os.path.exists(path):
        pytest.skip("run run_experiments.py first to create the trained Q-table")
    Q = np.load(path)
    _, _, v_fixed, _ = sc.run_day(sc.FixedTimeController(), seed=77, hours=2, start_hour=7,
                                  with_energy=False)
    _, _, v_rl, _ = sc.run_day(sc.QLearningController(Q, alpha=0, epsilon=0), seed=77, hours=2,
                               start_hour=7, with_energy=False)
    assert v_rl["delay"].mean() < v_fixed["delay"].mean()


def test_unity_trace_matches_simulation():
    """The Unity playback trace must describe exactly the simulated trips."""
    import simpy
    import export_unity_data as ex
    env = simpy.Environment()
    city = sc.City(env, sc.ActuatedController(), seed=3)
    city.trace = []
    env.run(until=2 * 3600.0)
    hops = ex.hops_table(city.trace)
    done = {r["vid"]: r for r in city.completed}
    # every completed vehicle has one hop per intersection crossed
    counts = hops.groupby("vid").size()
    assert all(counts[v] == r["hops"] for v, r in done.items())
    # departures never precede arrivals, and link travel takes at least the free-flow time
    departed = hops[hops.t_dep >= 0]
    assert (departed.t_dep >= departed.t_arr - 1e-6).all()
    h = hops.sort_values(["vid", "t_arr"])
    same = h.vid.values[1:] == h.vid.values[:-1]
    gaps = h.t_arr.values[1:][same] - h.t_dep.values[:-1][same]
    assert (gaps >= sc.TRAVEL_TIME - 1e-6).all() and (gaps <= sc.TRAVEL_TIME + sc.TICK + 1e-6).all()
    # the heading leaving one intersection is the approach heading at the next one
    assert (h["next"].values[:-1][same] == h["appr"].values[1:][same]).all()
