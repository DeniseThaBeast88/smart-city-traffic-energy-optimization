"""
AI-Driven Smart-City Traffic and Energy Optimization System
===========================================================
Core simulation module (SimPy discrete-event simulation).

City model
----------
* A 3 x 3 grid of signalized intersections (9 traffic-light agents).
* Road links of 300 m between neighboring intersections (free-flow travel 22 s).
* 12 boundary entry points with time-of-day Poisson demand.
* Four energy zones (Residential, Commercial, Industrial, Mixed-Use) with
  building loads, rooftop solar (PV), batteries, and street lighting.

Traffic-signal controllers
--------------------------
1. FixedTimeController   - 30 s green per phase (baseline).
2. ActuatedController    - rule-based queue-pressure logic (min/max green).
3. QLearningController   - tabular Q-learning, one shared Q-table used by all
                           nine intersection agents (parameter sharing).

Energy managers
---------------
1. Baseline  - scheduled street lights, greedy battery use, no zone sharing.
2. Optimized - regression day-ahead load forecast, forecast-based peak shaving,
               off-peak battery pre-charging, inter-zone solar sharing, and
               traffic-adaptive street lighting.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import simpy

# --------------------------------------------------------------------------
# Global constants
# --------------------------------------------------------------------------
GRID_ROWS, GRID_COLS = 3, 3
TICK = 2.0                 # saturation headway (s) -> 1,800 veh/h per lane
YELLOW_TICKS = 2           # 4 s yellow + all-red clearance
TRAVEL_TIME = 22.0         # free-flow link travel time (s), 300 m at 50 km/h
LINK_CAPACITY = 40         # vehicles stored per 300 m link (7.5 m per vehicle)
TURN_LEFT_P, TURN_RIGHT_P = 0.10, 0.10
HEADINGS = ("N", "S", "E", "W")
PHASE_HEADINGS = {0: ("N", "S"), 1: ("E", "W")}   # phase 0 = NS green, 1 = EW green
MOVE = {"N": (-1, 0), "S": (1, 0), "E": (0, 1), "W": (0, -1)}
LEFT = {"N": "W", "W": "S", "S": "E", "E": "N"}
RIGHT = {"N": "E", "E": "S", "S": "W", "W": "N"}

# Hourly demand multiplier (0-23 h): night trough, AM peak (8 h), PM peak (17 h)
DEMAND_PROFILE = np.array([
    0.10, 0.07, 0.06, 0.06, 0.10, 0.25, 0.55, 0.90, 1.00, 0.80, 0.60, 0.60,
    0.65, 0.62, 0.62, 0.70, 0.88, 1.00, 0.92, 0.65, 0.45, 0.35, 0.25, 0.15])


def entry_rate_vph(heading: str, index: int, hour: float) -> float:
    """Arrival rate (vehicles/h) for an entry; index is the column or row."""
    h = int(hour) % 24
    mult = DEMAND_PROFILE[h]
    if heading in ("N", "S"):
        base = 620.0 if index == 1 else 260.0        # column 1 is a N-S arterial
        am, pm = (1.2, 0.8) if heading == "S" else (0.8, 1.2)
    else:
        base = 420.0 if index == 1 else 220.0        # row 1 is an E-W arterial
        am, pm = (1.15, 0.85) if heading == "E" else (0.85, 1.15)
    if 6 <= h <= 10:
        mult *= am
    elif 15 <= h <= 19:
        mult *= pm
    return base * mult


# --------------------------------------------------------------------------
# Vehicles and intersections
# --------------------------------------------------------------------------
@dataclass
class Vehicle:
    vid: int
    t_enter: float
    heading: str
    hops: int = 0
    wait: float = 0.0
    t_ready: float = 0.0
    next_heading: str = ""


@dataclass
class Intersection:
    r: int
    c: int
    queues: dict = field(default_factory=lambda: {h: deque() for h in HEADINGS})
    transit: dict = field(default_factory=lambda: {h: deque() for h in HEADINGS})
    phase: int = 0
    state: str = "G"        # "G" green or "Y" yellow/all-red
    green_ticks: int = 0
    yellow_left: int = 0
    switches: int = 0

    @property
    def iid(self) -> int:
        return self.r * GRID_COLS + self.c

    def queue_len(self, headings=HEADINGS) -> int:
        return sum(len(self.queues[h]) for h in headings)

    def approaching(self, headings=HEADINGS) -> int:
        return sum(len(self.transit[h]) for h in headings)

    def link_load(self, heading: str) -> int:
        return len(self.queues[heading]) + len(self.transit[heading])

    def green_headings(self):
        return PHASE_HEADINGS[self.phase] if self.state == "G" else ()


# --------------------------------------------------------------------------
# Signal controllers
# --------------------------------------------------------------------------
class FixedTimeController:
    name = "Fixed-time"

    def __init__(self, green_s: float = 30.0):
        self.green_ticks = int(green_s / TICK)

    def decide(self, it: Intersection, now: float) -> bool:
        return it.green_ticks >= self.green_ticks


class ActuatedController:
    """Rule-based queue-pressure control with min/max green and gap-out."""
    name = "Rule-based actuated"

    def __init__(self, min_green=10.0, max_green=60.0, margin=3):
        self.min_t = int(min_green / TICK)
        self.max_t = int(max_green / TICK)
        self.margin = margin

    def decide(self, it: Intersection, now: float) -> bool:
        g = PHASE_HEADINGS[it.phase]
        r = PHASE_HEADINGS[1 - it.phase]
        qg, qr = it.queue_len(g), it.queue_len(r)
        if it.green_ticks < self.min_t:
            return False
        if it.green_ticks >= self.max_t and qr > 0:
            return True                                   # max-out
        if qg == 0 and it.approaching(g) == 0 and qr > 0:
            return True                                   # gap-out
        return qr > qg + self.margin                      # pressure rule


def _bin(x, edges):
    return int(np.searchsorted(edges, x, side="left"))   # bin i: edges[i-1] < x <= edges[i]


class QLearningController:
    """Tabular Q-learning traffic-signal agent with a shared Q-table.

    State  : (queue-pressure bin = red queue - green queue,
              green-queue bin, green-approaching bin, green-time bin,
              red-queue bin) -> 8 * 4 * 3 * 3 * 3 = 864 phase-symmetric states
    Action : 0 = keep current phase, 1 = switch phase
    Reward : negative mean number of queued vehicles at the intersection
             between decisions (equivalent to minimizing total waiting time).
    """
    name = "Q-learning (RL)"
    D_EDGES = np.array([-6, -3, 0, 2, 4, 7, 11])   # pressure bins (8)
    G_EDGES = np.array([0, 2, 6])                  # green queue: 0 | 1-2 | 3-6 | 7+
    A_EDGES = np.array([0, 3])                     # approaching: 0 | 1-3 | 4+
    T_EDGES = np.array([9, 19])                    # green time: <20 s | 20-40 s | >40 s
    R_EDGES = np.array([0, 2])                     # red queue: 0 | 1-2 | 3+
    SHAPE = (8, 4, 3, 3, 3, 2)

    def __init__(self, q_table=None, alpha=0.1, gamma=0.9, epsilon=0.0,
                 min_green=10.0, max_green=90.0, decision_s=2.0, seed=0):
        self.Q = np.zeros(self.SHAPE) if q_table is None else q_table
        self.alpha, self.gamma, self.epsilon = alpha, gamma, epsilon
        self.min_t = int(min_green / TICK)
        self.max_t = int(max_green / TICK)
        self.dec_t = max(1, int(decision_s / TICK))
        self.rng = np.random.default_rng(seed)
        self.memory = {}      # iid -> [state, action, accumulated queue, ticks]
        self.td_errors = []

    def state(self, it: Intersection):
        g = PHASE_HEADINGS[it.phase]
        r = PHASE_HEADINGS[1 - it.phase]
        qg, qr = it.queue_len(g), it.queue_len(r)
        return (_bin(qr - qg, self.D_EDGES),
                _bin(qg, self.G_EDGES),
                _bin(it.approaching(g), self.A_EDGES),
                _bin(it.green_ticks, self.T_EDGES),
                _bin(qr, self.R_EDGES))

    def observe(self, it: Intersection):
        """Called every tick to accumulate the reward signal."""
        m = self.memory.get(it.iid)
        if m is not None:
            m[2] += it.queue_len()
            m[3] += 1

    def decide(self, it: Intersection, now: float) -> bool:
        self.observe(it)
        if it.green_ticks < self.min_t:
            return False
        r = PHASE_HEADINGS[1 - it.phase]
        if it.green_ticks >= self.max_t and it.queue_len(r) > 0:
            self._learn(it, self.state(it), terminal_switch=True)
            return True                                   # safety max-green rule
        if (it.green_ticks - self.min_t) % self.dec_t != 0:
            return False
        s = self.state(it)
        self._learn(it, s)
        if self.rng.random() < self.epsilon:
            a = int(self.rng.integers(2))
        else:
            qv = self.Q[s]
            a = int(np.argmax(qv)) if qv[0] != qv[1] else int(s[0] >= 4)
        self.memory[it.iid] = [s, a, 0.0, 0]
        return a == 1

    def _learn(self, it, s_new, terminal_switch=False):
        m = self.memory.pop(it.iid, None)
        if m is None or self.alpha == 0:
            return
        s, a, qsum, n = m
        reward = -(qsum / max(n, 1)) / 10.0
        target = reward + self.gamma * np.max(self.Q[s_new])
        td = target - self.Q[s + (a,)]
        self.Q[s + (a,)] += self.alpha * td
        self.td_errors.append(abs(td))


# --------------------------------------------------------------------------
# Energy zones and managers
# --------------------------------------------------------------------------
ZONES = {
    # name: buildings, avg kW/building, profile type, PV kWp, battery kWh, battery kW,
    #       street lights, intersections served
    "Residential": dict(n=120, kw=1.3, kind="res", pv=420, bat=600, pkw=150,
                        lights=70, iids=[0, 1, 3]),
    "Commercial": dict(n=25, kw=36.0, kind="com", pv=380, bat=900, pkw=260,
                       lights=90, iids=[2, 4, 5]),
    "Industrial": dict(n=8, kw=55.0, kind="ind", pv=950, bat=500, pkw=200,
                       lights=50, iids=[7, 8]),
    "Mixed-Use": dict(n=30, kw=9.0, kind="mix", pv=380, bat=400, pkw=120,
                      lights=60, iids=[6]),
}
LIGHT_KW = 0.15            # LED street light (kW)
DT_H = 0.25                # energy step: 15 minutes
BAT_EFF = 0.95             # one-way battery efficiency
SHARE_EFF = 0.97           # inter-zone feeder efficiency

PROFILES = {   # normalized hourly building-use shapes (0-23 h)
    "res": [0.45, 0.40, 0.38, 0.38, 0.40, 0.50, 0.75, 0.85, 0.70, 0.55, 0.50, 0.50,
            0.52, 0.52, 0.55, 0.62, 0.78, 0.95, 1.00, 0.98, 0.90, 0.80, 0.65, 0.52],
    "com": [0.30, 0.28, 0.28, 0.28, 0.30, 0.35, 0.50, 0.75, 0.92, 1.00, 1.00, 1.00,
            1.00, 1.00, 1.00, 0.98, 0.92, 0.80, 0.60, 0.45, 0.38, 0.34, 0.32, 0.30],
    "ind": [0.70, 0.70, 0.70, 0.70, 0.72, 0.80, 0.92, 1.00, 1.00, 1.00, 1.00, 1.00,
            0.95, 1.00, 1.00, 1.00, 0.98, 0.90, 0.80, 0.75, 0.72, 0.70, 0.70, 0.70],
    "mix": [0.35, 0.32, 0.30, 0.30, 0.32, 0.40, 0.60, 0.80, 0.90, 0.92, 0.95, 0.98,
            1.00, 0.98, 0.95, 0.95, 0.95, 0.98, 0.95, 0.85, 0.70, 0.55, 0.45, 0.38],
}


def temperature_c(hour: float, day: int = 0, rng=None) -> float:
    """Daily outdoor temperature curve (late-October Houston-like climate)."""
    t = 23.0 + 6.0 * math.sin(2 * math.pi * (hour - 9.0) / 24.0)
    if rng is not None:
        t += rng.normal(0, 0.6)
    return t + 1.5 * math.sin(day * 1.3)


def solar_fraction(hour: float, cloud: float = 0.0) -> float:
    if hour < 7.0 or hour > 18.5:
        return 0.0
    return max(0.0, math.sin(math.pi * (hour - 7.0) / 11.5)) * (1 - cloud)


def is_dark(hour: float) -> bool:
    return hour < 7.0 or hour >= 18.75


def tariff(hour: float) -> float:
    h = int(hour) % 24
    if 14 <= h < 20:
        return 0.22       # on-peak USD/kWh
    if h >= 22 or h < 6:
        return 0.08       # off-peak
    return 0.12           # mid-peak


def building_load_kw(zone: str, hour: float, day: int, rng) -> float:
    z = ZONES[zone]
    shape = PROFILES[z["kind"]][int(hour) % 24]
    weekday = 0.85 if (day % 7) in (5, 6) and z["kind"] in ("com", "ind") else 1.0
    base = z["n"] * z["kw"] * shape * weekday
    hvac = 0.025 * z["n"] * z["kw"] * max(0.0, temperature_c(hour, day) - 22.0)
    return max(0.0, base + hvac + rng.normal(0, 0.03 * base))


def make_features(hour, day, zone_idx, traffic_density, lag_day_load):
    """Feature vector for the day-ahead regression model."""
    return [math.sin(2 * math.pi * hour / 24), math.cos(2 * math.pi * hour / 24),
            day % 7, temperature_c(hour, day), zone_idx, traffic_density, lag_day_load]


def generate_energy_history(days=21, seed=7):
    """Synthetic historical metering data used to train the forecaster."""
    rng = np.random.default_rng(seed)
    rows = []
    steps = int(24 / DT_H)
    prev = {z: [building_load_kw(z, k * DT_H, -1, rng) for k in range(steps)] for z in ZONES}
    for d in range(days):
        cur = {}
        for zi, z in enumerate(ZONES):
            cur[z] = []
            for k in range(steps):
                hr = k * DT_H
                dens = DEMAND_PROFILE[int(hr)] * (1 + rng.normal(0, 0.05))
                load = building_load_kw(z, hr, d, rng)
                cur[z].append(load)
                rows.append(dict(day=d, hour=hr, zone=z, zone_idx=zi, density=dens,
                                 lag_day=prev[z][k], load=load))
        prev = cur
    return pd.DataFrame(rows)


class EnergyLedger:
    """Tracks one energy-management policy for all four zones."""

    def __init__(self, name):
        self.name = name
        self.soc = {z: 0.5 * ZONES[z]["bat"] for z in ZONES}
        self.rows = []


class EnergyGridManager:
    """15-minute SimPy process that dispatches both energy policies side by side."""

    def __init__(self, env, city, forecaster=None, day=0, seed=11):
        self.env, self.city, self.day = env, city, day
        self.rng = np.random.default_rng(seed)
        self.forecaster = forecaster
        self.base = EnergyLedger("Baseline")
        self.opt = EnergyLedger("AI-optimized")
        self.cloud = np.clip(self.rng.normal(0.15, 0.08, 96), 0, 0.6)
        self.forecast = self._day_ahead_forecast()
        self.thresholds = self._peak_thresholds()
        self.precharge_goal = self._precharge_goals()
        env.process(self.run())

    # ---- forecasting -----------------------------------------------------
    def _day_ahead_forecast(self):
        steps = int(24 / DT_H)
        rng = np.random.default_rng(99)
        out = {}
        for zi, z in enumerate(ZONES):
            lag = [building_load_kw(z, k * DT_H, self.day - 1, rng) for k in range(steps)]
            if self.forecaster is None:
                out[z] = np.array(lag)
                continue
            X = [make_features(k * DT_H, self.day, zi, DEMAND_PROFILE[int(k * DT_H)], lag[k])
                 for k in range(steps)]
            out[z] = self.forecaster.predict(np.array(X))
        return out

    def _peak_thresholds(self):
        """Lowest net-load level the battery can hold the grid to (water-filling)."""
        th = {}
        hrs = np.arange(96) * DT_H
        pv_exp = np.array([solar_fraction(h, 0.15) for h in hrs])
        for z, zc in ZONES.items():
            net = self.forecast[z] - zc["pv"] * pv_exp
            usable = 0.9 * zc["bat"] * BAT_EFF
            lo, hi = 0.0, max(net.max(), 1.0)
            for _ in range(40):
                mid = (lo + hi) / 2
                need = np.clip(net - mid, 0, zc["pkw"]).sum() * DT_H
                lo, hi = (mid, hi) if need > usable else (lo, mid)
            th[z] = hi
        return th

    def _precharge_goals(self):
        """Off-peak pre-charge only what tomorrow's solar surplus will not supply."""
        hrs = np.arange(96) * DT_H
        pv_exp = np.array([solar_fraction(h, 0.15) for h in hrs])
        goals = {}
        for z, zc in ZONES.items():
            surplus = np.clip(zc["pv"] * pv_exp - self.forecast[z], 0, zc["pkw"]).sum() * DT_H
            goals[z] = float(np.clip(0.9 * zc["bat"] - surplus * BAT_EFF,
                                     0.2 * zc["bat"], 0.9 * zc["bat"]))
        return goals

    # ---- dispatch --------------------------------------------------------
    def zone_density(self, z):
        its = [self.city.intersections[i] for i in ZONES[z]["iids"]]
        veh = sum(it.queue_len() + it.approaching() for it in its)
        return veh / (len(its) * 4 * 10.0)        # 10 veh per approach = "busy"

    def run(self):
        k = 0
        while True:
            hr = (self.env.now / 3600.0) % 24
            idx = min(int(hr / DT_H), 95)
            loads = {z: building_load_kw(z, hr, self.day, self.rng) for z in ZONES}
            pv = {z: ZONES[z]["pv"] * solar_fraction(hr, self.cloud[idx]) for z in ZONES}
            dens = {z: self.zone_density(z) for z in ZONES}
            self._dispatch(self.base, hr, loads, pv, dens, optimized=False)
            self._dispatch(self.opt, hr, loads, pv, dens, optimized=True)
            k += 1
            yield self.env.timeout(DT_H * 3600)

    def _dispatch(self, L, hr, loads, pv, dens, optimized):
        res = {}
        for z, zc in ZONES.items():
            lights_full = zc["lights"] * LIGHT_KW if is_dark(hr) else 0.0
            need = min(1.0, 0.3 + 0.7 * min(1.0, dens[z] / 0.5)) if is_dark(hr) else 0.0
            if optimized:
                light_kw = zc["lights"] * LIGHT_KW * min(1.0, need * 1.1)
            else:
                light_kw = lights_full
            light_waste = max(0.0, light_kw - zc["lights"] * LIGHT_KW * need)
            load = loads[z] + light_kw
            net = load - pv[z]
            res[z] = dict(load=load, light_kw=light_kw, light_waste=light_waste, net=net,
                          pv=pv[z], bat_ch=0.0, bat_dis=0.0, grid=0.0, share_out=0.0,
                          share_in=0.0, curtail=0.0, dens=dens[z])
        # 1) surplus solar charges the local battery
        for z, r in res.items():
            zc = ZONES[z]
            if r["net"] < 0:
                room = (zc["bat"] - L.soc[z]) / BAT_EFF / DT_H
                ch = min(-r["net"], zc["pkw"], room)
                L.soc[z] += ch * BAT_EFF * DT_H
                r["bat_ch"] += ch
                r["net"] += ch
        # 2) optimized only: share remaining surplus with deficit zones
        if optimized:
            pool = sum(-r["net"] for r in res.values() if r["net"] < 0)
            for z, r in res.items():
                if r["net"] < 0:
                    r["share_out"] = -r["net"]
                    r["net"] = 0.0
            deficits = {z: r["net"] for z, r in res.items() if r["net"] > 0}
            tot = sum(deficits.values())
            if pool > 0 and tot > 0:
                for z, d in deficits.items():
                    got = min(d, pool * SHARE_EFF * d / tot)
                    res[z]["share_in"] = got
                    res[z]["net"] -= got
                used = sum(res[z]["share_in"] for z in deficits) / SHARE_EFF
                left = pool - used
                if left > 1e-9:   # leftover surplus is curtailed at the sources
                    for z, r in res.items():
                        if r["share_out"] > 0:
                            frac = r["share_out"] / pool
                            r["curtail"] = left * frac
                            r["share_out"] -= left * frac
            elif pool > 0:
                for z, r in res.items():
                    if r["share_out"] > 0:
                        r["curtail"] = r["share_out"]
                        r["share_out"] = 0.0
        # 3) deficits: battery discharge then grid
        for z, r in res.items():
            zc = ZONES[z]
            if r["net"] < 0:                      # baseline surplus is curtailed
                r["curtail"] += -r["net"]
                r["net"] = 0.0
                continue
            avail = L.soc[z] * BAT_EFF / DT_H
            if optimized:
                target = max(0.0, r["net"] - self.thresholds[z])
                if tariff(hr) >= 0.22:            # also use storage at on-peak price
                    target = r["net"]
                reserve = 0.1 * zc["bat"] if hr < 20 else 0.0
                avail = max(0.0, (L.soc[z] - reserve)) * BAT_EFF / DT_H
                dis = min(target, zc["pkw"], avail)
            else:
                dis = min(r["net"], zc["pkw"], avail)  # greedy self-consumption
            L.soc[z] -= dis / BAT_EFF * DT_H
            r["bat_dis"] = dis
            r["grid"] = r["net"] - dis
            # 4) optimized only: off-peak grid pre-charging for the evening peak
            if optimized and (hr >= 22 or hr < 6):
                goal = self.precharge_goal[z]
                room = max(0.0, goal - L.soc[z]) / BAT_EFF / DT_H
                ch = min(zc["pkw"] * 0.6, room)
                L.soc[z] += ch * BAT_EFF * DT_H
                r["bat_ch"] += ch
                r["grid"] += ch
            r["soc"] = L.soc[z]
            r["soc_frac"] = L.soc[z] / zc["bat"]
        for z, r in res.items():
            r.setdefault("soc", L.soc[z])
            r.setdefault("soc_frac", L.soc[z] / ZONES[z]["bat"])
            L.rows.append(dict(time_h=hr, zone=z, policy=L.name, price=tariff(hr), **r))

    def frames(self):
        return pd.DataFrame(self.base.rows + self.opt.rows)


# --------------------------------------------------------------------------
# City traffic simulation
# --------------------------------------------------------------------------
class City:
    def __init__(self, env, controller, seed=1, demand_scale=1.0, start_hour=0.0,
                 sample_s=60.0):
        self.env = env
        self.ctrl = controller
        self.rng = np.random.default_rng(seed)
        self.demand_scale = demand_scale
        self.start_hour = start_hour
        self.intersections = [Intersection(r, c) for r in range(GRID_ROWS)
                              for c in range(GRID_COLS)]
        self.vid = 0
        self.generated = 0
        self.completed = []          # finished vehicle records
        self.samples = []            # periodic state samples
        self.phase_log = []          # (time, iid, phase, state)
        self.violations = 0          # conflicting-green checks
        self.trace = None            # optional list of (event, vid, iid, heading, time) for Unity playback
        for h in HEADINGS:
            for idx in range(3):
                env.process(self.source(h, idx))
        for it in self.intersections:
            env.process(self.signal_process(it))
        env.process(self.sampler(sample_s))

    def hour(self):
        return (self.start_hour + self.env.now / 3600.0) % 24

    def at(self, r, c):
        if 0 <= r < GRID_ROWS and 0 <= c < GRID_COLS:
            return self.intersections[r * GRID_COLS + c]
        return None

    def entry_intersection(self, heading, idx):
        if heading == "S":
            return self.at(0, idx)
        if heading == "N":
            return self.at(GRID_ROWS - 1, idx)
        if heading == "E":
            return self.at(idx, 0)
        return self.at(idx, GRID_COLS - 1)

    def choose_turn(self, heading):
        u = self.rng.random()
        if u < TURN_LEFT_P:
            return LEFT[heading]
        if u < TURN_LEFT_P + TURN_RIGHT_P:
            return RIGHT[heading]
        return heading

    def source(self, heading, idx):
        """Non-homogeneous Poisson arrivals at a boundary entry (thinning)."""
        it = self.entry_intersection(heading, idx)
        lam_max = 900.0 * self.demand_scale / 3600.0
        while True:
            yield self.env.timeout(self.rng.exponential(1.0 / lam_max))
            lam = entry_rate_vph(heading, idx, self.hour()) * self.demand_scale / 3600.0
            if self.rng.random() < lam / lam_max:
                self.vid += 1
                self.generated += 1
                v = Vehicle(self.vid, self.env.now, heading, t_ready=self.env.now)
                v.next_heading = self.choose_turn(heading)
                it.queues[heading].append(v)
                if self.trace is not None:
                    self.trace.append(("A", v.vid, it.iid, heading, self.env.now))

    def signal_process(self, it: Intersection):
        """One SimPy process per intersection: control + saturation discharge."""
        yield self.env.timeout((it.iid % 3) * TICK)    # small offset per agent
        while True:
            now = self.env.now
            for h in HEADINGS:                          # vehicles reach stop line
                tr = it.transit[h]
                while tr and tr[0][0] <= now:
                    _, v = tr.popleft()
                    v.t_ready = now
                    v.next_heading = self.choose_turn(h)
                    it.queues[h].append(v)
                    if self.trace is not None:
                        self.trace.append(("A", v.vid, it.iid, h, now))
            if it.state == "Y":
                it.yellow_left -= 1
                if it.yellow_left <= 0:
                    it.state, it.phase, it.green_ticks = "G", 1 - it.phase, 0
                    self.phase_log.append((now, it.iid, it.phase, "G"))
            elif self.ctrl.decide(it, now):
                it.state, it.yellow_left = "Y", YELLOW_TICKS
                it.switches += 1
                self.phase_log.append((now, it.iid, it.phase, "Y"))
            if it.state == "G":
                g = it.green_headings()
                if set(g) & set(PHASE_HEADINGS[1 - it.phase]):
                    self.violations += 1
                for h in g:
                    self.discharge(it, h, now)
                it.green_ticks += 1
            elif hasattr(self.ctrl, "observe"):
                self.ctrl.observe(it)
            yield self.env.timeout(TICK)

    def discharge(self, it, h, now):
        q = it.queues[h]
        if not q:
            return
        v = q[0]
        nh = v.next_heading
        dr, dc = MOVE[nh]
        nxt = self.at(it.r + dr, it.c + dc)
        if nxt is not None and nxt.link_load(nh) >= LINK_CAPACITY:
            return                                      # spillback blocking
        q.popleft()
        if self.trace is not None:
            self.trace.append(("D", v.vid, it.iid, nh, now))
        v.wait += now - v.t_ready
        v.hops += 1
        v.heading = nh
        if nxt is None:
            t_exit = now + TRAVEL_TIME / 2              # leave the network
            free = v.hops * TRAVEL_TIME + TRAVEL_TIME / 2 - TRAVEL_TIME
            self.completed.append(dict(vid=v.vid, t_enter=v.t_enter, t_exit=t_exit,
                                       hops=v.hops, wait=v.wait,
                                       delay=max(0.0, t_exit - v.t_enter - free),
                                       hour=(self.start_hour + now / 3600.0) % 24))
        else:
            nxt.transit[nh].append((now + TRAVEL_TIME, v))

    def sampler(self, every):
        while True:
            hr = self.hour()
            for it in self.intersections:
                self.samples.append(dict(t=self.env.now, hour=hr, iid=it.iid,
                                         queue=it.queue_len(), approaching=it.approaching(),
                                         phase=it.phase, state=it.state,
                                         qN=len(it.queues["N"]), qS=len(it.queues["S"]),
                                         qE=len(it.queues["E"]), qW=len(it.queues["W"])))
            yield self.env.timeout(every)

    def vehicles_in_system(self):
        return sum(it.queue_len() + it.approaching() for it in self.intersections)


# --------------------------------------------------------------------------
# Experiment helpers
# --------------------------------------------------------------------------
def run_day(controller, seed=1, hours=24.0, start_hour=0.0, forecaster=None,
            with_energy=True, demand_scale=1.0, sample_s=60.0):
    env = simpy.Environment()
    city = City(env, controller, seed=seed, demand_scale=demand_scale,
                start_hour=start_hour, sample_s=sample_s)
    grid = EnergyGridManager(env, city, forecaster, seed=seed + 100) if with_energy else None
    env.run(until=hours * 3600.0)
    veh = pd.DataFrame(city.completed)
    samp = pd.DataFrame(city.samples)
    return city, grid, veh, samp


def traffic_metrics(city, veh, hours=24.0):
    return dict(controller=city.ctrl.name,
                vehicles_generated=city.generated,
                vehicles_completed=len(veh),
                avg_delay_s=veh["delay"].mean(),
                avg_wait_s=veh["wait"].mean(),
                p95_delay_s=veh["delay"].quantile(0.95),
                throughput_vph=len(veh) / hours,
                total_wait_veh_h=veh["wait"].sum() / 3600.0,
                signal_switches=sum(it.switches for it in city.intersections),
                left_in_system=city.vehicles_in_system(),
                conflict_violations=city.violations)


def energy_metrics(df):
    out = []
    for pol, g in df.groupby("policy"):
        grid_kwh = (g["grid"] * DT_H).sum()
        tot_t = g.groupby("time_h")["grid"].sum()
        load_kwh = (g["load"] * DT_H).sum()
        waste = ((g["curtail"] + g["light_waste"]) * DT_H).sum()
        cost = (g["grid"] * DT_H * g["price"]).sum()
        pv_kwh = (g["pv"] * DT_H).sum()
        out.append(dict(policy=pol, grid_import_kwh=grid_kwh, peak_grid_kw=tot_t.max(),
                        load_kwh=load_kwh, pv_generated_kwh=pv_kwh,
                        pv_curtailed_kwh=(g["curtail"] * DT_H).sum(),
                        lighting_kwh=(g["light_kw"] * DT_H).sum(),
                        lighting_waste_kwh=(g["light_waste"] * DT_H).sum(),
                        storage_losses_kwh=((g["bat_ch"] * (1 - BAT_EFF) + g["bat_dis"] * (1 / BAT_EFF - 1)) * DT_H).sum(),
                        sharing_losses_kwh=((g["share_out"] * DT_H).sum() - (g["share_in"] * DT_H).sum()),
                        energy_waste_kwh=waste, energy_cost_usd=cost,
                        solar_utilization=1 - (g["curtail"] * DT_H).sum() / pv_kwh,
                        energy_efficiency=1 - waste / (load_kwh + (g["curtail"] * DT_H).sum())))
    return pd.DataFrame(out)


def train_q_agent(episodes=240, episode_hours=2.0, seed=0, gamma=0.9, log=None):
    """Train the shared Q-table on randomized demand windows."""
    rng = np.random.default_rng(seed)
    Q = np.zeros(QLearningController.SHAPE)
    hist = []
    for ep in range(episodes):
        eps = max(0.05, 0.6 * (1 - ep / (0.7 * episodes)))
        start = float(rng.integers(0, 23))          # cover night, peak, off-peak
        scale = float(rng.uniform(0.8, 1.3))           # demand-scale randomization
        alpha = max(0.05, 0.3 * (1 - ep / episodes))
        ctrl = QLearningController(Q, alpha=alpha, gamma=gamma, epsilon=eps, seed=ep)
        city, _, veh, _ = run_day(ctrl, seed=1000 + ep, hours=episode_hours,
                                  start_hour=start, with_energy=False, demand_scale=scale)
        row = dict(episode=ep + 1, epsilon=eps, start_hour=start, demand_scale=scale,
                   avg_wait_s=veh["wait"].mean(), avg_delay_s=veh["delay"].mean(),
                   mean_td_error=float(np.mean(ctrl.td_errors)) if ctrl.td_errors else 0.0,
                   q_states_visited=int((np.abs(Q).sum(axis=-1) > 0).sum()))
        hist.append(row)
        if log:
            log(row)
    return Q, pd.DataFrame(hist)


def train_forecaster(seed=7):
    from sklearn.ensemble import GradientBoostingRegressor
    from sklearn.metrics import mean_absolute_error, r2_score
    df = generate_energy_history(days=21, seed=seed)
    feats = []
    for _, r in df.iterrows():
        feats.append(make_features(r.hour, r.day, r.zone_idx, r.density, r.lag_day))
    X = np.array(feats)
    y = df["load"].values
    train = df["day"] < 14
    model = GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                      random_state=seed)
    model.fit(X[train], y[train])
    pred = model.predict(X[~train])
    test = df[~train].copy()
    test["pred"] = pred
    naive = test["lag_day"].values
    metrics = dict(mae_kw=mean_absolute_error(test["load"], pred),
                   mape_pct=float(np.mean(np.abs(pred - test["load"]) / test["load"]) * 100),
                   r2=r2_score(test["load"], pred),
                   naive_mae_kw=mean_absolute_error(test["load"], naive),
                   n_train=int(train.sum()), n_test=int((~train).sum()))
    return model, metrics, test
