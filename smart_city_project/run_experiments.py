"""
Run the complete experiment pipeline:
  1. Train the Q-learning traffic-signal agent.
  2. Train the gradient-boosting day-ahead load forecaster.
  3. Evaluate Fixed-time, Rule-based actuated, and Q-learning control for
     five independent 24-hour days (random seeds) with the energy manager.
  4. Run a +20 % demand stress test.
  5. Save CSV results, figures, heatmaps, and the city animation (MP4 + GIF).

Usage:  python run_experiments.py            (about 5 minutes on a laptop)
        python run_experiments.py --quick    (smaller run for a smoke test)
"""
import argparse
import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.patches import Rectangle, Circle
import numpy as np
import pandas as pd

import smart_city_sim as sc

OUT = "outputs"
FIG = os.path.join(OUT, "figures")
DATA = os.path.join(OUT, "data")
COLORS = {"Fixed-time": "#7f7f7f", "Rule-based actuated": "#E08A1E", "Q-learning (RL)": "#1F5FA8",
          "Baseline": "#7f7f7f", "AI-optimized": "#2E8B57"}
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                     "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "figure.dpi": 150, "savefig.dpi": 300, "axes.grid": True,
                     "grid.alpha": 0.3})
LOG_LINES = []


def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG_LINES.append(line)


def save(fig, name):
    path = os.path.join(FIG, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    log(f"saved figure {path}")


# --------------------------------------------------------------------------
def fig_city_map():
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    zone_cells = {"Residential": [(0, 0), (0, 1), (1, 0)], "Commercial": [(0, 2), (1, 1), (1, 2)],
                  "Industrial": [(2, 1), (2, 2)], "Mixed-Use": [(2, 0)]}
    zc = {"Residential": "#F4E3B5", "Commercial": "#CFE0F3", "Industrial": "#E3D3EA",
          "Mixed-Use": "#D6EDD5"}
    for z, cells in zone_cells.items():
        for (r, c) in cells:
            ax.add_patch(Rectangle((c - 0.5, -r - 0.5), 1, 1, color=zc[z], zorder=0))
    for r in range(3):
        ax.plot([-0.9, 2.9], [-r, -r], color="#444", lw=3.5 if r == 1 else 1.8, zorder=1)
    for c in range(3):
        ax.plot([c, c], [0.9, -2.9], color="#444", lw=3.5 if c == 1 else 1.8, zorder=1)
    for it in range(9):
        r, c = divmod(it, 3)
        ax.add_patch(Circle((c, -r), 0.13, color="#C0392B", zorder=3))
        ax.text(c + 0.17, -r + 0.17, f"I{it}", fontsize=7, zorder=4, weight="bold")
    for z, cells in zone_cells.items():
        r, c = cells[0]
        p = sc.ZONES[z]
        ax.text(c - 0.45, -r - 0.32, f"{z}\nPV {p['pv']} kWp\nBat {p['bat']} kWh",
                fontsize=5.5, va="center", zorder=4,
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.75))
    ax.annotate("N-S arterial (col 1)", xy=(1, 0.85), xytext=(1.25, 1.05), fontsize=6,
                arrowprops=dict(arrowstyle="-", lw=0.5))
    ax.text(2.95, -1.12, "E-W arterial (row 1)", fontsize=6, ha="right", va="top")
    ax.set_xlim(-1, 3.0)
    ax.set_ylim(-3.0, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")
    save(fig, "fig01_city_map.png")


def fig_training(hist):
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.plot(hist.episode, hist.avg_wait_s, color="#9DB9DD", lw=0.8, label="Episode average wait")
    ax.plot(hist.episode, hist.avg_wait_s.rolling(20, min_periods=1).mean(), color=COLORS["Q-learning (RL)"],
            lw=1.6, label="20-episode moving average")
    ax.set_xlabel("Training episode (2 h of simulated traffic each)")
    ax.set_ylabel("Average wait per vehicle (s)")
    ax2 = ax.twinx()
    ax2.plot(hist.episode, hist.epsilon, color="#C0392B", lw=1, ls="--", label="Exploration rate")
    ax2.set_ylabel("Exploration rate", color="#C0392B")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", frameon=False)
    save(fig, "fig02_q_learning_training.png")


def fig_delay_by_hour(hourly):
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.fill_between(range(24), 0, sc.DEMAND_PROFILE * 60, color="#ddd", alpha=0.6,
                    label="Relative demand (scaled)")
    for ctrl, g in hourly.groupby("controller"):
        m = g.groupby("hour")["delay"].mean()
        s = g.groupby("hour")["delay"].std()
        ax.plot(m.index, m.values, color=COLORS[ctrl], lw=1.5, label=ctrl, marker="o", ms=2)
        ax.fill_between(m.index, m - s, m + s, color=COLORS[ctrl], alpha=0.15)
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average delay per vehicle (s)")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_ylim(0, 75)
    save(fig, "fig03_delay_by_hour.png")


def fig_traffic_bars(summary):
    order = ["Fixed-time", "Rule-based actuated", "Q-learning (RL)"]
    labels = ["Fixed", "Rule-based", "Q-learning"]
    fig, axes = plt.subplots(1, 3, figsize=(3.5, 1.9))
    for ax, (col, title) in zip(axes, [("avg_delay_s", "Mean delay (s)"),
                                       ("p95_delay_s", "P95 delay (s)"),
                                       ("throughput_vph", "Throughput (veh/h)")]):
        m = [summary.loc[o, (col, "mean")] for o in order]
        s = [summary.loc[o, (col, "std")] for o in order]
        bars = ax.bar(labels, m, yerr=s, capsize=2, color=[COLORS[o] for o in order])
        for b, v in zip(bars, m):
            ax.text(b.get_x() + b.get_width() / 2, v * 0.5 if v > 1000 else v * 1.03,
                    f"{v:.1f}" if v < 1000 else f"{v:.0f}", ha="center",
                    va="center" if v > 1000 else "bottom", fontsize=5.5,
                    rotation=90 if v > 1000 else 0, color="white" if v > 1000 else "black")
        ax.tick_params(axis="x", labelsize=5.5, rotation=30)
        ax.tick_params(axis="y", labelsize=6)
        ax.title.set_fontsize(6.5)
        ax.set_title(title)
        ax.set_ylim(0, max(m) * 1.2)
    fig.tight_layout(w_pad=0.6)
    save(fig, "fig04_traffic_metrics.png")


def fig_queue_heatmaps(samples):
    fig, axes = plt.subplots(3, 1, figsize=(3.5, 4.0), sharex=True,
                             gridspec_kw=dict(hspace=0.35))
    vmax = 0
    mats = {}
    for ctrl in ["Fixed-time", "Rule-based actuated", "Q-learning (RL)"]:
        g = samples[samples.controller == ctrl]
        mat = g.assign(h=g.hour.astype(int)).pivot_table(index="iid", columns="h", values="queue",
                                                         aggfunc="mean")
        mats[ctrl] = mat
        vmax = max(vmax, mat.values.max())
    for ax, (ctrl, mat) in zip(axes, mats.items()):
        im = ax.imshow(mat.values, aspect="auto", cmap="inferno_r", vmin=0, vmax=vmax)
        ax.set_title(ctrl)
        ax.set_xticks(range(0, 24, 3))
        ax.set_yticks(range(9))
        ax.set_yticklabels([f"I{i}" for i in range(9)], fontsize=6)
        ax.set_ylabel("Intersection")
        ax.grid(False)
    axes[-1].set_xlabel("Hour of day")
    cb = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
    cb.set_label("Mean queued vehicles")
    save(fig, "fig05_queue_heatmaps.png")


def fig_energy_heatmaps(edf):
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.1), sharex=True,
                             gridspec_kw=dict(hspace=0.35))
    vmax = 0
    mats = {}
    for pol in ["Baseline", "AI-optimized"]:
        g = edf[edf.policy == pol]
        mat = g.assign(h=g.time_h.astype(int)).pivot_table(index="zone", columns="h", values="grid",
                                                           aggfunc="mean")
        mat = mat.loc[list(sc.ZONES)]
        mats[pol] = mat
        vmax = max(vmax, mat.values.max())
    for ax, (pol, mat) in zip(axes, mats.items()):
        im = ax.imshow(mat.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=vmax)
        ax.set_title(f"{pol} grid import by zone")
        ax.set_yticks(range(4))
        ax.set_yticklabels(list(sc.ZONES))
        ax.grid(False)
    axes[1].set_xlabel("Hour of day")
    axes[1].set_xticks(range(0, 24, 3))
    cb = fig.colorbar(im, ax=axes, fraction=0.04, pad=0.02)
    cb.set_label("Grid import (kW)")
    save(fig, "fig06_energy_heatmaps.png")


def fig_dispatch(edf):
    fig, ax = plt.subplots(figsize=(3.5, 2.4))
    for pol, ls in [("Baseline", "--"), ("AI-optimized", "-")]:
        g = edf[edf.policy == pol].groupby("time_h")[["grid", "load", "pv", "bat_dis"]].sum()
        ax.plot(g.index, g.grid, color=COLORS[pol], lw=1.5, ls=ls, label=f"Grid import - {pol}")
    g = edf[edf.policy == "AI-optimized"].groupby("time_h")[["load", "pv"]].sum()
    ax.plot(g.index, g.load, color="black", lw=0.9, label="City demand")
    ax.fill_between(g.index, 0, g.pv, color="#F4D03F", alpha=0.45, label="Solar generation")
    ax.axvspan(14, 20, color="#C0392B", alpha=0.07, label="On-peak price window")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Power (kW)")
    ax.set_xticks(range(0, 25, 3))
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    ax.set_ylim(0, g.load.max() * 1.35)
    save(fig, "fig07_energy_dispatch.png")


def fig_forecast(test, fm):
    z = "Commercial"
    g = test[(test.zone == z)].copy()
    g["t"] = (g.day - g.day.min()) * 24 + g.hour
    g = g[g.t < 72]
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    ax.plot(g.t, g.load, color="black", lw=1, label="Actual load")
    ax.plot(g.t, g.pred, color=COLORS["AI-optimized"], lw=1.2, ls="--", label="Day-ahead forecast")
    ax.set_xlabel("Hours into held-out test week")
    ax.set_ylabel(f"{z} zone load (kW)")
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.set_ylim(g.load.min() * 0.8, g.load.max() * 1.25)
    ax.set_title(f"Held-out test, all zones: MAE {fm['mae_kw']:.1f} kW, MAPE {fm['mape_pct']:.1f}%, "
                 f"R$^2$ = {fm['r2']:.3f}")
    save(fig, "fig08_load_forecast.png")


def fig_density_energy(samples_rl, edf):
    dens = samples_rl.assign(h=samples_rl.hour.astype(int)).groupby(["t"]).agg(
        hour=("hour", "first"), veh=("queue", "sum"), app=("approaching", "sum"))
    dens["inside"] = dens.veh + dens.app
    hourly_d = dens.groupby(dens.hour.astype(int))["inside"].mean()
    e = edf[edf.policy == "AI-optimized"].groupby(edf.time_h.astype(int))[["load", "light_kw"]].sum() / 4
    eb = edf[edf.policy == "Baseline"].groupby(edf.time_h.astype(int))[["light_kw"]].sum() / 4
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 2.9), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.3, 1]))
    ax = axes[0]
    ax.bar(hourly_d.index, hourly_d.values, color="#9DB9DD", label="Vehicles in network")
    ax.set_ylabel("Vehicles in network")
    ax2 = ax.twinx()
    ax2.plot(e.index, e.load, color="#C0392B", lw=1.5, marker="o", ms=2, label="City electricity demand")
    ax2.set_ylabel("Demand (kW)", color="#C0392B")
    ax2.grid(False)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left", fontsize=6)
    ax = axes[1]
    ax.step(eb.index, eb.light_kw, where="mid", color=COLORS["Baseline"], ls="--", label="Scheduled lighting")
    ax.step(e.index, e.light_kw, where="mid", color=COLORS["AI-optimized"], label="Traffic-adaptive lighting")
    ax.set_ylabel("Street lighting (kW)")
    ax.set_xlabel("Hour of day")
    ax.set_xticks(range(0, 24, 3))
    ax.legend(frameon=False, fontsize=6, loc="upper center")
    save(fig, "fig09_density_vs_energy.png")


def fig_energy_bars(esum):
    cols = [("grid_import_kwh", "Grid import (MWh)", 1e-3), ("peak_grid_kw", "Peak demand (kW)", 1),
            ("energy_waste_kwh", "Energy wastage (kWh)", 1), ("energy_cost_usd", "Energy cost (USD)", 1)]
    fig, axes = plt.subplots(2, 2, figsize=(3.5, 3.0))
    for ax, (c, t, k) in zip(axes.flat, cols):
        m = [esum.loc[p, (c, "mean")] * k for p in ["Baseline", "AI-optimized"]]
        s = [esum.loc[p, (c, "std")] * k for p in ["Baseline", "AI-optimized"]]
        bars = ax.bar(["Baseline", "AI-optimized"], m, yerr=s, capsize=3,
                      color=[COLORS["Baseline"], COLORS["AI-optimized"]])
        for b, v in zip(bars, m):
            ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,.1f}" if v < 100 else f"{v:,.0f}",
                    ha="center", va="bottom", fontsize=6.5)
        ax.set_title(t)
        ax.set_ylim(0, max(m) * 1.25)
    fig.tight_layout()
    save(fig, "fig10_energy_metrics.png")


# --------------------------------------------------------------------------
def make_animation(samples, edf, phase_log, path_mp4, path_gif, step_min=10):
    """Animated city view: queues at each intersection, signal phase, zone load."""
    zone_cells = {"Residential": [(0, 0), (0, 1), (1, 0)], "Commercial": [(0, 2), (1, 1), (1, 2)],
                  "Industrial": [(2, 1), (2, 2)], "Mixed-Use": [(2, 0)]}
    opt = edf[edf.policy == "AI-optimized"]
    zone_max = opt.groupby("zone")["grid"].max().to_dict()
    times = np.arange(0, 24 * 3600, step_min * 60)
    s_idx = samples.set_index(["t", "iid"])
    city_q = samples.groupby("t")["queue"].sum()
    city_grid = opt.groupby("time_h")["grid"].sum()

    fig = plt.figure(figsize=(8, 4.5))
    axm = fig.add_axes([0.02, 0.05, 0.5, 0.88])
    axq = fig.add_axes([0.6, 0.57, 0.37, 0.33])
    axe = fig.add_axes([0.6, 0.1, 0.37, 0.33])
    cmap = plt.get_cmap("YlOrRd")
    patches = {}
    for z, cells in zone_cells.items():
        for (r, c) in cells:
            p = Rectangle((c - 0.5, -r - 0.5), 1, 1, color=cmap(0.1), zorder=0)
            axm.add_patch(p)
            patches.setdefault(z, []).append(p)
        r, c = cells[0]
        axm.text(c - 0.46, -r + 0.38, z, fontsize=7, zorder=5)
    for r in range(3):
        axm.plot([-0.9, 2.9], [-r, -r], color="#555", lw=4 if r == 1 else 2, zorder=1)
    for c in range(3):
        axm.plot([c, c], [0.9, -2.9], color="#555", lw=4 if c == 1 else 2, zorder=1)
    bars_ns, bars_ew, lights_ns, lights_ew = [], [], [], []
    for it in range(9):
        r, c = divmod(it, 3)
        bars_ns.append(axm.add_patch(Rectangle((c - 0.05, -r + 0.12), 0.1, 0, color="#1F5FA8", zorder=3)))
        bars_ew.append(axm.add_patch(Rectangle((c - 0.12, -r - 0.05), 0, 0.1, color="#8E44AD", zorder=3)))
        lights_ns.append(axm.add_patch(Circle((c + 0.12, -r + 0.12), 0.06, color="green", zorder=4)))
        lights_ew.append(axm.add_patch(Circle((c - 0.12, -r - 0.12), 0.06, color="red", zorder=4)))
    title = axm.set_title("", fontsize=10)
    axm.set_xlim(-1, 3)
    axm.set_ylim(-3, 1)
    axm.set_aspect("equal")
    axm.axis("off")
    axm.text(-0.95, -2.95, "Blue bar = N-S queue, purple bar = E-W queue; dots = signal state.\n"
             "Zone color = grid import (darker = higher). Controller: Q-learning, energy: AI-optimized.",
             fontsize=6)
    axq.plot(city_q.index / 3600, city_q.values, color="#ccc", lw=0.8)
    lq, = axq.plot([], [], color="#1F5FA8", lw=1.4)
    axq.set_title("Vehicles queued in the network", fontsize=8)
    axq.set_xlim(0, 24)
    axq.set_xticks(range(0, 25, 6))
    axe.plot(city_grid.index, city_grid.values, color="#ccc", lw=0.8)
    le, = axe.plot([], [], color="#2E8B57", lw=1.4)
    axe.set_title("City grid import (kW)", fontsize=8)
    axe.set_xlim(0, 24)
    axe.set_xticks(range(0, 25, 6))
    axe.set_xlabel("Hour of day")

    def update(k):
        t = times[k]
        ts = samples.t.values[np.searchsorted(samples.t.values, t)] if t <= samples.t.max() else samples.t.max()
        hr = t / 3600
        for it in range(9):
            row = s_idx.loc[(ts, it)]
            r, c = divmod(it, 3)
            qns = min(row.qN + row.qS, 30) / 30 * 0.75
            qew = min(row.qE + row.qW, 30) / 30 * 0.75
            bars_ns[it].set_height(qns)
            bars_ew[it].set_width(-qew)
            g_ns = row.phase == 0 and row.state == "G"
            g_ew = row.phase == 1 and row.state == "G"
            lights_ns[it].set_color("green" if g_ns else ("gold" if row.state == "Y" and row.phase == 0 else "red"))
            lights_ew[it].set_color("green" if g_ew else ("gold" if row.state == "Y" and row.phase == 1 else "red"))
        e_now = opt[np.isclose(opt.time_h, np.floor(hr * 4) / 4)]
        for z, ps in patches.items():
            v = e_now[e_now.zone == z]["grid"]
            frac = float(v.iloc[0]) / zone_max[z] if len(v) and zone_max[z] > 0 else 0
            for p in ps:
                p.set_color(cmap(0.08 + 0.8 * frac))
        title.set_text(f"Smart-city simulation  -  {int(hr):02d}:{int((hr % 1) * 60):02d}")
        m = city_q.index <= t
        lq.set_data(city_q.index[m] / 3600, city_q.values[m])
        me = city_grid.index <= hr
        le.set_data(city_grid.index[me], city_grid.values[me])
        return []

    ani = animation.FuncAnimation(fig, update, frames=len(times), blit=False)
    # Use the ffmpeg program bundled with the imageio-ffmpeg package, so a separate
    # ffmpeg installation is not needed (Windows usually does not have one).
    try:
        import imageio_ffmpeg
        plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as exc:   # package missing: fall back to an ffmpeg on the PATH
        log(f"imageio-ffmpeg not available ({exc}); trying ffmpeg on the PATH")
    try:
        ani.save(path_mp4, writer=animation.FFMpegWriter(fps=8, bitrate=1800), dpi=110)
        log(f"saved animation {path_mp4}")
    except (FileNotFoundError, OSError, RuntimeError) as exc:
        log(f"MP4 skipped because ffmpeg could not be started ({exc}); the GIF is still written")
    ani.save(path_gif, writer=animation.PillowWriter(fps=8), dpi=60)
    log(f"saved animation {path_gif}")
    for hh in (3, 8, 17):     # still frames for the report
        update(int(hh * 60 / step_min))
        fig.savefig(os.path.join(FIG, f"anim_frame_{hh:02d}h.png"), dpi=200)
    plt.close(fig)


# --------------------------------------------------------------------------
def main(quick=False):
    global OUT, FIG, DATA
    if quick:   # keep the full-run results used in the report untouched
        OUT = "outputs_quick"
        FIG = os.path.join(OUT, "figures")
        DATA = os.path.join(OUT, "data")
        log(f"quick mode: results go to {OUT}/ (the full results in outputs/ are not changed)")
    os.makedirs(FIG, exist_ok=True)
    os.makedirs(DATA, exist_ok=True)
    t0 = time.time()
    seeds = [1, 2] if quick else [1, 2, 3, 4, 5]
    episodes = 40 if quick else 240

    log("Step 1/5: building city map")
    fig_city_map()

    log(f"Step 2/5: training Q-learning agent ({episodes} episodes)")
    Q, hist = sc.train_q_agent(episodes=episodes,
                               log=lambda r: (r["episode"] % 20 == 0) and log(
                                   f"  episode {r['episode']:3d}  eps={r['epsilon']:.2f}  "
                                   f"avg wait={r['avg_wait_s']:.1f} s  states={r['q_states_visited']}"))
    np.save(os.path.join(DATA, "q_table.npy"), Q)
    hist.to_csv(os.path.join(DATA, "q_training_history.csv"), index=False)
    fig_training(hist)

    log("Step 3/5: training day-ahead load forecaster (gradient boosting)")
    model, fm, test = sc.train_forecaster()
    log(f"  forecaster MAE={fm['mae_kw']:.2f} kW  MAPE={fm['mape_pct']:.2f}%  R2={fm['r2']:.4f}  "
        f"naive MAE={fm['naive_mae_kw']:.2f} kW")
    test.to_csv(os.path.join(DATA, "forecast_test_week.csv"), index=False)
    fig_forecast(test, fm)

    log(f"Step 4/5: evaluating controllers on {len(seeds)} independent 24-hour days")
    rows, hourly, samples_all, energy_rows, edf_rl, samples_rl, phase_rl = [], [], [], [], None, None, None
    for seed in seeds:
        for ctrl in [sc.FixedTimeController(), sc.ActuatedController(),
                     sc.QLearningController(Q.copy(), alpha=0.0, epsilon=0.0)]:
            city, grid, veh, samp = sc.run_day(ctrl, seed=seed, forecaster=model)
            m = sc.traffic_metrics(city, veh)
            m["seed"] = seed
            rows.append(m)
            log(f"  seed {seed}  {ctrl.name:20s} delay={m['avg_delay_s']:.2f} s  "
                f"p95={m['p95_delay_s']:.1f} s  throughput={m['throughput_vph']:.0f} veh/h  "
                f"conflicts={m['conflict_violations']}")
            hv = veh.assign(hour=veh.hour.astype(int)).groupby("hour")["delay"].mean().reset_index()
            hv["controller"], hv["seed"] = ctrl.name, seed
            hourly.append(hv)
            if seed == seeds[0]:
                samp["controller"] = ctrl.name
                samples_all.append(samp)
            if isinstance(ctrl, sc.QLearningController):
                edf = grid.frames()
                em = sc.energy_metrics(edf)
                em["seed"] = seed
                energy_rows.append(em)
                if seed == seeds[0]:
                    edf_rl, samples_rl, phase_rl = edf, samp, city.phase_log
                    veh.to_csv(os.path.join(DATA, "vehicles_qlearning_seed1.csv"), index=False)
    tm = pd.DataFrame(rows)
    tm.to_csv(os.path.join(DATA, "traffic_metrics_by_seed.csv"), index=False)
    tsum = tm.groupby("controller")[["avg_delay_s", "p95_delay_s", "throughput_vph", "total_wait_veh_h",
                                     "signal_switches", "vehicles_completed"]].agg(["mean", "std"])
    tsum.to_csv(os.path.join(DATA, "traffic_metrics_summary.csv"))
    hourly = pd.concat(hourly)
    hourly.to_csv(os.path.join(DATA, "delay_by_hour.csv"), index=False)
    samples_all = pd.concat(samples_all)
    samples_all.to_csv(os.path.join(DATA, "intersection_samples_seed1.csv"), index=False)
    em = pd.concat(energy_rows)
    em.to_csv(os.path.join(DATA, "energy_metrics_by_seed.csv"), index=False)
    esum = em.groupby("policy").agg(["mean", "std"])
    esum.to_csv(os.path.join(DATA, "energy_metrics_summary.csv"))
    edf_rl.to_csv(os.path.join(DATA, "energy_dispatch_seed1.csv"), index=False)

    log("Step 5/5: stress test (+20% demand) and figures")
    stress = []
    for seed in seeds[:3]:
        for ctrl in [sc.FixedTimeController(), sc.ActuatedController(),
                     sc.QLearningController(Q.copy(), alpha=0.0, epsilon=0.0)]:
            city, _, veh, _ = sc.run_day(ctrl, seed=seed + 50, with_energy=False, demand_scale=1.2)
            m = sc.traffic_metrics(city, veh)
            m["seed"] = seed + 50
            stress.append(m)
    stress = pd.DataFrame(stress)
    stress.to_csv(os.path.join(DATA, "stress_test_120pct.csv"), index=False)
    ssum = stress.groupby("controller")[["avg_delay_s", "p95_delay_s", "throughput_vph",
                                         "left_in_system"]].mean()
    log("  stress test means:\n" + ssum.round(2).to_string())

    fig_delay_by_hour(hourly)
    fig_traffic_bars(tsum)
    fig_queue_heatmaps(samples_all)
    fig_energy_heatmaps(edf_rl)
    fig_dispatch(edf_rl)
    fig_density_energy(samples_rl, edf_rl)
    fig_energy_bars(esum)
    make_animation(samples_rl, edf_rl, phase_rl, os.path.join(OUT, "city_simulation.mp4"),
                   os.path.join(OUT, "city_simulation.gif"))

    summary = dict(traffic=tsum.round(3).to_dict(), energy=esum.round(3).to_dict(),
                   stress=ssum.round(3).to_dict(), forecaster=fm, runtime_s=time.time() - t0)
    with open(os.path.join(DATA, "summary.json"), "w") as f:
        json.dump(json.loads(pd.Series(summary).to_json()), f, indent=2)
    log("Traffic summary (mean over seeds):\n" + tsum.round(2).to_string())
    log("Energy summary (mean over seeds):\n" +
        esum.xs("mean", axis=1, level=1).round(2).T.to_string())
    log(f"Finished in {time.time() - t0:.1f} s")
    with open(os.path.join(OUT, "run_log.txt"), "w") as f:
        f.write("\n".join(LOG_LINES))


def replot():
    """Regenerate the static figures from the saved CSV files (no re-simulation)."""
    os.makedirs(FIG, exist_ok=True)
    fig_city_map()
    fig_training(pd.read_csv(os.path.join(DATA, "q_training_history.csv")))
    hourly = pd.read_csv(os.path.join(DATA, "delay_by_hour.csv"))
    fig_delay_by_hour(hourly)
    tm = pd.read_csv(os.path.join(DATA, "traffic_metrics_by_seed.csv"))
    fig_traffic_bars(tm.groupby("controller")[["avg_delay_s", "p95_delay_s", "throughput_vph"]].agg(["mean", "std"]))
    fig_queue_heatmaps(pd.read_csv(os.path.join(DATA, "intersection_samples_seed1.csv")))
    edf = pd.read_csv(os.path.join(DATA, "energy_dispatch_seed1.csv"))
    fig_energy_heatmaps(edf)
    fig_dispatch(edf)
    s = pd.read_csv(os.path.join(DATA, "intersection_samples_seed1.csv"))
    fig_density_energy(s[s.controller == "Q-learning (RL)"], edf)
    em = pd.read_csv(os.path.join(DATA, "energy_metrics_by_seed.csv"))
    fig_energy_bars(em.groupby("policy").agg(["mean", "std"]))
    test = pd.read_csv(os.path.join(DATA, "forecast_test_week.csv"))
    with open(os.path.join(DATA, "summary.json")) as f:
        fig_forecast(test, json.load(f)["forecaster"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--figures-only", action="store_true",
                    help="redraw figures from outputs/data without re-running the simulation")
    a = ap.parse_args()
    replot() if a.figures_only else main(a.quick)
