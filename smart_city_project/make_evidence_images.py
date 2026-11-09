import re
"""Render the console logs (run log and pytest results) as terminal-style PNG images."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def terminal_png(lines, path, title, width=7.0):
    h = 0.16 * len(lines) + 0.5
    fig = plt.figure(figsize=(width, h))
    fig.patch.set_facecolor("#1E1E1E")
    fig.text(0.01, 1 - 0.18 / h, title, color="#9CDCFE", family="monospace", fontsize=7.5,
             va="top", weight="bold")
    for i, ln in enumerate(lines):
        col = "#6A9955" if "PASSED" in ln or "passed" in ln else "#D4D4D4"
        fig.text(0.01, 1 - (0.42 + 0.16 * i) / h, ln, color=col, family="monospace",
                 fontsize=6.2, va="top")
    fig.savefig(path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


if __name__ == "__main__":
    t = [re.sub(r"\x1b\[[0-9;]*m", "", l).rstrip("\n")[:118] for l in open("outputs/pytest_results.txt")
         if l.strip() and not l.startswith(("cachedir", "plugins", "rootdir"))]
    terminal_png(t, "outputs/figures/evidence_pytest.png", "$ python -m pytest -v tests")
    run = [l.rstrip("\n")[:118] for l in open("outputs/run_log.txt")]
    keep = [l for l in run if l.startswith("[") and ("seed" in l or "Step" in l or "forecaster" in l or "episode" in l
                               or "Finished" in l) and "summary" not in l]
    terminal_png(keep, "outputs/figures/evidence_run_log.png", "$ python run_experiments.py")
    print("evidence images written")
