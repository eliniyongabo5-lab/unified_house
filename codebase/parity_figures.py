#!/usr/bin/env python3
"""Create held-out parity plots from the five saved runs of each configuration."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "evaluation" / "runs" / "corrected_v1"
OUTPUT = ROOT / "paper" / "figures" / "predicted_vs_actual.pdf"
CACHE = ROOT / ".mpl-cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402


CONFIGS = [
    ("A_independent", "A: Independent"),
    ("B_independent_prior", "B: Independent + prior"),
    ("C_shared", "C: Shared"),
    ("D_shared_prior", "D: Shared + prior"),
    ("E_shared_prior_gate", "E: Shared + prior + gate"),
]
TASKS = [
    ("rock", "Rock UCS", "MPa"),
    ("settlement", "Settlement", "mm"),
]
SEEDS = range(1, 6)
SEED_COLORS = ["#4477AA", "#EE6677", "#228833", "#CCBB44", "#AA3377"]


def regression_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = truth - prediction
    return {
        "mae": float(np.mean(np.abs(residual))),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "r2": float(1.0 - np.sum(residual**2) / np.sum((truth - truth.mean()) ** 2)),
    }


def load_and_verify() -> dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]]:
    """Load predictions and verify identity, finiteness, and stored metrics."""
    loaded: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]] = {}
    references: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    maximum_error = 0.0

    for config, _ in CONFIGS:
        loaded[config] = {}
        for task, _, _ in TASKS:
            loaded[config][task] = []
            for seed in SEEDS:
                run_dir = RUNS / config / f"seed_{seed}"
                with np.load(run_dir / f"predictions_{task}.npz", allow_pickle=False) as data:
                    row_id = np.asarray(data["row_id"])
                    truth = np.asarray(data["truth"], dtype=np.float64)
                    prediction = np.asarray(data["prediction"], dtype=np.float64)

                if len(row_id) != len(np.unique(row_id)):
                    raise ValueError(f"Duplicate row IDs in {config}, seed {seed}, {task}")
                if not (np.isfinite(truth).all() and np.isfinite(prediction).all()):
                    raise ValueError(f"Non-finite value in {config}, seed {seed}, {task}")
                if task not in references:
                    references[task] = (row_id.copy(), truth.copy())
                elif not (
                    np.array_equal(row_id, references[task][0])
                    and np.array_equal(truth, references[task][1])
                ):
                    raise ValueError(f"Held-out rows or targets differ for {config}, seed {seed}, {task}")

                with (run_dir / "test_metrics.json").open(encoding="utf-8") as handle:
                    record = json.load(handle)
                saved_task = record["tasks"][task]
                if saved_task["n_test"] != len(truth):
                    raise ValueError(f"n_test mismatch for {config}, seed {seed}, {task}")
                recalculated = regression_metrics(truth, prediction)
                for metric, value in recalculated.items():
                    error = abs(value - float(saved_task["metrics"][metric]))
                    maximum_error = max(maximum_error, error)
                    if not np.isclose(value, saved_task["metrics"][metric], rtol=1e-7, atol=1e-9):
                        raise ValueError(
                            f"{metric} mismatch for {config}, seed {seed}, {task}: "
                            f"recalculated={value}, saved={saved_task['metrics'][metric]}"
                        )
                loaded[config][task].append((truth, prediction))

    print(f"Verified 50 prediction files; maximum metric error = {maximum_error:.3e}")
    return loaded


def make_figure(
    loaded: dict[str, dict[str, list[tuple[np.ndarray, np.ndarray]]]]
) -> None:
    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
    })
    fig, axes = plt.subplots(5, 2, figsize=(7.0, 12.0), constrained_layout=True)

    task_limits: dict[str, tuple[float, float]] = {}
    for task, _, _ in TASKS:
        values = []
        for config, _ in CONFIGS:
            for truth, prediction in loaded[config][task]:
                values.extend((truth, prediction))
        low = min(float(np.min(v)) for v in values)
        high = max(float(np.max(v)) for v in values)
        padding = 0.035 * (high - low)
        task_limits[task] = (min(0.0, low - padding), high + padding)

    for row, (config, config_label) in enumerate(CONFIGS):
        for col, (task, task_label, unit) in enumerate(TASKS):
            low, high = task_limits[task]
            ax = axes[row, col]
            rmses = []
            for seed_index, (truth, prediction) in enumerate(loaded[config][task]):
                rmses.append(regression_metrics(truth, prediction)["rmse"])
                ax.scatter(
                    truth,
                    prediction,
                    s=7 if task == "rock" else 11,
                    color=SEED_COLORS[seed_index],
                    alpha=0.16 if task == "rock" else 0.34,
                    edgecolors="none",
                    rasterized=True,
                )
            ax.plot([low, high], [low, high], color="#222222", linewidth=0.9, zorder=5)
            ax.set(xlim=(low, high), ylim=(low, high), aspect="equal")
            ax.grid(color="#D8D8D8", linewidth=0.45)
            ax.set_axisbelow(True)
            if row == 0:
                ax.set_title(f"{task_label} ({unit})", pad=4)
            ax.text(
                0.04,
                0.95,
                f"5 runs\nRMSE {min(rmses):.2f}\N{EN DASH}{max(rmses):.2f} {unit}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.5},
            )
            ax.set_ylabel(f"{config_label}\nPredicted ({unit})")
            if row == len(CONFIGS) - 1:
                ax.set_xlabel(f"Recorded {task_label} ({unit})")

    handles = [
        Line2D([], [], marker="o", linestyle="none", color=color, markersize=4, label=f"Seed {seed}")
        for seed, color in zip(SEEDS, SEED_COLORS)
    ]
    handles.append(Line2D([], [], color="#222222", linewidth=0.9, label="Perfect agreement"))
    fig.legend(handles=handles, loc="outside upper center", ncol=6, frameon=False)
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT, dpi=300)
    plt.close(fig)
    print(f"Wrote {OUTPUT.relative_to(ROOT)}")


def main() -> None:
    make_figure(load_and_verify())


if __name__ == "__main__":
    main()
