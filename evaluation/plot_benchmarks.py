#!/usr/bin/env python3
"""Create primary-split benchmark performance and physics figures."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = HERE / "runs" / "benchmark_v3"
PRIMARY_SPLIT = 42

TASK_SPECS = {
    "slope": ("balanced_accuracy", "Balanced accuracy", True),
    "rock": ("rmse", "RMSE (MPa)", False),
    "settlement": ("rmse", "RMSE (mm)", False),
}
CLASSICAL_BY_TASK = {
    "slope": ("logistic_regression", "rbf_svc", "random_forest", "hist_gradient_boosting"),
    "rock": ("ridge", "rbf_svr", "random_forest", "hist_gradient_boosting"),
    "settlement": ("ridge", "rbf_svr", "random_forest", "hist_gradient_boosting"),
}
SHARED_MODELS = (
    "C_shared", "D_shared_prior", "E_shared_prior_gate",
    "Tuned_C_shared", "Tuned_E_shared_prior_gate",
)
MODEL_LABELS = {
    "linear": "Logistic / ridge regression",
    "rbf": "RBF support-vector model",
    "random_forest": "Random forest",
    "hist_gradient_boosting": "Histogram gradient boosting",
    "Calibrated_settlement_prior": "Calibrated settlement prior",
    "Tuned_MLP": "Tuned single-task MLP",
    "C_shared": "C · Shared",
    "D_shared_prior": "D · Shared + prior",
    "E_shared_prior_gate": "E · Shared + prior gate",
    "F_independent_prior_gate": "F · Independent + prior gate",
    "Tuned_C_shared": "Tuned C · Shared",
    "Tuned_E_shared_prior_gate": "Tuned E · Shared + prior gate",
}
DIAGNOSTICS = (
    ("slope", "cohesion_kPa", "Slope: cohesion"),
    ("slope", "friction_angle_deg", "Slope: friction angle"),
    ("rock", "Is50_MPa", "Rock: point-load index"),
    ("settlement", "Applied Load q (kPa)", "Settlement: applied load"),
    ("settlement", "Elastic Modulus E (MPa)", "Settlement: elastic modulus"),
)
PHYSICS_MODELS = (
    ("C_shared", "aggregate"),
    ("D_shared_prior", "aggregate"),
    ("E_shared_prior_gate", "aggregate"),
    ("F_independent_prior_gate", "task"),
)
COLORS = {
    "classical": "#6B7280",
    "independent": "#D97706",
    "shared": "#2563EB",
    "physics": "#059669",
}


def load_records(run_dir: Path) -> list[dict[str, Any]]:
    path = run_dir / "evaluations.json"
    if not path.is_file():
        raise FileNotFoundError(f"evaluations file does not exist: {path}")
    records = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(records, list):
        raise ValueError("evaluations.json must contain a list")
    return [row for row in records if row.get("split_seed") == PRIMARY_SPLIT]


def _selected_model_rows(records: list[dict[str, Any]], task: str) -> list[tuple[str, list[dict[str, Any]], bool]]:
    linear, rbf, random_forest, boosting = CLASSICAL_BY_TASK[task]
    specifications: list[tuple[str, str, str, bool]] = [
        ("linear", linear, "validation", True),
        ("rbf", rbf, "validation", True),
        (random_forest, random_forest, "validation", True),
        (boosting, boosting, "validation", True),
    ]
    specifications += [
        ("Calibrated_settlement_prior", "Calibrated_settlement_prior", "train_only", True),
        ("Tuned_MLP", f"Tuned_MLP_{task}", "task", False),
        *((model, model, "aggregate", False) for model in SHARED_MODELS),
        ("F_independent_prior_gate", "F_independent_prior_gate", "task", False),
    ]
    selected = []
    for label_key, model, selection, classical in specifications:
        if label_key == "Calibrated_settlement_prior" and task != "settlement":
            selected.append((label_key, [], classical))
            continue
        rows = [
            row for row in records
            if row.get("task") == task and row.get("model") == model and row.get("selection") == selection
        ]
        if not rows:
            raise ValueError(f"missing split {PRIMARY_SPLIT} result for {task}/{model}/{selection}")
        if classical:
            if len(rows) != 1 or rows[0].get("seed") is not None:
                raise ValueError(f"classical result must be one unseeded row: {task}/{model}")
        else:
            seeds = [row.get("seed") for row in rows]
            if len(rows) != 5 or len(set(seeds)) != 5 or any(seed is None for seed in seeds):
                raise ValueError(f"expected five unique training seeds for {task}/{model}/{selection}")
        selected.append((label_key, rows, classical))
    return selected


def _mean_sd(rows: list[dict[str, Any]], metric: str) -> tuple[float, float | None]:
    values = np.asarray([float(row["metrics"][metric]) for row in rows], dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"non-finite {metric} value")
    return float(values.mean()), float(values.std(ddof=1)) if len(values) > 1 else None


def performance_figure(records: list[dict[str, Any]], output: Path) -> None:
    plt.rcParams.update({
        "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 8,
        "axes.spines.top": False, "axes.spines.right": False,
    })
    fig, axes = plt.subplots(1, 3, figsize=(14.2, 6.3), constrained_layout=True)
    for panel, (ax, (task, (metric, axis_label, higher_better))) in enumerate(zip(axes, TASK_SPECS.items())):
        selected = _selected_model_rows(records, task)
        y = np.arange(len(selected))[::-1]
        for position, (label_key, rows, classical) in zip(y, selected):
            if not rows:
                ax.text(0.02, position, "N/A", transform=ax.get_yaxis_transform(),
                        color="#9CA3AF", va="center", ha="left", fontsize=7.5)
                continue
            mean, sd = _mean_sd(rows, metric)
            if classical:
                ax.scatter(mean, position, marker="D", s=35, color=COLORS["classical"], zorder=3)
            else:
                category = "independent" if label_key in {"Tuned_MLP", "F_independent_prior_gate"} else "shared"
                if "prior" in label_key.lower():
                    category = "physics"
                ax.errorbar(mean, position, xerr=sd, fmt="o", markersize=5.5, capsize=3,
                            elinewidth=1.2, color=COLORS[category], zorder=3)
        ax.set_yticks(y, [MODEL_LABELS[item[0]] for item in selected])
        ax.set_xlabel(axis_label)
        direction = "higher is better" if higher_better else "lower is better"
        ax.set_title(f"{chr(97 + panel)}) {task.title()}\n{direction}", loc="left", fontweight="bold")
        ax.grid(axis="x", color="#D1D5DB", linewidth=0.7, alpha=0.8)
        ax.set_axisbelow(True)
        if task == "slope":
            ax.set_xlim(0.0, 1.0)
    fig.suptitle("Primary split (42): held-out performance", fontsize=13, fontweight="bold")
    fig.text(0.5, -0.025, "Circles are means across five training seeds; whiskers show sample SD. Diamonds are single fits without whiskers (classical models are validation selected; the calibrated prior is train fitted).", ha="center", fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"primary_performance.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def _diagnostic_rates(records: list[dict[str, Any]], model: str, selection: str, task: str, name: str) -> tuple[float, float]:
    rows = [
        row for row in records
        if row.get("model") == model and row.get("selection") == selection and row.get("task") == task
    ]
    if len(rows) != 5 or len({row.get("seed") for row in rows}) != 5:
        raise ValueError(f"expected five diagnostic seeds for {task}/{model}/{selection}")
    rates = []
    for row in rows:
        diagnostic = row.get("diagnostics", {}).get(name)
        if not isinstance(diagnostic, dict) or diagnostic.get("rate") is None:
            raise ValueError(f"missing diagnostic {name!r} for {task}/{model}/{selection}")
        rate = float(diagnostic["rate"])
        if not math.isfinite(rate) or not 0 <= rate <= 1:
            raise ValueError(f"invalid diagnostic rate for {task}/{model}/{name}")
        rates.append(rate)
    values = np.asarray(rates)
    return float(values.mean()), float(values.std(ddof=1))


def physics_figure(records: list[dict[str, Any]], output: Path) -> None:
    fig, ax = plt.subplots(figsize=(11.2, 5.6), constrained_layout=True)
    x = np.arange(len(DIAGNOSTICS), dtype=float)
    width = 0.19
    palette = ("#2563EB", "#7C3AED", "#059669", "#D97706")
    for offset_index, ((model, selection), color) in enumerate(zip(PHYSICS_MODELS, palette)):
        means, sds = [], []
        for task, diagnostic, _ in DIAGNOSTICS:
            mean, sd = _diagnostic_rates(records, model, selection, task, diagnostic)
            means.append(mean)
            sds.append(sd)
        offset = (offset_index - 1.5) * width
        ax.bar(x + offset, means, width, yerr=sds, capsize=3, color=color, alpha=0.9,
               edgecolor="white", linewidth=0.6, label=MODEL_LABELS[model])
    ax.set_xticks(x, [label for _, _, label in DIAGNOSTICS], rotation=18, ha="right")
    ax.set_ylabel("Physical violation rate")
    ax.set_ylim(bottom=0)
    ax.set_title("Primary split (42): perturbation consistency", loc="left", fontsize=13, fontweight="bold")
    ax.grid(axis="y", color="#D1D5DB", linewidth=0.7, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.text(0.5, -0.035, "Bars show the mean violation rate across five training seeds; error bars show sample SD. C, D, and E use aggregate checkpoints; F uses task checkpoints.", ha="center", fontsize=8)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"primary_physics_violations.{suffix}", dpi=300 if suffix == "png" else None, bbox_inches="tight")
    plt.close(fig)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    records = load_records(run_dir)
    if not records:
        raise ValueError(f"no records found for primary split {PRIMARY_SPLIT}")
    output = run_dir / "analysis" / "figures"
    output.mkdir(parents=True, exist_ok=True)
    performance_figure(records, output)
    physics_figure(records, output)
    print(f"Wrote benchmark figures to {output}")


if __name__ == "__main__":
    main()
