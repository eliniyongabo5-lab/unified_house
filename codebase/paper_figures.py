#!/usr/bin/env python3
"""Generate the paper's performance and physical-diagnostic figures.

All plotted quantities are read from evaluation/analysis/summary.json.
"""

from __future__ import annotations

import json
import os
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUMMARY_PATH = ROOT / "evaluation" / "analysis" / "summary.json"
OUTPUT_DIR = ROOT / "paper" / "figures"
CACHE_DIR = ROOT / ".mpl-cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import PercentFormatter  # noqa: E402


METHOD_KEYS = [
    "A_independent",
    "B_independent_prior",
    "C_shared",
    "D_shared_prior",
    "E_shared_prior_gate",
]
METHOD_LABELS = ["A", "B", "C", "D", "E"]
COLORS = ["#4477AA", "#66CCEE", "#228833", "#CCBB44", "#AA3377"]


def load_summary() -> dict:
    with SUMMARY_PATH.open(encoding="utf-8") as handle:
        return json.load(handle)


def verify_metric(metric: dict) -> None:
    """Ensure stored summaries agree with their five underlying seed values."""
    values = metric["values"]
    assert metric["n"] == len(values) == 5
    assert np.isclose(metric["mean"], statistics.mean(values), rtol=0, atol=1e-12)
    assert np.isclose(
        metric["sample_sd"], statistics.stdev(values), rtol=0, atol=1e-12
    )


def performance_figure(summary: dict) -> None:
    panels = [
        ("slope", "accuracy", "Slope accuracy (higher is better)", "Accuracy", "a"),
        ("rock", "rmse", "Rock RMSE (lower is better)", "RMSE (MPa)", "b"),
        (
            "settlement",
            "rmse",
            "Settlement RMSE (lower is better)",
            "RMSE (mm)",
            "c",
        ),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.35))
    x = np.arange(len(METHOD_KEYS))

    for ax, (domain, metric_name, title, ylabel, letter) in zip(axes, panels):
        metrics = [summary["aggregate_metrics"][m][domain][metric_name] for m in METHOD_KEYS]
        for metric in metrics:
            verify_metric(metric)
        means = np.asarray([m["mean"] for m in metrics])
        sample_sds = np.asarray([m["sample_sd"] for m in metrics])
        baseline = summary["classical_baselines"][domain][metric_name]

        ax.bar(x, means, color=COLORS, width=0.7, edgecolor="black", linewidth=0.55)
        ax.errorbar(
            x,
            means,
            yerr=sample_sds,
            fmt="none",
            ecolor="black",
            elinewidth=0.9,
            capsize=3,
            capthick=0.9,
            zorder=3,
        )
        ax.axhline(
            baseline,
            color="#333333",
            linestyle="--",
            linewidth=1.2,
            label=f"{'Logistic regression' if domain == 'slope' else 'Ridge regression'} ({baseline:.3f})",
        )
        ax.set_xticks(x, METHOD_LABELS)
        ax.set_ylabel(ylabel)
        ax.set_title(f"({letter}) {title}", fontsize=9.5)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.legend(loc="best", fontsize=7.2, frameon=False, handlelength=2.6)
        if domain == "slope":
            ax.set_ylim(0, 1.0)
        else:
            ax.set_ylim(bottom=0)

    fig.supxlabel("Method", y=0.01, fontsize=9)
    fig.tight_layout(rect=(0, 0.03, 1, 1), w_pad=1.6)
    fig.savefig(OUTPUT_DIR / "performance_comparison.pdf", bbox_inches="tight")
    plt.close(fig)


def physical_violations_figure(summary: dict) -> None:
    diagnostics = [
        ("slope", "cohesion_kPa", "Slope\ncohesion"),
        ("slope", "friction_angle_deg", "Slope\nfriction angle"),
        ("rock", "Is50_MPa", "Rock\n$\\mathit{Is}_{50}$"),
        ("settlement", "Applied Load q (kPa)", "Settlement\n$q$"),
        ("settlement", "Elastic Modulus E (MPa)", "Settlement\n$E$"),
    ]
    models = ["C_shared", "D_shared_prior", "E_shared_prior_gate"]
    labels = ["C: shared", "D: shared + prior", "E: shared + prior + gate"]
    colors = ["#228833", "#CCBB44", "#AA3377"]
    physical = summary["physical_diagnostics"]

    fig, ax = plt.subplots(figsize=(9.0, 4.15))
    x = np.arange(len(diagnostics))
    width = 0.24
    maximum = 0.0
    for model_index, (model, label, color) in enumerate(zip(models, labels, colors)):
        rates = []
        count_labels = []
        for domain, diagnostic, _ in diagnostics:
            entry = physical[model][domain][diagnostic]
            violations = entry["pooled_violations"]
            denominator = entry["pooled_denominator"]
            rate = entry["pooled_rate"]
            assert denominator > 0
            assert 0 <= violations <= denominator
            assert np.isclose(rate, violations / denominator, rtol=0, atol=1e-12)
            rates.append(rate)
            count_labels.append(f"{violations}/{denominator}")
        offset = (model_index - 1) * width
        bars = ax.bar(
            x + offset,
            rates,
            width,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.55,
        )
        maximum = max(maximum, max(rates))
        for bar, text_label in zip(bars, count_labels):
            height = bar.get_height()
            ax.annotate(
                text_label,
                (bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=6.4,
            )

    ax.set_xticks(x, [item[2] for item in diagnostics])
    ax.set_ylabel("Pooled physical violation rate")
    ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
    ax.set_ylim(0, maximum + 0.105)
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.legend(ncol=3, loc="upper center", frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "physical_violations.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = load_summary()
    performance_figure(summary)
    physical_violations_figure(summary)
    print(f"Wrote {OUTPUT_DIR / 'performance_comparison.pdf'}")
    print(f"Wrote {OUTPUT_DIR / 'physical_violations.pdf'}")


if __name__ == "__main__":
    main()
