#!/usr/bin/env python3
"""Audit and summarize the corrected five-configuration evaluation.

Run from any directory. Requires NumPy. Outputs are written to
``evaluation/analysis/summary.json`` and ``paper/generated_tables.tex``.
No model fitting or test-set selection is performed here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np


HERE = Path(__file__).resolve().parent
RUNS = HERE / "runs" / "corrected_v1"
OUT = HERE / "analysis"
CONFIGS = [
    "A_independent",
    "B_independent_prior",
    "C_shared",
    "D_shared_prior",
    "E_shared_prior_gate",
]
TASKS = ["slope", "rock", "settlement"]
METRICS = {
    "slope": ["accuracy", "balanced_accuracy", "f1", "roc_auc"],
    "rock": ["mae", "rmse", "r2"],
    "settlement": ["mae", "rmse", "r2"],
}
SEEDS = [1, 2, 3, 4, 5]
T_CRIT_95_DF4 = 2.7764451051977987


def sample_summary(values: list[float]) -> dict[str, Any]:
    a = np.asarray(values, dtype=float)
    return {
        "values": [float(x) for x in a],
        "n": int(a.size),
        "mean": float(a.mean()),
        "sample_sd": float(a.std(ddof=1)) if a.size > 1 else None,
        "min": float(a.min()),
        "max": float(a.max()),
    }


def paired_summary(values: list[float]) -> dict[str, Any]:
    out = sample_summary(values)
    se = out["sample_sd"] / math.sqrt(out["n"])
    half = T_CRIT_95_DF4 * se
    out.update({
        "standard_error": float(se),
        "ci95_t": [float(out["mean"] - half), float(out["mean"] + half)],
        "note": "Descriptive paired-seed difference and t interval; n=5 is too small for strong inferential claims.",
    })
    return out


def recompute(task: str, truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    if task == "slope":
        y = truth.astype(int)
        p = prediction.astype(float)
        pred = (p >= 0.5).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        tpr = tp / (tp + fn) if tp + fn else math.nan
        tnr = tn / (tn + fp) if tn + fp else math.nan
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        # Mann-Whitney form of AUC, including half credit for ties.
        pos, neg = p[y == 1], p[y == 0]
        auc = float(((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum()) / (pos.size * neg.size))
        return {"accuracy": float((pred == y).mean()), "balanced_accuracy": float((tpr + tnr) / 2), "f1": float(f1), "roc_auc": auc}
    residual = prediction.astype(float) - truth.astype(float)
    ss_res = float(np.square(residual).sum())
    ss_tot = float(np.square(truth - truth.mean()).sum())
    return {
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "r2": float(1.0 - ss_res / ss_tot),
    }


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def fmt(mean: float, sd: float, metric: str) -> str:
    digits = 3 if metric in {"accuracy", "balanced_accuracy", "f1", "roc_auc", "r2"} else 2
    return f"{mean:.{digits}f} $\\pm$ {sd:.{digits}f}"


def main() -> None:
    metrics: dict[str, dict[str, dict[str, list[float]]]] = {
        c: {t: {m: [] for m in METRICS[t]} for t in TASKS} for c in CONFIGS
    }
    diagnostics: dict[str, dict[str, dict[str, list[dict[str, float]]]]] = {
        c: {t: {} for t in TASKS} for c in CONFIGS
    }
    gates: dict[str, dict[str, list[dict[str, float]]]] = {c: {t: [] for t in TASKS} for c in CONFIGS}
    reference: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    prior_reference: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    audit: dict[str, Any] = {
        "expected_metric_json_count": 25,
        "metric_json_count": 0,
        "prediction_file_count": 0,
        "shared_test_rows_and_truth": True,
        "metric_recalculation_max_abs_error": 0.0,
        "issues": [],
    }

    for config in CONFIGS:
        for seed in SEEDS:
            metric_path = RUNS / config / f"seed_{seed}" / "test_metrics.json"
            if not metric_path.exists():
                audit["issues"].append(f"Missing {metric_path.relative_to(RUNS)}")
                continue
            record = json.loads(metric_path.read_text())
            audit["metric_json_count"] += 1
            if record.get("config") != config or record.get("seed") != seed:
                audit["issues"].append(f"Manifest identity mismatch: {metric_path.relative_to(RUNS)}")
            for task in TASKS:
                task_record = record["tasks"][task]
                pred_path = metric_path.parent / f"predictions_{task}.npz"
                with np.load(pred_path, allow_pickle=False) as pred:
                    audit["prediction_file_count"] += 1
                    row_id = pred["row_id"]
                    truth = pred["truth"]
                    prediction = pred["prediction"]
                    key = task
                    if key not in reference:
                        reference[key] = (row_id.copy(), truth.copy())
                    elif not (np.array_equal(row_id, reference[key][0]) and np.array_equal(truth, reference[key][1])):
                        audit["shared_test_rows_and_truth"] = False
                        audit["issues"].append(f"Test rows/targets differ: {config}/seed_{seed}/{task}")
                    if len(row_id) != task_record["n_test"] or len(np.unique(row_id)) != len(row_id):
                        audit["issues"].append(f"Count or duplicate row ID problem: {config}/seed_{seed}/{task}")
                    if not all(np.isfinite(x).all() for x in (truth, prediction)):
                        audit["issues"].append(f"Non-finite truth/prediction: {config}/seed_{seed}/{task}")
                    recalculated = recompute(task, truth, prediction)
                    for metric in METRICS[task]:
                        saved = float(task_record["metrics"][metric])
                        metrics[config][task][metric].append(saved)
                        error = abs(saved - recalculated[metric])
                        audit["metric_recalculation_max_abs_error"] = max(audit["metric_recalculation_max_abs_error"], error)
                        if error > 1e-6:
                            audit["issues"].append(f"Metric mismatch ({error:.3g}): {config}/seed_{seed}/{task}/{metric}")
                    valid = pred["prior_valid"].astype(bool)
                    prior = pred["prior"].astype(float)
                    if task not in prior_reference:
                        prior_reference[task] = (prior.copy(), valid.copy())
                    elif not (np.array_equal(valid, prior_reference[task][1]) and np.allclose(prior, prior_reference[task][0], equal_nan=True)):
                        audit["issues"].append(f"Analytical prior/mask differs: {config}/seed_{seed}/{task}")
                    gate = pred["gate_weight"].astype(float)[valid]
                    gates[config][task].append({
                        "seed": seed,
                        "n_valid": int(valid.sum()),
                        "mean": float(gate.mean()) if gate.size else math.nan,
                        "sample_sd_across_cases": float(gate.std(ddof=1)) if gate.size > 1 else 0.0,
                        "min": float(gate.min()) if gate.size else math.nan,
                        "max": float(gate.max()) if gate.size else math.nan,
                        "saved_mean": float(task_record["mean_gate_weight_valid"]),
                    })
                    if gate.size and abs(gate.mean() - float(task_record["mean_gate_weight_valid"])) > 1e-6:
                        audit["issues"].append(f"Gate mean mismatch: {config}/seed_{seed}/{task}")
                for diagnostic, item in task_record["diagnostics"].items():
                    diagnostics[config][task].setdefault(diagnostic, []).append({
                        "seed": seed, "n": int(item["n"]), "violations": int(item["violations"]), "rate": float(item["rate"])
                    })
                    if item["n"] == 0 or abs(item["violations"] / item["n"] - item["rate"]) > 1e-12:
                        audit["issues"].append(f"Diagnostic denominator/rate mismatch: {config}/seed_{seed}/{task}/{diagnostic}")

    aggregate = {
        c: {t: {m: sample_summary(metrics[c][t][m]) for m in METRICS[t]} for t in TASKS} for c in CONFIGS
    }
    paired: dict[str, Any] = {}
    for label, left, right in [("E_minus_D", "E_shared_prior_gate", "D_shared_prior"), ("C_minus_A", "C_shared", "A_independent")]:
        paired[label] = {}
        for task in TASKS:
            paired[label][task] = {}
            for metric in METRICS[task]:
                diffs = np.asarray(metrics[left][task][metric]) - np.asarray(metrics[right][task][metric])
                paired[label][task][metric] = paired_summary(diffs.tolist())

    diag_out: dict[str, Any] = {}
    for config in CONFIGS:
        diag_out[config] = {}
        for task in TASKS:
            diag_out[config][task] = {}
            for name, rows in diagnostics[config][task].items():
                total_v = sum(r["violations"] for r in rows)
                total_n = sum(r["n"] for r in rows)
                diag_out[config][task][name] = {
                    "per_seed": rows,
                    "pooled_violations": total_v,
                    "pooled_denominator": total_n,
                    "pooled_rate": total_v / total_n,
                    "per_seed_rate": sample_summary([r["rate"] for r in rows]),
                    "note": "Pooled counts repeat the same eligible test cases across five trained seeds; denominator is prediction-case evaluations, not unique cases.",
                }

    gate_out: dict[str, Any] = {}
    for config in CONFIGS:
        gate_out[config] = {}
        for task in TASKS:
            rows = gates[config][task]
            gate_out[config][task] = {
                "per_seed_valid_cases": rows,
                "mean_gate_across_seeds": sample_summary([r["mean"] for r in rows]),
            }

    baselines = json.loads((RUNS / "baselines.json").read_text())
    analytical_prior: dict[str, Any] = {}
    for task in TASKS:
        truth = reference[task][1]
        prior, valid = prior_reference[task]
        analytical_prior[task] = {
            "n_valid": int(valid.sum()),
            "n_test": int(valid.size),
            "metrics_on_valid_cases": recompute(task, truth[valid], prior[valid]),
            "note": "Calculated only where the analytical prior is defined; this denominator differs from the full-test neural and classical metrics.",
        }
    observations = []
    for task in TASKS:
        for metric in METRICS[task]:
            b = float(baselines[task][metric])
            vals = {c: aggregate[c][task][metric]["mean"] for c in CONFIGS}
            higher = metric not in {"mae", "rmse"}
            best_config = max(vals, key=vals.get) if higher else min(vals, key=vals.get)
            best_value = vals[best_config]
            if (higher and b > best_value) or (not higher and b < best_value):
                observations.append(f"The classical {baselines[task]['model']} baseline outperforms every neural configuration on {task} {metric} ({b:.6g} versus best neural mean {best_value:.6g}, {best_config}).")
    observations += [
        "B_independent_prior settlement is seed-sensitive: seed 2 has MAE 2.29566 and RMSE 3.07608, while the other four seeds have MAE 0.742263-0.845137 and RMSE 1.04355-1.15305.",
        "E_shared_prior_gate slope ROC-AUC is visibly bimodal across these five runs (0.687822, 0.893189, 0.869453, 0.862745, 0.695046), so its mean has substantial seed uncertainty.",
        "The rock test maximum UCS is 560.31 MPa versus 318.2 MPa in training; evaluation at the upper end therefore measures extrapolation.",
        "Analytical-prior metrics use only prior-valid cases (slope 71/89, rock 267/608, settlement 75/75) and must not be compared as if they used the full test denominators.",
    ]
    for comparison in ("E_minus_D", "C_minus_A"):
        for task in TASKS:
            for metric, item in paired[comparison][task].items():
                lo, hi = item["ci95_t"]
                if lo <= 0 <= hi:
                    observations.append(f"{comparison} {task} {metric}: the descriptive 95% paired t interval spans zero ({lo:.6g}, {hi:.6g}); five seeds do not support a stable directional claim.")

    if audit["metric_json_count"] != 25:
        audit["issues"].append(f"Expected 25 metric JSONs, found {audit['metric_json_count']}")
    audit["status"] = "pass" if not audit["issues"] else "issues_found"
    splits = json.loads((RUNS / "prepared" / "splits.json").read_text())
    with np.load(RUNS / "prepared" / "rock.npz", allow_pickle=False) as rock:
        train_idx = np.asarray(splits["tasks"]["rock"]["indices"]["train"], dtype=int)
        rock_range_note = {
            "train_ucs_max_mpa": float(rock["y"][train_idx].max()),
            "test_ucs_max_mpa": float(reference["rock"][1].max()),
            "note": "The rock test target range extends substantially beyond the training target maximum; high-end prediction is extrapolation.",
        }
    result = {
        "source": str(RUNS.relative_to(HERE)),
        "methods": {
            "replicates": "Five matched training seeds (1-5); summaries use arithmetic mean and sample SD (ddof=1).",
            "paired_differences": "E-D isolates learned gate within shared+prior models; C-A compares shared versus independent without physics. Differences are left minus right by matched seed.",
            "uncertainty": "95% t intervals use df=4 and t*=2.7764. They are descriptive because there are only five training seeds; no p-values are reported.",
        },
        "audit": audit,
        "test_sets": {t: {"n": int(len(reference[t][0])), "row_ids": reference[t][0].astype(int).tolist()} for t in TASKS},
        "aggregate_metrics": aggregate,
        "paired_differences": paired,
        "physical_diagnostics": diag_out,
        "gate_weights": gate_out,
        "classical_baselines": baselines,
        "analytical_prior_valid_cases": analytical_prior,
        "rock_target_range_anomaly": rock_range_note,
        "evidence_notes": observations,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")

    from render_paper_tables import render
    render(result)
    print(f"Wrote {OUT / 'summary.json'}")
    print(f"Wrote {HERE.parent / 'paper' / 'generated_tables.tex'}")
    print(f"Audit: {audit['status']}; {audit['metric_json_count']} metric JSONs, {audit['prediction_file_count']} prediction files")


if __name__ == "__main__":
    main()
