#!/usr/bin/env python3
"""Audit benchmark predictions and write descriptive benchmark reports.

The unit of replication is a training seed. Results are summarized separately
for every split seed, task, model, and selection strategy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


HERE = Path(__file__).resolve().parent
DEFAULT_RUN_DIR = HERE / "runs" / "benchmark_v3"
TASK_METRICS = {
    "slope": ("accuracy", "balanced_accuracy", "f1", "roc_auc", "failure_recall", "brier"),
    "rock": ("mae", "rmse", "r2"),
    "settlement": ("mae", "rmse", "r2"),
}
REPORT_METRICS = {
    "slope": ("balanced_accuracy", "roc_auc"),
    "rock": ("rmse", "mae", "r2"),
    "settlement": ("rmse", "mae", "r2"),
}
COMPARISONS = (
    ("C_shared", "A_independent", ("aggregate",)),
    ("E_shared_prior_gate", "F_independent_prior_gate", None),
    ("E_shared_prior_gate", "D_shared_prior", None),
    ("F_independent_prior_gate", "B_independent_prior", None),
)


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be finite")
    return number


def recompute(task: str, truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    truth = np.asarray(truth).reshape(-1)
    prediction = np.asarray(prediction).reshape(-1)
    if task == "slope":
        numeric_truth = truth.astype(float)
        if not np.all(np.isin(numeric_truth, (0.0, 1.0))):
            raise ValueError("slope truth must contain only 0 and 1")
        y = numeric_truth.astype(int)
        p = prediction.astype(float)
        if np.any((p < 0) | (p > 1)):
            raise ValueError("slope predictions must be probabilities in [0, 1]")
        pred = (p >= 0.5).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        tn = int(((pred == 0) & (y == 0)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        if not (tp + fn) or not (tn + fp):
            raise ValueError("slope test data must contain both classes")
        pos, neg = p[y == 1], p[y == 0]
        auc = ((pos[:, None] > neg).sum() + 0.5 * (pos[:, None] == neg).sum()) / (pos.size * neg.size)
        return {
            "accuracy": float((pred == y).mean()),
            "balanced_accuracy": float((tp / (tp + fn) + tn / (tn + fp)) / 2),
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0,
            "roc_auc": float(auc),
            "failure_recall": float(tn / (tn + fp)),
            "brier": float(np.square(p - y).mean()),
        }
    residual = prediction.astype(float) - truth.astype(float)
    ss_tot = float(np.square(truth - truth.mean()).sum())
    return {
        "mae": float(np.abs(residual).mean()),
        "rmse": float(np.sqrt(np.square(residual).mean())),
        "r2": float(1.0 - np.square(residual).sum() / ss_tot) if ss_tot else math.nan,
    }


def _sample_sd(values: Iterable[float]) -> float | None:
    array = np.asarray(list(values), dtype=float)
    return float(array.std(ddof=1)) if array.size > 1 else None


def _fmt(mean: float, sd: float | None, metric: str) -> str:
    digits = 4 if metric in {"accuracy", "balanced_accuracy", "f1", "roc_auc", "failure_recall", "brier", "r2"} else 3
    return f"{mean:.{digits}f} ± {sd:.{digits}f}" if sd is not None else f"{mean:.{digits}f} (n=1)"


def _markdown_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    return [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


def audit_records(records: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError("evaluations.json must contain a JSON list")
    audited: list[dict[str, Any]] = []
    references: dict[tuple[int, str], tuple[np.ndarray, np.ndarray]] = {}
    identities: set[tuple[int, str, str, str, int | None]] = set()
    audit: dict[str, Any] = {
        "status": "pass",
        "record_count": len(records),
        "prediction_file_count": 0,
        "metric_recalculation_max_abs_error": 0.0,
        "issues": [],
    }
    for index, raw in enumerate(records):
        label = f"record {index}"
        if not isinstance(raw, dict):
            raise ValueError(f"{label} must be an object")
        missing = {"split_seed", "model", "seed", "task", "selection", "n_test", "metrics", "prediction_path"} - raw.keys()
        if missing:
            raise ValueError(f"{label} missing fields: {', '.join(sorted(missing))}")
        split_seed = raw["split_seed"]
        seed = raw["seed"]
        n_test = raw["n_test"]
        if isinstance(split_seed, bool) or not isinstance(split_seed, int):
            raise ValueError(f"{label} split_seed must be an integer")
        if seed is not None and (isinstance(seed, bool) or not isinstance(seed, int)):
            raise ValueError(f"{label} seed must be an integer or null")
        if isinstance(n_test, bool) or not isinstance(n_test, int) or n_test <= 0:
            raise ValueError(f"{label} n_test must be a positive integer")
        task, model, selection = raw["task"], raw["model"], raw["selection"]
        if task not in TASK_METRICS:
            raise ValueError(f"{label} has unsupported task {task!r}")
        if not isinstance(model, str) or not model or not isinstance(selection, str) or not selection:
            raise ValueError(f"{label} model and selection must be non-empty strings")
        identity = (split_seed, task, model, selection, seed)
        if identity in identities:
            raise ValueError(f"duplicate evaluation identity: {identity}")
        identities.add(identity)
        path = Path(raw["prediction_path"])
        if not path.is_absolute():
            raise ValueError(f"{label} prediction_path must be absolute: {path}")
        if not path.is_file():
            raise ValueError(f"{label} prediction file does not exist: {path}")
        with np.load(path, allow_pickle=False) as data:
            absent = {"row_id", "truth", "prediction"} - set(data.files)
            if absent:
                raise ValueError(f"{path} missing arrays: {', '.join(sorted(absent))}")
            row_id = np.asarray(data["row_id"]).reshape(-1)
            truth = np.asarray(data["truth"]).reshape(-1)
            prediction = np.asarray(data["prediction"]).reshape(-1)
        audit["prediction_file_count"] += 1
        if not (len(row_id) == len(truth) == len(prediction) == n_test):
            raise ValueError(f"{label} n_test/array length mismatch")
        if len(np.unique(row_id)) != n_test:
            raise ValueError(f"{label} contains duplicate row IDs")
        if not np.isfinite(truth).all() or not np.isfinite(prediction).all():
            raise ValueError(f"{label} contains non-finite truth or predictions")
        reference_key = (split_seed, task)
        if reference_key in references:
            ref_id, ref_truth = references[reference_key]
            if not np.array_equal(row_id, ref_id) or not np.array_equal(truth, ref_truth):
                raise ValueError(f"mismatched row IDs or truth within split {split_seed}, task {task}")
        else:
            references[reference_key] = (row_id.copy(), truth.copy())
        saved_metrics = raw["metrics"]
        if not isinstance(saved_metrics, dict):
            raise ValueError(f"{label} metrics must be an object")
        recalculated = recompute(task, truth, prediction)
        clean_metrics: dict[str, float] = {}
        for metric in TASK_METRICS[task]:
            if metric not in saved_metrics:
                raise ValueError(f"{label} missing metric {metric}")
            saved = _finite_number(saved_metrics[metric], f"{label} metric {metric}")
            calculated = recalculated[metric]
            if not math.isfinite(calculated):
                raise ValueError(f"{label} cannot recompute finite {metric}")
            error = abs(saved - calculated)
            audit["metric_recalculation_max_abs_error"] = max(audit["metric_recalculation_max_abs_error"], error)
            if error > 1e-6:
                raise ValueError(f"{label} metric mismatch for {metric}: saved={saved:.12g}, recalculated={calculated:.12g}")
            clean_metrics[metric] = saved
        audited.append({**raw, "metrics": clean_metrics})
    audit["groups_checked"] = len(references)
    return audited, audit


def aggregate_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[int, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["split_seed"], record["task"], record["model"], record["selection"])].append(record)
    output = []
    for key in sorted(groups):
        split_seed, task, model, selection = key
        rows = groups[key]
        item: dict[str, Any] = {"split_seed": split_seed, "task": task, "model": model, "selection": selection, "n_runs": len(rows)}
        for metric in TASK_METRICS[task]:
            values = [row["metrics"][metric] for row in rows]
            item[f"{metric}_mean"] = float(np.mean(values))
            item[f"{metric}_sample_sd"] = _sample_sd(values)
        output.append(item)
    return output


def paired_differences(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lookup = {(r["split_seed"], r["task"], r["model"], r["selection"], r["seed"]): r for r in records}
    output: list[dict[str, Any]] = []
    split_tasks_selections = sorted({(r["split_seed"], r["task"], r["selection"]) for r in records})
    for left, right, allowed_selections in COMPARISONS:
        for split_seed, task, selection in split_tasks_selections:
            if allowed_selections is not None and selection not in allowed_selections:
                continue
            left_seeds = {r["seed"] for r in records if (r["split_seed"], r["task"], r["model"], r["selection"]) == (split_seed, task, left, selection)}
            right_seeds = {r["seed"] for r in records if (r["split_seed"], r["task"], r["model"], r["selection"]) == (split_seed, task, right, selection)}
            common = sorted(left_seeds & right_seeds, key=lambda x: (-1 if x is None else x))
            if not common:
                continue
            for metric in TASK_METRICS[task]:
                differences = [
                    lookup[(split_seed, task, left, selection, seed)]["metrics"][metric]
                    - lookup[(split_seed, task, right, selection, seed)]["metrics"][metric]
                    for seed in common
                ]
                output.append({
                    "contrast": f"{left} − {right}", "split_seed": split_seed, "task": task,
                    "selection": selection, "metric": metric, "n_pairs": len(differences),
                    "mean_difference": float(np.mean(differences)), "sample_sd_difference": _sample_sd(differences),
                })
    split_model_tasks = sorted({(r["split_seed"], r["model"], r["task"]) for r in records})
    for split_seed, model, task in split_model_tasks:
        task_seeds = {
            r["seed"] for r in records
            if (r["split_seed"], r["model"], r["task"], r["selection"]) == (split_seed, model, task, "task")
        }
        aggregate_seeds = {
            r["seed"] for r in records
            if (r["split_seed"], r["model"], r["task"], r["selection"]) == (split_seed, model, task, "aggregate")
        }
        common = sorted(task_seeds & aggregate_seeds, key=lambda x: (-1 if x is None else x))
        if not common:
            continue
        for metric in TASK_METRICS[task]:
            differences = [
                lookup[(split_seed, task, model, "task", seed)]["metrics"][metric]
                - lookup[(split_seed, task, model, "aggregate", seed)]["metrics"][metric]
                for seed in common
            ]
            output.append({
                "contrast": f"{model}: task − aggregate", "split_seed": split_seed, "task": task,
                "selection": "task − aggregate", "metric": metric, "n_pairs": len(differences),
                "mean_difference": float(np.mean(differences)), "sample_sd_difference": _sample_sd(differences),
            })
    return output


def write_metrics_csv(path: Path, aggregates: list[dict[str, Any]]) -> None:
    metric_columns = [f"{metric}_{suffix}" for metric in dict.fromkeys(sum((list(v) for v in TASK_METRICS.values()), [])) for suffix in ("mean", "sample_sd")]
    fields = ["split_seed", "task", "model", "selection", "n_runs", *metric_columns]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in aggregates:
            writer.writerow(row)


def write_report(path: Path, records: list[dict[str, Any]], aggregates: list[dict[str, Any]], paired: list[dict[str, Any]]) -> None:
    lines = [
        "# Expanded benchmark report", "",
        "Results are descriptive. Each mean and sample SD is calculated across training seeds within one split, task, model, and selection strategy. Test rows, training seeds, and split seeds are not pooled as independent observations, and no significance tests are reported.", "",
        "## Interpretation caveats", "",
        "- These benchmark test sets are reused for exploratory model comparison, so reported performance is exploratory rather than a final untouched-test estimate.",
        "- The slope task uses a fixed test group across split seeds; its split-seed results therefore do not represent independent test samples.",
        "- Settlement data provenance is unknown, which limits claims about generalization and external validity.", "",
        "- For shared models, `selection=task` rows are separate task-specific checkpoint snapshots and cannot be combined into one deployable unified network. Independent models use separate task networks, so their task-selected snapshots can validly be combined.", "",
    ]
    for task in TASK_METRICS:
        lines += [f"## {task.title()}", ""]
        task_splits = sorted({row["split_seed"] for row in aggregates if row["task"] == task})
        for split_seed in task_splits:
            lines += [f"### Split seed {split_seed}", ""]
            rows = []
            for item in aggregates:
                if item["task"] != task or item["split_seed"] != split_seed:
                    continue
                cells = [item["model"], item["selection"], str(item["n_runs"])]
                for metric in REPORT_METRICS[task]:
                    cells.append(_fmt(item[f"{metric}_mean"], item[f"{metric}_sample_sd"], metric))
                rows.append(cells)
            lines += _markdown_table(["Model", "Selection", "Runs", *REPORT_METRICS[task]], rows) + [""]
    lines += ["## Paired training-seed differences", "", "Model differences are left model minus right model, paired only where split seed, task, selection strategy, and training seed match. Checkpoint contrasts are task-selected minus aggregate-selected results for the same model, task, split, and training seed. All values are descriptive means and sample SDs; no p-values or significance tests are used.", ""]
    paired_rows = []
    for item in paired:
        if item["metric"] not in REPORT_METRICS[item["task"]]:
            continue
        paired_rows.append([
            item["contrast"], str(item["split_seed"]), item["task"], item["selection"], item["metric"],
            str(item["n_pairs"]), _fmt(item["mean_difference"], item["sample_sd_difference"], item["metric"]),
        ])
    if paired_rows:
        lines += _markdown_table(["Contrast", "Split", "Task", "Selection", "Metric", "Pairs", "Mean difference ± sample SD"], paired_rows) + [""]
    else:
        lines += ["No complete requested model/seed pairs were available.", ""]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN_DIR)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.resolve()
    source = run_dir / "evaluations.json"
    if not source.is_file():
        parser.error(f"evaluations file does not exist: {source}")
    analysis_dir = run_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    audit_path = analysis_dir / "audit.json"
    try:
        records, audit = audit_records(json.loads(source.read_text(encoding="utf-8")))
    except Exception as exc:
        audit_path.write_text(json.dumps({"status": "fail", "source": str(source), "error": str(exc)}, indent=2) + "\n", encoding="utf-8")
        raise
    aggregates = aggregate_records(records)
    paired = paired_differences(records)
    audit.update({"source": str(source), "aggregate_group_count": len(aggregates), "paired_summary_count": len(paired)})
    audit_path.write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    write_metrics_csv(analysis_dir / "metrics.csv", aggregates)
    write_report(analysis_dir / "report.md", records, aggregates, paired)


if __name__ == "__main__":
    main()
