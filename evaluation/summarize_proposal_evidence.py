"""Create a compact, reproducible evidence summary from audited benchmarks."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from report_benchmarks import aggregate_records, audit_records


def table(headers, rows):
    return ["| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
            *["| " + " | ".join(map(str, row)) + " |" for row in rows], ""]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path(__file__).resolve().parent / "runs/benchmark_v3")
    args = parser.parse_args()
    root = args.run_dir.resolve()
    records, audit = audit_records(json.loads((root / "evaluations.json").read_text()))
    aggregates = aggregate_records(records)
    lookup = {(r["split_seed"], r["task"], r["model"], r["selection"]): r for r in aggregates}
    protocol = json.loads((root / "protocol.json").read_text())
    splits = protocol["split_seeds"]
    primary = splits[0]
    out = root / "analysis"
    out.mkdir(exist_ok=True)

    def value(split, task, model, selection, metric=None):
        metric = metric or ("balanced_accuracy" if task == "slope" else "rmse")
        row = lookup[(split, task, model, selection)]
        avg, sd = row[f"{metric}_mean"], row[f"{metric}_sample_sd"]
        return f"{avg:.3f}" + (f" ± {sd:.3f}" if sd is not None else "")

    lines = ["# Evidence for the Unified House proposal", "",
             "The completed experiments support a proposal about when parameter sharing and analytical guidance help heterogeneous geotechnical prediction. They do not establish that the complete Unified House model is consistently the strongest algorithm.", "",
             "All model choices use training/validation data. Test results are exploratory because the underlying data were inspected previously. Reported neural SDs describe initialization variability, not uncertainty across independent projects.", "",
             "## Predictive performance across partitions", "",
             "Independent models use task-specific checkpoints; shared models use one aggregate checkpoint. Classical controls are single selected fits. The full report also contains the controlled aggregate-versus-task comparisons.", ""]
    for task in ("slope", "rock", "settlement"):
        metric = "balanced accuracy (higher is better)" if task == "slope" else "RMSE in " + ("MPa" if task == "rock" else "mm") + " (lower is better)"
        lines += [f"### {task.title()}: {metric}", ""]
        controls = [("Logistic regression" if task == "slope" else "Ridge regression", "logistic_regression" if task == "slope" else "ridge", "validation"),
                    ("RBF support-vector model", "rbf_svc" if task == "slope" else "rbf_svr", "validation"),
                    ("Random forest", "random_forest", "validation"),
                    ("Histogram gradient boosting", "hist_gradient_boosting", "validation"),
                    ("Tuned single-task MLP", f"Tuned_MLP_{task}", "task"),
                    ("C shared", "C_shared", "aggregate"),
                    ("E shared with prior gate", "E_shared_prior_gate", "aggregate"),
                    ("F independent with prior gate", "F_independent_prior_gate", "task"),
                    ("Tuned C shared", "Tuned_C_shared", "aggregate"),
                    ("Tuned E shared with prior gate", "Tuned_E_shared_prior_gate", "aggregate")]
        if task == "settlement":
            controls.append(("Calibrated analytical prior", "Calibrated_settlement_prior", "train_only"))
        lines += table(["Model", *[f"Split {s}" for s in splits]],
                       [[label, *[value(s, task, model, selector) for s in splits]] for label, model, selector in controls])

    lines += ["## Isolating sharing and checkpoint choice", "",
              "The following mean differences pair identical training seeds and compare aggregate checkpoints on the same split. Lower RMSE differences favor the model on the left; higher balanced-accuracy differences favor the model on the left. These descriptive differences are not significance tests.", ""]
    comparisons = []
    for split in splits:
        for task in ("slope", "rock", "settlement"):
            metric = "balanced_accuracy" if task == "slope" else "rmse"
            for left, right in (("C_shared", "A_independent"), ("E_shared_prior_gate", "F_independent_prior_gate")):
                rows_left = {r["seed"]: r for r in records if (r["split_seed"], r["task"], r["model"], r["selection"]) == (split, task, left, "aggregate")}
                rows_right = {r["seed"]: r for r in records if (r["split_seed"], r["task"], r["model"], r["selection"]) == (split, task, right, "aggregate")}
                differences = [rows_left[s]["metrics"][metric] - rows_right[s]["metrics"][metric] for s in sorted(rows_left.keys() & rows_right.keys())]
                comparisons.append([split, task, left[0] + " − " + right[0], f"{np.mean(differences):.3f} ± {np.std(differences, ddof=1):.3f}"])
    lines += table(["Split", "Task", "Contrast", "Mean difference ± SD"], comparisons)

    lines += ["## Task-pair experiments", "", f"Split {primary}; shared aggregate checkpoints throughout. Metrics are slope balanced accuracy, rock RMSE (MPa), and settlement RMSE (mm). Missing cells mean that the task is absent from that pair.", ""]
    pair_names = sorted({r["model"] for r in records if r["model"].startswith("Pair_")})
    lines += table(["Model", "Slope", "Rock", "Settlement"],
                   [[model, *[value(primary, task, model, "aggregate") if (primary, task, model, "aggregate") in lookup else "—" for task in ("slope", "rock", "settlement")]] for model in pair_names])

    lines += ["## Fixed prior-weight sensitivity", "", f"Split {primary}; predeclared weights, not selected by test performance.", ""]
    weight_names = [("0.10", "D_shared_weight_0.1"), ("0.25", "D_shared_prior"), ("0.50", "D_shared_weight_0.5"), ("0.75", "D_shared_weight_0.75")]
    lines += table(["Prior weight", "Slope balanced accuracy", "Rock RMSE (MPa)", "Settlement RMSE (mm)"],
                   [[w, *[value(primary, t, name, "aggregate") for t in ("slope", "rock", "settlement")]] for w, name in weight_names])

    lines += ["## Settlement calibration", "", "The affine formula below is fitted on valid training priors only. A close match to a complex model would limit the evidence that sharing is responsible for the settlement gain. It does not prove how the targets were generated.", ""]
    calibrations = []
    for split in splits:
        calibration = json.loads((root / f"split_{split}/settlement_prior_calibration.json").read_text())
        calibrations.append([split, f"{calibration['coefficient']:.4f}", f"{calibration['intercept']:.4f}", value(split, "settlement", "Calibrated_settlement_prior", "train_only")])
    lines += table(["Split", "Prior multiplier", "Intercept (mm)", "Test RMSE (mm)"], calibrations)

    identity = ["split_seed", "model", "seed", "task", "selection"]
    subgroup_rows, diagnostic_rows = [], []
    for record in records:
        base = {k: record[k] for k in identity}
        for subgroup, info in record.get("subgroups", {}).items():
            subgroup_rows.append({**base, "subgroup": subgroup, "n": info["n"], **info["metrics"]})
        for feature, info in record.get("diagnostics", {}).items():
            diagnostic_rows.append({**base, "feature": feature, **info})
    for filename, rows in (("subgroups.csv", subgroup_rows), ("directional_diagnostics.csv", diagnostic_rows)):
        fields = list(dict.fromkeys(k for row in rows for k in row))
        with (out / filename).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    neural = [json.loads(p.read_text()) for p in root.glob("split_*/neural/**/training.json")]
    tuning = [json.loads(p.read_text()) for p in root.glob("split_*/tuning/**/training.json")]
    classical = [row for p in root.glob("split_*/classical/tuning_records.json") for row in json.loads(p.read_text())]
    timing = json.loads((root / "timing_all.json").read_text())
    counts = {"final_neural_fits": len(neural), "neural_tuning_fits": len(tuning),
              "classical_candidate_fits": len(classical), "failed_classical_candidates": sum(r["status"] != "ok" for r in classical),
              "evaluation_records": len(records), "elapsed_seconds": timing["seconds"], "audit": audit}
    (out / "experiment_summary.json").write_text(json.dumps(counts, indent=2) + "\n")
    lines += ["## Completion and remaining limitations", "",
              f"Completed {len(neural)} final neural fits, {len(tuning)} neural tuning fits and {len(classical)} classical candidate fits. The complete run took {timing['seconds'] / 60:.2f} minutes on the local CPU. All {len(records)} saved prediction records passed independent metric recomputation.", "",
              "The proposal can now describe a concrete benchmark and documented preliminary results. The main remaining research needs are:", "",
              "- Trace the settlement records to their original source or acquire a documented measured dataset.",
              "- Verify what the two slope grouping codes mean; repeated training partitions do not provide new independent slope test groups.",
              "- Validate on independent projects before making engineering-generalization claims.",
              "- Frame the contribution around identifying useful and harmful sharing, improving prior fusion and testing physical consistency. Broad algorithmic superiority is not established.", "",
              "`subgroups.csv` provides source, missingness, prior-valid and extrapolation results. `directional_diagnostics.csv` provides case counts and violation rates for every evaluated model. `report.md` contains all checkpoint-specific metrics and paired comparisons.", ""]
    (out / "proposal_evidence.md").write_text("\n".join(lines))
    print(f"Wrote {out / 'proposal_evidence.md'}")


if __name__ == "__main__":
    main()
