"""Versioned, resumable proposal experiments; never modifies corrected_v1.

Selection uses train/validation only. All test scoring is a final, separate phase.
The test sets were inspected in the earlier paper, so this is an exploratory
extension, not a new untouched confirmatory evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import sklearn
import torch
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, brier_score_loss,
                             f1_score, mean_absolute_error, mean_squared_error,
                             r2_score, recall_score, roc_auc_score)

import revised_data as rd
import revised_experiments as legacy
from benchmark_neural import BASE_SPEC, TUNING_GRID, load, train

ROOT = Path(__file__).resolve().parent.parent
VARIANTS = {
    "A_independent": dict(shared=False, physics=False, gate=False),
    "B_independent_prior": dict(shared=False, physics=True, gate=False),
    "C_shared": dict(shared=True, physics=False, gate=False),
    "D_shared_prior": dict(shared=True, physics=True, gate=False),
    "E_shared_prior_gate": dict(shared=True, physics=True, gate=True),
    "F_independent_prior_gate": dict(shared=False, physics=True, gate=True),
}


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def historical_hashes():
    return {str(p.relative_to(ROOT)): digest(p)
            for p in sorted((ROOT / "evaluation/runs/corrected_v1").rglob("*")) if p.is_file()}


def validate_output(output, protected):
    output, protected = Path(output).resolve(), Path(protected).resolve()
    if output == protected or protected in output.parents:
        raise ValueError("Refusing to write inside original paper results")
    if output.exists():
        if not output.is_dir():
            raise ValueError("Output must be a directory")
        if not (output / "protocol.json").exists() and any(output.iterdir()):
            raise ValueError("Nonempty output has no protocol; use a new empty output directory")


def prepare(seed):
    data = rd.load_raw()
    rd.SPLIT_SEED = seed
    manifest = rd.make_splits(data)
    transforms = rd.fit_transforms(data)
    legacy.TRANSFORMS = transforms
    arrays = legacy.tensors(data)
    return data, manifest, transforms, arrays


def metrics(task, truth, pred):
    if not np.isfinite(pred).all():
        raise FloatingPointError("Nonfinite predictions")
    if task == "slope":
        labels = pred >= .5
        return {"accuracy": float(accuracy_score(truth, labels)),
                "balanced_accuracy": float(balanced_accuracy_score(truth, labels)),
                "f1": float(f1_score(truth, labels, zero_division=0)),
                "roc_auc": float(roc_auc_score(truth, pred)) if len(np.unique(truth)) == 2 else None,
                "failure_recall": float(recall_score(truth, labels, pos_label=0, zero_division=0)),
                "brier": float(brier_score_loss(truth, pred))}
    return {"mae": float(mean_absolute_error(truth, pred)),
            "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
            "r2": float(r2_score(truth, pred)) if len(truth) > 1 else None}


def fit_neural(data, arrays, name, spec, seed, directory):
    folder = directory / name / f"seed_{seed}"
    metadata = folder / "training.json"
    if metadata.exists():
        info = json.loads(metadata.read_text())
        if info["spec"] != spec or info["seed"] != seed or info["tasks"] != list(data):
            raise ValueError(f"Incompatible resume: {folder}")
        if not all((folder / f"{key}.pt").exists() for key in ["aggregate", *data]):
            raise ValueError(f"Incomplete checkpoint: {folder}")
    else:
        _, info = train(data, arrays, spec, seed, folder)
    return {"model": name, "seed": seed, "directory": str(folder.resolve()), "tasks": list(data),
            "spec": spec, "seconds": info["seconds"]}


def tune_neural(data, arrays, name, base, directory):
    """Two tuning seeds distinct from final five seeds; no test predictions."""
    rows = []
    for i, candidate in enumerate(TUNING_GRID):
        spec = {**BASE_SPEC, **base, **candidate}
        values = []
        for seed in (101, 102):
            record = fit_neural(data, arrays, f"{name}/candidate_{i}", spec, seed, directory)
            if len(data) == 1:
                task = next(iter(data))
                model = load(data, record["directory"], task)
                t, a = data[task], arrays[task]
                idx = t.split["val"]
                pred, _, _ = legacy.predict(model, task, t, t.x[idx], a["prior"][idx].numpy(), t.prior_valid[idx])
                m = metrics(task, t.target[idx], pred)
                value = -m["balanced_accuracy"] if task == "slope" else m["rmse"]
            else:
                info = json.loads((Path(record["directory"]) / "training.json").read_text())
                value = info["best_validation_loss"]["aggregate"]
            values.append(value)
        rows.append({"candidate": i, "spec": spec, "values": values, "mean_selection_loss": float(np.mean(values))})
    selected = min(rows, key=lambda row: row["mean_selection_loss"])
    write_json(directory / name / "selection.json", {"candidates": rows, "selected": selected,
               "criterion": "negative balanced accuracy (slope) or RMSE (regression)" if len(data) == 1 else "equal average validation task loss"})
    return selected["spec"]


def fit_split(seed, primary_seed, training_seeds, output):
    from benchmark_classical import fit_baselines
    data, manifest, transforms, arrays = prepare(seed)
    directory = output / f"split_{seed}"
    rd.save_prepared(data, manifest, transforms, directory / "prepared")
    classical_manifest = directory / "classical" / "models.json"
    if classical_manifest.exists():
        classical = json.loads(classical_manifest.read_text())
        for record in classical:
            saved = Path(record["model_path"]).resolve()
            if not saved.is_file() or not saved.is_relative_to((directory / "classical").resolve()):
                raise ValueError(f"Missing or out-of-directory classical model: {saved}")
    else:
        classical = fit_baselines(data, directory / "classical", seed=seed)
        write_json(classical_manifest, classical)
    records = []
    for name, options in VARIANTS.items():
        for training_seed in training_seeds:
            records.append(fit_neural(data, arrays, name, {**BASE_SPEC, **options}, training_seed, directory / "neural"))
        print(f"split {seed}: {name} fitted", flush=True)
    # Tuned task-specific MLPs test whether the original independent networks
    # were weak controls. Tune shared C/E with the same six-candidate budget.
    for task in data:
        subset = {task: data[task]}
        name = f"Tuned_MLP_{task}"
        spec = tune_neural(subset, arrays, name, {"shared": False}, directory / "tuning")
        for training_seed in training_seeds:
            records.append(fit_neural(subset, arrays, name, spec, training_seed, directory / "neural"))
    for source_name in ("C_shared", "E_shared_prior_gate"):
        name = f"Tuned_{source_name}"
        spec = tune_neural(data, arrays, name, VARIANTS[source_name], directory / "tuning")
        for training_seed in training_seeds:
            records.append(fit_neural(data, arrays, name, spec, training_seed, directory / "neural"))
    if seed == primary_seed:
        # Same architecture and optimizer as controls, isolating the changed factor.
        for pair in itertools.combinations(data, 2):
            subset = {task: data[task] for task in pair}
            for config in ("C_shared", "E_shared_prior_gate"):
                name = f"Pair_{'_'.join(pair)}_{config}"
                for training_seed in training_seeds:
                    records.append(fit_neural(subset, arrays, name, {**BASE_SPEC, **VARIANTS[config]}, training_seed, directory / "neural"))
        for weight in (.1, .5, .75):
            name = f"D_shared_weight_{weight}"
            for training_seed in training_seeds:
                records.append(fit_neural(data, arrays, name,
                    {**BASE_SPEC, **VARIANTS["D_shared_prior"], "prior_weight": weight}, training_seed, directory / "neural"))
    write_json(directory / "fitted.json", {"classical": classical, "neural": records})
    print(f"split {seed}: fitting complete ({len(records)} neural final fits)", flush=True)


def subgroup_metrics(task, t, idx, pred):
    groups = {"prior_valid": t.prior_valid[idx], "prior_missing": ~t.prior_valid[idx],
              "complete_predictors": np.isfinite(t.raw[idx]).all(axis=1),
              "missing_predictors": ~np.isfinite(t.raw[idx]).all(axis=1)}
    if task == "rock":
        train_y = t.target[t.split["train"]]
        groups["above_training_max"] = t.target[idx] > train_y.max()
        groups["within_training_target_range"] = (t.target[idx] >= train_y.min()) & (t.target[idx] <= train_y.max())
        for group in sorted(set(t.groups[idx]), key=str):
            groups[f"source_{group}"] = t.groups[idx] == group
    return {key: {"n": int(mask.sum()), "metrics": metrics(task, t.target[idx][mask], pred[mask])}
            for key, mask in groups.items() if mask.sum() >= 2 and
            (task != "slope" or len(np.unique(t.target[idx][mask])) == 2)}


def perturbation_diagnostics(task, t, predictor):
    specs = {"slope": [(1, "multiply", 1), (2, "add", 1)],
             "rock": [(3, "multiply", 1)],
             "settlement": [(4, "multiply", 1), (5, "multiply", -1)]}[task]
    results = {}
    idx = t.split["test"]
    for column, mode, direction in specs:
        eligible = idx[t.prior_valid[idx] & np.isfinite(t.raw[idx, column])]
        if not len(eligible):
            continue
        x, prior = legacy.perturb(t, task, eligible, column, mode)
        original_prior = t.prior[eligible]
        if task == "slope":
            tr = legacy.TRANSFORMS[task]
            original_prior = 1 / (1 + np.exp(-np.clip(tr["fos_logit_coefficient"] * np.log(np.maximum(original_prior, .01)) + tr["fos_logit_intercept"], -30, 30)))
        base = predictor(t.x[eligible], original_prior, np.ones(len(eligible), dtype=bool))
        shifted = predictor(x, prior, np.ones(len(eligible), dtype=bool))
        nbad = int((direction * (shifted - base) < -1e-6).sum())
        results[t.columns[column]] = {"n": len(eligible), "violations": nbad, "rate": nbad / len(eligible)}
    return results


def evaluate_split(seed, output):
    data, _, _, arrays = prepare(seed)
    directory = output / f"split_{seed}"
    fitted = json.loads((directory / "fitted.json").read_text())
    records = []

    def emit(task, name, training_seed, selector, pred, predictor, directory_pred):
        t = data[task]
        idx = t.split["test"]
        directory_pred.mkdir(parents=True, exist_ok=True)
        path = directory_pred / f"{task}_{selector}.npz"
        np.savez_compressed(path, row_id=t.row_id[idx], truth=t.target[idx], prediction=pred,
                            prior_valid=t.prior_valid[idx])
        records.append({"split_seed": seed, "model": name, "seed": training_seed, "task": task,
                        "selection": selector, "n_test": len(idx), "metrics": metrics(task, t.target[idx], pred),
                        "prediction_path": str(path.resolve()), "subgroups": subgroup_metrics(task, t, idx, pred),
                        "diagnostics": perturbation_diagnostics(task, t, predictor)})

    for record in fitted["classical"]:
        task = record["task"]
        t = data[task]
        model = joblib.load(record["model_path"])
        def predictor(x, prior, valid, model=model, task=task):
            return model.predict_proba(x)[:, 1] if task == "slope" else np.maximum(model.predict(x), 0)
        pred = predictor(t.x[t.split["test"]], None, None)
        emit(task, record["family"], None, "validation", pred, predictor,
             directory / "predictions" / record["family"])
    for task, t in data.items():
        tr, te = t.split["train"], t.split["test"]
        fallback = float(t.target[tr].mean())
        prior = arrays[task]["prior"].numpy()
        def predictor(x, p, valid, fallback=fallback):
            return np.where(valid, p, fallback)
        emit(task, "Prior_with_training_mean_fallback", None, "train_only", predictor(None, prior[te], t.prior_valid[te]),
             predictor, directory / "predictions" / "Prior_with_training_mean_fallback")
        if task == "settlement":
            # Probe whether a simple calibration explains the apparent fusion gain.
            valid_train = tr[t.prior_valid[tr]]
            if len(valid_train) < 2:
                raise ValueError("Too few valid training priors for settlement calibration")
            coefficient, intercept = np.linalg.lstsq(np.column_stack([prior[valid_train], np.ones(len(valid_train))]), t.target[valid_train], rcond=None)[0]
            write_json(directory / "settlement_prior_calibration.json", {"coefficient": float(coefficient), "intercept": float(intercept), "fit_partition": "train", "n_valid_train": len(valid_train), "invalid_prior_fallback": fallback})
            def calibrated(x, p, valid):
                return np.where(valid, np.maximum(coefficient * p + intercept, 0), fallback)
            emit(task, "Calibrated_settlement_prior", None, "train_only", calibrated(None, prior[te], t.prior_valid[te]),
                 calibrated, directory / "predictions" / "Calibrated_settlement_prior")
    for record in fitted["neural"]:
        folder = Path(record["directory"])
        for task in record["tasks"]:
            t = data[task]
            # Task snapshots of SHARED models are explicitly diagnostic only.
            for selector in ("aggregate", "task"):
                model = load(data, folder, "aggregate" if selector == "aggregate" else task)
                def predictor(x, prior, valid, model=model, task=task, t=t):
                    return legacy.predict(model, task, t, x, prior, valid)[0]
                te = t.split["test"]
                pred = predictor(t.x[te], arrays[task]["prior"][te].numpy(), t.prior_valid[te])
                emit(task, record["model"], record["seed"], selector, pred, predictor,
                     directory / "predictions" / record["model"] / f"seed_{record['seed']}")
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "evaluation/runs/benchmark_v3")
    parser.add_argument("--split-seeds", type=int, nargs="+", default=[42, 137, 271])
    parser.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    parser.add_argument("--phase", choices=["fit", "evaluate", "all"], default="all")
    args = parser.parse_args()
    if len(set(args.split_seeds)) != len(args.split_seeds) or len(set(args.seeds)) != len(args.seeds):
        parser.error("Split seeds and training seeds must each be unique")
    torch.set_num_threads(1)
    output = args.output.resolve()
    validate_output(output, ROOT / "evaluation/runs/corrected_v1")
    output.mkdir(parents=True, exist_ok=True)
    code = [Path(__file__), Path(__file__).with_name("benchmark_neural.py"),
            Path(__file__).with_name("benchmark_classical.py"), Path(rd.__file__), Path(legacy.__file__)]
    protocol = {"version": "benchmark_v3", "split_seeds": args.split_seeds, "training_seeds": args.seeds,
                "neural_tuning_seeds": [101, 102], "neural_tuning_grid": TUNING_GRID,
                "base_spec": BASE_SPEC, "variants": VARIANTS,
                "code_sha256": {p.name: digest(p) for p in code},
                "data_sha256": {k: v.extras["sha256"] for k, v in rd.load_raw().items()},
                "software": {"python": platform.python_version(), "torch": torch.__version__,
                             "sklearn": sklearn.__version__, "numpy": np.__version__},
                "device": "cpu", "torch_threads": 1,
                "selection": "Train-only transforms and priors. Validation-only candidate and checkpoint selection. No train+validation refit.",
                "checkpoint": "Same trajectory retains aggregate/task snapshots; stops after ALL selectors exhaust 40-epoch patience; max 400 epochs.",
                "shared_task_snapshots": "Diagnostic only; different per-task snapshots are NOT one jointly deployed shared model.",
                "split_interpretation": "Slope test group stays fixed; only training/validation resampled. Rock source groups resampled. Settlement random rows, unknown provenance.",
                "test_status": "Exploratory extension on previously inspected data. Repeated partitions overlap; no independent-sample significance claim.",
                "budgets": "Neural 6 candidates x 2 tuning seeds; classical bounded family-specific grids. Budgets are disclosed, not equal compute.",
                "original_results_sha256": historical_hashes()}
    path = output / "protocol.json"
    if path.exists():
        if json.loads(path.read_text()) != protocol:
            raise ValueError("Protocol/code/data changed. Use a new output directory; refusing incompatible resume.")
    else:
        write_json(path, protocol)
        snapshot = output / "source_snapshot"
        snapshot.mkdir()
        for source in code:
            (snapshot / source.name).write_bytes(source.read_bytes())
    start = time.perf_counter()
    if args.phase in ("fit", "all"):
        for split_seed in args.split_seeds:
            fit_split(split_seed, args.split_seeds[0], args.seeds, output)
        write_json(output / "fit_complete.json", {"split_seeds": args.split_seeds})
    if args.phase in ("evaluate", "all"):
        if not (output / "fit_complete.json").exists():
            raise ValueError("Complete every configured fit before test scoring")
        rows = []
        for split_seed in args.split_seeds:
            rows.extend(evaluate_split(split_seed, output))
        write_json(output / "evaluations.json", rows)
        print(f"Scored {len(rows)} model/task/selector records", flush=True)
    if historical_hashes() != protocol["original_results_sha256"]:
        raise AssertionError("Original result tree changed (content, additions or deletions)")
    write_json(output / f"timing_{args.phase}.json", {"seconds": time.perf_counter() - start})
    print(f"Completed {args.phase} in {time.perf_counter() - start:.1f}s: {output}", flush=True)


if __name__ == "__main__":
    main()
