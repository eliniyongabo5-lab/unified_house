"""Controlled A--E ablations on the audited, unpaired geotechnical datasets.

Run: .venv/bin/python codebase/revised_experiments.py --prepare
     .venv/bin/python codebase/revised_experiments.py --run --seeds 1 2 3 4 5
The held-out partitions are scored only after every configuration is fitted.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import random
from pathlib import Path

import numpy as np
import sklearn
import torch
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score,
                             mean_absolute_error, mean_squared_error, r2_score,
                             roc_auc_score)

from revised_data import TASKS, _fos, fit_transforms, load_raw, make_splits, save_prepared


ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "evaluation" / "runs" / "corrected_v1"
CONFIGS = {
    "A_independent": (False, False, False),
    "B_independent_prior": (False, True, False),
    "C_shared": (True, False, False),
    "D_shared_prior": (True, True, False),
    "E_shared_prior_gate": (True, True, True),
}
MAX_EPOCHS = 400
PATIENCE = 40
MIN_DELTA = 1e-4
LR = 0.003
PRIOR_WEIGHT = 0.25


class Model(torch.nn.Module):
    def __init__(self, dims: dict[str, int], shared: bool, physics: bool, gate: bool):
        super().__init__()
        self.physics, self.gate = physics, gate
        self.encoders = torch.nn.ModuleDict({k: torch.nn.Sequential(
            torch.nn.Linear(d, 32), torch.nn.ReLU(),
            torch.nn.Linear(32, 32), torch.nn.ReLU()) for k, d in dims.items()})
        if shared:
            self.trunks = torch.nn.ModuleDict({"shared": torch.nn.Sequential(
                torch.nn.Linear(32, 32), torch.nn.ReLU())})
        else:
            self.trunks = torch.nn.ModuleDict({k: torch.nn.Sequential(
                torch.nn.Linear(32, 32), torch.nn.ReLU()) for k in dims})
        self.shared = shared
        self.heads = torch.nn.ModuleDict({k: torch.nn.Sequential(
            torch.nn.Linear(32, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1)) for k in dims})
        if gate:
            self.gates = torch.nn.ModuleDict({k: torch.nn.Linear(32, 1) for k in dims})
            for layer in self.gates.values():
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.constant_(layer.bias, np.log(PRIOR_WEIGHT / (1 - PRIOR_WEIGHT)))

    def forward(self, task: str, x: torch.Tensor, prior: torch.Tensor,
                valid: torch.Tensor, mean: float, std: float):
        h = self.trunks["shared" if self.shared else task](self.encoders[task](x))
        z = self.heads[task](h).squeeze(1)
        learned = torch.sigmoid(z) if task == "slope" else torch.nn.functional.softplus(z * std + mean)
        if not self.physics:
            return learned, h, torch.zeros_like(learned)
        if self.gate:
            weight = torch.sigmoid(self.gates[task](h).squeeze(1)) * valid
        else:
            weight = PRIOR_WEIGHT * valid
        return learned * (1 - weight) + prior * weight, h, weight


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def tensors(data):
    result = {}
    for name, t in data.items():
        p = t.prior.astype(np.float32).copy()
        if name == "slope":
            p = (1 / (1 + np.exp(-np.clip(
                TRANSFORMS[name]["fos_logit_coefficient"] * np.log(np.maximum(p, .01))
                + TRANSFORMS[name]["fos_logit_intercept"], -30, 30)))).astype(np.float32)
        # Invalid priors are never blended. Replace their arbitrary numbers with zero.
        p[~t.prior_valid] = 0
        result[name] = {
            "x": torch.tensor(t.x, dtype=torch.float32),
            "y": torch.tensor(t.target, dtype=torch.float32),
            "prior": torch.tensor(p, dtype=torch.float32),
            "valid": torch.tensor(t.prior_valid.astype(np.float32)),
        }
    return result


def split_loss(model, data, arrays, part):
    losses = []
    for task, t in data.items():
        idx = t.split[part]
        a = arrays[task]
        pred, _, _ = model(task, a["x"][idx], a["prior"][idx],
                           a["valid"][idx], t.y_mean, t.y_std)
        target = a["y"][idx]
        if task == "slope":
            loss = torch.nn.functional.binary_cross_entropy(pred.clamp(1e-6, 1-1e-6), target)
        else:
            loss = torch.nn.functional.smooth_l1_loss(
                (pred - t.y_mean) / t.y_std, (target - t.y_mean) / t.y_std)
        losses.append(loss)
    return torch.stack(losses).mean()


def train_one(data, arrays, config, seed, output):
    seed_everything(seed)
    shared, physics, gate = CONFIGS[config]
    model = Model({k: t.x.shape[1] for k, t in data.items()}, shared, physics, gate)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    best = float("inf")
    best_state = None
    wait = 0
    history = []
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        loss = split_loss(model, data, arrays, "train")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val = float(split_loss(model, data, arrays, "val"))
        history.append([epoch, float(loss.detach()), val])
        if val < best - MIN_DELTA:
            best, best_state, wait, best_epoch = val, copy.deepcopy(model.state_dict()), 0, epoch
        else:
            wait += 1
        if wait >= PATIENCE:
            break
    model.load_state_dict(best_state)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    (output / "history.json").write_text(json.dumps(history) + "\n")
    return model, {"best_epoch": best_epoch, "best_validation_loss": best,
                   "trained_epochs": len(history)}


def predict(model, task, t, x, prior, valid):
    model.eval()
    with torch.no_grad():
        p, h, w = model(task, torch.tensor(x, dtype=torch.float32),
                        torch.tensor(prior, dtype=torch.float32),
                        torch.tensor(valid.astype(np.float32)), t.y_mean, t.y_std)
    return p.numpy(), h.numpy(), w.numpy()


def perturb(t, task, arr, feature, mode):
    raw = t.raw.copy()
    old = raw[arr, feature]
    if mode == "add":
        new = old + 5
    else:
        new = old * 1.1
    raw[arr, feature] = new
    x = t.x[arr].copy()
    x[:, feature] = ((new - t.feature_mean[feature]) / t.feature_std[feature]).astype(np.float32)
    if task == "slope":
        base = _fos(raw[arr])
        coef = TRANSFORMS[task]["fos_logit_coefficient"]
        intercept = TRANSFORMS[task]["fos_logit_intercept"]
        p = 1 / (1 + np.exp(-np.clip(coef * np.log(np.maximum(base, .01)) + intercept, -30, 30)))
    elif task == "rock":
        p = np.maximum(raw[arr, 3] * TRANSFORMS[task]["point_load_coefficient"], 0)
    else:
        p = np.maximum(raw[arr, 4] * raw[arr, 3] * raw[arr, 6] / raw[arr, 5], 0)
    return x, p.astype(np.float32)


def diagnostics(model, task, t, arrays):
    idx = t.split["test"]
    a = arrays[task]
    p0, _, _ = predict(model, task, t, t.x[idx], a["prior"][idx].numpy(), t.prior_valid[idx])
    specs = {"slope": [(1, "multiply", 1), (2, "add", 1)],
             "rock": [(3, "multiply", 1)],
             "settlement": [(4, "multiply", 1), (5, "multiply", -1)]}[task]
    out = {}
    for j, mode, sign in specs:
        eligible = idx[np.isfinite(t.raw[idx, j]) & t.prior_valid[idx]]
        if not len(eligible):
            out[t.columns[j]] = {"n": 0, "violations": 0, "rate": None}
            continue
        x, prior = perturb(t, task, eligible, j, mode)
        p1, _, _ = predict(model, task, t, x, prior, np.ones(len(eligible), dtype=bool))
        base, _, _ = predict(model, task, t, t.x[eligible], a["prior"][eligible].numpy(), t.prior_valid[eligible])
        violations = int(np.sum(sign * (p1 - base) < -1e-6))
        out[t.columns[j]] = {"n": len(eligible), "violations": violations,
                             "rate": violations / len(eligible)}
    return out


def score(model, data, arrays, config, seed, output):
    results = {"config": config, "seed": seed, "tasks": {}}
    for task, t in data.items():
        idx = t.split["test"]
        a = arrays[task]
        pred, h, weight = predict(model, task, t, t.x[idx], a["prior"][idx].numpy(), t.prior_valid[idx])
        truth = t.target[idx]
        if task == "slope":
            label = pred >= .5
            metrics = {"accuracy": accuracy_score(truth, label),
                       "balanced_accuracy": balanced_accuracy_score(truth, label),
                       "f1": f1_score(truth, label),
                       "roc_auc": roc_auc_score(truth, pred)}
        else:
            metrics = {"mae": mean_absolute_error(truth, pred),
                       "rmse": np.sqrt(mean_squared_error(truth, pred)),
                       "r2": r2_score(truth, pred)}
        results["tasks"][task] = {"n_test": len(idx), "metrics": {k: float(v) for k,v in metrics.items()},
                                   "mean_gate_weight_valid": float(weight[t.prior_valid[idx]].mean()) if np.any(t.prior_valid[idx]) else None,
                                   "diagnostics": diagnostics(model, task, t, arrays)}
        np.savez_compressed(output / f"predictions_{task}.npz", row_id=t.row_id[idx], truth=truth,
                            prediction=pred, embedding=h, gate_weight=weight,
                            prior=a["prior"][idx].numpy(), prior_valid=t.prior_valid[idx])
    (output / "test_metrics.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def score_baselines(data):
    results = {}
    for name, t in data.items():
        tr, te = t.split["train"], t.split["test"]
        if name == "slope":
            m = LogisticRegression(max_iter=2000).fit(t.x[tr], t.target[tr])
            p = m.predict_proba(t.x[te])[:, 1]
            results[name] = {"model": "logistic_regression", "accuracy": float(accuracy_score(t.target[te], p >= .5)),
                             "balanced_accuracy": float(balanced_accuracy_score(t.target[te], p >= .5)),
                             "f1": float(f1_score(t.target[te], p >= .5)), "roc_auc": float(roc_auc_score(t.target[te], p))}
        else:
            m = Ridge(alpha=1.0).fit(t.x[tr], t.target[tr])
            p = np.maximum(m.predict(t.x[te]), 0)
            results[name] = {"model": "ridge_alpha_1", "mae": float(mean_absolute_error(t.target[te], p)),
                             "rmse": float(np.sqrt(mean_squared_error(t.target[te], p))),
                             "r2": float(r2_score(t.target[te], p))}
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Two training updates, validation only")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()
    torch.set_num_threads(1)
    data = load_raw()
    manifest = make_splits(data)
    global TRANSFORMS
    TRANSFORMS = fit_transforms(data)
    save_prepared(data, manifest, TRANSFORMS, OUT / "prepared")
    protocol = {"configs": CONFIGS, "seeds": args.seeds, "max_epochs": MAX_EPOCHS,
                "patience": PATIENCE, "min_delta": MIN_DELTA, "learning_rate": LR,
                "fixed_prior_weight": PRIOR_WEIGHT, "optimizer": "Adam", "training": "full-batch, equal mean of three task losses, aggregate-validation checkpoint",
                "architecture": "task encoder 2x32 ReLU; shared or separate 32 ReLU trunk; task head 16 ReLU; optional learned gate",
                "software": {"python": platform.python_version(), "torch": torch.__version__, "numpy": np.__version__, "sklearn": sklearn.__version__},
                "device": "cpu", "split_seed": 42,
                "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                                [Path(__file__), Path(__file__).with_name("revised_data.py")]}}
    (OUT / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    print("prepared", {k: {p: len(v) for p,v in t.split.items()} for k,t in data.items()}, flush=True)
    if args.smoke:
        arrays = tensors(data)
        seed_everything(1)
        for config, (shared, physics, gate) in CONFIGS.items():
            model = Model({k: t.x.shape[1] for k, t in data.items()}, shared, physics, gate)
            opt = torch.optim.Adam(model.parameters(), lr=LR)
            for _ in range(2):
                opt.zero_grad()
                loss = split_loss(model, data, arrays, "train")
                loss.backward()
                opt.step()
            with torch.no_grad():
                val = float(split_loss(model, data, arrays, "val"))
            print("smoke", config, "train", float(loss.detach()), "validation", val, flush=True)
        return
    if not args.run:
        return
    arrays = tensors(data)
    fitted = []
    for config in CONFIGS:
        for seed in args.seeds:
            output = OUT / config / f"seed_{seed}"
            model, train_info = train_one(data, arrays, config, seed, output)
            fitted.append((model, config, seed, output))
            (output / "training.json").write_text(json.dumps(train_info, indent=2) + "\n")
            print(config, seed, train_info, flush=True)
    all_results = [score(model, data, arrays, config, seed, output) for model,config,seed,output in fitted]
    (OUT / "all_test_metrics.json").write_text(json.dumps(all_results, indent=2) + "\n")
    (OUT / "baselines.json").write_text(json.dumps(score_baselines(data), indent=2) + "\n")
    print("test scoring complete", flush=True)


if __name__ == "__main__":
    main()
