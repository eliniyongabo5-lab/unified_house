"""Small CPU neural benchmarks with train/validation-only model selection."""
from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import torch

from revised_experiments import seed_everything


class BenchmarkNet(torch.nn.Module):
    def __init__(self, dims, shared=True, physics=False, gate=False, width=32,
                 dropout=0.0, prior_weight=0.25):
        super().__init__()
        self.shared, self.physics, self.gate = shared, physics, gate
        self.prior_weight = prior_weight
        self.encoders = torch.nn.ModuleDict({t: torch.nn.Sequential(
            torch.nn.Linear(d, width), torch.nn.ReLU(), torch.nn.Dropout(dropout),
            torch.nn.Linear(width, width), torch.nn.ReLU()) for t, d in dims.items()})
        self.trunks = torch.nn.ModuleDict({t: torch.nn.Sequential(
            torch.nn.Linear(width, width), torch.nn.ReLU())
            for t in (["shared"] if shared else dims)})
        self.heads = torch.nn.ModuleDict({t: torch.nn.Sequential(
            torch.nn.Linear(width, max(8, width // 2)), torch.nn.ReLU(),
            torch.nn.Linear(max(8, width // 2), 1)) for t in dims})
        if gate:
            self.gates = torch.nn.ModuleDict({t: torch.nn.Linear(width, 1) for t in dims})
            for layer in self.gates.values():
                torch.nn.init.zeros_(layer.weight)
                torch.nn.init.constant_(layer.bias, np.log(prior_weight / (1 - prior_weight)))

    def forward(self, task, x, prior, valid, mean, std):
        h = self.trunks["shared" if self.shared else task](self.encoders[task](x))
        z = self.heads[task](h).squeeze(1)
        pred = torch.sigmoid(z) if task == "slope" else torch.nn.functional.softplus(z * std + mean)
        weight = torch.zeros_like(pred)
        if self.physics:
            weight = (torch.sigmoid(self.gates[task](h).squeeze(1))
                      if self.gate else self.prior_weight) * valid
        return pred * (1 - weight) + prior * weight, h, weight


def task_losses(model, data, arrays, part):
    losses = {}
    for task, t in data.items():
        idx, a = t.split[part], arrays[task]
        pred, _, _ = model(task, a["x"][idx], a["prior"][idx], a["valid"][idx], t.y_mean, t.y_std)
        if task == "slope":
            loss = torch.nn.functional.binary_cross_entropy(pred.clamp(1e-6, 1-1e-6), a["y"][idx])
        else:
            loss = torch.nn.functional.smooth_l1_loss(
                (pred - t.y_mean) / t.y_std, (a["y"][idx] - t.y_mean) / t.y_std)
        losses[task] = loss
    return losses


def train(data, arrays, spec, seed, output, max_epochs=400, patience=40):
    """Retain aggregate and each task's best state on the SAME training path.

    Training ends when every selector has exhausted patience. Independent task
    states can be combined since they have no shared parameters. Shared task
    snapshots are diagnostics only, not a deployable single shared checkpoint.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    seed_everything(seed)
    constructor = {k: spec[k] for k in ("shared", "physics", "gate", "width", "dropout", "prior_weight")}
    model = BenchmarkNet({k: t.x.shape[1] for k, t in data.items()}, **constructor)
    optimizer = torch.optim.Adam(model.parameters(), lr=spec["lr"], weight_decay=spec["weight_decay"])
    best = {k: float("inf") for k in ["aggregate", *data]}
    states, epochs = {}, {}
    waits = dict.fromkeys(best, 0)
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        losses = task_losses(model, data, arrays, "train")
        loss = torch.stack(list(losses.values())).mean()
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite train loss: {spec}, seed={seed}")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            values = {k: float(v) for k, v in task_losses(model, data, arrays, "val").items()}
        values["aggregate"] = float(np.mean(list(values.values())))
        if not all(np.isfinite(v) for v in values.values()):
            raise FloatingPointError("Non-finite validation loss")
        history.append({"epoch": epoch, "train": float(loss.detach()), "val": values})
        for key, value in values.items():
            if value < best[key] - 1e-4:
                best[key], epochs[key], waits[key] = value, epoch, 0
                states[key] = copy.deepcopy(model.state_dict())
            else:
                waits[key] += 1
        if all(w >= patience for w in waits.values()):
            break
    for key, state in states.items():
        torch.save(state, output / f"{key}.pt")
    info = {"seed": seed, "spec": spec, "tasks": list(data), "best_validation_loss": best,
            "best_epoch": epochs, "trained_epochs": len(history),
            "seconds": time.perf_counter() - start,
            "parameters": sum(p.numel() for p in model.parameters()),
            "max_epochs": max_epochs, "patience": patience, "min_delta": 1e-4}
    (output / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    (output / "training.json").write_text(json.dumps(info, indent=2) + "\n")
    model.load_state_dict(states["aggregate"])
    return model, info


def load(data, directory, selector="aggregate"):
    directory = Path(directory)
    info = json.loads((directory / "training.json").read_text())
    spec = info["spec"]
    model = BenchmarkNet({t: data[t].x.shape[1] for t in info["tasks"]}, **{
        k: spec[k] for k in ("shared", "physics", "gate", "width", "dropout", "prior_weight")})
    model.load_state_dict(torch.load(directory / f"{selector}.pt", weights_only=True, map_location="cpu"))
    model.eval()
    return model


BASE_SPEC = dict(shared=True, physics=False, gate=False, width=32, dropout=0.0,
                 prior_weight=0.25, lr=0.003, weight_decay=0.0)

# Declared before test evaluation. A compact search, not a claim of exhaustive tuning.
TUNING_GRID = [
    dict(width=16, dropout=0.0, lr=0.001, weight_decay=1e-4),
    dict(width=16, dropout=0.1, lr=0.003, weight_decay=1e-3),
    dict(width=32, dropout=0.0, lr=0.003, weight_decay=0.0),
    dict(width=32, dropout=0.1, lr=0.001, weight_decay=1e-4),
    dict(width=64, dropout=0.0, lr=0.001, weight_decay=1e-4),
    dict(width=64, dropout=0.1, lr=0.003, weight_decay=1e-3),
]
