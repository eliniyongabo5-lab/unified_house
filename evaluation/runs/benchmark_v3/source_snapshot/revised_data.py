"""Verified datasets and frozen, train-only transforms for revision experiments.

The three corpora are separate. Returned row identifiers refer to source-file rows,
not to matched sites. Physics priors are evaluated only when their inputs were
originally observed. This module never reads historical model checkpoints.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, train_test_split


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "dataset"
SPLIT_SEED = 42
TASKS = ("slope", "rock", "settlement")


@dataclass
class TaskData:
    name: str
    columns: list[str]
    raw: np.ndarray
    target: np.ndarray
    row_id: np.ndarray
    groups: np.ndarray | None
    source: str
    extras: dict
    split: dict[str, np.ndarray] | None = None
    x: np.ndarray | None = None
    y_mean: float = 0.0
    y_std: float = 1.0
    feature_median: np.ndarray | None = None
    feature_mean: np.ndarray | None = None
    feature_std: np.ndarray | None = None
    prior: np.ndarray | None = None
    prior_valid: np.ndarray | None = None


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _numeric(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.column_stack([pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float) for c in columns])


def load_raw() -> dict[str, TaskData]:
    slope_path = DATA / "slope_stability_dataset" / "slope_stability_684case_real_dataset.csv"
    rock_path = DATA / "Rock_slope_(Hoek_Brown)" / "ROCK_sc_10.xlsx"
    settlement_path = DATA / "geotechnical_settlement_dataset" / "geotechnical_settlement_dataset.csv"

    s = pd.read_csv(slope_path)
    # Two source-1 rows have identical available predictors but opposite labels.
    # Neither outcome can be adjudicated from the supplied table.
    s = s.loc[~s.case_no.isin([394, 438])].copy()
    scols = ["unit_weight_kNm3", "cohesion_kPa", "friction_angle_deg", "slope_angle_deg", "slope_height_m", "pore_pressure_ratio"]
    if set(s.stability.unique()) != {"Stable", "Failure"}:
        raise ValueError("Unexpected slope labels")
    slope = TaskData("slope", scols, _numeric(s, scols), (s.stability == "Stable").to_numpy(dtype=float), s.case_no.to_numpy(), s.source_ref.to_numpy(), str(slope_path.relative_to(ROOT)), {"sha256": _sha(slope_path), "target_units": "class"})

    r = pd.read_excel(rock_path)
    # Use the actual workbook headers. The legacy position-based renaming is wrong.
    r = r[pd.to_numeric(r["σc"], errors="coerce").notna()].copy()
    # The first 18 columns are the populated source table; 21 rows duplicate
    # another record exactly, including the target. Keep the first occurrence.
    r = r.loc[~r.duplicated(subset=r.columns[:18])].copy()
    rcols = ["ρd", "n", "Vp", "Is (50)", "E"]
    rr = _numeric(r, rcols)
    # Vp is reported in m/s in the source workbook; km/s is easier to scale.
    rr[:, 2] /= 1000.0
    rock = TaskData("rock", ["rho_d_g_cm3", "porosity_percent", "Vp_km_s", "Is50_MPa", "E_source_units"], rr, pd.to_numeric(r["σc"]).to_numpy(dtype=float), r.index.to_numpy(dtype=int), r.Source.to_numpy(), str(rock_path.relative_to(ROOT)), {"sha256": _sha(rock_path), "target_units": "MPa", "E_units_note": "Workbook units are not independently confirmed; E is a predictor only, with no unit-sensitive physics calculation."})

    f = pd.read_csv(settlement_path)
    fcols = ["Depth (m)", "Moisture Content (%)", "SPT N-value", "Foundation Width B (m)", "Applied Load q (kPa)", "Elastic Modulus E (MPa)", "Influence Factor Is"]
    fr = _numeric(f, fcols)
    # Soil type is represented by train-fitted one-hot columns below.
    settlement = TaskData("settlement", fcols, fr, pd.to_numeric(f["Settlement S (mm)"], errors="coerce").to_numpy(dtype=float), np.arange(len(f)), None, str(settlement_path.relative_to(ROOT)), {"sha256": _sha(settlement_path), "target_units": "mm", "soil_type": f["Soil Type"].astype(str).to_numpy().tolist()})
    out = {t.name: t for t in (slope, rock, settlement)}
    if [len(out[k].target) for k in TASKS] != [507, 3976, 500]:
        raise ValueError("Source counts changed; inspect before experiments")
    return out


def make_splits(data: dict[str, TaskData]) -> dict:
    s = data["slope"]
    # The second published source is held out. The first source supplies train/val.
    train_val = np.flatnonzero(s.groups == 1)
    test = np.flatnonzero(s.groups == 2)
    if not len(train_val) or not len(test):
        raise ValueError("Slope source groups changed")
    train, val = train_test_split(train_val, test_size=0.15, random_state=SPLIT_SEED, stratify=s.target[train_val])
    s.split = {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}

    r = data["rock"]
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=SPLIT_SEED)
    train_val, test = next(gss.split(r.raw, r.target, r.groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.1764705882, random_state=SPLIT_SEED + 1)
    it, iv = next(gss2.split(r.raw[train_val], r.target[train_val], r.groups[train_val]))
    r.split = {"train": np.sort(train_val[it]), "val": np.sort(train_val[iv]), "test": np.sort(test)}

    f = data["settlement"]
    all_idx = np.arange(len(f.target))
    train_val, test = train_test_split(all_idx, test_size=0.15, random_state=SPLIT_SEED)
    train, val = train_test_split(train_val, test_size=0.1764705882, random_state=SPLIT_SEED + 1)
    f.split = {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test)}
    manifest = {"split_seed": SPLIT_SEED, "method": {"slope": "source_ref=2 held out; stratified validation within source_ref=1", "rock": "source-group held-out validation and test", "settlement": "random row split; original grouping unavailable"}, "tasks": {}}
    for t in data.values():
        idx = t.split
        assert idx is not None
        if len(np.unique(np.concatenate(list(idx.values())))) != len(t.target):
            raise AssertionError(f"Overlapping or missing split in {t.name}")
        if t.groups is not None and t.name == "rock":
            sets = [set(t.groups[v].tolist()) for v in idx.values()]
            if any(sets[a] & sets[b] for a in range(3) for b in range(a + 1, 3)):
                raise AssertionError("Rock source leakage")
        manifest["tasks"][t.name] = {"n": len(t.target), "source": t.source, **t.extras, "indices": {k: v.tolist() for k, v in idx.items()}, "row_ids": {k: t.row_id[v].tolist() for k, v in idx.items()}}
    return manifest


def _fos(x: np.ndarray) -> np.ndarray:
    gamma, cohesion, phi, beta, height, ru = x.T
    br = np.deg2rad(np.clip(beta, 1, 85))
    effective = np.maximum(gamma * height * (np.cos(br) ** 2 - ru), 1e-6)
    resisting = np.maximum(cohesion, 0) + effective * np.tan(np.deg2rad(np.clip(phi, 0, 60)))
    driving = np.maximum(gamma * height * np.sin(br) * np.cos(br), 1e-6)
    return np.clip(resisting / driving, 0.01, 50)


def fit_transforms(data: dict[str, TaskData]) -> dict:
    """Fit every estimated quantity on the train partition alone."""
    records = {}
    for t in data.values():
        tr = t.split["train"]
        raw_train = t.raw[tr]
        median = np.nanmedian(raw_train, axis=0)
        if not np.all(np.isfinite(median)):
            raise ValueError(f"All-missing feature in {t.name}")
        filled = np.where(np.isfinite(t.raw), t.raw, median)
        mean = filled[tr].mean(axis=0)
        std = filled[tr].std(axis=0)
        std[std < 1e-8] = 1.0
        z = (filled - mean) / std
        missing = (~np.isfinite(t.raw)).astype(float)
        xparts = [z, missing]
        vocab = []
        if t.name == "settlement":
            soil = np.array(t.extras["soil_type"])
            vocab = sorted(set(soil[tr].tolist()))
            xparts.append(np.column_stack([(soil == category).astype(float) for category in vocab]))
        t.x = np.concatenate(xparts, axis=1).astype(np.float32)
        t.feature_median, t.feature_mean, t.feature_std = median, mean, std
        if t.name != "slope":
            t.y_mean = float(t.target[tr].mean())
            t.y_std = float(t.target[tr].std())
            if t.y_std <= 0:
                raise ValueError(f"Zero target variance in {t.name}")
        if t.name == "slope":
            # Complete raw variables are required for analytical FoS.
            valid = np.isfinite(t.raw).all(axis=1) & (filled[:, 0] > 0) & (filled[:, 4] > 0)
            t.prior = _fos(filled)
        elif t.name == "rock":
            valid = np.isfinite(t.raw[:, 3]) & (t.raw[:, 3] > 0)
            ratio = t.target[tr][valid[tr]] / t.raw[tr, 3][valid[tr]]
            ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
            if len(ratio) < 30:
                raise ValueError("Too few measured Is50 values for rock prior")
            coefficient = float(np.median(ratio))
            t.prior = np.maximum(filled[:, 3] * coefficient, 0)
        else:
            q, b, modulus, influence = (filled[:, j] for j in (4, 3, 5, 6))
            valid = np.isfinite(t.raw[:, [3, 4, 5, 6]]).all(axis=1) & (modulus > 0)
            t.prior = np.maximum(q * b * influence / np.maximum(modulus, 1e-6), 0)
        t.prior_valid = valid
        records[t.name] = {"columns": t.columns, "median": median.tolist(), "mean": mean.tolist(), "std": std.tolist(), "target_mean": t.y_mean, "target_std": t.y_std, "n_train": len(tr), "n_prior_train": int(valid[tr].sum()), "soil_type_vocab": vocab, "prior": {"slope": "simplified infinite-slope Mohr-Coulomb FoS", "rock": "train-median UCS/Is50 coefficient; empirical index estimate", "settlement": "q*B*Is/E; q kPa, B m, E MPa, result mm"}[t.name]}
        if t.name == "rock":
            records[t.name]["point_load_coefficient"] = coefficient
        if t.name == "slope":
            # Logistic calibration uses only complete training data, never test labels.
            from sklearn.linear_model import LogisticRegression
            p = np.log(np.maximum(t.prior[tr][valid[tr]], 0.01)).reshape(-1, 1)
            model = LogisticRegression(max_iter=1000).fit(p, t.target[tr][valid[tr]])
            records[t.name]["fos_logit_coefficient"] = float(model.coef_[0, 0])
            records[t.name]["fos_logit_intercept"] = float(model.intercept_[0])
    return records


def save_prepared(data: dict[str, TaskData], split: dict, transforms: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "splits.json").write_text(json.dumps(split, indent=2) + "\n")
    (output / "transforms.json").write_text(json.dumps(transforms, indent=2) + "\n")
    for name, t in data.items():
        np.savez_compressed(output / f"{name}.npz", x=t.x, raw=t.raw, y=t.target, row_id=t.row_id, prior=t.prior, prior_valid=t.prior_valid)
