"""Descriptive held-out latent analysis; no claims of a common constitutive law."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score


ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "evaluation" / "runs" / "corrected_v1"
OUT = ROOT / "evaluation" / "analysis"
FIGURE = ROOT / "paper" / "figures" / "shared_embedding_pca.pdf"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    FIGURE.parent.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for seed in (1, 2, 3, 4, 5):
        rows = []
        task_label = []
        for i, task in enumerate(("slope", "rock", "settlement")):
            f = RUNS / "E_shared_prior_gate" / f"seed_{seed}" / f"predictions_{task}.npz"
            with np.load(f) as a:
                h = a["embedding"]
            rows.append(h)
            task_label.extend([i] * len(h))
        h = np.vstack(rows)
        labels = np.array(task_label)
        z = (h - h.mean(axis=0)) / np.maximum(h.std(axis=0), 1e-8)
        pca = PCA(n_components=2).fit(z)
        xy = pca.transform(z)
        outputs[str(seed)] = {"n": {t: len(rows[i]) for i,t in enumerate(("slope", "rock", "settlement"))},
                              "pca_variance_ratio": pca.explained_variance_ratio_.tolist(),
                              "task_silhouette_standardized_embedding": float(silhouette_score(z, labels))}
        if seed == 1:
            fig, ax = plt.subplots(figsize=(6, 4.2))
            for i, task in enumerate(("slope", "rock", "settlement")):
                mask = labels == i
                ax.scatter(xy[mask, 0], xy[mask, 1], s=12, alpha=.55, label=f"{task} (n={mask.sum()})")
            ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)")
            ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)")
            ax.set_title("Shared-trunk embeddings on held-out records (seed 1)")
            ax.legend(frameon=False, fontsize=8)
            fig.tight_layout()
            fig.savefig(FIGURE)
            plt.close(fig)
    (OUT / "embedding_summary.json").write_text(json.dumps(outputs, indent=2) + "\n")
    print(json.dumps(outputs, indent=2))


if __name__ == "__main__":
    main()
