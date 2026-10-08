"""Render compact LaTeX tables directly from the audited summary.json."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SUMMARY = ROOT / "evaluation" / "analysis" / "summary.json"
PAPER_OUT = ROOT / "paper" / "generated_tables.tex"
CONFIGS = ["A_independent", "B_independent_prior", "C_shared", "D_shared_prior", "E_shared_prior_gate"]
ROW = r"\\"


def table(lines, label, caption, columns, headings, rows):
    lines.extend([r"\begin{table}[htbp]", r"\centering\small",
                  f"\\caption{{{caption}\\label{{{label}}}}}",
                  f"\\begin{{tabular}}{{{columns}}}", r"\toprule",
                  " & ".join(headings) + " " + ROW, r"\midrule"])
    lines.extend(" & ".join(str(x) for x in row) + " " + ROW for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])


def render(summary: dict | None = None) -> None:
    d = summary or json.loads(SUMMARY.read_text())
    agg, baselines = d["aggregate_metrics"], d["classical_baselines"]
    prior = d["analytical_prior_valid_cases"]
    lines = ["% Machine-generated from evaluation/analysis/summary.json.",
             "% Neural values: mean and sample SD across five initialization seeds on a fixed split."]
    specs = {
        "slope": (["accuracy", "balanced_accuracy", "f1", "roc_auc"],
                  ["Configuration", "Accuracy", "Balanced accuracy", "F1", "AUC"],
                  "lcccc", "All metrics are dimensionless."),
        "rock": (["mae", "rmse", "r2"], ["Configuration", "MAE (MPa)", "RMSE (MPa)", "$R^2$"],
                 "lccc", "UCS errors are in MPa."),
        "settlement": (["mae", "rmse", "r2"], ["Configuration", "MAE (mm)", "RMSE (mm)", "$R^2$"],
                       "lccc", "Settlement errors are in mm."),
    }
    for task, (metrics, heading, cols, units) in specs.items():
        rows = []
        for config in CONFIGS:
            vals = []
            for metric in metrics:
                item = agg[config][task][metric]
                digits = 3 if metric in {"accuracy", "balanced_accuracy", "f1", "roc_auc", "r2"} else 2
                vals.append(f"{item['mean']:.{digits}f} $\\pm$ {item['sample_sd']:.{digits}f}")
            rows.append([config[0], *vals])
        for title, source in [("Classical baseline", baselines[task]),
                              (f"Prior ($n={prior[task]['n_valid']}$)", prior[task]["metrics_on_valid_cases"])]:
            vals = []
            for metric in metrics:
                digits = 3 if metric in {"accuracy", "balanced_accuracy", "f1", "roc_auc", "r2"} else 2
                vals.append(f"{source[metric]:.{digits}f}")
            rows.append([title, *vals])
        table(lines, f"tab:{task}_perf", f"{task.title()} held-out performance. Neural results are mean $\\pm$ sample SD over five seeds. {units} The prior is evaluated on cases with all required inputs; other models use the full test set.", cols, heading, rows)

    paired = d["paired_differences"]
    rows = []
    for key, name in [("C_minus_A", "C$-$A"), ("E_minus_D", "E$-$D")]:
        for task, metric in [("slope", "accuracy"), ("rock", "rmse"), ("settlement", "rmse")]:
            p = paired[key][task][metric]
            lo, hi = p["ci95_t"]
            units = "" if task == "slope" else (" (MPa)" if task == "rock" else " (mm)")
            rows.append([name, task.title(), ("Accuracy" if metric == "accuracy" else "RMSE") + units,
                         f"${p['mean']:.3f} \\pm {p['sample_sd']:.3f}$",
                         f"$[{lo:.3f}, {hi:.3f}]$"])
    table(lines, "tab:paired", "Differences between models paired by initialization seed (left minus right), with descriptive 95\\% $t$ intervals; five seeds do not quantify source or split uncertainty.", "lllcc", ["Contrast", "Task", "Metric", "Mean $\\pm$ SD", "95\\% interval"], rows)

    names = {"cohesion_kPa": "Cohesion", "friction_angle_deg": "Friction angle",
             "Is50_MPa": "Point-load index", "Applied Load q (kPa)": "Load",
             "Elastic Modulus E (MPa)": "Elastic modulus"}
    diag = d["physical_diagnostics"]
    rows = []
    for config in ("C_shared", "D_shared_prior", "E_shared_prior_gate"):
        for task in ("slope", "rock", "settlement"):
            for key, item in diag[config][task].items():
                rows.append([config[0], task.title(), names[key],
                             f"{item['pooled_violations']} / {item['pooled_denominator']}",
                             f"{100*item['pooled_rate']:.2f}\\%"])
    table(lines, "tab:physics", "Sign violations under held-out input perturbations. Each eligible case is evaluated once per seed, so denominators count five predictions per case.", "lllrr", ["Config.", "Task", "Perturbed input", "Violations / evaluations", "Rate"], rows)

    gates = d["gate_weights"]
    rows = []
    for config in CONFIGS:
        vals = []
        for task in ("slope", "rock", "settlement"):
            x = gates[config][task]["mean_gate_across_seeds"]
            vals.append(f"{x['mean']:.3f} $\\pm$ {x['sample_sd']:.3f}")
        rows.append([config[0], *vals])
    table(lines, "tab:gate", "Prior weights averaged over test cases with valid prior inputs; mean $\\pm$ sample SD across five seeds.", "lccc", ["Config.", "Slope", "Rock", "Settlement"], rows)

    output = "\n".join(lines) + "\n"
    PAPER_OUT.write_text(output)


if __name__ == "__main__":
    render()
