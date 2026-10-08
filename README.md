# Unified House — code and data

This package contains the Python experiment and evaluation scripts, all three
input datasets (two CSV files and the original rock-strength Excel workbook),
and saved experiment outputs and analysis. Git metadata/history, virtual
environments, caches, proposal documents and manuscript LaTeX are excluded.
The manuscript source is supplied separately in unified_house_latex.zip.

## Setup

From this directory, use Python 3.12:

```sh
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

## Reproduce the manuscript experiments

```sh
.venv/bin/python codebase/revised_experiments.py --prepare
.venv/bin/python codebase/revised_experiments.py --run --seeds 1 2 3 4 5
.venv/bin/python evaluation/analyze_results.py
.venv/bin/python codebase/embedding_analysis.py
.venv/bin/python codebase/paper_figures.py
.venv/bin/python codebase/parity_figures.py
```

Rerunning experiments replaces saved corrected_v1 results. Scripts create
publication tables and figures under paper/; to compile the manuscript, copy
the contents of the separate LaTeX package into that directory.

## Expanded benchmarks

Read evaluation/runs/BENCHMARK_PROTOCOL.md, then run:

```sh
.venv/bin/python codebase/test_benchmarks.py
.venv/bin/python codebase/run_benchmarks.py --phase all
.venv/bin/python evaluation/report_benchmarks.py
.venv/bin/python evaluation/plot_benchmarks.py
.venv/bin/python evaluation/summarize_proposal_evidence.py
```

The benchmark_v3 outputs include selected models, tuning records, predictions,
training histories and analysis reports. Earlier benchmark runs are retained.
PROJECT_README.md preserves the original workspace documentation; its paper
and proposal paths refer to the full workspace, not this code-only package.
