# Proposal benchmarking protocol

This extension tests whether Unified House adds value beyond stronger single-task
models and whether its gains come from shared training or from prior fusion.
The original `evaluation/runs/corrected_v1` results remain the paper's historical
experiment record. The verified results belong to `evaluation/runs/benchmark_v3`.
The preceding `benchmark_v2` batch is retained as an implementation-check run;
v3 adds stricter output/resume safeguards with the same experiment settings.

## Questions and controls

1. Can tuned nonlinear tabular baselines outperform the original linear controls
   and neural networks? Fit dummy, logistic/ridge, RBF support-vector, random
   forest and histogram gradient boosting models independently for each task.
2. Does joint checkpoint selection weaken independent models? Save aggregate
   and task-specific validation checkpoints on the same optimization trajectory.
3. Does the learned gate require sharing? Add F, an independent model with learned
   gating, to the existing A–E design. Compare E with F and F with B.
4. Which task combinations help? Train each of the three task pairs with shared
   models C and E on split 42, across five initialization seeds.
5. Is a fixed prior weight of 0.25 a reasonable choice? Evaluate 0.10, 0.50 and
   0.75 alongside D's 0.25. These are declared sensitivity experiments; do not
   choose a preferred weight from test performance.
6. Are conclusions stable to partitions? Repeat the main controls and tuning on
   split seeds 42, 137 and 271. Rock separates publication groups. Slope always
   holds file group 2 out and varies the training/validation division within
   group 1. Settlement uses random rows because source identifiers are absent.
7. Does a simple calibrated settlement formula explain the apparent neural gain?
   Fit an affine correction of the prior on training records only.

## Selection and budgets

All imputations, scaling, categorical vocabularies, target statistics and empirical
prior coefficients are fitted on training records. The same prepared predictors
and record partitions are used by all algorithm families within a split.

Classical candidates use validation balanced accuracy for slope and RMSE for
regression, with at most 12 candidates per family. Stable is class 1; class
decisions use a probability threshold of 0.5 for both selection and evaluation.
Failure recall is therefore recall of class 0. Regression predictions are clipped
to nonnegative values. No model is refitted on training plus validation records.

Neural tuning uses six declared capacity/regularization/learning-rate candidates
and seeds 101 and 102. Each single-task MLP selects candidates by validation
balanced accuracy or RMSE. Shared C and E select by the equal mean validation
task loss. Final neural fits use seeds 1–5. Candidate counts and selection rules
are disclosed; these are bounded searches, not exhaustive or equal-compute
optimization. A–F, task pairs and weight sensitivities retain the original fixed
architecture/settings to preserve controlled comparisons.

Neural checkpoints minimize task loss (binary cross entropy or standardized
Smooth L1) or its task average. Training ends when all tracked selectors have
exhausted 40 epochs of patience, up to 400 epochs, with improvement tolerance
0.0001. This extended stopping policy differs from the historical run. New
aggregate/task comparisons share one trajectory, so they isolate checkpoint
choice within that trajectory; they do not claim bit-for-bit reproduction of
the historical stopping rule.

Independent task-specific snapshots can be combined in one independent model.
Task-specific snapshots of a shared model are diagnostics: they are different
parameter states and cannot be presented as one simultaneously deployed shared
network. Use `aggregate` results for claims about a single shared model.

## Evaluation and interpretation

All configured fits finish before test evaluation starts. The code saves every
test prediction, tuning record, selected checkpoint, data/source hash and software
version. An independent reporting script recomputes metrics from predictions.

Report accuracy, balanced accuracy, stable-class F1, ROC AUC, failure recall and
Brier score for slope; MAE, RMSE and R-squared in original units for regression.
Report each split separately with mean and sample SD across neural seeds.
Single classical fits have no initialization SD. Split results overlap and must
not be treated as independent observations in a significance test.

Evaluate every model on the same full test records. For prior-only controls,
invalid priors fall back to the training target mean or class prevalence;
additionally report the common prior-valid subset. Analyze rock source groups,
missingness and targets above the training maximum. Evaluate directional
perturbations on the same prior-valid cases for classical and neural models.
These finite perturbations are diagnostics, not physical guarantees.

The previously inspected test data make this an exploratory extension, not an
untouched confirmatory test. The unknown settlement provenance and unverified
slope source-code meaning remain unresolved. A calibrated formula's success
does not prove synthetic data generation. External data are needed for claims
about independent engineering projects.

## Commands

From `unified_house`:

```sh
.venv/bin/python codebase/test_benchmarks.py
.venv/bin/python codebase/run_benchmarks.py --phase all
.venv/bin/python evaluation/report_benchmarks.py
.venv/bin/python evaluation/plot_benchmarks.py
.venv/bin/python evaluation/summarize_proposal_evidence.py
```

The runner resumes completed fits only when its protocol, code, data hashes and
settings match. A changed protocol requires a new output directory. Use
`--phase fit` and then `--phase evaluate` to separate fitting from scoring.
The default run includes all three split seeds and all five final training seeds.
The timed pilot is stored separately in `benchmark_v2_pilot`.
