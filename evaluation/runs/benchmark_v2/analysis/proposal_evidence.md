# Evidence for the Unified House proposal

The completed experiments support a proposal about when parameter sharing and analytical guidance help heterogeneous geotechnical prediction. They do not establish that the complete Unified House model is consistently the strongest algorithm.

All model choices use training/validation data. Test results are exploratory because the underlying data were inspected previously. Reported neural SDs describe initialization variability, not uncertainty across independent projects.

## Predictive performance across partitions

Independent models use task-specific checkpoints; shared models use one aggregate checkpoint. Classical controls are single selected fits. The full report also contains the controlled aggregate-versus-task comparisons.

### Slope: balanced accuracy (higher is better)

| Model | Split 42 | Split 137 | Split 271 |
| --- | --- | --- | --- |
| Logistic regression | 0.777 | 0.849 | 0.866 |
| RBF support-vector model | 0.708 | 0.681 | 0.655 |
| Random forest | 0.685 | 0.678 | 0.658 |
| Histogram gradient boosting | 0.626 | 0.619 | 0.639 |
| Tuned single-task MLP | 0.621 ± 0.027 | 0.737 ± 0.014 | 0.629 ± 0.020 |
| C shared | 0.654 ± 0.034 | 0.663 ± 0.053 | 0.640 ± 0.026 |
| E shared with prior gate | 0.695 ± 0.083 | 0.667 ± 0.059 | 0.708 ± 0.050 |
| F independent with prior gate | 0.682 ± 0.051 | 0.712 ± 0.060 | 0.683 ± 0.053 |
| Tuned C shared | 0.642 ± 0.019 | 0.654 ± 0.048 | 0.609 ± 0.017 |
| Tuned E shared with prior gate | 0.660 ± 0.020 | 0.671 ± 0.023 | 0.631 ± 0.057 |

### Rock: RMSE in MPa (lower is better)

| Model | Split 42 | Split 137 | Split 271 |
| --- | --- | --- | --- |
| Ridge regression | 52.588 | 36.575 | 37.110 |
| RBF support-vector model | 68.922 | 42.204 | 41.148 |
| Random forest | 63.067 | 51.096 | 37.665 |
| Histogram gradient boosting | 55.843 | 46.455 | 41.300 |
| Tuned single-task MLP | 54.968 ± 0.448 | 45.992 ± 0.336 | 42.300 ± 4.122 |
| C shared | 54.433 ± 3.407 | 48.614 ± 2.439 | 48.639 ± 1.775 |
| E shared with prior gate | 56.658 ± 3.484 | 48.468 ± 4.766 | 45.381 ± 2.274 |
| F independent with prior gate | 55.827 ± 6.045 | 45.015 ± 3.215 | 37.788 ± 1.620 |
| Tuned C shared | 50.251 ± 5.586 | 45.497 ± 2.062 | 46.158 ± 0.655 |
| Tuned E shared with prior gate | 57.579 ± 4.564 | 45.920 ± 3.630 | 46.419 ± 2.731 |

### Settlement: RMSE in mm (lower is better)

| Model | Split 42 | Split 137 | Split 271 |
| --- | --- | --- | --- |
| Ridge regression | 4.392 | 4.273 | 7.064 |
| RBF support-vector model | 1.849 | 2.731 | 3.489 |
| Random forest | 3.506 | 2.776 | 4.270 |
| Histogram gradient boosting | 2.518 | 2.894 | 4.471 |
| Tuned single-task MLP | 1.490 ± 0.213 | 2.050 ± 0.295 | 3.562 ± 0.334 |
| C shared | 1.575 ± 0.145 | 2.603 ± 0.578 | 3.358 ± 0.957 |
| E shared with prior gate | 1.015 ± 0.238 | 0.821 ± 0.131 | 0.875 ± 0.111 |
| F independent with prior gate | 0.778 ± 0.011 | 0.734 ± 0.035 | 0.788 ± 0.060 |
| Tuned C shared | 1.513 ± 0.090 | 2.196 ± 0.130 | 3.307 ± 0.220 |
| Tuned E shared with prior gate | 0.766 ± 0.006 | 0.690 ± 0.007 | 0.725 ± 0.051 |
| Calibrated analytical prior | 0.778 | 0.693 | 0.726 |

## Isolating sharing and checkpoint choice

The following mean differences pair identical training seeds and compare aggregate checkpoints on the same split. Lower RMSE differences favor the model on the left; higher balanced-accuracy differences favor the model on the left. These descriptive differences are not significance tests.

| Split | Task | Contrast | Mean difference ± SD |
| --- | --- | --- | --- |
| 42 | slope | C − A | -0.006 ± 0.030 |
| 42 | slope | E − F | -0.008 ± 0.063 |
| 42 | rock | C − A | -3.883 ± 1.437 |
| 42 | rock | E − F | 1.134 ± 4.165 |
| 42 | settlement | C − A | -0.022 ± 0.108 |
| 42 | settlement | E − F | -0.061 ± 0.375 |
| 137 | slope | C − A | -0.034 ± 0.059 |
| 137 | slope | E − F | -0.065 ± 0.069 |
| 137 | rock | C − A | 1.571 ± 5.107 |
| 137 | rock | E − F | 3.534 ± 6.076 |
| 137 | settlement | C − A | -0.237 ± 0.937 |
| 137 | settlement | E − F | -0.176 ± 0.333 |
| 271 | slope | C − A | 0.001 ± 0.038 |
| 271 | slope | E − F | 0.009 ± 0.027 |
| 271 | rock | C − A | 2.185 ± 1.833 |
| 271 | rock | E − F | 1.817 ± 2.711 |
| 271 | settlement | C − A | -0.173 ± 0.962 |
| 271 | settlement | E − F | -0.078 ± 0.166 |

## Task-pair experiments

Split 42; shared aggregate checkpoints throughout. Metrics are slope balanced accuracy, rock RMSE (MPa), and settlement RMSE (mm). Missing cells mean that the task is absent from that pair.

| Model | Slope | Rock | Settlement |
| --- | --- | --- | --- |
| Pair_rock_settlement_C_shared | — | 56.286 ± 5.878 | 1.819 ± 0.538 |
| Pair_rock_settlement_E_shared_prior_gate | — | 60.241 ± 6.302 | 0.955 ± 0.160 |
| Pair_slope_rock_C_shared | 0.650 ± 0.056 | 55.008 ± 3.511 | — |
| Pair_slope_rock_E_shared_prior_gate | 0.732 ± 0.062 | 57.050 ± 3.062 | — |
| Pair_slope_settlement_C_shared | 0.664 ± 0.040 | — | 1.505 ± 0.273 |
| Pair_slope_settlement_E_shared_prior_gate | 0.683 ± 0.063 | — | 1.085 ± 0.418 |

## Fixed prior-weight sensitivity

Split 42; predeclared weights, not selected by test performance.

| Prior weight | Slope balanced accuracy | Rock RMSE (MPa) | Settlement RMSE (mm) |
| --- | --- | --- | --- |
| 0.10 | 0.626 ± 0.038 | 54.036 ± 4.834 | 1.413 ± 0.113 |
| 0.25 | 0.650 ± 0.055 | 55.956 ± 2.624 | 1.534 ± 0.532 |
| 0.50 | 0.690 ± 0.083 | 57.861 ± 5.262 | 1.320 ± 0.808 |
| 0.75 | 0.696 ± 0.068 | 62.906 ± 5.865 | 0.771 ± 0.105 |

## Settlement calibration

The affine formula below is fitted on valid training priors only. A close match to a complex model would limit the evidence that sharing is responsible for the settlement gain. It does not prove how the targets were generated.

| Split | Prior multiplier | Intercept (mm) | Test RMSE (mm) |
| --- | --- | --- | --- |
| 42 | 0.8017 | 0.2339 | 0.778 |
| 137 | 0.8010 | 0.2538 | 0.693 |
| 271 | 0.7995 | 0.2775 | 0.726 |

## Completion and remaining limitations

Completed 210 final neural fits, 180 neural tuning fits and 294 classical candidate fits. The complete run took 3.44 minutes on the local CPU. All 1077 saved prediction records passed independent metric recomputation.

The proposal can now describe a concrete benchmark and documented preliminary results. The main remaining research needs are:

- Trace the settlement records to their original source or acquire a documented measured dataset.
- Verify what the two slope grouping codes mean; repeated training partitions do not provide new independent slope test groups.
- Validate on independent projects before making engineering-generalization claims.
- Frame the contribution around identifying useful and harmful sharing, improving prior fusion and testing physical consistency. Broad algorithmic superiority is not established.

`subgroups.csv` provides source, missingness, prior-valid and extrapolation results. `directional_diagnostics.csv` provides case counts and violation rates for every evaluated model. `report.md` contains all checkpoint-specific metrics and paired comparisons.
