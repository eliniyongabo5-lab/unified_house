# Expanded benchmark report

Results are descriptive. Each mean and sample SD is calculated across training seeds within one split, task, model, and selection strategy. Test rows, training seeds, and split seeds are not pooled as independent observations, and no significance tests are reported.

## Interpretation caveats

- These benchmark test sets are reused for exploratory model comparison, so reported performance is exploratory rather than a final untouched-test estimate.
- The slope task uses a fixed test group across split seeds; its split-seed results therefore do not represent independent test samples.
- Settlement data provenance is unknown, which limits claims about generalization and external validity.

- For shared models, `selection=task` rows are separate task-specific checkpoint snapshots and cannot be combined into one deployable unified network. Independent models use separate task networks, so their task-selected snapshots can validly be combined.

## Slope

### Split seed 42

| Model | Selection | Runs | balanced_accuracy | roc_auc |
| --- | --- | --- | --- | --- |
| A_independent | aggregate | 5 | 0.6595 ± 0.0489 | 0.7766 ± 0.0453 |
| A_independent | task | 5 | 0.6491 ± 0.0381 | 0.7760 ± 0.0455 |
| B_independent_prior | aggregate | 5 | 0.6628 ± 0.0281 | 0.8685 ± 0.0310 |
| B_independent_prior | task | 5 | 0.6628 ± 0.0281 | 0.8682 ± 0.0304 |
| C_shared | aggregate | 5 | 0.6536 ± 0.0339 | 0.7858 ± 0.0355 |
| C_shared | task | 5 | 0.6536 ± 0.0339 | 0.7862 ± 0.0335 |
| D_shared_prior | aggregate | 5 | 0.6497 ± 0.0549 | 0.8777 ± 0.0142 |
| D_shared_prior | task | 5 | 0.6583 ± 0.0663 | 0.8765 ± 0.0131 |
| D_shared_weight_0.1 | aggregate | 5 | 0.6260 ± 0.0377 | 0.8206 ± 0.0536 |
| D_shared_weight_0.1 | task | 5 | 0.6286 ± 0.0414 | 0.8195 ± 0.0519 |
| D_shared_weight_0.5 | aggregate | 5 | 0.6899 ± 0.0834 | 0.8745 ± 0.0495 |
| D_shared_weight_0.5 | task | 5 | 0.7011 ± 0.0949 | 0.8742 ± 0.0483 |
| D_shared_weight_0.75 | aggregate | 5 | 0.6965 ± 0.0684 | 0.8807 ± 0.0522 |
| D_shared_weight_0.75 | task | 5 | 0.6965 ± 0.0684 | 0.8795 ± 0.0508 |
| E_shared_prior_gate | aggregate | 5 | 0.6950 ± 0.0825 | 0.8017 ± 0.1013 |
| E_shared_prior_gate | task | 5 | 0.7029 ± 0.0786 | 0.8034 ± 0.0994 |
| F_independent_prior_gate | aggregate | 5 | 0.7029 ± 0.0612 | 0.8099 ± 0.1015 |
| F_independent_prior_gate | task | 5 | 0.6819 ± 0.0507 | 0.8041 ± 0.0909 |
| Pair_slope_rock_C_shared | aggregate | 5 | 0.6498 ± 0.0565 | 0.7718 ± 0.0296 |
| Pair_slope_rock_C_shared | task | 5 | 0.6551 ± 0.0565 | 0.7716 ± 0.0283 |
| Pair_slope_rock_E_shared_prior_gate | aggregate | 5 | 0.7319 ± 0.0621 | 0.8698 ± 0.0607 |
| Pair_slope_rock_E_shared_prior_gate | task | 5 | 0.7267 ± 0.0591 | 0.8682 ± 0.0599 |
| Pair_slope_settlement_C_shared | aggregate | 5 | 0.6641 ± 0.0397 | 0.8101 ± 0.0403 |
| Pair_slope_settlement_C_shared | task | 5 | 0.6622 ± 0.0374 | 0.8110 ± 0.0420 |
| Pair_slope_settlement_E_shared_prior_gate | aggregate | 5 | 0.6826 ± 0.0630 | 0.7894 ± 0.0929 |
| Pair_slope_settlement_E_shared_prior_gate | task | 5 | 0.6852 ± 0.0667 | 0.7892 ± 0.0925 |
| Prior_with_training_mean_fallback | train_only | 1 | 0.7007 (n=1) | 0.8532 (n=1) |
| Tuned_C_shared | aggregate | 5 | 0.6418 ± 0.0185 | 0.7587 ± 0.0340 |
| Tuned_C_shared | task | 5 | 0.6510 ± 0.0233 | 0.7671 ± 0.0132 |
| Tuned_E_shared_prior_gate | aggregate | 5 | 0.6601 ± 0.0198 | 0.7782 ± 0.0442 |
| Tuned_E_shared_prior_gate | task | 5 | 0.6522 ± 0.0164 | 0.7752 ± 0.0454 |
| Tuned_MLP_slope | aggregate | 5 | 0.6207 ± 0.0266 | 0.7453 ± 0.0324 |
| Tuned_MLP_slope | task | 5 | 0.6207 ± 0.0266 | 0.7453 ± 0.0324 |
| dummy | validation | 1 | 0.5000 (n=1) | 0.5000 (n=1) |
| hist_gradient_boosting | validation | 1 | 0.6259 (n=1) | 0.6780 (n=1) |
| logistic_regression | validation | 1 | 0.7768 (n=1) | 0.8782 (n=1) |
| random_forest | validation | 1 | 0.6850 (n=1) | 0.7753 (n=1) |
| rbf_svc | validation | 1 | 0.7077 (n=1) | 0.6749 (n=1) |

### Split seed 137

| Model | Selection | Runs | balanced_accuracy | roc_auc |
| --- | --- | --- | --- | --- |
| A_independent | aggregate | 5 | 0.6964 ± 0.0393 | 0.8421 ± 0.0216 |
| A_independent | task | 5 | 0.6911 ± 0.0454 | 0.8384 ± 0.0231 |
| B_independent_prior | aggregate | 5 | 0.6917 ± 0.0584 | 0.8965 ± 0.0228 |
| B_independent_prior | task | 5 | 0.6838 ± 0.0499 | 0.8920 ± 0.0233 |
| C_shared | aggregate | 5 | 0.6627 ± 0.0530 | 0.8231 ± 0.0315 |
| C_shared | task | 5 | 0.6470 ± 0.0245 | 0.8170 ± 0.0214 |
| D_shared_prior | aggregate | 5 | 0.6674 ± 0.0557 | 0.8837 ± 0.0192 |
| D_shared_prior | task | 5 | 0.6648 ± 0.0587 | 0.8885 ± 0.0141 |
| E_shared_prior_gate | aggregate | 5 | 0.6667 ± 0.0592 | 0.7606 ± 0.0847 |
| E_shared_prior_gate | task | 5 | 0.6450 ± 0.0297 | 0.7369 ± 0.0669 |
| F_independent_prior_gate | aggregate | 5 | 0.7319 ± 0.0590 | 0.8529 ± 0.0652 |
| F_independent_prior_gate | task | 5 | 0.7122 ± 0.0603 | 0.8484 ± 0.0649 |
| Prior_with_training_mean_fallback | train_only | 1 | 0.7337 (n=1) | 0.8532 (n=1) |
| Tuned_C_shared | aggregate | 5 | 0.6543 ± 0.0482 | 0.7938 ± 0.0259 |
| Tuned_C_shared | task | 5 | 0.6530 ± 0.0491 | 0.7965 ± 0.0293 |
| Tuned_E_shared_prior_gate | aggregate | 5 | 0.6707 ± 0.0232 | 0.8358 ± 0.0111 |
| Tuned_E_shared_prior_gate | task | 5 | 0.6695 ± 0.0209 | 0.8357 ± 0.0129 |
| Tuned_MLP_slope | aggregate | 5 | 0.7373 ± 0.0136 | 0.8333 ± 0.0077 |
| Tuned_MLP_slope | task | 5 | 0.7373 ± 0.0136 | 0.8333 ± 0.0077 |
| dummy | validation | 1 | 0.5000 (n=1) | 0.5000 (n=1) |
| hist_gradient_boosting | validation | 1 | 0.6192 (n=1) | 0.7477 (n=1) |
| logistic_regression | validation | 1 | 0.8491 (n=1) | 0.9185 (n=1) |
| random_forest | validation | 1 | 0.6780 (n=1) | 0.8075 (n=1) |
| rbf_svc | validation | 1 | 0.6814 (n=1) | 0.6878 (n=1) |

### Split seed 271

| Model | Selection | Runs | balanced_accuracy | roc_auc |
| --- | --- | --- | --- | --- |
| A_independent | aggregate | 5 | 0.6385 ± 0.0413 | 0.7931 ± 0.0287 |
| A_independent | task | 5 | 0.6411 ± 0.0427 | 0.7860 ± 0.0293 |
| B_independent_prior | aggregate | 5 | 0.6846 ± 0.0418 | 0.8868 ± 0.0181 |
| B_independent_prior | task | 5 | 0.7149 ± 0.0505 | 0.8996 ± 0.0099 |
| C_shared | aggregate | 5 | 0.6398 ± 0.0264 | 0.7611 ± 0.0485 |
| C_shared | task | 5 | 0.6398 ± 0.0264 | 0.7611 ± 0.0485 |
| D_shared_prior | aggregate | 5 | 0.6641 ± 0.0564 | 0.8729 ± 0.0166 |
| D_shared_prior | task | 5 | 0.6345 ± 0.0393 | 0.8570 ± 0.0187 |
| E_shared_prior_gate | aggregate | 5 | 0.7083 ± 0.0500 | 0.7917 ± 0.1121 |
| E_shared_prior_gate | task | 5 | 0.6865 ± 0.0739 | 0.7996 ± 0.0882 |
| F_independent_prior_gate | aggregate | 5 | 0.6991 ± 0.0581 | 0.8059 ± 0.0604 |
| F_independent_prior_gate | task | 5 | 0.6833 ± 0.0535 | 0.7866 ± 0.0509 |
| Prior_with_training_mean_fallback | train_only | 1 | 0.7337 (n=1) | 0.8532 (n=1) |
| Tuned_C_shared | aggregate | 5 | 0.6091 ± 0.0169 | 0.7231 ± 0.0226 |
| Tuned_C_shared | task | 5 | 0.6163 ± 0.0159 | 0.7260 ± 0.0265 |
| Tuned_E_shared_prior_gate | aggregate | 5 | 0.6313 ± 0.0568 | 0.8023 ± 0.0418 |
| Tuned_E_shared_prior_gate | task | 5 | 0.6320 ± 0.0456 | 0.7937 ± 0.0319 |
| Tuned_MLP_slope | aggregate | 5 | 0.6294 ± 0.0202 | 0.7284 ± 0.0302 |
| Tuned_MLP_slope | task | 5 | 0.6294 ± 0.0202 | 0.7284 ± 0.0302 |
| dummy | validation | 1 | 0.5000 (n=1) | 0.5000 (n=1) |
| hist_gradient_boosting | validation | 1 | 0.6391 (n=1) | 0.6780 (n=1) |
| logistic_regression | validation | 1 | 0.8656 (n=1) | 0.9169 (n=1) |
| random_forest | validation | 1 | 0.6584 (n=1) | 0.7988 (n=1) |
| rbf_svc | validation | 1 | 0.6551 (n=1) | 0.6476 (n=1) |

## Rock

### Split seed 42

| Model | Selection | Runs | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- |
| A_independent | aggregate | 5 | 58.315 ± 4.036 | 40.127 ± 1.809 | 0.3443 ± 0.0884 |
| A_independent | task | 5 | 56.208 ± 3.640 | 35.909 ± 3.769 | 0.3911 ± 0.0787 |
| B_independent_prior | aggregate | 5 | 53.085 ± 3.769 | 35.967 ± 3.091 | 0.4566 ± 0.0773 |
| B_independent_prior | task | 5 | 56.146 ± 4.959 | 36.932 ± 4.391 | 0.3907 ± 0.1079 |
| C_shared | aggregate | 5 | 54.433 ± 3.407 | 38.362 ± 1.697 | 0.4291 ± 0.0705 |
| C_shared | task | 5 | 55.341 ± 2.475 | 35.067 ± 3.228 | 0.4108 ± 0.0525 |
| D_shared_prior | aggregate | 5 | 55.956 ± 2.624 | 38.558 ± 2.300 | 0.3975 ± 0.0564 |
| D_shared_prior | task | 5 | 53.397 ± 3.615 | 35.057 ± 4.290 | 0.4503 ± 0.0770 |
| D_shared_weight_0.1 | aggregate | 5 | 54.036 ± 4.834 | 37.634 ± 3.165 | 0.4356 ± 0.0982 |
| D_shared_weight_0.1 | task | 5 | 54.468 ± 3.116 | 33.989 ± 4.189 | 0.4287 ± 0.0663 |
| D_shared_weight_0.5 | aggregate | 5 | 57.861 ± 5.262 | 39.290 ± 3.432 | 0.3527 ± 0.1178 |
| D_shared_weight_0.5 | task | 5 | 56.342 ± 5.068 | 38.045 ± 4.001 | 0.3863 ± 0.1136 |
| D_shared_weight_0.75 | aggregate | 5 | 62.906 ± 5.865 | 41.932 ± 2.762 | 0.2346 ± 0.1380 |
| D_shared_weight_0.75 | task | 5 | 60.003 ± 4.465 | 41.610 ± 2.992 | 0.3054 ± 0.1051 |
| E_shared_prior_gate | aggregate | 5 | 56.658 ± 3.484 | 38.608 ± 2.876 | 0.3816 ± 0.0754 |
| E_shared_prior_gate | task | 5 | 56.380 ± 6.508 | 37.760 ± 4.763 | 0.3829 ± 0.1440 |
| F_independent_prior_gate | aggregate | 5 | 55.524 ± 6.366 | 36.925 ± 4.149 | 0.4016 ± 0.1389 |
| F_independent_prior_gate | task | 5 | 55.827 ± 6.045 | 37.131 ± 3.936 | 0.3958 ± 0.1353 |
| Pair_rock_settlement_C_shared | aggregate | 5 | 56.286 ± 5.878 | 38.356 ± 4.332 | 0.3862 ± 0.1288 |
| Pair_rock_settlement_C_shared | task | 5 | 54.626 ± 1.311 | 32.693 ± 1.782 | 0.4266 ± 0.0273 |
| Pair_rock_settlement_E_shared_prior_gate | aggregate | 5 | 60.241 ± 6.302 | 40.001 ± 4.911 | 0.2969 ± 0.1443 |
| Pair_rock_settlement_E_shared_prior_gate | task | 5 | 54.329 ± 6.166 | 35.761 ± 5.432 | 0.4272 ± 0.1346 |
| Pair_slope_rock_C_shared | aggregate | 5 | 55.008 ± 3.511 | 39.015 ± 2.784 | 0.4169 ± 0.0734 |
| Pair_slope_rock_C_shared | task | 5 | 54.738 ± 3.231 | 34.469 ± 3.432 | 0.4229 ± 0.0679 |
| Pair_slope_rock_E_shared_prior_gate | aggregate | 5 | 57.050 ± 3.062 | 38.595 ± 2.751 | 0.3734 ± 0.0672 |
| Pair_slope_rock_E_shared_prior_gate | task | 5 | 55.213 ± 5.942 | 37.229 ± 4.806 | 0.4090 ± 0.1332 |
| Prior_with_training_mean_fallback | train_only | 1 | 54.307 (n=1) | 38.744 (n=1) | 0.4335 (n=1) |
| Tuned_C_shared | aggregate | 5 | 50.251 ± 5.586 | 34.875 ± 3.142 | 0.5102 ± 0.1063 |
| Tuned_C_shared | task | 5 | 55.824 ± 3.123 | 34.988 ± 3.698 | 0.3999 ± 0.0688 |
| Tuned_E_shared_prior_gate | aggregate | 5 | 57.579 ± 4.564 | 39.672 ± 3.811 | 0.3600 ± 0.0987 |
| Tuned_E_shared_prior_gate | task | 5 | 59.612 ± 1.575 | 41.491 ± 1.552 | 0.3171 ± 0.0359 |
| Tuned_MLP_rock | aggregate | 5 | 54.968 ± 0.448 | 33.869 ± 2.386 | 0.4196 ± 0.0094 |
| Tuned_MLP_rock | task | 5 | 54.968 ± 0.448 | 33.869 ± 2.386 | 0.4196 ± 0.0094 |
| dummy | validation | 1 | 72.232 (n=1) | 47.711 (n=1) | -0.0021 (n=1) |
| hist_gradient_boosting | validation | 1 | 55.843 (n=1) | 36.612 (n=1) | 0.4010 (n=1) |
| random_forest | validation | 1 | 63.067 (n=1) | 38.647 (n=1) | 0.2360 (n=1) |
| rbf_svr | validation | 1 | 68.922 (n=1) | 44.280 (n=1) | 0.0876 (n=1) |
| ridge | validation | 1 | 52.588 (n=1) | 34.852 (n=1) | 0.4688 (n=1) |

### Split seed 137

| Model | Selection | Runs | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- |
| A_independent | aggregate | 5 | 47.044 ± 4.276 | 36.540 ± 3.356 | -0.2334 ± 0.2281 |
| A_independent | task | 5 | 46.718 ± 4.076 | 36.654 ± 3.269 | -0.2158 ± 0.2165 |
| B_independent_prior | aggregate | 5 | 48.288 ± 2.897 | 36.960 ± 2.568 | -0.2947 ± 0.1540 |
| B_independent_prior | task | 5 | 43.707 ± 2.311 | 34.121 ± 1.705 | -0.0600 ± 0.1123 |
| C_shared | aggregate | 5 | 48.614 ± 2.439 | 37.053 ± 1.497 | -0.3111 ± 0.1299 |
| C_shared | task | 5 | 46.350 ± 2.991 | 36.524 ± 2.456 | -0.1934 ± 0.1530 |
| D_shared_prior | aggregate | 5 | 47.882 ± 2.773 | 36.685 ± 1.722 | -0.2728 ± 0.1430 |
| D_shared_prior | task | 5 | 46.668 ± 1.909 | 36.002 ± 1.728 | -0.2075 ± 0.0996 |
| E_shared_prior_gate | aggregate | 5 | 48.468 ± 4.766 | 36.981 ± 3.570 | -0.3107 ± 0.2448 |
| E_shared_prior_gate | task | 5 | 44.893 ± 4.031 | 34.757 ± 2.832 | -0.1230 ± 0.1979 |
| F_independent_prior_gate | aggregate | 5 | 44.934 ± 5.096 | 34.554 ± 3.782 | -0.1294 ± 0.2519 |
| F_independent_prior_gate | task | 5 | 45.015 ± 3.215 | 34.875 ± 2.326 | -0.1265 ± 0.1621 |
| Prior_with_training_mean_fallback | train_only | 1 | 43.259 (n=1) | 34.683 (n=1) | -0.0361 (n=1) |
| Tuned_C_shared | aggregate | 5 | 45.497 ± 2.062 | 35.526 ± 1.323 | -0.1480 ± 0.1053 |
| Tuned_C_shared | task | 5 | 43.731 ± 1.320 | 34.554 ± 0.990 | -0.0596 ± 0.0635 |
| Tuned_E_shared_prior_gate | aggregate | 5 | 45.920 ± 3.630 | 35.711 ± 2.623 | -0.1733 ± 0.1876 |
| Tuned_E_shared_prior_gate | task | 5 | 44.669 ± 1.090 | 34.294 ± 0.763 | -0.1052 ± 0.0538 |
| Tuned_MLP_rock | aggregate | 5 | 45.992 ± 0.336 | 36.231 ± 0.368 | -0.1712 ± 0.0171 |
| Tuned_MLP_rock | task | 5 | 45.992 ± 0.336 | 36.231 ± 0.368 | -0.1712 ± 0.0171 |
| dummy | validation | 1 | 43.302 (n=1) | 35.615 (n=1) | -0.0382 (n=1) |
| hist_gradient_boosting | validation | 1 | 46.455 (n=1) | 35.301 (n=1) | -0.1948 (n=1) |
| random_forest | validation | 1 | 51.096 (n=1) | 36.802 (n=1) | -0.4455 (n=1) |
| rbf_svr | validation | 1 | 42.204 (n=1) | 33.139 (n=1) | 0.0138 (n=1) |
| ridge | validation | 1 | 36.575 (n=1) | 29.026 (n=1) | 0.2593 (n=1) |

### Split seed 271

| Model | Selection | Runs | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- |
| A_independent | aggregate | 5 | 46.454 ± 2.888 | 34.890 ± 2.474 | -0.0237 ± 0.1280 |
| A_independent | task | 5 | 41.475 ± 0.890 | 31.747 ± 0.979 | 0.1862 ± 0.0349 |
| B_independent_prior | aggregate | 5 | 44.953 ± 3.013 | 33.819 ± 2.193 | 0.0409 ± 0.1298 |
| B_independent_prior | task | 5 | 42.127 ± 2.159 | 31.761 ± 1.645 | 0.1590 ± 0.0862 |
| C_shared | aggregate | 5 | 48.639 ± 1.775 | 36.590 ± 1.602 | -0.1200 ± 0.0817 |
| C_shared | task | 5 | 42.880 ± 2.155 | 32.554 ± 1.588 | 0.1287 ± 0.0884 |
| D_shared_prior | aggregate | 5 | 46.351 ± 2.536 | 34.723 ± 1.909 | -0.0185 ± 0.1117 |
| D_shared_prior | task | 5 | 42.468 ± 3.086 | 32.067 ± 2.223 | 0.1435 ± 0.1270 |
| E_shared_prior_gate | aggregate | 5 | 45.381 ± 2.274 | 33.847 ± 1.771 | 0.0241 ± 0.0979 |
| E_shared_prior_gate | task | 5 | 40.661 ± 5.130 | 31.230 ± 3.274 | 0.2081 ± 0.2066 |
| F_independent_prior_gate | aggregate | 5 | 43.563 ± 2.063 | 32.590 ± 1.393 | 0.1009 ± 0.0861 |
| F_independent_prior_gate | task | 5 | 37.788 ± 1.620 | 29.026 ± 0.931 | 0.3237 ± 0.0593 |
| Prior_with_training_mean_fallback | train_only | 1 | 45.288 (n=1) | 33.759 (n=1) | 0.0300 (n=1) |
| Tuned_C_shared | aggregate | 5 | 46.158 ± 0.655 | 34.998 ± 0.463 | -0.0078 ± 0.0286 |
| Tuned_C_shared | task | 5 | 42.531 ± 3.634 | 32.780 ± 2.409 | 0.1395 ± 0.1542 |
| Tuned_E_shared_prior_gate | aggregate | 5 | 46.419 ± 2.731 | 34.943 ± 2.317 | -0.0218 ± 0.1182 |
| Tuned_E_shared_prior_gate | task | 5 | 38.793 ± 2.639 | 29.762 ± 1.829 | 0.2857 ± 0.0984 |
| Tuned_MLP_rock | aggregate | 5 | 42.300 ± 4.122 | 32.488 ± 3.094 | 0.1474 ± 0.1677 |
| Tuned_MLP_rock | task | 5 | 42.300 ± 4.122 | 32.488 ± 3.094 | 0.1474 ± 0.1677 |
| dummy | validation | 1 | 47.267 (n=1) | 39.974 (n=1) | -0.0566 (n=1) |
| hist_gradient_boosting | validation | 1 | 41.300 (n=1) | 30.842 (n=1) | 0.1933 (n=1) |
| random_forest | validation | 1 | 37.665 (n=1) | 28.923 (n=1) | 0.3291 (n=1) |
| rbf_svr | validation | 1 | 41.148 (n=1) | 31.476 (n=1) | 0.1993 (n=1) |
| ridge | validation | 1 | 37.110 (n=1) | 28.773 (n=1) | 0.3487 (n=1) |

## Settlement

### Split seed 42

| Model | Selection | Runs | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- |
| A_independent | aggregate | 5 | 1.597 ± 0.059 | 1.128 ± 0.061 | 0.9760 ± 0.0018 |
| A_independent | task | 5 | 1.663 ± 0.122 | 1.168 ± 0.076 | 0.9739 ± 0.0038 |
| B_independent_prior | aggregate | 5 | 1.134 ± 0.081 | 0.804 ± 0.077 | 0.9879 ± 0.0018 |
| B_independent_prior | task | 5 | 1.230 ± 0.122 | 0.870 ± 0.125 | 0.9857 ± 0.0029 |
| C_shared | aggregate | 5 | 1.575 ± 0.145 | 1.087 ± 0.090 | 0.9765 ± 0.0044 |
| C_shared | task | 5 | 1.742 ± 0.270 | 1.170 ± 0.255 | 0.9710 ± 0.0097 |
| Calibrated_settlement_prior | train_only | 1 | 0.778 (n=1) | 0.597 (n=1) | 0.9943 (n=1) |
| D_shared_prior | aggregate | 5 | 1.534 ± 0.532 | 1.058 ± 0.306 | 0.9758 ± 0.0186 |
| D_shared_prior | task | 5 | 1.436 ± 0.170 | 1.026 ± 0.117 | 0.9804 ± 0.0044 |
| D_shared_weight_0.1 | aggregate | 5 | 1.413 ± 0.113 | 0.969 ± 0.089 | 0.9812 ± 0.0031 |
| D_shared_weight_0.1 | task | 5 | 1.604 ± 0.242 | 1.115 ± 0.227 | 0.9754 ± 0.0076 |
| D_shared_weight_0.5 | aggregate | 5 | 1.320 ± 0.808 | 0.890 ± 0.407 | 0.9787 ± 0.0274 |
| D_shared_weight_0.5 | task | 5 | 0.949 ± 0.152 | 0.711 ± 0.098 | 0.9914 ± 0.0027 |
| D_shared_weight_0.75 | aggregate | 5 | 0.771 ± 0.105 | 0.590 ± 0.085 | 0.9943 ± 0.0017 |
| D_shared_weight_0.75 | task | 5 | 0.733 ± 0.025 | 0.569 ± 0.022 | 0.9949 ± 0.0003 |
| E_shared_prior_gate | aggregate | 5 | 1.015 ± 0.238 | 0.746 ± 0.198 | 0.9899 ± 0.0047 |
| E_shared_prior_gate | task | 5 | 0.770 ± 0.040 | 0.555 ± 0.021 | 0.9944 ± 0.0006 |
| F_independent_prior_gate | aggregate | 5 | 1.076 ± 0.446 | 0.822 ± 0.367 | 0.9876 ± 0.0112 |
| F_independent_prior_gate | task | 5 | 0.778 ± 0.011 | 0.575 ± 0.026 | 0.9943 ± 0.0002 |
| Pair_rock_settlement_C_shared | aggregate | 5 | 1.819 ± 0.538 | 1.234 ± 0.310 | 0.9668 ± 0.0218 |
| Pair_rock_settlement_C_shared | task | 5 | 1.731 ± 0.253 | 1.089 ± 0.143 | 0.9714 ± 0.0085 |
| Pair_rock_settlement_E_shared_prior_gate | aggregate | 5 | 0.955 ± 0.160 | 0.682 ± 0.124 | 0.9912 ± 0.0029 |
| Pair_rock_settlement_E_shared_prior_gate | task | 5 | 0.774 ± 0.027 | 0.571 ± 0.012 | 0.9944 ± 0.0004 |
| Pair_slope_settlement_C_shared | aggregate | 5 | 1.505 ± 0.273 | 1.058 ± 0.197 | 0.9782 ± 0.0076 |
| Pair_slope_settlement_C_shared | task | 5 | 1.604 ± 0.136 | 1.168 ± 0.100 | 0.9757 ± 0.0039 |
| Pair_slope_settlement_E_shared_prior_gate | aggregate | 5 | 1.085 ± 0.418 | 0.823 ± 0.362 | 0.9876 ± 0.0096 |
| Pair_slope_settlement_E_shared_prior_gate | task | 5 | 0.753 ± 0.033 | 0.564 ± 0.017 | 0.9947 ± 0.0005 |
| Prior_with_training_mean_fallback | train_only | 1 | 3.577 (n=1) | 2.212 (n=1) | 0.8799 (n=1) |
| Tuned_C_shared | aggregate | 5 | 1.513 ± 0.090 | 1.031 ± 0.114 | 0.9785 ± 0.0026 |
| Tuned_C_shared | task | 5 | 1.510 ± 0.106 | 1.059 ± 0.121 | 0.9785 ± 0.0030 |
| Tuned_E_shared_prior_gate | aggregate | 5 | 0.766 ± 0.006 | 0.583 ± 0.005 | 0.9945 ± 0.0001 |
| Tuned_E_shared_prior_gate | task | 5 | 0.768 ± 0.005 | 0.584 ± 0.005 | 0.9945 ± 0.0001 |
| Tuned_MLP_settlement | aggregate | 5 | 1.490 ± 0.213 | 1.043 ± 0.135 | 0.9788 ± 0.0058 |
| Tuned_MLP_settlement | task | 5 | 1.490 ± 0.213 | 1.043 ± 0.135 | 0.9788 ± 0.0058 |
| dummy | validation | 1 | 10.358 (n=1) | 7.746 (n=1) | -0.0071 (n=1) |
| hist_gradient_boosting | validation | 1 | 2.518 (n=1) | 1.473 (n=1) | 0.9405 (n=1) |
| random_forest | validation | 1 | 3.506 (n=1) | 1.957 (n=1) | 0.8846 (n=1) |
| rbf_svr | validation | 1 | 1.849 (n=1) | 1.226 (n=1) | 0.9679 (n=1) |
| ridge | validation | 1 | 4.392 (n=1) | 2.983 (n=1) | 0.8190 (n=1) |

### Split seed 137

| Model | Selection | Runs | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- |
| A_independent | aggregate | 5 | 2.841 ± 0.647 | 1.686 ± 0.381 | 0.8975 ± 0.0420 |
| A_independent | task | 5 | 1.883 ± 0.301 | 1.067 ± 0.115 | 0.9559 ± 0.0131 |
| B_independent_prior | aggregate | 5 | 1.973 ± 0.593 | 1.178 ± 0.320 | 0.9491 ± 0.0282 |
| B_independent_prior | task | 5 | 1.455 ± 0.222 | 0.900 ± 0.094 | 0.9737 ± 0.0074 |
| C_shared | aggregate | 5 | 2.603 ± 0.578 | 1.470 ± 0.326 | 0.9141 ± 0.0405 |
| C_shared | task | 5 | 2.102 ± 0.407 | 1.107 ± 0.136 | 0.9445 ± 0.0201 |
| Calibrated_settlement_prior | train_only | 1 | 0.693 (n=1) | 0.550 (n=1) | 0.9941 (n=1) |
| D_shared_prior | aggregate | 5 | 1.897 ± 0.534 | 1.122 ± 0.295 | 0.9534 ± 0.0288 |
| D_shared_prior | task | 5 | 1.471 ± 0.133 | 0.881 ± 0.053 | 0.9734 ± 0.0047 |
| E_shared_prior_gate | aggregate | 5 | 0.821 ± 0.131 | 0.623 ± 0.091 | 0.9916 ± 0.0028 |
| E_shared_prior_gate | task | 5 | 0.725 ± 0.046 | 0.563 ± 0.038 | 0.9936 ± 0.0008 |
| F_independent_prior_gate | aggregate | 5 | 0.997 ± 0.298 | 0.747 ± 0.240 | 0.9870 ± 0.0080 |
| F_independent_prior_gate | task | 5 | 0.734 ± 0.035 | 0.569 ± 0.030 | 0.9934 ± 0.0006 |
| Prior_with_training_mean_fallback | train_only | 1 | 3.189 (n=1) | 2.109 (n=1) | 0.8760 (n=1) |
| Tuned_C_shared | aggregate | 5 | 2.196 ± 0.130 | 1.252 ± 0.073 | 0.9410 ± 0.0071 |
| Tuned_C_shared | task | 5 | 2.051 ± 0.150 | 1.208 ± 0.126 | 0.9485 ± 0.0075 |
| Tuned_E_shared_prior_gate | aggregate | 5 | 0.690 ± 0.007 | 0.525 ± 0.003 | 0.9942 ± 0.0001 |
| Tuned_E_shared_prior_gate | task | 5 | 0.689 ± 0.006 | 0.525 ± 0.005 | 0.9942 ± 0.0001 |
| Tuned_MLP_settlement | aggregate | 5 | 2.050 ± 0.295 | 1.202 ± 0.232 | 0.9479 ± 0.0159 |
| Tuned_MLP_settlement | task | 5 | 2.050 ± 0.295 | 1.202 ± 0.232 | 0.9479 ± 0.0159 |
| dummy | validation | 1 | 9.058 (n=1) | 7.008 (n=1) | -0.0003 (n=1) |
| hist_gradient_boosting | validation | 1 | 2.894 (n=1) | 1.679 (n=1) | 0.8979 (n=1) |
| random_forest | validation | 1 | 2.776 (n=1) | 1.684 (n=1) | 0.9061 (n=1) |
| rbf_svr | validation | 1 | 2.731 (n=1) | 1.482 (n=1) | 0.9091 (n=1) |
| ridge | validation | 1 | 4.273 (n=1) | 2.859 (n=1) | 0.7774 (n=1) |

### Split seed 271

| Model | Selection | Runs | rmse | mae | r2 |
| --- | --- | --- | --- | --- | --- |
| A_independent | aggregate | 5 | 3.531 ± 0.549 | 1.760 ± 0.255 | 0.9175 ± 0.0230 |
| A_independent | task | 5 | 2.726 ± 0.324 | 1.457 ± 0.155 | 0.9512 ± 0.0108 |
| B_independent_prior | aggregate | 5 | 2.741 ± 0.556 | 1.413 ± 0.225 | 0.9496 ± 0.0194 |
| B_independent_prior | task | 5 | 1.983 ± 0.309 | 1.147 ± 0.147 | 0.9740 ± 0.0080 |
| C_shared | aggregate | 5 | 3.358 ± 0.957 | 1.687 ± 0.336 | 0.9221 ± 0.0440 |
| C_shared | task | 5 | 2.483 ± 0.201 | 1.396 ± 0.149 | 0.9598 ± 0.0066 |
| Calibrated_settlement_prior | train_only | 1 | 0.726 (n=1) | 0.554 (n=1) | 0.9966 (n=1) |
| D_shared_prior | aggregate | 5 | 2.573 ± 0.676 | 1.372 ± 0.263 | 0.9546 ± 0.0230 |
| D_shared_prior | task | 5 | 1.792 ± 0.119 | 1.076 ± 0.075 | 0.9791 ± 0.0029 |
| E_shared_prior_gate | aggregate | 5 | 0.875 ± 0.111 | 0.625 ± 0.062 | 0.9950 ± 0.0013 |
| E_shared_prior_gate | task | 5 | 0.781 ± 0.055 | 0.567 ± 0.033 | 0.9960 ± 0.0006 |
| F_independent_prior_gate | aggregate | 5 | 0.954 ± 0.265 | 0.663 ± 0.144 | 0.9937 ± 0.0037 |
| F_independent_prior_gate | task | 5 | 0.788 ± 0.060 | 0.585 ± 0.026 | 0.9959 ± 0.0006 |
| Prior_with_training_mean_fallback | train_only | 1 | 3.948 (n=1) | 2.444 (n=1) | 0.8988 (n=1) |
| Tuned_C_shared | aggregate | 5 | 3.307 ± 0.220 | 1.458 ± 0.117 | 0.9287 ± 0.0096 |
| Tuned_C_shared | task | 5 | 3.066 ± 0.116 | 1.394 ± 0.069 | 0.9389 ± 0.0046 |
| Tuned_E_shared_prior_gate | aggregate | 5 | 0.725 ± 0.051 | 0.552 ± 0.034 | 0.9966 ± 0.0005 |
| Tuned_E_shared_prior_gate | task | 5 | 0.723 ± 0.042 | 0.552 ± 0.027 | 0.9966 ± 0.0004 |
| Tuned_MLP_settlement | aggregate | 5 | 3.562 ± 0.334 | 1.622 ± 0.163 | 0.9170 ± 0.0156 |
| Tuned_MLP_settlement | task | 5 | 3.562 ± 0.334 | 1.622 ± 0.163 | 0.9170 ± 0.0156 |
| dummy | validation | 1 | 12.615 (n=1) | 8.180 (n=1) | -0.0331 (n=1) |
| hist_gradient_boosting | validation | 1 | 4.471 (n=1) | 1.839 (n=1) | 0.8702 (n=1) |
| random_forest | validation | 1 | 4.270 (n=1) | 1.837 (n=1) | 0.8816 (n=1) |
| rbf_svr | validation | 1 | 3.489 (n=1) | 1.783 (n=1) | 0.9210 (n=1) |
| ridge | validation | 1 | 7.064 (n=1) | 3.838 (n=1) | 0.6761 (n=1) |

## Paired training-seed differences

Model differences are left model minus right model, paired only where split seed, task, selection strategy, and training seed match. Checkpoint contrasts are task-selected minus aggregate-selected results for the same model, task, split, and training seed. All values are descriptive means and sample SDs; no p-values or significance tests are used.

| Contrast | Split | Task | Selection | Metric | Pairs | Mean difference ± sample SD |
| --- | --- | --- | --- | --- | --- | --- |
| C_shared − A_independent | 42 | rock | aggregate | mae | 5 | -1.766 ± 1.762 |
| C_shared − A_independent | 42 | rock | aggregate | rmse | 5 | -3.883 ± 1.437 |
| C_shared − A_independent | 42 | rock | aggregate | r2 | 5 | 0.0848 ± 0.0339 |
| C_shared − A_independent | 42 | settlement | aggregate | mae | 5 | -0.041 ± 0.102 |
| C_shared − A_independent | 42 | settlement | aggregate | rmse | 5 | -0.022 ± 0.108 |
| C_shared − A_independent | 42 | settlement | aggregate | r2 | 5 | 0.0005 ± 0.0032 |
| C_shared − A_independent | 42 | slope | aggregate | balanced_accuracy | 5 | -0.0060 ± 0.0297 |
| C_shared − A_independent | 42 | slope | aggregate | roc_auc | 5 | 0.0092 ± 0.0321 |
| C_shared − A_independent | 137 | rock | aggregate | mae | 5 | 0.514 ± 3.745 |
| C_shared − A_independent | 137 | rock | aggregate | rmse | 5 | 1.571 ± 5.107 |
| C_shared − A_independent | 137 | rock | aggregate | r2 | 5 | -0.0777 ± 0.2749 |
| C_shared − A_independent | 137 | settlement | aggregate | mae | 5 | -0.215 ± 0.581 |
| C_shared − A_independent | 137 | settlement | aggregate | rmse | 5 | -0.237 ± 0.937 |
| C_shared − A_independent | 137 | settlement | aggregate | r2 | 5 | 0.0166 ± 0.0640 |
| C_shared − A_independent | 137 | slope | aggregate | balanced_accuracy | 5 | -0.0336 ± 0.0595 |
| C_shared − A_independent | 137 | slope | aggregate | roc_auc | 5 | -0.0190 ± 0.0238 |
| C_shared − A_independent | 271 | rock | aggregate | mae | 5 | 1.699 ± 1.496 |
| C_shared − A_independent | 271 | rock | aggregate | rmse | 5 | 2.185 ± 1.833 |
| C_shared − A_independent | 271 | rock | aggregate | r2 | 5 | -0.0963 ± 0.0802 |
| C_shared − A_independent | 271 | settlement | aggregate | mae | 5 | -0.073 ± 0.304 |
| C_shared − A_independent | 271 | settlement | aggregate | rmse | 5 | -0.173 ± 0.962 |
| C_shared − A_independent | 271 | settlement | aggregate | r2 | 5 | 0.0046 ± 0.0423 |
| C_shared − A_independent | 271 | slope | aggregate | balanced_accuracy | 5 | 0.0013 ± 0.0384 |
| C_shared − A_independent | 271 | slope | aggregate | roc_auc | 5 | -0.0320 ± 0.0485 |
| E_shared_prior_gate − F_independent_prior_gate | 42 | rock | aggregate | mae | 5 | 1.683 ± 2.232 |
| E_shared_prior_gate − F_independent_prior_gate | 42 | rock | aggregate | rmse | 5 | 1.134 ± 4.165 |
| E_shared_prior_gate − F_independent_prior_gate | 42 | rock | aggregate | r2 | 5 | -0.0201 ± 0.0920 |
| E_shared_prior_gate − F_independent_prior_gate | 42 | rock | task | mae | 5 | 0.630 ± 1.563 |
| E_shared_prior_gate − F_independent_prior_gate | 42 | rock | task | rmse | 5 | 0.553 ± 2.186 |
| E_shared_prior_gate − F_independent_prior_gate | 42 | rock | task | r2 | 5 | -0.0128 ± 0.0505 |
| E_shared_prior_gate − F_independent_prior_gate | 42 | settlement | aggregate | mae | 5 | -0.076 ± 0.299 |
| E_shared_prior_gate − F_independent_prior_gate | 42 | settlement | aggregate | rmse | 5 | -0.061 ± 0.375 |
| E_shared_prior_gate − F_independent_prior_gate | 42 | settlement | aggregate | r2 | 5 | 0.0023 ± 0.0091 |
| E_shared_prior_gate − F_independent_prior_gate | 42 | settlement | task | mae | 5 | -0.020 ± 0.014 |
| E_shared_prior_gate − F_independent_prior_gate | 42 | settlement | task | rmse | 5 | -0.008 ± 0.034 |
| E_shared_prior_gate − F_independent_prior_gate | 42 | settlement | task | r2 | 5 | 0.0001 ± 0.0005 |
| E_shared_prior_gate − F_independent_prior_gate | 42 | slope | aggregate | balanced_accuracy | 5 | -0.0079 ± 0.0630 |
| E_shared_prior_gate − F_independent_prior_gate | 42 | slope | aggregate | roc_auc | 5 | -0.0083 ± 0.1055 |
| E_shared_prior_gate − F_independent_prior_gate | 42 | slope | task | balanced_accuracy | 5 | 0.0211 ± 0.0798 |
| E_shared_prior_gate − F_independent_prior_gate | 42 | slope | task | roc_auc | 5 | -0.0007 ± 0.1066 |
| E_shared_prior_gate − F_independent_prior_gate | 137 | rock | aggregate | mae | 5 | 2.427 ± 4.500 |
| E_shared_prior_gate − F_independent_prior_gate | 137 | rock | aggregate | rmse | 5 | 3.534 ± 6.076 |
| E_shared_prior_gate − F_independent_prior_gate | 137 | rock | aggregate | r2 | 5 | -0.1813 ± 0.3076 |
| E_shared_prior_gate − F_independent_prior_gate | 137 | rock | task | mae | 5 | -0.118 ± 2.140 |
| E_shared_prior_gate − F_independent_prior_gate | 137 | rock | task | rmse | 5 | -0.122 ± 3.694 |
| E_shared_prior_gate − F_independent_prior_gate | 137 | rock | task | r2 | 5 | 0.0034 ± 0.1827 |
| E_shared_prior_gate − F_independent_prior_gate | 137 | settlement | aggregate | mae | 5 | -0.124 ± 0.285 |
| E_shared_prior_gate − F_independent_prior_gate | 137 | settlement | aggregate | rmse | 5 | -0.176 ± 0.333 |
| E_shared_prior_gate − F_independent_prior_gate | 137 | settlement | aggregate | r2 | 5 | 0.0046 ± 0.0088 |
| E_shared_prior_gate − F_independent_prior_gate | 137 | settlement | task | mae | 5 | -0.006 ± 0.059 |
| E_shared_prior_gate − F_independent_prior_gate | 137 | settlement | task | rmse | 5 | -0.009 ± 0.076 |
| E_shared_prior_gate − F_independent_prior_gate | 137 | settlement | task | r2 | 5 | 0.0002 ± 0.0013 |
| E_shared_prior_gate − F_independent_prior_gate | 137 | slope | aggregate | balanced_accuracy | 5 | -0.0652 ± 0.0685 |
| E_shared_prior_gate − F_independent_prior_gate | 137 | slope | aggregate | roc_auc | 5 | -0.0924 ± 0.0642 |
| E_shared_prior_gate − F_independent_prior_gate | 137 | slope | task | balanced_accuracy | 5 | -0.0671 ± 0.0814 |
| E_shared_prior_gate − F_independent_prior_gate | 137 | slope | task | roc_auc | 5 | -0.1115 ± 0.0970 |
| E_shared_prior_gate − F_independent_prior_gate | 271 | rock | aggregate | mae | 5 | 1.256 ± 1.973 |
| E_shared_prior_gate − F_independent_prior_gate | 271 | rock | aggregate | rmse | 5 | 1.817 ± 2.711 |
| E_shared_prior_gate − F_independent_prior_gate | 271 | rock | aggregate | r2 | 5 | -0.0768 ± 0.1159 |
| E_shared_prior_gate − F_independent_prior_gate | 271 | rock | task | mae | 5 | 2.203 ± 3.775 |
| E_shared_prior_gate − F_independent_prior_gate | 271 | rock | task | rmse | 5 | 2.874 ± 5.673 |
| E_shared_prior_gate − F_independent_prior_gate | 271 | rock | task | r2 | 5 | -0.1156 ± 0.2266 |
| E_shared_prior_gate − F_independent_prior_gate | 271 | settlement | aggregate | mae | 5 | -0.038 ± 0.091 |
| E_shared_prior_gate − F_independent_prior_gate | 271 | settlement | aggregate | rmse | 5 | -0.078 ± 0.166 |
| E_shared_prior_gate − F_independent_prior_gate | 271 | settlement | aggregate | r2 | 5 | 0.0012 ± 0.0025 |
| E_shared_prior_gate − F_independent_prior_gate | 271 | settlement | task | mae | 5 | -0.018 ± 0.014 |
| E_shared_prior_gate − F_independent_prior_gate | 271 | settlement | task | rmse | 5 | -0.008 ± 0.049 |
| E_shared_prior_gate − F_independent_prior_gate | 271 | settlement | task | r2 | 5 | 0.0001 ± 0.0005 |
| E_shared_prior_gate − F_independent_prior_gate | 271 | slope | aggregate | balanced_accuracy | 5 | 0.0092 ± 0.0267 |
| E_shared_prior_gate − F_independent_prior_gate | 271 | slope | aggregate | roc_auc | 5 | -0.0141 ± 0.0960 |
| E_shared_prior_gate − F_independent_prior_gate | 271 | slope | task | balanced_accuracy | 5 | 0.0033 ± 0.0482 |
| E_shared_prior_gate − F_independent_prior_gate | 271 | slope | task | roc_auc | 5 | 0.0130 ± 0.0843 |
| E_shared_prior_gate − D_shared_prior | 42 | rock | aggregate | mae | 5 | 0.050 ± 2.922 |
| E_shared_prior_gate − D_shared_prior | 42 | rock | aggregate | rmse | 5 | 0.702 ± 2.624 |
| E_shared_prior_gate − D_shared_prior | 42 | rock | aggregate | r2 | 5 | -0.0160 ± 0.0574 |
| E_shared_prior_gate − D_shared_prior | 42 | rock | task | mae | 5 | 2.703 ± 6.726 |
| E_shared_prior_gate − D_shared_prior | 42 | rock | task | rmse | 5 | 2.984 ± 8.115 |
| E_shared_prior_gate − D_shared_prior | 42 | rock | task | r2 | 5 | -0.0674 ± 0.1785 |
| E_shared_prior_gate − D_shared_prior | 42 | settlement | aggregate | mae | 5 | -0.312 ± 0.260 |
| E_shared_prior_gate − D_shared_prior | 42 | settlement | aggregate | rmse | 5 | -0.518 ± 0.457 |
| E_shared_prior_gate − D_shared_prior | 42 | settlement | aggregate | r2 | 5 | 0.0141 ± 0.0167 |
| E_shared_prior_gate − D_shared_prior | 42 | settlement | task | mae | 5 | -0.471 ± 0.123 |
| E_shared_prior_gate − D_shared_prior | 42 | settlement | task | rmse | 5 | -0.667 ± 0.188 |
| E_shared_prior_gate − D_shared_prior | 42 | settlement | task | r2 | 5 | 0.0140 ± 0.0047 |
| E_shared_prior_gate − D_shared_prior | 42 | slope | aggregate | balanced_accuracy | 5 | 0.0454 ± 0.0743 |
| E_shared_prior_gate − D_shared_prior | 42 | slope | aggregate | roc_auc | 5 | -0.0761 ± 0.1100 |
| E_shared_prior_gate − D_shared_prior | 42 | slope | task | balanced_accuracy | 5 | 0.0447 ± 0.0681 |
| E_shared_prior_gate − D_shared_prior | 42 | slope | task | roc_auc | 5 | -0.0731 ± 0.1099 |
| E_shared_prior_gate − D_shared_prior | 137 | rock | aggregate | mae | 5 | 0.295 ± 2.459 |
| E_shared_prior_gate − D_shared_prior | 137 | rock | aggregate | rmse | 5 | 0.586 ± 3.293 |
| E_shared_prior_gate − D_shared_prior | 137 | rock | aggregate | r2 | 5 | -0.0379 ± 0.1766 |
| E_shared_prior_gate − D_shared_prior | 137 | rock | task | mae | 5 | -1.245 ± 2.112 |
| E_shared_prior_gate − D_shared_prior | 137 | rock | task | rmse | 5 | -1.775 ± 3.103 |
| E_shared_prior_gate − D_shared_prior | 137 | rock | task | r2 | 5 | 0.0844 ± 0.1513 |
| E_shared_prior_gate − D_shared_prior | 137 | settlement | aggregate | mae | 5 | -0.500 ± 0.224 |
| E_shared_prior_gate − D_shared_prior | 137 | settlement | aggregate | rmse | 5 | -1.076 ± 0.422 |
| E_shared_prior_gate − D_shared_prior | 137 | settlement | aggregate | r2 | 5 | 0.0383 ± 0.0262 |
| E_shared_prior_gate − D_shared_prior | 137 | settlement | task | mae | 5 | -0.318 ± 0.027 |
| E_shared_prior_gate − D_shared_prior | 137 | settlement | task | rmse | 5 | -0.747 ± 0.134 |
| E_shared_prior_gate − D_shared_prior | 137 | settlement | task | r2 | 5 | 0.0201 ± 0.0047 |
| E_shared_prior_gate − D_shared_prior | 137 | slope | aggregate | balanced_accuracy | 5 | -0.0007 ± 0.0224 |
| E_shared_prior_gate − D_shared_prior | 137 | slope | aggregate | roc_auc | 5 | -0.1231 ± 0.0690 |
| E_shared_prior_gate − D_shared_prior | 137 | slope | task | balanced_accuracy | 5 | -0.0197 ± 0.0371 |
| E_shared_prior_gate − D_shared_prior | 137 | slope | task | roc_auc | 5 | -0.1516 ± 0.0564 |
| E_shared_prior_gate − D_shared_prior | 271 | rock | aggregate | mae | 5 | -0.877 ± 1.187 |
| E_shared_prior_gate − D_shared_prior | 271 | rock | aggregate | rmse | 5 | -0.970 ± 1.527 |
| E_shared_prior_gate − D_shared_prior | 271 | rock | aggregate | r2 | 5 | 0.0426 ± 0.0659 |
| E_shared_prior_gate − D_shared_prior | 271 | rock | task | mae | 5 | -0.837 ± 1.462 |
| E_shared_prior_gate − D_shared_prior | 271 | rock | task | rmse | 5 | -1.807 ± 2.343 |
| E_shared_prior_gate − D_shared_prior | 271 | rock | task | r2 | 5 | 0.0647 ± 0.0906 |
| E_shared_prior_gate − D_shared_prior | 271 | settlement | aggregate | mae | 5 | -0.747 ± 0.263 |
| E_shared_prior_gate − D_shared_prior | 271 | settlement | aggregate | rmse | 5 | -1.698 ± 0.701 |
| E_shared_prior_gate − D_shared_prior | 271 | settlement | aggregate | r2 | 5 | 0.0403 ± 0.0232 |
| E_shared_prior_gate − D_shared_prior | 271 | settlement | task | mae | 5 | -0.509 ± 0.107 |
| E_shared_prior_gate − D_shared_prior | 271 | settlement | task | rmse | 5 | -1.011 ± 0.166 |
| E_shared_prior_gate − D_shared_prior | 271 | settlement | task | r2 | 5 | 0.0169 ± 0.0033 |
| E_shared_prior_gate − D_shared_prior | 271 | slope | aggregate | balanced_accuracy | 5 | 0.0441 ± 0.0543 |
| E_shared_prior_gate − D_shared_prior | 271 | slope | aggregate | roc_auc | 5 | -0.0811 ± 0.1026 |
| E_shared_prior_gate − D_shared_prior | 271 | slope | task | balanced_accuracy | 5 | 0.0520 ± 0.0614 |
| E_shared_prior_gate − D_shared_prior | 271 | slope | task | roc_auc | 5 | -0.0574 ± 0.0888 |
| F_independent_prior_gate − B_independent_prior | 42 | rock | aggregate | mae | 5 | 0.958 ± 5.138 |
| F_independent_prior_gate − B_independent_prior | 42 | rock | aggregate | rmse | 5 | 2.439 ± 6.491 |
| F_independent_prior_gate − B_independent_prior | 42 | rock | aggregate | r2 | 5 | -0.0549 ± 0.1381 |
| F_independent_prior_gate − B_independent_prior | 42 | rock | task | mae | 5 | 0.199 ± 1.623 |
| F_independent_prior_gate − B_independent_prior | 42 | rock | task | rmse | 5 | -0.319 ± 2.246 |
| F_independent_prior_gate − B_independent_prior | 42 | rock | task | r2 | 5 | 0.0050 ± 0.0507 |
| F_independent_prior_gate − B_independent_prior | 42 | settlement | aggregate | mae | 5 | 0.019 ± 0.309 |
| F_independent_prior_gate − B_independent_prior | 42 | settlement | aggregate | rmse | 5 | -0.058 ± 0.369 |
| F_independent_prior_gate − B_independent_prior | 42 | settlement | aggregate | r2 | 5 | -0.0002 ± 0.0095 |
| F_independent_prior_gate − B_independent_prior | 42 | settlement | task | mae | 5 | -0.295 ± 0.128 |
| F_independent_prior_gate − B_independent_prior | 42 | settlement | task | rmse | 5 | -0.452 ± 0.117 |
| F_independent_prior_gate − B_independent_prior | 42 | settlement | task | r2 | 5 | 0.0086 ± 0.0028 |
| F_independent_prior_gate − B_independent_prior | 42 | slope | aggregate | balanced_accuracy | 5 | 0.0401 ± 0.0510 |
| F_independent_prior_gate − B_independent_prior | 42 | slope | aggregate | roc_auc | 5 | -0.0586 ± 0.0844 |
| F_independent_prior_gate − B_independent_prior | 42 | slope | task | balanced_accuracy | 5 | 0.0191 ± 0.0662 |
| F_independent_prior_gate − B_independent_prior | 42 | slope | task | roc_auc | 5 | -0.0641 ± 0.0930 |
| F_independent_prior_gate − B_independent_prior | 137 | rock | aggregate | mae | 5 | -2.407 ± 3.953 |
| F_independent_prior_gate − B_independent_prior | 137 | rock | aggregate | rmse | 5 | -3.354 ± 5.524 |
| F_independent_prior_gate − B_independent_prior | 137 | rock | aggregate | r2 | 5 | 0.1653 ± 0.2735 |
| F_independent_prior_gate − B_independent_prior | 137 | rock | task | mae | 5 | 0.754 ± 0.904 |
| F_independent_prior_gate − B_independent_prior | 137 | rock | task | rmse | 5 | 1.308 ± 1.233 |
| F_independent_prior_gate − B_independent_prior | 137 | rock | task | r2 | 5 | -0.0665 ± 0.0648 |
| F_independent_prior_gate − B_independent_prior | 137 | settlement | aggregate | mae | 5 | -0.431 ± 0.403 |
| F_independent_prior_gate − B_independent_prior | 137 | settlement | aggregate | rmse | 5 | -0.976 ± 0.597 |
| F_independent_prior_gate − B_independent_prior | 137 | settlement | aggregate | r2 | 5 | 0.0379 ± 0.0283 |
| F_independent_prior_gate − B_independent_prior | 137 | settlement | task | mae | 5 | -0.330 ± 0.074 |
| F_independent_prior_gate − B_independent_prior | 137 | settlement | task | rmse | 5 | -0.721 ± 0.200 |
| F_independent_prior_gate − B_independent_prior | 137 | settlement | task | r2 | 5 | 0.0197 ± 0.0070 |
| F_independent_prior_gate − B_independent_prior | 137 | slope | aggregate | balanced_accuracy | 5 | 0.0401 ± 0.0414 |
| F_independent_prior_gate − B_independent_prior | 137 | slope | aggregate | roc_auc | 5 | -0.0436 ± 0.0512 |
| F_independent_prior_gate − B_independent_prior | 137 | slope | task | balanced_accuracy | 5 | 0.0283 ± 0.0490 |
| F_independent_prior_gate − B_independent_prior | 137 | slope | task | roc_auc | 5 | -0.0436 ± 0.0653 |
| F_independent_prior_gate − B_independent_prior | 271 | rock | aggregate | mae | 5 | -1.229 ± 3.088 |
| F_independent_prior_gate − B_independent_prior | 271 | rock | aggregate | rmse | 5 | -1.390 ± 4.502 |
| F_independent_prior_gate − B_independent_prior | 271 | rock | aggregate | r2 | 5 | 0.0600 ± 0.1913 |
| F_independent_prior_gate − B_independent_prior | 271 | rock | task | mae | 5 | -2.735 ± 2.322 |
| F_independent_prior_gate − B_independent_prior | 271 | rock | task | rmse | 5 | -4.339 ± 3.108 |
| F_independent_prior_gate − B_independent_prior | 271 | rock | task | r2 | 5 | 0.1647 ± 0.1205 |
| F_independent_prior_gate − B_independent_prior | 271 | settlement | aggregate | mae | 5 | -0.750 ± 0.250 |
| F_independent_prior_gate − B_independent_prior | 271 | settlement | aggregate | rmse | 5 | -1.787 ± 0.636 |
| F_independent_prior_gate − B_independent_prior | 271 | settlement | aggregate | r2 | 5 | 0.0441 ± 0.0204 |
| F_independent_prior_gate − B_independent_prior | 271 | settlement | task | mae | 5 | -0.562 ± 0.159 |
| F_independent_prior_gate − B_independent_prior | 271 | settlement | task | rmse | 5 | -1.195 ± 0.319 |
| F_independent_prior_gate − B_independent_prior | 271 | settlement | task | r2 | 5 | 0.0220 ± 0.0080 |
| F_independent_prior_gate − B_independent_prior | 271 | slope | aggregate | balanced_accuracy | 5 | 0.0145 ± 0.0732 |
| F_independent_prior_gate − B_independent_prior | 271 | slope | aggregate | roc_auc | 5 | -0.0809 ± 0.0726 |
| F_independent_prior_gate − B_independent_prior | 271 | slope | task | balanced_accuracy | 5 | -0.0316 ± 0.0431 |
| F_independent_prior_gate − B_independent_prior | 271 | slope | task | roc_auc | 5 | -0.1130 ± 0.0462 |
| A_independent: task − aggregate | 42 | rock | task − aggregate | mae | 5 | -4.218 ± 3.737 |
| A_independent: task − aggregate | 42 | rock | task − aggregate | rmse | 5 | -2.107 ± 3.357 |
| A_independent: task − aggregate | 42 | rock | task − aggregate | r2 | 5 | 0.0468 ± 0.0767 |
| A_independent: task − aggregate | 42 | settlement | task − aggregate | mae | 5 | 0.040 ± 0.125 |
| A_independent: task − aggregate | 42 | settlement | task − aggregate | rmse | 5 | 0.066 ± 0.128 |
| A_independent: task − aggregate | 42 | settlement | task − aggregate | r2 | 5 | -0.0021 ± 0.0040 |
| A_independent: task − aggregate | 42 | slope | task − aggregate | balanced_accuracy | 5 | -0.0105 ± 0.0184 |
| A_independent: task − aggregate | 42 | slope | task − aggregate | roc_auc | 5 | -0.0006 ± 0.0007 |
| B_independent_prior: task − aggregate | 42 | rock | task − aggregate | mae | 5 | 0.965 ± 4.675 |
| B_independent_prior: task − aggregate | 42 | rock | task − aggregate | rmse | 5 | 3.062 ± 4.576 |
| B_independent_prior: task − aggregate | 42 | rock | task − aggregate | r2 | 5 | -0.0658 ± 0.0960 |
| B_independent_prior: task − aggregate | 42 | settlement | task − aggregate | mae | 5 | 0.066 ± 0.089 |
| B_independent_prior: task − aggregate | 42 | settlement | task − aggregate | rmse | 5 | 0.096 ± 0.139 |
| B_independent_prior: task − aggregate | 42 | settlement | task − aggregate | r2 | 5 | -0.0022 ± 0.0033 |
| B_independent_prior: task − aggregate | 42 | slope | task − aggregate | balanced_accuracy | 5 | 0.0000 ± 0.0000 |
| B_independent_prior: task − aggregate | 42 | slope | task − aggregate | roc_auc | 5 | -0.0003 ± 0.0007 |
| C_shared: task − aggregate | 42 | rock | task − aggregate | mae | 5 | -3.295 ± 2.494 |
| C_shared: task − aggregate | 42 | rock | task − aggregate | rmse | 5 | 0.908 ± 2.366 |
| C_shared: task − aggregate | 42 | rock | task − aggregate | r2 | 5 | -0.0183 ± 0.0507 |
| C_shared: task − aggregate | 42 | settlement | task − aggregate | mae | 5 | 0.084 ± 0.297 |
| C_shared: task − aggregate | 42 | settlement | task − aggregate | rmse | 5 | 0.167 ± 0.374 |
| C_shared: task − aggregate | 42 | settlement | task − aggregate | r2 | 5 | -0.0056 ± 0.0127 |
| C_shared: task − aggregate | 42 | slope | task − aggregate | balanced_accuracy | 5 | 0.0000 ± 0.0000 |
| C_shared: task − aggregate | 42 | slope | task − aggregate | roc_auc | 5 | 0.0004 ± 0.0022 |
| D_shared_prior: task − aggregate | 42 | rock | task − aggregate | mae | 5 | -3.501 ± 3.654 |
| D_shared_prior: task − aggregate | 42 | rock | task − aggregate | rmse | 5 | -2.559 ± 3.796 |
| D_shared_prior: task − aggregate | 42 | rock | task − aggregate | r2 | 5 | 0.0528 ± 0.0817 |
| D_shared_prior: task − aggregate | 42 | settlement | task − aggregate | mae | 5 | -0.032 ± 0.302 |
| D_shared_prior: task − aggregate | 42 | settlement | task − aggregate | rmse | 5 | -0.098 ± 0.535 |
| D_shared_prior: task − aggregate | 42 | settlement | task − aggregate | r2 | 5 | 0.0046 ± 0.0186 |
| D_shared_prior: task − aggregate | 42 | slope | task − aggregate | balanced_accuracy | 5 | 0.0086 ± 0.0173 |
| D_shared_prior: task − aggregate | 42 | slope | task − aggregate | roc_auc | 5 | -0.0012 ± 0.0049 |
| D_shared_weight_0.1: task − aggregate | 42 | rock | task − aggregate | mae | 5 | -3.644 ± 4.706 |
| D_shared_weight_0.1: task − aggregate | 42 | rock | task − aggregate | rmse | 5 | 0.432 ± 4.487 |
| D_shared_weight_0.1: task − aggregate | 42 | rock | task − aggregate | r2 | 5 | -0.0069 ± 0.0940 |
| D_shared_weight_0.1: task − aggregate | 42 | settlement | task − aggregate | mae | 5 | 0.146 ± 0.310 |
| D_shared_weight_0.1: task − aggregate | 42 | settlement | task − aggregate | rmse | 5 | 0.192 ± 0.339 |
| D_shared_weight_0.1: task − aggregate | 42 | settlement | task − aggregate | r2 | 5 | -0.0058 ± 0.0101 |
| D_shared_weight_0.1: task − aggregate | 42 | slope | task − aggregate | balanced_accuracy | 5 | 0.0026 ± 0.0059 |
| D_shared_weight_0.1: task − aggregate | 42 | slope | task − aggregate | roc_auc | 5 | -0.0011 ± 0.0025 |
| D_shared_weight_0.5: task − aggregate | 42 | rock | task − aggregate | mae | 5 | -1.246 ± 3.640 |
| D_shared_weight_0.5: task − aggregate | 42 | rock | task − aggregate | rmse | 5 | -1.519 ± 4.154 |
| D_shared_weight_0.5: task − aggregate | 42 | rock | task − aggregate | r2 | 5 | 0.0336 ± 0.0914 |
| D_shared_weight_0.5: task − aggregate | 42 | settlement | task − aggregate | mae | 5 | -0.180 ± 0.456 |
| D_shared_weight_0.5: task − aggregate | 42 | settlement | task − aggregate | rmse | 5 | -0.371 ± 0.896 |
| D_shared_weight_0.5: task − aggregate | 42 | settlement | task − aggregate | r2 | 5 | 0.0126 ± 0.0290 |
| D_shared_weight_0.5: task − aggregate | 42 | slope | task − aggregate | balanced_accuracy | 5 | 0.0112 ± 0.0250 |
| D_shared_weight_0.5: task − aggregate | 42 | slope | task − aggregate | roc_auc | 5 | -0.0003 ± 0.0025 |
| D_shared_weight_0.75: task − aggregate | 42 | rock | task − aggregate | mae | 5 | -0.321 ± 4.665 |
| D_shared_weight_0.75: task − aggregate | 42 | rock | task − aggregate | rmse | 5 | -2.903 ± 8.221 |
| D_shared_weight_0.75: task − aggregate | 42 | rock | task − aggregate | r2 | 5 | 0.0707 ± 0.1921 |
| D_shared_weight_0.75: task − aggregate | 42 | settlement | task − aggregate | mae | 5 | -0.021 ± 0.082 |
| D_shared_weight_0.75: task − aggregate | 42 | settlement | task − aggregate | rmse | 5 | -0.038 ± 0.084 |
| D_shared_weight_0.75: task − aggregate | 42 | settlement | task − aggregate | r2 | 5 | 0.0006 ± 0.0014 |
| D_shared_weight_0.75: task − aggregate | 42 | slope | task − aggregate | balanced_accuracy | 5 | 0.0000 ± 0.0000 |
| D_shared_weight_0.75: task − aggregate | 42 | slope | task − aggregate | roc_auc | 5 | -0.0012 ± 0.0028 |
| E_shared_prior_gate: task − aggregate | 42 | rock | task − aggregate | mae | 5 | -0.847 ± 4.075 |
| E_shared_prior_gate: task − aggregate | 42 | rock | task − aggregate | rmse | 5 | -0.277 ± 5.491 |
| E_shared_prior_gate: task − aggregate | 42 | rock | task − aggregate | r2 | 5 | 0.0014 ± 0.1232 |
| E_shared_prior_gate: task − aggregate | 42 | settlement | task − aggregate | mae | 5 | -0.191 ± 0.188 |
| E_shared_prior_gate: task − aggregate | 42 | settlement | task − aggregate | rmse | 5 | -0.246 ± 0.225 |
| E_shared_prior_gate: task − aggregate | 42 | settlement | task − aggregate | r2 | 5 | 0.0045 ± 0.0045 |
| E_shared_prior_gate: task − aggregate | 42 | slope | task − aggregate | balanced_accuracy | 5 | 0.0079 ± 0.0118 |
| E_shared_prior_gate: task − aggregate | 42 | slope | task − aggregate | roc_auc | 5 | 0.0018 ± 0.0040 |
| F_independent_prior_gate: task − aggregate | 42 | rock | task − aggregate | mae | 5 | 0.206 ± 3.437 |
| F_independent_prior_gate: task − aggregate | 42 | rock | task − aggregate | rmse | 5 | 0.303 ± 5.321 |
| F_independent_prior_gate: task − aggregate | 42 | rock | task − aggregate | r2 | 5 | -0.0059 ± 0.1152 |
| F_independent_prior_gate: task − aggregate | 42 | settlement | task − aggregate | mae | 5 | -0.247 ± 0.380 |
| F_independent_prior_gate: task − aggregate | 42 | settlement | task − aggregate | rmse | 5 | -0.298 ± 0.450 |
| F_independent_prior_gate: task − aggregate | 42 | settlement | task − aggregate | r2 | 5 | 0.0067 ± 0.0112 |
| F_independent_prior_gate: task − aggregate | 42 | slope | task − aggregate | balanced_accuracy | 5 | -0.0211 ± 0.0658 |
| F_independent_prior_gate: task − aggregate | 42 | slope | task − aggregate | roc_auc | 5 | -0.0058 ± 0.0648 |
| Pair_rock_settlement_C_shared: task − aggregate | 42 | rock | task − aggregate | mae | 5 | -5.663 ± 4.831 |
| Pair_rock_settlement_C_shared: task − aggregate | 42 | rock | task − aggregate | rmse | 5 | -1.660 ± 6.559 |
| Pair_rock_settlement_C_shared: task − aggregate | 42 | rock | task − aggregate | r2 | 5 | 0.0404 ± 0.1428 |
| Pair_rock_settlement_C_shared: task − aggregate | 42 | settlement | task − aggregate | mae | 5 | -0.145 ± 0.314 |
| Pair_rock_settlement_C_shared: task − aggregate | 42 | settlement | task − aggregate | rmse | 5 | -0.088 ± 0.339 |
| Pair_rock_settlement_C_shared: task − aggregate | 42 | settlement | task − aggregate | r2 | 5 | 0.0046 ± 0.0148 |
| Pair_rock_settlement_E_shared_prior_gate: task − aggregate | 42 | rock | task − aggregate | mae | 5 | -4.240 ± 6.262 |
| Pair_rock_settlement_E_shared_prior_gate: task − aggregate | 42 | rock | task − aggregate | rmse | 5 | -5.911 ± 9.016 |
| Pair_rock_settlement_E_shared_prior_gate: task − aggregate | 42 | rock | task − aggregate | r2 | 5 | 0.1303 ± 0.2020 |
| Pair_rock_settlement_E_shared_prior_gate: task − aggregate | 42 | settlement | task − aggregate | mae | 5 | -0.111 ± 0.127 |
| Pair_rock_settlement_E_shared_prior_gate: task − aggregate | 42 | settlement | task − aggregate | rmse | 5 | -0.181 ± 0.150 |
| Pair_rock_settlement_E_shared_prior_gate: task − aggregate | 42 | settlement | task − aggregate | r2 | 5 | 0.0031 ± 0.0027 |
| Pair_slope_rock_C_shared: task − aggregate | 42 | rock | task − aggregate | mae | 5 | -4.546 ± 3.163 |
| Pair_slope_rock_C_shared: task − aggregate | 42 | rock | task − aggregate | rmse | 5 | -0.270 ± 2.490 |
| Pair_slope_rock_C_shared: task − aggregate | 42 | rock | task − aggregate | r2 | 5 | 0.0060 ± 0.0501 |
| Pair_slope_rock_C_shared: task − aggregate | 42 | slope | task − aggregate | balanced_accuracy | 5 | 0.0053 ± 0.0076 |
| Pair_slope_rock_C_shared: task − aggregate | 42 | slope | task − aggregate | roc_auc | 5 | -0.0002 ± 0.0021 |
| Pair_slope_rock_E_shared_prior_gate: task − aggregate | 42 | rock | task − aggregate | mae | 5 | -1.365 ± 5.601 |
| Pair_slope_rock_E_shared_prior_gate: task − aggregate | 42 | rock | task − aggregate | rmse | 5 | -1.837 ± 7.211 |
| Pair_slope_rock_E_shared_prior_gate: task − aggregate | 42 | rock | task − aggregate | r2 | 5 | 0.0356 ± 0.1608 |
| Pair_slope_rock_E_shared_prior_gate: task − aggregate | 42 | slope | task − aggregate | balanced_accuracy | 5 | -0.0053 ± 0.0118 |
| Pair_slope_rock_E_shared_prior_gate: task − aggregate | 42 | slope | task − aggregate | roc_auc | 5 | -0.0015 ± 0.0025 |
| Pair_slope_settlement_C_shared: task − aggregate | 42 | settlement | task − aggregate | mae | 5 | 0.110 ± 0.216 |
| Pair_slope_settlement_C_shared: task − aggregate | 42 | settlement | task − aggregate | rmse | 5 | 0.100 ± 0.177 |
| Pair_slope_settlement_C_shared: task − aggregate | 42 | settlement | task − aggregate | r2 | 5 | -0.0025 ± 0.0051 |
| Pair_slope_settlement_C_shared: task − aggregate | 42 | slope | task − aggregate | balanced_accuracy | 5 | -0.0020 ± 0.0044 |
| Pair_slope_settlement_C_shared: task − aggregate | 42 | slope | task − aggregate | roc_auc | 5 | 0.0009 ± 0.0021 |
| Pair_slope_settlement_E_shared_prior_gate: task − aggregate | 42 | settlement | task − aggregate | mae | 5 | -0.258 ± 0.352 |
| Pair_slope_settlement_E_shared_prior_gate: task − aggregate | 42 | settlement | task − aggregate | rmse | 5 | -0.332 ± 0.424 |
| Pair_slope_settlement_E_shared_prior_gate: task − aggregate | 42 | settlement | task − aggregate | r2 | 5 | 0.0070 ± 0.0097 |
| Pair_slope_settlement_E_shared_prior_gate: task − aggregate | 42 | slope | task − aggregate | balanced_accuracy | 5 | 0.0026 ± 0.0059 |
| Pair_slope_settlement_E_shared_prior_gate: task − aggregate | 42 | slope | task − aggregate | roc_auc | 5 | -0.0002 ± 0.0008 |
| Tuned_C_shared: task − aggregate | 42 | rock | task − aggregate | mae | 5 | 0.112 ± 5.157 |
| Tuned_C_shared: task − aggregate | 42 | rock | task − aggregate | rmse | 5 | 5.573 ± 7.237 |
| Tuned_C_shared: task − aggregate | 42 | rock | task − aggregate | r2 | 5 | -0.1102 ± 0.1449 |
| Tuned_C_shared: task − aggregate | 42 | settlement | task − aggregate | mae | 5 | 0.028 ± 0.088 |
| Tuned_C_shared: task − aggregate | 42 | settlement | task − aggregate | rmse | 5 | -0.003 ± 0.112 |
| Tuned_C_shared: task − aggregate | 42 | settlement | task − aggregate | r2 | 5 | 0.0001 ± 0.0031 |
| Tuned_C_shared: task − aggregate | 42 | slope | task − aggregate | balanced_accuracy | 5 | 0.0092 ± 0.0102 |
| Tuned_C_shared: task − aggregate | 42 | slope | task − aggregate | roc_auc | 5 | 0.0084 ± 0.0220 |
| Tuned_E_shared_prior_gate: task − aggregate | 42 | rock | task − aggregate | mae | 5 | 1.819 ± 3.302 |
| Tuned_E_shared_prior_gate: task − aggregate | 42 | rock | task − aggregate | rmse | 5 | 2.033 ± 4.131 |
| Tuned_E_shared_prior_gate: task − aggregate | 42 | rock | task − aggregate | r2 | 5 | -0.0429 ± 0.0879 |
| Tuned_E_shared_prior_gate: task − aggregate | 42 | settlement | task − aggregate | mae | 5 | 0.000 ± 0.001 |
| Tuned_E_shared_prior_gate: task − aggregate | 42 | settlement | task − aggregate | rmse | 5 | 0.002 ± 0.002 |
| Tuned_E_shared_prior_gate: task − aggregate | 42 | settlement | task − aggregate | r2 | 5 | -0.0000 ± 0.0000 |
| Tuned_E_shared_prior_gate: task − aggregate | 42 | slope | task − aggregate | balanced_accuracy | 5 | -0.0079 ± 0.0118 |
| Tuned_E_shared_prior_gate: task − aggregate | 42 | slope | task − aggregate | roc_auc | 5 | -0.0030 ± 0.0049 |
| Tuned_MLP_rock: task − aggregate | 42 | rock | task − aggregate | mae | 5 | 0.000 ± 0.000 |
| Tuned_MLP_rock: task − aggregate | 42 | rock | task − aggregate | rmse | 5 | 0.000 ± 0.000 |
| Tuned_MLP_rock: task − aggregate | 42 | rock | task − aggregate | r2 | 5 | 0.0000 ± 0.0000 |
| Tuned_MLP_settlement: task − aggregate | 42 | settlement | task − aggregate | mae | 5 | 0.000 ± 0.000 |
| Tuned_MLP_settlement: task − aggregate | 42 | settlement | task − aggregate | rmse | 5 | 0.000 ± 0.000 |
| Tuned_MLP_settlement: task − aggregate | 42 | settlement | task − aggregate | r2 | 5 | 0.0000 ± 0.0000 |
| Tuned_MLP_slope: task − aggregate | 42 | slope | task − aggregate | balanced_accuracy | 5 | 0.0000 ± 0.0000 |
| Tuned_MLP_slope: task − aggregate | 42 | slope | task − aggregate | roc_auc | 5 | 0.0000 ± 0.0000 |
| A_independent: task − aggregate | 137 | rock | task − aggregate | mae | 5 | 0.114 ± 0.607 |
| A_independent: task − aggregate | 137 | rock | task − aggregate | rmse | 5 | -0.326 ± 0.430 |
| A_independent: task − aggregate | 137 | rock | task − aggregate | r2 | 5 | 0.0177 ± 0.0221 |
| A_independent: task − aggregate | 137 | settlement | task − aggregate | mae | 5 | -0.619 ± 0.320 |
| A_independent: task − aggregate | 137 | settlement | task − aggregate | rmse | 5 | -0.958 ± 0.407 |
| A_independent: task − aggregate | 137 | settlement | task − aggregate | r2 | 5 | 0.0583 ± 0.0316 |
| A_independent: task − aggregate | 137 | slope | task − aggregate | balanced_accuracy | 5 | -0.0053 ± 0.0253 |
| A_independent: task − aggregate | 137 | slope | task − aggregate | roc_auc | 5 | -0.0036 ± 0.0038 |
| B_independent_prior: task − aggregate | 137 | rock | task − aggregate | mae | 5 | -2.839 ± 2.714 |
| B_independent_prior: task − aggregate | 137 | rock | task − aggregate | rmse | 5 | -4.581 ± 2.981 |
| B_independent_prior: task − aggregate | 137 | rock | task − aggregate | r2 | 5 | 0.2347 ± 0.1541 |
| B_independent_prior: task − aggregate | 137 | settlement | task − aggregate | mae | 5 | -0.278 ± 0.323 |
| B_independent_prior: task − aggregate | 137 | settlement | task − aggregate | rmse | 5 | -0.518 ± 0.532 |
| B_independent_prior: task − aggregate | 137 | settlement | task − aggregate | r2 | 5 | 0.0246 ± 0.0270 |
| B_independent_prior: task − aggregate | 137 | slope | task − aggregate | balanced_accuracy | 5 | -0.0079 ± 0.0235 |
| B_independent_prior: task − aggregate | 137 | slope | task − aggregate | roc_auc | 5 | -0.0045 ± 0.0069 |
| C_shared: task − aggregate | 137 | rock | task − aggregate | mae | 5 | -0.530 ± 2.718 |
| C_shared: task − aggregate | 137 | rock | task − aggregate | rmse | 5 | -2.264 ± 3.013 |
| C_shared: task − aggregate | 137 | rock | task − aggregate | r2 | 5 | 0.1177 ± 0.1559 |
| C_shared: task − aggregate | 137 | settlement | task − aggregate | mae | 5 | -0.364 ± 0.315 |
| C_shared: task − aggregate | 137 | settlement | task − aggregate | rmse | 5 | -0.502 ± 0.700 |
| C_shared: task − aggregate | 137 | settlement | task − aggregate | r2 | 5 | 0.0304 ± 0.0470 |
| C_shared: task − aggregate | 137 | slope | task − aggregate | balanced_accuracy | 5 | -0.0158 ± 0.0353 |
| C_shared: task − aggregate | 137 | slope | task − aggregate | roc_auc | 5 | -0.0061 ± 0.0183 |
| D_shared_prior: task − aggregate | 137 | rock | task − aggregate | mae | 5 | -0.683 ± 2.554 |
| D_shared_prior: task − aggregate | 137 | rock | task − aggregate | rmse | 5 | -1.213 ± 2.935 |
| D_shared_prior: task − aggregate | 137 | rock | task − aggregate | r2 | 5 | 0.0653 ± 0.1498 |
| D_shared_prior: task − aggregate | 137 | settlement | task − aggregate | mae | 5 | -0.241 ± 0.330 |
| D_shared_prior: task − aggregate | 137 | settlement | task − aggregate | rmse | 5 | -0.425 ± 0.641 |
| D_shared_prior: task − aggregate | 137 | settlement | task − aggregate | r2 | 5 | 0.0201 ± 0.0324 |
| D_shared_prior: task − aggregate | 137 | slope | task − aggregate | balanced_accuracy | 5 | -0.0026 ± 0.0063 |
| D_shared_prior: task − aggregate | 137 | slope | task − aggregate | roc_auc | 5 | 0.0049 ± 0.0089 |
| E_shared_prior_gate: task − aggregate | 137 | rock | task − aggregate | mae | 5 | -2.224 ± 3.952 |
| E_shared_prior_gate: task − aggregate | 137 | rock | task − aggregate | rmse | 5 | -3.575 ± 5.362 |
| E_shared_prior_gate: task − aggregate | 137 | rock | task − aggregate | r2 | 5 | 0.1877 ± 0.2726 |
| E_shared_prior_gate: task − aggregate | 137 | settlement | task − aggregate | mae | 5 | -0.059 ± 0.084 |
| E_shared_prior_gate: task − aggregate | 137 | settlement | task − aggregate | rmse | 5 | -0.096 ± 0.128 |
| E_shared_prior_gate: task − aggregate | 137 | settlement | task − aggregate | r2 | 5 | 0.0020 ± 0.0028 |
| E_shared_prior_gate: task − aggregate | 137 | slope | task − aggregate | balanced_accuracy | 5 | -0.0217 ± 0.0333 |
| E_shared_prior_gate: task − aggregate | 137 | slope | task − aggregate | roc_auc | 5 | -0.0236 ± 0.0318 |
| F_independent_prior_gate: task − aggregate | 137 | rock | task − aggregate | mae | 5 | 0.321 ± 1.903 |
| F_independent_prior_gate: task − aggregate | 137 | rock | task − aggregate | rmse | 5 | 0.080 ± 3.100 |
| F_independent_prior_gate: task − aggregate | 137 | rock | task − aggregate | r2 | 5 | 0.0029 ± 0.1480 |
| F_independent_prior_gate: task − aggregate | 137 | settlement | task − aggregate | mae | 5 | -0.178 ± 0.214 |
| F_independent_prior_gate: task − aggregate | 137 | settlement | task − aggregate | rmse | 5 | -0.263 ± 0.265 |
| F_independent_prior_gate: task − aggregate | 137 | settlement | task − aggregate | r2 | 5 | 0.0064 ± 0.0074 |
| F_independent_prior_gate: task − aggregate | 137 | slope | task − aggregate | balanced_accuracy | 5 | -0.0197 ± 0.0424 |
| F_independent_prior_gate: task − aggregate | 137 | slope | task − aggregate | roc_auc | 5 | -0.0045 ± 0.0534 |
| Tuned_C_shared: task − aggregate | 137 | rock | task − aggregate | mae | 5 | -0.973 ± 1.408 |
| Tuned_C_shared: task − aggregate | 137 | rock | task − aggregate | rmse | 5 | -1.766 ± 2.058 |
| Tuned_C_shared: task − aggregate | 137 | rock | task − aggregate | r2 | 5 | 0.0884 ± 0.1047 |
| Tuned_C_shared: task − aggregate | 137 | settlement | task − aggregate | mae | 5 | -0.043 ± 0.072 |
| Tuned_C_shared: task − aggregate | 137 | settlement | task − aggregate | rmse | 5 | -0.145 ± 0.073 |
| Tuned_C_shared: task − aggregate | 137 | settlement | task − aggregate | r2 | 5 | 0.0075 ± 0.0036 |
| Tuned_C_shared: task − aggregate | 137 | slope | task − aggregate | balanced_accuracy | 5 | -0.0013 ± 0.0082 |
| Tuned_C_shared: task − aggregate | 137 | slope | task − aggregate | roc_auc | 5 | 0.0027 ± 0.0135 |
| Tuned_E_shared_prior_gate: task − aggregate | 137 | rock | task − aggregate | mae | 5 | -1.416 ± 2.502 |
| Tuned_E_shared_prior_gate: task − aggregate | 137 | rock | task − aggregate | rmse | 5 | -1.252 ± 3.704 |
| Tuned_E_shared_prior_gate: task − aggregate | 137 | rock | task − aggregate | r2 | 5 | 0.0681 ± 0.1895 |
| Tuned_E_shared_prior_gate: task − aggregate | 137 | settlement | task − aggregate | mae | 5 | 0.001 ± 0.004 |
| Tuned_E_shared_prior_gate: task − aggregate | 137 | settlement | task − aggregate | rmse | 5 | -0.001 ± 0.004 |
| Tuned_E_shared_prior_gate: task − aggregate | 137 | settlement | task − aggregate | r2 | 5 | 0.0000 ± 0.0001 |
| Tuned_E_shared_prior_gate: task − aggregate | 137 | slope | task − aggregate | balanced_accuracy | 5 | -0.0013 ± 0.0117 |
| Tuned_E_shared_prior_gate: task − aggregate | 137 | slope | task − aggregate | roc_auc | 5 | -0.0001 ± 0.0062 |
| Tuned_MLP_rock: task − aggregate | 137 | rock | task − aggregate | mae | 5 | 0.000 ± 0.000 |
| Tuned_MLP_rock: task − aggregate | 137 | rock | task − aggregate | rmse | 5 | 0.000 ± 0.000 |
| Tuned_MLP_rock: task − aggregate | 137 | rock | task − aggregate | r2 | 5 | 0.0000 ± 0.0000 |
| Tuned_MLP_settlement: task − aggregate | 137 | settlement | task − aggregate | mae | 5 | 0.000 ± 0.000 |
| Tuned_MLP_settlement: task − aggregate | 137 | settlement | task − aggregate | rmse | 5 | 0.000 ± 0.000 |
| Tuned_MLP_settlement: task − aggregate | 137 | settlement | task − aggregate | r2 | 5 | 0.0000 ± 0.0000 |
| Tuned_MLP_slope: task − aggregate | 137 | slope | task − aggregate | balanced_accuracy | 5 | 0.0000 ± 0.0000 |
| Tuned_MLP_slope: task − aggregate | 137 | slope | task − aggregate | roc_auc | 5 | 0.0000 ± 0.0000 |
| A_independent: task − aggregate | 271 | rock | task − aggregate | mae | 5 | -3.144 ± 1.696 |
| A_independent: task − aggregate | 271 | rock | task − aggregate | rmse | 5 | -4.979 ± 2.205 |
| A_independent: task − aggregate | 271 | rock | task − aggregate | r2 | 5 | 0.2099 ± 0.1006 |
| A_independent: task − aggregate | 271 | settlement | task − aggregate | mae | 5 | -0.303 ± 0.130 |
| A_independent: task − aggregate | 271 | settlement | task − aggregate | rmse | 5 | -0.805 ± 0.340 |
| A_independent: task − aggregate | 271 | settlement | task − aggregate | r2 | 5 | 0.0337 ± 0.0163 |
| A_independent: task − aggregate | 271 | slope | task − aggregate | balanced_accuracy | 5 | 0.0026 ± 0.0059 |
| A_independent: task − aggregate | 271 | slope | task − aggregate | roc_auc | 5 | -0.0071 ± 0.0159 |
| B_independent_prior: task − aggregate | 271 | rock | task − aggregate | mae | 5 | -2.058 ± 2.037 |
| B_independent_prior: task − aggregate | 271 | rock | task − aggregate | rmse | 5 | -2.826 ± 3.003 |
| B_independent_prior: task − aggregate | 271 | rock | task − aggregate | r2 | 5 | 0.1181 ± 0.1260 |
| B_independent_prior: task − aggregate | 271 | settlement | task − aggregate | mae | 5 | -0.266 ± 0.109 |
| B_independent_prior: task − aggregate | 271 | settlement | task − aggregate | rmse | 5 | -0.758 ± 0.393 |
| B_independent_prior: task − aggregate | 271 | settlement | task − aggregate | r2 | 5 | 0.0244 ± 0.0149 |
| B_independent_prior: task − aggregate | 271 | slope | task − aggregate | balanced_accuracy | 5 | 0.0303 ± 0.0512 |
| B_independent_prior: task − aggregate | 271 | slope | task − aggregate | roc_auc | 5 | 0.0128 ± 0.0230 |
| C_shared: task − aggregate | 271 | rock | task − aggregate | mae | 5 | -4.036 ± 1.135 |
| C_shared: task − aggregate | 271 | rock | task − aggregate | rmse | 5 | -5.759 ± 2.050 |
| C_shared: task − aggregate | 271 | rock | task − aggregate | r2 | 5 | 0.2487 ± 0.0871 |
| C_shared: task − aggregate | 271 | settlement | task − aggregate | mae | 5 | -0.291 ± 0.265 |
| C_shared: task − aggregate | 271 | settlement | task − aggregate | rmse | 5 | -0.875 ± 0.843 |
| C_shared: task − aggregate | 271 | settlement | task − aggregate | r2 | 5 | 0.0377 ± 0.0399 |
| C_shared: task − aggregate | 271 | slope | task − aggregate | balanced_accuracy | 5 | 0.0000 ± 0.0000 |
| C_shared: task − aggregate | 271 | slope | task − aggregate | roc_auc | 5 | 0.0000 ± 0.0000 |
| D_shared_prior: task − aggregate | 271 | rock | task − aggregate | mae | 5 | -2.657 ± 1.664 |
| D_shared_prior: task − aggregate | 271 | rock | task − aggregate | rmse | 5 | -3.883 ± 2.706 |
| D_shared_prior: task − aggregate | 271 | rock | task − aggregate | r2 | 5 | 0.1619 ± 0.1120 |
| D_shared_prior: task − aggregate | 271 | settlement | task − aggregate | mae | 5 | -0.296 ± 0.236 |
| D_shared_prior: task − aggregate | 271 | settlement | task − aggregate | rmse | 5 | -0.781 ± 0.627 |
| D_shared_prior: task − aggregate | 271 | settlement | task − aggregate | r2 | 5 | 0.0244 ± 0.0216 |
| D_shared_prior: task − aggregate | 271 | slope | task − aggregate | balanced_accuracy | 5 | -0.0296 ± 0.0499 |
| D_shared_prior: task − aggregate | 271 | slope | task − aggregate | roc_auc | 5 | -0.0159 ± 0.0230 |
| E_shared_prior_gate: task − aggregate | 271 | rock | task − aggregate | mae | 5 | -2.617 ± 3.047 |
| E_shared_prior_gate: task − aggregate | 271 | rock | task − aggregate | rmse | 5 | -4.719 ± 4.694 |
| E_shared_prior_gate: task − aggregate | 271 | rock | task − aggregate | r2 | 5 | 0.1840 ± 0.1846 |
| E_shared_prior_gate: task − aggregate | 271 | settlement | task − aggregate | mae | 5 | -0.058 ± 0.051 |
| E_shared_prior_gate: task − aggregate | 271 | settlement | task − aggregate | rmse | 5 | -0.095 ± 0.091 |
| E_shared_prior_gate: task − aggregate | 271 | settlement | task − aggregate | r2 | 5 | 0.0011 ± 0.0011 |
| E_shared_prior_gate: task − aggregate | 271 | slope | task − aggregate | balanced_accuracy | 5 | -0.0217 ± 0.0308 |
| E_shared_prior_gate: task − aggregate | 271 | slope | task − aggregate | roc_auc | 5 | 0.0078 ± 0.0297 |
| F_independent_prior_gate: task − aggregate | 271 | rock | task − aggregate | mae | 5 | -3.564 ± 1.394 |
| F_independent_prior_gate: task − aggregate | 271 | rock | task − aggregate | rmse | 5 | -5.776 ± 2.900 |
| F_independent_prior_gate: task − aggregate | 271 | rock | task − aggregate | r2 | 5 | 0.2228 ± 0.1153 |
| F_independent_prior_gate: task − aggregate | 271 | settlement | task − aggregate | mae | 5 | -0.078 ± 0.133 |
| F_independent_prior_gate: task − aggregate | 271 | settlement | task − aggregate | rmse | 5 | -0.165 ± 0.222 |
| F_independent_prior_gate: task − aggregate | 271 | settlement | task − aggregate | r2 | 5 | 0.0022 ± 0.0032 |
| F_independent_prior_gate: task − aggregate | 271 | slope | task − aggregate | balanced_accuracy | 5 | -0.0158 ± 0.0353 |
| F_independent_prior_gate: task − aggregate | 271 | slope | task − aggregate | roc_auc | 5 | -0.0193 ± 0.0400 |
| Tuned_C_shared: task − aggregate | 271 | rock | task − aggregate | mae | 5 | -2.219 ± 2.544 |
| Tuned_C_shared: task − aggregate | 271 | rock | task − aggregate | rmse | 5 | -3.627 ± 3.885 |
| Tuned_C_shared: task − aggregate | 271 | rock | task − aggregate | r2 | 5 | 0.1473 ± 0.1655 |
| Tuned_C_shared: task − aggregate | 271 | settlement | task − aggregate | mae | 5 | -0.064 ± 0.051 |
| Tuned_C_shared: task − aggregate | 271 | settlement | task − aggregate | rmse | 5 | -0.241 ± 0.151 |
| Tuned_C_shared: task − aggregate | 271 | settlement | task − aggregate | r2 | 5 | 0.0102 ± 0.0067 |
| Tuned_C_shared: task − aggregate | 271 | slope | task − aggregate | balanced_accuracy | 5 | 0.0072 ± 0.0160 |
| Tuned_C_shared: task − aggregate | 271 | slope | task − aggregate | roc_auc | 5 | 0.0029 ± 0.0074 |
| Tuned_E_shared_prior_gate: task − aggregate | 271 | rock | task − aggregate | mae | 5 | -5.180 ± 1.456 |
| Tuned_E_shared_prior_gate: task − aggregate | 271 | rock | task − aggregate | rmse | 5 | -7.626 ± 1.634 |
| Tuned_E_shared_prior_gate: task − aggregate | 271 | rock | task − aggregate | r2 | 5 | 0.3075 ± 0.0696 |
| Tuned_E_shared_prior_gate: task − aggregate | 271 | settlement | task − aggregate | mae | 5 | -0.000 ± 0.016 |
| Tuned_E_shared_prior_gate: task − aggregate | 271 | settlement | task − aggregate | rmse | 5 | -0.002 ± 0.021 |
| Tuned_E_shared_prior_gate: task − aggregate | 271 | settlement | task − aggregate | r2 | 5 | 0.0000 ± 0.0002 |
| Tuned_E_shared_prior_gate: task − aggregate | 271 | slope | task − aggregate | balanced_accuracy | 5 | 0.0007 ± 0.0216 |
| Tuned_E_shared_prior_gate: task − aggregate | 271 | slope | task − aggregate | roc_auc | 5 | -0.0086 ± 0.0105 |
| Tuned_MLP_rock: task − aggregate | 271 | rock | task − aggregate | mae | 5 | 0.000 ± 0.000 |
| Tuned_MLP_rock: task − aggregate | 271 | rock | task − aggregate | rmse | 5 | 0.000 ± 0.000 |
| Tuned_MLP_rock: task − aggregate | 271 | rock | task − aggregate | r2 | 5 | 0.0000 ± 0.0000 |
| Tuned_MLP_settlement: task − aggregate | 271 | settlement | task − aggregate | mae | 5 | 0.000 ± 0.000 |
| Tuned_MLP_settlement: task − aggregate | 271 | settlement | task − aggregate | rmse | 5 | 0.000 ± 0.000 |
| Tuned_MLP_settlement: task − aggregate | 271 | settlement | task − aggregate | r2 | 5 | 0.0000 ± 0.0000 |
| Tuned_MLP_slope: task − aggregate | 271 | slope | task − aggregate | balanced_accuracy | 5 | 0.0000 ± 0.0000 |
| Tuned_MLP_slope: task − aggregate | 271 | slope | task − aggregate | roc_auc | 5 | 0.0000 ± 0.0000 |
