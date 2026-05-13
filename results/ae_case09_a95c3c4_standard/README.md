# assist_09 AE Case-Study Evidence

Final small-experiment evidence package for the assist_09 A/E case study.

Generated from commit `359b346` with:

- full checkpoint: `checkpoints/abce_diag/ae_global_tutor_10ed8b6_20260512_1510/assist_09/seed42/best_full`
- no_A checkpoint: `checkpoints/abce_diag/ae_global_tutor_10ed8b6_20260512_1510/assist_09/seed42/best_no_A`
- no_E checkpoint: `checkpoints/abce_diag/ae_global_tutor_10ed8b6_20260512_1510/assist_09/seed42/best_no_E`

## Metrics

| variant | AUC | ACC | RMSE |
|---|---:|---:|---:|
| full | 0.7783116166 | 0.7406733216 | 0.4177640253 |
| no_A | 0.7671327299 | 0.7320414954 | 0.4222092188 |
| no_E | 0.7633669337 | 0.7308866706 | 0.4246636167 |
| E_shuffle | 0.5750942006 | 0.6333920532 | 0.5070431971 |
| E_mean | 0.6064734562 | 0.6636328049 | 0.4826892621 |

## Core Figures

- `figures/case_probability_comparison.png`: selected A/E rescue cases.
- `figures/A_roadmap_heatmap_A01_row39811.png`, `A02_row10377.png`, `A03_row7398.png`: A global roadmap cases.
- `figures/A_top_edges_by_case.png`, `A_evidence_source_mix.png`: A evidence-source diagnostics.
- `figures/E_counterfactual_probability_comparison.png`: actual E vs no_E, shuffled E, and mean E.
- `figures/E_tutor_heatmap_E01_row16884.png`, `E02_row49180.png`, `E03_row29063.png`: strict E cases where full is correct and no_E/shuffle/mean fail.
- `figures/E_same_global_map_posterior_by_student.png`: fixed C97 global A row with student-specific E posterior maps.
- `figures/E_same_global_map_delta_by_student.png`: fixed C97 global A row with student-specific posterior deltas.
- `figures/E_posterior_delta_by_student.png`, `E_student_state_by_support.png`: E posterior/state diagnostics.
- `figures/E_recent_history_strip.png`: recent related-concept history for selected E cases.

## Tables

- `case_predictions.csv`: full test predictions for full/no_A/no_E/E_shuffle/E_mean.
- `metrics_check.csv`, `case_study_summary.json`: metric and run summary.
- `a_candidate_pool.csv`, `a_selected_cases.csv`, `a_case_edges.csv`, `a_case_matrix.csv`: A evidence tables.
- `e_candidate_pool.csv`, `e_selected_cases.csv`, `e_case_edges.csv`, `e_case_history.csv`: strict E mechanism tables.
- `e_student_contrast_cases.csv`, `e_student_contrast_edges.csv`, `e_student_contrast_selected_cases.csv`: same-global-map E contrast tables.

The matching command log is stored in `logs/ae_case09_a95c3c4_standard/run.log`.
