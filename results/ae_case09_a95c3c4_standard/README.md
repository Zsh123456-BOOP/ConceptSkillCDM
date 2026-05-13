# assist_09 AE Case-Study Evidence

This directory contains the final small-experiment evidence package for the assist_09 A/E case study.

## Metrics

| variant | AUC | ACC | RMSE |
|---|---:|---:|---:|
| full | 0.7783116166 | 0.7406733216 | 0.4177640253 |
| no_A | 0.7671327299 | 0.7320414954 | 0.4222092188 |
| no_E | 0.7633669337 | 0.7308866706 | 0.4246636167 |

## Core Figures

- `figures/case_probability_comparison.png`: selected A/E rescue cases with full vs no_A/no_E probabilities.
- `figures/A_roadmap_heatmap_A01_row39811.png`: A global-roadmap single-concept rescue case.
- `figures/A_roadmap_heatmap_A02_row10377.png`: A global-roadmap sequence-evidence rescue case.
- `figures/A_roadmap_heatmap_A03_row7398.png`: A global-roadmap multi-concept rescue case.
- `figures/A_evidence_source_mix.png`: source mix of selected A edges.
- `figures/E_tutor_heatmap_E01_row45257.png`: E posterior reweighting case.
- `figures/E_posterior_delta_by_student.png`: E posterior shift over selected students.
- `figures/E_student_state_by_support.png`: student-specific mastery/recent states on the same support.
- `figures/E_recent_history_strip.png`: recent related-concept history for selected E cases.

## Tables

- `case_predictions.csv`: full test split predictions for full/no_A/no_E and deterministic rescue tags.
- `a_candidate_pool.csv`, `e_candidate_pool.csv`: ranked candidate pools.
- `a_selected_cases.csv`, `e_selected_cases.csv`, `selected_cases.csv`: final selected cases.
- `a_case_edges.csv`, `a_case_matrix.csv`: A local-roadmap edges and matrix values.
- `e_case_edges.csv`, `e_case_history.csv`: E posterior edges and recent history rows.
- `metrics_check.csv`, `case_study_summary.json`: metrics and run summary.

The matching command log is stored in `logs/ae_case09_a95c3c4_standard/run.log`.
