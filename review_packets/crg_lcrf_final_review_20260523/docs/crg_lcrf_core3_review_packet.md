# CRG/LCRF Core3 Review Packet

Generated from `results/crg_lcrf_core3_final_20260520/`.

## Scope

- Datasets: `assist_09`, `junyi`, `assist_17`.
- No retraining was performed by this aggregation script.
- Model structure was not changed.
- Existing checkpoints and existing inference/counterfactual CSVs were reused.

## Outputs

- Main table: `results/crg_lcrf_core3_final_20260520/main_table/table_main_ablation_core3.csv`
- Figure CSVs and plots: `results/crg_lcrf_core3_final_20260520/paper_figures/`
- CRG support audit: `results/crg_lcrf_core3_final_20260520/crg_support_audit/`
- LCRF same-query cases: `results/crg_lcrf_core3_final_20260520/lcrf_same_query/`
- CRG local route cases: `results/crg_lcrf_core3_final_20260520/crg_local_route_cases/`
- LCRF timeline: `results/crg_lcrf_core3_final_20260520/lcrf_student_timeline/`
- LCRF state-source audit: `results/crg_lcrf_core3_final_20260520/lcrf_state_source_audit/`

## Result Summary

| dataset | full AUC | no_CRG drop | no_LCRF drop | wording |
|---|---:|---:|---:|---|
| assist_09 | 0.7783 | 0.0112 | 0.0149 | balanced benchmark; both modules usable |
| junyi | 0.8291 | 0.0013 | 0.0005 | CRG reachability/data phenomenon, LCRF weak |
| assist_17 | 0.7847 | 0.0200 | 0.0018 | main CRG necessity evidence; LCRF state counterfactual strong |

## Claim Review

1. CRG sufficiency: supported by retrieval lift. This is the cleanest CRG evidence because it does not depend on prediction-head behavior.
2. CRG necessity: strongest on assist_17; assist_09 should be written as support-dependence rather than evidence-specific superiority; Junyi is weak at prediction-level corruption.
3. LCRF necessity: supported by no_filter/mean/shuffle counterfactual drops on assist_09 and assist_17. Junyi remains weak and should not be used as LCRF main evidence.
4. LCRF sufficiency: supported by same-query posterior variation on assist_09/assist_17. Use assist_17 as the primary visual case if its posterior variability remains highest.

## Risks

- Main `metrics_check.csv` files did not store global BCE for no_CRG/no_LCRF; AUC/ACC/RMSE are reliable, BCE is only available for support corruption and selected cases.
- `shuffle_id_keep_state` and `zero_id_keep_state` were not available from existing inference hooks. Do not claim student-ID shortcut is fully ruled out.
- CRG evidence-vs-degree-random gap is not universal. The paper should report weak results honestly for Junyi and cautious wording for assist_09.
