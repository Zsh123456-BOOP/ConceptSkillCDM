# CRG/LCRF Mechanism Review Packet

This packet is generated for reviewer-side auditing. It records what was run, what was skipped, and which claims are currently supported. No model architecture was changed by this packet generation.

## Dataset Story Cards

| dataset | single_concept_rate | item_edge_density | seq_edge_density | direct_unseen_rate | bridge_only_rate | recommended_role | run_priority |
| --- | --- | --- | --- | --- | --- | --- | --- |
| assist_12 | 1 | 0 | 0.4313 | 0.3399 | 0.3399 | core_crg | 1 |
| assist_12_clean15_item50 | 1 | 0 | 0.3195 | 0.3992 | 0.3951 | core_crg | 1 |
| assist_15 | 1 | 0 | 0.6283 | 0.5178 | 0.5178 | core_crg | 1 |
| junyi | 1 | 0 | 0.2519 | 1 | 0.9997 | core_crg | 2 |
| junyi_long | 1 | 0 | 0.2519 | 1 | 0.9997 | core_crg | 2 |
| assist_09 | 0.8283 | 0.008663 | 0.6433 | 0.031 | 0.031 | balanced_main | 3 |
| assist_17 | 0.7825 | 0.06178 | 0.7635 | 0.02781 | 0.02781 | core_lcrf | 3 |
| cdbd_a0910 | 0.8281 | 0.009104 | 0.6728 | 0.0306 | 0.0306 | balanced_main | 3 |
| ednet_kt1 | 0.5698 | 0.03892 | 0.9763 | 0.007984 | 0.007984 | appendix_contrast | 40 |
| frcsub | 0.15 | 0.75 | 1 | 0 | 0 | appendix_contrast | 40 |
| math2 | 0 | 0.3 | 1 | 0.01074 | 0.01074 | appendix_contrast | 40 |
| nips34_l3 | 0.9852 | 0.003133 | 0.9721 | 0.01576 | 0.01576 | appendix_contrast | 40 |
| junyi_sample | 1 | 0 | 0.973 | 0.07985 | 0.07985 | appendix_contrast | 60 |
| cdbd_lsat | 1 | 0 | 0 | 1 | 0 | skip | 99 |

## CRG Sufficiency: Held-out Transition Retrieval

| dataset | best_variant | best_hit@10 | random_hit@10 | degree_random_hit@10 | self_hit@10 | best_minus_random_hit@10 | best_over_random | best_over_self | retrieval_success | claim_status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| assist_09 | seq_only | 0.3673 | 0.1321 | 0.1358 | 0.1124 | 0.2352 | 2.781 | 3.268 | True | pass |
| junyi | fused_CRG | 0.1648 | 0.02191 | 0.01708 | 0.002048 | 0.1429 | 7.523 | 80.47 | True | pass |
| assist_17 | seq_only | 0.4113 | 0.1544 | 0.1618 | 0.132 | 0.2569 | 2.664 | 3.115 | True | pass |
| assist_12 | fused_CRG | 0.8761 | 0.02277 | 0.04109 | 0.02183 | 0.8534 | 38.47 | 40.14 | True | pass |
| assist_15 | fused_CRG | 0.5327 | 0.1124 | 0.1101 | 0.0947 | 0.4203 | 4.74 | 5.625 | True | pass |
| assist_12_clean15_item50 | fused_CRG | 0.4908 | 0.02131 | 0.03424 | 0.02059 | 0.4695 | 23.03 | 23.84 | True | pass |

Success rule: best CRG Hit@10 is at least 2x random or has absolute Hit@10 lift >= 0.05.

## Checkpoint Availability

| dataset | variant | checkpoint_dir | exists | action |
| --- | --- | --- | --- | --- |
| assist_09 | full | checkpoints/abce_diag/recover_ed553d3_assist09_gpu2_20260518_140623/assist_09/seed42/best_full | True | reuse_inference |
| assist_09 | no_A | checkpoints/abce_diag/recover_ed553d3_assist09_gpu2_20260518_140623/assist_09/seed42/best_no_A | True | reuse_inference |
| assist_09 | no_E | checkpoints/abce_diag/recover_ed553d3_assist09_gpu2_20260518_140623/assist_09/seed42/best_no_E | True | reuse_inference |
| junyi | full | checkpoints/abce_diag/recover_ed553d3_junyi17_gpu3_20260519_004530/junyi/seed42/best_full | True | reuse_inference |
| junyi | no_A | checkpoints/abce_diag/recover_ed553d3_junyi17_gpu3_20260519_004530/junyi/seed42/best_no_A | True | reuse_inference |
| junyi | no_E | checkpoints/abce_diag/recover_ed553d3_junyi17_gpu3_20260519_004530/junyi/seed42/best_no_E | True | reuse_inference |
| assist_17 | full | checkpoints/abce_diag/recover_ed553d3_junyi17_gpu3_20260519_004530/assist_17/seed42/best_full | True | reuse_inference |
| assist_17 | no_A | checkpoints/abce_diag/recover_ed553d3_junyi17_gpu3_20260519_004530/assist_17/seed42/best_no_A | True | reuse_inference |
| assist_17 | no_E | checkpoints/abce_diag/recover_ed553d3_junyi17_gpu3_20260519_004530/assist_17/seed42/best_no_E | True | reuse_inference |
| assist_12 | full | nan | False | missing_checkpoint_skip_or_train_if_new_candidate |
| assist_12 | no_A | nan | False | missing_checkpoint_skip_or_train_if_new_candidate |
| assist_12 | no_E | nan | False | missing_checkpoint_skip_or_train_if_new_candidate |
| assist_15 | full | nan | False | missing_checkpoint_skip_or_train_if_new_candidate |
| assist_15 | no_A | nan | False | missing_checkpoint_skip_or_train_if_new_candidate |
| assist_15 | no_E | nan | False | missing_checkpoint_skip_or_train_if_new_candidate |
| assist_12_clean15_item50 | full | nan | False | missing_checkpoint_skip_or_train_if_new_candidate |
| assist_12_clean15_item50 | no_A | nan | False | missing_checkpoint_skip_or_train_if_new_candidate |
| assist_12_clean15_item50 | no_E | nan | False | missing_checkpoint_skip_or_train_if_new_candidate |

## Current Claim Guidance

- CRG sufficiency can be claimed only for datasets whose retrieval status is `pass`.
- CRG necessity requires support-corruption/subgroup evidence; do not infer it from retrieval alone.
- LCRF necessity requires actual/mean/shuffle/no-filter counterfactuals from a trained full checkpoint.
- LCRF sufficiency requires same-query posterior variability with identical CRG support.
- Sequence transition must be worded as empirical learning route, not prerequisite.

## Next Required Steps

1. Train or locate full/no_CRG/no_LCRF checkpoints for new candidates that pass retrieval.
2. Run inference-only support corruption only for datasets with full checkpoints.
3. Run LCRF counterfactual and same-query posterior only for datasets with full checkpoints.
4. Update the paper outline after actual prediction-level evidence is available.

## Main Ablation Initial Screen

### assist_12

| variant | auc | bce | acc | rmse | checkpoint |
|---|---:|---:|---:|---:|---|
| full | 0.7019780936804425 | nan | 0.690196604710461 | 0.450954217345494 | `checkpoints/mechanism/crg_lcrf_story_extension_20260520_ablation/phase2/assist_12/full/best_model.pth` |
| no_CRG | 0.7010050840801034 | nan | 0.7542776622156613 | 0.4140138420691048 | `checkpoints/mechanism/crg_lcrf_story_extension_20260520_ablation/phase2/assist_12/no_A/best_model.pth` |
| no_LCRF | 0.6941951171208838 | nan | 0.7543447627994364 | 0.4149534824099575 | `checkpoints/mechanism/crg_lcrf_story_extension_20260520_ablation/phase2/assist_12/no_E/best_model.pth` |

Interpretation rule: promote only if full is stable and either no_CRG/no_LCRF or later counterfactual evidence shows a clear mechanism signal.

### assist_15

| variant | auc | bce | acc | rmse | checkpoint |
|---|---:|---:|---:|---:|---|
| full | 0.656585451877729 | nan | 0.7098944888599283 | 0.438035562192635 | `checkpoints/mechanism/crg_lcrf_story_extension_20260520_ablation/phase2/assist_15/full/best_model.pth` |

Interpretation rule: promote only if full is stable and either no_CRG/no_LCRF or later counterfactual evidence shows a clear mechanism signal.

## Reviewer Decision After Extension Screen

### What was run

- Step 1 data profile confirmation completed for the server profile CSV.
- Step 2 CRG held-out transition retrieval completed for `assist_09`, `junyi`, `assist_17`, `assist_12`, `assist_15`, and optional `assist_12_clean15_item50`.
- Step 3 one-seed initial training screen completed for `assist_12`: full / no_CRG / no_LCRF.
- Step 3 one-seed initial training screen completed only for `assist_15/full`; no_CRG and no_LCRF were stopped because full AUC was too low to justify further mechanism training.
- No model architecture was changed. The only code change in this round was result/audit tooling.

### Main conclusions

| dataset | retrieval claim | prediction-level screen | decision |
|---|---|---|---|
| assist_12 | Very strong CRG sufficiency: fused CRG Hit@10 = 0.876 vs random = 0.0228. | full AUC = 0.7020; no_CRG AUC = 0.7010; no_LCRF AUC = 0.6942. CRG prediction necessity is weak; LCRF has only a small signal. | Use as CRG retrieval/data-phenomenon supplement only. Do not promote to main dataset. |
| assist_15 | Strong CRG sufficiency: fused CRG Hit@10 = 0.533 vs random = 0.112. | full AUC = 0.6566, best val AUC = 0.6595. Initial prediction screen failed. | Keep as negative/weak candidate. Do not run no_CRG/no_LCRF unless a future data-processing reason is found. |
| assist_12_clean15_item50 | Strong CRG retrieval: fused CRG Hit@10 = 0.491 vs random = 0.0213. | Not trained because assist_12 and assist_15 did not pass prediction-level screen. | Appendix candidate only, not current main claim. |

### Recommended paper wording

- Write that `assist_12` and `assist_15` confirm the data-side reachability phenomenon, especially single-concept 100%, item-edge 0, and high bridge-only rate.
- Do not write that the current CRG/LCRF model is prediction-level effective on `assist_12` or `assist_15`.
- Keep the main evidence chain on `assist_09`, `junyi`, and `assist_17`.
- Use `assist_12` and `assist_15` as honest supplementary evidence: CRG can retrieve empirical learning routes, but strong retrieval alone does not guarantee downstream CDM gains.

### Remaining risks

- CRG retrieval is strong on new single-concept datasets, but prediction-level no_CRG is not strong on `assist_12`.
- `assist_15` may have preprocessing, label-distribution, or split-specific issues because full AUC is too low; this should not be fixed by adding residuals.
- If the paper needs more than three core datasets, the next action should be data-processing validation, not model structure changes.
