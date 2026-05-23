# Main Problem Experiment Review Packet

This packet uses existing checkpoints only. No retraining or model-structure changes were performed.

## Experiment 1: History-to-Query Concept Route Retrieval

| dataset | exp | random_hit10 | seq_hit10 | fused_hit10 | success |
| --- | --- | --- | --- | --- | --- |
| assist_09 | exp1_history_to_query_retrieval | 0.067308 | 0.178138 | 0.157389 | True |
| assist_17 | exp1_history_to_query_retrieval | 0.120141 | 0.077749 | 0.093299 | False |
| junyi | exp1_history_to_query_retrieval | 0.019098 | 0.137885 | 0.137885 | True |

## Experiment 2: Coverage-conditioned Prediction

| dataset | subgroup | variant | n_eval | auc | bce | auc_gap_full_minus_variant | bce_gap_variant_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| junyi | direct_unseen_bridgeable | full | 60478 | 0.829116 | 0.489031 | 0.000000 | 0.000000 |
| junyi | direct_unseen_bridgeable | no_CRG | 60478 | 0.827906 | 0.490299 | 0.001210 | 0.001268 |
| junyi | direct_unseen_bridgeable | no_LCRF | 60478 | 0.828627 | 0.485876 | 0.000489 | -0.003156 |
| junyi | direct_unseen_bridgeable | self_only | 60478 | 0.827581 | 0.500575 | 0.001535 | 0.011544 |
| junyi | direct_unseen_bridgeable | degree_random_support | 60478 | 0.828546 | 0.488556 | 0.000569 | -0.000475 |
| junyi | weak_direct_evidence | full | 60494 | 0.829113 | 0.489137 | 0.000000 | 0.000000 |
| junyi | weak_direct_evidence | no_CRG | 60494 | 0.827844 | 0.490455 | 0.001269 | 0.001318 |
| junyi | weak_direct_evidence | no_LCRF | 60494 | 0.828603 | 0.485984 | 0.000510 | -0.003153 |
| junyi | weak_direct_evidence | self_only | 60494 | 0.827562 | 0.500702 | 0.001551 | 0.011566 |
| junyi | weak_direct_evidence | degree_random_support | 60494 | 0.828538 | 0.488664 | 0.000575 | -0.000472 |
| junyi | high_route_mass | full | 18148 | 0.798027 | 0.385576 | 0.000000 | 0.000000 |
| junyi | high_route_mass | no_CRG | 18148 | 0.795512 | 0.387356 | 0.002515 | 0.001780 |
| junyi | high_route_mass | no_LCRF | 18148 | 0.796300 | 0.384335 | 0.001728 | -0.001241 |
| junyi | high_route_mass | self_only | 18148 | 0.798892 | 0.394169 | -0.000865 | 0.008593 |
| junyi | high_route_mass | degree_random_support | 18148 | 0.797643 | 0.384961 | 0.000385 | -0.000615 |
| assist_17 | direct_seen | full | 71859 | 0.787490 | 0.547528 | 0.000000 | 0.000000 |
| assist_17 | direct_seen | no_CRG | 71859 | 0.767384 | 0.565828 | 0.020106 | 0.018299 |
| assist_17 | direct_seen | no_LCRF | 71859 | 0.786282 | 0.557109 | 0.001208 | 0.009580 |
| assist_17 | direct_seen | self_only | 71859 | 0.781293 | 0.563022 | 0.006197 | 0.015494 |
| assist_17 | direct_seen | degree_random_support | 71859 | 0.787490 | 0.547556 | -0.000000 | 0.000028 |
| assist_17 | direct_unseen_bridgeable | full | 5402 | 0.735481 | 0.613744 | 0.000000 | 0.000000 |
| assist_17 | direct_unseen_bridgeable | no_CRG | 5402 | 0.729378 | 0.632852 | 0.006103 | 0.019108 |
| assist_17 | direct_unseen_bridgeable | no_LCRF | 5402 | 0.740934 | 0.662803 | -0.005453 | 0.049059 |
| assist_17 | direct_unseen_bridgeable | self_only | 5402 | 0.722800 | 0.725041 | 0.012681 | 0.111297 |
| assist_17 | direct_unseen_bridgeable | degree_random_support | 5402 | 0.734647 | 0.614571 | 0.000834 | 0.000827 |
| assist_17 | weak_direct_evidence | full | 11552 | 0.762727 | 0.582973 | 0.000000 | 0.000000 |
| assist_17 | weak_direct_evidence | no_CRG | 11552 | 0.747624 | 0.601465 | 0.015103 | 0.018492 |
| assist_17 | weak_direct_evidence | no_LCRF | 11552 | 0.763003 | 0.609924 | -0.000277 | 0.026951 |
| assist_17 | weak_direct_evidence | self_only | 11552 | 0.749602 | 0.652477 | 0.013125 | 0.069504 |
| assist_17 | weak_direct_evidence | degree_random_support | 11552 | 0.762315 | 0.583544 | 0.000411 | 0.000571 |
| assist_17 | high_route_mass | full | 23187 | 0.787593 | 0.550806 | 0.000000 | 0.000000 |
| assist_17 | high_route_mass | no_CRG | 23187 | 0.772535 | 0.564375 | 0.015058 | 0.013569 |
| assist_17 | high_route_mass | no_LCRF | 23187 | 0.785608 | 0.562817 | 0.001985 | 0.012011 |
| assist_17 | high_route_mass | self_only | 23187 | 0.782758 | 0.559071 | 0.004835 | 0.008265 |
| assist_17 | high_route_mass | degree_random_support | 23187 | 0.787556 | 0.550868 | 0.000037 | 0.000062 |
| assist_09 | direct_seen | full | 49506 | 0.775769 | 0.527673 | 0.000000 | 0.000000 |
| assist_09 | direct_seen | no_CRG | 49506 | 0.765220 | 0.535971 | 0.010550 | 0.008298 |
| assist_09 | direct_seen | no_LCRF | 49506 | 0.761770 | 0.542749 | 0.013999 | 0.015075 |
| assist_09 | direct_seen | self_only | 49506 | 0.766488 | 0.545822 | 0.009282 | 0.018149 |
| assist_09 | direct_seen | degree_random_support | 49506 | 0.775334 | 0.527876 | 0.000435 | 0.000203 |
| assist_09 | direct_unseen_bridgeable | full | 1584 | 0.826429 | 0.381085 | 0.000000 | 0.000000 |
| assist_09 | direct_unseen_bridgeable | no_CRG | 1584 | 0.831376 | 0.402949 | -0.004947 | 0.021863 |
| assist_09 | direct_unseen_bridgeable | no_LCRF | 1584 | 0.826311 | 0.419262 | 0.000118 | 0.038176 |
| assist_09 | direct_unseen_bridgeable | self_only | 1584 | 0.804941 | 0.437692 | 0.021488 | 0.056607 |
| assist_09 | direct_unseen_bridgeable | degree_random_support | 1584 | 0.825226 | 0.382447 | 0.001203 | 0.001362 |
| assist_09 | weak_direct_evidence | full | 3882 | 0.809960 | 0.416933 | 0.000000 | 0.000000 |
| assist_09 | weak_direct_evidence | no_CRG | 3882 | 0.805864 | 0.432300 | 0.004096 | 0.015366 |
| assist_09 | weak_direct_evidence | no_LCRF | 3882 | 0.791183 | 0.451914 | 0.018777 | 0.034980 |
| assist_09 | weak_direct_evidence | self_only | 3882 | 0.794655 | 0.448223 | 0.015305 | 0.031289 |
| assist_09 | weak_direct_evidence | degree_random_support | 3882 | 0.808654 | 0.418095 | 0.001306 | 0.001162 |
| assist_09 | high_route_mass | full | 15327 | 0.756461 | 0.502038 | 0.000000 | 0.000000 |
| assist_09 | high_route_mass | no_CRG | 15327 | 0.748333 | 0.507969 | 0.008129 | 0.005931 |
| assist_09 | high_route_mass | no_LCRF | 15327 | 0.744548 | 0.512769 | 0.011914 | 0.010731 |
| assist_09 | high_route_mass | self_only | 15327 | 0.744705 | 0.513916 | 0.011756 | 0.011877 |
| assist_09 | high_route_mass | degree_random_support | 15327 | 0.755856 | 0.502435 | 0.000606 | 0.000397 |

## Experiment 3: Direct Concept Evidence Removal Counterfactual

| dataset | subgroup | variant | n_eval | auc | bce | rmse | acc | mask_type | paired_n | target_bce_increase | random_bce_increase |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| assist_17 | direct_seen_sample | full | 500.000000 | 0.817940 | 0.519265 | 0.417380 | 0.728000 | original |  |  |  |
| assist_17 | direct_seen_sample | full | 500.000000 | 0.817859 | 0.519376 | 0.417464 | 0.728000 | random_history_mask |  |  |  |
| assist_17 | direct_seen_sample | full | 500.000000 | 0.810082 | 0.535074 | 0.424041 | 0.734000 | target_concept_mask |  |  |  |
| assist_17 | direct_seen_sample | no_CRG | 500.000000 | 0.788629 | 0.551922 | 0.432633 | 0.724000 | original |  |  |  |
| assist_17 | direct_seen_sample | no_CRG | 500.000000 | 0.788629 | 0.551922 | 0.432633 | 0.724000 | random_history_mask |  |  |  |
| assist_17 | direct_seen_sample | no_CRG | 500.000000 | 0.788629 | 0.551922 | 0.432633 | 0.724000 | target_concept_mask |  |  |  |
| assist_17 | direct_seen_sample | no_LCRF | 500.000000 | 0.810973 | 0.536351 | 0.425286 | 0.722000 | original |  |  |  |
| assist_17 | direct_seen_sample | no_LCRF | 500.000000 | 0.810973 | 0.536351 | 0.425286 | 0.722000 | random_history_mask |  |  |  |
| assist_17 | direct_seen_sample | no_LCRF | 500.000000 | 0.810973 | 0.536351 | 0.425286 | 0.722000 | target_concept_mask |  |  |  |
| assist_17 | direct_seen_sample | self_only | 500.000000 | 0.809417 | 0.543004 | 0.429566 | 0.732000 | original |  |  |  |
| assist_17 | direct_seen_sample | self_only | 500.000000 | 0.809806 | 0.542640 | 0.429454 | 0.732000 | random_history_mask |  |  |  |
| assist_17 | direct_seen_sample | self_only | 500.000000 | 0.801559 | 0.539982 | 0.427027 | 0.708000 | target_concept_mask |  |  |  |
| assist_17 | paired_bce_increase | full |  |  |  |  |  |  | 500.000000 | 0.015809 | 0.000111 |
| assist_17 | paired_bce_increase | no_CRG |  |  |  |  |  |  | 500.000000 | 0.000000 | 0.000000 |
| assist_17 | paired_bce_increase | no_LCRF |  |  |  |  |  |  | 500.000000 | 0.000000 | 0.000000 |
| assist_17 | paired_bce_increase | self_only |  |  |  |  |  |  | 500.000000 | -0.003022 | -0.000364 |
| assist_09 | direct_seen_sample | full | 500.000000 | 0.742029 | 0.538739 | 0.424925 | 0.732000 | original |  |  |  |
| assist_09 | direct_seen_sample | full | 500.000000 | 0.741431 | 0.539767 | 0.425558 | 0.732000 | random_history_mask |  |  |  |
| assist_09 | direct_seen_sample | full | 500.000000 | 0.705489 | 0.619865 | 0.452931 | 0.714000 | target_concept_mask |  |  |  |
| assist_09 | direct_seen_sample | no_CRG | 500.000000 | 0.741861 | 0.534186 | 0.423969 | 0.734000 | original |  |  |  |
| assist_09 | direct_seen_sample | no_CRG | 500.000000 | 0.741861 | 0.534186 | 0.423969 | 0.734000 | random_history_mask |  |  |  |
| assist_09 | direct_seen_sample | no_CRG | 500.000000 | 0.741861 | 0.534186 | 0.423969 | 0.734000 | target_concept_mask |  |  |  |
| assist_09 | direct_seen_sample | no_LCRF | 500.000000 | 0.723516 | 0.560436 | 0.432283 | 0.728000 | original |  |  |  |
| assist_09 | direct_seen_sample | no_LCRF | 500.000000 | 0.723516 | 0.560436 | 0.432283 | 0.728000 | random_history_mask |  |  |  |
| assist_09 | direct_seen_sample | no_LCRF | 500.000000 | 0.723516 | 0.560436 | 0.432283 | 0.728000 | target_concept_mask |  |  |  |
| assist_09 | direct_seen_sample | self_only | 500.000000 | 0.726657 | 0.574471 | 0.440245 | 0.710000 | original |  |  |  |
| assist_09 | direct_seen_sample | self_only | 500.000000 | 0.726657 | 0.574471 | 0.440245 | 0.710000 | random_history_mask |  |  |  |
| assist_09 | direct_seen_sample | self_only | 500.000000 | 0.682057 | 0.598796 | 0.447074 | 0.710000 | target_concept_mask |  |  |  |
| assist_09 | paired_bce_increase | full |  |  |  |  |  |  | 500.000000 | 0.081126 | 0.001028 |
| assist_09 | paired_bce_increase | no_CRG |  |  |  |  |  |  | 500.000000 | 0.000000 | 0.000000 |
| assist_09 | paired_bce_increase | no_LCRF |  |  |  |  |  |  | 500.000000 | 0.000000 | 0.000000 |
| assist_09 | paired_bce_increase | self_only |  |  |  |  |  |  | 500.000000 | 0.024324 | 0.000000 |

## Recommendation

Use Experiment 1 as the safest main-problem evidence if prediction-level subgroup gaps are weak. Promote Experiment 2 to the main text only when direct-unseen-bridgeable or high-route-mass gaps meet the pre-defined thresholds. Use Experiment 3 as a mechanism counterfactual only as a buffer-level state-removal analysis; do not claim exact raw-history recomputation.

## Missing or downgraded controls

- exp3 is a sampled buffer-level student-concept state mask diagnostic, not exact raw-history recomputation
- global_only control was not evaluated unless an existing exact checkpoint was present
- plot-only rerun: reused existing CSV outputs without re-running inference
