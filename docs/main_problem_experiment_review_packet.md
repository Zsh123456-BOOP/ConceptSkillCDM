# Main Problem Experiment Review Packet

This packet uses existing checkpoints only. No retraining or model-structure changes were performed.

## Experiment 1: History-to-Query Concept Route Retrieval

| dataset | exp | random_hit10 | seq_hit10 | fused_hit10 | success |
| --- | --- | --- | --- | --- | --- |
| assist_09_chold | exp1_history_to_query_retrieval | 0.087254 | 0.510538 | 0.488152 | True |
| assist_17_chold | exp1_history_to_query_retrieval | 0.049506 | 0.757700 | 0.650072 | True |
| junyi_chold | exp1_history_to_query_retrieval | 0.014506 | 0.185689 | 0.185689 | True |

## Experiment 2: Coverage-conditioned Prediction

| dataset | subgroup | variant | n_eval | auc | bce | auc_gap_full_minus_variant | bce_gap_variant_minus_full |
| --- | --- | --- | --- | --- | --- | --- | --- |
| junyi_chold | direct_unseen_bridgeable | full | 48118 | 0.821305 | 0.498434 | 0.000000 | 0.000000 |
| junyi_chold | direct_unseen_bridgeable | no_CRG | 48118 | 0.813955 | 0.495166 | 0.007350 | -0.003268 |
| junyi_chold | direct_unseen_bridgeable | self_only | 48118 | 0.818840 | 0.503094 | 0.002465 | 0.004660 |
| junyi_chold | direct_unseen_bridgeable | degree_random_support | 48118 | 0.821069 | 0.497832 | 0.000236 | -0.000602 |
| junyi_chold | weak_direct_evidence | full | 48133 | 0.821400 | 0.498402 | 0.000000 | 0.000000 |
| junyi_chold | weak_direct_evidence | no_CRG | 48133 | 0.814029 | 0.495144 | 0.007371 | -0.003258 |
| junyi_chold | weak_direct_evidence | self_only | 48133 | 0.818927 | 0.503064 | 0.002473 | 0.004662 |
| junyi_chold | weak_direct_evidence | degree_random_support | 48133 | 0.821161 | 0.497801 | 0.000239 | -0.000602 |
| junyi_chold | high_route_mass | full | 14440 | 0.784972 | 0.388068 | 0.000000 | 0.000000 |
| junyi_chold | high_route_mass | no_CRG | 14440 | 0.776334 | 0.387243 | 0.008638 | -0.000825 |
| junyi_chold | high_route_mass | self_only | 14440 | 0.782847 | 0.394760 | 0.002125 | 0.006692 |
| junyi_chold | high_route_mass | degree_random_support | 14440 | 0.784728 | 0.387348 | 0.000243 | -0.000720 |
| assist_17_chold | direct_unseen_bridgeable | full | 72435 | 0.757754 | 0.591537 | 0.000000 | 0.000000 |
| assist_17_chold | direct_unseen_bridgeable | no_CRG | 72435 | 0.739758 | 0.595140 | 0.017997 | 0.003603 |
| assist_17_chold | direct_unseen_bridgeable | self_only | 72435 | 0.752840 | 0.595517 | 0.004915 | 0.003979 |
| assist_17_chold | direct_unseen_bridgeable | degree_random_support | 72435 | 0.757709 | 0.591565 | 0.000046 | 0.000028 |
| assist_17_chold | weak_direct_evidence | full | 72435 | 0.757754 | 0.591537 | 0.000000 | 0.000000 |
| assist_17_chold | weak_direct_evidence | no_CRG | 72435 | 0.739758 | 0.595140 | 0.017997 | 0.003603 |
| assist_17_chold | weak_direct_evidence | self_only | 72435 | 0.752840 | 0.595517 | 0.004915 | 0.003979 |
| assist_17_chold | weak_direct_evidence | degree_random_support | 72435 | 0.757709 | 0.591565 | 0.000046 | 0.000028 |
| assist_17_chold | high_route_mass | full | 21754 | 0.754502 | 0.591733 | 0.000000 | 0.000000 |
| assist_17_chold | high_route_mass | no_CRG | 21754 | 0.747676 | 0.593905 | 0.006825 | 0.002172 |
| assist_17_chold | high_route_mass | self_only | 21754 | 0.747647 | 0.602738 | 0.006855 | 0.011005 |
| assist_17_chold | high_route_mass | degree_random_support | 21754 | 0.754374 | 0.591904 | 0.000127 | 0.000171 |
| assist_09_chold | direct_seen | full | 4372 | 0.724071 | 0.689550 | 0.000000 | 0.000000 |
| assist_09_chold | direct_seen | no_CRG | 4372 | 0.717984 | 0.652612 | 0.006087 | -0.036938 |
| assist_09_chold | direct_seen | self_only | 4372 | 0.721498 | 0.653068 | 0.002572 | -0.036482 |
| assist_09_chold | direct_seen | degree_random_support | 4372 | 0.721174 | 0.701142 | 0.002897 | 0.011592 |
| assist_09_chold | direct_unseen_bridgeable | full | 41262 | 0.732771 | 0.609894 | 0.000000 | 0.000000 |
| assist_09_chold | direct_unseen_bridgeable | no_CRG | 41262 | 0.723915 | 0.591054 | 0.008856 | -0.018840 |
| assist_09_chold | direct_unseen_bridgeable | self_only | 41262 | 0.729845 | 0.583809 | 0.002925 | -0.026085 |
| assist_09_chold | direct_unseen_bridgeable | degree_random_support | 41262 | 0.732475 | 0.609991 | 0.000295 | 0.000097 |
| assist_09_chold | weak_direct_evidence | full | 45634 | 0.733645 | 0.617525 | 0.000000 | 0.000000 |
| assist_09_chold | weak_direct_evidence | no_CRG | 45634 | 0.724803 | 0.596952 | 0.008842 | -0.020574 |
| assist_09_chold | weak_direct_evidence | self_only | 45634 | 0.730668 | 0.590444 | 0.002977 | -0.027081 |
| assist_09_chold | weak_direct_evidence | degree_random_support | 45634 | 0.732732 | 0.618724 | 0.000913 | 0.001198 |
| assist_09_chold | high_route_mass | full | 13711 | 0.724889 | 0.569675 | 0.000000 | 0.000000 |
| assist_09_chold | high_route_mass | no_CRG | 13711 | 0.694023 | 0.573229 | 0.030866 | 0.003553 |
| assist_09_chold | high_route_mass | self_only | 13711 | 0.717557 | 0.551768 | 0.007332 | -0.017908 |
| assist_09_chold | high_route_mass | degree_random_support | 13711 | 0.722148 | 0.570841 | 0.002741 | 0.001166 |

## Recommendation

Use Experiment 1 as the safest main-problem evidence if prediction-level subgroup gaps are weak. Promote Experiment 2 to the main text only when direct-unseen-bridgeable or high-route-mass gaps meet the pre-defined thresholds. Use Experiment 3 as a mechanism counterfactual only as a buffer-level state-removal analysis; do not claim exact raw-history recomputation.

## Missing or downgraded controls

- assist_09_chold: global_only control skipped; no existing checkpoint or exact inference hook for a pure global-only prediction path
- assist_09_chold: missing no_LCRF checkpoint in main table
- assist_17_chold: global_only control skipped; no existing checkpoint or exact inference hook for a pure global-only prediction path
- assist_17_chold: missing no_LCRF checkpoint in main table
- junyi_chold: global_only control skipped; no existing checkpoint or exact inference hook for a pure global-only prediction path
- junyi_chold: missing no_LCRF checkpoint in main table
