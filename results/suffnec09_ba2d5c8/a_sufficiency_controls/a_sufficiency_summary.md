# A Sufficiency Controls

## Prediction Necessity
Use `A_fused_neutralE` vs `no_A_fair` from `../mechanism_results.csv` for the main prediction necessity result.

## A-Relevant Monotonicity
| group | n | support_mass_mean | A_fused_minus_no_A_auc | no_A_minus_A_fused_bce |
| --- | --- | --- | --- | --- |
| q2_midlow | 12622 | 0.560125 | 0.002161 | -0.010940 |
| q1_low | 12644 | 0.253371 | 0.005164 | -0.012233 |
| q4_high | 12630 | 0.947391 | 0.006915 | 0.002957 |
| q3_midhigh | 12633 | 0.838008 | 0.005778 | -0.001302 |
| zero | 561 | 0.000000 | 0.025202 | 0.023824 |

## Edge Deletion
| delete_mode | fraction | trial | group | auc_drop_vs_baseline | bce_increase_vs_baseline |
| --- | --- | --- | --- | --- | --- |
| top | 0.100000 | 0 | high_support_mass | 0.029086 | 0.026923 |
| random | 0.100000 | 0 | high_support_mass | 0.002232 | 0.000039 |
| random | 0.100000 | 1 | high_support_mass | 0.005034 | 0.006133 |
| random | 0.100000 | 2 | high_support_mass | 0.001635 | 0.000629 |
| random | 0.100000 | 3 | high_support_mass | 0.004890 | 0.005504 |
| random | 0.100000 | 4 | high_support_mass | 0.003365 | 0.002105 |

## Held-Out Transition Retrieval
| variant | hit@10 | mrr |
| --- | --- | --- |
| A_fused_prior | 0.364922 | 0.159096 |
| A_item_only | 0.163579 | 0.083493 |
| A_seq_only | 0.367313 | 0.169874 |
| A_support_uniform | 0.134532 | 0.060045 |
| A_degree_random | 0.135773 | 0.057559 |
| A_uniform_offdiag | 0.132096 | 0.058899 |
| A_self_only | 0.112380 | 0.046137 |
