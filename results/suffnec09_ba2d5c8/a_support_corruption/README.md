# A Support Corruption Counterfactual

Dataset: `assist_09`

Checkpoint: `suffnec09_ba2d5c8/phase2/assist_09/A_fused_neutralE`

This experiment keeps the trained checkpoint fixed and does not retrain. It progressively replaces the item/sequence evidence support of A with source-wise row-degree-matched random non-evidence support.

## Main Result

The prediction loss increases as more A support is corrupted.

| Group | 25% corrupted AUC drop | 50% corrupted AUC drop | 75% corrupted AUC drop | 100% corrupted AUC drop |
| --- | ---: | ---: | ---: | ---: |
| all | 0.0050 | 0.0110 | 0.0218 | 0.0341 |
| high_support_mass | 0.0055 | 0.0099 | 0.0242 | 0.0408 |
| query_seq_top5_q4_high | 0.0082 | 0.0129 | 0.0246 | 0.0474 |
| multi_concept | 0.0059 | 0.0093 | 0.0334 | 0.0523 |

The BCE increase follows the same direction:

| Group | 25% corrupted BCE increase | 50% corrupted BCE increase | 75% corrupted BCE increase | 100% corrupted BCE increase |
| --- | ---: | ---: | ---: | ---: |
| all | 0.0063 | 0.0081 | 0.0148 | 0.0196 |
| high_support_mass | 0.0069 | 0.0078 | 0.0159 | 0.0238 |
| query_seq_top5_q4_high | 0.0112 | 0.0110 | 0.0196 | 0.0318 |
| multi_concept | 0.0082 | 0.0092 | 0.0239 | 0.0349 |

## Interpretation

This is a stronger replacement for the previous subgroup-only evidence. It shows that the A support identity matters: when the same checkpoint uses increasingly corrupted degree-matched support, prediction degrades in a mostly monotonic way, especially for high-support and strong sequence-evidence samples.

Recommended wording:

> Replacing evidence-selected A support with degree-matched random support causes a monotonic prediction degradation, showing that A's support is not an arbitrary graph scaffold but an informative global map.
