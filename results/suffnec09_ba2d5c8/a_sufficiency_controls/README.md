# A Sufficiency/Necessity Evidence Notes

Dataset: `assist_09`

Run id: `suffnec09_ba2d5c8`

## What Is Strong

1. Edge deletion necessity is strong.

   Using the fixed `A_fused_neutralE` checkpoint, deleting top evidence edges hurts much more than deleting the same number of random support edges.

   Representative AUC drops:

   | Group | Delete 10% top edges | Delete 10% random edges | Delete 50% top edges | Delete 50% random edges |
   | --- | ---: | ---: | ---: | ---: |
   | all | 0.0216 | 0.0024 | 0.0320 | 0.0131 |
   | high_support_mass | 0.0291 | 0.0034 | 0.0398 | 0.0212 |
   | multi_concept | 0.0410 | 0.0047 | 0.0727 | 0.0224 |

   This supports the claim that the selected high-evidence A edges are necessary and are not interchangeable with arbitrary support edges.

2. Held-out transition retrieval is strong.

   The A prior predicts held-out student concept transitions much better than uniform, random, or self-only graphs.

   | Variant | Hit@10 mean | Bootstrap 95% CI |
   | --- | ---: | --- |
   | A_seq_only | 0.3672 | [0.3653, 0.3691] |
   | A_fused_prior | 0.3649 | [0.3629, 0.3668] |
   | A_item_only | 0.1636 | [0.1623, 0.1649] |
   | A_degree_random | 0.1358 | [0.1345, 0.1371] |
   | A_support_uniform | 0.1345 | [0.1333, 0.1360] |
   | A_uniform_offdiag | 0.1322 | [0.1309, 0.1334] |
   | A_self_only | 0.1123 | [0.1112, 0.1136] |

   This supports the claim that A captures a train-only learning-path map, not just generic graph smoothing.

## What Is Only Auxiliary

The A-relevant subgroup result is useful but not a clean monotonic proof.

Positive examples:

| Group | AUC gain over no_A | BCE gain |
| --- | ---: | ---: |
| high_support_mass | 0.0069 | 0.0030 |
| query_seq_top5 q4_high | 0.0108 | 0.0033 |

However, several bins have negative BCE gain or non-monotonic behavior. Use this figure only as scene-level evidence: A helps more in high support / strong sequence-evidence conditions. Do not claim every train-only relevance score produces a strictly monotonic AUC gain.

## Recommended Paper Wording

Safe claim:

> The global map is useful because its selected high-evidence edges are necessary for prediction and its support recovers held-out concept transitions far above degree-matched random controls. Its prediction gains concentrate in high-support and high-sequence-evidence samples, although not every relevance bin is strictly monotonic.

Avoid:

> A_fused weights are always better than support-uniform/random in full-test AUC.

> A-relevant subgroup gains are strictly monotonic for all relevance definitions.
