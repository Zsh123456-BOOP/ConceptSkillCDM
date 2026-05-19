# CRG/LCRF Small Experiments

Run: `crg_lcrf_small_core_20260519`

## Summary

- Paper names: `CRG` = Concept Reachability Graph, `LCRF` = Learner-Conditioned Reachability Filter.
- Full raw results remain on the server at `results/crg_lcrf_small_core_20260519` because transition-pair and candidate-pool CSVs are large.

## Dataset Phenomenon

| dataset | train rows | concepts | multi-concept rate | item edges | seq density | median history | direct seen | neighbor reachable | bridge-only |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| assist_09 | 190623 | 123 | 0.172 | 130 | 0.643 | 41 | 0.969 | 0.989 | 0.031 |
| junyi | 235446 | 697 | 0.000 | 0 | 0.252 | 25 | 0.000 | 1.000 | 1.000 |
| assist_17 | 282340 | 101 | 0.217 | 624 | 0.763 | 148 | 0.972 | 1.000 | 0.028 |
| nips34 | 1104293 | 86 | 1.000 | 464 | 0.982 | 191 | 1.000 | 1.000 | 0.000 |

## CRG Retrieval

| dataset | best variant | Hit@10 | NDCG@10 | MRR | random Hit@10 | self Hit@10 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| assist_09 | CRG_seq_only | 0.367 | 0.196 | 0.170 | 0.136 | 0.112 |
| junyi | CRG_fused_prior / CRG_seq_only | 0.165 | 0.078 | 0.072 | 0.017 | 0.002 |
| assist_17 | CRG_seq_only | 0.411 | 0.229 | 0.199 | 0.162 | 0.132 |

## CRG Support Corruption

| dataset | all AUC drop at 100% | high-support AUC drop at 100% | high-seq AUC drop at 100% | note |
| --- | ---: | ---: | ---: | --- |
| assist_09 | 0.0138 | 0.0121 | 0.0155 | strong |
| junyi | 0.0029 | 0.0022 | 0.0018 | weak |
| assist_17 | 0.0028 | 0.0034 | 0.0009 | weak/moderate in AUC; BCE sensitivity is clearer |

## LCRF Counterfactual

| dataset | full AUC | no_LCRF AUC | shuffle AUC | mean AUC | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| assist_09 | 0.7783 | 0.7634 | 0.5751 | 0.6065 | strong |
| junyi | 0.8291 | 0.8286 | 0.8188 | 0.8259 | weak/global, usable cautiously |
| assist_17 | 0.7847 | 0.7829 | 0.5966 | 0.6441 | strong counterfactual despite weak no_LCRF |
| nips34 | 0.7903 | 0.7733 | 0.5083 | 0.5050 | strong |

## Interpretation

- CRG sufficiency is strong on `assist_09`, `junyi`, and `assist_17`: train-only sequence/fused reachability retrieves held-out concept transitions far above random/self controls.
- CRG necessity is strongest on `assist_09`: 100% support corruption drops AUC by about 1.4 points overall and 1.55 points on high sequence-support samples. `assist_17` is weaker in AUC but has clearer BCE sensitivity. `junyi` corruption is weak and should be used mainly as a reachability-phenomenon/retrieval dataset.
- LCRF necessity is strong on `assist_09`, `assist_17`, and especially `nips34`: shuffled/mean learner state collapses far below full, so the personalized filter is not a fixed global patch. `junyi` is weaker globally, so use it cautiously for LCRF.

## Key Figures

- CRG retrieval: `crg_retrieval/<dataset>/figures/crg_transition_retrieval.png`
- CRG corruption: `crg_support_corruption/<dataset>/figures/crg_support_corruption_counterfactual.png`
- LCRF counterfactual/cases: `lcrf_case_studies/<dataset>/figures/`
