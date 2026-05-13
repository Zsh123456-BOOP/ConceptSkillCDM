# suffnec09_ba2d5c8 Evidence Package

This package keeps the clean A-module evidence for `assist_09`.

## Kept Evidence

- `mechanism_results.csv`: completed prediction rows for `no_A_fair` and `A_fused_neutralE`.
- `a_necessity_preserved.csv`: preserved copy of the same necessary prediction comparison.
- `figures/auc_by_variant.png` and `figures/mechanism_drops.png`: compact prediction comparison figures.
- `a_support_evidence/a_transition_retrieval.csv`: held-out concept transition retrieval metrics.
- `a_support_evidence/figures/a_transition_retrieval.png`: retrieval figure.
- `a_support_evidence/a_heldout_transition_pairs.csv`: held-out transition pairs used to compute retrieval metrics.
- `a_support_evidence/a_support_evidence_summary.json`: compact metadata for the retrieval analysis.

## Interpretation Boundary

Use `A_fused_neutralE` vs `no_A_fair` for the prediction necessity claim:

- `A_fused_neutralE` test AUC: `0.778686`
- `no_A_fair` test AUC: `0.768674`
- A-side gain: about `+0.0100` AUC

Use the held-out transition retrieval table for the map-quality claim. It shows whether the train-only A map predicts future concept transitions better than random, uniform, and self-only controls.

Do not use the stopped `A_support_uniform_neutralE`, `A_degree_random_neutralE`, or `A_self_neutralE` partial training rows as final AUC evidence. Those runs were stopped because their validation curves made the global-AUC control route weak and low value for the current proof.
