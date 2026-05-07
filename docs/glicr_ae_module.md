# GLICR-AE module

`GLICR-AE` is the project name for the current A/E module:

**Global-Local Interpretable Concept Relation Module**

Chinese name: **全局-局部可解释概念关系模块**.

## Component names

| Short name | Paper/code name | Meaning |
| --- | --- | --- |
| A | Evidence-Guided Global Concept Relation Graph | A train-only, student-independent concept relation graph. It fuses item co-occurrence evidence, sequence transition evidence, self-loop retention, and a small interpretable receiver bias. |
| E | Student-Conditioned Local Posterior Reweighting | A personalized posterior over the same support supplied by A. It reweights existing support edges using current student concept-state contrast and does not create arbitrary new edges. |
| AE reliability | Train-only Evidence Reliability Residual | A joint A+E residual based on standardized train-only count features, such as student count, exercise count, concept count, and student-concept count. It is active only when both A and E are active. |

## Why this name

The name avoids saying that A is only a multi-concept co-occurrence graph. A is now a global evidence-fused concept relation graph, so it still has a clear interpretation on single-concept datasets through train-only sequence evidence.

E remains local and constrained: it does not generate a separate dense student graph, does not read student-id embedding as a shortcut, and only changes edge weights inside A's support.

## Ablation boundary

`no_A` removes the global relation substrate. This includes the evidence-fused graph support and A-side relation residual.

`no_E` keeps A but removes the student-conditioned local posterior. The train-only reliability residual is also disabled because it is defined as a joint A+E term.

This makes the final full/no_A/no_E comparison a direct test of GLICR-AE instead of a hidden prediction-head change.
