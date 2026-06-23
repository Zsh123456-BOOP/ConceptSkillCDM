# Evidence-Grounded Concept Reachability (CRG/LCRF refactor)

## The critique this addresses

> "CRG 基本上就是一个数据统计结果。统计是通过训练集统计的，跟问题关系不大，因为问题是未观测知识点，统计出来的结果其实没有利用上未观测数据。"

In the **per-student concept-holdout (chold)** setting, a test query targets a concept the
student has **not** practiced (it is still observed by *other* students, so it is globally
trainable). The old model failed this scenario for two concrete reasons:

1. **The student's observed evidence never entered the graph.** The GNN initial state was
   `concept_emb[c] + student_global(student_id)` - a memorized per-student ID vector that is
   identical regardless of which concept is queried. Nothing about *what this student actually
   did* was propagated to the unobserved target concept.
2. **CRG edge weights were frozen statistics.** Edges came from train co-occurrence /
   sequence-transition log-ratios scaled by a scalar; only temperature/bias/self-loop were
   learned. The support set was hard-frozen from counts.

Symptom in the logs: `A_degree_random ≈ full` and `E_shuffle_student ≥ full` - the graph
carried no student-specific, transferable signal, so randomizing or shuffling it cost nothing.

## What changed (two modules + minimal glue)

### CRG - `src/model_graph.py`

- **Evidence-grounded initial state.** `StudentKnowledgeEncoder.encode_evidence(mastery,
  recent, count)` builds a `(B, C, D)` node-feature state from the **train-only** buffers
  `ae_student_concept_prior_logit` / `ae_student_concept_recent_logit` /
  `ae_student_concept_observed_count`. It is **reliability-gated** (`vec *= count/(count+s)`),
  so a concept the student never observed (including the held-out target, `count=0`)
  contributes **exactly zero** and can only be estimated by graph propagation from the
  student's observed concepts. The final projection layer is **zero-initialized**.
- **Learnable edge weights.** `MultiHeadRelationLearning` adds a per-head low-rank bilinear
  term over concept embeddings (`edge_query_proj`, `edge_key_proj`) on top of the
  co-occurrence/transition prior, with a **zero-initialized** per-head `learned_edge_scale`.
  Edge weights are now trained end-to-end (no longer a frozen statistic) while support stays
  prior-defined (interpretable, sparse).

### LCRF - `src/model_personal.py`

- **Reliability-preferring route.** `PersonalRelationGenerator` adds
  `reliability_pref * normalized(support_reliability)` (zero-initialized scalar) to the route
  residual, so the per-student route prefers support concepts the student has *actually
  observed* (real evidence). This makes `E_shuffle_student` a genuine counterfactual.

### Glue

- `model_structure.py` / `model_structure_forward.py` thread the evidence state into the
  encoder (and the diagnostic initial-state).
- `model_cdm.py` / `model_cdm_forward.py` add `graph_evidence_scale` (default 1.0) and
  `graph_evidence_reliability_smoothing` (default 8.0) and feed the train-only buffers to the
  graph whenever the concept graph is active (independent of the LCRF mastery/recency scales,
  which still self-gate).
- `main.py` + `src/trainer.py`: `--graph_evidence_scale` / `--graph_evidence_reliability_smoothing`
  CLI flags wired at all construction sites.

## Design guarantees and required checks

- **Zero regression at init.** With zero-initialized evidence projection, edge scale, and
  reliability preference, the new paths are inactive at initialization. It only *learns* to
  use the new signal, so the well-tuned standard-split baselines are not perturbed
  structurally.
- **No leakage.** Evidence buffers are train-only and `= 0` on the held-out target at eval.
- **Held-out routing target.** The held-out (`count=0`) query node should receive evidence
  only through graph-routed neighbors, not direct target-concept observations.
- **Controls to verify.** `E_shuffle_student` should change predictions; `no_A` should remove
  the routing substrate; `degree_random` should route evidence through irrelevant concepts.

## Suggested experiments (existing controls become meaningful, no new flags needed)

`full` vs `no_A` vs `A_self_only` vs `A_degree_random` vs `E_shuffle_student`, plus a new
clean isolation of the evidence component:

```bash
# evidence-grounded reachability OFF (graph kept, only ID embedding) - isolates the new signal
python main.py --dataset_name assist_09_chold ... --graph_evidence_scale 0
```

Expected story: `full` > `--graph_evidence_scale 0` (evidence matters), `full` > `no_A`
(graph routing matters), `full` > `E_shuffle_student` (the *right* student's evidence matters),
`full` >= `A_degree_random` (routing through relevant concepts beats random).
