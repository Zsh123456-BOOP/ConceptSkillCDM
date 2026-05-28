# Concept Evidence Gap Main-Text Update Packet

This update is inference-only / aggregation-only / plotting-only. It uses ASSIST09, Junyi, and ASSIST17 with existing CRG/LCRF checkpoints and existing diagnostic artifacts. No retraining, model-structure edits, dataset additions, or parameter tuning were performed.

## Generated Artifacts

- `results/main_problem_experiments_20260523/main_text/evidence_gap_impact_curve.csv`
- `results/main_problem_experiments_20260523/main_text/coverage_conditioned_prediction.csv`
- `results/main_problem_experiments_20260523/main_text/crg_evidence_source_decomposition.csv`
- `results/main_problem_experiments_20260523/main_text/lcrf_posterior_state_alignment.csv`
- `results/main_problem_experiments_20260523/main_text/caption_ready_lcrf_case_summary.csv`
- `results/main_problem_experiments_20260523/main_text/direct_evidence_removal_boundary_summary.csv`
- `results/main_problem_experiments_20260523/main_text/fig_concept_gap_diagnosis.pdf`
- `results/main_problem_experiments_20260523/main_text/fig_concept_gap_diagnosis.png`
- `results/main_problem_experiments_20260523/main_text/diagnostic_extension_manifest.csv`

## Success and Weakness Summary

- Evidence gap impact: ASSIST17 shows higher BCE in the target-count-zero bin, but ASSIST09 is not monotonic and Junyi has almost all test events in the zero-count bin. The safe conclusion is dataset-dependent association, not a universal difficulty law.
- History-to-query retrieval: ASSIST09 and Junyi pass the positive criterion. ASSIST09 random Hit@10 is 0.0673, seq-only is 0.1781, and fused CRG is 0.1574. Junyi random Hit@10 is 0.0191, while seq-only and fused CRG both reach 0.1379. ASSIST17 is not a positive case for this retrieval setting.
- Coverage-conditioned prediction: results are mixed because some route-supported groups improve but low-route or unbridgeable controls are not consistently weak. Do not use this experiment as main positive evidence.
- LCRF posterior-state alignment: selected ASSIST09 and ASSIST17 cases show case-level posterior shifts associated with learner-state variables. This supports case-level interpretation, not broad statistical proof.
- Direct evidence removal: the available experiment is only a buffer-level state-mask diagnostic. It is not promoted as main-text evidence; the paper only keeps the general boundary that direct target-concept history remains important.

## Figure Recommendation

Use `fig_concept_gap_diagnosis` as the single main-text experimental figure for the concept evidence gap subsection. It now keeps only the stable positive retrieval results and drops the mixed coverage-conditioned plot from the main text:

- Panel A: history-to-query retrieval on direct-unseen-bridgeable samples for ASSIST09 and Junyi.
- Panel B: held-out transition retrieval source decomposition on ASSIST09, Junyi, and ASSIST17.

## `main_en.tex` Changes

- Replaced `Route Retrieval and Coverage-Conditioned Prediction` with `Diagnosis under Concept Evidence Gaps`.
- Added one main-text figure: `fig_concept_gap_diagnosis.pdf`, containing only retrieval/source-decomposition panels.
- Removed the separate held-out transition figure from this subsection and summarized source decomposition in prose.
- Added one LCRF case-level alignment sentence after the same-query posterior case.
- Kept the limitation wording general; direct-evidence-removal is not used as a main-text result.

## Safe Claim Wording

- Direct coverage count alone is not used as a main claim because the pattern is not stable across datasets.
- CRG retrieves target concepts from learner histories on ASSIST09 and Junyi.
- Held-out transition retrieval supports the empirical concept-support role of sequence/fused CRG variants.
- Junyi supports the route-support signal, while prediction-level gains are small or mixed.
- Same-query posterior cases illustrate learner-conditioned filtering at the case level.
- Direct target-concept history remains a strong diagnostic signal.

## Claims Not Allowed

- Do not write that CRG learns prerequisites.
- Do not write that CRG universally beats arbitrary graphs.
- Do not write that Junyi proves LCRF necessity.
- Do not write that student-ID shortcut is fully ruled out.
- Do not write that CRG/LCRF completely solves direct concept coverage absence.
- Do not write that direct evidence removal proves robustness.

## Missing Inputs

- `global_only` was not evaluated because there is no exact existing checkpoint or inference hook for a pure global-only prediction path.
- No baseline sample-level predictions were available for the new subgroup diagnostics.

