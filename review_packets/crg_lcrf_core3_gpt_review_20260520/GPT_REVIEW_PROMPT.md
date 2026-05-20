# Prompt To Give GPT Pro

You are a cognitive diagnosis paper reviewer and experiment-design advisor. Please review this uploaded folder as a self-contained evidence packet for a paper about CRG and LCRF.

Do not assume access to the original repository. Use only the files in this packet.

## Files To Read First

1. `docs/crg_lcrf_paper_outline.md`
2. `docs/crg_lcrf_core3_review_packet.md`
3. `docs/top20_cd_paper_story_review.md`
4. `tables/paper_figure_summary_core3.csv`
5. `tables/table_main_ablation_core3.csv`
6. `tables/fig2_core3_retrieval_summary.csv`
7. `data/dataset_story_cards_core3.csv`
8. `data/crg_retrieval_full_core3.csv`
9. `data/crg_support_gap_audit_core3.csv`
10. `data/crg_subgroup_support_dependence_core3.csv`
11. `data/lcrf_counterfactual_delta_core3.csv`
12. `data/lcrf_same_query_candidate_summary_core3.csv`
13. `data/lcrf_same_query_annotated_core3.csv`
14. `data/lcrf_two_student_path_case_core3.csv`
15. `data/lcrf_state_source_audit_core3.csv`
16. `figures/main/*.png`
17. `figures/appendix/*.png`

## Review Goals

Please judge whether the current paper story, experiments, and figures can support the following claims:

### Claim 1: Data Phenomenon

Real CD datasets contain sparse response evidence where a queried concept may not be directly observed in a learner history, but can often be reached through train-only concept routes.

Check:

- Whether `assist_09`, `junyi`, and `assist_17` are enough for this claim.
- Whether the dataset cards in Figure 2 make this problem clear.
- Whether the wording should avoid overemphasizing "single-concept items" and instead emphasize sparse direct evidence plus bridgeable concept routes.

### Claim 2: CRG Sufficiency

CRG is sufficient as a train-only roadmap because it retrieves held-out concept transitions better than self-only and random baselines.

Check:

- Whether `fig2_core3_data_and_crg_retrieval_final.png` and the retrieval CSVs support this claim.
- Whether the paper should say "empirical learning route" instead of "prerequisite".
- Whether the claim should be strongest on Junyi and assist_17, or balanced across all three datasets.

### Claim 3: CRG Necessity / Support Dependence

The model depends on CRG support, but the evidence is dataset-dependent.

Current intended wording:

- `assist_17`: strongest evidence for CRG necessity.
- `assist_09`: support-dependence evidence, not exclusive evidence-edge superiority.
- `junyi`: weak prediction-level corruption result, should be reported cautiously.

Check:

- Whether `fig3_core3_support_corruption_final.png`, `figS_crg_subgroup_support_dependence_core3.png`, and the support audit CSVs support this cautious claim.
- Whether the figure caption and paper wording should avoid saying CRG beats any arbitrary graph on every dataset.
- Whether support-dependence is enough for a main-text claim, or should be moved partly to appendix.

### Claim 4: LCRF Necessity

LCRF improves prediction when real learner-conditioned state is used; shuffled or mean state weakens the result.

Check:

- Whether `fig4_core3_lcrf_counterfactual_final.png` and `data/lcrf_counterfactual_delta_core3.csv` support this claim.
- Whether Junyi should be greyed out / marked weak.
- Whether no-filter, mean-state, and shuffle-state should be interpreted separately.

### Claim 5: LCRF Sufficiency / Interpretability

Under the same CRG support, different learners receive different posterior routes, and these posterior changes can be linked to learner mastery/recent state.

Check:

- Whether `fig5_core3_lcrf_same_query_posterior_final.png` supports this claim.
- Whether assist_17 should be the main case and assist_09 should be appendix/supporting.
- Whether the two-student local path panel is understandable enough.
- Whether the same-query evidence should be written as a mechanism case rather than a broad statistical proof.

### Claim 6: Limitation

The state-source audit should be presented as a limitation. It cannot fully rule out a student-ID shortcut.

Check:

- Whether `figS_lcrf_state_source_audit_core3.png` should remain appendix-only.
- Whether the outline has removed strong claims such as "LCRF does not use student-ID shortcut".

## Required Output

Please give a concrete review with four sections:

1. **Acceptable Main Story**
   - State whether the CRG/LCRF story is coherent.
   - Say which claims are safe for the main paper.

2. **Figure-by-Figure Review**
   - For Figure 2, Figure 3, Figure 4, Figure 5, and appendix figures:
     - keep / revise / move to appendix / remove
     - exact reason
     - exact caption wording to avoid overclaiming

3. **Paper Writing Edits**
   - Give concrete wording for Introduction, Method, Experiment, Mechanism Analysis, Limitation.
   - Include warnings about forbidden claims.

4. **Missing Evidence Or Optional Improvements**
   - If no more experiments are needed, say so.
   - If one or two small no-retraining diagnostics would materially improve the paper, specify:
     - input CSV/checkpoint needed
     - output CSV
     - figure type
     - success/failure criterion

Do not ask for new large-scale training unless absolutely necessary. Prefer honest claim boundaries over forcing every dataset to be positive.

