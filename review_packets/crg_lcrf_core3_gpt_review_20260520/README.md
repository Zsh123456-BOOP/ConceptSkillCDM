# CRG/LCRF Core-3 GPT Review Packet

This folder is the minimal packet to upload to GPT Pro for checking the paper story, experiment evidence, and figures. It contains only the current core-3 setting:

- `assist_09`
- `junyi`
- `assist_17`

No extra datasets, old intermediate logs, checkpoints, or obsolete figures are included.

## What To Upload

Upload this whole folder:

```text
review_packets/crg_lcrf_core3_gpt_review_20260520/
```

If the upload interface accepts one archive more easily, upload the generated zip:

```text
review_packets/crg_lcrf_core3_gpt_review_20260520.zip
```

## Folder Contents

### `docs/`

- `crg_lcrf_paper_outline.md`: current paper outline, figure plan, claim boundaries.
- `crg_lcrf_core3_review_packet.md`: summary of final core-3 diagnostics and reviewer-facing risks.
- `top20_cd_paper_story_review.md`: top-conference / top-journal CD paper story review notes.

### `figures/main/`

Main-text candidate figures:

- `fig2_core3_data_and_crg_retrieval_final.png`
- `fig3_core3_support_corruption_final.png`
- `fig4_core3_lcrf_counterfactual_final.png`
- `fig5_core3_lcrf_same_query_posterior_final.png`

### `figures/appendix/`

Appendix candidate figures:

- `figS_crg_local_route_cases_core3.png`
- `figS_crg_subgroup_support_dependence_core3.png`
- `figS_lcrf_specific_student_timeline_core3.png`
- `figS_lcrf_state_source_audit_core3.png`

### `tables/`

Compact paper tables and figure summaries:

- `table_main_ablation_core3.csv`
- `table_main_ablation_core3.tex`
- `fig2_core3_retrieval_summary.csv`
- `paper_figure_summary_core3.csv`

### `data/`

CSV evidence behind the figures and claims:

- dataset story cards
- CRG retrieval
- CRG support corruption / subgroup audit
- LCRF counterfactual
- LCRF same-query posterior cases
- CRG local route cases
- LCRF timeline and state-source limitation audit
- run manifest

### `scripts/`

- `plot_crg_lcrf_core3_final.R`: the R plotting script used for the final figures.

## Key Claim Boundaries

Use these boundaries when asking GPT to review the material:

1. CRG is the main contribution: a train-only concept reachability roadmap.
2. LCRF is the secondary contribution: learner-conditioned filtering inside CRG support.
3. Sequence transition should be described as an empirical learning route, not as a prerequisite relation.
4. CRG necessity is dataset-dependent:
   - `assist_17`: strongest CRG necessity evidence.
   - `assist_09`: support-dependence evidence.
   - `junyi`: strong data/retrieval story but weak prediction-level corruption.
5. Junyi should not be used as strong LCRF evidence.
6. LCRF should not be described as generating new graph edges.
7. The state-source audit is a limitation analysis and does not fully rule out student-ID shortcut.

