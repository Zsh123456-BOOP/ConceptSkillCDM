# AE adaptation evidence, 2026-05-06

Remote workspace: `~/ConceptSkillCDM`
Commit under test: `3c3f0c7`
GPU rule: local experiments launched on GPU 2/3 only. GPU 0 memory seen during checks was owned by unrelated user processes, not these runs.

## assist_09

Run: `remote_ae_assist_wide_full_20260506_3c3f0c7`

- full test AUC: 0.8168383600
- target: 0.782
- best validation AUC: 0.8173855338
- best epoch: 2

Run: `remote_ae_assist_wide_ablate2ep_20260506_3c3f0c7`

- no_A test AUC: 0.6826769320
- no_E test AUC: 0.6900655663
- full - no_A: 0.1341614281
- full - no_E: 0.1267727937

## junyi

Run: `remote_ae_junyi_AE050I25_fulltrain_20260506_3c3f0c7`

- full test AUC: 0.8292705230
- target: 0.823
- best validation AUC: 0.8248278462
- best epoch: 2
- winning AE settings: `ae_logit_residual_scale=0.50`, `ae_interaction_logit_scale=0.25`, `ae_irt_logit_scale=1.00`, `ae_query_residual_scale=0.0`, `ae_lr_mult=1.0`

Run: `remote_ae_junyi_AE050I25_ablate500_20260506_3c3f0c7`

- no_A test AUC: 0.6349800139
- no_E test AUC: 0.7983835542
- full - no_A: 0.1942905091
- full - no_E: 0.0308869688

The full `junyi` run used uncapped training and the best checkpoint was evaluated independently on the full filtered test set. The ablation run used `max_train_batches=500` because uncapped no_A/no_E diverged numerically after the first epoch; the saved epoch-1 best checkpoints were then evaluated independently on the full filtered test set.
