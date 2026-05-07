# 2026-05-07 GLICR-AE ablation results

Run id: `ae_reliability_gpu23_20260507_170717`

Remote source: `/home/zsh/ConceptSkillCDM`

Remote code commit: `96e39d3 Add train-only AE reliability features`

Local backup archive:

`C:\Users\zsh\Desktop\test_xph\ConceptSkillCDM_artifact_backups\ae_reliability_gpu23_20260507_170717_20260508_073034.zip`

SHA256:

`258E1A21860368B5B5A870EBB664494163BA62A92023E6689BCFD9306A57903E`

## Summary

| Dataset | Variant | Test AUC | Best Val AUC | Epoch | Test ACC | Test RMSE |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| assist_09 | full | 0.779646 | 0.778956 | 6 | 0.740712 | 0.417428 |
| assist_09 | no_A | 0.681722 | 0.681624 | 1 | 0.679291 | 0.457334 |
| assist_09 | no_E | 0.692737 | 0.693441 | 3 | 0.686945 | 0.448478 |
| junyi | full | 0.829135 | 0.824817 | 20 | 0.766721 | 0.399369 |
| junyi | no_A | 0.635569 | 0.641430 | 1 | 0.611135 | 0.484923 |
| junyi | no_E | 0.798560 | 0.789544 | 1 | 0.738453 | 0.425556 |

## Target check

`assist_09` target: `>= 0.778`; observed full test AUC: `0.779646`.

`junyi` target: `>= 0.823`; observed full test AUC: `0.829135`.

## Committed artifacts

Result CSV files:

- `results/experiment_results.csv`
- `results/abce_ablation_diagnosis.csv`
- `results/abce_ablation_summary.csv`
- `results/abce_ablation_summary_mean.csv`

Training logs:

- `logs/abce_diag/ae_reliability_gpu23_20260507_170717/assist_09/seed42/best_full/train_20260507_170720.log`
- `logs/abce_diag/ae_reliability_gpu23_20260507_170717/assist_09/seed42/best_no_A/train_20260507_170720.log`
- `logs/abce_diag/ae_reliability_gpu23_20260507_170717/assist_09/seed42/best_no_E/train_20260507_172431.log`
- `logs/abce_diag/ae_reliability_gpu23_20260507_170717/junyi/seed42/best_full/train_20260507_172951.log`
- `logs/abce_diag/ae_reliability_gpu23_20260507_170717/junyi/seed42/best_no_A/train_20260507_173041.log`
- `logs/abce_diag/ae_reliability_gpu23_20260507_170717/junyi/seed42/best_no_E/train_20260507_174642.log`

Runner log:

- `server_logs/ae_reliability_gpu23_20260507_170717.out`

The stale runner pid file was intentionally not committed.
