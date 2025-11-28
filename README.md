# ConceptSkillCDM: Graph-based Concept & Skill Disentangled Cognitive Diagnosis

ConceptSkillCDM learns unsupervised concept relations, propagates student knowledge over the learned graph, and disentangles knowledge mastery from test-taking skills to predict responses.

## Project Structure
```
ConceptSkillCDM/
├── data/
│   ├── assist_09/{train,valid,test}.csv
│   ├── junyi/{train,valid,test}.csv
│   └── assist_17/{train,valid,test}.csv
├── logs/                  # logs and saved checkpoints
├── src/
│   ├── config.py          # hyperparameters and CLI
│   ├── dataset.py         # CSV loading and variable-length batching
│   ├── graph_learner.py   # multi-head attention concept graph
│   ├── gnn_layers.py      # concept propagation
│   ├── model.py           # ConceptSkillCDM core model
│   ├── losses.py          # BCE + graph regularizers + disentanglement
│   ├── trainer.py         # train/validation/test loops
│   ├── metrics.py         # AUC / ACC / RMSE
│   └── utils.py           # seeds, logging, device helpers
├── main.py                # single-dataset training entry
├── run_all_datasets.py    # run assist_09, assist_17, junyi, sample sequentially
└── requirements.txt
```

## Data Format
Each CSV row is one interaction with mandatory columns:
- `student_id` (int)
- `item_id` (int)
- `correct` (0/1)
- `concept_ids` (str) e.g. `"3;7;10"` for linked concepts

## Model Highlights
1. **Embeddings**: student/item/concept embeddings.
2. **Concept graph learner**: multi-head self-attention discovers concept relations (precedence, similarity, other).
3. **GNN propagation**: spreads student knowledge over the learned graph.
4. **Disentanglement**: splits knowledge state vs. test-taking skill vector.
5. **Fusion for prediction**: soft-min aggregation over linked concepts + skill offset for final logits.

## Loss
`L = L_BCE + L_graph + L_de`, where:
- `L_BCE`: binary cross-entropy on logits.
- `L_graph`: sparsity + symmetry (similarity head) + DAG soft constraint (precedence head).
- `L_de`: disentanglement via orthogonality + InfoNCE-style mutual information minimization.

## Quick Start
```bash
conda create -n conceptskillcdm python=3.10
conda activate conceptskillcdm
pip install -r requirements.txt

# Single dataset
python main.py --dataset assist_09 --device auto --seed 42
python main.py --dataset assist_17 --device auto --seed 42
python main.py --dataset junyi     --device auto --seed 42
python main.py --dataset sample    --device auto --seed 42   # junyi 的小规模子集快速调试

# Run all four (默认 assist_09, assist_17, junyi, sample，跳过 junyi copy)
python run_all_datasets.py --device auto
```

Logs include per-epoch train loss and valid AUC/ACC/RMSE, plus final best-model metrics on test. Saved checkpoints are written to `logs/{dataset}_best.pt`.

## Dataset Notes
- assist_09, assist_17, junyi, sample 是默认会一起跑的四个数据集。
- sample 是 junyi 的小规模子集，用于快速调试，示例：`python main.py --dataset sample --device auto`
- `junyi copy` 只是人工备份目录，任何脚本默认都不会使用；仅当你手动指定 `--dataset "junyi copy"` 时才会加载。

## Device Auto-Selection
- 当 `--device auto`（默认）时：如果检测到 CUDA，则调用 `gpu_utils.get_best_gpu()` 自动选择剩余显存最多的 GPU；否则自动退回 CPU。
- 也可以显式指定：`--device cpu`、`--device cuda`、`--device cuda:1` 等。

## Target Performance Reference (for guidance)
- Assist09: AUC ≈ 0.782 / ACC ≈ 0.744 / RMSE ≈ 0.415
- Junyi:   AUC ≈ 0.823 / ACC ≈ 0.781 / RMSE ≈ 0.389
- Assist17: AUC ≈ 0.896 / ACC ≈ 0.870 / RMSE ≈ 0.301

These are research goals, not enforced assertions.
# ConceptSkillCDM
