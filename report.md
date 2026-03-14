# CognitiveDiagnosisModel Report

## Model overview
- Dual-branch design: cognitive IRT 2PL + MF residual correction
- Shared concept embeddings with multi-head relation learning (row-stochastic adjacency)
- Graph propagation with entropy sparsity regularizer
- Soft prototype correction (optional)
- Personal graph (optional) via student-conditioned low-rank adjacency and adaptive gate

## Mathematical form
Notation:
- student s, exercise e, concept c
- Q-mask q_e in {0,1}^C from data
- knowledge state H_s in R^{C x D}

Concept graph encoding:
```
H_s^0 = C + s
H_s^{l+1} = LN(H_s^l + gamma * GNN(H_s^l, A))
```
where A is a learned multi-head row-stochastic adjacency (or identity in ablation).

Cognitive (IRT) branch:
```
theta_c = w^T H_s[c] + b
theta_e = sum_c q_e[c] * theta_c / sum_c q_e[c]
logit_irt = a_e * (theta_e - b_e)
```

MF residual branch:
```
u = normalize(W_u z_s)
v = normalize(W_v z_e)
logit_mf = softplus(s) * <u, v> + b_mf + b_s + b_e
```
where z_s is student latent, and z_e is exercise latent with optional Q-conditioned concept mix.

Fusion:
```
gate = sigmoid(W_g [logit_irt, logit_mf])
logit = logit_irt + gate * logit_mf
p = sigmoid(logit)
```

Soft prototype (optional):
```
p_s = softmax(sim(h_s, P) / tau) * P
H_s <- (1 - lambda_proto) * H_s + lambda_proto * p_s
```

Personal graph (optional):
```
A_personal = softmax(U_s V_s^T)
A_used = (1 - alpha_s) * A_global + alpha_s * A_personal
```

## Training setup
- Data filtering (train/val/test jointly):
  - Remove students with interactions < min_stu_interactions
  - Remove exercises with interactions < min_exer_interactions
  - Remove "poison" items with count >= min_poison_count and correctness = 0 or 1
- Loss: BCEWithLogitsLoss on logits
- Regularization:
  - Graph entropy sparsity (lambda_sparse)
  - L2 on MF + IRT parameters (exercise_l2_lambda)
  - Prototype diversity/usage (lambda_proto_div, lambda_proto_usage)
  - Personal graph sparsity + alpha penalty (lambda_sparse_personal, lambda_alpha)
- Optimizer: Adam + ReduceLROnPlateau scheduler
- Early stopping: best validation AUC with patience early_stop_patience
- Seeds: numpy + torch + CUDA deterministic flags from main.py

## Ablation definitions
- full: all modules enabled (concept graph, MF residual, soft prototype)
- ablate_soft_prototype:
  - Disable soft prototype correction only
  - Keep concept graph, MF residual, gating, personal graph (if enabled)
- ablate_skill_encoder (pure IRT 2PL):
  - Disable MF residual branch entirely
  - Remove mf_bias / student_bias / exercise_bias contributions
  - Final logit = a_e * (theta_e - b_e)
- ablate_concept_graph:
  - Replace learned concept adjacency with identity rows (self-loop only)
  - Keep GNN layers but remove cross-concept message passing

## Repro commands
Full model:
```
python main.py --dataset_name assist_09 --model_variant full --save_dir ./checkpoints/full --log_dir ./logs/full
```

Ablation: no soft prototype
```
python main.py --dataset_name assist_09 --model_variant no_soft_proto --ablate_soft_prototype --save_dir ./checkpoints/no_soft_proto --log_dir ./logs/no_soft_proto
```

Ablation: pure IRT (no MF branch)
```
python main.py --dataset_name assist_09 --model_variant no_skill --ablate_skill_encoder --save_dir ./checkpoints/no_skill --log_dir ./logs/no_skill
```

Ablation: no concept graph (alias flag kept for legacy scripts)
```
python main.py --dataset_name assist_09 --model_variant no_concept_graph --ablate_concept_graph --save_dir ./checkpoints/no_concept_graph --log_dir ./logs/no_concept_graph
```
