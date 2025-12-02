import time
import numpy as np
import torch
import torch.optim as optim
from sklearn.metrics import roc_auc_score, accuracy_score, mean_squared_error


class Trainer:
    """
    适配 DisentangledCDM 的 Trainer：
    - 训练使用 CDMLoss（BCE + DAG + Sparse + HSIC）
    - DataLoader batch 期望格式：
        (student_ids, exercise_ids, labels, q_mask)
    """
    def __init__(self, model, loss_fn, loaders, args, logger):
        self.model = model
        self.loss_fn = loss_fn
        self.args = args
        self.logger = logger

        (self.train_loader,
         self.val_loader,
         self.test_loader,
         self.high_loader,
         self.med_loader,
         self.low_loader) = loaders

        self.device = next(self.model.parameters()).device
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )

    def _move_batch_to_device(self, batch):
        """假定 batch = (stu_ids, exer_ids, labels, q_mask)"""
        stu_ids, exer_ids, labels, q_mask = batch
        stu_ids = stu_ids.to(self.device)
        exer_ids = exer_ids.to(self.device)
        labels = labels.float().to(self.device)
        q_mask = q_mask.float().to(self.device)
        return stu_ids, exer_ids, labels, q_mask

    def train_epoch(self, epoch):
        self.model.train()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_dag_loss = 0.0
        total_sparse_loss = 0.0
        total_hsic_loss = 0.0

        start = time.time()

        for batch in self.train_loader:
            stu_ids, exer_ids, labels, q_mask = self._move_batch_to_device(batch)

            self.optimizer.zero_grad()

            pred, adj_dag, h_knowledge, z_skill = self.model(
                stu_ids, exer_ids, q_mask
            )
            loss, loss_dict = self.loss_fn(
                pred, labels, adj_dag, h_knowledge, z_skill
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            total_pred_loss += loss_dict.get("pred", 0.0)
            total_dag_loss += loss_dict.get("dag", 0.0)
            total_sparse_loss += loss_dict.get("sparse", 0.0)
            total_hsic_loss += loss_dict.get("hsic", 0.0)

        n = len(self.train_loader) or 1
        avg_total = total_loss / n
        avg_pred = total_pred_loss / n
        avg_dag = total_dag_loss / n
        avg_sparse = total_sparse_loss / n
        avg_hsic = total_hsic_loss / n

        self.logger.info(
            f"Epoch {epoch} | T: {time.time() - start:.1f}s | "
            f"Loss: {avg_total:.4f} "
            f"(Pred={avg_pred:.4f}, DAG={avg_dag:.4f}, "
            f"Sparse={avg_sparse:.4f}, HSIC={avg_hsic:.4f})"
        )

        return avg_total

    def evaluate(self, loader, name):
        if loader is None:
            return 0.0, 0.0, 0.0

        self.model.eval()
        preds, labels = [], []

        with torch.no_grad():
            for batch in loader:
                stu_ids, exer_ids, lbls, q_mask = self._move_batch_to_device(batch)
                pred, _, _, _ = self.model(stu_ids, exer_ids, q_mask)
                preds.extend(pred.detach().cpu().numpy().tolist())
                labels.extend(lbls.detach().cpu().numpy().tolist())

        if len(labels) == 0:
            return 0.0, 0.0, 0.0

        preds = np.array(preds)
        labels = np.array(labels)

        try:
            auc = roc_auc_score(labels, preds)
        except ValueError:
            auc = 0.5

        acc = accuracy_score(labels, preds > 0.5)
        rmse = np.sqrt(mean_squared_error(labels, preds))

        self.logger.info(f"[{name}] AUC: {auc:.4f} | ACC: {acc:.4f} | RMSE: {rmse:.4f}")
        return auc, acc, rmse
