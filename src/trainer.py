import copy
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from .losses import bce_loss, graph_loss, disentangle_loss
from .metrics import compute_auc, compute_acc, compute_rmse
from .utils import ensure_dir


class Trainer:
    def __init__(self, model: nn.Module, config, train_loader, valid_loader, test_loader, device, logger=None):
        self.model = model.to(device)
        self.config = config
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        self.device = device
        self.logger = logger
        ensure_dir(config.save_dir)
        self.save_path = os.path.join(config.save_dir, f"{config.dataset}_best.pt")

        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )
        self.best_state = None
        self.best_valid_auc = -1.0

    def train(self):
        for epoch in range(1, self.config.epochs + 1):
            train_stats = self._train_one_epoch()
            val_metrics = self.evaluate(self.valid_loader, split_name="valid")
            val_auc = val_metrics["AUC"]
            if val_auc > self.best_valid_auc:
                self.best_valid_auc = val_auc
                self.best_state = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_state, self.save_path)
            self._log(
                f"[Epoch {epoch}] TrainLoss={train_stats['loss']:.4f} "
                f"(BCE={train_stats['bce']:.4f}, Graph={train_stats['graph']:.4f}, "
                f"De={train_stats['de']:.4f}, Orth={train_stats['orth']:.4f}, MI={train_stats['mi']:.4f}) "
                f"Valid AUC={val_metrics['AUC']:.4f} ACC={val_metrics['ACC']:.4f} RMSE={val_metrics['RMSE']:.4f}"
            )

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        test_metrics = self.evaluate(self.test_loader, split_name="test")
        self._log(
            f"[Test] AUC={test_metrics['AUC']:.4f} ACC={test_metrics['ACC']:.4f} RMSE={test_metrics['RMSE']:.4f}"
        )
        return test_metrics

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_bce = total_graph = total_de = total_orth = total_mi = 0.0
        for batch in self.train_loader:
            batch = self._to_device(batch)
            labels = batch["labels"]
            logits, A_list, S_prop, z_skill, z_know = self.model(batch)

            L_bce = bce_loss(logits, labels)
            L_graph, graph_regs = graph_loss(
                A_list, self.model.head_types, self.config, self.model.graph_learner
            )
            L_de, L_orth, L_mi = disentangle_loss(
                S_prop, z_skill, self.config, z_know=z_know, critic=None
            )
            loss = L_bce + L_graph + L_de

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_bce += L_bce.item()
            total_graph += L_graph.item()
            total_de += L_de.item()
            total_orth += L_orth.item()
            total_mi += L_mi.item()
        denom = max(len(self.train_loader), 1)
        stats = {
            "loss": total_loss / denom,
            "bce": total_bce / denom,
            "graph": total_graph / denom,
            "de": total_de / denom,
            "orth": total_orth / denom,
            "mi": total_mi / denom,
        }
        self.last_train_stats = stats
        return stats

    def evaluate(self, loader, split_name: str = "valid") -> Dict[str, float]:
        self.model.eval()
        logits_all = []
        labels_all = []
        with torch.no_grad():
            for batch in loader:
                batch = self._to_device(batch)
                labels = batch["labels"]
                logits, _, _, _, _ = self.model(batch)
                logits_all.append(logits.detach().cpu())
                labels_all.append(labels.detach().cpu())

        logits_all = torch.cat(logits_all).view(-1)
        labels_all = torch.cat(labels_all).view(-1)
        probs = torch.sigmoid(logits_all).numpy()
        labels_np = labels_all.numpy()

        metrics = {
            "AUC": compute_auc(labels_np, probs),
            "ACC": compute_acc(labels_np, probs),
            "RMSE": compute_rmse(labels_np, probs),
        }
        return metrics

    def _to_device(self, batch):
        return {k: v.to(self.device) for k, v in batch.items()}

    def _log(self, msg: str):
        if self.logger is None:
            print(msg)
        else:
            self.logger.info(msg)


__all__ = ["Trainer"]
