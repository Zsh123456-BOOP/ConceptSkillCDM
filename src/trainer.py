import copy
import os
import time
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
        self.best_epoch = 0
        self.patience = getattr(config, "patience", 0)

    def train(self):
        no_improve = 0
        for epoch in range(1, self.config.epochs + 1):
            start_time = time.time()
            train_stats = self._train_one_epoch()
            epoch_time = time.time() - start_time

            val_metrics = self.evaluate(self.valid_loader, split_name="valid")
            val_auc = val_metrics["AUC"]
            val_acc = val_metrics["ACC"]
            val_rmse = val_metrics["RMSE"]

            self._log(
                f"Epoch {epoch:02d} | T: {epoch_time:.1f}s | "
                f"Train: Loss={train_stats['loss']:.4f} "
                f"(BCE={train_stats['bce']:.4f}, Graph={train_stats['graph']:.6f}, "
                f"De={train_stats['de']:.6f}, Orth={train_stats['orth']:.6f}, MI={train_stats['mi']:.4f})"
            )
            self._log(
                f"[Valid] AUC={val_auc:.4f} | ACC={val_acc:.4f} | RMSE={val_rmse:.4f} | "
                f"Best AUC={(self.best_valid_auc if self.best_valid_auc >= 0 else float('nan')):.4f} "
                f"(epoch={self.best_epoch if self.best_epoch > 0 else '-'})"
            )

            if val_auc > self.best_valid_auc:
                prev_best = self.best_valid_auc
                self.best_valid_auc = val_auc
                self.best_epoch = epoch
                self.best_state = copy.deepcopy(self.model.state_dict())
                torch.save(self.best_state, self.save_path)
                no_improve = 0

                if prev_best < 0:
                    self._log(f"*** New Best Valid AUC: {val_auc:.4f} (epoch={epoch}) ***")
                else:
                    self._log(
                        f"*** New Best Valid AUC: {val_auc:.4f} (epoch={epoch}, prev={prev_best:.4f}) ***"
                    )
            else:
                no_improve += 1
                if self.patience and self.patience > 0 and no_improve >= self.patience:
                    self._log(
                        f"Early stopping triggered: no AUC improvement for {no_improve} epochs "
                        f"(patience={self.patience})."
                    )
                    break

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
        test_metrics = self.evaluate(self.test_loader, split_name="test")
        test_auc = test_metrics["AUC"]
        test_acc = test_metrics["ACC"]
        test_rmse = test_metrics["RMSE"]
        self._log(
            f"[Test] AUC={test_auc:.4f} | ACC={test_acc:.4f} | RMSE={test_rmse:.4f}"
        )
        self._log(
            f"[Summary] Best Valid AUC={self.best_valid_auc:.4f} at epoch={self.best_epoch}; "
            f"Final Test AUC={test_auc:.4f}, ACC={test_acc:.4f}, RMSE={test_rmse:.4f}"
        )
        return test_metrics

    def _train_one_epoch(self):
        self.model.train()
        total_loss = 0.0
        total_bce = total_graph = total_de = total_orth = total_mi = 0.0
        num_batches = 0
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
            num_batches += 1
        denom = max(num_batches, 1)
        return {
            "loss": total_loss / denom,
            "bce": total_bce / denom,
            "graph": total_graph / denom,
            "de": total_de / denom,
            "orth": total_orth / denom,
            "mi": total_mi / denom,
        }

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
