from typing import Dict, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class ExerciseDifficultyEncoder(nn.Module):
    """固定的 IRT 题目参数编码器，只负责输出 b/a。"""

    def __init__(self, num_exercises: int):
        super().__init__()
        self.b = nn.Embedding(num_exercises, 1)
        self.a_raw = nn.Embedding(num_exercises, 1)
        self._initialize_weights()

    def _initialize_weights(self) -> None:
        nn.init.zeros_(self.b.weight)
        nn.init.normal_(self.a_raw.weight, mean=0.0, std=0.02)

    def forward(self, exercise_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        b = self.b(exercise_ids).squeeze(-1)
        a = F.softplus(self.a_raw(exercise_ids).squeeze(-1)) + 1e-6
        return b, a


class CognitiveDiagnosisHead(nn.Module):
    """
    固定预测头 D：2PL-IRT
    - theta_c：对每个概念的能力
    - theta_e：按 Q-mask 聚合后的题目能力
    - irt_logit = a * (theta_e - b)
    """

    def __init__(self, knowledge_dim: int):
        super().__init__()
        self.theta_proj = nn.Linear(knowledge_dim, 1, bias=True)
        nn.init.normal_(self.theta_proj.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.theta_proj.bias)

    def forward(
        self,
        knowledge_state: torch.Tensor,
        concept_mask: torch.Tensor,
        b: torch.Tensor,
        a: torch.Tensor,
        return_details: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        theta_c = self.theta_proj(knowledge_state).squeeze(-1)

        mask = concept_mask.float()
        denom = mask.sum(dim=1).clamp(min=1.0)
        theta_e = (theta_c * mask).sum(dim=1) / denom

        irt_logit = a * (theta_e - b)

        if not return_details:
            return irt_logit

        details = {
            "theta_c": theta_c.detach(),
            "theta_e": theta_e.detach(),
            "irt_logit": irt_logit.detach(),
        }
        return irt_logit, details
