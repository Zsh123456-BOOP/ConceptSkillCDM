import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.model_graph import StudentKnowledgeEncoder


def _encoder(alpha: float, student_global_scale: float = 1.0) -> StudentKnowledgeEncoder:
    torch.manual_seed(7)
    enc = StudentKnowledgeEncoder(
        num_students=3,
        num_concepts=4,
        knowledge_dim=5,
        num_gnn_layers=1,
        num_relation_heads=2,
        dropout=0.0,
        gnn_residual_weight=0.5,
        propagation_alpha=alpha,
        student_global_scale=student_global_scale,
    )
    enc.eval()
    return enc


def _relations() -> torch.Tensor:
    return torch.tensor(
        [
            [
                [0.70, 0.20, 0.10, 0.00],
                [0.10, 0.70, 0.20, 0.00],
                [0.00, 0.20, 0.70, 0.10],
                [0.10, 0.00, 0.20, 0.70],
            ],
            [
                [0.60, 0.00, 0.30, 0.10],
                [0.20, 0.60, 0.00, 0.20],
                [0.10, 0.30, 0.60, 0.00],
                [0.00, 0.20, 0.20, 0.60],
            ],
        ],
        dtype=torch.float32,
    )


def test_graph_propagation_alpha_zero_preserves_initial_state() -> None:
    student_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    enc = _encoder(alpha=0.0)
    with torch.no_grad():
        initial = enc.compose_initial_state(student_ids)
        out = enc(student_ids, _relations())
    assert torch.allclose(out, initial, atol=1e-6), "alpha=0 should disable graph state updates"


def test_graph_propagation_alpha_controls_update_strength() -> None:
    student_ids = torch.tensor([0, 1, 2], dtype=torch.long)
    low = _encoder(alpha=0.05)
    high = _encoder(alpha=0.50)
    high.load_state_dict(low.state_dict())
    with torch.no_grad():
        initial = low.compose_initial_state(student_ids)
        low_delta = (low(student_ids, _relations()) - initial).pow(2).mean().sqrt()
        high_delta = (high(student_ids, _relations()) - initial).pow(2).mean().sqrt()
    assert high_delta > low_delta * 2.0, "larger alpha should produce a stronger graph update"


def test_student_global_scale_zero_removes_student_id_state_shortcut() -> None:
    student_ids = torch.tensor([0, 1], dtype=torch.long)
    scaled = _encoder(alpha=0.0, student_global_scale=1.0)
    disabled = _encoder(alpha=0.0, student_global_scale=0.0)
    disabled.load_state_dict(scaled.state_dict())
    with torch.no_grad():
        scaled_state = scaled.compose_initial_state(student_ids)
        disabled_state = disabled.compose_initial_state(student_ids)
    assert not torch.allclose(
        scaled_state[0], scaled_state[1], atol=1e-6
    ), "default student-global embedding should make different students distinct"
    assert torch.allclose(
        disabled_state[0], disabled_state[1], atol=1e-6
    ), "scale=0 should remove the student-id shortcut from the initial concept state"
