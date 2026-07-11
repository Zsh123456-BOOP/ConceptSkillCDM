"""Query-safe collaborative aggregation on a train-only response graph."""

from __future__ import annotations

import hashlib
from typing import Dict, Tuple

import torch
import torch.nn as nn


class ResponseGraphEncoder(nn.Module):
    """Aggregate student/item neighbours with exact target-edge exclusion.

    The input is a sparse binary ``student x item`` support matrix. Edge values
    are ignored, so outcomes cannot enter the graph. For a training query
    ``(student, item)``, its own edge is subtracted from both directional
    aggregates and from both degrees. Validation/test query edges are absent
    already, making the same forward rule consistent across splits.

    Neighbours are stored as two compact CSR adjacency lists. A forward pass
    expands only the rows required by the batch's unique students and items;
    it never multiplies the complete graph by an embedding table. The output is
    the fixed mean of the node's base embedding and its neighbour mean. Isolated
    nodes (including degree-one nodes after target exclusion) fall back exactly
    to their base embedding.

    The graph arrays are non-persistent buffers, while a small schema and
    fingerprint are stored as module extra state. Strict checkpoint loading can
    therefore reject a same-shaped but different graph without serializing the
    complete graph twice.
    """

    GRAPH_SCHEMA = "response_graph_binary_csr_v1"

    def __init__(self, student_item_adjacency: torch.Tensor) -> None:
        super().__init__()
        (
            student_crow_indices,
            student_col_indices,
            item_crow_indices,
            item_col_indices,
            edge_keys,
            shape,
        ) = self._prepare_graph(student_item_adjacency)
        self.num_students = int(shape[0])
        self.num_items = int(shape[1])
        self.num_edges = int(edge_keys.numel())
        self.graph_fingerprint = self._fingerprint_graph(
            self.num_students,
            self.num_items,
            edge_keys,
        )

        # These arrays are reconstructed from the train-only graph in the
        # checkpoint's info_dict. Persisting them here would duplicate the graph.
        self.register_buffer("student_crow_indices", student_crow_indices, persistent=False)
        self.register_buffer("student_col_indices", student_col_indices, persistent=False)
        self.register_buffer("item_crow_indices", item_crow_indices, persistent=False)
        self.register_buffer("item_col_indices", item_col_indices, persistent=False)
        self.register_buffer("edge_keys", edge_keys, persistent=False)

    @staticmethod
    def _prepare_graph(
        adjacency: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        Tuple[int, int],
    ]:
        if not isinstance(adjacency, torch.Tensor):
            raise TypeError("student_item_adjacency must be a sparse torch.Tensor")
        if adjacency.layout not in (torch.sparse_coo, torch.sparse_csr):
            raise TypeError(
                "student_item_adjacency must use sparse COO or CSR layout; "
                f"got {adjacency.layout}"
            )
        if adjacency.dim() != 2 or adjacency.size(0) <= 0 or adjacency.size(1) <= 0:
            raise ValueError(
                "student_item_adjacency must have non-empty shape "
                f"(num_students, num_items), got {tuple(adjacency.shape)}"
            )
        coo = (
            adjacency.coalesce()
            if adjacency.layout == torch.sparse_coo
            else adjacency.to_sparse_coo().coalesce()
        )
        indices = coo.indices()
        if indices.size(1) == 0:
            raise ValueError("student_item_adjacency must contain at least one edge")

        num_students = int(coo.size(0))
        num_items = int(coo.size(1))
        student_rows = indices[0]
        item_rows = indices[1]

        # A coalesced COO tensor is sorted by (row, column), so its item indices
        # already form the student-side CSR column array.
        student_degree = torch.bincount(student_rows, minlength=num_students)
        student_crow_indices = torch.zeros(
            num_students + 1,
            device=indices.device,
            dtype=torch.long,
        )
        student_crow_indices[1:] = student_degree.cumsum(dim=0)
        student_col_indices = item_rows.contiguous()

        # Sort by (item, student) to form the reverse CSR adjacency list.
        reverse_keys = item_rows * num_students + student_rows
        reverse_order = reverse_keys.argsort()
        item_degree = torch.bincount(item_rows, minlength=num_items)
        item_crow_indices = torch.zeros(
            num_items + 1,
            device=indices.device,
            dtype=torch.long,
        )
        item_crow_indices[1:] = item_degree.cumsum(dim=0)
        item_col_indices = student_rows[reverse_order].contiguous()

        edge_keys = (student_rows * num_items + item_rows).sort().values.contiguous()
        return (
            student_crow_indices,
            student_col_indices,
            item_crow_indices,
            item_col_indices,
            edge_keys,
            (num_students, num_items),
        )

    @classmethod
    def _fingerprint_graph(
        cls,
        num_students: int,
        num_items: int,
        edge_keys: torch.Tensor,
    ) -> str:
        digest = hashlib.sha256()
        header = (
            f"{cls.GRAPH_SCHEMA}|{int(num_students)}|{int(num_items)}|"
            f"{int(edge_keys.numel())}|"
        )
        digest.update(header.encode("ascii"))
        canonical_keys = edge_keys.detach().to(device="cpu", dtype=torch.long).contiguous()
        digest.update(canonical_keys.numpy().tobytes(order="C"))
        return digest.hexdigest()

    def _graph_extra_state(self) -> Dict[str, object]:
        return {
            "schema": self.GRAPH_SCHEMA,
            "num_students": self.num_students,
            "num_items": self.num_items,
            "num_edges": self.num_edges,
            "fingerprint": self.graph_fingerprint,
        }

    def get_extra_state(self) -> Dict[str, object]:
        """Persist only enough graph identity to validate a reconstructed graph."""
        return dict(self._graph_extra_state())

    def set_extra_state(self, state: Dict[str, object]) -> None:
        """Reject checkpoints built from a different response graph."""
        expected = self._graph_extra_state()
        if not isinstance(state, dict):
            raise RuntimeError(
                "Response graph checkpoint extra state must be a dictionary; "
                f"got {type(state).__name__}."
            )
        missing = sorted(set(expected) - set(state))
        if missing:
            raise RuntimeError(
                "Response graph checkpoint extra state is missing fields: "
                f"{missing}."
            )
        mismatches = {
            key: (expected[key], state.get(key))
            for key in expected
            if state.get(key) != expected[key]
        }
        if mismatches:
            mismatch_text = ", ".join(
                f"{key}: current={current!r}, checkpoint={checkpoint!r}"
                for key, (current, checkpoint) in mismatches.items()
            )
            raise RuntimeError(f"Response graph schema/fingerprint mismatch ({mismatch_text}).")

    def _validate_embeddings(
        self,
        student_base: torch.Tensor,
        item_base: torch.Tensor,
    ) -> None:
        expected_student = (self.num_students, student_base.size(-1))
        expected_item = (self.num_items, student_base.size(-1))
        if student_base.dim() != 2 or tuple(student_base.shape) != expected_student:
            raise ValueError(
                f"student_base must have shape {expected_student}, got {tuple(student_base.shape)}"
            )
        if item_base.dim() != 2 or tuple(item_base.shape) != expected_item:
            raise ValueError(
                f"item_base must have shape {expected_item}, got {tuple(item_base.shape)}"
            )
        if student_base.device != item_base.device or student_base.dtype != item_base.dtype:
            raise ValueError("student and item embeddings must share device and dtype")
        if not student_base.is_floating_point() or not item_base.is_floating_point():
            raise TypeError("student and item embeddings must be floating point")

    @staticmethod
    def _validate_ids(ids: torch.Tensor, size: int, name: str) -> None:
        if not isinstance(ids, torch.Tensor) or ids.dtype != torch.long:
            raise TypeError(f"{name} must be a torch.long tensor")
        if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= size):
            raise IndexError(f"{name} values must lie in [0, {size})")

    def _edge_present(self, student_ids: torch.Tensor, item_ids: torch.Tensor) -> torch.Tensor:
        query_keys = student_ids * self.num_items + item_ids
        positions = torch.searchsorted(self.edge_keys, query_keys)
        safe_positions = positions.clamp(max=self.edge_keys.numel() - 1)
        return (positions < self.edge_keys.numel()) & (
            self.edge_keys[safe_positions] == query_keys
        )

    @staticmethod
    def _aggregate_csr_rows(
        crow_indices: torch.Tensor,
        col_indices: torch.Tensor,
        row_ids: torch.Tensor,
        neighbour_base: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Aggregate only CSR rows referenced by ``row_ids``.

        Duplicate queried rows are expanded once and mapped back with
        ``inverse``. Gather and ``index_add`` keep the result differentiable
        with respect to the dense neighbour embedding table.
        """
        unique_rows, inverse = torch.unique(row_ids, sorted=True, return_inverse=True)
        if unique_rows.numel() == 0:
            empty_shape = (*row_ids.shape, int(neighbour_base.size(-1)))
            return neighbour_base.new_zeros(empty_shape), neighbour_base.new_zeros(row_ids.shape)

        starts = crow_indices[unique_rows]
        lengths = crow_indices[unique_rows + 1] - starts
        total_neighbours = int(lengths.sum().item())
        unique_sums = neighbour_base.new_zeros(
            (int(unique_rows.numel()), int(neighbour_base.size(-1)))
        )
        if total_neighbours > 0:
            unique_offsets = lengths.cumsum(dim=0) - lengths
            owner_rows = torch.repeat_interleave(
                torch.arange(unique_rows.numel(), device=row_ids.device),
                lengths,
                output_size=total_neighbours,
            )
            edge_positions = (
                torch.repeat_interleave(starts, lengths, output_size=total_neighbours)
                + torch.arange(total_neighbours, device=row_ids.device)
                - torch.repeat_interleave(
                    unique_offsets,
                    lengths,
                    output_size=total_neighbours,
                )
            )
            neighbours = col_indices[edge_positions]
            unique_sums = unique_sums.index_add(0, owner_rows, neighbour_base[neighbours])

        return unique_sums[inverse], lengths.to(dtype=neighbour_base.dtype)[inverse]

    @staticmethod
    def _mix_base_and_neighbours(
        base: torch.Tensor,
        neighbour_sum: torch.Tensor,
        degree: torch.Tensor,
    ) -> torch.Tensor:
        observed = degree > 0
        neighbour_mean = neighbour_sum / degree.clamp(min=1.0).unsqueeze(-1)
        mixed = 0.5 * (base + neighbour_mean)
        return torch.where(observed.unsqueeze(-1), mixed, base)

    def forward(
        self,
        student_base: torch.Tensor,
        item_base: torch.Tensor,
        student_ids: torch.Tensor,
        item_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return query-specific contexts with the current edge left out."""
        self._validate_embeddings(student_base, item_base)
        self._validate_ids(student_ids, self.num_students, "student_ids")
        self._validate_ids(item_ids, self.num_items, "item_ids")
        if tuple(student_ids.shape) != tuple(item_ids.shape):
            raise ValueError("student_ids and item_ids must have the same shape")

        student_sums, student_degree = self._aggregate_csr_rows(
            self.student_crow_indices,
            self.student_col_indices,
            student_ids,
            item_base,
        )
        item_sums, item_degree = self._aggregate_csr_rows(
            self.item_crow_indices,
            self.item_col_indices,
            item_ids,
            student_base,
        )
        edge_present = self._edge_present(student_ids, item_ids).to(student_base.dtype)
        student_sums = student_sums - edge_present.unsqueeze(-1) * item_base[item_ids]
        item_sums = item_sums - edge_present.unsqueeze(-1) * student_base[student_ids]
        student_degree = student_degree - edge_present
        item_degree = item_degree - edge_present
        student_context = self._mix_base_and_neighbours(
            student_base[student_ids], student_sums, student_degree
        )
        item_context = self._mix_base_and_neighbours(
            item_base[item_ids], item_sums, item_degree
        )
        return student_context, item_context

    def encode_students(
        self,
        student_base: torch.Tensor,
        item_base: torch.Tensor,
        student_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Encode students against their complete training neighbourhood."""
        self._validate_embeddings(student_base, item_base)
        self._validate_ids(student_ids, self.num_students, "student_ids")
        neighbour_sums, student_degree = self._aggregate_csr_rows(
            self.student_crow_indices,
            self.student_col_indices,
            student_ids,
            item_base,
        )
        return self._mix_base_and_neighbours(
            student_base[student_ids],
            neighbour_sums,
            student_degree,
        )

    def extra_repr(self) -> str:
        return (
            f"num_students={self.num_students}, num_items={self.num_items}, "
            f"num_edges={self.num_edges}, schema={self.GRAPH_SCHEMA}, "
            f"fingerprint={self.graph_fingerprint[:12]}, target_edge_exclusion=True"
        )
