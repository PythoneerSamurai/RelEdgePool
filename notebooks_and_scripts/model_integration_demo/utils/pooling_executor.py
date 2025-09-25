import sys
from pathlib import Path

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent.parent
components_dir = root_dir / "components"
sys.path.append(str(components_dir))

from edge_based_vertex_averaging import EdgeBasedVertexAveraging


class PoolingExecutor:
    def __init__(self, vertices_batch: Tensor, edges_batch: list) -> None:
        self.vertices_batch = vertices_batch
        self.edges_batch = edges_batch

        self.updated_vertices = []
        self.updated_edges = []

    def pool(self) -> tuple:
        padding = torch.zeros((1, self.vertices_batch.shape[2]), dtype=torch.float32)
        for vertices, edges in zip(self.vertices_batch, self.edges_batch):
            padding_removal_mask = ~torch.all((vertices == padding), dim=1)
            vertices = vertices[padding_removal_mask]
            updated_vertices, updated_edges = EdgeBasedVertexAveraging(vertices, edges, "cpu").pool()
            self.updated_vertices.append(updated_vertices)
            self.updated_edges.append(updated_edges)

        self.updated_vertices = pad_sequence(self.updated_vertices, batch_first=True, padding_value=0.)

        return self.updated_vertices, self.updated_edges
