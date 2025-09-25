import torch
from pytorch3d.ops import GraphConv
from torch.nn.utils.rnn import pad_sequence


class GraphConvolver:
    def __init__(self, in_channels: int, out_channels: int) -> None:
        self.gc = GraphConv(input_dim=in_channels, output_dim=out_channels)

    def convolve(self, vertices_batch: torch.Tensor, edges_batch: list) -> torch.Tensor:
        updates_vertices_list = []
        padding = torch.zeros((1, vertices_batch.shape[2]), dtype=torch.float32)
        for vertices, edges in zip(vertices_batch, edges_batch):
            padding_removal_mask = ~torch.all((vertices == padding), dim=1)
            vertices = vertices[padding_removal_mask]
            updated_vertices = self.gc(vertices, edges)
            updates_vertices_list.append(updated_vertices)

        updates_vertices = pad_sequence(updates_vertices_list, batch_first=True, padding_value=0.)

        return updates_vertices
