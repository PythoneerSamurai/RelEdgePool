import numpy as np
import point_cloud_utils as pcu
from scipy.stats import wasserstein_distance
from torch import Tensor


class MeshQualityMetrics:
    def __init__(
            self,
            original_mesh_verts: Tensor,
            original_mesh_edges: Tensor,
            pooled_mesh_verts: Tensor,
            pooled_mesh_edges: Tensor
    ):
        self.original_mesh_verts = original_mesh_verts.numpy()
        self.original_mesh_edges = original_mesh_edges.tolist()

        self.pooled_mesh_verts = pooled_mesh_verts.numpy()
        self.pooled_mesh_edges = pooled_mesh_edges.tolist()

    def compute_hausdorff(self):
        return pcu.hausdorff_distance(self.original_mesh_verts, self.pooled_mesh_verts)

    def compute_edge_length_distribution_similarity(self):
        orig_edge_coords = self.original_mesh_verts[self.original_mesh_edges]
        pool_edge_coords = self.pooled_mesh_verts[self.pooled_mesh_edges]

        orig_lengths = np.linalg.norm(orig_edge_coords[:, 1] - orig_edge_coords[:, 0], axis=1)
        pool_lengths = np.linalg.norm(pool_edge_coords[:, 1] - pool_edge_coords[:, 0], axis=1)

        return wasserstein_distance(orig_lengths, pool_lengths).tolist()

    def compute_spectral_similarity(self):
        def build_laplacian(vertices, edges):
            n_vertices = len(vertices)
            adj_matrix = np.zeros((n_vertices, n_vertices))
            for edge in edges:
                adj_matrix[edge[0], edge[1]] = 1
                adj_matrix[edge[1], edge[0]] = 1

            degree_matrix = np.diag(np.sum(adj_matrix, axis=1))

            laplacian = degree_matrix - adj_matrix

            eigenvals = np.linalg.eigvals(laplacian)
            return np.sort(np.real(eigenvals))[:10]

        orig_spectrum = build_laplacian(self.original_mesh_verts, self.original_mesh_edges)
        pool_spectrum = build_laplacian(self.pooled_mesh_verts, self.pooled_mesh_edges)

        orig_spectrum = orig_spectrum / np.max(np.abs(orig_spectrum))
        pool_spectrum = pool_spectrum / np.max(np.abs(pool_spectrum))

        return np.mean(np.abs(orig_spectrum - pool_spectrum)).tolist()
