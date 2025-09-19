import torch
import torch.nn as nn

from vertex_relational_clustering import VertexRelationalClustering


class EdgeBasedVertexAveraging(nn.Module):
    def __init__(
            self,
            vertices: torch.Tensor,
            edges: torch.Tensor,
            device="cpu"
    ) -> None:
        """
            Edge-Based Vertex Averaging component of RelEdgePool algorithm.
            Performs dimensionality reduction by computing means of local vertex clusters.
            Identifies valid and auxiliary clusters for edge re-establishment.
        """

        super().__init__()
        self.device = device
        self.vertices = vertices.clone()
        self.edges = edges.clone()

    def _compute_adjacency_matrix(self) -> None:
        """
            Creates a sparse adjacency matrix using the tensor of edges.
        """

        num_vertices = self.vertices.shape[0]
        
        # Create bidirectional edges by concatenating original and flipped edges
        indices = torch.cat([self.edges, self.edges.flip(1)], dim=0).t()

        values = torch.ones(indices.shape[1], device=self.device)

        adj_matrix = torch.sparse_coo_tensor(
            indices,
            values,
            size=(num_vertices, num_vertices),
            device=self.device
        )

        adj_matrix = adj_matrix.coalesce()
        return adj_matrix

    def _compute_means(
            self,
            vertices: torch.Tensor,
            adj_matrix: torch.Tensor
    ) -> tuple:
        """
            Core averaging algorithm that implements the first phase of RelEdgePool.
            Details mentioned in the paper.
        """
        num_vertices = vertices.shape[0]

        assert vertices.requires_grad, "Input vertices tensor must have requires_grad=True"

        # Extract row and column indices from sparse adjacency matrix
        edge_indices = adj_matrix.indices()
        row_indices = edge_indices[0]
        col_indices = edge_indices[1]

        # Track which vertices haven't been processed yet
        unprocessed_mask = torch.ones(num_vertices, dtype=torch.bool, device=self.device)

        # Pre-allocate storage for averaged vertices
        max_possible_clusters = num_vertices
        averaged_vertices = torch.zeros(max_possible_clusters, vertices.shape[1],
                                        dtype=vertices.dtype, device=self.device)

        # Storage for cluster information needed for edge reconstruction
        valid_clusters = []  # Clusters where core was unprocessed
        auxiliary_clusters = []  # Clusters where core was already processed
        vertex_mappings = torch.tensor([-1], dtype=torch.int8)  # Stores core vertices
        mean_mappings = torch.tensor([-1], dtype=torch.int8)  # Stores indices of averaged vertices
        num_clusters = 0

        for vertex_idx in range(num_vertices):
            vertex_idx = torch.tensor(vertex_idx, device=self.device)
            mask = (row_indices == vertex_idx)
            connections = col_indices[mask]

            # Form cluster: connections + core vertex (core is always last)
            cluster = torch.cat([connections, vertex_idx.unsqueeze(0)]) if len(connections) > 0 \
                else vertex_idx.unsqueeze(0)

            if unprocessed_mask[vertex_idx]:
                # Valid cluster: core vertex hasn't been processed
                valid_clusters.append(cluster)

                # Only average unprocessed vertices in the cluster
                vertices_for_mean = cluster[unprocessed_mask[cluster]]

                # Record mappings for edge reconstruction phase
                vertex_mappings = torch.hstack((vertex_mappings, torch.tensor([vertex_idx.cpu().item()])))
                mean_mappings = torch.hstack((mean_mappings, torch.tensor([num_clusters])))

                # Compute cluster centroid
                if len(vertices_for_mean) == 1:
                    averaged_vertices[num_clusters] = vertices[vertex_idx]
                else:
                    cluster_vertices = vertices[vertices_for_mean]
                    mean = cluster_vertices.mean(dim=0)
                    averaged_vertices[num_clusters] = mean

                # Mark all vertices in cluster as processed
                new_mask = unprocessed_mask.clone()
                new_mask[vertices_for_mean] = False
                unprocessed_mask = new_mask

                num_clusters += 1

            else:
                # Auxiliary cluster: core was already processed, but save for edge reconstruction
                auxiliary_clusters.append(cluster)

        # Trim to actual number of clusters created
        averaged_vertices = averaged_vertices[:num_clusters].clone()

        return averaged_vertices, vertex_mappings[1:], mean_mappings[1:], valid_clusters, auxiliary_clusters

    def pool(self) -> tuple:
        """
            Execute complete RelEdgePool algorithm:
            1. Perform edge-based vertex averaging for dimensionality reduction
            2. Use vertex relational clustering to reconstruct connectivity
        """

        adj_matrix = self._compute_adjacency_matrix()
        updated_vertices, vertex_mappings, mean_mappings, valid_clusters, auxiliary_clusters = self._compute_means(
            self.vertices.clone(), adj_matrix)

        # Use vertex relational clustering to establish edges between averaged vertices
        updated_edges = VertexRelationalClustering(valid_clusters, auxiliary_clusters, vertex_mappings, mean_mappings,
                                                   device=self.device).get_edges()

        return updated_vertices, updated_edges
