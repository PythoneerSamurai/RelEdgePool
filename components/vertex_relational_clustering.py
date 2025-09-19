from itertools import product

import torch
from torch.nn.utils.rnn import pad_sequence


class VertexRelationalClustering:
    def __init__(
            self,
            valid_clusters: list,
            auxiliary_clusters: list,
            vertex_mapping,
            mean_mapping,
            device="cpu"
    ) -> None:
        """
           Vertex Relational Clustering component of RelEdgePool algorithm.
           Reconstructs connectivity between pooled vertices using
           both direct and indirect relational checks.
       """

        self.device = device

        # Pad cluster sequences to uniform length for tensor operations
        self.valid_clusters = pad_sequence(valid_clusters, batch_first=True, padding_value=-1).to(device)

        try:
            self.aux_clusters = pad_sequence(auxiliary_clusters, batch_first=True, padding_value=-1).to(device)
        except:
            self.aux_clusters = torch.full((1, self.valid_clusters.shape[1]), -4, dtype=torch.int8, device=device)

        self.tensor_to_remove = torch.full((1, self.valid_clusters.shape[1]), -1, dtype=torch.int8, device=device)

        # Mappings between valid core vertices and the averaged vertices of their clusters
        self.vertex_mapping = vertex_mapping.to(device) if hasattr(vertex_mapping, 'to') else torch.tensor(
            vertex_mapping, device=device)
        self.mean_mapping = mean_mapping.to(device) if hasattr(mean_mapping, 'to') else torch.tensor(mean_mapping,
                                                                                                     device=device)
        # Initialize tensor for storing updated edges
        self.updated_edges = torch.full((1, 2), -1, dtype=torch.int8, device=self.device)

    def _check_row_matches(
            self,
            clusters: torch.Tensor,
            test_values: torch.Tensor
    ) -> torch.Tensor:
        """
            Check if any value in test_values appears in any row of clusters.
            Used to filter clusters based on the presence of test_values.
        """

        flat_clusters = clusters.reshape(-1)
        matches = torch.isin(flat_clusters, test_values)

        return matches.reshape(clusters.shape).any(dim=1)

    def _check_specific_aux_in_valid(
            self,
            valid_clusters: torch.Tensor,
            aux_clusters: torch.Tensor,
            aux_row_mask: torch.Tensor
    ) -> tuple:
        """
           Check if auxiliary cluster cores appear in valid clusters.
           Used for indirect edge detection when vertices aren't directly connected
           but share connections through auxiliary clusters.
        """

        masked_aux = aux_clusters[aux_row_mask]

        if masked_aux.numel() == 0:
            return self.tensor_to_remove

        # Extract core vertices (last non-padding element) from auxiliary clusters
        aux_core_mask = masked_aux != -1
        last_non_minus_one_indices = aux_core_mask.sum(dim=1) - 1
        aux_cluster_cores = masked_aux[torch.arange(masked_aux.size(0)), last_non_minus_one_indices]
        flat_valid = valid_clusters.reshape(-1)
        matches = torch.isin(flat_valid, aux_cluster_cores)

        return matches.reshape(valid_clusters.shape).any(dim=1), aux_cluster_cores

    def _compute_cartesian_product(
            self,
            tensor1: torch.Tensor,
            tensor2: torch.Tensor
    ) -> list:
        """
            Compute cartesian product between two tensors to form all possible edge combinations.
            Used to create edges between averaged vertices of connected clusters.
        """

        cartesian_product = list(product(tensor1, tensor2))
        return cartesian_product

    def _direct_and_indirect_edge_detector(self, index: int) -> None:
        """
            Core edge reconstruction algorithm implementing both direct and indirect edge detection.
            Direct and Indirect relational checks are performed for all vertices in a cluster in parallel.
            Details mentioned in the paper.
        """

        source_cluster = self.valid_clusters[index]

        # Extract cluster vertices, removing padding (-1) and processed markers (-2)
        trimmed_cluster = source_cluster[source_cluster != -1]
        trimmed_cluster = trimmed_cluster[trimmed_cluster != -2]
        source_core = [trimmed_cluster[-1]] # Core vertex is always last

        # Remove current cluster from consideration to avoid self-connections
        source_removal_mask = ~torch.all((self.valid_clusters == source_cluster), dim=1)
        valid_clusters = self.valid_clusters[source_removal_mask]

        # DIRECT EDGE DETECTION
        # Find valid clusters that share vertices with current cluster
        valid_matches = self._check_row_matches(valid_clusters, trimmed_cluster)
        valid_mask = valid_matches.unsqueeze(1).expand_as(valid_clusters)

        # Filter clusters to only those with connections
        unfiltered_valid_connections = torch.where(valid_mask, valid_clusters, self.tensor_to_remove)
        valid_cluster_filteration_mask = ~torch.all(unfiltered_valid_connections == self.tensor_to_remove, dim=1)
        filtered_valid_connections = unfiltered_valid_connections[valid_cluster_filteration_mask]

        # Identify which vertices have direct connections and those that don't
        vertices_with_valid_cons = filtered_valid_connections[
            torch.isin(filtered_valid_connections, trimmed_cluster)
        ].unique()
        vertices_without_valid_cons = trimmed_cluster[~torch.isin(trimmed_cluster, vertices_with_valid_cons)]

        # INDIRECT EDGE DETECTION
        try:
            # Check if vertices without direct connections appear in auxiliary clusters
            aux_matches = self._check_row_matches(self.aux_clusters, vertices_without_valid_cons)
            aux_in_valid_matches, aux_cluster_cores = self._check_specific_aux_in_valid(
                valid_clusters,
                self.aux_clusters,
                aux_matches
            )
            aux_in_valid_mask = aux_in_valid_matches.unsqueeze(1).expand_as(valid_clusters)
            unfiltered_aux_connections = torch.where(aux_in_valid_mask, valid_clusters, self.tensor_to_remove)
            filtered_aux_connections = unfiltered_aux_connections[
                ~torch.all(unfiltered_aux_connections == self.tensor_to_remove, dim=1)
            ]

            aux_with_valid_cons = filtered_aux_connections[
                torch.isin(filtered_aux_connections, aux_cluster_cores)
            ].unique()

            # Combine direct and indirect connections
            total_connections = torch.vstack((filtered_valid_connections, filtered_aux_connections)).unique(dim=0)
            total_vertices_with_valid_cons = torch.hstack((vertices_with_valid_cons, aux_with_valid_cons))

            # Mark connected vertices, in all but the source valid cluster, as processed (-2) to prevent redundant edges
            self.valid_clusters[source_removal_mask] = torch.where(
                torch.isin(self.valid_clusters[source_removal_mask], total_vertices_with_valid_cons), -2,
                self.valid_clusters[source_removal_mask])

        except ValueError:
            # Fallback to only direct connections if no indirect connections are found
            self.valid_clusters[source_removal_mask] = torch.where(
                torch.isin(self.valid_clusters[source_removal_mask], vertices_with_valid_cons), -2,
                self.valid_clusters[source_removal_mask])
            total_connections = filtered_valid_connections

        # Extract core vertices from connected clusters
        core_mask = total_connections != -1
        last_non_minus_one_indices = core_mask.sum(dim=1) - 1
        cluster_cores = total_connections[torch.arange(total_connections.size(0)), last_non_minus_one_indices]

        # Map original core vertex indices to corresponding averaged vertex indices
        source_mean_mask = torch.isin(self.vertex_mapping, torch.tensor(source_core, device=self.device))
        source_mean = self.mean_mapping[source_mean_mask]
        core_means_mask = torch.isin(self.vertex_mapping, cluster_cores)
        core_means = self.mean_mapping[core_means_mask]

        # Create edges
        cartesian_product = self._compute_cartesian_product(source_mean, core_means)

        if len(cartesian_product) != 0:
            self.updated_edges = torch.vstack((self.updated_edges, torch.tensor(cartesian_product, device=self.device)))

        # Sort edges to avoid duplication
        self.updated_edges, _ = self.updated_edges.sort(dim=1)

    def get_edges(self) -> torch.Tensor:
        """
            Execute complete edge reconstruction process and return unique edge list.
            This is the main entry point for the vertex relational clustering phase.
        """

        for index in range(len(self.valid_clusters)):
            self._direct_and_indirect_edge_detector(index)
            
        return self.updated_edges[1:].unique(dim=0)  # Remove initial placeholder and duplicates
