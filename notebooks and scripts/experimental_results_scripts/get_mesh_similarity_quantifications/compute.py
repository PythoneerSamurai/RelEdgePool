import os
from pathlib import Path
from pprint import pprint

from pytorch3d.io import IO
from torch import tensor, float32, int64

from mesh_quality_metrics import MeshQualityMetrics

current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent.parent
datasets_dir = root_dir / "datasets/single_iteration_pooled_meshes"
original_meshes_dir = root_dir / "datasets/single_iteration_pooled_meshes/original_meshes"

assert os.path.exists(datasets_dir), ("Pooled meshes not found. Execute the relevant script to automatically download "
                                      "and extract the pooled meshes.")

def load_wireframe(filename: str):
    vertices = []
    edges = []
    with open(filename, "r") as f:
        for line in f:
            if line.startswith("v "):  # Vertex
                _, x, y, z = line.strip().split()
                vertices.append([float(x), float(y), float(z)])
            elif line.startswith("l "):  # Face
                edge = [int(part.split('/')[0]) - 1 for part in line.strip().split()[1:]]
                edges.append(edge)

    vertices = tensor(vertices, dtype=float32)
    edges = tensor(edges, dtype=int64)
    return vertices, edges


results = {
    "meshcnn": {
        "SHREC": {},
        "Cubes": {}
    },
    "edge_contraction_pooling": {
        "SHREC": {},
        "Cubes": {}
    },
    "mesh_conv_3d": {
        "SHREC": {},
        "Cubes": {}
    },
    "reledgepool": {
        "SHREC": {},
        "Cubes": {}
    }
}

hausdorff_distance = []
edge_length_distribution_similarity = []
spectral_similarity = []

for algorithm in os.listdir(datasets_dir):
    if algorithm.lower() == "original_meshes":
        continue
    print(f"Algorithm: {algorithm}")
    algorithm_path = os.path.join(datasets_dir, algorithm)
    for dataset in os.listdir(algorithm_path):
        print(f"Computing metrics for {dataset}")
        dataset_dir = os.path.join(algorithm_path, dataset)
        for obj in os.listdir(dataset_dir):
            obj_path = os.path.join(dataset_dir, obj)
            original_mesh_path = f"{original_meshes_dir}/{dataset}/{obj}"
            original_mesh = IO().load_mesh(original_mesh_path, False)
            original_verts, original_edges = original_mesh.verts_packed(), original_mesh.edges_packed()

            if algorithm.lower() == "reledgepool":
                pooled_verts, pooled_edges = load_wireframe(obj_path)
            else:
                pooled_mesh = IO().load_mesh(obj_path, False)
                pooled_verts, pooled_edges = pooled_mesh.verts_packed(), pooled_mesh.edges_packed()

            metrics = MeshQualityMetrics(original_verts, original_edges, pooled_verts, pooled_edges)

            hausdorff_distance.append(metrics.compute_hausdorff())
            edge_length_distribution_similarity.append(metrics.compute_edge_length_distribution_similarity())
            spectral_similarity.append(metrics.compute_spectral_similarity())

        results[algorithm][dataset]["Hausdorff Distance"] = round(sum(hausdorff_distance) / len(hausdorff_distance), 5)
        results[algorithm][dataset]["Edge Length Distribution Similarity"] = round(
            sum(edge_length_distribution_similarity) / len(edge_length_distribution_similarity), 5)
        results[algorithm][dataset]["Spectral Similarity"] = round(sum(spectral_similarity) / len(spectral_similarity), 5)
        print(f"Computation finished for {dataset}")
        hausdorff_distance = []
        edge_length_distribution_similarity = []
        spectral_similarity = []

print("---------------------------------------------------------------------------------------------------------------")
print("Results:\n")
pprint(results)
