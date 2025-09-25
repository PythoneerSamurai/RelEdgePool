import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from pytorch3d.io import IO

current_dir = Path(__file__).parent
root_dir = current_dir.parent.parent
components_dir = root_dir / "components"
sys.path.append(str(components_dir))

from edge_based_vertex_averaging import EdgeBasedVertexAveraging


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = x_limits[1] - x_limits[0]
    y_range = y_limits[1] - y_limits[0]
    z_range = z_limits[1] - z_limits[0]

    max_range = max(x_range, y_range, z_range)

    x_middle = np.mean(x_limits)
    y_middle = np.mean(y_limits)
    z_middle = np.mean(z_limits)

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


def plot_mesh(vertices: np.ndarray, edges: np.ndarray, title: str, show: bool = False, save_path: str = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    vertices = vertices[:, [0, 2, 1]]
    vertices[:, 1] *= -1

    ax.scatter(vertices[:, 0], vertices[:, 1], vertices[:, 2], c='red', s=20, label='Vertices')
    ax.legend(loc="upper right")
    for start, end in edges:
        x = [vertices[start, 0], vertices[end, 0]]
        y = [vertices[start, 1], vertices[end, 1]]
        z = [vertices[start, 2], vertices[end, 2]]
        ax.plot(x, y, z, color='blue', linewidth=1.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title(title)
    plt.tight_layout()

    set_axes_equal(ax)

    if save_path:
        plt.savefig(save_path)
    if show:
        plt.show()
    plt.close(fig)


def load_mesh(mesh_path: str, topology: str) -> tuple:
    if "quadrilateral" in topology.lower() or "non-manifold" in topology.lower():
        vertices = []
        faces = []
        edges_set = set()

        with open(mesh_path, "r") as f:
            for line in f:
                if line.startswith("v "):  # Vertex
                    _, x, y, z = line.strip().split()
                    vertices.append([float(x), float(y), float(z)])
                elif line.startswith("f "):  # Face
                    face = [int(part.split('/')[0]) - 1 for part in line.strip().split()[1:]]
                    faces.append(face)

        for face in faces:
            n = len(face)
            for i in range(n):
                v0, v1 = face[i], face[(i + 1) % n]
                edges_set.add(tuple(sorted((v0, v1))))

        vertices = torch.tensor(vertices, dtype=torch.float32)
        edges = torch.tensor(list(edges_set), dtype=torch.int64)

        return vertices, edges
    else:
        mesh = IO().load_mesh(mesh_path, include_textures=False)
        vertices = mesh.verts_packed()
        vertices.requires_grad = True
        edges = mesh.edges_packed()
        return vertices, edges


def process_datasets() -> None:
    meshes_dir = root_dir / "datasets/meshes"
    if not os.path.exists(meshes_dir):
        print("meshes directory not found. Execute get_meshes.sh to automatically download and extract the meshes dir.")
    else:
        if not os.path.exists(f"{current_dir}/plots"):
            os.mkdir(f"{current_dir}/plots")
        for sub_dir in os.listdir(meshes_dir):
            sub_dir_path = os.path.join(meshes_dir, sub_dir)
            save_dir_path = f"{current_dir}/plots/{sub_dir}"
            if not os.path.exists(save_dir_path):
                os.mkdir(save_dir_path)
            for mesh in os.listdir(sub_dir_path):
                mesh_path = os.path.join(sub_dir_path, mesh)
                mesh_name = mesh.replace(".obj", "")
                mesh_plots_folder_path = f"{save_dir_path}/{mesh_name}"
                if not os.path.exists(mesh_plots_folder_path):
                    os.mkdir(mesh_plots_folder_path)
                vertices, edges = load_mesh(mesh_path, sub_dir)
                vertices.requires_grad = True
                plot_mesh(
                    vertices.detach().numpy(),
                    edges.numpy(),
                    title=f"Original Mesh, Vertices: {len(vertices)}",
                    save_path=f"{mesh_plots_folder_path}/original_mesh, vertices: {len(vertices)}.png"
                )
                iteration = 1
                print(f"Pooling {mesh}")
                while len(edges) != 0:
                    vertices, edges = EdgeBasedVertexAveraging(vertices, edges).pool()
                    plot_mesh(
                        vertices.detach().numpy(),
                        edges.numpy(),
                        title=f"Pooling Step: {iteration}, Vertices: {len(vertices)}",
                        save_path=f"{mesh_plots_folder_path}/pooling_iteration: {iteration}, vertices: {len(vertices)}.png"
                    )
                    print(f"Iteration {iteration} finished. Vertices: {len(vertices)}, Edges: {len(edges)}")
                    iteration += 1


process_datasets()
