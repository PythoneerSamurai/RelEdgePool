# RelEdgePool

Official Implementation of **RelEdgePool: Relational Edge-Aware Pooling for n-gonal manifold and non-manifold 3D meshes**

---

## Overview

**RelEdgePool** is a novel, deterministic, and efficient mesh pooling algorithm designed specifically for deep learning and geometric processing on 3D meshes of arbitrary topology—including n-gonal, manifold, and non-manifold meshes. Unlike conventional approaches that are limited to triangular meshes or require extensive pre-processing (triangulation or remeshing), RelEdgePool operates natively on diverse mesh structures, preserving geometric resemblance while efficiently reducing mesh complexity.

### Key Features

- **Edge-Based Vertex Averaging**: Differentiable pooling via local edge-based vertex aggregation.
- **Vertex Relational Clustering**: A novel, non-differentiable structural step for edge re-establishment and mesh connectivity reconstruction.
- **Broad Applicability**: Direct support for n-gonal (triangular, quadrilateral, etc.), manifold, and non-manifold meshes—even those with irregular connectivity.
- **Determinism**: Consistent results for identical inputs, reducing variability in gradient flow and improving training stability.
- **Efficient Vertex Reduction**: Removes approximately ~72% of vertices per pooling iteration for triangular meshes, and 50% for quadrilateral meshes.
- **PyTorch Implementation**: Optimized for GPU acceleration and easy integration into deep learning pipelines.

---

## Algorithmic Highlights

The RelEdgePool workflow is comprised of two main stages:
1. **Edge-Based Vertex Averaging**: Clusters vertices based on edge connectivity and computes their centroids, thus reducing the vertex set while maintaining local geometry.
2. **Vertex Relational Clustering**: Reconstructs the mesh's edge structure by analyzing direct and indirect relationships between clusters, ensuring valid and consistent connectivity in the pooled mesh.

The algorithm is highly generalizable, capable of processing a wide range of mesh topologies, including those that are highly irregular or non-manifold. It enables hierarchical feature extraction and can pool meshes down to a single vertex for applications such as shape classification or segmentation.

---

## Experimental Results

- **Generalizability**: Successfully pools both manifold and non-manifold, triangular, quadrilateral, and higher n-gonal meshes.
- **Performance**: Outperforms existing mesh pooling algorithms in terms of determinism, vertex reduction, and geometric preservation.
- **Robustness**: Handles challenging scenarios such as meshes with regions of no thickness, shared edges among multiple faces, interior faces, and inconsistent normals.

Please refer to the [paper](https://doi.org/10.22541/au.175856984.48726132/v1) for detailed results, ablation studies, and benchmarks.

---

## Getting Started

### Prerequisites

The implementation is based on **Python** and **PyTorch**. The following packages are required for the totality of this repository:

- `torch`
- `numpy`
- `matplotlib`
- `scipy`
- `pytorch3d`
- `point_cloud_utils`

---

## Usage

The core algorithm is implemented in two main Python files:

- `code_implementation/edge_based_vertex_averaging.py`: Implements the edge-based vertex averaging pooling step.
- `code_implementation/vertex_relational_clustering.py`: Implements the vertex relational clustering algorithm for edge re-establishment.

You may use these modules directly in your mesh processing or neural network pipeline. Refer to the [notebook](notebooks_and_scripts/model_integration_demo/pipeline.ipynb) for an in-depth integration guideline.

---

## Citation

If you use this code or ideas from RelEdgePool in your research, please cite:

```
@online{rashid2025reledgepool,
  author       = {Haroon Rashid},
  title        = {RelEdgePool: Relational Edge-Aware Pooling for n-gonal manifold and non-manifold 3D meshes},
  date         = {2025-09-22},
  doi          = {10.22541/au.175856984.48726132/v1},
  publisher    = {Authorea},
  note         = {Preprint}
}

```

---

## Contact

For questions or collaboration:

- **Author:** Haroon Rashid  
- **Email:** [haroonrashidmcvd@proton.me](mailto:haroonrashidmcvd@proton.me)
- **ORCID:** [0009-0002-1150-3829](https://orcid.org/0009-0002-1150-3829)

---

## License

See [`LICENSE`](LICENSE) file for details.

---

## Acknowledgements

This repository accompanies the paper _RelEdgePool: Relational Edge-Aware Pooling for n-gonal manifold and non-manifold 3D meshes_.  
For more details, algorithmic explanations, and experimental results, please refer to the [paper](https://doi.org/10.22541/au.175856984.48726132/v1).
