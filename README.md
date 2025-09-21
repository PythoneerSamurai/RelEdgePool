# RelEdgePool: Relational Edge-Aware Pooling for n-gonal manifold and non-manifold 3D meshes

RelEdgePool is a novel deterministic algorithm for pooling 3D meshes that operates natively on n-gonal manifold and non-manifold meshes without requiring pre-processing steps like triangulation or remeshing. The algorithm combines edge-based vertex averaging for pooling with a novel "vertex relational clustering" approach for re-establishing connectivity in pooled meshes.

![Teaser](repository_images/human_collage.jpeg)
*Progressive reduction in mesh complexity while maintaining geometric similarity using RelEdgePool*

## Key Features

- **Native Support for Diverse Mesh Types**: Works directly with:
  - N-gonal manifold meshes
  - Non-manifold meshes
  - Meshes with irregular connectivities
  - No pre-processing required

- **Vertex Removal**:
  - Removes ~72% vertices in a single pooling iteration for triangular meshes
  - Removes ~50% vertices in a single pooling iteration for quadrilateral meshes
  - Maintains topological resemblance between input and pooled meshes

- **Deterministic Operation**:
  - Ensures consistent outputs during training
  - Reduces variability in gradient flow
  - Improves optimization efficiency

- **Deep Learning Integration**:
  - Fully compatible with deep learning workflows
  - GPU-accelerated implementation using PyTorch
  - Supports end-to-end training

## Implementation Components

### 1. Edge-Based Vertex Averaging
- Core pooling functionality
- Performs dimensionality reduction through local vertex cluster averaging
- Fully differentiable for gradient flow during training

### 2. Vertex Relational Clustering
- Novel approach for edge re-establishment
- Comprises direct and indirect edge detection mechanisms
- Non-differentiable structural operation for mesh reconstruction

## Usage

```python
# Example code will be added with implementation
```

## Results

RelEdgePool has been extensively tested and compared with existing mesh pooling techniques across multiple parameters:

1. **Generalizability**: Successfully tested on:
   - Triangular meshes
   - Quadrilateral meshes
   - Non-manifold meshes with:
     - Regions having no thickness
     - Shared edges between multiple faces
     - Interior faces
     - Inconsistent face normals

2. **Performance Metrics**:
   - Hausdorff Distance
   - Edge Length Distribution Similarity
   - Spectral Similarity

## Requirements

- PyTorch
- PyTorch3D

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@article{rashid2025reledgepool,
    title={RelEdgePool: Relational Edge-Aware Pooling for n-gonal manifold and non-manifold 3D meshes},
    author={Haroon Rashid},
    year={2025}
}
```

## License

[MIT License](LICENSE)

## Contact

Haroon Rashid - haroonrashidmcvd@proton.me
