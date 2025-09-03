# HIFT-Engine 1.0 (Prototype)

This is a minimal, open-source-friendly prototype to **explore** a trinary (C3, C5, C8) quasicrystal
inspired by **E8** projections and **cut-and-project** ideas. It is a *toy engine* meant to support
geometric/topological experimentation and visualization—not a full physical simulator.

## What it does

- Samples points from the **E8 lattice** via the standard construction: `E8 = D8 ∪ (D8 + 1/2)`.
- Projects 8D points to 3D using simple, **symmetry-seeded star maps** for target symmetries: C3, C5, C8.
- **Carves three spatial wedges** so the point clouds meet along **seams** and a **triple junction**.
- Voxelizes space to mark **bi-seam** voxels (≥2 domains present) and **triple-junction** voxels (all 3 present).

## Quick start

```bash
python demo.py
```

This will:
- Generate a small E8 sample
- Project to three 3D clouds
- Carve the wedge domains
- Print counts and seam/junction stats
- Show four basic 3D plots (matplotlib)

## Files

- `hift_engine.py` — library with all core functions, well-commented
- `demo.py` — minimal runnable example using the library
- `example_pointclouds.npz` — saved domain clouds and triple-junction centers (from the notebook run)

## Caveats (important)

- The projection bases are deliberately simple and symmetry-seeded; they **do not** implement a faithful
  physical E8→3D quasicrystal projection. They *do* provide a reproducible geometry for testing seam logic.
- Wedge carving is an explicit, controlled way to create **interfaces**; real grain boundaries are subtler.
- Seam/triple detection uses simple **voxel occupancy**.

## Next steps

- Replace symmetry-seeded star maps with a proper **cut-and-project** from E8 using a chosen physical/internal split.
- Implement **domain rotations** and **misorientation angles** rather than fixed wedges; compare with grain boundary theory.
- Introduce **acceptance windows** in the internal space to obtain cleaner quasicrystalline point sets.
- Add topological analysis: local coordination, defect charge counting, and line-tension estimators.
- Export to PLY/VTK for 3D viewers, and add unit tests.
