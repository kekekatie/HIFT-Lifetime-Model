#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""HIFT-Engine (toy, NumPy-only) — two-pass cut-and-project from an E8 lattice sample
and seam identification for trinary (3,5,8) phases.

Run:
    python3 hift_engine_np.py
Outputs:
    - e8_trinary_dataset.npz
    - P3_labels_label*.png, P3_seams.png, P3_triples.png
"""

import os, json, itertools, time
from dataclasses import dataclass
import numpy as np
from numpy.linalg import norm, qr
import matplotlib.pyplot as plt

outdir = os.path.dirname(__file__) if os.path.dirname(__file__) else "."

def random_orthonormal_matrix(n, seed=12345):
    rng = np.random.default_rng(seed)
    A = rng.normal(size=(n, n))
    Q, R = np.linalg.qr(A)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q

@dataclass
class E8Params:
    M_int: int = 2
    R8_int: float = 2.6
    N_half_samples: int = 120_000
    R8_half: float = 2.6
    target_half: int = 4000

def generate_e8_sample(params: E8Params, seed=20250902):
    rng = np.random.default_rng(seed)
    pts = []
    coords = range(-params.M_int, params.M_int+1)
    R2_int = params.R8_int**2
    for tup in itertools.product(coords, repeat=8):
        if (sum(tup) % 2) != 0:
            continue
        v = np.array(tup, dtype=float)
        if v.dot(v) <= R2_int + 1e-12:
            pts.append(v)
    R2_half = params.R8_half**2
    low = -params.M_int - 1
    high = params.M_int
    keep_half = 0
    for _ in range(params.N_half_samples):
        n = rng.integers(low, high+1, size=8)
        if (np.sum(n) % 2) != 0:
            continue
        v = n.astype(float) + 0.5
        if v.dot(v) <= R2_half + 1e-12:
            pts.append(v)
            keep_half += 1
            if keep_half >= params.target_half:
                break
    return np.array(pts, dtype=float)

@dataclass
class ProjectionParams:
    seed_rot: int = 314159
    window_radius: float = 0.52

def two_pass_project(E8_pts: np.ndarray, proj_params: ProjectionParams):
    Q = random_orthonormal_matrix(8, seed=proj_params.seed_rot)
    U_phys = Q[:3,:]
    U_int  = Q[3:,:]
    P3 = (U_phys @ E8_pts.T).T
    I5 = (U_int  @ E8_pts.T).T
    r = np.linalg.norm(I5, axis=1)
    mask = r <= proj_params.window_radius
    return P3[mask], I5[mask], mask

@dataclass
class PhaseParams:
    n3: int = 3
    n5: int = 5
    n8: int = 8
    seed_planes: int = 271828

def random_plane_basis_5D(seed, k):
    rng = np.random.default_rng(seed + k*101)
    A = rng.normal(size=(5,2))
    v1 = A[:,0]; v1 = v1 / norm(v1)
    v2 = A[:,1] - (A[:,1] @ v1) * v1; v2 = v2 / norm(v2)
    return v1, v2

def plane_angle(v1, v2, X):
    x1 = X @ v1; x2 = X @ v2
    return np.arctan2(x2, x1)

def residuals_to_nearest_minima(theta, n):
    x = (theta + np.pi) % (2*np.pi) - np.pi
    step = 2*np.pi / n
    nearest = np.round(x / step) * step
    res = np.abs(x - nearest)
    return np.minimum(res, step - res)

def compute_phase_labels(I5: np.ndarray, phase_params: PhaseParams):
    v1_3, v2_3 = random_plane_basis_5D(phase_params.seed_planes, 3)
    v1_5, v2_5 = random_plane_basis_5D(phase_params.seed_planes, 5)
    v1_8, v2_8 = random_plane_basis_5D(phase_params.seed_planes, 8)
    th3 = plane_angle(v1_3, v2_3, I5)
    th5 = plane_angle(v1_5, v2_5, I5)
    th8 = plane_angle(v1_8, v2_8, I5)
    r3 = residuals_to_nearest_minima(th3, 3)
    r5 = residuals_to_nearest_minima(th5, 5)
    r8 = residuals_to_nearest_minima(th8, 8)
    R = np.stack([r3, r5, r8], axis=1)
    labels = np.argmin(R, axis=1)
    return labels

@dataclass
class SeamParams:
    k_neighbors: int = 8
    seam_min_labels: int = 2
    triple_min_labels: int = 3

def knn_bruteforce(P3: np.ndarray, k: int):
    N = P3.shape[0]
    D2 = np.sum(P3**2, axis=1, keepdims=True) + np.sum(P3**2, axis=1) - 2*(P3 @ P3.T)
    np.fill_diagonal(D2, np.inf)
    idxs = np.argpartition(D2, kth=k-1, axis=1)[:, :k]
    row_indices = np.arange(N)[:, None]
    ordered = idxs[row_indices, np.argsort(D2[row_indices, idxs])]
    return ordered

def identify_seams(P3: np.ndarray, labels: np.ndarray, seam_params: SeamParams):
    nn_idx = knn_bruteforce(P3, k=seam_params.k_neighbors)
    neighbor_labels = labels[nn_idx]
    unique_counts = np.array([len(set(row.tolist())) for row in neighbor_labels])
    seam_points = unique_counts >= seam_params.seam_min_labels
    triple_points = unique_counts >= seam_params.triple_min_labels
    return seam_points, triple_points

def scatter3_group(P3, mask, title, fname):
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(projection='3d')
    X = P3[mask]
    ax.scatter(X[:,0], X[:,1], X[:,2], s=4, alpha=0.9)
    ax.set_title(title)
    ax.set_xlabel('x'); ax.set_ylabel('y'); ax.set_zlabel('z')
    plt.tight_layout()
    fig.savefig(os.path.join(outdir, fname), dpi=160)
    plt.close(fig)

def scatter3_labels(P3, labels, title, fname_prefix):
    for lab in range(3):
        mask = (labels == lab)
        scatter3_group(P3, mask, f"{title} — label {lab}", f"{fname_prefix}_label{lab}.png")

def main():
    e8p = E8Params()
    E8 = generate_e8_sample(e8p, seed=20250902)
    projp = ProjectionParams()
    P3, I5, mask = two_pass_project(E8, projp)
    phasep = PhaseParams()
    labels = compute_phase_labels(I5, phasep)
    seamp = SeamParams()
    seam_points, triple_points = identify_seams(P3, labels, seamp)

    meta = dict(
        E8_count=int(E8.shape[0]), kept_count=int(P3.shape[0]),
        params=dict(E8Params=e8p.__dict__, ProjectionParams=projp.__dict__,
                    PhaseParams=phasep.__dict__, SeamParams=seamp.__dict__)
    )
    np.savez_compressed(os.path.join(outdir, "e8_trinary_dataset.npz"),
                        P3=P3, I5=I5, labels=labels.astype(np.int16),
                        seam_points=seam_points.astype(np.bool_),
                        triple_points=triple_points.astype(np.bool_),
                        meta=json.dumps(meta))
    scatter3_labels(P3, labels, "Projected 3D point cloud (labels)", "P3_labels")
    scatter3_group(P3, seam_points, "Seam points (≥2 neighbor labels)", "P3_seams.png")
    scatter3_group(P3, triple_points, "Triple-junction points (all 3 labels)", "P3_triples.png")
    print("Done. Files saved in", outdir)

if __name__ == "__main__":
    main()
