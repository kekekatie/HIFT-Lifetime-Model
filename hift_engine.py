from dataclasses import dataclass
import numpy as np

def generate_e8_points(n_points: int = 8000, coord_limit: int = 2, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = []
    target = n_points
    batch = max(2000, n_points // 2)
    half = np.full(8, 0.5)
    while len(pts) < target:
        Z = rng.integers(-coord_limit, coord_limit + 1, size=(batch, 8))
        parity = (Z.sum(axis=1) % 2 == 0)
        D8 = Z[parity]
        S = D8 + half
        pts.append(D8)
        pts.append(S)
    P = np.vstack(pts)[:target]
    return P.astype(float)

def projection_basis_symmetry(s: int, phase: float = 0.0) -> np.ndarray:
    k = np.arange(8, dtype=float)
    theta = (2.0 * np.pi * k / float(s)) + (phase / float(max(1, s)))
    r1 = np.cos(theta)
    r2 = np.sin(theta)
    r3 = np.cos(2.0 * theta + phase)
    R = np.vstack([r1, r2, r3])
    R = R / np.linalg.norm(R, axis=1, keepdims=True)
    return R

def project_points(X8: np.ndarray, B3x8: np.ndarray) -> np.ndarray:
    return X8 @ B3x8.T

def radial_filter_xy(P3: np.ndarray, rmax: float = 18.0) -> np.ndarray:
    r2 = np.sum(P3[:, :2]**2, axis=1)
    return P3[r2 <= rmax**2]

def angle_xy(p3: np.ndarray) -> np.ndarray:
    return np.arctan2(p3[:,1], p3[:,0])

def default_triple_wedge():
    import numpy as np
    return np.array([-np.pi, -np.pi/3.0, np.pi/3.0], dtype=float)

def assign_wedge_labels(P: np.ndarray, boundaries: np.ndarray) -> np.ndarray:
    θ = angle_xy(P)
    θ0, θ1, θ2 = boundaries
    labels = np.empty(P.shape[0], dtype=int)
    in0 = (θ >= θ0) & (θ < θ1)
    in1 = (θ >= θ1) & (θ < θ2)
    labels[in0] = 0
    labels[in1] = 1
    labels[~(in0 | in1)] = 2
    return labels

def carve_domains(P3_s, boundaries):
    domain_points = {}
    mapping = {0:3, 1:5, 2:8}
    for domain_idx, s in mapping.items():
        P = P3_s[s]
        wedge_labels = assign_wedge_labels(P, boundaries)
        domain_points[domain_idx] = P[wedge_labels == domain_idx]
    return domain_points

def voxelize_and_characterize(domains, voxels=32):
    labels = sorted(domains.keys())
    allP = np.vstack([domains[l] for l in labels if domains[l].size])
    xmin, ymin, zmin = allP.min(axis=0)
    xmax, ymax, zmax = allP.max(axis=0)
    pad = 1e-6
    bounds = (xmin - pad, xmax + pad, ymin - pad, ymax + pad, zmin - pad, zmax + pad)
    nx = ny = nz = int(voxels)
    dx = (bounds[1] - bounds[0]) / nx
    dy = (bounds[3] - bounds[2]) / ny
    dz = (bounds[5] - bounds[4]) / nz

    def index_coords(P):
        ix = np.clip(((P[:,0] - bounds[0]) / dx).astype(int), 0, nx-1)
        iy = np.clip(((P[:,1] - bounds[2]) / dy).astype(int), 0, ny-1)
        iz = np.clip(((P[:,2] - bounds[4]) / dz).astype(int), 0, nz-1)
        return ix, iy, iz

    voxel_counts = {l: np.zeros((nx,ny,nz), dtype=int) for l in labels}
    for l in labels:
        P = domains[l]
        if P.size == 0:
            continue
        ix, iy, iz = index_coords(P)
        for a,b,c in zip(ix,iy,iz):
            voxel_counts[l][a,b,c] += 1

    occ = np.zeros((nx,ny,nz), dtype=int)
    for l in labels:
        occ += (voxel_counts[l] > 0).astype(int)

    seam_mask = (occ >= 2)
    triple_mask = (occ == 3)

    triple_coords = np.argwhere(triple_mask)
    centers = []
    for (i,j,k) in triple_coords:
        cx = bounds[0] + (i + 0.5) * dx
        cy = bounds[2] + (j + 0.5) * dy
        cz = bounds[4] + (k + 0.5) * dz
        centers.append((cx,cy,cz))
    centers = np.array(centers) if len(centers) else np.empty((0,3))

    return dict(
        bounds=bounds,
        voxel_size=(dx,dy,dz),
        voxel_counts=voxel_counts,
        seam_mask=seam_mask,
        triple_mask=triple_mask,
        triple_voxel_centers=centers,
        occ=occ
    )
