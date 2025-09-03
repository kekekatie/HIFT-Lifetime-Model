import numpy as np
import matplotlib.pyplot as plt
from hift_engine import (
    generate_e8_points, projection_basis_symmetry, project_points, radial_filter_xy,
    default_triple_wedge, carve_domains, voxelize_and_characterize
)

X8 = generate_e8_points(n_points=9000, coord_limit=2, seed=7)

B3 = projection_basis_symmetry(3, phase=0.15)
B5 = projection_basis_symmetry(5, phase=0.35)
B8 = projection_basis_symmetry(8, phase=0.55)

P3 = {
    3: radial_filter_xy(project_points(X8, B3), rmax=18.0),
    5: radial_filter_xy(project_points(X8, B5), rmax=18.0),
    8: radial_filter_xy(project_points(X8, B8), rmax=18.0),
}

boundaries = default_triple_wedge()
domains = carve_domains(P3, boundaries)
summary = voxelize_and_characterize(domains, voxels=32)

print("Counts:", {k: domains[k].shape[0] for k in domains})
print("Bi-seam voxels:", int(summary["seam_mask"].sum()))
print("Triple-junction voxels:", int(summary["triple_mask"].sum()))

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(domains[0][:,0], domains[0][:,1], domains[0][:,2], s=3, marker='o')
ax.set_title("Domain 0 (C3)")
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(domains[1][:,0], domains[1][:,1], domains[1][:,2], s=3, marker='^')
ax.set_title("Domain 1 (C5)")
plt.tight_layout()
plt.show()

fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(domains[2][:,0], domains[2][:,1], domains[2][:,2], s=3, marker='s')
ax.set_title("Domain 2 (C8)")
plt.tight_layout()
plt.show()

tvc = summary["triple_voxel_centers"]
fig = plt.figure(figsize=(6,5))
ax = fig.add_subplot(111, projection='3d')
if tvc.size:
    ax.scatter(tvc[:,0], tvc[:,1], tvc[:,2], s=12, marker='x')
ax.set_title("Triple-junction voxel centers")
plt.tight_layout()
plt.show()
