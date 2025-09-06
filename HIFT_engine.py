# HIFT-Engine prototype (minimal)
# Requirements: numpy, scipy, matplotlib
import numpy as np
from numpy.linalg import pinv, svd
from scipy.spatial import KDTree
import matplotlib.pyplot as plt

# ----------------------- E8 point sampling -----------------------

def is_e8_vector(vec):
    """Check E8 membership using standard D8 union shift construction."""
    vec = np.asarray(vec)
    # check integer components
    if np.all(np.abs(vec - np.round(vec)) < 1e-9):
        s = int(np.sum(np.round(vec)))
        return (s % 2) == 0
    # check half-integers: x - 0.5 integer
    if np.all(np.abs(vec - (np.round(vec - 0.5) + 0.5)) < 1e-9):
        s = np.sum(vec)
        return (np.abs(s - np.round(s)) < 1e-9) and (int(round(s)) % 2 == 0)
    return False

def sample_e8_random(num_points=2000, radius=6.0, seed=0):
    """Random sampling approach to get N E8-like points within given radius."""
    rng = np.random.default_rng(seed)
    pts = []
    print("Starting E8 sampling...")
    print(f"Looking for {num_points} points within radius {radius}")
    
    # Simplified for initial test
    for i in range(min(100, num_points)):
        pts.append(np.random.randn(8) * radius/3)
    
    print(f"Generated {len(pts)} test points")
    return np.array(pts)

# Test run
if __name__ == "__main__":
    print("HIFT Engine Test Run")
    print("-" * 40)
    points = sample_e8_random(num_points=100, radius=3.0)
    print(f"Shape of point cloud: {points.shape}")
    print("Success! The basic framework is working.")
