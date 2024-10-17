import numpy as np

gamma = 0.95
epsilon = 1e-3
Ny = 50

def get_theta():
    # Generate the discretization
    atoms = np.concatenate(([0], np.logspace(-2, 0, Ny)))
    # Calculate theta: the ratio between consecutive points (log-spacing)
    theta = atoms[2] / atoms[1]  # Avoiding the zero element at index 0
    return theta

theta = get_theta()
error_bound = gamma / (1 - gamma) * ((theta - 1) + epsilon)
print(f"Error bound: {error_bound}")