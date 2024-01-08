import numpy as np
import matplotlib.pyplot as plt

# Define the problem and grid
L = 1.0  # Length of the square domain
N = 50   # Number of grid points in each direction
dx = L / (N - 1)  # Grid spacing

# Initialize the solution matrix
u = np.zeros((N, N))

# Set boundary conditions (u = 0 on all boundaries)
u[:, 0] = 0
u[:, -1] = 0
u[0, :] = 0
u[-1, :] = 0

# Define the source term
f = 2.0

# Perform the Finite Difference Method iteration
for it in range(1000):  # Adjust the number of iterations as needed
    u_new = np.copy(u)  # Create a copy to avoid modifying values during iteration
    for i in range(1, N-1):
        for j in range(1, N-1):
            u_new[i, j] = 0.25 * (u[i+1, j] + u[i-1, j] + u[i, j+1] + u[i, j-1] + dx**2 * f)

    # Check for convergence (adjust the tolerance as needed)
    if np.linalg.norm(u_new - u) < 1e-5:
        break

    u = np.copy(u_new)

# Plot the solution
x = np.linspace(0, L, N)
y = np.linspace(0, L, N)
X, Y = np.meshgrid(x, y)

plt.contourf(X, Y, u, cmap='viridis', levels=20)
plt.colorbar(label='Solution (u)')

# Plot grid lines
plt.axhline(0, color='black', linewidth=0.5)
plt.axhline(L, color='black', linewidth=0.5)
plt.axvline(0, color='black', linewidth=0.5)
plt.axvline(L, color='black', linewidth=0.5)

plt.title('Poisson Equation Solution with Grid Lines')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
