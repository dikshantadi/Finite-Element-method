import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# Define the problem and grid
L = 1.0  # Length of the square domain
N = 10   # Number of nodes in each direction
dx = L / (N - 1)  # Grid spacing

# Generate nodes
x_nodes = np.linspace(0, L, N)
y_nodes = np.linspace(0, L, N)
nodes = np.array([[x, y] for y in y_nodes for x in x_nodes])

# Generate elements (triangular elements)
elements = np.array([[i, i + 1, i + N + 1] for i in range(0, (N - 1) * N, N - 1)])
elements = np.append(elements, np.array([[i, i + N + 1, i + N] for i in range(0, (N - 1) * N, N - 1)]), axis=0)

# Assemble the stiffness matrix and load vector
num_nodes = N * N
K = lil_matrix((num_nodes, num_nodes))
F = np.zeros(num_nodes)

# Define the source term
f = 2.0

# Assemble the stiffness matrix and load vector
for element in elements:
    x, y = nodes[element].T
    area = 0.5 * np.abs(x[0] * (y[1] - y[2]) + x[1] * (y[2] - y[0]) + x[2] * (y[0] - y[1]))
    b = np.array([[y[1] - y[2], y[2] - y[0], y[0] - y[1]],
                  [x[2] - x[1], x[0] - x[2], x[1] - x[0]]]) / (2 * area)
    
    C = np.dot(b.T, b) * area
    for i in range(3):
        for j in range(3):
            K[element[i], element[j]] += C[i, j]

    F[element] += area * f / 3  # Assuming f is constant

# Apply Dirichlet boundary conditions
boundary_nodes = np.arange(0, N)  # Nodes on the bottom edge
K[boundary_nodes, :] = 0
K[:, boundary_nodes] = 0
K[boundary_nodes, boundary_nodes] = 1
F[boundary_nodes] = 0

# Solve the linear system
U = spsolve(csr_matrix(K), F)

# Visualize the solution
plt.triplot(nodes[:, 0], nodes[:, 1], elements)
plt.tripcolor(nodes[:, 0], nodes[:, 1], elements, U, shading='flat', cmap='viridis')
plt.colorbar(label='Solution (u)')
plt.title('Poisson Equation Solution (FEM)')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
