import numpy as np
import matplotlib.pyplot as plt

# Define the problem
L = 1.0  # Length of the domain
num_elements = 5  # Number of elements
num_nodes = num_elements + 1  # Number of nodes

# Generate nodes
nodes = np.linspace(0, L, num_nodes)

# Generate elements (connectivity)
elements = np.column_stack((np.arange(num_elements), np.arange(1, num_nodes)))

# Visualize the mesh
plt.plot(nodes, np.zeros(num_nodes), 'o-', label='Nodes')
for element in elements:
    plt.plot(nodes[element], [0, 0], 'k-')
plt.title('Mesh')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.legend()
plt.show()

# Define the material properties
E = 1.0  # Young's modulus
A = 1.0  # Cross-sectional area

# Define the Dirichlet boundary conditions
u_left = 0.0
u_right = 1.0

# Assemble the stiffness matrix and load vector
K = np.zeros((num_nodes, num_nodes))
F = np.zeros(num_nodes)

for element in elements:
    length = nodes[element[1]] - nodes[element[0]]
    ke = E * A / length * np.array([[1, -1], [-1, 1]])  # Element stiffness matrix

    K[np.ix_(element, element)] += ke

# Apply Dirichlet boundary conditions
K[0, :] = 0
K[:, 0] = 0
K[0, 0] = 1
F[0] = u_left

K[-1, :] = 0
K[:, -1] = 0
K[-1, -1] = 1
F[-1] = u_right

# Solve the linear system
U = np.linalg.solve(K, F)

# Visualize the solution
plt.plot(nodes, U, 'o-', label='Numerical Solution')
plt.title('Poisson Equation Solution')
plt.xlabel('X-axis')
plt.ylabel('Solution')
plt.legend()
plt.show()
