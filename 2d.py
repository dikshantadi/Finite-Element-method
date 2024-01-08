import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

# Define the domain
Lx, Ly = 1.0, 1.0  # Dimensions of the domain
num_nodes_x, num_nodes_y = 5, 5  # Number of nodes along x and y axes

# Generate nodes
x = np.linspace(0, Lx, num_nodes_x)
y = np.linspace(0, Ly, num_nodes_y)
x, y = np.meshgrid(x, y)
nodes = np.column_stack((x.flatten(), y.flatten()))

# Create Delaunay triangulation
triangulation = Delaunay(nodes)

# Visualize the mesh
plt.triplot(nodes[:, 0], nodes[:, 1], triangulation.simplices)
plt.plot(nodes[:, 0], nodes[:, 1], 'o')
plt.title('Mesh')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()

# Define the material properties
E = 1.0  # Young's modulus
nu = 0.3  # Poisson's ratio

# Define the function for the right-hand side (RHS) of the Poisson equation
def rhs_function(x, y):
    return 1.0  # You can modify this function as needed

# Assemble the stiffness matrix and load vector
num_nodes = nodes.shape[0]
K = np.zeros((num_nodes, num_nodes))
F = np.zeros(num_nodes)

for simplex in triangulation.simplices:
    vertices = nodes[simplex]
    area = np.abs(np.linalg.det(np.column_stack((vertices[1] - vertices[0], vertices[2] - vertices[0])))) / 2.0

    # Compute the element stiffness matrix and load vector
    D = E / (1 - nu**2) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1 - nu) / 2]])
    B = np.linalg.inv(np.column_stack((vertices[1] - vertices[0], vertices[2] - vertices[0])))

    element_stiffness = area * np.dot(np.dot(B.T, D), B)
    element_load = area / 3.0 * np.array([rhs_function(*vertices[0]), rhs_function(*vertices[1]), rhs_function(*vertices[2])])

    # Assemble the global stiffness matrix and load vector
    for i in range(3):
        for j in range(3):
            K[simplex[i], simplex[j]] += element_stiffness[i, j]
        F[simplex[i]] += element_load[i]

# Apply Dirichlet boundary conditions (for simplicity, fixing the left and bottom nodes)
fixed_nodes = [i for i in range(num_nodes) if nodes[i, 0] == 0.0 or nodes[i, 1] == 0.0]
for i in fixed_nodes:
    K[i, :] = 0
    K[:, i] = 0
    K[i, i] = 1
    F[i] = 0

# Solve the linear system
U = np.linalg.solve(K, F)

# Visualize the solution
plt.tripcolor(nodes[:, 0], nodes[:, 1], triangulation.simplices, U, shading='flat', cmap=plt.cm.viridis)
plt.colorbar(label='Solution')
plt.title('Solution to Poisson Equation')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()
