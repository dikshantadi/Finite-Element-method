from fenics import *

# Define mesh and function space
num_elements = 5
length = 1.0
mesh = IntervalMesh(num_elements, 0, length)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary conditions
def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, Constant(0), 'near(x[0], 0)')

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
a = inner(grad(u), grad(v)) * dx
L = Constant(1) * v * dx

# Compute solution
u = Function(V)
solve(a == L, u, bc)

# Plot the solution
plot(u)
plt.show()
