import numpy as np
import matplotlib.pyplot as plt

# Parameters
L = 1.0  # Length of the rod
T = 1.0  # Total simulation time
alpha = 0.01  # Thermal diffusivity
num_elements = 10  # Number of elements
num_nodes = num_elements + 1
dx = L / num_elements
dt = 0.001  # Time step
num_steps = int(T / dt)

# Initialize temperature field
u = np.zeros(num_nodes)

# Initial condition
u[:num_nodes // 2] = 100.0

# Finite Element Method
for step in range(num_steps):
    u_new = np.copy(u)
    for i in range(1, num_nodes - 1):
        # Finite difference approximation for the second spatial derivative
        d2u_dx2 = (u[i + 1] - 2 * u[i] + u[i - 1]) / dx**2
        # Update temperature using the heat equation
        u_new[i] = u[i] + alpha * dt * d2u_dx2

    u = np.copy(u_new)

# Plot the final temperature distribution
x = np.linspace(0, L, num_nodes)
plt.plot(x, u)
plt.xlabel('Position (m)')
plt.ylabel('Temperature')
plt.title('Temperature Distribution in a Rod')
plt.show()
