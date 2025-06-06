import numpy as np
import matplotlib.pyplot as plt

# Gravitational constant
G = 10

# Initial conditions (same as original so.py)
A = (100, -10, 10, -11, -3, 0, 0)  # mass, x, y, z, vx, vy, vz
B = (100, 0, 0, 0, 3, 0, 0)
C = (0, 10, 14, 12, 3, 0, 0)
bodies = np.array([A, B, C])

m = bodies[:, 0]  # Masses
x0 = bodies[:, 1:4]  # Initial positions
v0 = bodies[:, 4:7]  # Initial velocities


# Acceleration function (vectorized, from original so.py)
def acc(x):
    dx = x[None, :, :] - x[:, None, :]  # Position differences: x[j] - x[i]
    r3 = np.sum(dx**2, axis=2) ** 1.5  # Distance cubed
    return -G * np.sum(m[:, None, None] * dx / (1e-40 + r3[:, :, None]), axis=0)


# Energy calculation functions (from original so.py)
def kineticEnergy(m, v):
    imp = 0.5 * m * np.sum(v**2, axis=1).T
    return np.sum(imp.T, axis=0)


def potentialEnergy(m1, x1, m2, x2):
    dx = x2 - x1
    r = np.sum(dx**2, axis=0) ** 0.5
    return -G * m1 * m2 / r


def totalEnergy(m, x, v):
    pot = 0
    for i in range(len(m)):
        for j in range(i):
            pot += potentialEnergy(m[i], x[i], m[j], x[j])
    return pot + kineticEnergy(m, v)


# Simulation parameters to match solvers.py
h = 0.02  # Step size
simulation_length = 150  # Total time
num_steps = int(simulation_length / h)  # 7500 steps

# Initialize arrays for RK4 (3 objects, 3 dimensions, 7500 steps)
x_RK4 = np.zeros([3, 3, num_steps], dtype=np.float64)
v_RK4 = np.zeros([3, 3, num_steps], dtype=np.float64)

# RK4 integration loop
xi = x0  # Initial positions
vi = v0  # Initial velocities
for i in range(num_steps):
    # RK4 coefficients
    k1x = h * vi
    k1v = h * acc(xi)
    k2x = h * (vi + 0.5 * k1v)
    k2v = h * acc(xi + 0.5 * k1x)
    k3x = h * (vi + 0.5 * k2v)
    k3v = h * acc(xi + 0.5 * k2x)
    k4x = h * (vi + k3v)
    k4v = h * acc(xi + k3x)

    # Update positions and velocities
    xi = xi + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
    vi = vi + (k1v + 2 * k2v + 2 * k3v + k4v) / 6

    # Store results
    x_RK4[:, :, i] = xi
    v_RK4[:, :, i] = vi

# Calculate total energy
E_RK4 = totalEnergy(m, x_RK4, v_RK4)

# Output shapes
print("x_RK4 shape:", x_RK4.shape)  # (3, 3, 7500)
print("v_RK4 shape:", v_RK4.shape)  # (3, 3, 7500)

t_RK4 = np.linspace(0, simulation_length, num_steps)
plt.plot(t_RK4, E_RK4, label="RK4")
plt.title("Total Energy vs Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Total Energy (Joules)")
plt.legend()
plt.grid()
plt.show()
plt.close()

# class RK4Solver(Solver):
#     def __init__(self, system: System):
#         super().__init__(system)

#     def solve_step(self):
#         h = self.system.h
#         # Extract initial positions and velocities from system objects
#         xi = np.array([obj.position for obj in self.system.objs])
#         vi = np.array([obj.velocity for obj in self.system.objs])
#         masses = np.array([obj.mass for obj in self.system.objs])

#         # Define the acceleration function, vectorized as in so.py
#         def acc(x):
#             dx = x[None, :, :] - x[:, None, :]  # Position differences: x[j] - x[i]
#             r3 = np.sum(dx**2, axis=2) ** 1.5  # Distance cubed
#             # Compute accelerations with a small constant to avoid division by zero
#             return -g * np.sum(
#                 masses[:, None, None] * dx / (1e-40 + r3[:, :, None]), axis=0
#             )

#         # Compute RK4 increments, matching the structure in so.py
#         k1x = h * vi
#         k1v = h * acc(xi)
#         k2x = h * (vi + 0.5 * k1v)
#         k2v = h * acc(xi + 0.5 * k1x)
#         k3x = h * (vi + 0.5 * k2v)
#         k3v = h * acc(xi + 0.5 * k2x)
#         k4x = h * (vi + k3v)
#         k4v = h * acc(xi + k3x)

#         # Update positions and velocities with the weighted average
#         xi_new = xi + (k1x + 2 * k2x + 2 * k3x + k4x) / 6
#         vi_new = vi + (k1v + 2 * k2v + 2 * k3v + k4v) / 6

#         # Update the system objects with new positions and velocities
#         for i, obj in enumerate(self.system.objs):
#             obj.position = xi_new[i]
#             obj.velocity = vi_new[i]
#             self.system.positions[i].append(np.copy(obj.position))
#             self.system.velocities[i].append(np.copy(obj.velocity))

#         # Update accelerations for consistency with the system state
#         new_acc = acc(xi_new)
#         for i, obj in enumerate(self.system.objs):
#             obj.acceleration = new_acc[i]
