import scipy as sci

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

plt.style.use("seaborn-v0_8")

# Define universal constants
G = 6.67408e-11  # N-m^2/kg^2


class Body:
    def __init__(self, mass, position, velocity):
        self.mass = mass
        self.position = np.array(position, dtype="float64")
        self.velocity = np.array(velocity, dtype="float64")


class TwoBodySystem:
    def __init__(self, body1, body2):
        self.body1 = body1
        self.body2 = body2
        self.total_energy = []
        self.total_ke = []
        self.total_gpe = []

    def equations(self, w, t):
        r1, r2 = w[:3], w[3:6]
        v1, v2 = w[6:9], w[9:12]
        r = np.linalg.norm(r2 - r1)
        dv1bydt = G * self.body2.mass * (r2 - r1) / r**3
        dv2bydt = G * self.body1.mass * (r1 - r2) / r**3

        gpe = -((G * self.body1.mass * self.body2.mass) / r)
        ke1 = 0.5 * self.body1.mass * np.sum(v1**2)
        ke2 = 0.5 * self.body2.mass * np.sum(v2**2)

        self.total_energy.append(gpe + ke1 + ke2)
        self.total_ke.append(ke1 + ke2)
        self.total_gpe.append(gpe)
        dr1bydt = v1
        dr2bydt = v2
        return np.concatenate((dr1bydt, dr2bydt, dv1bydt, dv2bydt))

    def solve(self, init_conditions, time_span):
        return sci.integrate.odeint(self.equations, init_conditions, time_span)


class ThreeBodySystem:
    def __init__(self, body1, body2, body3):
        self.body1 = body1
        self.body2 = body2
        self.body3 = body3
        self.total_energy = []
        self.total_ke = []
        self.total_gpe = []

    def equations(self, w, t):
        r1, r2, r3 = w[:3], w[3:6], w[6:9]
        v1, v2, v3 = w[9:12], w[12:15], w[15:18]
        r12 = np.linalg.norm(r2 - r1)
        r13 = np.linalg.norm(r3 - r1)
        r23 = np.linalg.norm(r3 - r2)
        dv1bydt = (
            G * self.body2.mass * (r2 - r1) / r12**3
            + G * self.body3.mass * (r3 - r1) / r13**3
        )
        dv2bydt = (
            G * self.body1.mass * (r1 - r2) / r12**3
            + G * self.body3.mass * (r3 - r2) / r23**3
        )
        dv3bydt = (
            G * self.body1.mass * (r1 - r3) / r13**3
            + G * self.body2.mass * (r2 - r3) / r23**3
        )

        dr1bydt = v1
        dr2bydt = v2
        dr3bydt = v3
        return np.concatenate((dr1bydt, dr2bydt, dr3bydt, dv1bydt, dv2bydt, dv3bydt))

    def solve(self, init_conditions, time_span):
        return sci.integrate.odeint(self.equations, init_conditions, time_span)


# Plotting and animations (unchanged from original code)
fig = plt.figure(figsize=(15, 7.1))
ax1 = fig.add_subplot(121, projection="3d")
ax2 = fig.add_subplot(122, projection="3d")

# Create body instances for the two-body system
body1 = Body(mass=2e30, position=[-1e11, 0, 0], velocity=[0, 1e4, 0])
body2 = Body(mass=6e24, position=[1e11, 0, 0], velocity=[0, -1e4, 0])
two_body_system = TwoBodySystem(body1, body2)

# Solve two-body system
init_conditions = np.concatenate(
    (body1.position, body2.position, body1.velocity, body2.velocity)
)
time_span_2b = np.linspace(0, 3.154e7, 500)  # 1 year in seconds
two_body_sol = two_body_system.solve(init_conditions, time_span_2b)
r1_sol_2b, r2_sol_2b = two_body_sol[:, :3], two_body_sol[:, 3:6]


def animate_two_body(i):
    ax1.cla()
    ax1.set_title("2-Body Problem Animation", fontsize=14)
    ax1.set_xlabel("X Coordinate", fontsize=12)
    ax1.set_ylabel("Y Coordinate", fontsize=12)
    ax1.set_zlabel("Z Coordinate", fontsize=12)
    ax1.plot(r1_sol_2b[:i, 0], r1_sol_2b[:i, 1], r1_sol_2b[:i, 2], color="darkblue")
    ax1.plot(r2_sol_2b[:i, 0], r2_sol_2b[:i, 1], r2_sol_2b[:i, 2], color="tab:red")
    ax1.scatter(
        r1_sol_2b[i - 1, 0],
        r1_sol_2b[i - 1, 1],
        r1_sol_2b[i - 1, 2],
        color="darkblue",
        marker="o",
        s=100,
    )
    ax1.scatter(
        r2_sol_2b[i - 1, 0],
        r2_sol_2b[i - 1, 1],
        r2_sol_2b[i - 1, 2],
        color="tab:red",
        marker="o",
        s=100,
    )
    ax1.set_xlim([-2, 2])
    ax1.set_ylim([-2, 2])
    ax1.set_zlim([-2, 2])


ani1 = animation.FuncAnimation(
    fig, animate_two_body, frames=len(time_span_2b), interval=1
)

plt.show()


plt.plot(
    [i for i in range(len(two_body_system.total_energy))],
    two_body_system.total_energy,
    label="total_energy",
)
plt.plot(
    [i for i in range(len(two_body_system.total_gpe))],
    two_body_system.total_gpe,
    label="total_gpe",
)
plt.plot(
    [i for i in range(len(two_body_system.total_ke))],
    two_body_system.total_ke,
    label="total_ke",
)
plt.legend()
plt.xlabel("Time steps")
plt.ylabel("Total energy in system")
plt.show()
plt.close()
