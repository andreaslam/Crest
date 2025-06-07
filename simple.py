import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


def dydx(x):
    return np.cos(x)


class RK4Solver:
    def __init__(self, x, y, h, num_steps):
        self.x = x
        self.y = y
        self.h = h
        self.num_steps = num_steps
        self.xs = []
        self.ys = []

    def solve(self):
        for _ in range(self.num_steps):
            self.x, self.y = self.solve_step()
            self.xs.append(self.x)
            self.ys.append(self.y)

    def solve_step(self):
        x0 = self.x
        y0 = self.y

        k1 = dydx(x0)
        k2 = dydx(x0 + self.h / 2)
        k3 = dydx(x0 + self.h / 2)
        k4 = dydx(x0 + self.h)

        x_new = x0 + self.h
        y_new = y0 + (self.h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        return x_new, y_new


class EulerSolver:
    def __init__(self, x, y, h, num_steps):
        self.x = x
        self.y = y
        self.h = h
        self.num_steps = num_steps
        self.xs = []
        self.ys = []

    def solve(self):
        for _ in range(self.num_steps):
            self.x, self.y = self.solve_step()
            self.xs.append(self.x)
            self.ys.append(self.y)

    def solve_step(self):
        x0 = self.x
        y0 = self.y

        x_new = x0 + self.h
        y_new = y0 + self.h * dydx(x0)

        return x_new, y_new


if __name__ == "__main__":
    hs = np.logspace(
        -3, -1, 100, dtype=np.float64
    )  # -9 for euler to break, -3 for rk4 to break
    x = 0
    y = 0
    x_end = 100
    errors_rk4 = []
    errors_euler = []

    for h in tqdm(hs):
        num_steps = int((x_end - x) / h)
        x_model = x + h * np.arange(1, num_steps + 1, dtype=np.float64)
        y_model = np.sin(x_model)

        rk4 = RK4Solver(x, y, h, num_steps)
        euler = EulerSolver(x, y, h, num_steps)

        rk4.solve()
        euler.solve()
        rk4_error = np.mean(
            np.abs(np.array(rk4.ys, dtype=np.float64) - y_model), dtype=np.float64
        )
        euler_error = np.mean(
            np.abs(np.array(euler.ys, dtype=np.float64) - y_model), dtype=np.float64
        )

        errors_rk4.append(rk4_error)
        errors_euler.append(euler_error)

    plt.scatter(hs, errors_rk4, label="RK4")
    plt.vlines(
        3.16e-2,
        ymin=min(errors_rk4 + errors_euler),
        ymax=max(errors_rk4 + errors_euler),
        color="red",
        label="float64",
    )
    plt.vlines(
        0.5 * (3.16e-2 + 1.38e-3),
        ymin=min(errors_rk4 + errors_euler),
        ymax=max(errors_rk4 + errors_euler),
        color="green",
        label="avg",
    )
    plt.vlines(
        1.38e-3,
        ymin=min(errors_rk4 + errors_euler),
        ymax=max(errors_rk4 + errors_euler),
        color="blue",
        label="float32",
    )

    plt.scatter(hs, errors_euler, label="Euler")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Step Size (h)")
    plt.ylabel("Mean Absolute Error")
    plt.title("Comparison of RK4 and Euler Methods")
    plt.legend()
    plt.grid(True)
    plt.show()
