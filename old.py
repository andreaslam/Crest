from __future__ import annotations
from pathlib import Path
import time
import datetime
import os
import csv
from abc import ABC, abstractmethod
from copy import deepcopy
from itertools import combinations
from typing import Any

import matplotlib.pyplot as plt

import numpy as np
import tqdm
from matplotlib.animation import FuncAnimation

g = 6.6743e-11


class Mass:
    """
    Represents a celestial object used to model in n-body simulation. This assumes a point mass and ignoring relativistic effects in orbit.

    Attributes
    ----------
    self.mass (float): mass of the celstial object in kg.
    self.velocity (np.ndarray): initial velocity in m/s of the celstial object as an `np.ndarray`, with components of velocity (x,y,z)
    self.position (np.ndarray): initial position of the celstial object as an `np.ndarray`, with positions as (x,y,z) on a Cartesian space in meters.

    """

    def __init__(self, mass: float, velocity: np.ndarray, position: np.ndarray) -> None:
        self.mass = mass
        self.velocity = velocity
        self.position = position
        self.acceleration = np.zeros(3, dtype=np.float64)

    def reset_acceleration(self):
        """
        Method to reset acceleration for the object, replacing current values of `self.acceleration` with (0,0,0)
        """
        self.acceleration = np.zeros(3, dtype=np.float64)

    def __repr__(self) -> str:
        return f"Mass(mass={self.mass}, velocity={self.velocity}, position={self.position}, acceleration={self.acceleration})"


class System:
    def __init__(
        self,
        objs: list[Mass],
        num_steps: int,
        solver,
        h: float = 0.1,
        energy_check_interval: int = 1,
    ) -> None:
        """
        An n-body system containing celestial objects.


        Attributes
        ----------
        self.objs (list[Mass]): objects involved in simulations as a list of `Mass`
        self.num_steps (int): number of time steps in total for simulation.
        self.h (float): step size of simulation.
        self.starting_positions (list[np.ndarray]): list of starting positions for each celestial object.
        self.solver: method of solver used to numerically approximate orbits.
        self.energy_check_interval (int): how often to calculate the total energy of the system to check for energy conservation and drift.
        self.positions (dict): dictionary containing mass: (cumulative) position vector pairs of each object for all time steps.
        self.velocities (dict): dictionary containing mass: (cumulative) velocity vector pairs of each object for all time steps.
        self.total_energy (list): list containing values of the total energy in the system.
        """
        self.objs = objs
        self.h = h
        self.num_steps = int(num_steps / self.h)
        assert self.num_steps > 0
        self.starting_positions = [obj.position for obj in self.objs]

        self.solver = solver(self)

        self.positions = {id(mass): [] for mass in objs}
        self.velocities = {id(mass): [] for mass in objs}
        self.energy_check_interval = energy_check_interval
        self.total_energy = []

    def solve(self):
        """
        Computes an orbit for the number of given time steps.
        """
        self.solver.solve()

    def get_acceleration(self, obj_or_y):
        if isinstance(obj_or_y, Mass):
            # Initialise a fresh acceleration vector.
            accel = np.zeros(3, dtype=np.float64)
            obj = obj_or_y
            for other_obj in self.objs:
                if id(other_obj) != id(obj):
                    r_vec = other_obj.position - obj.position
                    r = np.linalg.norm(r_vec)
                    if r == 0:
                        continue
                    accel += (g * other_obj.mass * r_vec) / (r**3)
            return accel

        elif isinstance(obj_or_y, np.ndarray):
            # (Keep the existing np.ndarray branch)
            y = obj_or_y
            num_objs = len(self.objs)
            positions = y[: 3 * num_objs].reshape((num_objs, 3))
            velocities = y[3 * num_objs :].reshape((num_objs, 3))

            accelerations = np.zeros_like(positions)
            for i, obj in enumerate(self.objs):
                for j, other in enumerate(self.objs):
                    if i != j:
                        r_vec = positions[j] - positions[i]
                        r = np.linalg.norm(r_vec)
                        if r > 0:
                            accelerations[i] += (g * other.mass * r_vec) / (r**3)
            return np.hstack([velocities.flatten(), accelerations.flatten()])
        else:
            raise ValueError("Invalid arguments for get_acceleration function.")

    def calculate_total_ke(self):
        """
        Calcualates the total kinetic energy of the system using 0.5mv^2
        """
        return sum(
            [0.5 * obj.mass * (np.linalg.norm(obj.velocity) ** 2) for obj in self.objs]
        )

    def calculate_total_pe(self):
        """
        Calculates the total Gravitational Potential Energy (GPE) of the system using g*m1m2/r. Value of GPE is negative as it is an attractive force.
        """
        return -sum(
            [
                (g * pair[0].mass * pair[1].mass)
                / (np.linalg.norm(pair[0].position - pair[1].position))
                for pair in list(combinations(self.objs, 2))
            ]
        )

    def calculate_total_energy(self):
        """
        Calculates the total energy (kinetic + gravitational potential) of the system. Appends to `self.total_energy` within the function.
        """
        self.total_energy.append(self.calculate_total_ke() + self.calculate_total_pe())

    def total_num_positions(self):
        """
        Gets the total number of positions for visualisation.
        """
        return int(
            sum([len(pos) for pos in self.positions.values()])
            / len(self.positions.keys())
        )

    def plot_1d_phase_space(self, nth_obj, nth_dim, velocities, positions, ax):
        """
        Plots a phase space diagram (momentum-position) for 1 component of velocity.

        Args:
        nth_obj (int): the nth object modelled inside the n-body simulation
        n_th_dim (int): which component of velocity to plot, indexed from 0 ("xyz")
        velocities (np.ndarray): velocities for the object over time.
        """
        dim_order = "xyz"
        axis = dim_order[nth_dim]
        ax.plot(positions, velocities, label=f"Phase {axis}-component")
        ax.set_title(f"{axis}-velocity vs {axis}-position (Object {nth_obj + 1})")
        ax.set_xlabel(f"{axis}-position (m)")
        ax.set_ylabel(f"{axis}-momentum (kgm/s)")
        ax.legend()
        ax.grid(True)

    def plot_object_phase_space(self, save=True, img_path="phase_diagram.png"):
        """
        Plots the phase diagram plots for all celestial bodies.

        save (bool): whether to save plot as an image. Defaults to True
        img_path (str): path of where image is saved. Defaults to "phase_diagram.png"
        """
        num_objects = len(self.positions)
        num_dimensions = len(next(iter(self.positions.values()))[0])

        fig, axes = plt.subplots(
            num_objects,
            num_dimensions,
            figsize=(6 * num_dimensions, 2.25 * num_objects),
        )  # w, h
        fig.suptitle(
            f"Phase diagram of {len(self.objs)}-body system using {self.solver}"
        )

        for nth_obj, (position_list, velocity_list) in enumerate(
            zip(self.positions.values(), self.velocities.values())
        ):
            position_data = np.array(position_list)
            velocity_data = np.array(velocity_list)

            if position_data.shape[0] < 2 or velocity_data.shape[0] < 2:
                raise ValueError(f"Not enough data to plot for object {nth_obj + 1}.")

            for nth_dim in range(position_data.shape[1]):
                ax = axes[nth_obj, nth_dim]
                self.plot_1d_phase_space(
                    nth_obj,
                    nth_dim,
                    velocities=velocity_data[:, nth_dim] * self.objs[nth_obj].mass,
                    positions=position_data[:, nth_dim],
                    ax=ax,
                )
        plt.tight_layout()
        if save:
            plt.savefig(f"{Path(img_path).stem}_{str(self.solver)}.png")
        plt.show()


class Solver(ABC):
    def __init__(self, system: System):
        self.system = system
        self.progress_bar_description = (
            f"Solving {len(self.system.objs)}-body system using {type(self).__name__}"
        )
        self.execution_duration = 0

    def solve(self):
        start = time.time()
        for i in tqdm.tqdm(
            range(self.system.num_steps), desc=self.progress_bar_description
        ):
            if i == 0:
                self.system.calculate_total_energy()
            self.solve_step()
            if i % self.system.energy_check_interval == 0 and i != 0:
                self.system.calculate_total_energy()
        end = time.time()
        self.execution_duration = end - start

    @abstractmethod
    def solve_step(self) -> Any:
        pass

    def __repr__(self) -> str:
        return type(self).__name__


class EulerSolver(Solver):
    def __init__(self, system: System):
        super().__init__(system)

    def solve_step(self):
        for obj in self.system.objs:
            obj.acceleration = self.system.get_acceleration(obj)
            obj.velocity += self.system.h * obj.acceleration

        for obj in self.system.objs:
            obj.position += self.system.h * obj.velocity
            self.system.positions[id(obj)].append(np.copy(obj.position))
            self.system.velocities[id(obj)].append(np.copy(obj.velocity))
            obj.reset_acceleration()


class RK4Solver(Solver):
    def __init__(self, system: System):
        super().__init__(system)

    def solve_step(self):
        h = self.system.h
        num_objs = len(self.system.objs)

        # Flatten initial state
        y0 = np.hstack(
            [
                np.ravel([obj.position for obj in self.system.objs]),
                np.ravel([obj.velocity for obj in self.system.objs]),
            ]
        )

        k1 = self.system.get_acceleration(y0)
        k2 = self.system.get_acceleration(y0 + (h / 2) * k1)
        k3 = self.system.get_acceleration(y0 + (h / 2) * k2)
        k4 = self.system.get_acceleration(y0 + h * k3)

        y_new = y0 + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)

        # Unpack updated positions and velocities
        positions = y_new[: 3 * num_objs].reshape((num_objs, 3))
        velocities = y_new[3 * num_objs :].reshape((num_objs, 3))

        # Update objects
        for i, obj in enumerate(self.system.objs):
            obj.position[:] = positions[i]
            obj.velocity[:] = velocities[i]
            self.system.positions[id(obj)].append(positions[i].copy())
            self.system.velocities[id(obj)].append(velocities[i].copy())


class LyapunovCalculator:
    def __init__(self, orbits) -> None:
        self.orbits = orbits
        self.separation = []

    def get_separation(self, skip_interval=50):
        for i, (obj, other_obj) in enumerate(zip(*self.orbits)):
            if i % skip_interval == 0:
                self.separation.append(
                    np.linalg.norm(obj.position - other_obj.position)
                )

    def calculate_lyapunov_exponent(self, num_points=500, skip_interval=50):
        lyapunov_sum = 0.0
        num_pairs = len(self.orbits[0])
        for i in range(num_points):
            self.get_separation(skip_interval)
            for j in range(num_pairs):
                lyapunov_sum += np.log(
                    self.separation[j]
                    / np.linalg.norm(
                        self.orbits[0][j].position - self.orbits[1][j].position
                    )
                )
                self.orbits[1][j].position = deepcopy(self.orbits[0][j].position)
                self.orbits[1][j].velocity = deepcopy(self.orbits[0][j].velocity)
        return lyapunov_sum / (num_points * num_pairs)


class VelocityVerletSolver(Solver):
    def __init__(self, system: System):
        super().__init__(system)
        # Compute initial accelerations
        for obj in self.system.objs:
            obj.acceleration = self.system.get_acceleration(obj)

    def solve_step(self):
        h = self.system.h
        h2_half = 0.5 * h * h  # Precompute factor

        # Save the old accelerations
        old_accels = [obj.acceleration.copy() for obj in self.system.objs]

        # First update: position using current velocity and acceleration
        for obj in self.system.objs:
            obj.position += obj.velocity * h + obj.acceleration * h2_half

        # Compute new accelerations after position update
        new_accels = [self.system.get_acceleration(obj) for obj in self.system.objs]

        # Second update: velocity using the average of old and new acceleration
        for obj, old_a, new_a in zip(self.system.objs, old_accels, new_accels):
            obj.velocity += 0.5 * h * (old_a + new_a)
            obj.acceleration = new_a  # Update acceleration for the next step

            # Store updated values
            self.system.positions[id(obj)].append(np.copy(obj.position))
            self.system.velocities[id(obj)].append(np.copy(obj.velocity))


class ExperimentManager:
    def __init__(
        self,
        init_conditions: list[Mass],
        num_steps: int,
        solvers: list[type],
        experiment_csv_path: str = "experiment.csv",
        note: str = "",
    ):
        self.init_conditions = init_conditions
        self.num_steps = num_steps
        self.all_systems_total_energy = []
        self.systems = [
            System(deepcopy(init_conditions), num_steps, solver) for solver in solvers
        ]
        self.experiment_csv_path = experiment_csv_path
        self.note = note

    def solve_all(self):
        for system in self.systems:
            system.solve()
            print("std:", np.std(system.total_energy))
            print(
                "average abs energy difference since initial condition:",
                np.mean([abs(system.total_energy[0] - t) for t in system.total_energy]),
            )
            print(
                "average abs squared energy difference since initial condition:",
                np.mean(
                    [abs(system.total_energy[0] - t) ** 2 for t in system.total_energy]
                ),
            )
            self.all_systems_total_energy.append(system.total_energy)

    def plot_object_phase_space(self, save=True, img_path="phase_diagram.png"):
        for system in self.systems:
            system.plot_object_phase_space(save=save, img_path=img_path)

    def plot_results(self):
        for system, total_energy in zip(self.systems, self.all_systems_total_energy):
            plt.plot(
                [i * system.energy_check_interval for i in range(len(total_energy))],
                total_energy,
                label=system.solver,
            )

        plt.title("Total energy (J) in system")
        plt.legend()
        plt.show()

    def export_experiment(self):
        headings = [
            "date",
            "masses",
            "initial_velocities",
            "initial_positions",
            "solver",
            "n_steps",
            "step_size",
            "std_energy_loss",
            "execution_duration",
            "notes",
        ]

        if not os.path.exists(self.experiment_csv_path) or not os.path.getsize(
            self.experiment_csv_path
        ):
            with open(self.experiment_csv_path, "w") as f:
                writer = csv.DictWriter(f, fieldnames=headings)
                writer.writeheader()
        for system in self.systems:
            experiment_data = [
                datetime.datetime.now(),
                [mass.mass for mass in system.objs],
                [list(mass.velocity) for mass in self.init_conditions],
                [list(mass.position) for mass in self.init_conditions],
                system.solver,
                system.num_steps,
                system.h,
                np.std(system.total_energy),
                system.solver.execution_duration,
                self.note,
            ]
            with open(self.experiment_csv_path, "a") as f:
                writer = csv.DictWriter(f, fieldnames=headings)
                writer.writerow({k: v for k, v in zip(headings, experiment_data)})


def animate_orbits_3d(manager: ExperimentManager):
    num_solvers = len(manager.systems)
    fig = plt.figure(figsize=(5 * num_solvers, 5))

    axes = [
        fig.add_subplot(1, num_solvers, i + 1, projection="3d")
        for i in range(num_solvers)
    ]

    lines_per_solver = [
        [ax.plot([], [], [], label=f"Mass {obj.mass} kg")[0] for obj in system.objs]
        for ax, system in zip(axes, manager.systems)
    ]
    points_per_solver = [
        [ax.plot([], [], [], "o")[0] for _ in system.objs]
        for ax, system in zip(axes, manager.systems)
    ]

    for ax, system in zip(axes, manager.systems):
        ax.set_title(f"{type(system.solver).__name__} Simulation")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_zlabel("Z Position (m)")
        ax.set_xlim([-250, 250])
        ax.set_ylim([-250, 250])
        ax.set_zlim([-250, 250])
        ax.legend()

    def init():
        for lines, points in zip(lines_per_solver, points_per_solver):
            for line, point in zip(lines, points):
                line.set_data([], [])
                line.set_3d_properties([])
                point.set_data([], [])
                point.set_3d_properties([])
        return [
            item
            for sublist in (lines_per_solver + points_per_solver)
            for item in sublist
        ]

    def update(frame):
        for system, lines, points in zip(
            manager.systems, lines_per_solver, points_per_solver
        ):
            for obj, line, point in zip(system.objs, lines, points):
                positions = np.array(system.positions[id(obj)])
                if frame < len(positions):
                    line.set_data(positions[:frame, 0], positions[:frame, 1])
                    line.set_3d_properties(positions[:frame, 2])
                    point.set_data(positions[frame, 0], positions[frame, 1])
                    point.set_3d_properties(positions[frame, 2])
        return [
            item
            for sublist in (lines_per_solver + points_per_solver)
            for item in sublist
        ]

    max_frames = max(system.total_num_positions() for system in manager.systems)

    ani = FuncAnimation(
        fig,
        update,
        frames=max_frames,
        init_func=init,
        interval=1,
        blit=True,
    )
    plt.show()


if __name__ == "__main__":
    # TODO implement non-dimensionalisation function so that it works with real life values without introducing large numerical values
    celestial_objects = [
        Mass(
            mass=100000e3,
            velocity=np.array([-0.0001, -0.000, 0.0]),
            position=np.array([-10.0, 10.0, 10.0]),
        ),  # 3
        Mass(
            mass=1500e3,
            velocity=np.array([0.005, -0.01, -0.002]),
            position=np.array([-50.0, 7.0, 0.0]),
        ),  # 3
        Mass(
            mass=500e3,
            velocity=np.array([0.0, 0.01, 0.0]),
            position=np.array([-100, -0.35, 0.2]),
        ),  # 3
        Mass(
            mass=50e3,
            velocity=np.array([0.0, -0.001, 0.001]),
            position=np.array([240.3, 2.5, -3.2]),
        ),
        Mass(
            mass=100e3,
            velocity=np.array([0.001, -0.01, 0.002]),
            position=np.array([-75.2, -48.3, -2.4]),
        ),
        Mass(
            mass=100e3,
            velocity=np.array([0.0, 0.0, 0.001]),
            position=np.array([-90.2, 210.3, -25.0]),
        ),
        Mass(
            mass=10e3,
            velocity=np.array([-0.0005, -0.0001, -0.005]),
            position=np.array([200.0, 200.0, 200.0]),
        ),
        Mass(
            mass=10e3,
            velocity=np.array([0.0, 0.01, 0.0]),
            position=np.array([25.0, -75.0, -20.2]),
        ),
    ]

    experiment = ExperimentManager(
        celestial_objects, 10000, [EulerSolver, VelocityVerletSolver, RK4Solver]
    )

    experiment.solve_all()
    experiment.plot_results()
    experiment.plot_object_phase_space(save=True)
    experiment.export_experiment()
    animate_orbits_3d(experiment)
