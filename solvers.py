from __future__ import annotations
import rebound
import csv
import datetime
import os
import sys
import time
from abc import ABC, abstractmethod
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from itertools import combinations
import random
from pathlib import Path
from typing import Any
from lyapunov import modify_init_conditions, LyapunovCalculator
import matplotlib.pyplot as plt
import numpy as np
import tqdm
from matplotlib.animation import FuncAnimation

from metrics import calculate_trajectory_deviance_pair

g = 6.6743e-11


class Mass:
    """
    Represents a celestial object used in an n-body simulation modelled as a point-mass.
    """

    def __init__(self, mass: float, velocity: np.ndarray, position: np.ndarray) -> None:
        self.mass = mass
        self.velocity = velocity
        self.position = position
        self.acceleration = np.zeros(3, dtype=np.float64)

    def reset_acceleration(self):
        self.acceleration = np.zeros(3, dtype=np.float64)

    def __repr__(self) -> str:
        return f"Mass(mass={self.mass}, velocity={self.velocity}, position={self.position}, acceleration={self.acceleration})"


class System:
    def __init__(
        self,
        objs: list[Mass],
        simulation_length: int,
        solver,
        h: float = 1000,
        energy_check_interval: int = 1,
        scaled=False,
        conv=None,
    ) -> None:
        self.scaled = scaled
        self.objs = objs
        self.h = h
        self.simulation_length = simulation_length
        self.num_steps = int(self.simulation_length / self.h)
        assert self.num_steps > 1
        self.starting_positions = [obj.position for obj in self.objs]
        self.solver = solver(self)
        self.positions = {i: [] for i in range(len(objs))}
        self.velocities = {i: [] for i in range(len(objs))}
        self.energy_check_interval = energy_check_interval
        self.total_energy = []
        self.conv = conv
        self.energy_thresholds = np.logspace(-20, -1, num=20, dtype=float)
        self.idx_energy_exceeded = {t: None for t in self.energy_thresholds}
        self.solved = False
        self.acc_energy = 0.0

    def solve(self):
        self.solver.solve()
        self.solved = True

    def get_acceleration(self, obj):
        positions = np.array([other.position for other in self.objs])
        r_vec = positions - obj.position
        r = np.linalg.norm(r_vec, axis=1)
        mask = r != 0  # Avoid division by zero
        accel = np.zeros(3, dtype=np.float64)
        if np.any(mask):
            accel = np.sum(
                (
                    (1 if self.scaled else g)
                    * np.array([o.mass for o in self.objs])[mask, None]
                    * r_vec[mask]
                )
                / (r[mask, None] ** 3),
                axis=0,
            )
        return accel

    def calculate_total_ke(self):
        return sum(
            [0.5 * obj.mass * (np.linalg.norm(obj.velocity) ** 2) for obj in self.objs]
        )

    def calculate_total_pe(self):
        return -sum(
            [
                ((1 if self.scaled else g) * pair[0].mass * pair[1].mass)
                / (np.linalg.norm(pair[0].position - pair[1].position))
                for pair in list(combinations(self.objs, 2))
            ]
        )

    def calculate_total_energy(self):
        total_energy = (
            self.conv.convert_energy_to_joules(
                self.calculate_total_ke() + self.calculate_total_pe()
            )
            if self.conv and self.scaled
            else self.calculate_total_ke() + self.calculate_total_pe()
        )
        if not self.total_energy:
            print("init total_energy", total_energy)
            assert (
                total_energy < 0
            )  # negative to ensure that system is gravitationally bound
        else:
            self.acc_energy += abs((total_energy - self.total_energy[-1])) / abs(
                self.total_energy[-1]
            )
        self.total_energy.append(total_energy)
        for t in self.energy_thresholds:
            if (
                total_energy < self.total_energy[0] * (1 + t)
                or total_energy > self.total_energy[0] * (1 - t)
            ) and not self.idx_energy_exceeded[t]:
                self.idx_energy_exceeded[t] = (
                    len(self.total_energy) - 1
                ) / self.num_steps
        return total_energy

    def total_num_positions(self):
        return int(
            sum(len(pos) for pos in self.positions.values()) / len(self.positions)
        )

    def plot_1d_phase_space(self, nth_obj, nth_dim, momenta, positions, ax):
        dim_order = "xyz"
        axis = dim_order[nth_dim]
        ax.plot(positions, momenta, label=f"Phase {axis}-component")
        ax.set_title(f"{axis}-momentum vs {axis}-position (Object {int(nth_obj):e} kg)")
        ax.set_xlabel(f"{axis}-position (m)")
        ax.set_ylabel(f"{axis}-momentum (kgm/s)")
        ax.legend()
        ax.grid(True)

    def plot_object_phase_space(
        self, save=True, img_path="phase_diagram.png", show=False
    ):
        first_obj_positions = next(iter(self.positions.values()))
        if not first_obj_positions:
            raise ValueError(
                "No position data recorded. Ensure the simulation has run."
            )
        num_dimensions = len(first_obj_positions[0])
        num_objects = len(self.positions)
        fig, axes = plt.subplots(
            num_objects,
            num_dimensions,
            figsize=(6 * num_dimensions, 2.25 * num_objects),
        )
        fig.suptitle(
            f"Phase diagram of {len(self.objs)}-body system using {str(self.solver)}"
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
                    self.objs[nth_obj].mass,
                    nth_dim,
                    momenta=velocity_data[:, nth_dim] * self.objs[nth_obj].mass,
                    positions=position_data[:, nth_dim],
                    ax=ax,
                )
        plt.tight_layout()
        if save:
            if not os.path.exists(folder_path := "experiment_graphs"):
                os.makedirs(folder_path)
            plt.savefig(f"{folder_path}/{Path(img_path).stem}_{str(self.solver)}.png")
        if show:
            plt.show()
        plt.close()

    def __repr__(self) -> str:
        return f"System(num_objs={len(self.objs)}, solver={self.solver}, h={self.h}, self.num_steps={self.num_steps})"


class Solver(ABC):
    def __init__(self, system: System, patience=100):
        self.system = system
        self.progress_bar_description = (
            f"Solving {len(self.system.objs)}-body system using {type(self).__name__}"
        )
        self.execution_duration = 0
        self.zero_acceleration_threshold = 100

    def solve(self):
        start = time.time()
        self.system.calculate_total_energy()
        for i in tqdm.tqdm(
            range(self.system.num_steps), desc=self.progress_bar_description
        ):
            self.solve_step()
            if (i + 1) % self.system.energy_check_interval == 0:
                self.system.calculate_total_energy()
        end = time.time()
        self.execution_duration = end - start

    @abstractmethod
    def solve_step(self) -> Any:
        pass

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.system.h:.5g})"


class EulerSolver(Solver):
    def __init__(self, system: System):
        super().__init__(system)

    def solve_step(self):
        for obj in self.system.objs:
            obj.acceleration = self.system.get_acceleration(obj)
            obj.velocity += self.system.h * obj.acceleration

        for i, obj in enumerate(self.system.objs):
            obj.position += self.system.h * obj.velocity
            self.system.positions[i].append(np.copy(obj.position))
            self.system.velocities[i].append(np.copy(obj.velocity))
            obj.reset_acceleration()


class VelocityVerletSolver(Solver):
    def __init__(self, system: System):
        super().__init__(system)
        for obj in self.system.objs:
            obj.acceleration = self.system.get_acceleration(obj)

    def solve_step(self):
        h = self.system.h
        h2_half = 0.5 * h * h
        old_accels = [obj.acceleration.copy() for obj in self.system.objs]
        for obj in self.system.objs:
            obj.position += obj.velocity * h + obj.acceleration * h2_half
        new_accels = [self.system.get_acceleration(obj) for obj in self.system.objs]
        for i, (obj, old_a, new_a) in enumerate(
            zip(self.system.objs, old_accels, new_accels)
        ):
            obj.velocity += 0.5 * h * (old_a + new_a)
            obj.acceleration = new_a
            self.system.positions[i].append(np.copy(obj.position))
            self.system.velocities[i].append(np.copy(obj.velocity))


class RK4Solver(Solver):
    def __init__(self, system: System):
        super().__init__(system)

    def solve_step(self):
        h = self.system.h
        objs = self.system.objs

        # Save initial states
        pos0 = [np.copy(obj.position) for obj in objs]
        vel0 = [np.copy(obj.velocity) for obj in objs]

        def compute_accelerations(positions):
            # Temporarily assign positions to objects to compute acceleration correctly
            for obj, p in zip(objs, positions):
                obj.position = p
            return [self.system.get_acceleration(obj) for obj in objs]

        # k1: derivatives at initial state
        a1 = compute_accelerations(pos0)
        k1_pos = vel0
        k1_vel = a1

        # k2: derivatives at midpoint using k1
        pos_k2 = [p + 0.5 * h * v for p, v in zip(pos0, k1_pos)]
        vel_k2 = [v + 0.5 * h * a for v, a in zip(vel0, k1_vel)]
        a2 = compute_accelerations(pos_k2)
        k2_pos = vel_k2
        k2_vel = a2

        # k3: derivatives at midpoint using k2
        pos_k3 = [p + 0.5 * h * v for p, v in zip(pos0, k2_pos)]
        vel_k3 = [v + 0.5 * h * a for v, a in zip(vel0, k2_vel)]
        a3 = compute_accelerations(pos_k3)
        k3_pos = vel_k3
        k3_vel = a3

        # k4: derivatives at end using k3
        pos_k4 = [p + h * v for p, v in zip(pos0, k3_pos)]
        vel_k4 = [v + h * a for v, a in zip(vel0, k3_vel)]
        a4 = compute_accelerations(pos_k4)
        k4_pos = vel_k4
        k4_vel = a4

        # Combine increments weighted sum for next step
        for i, obj in enumerate(objs):
            obj.position = pos0[i] + (h / 6) * (
                k1_pos[i] + 2 * k2_pos[i] + 2 * k3_pos[i] + k4_pos[i]
            )
            obj.velocity = vel0[i] + (h / 6) * (
                k1_vel[i] + 2 * k2_vel[i] + 2 * k3_vel[i] + k4_vel[i]
            )
            obj.acceleration = self.system.get_acceleration(obj)

            self.system.positions[i].append(np.copy(obj.position))
            self.system.velocities[i].append(np.copy(obj.velocity))


class ReboundSolver(Solver):
    """
    A solver that uses the REBOUND library for high-precision N-body integration.
    This class acts as a wrapper to make REBOUND compatible with the System and
    ExperimentManager framework.
    """

    def __init__(self, system: System):
        super().__init__(system)
        self.sim = rebound.Simulation()
        self.sim.integrator = "ias15"
        self.sim.dt = self.system.h
        if self.system.scaled:
            self.sim.G = 1
        else:
            self.sim.G = g

        for obj in self.system.objs:
            self.sim.add(
                m=obj.mass,
                x=obj.position[0],
                y=obj.position[1],
                z=obj.position[2],
                vx=obj.velocity[0],
                vy=obj.velocity[1],
                vz=obj.velocity[2],
            )
        self.sim.move_to_com()

    def solve(self):
        """
        Overrides the base solve method to use REBOUND's integration logic.
        """
        start_time = time.time()
        self.system.calculate_total_energy()  # Initial energy

        times = np.linspace(
            self.sim.t, self.system.simulation_length, self.system.num_steps
        )

        for i, t in enumerate(tqdm.tqdm(times[1:], desc=self.progress_bar_description)):
            self.sim.integrate(t)
            for j, p in enumerate(self.sim.particles):
                self.system.positions[j].append(np.array([p.x, p.y, p.z]))
                self.system.velocities[j].append(np.array([p.vx, p.vy, p.vz]))

            if (i + 1) % self.system.energy_check_interval == 0:
                # Use REBOUND's optimized energy calculation
                energy = self.sim.energy()
                if self.system.conv and self.system.scaled:
                    energy = self.system.conv.convert_energy_to_joules(energy)
                self.system.total_energy.append(energy)

        self.execution_duration = time.time() - start_time

    def solve_step(self):
        """Not used by this solver, as solve() is overridden."""
        pass


class ExperimentManager:
    def __init__(
        self,
        init_conditions: list[Mass],
        simulation_length: float,
        solvers: list[Solver],
        h_values: list[float] | float,
        experiment_note: str = "",
        scaled=False,
        conv=None,
    ):
        self.scaled = scaled
        self.init_conditions = init_conditions
        self.simulation_length = simulation_length
        self.all_systems_total_energy = []
        if isinstance(h_values, (float, int)):
            h_values = [h_values] * len(solvers)
        elif len(h_values) == 1:
            h_values = h_values * len(solvers)
        elif len(solvers) == 1 and len(h_values) > 1:
            solvers *= len(h_values)
        self.solvers = solvers
        assert len(h_values) == len(solvers)
        assert len(init_conditions) >= 1
        self.h_values = h_values
        self.systems = [
            System(
                deepcopy(init_conditions),
                simulation_length,
                solver,
                h=h,
                scaled=scaled,
                conv=conv,
            )
            for solver, h in zip(solvers, self.h_values)
        ]
        self.systems_trajectories = []
        self.experiment_note = experiment_note
        self.num_steps = [system.num_steps for system in self.systems]
        self.separation = {}
        measure_duration = simulation_length
        measure_h = 1000 / conv.time_sf
        self.conv = conv
        self.baseline = System(
            deepcopy(init_conditions),
            measure_duration,
            ReboundSolver,
            h=measure_h,
            scaled=self.scaled,
            conv=self.conv,
        )
        self.modified = System(
            modify_init_conditions(deepcopy((init_conditions)), threshold=1e-3),
            measure_duration,
            ReboundSolver,
            h=measure_h,
            scaled=self.scaled,
            conv=self.conv,
        )
        self.lyapunov = None

    def get_lyapunov(self):
        solved_systems = []
        with ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(solve_system, system)
                for system in [self.baseline, self.modified]
            ]
        for future in as_completed(futures):
            system = future.result()
            solved_systems.append(system)
        self.baseline, self.modified = solved_systems
        calc = LyapunovCalculator(self.baseline, self.modified)
        self.lyapunov = calc.calculate_lyapunov_exponent()
        return self.lyapunov / self.conv.time_sf if self.conv else self.lyapunov

    def solve_all(self):
        solved_systems = []
        with ProcessPoolExecutor() as executor:
            futures = [executor.submit(solve_system, system) for system in self.systems]
            for future in as_completed(futures):
                system = future.result()
                print("std:", np.std(system.total_energy))
                print(
                    "average abs energy difference since initial condition:",
                    np.mean(
                        [abs(system.total_energy[0] - t) for t in system.total_energy]
                    ),
                )
                print(
                    "average abs squared energy difference since initial condition:",
                    np.mean(
                        [
                            abs(system.total_energy[0] - t) ** 2
                            for t in system.total_energy
                        ]
                    ),
                )
                solved_systems.append(system)
                self.all_systems_total_energy.append(system.total_energy)
        self.systems = solved_systems

    def plot_object_phase_space(
        self, save=True, img_path="phase_diagram.png", show=False
    ):
        assert all(s.solved for s in self.systems)
        for system in self.systems:
            system.plot_object_phase_space(save=save, img_path=img_path, show=show)

    def common_time_axis(self, y, max_sim_len):
        max_points = max(len(e) for e in y)
        return np.linspace(0, max_sim_len, max_points)

    def plot_energy_conservation(
        self, save=True, img_path="total_energy.png", show=False
    ):
        if not self.systems:
            return
        assert all(s.solved for s in self.systems)
        simulation_length = self.systems[0].simulation_length
        all_times = []
        all_energies = []

        for system in self.systems:
            h = system.h
            interval = system.energy_check_interval
            times = [i * interval * h for i in range(len(system.total_energy))]
            all_times.append(times)
            all_energies.append(system.total_energy)

        common_time = self.common_time_axis(all_energies, simulation_length)

        if self.systems[0].scaled and self.systems[0].conv:
            time_sf = self.systems[0].conv.time_sf
            common_time_physical = common_time * time_sf
        else:
            common_time_physical = common_time

        interpolated_energies = []
        for times, energy in zip(all_times, all_energies):
            if len(times) < 2:
                interpolated = np.full_like(common_time, energy[0]) if energy else []
            else:
                interpolated = np.interp(common_time, times, energy)
            interpolated_energies.append(interpolated)

        plt.figure()
        for system, energy in zip(self.systems, interpolated_energies):
            plt.plot(common_time_physical, energy, label=system.solver)

        plt.title("Total Energy Conservation Across Systems")
        plt.xlabel("Time (seconds)")
        plt.ylabel("Total Energy (Joules)")
        plt.grid()
        plt.legend()

        if save:
            os.makedirs("experiment_graphs", exist_ok=True)
            plt.savefig(f"experiment_graphs/{Path(img_path).stem}.png")
        if show:
            plt.show()
        plt.close()

    def calculate_trajectory_deviance(self, metric, show=False, save=True):
        assert all(s.solved for s in self.systems)
        all_trajectories = {}

        self.systems_trajectories = [
            deepcopy(system.positions) for system in self.systems
        ]

        for pair, system_pair in zip(
            combinations(self.systems_trajectories, 2),
            combinations([sys.solver for sys in self.systems], 2),
        ):
            separation, step_separation = self.calculate_trajectory_deviance_pair(
                metric, pair
            )

            self.separation[system_pair] = separation
            all_trajectories[system_pair] = step_separation
            self.plot_trajectory_deviance(
                all_trajectories, metric, system_pair, show, save
            )

    def calculate_trajectory_deviance_pair(self, metric, pair, metric_score_only=False):
        traj1, traj2 = pair
        len1 = len(next(iter(traj1.values()))) if traj1 else 0
        len2 = len(next(iter(traj2.values()))) if traj2 else 0
        if len1 > len2 > 0:
            step = len1 // len2
            for obj in traj1:
                traj1[obj] = traj1[obj][::step]
        elif len2 > len1 > 0:
            step = len2 // len1
            for obj in traj2:
                traj2[obj] = traj2[obj][::step]
        final_len1 = len(next(iter(traj1.values()))) if traj1 else 0
        final_len2 = len(next(iter(traj2.values()))) if traj2 else 0

        min_len = min(final_len1, final_len2)
        if min_len > 0:
            for obj in traj1:
                traj1[obj] = traj1[obj][:min_len]
            for obj in traj2:
                traj2[obj] = traj2[obj][:min_len]

        separation, step_separation = calculate_trajectory_deviance_pair(
            (traj1, traj2), metric
        )
        return separation if metric_score_only else (separation, step_separation)

    def plot_trajectory_deviance(
        self, trajectories, metric, solver_pair, show=False, save=True
    ):
        assert all(s.solved for s in self.systems)
        data = trajectories[solver_pair]
        common_time = np.linspace(0, solver_pair[0].system.simulation_length, len(data))
        label = f"{solver_pair[0]} vs {solver_pair[1]}"

        plt.figure()
        plt.plot(common_time, data, label=label)
        plt.title(f"Trajectory Deviance ({metric})")
        plt.xlabel("Time (s)")
        plt.ylabel("Deviance")
        plt.legend()
        plt.grid(True)
        if save:
            os.makedirs("experiment_graphs", exist_ok=True)
            plt.savefig(
                f"experiment_graphs/{Path(metric).stem}_{label.replace(' ', '_')}.png"
            )
        if show:
            plt.show()
        plt.close()

    def export_experiment(self, overwrite=False):
        headings = [
            "experiment_id",
            "date",
            "masses",
            "initial_velocities",
            "initial_positions",
            "solver",
            "n_steps",
            "step_size",
            "std_energy_loss",
            "execution_duration",
            "energy_thresholds",
            "lyapunov",
            "simulated_time_conv_scaled",
            "mae_deviance",
            "notes",
        ]
        assert all(s.solved for s in self.systems)
        if not os.path.exists(folder_path := "experiment_data"):
            os.makedirs(folder_path)
        file_path = f"{folder_path}/experiment.csv"

        if overwrite:
            with open(file_path, "w") as f:
                f.write("")

        # Determine next experiment ID
        next_experiment_id = 0
        if os.path.exists(file_path) and os.path.getsize(file_path):
            with open(file_path, newline="") as f:
                reader = csv.DictReader(f)
                ids = [
                    int(row["experiment_id"])
                    for row in reader
                    if row.get("experiment_id")
                ]
                if ids:
                    next_experiment_id = max(ids) + 1

        # Write header if needed
        if not os.path.exists(file_path) or not os.path.getsize(file_path):
            with open(file_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headings)
                writer.writeheader()

        if not self.lyapunov:
            self.get_lyapunov()

        reb_sim = System(
            self.init_conditions,
            self.simulation_length,
            ReboundSolver,
            1000.0 / self.conv.time_sf if self.conv else 1000.0,
            1,
            self.scaled,
            self.conv,
        )
        reb_sim.solve()
        for system in self.systems:
            experiment_data = [
                next_experiment_id,
                datetime.datetime.now(),
                [float(mass.mass) for mass in system.objs],
                [list(mass.velocity) for mass in self.init_conditions],
                [
                    [float(position) for position in list(mass.position)]
                    for mass in self.init_conditions
                ],
                system.solver,
                system.num_steps,
                (
                    system.h * system.conv.time_sf
                    if system.conv and system.scaled
                    else system.h
                ),
                abs(np.std(system.total_energy) / (system.total_energy[0])),
                # system.acc_energy / system.total_energy[0],
                system.solver.execution_duration,
                system.idx_energy_exceeded,
                self.lyapunov,
                system.simulation_length,
                (
                    self.calculate_trajectory_deviance_pair(
                        "mae",
                        [reb_sim.positions, system.positions],
                        metric_score_only=True,
                    )
                    * system.conv.dist_sf
                )
                / len(system.objs),
                self.experiment_note,
            ]
            with open(file_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=headings)
                writer.writerow({k: v for k, v in zip(headings, experiment_data)})

    def export_positions(self, clear_file=True):
        assert all(s.solved for s in self.systems)
        headings_positions = ["mass_id", "x", "y", "z"]
        headings_metadata = ["mass_id", "mass"]
        if not os.path.exists(folder_path := "experiment_data"):
            os.makedirs(folder_path)
        for system in self.systems:
            path_positions = f"{folder_path}/{str(system.solver)}_positions.csv"
            metadata_path = f"{folder_path}/{str(system.solver)}_metadata.csv"
            if clear_file:
                with open(path_positions, "w") as f:
                    f.write("")
                with open(metadata_path, "w") as f:
                    f.write("")

            if not os.path.exists(path_positions) or not os.path.getsize(
                path_positions
            ):
                with open(path_positions, "w") as f:
                    writer = csv.DictWriter(f, fieldnames=headings_positions)
                    writer.writeheader()

            if not os.path.exists(metadata_path) or not os.path.getsize(metadata_path):
                with open(metadata_path, "w") as f:
                    writer = csv.DictWriter(f, fieldnames=headings_metadata)
                    writer.writeheader()

            for i, obj in enumerate(system.objs):
                with open(path_positions, "a") as f:
                    writer = csv.DictWriter(f, fieldnames=headings_positions)
                    for pos in system.positions[i]:
                        writer.writerow(
                            {
                                "mass_id": i,
                                "x": np.round(pos[0], decimals=5),
                                "y": np.round(pos[1], decimals=5),
                                "z": np.round(pos[2], decimals=5),
                            }
                        )
                with open(metadata_path, "a") as f:
                    writer = csv.DictWriter(f, fieldnames=headings_metadata)
                    writer.writerow(
                        {
                            "mass_id": i,
                            "mass": obj.mass,
                        }
                    )

    def __repr__(self) -> str:
        exp = "\n\t".join([str(sys) for sys in self.systems])
        return f"ExperimentManager(num_experiments={len(self.systems)}, experiments=[\n\t{exp}\n])"


def solve_system(system: System) -> System:
    """
    Helper function to run a system's simulation.
    """
    system.solve()
    return system


def animate_orbits_3d(manager: ExperimentManager):
    num_solvers = len(manager.systems)
    fig = plt.figure(figsize=(5 * num_solvers, 5))
    axes = [
        fig.add_subplot(1, num_solvers, i + 1, projection="3d")
        for i in range(num_solvers)
    ]
    lines_per_solver = [
        [ax.plot([], [], [], label=f"Mass {obj.mass:.3g} kg")[0] for obj in system.objs]
        for ax, system in zip(axes, manager.systems)
    ]
    points_per_solver = [
        [ax.plot([], [], [], "o")[0] for _ in system.objs]
        for ax, system in zip(axes, manager.systems)
    ]
    for ax, system in zip(axes, manager.systems):
        ax.set_title(f"{system.solver} Simulation")
        ax.set_xlabel("X Position (m)")
        ax.set_ylabel("Y Position (m)")
        ax.set_zlabel("Z Position (m)")
        ax.set_xlim([-2, 2])
        ax.set_ylim([-2, 2])
        ax.set_zlim([-2, 2])
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
            for i, (obj, line, point) in enumerate(zip(system.objs, lines, points)):
                positions = np.array(system.positions[i])
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
    plt.close()


if __name__ == "__main__":
    from lyapunov import LyapunovCalculator, modify_init_conditions

    from scale import UnitConverter

    seed = random.randrange(sys.maxsize)
    rng = random.Random(seed)
    print("Seed was:", seed)
    # celestial_objects = [
    #     Mass(
    #         mass=1,
    #         velocity=np.array([0.4662036850, 0.4323657300, 0.0]),
    #         position=np.array([0.97000436, -0.24308753, 0.0]),
    #     ),
    #     Mass(
    #         mass=1,
    #         velocity=np.array([0.4662036850, 0.4323657300, 0.0]),
    #         position=np.array([-0.97000436, 0.24308753, 0.0]),
    #     ),
    #     Mass(
    #         mass=1,
    #         velocity=np.array([-0.93240737, -0.8643146, 0.0]),
    #         position=np.array([0.0, 0.0, 0.0]),
    #     ),
    # ]

    celestial_objects = [
        Mass(
            mass=100,
            velocity=np.array([-3.0, 0.0, 0.0]),
            position=np.array([-10.0, -10.0, -11.0]),
        ),
        Mass(
            mass=100,
            velocity=np.array([3.0, 0.0, 0.0]),
            position=np.array([0.0, 0.0, 0.0]),
        ),
        Mass(
            mass=0,
            velocity=np.array([3.0, 0.0, 0.0]),
            position=np.array([10.0, 14.0, 12.0]),
        ),
    ]

    conv = UnitConverter(celestial_objects)
    celestial_objects = conv.convert_initial_conditions()
    solvers = [RK4Solver, EulerSolver, VelocityVerletSolver]
    step_size = [0.02 / conv.time_sf]

    sim_solve = []
    sim_step_sizes = []

    for so in solvers:
        for st in step_size:
            sim_solve.append(so)
            sim_step_sizes.append(st)

    experiment = ExperimentManager(
        celestial_objects,
        150 / conv.time_sf,
        sim_solve,
        h_values=sim_step_sizes,
        conv=conv,
        scaled=True,
    )
    print(conv)
    print(experiment.get_lyapunov())
    experiment.solve_all()
    experiment.plot_energy_conservation(save=True)
    animate_orbits_3d(experiment)
