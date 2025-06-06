import random
from orbit import generate_random_orbits
from scale import UnitConverter
from solvers import (
    ExperimentManager,
    EulerSolver,
    VelocityVerletSolver,
    RK4Solver,
    animate_orbits_3d,
)
import numpy as np


def main():
    # random.seed(42)
    # np.random.seed(42)
    masses = generate_random_orbits(3, 1)
    conv = UnitConverter(masses)
    masses = conv.convert_initial_conditions()
    print(conv)
    print(masses)
    exp_time = 1e8 / conv.time_sf
    h = [10e4 / conv.time_sf]
    experiment = ExperimentManager(
        masses,
        exp_time,
        [EulerSolver, VelocityVerletSolver, RK4Solver],
        h,
        scaled=True,
        conv=conv,
    )

    experiment.solve_all()
    animate_orbits_3d(experiment)
    experiment.plot_energy_conservation(show=True)
    # experiment.plot_object_phase_space()


if __name__ == "__main__":
    main()
