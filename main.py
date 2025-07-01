# import random
# from orbit import MACHINE_EPS, generate_random_orbits
# from scale import UnitConverter
# from solvers import (
#     ExperimentManager,
#     ReboundSolver,
#     VelocityVerletSolver,
#     RK4Solver,
#     animate_orbits_3d,
# )
# import numpy as np


# def main():
#     random.seed(42)
#     np.random.seed(42)
#     masses = generate_random_orbits(3, 1)
#     conv = UnitConverter(masses)
#     masses = conv.convert_initial_conditions()
#     print(conv)
#     print(masses)
#     # exp_time = 1e10 / conv.time_sf
#     exp_time = 1e7 / conv.time_sf
#     h = list(
#         np.logspace(np.log10(MACHINE_EPS**0.2 * conv.time_sf), 6, num=10, dtype=float)
#         / conv.time_sf
#     )
#     # h = 1e3 / conv.time_sf
#     experiment = ExperimentManager(
#         masses,
#         exp_time,
#         [VelocityVerletSolver],
#         h,
#         scaled=True,
#         conv=conv,
#     )
#     print(np.logspace(3, 6, num=10, dtype=float))
#     experiment.solve_all()
#     print([s.idx_energy_exceeded for s in experiment.systems])
#     experiment.export_experiment()
#     # animate_orbits_3d(experiment)
#     # experiment.plot_energy_conservation()
#     # experiment.calculate_trajectory_deviance("mae")
#     # experiment.export_positions()


# if __name__ == "__main__":
#     main()

import random
from orbit import MACHINE_EPS, generate_random_orbits
from scale import UnitConverter
from solvers import (
    ExperimentManager,
    ReboundSolver,
    VelocityVerletSolver,
    RK4Solver,
    animate_orbits_3d,
)
import numpy as np


def main():
    random.seed(42)
    np.random.seed(42)
    masses = generate_random_orbits(3, 1)
    conv = UnitConverter(masses)
    masses = conv.convert_initial_conditions()
    print(conv)
    print(masses)
    # exp_time = 1e10 / conv.time_sf
    exp_time = 1e7 / conv.time_sf
    # h = list(
    #     np.logspace(np.log10(MACHINE_EPS**0.2 * conv.time_sf), 6, num=10, dtype=float)
    #     / conv.time_sf
    # )
    h = [3000 / conv.time_sf] * 2
    experiment = ExperimentManager(
        masses,
        exp_time,
        [VelocityVerletSolver, ReboundSolver],
        h,
        scaled=True,
        conv=conv,
    )
    print(np.logspace(3, 6, num=10, dtype=float))
    experiment.solve_all()
    print(
        experiment.calculate_trajectory_deviance_pair(
            "mae", [s.positions for s in experiment.systems], metric_score_only=True
        )
        * experiment.conv.dist_sf
        / 1000
    )


if __name__ == "__main__":
    main()
