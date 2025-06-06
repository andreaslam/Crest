from solvers import *
from copy import deepcopy
from orbit import *

import numpy as np

if __name__ == "__main__":
    seed = random.randrange(sys.maxsize)
    rng = random.Random(seed)
    print("Seed was:", seed)

    time = 1e4

    sim_solve = []
    sim_steps = []

    step_size = list(np.logspace(-2, 1, num=10, dtype=float))
    print(step_size)
    masses = [
        Mass(
            mass=1,
            velocity=np.array([0.4662036850, 0.4323657300, 0.0]),
            position=np.array([0.97000436, -0.24308753, 0.0]),
        ),
        Mass(
            mass=1,
            velocity=np.array([0.4662036850, 0.4323657300, 0.0]),
            position=np.array([-0.97000436, 0.24308753, 0.0]),
        ),
        Mass(
            mass=1,
            velocity=np.array([-0.93240737, -0.8643146, 0.0]),
            position=np.array([0.0, 0.0, 0.0]),
        ),
    ]
    for _ in range(100000):
        solvers = [RK4Solver, VelocityVerletSolver, EulerSolver]
        for so in solvers:
            for st in step_size:
                sim_solve.append(deepcopy(so))
                sim_steps.append(deepcopy(st))
        experiment = ExperimentManager(
            modify_init_conditions(masses, threshold=0.001),
            time,
            sim_solve,
            h_values=sim_steps,
        )
        experiment.solve_all()
        experiment.export_experiment()
