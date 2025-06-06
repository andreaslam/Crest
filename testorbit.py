from solvers import *

from orbit import *

if __name__ == "__main__":
    seed = random.randrange(sys.maxsize)
    rng = random.Random(seed)
    print("Seed was:", seed)
    time = 1e7
    sim_steps = []
    num_masses = 5
    central_masses = 3

    masses = generate_random_orbits(num_masses, central_masses)

    experiment = ExperimentManager(
        masses,
        time,
        [VelocityVerletSolver],
        h_values=[100],
    )

    # print(experiment.get_lyapunov())

    for mass in masses:
        print(mass)

    experiment.solve_all()
    # animate_orbits_3d(experiment)
    # experiment.calculate_trajectory_deviance("mae", save=True)
    # experiment.calculate_trajectory_deviance("mse", save=True)
    experiment.plot_energy_conservation(save=True)
    experiment.plot_object_phase_space(save=True)
    # experiment.export_experiment()
