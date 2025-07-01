from solvers import *
from scale import UnitConverter
from orbit import generate_random_orbits

MACHINE_EPS = 2**-52

if __name__ == "__main__":
    n_orbits_per_mass = 10
    min_masses = 3
    max_num_masses = 20
    max_num_central_masses = 10
    time = 1e7
    breakdown_step_size = MACHINE_EPS**0.2
    for num_masses in range(min_masses, max_num_masses + 1):
        for m in range(n_orbits_per_mass):
            for central_mass in range(1, min(m, max_num_central_masses + 1)):
                masses = generate_random_orbits(num_masses, central_mass)
                conv = UnitConverter(masses)
                masses = conv.convert_initial_conditions()
                print("masses", masses)
                print(conv)
                solvers = [RK4Solver, EulerSolver, VelocityVerletSolver, ReboundSolver]
                sim_solve = []
                sim_step_sizes = []
                # smallest_step_size = (
                #     np.log10(
                #         breakdown_step_size * conv.time_sf
                #         if conv.time_sf
                #         else breakdown_step_size
                #     )
                #     + 1
                # )
                # step_size = list(
                #     np.logspace(
                #         max(3, smallest_step_size),
                #         np.log10(time) - 1,
                #         num=3,
                #         dtype=float,
                #     )
                # )
                step_size = [1000]
                for so in solvers:
                    for st in step_size:
                        sim_solve.append(deepcopy(so))
                        sim_step_sizes.append(deepcopy(st / conv.time_sf))
                print("time", time / conv.time_sf)
                print("sim_step_sizes", sim_step_sizes)
                experiment = ExperimentManager(
                    masses,
                    time / conv.time_sf,
                    sim_solve,
                    h_values=sim_step_sizes,
                    scaled=True,
                    conv=conv,
                )
                exp = experiment.get_lyapunov()
                print("Lyapunov exponent:", exp)
                experiment.solve_all()
                experiment.calculate_trajectory_deviance("mae", save=True, show=True)
                # animate_orbits_3d(experiment)
