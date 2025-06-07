import random
from copy import deepcopy
from solvers import *
import numpy as np
from scale import UnitConverter


def generate_orbital_velocity(relative_position, central_mass, eccentricity=0.0):
    """
    Generate a velocity vector for a bound orbit using the vis-viva equation.

    Parameters:
        relative_position (np.ndarray): Position vector relative to central mass (shape: (3,))
        central_mass (float): Mass of the central object (in kg)
        eccentricity (float): 0 for circular, (0 < e < 1) for elliptical

    Returns:
        np.ndarray: Velocity vector (shape: (3,))
    """
    r_vec = np.array(relative_position, dtype=float)
    r = np.linalg.norm(r_vec)

    # Define semi-major axis from r and eccentricity
    # If e == 0 (circular), then a = r
    # Otherwise, we assume this point is periapsis: r = a * (1 - e)
    a = r / (1 - random.random()) if random.random() > 0 else r

    # Vis-viva equation
    speed = np.sqrt(g * central_mass * (2 / r - 1 / a))

    # Create a perpendicular tangent vector
    # If near z-axis, use x-axis to avoid degeneracy
    if np.allclose([r_vec[0], r_vec[1]], [0, 0], atol=1e-8):
        tangent = np.array([1.0, 0.0, 0.0])
    else:
        tangent = np.cross(r_vec, [0, 0, 1.0])
    tangent /= np.linalg.norm(tangent)

    return speed * tangent


def generate_unique_position(existing_masses, bounds, min_distance=1e7, max_tries=200):
    """Generate a position that is not too close to any existing mass."""
    for _ in range(max_tries):
        position = np.random.uniform(-bounds, bounds, 3)
        if all(
            np.linalg.norm(position - mass.position) >= min_distance
            for mass in existing_masses
        ):
            return position
    raise RuntimeError(
        f"Failed to place mass with sufficient spacing after {max_tries} tries"
    )


def generate_random_orbits(n_masses, n_central_masses):
    """
    Generate a mixture of 'central' bodies (stars) and 'orbiting' bodies (planets),
    all with positions/velocities that guarantee negative total energy.

    Returns:
        list of Mass: each has .mass (kg), .position (m), .velocity (m/s).
    """

    # ————————————————————————————————————————————————
    # STEP A: create the 'central' bodies (e.g. 1 or 2 stars)
    # ————————————————————————————————————————————————
    central_masses = []
    num_central_masses = min(n_masses, n_central_masses)

    for i in range(num_central_masses):
        # Pick a random “star” mass between 2e29 kg and 3e30 kg
        mass_value = random.uniform(2e29, 3e30)

        # Place each star separated by up to ~1 AU (1e11 m), at least 0.1 AU apart
        position = generate_unique_position(
            central_masses,
            bounds=1e11,  # ±1e11 m  (~0.7 AU)
            min_distance=1e10,  # 1e10 m (~0.07 AU)
        )

        if i == 0:
            # First star sits at rest in the inertial frame
            velocity = np.zeros(3)
        else:
            # For the 2nd (and beyond) star, pick one existing star to orbit:
            central = random.choice(central_masses)

            rel_pos = position - central.position
            # Compute velocity RELATIVE to 'central' via vis-viva
            rel_vel = generate_orbital_velocity(rel_pos, central.mass)

            # SHIFT back to inertial frame by adding central.velocity
            velocity = central.velocity + rel_vel

        central_masses.append(
            Mass(mass=mass_value, velocity=velocity, position=position)
        )

    # ————————————————————————————————————————————————
    # STEP B: create the 'orbiting' bodies (e.g. planets)
    # ————————————————————————————————————————————————
    orbiting_masses = []
    for _ in range(n_masses - num_central_masses):
        # Pick a “planet” mass between 1e21 kg and 1e25 kg
        mass_value = random.uniform(1e21, 1e25)

        # Place it between ~0.2 AU and ~1 AU from whichever star we pick
        position = generate_unique_position(
            central_masses + orbiting_masses,
            bounds=1e11,  # ±1e11 m (0.7 AU)
            min_distance=1e10,  # ensure ≥0.07 AU clearance
        )

        # Choose a star to orbit
        central = random.choice(central_masses)

        rel_pos = position - central.position
        rel_vel = generate_orbital_velocity(rel_pos, central.mass)

        # SHIFT from central frame → inertial frame
        velocity = central.velocity + rel_vel

        orbiting_masses.append(
            Mass(mass=mass_value, velocity=velocity, position=position)
        )

    return orbiting_masses + central_masses


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)
    n_orbits_per_mass = 10
    min_masses = 3
    max_num_masses = 20
    max_num_central_masses = 10
    time = 1e8
    breakdown_step_size = (2 ** -52) ** 0.2
    for num_masses in range(min_masses, max_num_masses + 1):
        for m in range(n_orbits_per_mass):
            for central_mass in range(1, min(m, max_num_central_masses + 1)):
                while True:
                    masses = generate_random_orbits(num_masses, central_mass)
                    conv = UnitConverter(masses)
                    masses = conv.convert_initial_conditions()
                    print("masses", masses)
                    print(conv)
                    solvers = [VelocityVerletSolver, EulerSolver, RK4Solver]
                    sim_solve = []
                    sim_step_sizes = []
                    smallest_step_size = np.log10(
                        breakdown_step_size * conv.time_sf
                        if conv
                        else breakdown_step_size
                    ) + 1
                    step_size = list(
                        np.logspace(
                            max(3, smallest_step_size), np.log10(time) - 1, num=20, dtype=float
                        )
                    )
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
                    if exp < 0.05:
                        break

                experiment.solve_all()
                experiment.export_experiment()
                print(
                    *[(sys.idx_energy_exceeded, sys) for sys in experiment.systems],
                    sep="\n",
                )
