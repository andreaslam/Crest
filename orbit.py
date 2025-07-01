import random
from copy import deepcopy
from solvers import *
from scale import UnitConverter
import numpy as np


def generate_orbital_velocity(
    relative_position,
    central_mass,
    eccentricity=0.0,
    use_random_eccentricity=True,
    random_eccentricity_range=(0.0, 0.75),
):
    """
    Generate a velocity vector for a two-body bound orbit using the vis-viva equation.

    This revised version generates a velocity vector perpendicular to the position
    vector but in a randomly oriented plane, providing a balance between purely
    planar and unstable random orbits.

    Parameters:
        relative_position (np.ndarray): Position vector relative to the central mass (m).
        central_mass (float): Mass of the central object (kg).
        eccentricity (float): Orbital eccentricity (0 for circular, <1 for elliptical).
        use_random_eccentricity (bool): If True, generate a random eccentricity.
        random_eccentricity_range (tuple): Range for random eccentricity.

    Returns:
        np.ndarray: Velocity vector (m/s).
    """
    r_vec = np.array(relative_position, dtype=float)
    r_norm = np.linalg.norm(r_vec)

    if r_norm == 0:
        return np.zeros(3)

    # Determine eccentricity
    e = (
        random.uniform(*random_eccentricity_range)
        if use_random_eccentricity
        else eccentricity
    )

    # We assume the given position 'r' is the periapsis (closest point)
    # where r_p = a * (1 - e), so the semi-major axis 'a' is r_norm / (1 - e).
    # This calculation ensures 'a' is positive for any valid eccentricity e < 1.
    if e >= 1.0:  # Ensure eccentricity is valid for a bound orbit
        e = 0.99
    a = r_norm / (1 - e)

    # Vis-viva equation to find the speed for a bound orbit
    speed_sq = g * central_mass * (2 / r_norm - 1 / a)
    if speed_sq < 0:
        # This can happen with extreme eccentricities, default to a circular orbit as a fallback
        a = r_norm
        speed_sq = g * central_mass * (2 / r_norm - 1 / a)

    speed = np.sqrt(speed_sq)

    # --- MODIFIED VELOCITY VECTOR LOGIC ---
    # To create a velocity vector, we need a direction tangent to the orbit
    # (and perpendicular to the position vector r_vec).
    #
    # Old Method Issues:
    # 1. Crossing with a fixed axis (e.g., [0,0,1]) -> All orbits are "boring" and co-planar.
    # 2. Crossing with a naive random vector -> Can be "unstable" if the random
    #    vector is nearly parallel to r_vec, leading to a near-zero cross product.
    #
    # New Method ("The Balance"):
    # Generate a random direction, and create a perpendicular vector via the cross
    # product. Handle the edge case where the random direction is parallel to r_vec.
    # This creates a stable, randomly oriented orbital plane for each body.

    # 1. Generate a random direction vector. Using np.random.randn(3) is ideal
    #    as it provides a statistically uniform distribution of directions.
    random_dir = np.random.randn(3)

    # 2. Create the tangent vector by crossing the position with the random direction.
    tangent = np.cross(r_vec, random_dir)

    # 3. Handle the edge case where the random vector is nearly parallel to the
    #    position vector. The cross product would be a near-zero vector.
    tangent_norm = np.linalg.norm(tangent)

    # Loop to ensure the generated tangent is valid.
    while tangent_norm < 1e-9:
        # If the vectors were parallel, generate a new random direction and retry.
        random_dir = np.random.randn(3)
        tangent = np.cross(r_vec, random_dir)
        tangent_norm = np.linalg.norm(tangent)

    # 4. Normalize the resulting tangent vector to get a unit direction vector.
    tangent /= tangent_norm
    # --- END OF MODIFIED LOGIC ---

    return speed * tangent


def generate_unique_position(existing_masses, bounds, min_distance=1e9, max_tries=500):
    """
    Generate a unique position for a new mass, ensuring it's not too close to existing ones.
    """
    for _ in range(max_tries):
        position = np.random.uniform(-bounds, bounds, 3)
        if not existing_masses or all(
            np.linalg.norm(position - mass.position) >= min_distance
            for mass in existing_masses
        ):
            return position
    raise RuntimeError(f"Failed to find a unique position after {max_tries} attempts.")


def generate_random_orbits(n_masses, n_central_masses):
    """
    Generate a set of initial conditions for N bodies that guarantees a gravitationally
    bound system (i.e., total energy is negative).

    Args:
        n_masses (int): Total number of masses in the system.
        n_central_masses (int): Number of heavier "central" bodies (e.g., stars).

    Returns:
        list of Mass: A list of Mass objects with properties ensuring a stable, bound system.
    """
    masses = []
    central_masses = []

    # Step 1: Generate initial positions and masses
    for i in range(n_masses):
        is_central = i < n_central_masses
        mass_value = (
            random.uniform(3e29, 4e29) if is_central else random.uniform(1e22, 5e24)
        )

        # Place central masses closer to the origin, orbiting bodies further out
        bounds = 50e10 if is_central else 1e11
        min_dist = 10e10 if is_central else 1e10

        position = generate_unique_position(
            masses, bounds=bounds, min_distance=min_dist
        )
        mass = Mass(mass=mass_value, position=position, velocity=np.zeros(3))
        masses.append(mass)
        if is_central:
            central_masses.append(mass)

    # Step 2: Assign velocities pairwise
    # The first central mass is the initial frame of reference
    if not central_masses:  # Handle case with no central masses
        return masses

    for i in range(len(masses)):
        # Skip the very first central mass, which acts as our initial reference point
        if masses[i] in central_masses and central_masses.index(masses[i]) == 0:
            continue

        # Each new body orbits a randomly chosen central body
        central_body = random.choice(central_masses)
        # Avoid a body orbiting itself
        if masses[i] is central_body:
            continue

        relative_pos = masses[i].position - central_body.position

        # Calculate velocity relative to the central body
        relative_vel = generate_orbital_velocity(
            relative_pos,
            central_body.mass,
            use_random_eccentricity=True,
            random_eccentricity_range=(0.0, 0.6),
        )

        # Set the absolute velocity
        masses[i].velocity = central_body.velocity + relative_vel

    # Step 3: Shift to the Barycenter (Center of Mass) Frame to guarantee stability
    total_mass = sum(m.mass for m in masses)
    if total_mass == 0:
        return masses  # Avoid division by zero

    # Calculate center of mass position and velocity
    com_pos = sum(m.mass * m.position for m in masses) / total_mass
    com_vel = sum(m.mass * m.velocity for m in masses) / total_mass

    # Adjust all positions and velocities to be relative to the barycenter
    for m in masses:
        m.position -= com_pos
        m.velocity -= com_vel

    # Step 4: Calculate total energy and ensure it's negative
    kinetic_energy = 0.5 * sum(m.mass * np.linalg.norm(m.velocity) ** 2 for m in masses)
    potential_energy = 0
    for i in range(len(masses)):
        for j in range(i + 1, len(masses)):
            r_ij = np.linalg.norm(masses[i].position - masses[j].position)
            if r_ij > 0:
                potential_energy -= (g * masses[i].mass * masses[j].mass) / r_ij

    total_energy = kinetic_energy + potential_energy

    # Step 5: If total energy is positive, scale down kinetic energy to make it negative
    if total_energy >= 0:
        # To guarantee a bound system, the total energy must be negative (KE < |PE|).
        # We find a scaling factor to reduce the kinetic energy to just below the
        # magnitude of the potential energy, with a 0.9 safety margin.
        if kinetic_energy > 0:
            scaling_factor = np.sqrt(abs(potential_energy) / kinetic_energy) * 0.9
            for m in masses:
                m.velocity *= scaling_factor
            print(
                f"System energy was positive ({total_energy:.2e}). Velocities scaled by {scaling_factor:.2f} to guarantee bound orbit."
            )

    return masses


MACHINE_EPS = 2**-52

if __name__ == "__main__":
    n_orbits_per_mass = 10
    min_masses = 3
    max_num_masses = 10
    max_num_central_masses = 10
    time = 60 * 60 * 24 * 365
    breakdown_step_size = MACHINE_EPS**0.2
    for num_masses in range(min_masses, max_num_masses + 1):
        for m in range(n_orbits_per_mass):
            for central_mass in range(1, min(m, max_num_central_masses + 1)):
                masses = generate_random_orbits(num_masses, central_mass)
                conv = UnitConverter(masses)
                masses = conv.convert_initial_conditions()
                print("masses", masses)
                print(conv)
                solvers = [VelocityVerletSolver, EulerSolver, RK4Solver]
                sim_solve = []
                sim_step_sizes = []
                smallest_step_size = (
                    np.log10(
                        breakdown_step_size * conv.time_sf
                        if conv.time_sf
                        else breakdown_step_size
                    )
                )
                print("smallest_step_size", smallest_step_size)
                step_size = list(
                    np.logspace(
                        max(3, smallest_step_size),
                        np.log10(time) - 1,
                        num=40,
                        dtype=float,
                    )
                )
                print("step_size", step_size)
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
                experiment.export_experiment()
                print(
                    *[(sys.idx_energy_exceeded, sys) for sys in experiment.systems],
                    sep="\n",
                )
