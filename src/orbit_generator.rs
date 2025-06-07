// src/orbit_generator.rs

use crate::mass::Mass;
use nalgebra::Vector3;
use rand::Rng;
use rand_distr::Uniform;

const G: f64 = 6.6743e-11;

/// Generates a velocity vector for a bound orbit using the vis-viva equation.
fn generate_orbital_velocity(relative_position: Vector3<f64>, central_mass: f64) -> Vector3<f64> {
    let mut rng = rand::rng();
    let r_vec = relative_position;
    let r = r_vec.norm();

    // Assume this point is periapsis for simplicity, similar to the Python version
    let a = r / (1.0 - rng.random::<f64>());

    let speed = (G * central_mass * (2.0 / r - 1.0 / a)).sqrt();

    let tangent = if r_vec.xy().norm_squared() < 1e-16 {
        Vector3::x()
    } else {
        r_vec.cross(&Vector3::z())
    };

    speed * tangent.normalize()
}

/// Generate a position that is not too close to any existing mass.
fn generate_unique_position(
    existing_masses: &[Mass],
    bounds: f64,
    min_distance: f64,
) -> Option<Vector3<f64>> {
    let mut rng = rand::rng();
    let range = Uniform::new_inclusive(-bounds, bounds).unwrap();
    for _ in 0..200 {
        let position = Vector3::new(rng.sample(&range), rng.sample(&range), rng.sample(&range));
        if existing_masses
            .iter()
            .all(|mass| (position - mass.position).norm() >= min_distance)
        {
            return Some(position);
        }
    }
    None
}

/// Generates a complete N-body system with central and orbiting bodies.
pub fn generate_random_orbits(n_masses: u32, n_central_masses: u32) -> Option<Vec<Mass>> {
    let mut rng = rand::rng();
    let mut central_masses = Vec::new();

    // Step A: Create central bodies (stars)
    for i in 0..n_central_masses {
        let mass_value = rng.random_range(2e29..3e30);
        let position = generate_unique_position(&central_masses, 1e11, 1e10)?;

        let velocity = if i == 0 {
            Vector3::zeros()
        } else {
            let central_idx = rng.random_range(0..central_masses.len());
            let central = &central_masses[central_idx];
            let rel_pos = position - central.position;
            let rel_vel = generate_orbital_velocity(rel_pos, central.mass);
            central.velocity + rel_vel
        };

        central_masses.push(Mass::new(mass_value, position, velocity));
    }

    // Step B: Create orbiting bodies (planets)
    let mut orbiting_masses = Vec::new();
    for _ in 0..(n_masses - n_central_masses) {
        let mass_value = rng.random_range(1e21..1e25);
        let all_masses: Vec<Mass> = central_masses
            .iter()
            .chain(orbiting_masses.iter())
            .cloned()
            .collect();
        let position = generate_unique_position(&all_masses, 1e11, 1e10)?;

        let central_idx = rng.random_range(0..central_masses.len());
        let central = &central_masses[central_idx];

        let rel_pos = position - central.position;
        let rel_vel = generate_orbital_velocity(rel_pos, central.mass);
        let velocity = central.velocity + rel_vel;

        orbiting_masses.push(Mass::new(mass_value, position, velocity));
    }

    central_masses.append(&mut orbiting_masses);
    Some(central_masses)
}
