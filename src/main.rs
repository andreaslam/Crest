use csv;
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;

const G: f64 = 6.67408e-11;

#[inline(always)]
fn norm(array: &[f64; 3]) -> f64 {
    array.iter().map(|x| x.powi(2)).sum::<f64>().sqrt()
}

#[derive(Clone)]
struct Mass {
    mass: f64,
    velocity: [f64; 3],
    position: [f64; 3],
    acceleration: [f64; 3],
}

impl Mass {
    fn new(mass: f64, velocity: [f64; 3], position: [f64; 3]) -> Self {
        Self {
            mass,
            velocity,
            position,
            acceleration: [0.0; 3],
        }
    }
}

struct System {
    masses: Vec<Mass>,
    num_steps: usize,
    h: f64,
    energy_check_interval: usize,
    kinetic_energy_vec: Vec<f64>,
    potential_energy_vec: Vec<f64>,
    total_energy_vec: Vec<f64>,
    history: Vec<Vec<(usize, f64, [f64; 3], [f64; 3])>>, // Include mass in history
}

impl System {
    fn new(masses: Vec<Mass>, num_steps: usize, h: f64, energy_check_interval: usize) -> Self {
        Self {
            masses,
            num_steps,
            h,
            energy_check_interval,
            kinetic_energy_vec: Vec::new(),
            potential_energy_vec: Vec::new(),
            total_energy_vec: Vec::new(),
            history: Vec::new(),
        }
    }

    fn calculate_acceleration(&self, index: usize) -> [f64; 3] {
        let mut accel = [0.0; 3];
        let current_mass = &self.masses[index];

        for (j, other) in self.masses.iter().enumerate() {
            if index != j {
                let diff = [
                    other.position[0] - current_mass.position[0],
                    other.position[1] - current_mass.position[1],
                    other.position[2] - current_mass.position[2],
                ];
                let r = norm(&diff);

                if r > 0.0 {
                    let factor = G * other.mass / (r * r * r);
                    accel[0] += diff[0] * factor;
                    accel[1] += diff[1] * factor;
                    accel[2] += diff[2] * factor;
                }
            }
        }

        accel
    }

    fn verlet_step(&mut self) {
        let h = self.h;

        // Step 1: Update positions
        self.masses.par_iter_mut().for_each(|mass| {
            let prev_acceleration = mass.acceleration;
            mass.position = [
                mass.position[0] + mass.velocity[0] * h + prev_acceleration[0] * (h * h / 2.0),
                mass.position[1] + mass.velocity[1] * h + prev_acceleration[1] * (h * h / 2.0),
                mass.position[2] + mass.velocity[2] * h + prev_acceleration[2] * (h * h / 2.0),
            ];
        });

        // Step 2: Recalculate accelerations in parallel
        let new_accelerations: Vec<[f64; 3]> = (0..self.masses.len())
            .into_par_iter()
            .map(|i| self.calculate_acceleration(i))
            .collect();

        // Step 3: Update velocities
        self.masses
            .iter_mut()
            .zip(new_accelerations.into_iter())
            .for_each(|(mass, new_accel)| {
                let prev_acceleration = mass.acceleration;
                mass.velocity = [
                    mass.velocity[0] + (new_accel[0] + prev_acceleration[0]) * (h / 2.0),
                    mass.velocity[1] + (new_accel[1] + prev_acceleration[1]) * (h / 2.0),
                    mass.velocity[2] + (new_accel[2] + prev_acceleration[2]) * (h / 2.0),
                ];
                mass.acceleration = new_accel;
            });
    }

    fn calculate_kinetic_energy(&self) -> f64 {
        self.masses
            .par_iter()
            .map(|mass| {
                0.5 * mass.mass
                    * (mass.velocity[0].powi(2)
                        + mass.velocity[1].powi(2)
                        + mass.velocity[2].powi(2))
            })
            .sum()
    }

    fn calculate_potential_energy(&self) -> f64 {
        (0..self.masses.len())
            .into_par_iter()
            .flat_map(|i| {
                (i + 1..self.masses.len()).into_par_iter().map(move |j| {
                    let diff = [
                        self.masses[i].position[0] - self.masses[j].position[0],
                        self.masses[i].position[1] - self.masses[j].position[1],
                        self.masses[i].position[2] - self.masses[j].position[2],
                    ];
                    let r = norm(&diff);
                    if r > 0.0 {
                        -G * self.masses[i].mass * self.masses[j].mass / r
                    } else {
                        0.0
                    }
                })
            })
            .sum()
    }

    fn solve(&mut self) {
        let pb = ProgressBar::new(self.num_steps as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template(
                    "{spinner:.green} [{elapsed_precise}] [{bar:40.cyan/blue}] {pos}/{len} ({eta})",
                )
                .unwrap()
                .progress_chars("#>-"),
        );

        for step in 0..self.num_steps {
            if step % 100 == 0 {
                pb.set_position(step as u64);
            }

            self.verlet_step();

            // Record positions, velocities, and masses at this step
            let snapshot = self
                .masses
                .iter()
                .enumerate()
                .map(|(i, m)| (i, m.mass, m.position, m.velocity)) // Include mass here
                .collect();
            self.history.push(snapshot);

            if step % self.energy_check_interval == 0 && step != 0 {
                let ke = self.calculate_kinetic_energy();
                let pe = self.calculate_potential_energy();
                let te = ke + pe;

                // Store energies in vectors
                self.kinetic_energy_vec.push(ke);
                self.potential_energy_vec.push(pe);
                self.total_energy_vec.push(te);
            }
        }

        pb.finish_with_message("Simulation complete");
    }

    fn save_to_csv(&self, filename: &str) {
        let mut wtr = csv::Writer::from_path(filename).unwrap();

        wtr.write_record(&[
            "Time Step",
            "Object",
            "Mass",
            "Position X",
            "Position Y",
            "Position Z",
            "Velocity X",
            "Velocity Y",
            "Velocity Z",
        ])
        .unwrap();

        for (step, snapshot) in self.history.iter().enumerate() {
            for &(i, mass, position, velocity) in snapshot.iter() {
                wtr.write_record(&[
                    step.to_string(),        // Time Step
                    i.to_string(),           // Object index
                    mass.to_string(),        // Mass
                    position[0].to_string(), // Position X
                    position[1].to_string(), // Position Y
                    position[2].to_string(), // Position Z
                    velocity[0].to_string(), // Velocity X
                    velocity[1].to_string(), // Velocity Y
                    velocity[2].to_string(), // Velocity Z
                ])
                .unwrap();
            }
        }

        wtr.flush().unwrap();
    }
}

fn main() {
    let masses = vec![
        Mass::new(100000e3, [-0.0001, -0.0, 0.0], [-10.0, 10.0, 10.0]),
        Mass::new(1500e3, [0.005, -0.01, -0.002], [-50.0, 7.0, 0.0]),
        Mass::new(10e3, [0.0, 0.01, 0.0], [25.0, -75.0, -20.2]),
        Mass::new(500e3, [0.0, 0.01, 0.0], [-100.0, -0.35, 0.2]),
        Mass::new(50e3, [0.0, -0.001, 0.001], [240.3, 2.5, -3.2]),
        Mass::new(100e3, [0.0, 0.0, 0.001], [-90.2, 210.3, -25.0]),
        Mass::new(10e3, [-0.0005, 0.0001, -0.005], [-150.0, 200.0, 200.0]),
        Mass::new(13e3, [0.0005, -0.0001, -0.005], [50.0, -20.0, 200.0]),
        Mass::new(25e3, [-0.0005, 0.0001, 0.005], [-200.0, 230.0, -200.0]),
    ];

    let mut system = System::new(masses, 500000, 1.0, 100);

    system.solve();
    system.save_to_csv("nbody_simulation.csv");
}
