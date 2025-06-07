use crate::mass::Mass;
use crate::solvers::System;
use nalgebra::Vector3;
use rand::Rng;
use rand_distr::Uniform;

pub fn modify_init_conditions(init_conditions: &mut [Mass], threshold: f64) {
    let mut rng = rand::rng();
    let range = Uniform::new_inclusive(-threshold, threshold).unwrap();
    for obj in init_conditions.iter_mut() {
        let pos_perturbation =
            Vector3::new(rng.sample(&range), rng.sample(&range), rng.sample(&range));
        let vel_perturbation =
            Vector3::new(rng.sample(&range), rng.sample(&range), rng.sample(&range));
        obj.position += pos_perturbation;
        obj.velocity += vel_perturbation;
    }
}

pub struct LyapunovCalculator {
    system1: System,
    system2: System,
}

impl LyapunovCalculator {
    pub fn new(system1: System, system2: System) -> Self {
        assert_eq!(system1.h, system2.h, "Step sizes must match");
        assert_eq!(
            system1.num_steps, system2.num_steps,
            "Number of steps must match"
        );
        LyapunovCalculator { system1, system2 }
    }

    pub fn compute_separation(&self) -> Vec<f64> {
        let num_steps = self.system1.positions[0].len();
        let num_masses = self.system1.objs.len();
        let mut separations = vec![0.0; num_steps];

        for step in 0..num_steps {
            let mut total_sep = 0.0;
            for mass in 0..num_masses {
                let pos1 = self.system1.positions[mass][step];
                let pos2 = self.system2.positions[mass][step];
                let sep = (pos1 - pos2).norm();
                total_sep += sep;
            }
            separations[step] = total_sep / num_masses as f64;
        }
        separations
    }

    pub fn calculate_lyapunov_exponent(&self) -> f64 {
        let separations = self.compute_separation();
        let initial_sep = separations[0];
        assert!(
            initial_sep > 0.0,
            "Initial separation must be greater than zero"
        );

        let num_steps = separations.len();
        let total_time = num_steps as f64 * self.system1.h;
        let sum: f64 = separations
            .iter()
            .skip(1)
            .map(|&sep| (sep / initial_sep).ln())
            .sum();
        sum / total_time / (num_steps - 1) as f64
    }
}
