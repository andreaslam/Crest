// src/scale.rs

use crate::mass::Mass;
use itertools::{izip, Itertools};
use nalgebra::Vector3;

const G: f64 = 6.6743e-11;

#[derive(Debug, Clone)]
pub struct UnitConverter {
    pub mass_sf: f64,
    pub dist_sf: f64,
    pub time_sf: f64,
    raw_initial_masses: Vec<f64>,
    raw_initial_positions: Vec<Vector3<f64>>,
    raw_initial_velocities: Vec<Vector3<f64>>,
}

impl UnitConverter {
    pub fn new(system: &[Mass]) -> Self {
        let raw_initial_masses = system.iter().map(|obj| obj.mass).collect();
        let raw_initial_positions = system.iter().map(|obj| obj.position).collect();
        let raw_initial_velocities = system.iter().map(|obj| obj.velocity).collect();

        let mut converter = Self {
            mass_sf: 1.0,
            dist_sf: 1.0,
            time_sf: 1.0,
            raw_initial_masses,
            raw_initial_positions,
            raw_initial_velocities,
        };
        converter.compute_scale_factors();
        converter
    }

    fn compute_scale_factors(&mut self) {
        self.mass_sf =
            self.raw_initial_masses.iter().sum::<f64>() / self.raw_initial_masses.len() as f64;

        let seps: Vec<f64> = self
            .raw_initial_positions
            .iter()
            .combinations(2)
            .map(|p| (p[0] - p[1]).norm())
            .collect();

        self.dist_sf = seps.iter().sum::<f64>() / seps.len() as f64;
        self.time_sf = (self.dist_sf.powi(3) / (G * self.mass_sf)).sqrt();
    }

    pub fn convert_initial_conditions(&self) -> Vec<Mass> {
        let scaled_masses = self.raw_initial_masses.iter().map(|&m| m / self.mass_sf);
        let scaled_positions = self.raw_initial_positions.iter().map(|&p| p / self.dist_sf);
        let scaled_velocities = self
            .raw_initial_velocities
            .iter()
            .map(|&v| v * self.time_sf / self.dist_sf);

        izip!(scaled_masses, scaled_positions, scaled_velocities)
            .map(|(m, p, v)| Mass::new(m, p, v))
            .collect()
    }

    pub fn convert_energy_to_joules(&self, energy: f64) -> f64 {
        energy * (G * self.mass_sf.powi(2)) / self.dist_sf
    }
}
