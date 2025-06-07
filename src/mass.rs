// src/mass.rs

use nalgebra::Vector3;
use std::fmt;

#[derive(Debug, Clone, Copy)]
pub struct Mass {
    pub mass: f64,
    pub position: Vector3<f64>,
    pub velocity: Vector3<f64>,
    pub acceleration: Vector3<f64>,
}

impl Mass {
    pub fn new(mass: f64, position: Vector3<f64>, velocity: Vector3<f64>) -> Self {
        Mass {
            mass,
            position,
            velocity,
            acceleration: Vector3::zeros(),
        }
    }
}

impl fmt::Display for Mass {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Mass(m={:.2e}, p=[{:.2e}, {:.2e}, {:.2e}], v=[{:.2e}, {:.2e}, {:.2e}])",
            self.mass,
            self.position.x,
            self.position.y,
            self.position.z,
            self.velocity.x,
            self.velocity.y,
            self.velocity.z
        )
    }
}
