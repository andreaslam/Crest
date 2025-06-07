use crate::mass::Mass;
use indicatif::ProgressBar;
use itertools::izip;
use nalgebra::Vector3;
use ordered_float::OrderedFloat;
use std::collections::HashMap;
use std::time::Instant;

const G: f64 = 6.6743e-11;
#[derive(Clone)]
pub struct System {
    pub objs: Vec<Mass>,
    pub h: f64,
    simulation_length: f64,
    pub num_steps: usize,
    pub positions: Vec<Vec<Vector3<f64>>>,
    pub total_energy: Vec<f64>,
    scaled: bool,
    pub energy_thresholds: Vec<f64>,
    pub idx_energy_exceeded: HashMap<OrderedFloat<f64>, Option<f64>>,
    initial_energy: f64,
}

impl System {
    pub fn new(objs: Vec<Mass>, simulation_length: f64, h: f64, scaled: bool) -> Self {
        let num_steps = (simulation_length / h).ceil() as usize;
        let energy_thresholds = vec![
            0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 10.0,
        ];
        let idx_energy_exceeded = energy_thresholds
            .iter()
            .map(|&t| (OrderedFloat(t), None))
            .collect();
        let initial_energy = calculate_total_energy(&objs, scaled);
        let num_objs = objs.len();
        Self {
            objs,
            h,
            simulation_length,
            num_steps,
            positions: vec![Vec::with_capacity(num_steps); num_objs],
            total_energy: Vec::with_capacity(num_steps),
            scaled,
            initial_energy,
            energy_thresholds,
            idx_energy_exceeded,
        }
    }

    fn get_acceleration(&self, obj_idx: usize) -> Vector3<f64> {
        let obj_pos = self.objs[obj_idx].position;
        let obj_mass = self.objs[obj_idx].mass;
        let grav_const = if self.scaled { 1.0 } else { G };

        self.objs
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != obj_idx)
            .map(|(_, other)| {
                let r_vec = other.position - obj_pos;
                let r = r_vec.norm();
                if r > 1e-9 {
                    grav_const * other.mass * r_vec / r.powi(3)
                } else {
                    Vector3::zeros()
                }
            })
            .sum()
    }

    pub fn calculate_total_energy(&self) -> f64 {
        calculate_total_energy(&self.objs, self.scaled)
    }

    fn record_state(&mut self, step: usize) {
        for (i, obj) in self.objs.iter().enumerate() {
            self.positions[i].push(obj.position);
        }
        let current_energy = self.calculate_total_energy();
        self.total_energy.push(current_energy);
        for &t in &self.energy_thresholds {
            if let Some(val) = self.idx_energy_exceeded.get_mut(&OrderedFloat(t)) {
                if val.is_none() {
                    if current_energy > self.initial_energy * (1.0 + t)
                        || current_energy < self.initial_energy * (1.0 - t)
                    {
                        let fraction = step as f64 / self.num_steps as f64;
                        *val = Some(fraction);
                    }
                }
            }
        }
    }
}

fn calculate_total_energy(objs: &[Mass], scaled: bool) -> f64 {
    let kinetic_energy = objs
        .iter()
        .map(|obj| 0.5 * obj.mass * obj.velocity.norm_squared())
        .sum::<f64>();
    let mut potential_energy = 0.0;
    let grav_const = if scaled { 1.0 } else { G };
    for i in 0..objs.len() {
        for j in (i + 1)..objs.len() {
            let r = (objs[i].position - objs[j].position).norm();
            if r > 1e-9 {
                potential_energy -= grav_const * objs[i].mass * objs[j].mass / r;
            }
        }
    }
    kinetic_energy + potential_energy
}

pub trait Solver {
    fn solve(&self, system: &mut System) -> f64;
    fn name(&self) -> String;
}

pub struct EulerSolver;
impl Solver for EulerSolver {
    fn name(&self) -> String {
        "Euler".to_string()
    }
    fn solve(&self, system: &mut System) -> f64 {
        let start = Instant::now();
        let pb = ProgressBar::new(system.num_steps as u64);
        pb.set_message(format!("Solving with {}", self.name()));
        for step in 0..system.num_steps {
            for i in 0..system.objs.len() {
                let acc = system.get_acceleration(i);
                system.objs[i].velocity += system.h * acc;
            }
            let velocities: Vec<_> = system.objs.iter().map(|obj| obj.velocity).collect();
            for (i, vel) in velocities.iter().enumerate() {
                system.objs[i].position += system.h * *vel;
            }
            system.record_state(step);
            pb.inc(1);
        }
        pb.finish();
        start.elapsed().as_secs_f64()
    }
}

pub struct VelocityVerletSolver;
impl Solver for VelocityVerletSolver {
    fn name(&self) -> String {
        "VelocityVerlet".to_string()
    }
    fn solve(&self, system: &mut System) -> f64 {
        let start = Instant::now();
        let pb = ProgressBar::new(system.num_steps as u64);
        pb.set_message(format!("Solving with {}", self.name()));
        for i in 0..system.objs.len() {
            system.objs[i].acceleration = system.get_acceleration(i);
        }
        for step in 0..system.num_steps {
            let h = system.h;
            let h2_half = 0.5 * h * h;
            for obj in &mut system.objs {
                obj.position += obj.velocity * h + obj.acceleration * h2_half;
            }
            let old_accels: Vec<_> = system.objs.iter().map(|o| o.acceleration).collect();
            let new_accels: Vec<_> = (0..system.objs.len())
                .map(|i| system.get_acceleration(i))
                .collect();
            for i in 0..system.objs.len() {
                system.objs[i].velocity += 0.5 * h * (old_accels[i] + new_accels[i]);
                system.objs[i].acceleration = new_accels[i];
            }
            system.record_state(step);
            pb.inc(1);
        }
        pb.finish();
        start.elapsed().as_secs_f64()
    }
}

pub struct RK4Solver;
impl Solver for RK4Solver {
    fn name(&self) -> String {
        "RK4".to_string()
    }

    fn solve(&self, system: &mut System) -> f64 {
        let start = Instant::now();
        let pb = ProgressBar::new(system.num_steps as u64);
        pb.set_message(format!("Solving with {}", self.name()));

        let h = system.h;
        let h2 = h / 2.0;
        let h6 = h / 6.0;
        let n = system.objs.len();
        let masses: Vec<f64> = system.objs.iter().map(|o| o.mass).collect();

        for step in 0..system.num_steps {
            let pos0: Vec<_> = system.objs.iter().map(|o| o.position).collect();
            let vel0: Vec<_> = system.objs.iter().map(|o| o.velocity).collect();

            // k1
            let a1 = compute_all_accelerations(&pos0, &masses, system.scaled);
            let k1_pos = vel0.clone();
            let k1_vel = a1;

            // k2
            let pos_k2: Vec<_> = izip!(&pos0, &k1_pos).map(|(p, v)| p + h2 * v).collect();
            let vel_k2: Vec<_> = izip!(&vel0, &k1_vel).map(|(v, a)| v + h2 * a).collect();
            let a2 = compute_all_accelerations(&pos_k2, &masses, system.scaled);
            let k2_pos = vel_k2.clone();
            let k2_vel = a2;

            // k3
            let pos_k3: Vec<_> = izip!(&pos0, &k2_pos).map(|(p, v)| p + h2 * v).collect();
            let vel_k3: Vec<_> = izip!(&vel0, &k2_vel).map(|(v, a)| v + h2 * a).collect();
            let a3 = compute_all_accelerations(&pos_k3, &masses, system.scaled);
            let k3_pos = vel_k3.clone();
            let k3_vel = a3;

            // k4
            let pos_k4: Vec<_> = izip!(&pos0, &k3_pos).map(|(p, v)| p + h * v).collect();
            let vel_k4: Vec<_> = izip!(&vel0, &k3_vel).map(|(v, a)| v + h * a).collect();
            let a4 = compute_all_accelerations(&pos_k4, &masses, system.scaled);
            let k4_pos = vel_k4;
            let k4_vel = a4;

            for i in 0..n {
                system.objs[i].position = pos0[i]
                    + h6 * (k1_pos[i] + 2.0 * k2_pos[i] + 2.0 * k3_pos[i] + k4_pos[i]);
                system.objs[i].velocity = vel0[i]
                    + h6 * (k1_vel[i] + 2.0 * k2_vel[i] + 2.0 * k3_vel[i] + k4_vel[i]);
            }

            system.record_state(step);
            pb.inc(1);
        }

        pb.finish();
        start.elapsed().as_secs_f64()
    }
}

fn compute_all_accelerations(
    positions: &[Vector3<f64>],
    masses: &[f64],
    scaled: bool,
) -> Vec<Vector3<f64>> {
    let grav_const = if scaled { 1.0 } else { G };
    let mut accelerations = vec![Vector3::zeros(); positions.len()];
    for i in 0..positions.len() {
        for j in 0..positions.len() {
            if i == j {
                continue;
            }
            let r_vec = positions[j] - positions[i];
            let r_sq = r_vec.norm_squared();
            if r_sq > 1e-18 {
                let r = r_sq.sqrt();
                accelerations[i] += grav_const * masses[j] * r_vec / r.powi(3);
            }
        }
    }
    accelerations
}