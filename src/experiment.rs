use crate::mass::Mass;
use crate::solvers::System;
use chrono::Utc;
use serde::Serialize;
use std::error::Error;

#[derive(Serialize)]
struct ExperimentRecord {
    date: String,
    masses: String,
    initial_velocities: String,
    initial_positions: String,
    solver: String,
    n_steps: usize,
    step_size: f64,
    std_energy_loss: f64,
    execution_duration: f64,
    energy_thresholds: String,
    lyapunov: f64,
    notes: String,
}

pub fn export_experiment(
    system: &System,
    init_conditions: &[Mass],
    solver_name: &str,
    execution_duration: f64,
    lyapunov_exp: f64,
    notes: &str,
) -> Result<(), Box<dyn Error>> {
    let _initial_energy = system.total_energy[0];
    let energy_std_dev = {
        let mean = system.total_energy.iter().sum::<f64>() / system.total_energy.len() as f64;
        let variance = system
            .total_energy
            .iter()
            .map(|val| (*val - mean).powi(2))
            .sum::<f64>()
            / system.total_energy.len() as f64;
        variance.sqrt()
    };
    let mut energy_thresholds_vec: Vec<(f64, Option<f64>)> = system
        .idx_energy_exceeded
        .iter()
        .map(|(k, v)| (k.into_inner(), *v))
        .collect();
    energy_thresholds_vec
        .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let energy_thresholds_json = serde_json::to_string(&energy_thresholds_vec)?;
    let record = ExperimentRecord {
        date: Utc::now().to_rfc3339(),
        masses: format!(
            "{:?}",
            init_conditions.iter().map(|m| m.mass).collect::<Vec<_>>()
        ),
        initial_velocities: format!(
            "{:?}",
            init_conditions
                .iter()
                .map(|m| m.velocity)
                .collect::<Vec<_>>()
        ),
        initial_positions: format!(
            "{:?}",
            init_conditions
                .iter()
                .map(|m| m.position)
                .collect::<Vec<_>>()
        ),
        solver: solver_name.to_string(),
        n_steps: system.num_steps,
        step_size: system.h,
        std_energy_loss: energy_std_dev,
        execution_duration,
        energy_thresholds: energy_thresholds_json,
        lyapunov: lyapunov_exp,
        notes: notes.to_string(),
    };

    let file_path = "experiment_data/experiment.csv";
    let file_exists = std::path::Path::new(file_path).exists();
    let file = std::fs::OpenOptions::new()
        .append(true)
        .create(true)
        .open(file_path)?;

    let mut wtr = csv::WriterBuilder::new()
        .has_headers(!file_exists)
        .from_writer(file);

    wtr.serialize(record)?;
    wtr.flush()?;
    Ok(())
}

pub fn export_positions(system: &System, solver_name: &str) -> Result<(), Box<dyn Error>> {
    let positions_path = format!("experiment_data/{}_positions.csv", solver_name);
    let metadata_path = format!("experiment_data/{}_metadata.csv", solver_name);

    let mut wtr_meta = csv::Writer::from_path(metadata_path)?;
    wtr_meta.write_record(&["mass_id", "mass"])?;
    for (i, obj) in system.objs.iter().enumerate() {
        wtr_meta.write_record(&[i.to_string(), obj.mass.to_string()])?;
    }
    wtr_meta.flush()?;

    let mut wtr_pos = csv::Writer::from_path(positions_path)?;
    wtr_pos.write_record(&["mass_id", "x", "y", "z"])?;
    for (i, positions) in system.positions.iter().enumerate() {
        for pos in positions {
            wtr_pos.write_record(&[
                i.to_string(),
                pos.x.to_string(),
                pos.y.to_string(),
                pos.z.to_string(),
            ])?;
        }
    }
    wtr_pos.flush()?;

    Ok(())
}

pub fn calculate_trajectory_deviance(system1: &System, system2: &System) -> Vec<f64> {
    let num_masses = system1.objs.len();
    assert_eq!(num_masses, system2.objs.len());
    let num_steps = system1.positions[0].len();
    assert_eq!(num_steps, system2.positions[0].len());
    (0..num_steps)
        .map(|step| {
            let mut sum_sq_diff = 0.0;
            for mass in 0..num_masses {
                let pos1 = system1.positions[mass][step];
                let pos2 = system2.positions[mass][step];
                let diff = pos1 - pos2;
                sum_sq_diff += diff.norm_squared();
            }
            let mse: f64 = sum_sq_diff / num_masses as f64;
            mse.sqrt()
        })
        .collect()
}
