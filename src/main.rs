use rayon::prelude::*;
use std::error::Error;
use Crest::experiment;
use Crest::lyapunov::{modify_init_conditions, LyapunovCalculator};
use Crest::orbit_generator;
use Crest::solvers::{EulerSolver, RK4Solver, Solver, System, VelocityVerletSolver};

fn logspace(log_start: f64, log_end: f64, num: usize) -> Vec<f64> {
    let step = (log_end - log_start) / (num - 1) as f64;
    (0..num).map(|i| 10f64.powf(log_start + i as f64 * step)).collect()
}

fn main() -> Result<(), Box<dyn Error>> {
    let n_masses = 3;
    let n_central_masses = 1;
    let sim_time = 1e9;

    loop {
        let init_conditions = orbit_generator::generate_random_orbits(n_masses, n_central_masses)
            .expect("Failed to generate orbits.");
        println!("Generated {} masses for simulation.", init_conditions.len());

        let ss = logspace(3.0, 3.1, 10);

        ss.par_iter().for_each(|&step_size| {
            // Ensure all this work is thread-safe and isolated
            let solvers: Vec<Box<dyn Solver + Send + Sync>> = vec![
                Box::new(EulerSolver),
                Box::new(VelocityVerletSolver),
                Box::new(RK4Solver),
            ];

            println!("\n[Step Size = {:.3e}] Running {} solvers in sequence...", step_size, solvers.len());

            let results: Vec<(String, f64, System)> = solvers
                .iter()
                .map(|solver| {
                    println!("   [Thread-{:?}] Solver: {}", std::thread::current().id(), solver.name());
                    let mut system = System::new(init_conditions.clone(), sim_time, step_size, false);
                    let duration = solver.solve(&mut system);
                    (solver.name().to_string(), duration, system)
                })
                .collect();

            for (solver_name, duration, system) in results {
                println!("\n----- Results for {} (dt = {:.3e}) -----", solver_name, step_size);
                println!("Execution finished in {:.2} seconds.", duration);

                if let (Some(initial_e), Some(final_e)) =
                    (system.total_energy.first(), system.total_energy.last())
                {
                    println!("Initial Energy: {:.4e} J", initial_e);
                    println!("Final Energy:   {:.4e} J", final_e);
                }

                let mut perturbed_conditions = init_conditions.clone();
                modify_init_conditions(&mut perturbed_conditions, 0.01);

                let measure_h = 0.1;
                let measure_sim_time = 1000.0;
                let mut baseline = System::new(init_conditions.clone(), measure_sim_time, measure_h, false);
                let mut modified = System::new(perturbed_conditions, measure_sim_time, measure_h, false);

                let solver = VelocityVerletSolver;
                solver.solve(&mut baseline);
                solver.solve(&mut modified);

                let lyap = LyapunovCalculator::new(baseline, modified).calculate_lyapunov_exponent();

                if let Err(e) = experiment::export_experiment(
                    &system,
                    &init_conditions,
                    &solver_name,
                    duration,
                    lyap,
                    "Rust run",
                ) {
                    eprintln!("Failed to export: {}", e);
                } else {
                    println!("Exported results to experiment_data.csv");
                }

                println!("---------------------------------");
            }
        });
    }
}
