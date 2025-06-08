import rebound
import pandas as pd
import ast


def main():
    df = pd.read_csv("experiment_data/experiment.csv")

    for col in ["masses", "initial_positions", "initial_velocities"]:
        df[col] = df[col].apply(ast.literal_eval)

    df["masses"] = df["masses"].apply(lambda x: tuple(x))
    df["initial_positions"] = df["initial_positions"].apply(
        lambda x: tuple(tuple(p) for p in x)
    )
    df["initial_velocities"] = df["initial_velocities"].apply(
        lambda x: tuple(tuple(v) for v in x)
    )
    df["step_size"] = df["step_size"].astype(float)
    unique_conditions = df.drop_duplicates(
        subset=["masses", "initial_positions", "initial_velocities", "step_size"]
    )

    for _, row in unique_conditions.iterrows():
        masses = row["masses"]
        positions = row["initial_positions"]
        velocities = row["initial_velocities"]
        sim_time = row["simulated_time"]
        step_size = row["step_size"]

        sim = rebound.Simulation()
        for m, pos, vel in zip(masses, positions, velocities):
            sim.add(m=m, x=pos[0], y=pos[1], z=pos[2], vx=vel[0], vy=vel[1], vz=vel[2])
        sim.move_to_com()
        for _ in range(int(sim_time)):
            sim.integrate(step_size)
            for p in sim.particles:
                print(sim.t, p.x, p.y, p.z)


if __name__ == "__main__":
    main()
