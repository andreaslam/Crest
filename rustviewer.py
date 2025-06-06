import pandas as pd

import numpy as np
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class MassFromCSV:
    mass: float
    positions: np.ndarray
    velocities: np.ndarray


def load_csv_data(filename: str) -> list[MassFromCSV]:
    # Read CSV file
    df = pd.read_csv(filename)

    # Group by unique objects
    grouped = df.groupby("Object")
    masses = []

    for obj_name, group in grouped:
        # Sort by time step to ensure correct order
        group = group.sort_values("Time Step")

        # Extract mass (assuming constant mass per object)
        mass = group["Mass"].iloc[0]

        # Create position and velocity arrays
        positions = group[["Position X", "Position Y", "Position Z"]].values
        velocities = group[["Velocity X", "Velocity Y", "Velocity Z"]].values

        masses.append(
            MassFromCSV(mass=mass, positions=positions, velocities=velocities)
        )

    return masses


def animate_csv_orbits_3d(masses: list[MassFromCSV]):
    fig = plt.figure(figsize=(20, 7))
    ax = fig.add_subplot(111, projection="3d")

    # Create lines and points for each mass
    lines = [ax.plot([], [], [], label=f"Mass {obj.mass:.2f} kg")[0] for obj in masses]
    points = [ax.plot([], [], [], "o")[0] for _ in masses]

    ax.set_title("Orbital Motion Simulation")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_zlabel("Z Position (m)")

    # Set limits based on data
    ax.set_xlim([-500, 500])
    ax.set_ylim([-500, 500])
    ax.set_zlim([-500, 500])
    ax.legend()

    def init():
        for line, point in zip(lines, points):
            line.set_data([], [])
            line.set_3d_properties([])
            point.set_data([], [])
            point.set_3d_properties([])
        return lines + points

    def update(frame):
        for mass, line, point in zip(masses, lines, points):
            positions = mass.positions
            if frame < len(positions):
                line.set_data(positions[:frame, 0], positions[:frame, 1])
                line.set_3d_properties(positions[:frame, 2])
                point.set_data(
                    positions[frame : frame + 1, 0], positions[frame : frame + 1, 1]
                )
                point.set_3d_properties(positions[frame : frame + 1, 2])
        return lines + points

    max_frames = min(len(mass.positions) for mass in masses)

    ani = FuncAnimation(
        fig, update, frames=max_frames, init_func=init, interval=1, blit=True
    )

    plt.show()
    plt.close()


# Example usage
if __name__ == "__main__":
    # Load and animate the data
    masses = load_csv_data("nbody_simulation.csv")
    animate_csv_orbits_3d(masses)
