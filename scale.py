from itertools import combinations

import numpy as np

from solvers import Mass, g


class UnitConverter:
    def __init__(self, system: list[Mass]) -> None:
        self.objs = system
        self.raw_initial_masses = [obj.mass for obj in self.objs]
        self.raw_initial_velocities = [obj.velocity for obj in self.objs]
        self.raw_initial_positions = [obj.position for obj in self.objs]
        self.mass_sf = None
        self.dist_sf = None
        self.time_sf = None

    def compute_scale_factors(self):
        """Compute mass, distance, and time scale factors."""

        self.mass_sf = np.mean(self.raw_initial_masses)
        seps = []
        for pair in list(combinations(self.objs, 2)):
            seps.append(np.linalg.norm(pair[0].position - pair[1].position))
        self.dist_sf = np.mean(np.array(seps))

        self.time_sf = np.sqrt(self.dist_sf**3 / (g * self.mass_sf))

    def normalise_masses(self):
        """Return masses in dimensionless units."""
        if self.mass_sf is None:
            self.compute_scale_factors()
        return [mass / self.mass_sf for mass in self.raw_initial_masses]

    def normalise_positions(self):
        """Return positions in dimensionless units."""
        if self.dist_sf is None:
            self.compute_scale_factors()
        return [position / self.dist_sf for position in self.raw_initial_positions]

    def normalise_velocities(self):
        """Return velocities in dimensionless units."""
        if self.dist_sf is None or self.time_sf is None:
            self.compute_scale_factors()

        return [
            velocity * self.time_sf / self.dist_sf
            for velocity in self.raw_initial_velocities
        ]

    def convert_initial_conditions(self):
        """Return scaled initial conditions."""
        if self.mass_sf is None or self.dist_sf is None or self.time_sf is None:
            self.compute_scale_factors()
        scaled_positions = self.normalise_positions()
        scaled_velocities = self.normalise_velocities()
        scaled_masses = self.normalise_masses()
        return [
            Mass(m, v, p)
            for p, v, m in zip(scaled_positions, scaled_velocities, scaled_masses)
        ]

    def convert_energy_to_joules(self, energy):
        if self.mass_sf is None or self.dist_sf is None:
            self.compute_scale_factors()
        return energy * ((g * (self.mass_sf**2))) / self.dist_sf

    def __repr__(self) -> str:
        """Display the scale factors."""
        if self.mass_sf is None or self.dist_sf is None or self.time_sf is None:
            return "Scale factors not computed yet."
        vel_sf = self.dist_sf / self.time_sf
        return f"""Mass: 1 arbitrary unit = {self.mass_sf:.3g} kg
Distance: 1 arbitrary unit = {self.dist_sf:.3g} m
Velocity: 1 arbitrary unit = {vel_sf:.3g} m/s
Time: 1 arbitrary unit = {self.time_sf:.3g} s
Gravitational constant G: 1 arbitrary unit
Energy: 1 arbitrary unit = {((g * (self.mass_sf**2))) / self.dist_sf} J"""
