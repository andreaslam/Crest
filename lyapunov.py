from __future__ import annotations

import numpy as np
import random

from solvers import *


def modify_init_conditions(init_conditions, threshold=0.01):
    for obj in init_conditions:
        obj.position *= np.random.uniform(-threshold, threshold, 3) * [
            random.choice([-1, 1]) for _ in range(3)
        ]
        obj.velocity *= np.random.uniform(-threshold, threshold, 3) * [
            random.choice([-1, 1]) for _ in range(3)
        ]

    return init_conditions


class LyapunovCalculator:
    """
    Calculates the (largest) Lyapunov exponent for two nearly identical orbits.
    """

    def __init__(self, system1: System, system2: System) -> None:
        # Each orbit is a list of Mass objects.
        assert (
            system1.solved and system2.solved
        ), "must solve the system before using LyapunovCalculator"
        assert system1.h == system2.h
        assert system1.num_steps == system2.num_steps
        self.system1 = system1
        self.system2 = system2

    def compute_separation(self) -> np.ndarray:
        separations = np.array(
            [
                np.linalg.norm(
                    np.array(self.system1.positions[m1])
                    - np.array(self.system2.positions[m2]),
                    axis=1,
                )
                for m1, m2 in zip(self.system1.positions, self.system2.positions)
            ]
        )
        return np.mean(separations, axis=0)

    def calculate_lyapunov_exponent(
        self,
    ) -> float:
        """
        Evolves both orbits for a given number of steps.
        At each step, it measures the separation and adds the logarithm
        of the ratio (current separation / initial separation).
        """
        # Record the initial separation
        all_seps = self.compute_separation()
        initial_sep = all_seps[0]
        assert initial_sep != 0, "must have initial separation"

        # ln(|delta(t)|/|delta_0|)/t is the Lyapunov exponent

        return np.mean(
            np.log((all_seps / initial_sep)) / (self.system1.num_steps * self.system1.h)
        )
