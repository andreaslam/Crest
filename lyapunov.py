from __future__ import annotations

import numpy as np

from solvers import *


def modify_init_conditions(init_conditions, threshold=1e-6):
    """
    Add a tiny random displacement to each object's position & velocity.
    """
    for obj in init_conditions:
        dp = np.random.uniform(-threshold, threshold, size=3)
        dv = np.random.uniform(-threshold, threshold, size=3)
        obj.position = obj.position + dp
        obj.velocity = obj.velocity + dv
    return init_conditions


class LyapunovCalculator:
    """
    Computes the largest Lyapunov exponent by linear-fitting
    log(separation) vs time.
    """

    def __init__(self, system1: System, system2: System) -> None:
        assert (
            system1.solved and system2.solved
        ), "Both systems must be solved (i.e. integrated) first."
        assert system1.h == system2.h, "Step sizes must match"
        assert system1.num_steps == system2.num_steps, "Number of steps must match"
        self.sys1 = system1
        self.sys2 = system2
        self.h = system1.h
        self.num_steps = system1.num_steps

    def compute_separation(self) -> np.ndarray:
        """
        Returns an array delta of length num_steps+1,
        where delta[i] is the mean distance between the two
        systems' mass configurations at step i.
        """
        # assume system.positions is dict or list: mass_id -> array shape (num_steps+1,3)
        # stack over masses then take mean over masses
        all_dists = []
        for m in self.sys1.positions:
            pos1 = np.asarray(self.sys1.positions[m])  # shape (N+1,3)
            pos2 = np.asarray(self.sys2.positions[m])
            d = np.linalg.norm(pos1 - pos2, axis=1)  # shape (N+1,)
            all_dists.append(d)
        # shape (num_masses, N+1) -> mean over masses
        return np.mean(np.vstack(all_dists), axis=0)

    def calculate_lyapunov_exponent(self) -> float:
        delta = self.compute_separation()
        delta0 = delta[0]
        if delta0 <= 0:
            raise ValueError("Initial separation must be > 0")
        t = np.arange(len(delta)) * self.h
        y = np.log(delta / delta0)
        # np.polyfit returns [slope, intercept]
        lyapunov, intercept = np.polyfit(t, y, 1)
        print("rate of separation:", delta0 * np.e * (lyapunov * len(delta) * self.h))
        return lyapunov
