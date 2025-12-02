from contextlib import contextmanager
from typing import List

import numpy as np
import pybullet as p

from xArm import XArm


class CollisionChecker:
    def __init__(self, robot: XArm, obstacles: List[int]):
        self.robot = robot
        self.obstacles = list(obstacles)

    def set_q(self, q: np.ndarray):
        for k, j in enumerate(self.robot.joints):
            p.resetJointState(self.robot.id, j, float(q[k]))

    @contextmanager
    def _temp_config(self, q: np.ndarray):
        """Temporarily set robot configuration, then restore"""
        # Save current configuration
        original_q = self.robot.get_q()
        try:
            # Set temporary configuration
            self.set_q(q)
            yield
        finally:
            # Restore original configuration
            self.set_q(original_q)

    def is_state_valid(self, q: np.ndarray) -> bool:
        with self._temp_config(q):
            # self-collision (skip adjacent)
            num = len(self.robot.joints)
            for a in range(num):
                for b in range(a+2, num):
                    if p.getClosestPoints(self.robot.id, self.robot.id, distance=0.02, linkIndexA=self.robot.joints[a], linkIndexB=self.robot.joints[b]):
                        return False
            # environment
            for obs in self.obstacles:
                if p.getClosestPoints(self.robot.id, obs, distance=0.0):
                    return False
            # keep EE above ground a bit
            ee_pos, _ = self.robot.fk()
            if ee_pos[2] < 0.0:
                return False
        return True

    def path_is_valid(self, q0: np.ndarray, q1: np.ndarray, step: float = 0.02) -> bool:
        d = np.linalg.norm(q1-q0)
        n = max(2, int(d/step))
        for i in range(n+1):
            q = q0 + (q1-q0)*(i/n)
            if not self.is_state_valid(q):
                return False
        return True