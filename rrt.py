from typing import List, Optional
import numpy as np
from collision_checker import CollisionChecker


class RRT:
    def __init__(self, checker: CollisionChecker, q_min: np.ndarray, q_max: np.ndarray,
                 step_size: float = 0.1, goal_bias: float = 0.1, max_iter: int = 5000,
                 goal_threshold: float = 0.1, seed: int = 42):
        self.cc = checker
        self.q_min = q_min
        self.q_max = q_max
        self.step_size = step_size
        self.goal_bias = goal_bias
        self.max_iter = max_iter
        self.goal_threshold = goal_threshold
        self.rng = np.random.default_rng(seed)

    def _sample(self, q_goal: np.ndarray) -> np.ndarray:
        """Sample random configuration with goal bias"""
        if self.rng.random() < self.goal_bias:
            return q_goal.copy()
        return self.rng.uniform(self.q_min, self.q_max)

    def _nearest(self, tree: List[np.ndarray], q_sample: np.ndarray) -> int:
        """Find nearest node in tree"""
        distances = [np.linalg.norm(node - q_sample) for node in tree]
        return int(np.argmin(distances))

    def _steer(self, q_from: np.ndarray, q_to: np.ndarray) -> np.ndarray:
        """Steer from q_from toward q_to by step_size"""
        direction = q_to - q_from
        distance = np.linalg.norm(direction)

        if distance <= self.step_size:
            return q_to.copy()

        return q_from + (direction / distance) * self.step_size

    def _extract_path(self, tree: List[np.ndarray], parents: List[int], goal_idx: int) -> List[np.ndarray]:
        """Extract path from start to goal"""
        path = []
        current = goal_idx

        while current != -1:
            path.append(tree[current])
            current = parents[current]

        return list(reversed(path))

    def plan(self, q_start: np.ndarray, q_goal: np.ndarray) -> Optional[List[np.ndarray]]:
        """Plan path from start to goal using RRT"""
        # Validate start and goal
        if not self.cc.is_state_valid(q_start):
            print("Start configuration invalid")
            return None
        if not self.cc.is_state_valid(q_goal):
            print("Goal configuration invalid")
            return None

        # Initialize tree
        tree = [q_start.copy()]
        parents = [-1]  # Root has no parent

        for i in range(self.max_iter):
            # Sample configuration
            q_sample = self._sample(q_goal)

            # Find nearest node
            nearest_idx = self._nearest(tree, q_sample)
            q_nearest = tree[nearest_idx]

            # Steer toward sample
            q_new = self._steer(q_nearest, q_sample)

            # Check if path is collision-free
            if self.cc.path_is_valid(q_nearest, q_new, step=self.step_size * 0.5):
                # Add new node
                tree.append(q_new)
                parents.append(nearest_idx)

                # Check if goal reached
                if np.linalg.norm(q_new - q_goal) < self.goal_threshold:
                    print(f"Goal reached in {i + 1} iterations")
                    self.cc.set_q(q_goal)
                    return self._extract_path(tree, parents, len(tree) - 1)

        print(f"Failed to reach goal in {self.max_iter} iterations")
        return None