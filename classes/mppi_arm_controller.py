import numpy as np
import pybullet as p


class MppiArmController:
    """
    MPPI controller for a 7-DoF arm using a kinematic rollout model in PyBullet.

    - Rollouts are performed by resetting joint states (no dynamics).
    - Cost = distance-to-target + collision penalty + jerk penalty.
    - Draws the best rollout (green) + a few colliding rollouts (red).
    """

    def __init__(self, robot_id, dt=0.05, horizon=30, n_samples=30):
        self.robot_id = robot_id

        # MPPI parameters
        self.dt = dt
        self.H = horizon
        self.K = n_samples
        self.dof = 7
        self.lambda_ = 0.001
        self.sigma = 0.3

        # Cost weights
        self.dist_weight = 10.0
        self.collision_cost = 10_000.0
        self.jerk_weight = 1.0

        # Robot indexing (matches your URDF setup)
        self.arm_indices = [4, 5, 6, 7, 8, 9, 10]  # joints j1..j7
        self.ee_idx = 11                           # end-effector link index

        # Joint limits
        self.joint_limits_lower = np.array([-5.89, -3.76, -2.89, -3.07, -2.89, -0.01, -2.89])
        self.joint_limits_upper = np.array([ 5.89,  3.76,  2.89, -0.06,  2.89,  3.75,  2.89])

        # Nominal control sequence (warm-started across calls)
        self.U = np.zeros((self.H, self.dof))

    def compute_action(self, current_q, target_pos, target_body_id=None):
        """
        Returns 7 joint velocity commands (clipped to [-1, 1]) for the current step.

        target_body_id is optional: if provided, contacts with that body are ignored.
        (With Fix A, the target marker has no collision shape anyway.)
        """
        # Save/restore simulator state so rollouts don't affect the real simulation.
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        state_id = p.saveState()

        noise = np.random.normal(0.0, self.sigma, size=(self.K, self.H, self.dof))
        costs = np.zeros(self.K)

        # Store EE paths for drawing (K rollouts, H steps, 3D point)
        rollout_paths = np.zeros((self.K, self.H, 3))
        collision_mask = np.zeros(self.K, dtype=bool)

        for k in range(self.K):
            self._reset_arm(current_q)
            u_prev = np.zeros(self.dof)

            for t in range(self.H):
                u_t = np.clip(self.U[t] + noise[k, t], -1.0, 1.0)
                self._step_kinematics(u_t)

                # Make sure contact queries reflect the new joint states
                p.performCollisionDetection()

                ee_pos = np.array(p.getLinkState(self.robot_id, self.ee_idx)[0])
                rollout_paths[k, t] = ee_pos

                dist = np.linalg.norm(ee_pos - np.array(target_pos))
                collided, hit_body_id = self._check_collision(target_body_id)

                if collided:
                    collision_mask[k] = True
                    # Note: this can be spammy; kept because your original code prints.
                    print(f"CRASH: Hit object ID {hit_body_id}")

                jerk = np.linalg.norm(u_t - u_prev)
                u_prev = u_t

                costs[k] += (
                    self.dist_weight * dist
                    + (self.collision_cost if collided else 0.0)
                    + self.jerk_weight * jerk
                )

        # Restore the real world state before returning an action.
        p.restoreState(state_id)
        p.removeState(state_id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        best_k = int(np.argmin(costs))
        self._draw_rollouts(paths=rollout_paths, collision_mask=collision_mask, best_k=best_k)

        # MPPI update: weight rollouts by cost and shift the control sequence forward.
        min_cost = float(np.min(costs))
        weights = np.exp(-(costs - min_cost) / self.lambda_)
        weights /= (np.sum(weights) + 1e-10)

        for t in range(self.H):
            self.U[t] += np.sum(weights[:, None] * noise[:, t, :], axis=0)

        action = self.U[0].copy()
        self.U[:-1] = self.U[1:]
        self.U[-1] = 0.0

        return action

    # -----------------------
    # Collision + visualization
    # -----------------------

    def _check_collision(self, target_body_id):
        """
        Returns (collided: bool, hit_body_id: int|None)
        Filters: self-collisions, target object (if provided), and plane.
        """
        contacts = p.getContactPoints(bodyA=self.robot_id)
        for c in contacts:
            hit_body_id = c[2]

            # Ignore self
            if hit_body_id == self.robot_id:
                continue

            # Ignore target if requested
            if target_body_id is not None and hit_body_id == target_body_id:
                continue

            # Ignore floor/plane (name check kept, as in your original code)
            try:
                name = p.getBodyInfo(hit_body_id)[1].decode("utf-8")
                if name == "plane":
                    continue
            except Exception:
                pass

            return True, hit_body_id

        return False, None

    def _draw_rollouts(self, paths, collision_mask, best_k):
        """Draw best rollout in green and a few colliding rollouts in red."""
        self._draw_single_path(paths[best_k], color=[0, 1, 0], width=3.0)

        crash_indices = np.where(collision_mask)[0]
        np.random.shuffle(crash_indices)

        drawn = 0
        for idx in crash_indices:
            if idx == best_k:
                continue
            self._draw_single_path(paths[idx], color=[1, 0, 0], width=1.0)
            drawn += 1
            if drawn >= 3:
                break

    def _draw_single_path(self, path, color, width, stride=3, life_time=0.2):
        """Draw a rollout path using debug lines (strided to reduce line count)."""
        for t in range(0, self.H - stride, stride):
            p.addUserDebugLine(
                lineFromXYZ=path[t],
                lineToXYZ=path[t + stride],
                lineColorRGB=color,
                lineWidth=width,
                lifeTime=life_time,
            )

    # -----------------------
    # Kinematic rollout model
    # -----------------------

    def _reset_arm(self, q):
        """Reset arm joints to q (length 7)."""
        for i, joint_idx in enumerate(self.arm_indices):
            p.resetJointState(self.robot_id, joint_idx, q[i])

    def _step_kinematics(self, u):
        """
        Integrate joint velocities for one timestep and clamp to joint limits.
        This is a kinematic step (no dynamics).
        """
        for i, joint_idx in enumerate(self.arm_indices):
            curr = p.getJointState(self.robot_id, joint_idx)[0]
            new_pos = curr + u[i] * self.dt
            new_pos = np.clip(new_pos, self.joint_limits_lower[i], self.joint_limits_upper[i])
            p.resetJointState(self.robot_id, joint_idx, new_pos)
