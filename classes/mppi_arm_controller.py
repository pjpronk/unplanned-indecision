import numpy as np
import pybullet as p


class MppiArmController:
    """
    MPPI controller for a 7-DoF arm using a kinematic rollout model in PyBullet.

    - Rollouts are performed by resetting joint states (no dynamics).
    - Cost = exponential distance-to-target + terminal cost + collision penalty + jerk penalty.
    - Actions are scaled down when close to target to prevent overshoot.
    - Draws the best rollout (green) + a few colliding rollouts (red).
    """

    # Adaptive exploration constants
    SIGMA_MIN_FACTOR = 0.3  # Minimum exploration (when close to target)
    SIGMA_MAX_FACTOR = 1.0  # Maximum exploration (when far from target)
    ADAPTIVE_DISTANCE = 0.5  # Distance threshold for adaptive scaling (meters)

    # Warm-start threshold
    TARGET_CHANGE_THRESHOLD = 0.01  # Re-initialize if target moves >1cm

    # Visualization constants
    MAX_COLLISION_PATHS_DRAWN = 3
    ROLLOUT_STRIDE = 3
    DEBUG_LINE_LIFETIME = 0.2

    def __init__(self, robot_id, robot_config, dt, horizon, n_samples, lambda_, sigma,
                 dist_weight, terminal_dist_weight, collision_cost, jerk_weight, exp_decay_rate):
        self.robot_id = robot_id
        self.config = robot_config

        # MPPI parameters
        self.dt = dt
        self.H = horizon
        self.K = n_samples
        self.lambda_ = lambda_
        self.sigma = sigma

        # Cost weights
        self.dist_weight = dist_weight
        self.terminal_dist_weight = terminal_dist_weight
        self.collision_cost = collision_cost
        self.jerk_weight = jerk_weight
        self.exp_decay_rate = exp_decay_rate

        # Nominal control sequence (warm-started across calls)
        self.U = np.zeros((self.H, self.config.arm_dof))

        # Track previous target for re-initialization detection
        self.prev_target = None

    def compute_action(self, current_q, target_pos, target_body_id=None):
        """
        Returns 7 joint velocity commands (clipped to [-1, 1]) for the current step.

        target_body_id is optional: if provided, contacts with that body are ignored.
        (With Fix A, the target marker has no collision shape anyway.)
        """
        # Warm-start control sequence with IK if target changed or first call
        target_array = np.array(target_pos)
        if self._should_reinitialize(target_array):
            self._warm_start_with_ik(current_q, target_pos)
            self.prev_target = target_array.copy()

        # Calculate adaptive exploration based on distance to target
        adaptive_sigma = self._calculate_adaptive_sigma(target_array)

        # Calculate current distance to target for action scaling
        current_ee_pos = np.array(p.getLinkState(self.robot_id, self.config.ee_link_index)[0])
        dist_to_target = np.linalg.norm(current_ee_pos - target_array)

        # Generate noise for all rollouts
        noise = np.random.normal(0.0, adaptive_sigma, size=(self.K, self.H, self.config.arm_dof))

        # Perform rollouts and calculate costs
        rollout_paths, collision_mask, costs = self._perform_rollouts(
            current_q, target_array, noise, target_body_id
        )

        # Update control sequence using MPPI
        action = self._update_control_sequence(costs, noise)

        # Scale down action when close to target to avoid overshoot
        # Use exponential decay: scale = 1.0 when far, approaches 0.1 when very close
        close_threshold = 0.1  # meters
        if dist_to_target < close_threshold:
            action_scale = 0.1 + 0.9 * (dist_to_target / close_threshold)
            action *= action_scale

        # Visualize rollouts
        best_k = int(np.argmin(costs))
        self._draw_rollouts(paths=rollout_paths, collision_mask=collision_mask, best_k=best_k)

        return action

    def _should_reinitialize(self, target_array: np.ndarray) -> bool:
        """Check if control sequence should be re-initialized with IK."""
        if self.prev_target is None:
            return True
        target_moved = (
            np.linalg.norm(target_array - self.prev_target) > self.TARGET_CHANGE_THRESHOLD
        )
        return target_moved

    def _calculate_adaptive_sigma(self, target_array: np.ndarray) -> float:
        """Calculate adaptive exploration noise based on distance to target."""
        current_ee_pos = np.array(p.getLinkState(self.robot_id, self.config.ee_link_index)[0])
        dist_to_target = np.linalg.norm(current_ee_pos - target_array)

        # Scale from SIGMA_MIN_FACTOR (close) to SIGMA_MAX_FACTOR (far)
        scale_factor = self.SIGMA_MIN_FACTOR + (
            self.SIGMA_MAX_FACTOR - self.SIGMA_MIN_FACTOR
        ) * min(dist_to_target / self.ADAPTIVE_DISTANCE, 1.0)

        return self.sigma * scale_factor

    def _perform_rollouts(self, current_q, target_array, noise, target_body_id):
        """
        Perform K rollout simulations and calculate costs.
        Returns: (rollout_paths, collision_mask, costs)
        """
        # Save/restore simulator state so rollouts don't affect the real simulation
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        state_id = p.saveState()

        costs = np.zeros(self.K)
        rollout_paths = np.zeros((self.K, self.H, 3))
        collision_mask = np.zeros(self.K, dtype=bool)

        for k in range(self.K):
            self._reset_arm(current_q)
            u_prev = np.zeros(self.config.arm_dof)

            for t in range(self.H):
                u_t = np.clip(self.U[t] + noise[k, t], -1.0, 1.0)
                self._step_kinematics(u_t)
                p.performCollisionDetection()

                ee_pos = np.array(p.getLinkState(self.robot_id, self.config.ee_link_index)[0])
                rollout_paths[k, t] = ee_pos

                dist = np.linalg.norm(ee_pos - target_array)
                collided, hit_body_id = self._check_collision(target_body_id)

                if collided:
                    collision_mask[k] = True
                    if k == 0:  # Only print for first rollout
                        print(f"CRASH: Hit object ID {hit_body_id}")

                jerk = np.linalg.norm(u_t - u_prev)
                u_prev = u_t

                # Exponential distance cost for stronger gradient near target
                # exp(decay * dist) - 1 gives exponential growth, stronger near target
                dist_cost = self.dist_weight * (np.exp(self.exp_decay_rate * dist) - 1.0)

                # Terminal cost: heavily penalize final state if far from target
                if t == self.H - 1:
                    terminal_cost = self.terminal_dist_weight * (
                        np.exp(self.exp_decay_rate * dist) - 1.0
                    )
                    dist_cost += terminal_cost

                costs[k] += (
                    dist_cost + (self.collision_cost if collided else 0.0) + self.jerk_weight * jerk
                )

        # Restore the real world state
        p.restoreState(state_id)
        p.removeState(state_id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)

        return rollout_paths, collision_mask, costs

    def _update_control_sequence(self, costs, noise):
        """Update control sequence using MPPI weighted average and return first action."""
        min_cost = float(np.min(costs))
        weights = np.exp(-(costs - min_cost) / self.lambda_)
        weights /= np.sum(weights) + 1e-10

        for t in range(self.H):
            self.U[t] += np.sum(weights[:, None] * noise[:, t, :], axis=0)

        action = self.U[0].copy()
        action = np.clip(action, -2.0, 2.0)
        self.U[:-1] = self.U[1:]
        self.U[-1] = 0.0

        return action

    def _warm_start_with_ik(self, current_q, target_pos):
        """
        Warm-start the control sequence using IK-based velocities.
        Calculates desired joint configuration and initializes velocities to move towards it.
        """
        # Use PyBullet IK to get target joint configuration
        ik_solution = p.calculateInverseKinematics(
            self.robot_id,
            self.config.ee_link_index,
            target_pos,
            maxNumIterations=100,
            residualThreshold=0.001,
        )

        # Extract arm joints from IK solution
        target_q = np.array(ik_solution[self.config.arm_obs_slice])

        # Clamp to joint limits
        target_q = np.clip(
            target_q, self.config.arm_joint_limits_lower, self.config.arm_joint_limits_upper
        )

        # Calculate velocity needed to reach target over the horizon
        delta_q = target_q - current_q

        # Initialize control sequence with decaying velocities toward target
        for t in range(self.H):
            # Decay factor: move quickly initially, slower later
            decay = 1.0 - (t / self.H)
            self.U[t] = (delta_q / (self.H * self.dt)) * decay
            # Clip to action limits
            self.U[t] = np.clip(self.U[t], -1.0, 1.0)

        print(f"Warm-started MPPI: current_q={current_q.round(2)}, target_q={target_q.round(2)}")

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

            # Ignore floor/plane
            try:
                name = p.getBodyInfo(hit_body_id)[1].decode("utf-8")
                if name == "plane":
                    continue
            except (IndexError, AttributeError, UnicodeDecodeError):
                # Body might not have info or name field
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
            if drawn >= self.MAX_COLLISION_PATHS_DRAWN:
                break

    def _draw_single_path(self, path, color, width):
        """Draw a rollout path using debug lines (strided to reduce line count)."""
        for t in range(0, self.H - self.ROLLOUT_STRIDE, self.ROLLOUT_STRIDE):
            p.addUserDebugLine(
                lineFromXYZ=path[t],
                lineToXYZ=path[t + self.ROLLOUT_STRIDE],
                lineColorRGB=color,
                lineWidth=width,
                lifeTime=self.DEBUG_LINE_LIFETIME,
            )

    # -----------------------
    # Kinematic rollout model
    # -----------------------

    def _reset_arm(self, q):
        """Reset arm joints to q and grippers to closed position."""
        # Reset arm joints
        for i, joint_idx in enumerate(self.config.arm_joint_indices):
            p.resetJointState(self.robot_id, joint_idx, q[i])

        # Reset grippers to closed (0 position)
        for gripper_idx in self.config.gripper_joint_indices:
            p.resetJointState(self.robot_id, gripper_idx, 0.0)

    def _step_kinematics(self, u):
        """
        Integrate joint velocities for one timestep and clamp to joint limits.
        This is a kinematic step (no dynamics).
        """
        for i, joint_idx in enumerate(self.config.arm_joint_indices):
            curr = p.getJointState(self.robot_id, joint_idx)[0]
            new_pos = curr + u[i] * self.dt
            new_pos = np.clip(
                new_pos,
                self.config.arm_joint_limits_lower[i],
                self.config.arm_joint_limits_upper[i],
            )
            p.resetJointState(self.robot_id, joint_idx, new_pos)
