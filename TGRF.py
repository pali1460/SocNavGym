import numpy as np
from socnavgym.envs.socnavenv_v1 import SocNavEnv_v1, EntityObs
from socnavgym.envs.rewards.reward_api import RewardAPI


class Reward(RewardAPI):
    """
    Transformable Gaussian Reward Function (TGRF) for SocNavGym.
    
    Based on the paper: "Transformable Gaussian Reward Function for Socially-Aware 
    Navigation with Deep Reinforcement Learning" (Kim et al., 2024)
    Paper: https://arxiv.org/abs/2402.14569
    
    The TGRF uses a Gaussian distribution to create flexible, tunable reward functions
    with minimal hyperparameters for socially-aware robot navigation.
    """
    
    def __init__(self, env: SocNavEnv_v1) -> None:
        super().__init__(env)
        
        # Terminal rewards
        self.r_reach = 10.0  # Reward for reaching goal
        self.r_collision = -10.0  # Penalty for collision
        
        # TGRF hyperparameters for discomfort distance reward
        self.h_TGRF = 8.0  # Height/weight of TGRF (scales the reward)
        self.sigma_TGRF = 3.0  # Variance of Gaussian (controls width/sensitivity)
        self.d_disc = 2.0  # Discomfort distance threshold (danger zone)
        
        # Potential field reward parameters
        self.goal_factor = 1.0  # Weight for potential field reward
        self.sigma_potential = 1000.0  # Very high sigma for constant-like TGRF
        
        # Track previous distance for potential reward
        self.prev_distance_to_goal = None
    
    def _tgrf(self, x: float, h: float, sigma: float, mu: float = 0.0) -> float:
        """
        Compute the Transformable Gaussian Reward Function (TGRF).
        
        TGRF(x) = h * exp(-(x - mu)^2 / (2 * sigma^2)) / exp(-mu^2 / (2 * sigma^2))
        
        The normalization ensures TGRF has a maximum value of h at x = mu.
        
        Args:
            x: Input value (e.g., distance)
            h: Height/weight parameter (scales the maximum value)
            sigma: Variance parameter (controls width of Gaussian)
            mu: Mean parameter (center of Gaussian, default 0)
            
        Returns:
            TGRF value
        """
        normalization = np.exp(-(mu ** 2) / (2 * sigma ** 2))
        gaussian = np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))
        return h * gaussian / normalization
    
    def _compute_dmin(self) -> float:
        """
        Compute minimum distance to nearest human.
        
        Returns:
            Minimum distance to any human in the environment
        """
        dmin = float('inf')
        
        # Collect all humans from different sources
        all_humans = []
        
        # Static and dynamic humans
        for human in self.env.static_humans + self.env.dynamic_humans:
            all_humans.append(human)
        
        # Humans in interactions
        for interaction in self.env.static_interactions + self.env.moving_interactions:
            for h in interaction.humans:
                all_humans.append(h)
        
        # Humans in human-laptop interactions
        for interaction in self.env.h_l_interactions:
            all_humans.append(interaction.human)
        
        # Compute minimum distance
        for human in all_humans:
            dx = human.x - self.env.robot.x
            dy = human.y - self.env.robot.y
            distance = np.sqrt(dx ** 2 + dy ** 2)
            
            # Account for human and robot radii
            distance = distance - self.env.HUMAN_DIAMETER / 2 - self.env.ROBOT_RADIUS
            
            if distance < dmin:
                dmin = distance
        
        return dmin
    
    def _compute_r_disc(self) -> float:
        """
        Compute discomfort distance reward using TGRF.
        
        This reward penalizes the robot for being too close to humans,
        with the penalty following a Gaussian distribution around each human.
        
        Returns:
            Discomfort reward (negative value when in danger zone)
        """
        dmin = self._compute_dmin()
        
        # Only apply penalty if within danger zone
        if dmin < self.d_disc:
            # Apply TGRF: negative reward that increases as distance decreases
            penalty = self._tgrf(dmin, self.h_TGRF, self.sigma_TGRF)
            return -penalty  # Negative because it's a penalty
        
        return 0.0
    
    def _compute_r_potential(self) -> float:
        """
        Compute potential field reward for progress toward goal.
        
        This uses a constant-like TGRF (very high sigma) to provide
        consistent positive/negative reward based on movement toward goal.
        
        Returns:
            Potential reward based on change in distance to goal
        """
        # Calculate current distance to goal
        current_distance = np.sqrt(
            (self.env.robot.goal_x - self.env.robot.x) ** 2 + 
            (self.env.robot.goal_y - self.env.robot.y) ** 2
        )
        
        # Compute reward based on change in distance
        reward = 0.0
        if self.prev_distance_to_goal is not None:
            # Positive reward for getting closer, negative for getting farther
            progress = self.prev_distance_to_goal - current_distance
            
            # Apply constant-like TGRF scaling
            # With very high sigma, TGRF ≈ h (constant)
            reward = progress * self.goal_factor * self._tgrf(0, 1.5, self.sigma_potential)
        
        # Update previous distance
        self.prev_distance_to_goal = current_distance
        
        return reward
    
    def compute_reward(self, action, prev_obs: EntityObs, curr_obs: EntityObs) -> float:
        """
        Compute total reward using TGRF components.
        
        Total reward = r_reach OR r_collision OR (r_disc + r_potential)
        
        Args:
            action: Action taken by robot
            prev_obs: Previous observation
            curr_obs: Current observation
            
        Returns:
            Total reward value
        """
        # Check terminal conditions first
        if self.check_reached_goal():
            self.prev_distance_to_goal = None  # Reset for next episode
            return self.r_reach
        
        if self.check_collision():
            self.prev_distance_to_goal = None
            return self.r_collision
        
        if self.check_out_of_map():
            self.prev_distance_to_goal = None
            return self.r_collision  # Treat as collision
        
        if self.check_timeout():
            self.prev_distance_to_goal = None
            return self.r_collision  # Treat as collision
        
        # Compute step rewards using TGRF
        r_disc = self._compute_r_disc()
        r_potential = self._compute_r_potential()
        
        # Store components in info for logging/debugging
        self.info["r_disc"] = r_disc
        self.info["r_potential"] = r_potential
        self.info["dmin"] = self._compute_dmin()
        self.info["distance_to_goal"] = self.prev_distance_to_goal
        
        # Total reward
        total_reward = r_disc + r_potential
        
        return total_reward
