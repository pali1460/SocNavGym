import numpy as np
from socnavgym.envs.socnavenv_v1 import SocNavEnv_v1, EntityObs
from socnavgym.envs.rewards.reward_api import RewardAPI


class Reward(RewardAPI):
    """
    Simple distance-based reward function for SocNavGym.
    
    Rewards the robot for:
    - Reaching the goal
    - Getting closer to the goal
    
    Penalizes the robot for:
    - Collisions
    - Going out of map bounds
    - Timeout
    - Being alive (small penalty to encourage efficiency)
    """
    
    def __init__(self, env: SocNavEnv_v1) -> None:
        super().__init__(env)
        
        # Terminal state rewards
        self.reach_reward = 1.0
        self.collision_reward = -0.25
        self.out_of_map_reward = -0.25
        self.timeout_reward = -0.25
        
        # Step-based rewards
        self.distance_reward_scale = 1.0  # Scale factor for progress toward goal
        self.alive_penalty = -0.01  # Small penalty per step to encourage efficiency
        
        # Track previous distance to goal
        self.prev_distance = None
    
    def compute_reward(self, action, prev_obs: EntityObs, curr_obs: EntityObs):
        """
        Compute reward based on current state and action taken.
        
        Args:
            action: The action taken by the robot
            prev_obs: Previous observation (EntityObs)
            curr_obs: Current observation (EntityObs)
            
        Returns:
            float: The computed reward value
        """
        # Check terminal conditions first
        if self.check_reached_goal():
            self.prev_distance = None  # Reset for next episode
            return self.reach_reward
        
        if self.check_collision():
            self.prev_distance = None
            return self.collision_reward
        
        if self.check_out_of_map():
            self.prev_distance = None
            return self.out_of_map_reward
        
        if self.check_timeout():
            self.prev_distance = None
            return self.timeout_reward
        
        # Compute distance-based reward for normal steps
        current_distance = np.sqrt(
            (self.env.robot.goal_x - self.env.robot.x) ** 2 + 
            (self.env.robot.goal_y - self.env.robot.y) ** 2
        )
        
        # Reward for getting closer to goal (or penalize for getting farther)
        distance_reward = 0.0
        if self.prev_distance is not None:
            progress = self.prev_distance - current_distance
            distance_reward = progress * self.distance_reward_scale
        
        # Update previous distance for next step
        self.prev_distance = current_distance
        
        # Store reward components in info dict for logging/debugging
        self.info["distance_reward"] = distance_reward
        self.info["alive_penalty"] = self.alive_penalty
        self.info["current_distance_to_goal"] = current_distance
        
        # Total reward is progress toward goal minus small alive penalty
        total_reward = distance_reward + self.alive_penalty
        
        return total_reward
