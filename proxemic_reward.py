import socnavgym
from socnavgym.envs.rewards.reward_api import RewardAPI
from socnavgym.envs.socnavenv_v1 import SocNavEnv_v1, EntityObs
from socnavgym.envs.utils.utils import point_to_segment_dist
import numpy as np

class Reward(RewardAPI):
    def __init__(self, env: SocNavEnv_v1) -> None:
        super().__init__(env)
        
        # INCREASED MAGNITUDE to prevent accumulation of small penalties outweighing collision
        self.reach_reward = 10.0          
        self.collision_reward = -20.0     
        self.out_of_map_reward = -10.0    
        self.max_steps_reward = -10.0     
        
        # --- Step Rewards ---
        self.alive_reward = -0.001        
        
        # --- Factors ---
        # Smoothness
        self.smoothness_factor = 0.1
        
        # Proxemics
        self.intimate_distance = 0.45  
        self.personal_distance = 1.2  
        self.social_distance = 3.6
        self.intimate_penalty = -0.1  
        self.personal_penalty = -0.05
        self.social_penalty = 0.0   
        
        # Freezing
        self.min_forward_speed = 0.1
        self.freezing_penalty = -0.05
        self.safety_stop_distance = 1.0 # Don't penalize freezing if within this dist of human

        # Orientation
        self.orientation_alignment_factor = 0.1

        # Obstacle Avoidance
        self.obstacle_safety_distance = 0.6 
        self.obstacle_penalty_factor = -0.1
        self.sensor_range = 4.0 # Optimization: Only check obstacles within 4m

        # State tracking
        self.prev_action = None
        self.prev_position = None
        self.last_step_count = -1

    def _reset_internal_state(self):
        """Resets the internal history when a new episode starts."""
        self.prev_action = None
        self.prev_position = None

    def _get_all_humans(self):
        all_humans = []
        all_humans.extend(self.env.static_humans)
        all_humans.extend(self.env.dynamic_humans)
        for interaction in self.env.static_interactions + self.env.moving_interactions:
            all_humans.extend(interaction.humans)
        for interaction in self.env.h_l_interactions:
            all_humans.append(interaction.human)
        return all_humans

    def compute_smoothness_reward(self, action):
        if self.prev_action is None:
            return 0.0
        
        delta_linear = np.sqrt((action[0] - self.prev_action[0])**2 + (action[1] - self.prev_action[1])**2)
        delta_angular = np.abs(action[2] - self.prev_action[2])
        
        smoothness_penalty = -(delta_linear + delta_angular) * self.smoothness_factor
        return smoothness_penalty

    def compute_proxemics_reward(self):
        reward = 0.0
        all_humans = self._get_all_humans()
        
        for human in all_humans:
            dist_sq = (human.x - self.env.robot.x)**2 + (human.y - self.env.robot.y)**2
            # Optimization: Quick check before sqrt
            if dist_sq > self.social_distance**2:
                continue

            distance = np.sqrt(dist_sq) - (self.env.HUMAN_DIAMETER / 2 + self.env.ROBOT_RADIUS)
            
            if distance < self.intimate_distance:
                reward += self.intimate_penalty
            elif distance < self.personal_distance:
                reward += self.personal_penalty
            elif distance < self.social_distance:
                reward += self.social_penalty
        
        return reward

    def compute_freezing_penalty(self, action):
        forward_speed = action[0]
        
        # If we are moving fast enough, no penalty
        if forward_speed >= self.min_forward_speed:
            return 0.0
            
        # CONTEXT AWARENESS: If we are freezing, check if it is justified (human nearby)
        all_humans = self._get_all_humans()
        for human in all_humans:
            dist = np.sqrt((human.x - self.env.robot.x)**2 + (human.y - self.env.robot.y)**2)
            if dist < self.safety_stop_distance:
                # Justified stop: No penalty
                return 0.0
                
        return self.freezing_penalty

    def _get_nearest_point_on_rectangle(self, rect_x, rect_y, rect_length, rect_width, rect_orientation, point_x, point_y):
        dx = point_x - rect_x
        dy = point_y - rect_y
        cos_theta = np.cos(-rect_orientation)
        sin_theta = np.sin(-rect_orientation)
        local_x = dx * cos_theta - dy * sin_theta
        local_y = dx * sin_theta + dy * cos_theta
        
        half_length = rect_length / 2
        half_width = rect_width / 2
        clamped_x = np.clip(local_x, -half_length, half_length)
        clamped_y = np.clip(local_y, -half_width, half_width)
        
        world_x = clamped_x * cos_theta + clamped_y * sin_theta + rect_x
        world_y = -clamped_x * sin_theta + clamped_y * cos_theta + rect_y
        return world_x, world_y

    def compute_obstacle_avoidance_reward(self):
        penalty = 0.0
        min_distance = float('inf')
        rx, ry = self.env.robot.x, self.env.robot.y
        
        # Helper to reduce code duplication and add sensor range check
        def check_rect_obstacles(obstacles, is_wall=False):
            local_pen = 0.0
            local_min = float('inf')
            for obj in obstacles:
                # Optimization: Distance check before expensive math
                if abs(obj.x - rx) > self.sensor_range or abs(obj.y - ry) > self.sensor_range:
                    continue

                width = obj.thickness if is_wall else obj.width
                nx, ny = self._get_nearest_point_on_rectangle(
                    obj.x, obj.y, obj.length, width, obj.orientation, rx, ry
                )
                dist = np.sqrt((nx - rx)**2 + (ny - ry)**2) - self.env.ROBOT_RADIUS
                local_min = min(local_min, dist)
                
                if dist < self.obstacle_safety_distance:
                    ratio = (self.obstacle_safety_distance - dist) / self.obstacle_safety_distance
                    local_pen += self.obstacle_penalty_factor * ratio
            return local_pen, local_min

        # Check all lists
        pen_w, min_w = check_rect_obstacles(self.env.walls, is_wall=True)
        pen_t, min_t = check_rect_obstacles(self.env.tables)
        pen_c, min_c = check_rect_obstacles(self.env.chairs)
        pen_l, min_l = check_rect_obstacles(self.env.laptops)
        
        # Interactions laptops
        int_laptops = [i.laptop for i in self.env.h_l_interactions]
        pen_il, min_il = check_rect_obstacles(int_laptops)

        # Plants (Circles)
        for plant in self.env.plants:
            if abs(plant.x - rx) > self.sensor_range or abs(plant.y - ry) > self.sensor_range:
                continue
            dist = np.sqrt((plant.x - rx)**2 + (plant.y - ry)**2) - plant.radius - self.env.ROBOT_RADIUS
            min_distance = min(min_distance, dist)
            if dist < self.obstacle_safety_distance:
                ratio = (self.obstacle_safety_distance - dist) / self.obstacle_safety_distance
                penalty += self.obstacle_penalty_factor * ratio

        total_penalty = penalty + pen_w + pen_t + pen_c + pen_l + pen_il
        total_min = min(min_distance, min_w, min_t, min_c, min_l, min_il)
        
        return total_penalty, total_min

    def compute_orientation_alignment_reward(self):
        dx = self.env.robot.goal_x - self.env.robot.x
        dy = self.env.robot.goal_y - self.env.robot.y
        dist_to_goal = np.sqrt(dx**2 + dy**2)
        
        if dist_to_goal < 2.0:
            desired_orientation = np.arctan2(dy, dx)
            current_ori = self.env.robot.orientation
            # Normalize angle diff to [-pi, pi]
            angle_diff = np.arctan2(np.sin(desired_orientation - current_ori), np.cos(desired_orientation - current_ori))
            
            proximity_weight = (2.0 - dist_to_goal) / 2.0
            # Use absolute difference for penalty
            return -np.abs(angle_diff) * self.orientation_alignment_factor * proximity_weight
        
        return 0.0

    def compute_reward(self, action, prev_obs: EntityObs, curr_obs: EntityObs):
        # Detect new episode via checking step count or internal flags if available
        # Heuristic: If prev_position is very far from current, it might be a reset
        # Ideally, use an explicit reset call, but here we handle startup:
        if self.prev_position is None:
            self._reset_internal_state()
            # Skip progress/smoothness calculation on first step to avoid huge spikes
            self.prev_action = action.copy()
            self.prev_position = np.array([self.env.robot.x, self.env.robot.y])
            return 0.0

        # 1. Terminal Checks
        if self.check_out_of_map(): return self.out_of_map_reward
        elif self.check_reached_goal(): return self.reach_reward
        elif self.check_collision(): return self.collision_reward
        elif self.check_timeout(): return self.max_steps_reward

        # 2. Calculate Component Rewards
        smoothness_reward = self.compute_smoothness_reward(action)
        proxemics_reward = self.compute_proxemics_reward()
        freezing_penalty = self.compute_freezing_penalty(action)
        orientation_reward = self.compute_orientation_alignment_reward()
        obstacle_penalty, min_obstacle_dist = self.compute_obstacle_avoidance_reward()

        # 3. Update Info
        self.info = {
            "alive_reward": self.alive_reward,
            "smoothness_reward": smoothness_reward,
            "proxemics_reward": proxemics_reward,
            "freezing_penalty": freezing_penalty,
            "orientation_alignment_reward": orientation_reward,
            "obstacle_avoidance_penalty": obstacle_penalty,
            "min_obstacle_distance": min_obstacle_dist
        }
        
        # 4. Update State
        self.prev_action = action.copy()
        self.prev_position = np.array([self.env.robot.x, self.env.robot.y])

        # 5. Aggregate
        total_reward = (
            smoothness_reward + 
            proxemics_reward + 
            freezing_penalty + 
            orientation_reward +
            obstacle_penalty +
            self.alive_reward
        )

        return total_reward