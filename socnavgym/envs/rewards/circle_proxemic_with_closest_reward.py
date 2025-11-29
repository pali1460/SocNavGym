import socnavgym
from socnavgym.envs.socnavenv_v1 import SocNavEnv_v1, EntityObs
from socnavgym.envs.utils.utils import point_to_segment_dist
import numpy as np
from socnavgym.envs.rewards.reward_api import RewardAPI
from enum import Enum
import matplotlib.patches as patches


class Reward(RewardAPI):
    def __init__(self, env: SocNavEnv_v1) -> None:
        super().__init__(env)
        self.reach_reward = 1.0
        self.out_of_map_reward = -1.0 
        self.max_steps_reward = -1.0 
        self.alive_reward = -0.00001 
        self.collision_reward = -1.0
        self.distance_reward_scaler = 5.0
        self.discomfort_distance = 0.6
        self.discomfort_penalty_factor = 0.5
        self.prev_distance = None

        # proxemics
        self.proxemic_scalar = 0.5

        # Obstacle Avoidance
        self.obstacle_safety_distance = 0.6 
        self.obstacle_penalty_factor = -0.1
        self.sensor_range = 4.0 # Optimization: Only check obstacles within 4m


    # compute the minimum distance between the robot and all humans/interactions over the next timestep
    def compute_dmin(self, action):
        dmin = float('inf')

        all_humans = []
        for human in self.env.static_humans + self.env.dynamic_humans : all_humans.append(human)

        for i in self.env.static_interactions + self.env.moving_interactions:
            for h in i.humans: all_humans.append(h)
        
        for i in self.env.h_l_interactions: all_humans.append(i.human)

        for human in all_humans:
            px = human.x - self.env.robot.x
            py = human.y - self.env.robot.y

            vx = human.speed*np.cos(human.orientation) - action[0] * np.cos(action[2]*self.env.TIMESTEP + self.env.robot.orientation) - action[1] * np.cos(action[2]*self.env.TIMESTEP + self.env.robot.orientation + np.pi/2)
            vy = human.speed*np.sin(human.orientation) - action[0] * np.sin(action[2]*self.env.TIMESTEP + self.env.robot.orientation) - action[1] * np.sin(action[2]*self.env.TIMESTEP + self.env.robot.orientation + np.pi/2)

            ex = px + vx * self.env.TIMESTEP
            ey = py + vy * self.env.TIMESTEP

            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - self.env.HUMAN_DIAMETER/2 - self.env.ROBOT_RADIUS

            if closest_dist < dmin:
                dmin = closest_dist

        for human in self.env.static_humans + self.env.dynamic_humans:
            px = human.x - self.env.robot.x
            py = human.y - self.env.robot.y

            vx = human.speed*np.cos(human.orientation) - action[0] * np.cos(action[2]*self.env.TIMESTEP + self.env.robot.orientation) - action[1] * np.cos(action[2]*self.env.TIMESTEP + self.env.robot.orientation + np.pi/2)
            vy = human.speed*np.sin(human.orientation) - action[0] * np.sin(action[2]*self.env.TIMESTEP + self.env.robot.orientation) - action[1] * np.sin(action[2]*self.env.TIMESTEP + self.env.robot.orientation + np.pi/2)

            ex = px + vx * self.env.TIMESTEP
            ey = py + vy * self.env.TIMESTEP

            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - self.env.HUMAN_DIAMETER/2 - self.env.ROBOT_RADIUS

            if closest_dist < dmin:
                dmin = closest_dist

        for interaction in (self.env.moving_interactions + self.env.static_interactions + self.env.h_l_interactions):
            px = interaction.x - self.env.robot.x
            py = interaction.y - self.env.robot.y

            speed = 0
            if interaction.name == "human-human-interaction":
                for h in interaction.humans:
                    speed += h.speed
                speed /= len(interaction.humans)

            vx = speed*np.cos(human.orientation) - action[0] * np.cos(action[2]*self.env.TIMESTEP + self.env.robot.orientation) - action[1] * np.cos(action[2]*self.env.TIMESTEP + self.env.robot.orientation + np.pi/2)
            vy = speed*np.sin(human.orientation) - action[0] * np.sin(action[2]*self.env.TIMESTEP + self.env.robot.orientation) - action[1] * np.sin(action[2]*self.env.TIMESTEP + self.env.robot.orientation + np.pi/2)

            ex = px + vx * self.env.TIMESTEP
            ey = py + vy * self.env.TIMESTEP

            closest_dist = point_to_segment_dist(px, py, ex, ey, 0, 0) - self.env.HUMAN_DIAMETER/2 - self.env.ROBOT_RADIUS

            if closest_dist < dmin:
                dmin = closest_dist
        
        return dmin
        

    def compute_reward(self, action, prev_obs: EntityObs, curr_obs: EntityObs):
        if self.check_out_of_map(): return self.out_of_map_reward
        elif self.check_reached_goal(): return self.reach_reward
        elif self.check_collision(): return self.collision_reward
        elif self.check_timeout(): return self.max_steps_reward
        else:
            dsrnn_reward = 0.0
            dmin = self.compute_dmin(action)
            if dmin < self.discomfort_distance:
                dsrnn_reward = (dmin - self.discomfort_distance) * self.discomfort_penalty_factor * self.env.TIMESTEP

            distance_to_goal = np.sqrt((self.env.robot.goal_x - self.env.robot.x)**2 + (self.env.robot.goal_y - self.env.robot.y)**2)
            distance_reward = 0.0
            if self.prev_distance is not None:
                distance_reward = -(distance_to_goal-self.prev_distance) * self.distance_reward_scaler
            
            self.prev_distance = distance_to_goal

            self.info["DISCOMFORT_SNGNN"] = 0.0
            self.info["DISCOMFORT_DSRNN"] = dsrnn_reward
            self.info["distance_reward"] = distance_reward
            self.info["alive_reward"] = self.alive_reward
            self.info["proxemics_reward"] = self.compute_proxemics()

            return dsrnn_reward + distance_reward + self.alive_reward + self.info["proxemics_reward"]
    
    def compute_proxemics(self):
        reward = 0
        robot_pos = np.array(self.env.robot.x, self.env.robot.y)

        nearby = 0

        for human in self.env.static_humans + self.env.dynamic_humans:
            dist = np.linalg.norm(robot_pos - np.array(human.x, human.y))
            proxemic_zone = self.get_proxemic_zone(dist)
            
            # these are just arbitarily chosen values for now
            match proxemic_zone:
                case self.Proxemic_Zone.INTIMETE:
                    reward -= 0.5
                    nearby += 1
                case self.Proxemic_Zone.PERSONAL:
                    reward -= 0.2
                    nearby += 1
                case self.Proxemic_Zone.SOCIAL:
                    reward -= 0.02
                    nearby += 1
                case self.Proxemic_Zone.PUBLIC:
                    reward += 0.0
            
        reward /= (nearby + 1e-6)  # normalize by number of nearby humans to prevent scaling with crowd size

        return reward * self.proxemic_scalar
    
    class Proxemic_Zone(Enum):
        INTIMETE = 0
        PERSONAL = 1
        SOCIAL = 2
        PUBLIC = 3
    
    # Circle proxemic model, so just check distance to determine zone
    # Values based on Hall's paper on proxemics, in meters
    # May differ for robot interaction & cultures & scenarios
    def get_proxemic_zone(self, distance):
        if distance < 0:
            raise ValueError("Distance cannot be negative")
        
        if distance < 0.46:
            return self.Proxemic_Zone.INTIMETE
        elif distance < 1.2:
            return self.Proxemic_Zone.PERSONAL
        elif distance < 3.7:
            return self.Proxemic_Zone.SOCIAL
        else:
            return self.Proxemic_Zone.PUBLIC
        
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
    
    