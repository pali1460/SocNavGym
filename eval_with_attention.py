import gymnasium as gym
import socnavgym
import torch
import torch.nn as nn
import numpy as np
from gymnasium import ObservationWrapper
from socnavgym.wrappers import DiscreteActions
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import argparse
from tqdm import tqdm
from stable_baselines3.common.monitor import Monitor
import sys
import imageio
import os

class FixedPaddingWrapper(ObservationWrapper):
    """
    Wrapper that pads observations to a fixed size independent of the 
    environment's max_* config parameters.
    """
    
    def __init__(self, env, global_max_entities):
        super().__init__(env)
        self.global_max_entities = global_max_entities
        
        # Build new observation space with fixed dimensions
        new_obs_space = {}
        
        # Robot observation is always (9,)
        new_obs_space["robot"] = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(9,), dtype=np.float32
        )
        
        # Each entity type padded to global max
        for entity_type, max_count in global_max_entities.items():
            # Each entity has 14 features
            obs_dim = max_count * 14
            new_obs_space[entity_type] = gym.spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(obs_dim,), 
                dtype=np.float32
            )
        
        self.observation_space = gym.spaces.Dict(new_obs_space)
    
    def observation(self, obs):
        """Pad observations to fixed global maximum"""
        new_obs = {"robot": obs["robot"]}
        
        for entity_type, max_count in self.global_max_entities.items():
            # Get current entity observations (may be empty array)
            entity_obs = obs.get(entity_type, np.array([], dtype=np.float32))
            
            current_size = len(entity_obs)
            target_size = max_count * 14
            
            if current_size < target_size:
                # Pad with zeros
                padding = np.zeros(target_size - current_size, dtype=np.float32)
                new_obs[entity_type] = np.concatenate([entity_obs, padding])
            elif current_size > target_size:
                # Warn if actual observations exceed global max
                print(f"WARNING: {entity_type} has {current_size//14} entities "
                      f"but global max is {max_count}. Truncating!")
                new_obs[entity_type] = entity_obs[:target_size]
            else:
                new_obs[entity_type] = entity_obs
        
        return new_obs


class AttentionFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor using attention mechanism to handle variable numbers of entities.
    This allows the network to focus on relevant entities and ignore padding.
    MUST MATCH THE ARCHITECTURE USED DURING TRAINING.
    """
    
    def __init__(self, observation_space, features_dim=256, entity_embed_dim=128, num_heads=4):
        super().__init__(observation_space, features_dim)
        
        self.entity_embed_dim = entity_embed_dim
        
        # Entity encoder - processes each entity's 14 features
        self.entity_encoder = nn.Sequential(
            nn.Linear(14, 64),
            nn.ReLU(),
            nn.Linear(64, entity_embed_dim),
            nn.ReLU()
        )
        
        # Robot state encoder - processes robot's 9 features
        self.robot_encoder = nn.Sequential(
            nn.Linear(9, 64),
            nn.ReLU(),
            nn.Linear(64, entity_embed_dim),
            nn.ReLU()
        )
        
        # Multi-head attention layer
        self.attention = nn.MultiheadAttention(
            embed_dim=entity_embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.1
        )
        
        # Layer norm for stability
        self.layer_norm = nn.LayerNorm(entity_embed_dim)
        
        # Final aggregation network
        self.aggregator = nn.Sequential(
            nn.Linear(entity_embed_dim * 2, 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations):
        """Forward pass with attention mechanism"""
        robot_state = observations["robot"]
        batch_size = robot_state.shape[0]
        
        # Collect all entity observations
        all_entities = []
        entity_types = ["humans", "tables", "laptops", "plants", "walls"]
        
        for entity_type in entity_types:
            if entity_type in observations:
                entity_obs = observations[entity_type]
                num_entities = entity_obs.shape[1] // 14
                entity_obs = entity_obs.reshape(batch_size, num_entities, 14)
                all_entities.append(entity_obs)
        
        # Concatenate all entities
        if len(all_entities) > 0:
            all_entities = torch.cat(all_entities, dim=1)
        else:
            all_entities = torch.zeros(batch_size, 1, 14, device=robot_state.device)
        
        num_entities = all_entities.shape[1]
        
        # Create mask for padding
        entity_mask = (all_entities.abs().sum(dim=-1) > 1e-6)
        
        # Encode entities
        entities_flat = all_entities.reshape(-1, 14)
        encoded_entities = self.entity_encoder(entities_flat)
        encoded_entities = encoded_entities.reshape(batch_size, num_entities, self.entity_embed_dim)
        
        # Encode robot state
        robot_encoded = self.robot_encoder(robot_state)
        robot_query = robot_encoded.unsqueeze(1)
        
        # Apply attention
        padding_mask = ~entity_mask
        attended, attention_weights = self.attention(
            query=robot_query,
            key=encoded_entities,
            value=encoded_entities,
            key_padding_mask=padding_mask,
            need_weights=True
        )
        
        attended = attended.squeeze(1)
        attended = self.layer_norm(attended)
        
        # Combine features
        combined = torch.cat([robot_encoded, attended], dim=-1)
        features = self.aggregator(combined)
        
        return features


# Global maximum entities - MUST MATCH TRAINING
GLOBAL_MAX_ENTITIES = {
    "humans": 20,
    "tables": 10,
    "laptops": 5,
    "plants": 10,
    "walls": 80
}


def eval(model, num_episodes, env):
    # initialising metrics
    discomfort_sngnn = 0
    discomfort_dsrnn = 0
    timeout = 0
    success_rate = 0
    time_taken = 0
    closest_human_dist = 0
    closest_obstacle_dist = 0
    collision_rate = 0
    collision_rate_human = 0
    collision_rate_object = 0
    collision_rate_wall = 0
    total_psc = 0
    total_stl = 0
    total_spl = 0
    total_failure_to_progress = 0
    total_stalled_time = 0
    total_path_length = 0
    total_vel_min = 0
    total_vel_max = 0
    total_vel_avg = 0
    total_a_min = 0
    total_a_max = 0
    total_a_avg = 0
    total_jerk_min = 0
    total_jerk_max = 0
    total_jerk_avg = 0
    total_avg_obstacle_distance = 0
    total_minimum_time_to_collision = 0
    total_time_to_reach_goal = 0
    
    total_reward = 0
    print(f"Evaluating model for {num_episodes} episodes")

    for i in tqdm(range(num_episodes)):
        obs, _ = env.reset()
        done = False
        episode_reward = 0
        has_reached_goal = 0
        has_collided = 0
        has_collided_human = 0
        has_collided_object = 0
        has_collided_wall = 0
        has_timed_out = 0
        steps = 0
        count = 0
        episode_discomfort_sngnn = 0
        episode_discomfort_dsrnn = 0
        psc = 0
        stl = 0
        spl = 0
        failure_to_progress = 0
        stalled_time = 0

        # Access the base environment through the wrapper chain
        base_env = env
        while hasattr(base_env, 'env'):
            base_env = base_env.env
        time_to_reach_goal = base_env.EPISODE_LENGTH

        path_length = 0
        vel_min = 0
        vel_max = 0
        vel_avg = 0
        a_min = 0
        a_max = 0
        a_avg = 0
        jerk_min = 0
        jerk_max = 0
        jerk_avg = 0
        min_human_dist = float('inf')
        min_obstacle_dist = float('inf')
        avg_obstacle_dist = 0
        avg_minimum_time_to_collision = 0

        frames = []
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

            base = env.unwrapped
            frame = base.world_image
            
            if frame is not None:
                frames.append(frame)
            
            steps += 1
            count += 1

            # storing the rewards
            episode_reward += reward

            # storing discomforts
            episode_discomfort_sngnn += info["sngnn_reward"]
            episode_discomfort_dsrnn += info["DISCOMFORT_DSRNN"]

            # storing whether the agent reached the goal
            if info["SUCCESS"]:
                has_reached_goal = 1
                stl = info["STL"]
                spl = info["SPL"]
                time_to_reach_goal = info["TIME_TO_REACH_GOAL"]
            
            if info["COLLISION"]:
                has_collided = 1
                if info["COLLISION_HUMAN"]:
                    has_collided_human = 1
                if info["COLLISION_OBJECT"]:
                    has_collided_object = 1
                if info["COLLISION_WALL"]:
                    has_collided_wall = 1

                steps = base_env.EPISODE_LENGTH
            
            if info["TIMEOUT"]:
                has_timed_out = 1

            min_human_dist = min(min_human_dist, info["MINIMUM_DISTANCE_TO_HUMAN"])
            min_obstacle_dist = min(min_obstacle_dist, info["MINIMUM_OBSTACLE_DISTANCE"])
            avg_obstacle_dist += info["AVERAGE_OBSTACLE_DISTANCE"]
            if info["TIME_TO_COLLISION"] != -1: 
                avg_minimum_time_to_collision += info["TIME_TO_COLLISION"]
            else: 
                avg_minimum_time_to_collision += base_env.EPISODE_LENGTH
            
            obs = new_state
            
            if done:
                psc = info["PERSONAL_SPACE_COMPLIANCE"]
                failure_to_progress = info["FAILURE_TO_PROGRESS"]
                stalled_time = info["STALLED_TIME"]
                path_length = info["PATH_LENGTH"]
                vel_min = info["V_MIN"]
                vel_avg = info["V_AVG"]
                vel_max = info["V_MAX"]
                a_min = info["A_MIN"]
                a_avg = info["A_AVG"]
                a_max = info["A_MAX"]
                jerk_min = info["JERK_MIN"]
                jerk_avg = info["JERK_AVG"]
                jerk_max = info["JERK_MAX"]
        # After episode finishes
        if has_reached_goal and save_dir!= False:
            video_path = f"{save_dir}/{i}.mp4"
            imageio.mimwrite(video_path, frames, fps=30)
            print(f"Saved success episode to {video_path}")
        discomfort_sngnn += episode_discomfort_sngnn
        discomfort_dsrnn += episode_discomfort_dsrnn
        timeout += has_timed_out
        success_rate += has_reached_goal
        time_taken += steps
        closest_human_dist += min_human_dist
        closest_obstacle_dist += min_obstacle_dist
        collision_rate += has_collided
        collision_rate_human += has_collided_human
        collision_rate_object += has_collided_object
        collision_rate_wall += has_collided_wall
        total_psc += psc
        total_stl += stl
        total_spl += spl
        total_failure_to_progress += failure_to_progress
        total_stalled_time += stalled_time
        total_path_length += path_length
        total_vel_min += vel_min 
        total_vel_max += vel_max 
        total_vel_avg += vel_avg 
        total_a_min += a_min 
        total_a_max += a_max
        total_a_avg += a_avg 
        total_jerk_min += jerk_min 
        total_jerk_max += jerk_max 
        total_jerk_avg += jerk_avg
        total_avg_obstacle_distance += (avg_obstacle_dist / count)
        total_minimum_time_to_collision += (avg_minimum_time_to_collision / count)
        total_time_to_reach_goal += time_to_reach_goal

    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS ({num_episodes} episodes)")
    print(f"{'='*60}")
    print(f"Average discomfort_sngnn: {discomfort_sngnn/num_episodes:.4f}") 
    print(f"Average discomfort_dsrnn: {discomfort_dsrnn/num_episodes:.4f}") 
    
    print(f"\nSuccess Metrics:")
    print(f"  Success rate: {success_rate/num_episodes:.4f}") 
    print(f"  STL: {total_stl/num_episodes:.4f}")
    print(f"  SPL: {total_spl/num_episodes:.4f}")
    
    print(f"\nCollision Metrics:")
    print(f"  Overall collision rate: {collision_rate/num_episodes:.4f}")
    print(f"  Human collision rate: {collision_rate_human/num_episodes:.4f}")
    print(f"  Object collision rate: {collision_rate_object/num_episodes:.4f}")
    print(f"  Wall collision rate: {collision_rate_wall/num_episodes:.4f}")
    
    print(f"\nTime Metrics:")
    print(f"  Timeout rate: {timeout/num_episodes:.4f}") 
    print(f"  Average time taken: {time_taken/num_episodes:.2f} steps") 
    print(f"  Average time to reach goal: {total_time_to_reach_goal/num_episodes:.2f} steps")
    print(f"  Average failure to progress: {total_failure_to_progress/num_episodes:.2f}")
    print(f"  Average stalled time: {total_stalled_time/num_episodes:.2f}")
    
    print(f"\nPath Metrics:")
    print(f"  Average path length: {total_path_length/num_episodes:.4f} m")
    
    print(f"\nVelocity Metrics:")
    print(f"  Min velocity: {total_vel_min/num_episodes:.4f} m/s")
    print(f"  Avg velocity: {total_vel_avg/num_episodes:.4f} m/s")
    print(f"  Max velocity: {total_vel_max/num_episodes:.4f} m/s")
    
    print(f"\nAcceleration Metrics:")
    print(f"  Min acceleration: {total_a_min/num_episodes:.4f} m/s²")
    print(f"  Avg acceleration: {total_a_avg/num_episodes:.4f} m/s²")
    print(f"  Max acceleration: {total_a_max/num_episodes:.4f} m/s²")
    
    print(f"\nJerk Metrics:")
    print(f"  Min jerk: {total_jerk_min/num_episodes:.4f} m/s³")
    print(f"  Avg jerk: {total_jerk_avg/num_episodes:.4f} m/s³")
    print(f"  Max jerk: {total_jerk_max/num_episodes:.4f} m/s³")
    
    print(f"\nDistance Metrics:")
    print(f"  Closest human distance: {closest_human_dist/num_episodes:.4f} m") 
    print(f"  Closest obstacle distance: {closest_obstacle_dist/num_episodes:.4f} m") 
    print(f"  Average obstacle distance: {total_avg_obstacle_distance/num_episodes:.4f} m")
    print(f"  Personal space compliance: {total_psc/num_episodes:.4f}")
    print(f"  Minimum time to collision: {total_minimum_time_to_collision/num_episodes:.4f} s")
    print(f"{'='*60}\n")

    if(save_dir != False):
        results_text = (
            f"\n{'='*60}\n"
            f"EVALUATION RESULTS ({num_episodes} episodes)\n"
            f"{'='*60}\n"
            f"Average discomfort_sngnn: {discomfort_sngnn/num_episodes:.4f}\n"
            f"Average discomfort_dsrnn: {discomfort_dsrnn/num_episodes:.4f}\n"
            f"\nSuccess Metrics:\n"
            f"  Success rate: {success_rate/num_episodes:.4f}\n"
            f"  STL: {total_stl/num_episodes:.4f}\n"
            f"  SPL: {total_spl/num_episodes:.4f}\n"
            f"\nCollision Metrics:\n"
            f"  Overall collision rate: {collision_rate/num_episodes:.4f}\n"
            f"  Human collision rate: {collision_rate_human/num_episodes:.4f}\n"
            f"  Object collision rate: {collision_rate_object/num_episodes:.4f}\n"
            f"  Wall collision rate: {collision_rate_wall/num_episodes:.4f}\n"
            f"\nTime Metrics:\n"
            f"  Timeout rate: {timeout/num_episodes:.4f}\n"
            f"  Average time taken: {time_taken/num_episodes:.2f} steps\n"
            f"  Average time to reach goal: {total_time_to_reach_goal/num_episodes:.2f} steps\n"
            f"  Average failure to progress: {total_failure_to_progress/num_episodes:.2f}\n"
            f"  Average stalled time: {total_stalled_time/num_episodes:.2f}\n"
            f"\nPath Metrics:\n"
            f"  Average path length: {total_path_length/num_episodes:.4f} m\n"
            f"\nVelocity Metrics:\n"
            f"  Min velocity: {total_vel_min/num_episodes:.4f} m/s\n"
            f"  Avg velocity: {total_vel_avg/num_episodes:.4f} m/s\n"
            f"  Max velocity: {total_vel_max/num_episodes:.4f} m/s\n"
            f"\nAcceleration Metrics:\n"
            f"  Min acceleration: {total_a_min/num_episodes:.4f} m/s²\n"
            f"  Avg acceleration: {total_a_avg/num_episodes:.4f} m/s²\n"
            f"  Max acceleration: {total_a_max/num_episodes:.4f} m/s²\n"
            f"\nJerk Metrics:\n"
            f"  Min jerk: {total_jerk_min/num_episodes:.4f} m/s³\n"
            f"  Avg jerk: {total_jerk_avg/num_episodes:.4f} m/s³\n"
            f"  Max jerk: {total_jerk_max/num_episodes:.4f} m/s³\n"
            f"\nDistance Metrics:\n"
            f"  Closest human distance: {closest_human_dist/num_episodes:.4f} m\n"
            f"  Closest obstacle distance: {closest_obstacle_dist/num_episodes:.4f} m\n"
            f"  Average obstacle distance: {total_avg_obstacle_distance/num_episodes:.4f} m\n"
            f"  Personal space compliance: {total_psc/num_episodes:.4f}\n"
            f"  Minimum time to collision: {total_minimum_time_to_collision/num_episodes:.4f} s\n"
            f"{'='*60}\n"
        )
        
        import os

        filename = "evaluation_results.txt"
        filepath = os.path.join(save_dir, filename)
        
        with open(filepath, "w") as f:
            f.write(results_text)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-n", "--num_episodes", type=int, required=True, help="number of episodes")
    ap.add_argument("-w", "--weight_path", type=str, required=True, help="path to weight file")
    ap.add_argument("-c", "--config", type=str, required=True, help="path to config file")
    ap.add_argument("-s", "--save_dir", type=str, required=False, help="path to save episodes")
    args = vars(ap.parse_args())

    # Create folder if needed
    save_dir = False
    if args['save_dir']:
        save_dir = args['save_dir']
        os.makedirs(save_dir, exist_ok=True)
    
    
    print(f"Loading environment from config: {args['config']}")
    print(f"Using global max entities: {GLOBAL_MAX_ENTITIES}")
    
    # Create environment with fixed padding - MUST MATCH TRAINING
    env = gym.make("SocNavGym-v1", config=args["config"])
    env = FixedPaddingWrapper(env, GLOBAL_MAX_ENTITIES)
    env = DiscreteActions(env)
    
    print(f"Environment observation space: {env.observation_space}")
    print(f"Loading model from: {args['weight_path']}")

    try:
        # Load the model - it will automatically use the saved architecture
        # including the AttentionFeatureExtractor
        model = DQN.load(args["weight_path"])
        print("Successfully loaded model with attention-based feature extractor")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("\nIf you get an error about the feature extractor, make sure:")
        print("  1. The AttentionFeatureExtractor class is defined above")
        print("  2. GLOBAL_MAX_ENTITIES matches what was used during training")
        print("  3. You're using the same version of stable-baselines3")
        sys.exit(1)
        
    eval(model, args["num_episodes"], env)
