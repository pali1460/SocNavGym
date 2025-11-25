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
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import safe_mean


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
        # Robot (query) attends to entities (keys/values)
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
            nn.Linear(entity_embed_dim * 2, 256),  # robot + attended entities
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )
        
    def forward(self, observations):
        """
        Forward pass with attention mechanism
        
        Args:
            observations: Dict with keys 'robot', 'humans', 'tables', etc.
        
        Returns:
            features: Tensor of shape (batch_size, features_dim)
        """
        robot_state = observations["robot"]  # (batch_size, 9)
        batch_size = robot_state.shape[0]
        
        # Collect all entity observations and filter padding
        all_entities = []
        entity_types = ["humans", "tables", "laptops", "plants", "walls"]
        
        for entity_type in entity_types:
            if entity_type in observations:
                entity_obs = observations[entity_type]  # (batch_size, N*14)
                # Reshape to (batch_size, N, 14)
                num_entities = entity_obs.shape[1] // 14
                entity_obs = entity_obs.reshape(batch_size, num_entities, 14)
                all_entities.append(entity_obs)
        
        # Concatenate all entities: (batch_size, total_entities, 14)
        if len(all_entities) > 0:
            all_entities = torch.cat(all_entities, dim=1)
        else:
            # No entities, create empty tensor
            all_entities = torch.zeros(batch_size, 1, 14, device=robot_state.device)
        
        num_entities = all_entities.shape[1]
        
        # Create mask for padding (entities with all zeros are padding)
        # entity_mask: True for real entities, False for padding
        entity_mask = (all_entities.abs().sum(dim=-1) > 1e-6)  # (batch_size, num_entities)
        
        # Encode entities
        # Flatten for batch processing: (batch_size * num_entities, 14)
        entities_flat = all_entities.reshape(-1, 14)
        encoded_entities = self.entity_encoder(entities_flat)
        # Reshape back: (batch_size, num_entities, entity_embed_dim)
        encoded_entities = encoded_entities.reshape(batch_size, num_entities, self.entity_embed_dim)
        
        # Encode robot state: (batch_size, entity_embed_dim)
        robot_encoded = self.robot_encoder(robot_state)
        # Add sequence dimension: (batch_size, 1, entity_embed_dim)
        robot_query = robot_encoded.unsqueeze(1)
        
        # Apply attention: robot attends to entities
        # key_padding_mask: True for positions to IGNORE (padding)
        # So we need to invert our entity_mask
        padding_mask = ~entity_mask  # True where we should ignore
        
        # Attention output: (batch_size, 1, entity_embed_dim)
        attended, attention_weights = self.attention(
            query=robot_query,
            key=encoded_entities,
            value=encoded_entities,
            key_padding_mask=padding_mask,
            need_weights=True
        )
        
        # Remove sequence dimension: (batch_size, entity_embed_dim)
        attended = attended.squeeze(1)
        
        # Apply layer norm
        attended = self.layer_norm(attended)
        
        # Combine robot state and attended entity features
        combined = torch.cat([robot_encoded, attended], dim=-1)  # (batch_size, entity_embed_dim*2)
        
        # Final feature extraction
        features = self.aggregator(combined)  # (batch_size, features_dim)
        
        return features


# Global maximum entities across ALL environments
# These can be more conservative than before since attention handles padding well
GLOBAL_MAX_ENTITIES = {
    "humans": 20,
    "tables": 10,
    "laptops": 5,
    "plants": 10,
    "walls": 80  # Reduced from 200 - attention makes this less critical
}


# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--env_config", help="path to environment config", required=True)
ap.add_argument("-r", "--run_name", help="name of comet_ml run", required=True)
ap.add_argument("-s", "--save_path", help="path to save the model", required=True)
ap.add_argument("-p", "--project_name", help="project name in comet ml", required=False, default=None)
ap.add_argument("-a", "--api_key", help="api key to your comet ml profile", required=False, default=None)
ap.add_argument("-g", "--gpu", help="gpu id to use", required=False, default="0")
ap.add_argument("-t", "--total_timesteps", help="total timesteps to train", required=False, default=1000000, type=int)
ap.add_argument("-b", "--buffer_size", help="replay buffer size", required=False, default=100000, type=int)
ap.add_argument("-f", "--features_dim", help="dimension of attention feature extractor output", required=False, default=256, type=int)
ap.add_argument("-n", "--num_heads", help="number of attention heads", required=False, default=4, type=int)
args = vars(ap.parse_args())


# Setup Comet ML callback if API key provided
if args["api_key"] is not None:
    from comet_ml import Experiment

    class CometMLCallback(BaseCallback):
        """Custom callback for logging to Comet ML"""
        
        def __init__(self, run_name:str, project_name:str, api_key:str, log_freq=1000, verbose=0):
            super(CometMLCallback, self).__init__(verbose)
            print("Logging using comet_ml")
            self.run_name = run_name
            self.log_freq = log_freq
            self.experiment = Experiment(
                api_key=api_key,
                project_name=project_name,
                parse_args=False   
            )
            self.experiment.set_name(self.run_name)
            # Log configuration
            self.experiment.log_parameters(GLOBAL_MAX_ENTITIES)
            self.experiment.log_parameter("buffer_size", args["buffer_size"])
            self.experiment.log_parameter("features_dim", args["features_dim"])
            self.experiment.log_parameter("num_attention_heads", args["num_heads"])
            self.experiment.log_parameter("feature_extractor", "attention")

        def _on_step(self) -> bool:
            """Log metrics every log_freq steps"""
            if self.n_calls % self.log_freq == 0:
                metrics = {}
                
                # Log episode statistics if available
                if len(self.model.ep_info_buffer) > 0:
                    metrics["rollout/ep_rew_mean"] = safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer])
                    metrics["rollout/ep_len_mean"] = safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
                
                if len(self.model.ep_success_buffer) > 0:
                    metrics["rollout/success_rate"] = safe_mean(self.model.ep_success_buffer)

                # Log all available training metrics
                if self.logger is not None:
                    for key, value in self.logger.name_to_value.items():
                        metrics[key] = value

                step = self.model.num_timesteps

                if metrics:
                    self.experiment.log_metrics(metrics, step=step)
                    
                    if self.verbose > 0:
                        print(f"Step {step}: Logged {len(metrics)} metrics to Comet ML")
            
            return True
        
        def _on_training_end(self) -> None:
            """Finalize experiment at end of training"""
            print("Training completed. Experiment will be finalized.")


# Create environment with fixed padding
print("Creating environment...")
env = gym.make("SocNavGym-v1", config=args["env_config"])
env = FixedPaddingWrapper(env, GLOBAL_MAX_ENTITIES)
env = DiscreteActions(env)

print(f"\nEnvironment Configuration:")
print(f"  Observation space: {env.observation_space}")
print(f"  Global max entities: {GLOBAL_MAX_ENTITIES}")
print(f"  Action space: {env.action_space}")

# Calculate total observation dimension
total_obs_dim = sum([space.shape[0] for space in env.observation_space.spaces.values()])
print(f"  Total observation dimension: {total_obs_dim}")

# Setup policy with attention-based feature extractor
print(f"\nModel Configuration:")
print(f"  Feature extractor: Attention-based")
print(f"  Features dimension: {args['features_dim']}")
print(f"  Number of attention heads: {args['num_heads']}")
print(f"  Replay buffer size: {args['buffer_size']}")

policy_kwargs = {
    "features_extractor_class": AttentionFeatureExtractor,
    "features_extractor_kwargs": {
        "features_dim": args["features_dim"],
        "num_heads": args["num_heads"]
    },
    "net_arch": [256, 128]  # Smaller Q-network since feature extractor does heavy lifting
}

device = 'cuda:'+str(args["gpu"]) if torch.cuda.is_available() else 'cpu'
print(f"  Device: {device}")

# Create PPO model
model = PPO(
    "MultiInputPolicy",
    env,
    policy_kwargs=policy_kwargs,  # Same attention extractor!
    n_steps=2048,
    batch_size=128,
    n_epochs=10,
    learning_rate=3e-4,
    clip_range=0.2,
    verbose=1
)

# Setup callback
if args["api_key"] is not None:
    callback = CometMLCallback(args["run_name"], args["project_name"], args["api_key"], verbose=1)
else:
    callback = None
    
print(f"\nStarting training for {args['total_timesteps']} timesteps...")
print("="*60)

# Train
model.learn(total_timesteps=args["total_timesteps"], callback=callback)

# Save final model locally
final_save_path = args["save_path"] + "_final"
model.save(final_save_path)
print(f"\nModel saved locally to {final_save_path}.zip")

# Upload to Comet ML if using it
if args["api_key"] is not None:
    try:
        callback.experiment.log_model("final_model", final_save_path + ".zip")
        print(f"Model uploaded to Comet ML")
    except Exception as e:
        print(f"Warning: Could not upload model to Comet ML: {e}")
    
    callback.experiment.end()
    print("Comet ML experiment ended")

print("\n" + "="*60)
print("Training complete!")
print("="*60)
