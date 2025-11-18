import gymnasium as gym
import socnavgym
import torch
import torch.nn as nn
import numpy as np
from gymnasium import ObservationWrapper
from socnavgym.wrappers import DiscreteActions
from stable_baselines3 import DQN
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from math import sqrt
import argparse
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.results_plotter import ts2xy, plot_results
from stable_baselines3.common.utils import safe_mean
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor


class FixedPaddingWrapper(ObservationWrapper):
    """
    Wrapper that pads observations to a fixed size independent of the 
    environment's max_* config parameters. This allows different environments
    to have different entity distributions while maintaining compatible
    observation spaces for policy transfer.
    """
    
    def __init__(self, env, global_max_entities):
        """
        Args:
            env: The SocNavGym environment
            global_max_entities: Dict specifying the global maximum for padding
        """
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


# Global maximum entities across ALL environments
GLOBAL_MAX_ENTITIES = {
    "humans": 20,
    "tables": 10,
    "laptops": 5,
    "plants": 10,
    "walls": 200
}


ap = argparse.ArgumentParser()
ap.add_argument("-e", "--env_config", help="path to environment config", required=True)
ap.add_argument("-r", "--run_name", help="name of comet_ml run", required=True)
ap.add_argument("-s", "--save_path", help="path to save the model", required=True)
ap.add_argument("-p", "--project_name", help="project name in comet ml", required=False, default=None)
ap.add_argument("-a", "--api_key", help="api key to your comet ml profile", required=False, default=None)
ap.add_argument("-d", "--use_deep_net", help="True or False, based on whether you want a transformer based feature extractor", required=False, default=False)
ap.add_argument("-g", "--gpu", help="gpu id to use", required=False, default="0")
ap.add_argument("-t", "--total_timesteps", help="total timesteps to train", required=False, default=1000000, type=int)
ap.add_argument("-b", "--buffer_size", help="replay buffer size", required=False, default=50000, type=int)
args = vars(ap.parse_args())


if args["api_key"] is not None:
    from comet_ml import Experiment

    class CometMLCallback(BaseCallback):
        """
        A custom callback for logging to Comet ML without saving checkpoints.
        """
        def __init__(self, run_name:str, project_name:str, api_key:str, verbose=0):
            super(CometMLCallback, self).__init__(verbose)
            print("Logging using comet_ml")
            self.run_name = run_name
            self.experiment = Experiment(
                api_key=api_key,
                project_name=project_name,
                parse_args=False   
            )
            self.experiment.set_name(self.run_name)
            # Log the global max entities configuration
            self.experiment.log_parameters(GLOBAL_MAX_ENTITIES)
            self.experiment.log_parameter("buffer_size", args["buffer_size"])

        def _on_step(self) -> bool:
            """
            This method is called after each step.
            Return False to stop training.
            """
            return True

        def _on_rollout_end(self) -> None:
            """
            This event is triggered before updating the policy.
            """
            metrics = {
                "rollout/ep_rew_mean": safe_mean([ep_info["r"] for ep_info in self.model.ep_info_buffer]),
                "rollout/ep_len_mean": safe_mean([ep_info["l"] for ep_info in self.model.ep_info_buffer])
            }
            if len(self.model.ep_success_buffer) > 0:
                metrics["rollout/success_rate"] = safe_mean(self.model.ep_success_buffer)

            l = [
                "train/loss",
                "train/n_updates",
            ]

            for val in l:
                if val in self.logger.name_to_value.keys():
                    metrics[val] = self.logger.name_to_value[val]

            step = self.model.num_timesteps

            self.experiment.log_metrics(metrics, step=step)
        
        def _on_training_end(self) -> None:
            """
            This event is triggered at the end of training.
            """
            print("Training completed. Experiment will be finalized.")


# Create environment with fixed padding
env = gym.make("SocNavGym-v1", config=args["env_config"])
env = FixedPaddingWrapper(env, GLOBAL_MAX_ENTITIES)
env = DiscreteActions(env)

print(f"Environment observation space: {env.observation_space}")
print(f"Using global max entities: {GLOBAL_MAX_ENTITIES}")
print(f"Replay buffer size: {args['buffer_size']}")

net_arch = {}

if not args["use_deep_net"]:
    net_arch = [512, 256, 128, 64]
else:
    net_arch = [512, 256, 256, 256, 128, 128, 64]

policy_kwargs = {"net_arch": net_arch}

device = 'cuda:'+str(args["gpu"]) if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# Create DQN model with reduced buffer size
model = DQN(
    "MultiInputPolicy", 
    env, 
    verbose=1, 
    policy_kwargs=policy_kwargs, 
    device=device,
    buffer_size=args["buffer_size"],  # Reduced from default 1000000
    learning_starts=1000,  # Start learning after 1000 steps
    batch_size=128,  # Batch size for training
    tau=1.0,  # Soft update coefficient
    gamma=0.99,  # Discount factor
    train_freq=4,  # Update the model every 4 steps
    gradient_steps=1,  # How many gradient steps after each rollout
    target_update_interval=1000,  # Update target network every 1000 steps
)

if args["api_key"] is not None:
    callback = CometMLCallback(args["run_name"], args["project_name"], args["api_key"])
else:
    callback = None
    
print(f"Starting training for {args['total_timesteps']} timesteps...")
model.learn(total_timesteps=args["total_timesteps"], callback=callback)

# Save final model locally
final_save_path = args["save_path"] + "_final"
model.save(final_save_path)
print(f"Model saved locally to {final_save_path}.zip")

# Upload model to Comet ML if using it
if args["api_key"] is not None:
    try:
        callback.experiment.log_model("final_model", final_save_path + ".zip")
        print(f"Model uploaded to Comet ML")
    except Exception as e:
        print(f"Warning: Could not upload model to Comet ML: {e}")
    
    # End the experiment
    callback.experiment.end()
    print("Comet ML experiment ended")

print("Training complete!")
