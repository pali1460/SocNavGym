import os
import re
import torch
import argparse
import gymnasium as gym
import numpy as np
import random
from socnavgym.wrappers import DiscreteActions
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.utils import safe_mean

# ---------------------------------------
# ARGUMENTS
# ---------------------------------------
ap = argparse.ArgumentParser()
ap.add_argument("-e", "--env_config", help="path to environment config", required=True)
ap.add_argument("-r", "--run_name", help="name of the run", required=True)
ap.add_argument("-s", "--save_path", help="path to save checkpoints/models", required=True)
ap.add_argument("-p", "--project_name", help="Comet project name", default=None)
ap.add_argument("-a", "--api_key", help="Comet API key", default=None)
ap.add_argument("-d", "--use_deep_net", help="True or False, based on whether you want a transformer based feature extractor", required=False, default=False)
ap.add_argument("-g", "--gpu", help="GPU id to use", default="0")
ap.add_argument("--resume_from", help="path to specific checkpoint to resume from", default=None)
ap.add_argument("--seed", help="Random seed for reproducibility", type=int, default=42)
args = vars(ap.parse_args())

os.makedirs(args["save_path"], exist_ok=True)
TOTAL_TIMESTEPS = 200_000

# ---------------------------------------
# SET SEEDS FOR REPRODUCIBILITY
# ---------------------------------------
def set_seed(seed):
    """Set seeds for reproducibility across all random number generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Make PyTorch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"🎲 Set all random seeds to {seed}")

set_seed(args["seed"])

# ---------------------------------------
# DEVICE
# ---------------------------------------
device = f"cuda:{args['gpu']}" if torch.cuda.is_available() else "cpu"
print(f"🖥️ Using device: {device}")

# ---------------------------------------
# ENVIRONMENT
# ---------------------------------------
env = gym.make("SocNavGym-v1", config=args["env_config"])
env = DiscreteActions(env)
# Set environment seed for reproducibility
env.reset(seed=args["seed"])
env.action_space.seed(args["seed"])

net_arch = [512, 256, 128, 64] if not args["use_deep_net"] else [512, 256, 256, 256, 128, 128, 64]
policy_kwargs = {"net_arch": net_arch}

# ---------------------------------------
# CHECKPOINT LOGIC
# ---------------------------------------
def find_latest_checkpoint(path: str):
    """Find the latest checkpoint in the given directory."""
    ckpts = [f for f in os.listdir(path) if f.endswith(".zip") and not f.endswith("_final.zip")]
    if not ckpts:
        return None, 0
    latest = max(ckpts, key=lambda f: os.path.getmtime(os.path.join(path, f)))
    latest_path = os.path.join(path, latest)
    step_file = latest_path.replace(".zip", ".steps")
    prev_steps = 0
    if os.path.exists(step_file):
        with open(step_file) as f:
            prev_steps = int(f.read().strip())
    return latest_path, prev_steps

def load_full_checkpoint(ckpt_path, env, device):
    """Load model with replay buffer, RNG states, and return previous steps."""
    import gzip
    import shutil
    import pickle
    
    print(f"🔄 Loading checkpoint: {ckpt_path}")
    
    # Load the model
    model = DQN.load(ckpt_path, env=env, device=device)
    print(f"✅ Model loaded successfully")
    
    # Load replay buffer (try compressed first, then uncompressed)
    replay_buffer_path = ckpt_path.replace(".zip", "_replay_buffer.pkl")
    replay_buffer_gz_path = f"{replay_buffer_path}.gz"
    
    replay_loaded = False
    if os.path.exists(replay_buffer_gz_path):
        # Decompress the replay buffer
        print(f"📦 Found compressed replay buffer, decompressing...")
        with gzip.open(replay_buffer_gz_path, 'rb') as f_in:
            with open(replay_buffer_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
        model.load_replay_buffer(replay_buffer_path)
        replay_loaded = True
        print(f"🧠 Loaded replay buffer from {replay_buffer_gz_path}")
    elif os.path.exists(replay_buffer_path):
        model.load_replay_buffer(replay_buffer_path)
        replay_loaded = True
        print(f"🧠 Loaded replay buffer from {replay_buffer_path}")
    else:
        print("⚠️ No replay buffer found — starting with empty memory.")
    
    # Verify replay buffer was loaded
    if replay_loaded:
        buffer_size = model.replay_buffer.size()
        print(f"📊 Replay buffer size: {buffer_size} transitions")
        if buffer_size == 0:
            print("⚠️ WARNING: Replay buffer is EMPTY! This will cause training from scratch.")
    
    # Load RNG states for determinism
    rng_state_file = ckpt_path.replace(".zip", "_rng_state.pkl")
    if os.path.exists(rng_state_file):
        with open(rng_state_file, 'rb') as f:
            rng_states = pickle.load(f)
        
        random.setstate(rng_states['python'])
        np.random.set_state(rng_states['numpy'])
        torch.set_rng_state(rng_states['torch'])
        if rng_states['torch_cuda'] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_states['torch_cuda'])
        print(f"🎲 Restored RNG states for deterministic continuation")
    else:
        print("⚠️ No RNG state file found — randomness may differ from original run")
    
    # Get previous step count
    step_file = ckpt_path.replace(".zip", ".steps")
    prev_steps = 0
    if os.path.exists(step_file):
        with open(step_file) as f:
            prev_steps = int(f.read().strip())
        print(f"📊 Resuming from step {prev_steps}")
    
    # Verify exploration rate
    print(f"🎯 Current exploration rate: {model.exploration_rate:.4f}")
    
    return model, prev_steps

def save_full_checkpoint(model, save_path, run_name, current_steps, experiment=None):
    """Save model, replay buffer, step count, and RNG states."""
    import gzip
    import shutil
    import pickle
    
    ckpt_file = os.path.join(save_path, f"{run_name}_{current_steps}_steps.zip")
    replay_buffer_file = ckpt_file.replace(".zip", "_replay_buffer.pkl")
    step_file = ckpt_file.replace(".zip", ".steps")
    rng_state_file = ckpt_file.replace(".zip", "_rng_state.pkl")
    
    # Save model weights
    model.save(ckpt_file)
    
    # Save replay buffer
    model.save_replay_buffer(replay_buffer_file)
    
    # Save step count
    with open(step_file, "w") as f:
        f.write(str(current_steps))
    
    # Save RNG states for full determinism
    rng_states = {
        'python': random.getstate(),
        'numpy': np.random.get_state(),
        'torch': torch.get_rng_state(),
        'torch_cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }
    with open(rng_state_file, 'wb') as f:
        pickle.dump(rng_states, f)
    
    print(f"💾 Saved full checkpoint: {ckpt_file}")
    print(f"   ├─ Weights: {ckpt_file}")
    print(f"   ├─ Replay buffer: {replay_buffer_file}")
    print(f"   ├─ RNG states: {rng_state_file}")
    print(f"   └─ Steps: {step_file}")
    
    # Upload to Comet if available
    if experiment:
        try:
            # Upload model weights and steps
            experiment.log_asset(ckpt_file, step=current_steps)
            experiment.log_asset(step_file, step=current_steps)
            experiment.log_asset(rng_state_file, step=current_steps)
            
            # Compress replay buffer before uploading
            replay_buffer_gz = f"{replay_buffer_file}.gz"
            with open(replay_buffer_file, 'rb') as f_in:
                with gzip.open(replay_buffer_gz, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            
            rb_size_mb = os.path.getsize(replay_buffer_file) / (1024 * 1024)
            rb_gz_size_mb = os.path.getsize(replay_buffer_gz) / (1024 * 1024)
            print(f"   📦 Compressed replay buffer: {rb_size_mb:.1f}MB → {rb_gz_size_mb:.1f}MB")
            
            # Upload compressed replay buffer
            experiment.log_asset(replay_buffer_gz, step=current_steps)
            
            # Clean up compressed file
            os.remove(replay_buffer_gz)
            
            print("☁️ Uploaded checkpoint to Comet.")
        except Exception as e:
            print(f"⚠️ Comet upload failed: {e}")
    
    return ckpt_file

# Determine resume checkpoint
if args["resume_from"]:
    ckpt_path = args["resume_from"]
    print(f"📂 Resuming from user-specified checkpoint: {ckpt_path}")
elif os.path.isdir(args["save_path"]):
    ckpt_path, _ = find_latest_checkpoint(args["save_path"])
else:
    ckpt_path = None

# ---------------------------------------
# MODEL INIT
# ---------------------------------------
if ckpt_path and os.path.exists(ckpt_path):
    model, prev_steps = load_full_checkpoint(ckpt_path, env, device)
    # CRITICAL: Update model's internal timestep counter for schedules
    model.num_timesteps = prev_steps
    model._total_timesteps = TOTAL_TIMESTEPS
    print(f"🔧 Set model.num_timesteps = {prev_steps} for schedule continuation")
else:
    print("🚀 No checkpoint found — starting new model from scratch")
    model = DQN("MultiInputPolicy", env, verbose=1, policy_kwargs=policy_kwargs, device=device, seed=args["seed"])
    prev_steps = 0

# ---------------------------------------
# CALLBACKS
# ---------------------------------------
if args["api_key"]:
    from comet_ml import Experiment

    print(f"🔗 Logging to Comet (project: {args['project_name']})")
    experiment = Experiment(
        api_key=args["api_key"],
        project_name=args["project_name"],
        parse_args=False
    )
    experiment.set_name(args["run_name"])

    class CometCallback(CheckpointCallback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prev_steps = prev_steps
            
        def _on_step(self):
            if self.num_timesteps % self.save_freq == 0:
                current_steps = self.num_timesteps + self.prev_steps
                save_full_checkpoint(
                    self.model, 
                    self.save_path, 
                    args['run_name'], 
                    current_steps,
                    experiment
                )
            return True

        def _on_rollout_end(self):
            try:
                ep_rew = safe_mean([ep_info["r"] for ep_info in self.locals["self"].ep_info_buffer])
                ep_len = safe_mean([ep_info["l"] for ep_info in self.locals["self"].ep_info_buffer])
                experiment.log_metrics({
                    "ep_rew_mean": ep_rew,
                    "ep_len_mean": ep_len
                }, step=self.num_timesteps + self.prev_steps)
            except Exception as e:
                print(f"⚠️ Metric logging failed: {e}")

        def _on_training_end(self):
            final_path = os.path.join(self.save_path, f"{args['run_name']}_final.zip")
            
            # Only save model weights for final checkpoint
            self.model.save(final_path)
            print("🏁 Training complete — final model saved (weights only).")
            
            # Only upload weights to Comet
            experiment.log_asset(final_path)
            experiment.end()

    callback = CometCallback(save_freq=25_000, save_path=args["save_path"], verbose=1)
else:
    # For local-only checkpointing
    class LocalCheckpointCallback(CheckpointCallback):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.prev_steps = prev_steps
            
        def _on_step(self):
            if self.num_timesteps % self.save_freq == 0:
                current_steps = self.num_timesteps + self.prev_steps
                save_full_checkpoint(
                    self.model, 
                    self.save_path, 
                    args['run_name'], 
                    current_steps
                )
            return True
        
        def _on_training_end(self):
            final_path = os.path.join(self.save_path, f"{args['run_name']}_final.zip")
            
            # Only save model weights for final checkpoint
            self.model.save(final_path)
            print("🏁 Training complete — final model saved locally (weights only).")
    
    callback = LocalCheckpointCallback(
        save_freq=25_000,
        save_path=args["save_path"],
        verbose=1
    )

# ---------------------------------------
# TRAINING
# ---------------------------------------
remaining_steps = max(0, TOTAL_TIMESTEPS - prev_steps)

if remaining_steps == 0:
    print("✅ Training already complete — no steps left.")
else:
    print(f"🏋️ Resuming training for {remaining_steps} timesteps (already done {prev_steps}).")
    model.learn(total_timesteps=remaining_steps, callback=callback)

# Save final checkpoint (weights only - no replay buffer needed)
final_path = os.path.join(args["save_path"], f"{args['run_name']}_final.zip")
model.save(final_path)

print(f"✅ Training complete. Final model saved to {final_path}")
