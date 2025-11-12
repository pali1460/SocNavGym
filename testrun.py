import gymnasium as gym
from stable_baselines3 import DQN
import socnavgym
from socnavgym.wrappers import DiscreteActions


# Load your trained model
model = DQN.load("sngnn_exp1_final-200000.zip")

# Create the SocNavGym environment
env = gym.make("SocNavGym-v1", config="environment_configs/exp2_with_sngnn.yaml")  # you can pass any config
env = DiscreteActions(env)  # creates an env with discrete action space

obs, info = env.reset()

# Run a few episodes to see it move
for episode in range(5):
    obs, info = env.reset()
    done = False
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        env.render()
        
env.close()
