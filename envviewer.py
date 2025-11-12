import socnavgym
import gymnasium as gym
env = gym.make("SocNavGym-v1", config="./exp1_simple_custom.yaml") 
obs, _ = env.reset()


for i in range(1000):
    obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
    env.render()
    if terminated or truncated:
        env.reset()
