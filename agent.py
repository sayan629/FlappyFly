import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN
from experience_replay import ReplayMemory
import itertools
import yaml
import torch
import torch.nn as nn
import torch.optim as optim

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


class Agent():
    
    def __init__(self, param_set):
        self.param_set = param_set
        
        with open("parameters.yaml", "r") as f:
            all_param_set = yaml.safe_load(f)
            params = all_param_set[param_set]
        
        self.alpha = params["alpha"]
        self.gamma = params["gamma"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_init = params["epsilon_init"]
        self.epsilon_init = params["epsilon_init"]
            
            
            
    def run(self, is_training=True, render=False):
        env = gym.make("FlappyBird-v0", render_mode="human" if render else None)
        
        num_states = env.observation_space.shape[0] # input dim
        num_actions = env.action_space.n # output dim
        
        policy_dqn = DQN(num_states, num_actions).to(device)
    
        
        if is_training:
            memory = ReplayMemory(10000)
            
        for episode in itertools.count():
            
            state, _ = env.reset()
            episode_rewards = 0
            terminated = False

            while not terminated:
                # Next ation
                # (feed the obervation to your agent and get the next action)
                action = env.action_space.sample()  # This is just a random action, replace it with your agent's action
                
                # Processing: terminated => done
                next_state, reward, terminated, _, _ = env.step(action)
                
                if is_training:
                    # Store the transition in memory
                    memory.append((state, action, next_state, reward, terminated))
                
                state = next_state
                episode_rewards += reward
            
            print(f"episode: {episode+1} with total reward: {episode_rewards}")

    # env.close() -- manually stop the game by closing the window, otherwise it will run forever



