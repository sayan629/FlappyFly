import flappy_bird_gymnasium
import gymnasium as gym
from dqn import DQN

if torch.backends.mps.is_available():
    device = "mps"
elif torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"



def run(self, is_training=True, render=False):
    env = gym.make("FlappyBird-v0", render_mode="human" if render else None)
    state, _ = env.reset()

    while True:
        # Next ation
        # (feed the obervation to your agent and get the next action)
        action = env.action_space.sample()  # This is just a random action, replace it with your agent's action
        
        # Processing: terminated => done
        next_state, reward, terminated, _, _ = env.step(action)
        
        # checking if the player is still alive
        if terminated:
            break

    env.close() 

