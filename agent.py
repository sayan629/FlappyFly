import flappy_bird_gymnasium
import gymnasium as gym







env = gym.make("FlappyBird-v0", render_mode="human")
state, _ = env.reset()

While True:
    # Next ation
    # (feed the obervation to your agent and get the next action)