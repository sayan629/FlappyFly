import gymnasium as gym
import flappy_bird_gymnasium
import pygame

# Creating our env
env = gym.make("FlappyBird-v0", render_mode="human")
state, info = env.reset()
done = False