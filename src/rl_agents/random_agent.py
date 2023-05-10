import gym
import numpy as np


class RandomAgent:
    def __init__(self, action_space):
        self.action_space = action_space

    def choose_action(self, observation):
        return self.action_space.sample()  # Choose a random action
