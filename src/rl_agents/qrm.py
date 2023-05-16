"""
https://github.com/RodrigoToroIcarte/reward_machines/blob/master/reward_machines/rl_agents/qlearning/qlearning.py
"""

import random
import time

import IPython

import baselines_logger as logger
from consts import *

# from baselines import logger


def get_qmax(Q, s, actions, q_init):
    if s not in Q:
        Q[s] = dict([(a, q_init) for a in actions])
    return max(Q[s].values())


def get_best_action(Q, s, actions, q_init):
    qmax = get_qmax(Q, s, actions, q_init)
    best = [a for a in actions if Q[s][a] == qmax]
    return random.choice(best)


def learn(env,
          network=None,
          lr=0.1,
          total_timesteps=DEFAULT_STEPS,
          epsilon=0.1,
          print_freq=10000,
          gamma=0.9,
          q_init=DEFAULT_Q_INIT):
    """Train a tabular q-learning model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        This is just a placeholder to be consistent with the openai-baselines interface, but we don't really use state-approximation in tabular q-learning
    lr: float
        learning rate
    total_timesteps: int
        number of env steps to optimizer for
    epsilon: float
        epsilon-greedy exploration
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    gamma: float
        discount factor
    q_init: float
        initial q-value for unseen states
    """

    # Running Q-Learning
    reward_total = 0
    step = 0
    num_episodes = 0
    Q = {}
    actions = list(range(env.action_space.n))

    while step < total_timesteps:
        s = tuple(env.reset())
        if s not in Q:
            Q[s] = dict([(a, q_init) for a in actions])
        while True:
            # Selecting and executing the action
            a = random.choice(actions) if random.random(
            ) < epsilon else get_best_action(Q, s, actions, q_init)
            sn, r, done, info = env.step(a)
            sn = tuple(sn)

            experiences = []
            # Adding counterfactual experience (this will alrady include shaped rewards if use_rs=True)
            for _s, _a, _r, _sn, _done in info["crm-experience"]:
                experiences.append((tuple(_s), _a, _r, tuple(_sn), _done))

            for _s, _a, _r, _sn, _done in experiences:
                if _s not in Q:
                    Q[_s] = dict([(b, q_init) for b in actions])
                if _done:
                    _delta = _r - Q[_s][_a]
                else:
                    _delta = _r + gamma * \
                        get_qmax(Q, _sn, actions, q_init) - Q[_s][_a]
                Q[_s][_a] += lr*_delta

            # moving to the next state
            reward_total += r
            step += 1
            if step % print_freq == 0:
                logger.record_tabular("steps", step)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular(
                    "reward per step", reward_total / print_freq)
                logger.dump_tabular()
                reward_total = 0
            if done:
                num_episodes += 1
                break
            s = sn
    return Q
