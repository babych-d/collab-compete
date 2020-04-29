import time

import numpy as np
import torch
from unityagents import UnityEnvironment

from maddpg import MADDPG


def run_agents(n_episodes=5):
    env = UnityEnvironment(file_name="envs/Tennis.app")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size
    num_agents = env_info.vector_observations.shape[0]
    maddpg = MADDPG(state_size=state_size, action_size=action_size, num_agents=num_agents)

    for i, agent in enumerate(maddpg.agents):
        agent.actor_local.load_state_dict(torch.load(f'models/checkpoint_actor_local_{i}.pth'))
        agent.critic_local.load_state_dict(torch.load(f'models/checkpoint_critic_local_{i}.pth'))

    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        while True:
            actions = maddpg.act(states, add_noise=True)
            env_info = env.step(actions)[brain_name]
            states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += rewards
            if any(dones):
                break
        print(f"Epsiode {i_episode}. Rewards of two agents: {scores}")


if __name__ == '__main__':
    run_agents()
