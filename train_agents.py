import random
from collections import deque

import numpy as np
import torch
import matplotlib.pyplot as plt
from unityagents import UnityEnvironment

from maddpg import MADDPG

PRINT_EVERY = 100


def seeding(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_agents(n_episodes=10000, t_max=1000):
    env = UnityEnvironment(file_name="envs/Tennis.app")
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]
    env_info = env.reset(train_mode=True)[brain_name]

    seeding(seed=42)
    state_size = env_info.vector_observations.shape[1]
    action_size = brain.vector_action_space_size
    num_agents = env_info.vector_observations.shape[0]
    maddpg = MADDPG(state_size=state_size, action_size=action_size, num_agents=num_agents)

    scores_deque = deque(maxlen=100)
    scores_list = []
    for i_episode in range(1, n_episodes + 1):
        env_info = env.reset(train_mode=True)[brain_name]
        states = env_info.vector_observations
        scores = np.zeros(num_agents)
        for _ in range(t_max):
            actions = maddpg.act(states)
            env_info = env.step(actions)[brain_name]
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += rewards
            maddpg.step(states, actions, rewards, next_states, dones)
            states = next_states
            if np.any(dones):
                break

        scores_deque.append(np.max(scores))
        scores_list.append(np.max(scores))

        print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque)}', end="")
        if i_episode % PRINT_EVERY == 0:
            print(f'\rEpisode {i_episode}\tAverage Score: {np.mean(scores_deque) : .3f}')

        if np.mean(scores_deque) >= 2.0 and len(scores_deque) >= 100:
            for i, agent in enumerate(maddpg.agents):
                torch.save(agent.actor_local.state_dict(), f'models/checkpoint_actor_local_{i}.pth')
                torch.save(agent.critic_local.state_dict(), f'models/checkpoint_critic_local_{i}.pth')
            print(f'\nSaved Model: Episode {i_episode}\tAverage Score: {np.mean(scores_deque) : .3f}')
            break

    return scores_list


def save_scores(scores):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(1, len(scores) + 1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig("models/scores.png")


def main():
    scores = train_agents()
    save_scores(scores)


if __name__ == '__main__':
    main()
