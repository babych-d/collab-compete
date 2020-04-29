import numpy as np
import random
from collections import namedtuple, deque

import torch

from ddpg import DDPG


BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
GAMMA = 0.99
NOISE_END = 0.1
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class MADDPG:
    """The main class that defines and trains all the agents"""

    def __init__(self, state_size, action_size, num_agents):
        self.state_size = state_size
        self.action_size = action_size
        self.num_agents = num_agents
        self.whole_action_dim = self.action_size * self.num_agents
        self.memory = ReplayBuffer(BUFFER_SIZE, BATCH_SIZE)
        self.agents = [DDPG(state_size, action_size, num_agents),
                       DDPG(state_size, action_size, num_agents)]

    def step(self, states, actions, rewards, next_states, dones):
        """Save experience in replay memory, and use random sample from buffer to learn."""
        full_states = np.reshape(states, newshape=(-1))
        full_next_states = np.reshape(next_states, newshape=(-1))

        self.memory.add(full_states, states, actions, rewards, full_next_states, next_states, dones)

        if len(self.memory) > BATCH_SIZE:
            for agent_no in range(self.num_agents):
                samples = self.memory.sample()
                self.learn(samples, agent_no, GAMMA)

    def learn(self, samples, agent_no, gamma):
        full_states, states, actions, rewards, full_next_states, next_states, dones = samples

        critic_full_next_actions = torch.zeros(states.shape[:2] + (self.action_size,), dtype=torch.float, device=DEVICE)
        for agent_id, agent in enumerate(self.agents):
            agent_next_state = next_states[:, agent_id, :]
            critic_full_next_actions[:, agent_id, :] = agent.actor_target.forward(agent_next_state)
        critic_full_next_actions = critic_full_next_actions.view(-1, self.whole_action_dim)

        agent = self.agents[agent_no]
        agent_state = states[:, agent_no, :]
        actor_full_actions = actions.clone()
        actor_full_actions[:, agent_no, :] = agent.actor_local.forward(agent_state)
        actor_full_actions = actor_full_actions.view(-1, self.whole_action_dim)

        full_actions = actions.view(-1, self.whole_action_dim)

        agent_rewards = rewards[:, agent_no].view(-1, 1)
        agent_dones = dones[:, agent_no].view(-1, 1)
        experiences = (full_states, actor_full_actions, full_actions, agent_rewards,
                       agent_dones, full_next_states, critic_full_next_actions)
        agent.learn(experiences, gamma)

    def act(self, full_states, add_noise=True):
        actions = []
        for agent_id, agent in enumerate(self.agents):
            action = agent.act(np.reshape(full_states[agent_id, :], newshape=(1, -1)), add_noise)
            action = np.reshape(action, newshape=(1, -1))
            actions.append(action)
        actions = np.concatenate(actions, axis=0)
        return actions


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["full_state", "state", "action", "reward",
                                                                "full_next_state", "next_state", "done"])

    def add(self, full_state, state, action, reward, full_next_state, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(full_state, state, action, reward, full_next_state, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        full_states = torch.from_numpy(np.array([e.full_state for e in experiences if e is not None])).float().to(
            DEVICE)
        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float().to(DEVICE)
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).float().to(DEVICE)
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float().to(DEVICE)
        full_next_states = torch.from_numpy(
            np.array([e.full_next_state for e in experiences if e is not None])).float().to(DEVICE)
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float().to(
            DEVICE)
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            DEVICE)

        return full_states, states, actions, rewards, full_next_states, next_states, dones

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
