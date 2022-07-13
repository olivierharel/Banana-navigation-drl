import numpy as np
import random
from collections import deque, namedtuple
#from recordclass import recordclass

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 64         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate 
UPDATE_EVERY = 4        # how often to update the network

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, use_double_dqn=False, prioritized_replay=(False,0.,0.,0.,1.01), use_dueling=False):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            use_double_dqn (bool): if true use double dqn for learning
            prioritized_replay (tuple(bool, float, float, float, float)): enable / epsilon (>0) / alpha (0.0 < <= 1.0) / beta (0.0 < <=1.0) / beta decay (>= 1)
            use_dueling (bool): if true use dueling networks
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.use_double_dqn = use_double_dqn

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed, use_dueling=use_dueling).to(device)
        self.qnetwork_target = QNetwork(state_size, action_size, seed, use_dueling=use_dueling).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed, prioritized_replay)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones, exp_indices, is_weights = experiences

        # Get max predicted Q values (for next states) from target model
        if self.use_double_dqn:
            local_max_actions = self.qnetwork_local(next_states).detach().max(1)[1].unsqueeze(1)
            Q_targets_next    = self.qnetwork_target(next_states).detach().gather(1, local_max_actions)
        else:
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)

        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        #Update experience replay buffer weights
        self.memory.update_weight((Q_expected - Q_targets).detach(), exp_indices)

        # Compute loss
        #loss = F.mse_loss(Q_expected, Q_targets)
        loss = F.mse_loss((Q_expected - Q_targets)*is_weights,
                          torch.zeros(Q_expected.size()).to(device))

        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)                     

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, prioritized_replay):
        """Initialize a ReplayBuffer object.

        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
            prioritized_replay (tuple(bool, float, float, float, float)): enable / epsilon (>0) / alpha (0.0 < <= 1.0) / beta (0.0 < <=1.0) / beta decay (>= 1)
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        #self.experience = recordclass("Experience", "state action reward next_state done weight")
        self.seed = random.seed(seed)
        self.prioritized_replay = list(prioritized_replay)
        if prioritized_replay[0]:
            self.default_weight = prioritized_replay[1]**prioritized_replay[2]
        else:
            self.default_weight = 0
        self.weight        = np.zeros(buffer_size)
        self.min_weight    = self.default_weight
        self.sum_weight    = 0
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        # Sampling weight is set to self.prioritized_replay 'epsilon' by default
        self.memory.append(e)        
        if len(self.memory) == BUFFER_SIZE:
            removed = self.weight[0]
            self.sum_weight -= removed
        self.weight = np.roll(self.weight,-1)
        self.weight[-1] = self.default_weight
        self.sum_weight += self.default_weight
        if len(self.memory) == BUFFER_SIZE and removed == self.min_weight:
            self.min_weight = self.weight[-len(self.memory):].min()
        else: 
            self.min_weight = min(self.min_weight, self.default_weight)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        if not self.prioritized_replay[0]:
            exp_indices = None
            experiences = random.sample(self.memory, k=self.batch_size)
            is_weights = torch.ones((len(experiences),1)).float().to(device)
        else:
            min_weight = self.min_weight
            exp_indices = np.random.choice(np.arange(len(self.memory),dtype=int),
                                           self.batch_size,
                                           p=self.weight[-len(self.memory):]/self.sum_weight)
                                    
            experiences = [self.memory[i] for i in exp_indices]
            is_weights = torch.from_numpy(
                            np.vstack([pow(min_weight/self.weight[-len(self.memory)+i],self.prioritized_replay[3]) for i in exp_indices])).float().to(device).reshape((len(experiences),1))

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones, exp_indices, is_weights)

    def update_weight(self, delta, exp_indices):
        recalc_min = False
        if self.prioritized_replay[0]:
            for idx, exp_idx in enumerate(exp_indices):
                if self.weight[exp_idx+BUFFER_SIZE-len(self.memory)] == self.min_weight:
                    recalc_min = True
                self.sum_weight -= self.weight[exp_idx+BUFFER_SIZE-len(self.memory)]
                update_weight = pow(abs(float(delta[idx])) + self.prioritized_replay[1],self.prioritized_replay[2])
                self.weight[exp_idx+BUFFER_SIZE-len(self.memory)] = update_weight
                self.sum_weight += update_weight
                self.min_weight = min(update_weight, self.min_weight)
            if recalc_min:
                self.min_weight = self.weight[-len(self.memory):].min()
                    
            one_m_beta = 1 - self.prioritized_replay[3]
            one_m_beta *= (self.prioritized_replay[4] ** len(exp_indices))
            self.prioritized_replay[3] = max(0.01, min(1, 1 - one_m_beta))

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)