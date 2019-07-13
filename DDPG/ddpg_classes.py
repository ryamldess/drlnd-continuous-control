"""
The ReacherCCTask class.
"""

import numpy as np
from unityagents import UnityEnvironment

class ReacherCCTask:
    """Task (environment) that defines the goal and provides feedback to the agent."""
    def __init__(self, env, brain_name, state_size, action_size, action_low, action_high):
        self.env = env
        self.brain_name = brain_name
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high

    def step(self, action):
        """Uses action to obtain next state, reward, done."""
        action = np.clip(action, -1, 1)
        env_info = self.env.step(action)[self.brain_name]
        next_state = env_info.vector_observations   # next state
        reward = env_info.rewards                   # reward
        #print(reward)
        done = env_info.local_done
        
        if np.any(done):
            next_state = self.reset()
            
        return np.array(next_state), np.array(reward), np.array(done)

    def reset(self):
        """Reset the sim to start a new episode."""
        env_info = self.env.reset(train_mode=True)[self.brain_name]
        return np.array(env_info.vector_observations)

""""
The ReplayBuffer_Keras class.
"""

import random
from collections import namedtuple, deque

class ReplayBuffer_Keras:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, batch_size):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size: maximum size of buffer
            batch_size: size of each training batch
        """
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self, batch_size=64):
        """Randomly sample a batch of experiences from memory."""
        return random.sample(self.memory, k=self.batch_size)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

"""
The ReplayBuffer_Torch class.
"""
    
from keras import layers, models, optimizers
from keras import backend as K
from keras.layers.normalization import BatchNormalization

from collections import namedtuple, deque
import numpy as np
import random
import torch

class ReplayBuffer_Torch:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed, device):
        """Initialize a ReplayBuffer object.
        Params
        ======
            action_size (int): dimension of each action
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
        self.device = device
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)
        
        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(self.device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(self.device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(self.device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(self.device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(self.device)
  
        return (states, actions, rewards, next_states, dones)
        
    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

"""
The Actor_Keras class.
"""

class Actor_Keras:
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, action_low, action_high):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            action_low (array): Min value of each action dimension
            action_high (array): Max value of each action dimension
        """
        self.state_size = state_size
        self.action_size = action_size
        self.action_low = action_low
        self.action_high = action_high
        self.action_range = self.action_high - self.action_low

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build an actor (policy) network that maps states -> actions."""
        # Define input layer (states)
        states = layers.Input(shape=(self.state_size,), name='states')

        # Add hidden layers
        net = layers.Dense(units=32, activation='relu')(states)
        bn1 = BatchNormalization()(net)
        drop1 = layers.Dropout(0.5)(bn1)
        net = layers.Dense(units=64, activation='relu')(drop1)
        bn2 = BatchNormalization()(net)
        drop2 = layers.Dropout(0.5)(bn2)
        net = layers.Dense(units=32, activation='relu')(drop2)
        #net = layers.Dense(units=32, activation='sigmoid')(net)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Add final output layer with sigmoid activation
        raw_actions = layers.Dense(units=self.action_size, activation='sigmoid',
            name='raw_actions')(net)

        # Scale [0, 1] output for each action dimension to proper range
        actions = layers.Lambda(lambda x: (x * self.action_range) + self.action_low,
            name='actions')(raw_actions)

        # Create Keras model
        self.model = models.Model(inputs=states, outputs=actions)

        # Define loss function using action value (Q value) gradients
        action_gradients = layers.Input(shape=(self.action_size,))
        loss = K.mean(-action_gradients * actions)

        # Incorporate any additional losses here (e.g. from regularizers)

        # Define optimizer and training function
        optimizer = optimizers.Adam(lr=0.0001)
        updates_op = optimizer.get_updates(params=self.model.trainable_weights, loss=loss)
        self.train_fn = K.function(
            inputs=[self.model.input, action_gradients, K.learning_phase()],
            outputs=[],
            updates=updates_op)

"""
Global hidden_init function for Actor_Torch and Critic_Torch.
"""

import numpy as np

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

"""
The Actor_Torch class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Actor_Torch(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_units, seed, gate=F.relu, final_gate=F.tanh):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_units (array): Number of nodes for layers
            seed (int): Random seed
            gate (function): activation function
            final_gate (function): final activation function
        """
        super(Actor_Torch, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.gate = gate
        self.final_gate = final_gate
        self.normalizer = nn.BatchNorm1d(state_size)
        dims = (state_size, ) + hidden_units
        self.layers = nn.ModuleList([nn.Linear(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:])])
        self.output = nn.Linear(dims[-1], action_size)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states):
        """Build an actor (policy) network that maps states -> actions."""
        x = self.normalizer(states)
        for layer in self.layers:
            x = self.gate(layer(x))
        return self.final_gate(self.output(x))

"""
The Critic_Keras class.
"""
        
class Critic_Keras:
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size):
        """Initialize parameters and build model.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
        """
        self.state_size = state_size
        self.action_size = action_size

        # Initialize any other variables here

        self.build_model()

    def build_model(self):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        # Define input layers
        states = layers.Input(shape=(self.state_size,), name='states')
        actions = layers.Input(shape=(self.action_size,), name='actions')

        # Add hidden layer(s) for state pathway
        net_states = layers.Dense(units=32, activation='relu')(states)
        bn1 = BatchNormalization()(net_states)
        #drop1 = layers.Dropout(0.5)(bn1)
        net_states = layers.Dense(units=64, activation='relu')(bn1)
        
        # Add hidden layer(s) for action pathway
        net_actions = layers.Dense(units=32, activation='relu')(actions)
        bn2 = BatchNormalization()(net_actions)
        #drop2 = layers.Dropout(0.5)(bn2)
        net_actions = layers.Dense(units=64, activation='relu')(bn2)

        # Try different layer sizes, activations, add batch normalization, regularizers, etc.

        # Combine state and action pathways
        net = layers.Add()([net_states, net_actions])
        net = layers.Activation('relu')(net)

        # Add more layers to the combined network if needed

        # Add final output layer to prduce action values (Q values)
        Q_values = layers.Dense(units=1, name='q_values')(net)

        # Create Keras model
        self.model = models.Model(inputs=[states, actions], outputs=Q_values)

        # Define optimizer and compile model for training with built-in loss function
        optimizer = optimizers.Adam(lr=0.0001)
        self.model.compile(optimizer=optimizer, loss='mse')

        # Compute action gradients (derivative of Q values w.r.t. to actions)
        action_gradients = K.gradients(Q_values, actions)

        # Define an additional function to fetch action gradients (to be used by actor model)
        self.get_action_gradients = K.function(
            inputs=[*self.model.input, K.learning_phase()],
            outputs=action_gradients)

"""
The Critic_Torch class.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class Critic_Torch(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, hidden_units, seed, gate=F.relu, dropout=0.2):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            hidden_units (array): Number of nodes for layers
            seed (int): Random seed
            gate (function): activation function
        """
        super(Critic_Torch, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.gate = gate
        self.dropout = nn.Dropout(p=dropout)
        self.normalizer = nn.BatchNorm1d(state_size)
        dims = (state_size, ) + hidden_units
        self.layers = nn.ModuleList()
        count = 0
        for dim_in, dim_out in zip(dims[:-1], dims[1:]):
            if count == 1:
                self.layers.append(nn.Linear(dim_in+action_size, dim_out))
            else:
                self.layers.append(nn.Linear(dim_in, dim_out))
            count += 1
        self.output = nn.Linear(dims[-1], 1)
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.layers:
            layer.weight.data.uniform_(*hidden_init(layer))
        self.output.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, states, actions):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        xs = self.normalizer(states)
        xs = self.gate(self.layers[0](xs))
        x = torch.cat((xs, actions), dim=1)
        for i in range(1, len(self.layers)):
            x = self.gate(self.layers[i](x))
        x = self.dropout(x)
        return self.output(x)

"""
The Ornstein-Uhlenbeck Noise class.
"""

import numpy as np
import copy

class OUNoise:
    """Ornstein-Uhlenbeck process."""

    def __init__(self, size, mu, theta, sigma):
        """Initialize parameters and noise process."""
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        """Reset the internal state (= noise) to mean (mu)."""
        self.state = copy.copy(self.mu)

    def sample(self):
        """Update internal state and return it as a noise sample."""
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state

"""
The DDPGAgent_Keras class.
"""
    
class DDPGAgent_Keras():
    """Reinforcement Learning agent that learns using DDPG."""
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high

        # Actor (Policy) Model
        self.actor_local = Actor_Keras(self.state_size, self.action_size, self.action_low, self.action_high)
        self.actor_target = Actor_Keras(self.state_size, self.action_size, self.action_low, self.action_high)

        # Critic (Value) Model
        self.critic_local = Critic_Keras(self.state_size, self.action_size)
        self.critic_target = Critic_Keras(self.state_size, self.action_size)

        # Initialize target model parameters with local model parameters
        self.critic_target.model.set_weights(self.critic_local.model.get_weights())
        self.actor_target.model.set_weights(self.actor_local.model.get_weights())

        # Noise process
        self.exploration_mu = 0#1.2
        self.exploration_theta = 0.15#1
        self.exploration_sigma = 0.3#0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)

        # Replay memory
        self.buffer_size = 100000#1000000
        self.batch_size = 64
        self.memory = ReplayBuffer_Keras(self.buffer_size, self.batch_size)

        # Algorithm parameters
        self.gamma = 0.99  # discount factor
        self.tau = 0.005#0.01  # for soft update of target parameters

    def reset_episode(self):
        self.noise.reset()
        state = self.task.reset()
        self.last_state = state
        return state

    def step(self, action, reward, next_state, done):
         # Save experience / reward
        self.memory.add(self.last_state, action, reward, next_state, done)

        # Learn, if enough samples are available in memory
        if len(self.memory) > self.batch_size:
            experiences = self.memory.sample()
            self.learn(experiences)

        # Roll over last state and action
        self.last_state = next_state

    def act(self, states):
        """Returns actions for given state(s) as per current policy."""
        #
        '''
        #state = np.reshape(state, [-1, self.state_size])
        #action = self.actor_local.model.predict(state)[0]

        return list(actions + self.noise.sample())  # add some noise for exploration
        #
        '''
        #'''
        states = np.reshape(states, [-1, self.state_size])
        actions = self.actor_local.model.predict(states)
        actions += self.noise.sample()

        return np.clip(actions, -1, 1)
        #'''

    def learn(self, experiences):
        """Update policy and value parameters using given batch of experience tuples."""
        # Convert experience tuples to separate arrays for each element (states, actions, rewards, etc.)
        states = np.vstack([e.state for e in experiences if e is not None])
        actions = np.array([e.action for e in experiences if e is not None]).astype(np.float32).reshape(-1, self.action_size)
        rewards = np.array([e.reward for e in experiences if e is not None]).astype(np.float32).reshape(-1, 1)
        dones = np.array([e.done for e in experiences if e is not None]).astype(np.uint8).reshape(-1, 1)
        next_states = np.vstack([e.next_state for e in experiences if e is not None])

        # Get predicted next-state actions and Q values from target models
        #     Q_targets_next = critic_target(next_state, actor_target(next_state))
        actions_next = self.actor_target.model.predict_on_batch(next_states)
        Q_targets_next = self.critic_target.model.predict_on_batch([next_states, actions_next])

        # Compute Q targets for current states and train critic model (local)
        Q_targets = rewards + self.gamma * Q_targets_next * (1 - dones)
        self.critic_local.model.train_on_batch(x=[states, actions], y=Q_targets)

        # Train actor model (local)
        action_gradients = np.reshape(self.critic_local.get_action_gradients([states, actions, 0]), (-1, self.action_size))
        self.actor_local.train_fn([states, action_gradients, 1])  # custom training function

        # Soft-update target models
        self.soft_update(self.critic_local.model, self.critic_target.model)
        self.soft_update(self.actor_local.model, self.actor_target.model)   

    def soft_update(self, local_model, target_model):
        """Soft update model parameters."""
        local_weights = np.array(local_model.get_weights())
        target_weights = np.array(target_model.get_weights())

        assert len(local_weights) == len(target_weights), "Local and target model parameters must have the same size"

        new_weights = self.tau * local_weights + (1 - self.tau) * target_weights
        target_model.set_weights(new_weights)

"""
The DDPGAgent_Torch class.
"""

import random
import numpy as np
import torch
import torch.nn.functional as F
import copy

class DDPGAgent_Torch:
    def __init__(self, task):
        self.task = task
        self.state_size = task.state_size
        self.action_size = task.action_size
        self.action_low = task.action_low
        self.action_high = task.action_high
        self.actor_learning_rate = 1e-4
        self.hidden_units = (512, 256)
        self.discount = 0.99
        self.tau = 1e-3
        
        # Actor (Policy) Model
        self.actor_local = Actor_Torch(self.state_size, self.action_size, self.hidden_units, 2).to('cpu')
        self.actor_target = Actor_Torch(self.state_size, self.action_size, self.hidden_units, 2).to('cpu')
        self.actor_optimizer = torch.optim.Adam(self.actor_local.parameters(), lr=self.actor_learning_rate)

        # Critic (Value) Model
        self.critic_local = Critic_Torch(self.state_size, self.action_size, self.hidden_units, 2).to('cpu')
        self.critic_target = Critic_Torch(self.state_size, self.action_size, self.hidden_units, 2).to('cpu')
        self.critic_optimizer = torch.optim.Adam(self.critic_local.parameters(), lr=self.actor_learning_rate)
        
        # Initialize networks
        self.soft_update(self.critic_local, self.critic_target, 1)
        self.soft_update(self.actor_local, self.actor_target, 1)

        # Noise process
        self.exploration_mu = 0#1.2
        self.exploration_theta = 0.15#1
        self.exploration_sigma = 0.3#0.2
        self.noise = OUNoise(self.action_size, self.exploration_mu, self.exploration_theta, self.exploration_sigma)
        
        # Replay memory
        self.buffer_size = 100000#1000000
        self.batch_size = 64
        self.memory = ReplayBuffer_Torch(self.action_size, self.buffer_size, self.batch_size, 2, 'cpu')

        self.t_step = 0
        
    def reset(self):
        self.noise.reset()
        
    def act(self, states):            
        """Returns actions for given state as per current policy."""
        states = torch.from_numpy(states).float().to('cpu')
        self.actor_local.eval()
        with torch.no_grad():
            actions = self.actor_local(states).cpu().data.numpy()
        self.actor_local.train()
        actions += self.noise.sample()
        return np.clip(actions, -1, 1)
        
    def step(self, states, actions, rewards, next_states, dones):
        # Save experience in replay memory
        for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
            self.memory.add(state, action, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % 4
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, self.discount)
                
    def learn(self, experiences, gamma):
        """Update policy and value parameters using given batch of experience tuples.
        Q_targets = r + γ * critic_target(next_state, actor_target(next_state))
        where:
            actor_target(state) -> action
            critic_target(state, action) -> Q-value
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
        """

        states, actions, rewards, next_states, dones = experiences
        
        # Compute Q targets for current states and train critic model (local)
        # Get predicted next-state actions and Q values from target models
        actions_next = self.actor_target(next_states)
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)
        self.critic_optimizer.step()

        # Train actor model (local)
        # Compute actor loss
        actions_pred = self.actor_local(states)
        actor_loss = -self.critic_local(states, actions_pred).mean()
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft-update target models
        self.soft_update(self.critic_local, self.critic_target, self.tau)
        self.soft_update(self.actor_local, self.actor_target, self.tau)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)
