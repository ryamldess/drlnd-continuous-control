[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Project 2: Continuous Control - Report

### DDPG ('Deep Deterministic Policy Gradient') Algorithm and Hyperparameters

I used DDPG for the assignment, although I had originally wanted to use PPO, which apparently tends to converge faster with better solutions. Unfortunately, I couldn't get PPO to function properly. DDPG is a policy gradient method. Policy gradient methods are a subset of poliy-based methods. While the latter search directly for the optimal policy, policy gradient methods estimate the best weights by gradient descent. Essentially, they estimate the gradient via a neural network rather than making direct guesses. It is similar to actor-critic methods, except the actor maps states to actions directly rather than stochastically. The target networks are time-delayed to avoid interdependence on the outputs of the original networks.

I used the following hyperparameters as constants in ddpg_classes.py for the PyTorch DDPG agent class:

exploration_mu = 0
exploration_theta = 0.15
exploration_sigma = 0.3
gamma = 0.99 (discount factor)
self.tau = 0.005 (for soft update of target parameters)

For experience replay:

buffer_size = 100000
batch_size = 64
seed = 2

The actor network consists of the following structure:

- 512 linear layers with size of 256 + 33 (for state space size).
- A linear output layer with 512 + 33 inputs and outputs corresponding to the number of actions.

It uses torch.nn.BatchNorm1d for normalization.

The critic network consists of the following structure:

- 1 linear layer with size of 256 + 33 (for state space size) + number of actions .
- 511 linear layers with size of 256 + 33 (for state space size).
- A linear output layer with 512 + 33 inputs and outputs corresponding to the number of actions.

It uses torch.nn.BatchNorm1d for normalization and dropout with a value of 0.2.

### Thoughts

Similar to the last assignment, I felt the lesson material was great on theory and math but didn't do enough to prepare me for an implementation of any of these algorithms. Unlike in assignments for past Nanodegrees, I felt entirely lost for several days.

### Room for improvement

The main thing I would improve is simply to get PPO working, as it should converge faster with a smoother curve.
