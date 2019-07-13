[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Project 2: Continuous Control

### Introduction

This is my submission for Project 2 - Continuous Control in the Udacity Deep Reinforcement Learning Nanodegree.

![Trained Agent][image1]

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Distributred Training

For this project, Udacity provided us with two separate versions of the Unity environment:

- The first version contains a single agent.
- The second version contains 20 identical agents, each with its own copy of the environment.  

The second version is useful for algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb) that use multiple (non-interacting, parallel) copies of the same agent to distribute the task of gathering experience.  

### Solving the Environment

The project submission needed only solve one of the two versions of the environment. I chose to solve Option 2, solving 20 identical agents:

The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents.  In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents).  Specifically,
- After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent.  This yields 20 (potentially different) scores.  We then take the average of these 20 scores. 
- This yields an **average score** for each episode (where the average is over all 20 agents).

The environment is considered solved, when the average (over 100 episodes) of those average scores is at least +30.

### Getting Started

1. Download the environment from one of the links below.  You need only select the environment that matches your operating system:

    - **_Version 2: Twenty (20) Agents_**
        - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
        - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
        - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
        - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)
    
    (_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

    (_For AWS_) If you'd like to train the agent on AWS (and have not [enabled a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md)), then please use [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux_NoVis.zip) (version 1) or [this link](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux_NoVis.zip) (version 2) to obtain the "headless" version of the environment.  You will **not** be able to watch the agent without enabling a virtual screen, but you will be able to train the agent.  (_To watch the agent, you should follow the instructions to [enable a virtual screen](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Training-on-Amazon-Web-Service.md), and then download the environment for the **Linux** operating system above._)

2. Place the file in the cloned repository, in a folder named `Reacher_Windows_x86_64-20_Agents/` folder, and unzip (or decompress) the file.

3. As an early attempt utilized Keras instead of PyTorch, you may need to install Keras in your environment via the command 'pip install keras'.

### Instructions

Follow the instructions in `Continuous_Control_DDPG_Torch.ipynb` to get started with training your own agent! To train with my submission, you can simply run the last code cell in the notebook to execute my code; it contains all imports and other code necessary for the assignment.

### My Project Submission - Overview

My submission includes the following files:

- Continuous_Control_DDPG_Torch.ipynb
- This file, README.md
- Report.md
- DDPG.ddpg_classes.py, which contains all of my Python classes
- The final checkpoint_actor.pth and checkpoint_critic.pth files

1. I followed the instructions specified as far as downloading the DRLND repo.

2. I followed the instructions in the notebook to explore the environment provided.

### The Assignment

1. I actually went through 4 notebooks trying to figure out this assignment. I had a very tough time trying to figure out where to begin (the first 3 notebooks aren't included in my submission). The first notebook is full of messy, uncommented code; I never achieved a score above a zero. I tried to adapt the Shangtong Zhang code reviewed in the materials, but couldn't figure out how to interface that code, which is openai-gym-specific, to ml-agents. I also tried a DQN from the first assignment, just to naively get something working. Both approaches failed completely.

2. I looked into adapting the PPO pong code from and earlier lesson, but it didn't seem fruitful. After numerous rehashed, frustrating attempts with the previous two techniques, I finally loooked up a previous student's implementation, which proved that the approach adapting Shangtong Zhang's code could actually work. Using that as a jumping off point, I slogged again through adapting that code, which had previously put me off due to errors it was throwing and the highly deep, abstracted nature of the code. It turned out that the errors were trivial, amounting mainly to logging issues. I was able to get PPO working this way, but is was very disappointing, also producing scores of 0.00 after 25 episodes in a second notebook, and I abandoned it.

3. I found that a number of people were using DDPG to address the assignment, so I switched to that algorithm, as I had spent several days trying to get PPO to work without success. I first copied and adapted code that I had modified from the quadcopter assignment in the DLND program that had been provided for that Nanodegree, and which I had modified for that assignment. I got it to work in a third notebook titled Continuous_Control_DDPG_Keras with satisfactory results; however the assignment requires that we use PyTorch, not Keras.

4. I adapted the previous code for PyTorch. I used this previous student's code as a starting point: https://github.com/Catastropha/continuous-control/blob/master/agent.py. It appears to have been based on the same code used in the DLND that I started with, and was very similar to my existing code for Keras from the quadcopter assignment. As a result, I did not re-write every line from scratch, I mostly modified it to fit my own architecture. I don't know if this satisfies the plagiarism rules in the honor code or not; I hope so, but it's a bit fuzzy.
