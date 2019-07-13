[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"

# Project 2: Continuous Control

### Introduction

This is my submission for Project 2 - Continuous Control in the Udacity Deep Reinforcement Learning Nanodegree.

![Trained Agent][image1]

### Getting Started

My submission includes the following files:

- Continuous_Control_DDPG_Torch.ipynb
- This file, README.md
- Report.md
- DDPG.ddpg_classes.py, which contains all of my Python classes
- The final checkpoint_actor.pth and checkpoint_critic.pth files

You can simply run the last code cell in the notebook to execute my code; it contains all imports and other code necessary for the assignment.

1. I followed the instructions specified as far as downloading the DRLND repo.

2. I followed the instructions in the notebook to explore the environment provided.

### The Assignment

1. I actually went through 4 notebooks trying to figure out this assignment. I had a very tough time trying to figure out where to begin (the first 3 notebooks aren't included in my submission). The first notebook is full of messy, uncommented code; I never achieved a score above a zero. I tried to adapt the Shangtong Zhang code reviewed in the materials, but couldn't figure out how to interface that code, which is openai-gym-specific, to ml-agents. I also tried a DQN from the first assignment, just to naively get something working. Both approaches failed completely.

2. I looked into adapting the PPO pong code from and earlier lesson, but it didn't seem fruitful. After numerous rehashed, frustrating attempts with the previous two techniques, I finally loooked up a previous student's implementation, which proved that the approach adapting Shangtong Zhang's code could actually work. Using that as a jumping off point, I slogged again through adapting that code, which had previously put me off due to errors it was throwing and the highly deep, abstracted nature of the code. It turned out that the errors were trivial, amounting mainly to logging issues. I was able to get PPO working this way, but is was very disappointing, also producing scores of 0.00 after 25 episodes in a second notebook, and I abandoned it.

3. I found that a number of people were using DDPG to address the assignment, so I switched to that algorithm, as I had spent several days trying to get PPO to work without success. I first copied and adapted code that I had modified from the quadcopter assignment in the DLND program that had been provided for that Nanodegree, and which I had modified for that assignment. I got it to work in a third notebook titled Continuous_Control_DDPG_Keras with satisfactory results; however the assignment requires that we use PyTorch, not Keras.

4. I adapted the previous code for PyTorch. I used this previous student's code as a starting point: https://github.com/Catastropha/continuous-control/blob/master/agent.py. It appears to have been based on the same code used in the DLND that I started with, and was very similar to my existing code for Keras from the quadcopter assignment. As a result, I did not re-write every line from scratch, I mostly modified it to fit my own architecture. I don't know if this satisfies the plagiarism rules in the honor code or not; I hope so, but it's a bit fuzzy.
