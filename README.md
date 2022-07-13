# Banana-navigation-drl
Deep-Reinforcement-Learning based navigation in Unity "banana collection".

## Overview

This project was completed as part of the Udacity Deep Reinforcement Learning nanodegree (Deep Q-Learning).

The code trains and tests an ML-agent to collect bananas in a Unity environment. The environment can be obtained here:
* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

Pull the git repository, unzip the downloaded environement file and place it in the repository (same folder as 'Nativation.ipynb').

The goal of the agent is collect as many yellow bananas as possible and avoid collecting blue bananas. 
The environment provides a descriptive state with 37 dimensions, including velocity and ray-based perceptions of objects relatively to the forward direction of the agent. In that environment, the agent can take 4 actions:
* Move forward
* Move backward
* Turn left (fixed angle)
* Turn right

## Other dependencies 
(in development environment - may not all be required by this specific project)
* matplotlib
* numpy>=1.11.0
* jupyter
* pytest>=3.2.2
* docopt
* pyyaml
* protobuf>=3.5.2
* grpcio>=1.11.0
* torch>=0.4.0
* pandas
* scipy
* ipykernel

## How to use

There 3 primary code files:
1. Navigation.ipynb: trains and test the agents
2. agent.py: agent definition with learning and action methods
3. model.py: the underlying Deep Q-Learning network used to model the Q-Value function

The navigation notebook allows you to:
* Section 1: set the environment (load modules)
* Section 2: explore the environment (actions, states) and set corresponding properties variables
* Section 3: play the environment and observe an untrained agent taking random actions
* Section 4: define 5 agents
  * DQN: Vanilla Deep Q-network
  * DDQN: double DQN leveraging target Q network to evaluate the Q value) during training (https://arxiv.org/abs/1509.06461)
  * Prioritized: DQN trained with prioritized experience replay (https://arxiv.org/pdf/1511.05952.pdf) during training
  * Dueling: Modified DQN with Q(state, action) = Q0(state) + A(state, action) (https://arxiv.org/abs/1511.06581)
  * DDQN+Prioritized+Dueling
* Section 5: train the agents
The dqn() method is used to train all 5 agents.
By default the maximum number of training episodes is set to 1000 and the 'success threshold' is set to 17: if an agent reaches this score (average of score over last 100 episodes), training is ended early. Note that all agents typically reach a score of at least 15.
* Section 6: test the agents
The average and median scores of the agents are measured over 40 episodes each with at most 500 steps.
All agents perform better than the vanilla DQN, in particular the 'prioitized replay' agent.

For training, execute sections 1, 2, 4 and 5.
For testing (after training), run sections 1, 2, 4 and 6.
