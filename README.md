## Collaboration competition project

This repository provides way to train Deep Reinforcement 
Learning agents in order to play tennis 
in virtual environment.

In order to run this code, you'll need Unity Environment that 
Udacity team provided in Deep Reinforcement Learning Nanodegree. 

### Environment

In this environment, two agents control rackets to bounce a ball over a net. 
If an agent hits the ball over the net, it receives a reward 
of +0.1. If an agent lets a ball hit the ground or hits the ball 
out of bounds, it receives a reward of -0.01. Thus, the goal 
of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding 
to the position and velocity of the ball and racket. Each 
agent receives its own, local observation. Two continuous 
actions are available, corresponding to movement toward 
(or away from) the net, and jumping. 

The task is episodic, and in order to solve the environment, 
your agents must get an average score of +0.5 (over 100 
consecutive episodes, after taking the maximum over both agents).

### Training 

For training, see file `train_agent.py`

### Running

I committed some trained files in models/ folder that you can use 
to run this code without training. Just run:
```
python run_agent.py
```
