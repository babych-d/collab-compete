##Report

This implementation uses Multi-agent DDPG method.

Agent uses following hyperparameters: 
```
BUFFER_SIZE = int(1e5)
BATCH_SIZE = 128
GAMMA = 0.99
NOISE_END = 0.1
LR_ACTOR = 1e-4
LR_CRITIC = 3e-4
NOISE_START = 1.0
NOISE_REDUCTION = 0.9999
TAU = 1e-3
```

Plot with rewards can be found in model/scores.png
