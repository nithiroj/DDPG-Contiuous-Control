We decided to solve the project by making some amendments to the **Deep Deterministic Policy Gradients (DDPG)** algorithm. You can find the detail of the algorithm in [DDPG paper](https://arxiv.org/abs/1509.02971). The solution is for the **first version (single agent)**.

## Begining Parameters

```
BATCH_SIZE = 128        # minibatch size
BUFFER_SIZE = int(1e5)  # replay buffer size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-3        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
THETA = 0.15            # for noise process
SIGMA = 0.2             # for noise process
```

![Begining](./images/rewards_0.png)

## Experimental Parameters
```
BUFFER_SIZE = int(1e6)  # replay buffer size
LR_CRITIC = 1e-4        # learning rate of the critic
```
![Experiment1](./images/rewards_1.png)

```
BUFFER_SIZE = int(1e6)  # replay buffer size
LR_CRITIC = 1e-4        # learning rate of the critic
SIGMA = 0.1             # for noise process
```
![Experiment2](./images/rewards_2.png)

## Solution Parameters
The saved models and plot of rewards are in `/output`
```
BATCH_SIZE = 64         # minibatch size
BUFFER_SIZE = int(1e6)  # replay buffer size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR_ACTOR = 1e-4         # learning rate of the actor 
LR_CRITIC = 1e-4        # learning rate of the critic
WEIGHT_DECAY = 0        # L2 weight decay
THETA = 0.15            # for noise process
SIGMA = 0.1             # for noise process
```
![Solution](./images/rewards_ok.png)

## Next Steps
- So far, we update the networks every step. Try lower the update learning frequency of the networks, for example two steps at a time.
- Test on option two (20 agents) to see if the parallel learnings help better the results.
- Try different algorithms like [PPO](https://arxiv.org/pdf/1707.06347.pdf), [A3C](https://arxiv.org/pdf/1602.01783.pdf), and [D4PG](https://openreview.net/pdf?id=SyZipzbCb)