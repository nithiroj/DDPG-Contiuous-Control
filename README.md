# Project 2: Continuous Control
This is second project of my Udacity Deep Reinforcement Learning Nanodegree program. Most codes are referred to what I've learned from the course.

## Introduction

For this project, we will work with the [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment provided by Unity Machine Learning Agents (ML-Agents)

![Reacher Agent](./images/reacher.gif)

A reward of +0.1 is given for each step if the agent's is in the goal location.  Our goal is to maintain its position at the target location for as many time steps as possible. In particular, the agent must get an average score of +30 over 100 consecutive episodes.

## Getting Started

1. Create (and activate) a new environment with Python 3.6.
    - __Linux__ or __Mac__: 
        ```bash
        conda create --name env python=3.6
        source activate drlnd
        ```
    - __Windows__: 
        ```bash
        conda create --name env python=3.6 
        activate drlnd
        ```

2. Clone this repository and navigate to the `python/` folder.  Then, install several dependencies.
    ```bash
    git clone https://github.com/nithiroj/DDPG-Contiuous-Control.git
    cd DDPG-Contiuous-Control/python
    pip install .
    ```

3. Download the environment that matches your operating system, place it in `DDPG-Contiuous-Control/` folder, and unzip it.

    Version 1: One (1) Agent
    - Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Linux.zip)
    - Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher.app.zip)
    - Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86.zip)
    - Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/one_agent/Reacher_Windows_x86_64.zip)

## Training Instructions
If you wouldlike to train a new agent , run this script.

```python train.py --env path/to/Reacher.app --episodes 1000```

Change the `file_name` parameter to match the location of the Unity environment that you downloaded.

- **Mac**: `"path/to/Reacher.app"`
- **Windows** (x86): `"path/to/Reacher_Windows_x86/Reacher.exe"`
- **Windows** (x86_64): `"path/to/Reacher_Windows_x86_64/Reacher.exe"`
- **Linux** (x86): `"path/to/Reacher_Linux/Reacher.x86"`
- **Linux** (x86_64): `"path/to/Reacher_Linux/Reacher.x86_64"`
- **Linux** (x86, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86"`
- **Linux** (x86_64, headless): `"path/to/Reacher_Linux_NoVis/Reacher.x86_64"`

## More informations
You can find more detais in implementation-algorithms (DDPG), model architectures,and choosen hyperparameters-and the achieved rewards in [Report.md](./Report.md).