import os
import numpy as np 
from collections import deque
import matplotlib.pyplot as plt 
import logging
import argparse
import time

from unityagents import UnityEnvironment
import torch
from ddpg_agent import Agent

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--env', default='./Reacher_Linux_NoVis/Reacher.x86_64', help='Path to the Reacher Unity environment')
parser.add_argument('--episodes', default=500, help='Number of episodes', type=int)
parser.add_argument('--output', default='output', help='Location to output')

def plot_rewards(scores, output):
    plt.plot(np.arange(1, len(scores)+1), scores)
    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.savefig(f'{output}/rewards.png')

def train(env_pth, n_episodes=500, output='output'):

    BATCH_SIZE = 64      # minibatch size
    SEED = 2
    os.makedirs(output, exist_ok=True)

    # load environment
    env = UnityEnvironment(file_name=env_pth)

    # get the default brain name
    brain_name = env.brain_names[0]
    brain = env.brains[brain_name]

    # reset the evironment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    logger.info(f'Number of agents: {num_agents}')

    # size of each action
    action_size = brain.vector_action_space_size
    logger.info(f'Size of each action: {action_size}')

    # examine the state space
    states = env_info.vector_observations
    state_size = states.shape[1]
    logger.info(f'There are {states.shape[0]} agent(s). Each observes a state with length: {state_size}')
    logger.info(f'The state for the first agent looks like: {states[0]}')

    # create agent
    agent = Agent(state_size, action_size, SEED, BATCH_SIZE)

    def ddpg(n_episodes, average_window=100, output='output'):
        scores_deque = deque(maxlen=average_window)
        scores_all = []

        for i_episode in range(1, n_episodes+1):
            env_info = env.reset(train_mode=True)[brain_name]        # reset the environment
            states = env_info.vector_observations                    # get the current state (for each agent)
            scores = np.zeros(num_agents)                            # initialize the score (for each agent)

            while True:
                actions = agent.act(states)                          # select an action (for each agent)
                env_info = env.step(actions)[brain_name]             # send all actions to tne environment
                next_states = env_info.vector_observations           # get next state (for each agent)
                rewards = env_info.rewards                           # get reward (for each agent)
                dones = env_info.local_done                          # see if episode finished

                for state, action, reward, next_state, done in zip(states, actions, rewards, next_states, dones):
                    agent.step(state, action, reward, next_state, done)

                scores += rewards                                    # update the score (for each agent)
                states = next_states                                 # roll over states to next time step
                if np.any(dones):                                    # exit loop if episode finished
                    break

            average_score_episode = np.mean(scores)
            scores_deque.append(average_score_episode)
            scores_all.append(average_score_episode)
            average_score_queue = np.mean(scores_deque)

            logger.info(f'\rEpisode {i_episode}\tScores: {average_score_episode:.2f}\tAverage Score: {average_score_queue:.2f}')
            torch.save(agent.actor_local.state_dict(), f'{output}/checkpoint_actor.pth')
            torch.save(agent.critic_local.state_dict(), f'{output}/checkpoint_critic.pth')
            if i_episode > average_window and average_score_queue > 30:
                break
        
        return scores_all

    scores = ddpg(n_episodes=n_episodes, output=output)
    plot_rewards(scores, output)

    env.close()

if __name__ =='__main__':
    args = parser.parse_args()
    train(args.env, args.episodes, args.output)
    logger.info(f'Done - elapsed time {time.process_time() / 60} mins')

