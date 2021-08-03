
from dql_agent import DQNAgent
from event_buffer import Buffer
from BetterGameSimulator import BetterGameSimulator
from neural_networks.base.NetworkInteraction import get_game_params_from_dict
from neural_networks.base.ParameterHost import get_parameters
from Receivers import *
from Adversaries import *
from PolicyMakers import *
from SimpleGameSimulator import *

from copy import deepcopy

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from sklearn.linear_model import LinearRegression


def train(agent, num_episodes, min_games_per_episode, updates_per_episode, policy_maker, receiver, adversary, internalAdversary):
    batch_size = 64
    episode_rewards = []
    episode_switches = []
    for episode in range(num_episodes):
        game_rewards = []
        i = 0

        while i < min_games_per_episode or agent.buffer.current_length < 8*batch_size:
            game_reward = BetterGameSimulator(params, policy_maker, agent, receiver, adversary, internalAdversary).simulate_game()
            game_rewards.append(game_reward)
            i += 1

        for k in range(updates_per_episode):
            loss = agent.update(batch_size)


        if episode%5 ==4:
            agent.buffer = Buffer(10*batch_size) #should be same as dql_agents
            agent.buffer.current_length = 0

        episode_reward = sum(game_rewards)/len(game_rewards)
        episode_rewards.append(episode_reward)
        print("Episode " + str(episode) + ": " + str(episode_reward))
    return episode_rewards


if __name__ == "__main__":
    params_dict = get_parameters("GAME_PARAMS")
    params = get_game_params_from_dict(params_dict)


    params.T = 50
    params.N = 5
    params.M = 10

    NUM_EPISODES = 300
    UPDATES_PER_EPISODE = 30
    MIN_GAMES_PER_EPISODE = 3




    policy_maker = RandomDeterministicPolicyMaker(params)
    stateSize = 2*params.N

    transmitter = DQNAgent(stateSize, params.N)
    receiver = ExampleReceiver()
    adversary = GammaAdversary()
    internalAdversary = GammaAdversary()

    rewards = train(transmitter, NUM_EPISODES, MIN_GAMES_PER_EPISODE, UPDATES_PER_EPISODE, policy_maker, receiver, adversary, internalAdversary)

    X = np.array([i for i in range(NUM_EPISODES)]).reshape((-1, 1))
    print(sum(rewards[-10:])/10)
    Y = np.array(rewards)
    #Z = np.array(switches)
    model = LinearRegression()  # create object for the class
    model.fit(X, Y)  # perform linear regression
    Y_pred = model.predict(X)  # make predictions

    r_sq = model.score(X, Y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)


    plt.scatter(X, Y)
    plt.plot(X, Y_pred, color='red')
    plt.show()






