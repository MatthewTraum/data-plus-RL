
from dql_agent import DQNAgent
from event_buffer import Buffer
from BetterGameSimulator import BetterGameSimulator
from neural_networks.base.NetworkInteraction import get_game_params_from_dict
from neural_networks.base.ParameterHost import get_parameters
from Receivers import *
from Adversaries import *
from PolicyMakers import *
from SimpleGameSimulator import SimpleGameSimulator

from copy import deepcopy

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from sklearn.linear_model import LinearRegression


def train(agent, num_episodes, min_games_per_episode, updates_per_episode, policy_maker, receiver, adversary):
    episode_rewards=[]
    batch_size = params.N**2
    for episode in range(num_episodes):
        game_reward = SimpleGameSimulator2(params, policy_maker, agent, receiver, adversary).simulate_game()
        episode_rewards.append(game_reward)


        for k in range(updates_per_episode):
            loss = agent.update(batch_size)
        agent.buffer.current_length = 0

        print("Episode " + str(episode) + ": " + str(game_reward))
    return episode_rewards


if __name__ == "__main__":
    params_dict = get_parameters("GAME_PARAMS")
    params = get_game_params_from_dict(params_dict)

    params.T = 10
    params.N = 5
    params.M = 10

    NUM_EPISODES = 150
    UPDATES_PER_EPISODE = 100
    MIN_GAMES_PER_EPISODE = 30




    policy_maker = RandomDeterministicPolicyMaker(params)
    stateSize = 1*params.N

    transmitter = DQNAgent(stateSize, params.N)
    receiver = ExampleReceiver()
    adversary = GammaAdversary()

    rewards = train(transmitter, NUM_EPISODES, MIN_GAMES_PER_EPISODE, UPDATES_PER_EPISODE, policy_maker, receiver, adversary)


    X = np.array([i for i in range(NUM_EPISODES)]).reshape((-1, 1))
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






