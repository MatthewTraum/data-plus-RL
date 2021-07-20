
from dql_agent import DQNAgent
from event_buffer import Buffer
from BetterGameSimulator import BetterGameSimulator
from neural_networks.base.NetworkInteraction import get_game_params_from_dict
from neural_networks.base.ParameterHost import get_parameters
from Receivers import *
from Adversaries import *
from PolicyMakers import *

from copy import deepcopy

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from sklearn.linear_model import LinearRegression


def train(agent, max_episodes, batch_size, policy_maker, receiver, adversary):
    episode_rewards = []
    i=1
    for episode in range(max_episodes):
        game_rewards = []
        while agent.buffer.current_length <= batch_size*i:
        #for _ in range(10):
            game_reward = BetterGameSimulator(params, policy_maker, agent, receiver, adversary).simulate_game()
            game_rewards.append(game_reward)
        i+=1

        for k in range(3):
            agent.update(batch_size)



        if episode%5 ==4:
            agent.buffer = Buffer(50000) #should be same as dql_agents
            agent.buffer.current_length = 0
            i=1


        #Getting some errors with divide by 0 should doo
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

    MAX_EPISODE = 150
    BATCH_SIZE = 7 * (params.T+params.N-1)

    policy_maker = RandomDeterministicPolicyMaker(params)
    stateSize = 9 + 3*params.N+params.M

    transmitter = DQNAgent(stateSize, params.N)
    receiver = ExampleReceiver()
    adversary = GammaAdversary()

    ret = train(transmitter, MAX_EPISODE, BATCH_SIZE, policy_maker, receiver, adversary)


    X = np.array([i for i in range(10,MAX_EPISODE)]).reshape((-1, 1))
    Y = np.array(ret[10:])
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






