
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
    batch_size = 1000
    episode_rewards = []
    episode_switches = []
    for episode in range(num_episodes):
        game_rewards = []
        i = 0

        while i < min_games_per_episode or agent.buffer.current_length < 8*batch_size:
        #for _ in range(games_per_episode):
            game_reward = BetterGameSimulator(params, policy_maker, agent, receiver, adversary).simulate_game()
            game_rewards.append(game_reward)
            #switches.append(switch)
            i += 1

        for k in range(updates_per_episode):
            loss = agent.update(batch_size)
            #print(loss)
            #1.0107 is 300th root of 5. Grows 5 times
        batch_size=int(batch_size*1.00537920931)+1



        if episode%5 ==4:
            agent.buffer = Buffer(10*batch_size) #should be same as dql_agents
            agent.buffer.current_length = 0

        #batch_size = batch_size + games_per_episode * (params.T)//20
        #if batch_size* params.T* games_per_episode<= batch_size * :


        #Getting some errors with divide by 0 should do something to make sure each episode has games
        episode_reward = sum(game_rewards)/len(game_rewards)
        episode_rewards.append(episode_reward)
        print("Episode " + str(episode) + ": " + str(episode_reward))
    return episode_rewards


if __name__ == "__main__":
    params_dict = get_parameters("GAME_PARAMS")
    params = get_game_params_from_dict(params_dict)

    params.T = 100
    params.N = 5
    params.M = 10

    NUM_EPISODES = 100
    UPDATES_PER_EPISODE = 10
    MIN_GAMES_PER_EPISODE = 8




    policy_maker = RandomDeterministicPolicyMaker(params)
    stateSize = 1*params.N +1

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






