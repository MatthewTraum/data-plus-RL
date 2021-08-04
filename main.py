from dql_agent import DQNAgent
from event_buffer import Buffer
from BetterGameSimulator import BetterGameSimulator
from neural_networks.base.NetworkInteraction import get_game_params_from_dict
from neural_networks.base.ParameterHost import get_parameters
from Receivers import *
from Adversaries import *
from RNN_Adversary import *
from PolicyMakers import *
from SimpleGameSimulator import *

from copy import deepcopy

import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from sklearn.linear_model import LinearRegression


def train(agent, num_episodes, min_games_per_episode, updates_per_episode, policy_maker, receiver, adversary, internalAdversary):
    batch_size = 128
    episode_rewards = []
    episode_switches = []
    for episode in range(num_episodes):
        game_rewards = []
        i = 0

        while i < min_games_per_episode or agent.buffer.current_length < 8*batch_size:
            gameState = GameState(params, policy_maker.get_policy_list())
            game_reward, switches = BetterGameSimulator(gameState, agent, receiver, adversary, internalAdversary, True).simulate_game()
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


class GameState():
    """
    The publicly available information about the game.
    (Everything except the players.)

    EDIT: Modified to allow access to the policy history.
    """

    def __init__(self, params: GameParameterSet, policy_list: 'list[Policy]'):
        self.params = params
        self.t = 0
        self.policy_list = policy_list
        self.rounds = []

    def __deepcopy__(self, memo):
        new = GameState(self.params, self.policy_list)

        new.t = self.t
        new.rounds = self.rounds[:]

        return new

if __name__ == "__main__":
    params_dict = get_parameters("GAME_PARAMS")
    params = get_game_params_from_dict(params_dict)


    params.T = 50
    params.N = 20
    params.M = 5


    NUM_EPISODES = 150
    UPDATES_PER_EPISODE = 50
    MIN_GAMES_PER_EPISODE = 3




    policy_maker = RandomDeterministicPolicyMaker(params)
    stateSize = 2*params.N+2

    transmitter = DQNAgent(stateSize, params.N)
    receiver = ExampleReceiver()
    internalAdversary = GammaAdversary()

    RL_RNN = {
        "NUM_LAYERS": 2,
        "LEARNING_RATE": 0.001,
        "LOOKBACK": 5,  # CHECK IF NEEDS TO BE 5
        "HIDDEN_DIM": 16,
        "REPETITIONS": 5,  # CHECK IF NEEDS TO BE 5
    }
    adversary = PriyaRLAdversary(params.N, RL_RNN)


    rewards = train(transmitter, NUM_EPISODES, MIN_GAMES_PER_EPISODE, UPDATES_PER_EPISODE, policy_maker, receiver, adversary, internalAdversary)
    print("hi")
    NUM_SIMULATED_GAMES = 10
    rewards2 = []
    switches = []
    for i in range(NUM_SIMULATED_GAMES):
        gameState = GameState(params, policy_maker.get_policy_list())
        reward, switch = BetterGameSimulator(gameState, transmitter, receiver, adversary, internalAdversary, False).simulate_game()
        rewards2.append(reward)
        switches.append(switch)


    X = np.array([i for i in range(NUM_EPISODES)]).reshape((-1, 1))
    print(sum(rewards[-10:])/10)
    print(sum(rewards2[-10:])/ 10)
    print(sum(switches)/ 10)

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






