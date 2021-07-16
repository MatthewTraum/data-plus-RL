
from dql_agent import DQNAgent
from event_buffer import Buffer
from BetterGameSimulator import better_simulate_game
from neural_networks.base.NetworkInteraction import get_game_params_from_dict
from neural_networks.base.ParameterHost import get_parameters
from Transmitters import *
from Receivers import *
from Adversaries import *
from PolicyMakers import *




def train(agent, max_episodes, batch_size, parameter_set, policy_maker, receiver, adversary):
    episode_rewards = []
    for episode in range(max_episodes):
        "Should reset Adversary ect:"
        #Do I need to implement
        #state = env.reset()
        episode_reward = 0


        #for _ in range(T):
            #action = agent.get_action(state)
            #next_state, reward, _ = env.step(action)
            #agent.replay_buffer.push(state, action, reward, next_state)
            #episode_reward += reward

        #TODO find better way of getting 46
        agent2 = DQNAgent(46, params.N)
        while agent2.replay_buffer.current_length <= batch_size:
            states, actions, game_rewards = better_simulate_game(params, policy_maker, agent, receiver, adversary)
            agent2 = DQNAgent(len(states[0]), params.N)

            #Figure out Terminal/First State.
            for i in range(1, len(states)):
                agent2.replay_buffer.push(states[i-1], actions[i], game_rewards[i], states[i])
                print(agent2.replay_buffer.current_length)

        agent2.update(batch_size)

        if episode%4 ==3:
            agent2.replay_buffer.buffer = Buffer(10000) #HardCoded in as 10000
            agent2.replay_buffer.current_length = 0



        episode_reward = sum(game_rewards)/len(game_rewards)
        episode_rewards.append(episode_reward)
        print("Episode " + str(episode) + ": " + str(episode_reward))
    return episode_rewards


if __name__ == "__main__":
    MAX_EPISODE = 100
    BATCH_SIZE = 45
    BUFFER_SIZE= 10000 #Also initialized seperately by agent


    params_dict = get_parameters("GAME_PARAMS")
    params = get_game_params_from_dict(params_dict)

    params.T = 10
    params.N = 12
    params.M = 10

    policy_maker = RandomDeterministicPolicyMaker(params)
    transmitter = RandomTransmitter(params.N)
    receiver = ExampleReceiver()
    adversary = GammaAdversary()

    ret = train(transmitter,MAX_EPISODE, BATCH_SIZE, params, policy_maker, receiver, adversary)

    print(sum(ret[:49]))
    print(sum(ret[49:]))

