
from dql_agent import DQNAgent
from event_buffer import Buffer

def train(env, agent, max_episodes, T, batch_size):
    episode_rewards = []
    for episode in range(max_episodes):
        "Should reset Adversary ect:"
        #Do I need to impement
        state = env.reset()
        episode_reward = 0


        #for _ in range(T):
            #action = agent.get_action(state)
            #next_state, reward, _ = env.step(action)
            #agent.replay_buffer.push(state, action, reward, next_state)
            #episode_reward += reward

        [states], [actions], [game_rewards] = simulateGame(env, agent)
        #Figure out Terminal State.
        for i in range(0,len(states)):
            agent.replay_buffer.push(states[i], actions[i], game_rewards[i], states[i+1])

        if len(agent.replay_buffer) > batch_size:
            agent.update(batch_size)

        if episode%4 ==3:
            agent.replay_buffer = Buffer(10000) #HardCoded in as 10000, which is bad.

        episode_rewards.append(episode_reward)
        print("Episode " + str(episode) + ": " + str(episode_reward))
    return episode_rewards


if __name__ == __main__:
    MAX_EPISODE = 1024
    BATCH_SIZE = 8
    BUFFER_SIZE= 10000 #Also initialized seperately by agent


    N = 5
    M = 4
    T = 500
    R1 = 5
    R2 = 8
    R3 = 2

    #TODO Initialize Game Enviorment with Params

    train(env, DQNAgent, max_episodes, T, batch_size)
