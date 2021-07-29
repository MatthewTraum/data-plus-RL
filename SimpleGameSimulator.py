from GameElements import *

class SimpleGameSimulator:
    """
    SimpleGameSimulator is designed to test whether we can converge to a strategy of never
    switching when the only reward is a penalty for not switching. The only parameter for
    the NN is a one-hot encoding of the previous policy.
    This NN also trains
    """
    def __init__(self, params: GameParameterSet, policy_maker: PolicyMaker,
                 transmitter: Transmitter, receiver: Receiver, adversary: Adversary):
        self.params = params
        self.transmitter = transmitter
        self.receiver = receiver
        self.adversary = adversary
        self.policy_list = policy_maker.get_policy_list()

        "Size of state is N"
        self.state = [0]*params.N
        self.lastAct = -1


    def simulate_game(self):
        # Initialize the return lists
        gameReward = 0

        # Run the game

        state = self.state
        gameReward = 0

        for t in range(self.params.T):
            trueAct = self.transmitter.get_policy(self.state, self.params.N)
            #trueAct = self.lastAct
            self.state = [0]*self.params.N
            self.state[trueAct] = 1
            if trueAct != self.lastAct:
                gameReward -= self.params.R2
                self.lastAct = trueAct
            for action in range(self.params.N):
                nextState = [0]*self.params.N
                nextState[action] = 1
                if action != self.lastAct:
                    reward = -self.params.R2
                else:
                    reward = 0
                if t == self.params.T-1:
                    self.transmitter.buffer.push(state, action, reward, nextState, 1)
                else:
                    self.transmitter.buffer.push(state, action, reward, nextState, 0)

        return gameReward


    def simulate_game2(self) -> int:
        #simulates all possible states & actions
        gameReward = 0
        for pol in range(self.params.N):
            state = [0]*self.params.N
            state[pol] = 1
            realAction = self.transmitter.get_policy(state, self.params.N)
            if realAction != pol:
                gameReward -= self.params.R2
            for action in range(self.params.N):
                reward = 0
                if pol != action:
                    reward = -self.params.R2
                nextState = [0] * self.params.N
                nextState[action] = 1
                self.transmitter.buffer.push(state, action, reward, nextState)
        return gameReward


