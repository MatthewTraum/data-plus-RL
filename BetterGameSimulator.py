
from GameElements import *

class BetterGameSimulator:

    def __init__(self, params, policy_maker, transmitter, receiver, adversary, internalAdversary):
        self.params = params
        self.transmitter = transmitter
        self.receiver = receiver
        self.adversary = adversary
        self.internalAdversary = internalAdversary
        self.policy_list = policy_maker.get_policy_list()

        "Size of state is 2N"
        self.state = {
            "1_hot_last_policy": [0 for _ in range(params.N)],
            "internal_predictions" : [0 for _ in range(params.N)]
            #"t%": 0,
        }
        self.t = 0
        self.last_policy = -1
        self.rounds = []

    def simulate_game(self) -> int:
        # Initialize the return lists
        gameReward = 0

        # Run the game

        state = []
        for item in self.state:
            if type(self.state[item]) is list:
                for elm in self.state[item]:
                    state.append(elm)
            else:
                state.append(self.state[item])
        self.nnInput = state

        done = False
        while not done:
            item = self.advance_time()
            action, reward, next_state, done = item
            if done:
                for i in range(5):
                    self.transmitter.buffer.push(state, action, 0, next_state, 0)
            else:
                self.transmitter.buffer.push(state, action, reward, next_state, 1)


            state = next_state
            self.nnInput = state
            gameReward += reward

        return gameReward


    def advance_time(self)-> (int, int, list, bool):
        """
        Advances the time in the game, and calls on each player to make a
        move.
        Returns action, reward, nextState, and whether the game is done.
        """

        reward = 0
        t = self.t
        # Select policy --------------------------------
        action = self.transmitter.get_policy(self.nnInput, self.params.N)
        # Transmit based on the selected policy --------------------------------
        policy = self.policy_list[action]

        transmission_band = policy.get_bandwidth(t)
        receiver_guess = transmission_band
        adversary_guess = self.adversary.predict_policy(self.policy_list, self.rounds, self.params.M, t)


        self.rounds.append(Round(transmission_band, receiver_guess, adversary_guess))

        if (adversary_guess != transmission_band):
            reward += self.params.R1
        if action == self.last_policy:
            reward += self.params.R2

        # Advance the time ----------------------------------------------------
        self.updateState(action)

        nextState = []
        for item in self.state:
            if type(self.state[item]) is list:
                for elm in self.state[item]:
                    nextState.append(elm)
            else:
                nextState.append(self.state[item])

        if self.t >= self.params.T:
            return action, reward, nextState, True
        else:
            return action, reward, nextState, False



    def updateState(self,action):
        # Update states, actions, rewards based on what happened in the game

        t = self.t
        round = self.rounds[t]


        self.state["1_hot_last_policy"] = [0 for _ in range(self.params.N)]
        self.state["1_hot_last_policy"][action]=1
        bands= self.internalAdversary.bandwidth_prediction_vals(self.policy_list, self.rounds, self.params.M, t)

        for i, policy in enumerate(self.policy_list):
            self.state["internal_predictions"][i] = bands[policy.get_bandwidth(t)]
        #self.state["t%"] = t/self.params.T
        self.last_policy = action

        t = t+1
        self.t = t
