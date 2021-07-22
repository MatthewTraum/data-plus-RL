print("Importing pacakges...")

from neural_networks.base.NetworkInteraction import get_game_params_from_dict
from neural_networks.base.ParameterHost import get_parameters
from Transmitters import *
from Receivers import *
from Adversaries import *
from PolicyMakers import *
from GameElements import *
from collections import deque
from GameSimulator import simulate_game
from copy import deepcopy

class BetterGameSimulator:

    def __init__(self, params: GameParameterSet, policy_maker: PolicyMaker,
                 transmitter: Transmitter, receiver: Receiver, adversary: Adversary):
        self.params = params
        self.transmitter = transmitter
        self.receiver = receiver
        self.adversary = adversary
        self.policy_list = policy_maker.get_policy_list()

        "Size of state is 9+3N+M"
        self.state = {
            "1_hot_last_policy": [0 for _ in range(params.N)],
            #"bad_switch" : [0 for _in range(params.N)],
            "t": 0,
            "time_since_last_switch": 0,
            "adversary_correct_since_last_switch": 0,
            "percent_of_time_on_each_policy": [0 for _ in range(params.N)],
            "percent_time_adversary_guessed_each_band": [0 for _ in range(params.M)],
            "adversary_accuracy_for_each_policy": [0 for _ in range(params.N)],
            "adversary_accuracy_since_last_switch": 0,
            "adversary_success_in_last_4_turns": [0, 0, 0, 0],
            "adversary_accuracy_in_last_20_turns": 0,
        }
        self.last_policy = -1
        self.adversary_correct_since_last_switch = 0
        self.policy_use_count = [0 for _ in range(params.N)]
        self.adversary_policy_correct_count = [0 for _ in range(params.N)]
        self.bandwidth_use_count = [0 for _ in range(params.M)]
        self.adversary_band_guess_count = [0 for _ in range(params.M)]
        self.adversary_correct_hx20 = []
        self.rounds = []
        self.nnInput = []



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
        self.nnInput=state

        done = False
        while not done:
            item = self.advance_time()
            action, reward, next_state, done = item[:]
            self.transmitter.buffer.push(state, action, reward, next_state)
            #Need to teach the game ends
            if done:
                #really want to teach game over 0 reward.
                for _ in range(int(self.params.T/self.params.N/4)):
                    for i in range(self.params.N):
                        self.transmitter.buffer.push(next_state, i, 0, next_state)
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
        t = self.state["t"]
        # Select policy --------------------------------
        action = self.transmitter.get_policy(self.nnInput, self.params.N)
        # Transmit based on the selected policy --------------------------------
        policy = self.policy_list[action]

        transmission_band = policy.get_bandwidth(t)
        receiver_guess = transmission_band
        adversary_guess = self.adversary.predict_policy(self.policy_list, self.rounds,t)
        #adversary_guess = random.randrange(0,self.params.M)

        self.rounds.append(Round(transmission_band, receiver_guess, adversary_guess))

        if (adversary_guess != transmission_band):
            reward += self.params.R1
        if action != self.last_policy:
            reward -= self.params.R2

        # Advance the time ----------------------------------------------------
        self.updateState(action)

        nextState = []
        for item in self.state:
            if type(self.state[item]) is list:
                for elm in self.state[item]:
                    nextState.append(elm)
            else:
                nextState.append(self.state[item])


        if self.state["t"] >= self.params.T:
            return action, reward, nextState, True
        else:
            return action, reward, nextState, False



    def updateState(self,action):
        # Update states, actions, rewards based on what happened in the game
        t = self.state["t"]
        round = self.rounds[t]

        self.adversary_band_guess_count[round.adversary_guess] += 1
        if self.last_policy != action:
            self.last_policy = action
            self.state["time_since_last_switch"] = 0
        else:
            self.state["time_since_last_switch"] += 1
        #print(self.state["time_since_last_switch"])

        self.state["1_hot_last_policy"] = [0 for i in range(self.params.N)]
        self.state["1_hot_last_policy"][action]=1

        if round.adversary_guess == round.transmission_band:
            # The adversary was correct
            self.adversary_policy_correct_count[action] += 1
            self.adversary_correct_since_last_switch += 1
            self.state["adversary_success_in_last_4_turns"].append(0)
            self.state["adversary_success_in_last_4_turns"].pop(0)
            self.adversary_correct_hx20.append(1)

        else:
            # The adversary was incorrect
            self.state["adversary_success_in_last_4_turns"].append(1)
            self.state["adversary_success_in_last_4_turns"].pop(0)
            self.adversary_correct_hx20.append(0)

        self.adversary_correct_hx20 = self.adversary_correct_hx20[-20:]

        self.policy_use_count[action] += 1
        self.bandwidth_use_count[round.transmission_band] += 1
        t = t+1
        self.state["t"] = t
        self.state["percent_of_time_on_each_policy"] = [
            self.policy_use_count[i] / (t) for i in range(self.params.N)]
        self.state["percent_time_adversary_guessed_each_band"] = [
            self.adversary_band_guess_count[i] / (t) for i in range(self.params.M)]
        # NOTE: consider (4 lines below) what value to insert for
        # unused policies (currently 0)
        self.state["adversary_accuracy_for_each_policy"] = [
            self.adversary_policy_correct_count[i] / self.policy_use_count[i]
            if self.policy_use_count[i] != 0 else 0 for i in range(self.params.N)]
        if self.state["time_since_last_switch"] != 0:
            self.state["adversary_accuracy_since_last_switch"] = \
            self.adversary_correct_since_last_switch / self.state["time_since_last_switch"]
        else:
            self.state["adversary_accuracy_since_last_switch"] = 0
        self.state["adversary_accuracy_in_last_20_turns"] = sum(
            self.adversary_correct_hx20) / min(t, 20)



    def test_better_sim():
        params_dict = get_parameters("GAME_PARAMS")
        params = get_game_params_from_dict(params_dict)

        params.T = 3
        params.N = 18
        params.M = 10

        policy_maker = RandomDeterministicPolicyMaker(params)
        transmitter = HumanTransmitter(params.N)
        receiver = ExampleReceiver()
        adversary = GammaAdversary()

        s, a, r = better_simulate_game(params, policy_maker, transmitter, receiver,
                                       adversary)

        print("STATES", s, "ACTIONS", a, "REWARDS", r, sep="\n", end="\n")


if __name__ == "__main__":
    test_better_sim()
