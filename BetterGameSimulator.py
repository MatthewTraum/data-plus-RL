print("Importing pacakges...")

from neural_networks.base.NetworkInteraction import get_game_params_from_dict
from neural_networks.base.ParameterHost import get_parameters
from Transmitters import *
from Receivers import *
from Adversaries import *
from PolicyMakers import *
from GameElements import *
from GameSimulator import simulate_game

from copy import deepcopy

def better_simulate_game(params: GameParameterSet, policy_maker: PolicyMaker,
        transmitter: Transmitter, receiver: Receiver, adversary: Adversary):
    # Run the game
    game = simulate_game(params, policy_maker, transmitter, receiver, adversary)

    # Initialize tracking
    relevant_info = {
        "current_policy" : 0,
        "next_bandwidth_for_current_policy": 0,
        "current_time": 0,
        "time_since_last_switch": 1,
        "percent_of_time_on_each_policy": [0 for _ in range(params.N)],
        "percent_time_adversary_guessed_each_band": [0 for _ in range(params.M)],
        "adversary_accuracy_for_each_policy": [0 for _ in range(params.N)],
        "adversary_accuracy_since_last_switch": 0,
        "adversary_success_in_last_4_turns": [False,False,False,False],
        "adversary_accuracy_in_last_20_turns": 0,
        "adversary_accuracy_all_game": 0,
        "average_duration_between_switches": 0
    }

    policy_use_count = [0 for _ in range(params.N)]
    adversary_policy_correct_count = [0 for _ in range(params.N)]
    bandwidth_use_count = [0 for _ in range(params.M)]
    adversary_band_guess_count = [0 for _ in range(params.M)]

    # Initialize the return lists
    states = []  # A list of lists containing the relevant info
    actions = [] # A list containing tuples of (policy: int, communication: bool)
    rewards = [] # Reward after the current action (a_i leads to reward r_i)

    # Prepare for the first turn
    current_policy = transmitter.start_policy
    adversary_correct_since_last_switch = 0
    policy_record = [current_policy]
    adversary_correct_hx20 = []
    adversary_correct_total = 0
    switch_count = 1


    # Update states, actions, rewards based on what happened in the game
    for t in range(params.T):
        adversary_band_guess_count[
            game.state.rounds[t].adversary_guess] += 1

        if game.state.rounds[t].adversary_guess == \
                game.state.rounds[t].transmission_band:
            # The adversary was correct
            adversary_policy_correct_count[policy_record[-1]] += 1
            adversary_correct_since_last_switch += 1
            relevant_info["adversary_success_in_last_4_turns"].append(
                True)
            adversary_correct_hx20.append(1)
            adversary_correct_total += 1

            rewards.append(0)

        else:
            # The adversary was incorrect
            relevant_info["adversary_success_in_last_4_turns"].append(
                False)
            adversary_correct_hx20.append(0)

            if game.state.rounds[t].transmission_band == \
                    game.state.rounds[t].receiver_guess:
                # Team A earns points
                rewards.append(params.R1)
            else:
                rewards.append(0)
    
        relevant_info["adversary_success_in_last_4_turns"] = \
            relevant_info["adversary_success_in_last_4_turns"][-4:]
        adversary_correct_hx20 = adversary_correct_hx20[-20:]
        
        policy_record.append(current_policy)  
        policy_use_count[current_policy] += 1
        bandwidth_use_count[game.state.policy_list[
            current_policy].get_bandwidth(t)] += 1


        relevant_info["current_policy"] = current_policy
        relevant_info["next_bandwidth_for_current_policy"] = \
            game.state.policy_list[current_policy].get_bandwidth(t + 1)
        relevant_info["current_time"] = t
        relevant_info["percent_of_time_on_each_policy"] = [
            policy_use_count[i] / (t + 1) for i in range(params.N)]
        relevant_info["percent_time_adversary_guessed_each_band"] = [
            adversary_band_guess_count[i] / (t  + 1) for i in range(params.M)]
        # NOTE: consider (4 lines below) what value to insert for 
        # unused policies (currently math.nan)
        relevant_info["adversary_accuracy_for_each_policy"] = [
            adversary_policy_correct_count[i] / policy_use_count[i]
            if policy_use_count[i] != 0 else math.nan for i in range(params.N)]
        relevant_info["adversary_accuracy_since_last_switch"] = \
            adversary_correct_since_last_switch / relevant_info["time_since_last_switch"]
        relevant_info["adversary_accuracy_in_last_20_turns"] = sum(
            adversary_correct_hx20) / min(t + 1, 20)
        relevant_info["adversary_accuracy_all_game"] = \
            adversary_correct_total / (t + 1)
        relevant_info["average_duration_between_switches"] = \
            (t + 1) / switch_count
        
        policy = game.policy_record[t]
        communication = game.communication_record[t] if t < params.T - 1 \
            else False

        if policy == current_policy:
            relevant_info["time_since_last_switch"] += 1
        else:
            if t < params.T - 1:
                rewards[-1] -= params.R2 * math.log2(params.N)
            relevant_info["time_since_last_switch"] = 1
            adversary_correct_since_last_switch = 0
            current_policy = policy        
            switch_count += 1

        if communication and t > 0:
            rewards[-1] -= params.R3

        state = []
        for item in relevant_info:
            if type(relevant_info[item]) is list:
                for elm in relevant_info[item]:
                    state.append(elm)
            else:
                state.append(relevant_info[item])

        states.append(state)
        actions.append(policy)


    return states, actions, rewards

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