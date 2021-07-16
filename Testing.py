print("\nLoading...\n")

import pickle
from copy import deepcopy

from neural_networks.base.NetworkInteraction import get_adversaries, \
    get_policy_makers, get_receivers, get_transmitters, play_games, \
    get_game_params_from_dict
from neural_networks.base.ParameterHost import get_parameters

policy_maker_list = get_policy_makers()
transmitter_list = get_transmitters()
receiver_list = get_receivers()
adversary_list = get_adversaries()

# Choose the players
policy_maker = policy_maker_list[1]
transmitter = transmitter_list[2]
receiver = receiver_list[0]
adversary = adversary_list[1]

# Number of repeats for each simulation
repeats = 250

# Print info about what we are doing
print(f"Running {repeats} sims. ea. for P = {str(policy_maker)}, \n" + 
      f"T = {str(transmitter)}, R = {str(receiver)}, A = {str(adversary)}\n")

default_params = get_parameters("GAME_PARAMS")

""" 
Score storage format
[
num_bands 10 [ num_policies 10 ... 100]
 .
 .
 .
num_bands 100 [ num_policies 10 ... 100]
]
"""
b_scores = []

count = 1

for num_bands in range(10, 100):
    b_scores.append([])

    for num_policies in range(10, 100, 10):

        print(f"\nRun {count} of 810...")

        new_params = deepcopy(default_params)
        new_params["M"] = num_bands
        new_params["N"] = num_policies

        completed_games = play_games(
            train_model=False,
            print_each_game=False,
            nnet_params=None,
            game_params=get_game_params_from_dict(new_params),
            policy_maker=policy_maker,
            transmitter=transmitter,
            receiver=receiver,
            adversary=adversary,
            count=repeats,
            show_output=True
        )

        b_avg_score = sum([game.state.score_b for game in completed_games]) / \
            len(completed_games)

        b_scores[-1].append(b_avg_score)

        count += 1

pickle.dump(b_scores, open("all-data.pkl", 'wb'))