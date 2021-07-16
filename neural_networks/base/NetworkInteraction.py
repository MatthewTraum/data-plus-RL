import inspect, copy
from tqdm import tqdm

import Adversaries, Transmitters, Receivers, PolicyMakers

from Util import confirm, get_integer, select_option
from ShowInfo import get_game_info_string

from GameElements import Game, Transmitter, Receiver, Adversary, PolicyMaker
from GameParameters import GameParameterSet
from GameSimulator import simulate_game

from neural_networks.base.NeuralNetworks import GameAgent, \
    get_available_networks, device, save_networks_to_disk, add_trained_model
from neural_networks.base.ParameterHost import get_parameters
from neural_networks.jonathan.SimpleRNNs import SimpleRNN_Adversary

class ZipPlayer():
    """
    Wrapper class to keep up with the players of the game, their "names",
    and whether or not they have an associated neural network.
    """
    def __init__(self, agent: GameAgent, player=None):
        self.player = player
        self.agent = agent

    def __str__(self) -> str:
        return self.agent.name + (" (NEURAL NETWORK)" 
            if self.agent.nnet else "")

#   -----------------------------------------------
#   GETTER METHODS to retrieve all possible players
#   -----------------------------------------------

def get_adversaries() -> 'list[ZipPlayer]':
    adversaries = [ZipPlayer(GameAgent("Adversary", a[0]), a[1]) 
        for a in inspect.getmembers(Adversaries, inspect.isclass) 
            if a[1].__module__ == "Adversaries"]
    for agent in get_available_networks():
        if agent.role == "Adversary":
            adversaries.append(
                ZipPlayer(agent)
            )
    return adversaries

def get_transmitters() -> 'list[ZipPlayer]':
    transmitters = [ZipPlayer(GameAgent("Transmitter", t[0]), t[1]) 
        for t in inspect.getmembers(Transmitters, inspect.isclass) 
            if t[1].__module__ == "Transmitters"]
    for agent in get_available_networks():
        if agent.role == "Transmitter":
            transmitters.append(
                ZipPlayer(agent)
            )
    return transmitters

def get_receivers() -> 'list[ZipPlayer]':
    receivers = [ZipPlayer(GameAgent("Receiver", r[0]), r[1]) 
        for r in inspect.getmembers(Receivers, inspect.isclass) 
            if r[1].__module__ == "Receivers"]
    for agent in get_available_networks():
        if agent.role == "Receiver":
            receivers.append(
                ZipPlayer(agent)
            )
    return receivers

def get_policy_makers() -> 'list[ZipPlayer]':
    policy_makers = [ZipPlayer(GameAgent("PolicyMaker", p[0]), p[1]) 
        for p in inspect.getmembers(PolicyMakers, inspect.isclass) 
            if p[1].__module__ == "PolicyMakers"]
    for agent in get_available_networks():
        if agent.role == "PolicyMaker":
            policy_makers.append(
                ZipPlayer(agent)
            )
    return policy_makers

#   --------------------------------
#   TRAINING & GAMEPLAYING FUNCTIONS
#   --------------------------------

def train_demo():
    
    transmitter_agent = GameAgent("Transmitter", "ExampleTransmitter")
    receiver_agent = GameAgent("Receiver", "ExampleReceiver")
    policy_maker_agent = GameAgent("PolicyMaker", "RandomDeterministicPolicyMaker")
    
    adversary_agent = GameAgent("Adversary", "Jonathan's Example Adversary",
        SimpleRNN_Adversary)
    

    train_models(
        get_parameters("TRAINING_SIMPLE_RNN_ADV"), 
        get_game_params_from_dict(get_parameters("GAME_PARAMS")),
        ZipPlayer(policy_maker_agent, 
            PolicyMakers.RandomDeterministicPolicyMaker), 
        ZipPlayer(transmitter_agent, Transmitters.ExampleTransmitter),
        ZipPlayer(receiver_agent, Receivers.ExampleReceiver), 
        ZipPlayer(adversary_agent)
    )

def train_models(*args):
    play_games(True, False, *args)

def play_print_games():
    play_games(print_each_game=True)

def get_stats(**kwargs):
    completed_games = play_games(**kwargs)
    a_avg_score = sum([game.state.score_a for game in completed_games]) / \
        len(completed_games)
    b_avg_score = sum([game.state.score_b for game in completed_games]) / \
        len(completed_games)
    print(f"Avg. score A: {a_avg_score}, Avg. score B: {b_avg_score}")
    print("A scores: ")
    print([game.state.score_a for game in completed_games])
    print("B scores:")
    print([game.state.score_b for game in completed_games])


def play_games(train_model: bool=False, print_each_game: bool=False, 
        nnet_params: dict=None, game_params: GameParameterSet=None, 
        policy_maker: ZipPlayer=None, transmitter: ZipPlayer=None, 
        receiver: ZipPlayer=None, adversary: ZipPlayer=None, count: int=-1,
        show_output: bool=True) -> 'list[Game]':

    """
    This is the main function used to simulate multiple games
    between the various agents in the game.
    """

    if show_output:
        print("Playing games...")

    if game_params == None:
        p = get_parameters("GAME_PARAMS", default=
            confirm("Use default game parameters?"))
        game_params = get_game_params_from_dict(p)
        print()

    if nnet_params == None and train_model:
        nnet_params = get_parameters("TRAINING", default=confirm(
            "Use default training parameters for neural network?"
        ))

    if show_output:
        print()

    if policy_maker == None:
        print("Select a policy maker.\n")
        policy_maker = select_option(get_policy_makers())
        print()
    
    if transmitter == None:
        print("Select a transmitter.\n")
        transmitter = select_option(get_transmitters())
        print()

    if receiver == None:
        print("Select a receiver.\n")
        receiver = select_option(get_receivers())
        print()

    if adversary == None:
        print("Select an adversary.\n")
        adversary = select_option(get_adversaries())
        print()
   
    if train_model:
        count = nnet_params["COUNT"]
    elif count < 0:
        count = get_integer("How many games would you like to play?")

    if show_output:
        print("Please wait while the neural networks are initialized...\n")

    for zip_player in [policy_maker, transmitter, receiver, adversary]:
        # If player is using a NNET, we need to initialize it w/ parameters
        if zip_player.agent.nnet != None:

            # Create a deep copy to prevent overwrites
            if zip_player.agent.role == "Transmitter":
                transmitter = copy.deepcopy(transmitter)
                zip_player = transmitter
            elif zip_player.agent.role == "Receiver":
                receiver = copy.deepcopy(receiver)
                zip_player = receiver
            elif zip_player.agent.role == "Adversary":
                adversary = copy.deepcopy(adversary)
                zip_player = adversary
            elif zip_player.agent.role == "PolicyMaker":
                policy_maker = copy.deepcopy(policy_maker)
                zip_player = policy_maker

            if zip_player.agent.nnet_instance == None:
                zip_player.agent.nnet_instance = zip_player.agent.nnet(
                    game_params, device)
                zip_player.agent.working_parameters = game_params
            else:
                if not game_params.are_equal_to(zip_player.agent.working_parameters):
                    raise ValueError(f"Game parameters {str(game_params)} " + 
                        "and model parameters " + 
                       f"{str(zip_player.agent.working_parameters)}" + 
                        " are not compatible!")
            
            zip_player.agent.nnet_instance = zip_player.agent.nnet_instance.to(device)

            if zip_player.agent.role == "Transmitter":
                # Initialize the transmitter's functions
                transmitter.player = lambda num_policies: Transmitter(
                    transmitter.agent.nnet_instance.select_policy,
                    transmitter.agent.nnet_instance.get_start_policy()
                )
            elif zip_player.agent.role == "Receiver":
                # Initialize the receiver's functions
                receiver.player = lambda: Receiver(
                    receiver.agent.nnet_instance.get_prediction,
                    receiver.agent.nnet_instance.communicate_bandwidth
                )
            elif zip_player.agent.role == "Adversary":
                # Initialize the adversary's prediction function
                adversary.player = lambda: Adversary(
                    adversary.agent.nnet_instance.get_prediction)
            elif zip_player.agent.role == "PolicyMaker":
                policy_maker.player = lambda params: PolicyMaker(
                    params, policy_maker.agent.nnet_instance.get_policy_list
                )
        
    completed_games = []

    iter = range(count)
    if show_output:
        iter = tqdm(iter)
    for _ in iter:
        game = simulate_game(game_params, policy_maker.player(game_params), 
            transmitter.player(game_params.N), receiver.player(), 
            adversary.player())
        completed_games.append(game)
        if print_each_game:
            print(get_game_info_string(game.state))

    if show_output:
        print(f"Completed {count} games.")

    if train_model:
        print("Training the network...")
        for zip_player in [transmitter, receiver, policy_maker, adversary]:
            if zip_player.agent.nnet != None:
                zip_player.agent.nnet_instance.train(completed_games,
                    nnet_params)
                add_trained_model(zip_player.agent)

        save_networks_to_disk()
        print("\nDone training!\n")

    return completed_games

def get_game_params_from_dict(p: dict) -> GameParameterSet:
    return GameParameterSet(
            p["M"],
            p["N"],
            p["T"],
            p["R1"],
            p["R2"],
            p["R3"]
        )
