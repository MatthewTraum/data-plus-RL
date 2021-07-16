from Transmitters import *
from Receivers import *
from Adversaries import *
from PolicyMakers import *
from GameElements import *
from datetime import datetime
import os

def simulate_game(params: GameParameterSet, policy_maker: PolicyMaker,
    transmitter: Transmitter, receiver: Receiver, adversary: Adversary) -> Game:

        policy_list = policy_maker.get_policy_list()
        game = Game(transmitter, receiver, adversary, policy_list, params)

        # Communicate the initial policy (very important)
        receiver.communicate(transmitter.start_policy)

        while(game.advance_time()):
            pass

        return game

def repeat_game(params: GameParameterSet, policy_maker: PolicyMaker,
    transmitter: Transmitter, receiver: Receiver, adversary: Adversary, 
    count: int, save_output: bool = False) -> None:

    output = ""

    for i in range(0, count):
        result = simulate_game(params, policy_maker, transmitter, receiver, 
            adversary)
        score_a = result.state.score_a
        score_b = result.state.score_b
        winner = 'A' if score_a > score_b else \
            'B' if score_b > score_a else '-'
        print("Trial {:3d}  |  Score A: {:4d}  |  Score B: {:4d}"
            .format(i + 1, score_a, score_b, winner ))

        if save_output:
            output += str(result)

    if save_output:
        timestamp = "{:%Y.%m.%d %H-%M-%S}".format(datetime.now())
        if not os.path.exists('output'):
            os.makedirs('output')
        with open(f'output/Games {timestamp}.txt', 'w') as file:
            file.write(output)


def demo():
    params = GameParameterSet(
        M = 3,
        N = 2,
        T = 100,
        R1 = 10,
        R2 = 5,
        R3 = 20
    )

    policy_maker = ExamplePolicyMaker(params)
    
    transmitter = ExampleTransmitter()
    receiver = ExampleReceiver()
    adversary = ExampleAdversary()

    repeat_game(params, policy_maker, transmitter, receiver, adversary, 
        count=10, save_output=True)


def main():
    demo()

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n\nKeyboardInterrupt")