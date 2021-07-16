from GameSimulator import *

params = GameParameterSet(
    M = 3,
    N = 2,
    T = 100,
    R1 = 10,
    R2 = 5,
    R3 = 20
)

policy_maker = ExamplePolicyMaker(params)

def main():
    transmitter = None
    receiver = None
    adversary = None
    role_selected = False

    print(f"\nParamters: {params}\n")

    for index, policy in enumerate(policy_maker.get_policy_list()):
        print(f"Policy {index}: {policy}")

    print("")

    while not role_selected: 
        role = input(f"Who would you like to play as?\nt = Transmitter, r \
= receiver, a = adversary > ")
        role_selected = True

        if role == 't':
            print("\nWelcome, transmitter.\n")
            transmitter = HumanTransmitter(num_policies=params.N)
        elif role == 'r':
            print("\nWelcome, receiver.\n")
            receiver = HumanReceiver()
        elif role == 'a':
            print("\nWelcome, adversary.\n")
            adversary = HumanAdversary()
        else:
            print(f"\nERROR: Invalid role entered. Try again or press CTRL-C to exit.\n")
            role_selected = False

    simulate_game(params, policy_maker, 
        transmitter if transmitter else ExampleTransmitter(params.N),
        receiver if receiver else ExampleReceiver(),
        adversary if adversary else ExampleAdversary())

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nKeyboardInterrupt")