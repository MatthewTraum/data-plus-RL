# neural-net-basic-game

## How to contribute *(Duke/Data+ Project 13 members only)*

 - **To get started making agent that makes decisions according to a mathematical formula,** place the agent's class code in the appropriate file in the root directory, i.e. in one of the following: [Transmitters.py](https://github.com/jpiland16/neural-net-basic-game/blob/master/Transmitters.py) / [Receivers.py](https://github.com/jpiland16/neural-net-basic-game/blob/master/Receivers.py) / [Adversaries.py](https://github.com/jpiland16/neural-net-basic-game/blob/master/Adversaries.py) / [PolicyMakers.py](https://github.com/jpiland16/neural-net-basic-game/blob/master/PolicyMakers.py) (examples are provided to demonstrate the correct syntax)

 - **To get started building a neural network,** check out the neural networks [README](https://github.com/jpiland16/neural-net-basic-game/tree/master/neural_networks/base/) or some of my [examples](https://github.com/jpiland16/neural-net-basic-game/tree/master/neural_networks/jonathan).

## How to get started running the scripts

 - Running `Main.py` is a great place to start.

## Structural overview
 
 - To understand the members of classes in the game, check out [GameElements.py](https://github.com/jpiland16/neural-net-basic-game/blob/master/GameElements.py) and [GameParameters.py](https://github.com/jpiland16/neural-net-basic-game/blob/master/GameParameters.py).

## Command line flags

```   
   -c --cpu: run on CPU even if GPU is available
   -g --gpu: run on GPU if available, without asking confirmation
   -o --option <option>: select option <option> at the first prompt
   -h --help: show help
```

## Options available at the main menu (`Main.py`)
 0. **Play against the computer**
 1. **Train neural networks** (gives you the option to select the players of the game. Only neural-network type players will be trained.)
 2. **Train neural networks, with demo options pre-selected** (a shortcut option I used for testing. Uses the `RandomDeterministicPolicyMaker`, `SimpleRNN_Adversary`, and example players everywhere else.)
 3. **Pit agents against each other and view the output** (allows you to select agents and view the full results of a user-defined number of games)
 4. **View the stats after many games** (similar to the above, but shows summary stastics rather than the full games)
 5. **Exit**