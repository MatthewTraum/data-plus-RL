# Python libraries
import getopt, sys
from sys import exit

USAGE = "   -c --cpu: run on CPU even if GPU is available\n"\
    +   "   -g --gpu: run on GPU if available, without asking confirmation\n" \
    +   "   -o --option <option>: select option <option> at the first prompt\n"\
    +   "   -h --help: show help\n"

default_device = None
initial_option = None

# Check for command-line arguments
try:
    opts, args = getopt.getopt(sys.argv[1:], "cgo:h",
        ["cpu", "gpu", "option=", "help"])

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            print(USAGE)
            exit()
        elif opt in ("-c", "--cpu"):
            default_device = "CPU"
        elif opt in ("-g", "--gpu"):
            default_device = "GPU"
        elif opt in ("-o", "--option"):
            initial_option = int(arg)

except getopt.GetoptError as e:
    print(e)
    print(USAGE)
    exit()

print("\nImporting necessary packages...")

# Personal code
import HumanVsComputers
from Util import select_option
from neural_networks.base.NeuralNetworks import do_gpu_check

do_gpu_check(default_device)

from neural_networks.base.NetworkInteraction import get_stats, train_demo, train_models, \
    play_print_games

class Thing():
    """
    Wrapper for a function that provides human-readable 
    command-line summaries.
    """
    def __init__(self, function: 'function', description: str) -> None:
        self.function = function
        self.description = description
    
    def __str__(self):
        return self.description

things_to_do = [
    Thing(HumanVsComputers.main, "Play against the computer"),
    Thing(train_models, "Train neural networks"),
    Thing(train_demo, "Train neural networks, with demo options pre-selected"),
    Thing(play_print_games, 
        "Pit agents against each other and view the output"),
    Thing(get_stats, "View the stats after many games"),
    Thing(exit, "Exit")
]

def do_something(msg: str):
    """
    Ask the user what they would like to do.
    """
    print(f"\n{msg}\n")
    option = select_option(things_to_do)
            
    print()
    option.function()

def main():

    if initial_option != None:
        things_to_do[initial_option].function()
    else:
        while True:
            do_something("What would you like to do? (CTRL-C to exit)")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print()