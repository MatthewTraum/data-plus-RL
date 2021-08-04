ALLOW_ENTER_MEANS_DEFAULT = True

class Parameter():
    """
    Wrapper for a parameter for a neural network.
    """
    def __init__(self, shortname: str, longname: str,
            type: str, value) -> None:
        self.shortname = shortname
        self.longname = longname
        self.type = type
        self.value = value
    
default_params = {
    "TRAINING": [
        Parameter("COUNT", "number of games to play", "int", 100),
        Parameter("LEARNING_RATE", "learning rate", "float", 0.02),
        Parameter('BATCH_SIZE', "mini-batch size for training", "int", 100),
        Parameter('NUM_EPOCHS', "number of training epochs", "int", 500),
        Parameter('SEQ_LEN', "sequence length for training", "int", 10)
    ],
    "TRAINING_SIMPLE_RNN_ADV": [
        Parameter("COUNT", "number of games to play", "int", 100),
        Parameter("LEARNING_RATE", "learning rate", "float", 0.03),
        Parameter('BATCH_SIZE', "mini-batch size for training", "int", 20),
        Parameter('NUM_EPOCHS', "number of training epochs", "int", 100),
        Parameter('SEQ_LEN', "sequence length for training", "int", 25)
    ],
    "GAME_PARAMS": [
        Parameter("M", "number of available bands", "int", 100),
        Parameter("N", "number of available policies", "int", 40),
        Parameter("T", "length of the game", "int", 100),
        Parameter("R1", "Reward #1", "float", 10),
        Parameter("R2", "Reward #2", "float", 6),
        Parameter("R3", "Reward #3", "float", 0)
    ]
}

def get_parameters(group_id: str, default=True) -> dict:
    """
    Get the parameters governing the neural network, or allow the 
    user to set them if `default` == `False`.
    """
    param_set = {}
    if not default:
        print("Please enter parameters below " +
            "(default values given in parentheses)\n")

        if ALLOW_ENTER_MEANS_DEFAULT:
            print("Press ENTER to accept default for any parameter.\n")

    for parameter in default_params[group_id]:
        if default:
            # Use the default value of the parameter as defined above
            param_set[parameter.shortname] = parameter.value
        else:
            # Ask the user to enter the parameters
            input_accepted = False
            while not input_accepted:
                try:
                    # Ask for input
                    val = input(f"{parameter.longname} " + 
                        f"({parameter.shortname} = {parameter.value}) > ")
                    # Ensure their entry is of the correct type
                    if parameter.type == 'int':
                        val = int(val)
                    elif parameter.type == 'float':
                        val = float(val)
                    param_set[parameter.shortname] = val
                    input_accepted = True
                except KeyboardInterrupt:
                    raise
                except:
                    if ALLOW_ENTER_MEANS_DEFAULT and val == "":
                            param_set[parameter.shortname] = parameter.value
                            input_accepted = True
                    else:
                        print("Invalid entry!")
    return param_set