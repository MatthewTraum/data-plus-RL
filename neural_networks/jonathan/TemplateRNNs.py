import torch
from torch import nn

from GameElements import GameState, Game, Policy
from GameParameters import GameParameterSet

################################################################################
######################            TRANSMITTER            #######################
################################################################################

class TemplateRNN_Transmitter(nn.Module):
    """
    An empty framework for the receiver that could potentially include 
    a neural network.
    """
    def __init__(self, params: GameParameterSet, device: torch.device) -> None:
        """
        The initialization function receives two parameters:
         - params: a GameParameterSet
         - device: device on which to store tensors (CPU/GPU)

        You can save these parameters using `self` and/or use them
        to create a neural network.
        """
        
        # Must have this line below in order to use neural network capabilities
        # (Otherwise there will be an error in NetworkInteraction.py
        # when we try to move to the specified device.)
        super(TemplateRNN_Transmitter, self).__init__()

    def get_start_policy(self) -> int:
        """
        Should return an integer between 0 and (M - 1). This policy will be 
        automatically communicated to the receiver. 
        """
        return 0

    def select_policy(self, game_state: GameState) -> 'tuple[int, bool]':
        """
        Function that is called at the start of each turn when the transmitter 
        must choose whether to switch the policy and/or communicate the policy.
        Accepts a GameState and returns the following:
         - int: the selected policy
         - bool: whether to communicate or not
        """
        return 0, False

    def train(self, completed_games: 'list[Game]', params: dict) -> None:
        """
        This function is called whenever the neural network should be trained.
        It accepts a list of completed games and a set of parameters
        (see ParameterHost.py for examples of parameter sets).
        """
        pass

################################################################################
######################              RECEIVER              ######################
################################################################################

class TemplateRNN_Receiver(nn.Module):
    """
    An empty framework for the receiver that could potentially include 
    a neural network.
    """
    def __init__(self, params: GameParameterSet, device: torch.device) -> None:
        """
        The initialization function receives two parameters:
         - params: a GameParameterSet
         - device: device on which to store tensors (CPU/GPU)

        You can save these parameters using `self` and/or use them
        to create a neural network.
        """
        
        # Must have this line below in order to use neural network capabilities
        # (Otherwise there will be an error in NetworkInteraction.py
        # when we try to move to the specified device.)
        super(TemplateRNN_Receiver, self).__init__()

    def communicate_bandwidth(self, communicated_band: int) -> None:
        """
        This function is called whenever the transmitter chooses to communicate
        a bandwidth to the receiver.
        """
        pass

    def get_prediction(self, game_state: GameState) -> int:
        """
        This function accepts a GameState and should return an integer
        from 0 to (M - 1).
        """
        return 0


    def train(self, completed_games: 'list[Game]', params: dict) -> None:
        """
        This function is called whenever the neural network should be trained.
        It accepts a list of completed games and a set of parameters
        (see ParameterHost.py for examples of parameter sets).
        """
        pass

################################################################################
#####################              ADVERSARY              ######################
################################################################################

class TemplateRNN_Adversary(nn.Module):
    """
    An empty framework for the adversary that could potentially include 
    a neural network.
    """
    def __init__(self, params: GameParameterSet, device: torch.device) -> None:
        """
        The initialization function receives two parameters:
         - params: a GameParameterSet
         - device: device on which to store tensors (CPU/GPU)

        You can save these parameters using `self` and/or use them
        to create a neural network.
        """
        
        # Must have this line below in order to use neural network capabilities
        # (Otherwise there will be an error in NetworkInteraction.py
        # when we try to move to the specified device.)
        super(TemplateRNN_Adversary, self).__init__()

    def get_prediction(self, game_state: GameState) -> int:
        """
        This function accepts a GameState and should return an integer
        from 0 to (M - 1).
        """
        return 0


    def train(self, completed_games: 'list[Game]', params: dict) -> None:
        """
        This function is called whenever the neural network should be trained.
        It accepts a list of completed games and a set of parameters
        (see ParameterHost.py for examples of parameter sets).
        """
        pass

################################################################################
######################            POLICY MAKER            ######################
################################################################################

class TemplateRNN_PolicyMaker(nn.Module):
    """
    An empty framework for the receiver that could potentially include 
    a neural network.
    """
    def __init__(self, params: GameParameterSet, device: torch.device):
        """
        The initialization function receives two parameters:
         - params: a GameParameterSet
         - device: device on which to store tensors (CPU/GPU)

        You can save these parameters using `self` and/or use them
        to create a neural network.
        """
        
        # Must have this line below in order to use neural network capabilities
        # (Otherwise there will be an error in NetworkInteraction.py
        # when we try to move to the specified device.)
        super(TemplateRNN_PolicyMaker, self).__init__()

        self.num_policies = params.N

    def get_policy_list(self) -> 'list[Policy]':
        """
        This function is called whenever the game needs a list of policies.
        """
        return [Policy(lambda x: 0, "Always return zero") 
            for _ in range(self.num_policies)]


    def train(self, completed_games: 'list[Game]', params: dict) -> None:
        """
        This function is called whenever the neural network should be trained.
        It accepts a list of completed games and a set of parameters
        (see ParameterHost.py for examples of parameter sets).
        """
        pass