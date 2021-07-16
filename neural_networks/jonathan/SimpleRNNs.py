from GameParameters import GameParameterSet
import random
import torch
import numpy as np
from torch import nn
from tqdm import tqdm

# Personal code
from GameElements import Game, GameState
from Util import one_hot_encode

class SimpleRNN_Adversary(nn.Module):
    """
    A simple recurrent neural network with 1 hidden layer.
    Based on the tutorial available at 
    https://blog.floydhub.com/a-beginners-guide-on-recurrent-neural-networks-with-pytorch/
    """
    def __init__(self, params: GameParameterSet, device: torch.device):
        
        super(SimpleRNN_Adversary, self).__init__()

        # Defining some parameters ------------------------
        self.hidden_layer_size = 20
        self.n_layers = 1
        self.to(device)
        self.my_device = device
        self.M = params.M

        # Defining the layers ------------------------------
        # RNN Layer
        self.rnn = nn.RNN((params.N + 3) * params.M, self.hidden_layer_size, 
            self.n_layers, batch_first=True)   
        # Fully connected layer
        self.fc = nn.Linear(self.hidden_layer_size, params.M)
    
    def forward(self, input: torch.Tensor):
        """
        Return the output(s) from each timestep in the input sequence.
        """        
        input_size = input.size(0)

        # Initializing hidden state for first input using method defined below
        hidden_state = self.init_hidden(input_size)

        # Passing in the input and hidden state into the model and 
        # obtaining outputs
        output, final_hidden_state = self.rnn(input, hidden_state)
        
        # Reshaping the outputs such that it can be fit into the 
        # fully connected layer
        output = output.contiguous().view(-1, self.hidden_layer_size)
        output = self.fc(output)
        return output
    
    def init_hidden(self, batch_size):
        """
        Initialize the neural network's hidden layer.
        """
        # This method generates the first hidden state of zeros
        # which we'll use in the forward pass, also sends the tensor 
        # holding the hidden state to the device specified 
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_layer_size)
        hidden = hidden.to(self.my_device)
        return hidden

    def get_prediction(self, game_state: GameState):
        timesteps = []
        for time in range(game_state.t):
            network_input = self.get_input_vector_at_time(
                game_state, time)
            timesteps.append(network_input)

        # The current time, when the guess is being made
        last_timestep = self.get_half_empty_vector_at_time(game_state, 
            game_state.t)
        
        timesteps.append(last_timestep)

        # Run the RNN, then decode the last output
        output = self(torch.Tensor([timesteps]).to(self.my_device))[-1]
        adversary_guess = torch.argmax(output).item()
        return adversary_guess


    def train(self, completed_games: 'list[Game]', params: dict) -> None:

        # Define Loss, Optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), 
            lr=params["LEARNING_RATE"])

        iter = tqdm(range(params["NUM_EPOCHS"]))
        for _ in iter:
            # Get a new batch of training data
            training_input, training_target_output = self.get_train_set(
                completed_games, params["SEQ_LEN"], params["BATCH_SIZE"])

            # Move to GPU, if available
            training_input = training_input.to(self.my_device)
            training_target_output = training_target_output.to(self.my_device)

            optimizer.zero_grad() # Clears existing grads. from previous epoch
            output = self(training_input)
            training_target_output = training_target_output.view(-1).long()
            loss = criterion(output, training_target_output)
            loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly

            iter.set_description(f"loss: {round(loss.item(), 3):5}")


    def get_input_vector_at_time(self, game_state: GameState, t: int):
        """
        Takes the game state at time `t` and converts it to an input vector.
        """
        round = game_state.rounds[t]
        input = np.array([
            one_hot_encode(round.transmission_band, vector_size = self.M),
            one_hot_encode(round.receiver_guess, vector_size = self.M),
            one_hot_encode(round.adversary_guess, vector_size = self.M)] 
            + [ one_hot_encode(policy.get_bandwidth(t), self.M) 
                for policy in game_state.policy_list ]
        )
        input = input.reshape(-1) # Convert to a single column
        return input

    def get_half_empty_vector_at_time(self, game_state: GameState, t: int):
        """
        Returns a vector containing only the policy predictions for time `t`.
        (Omits the guesses of transmitter, receiver, and adversary, even
        if they are available.)
        """
        vector = np.array([ [0 for _ in range(self.M)] for _ in range(3) ] + \
            [ one_hot_encode(policy.get_bandwidth(t), self.M)
                for policy in game_state.policy_list ])

        return vector.reshape(-1) # Convert to a single column

    def get_train_set(self, completed_games: 'list[Game]', seq_len: int,
            batch_size: int):

        train_x = []
        train_y = []

        for _ in range(batch_size):

            # Choose a random game
            game = completed_games[random.randint(0, len(completed_games) - 1)]

            # Choose a random sequence within the game (subtract 1 for 0-based 
            # array indices)
            if game.state.t - seq_len - 1 < 0:
                raise ValueError("Length of the game must be at least 1 " + 
                    "greater than the length of the sequence")
            start_pos = random.randint(0, game.state.t - seq_len - 1)

            training_example_x = []
            training_example_y = []

            for t in range(start_pos, start_pos + seq_len - 1):
                training_example_x.append(
                    self.get_input_vector_at_time(game.state, t))
                training_example_y.append(
                    game.state.rounds[t + 1].transmission_band)

            training_example_x.append(
                self.get_half_empty_vector_at_time(game.state, 
                    start_pos + seq_len - 1)
            )

            training_example_y.append(
                game.state.rounds[start_pos + seq_len].transmission_band)

            train_x.append(training_example_x)
            train_y.append(training_example_y)

        return torch.Tensor(train_x).to(self.my_device), \
            torch.Tensor(train_y).to(self.my_device)

