# Building a neural network into the game

## TL;DR

 - Create a subclass of `nn.Module` along the specifications of [`TemplateRNNs.py`](https://github.com/jpiland16/neural-net-basic-game/blob/master/neural_networks/jonathan/TemplateRNNs.py)

 - Edit [`NeuralNetworks.py`](https://github.com/jpiland16/neural-net-basic-game/blob/master/neural_networks/base/NeuralNetworks.py) by importing your class at the top of the script, then adding a `GameAgent` containing the neural network to the `untrained_networks` list

## Some specific notes about `SimpleRNN_Adversary`

  > **IMPORTANT: This is probably not a great example of an effective solution to the problem.** (The network was performing no better than chance at guessing the bandwidth, even though the policy was never changed.)

### Structure

  - This neural network consists of a single recurrent layer and then a single fully connected (linear) layer.
    - The state of the recurrent layer is influenced by two variables: the state of the recurrent layer in the previous timestep (AKA hidden state) and the input at a given time.
    - This means that the activations in the recurrent layer are a linear combination of the previous hidden state activations and the input activations.
    - The hidden state activations are 0 for the first timestep.
    - The hidden layer has arbitrary size of 20.
  
### Input and output

   - The input to the network is of the following shape:
     > (number of batches, number of timesteps, (N + 3) * M)

     - The (N + 3) * M comes from the fact that each bandwidth is encoded as a one-hot vector, and I am passing the bandwidths predicted by N policies plus the guesses of the transmitter, receiver, and adversary.

   - The output of the model is of the following shape:
     > (number of batches, number of timesteps, M)
      - This is because only a single one-hot vector is needed to encode the bandwidth prediction.
   - I am predicting the bandwidth using the index of the maximum value of the one-hot vector.

### Training
   -  The model is being trained using randomly selected sequences from within a fixed number of games.
      > **IMPORTANT: the tensors used for training are (counterintuitively) not the same shape as the output.** Instead, they are of the shape (number of batches, number of timesteps) where the value at each time step is an **integer**, rather than a one-hot vector.

## General information about testing and training

 - To train your neural network, it will probably be best to select option (1) from the main menu (i.e., the prompt given when running `Main.py`).
 - To see the results of your neural network, try options (3) or (4), which print entire games and overall statistics, respectively.

