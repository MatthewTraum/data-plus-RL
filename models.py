import torch
import torch.nn as nn
import torch.autograd as autograd

from GameParameters import GameParameterSet


#class ConvDQN(nn.Module):

    #def __init__(self, input_dim, output_dim):
        #self.input_dim = input_dim
        #self.output_dim = output_dim
        #self.fc_input_dim = self.feature_size()

        #self.conv = nn.Sequential(
            #nn.Conv2d(self.input_dim[0], 32, kernel_size=8, stride=4),
            #nn.ReLU(),
            #nn.Conv2d(32, 64, kernel_size=4, stride=2),
            #nn.ReLU(),
            #nn.Conv2d(64, 64, kernel_size=3, stride=1),
            #nn.ReLU()
        #)

        #self.fc = nn.Sequential(
            #nn.Linear(self.fc_input_dim, 128),
            #nn.Linear(128, 256),
            #nn.ReLU(),
            #nn.Linear(256, self.output_dim)
        #)

    #def forward(self, state):
        #features = self.conv_net(state)
        #features = features.view(features.size(0), -1)
        #qvals = self.fc(features)
        #return qvals

    #def feature_size(self):
        #return self.conv_net(autograd.Variable(torch.zeros(1, *self.input_dim))).view(1, -1).size(1)


class DQN(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        #ShouldBeAParameterAndNotCalculatedInteranlly
        input_size = GameParameterSet.M+GameParameterSet.N*4+10


        self.fc = nn.Sequential(
            nn.Linear(self.input_dim[0], input_size),
            nn.ReLU(),
            nn.Linear(input_size, 2*input_size),
            nn.ReLU(),
            nn.Linear(2*input_size, self.output_dim)
        )


    def  forward(self, state):
        "Input's parameters for state, outputs a vector of n qvals"
        qvals = self.fc(state)
        return qvals