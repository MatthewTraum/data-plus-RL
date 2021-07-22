import torch
import torch.nn as nn
import torch.autograd as autograd
import numpy as np

from random import randrange
from event_buffer import Buffer
from models import DQN

class DQNAgent:

    def __init__(self, NumObs, NumActions,learning_rate=3e-4, gamma=0.5, buffer_size=100000):

        self.learning_rate = learning_rate
        self.gamma = gamma
        self.buffer = Buffer(buffer_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = DQN(NumObs, NumActions).to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters())
        self.MSE_loss = nn.MSELoss()

        #TODO fix starter policy
        self.start_policy = 0




    def get_policy(self, state, numPolicies, eps=.2):
        state = torch.FloatTensor(state).to(self.device)
        qvals = self.model.forward(state)
        action = np.argmax(qvals.cpu().detach().numpy())

        if (np.random.randn() < eps):
            return randrange(numPolicies)

        return action

    def compute_loss(self, batch):
        states, actions, rewards, next_states = batch
        states = torch.tensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.tensor(next_states).to(self.device)


        curr_Q = self.model.forward(states).gather(1, actions.unsqueeze(1))
        curr_Q = curr_Q.squeeze(1)
        next_Q = self.model.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        #expected_Q = rewards.squeeze(1) + self.gamma * max_next_Q
        expected_Q = rewards + max_next_Q


        loss = self.MSE_loss(curr_Q, expected_Q)
        #print("loss is "+str(loss.data))
        return loss

    def update(self, batch_size):
        batch = self.buffer.sample(batch_size)
        loss = self.compute_loss(batch)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

