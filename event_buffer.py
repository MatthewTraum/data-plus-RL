import torch
import torch.nn as nn
import torch.autograd as autograd

import random
import numpy as np
from collections import deque
from sum_tree import SumTree

class Buffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.current_length = 0
        self.buffer = deque(maxlen=max_size)

    def push(self, state, action, reward, next_state, notDone):
        experience = (state, action, reward, next_state, notDone)
        self.current_length +=1
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        not_done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, notDone= experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            not_done_batch.append(notDone)

        return state_batch, action_batch, reward_batch, next_state_batch, not_done_batch


    def __len__(self):
        return len(self.buffer)


