import math
import random

from GameParameters import GameParameterSet
from GameElements import Policy, PolicyMaker

class ExamplePolicyMaker(PolicyMaker):
    def __init__(self, params: GameParameterSet) -> None:
        super(ExamplePolicyMaker, self).__init__(params, self.get_policy_list)

    def get_policy_list(self):
        return [
            Policy(lambda t: t % self.params.M, "t % M"),
            Policy(lambda t: (t ** 2) % self.params.M, "t^2 % M")
        ]

class RandomDeterministicPolicyMaker(PolicyMaker):
    def __init__(self, params: GameParameterSet) -> None:
        random.seed(10)
        self.random_sequence_set = [
            [   random.randint(0, params.M - 1) for _ in range(params.T)] 
                for _ in range(params.N)
        ]
        super(RandomDeterministicPolicyMaker, self).__init__(
            params, self.get_policy_list)

    def get_sequence_value_at_time(self, seq_index: int, time: int):
        return self.random_sequence_set[seq_index][time] \
            if time < len(self.random_sequence_set[seq_index]) else math.nan

    def get_policy_from_seq_index(self, seq_index: int):
        return Policy(lambda t: self.get_sequence_value_at_time(seq_index, t),
            f"Random Deterministic Policy #{seq_index}")

    def get_policy_list(self):
        return [self.get_policy_from_seq_index(i) for i in range(self.params.N)]
