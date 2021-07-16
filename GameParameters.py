class GameParameterSet():
    def __init__(self, M: int, N: int, T: int, R1: float, 
        R2: float, R3: float):
            self.M = M
            self.N = N
            self.T = T
            self.R1 = R1
            self.R2 = R2
            self.R3 = R3

    def __str__(self):
        return f"(M: {self.M}, N: {self.N}, T: {self.T}, \
R1: {self.R1}, R2: {self.R2}, R3: {self.R3})"

    def are_equal_to(self, other_param_set: 'GameParameterSet'):
        return (
            self.M == other_param_set.M and
            self.N == other_param_set.N and
            self.T == other_param_set.T and
            self.R1 == other_param_set.R1 and
            self.R2 == other_param_set.R2 and
            self.R3 == other_param_set.R3
        )