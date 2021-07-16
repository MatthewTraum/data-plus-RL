from GameParameters import GameParameterSet
import math

class Round():
    """
    A single round in the game, which contains a transmission
    on a given band along with guesses of both the receiver
    and the adversary.
    """
    def __init__(self, transmission_band: int, receiver_guess: int, 
        adversary_guess: int) -> None:
            self.transmission_band = transmission_band
            self.receiver_guess = receiver_guess
            self.adversary_guess = adversary_guess

    def __str__(self):
        return f"(T: {self.transmission_band} R: {self.receiver_guess} \
A: {self.adversary_guess})"

class Policy():
    """
    A wrapper used to store some function (`bandwidth_selector_function`)
    that takes time as an input and returns a value from 1 to M, 
    where M is the number of available bands. Also should contain a string
    representing the policy if possible.
    """
    def __init__(self, bandwidth_selector_function: 'function', desc: str) \
        -> None:
        self.get_bandwidth = bandwidth_selector_function
        self.desc = desc

    def __str__(self):
        return self.desc


class PolicyMaker():
    """
    The agent who determines the policies before the game starts.
    """
    def __init__(self, params: GameParameterSet, 
        policy_making_function: 'function') -> None:
            self.params = params
            self.get_policy_list = policy_making_function


class Transmitter():
    """
    The transmitter on Team A. Is the only one who can change the current
    policy, and/or communicate this change to the Receiver. Includes a 
    reference for a `policy_selector_function` "PSF" which would be called 
    during the game to determine whether a switch is necessary. 

    Syntax: 
      - PSF accepts a GameState
      - PSF returns (new_policy: int, communicate: boolean)
        - If new_policy is -1 then communication is ignored (nothing should happen)
        - Else set the new policy
          - Cost of R3 * log_2(N) if communication is true (even if policy
            stays the same)
          - Else if communication is false
            - Cost of R2 

    """
    def __init__(self, policy_selector_function: 'function', 
        start_policy: int) -> None:
            self.get_policy = policy_selector_function
            self.start_policy = start_policy

class Receiver():
    """
    The receiver on Team A. Must use available information (open
    list of past actions and any communication from Transmitter) to 
    select a predicted bandwidth.

    `bandwidth_prediction_function` should take a GameState as an input 
    and return an integer from 0 to M - 1.

    `communicate` should accept an integer and return None (this is how
    the transmitter will communicate a change in policy, if they choose)
    """

    def __init__(self, bandwidth_prediction_function: 'function',
        communication_channel: 'function') -> None:
            self.predict_policy = bandwidth_prediction_function
            self.communicate = communication_channel

class Adversary():
    """
    The adversary - the sole member of Team B. Must use available 
    information (list of past actions only) to select a predicted bandwidth.

    `bandwidth_prediction_function` should take a GameState as an input 
    and return an integer from 0 to M-1.
    """

    def __init__(self, bandwidth_prediction_function: 'function') -> None:
            self.predict_policy = bandwidth_prediction_function

class GameState():
    """
    The publicly available information about the game. 
    (Everything except the players.)
    """
    def __init__(self, params: GameParameterSet, policy_list: 'list[Policy]'):
        self.params = params
        self.t = 0
        self.score_a = 0
        self.score_b = 0
        self.policy_list = policy_list
        self.rounds = []

class Game():
    """
    A group containing a transmitter, receiver, adversary,
    policies, and actions, along with several other variables:
     - N (number of policies)
     - M (number of available bands)
     - T (length of game)
     - t (current time of the game)
     - R1, R2, R3 (rewards / costs)

    Redundant variables: (could be calculated from the above)
     - score_a
     - score_b

    Private variables: (not available to players)
     - policy_record
     - communication_record
     - current_policy_id
    """
    def __init__(self, transmitter: Transmitter, receiver: Receiver,
        adversary: Adversary, policy_list: 'list[Policy]', 
        params: GameParameterSet):
            self.current_policy_id = transmitter.start_policy
            self.transmitter = transmitter
            self.receiver = receiver
            self.adversary = adversary
            self.state = GameState(params, policy_list)
            self.policy_record = []
            self.communication_record = [True]

    def advance_time(self) -> bool:
        """
        Advances the time in the game, and calls on each player to make a 
        move. Returns true unless the game is over (when t is greater than 
        or equal to T).
        """
        if self.state.t >= self.state.params.T:
            return False

        # Transmit based on the selected policy --------------------------------

        self.policy_record.append(self.current_policy_id)
        policy = self.state.policy_list[self.current_policy_id]

        transmission_band = policy.get_bandwidth(self.state.t)

        receiver_guess = self.receiver.predict_policy(self.state)
        adversary_guess = self.adversary.predict_policy(self.state)

        self.state.rounds.append(Round(transmission_band, receiver_guess, 
            adversary_guess))
        
        if (adversary_guess == transmission_band):
            self.state.score_b += self.state.params.R3
        elif (receiver_guess == transmission_band):
            self.state.score_a += self.state.params.R1

        # After transmission, determine if a new policy is needed --------------
        if self.state.t + 1 < self.state.params.T:
            new_policy_id, communication = \
                self.transmitter.get_policy(self.state)

            if new_policy_id != -1:
                self.current_policy_id = new_policy_id
                if communication:
                    # Change the policy and communicate the change
                    self.receiver.communicate(new_policy_id)
                    self.state.score_a -= self.state.params.R3 * \
                        math.log2(self.state.params.N) + self.state.params.R2
                    self.communication_record.append(True)
                else:
                    self.communication_record.append(False)
                    # Change the policy and don't communicate the change
                    if new_policy_id != self.current_policy_id:
                        self.state.score_a -= self.state.params.R2
                    else:
                        # The transmitter didn't change the policy and 
                        # didn't communicate the policy number
                        pass
            elif communication:
                # Communicating the policy without switching it
                # QUESTION - would this ever happen?
                self.communication_record.append(True)
                self.state.score_a -= self.state.params.R3 * \
                    math.log2(self.state.params.N)
            else:
                # No policy change, and no communication
                self.communication_record.append(False)

        # Advance the time -----------------------------------------------------

        self.state.t += 1

        return True

    def __str__(self):
        game_str = f"--GAME--\n"
        for i in range(0, len(self.state.rounds)):
            game_str += f"Time {i}\t\
Policy {self.policy_record[i]}\t\
Choice {self.state.rounds[i]}\tPre-comm. \
{self.communication_record[i]}\n"

        return game_str