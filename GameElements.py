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

    def advance_time(self, BigState):
        """
        Advances the time in the game, and calls on each player to make a 
        move. Returns true unless the game is over (when t is greater than 
        or equal to T).
        """

        reward = 0

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
            reward += self.state.params.R1

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
                reward -= self.state.params.R3 * \
                    math.log2(self.state.params.N)
            else:
                # No policy change, and no communication
                self.communication_record.append(False)

        # Advance the time -----------------------------------------------------

        self.state.t += 1
        print(self.state.t)
        if self.state.t >= self.state.params.T:
            done = True
        else:
            done = False
        nextState = self.updateRelevantInfo(self.transmitter, self.game_state)






        return new_policy_id, reward, nextState, done

    def updateRelevantInfo(transmitter, game_state: GameState) -> int:

        if transmitter.relevant_info == None:
            transmitter.initialize_relevant_info(game_state)

        # Check the adversary's guess on the last turn, then update info ------

        transmitter.adversary_band_guess_count[
            game_state.rounds[-1].adversary_guess] += 1

        if game_state.rounds[-1].adversary_guess == \
                game_state.rounds[-1].transmission_band:
            # The adversary was correct
            transmitter.adversary_policy_correct_count[transmitter.policy_record[-1]] += 1
            transmitter.adversary_correct_since_last_switch += 1
            transmitter.relevant_info["adversary_success_in_last_4_turns"].append(
                True)
            transmitter.adversary_correct_hx20.append(1)
            transmitter.adversary_correct_total += 1

        else:
            # The adversary was incorrect
            transmitter.relevant_info["adversary_success_in_last_4_turns"].append(
                False)
            transmitter.adversary_correct_hx20.append(0)

        transmitter.relevant_info["adversary_success_in_last_4_turns"] = \
            transmitter.relevant_info["adversary_success_in_last_4_turns"][-4:]
        transmitter.adversary_correct_hx20 = transmitter.adversary_correct_hx20[-20:]

        transmitter.policy_record.append(transmitter.current_policy)
        transmitter.policy_use_count[transmitter.current_policy] += 1
        transmitter.bandwidth_use_count[game_state.policy_list[
            transmitter.current_policy].get_bandwidth(game_state.t)] += 1

        transmitter.relevant_info["current_policy"] = transmitter.current_policy
        transmitter.relevant_info["next_bandwidth_for_current_policy"] = \
            game_state.policy_list[transmitter.current_policy].get_bandwidth(
                game_state.t + 1)
        transmitter.relevant_info["current_time"] = game_state.t
        transmitter.relevant_info["percent_of_time_on_each_policy"] = [
            transmitter.policy_use_count[i] / (game_state.t + 1)
            for i in range(game_state.params.N)]
        self.relevant_info["percent_time_adversary_guessed_each_band"] = [
            self.adversary_band_guess_count[i] / (game_state.t + 1)
            for i in range(game_state.params.M)]
        # NOTE: consider (5 lines below) what value to insert for
        # unused policies (currently math.nan)
        self.relevant_info["adversary_accuracy_for_each_policy"] = [
            self.adversary_policy_correct_count[i] / self.policy_use_count[i]
            if self.policy_use_count[i] != 0 else math.nan
            for i in range(game_state.params.N)]
        self.relevant_info["adversary_accuracy_since_last_switch"] = \
            self.adversary_correct_since_last_switch / self.relevant_info["time_since_last_switch"]
        self.relevant_info["adversary_accuracy_in_last_20_turns"] = sum(
            self.adversary_correct_hx20) / min(game_state.t + 1, 20)
        self.relevant_info["adversary_accuracy_all_game"] = \
            self.adversary_correct_total / (game_state.t + 1)
        self.relevant_info["average_duration_between_switches"] = \
            (game_state.t + 1) / self.switch_count

        # Ask the player to choose policy -------------------------------------

        self.show_relevant_info()
        policy = get_integer("New policy [-1 = no change]? (0 - {:d})".format(
            game_state.params.N - 1), min=-1, max=game_state.params.N - 1)

        # Update current policy info ------------------------------------------

        if policy == -1:
            self.relevant_info["time_since_last_switch"] += 1
        else:
            self.relevant_info["time_since_last_switch"] = 1
            self.adversary_correct_since_last_switch = 0
            self.current_policy = policy
            self.switch_count += 1

    def __str__(self):
        game_str = f"--GAME--\n"
        for i in range(0, len(self.state.rounds)):
            game_str += f"Time {i}\t\
Policy {self.policy_record[i]}\t\
Choice {self.state.rounds[i]}\tPre-comm. \
{self.communication_record[i]}\n"

        return game_str