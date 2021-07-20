import random, math
from GameElements import Transmitter, GameState
from ShowInfo import show_game_info, show_string
from Util import get_integer, confirm

class ExampleTransmitter(Transmitter):

    def policy_selector_function(self, game_state: GameState) -> int:
        # Always choose policy 0 and don't communicate it
        return 0, False

    def __init__(self, num_policies: int) -> None:
        super(ExampleTransmitter, self).__init__(self.policy_selector_function, 
            start_policy = 0)

class HumanTransmitter(Transmitter):

    def policy_selector_function(self, game_state: GameState) -> int:
        # Ask the player to choose policy and communication
        show_game_info(game_state)
        policy = get_integer("New policy [-1 = no change]? (0 - {:d})".format(
            game_state.params.N - 1), min=-1, max=game_state.params.N - 1)
        comm = confirm("Communicate the policy?")
        return policy, comm

    def __init__(self, num_policies: int) -> None:
        super(HumanTransmitter, self).__init__(self.policy_selector_function, 
            start_policy = get_integer("\nEnter start policy " + 
                f"(0-{num_policies - 1})", min=0, max=num_policies - 1))
    
class RandomTransmitter(Transmitter):

    def policy_selector_function(self, game_state: GameState) -> int:
        if random.random() < self.sw_prob:
            new_policy = random.randint(0, game_state.params.N - 2)
            if new_policy >= self.last_policy:
                new_policy += 1
            self.last_policy = new_policy
            return self.last_policy, True
        return self.last_policy, False

    def __init__(self, num_policies: int) -> None:
        self.last_policy = 0
        self.sw_prob = 0.2
        super().__init__(self.policy_selector_function, start_policy = 0)


class HumanInfoTransmitter(Transmitter):
    """
    A human transmitter that also keeps track of relevant game information.
    """
    def show_relevant_info(self):
        string = ""
        for k in self.relevant_info:
            string += f"{k}: {self.relevant_info[k]}\n"
        show_string(string)

    def initialize_relevant_info(self, game_state: GameState):
        self.relevant_info = {
            "current_policy" : 0,
            "next_bandwidth_for_current_policy": 0,
            "current_time": 0,
            "time_since_last_switch": 1,
            "percent_of_time_on_each_policy": [0 for _ in range(
                game_state.params.N)],
            "percent_time_adversary_guessed_each_band": [0 for _ in range(
                game_state.params.M)],
            "adversary_accuracy_for_each_policy": [0 for _ in range(
                game_state.params.N)],
            "adversary_accuracy_since_last_switch": 0,
            "adversary_success_in_last_4_turns": [],
            "adversary_accuracy_in_last_20_turns": 0,
            "adversary_accuracy_all_game": 0,
            "average_duration_between_switches": 0
        }

        self.policy_use_count = [0 for _ in range(game_state.params.N)]
        self.adversary_policy_correct_count = [0 for _ in range(
            game_state.params.N)]
        self.bandwidth_use_count = [0 for _ in range(game_state.params.M)]
        self.adversary_band_guess_count = [0 for _ in range(
            game_state.params.M)]

    def policy_selector_function(self, game_state: GameState) -> int:
        
        if self.relevant_info == None:
            self.initialize_relevant_info(game_state)

        # Check the adversary's guess on the last turn, then update info ------
    
        self.adversary_band_guess_count[
            game_state.rounds[-1].adversary_guess] += 1

        if game_state.rounds[-1].adversary_guess == \
                game_state.rounds[-1].transmission_band:
            # The adversary was correct
            self.adversary_policy_correct_count[self.policy_record[-1]] += 1
            self.adversary_correct_since_last_switch += 1
            self.relevant_info["adversary_success_in_last_4_turns"].append(
                True)
            self.adversary_correct_hx20.append(1)
            self.adversary_correct_total += 1

        else:
            # The adversary was incorrect
            self.relevant_info["adversary_success_in_last_4_turns"].append(
                False)
            self.adversary_correct_hx20.append(0)
    
        self.relevant_info["adversary_success_in_last_4_turns"] = \
            self.relevant_info["adversary_success_in_last_4_turns"][-4:]
        self.adversary_correct_hx20 = self.adversary_correct_hx20[-20:]
        
        self.policy_record.append(self.current_policy)  
        self.policy_use_count[self.current_policy] += 1
        self.bandwidth_use_count[game_state.policy_list[
            self.current_policy].get_bandwidth(game_state.t)] += 1


        self.relevant_info["current_policy"] = self.current_policy
        self.relevant_info["next_bandwidth_for_current_policy"] = \
            game_state.policy_list[self.current_policy].get_bandwidth(
                game_state.t + 1)
        self.relevant_info["current_time"] = game_state.t
        self.relevant_info["percent_of_time_on_each_policy"] = [
            self.policy_use_count[i] / (game_state.t  + 1)
            for i in range(game_state.params.N)]
        self.relevant_info["percent_time_adversary_guessed_each_band"] = [
            self.adversary_band_guess_count[i] / (game_state.t  + 1)
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

        return policy, True # always communicate

    def __init__(self, num_policies: int) -> None:
        super(HumanInfoTransmitter, self).__init__(self.policy_selector_function, 
            start_policy = get_integer("\nEnter start policy " + 
                f"(0-{num_policies - 1})", min=0, max=num_policies - 1))
        self.current_policy = self.start_policy
        self.relevant_info = None
        self.adversary_correct_since_last_switch = 0
        self.policy_record = [self.current_policy]
        self.adversary_correct_hx20 = []
        self.adversary_correct_total = 0
        self.switch_count = 1

