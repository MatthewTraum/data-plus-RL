from GameElements import Receiver, GameState
from ShowInfo import show_info_with_extra
from Util import get_integer

class ExampleReceiver(Receiver):

    def bandwidth_predictor_function(self, game_state: GameState) -> int:
        # Guess based on whatever the last policy was
        predicted_policy = game_state.policy_list \
            [self.last_policy_communicated]
        return predicted_policy.get_bandwidth(game_state.t)

    def communication_channel(self, policy) -> None:
        self.last_policy_communicated = policy

    def __init__(self) -> None:
        super(ExampleReceiver, self).__init__(self.bandwidth_predictor_function, 
            self.communication_channel)


class HumanReceiver(Receiver):

    def bandwidth_predictor_function(self, game_state: GameState) -> int:
        show_info_with_extra(game_state, self.communication + f"\n\n")
        self.communication = "Policy not communicated."
        return get_integer("Predicted band? (0 - {:d})".format(
            game_state.params.M - 1), min=0, max=game_state.params.M - 1)

    def communication_channel(self, policy) -> None:
        self.communication = "Communicated policy: {:d}".format(policy)

    def __init__(self) -> None:
        super(HumanReceiver, self).__init__(self.bandwidth_predictor_function, 
            self.communication_channel)
        self.communication = "Policy not communicated."