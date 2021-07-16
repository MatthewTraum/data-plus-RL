from GameElements import GameState
import os

HEAD_TITLE_POLICY = "Pol.Prd"
HEAD_TITLE_GUESSES = "Guesses"
SUFFIXES = ["th", "st", "nd", "rd"]

def get_nth_str(n: int) -> str:
    m = n % 100
    if m > 10 and m < 20:
        return f"{n}th"
    if n % 10 < 4:
        return f"{m}{SUFFIXES[n % 10]}"
    return f"{n}th"
    

def get_game_info_string(game_state: GameState) -> str:
    """
    Prints the available info about the game in a human-readable format.
    Suggested limits (based on printing the output) are 100 bands, 10 policies,
    T = 1,000,000.
    """
    output = f"** state of the game before the \
{get_nth_str(len(game_state.rounds) + 1)} turn **\n\n"

    for index, policy in enumerate(game_state.policy_list):
        output += f"Policy {index}: {policy}\n"

    output += f"\nParameters: {game_state.params}\n\n"

    output += " " * 9 + "Score A: {:d} // Score B: {:d}".format(
        int(game_state.score_a), int(game_state.score_b))

    output += f"\n\n" + " " * 8 + "|"
    num_policies = len(game_state.policy_list)

    whitespace = ((num_policies * 3 + 1) - len(HEAD_TITLE_POLICY))
    output += " " * (whitespace // 2) + HEAD_TITLE_POLICY + " " * (whitespace \
        // 2 + (1 if whitespace % 2 == 1 else 0)) + "|"

    whitespace = (10 - len(HEAD_TITLE_GUESSES))
    output += " " * (whitespace // 2) + HEAD_TITLE_GUESSES + " " * (whitespace \
        // 2 + (1 if whitespace % 2 == 1 else 0)) + f"|\n"

    output += " " * 8 + "| "

    for index, _ in enumerate(game_state.policy_list):
        output += "{:2d} ".format(index)

    output += f"|  T  R  A |\n"

    output += "-" * 8 + "|"
    output += "-" * (num_policies * 3 + 1) + "|" + "-" * 10 + f"|"

    for index, round in enumerate(game_state.rounds):
        line = f"\n"
        line += "t:{:6d}|".format(index)
        for policy in game_state.policy_list:
            line += " {:2d}".format(policy.get_bandwidth(index))
        line += " | {:2d} {:2d} {:2d} |".format(
            round.transmission_band,
            round.receiver_guess,
            round.adversary_guess
        )
        output += line

    if len(game_state.rounds) < game_state.params.T:
        line = f"\n"
        line += "t:{:6d}|".format(len(game_state.rounds))
        for policy in game_state.policy_list:
            line += " {:2d}".format(policy.get_bandwidth(len(game_state.rounds)))
        line += " |"

    output += line

    return output    

def show_string(info: str) -> None:
    if os.name == 'nt':
        info = info.replace(f"\n\n", "& echo. & echo ") \
            .replace(f"\n", "& echo ").replace("^", "^^").replace("|", "^|")
        os.system("start cmd /k \"echo " + info + " & echo. & pause & exit\"")
    else:
        print(info + "\n")

def show_game_info(game_state: GameState) -> None:
    show_string(get_game_info_string(game_state))

def show_info_with_extra(game_state: GameState, extra: str) -> None:
    show_string(extra + get_game_info_string(game_state))
