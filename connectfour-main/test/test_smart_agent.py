from __future__ import annotations

"""
Unit and integration tests for SmartAgent.

Exercice 3 : Tâche 3.5 (tests unitaires) et Tâche 3.6 (test d'intégration).
"""


from typing import Dict, Tuple, List

import numpy as np
from pettingzoo.classic import connect_four_v3

from random_agent import RandomAgent
from smart_agent import SmartAgent


# ----------------------------------------------------------------------
# Outil commun : jouer une partie entre deux agents donnés
# ----------------------------------------------------------------------
def play_one_game_with_agents(
    env,
    agents: Dict[str, object],
    verbose: bool = False,
) -> Tuple[str | None, int]:
    """
    Play a single game with the given mapping agent_name -> agent object.

    Returns
    -------
    winner : str or None
    n_moves : int
    """
    n_moves = 0
    winner: str | None = None

    for agent_name in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            if reward == 1:
                winner = agent_name
                if verbose:
                    print(f"{agent_name} wins!")
            elif reward == -1:
                other = "player_1" if agent_name == "player_0" else "player_0"
                winner = other
                if verbose:
                    print("The other player wins!")
            else:
                winner = None
                if verbose:
                    print("It's a draw!")
            break
        else:
            obs_array = observation["observation"]
            mask = observation["action_mask"]

            action = agents[agent_name].choose_action(
                observation=obs_array,
                reward=reward,
                terminated=termination,
                truncated=truncation,
                info=info,
                action_mask=mask,
            )

            if verbose:
                print(f"{agent_name} plays column {action}")

            env.step(action)
            n_moves += 1

    return winner, n_moves


# ----------------------------------------------------------------------
# Tests unitaires sur les fonctions auxiliaires
# ----------------------------------------------------------------------
def _make_dummy_agent() -> SmartAgent:
    """Créer un SmartAgent avec un environnement Connect Four vide."""
    env = connect_four_v3.env()
    env.reset(seed=0)
    return SmartAgent(env, player_name="test_agent")


def test_get_valid_actions():
    agent = _make_dummy_agent()

    mask = np.array([1, 1, 1, 1, 1, 1, 1], dtype=int)
    assert agent._get_valid_actions(mask) == [0, 1, 2, 3, 4, 5, 6]

    mask = np.array([0, 1, 0, 1, 0, 1, 0], dtype=int)
    assert agent._get_valid_actions(mask) == [1, 3, 5]


def test_get_next_row():
    agent = _make_dummy_agent()
    board = np.zeros((6, 7, 2), dtype=int)

    assert agent._get_next_row(board, 3) == 5

    board[5, 3, 0] = 1
    assert agent._get_next_row(board, 3) == 4


def test_check_win_from_position_horizontal():
    agent = _make_dummy_agent()
    board = np.zeros((6, 7, 2), dtype=int)

    row = 5
    for c in range(4):
        board[row, c, 0] = 1
    assert agent._check_win_from_position(board, row, 3, channel=0)
    assert agent._check_win_from_position(board, row, 0, channel=0)


def test_check_win_from_position_vertical():
    agent = _make_dummy_agent()
    board = np.zeros((6, 7, 2), dtype=int)

    col = 2
    for r in range(2, 6):
        board[r, col, 1] = 1
    assert agent._check_win_from_position(board, 5, col, channel=1)
    assert agent._check_win_from_position(board, 2, col, channel=1)


def test_find_winning_move():
    agent = _make_dummy_agent()
    board = np.zeros((6, 7, 2), dtype=int)

    board[5, 0, 0] = 1
    board[5, 1, 0] = 1
    board[5, 2, 0] = 1

    valid_actions = list(range(7))
    col = agent._find_winning_move(board, valid_actions, channel=0)
    assert col == 3


# ----------------------------------------------------------------------
# Test d'intégration : SmartAgent vs RandomAgent
# ----------------------------------------------------------------------
def tournament_smart_vs_random(
    num_games: int = 100,
    render_mode: str | None = None,
    verbose: bool = False,
) -> Tuple[Dict[str, int], List[int]]:
    """
    Run a tournament SmartAgent vs RandomAgent.

    Returns
    -------
    stats : dict
        keys = 'smart', 'random', 'draw'
    move_counts : list[int]
        number of moves for each game
    """
    stats = {"smart": 0, "random": 0, "draw": 0}
    move_counts: List[int] = []

    for i in range(num_games):
        env = connect_four_v3.env(render_mode=render_mode)
        env.reset(seed=i)

        # player_0 = Smart, player_1 = Random
        agents = {
            "player_0": SmartAgent(env, player_name="smart"),
            "player_1": RandomAgent(env, player_name="random"),
        }

        winner, n_moves = play_one_game_with_agents(env, agents, verbose=verbose)
        move_counts.append(n_moves)

        if winner is None:
            stats["draw"] += 1
        elif winner == "player_0":
            stats["smart"] += 1
        elif winner == "player_1":
            stats["random"] += 1

        env.close()

    return stats, move_counts


if __name__ == "__main__":
    print("Running simple unit tests...")
    test_get_valid_actions()
    test_get_next_row()
    test_check_win_from_position_horizontal()
    test_check_win_from_position_vertical()
    test_find_winning_move()
    print("All simple unit tests passed.\n")

    stats, move_counts = tournament_smart_vs_random(
        num_games=100,
        render_mode=None,
        verbose=False,
    )

    total_games = len(move_counts)
    avg_moves = sum(move_counts) / total_games

    print("=== SmartAgent vs RandomAgent ===")
    print(f"Number of games : {total_games}")
    print(f"Wins SmartAgent : {stats['smart']}")
    print(f"Wins RandomAgent: {stats['random']}")
    print(f"Draws           : {stats['draw']}")
    print(f"Average number of moves: {avg_moves:.2f}")

