from __future__ import annotations

"""
Système de comparaison par tournoi entre plusieurs agents Connect Four.

Usage:
    python tournament.py
"""

from typing import Dict, Type, Tuple, List, Optional

from pettingzoo.classic import connect_four_v3

from random_agent import RandomAgent
from smart_agent import SmartAgent


def play_one_game_with_agents(
    env,
    agents: Dict[str, object],
    verbose: bool = False,
) -> Tuple[Optional[str], int]:
    """Identique à la fonction du test_suite, réécrite ici pour être autonome."""
    n_moves = 0
    winner: Optional[str] = None

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
                    print(f"{other} wins!")
            else:
                winner = None
                if verbose:
                    print("It's a draw!")
            break

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


def run_matchup(
    agent_cls_A: Type[object],
    agent_cls_B: Type[object],
    num_games: int = 100,
) -> Tuple[Dict[str, int], List[int]]:
    """
    Fait s'affronter agent_cls_A (player_0) et agent_cls_B (player_1) num_games fois.

    Retourne:
        stats: dict avec clés 'A', 'B', 'draw'
        move_counts: liste du nombre de coups par partie
    """
    stats = {"A": 0, "B": 0, "draw": 0}
    move_counts: List[int] = []

    for i in range(num_games):
        env = connect_four_v3.env()
        env.reset(seed=i)

        agents = {
            "player_0": agent_cls_A(env, player_name="A"),
            "player_1": agent_cls_B(env, player_name="B"),
        }

        winner, n_moves = play_one_game_with_agents(env, agents, verbose=False)
        move_counts.append(n_moves)

        if winner is None:
            stats["draw"] += 1
        elif winner == "player_0":
            stats["A"] += 1
        elif winner == "player_1":
            stats["B"] += 1

        env.close()

    return stats, move_counts


def main():
    # Liste des agents à comparer
    agent_classes: Dict[str, Type[object]] = {
        "Random": RandomAgent,
        "Smart": SmartAgent,    }

    names = list(agent_classes.keys())

    print("=== Tournament between agents ===")
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            nameA, nameB = names[i], names[j]
            clsA, clsB = agent_classes[nameA], agent_classes[nameB]

            print(f"\n--- Matchup {nameA} (A) vs {nameB} (B) ---")
            stats, moves = run_matchup(clsA, clsB, num_games=50)

            total = sum(stats.values())
            avg_moves = sum(moves) / len(moves)

            win_rate_A = stats["A"] / total
            win_rate_B = stats["B"] / total

            print(f"{nameA} wins : {stats['A']} ({win_rate_A:.1%})")
            print(f"{nameB} wins : {stats['B']} ({win_rate_B:.1%})")
            print(f"Draws       : {stats['draw']}")
            print(f"Average moves per game: {avg_moves:.2f}")

