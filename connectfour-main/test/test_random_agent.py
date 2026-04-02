from __future__ import annotations

from typing import Dict, Tuple, List

from pettingzoo.classic import connect_four_v3

from random_agent import RandomAgent, WeightedRandomAgent


def play_one_game(env, agent_cls=RandomAgent, verbose: bool = True) -> Tuple[str | None, int]:
    """
    Play a single game between two agents of type `agent_cls`.

    Returns
    -------
    winner : str or None
        Name of the PettingZoo agent who won ('player_0', 'player_1') or None for draw.
    n_moves : int
        Number of moves played in the game.
    """
    agents = {
        name: agent_cls(env, player_name=name)
        for name in env.possible_agents
    }

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
                winner = None
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


def run_single_game(render_mode: str = "human", agent_cls=RandomAgent) -> None:
    """Tâche 2.5 : Exécuter une partie simple pour vérifier que tout marche."""
    env = connect_four_v3.env(render_mode=render_mode)
    env.reset(seed=42)

    winner, n_moves = play_one_game(env, agent_cls=agent_cls, verbose=True)

    print(f"Game finished in {n_moves} moves, winner = {winner}")
    input("Press Enter to close the window...")
    env.close()


def run_multiple_games(
    num_games: int = 100,
    agent_cls=RandomAgent,
    render_mode: str | None = None,
    verbose: bool = False,
) -> Tuple[Dict[str, int], List[int]]:
    """
    Tâche 2.6 : Tester plusieurs parties.

    Parameters
    ----------
    num_games : int
        Number of games to play.
    agent_cls :
        Class of the agent to use (RandomAgent or WeightedRandomAgent).
    render_mode : str or None
        None for no rendering (faster).
    verbose : bool
        If True, print each move (slow).

    Returns
    -------
    stats : dict
        Counts of wins for 'player_0', 'player_1' and draws.
    move_counts : list[int]
        Number of moves for each game.
    """
    stats = {"player_0": 0, "player_1": 0, "draw": 0}
    move_counts: List[int] = []

    for i in range(num_games):
        env = connect_four_v3.env(render_mode=render_mode)
        env.reset(seed=i) 

        winner, n_moves = play_one_game(env, agent_cls=agent_cls, verbose=verbose)
        move_counts.append(n_moves)

        if winner is None:
            stats["draw"] += 1
        elif winner == "player_0":
            stats["player_0"] += 1
        elif winner == "player_1":
            stats["player_1"] += 1
        else:
            stats.setdefault(winner, 0)
            stats[winner] += 1
        input("Press Enter to close the window...")
        env.close()

    return stats, move_counts


if __name__ == "__main__":
    run_single_game(render_mode="human", agent_cls=RandomAgent)

    stats, move_counts = run_multiple_games(
        num_games=100,
        agent_cls=RandomAgent,
        render_mode=None,
        verbose=False,
    )

    total_games = len(move_counts)
    avg_moves = sum(move_counts) / total_games

    print("\n=== RandomAgent statistics ===")
    print(f"Number of games: {total_games}")
    print(f"Wins player_0 : {stats['player_0']}")
    print(f"Wins player_1 : {stats['player_1']}")
    print(f"Draws        : {stats['draw']}")
    print(f"Average number of moves: {avg_moves:.2f}")
