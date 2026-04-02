import time
import numpy as np
from pettingzoo.classic import connect_four_v3

from smart_agent import SmartAgent
from minimax_agent import MinimaxAgent


def play_one_game(env, agent0, agent1, collect_times=True):
    env.reset()

    a0_name, a1_name = env.agents[0], env.agents[1]

    agents = {a0_name: agent0, a1_name: agent1}
    times = {a0_name: [], a1_name: []}

    for agent_name in env.agent_iter():
        obs, reward, terminated, truncated, info = env.last()

        if terminated or truncated:
            env.step(None)
            continue

        if isinstance(obs, dict):
            board = obs["observation"]
            action_mask = obs["action_mask"]
        else:
            board = obs
            action_mask = info.get("action_mask", None) if info else None

        a = agents[agent_name]

        t0 = time.perf_counter()
        action = a.choose_action(
            board,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            action_mask=action_mask,
        )
        t1 = time.perf_counter()

        if collect_times:
            times[agent_name].append(t1 - t0)

        env.step(action)


    r0 = env.rewards.get(a0_name, 0)
    r1 = env.rewards.get(a1_name, 0)

    if r0 > r1:
        winner = 0
    elif r1 > r0:
        winner = 1
    else:
        winner = None

    return winner, times[a0_name], times[a1_name]


def evaluate_depth(depth, num_games=20, seed=0):
 
    rng = np.random.default_rng(seed)

    wins = 0
    losses = 0
    draws = 0

    all_times_minimax = []

    for g in range(num_games):
        env = connect_four_v3.env(render_mode=None)
        env.reset(seed=seed)  
        minimax = MinimaxAgent(env, depth=depth)
        smart = SmartAgent(env)

        if g % 2 == 0:
            agent0, agent1 = minimax, smart
            minimax_index = 0
        else:
            agent0, agent1 = smart, minimax
            minimax_index = 1

        winner, times0, times1 = play_one_game(env, agent0, agent1)

        minimax_times = times0 if minimax_index == 0 else times1
        all_times_minimax.extend(minimax_times)

        if winner is None:
            draws += 1
        elif winner == minimax_index:
            wins += 1
        else:
            losses += 1

        env.close()

    avg_t = float(np.mean(all_times_minimax)) if all_times_minimax else 0.0
    max_t = float(np.max(all_times_minimax)) if all_times_minimax else 0.0

    return {
        "depth": depth,
        "games": num_games,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "win_rate": wins / num_games,
        "avg_time_per_move": avg_t,
        "max_time_per_move": max_t,
    }


def main():
    for depth in [2, 3, 4, 5, 6]:
        stats = evaluate_depth(depth, num_games=20)
        print(
            f"d={stats['depth']} | "
            f"W/L/D={stats['wins']}/{stats['losses']}/{stats['draws']} | "
            f"win_rate={stats['win_rate']:.2f} | "
            f"avg_t={stats['avg_time_per_move']*1000:.1f}ms | "
            f"max_t={stats['max_time_per_move']*1000:.1f}ms"
        )


if __name__ == "__main__":
    main()
