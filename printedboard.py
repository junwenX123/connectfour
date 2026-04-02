from pettingzoo.classic import connect_four_v3
import numpy as np


def print_board(observation: np.ndarray) -> None:
    """
    Print a human-readable version of the board.

    observation: numpy array of shape (6, 7, 2)
        observation[:, :, 0] = current player's pieces
        observation[:, :, 1] = opponent's pieces
    """
    board = observation  # alias

    # On parcourt les lignes de haut en bas (0 -> 5)
    for row in range(board.shape[0]):
        line_symbols = []
        for col in range(board.shape[1]):
            me = board[row, col, 0]
            opp = board[row, col, 1]

            if me == 1 and opp == 0:
                symbol = "X"   # current player
            elif me == 0 and opp == 1:
                symbol = "O"   # opponent
            else:
                symbol = "."   # empty
            line_symbols.append(symbol)

        print(" ".join(line_symbols))
    print()  # ligne vide pour séparer les états


# Test your function
env = connect_four_v3.env()
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        break

    print(f"\nAgent: {agent}")
    print_board(observation['observation'])

    # Make a few moves to see the board change
    env.step(3)
    if agent == env.agents[0]:
        break


env.close()
