"""
Minimax agent with alpha-beta pruning for Connect Four (PettingZoo connect_four_v3)
"""

import numpy as np
import random
from math import inf


class MinimaxAgent:
    """
    Agent using minimax algorithm with alpha-beta pruning
    """

    def __init__(self, env, depth=4, player_name=None):
        """
        Initialize minimax agent

        Parameters:
            env: PettingZoo environment
            depth: How many moves to look ahead
            player_name: Optional name
        """
        self.env = env
        self.action_space = env.action_space(env.agents[0])
        self.depth = depth
        self.player_name = player_name or f"Minimax(d={depth})"

    # ---------- Public API ----------

    def choose_action(
        self,
        observation,
        reward=0.0,
        terminated=False,
        truncated=False,
        info=None,
        action_mask=None,
    ):
        """
        Choose action using minimax algorithm.

        observation: numpy array (6, 7, 2)
        action_mask: length-7 array, 1 = legal move
        """
        valid_actions = [i for i, valid in enumerate(action_mask) if valid == 1]

        # fallback: random
        if not valid_actions:
            return self.action_space.sample()

        best_action = None
        best_value = -inf

        board = observation.copy()

        for action in valid_actions:
            # simulate my move (channel 0 = this agent)
            new_board = self._simulate_move(board, action, channel=0)
            # then it's opponent's turn -> minimizing
            value = self._minimax(new_board, self.depth - 1, -inf, inf, maximizing=False)

            if value > best_value:
                best_value = value
                best_action = action

        if best_action is None:
            best_action = random.choice(valid_actions)
        return best_action

    # ---------- Minimax core ----------

    def _minimax(self, board, depth, alpha, beta, maximizing):
        """
        Minimax algorithm with alpha-beta pruning

        board: current board (6, 7, 2)
        depth: remaining depth
        maximizing: True if it's our (player 0) turn
        """
        # Terminal tests
        if self._check_win(board, 0):
            return 1_000_000  # huge positive
        if self._check_win(board, 1):
            return -1_000_000  # huge negative

        valid_moves = self._get_valid_moves(board)

        # No more moves (draw) or depth limit
        if depth == 0 or not valid_moves:
            return self._evaluate(board)

        if maximizing:
            value = -inf
            for col in valid_moves:
                child_board = self._simulate_move(board, col, channel=0)
                value = max(
                    value,
                    self._minimax(child_board, depth - 1, alpha, beta, maximizing=False),
                )
                alpha = max(alpha, value)
                if alpha >= beta:  # beta cut-off
                    break
            return value
        else:
            value = inf
            for col in valid_moves:
                child_board = self._simulate_move(board, col, channel=1)
                value = min(
                    value,
                    self._minimax(child_board, depth - 1, alpha, beta, maximizing=True),
                )
                beta = min(beta, value)
                if alpha >= beta:  # alpha cut-off
                    break
            return value

    # ---------- Board helpers ----------

    def _simulate_move(self, board, col, channel):
        """
        Simulate placing a piece without modifying original board

        board: (6, 7, 2)
        col: column index 0..6
        channel: 0 for this agent, 1 for opponent

        returns: new_board
        """
        new_board = board.copy()

        # Find lowest empty row in column
        for row in range(5, -1, -1):  # from bottom (row=5) to top (row=0)
            if new_board[row, col, 0] == 0 and new_board[row, col, 1] == 0:
                new_board[row, col, channel] = 1
                return new_board

        # Column full -> return board unchanged (should not happen if we check valid moves)
        return new_board

    def _get_valid_moves(self, board):
        """
        Get list of valid column indices (not full)
        """
        valid = []
        # cell is empty if both channels are 0
        top_occupancy = board[0, :, 0] + board[0, :, 1]
        for col in range(7):
            if top_occupancy[col] == 0:
                valid.append(col)
        return valid

    # ---------- Evaluation ----------

    def _evaluate(self, board):
        """
        Evaluate board position from player 0's perspective.
        Positive = good for us, negative = good for opponent.
        """

        # Immediate win/lose already handled in minimax;
        if self._check_win(board, 0):
            return 1_000_000
        if self._check_win(board, 1):
            return -1_000_000

        score = 0
        my_channel = 0
        opp_channel = 1

        # 1) center column preference
        center_col = 3  # 0..6
        center_array = board[:, center_col, my_channel]
        center_count = np.sum(center_array)
        score += center_count * 3

        # helper: all 4-cell windows
        def score_window(window_my, window_opp):
            cnt_my = np.sum(window_my)
            cnt_opp = np.sum(window_opp)
            cnt_empty = 4 - cnt_my - cnt_opp
            s = 0
            if cnt_my == 4:
                s += 100
            elif cnt_my == 3 and cnt_empty == 1:
                s += 5
            elif cnt_my == 2 and cnt_empty == 2:
                s += 2

            # threat from opponent
            if cnt_opp == 3 and cnt_empty == 1:
                s -= 4
            if cnt_opp == 4:
                s -= 100
            return s

        # horizontal windows
        for row in range(6):
            for col in range(4):  # 0..3
                w_my = board[row, col : col + 4, my_channel]
                w_opp = board[row, col : col + 4, opp_channel]
                score += score_window(w_my, w_opp)

        # vertical windows
        for col in range(7):
            for row in range(3):  # 0..2
                w_my = board[row : row + 4, col, my_channel]
                w_opp = board[row : row + 4, col, opp_channel]
                score += score_window(w_my, w_opp)

        # positive diagonal (\)
        for row in range(3):  # 0..2
            for col in range(4):  # 0..3
                w_my = np.array(
                    [
                        board[row + i, col + i, my_channel]
                        for i in range(4)
                    ]
                )
                w_opp = np.array(
                    [
                        board[row + i, col + i, opp_channel]
                        for i in range(4)
                    ]
                )
                score += score_window(w_my, w_opp)

        # negative diagonal (/)
        for row in range(3, 6):  # 3..5
            for col in range(4):  # 0..3
                w_my = np.array(
                    [
                        board[row - i, col + i, my_channel]
                        for i in range(4)
                    ]
                )
                w_opp = np.array(
                    [
                        board[row - i, col + i, opp_channel]
                        for i in range(4)
                    ]
                )
                score += score_window(w_my, w_opp)

        return score

    # ---------- Win detection ----------

    def _check_win(self, board, channel):
        """
        Check if player (channel 0 or 1) has 4 in a row.
        """

        # horizontal
        for row in range(6):
            for col in range(4):
                if np.sum(board[row, col : col + 4, channel]) == 4:
                    return True

        # vertical
        for col in range(7):
            for row in range(3):
                if np.sum(board[row : row + 4, col, channel]) == 4:
                    return True

        # positive diagonal (\)
        for row in range(3):
            for col in range(4):
                if (
                    board[row, col, channel]
                    + board[row + 1, col + 1, channel]
                    + board[row + 2, col + 2, channel]
                    + board[row + 3, col + 3, channel]
                    == 4
                ):
                    return True

        # negative diagonal (/)
        for row in range(3, 6):
            for col in range(4):
                if (
                    board[row, col, channel]
                    + board[row - 1, col + 1, channel]
                    + board[row - 2, col + 2, channel]
                    + board[row - 3, col + 3, channel]
                    == 4
                ):
                    return True

        return False

