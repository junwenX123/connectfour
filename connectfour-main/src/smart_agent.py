# -*- coding: utf-8 -*-
from __future__ import annotations

"""
Created on Fri Nov 28 00:11:22 2025

@author: User
"""

"""
SmartAgent for Connect Four (rule-based).

Exercice 3 : Implémenter un agent basé sur des règles.

  Strategy priority:
    1. Win if possible
    2. Block opponent from winning
    3. Create a double threat if possible
    4. Play center if available
    5. Random valid move
"""


import random
from typing import Optional, Dict, Any, List

import numpy as np
from pettingzoo.classic import connect_four_v3

# Logging optionnel avec loguru (comme dans random_agent.py)
try:
    from loguru import logger
except ImportError:
    logger = None


class SmartAgent:
    """
    A rule-based agent that plays strategically.
    """

    def __init__(self, env, player_name: Optional[str] = None):
        """
        Initialize the smart agent.

        Parameters
        ----------
        env : PettingZoo environment
            The shared Connect Four environment.
        player_name : str, optional
            Optional display name for the agent.
        """
        self.env = env
        self.player_name = player_name or "SmartAgent"

        any_agent = env.possible_agents[0]
        self.action_space = env.action_space(any_agent)

    def reset(self) -> None:
        """Reset internal state before a new game (stateless agent)."""
        return

    # ------------------------------------------------------------------
    # Choix d'action : stratégie à règles
    # ------------------------------------------------------------------
    def choose_action(
        self,
        observation: np.ndarray,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        info: Optional[Dict[str, Any]] = None,
        action_mask: Optional[np.ndarray] = None,
    ) -> Optional[int]:
        """
        Choose an action using the rule-based strategy.

        Parameters
        ----------
        observation : np.ndarray
            Current board state (6, 7, 2).
            Channel 0 = current player's pieces
            Channel 1 = opponent's pieces
        action_mask : np.ndarray, shape (7,)
            Binary vector where 1 = legal move, 0 = illegal (full column).

        Returns
        -------
        int or None
            Column index (0–6) to play, or None if no action is possible.
        """
        if terminated or truncated:
            return None

        if action_mask is None:
            valid_actions = list(range(7))
        else:
            valid_actions = self._get_valid_actions(action_mask)

        if not valid_actions:
            if logger is not None:
                logger.warning(f"[{self.player_name}] no valid actions available")
            return None

        board = observation  # alias
        if action_mask is None:
            valid_actions = list(range(7))
        else:
            valid_actions = self._get_valid_actions(action_mask)

        valid_actions = [c for c in valid_actions if self._get_next_row(board, c) is not None]

        if not valid_actions:
            if logger is not None:
                logger.warning(f"[{self.player_name}] no valid actions (all full?)")
            return None
        # ---- Règle 1 : tenter de gagner immédiatement (canal 0) ----
        winning_move = self._find_winning_move(board, valid_actions, channel=0)
        if winning_move is not None:
            if logger is not None:
                logger.success(
                    f"[{self.player_name}] WINNING MOVE -> column {winning_move}"
                )
            return winning_move

        # ---- Règle 2 : bloquer l'adversaire (canal 1) ----
        blocking_move = self._find_winning_move(board, valid_actions, channel=1)
        if blocking_move is not None:
            if logger is not None:
                logger.warning(
                    f"[{self.player_name}] BLOCKING -> column {blocking_move}"
                )
            return blocking_move
        # ---- Rule 3  double threat----
        for col in valid_actions:
            if self._creates_double_threat(board, col, channel=0):
                if logger is not None:
                    logger.info(
                        f"[{self.player_name}] DOUBLE THREAT -> column {col}"
                    )
                return col

        # ---- Règle  : préférer le centre si possible ----
        center_preference = [3, 2, 4, 1, 5, 0, 6]
        for col in center_preference:
            if col in valid_actions:
          
                if logger is not None:
                    logger.info( f"[{self.player_name}] CENTER PREFERENCE -> column {col}"
                                    )
                return col

        # ---- Règle 4 : coup aléatoire parmi les actions valides ----
        action = int(random.choice(valid_actions))
        if logger is not None:
            logger.debug(
                f"[{self.player_name}] RANDOM CHOICE -> column {action} "
                f"among {valid_actions}"
            )
        return action

    # ------------------------------------------------------------------
    # Fonctions auxiliaires (Tâche 3.4)
    # ------------------------------------------------------------------
    def _get_valid_actions(self, action_mask: np.ndarray) -> List[int]:
        """
        Convert the action mask into a list of valid column indices.

        Parameters
        ----------
        action_mask : np.ndarray, shape (7,)
            1 for valid columns, 0 for invalid.

        Returns
        -------
        list[int]
            Indices of valid columns.
        """
        return [i for i, m in enumerate(action_mask) if m == 1]

    def _get_next_row(self, board: np.ndarray, col: int) -> Optional[int]:
        """
        Find which row a piece would land in if dropped in column `col`.

        Parameters
        ----------
        board : np.ndarray, shape (6, 7, 2)
            Current board.
        col : int
            Column index (0–6).

        Returns
        -------
        int or None
            Row index (0–5) if space available, None if column is full.
        """
        for row in range(board.shape[0] - 1, -1, -1):  # 5, 4, ..., 0
            if board[row, col, 0] == 0 and board[row, col, 1] == 0:
                return row
        return None  # column is full
    def _creates_double_threat(
        self,
        board: np.ndarray,
        col: int,
        channel: int,
    ) ->  bool:
        """
        Check if playing in column `col` creates TWO different winning threats
        for the given `channel` on the NEXT move (double threat).

        Parameters
        ----------
        board : np.ndarray, shape (6, 7, 2)
            Current board state.
        col : int
            Column we are considering to play now.
        channel : int
            0 for current player, 1 for opponent.

    Returns
    -------
    bool
        True if this move creates a double threat, False otherwise.
    """
        row = self._get_next_row(board, col)
        if row is None:
        # 
            return False

        board_after = board.copy()
        board_after[row, col, channel] = 1

        n_cols = board_after.shape[1]
        valid_next_actions: List[int] = []
        for c in range(n_cols):
            if board_after[0, c, 0] == 0 and board_after[0, c, 1] == 0:
                valid_next_actions.append(c)

        winning_next_moves = 0
        for next_col in valid_next_actions:
            next_row = self._get_next_row(board_after, next_col)
            if next_row is None:
                continue

            tmp_board = board_after.copy()
            tmp_board[next_row, next_col, channel] = 1

            if self._check_win_from_position(tmp_board, next_row, next_col, channel):
                winning_next_moves += 1
                if winning_next_moves >= 2:
                    return True

        return False

    def _check_win_from_position(
        self,
        board: np.ndarray,
        row: int,
        col: int,
        channel: int,
    ) -> bool:
        """
        Check if placing a piece at (row, col) for `channel` creates 4 in a row.

        Parameters
        ----------
        board : np.ndarray, shape (6, 7, 2)
        row : int
        col : int
        channel : int
            0 for current player, 1 for opponent.

        Returns
        -------
        bool
            True if this position creates 4 in a row/column/diag, False otherwise.
        """
        directions = [
            (0, 1),   # horizontal
            (1, 0),   # vertical
            (1, 1),   # diagonal /
            (1, -1),  # diagonal \
        ]
        n_rows, n_cols = board.shape[0], board.shape[1]

        for dr, dc in directions:
            count = 1  # compter (row, col) lui-même

            # direction + (dr, dc)
            r, c = row + dr, col + dc
            while 0 <= r < n_rows and 0 <= c < n_cols and board[r, c, channel] == 1:
                count += 1
                r += dr
                c += dc

            # direction - (dr, dc)
            r, c = row - dr, col - dc
            while 0 <= r < n_rows and 0 <= c < n_cols and board[r, c, channel] == 1:
                count += 1
                r -= dr
                c -= dc

            if count >= 4:
                return True

        return False

    def _find_winning_move(
        self,
        observation: np.ndarray,
        valid_actions: List[int],
        channel: int,
    ) -> Optional[int]:
        """
        Find a move that creates a 4 in a row for the specified player.

        Parameters
        ----------
        observation : np.ndarray, shape (6, 7, 2)
            Current board state.
        valid_actions : list[int]
            List of valid column indices.
        channel : int
            0 for current player, 1 for opponent.

        Returns
        -------
        int or None
            Column index if a winning move exists, None otherwise.
        """
        board = observation

        for col in valid_actions:
            row = self._get_next_row(board, col)
            if row is None:
                continue  # column is full, skip

            board_copy = board.copy()
            board_copy[row, col, channel] = 1

            if self._check_win_from_position(board_copy, row, col, channel):
                return col

        return None


# Petit test manuel (comme dans random_agent.py)
if __name__ == "__main__":
    env = connect_four_v3.env(render_mode="human")
    env.reset(seed=0)

    agents = {
        name: SmartAgent(env, player_name=name)
        for name in env.possible_agents
    }

    for agent_name in env.agent_iter():
        obs, reward, term, trunc, info = env.last()

        if term or trunc:
            if reward == 1:
                print(f"{agent_name} wins!")
            elif reward == -1:
                print("The other player wins!")
            else:
                print("It's a draw!")
            break
        else:
            obs_array = obs["observation"]
            mask = obs["action_mask"]

            action = agents[agent_name].choose_action(
                observation=obs_array,
                reward=reward,
                terminated=term,
                truncated=trunc,
                info=info,
                action_mask=mask,
            )
            print(f"{agent_name} plays column {action}")

        env.step(action)

    input("Press Enter to close...")
    env.close()
