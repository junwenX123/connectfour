"""
Monte Carlo Tree Search agent for Connect Four (PettingZoo connect_four_v3)
"""

import numpy as np
import random
import math
import time


class MCTSNode:
    """
    Node in the MCTS tree
    """

    def __init__(self, board, player, parent=None, move=None):
        self.board = board        # numpy array (6, 7, 2)
        self.player = player      # whose turn at this node: 0 or 1
        self.parent = parent      # parent node
        self.move = move          # move (column) that led to this node from parent
        self.children = []        # list of child nodes
        self.visits = 0
        self.wins = 0.0           # total wins from perspective of player 0

    # ----- Node helpers -----

    def is_fully_expanded(self):
        """Check if all children for all valid moves have been added."""
        valid_moves = self._get_valid_moves()
        return len(self.children) == len(valid_moves)

    def best_child(self, c=1.41):
        """
        Select best child using UCB1:
            (wins/visits) + c * sqrt( ln(parent.visits) / child.visits )
        """
        best_value = -float("inf")
        best_nodes = []

        for child in self.children:
            if child.visits == 0:
                ucb = float("inf")
            else:
                exploit = child.wins / child.visits
                explore = c * math.sqrt(math.log(self.visits) / child.visits)
                ucb = exploit + explore

            if ucb > best_value:
                best_value = ucb
                best_nodes = [child]
            elif ucb == best_value:
                best_nodes.append(child)

        return random.choice(best_nodes)

    def _get_valid_moves(self):
        """Get valid column indices from this state's board."""
        board = self.board
        top_occupancy = board[0, :, 0] + board[0, :, 1]
        return [c for c in range(7) if top_occupancy[c] == 0]


class MCTSAgent:
    """
    Agent using Monte Carlo Tree Search
    """

    def __init__(self, env, time_limit=0.95, player_name=None):
        """
        env: PettingZoo environment
        time_limit: time budget per move in seconds
        """
        self.env = env
        self.time_limit = time_limit
        self.player_name = player_name or "MCTS"

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
        Choose action using MCTS.

        observation: numpy array (6, 7, 2) from current player's perspective.
        """
        root = MCTSNode(board=observation.copy(), player=0, parent=None, move=None)

        start_time = time.time()
        simulations = 0

        while time.time() - start_time < self.time_limit:
            # 1) selection
            node = self._select(root)

            # 2) expansion
            if not self._is_terminal(node):
                node = self._expand(node)

            # 3) simulation
            result = self._simulate(node)

            # 4) backpropagation
            self._backpropagate(node, result)

            simulations += 1

        # After time limit: choose child with best empirical win rate (no exploration)
        if not root.children:
            # no move? choose random legal one based on mask
            valid_actions = [i for i, valid in enumerate(action_mask) if valid == 1]
            return random.choice(valid_actions)

        best_child = max(root.children, key=lambda c: c.wins / c.visits if c.visits > 0 else -1)
        return best_child.move

    # ---------- Core MCTS steps ----------

    def _select(self, node):
        """
        Select a node to expand: go down while node is fully expanded and non-terminal.
        """
        while node.children and node.is_fully_expanded() and not self._is_terminal(node):
            node = node.best_child(c=1.41)
        return node

    def _expand(self, node):
        """
        Expand one untried move from the given node and return the new child.
        """
        valid_moves = node._get_valid_moves()
        tried_moves = [child.move for child in node.children]
        untried_moves = [m for m in valid_moves if m not in tried_moves]

        if not untried_moves:
            # nothing to expand
            return node

        move = random.choice(untried_moves)
        new_board = self._simulate_move(node.board, move, node.player)
        next_player = 1 - node.player

        child = MCTSNode(board=new_board, player=next_player, parent=node, move=move)
        node.children.append(child)
        return child

    def _simulate(self, node):
        """
        Play random game from node until terminal state.

        return:
            result in {1, 0, 0.5} from the view of player 0.
        """
        board = node.board.copy()
        current_player = node.player

        while True:
            # terminal check
            if self._check_win(board, 0):
                winner = 0
                break
            if self._check_win(board, 1):
                winner = 1
                break

            valid_moves = self._get_valid_moves_board(board)
            if not valid_moves:
                winner = None  # draw
                break

            # random move
            move = random.choice(valid_moves)
            board = self._simulate_move(board, move, current_player)
            current_player = 1 - current_player

        # Result: from perspective of player 0
        if winner is None:
            return 0.5
        elif winner == 0:
            return 1.0
        else:
            return 0.0

    def _backpropagate(self, node, result):
        """
        Propagate simulation result back to root.

        result: win probability for player 0 ∈ {0, 0.5, 1}
        """
        current = node
        while current is not None:
            current.visits += 1

            # wins are always stored from the perspective of player 0
            current.wins += result

            current = current.parent

    def _is_terminal(self, node):
        """Check whether node is terminal."""
        board = node.board
        if self._check_win(board, 0) or self._check_win(board, 1):
            return True
        if not node._get_valid_moves():
            return True
        return False

    # ---------- Board helpers ----------

    def _simulate_move(self, board, col, player):
        """Place a piece for 'player' (0 or 1) in column 'col' on a copy of board."""
        new_board = board.copy()
        for row in range(5, -1, -1):
            if new_board[row, col, 0] == 0 and new_board[row, col, 1] == 0:
                new_board[row, col, player] = 1
                return new_board
        return new_board  # column full -> unchanged

    def _get_valid_moves_board(self, board):
        top_occupancy = board[0, :, 0] + board[0, :, 1]
        return [c for c in range(7) if top_occupancy[c] == 0]

    def _check_win(self, board, channel):
        """Same win check as in MinimaxAgent."""

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
