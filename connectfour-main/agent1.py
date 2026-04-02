import time
import random

class Agent:
    def __init__(self, env, player_name=None):
        self.env = env
        self.time_limit = 0.95
        self.transposition_table = {}
        self.start_time = 0.0
        self.column_order = [3, 2, 4, 1, 5, 0, 6]

        self.ROWS = 6
        self.COLS = 7
        self.H = 7  # 6 rows + 1 sentinel in bitboard layout

    def choose_action(self, observation, reward=0.0, terminated=False,
                      truncated=False, info=None, action_mask=None):
        """
        Interface principale.
        Observation PettingZoo connect_four_v3:
          dict { "observation": (6,7,2), "action_mask": (7,) }
        plane 0 = current observing agent, plane 1 = opponent.
        """
        self.start_time = time.time()

        self.transposition_table = {}

        if isinstance(observation, dict):
            obs = observation.get("observation", observation)
            mask_vec = observation.get("action_mask", None)
        else:
            obs = observation
            mask_vec = None

        if mask_vec is None:
            mask_vec = action_mask if action_mask is not None else [1] * self.COLS

        # 2) numpy -> bitboard
        position, mask_board = self._numpy_to_bitboard(obs)

        # 3) Iterative deepening + negamax
        best_move = self._iterative_deepening(position, mask_board, mask_vec)

        try:
            if hasattr(mask_vec, "__len__"):
                if mask_vec[best_move] != 1:
                    # fallback to any legal
                    legal = [i for i, v in enumerate(mask_vec) if v == 1]
                    return legal[0] if legal else 0
        except Exception:
            pass

        return best_move
        
    # ---------- Conversion ----------

    def _numpy_to_bitboard(self, obs):
        """
        obs shape (6,7,2)
        bit index = col*7 + row
        """
        position = 0
        mask_board = 0

        current_player_grid = obs[:, :, 0]
        opponent_grid = obs[:, :, 1]

        for col in range(self.COLS):
            for row in range(self.ROWS):
                bit_index = col * self.H + row

                r_np = 5 - row
                if current_player_grid[r_np][col] == 1:
                    position |= (1 << bit_index)
                    mask_board |= (1 << bit_index)
                elif opponent_grid[r_np][col] == 1:
                    mask_board |= (1 << bit_index)

        return position, mask_board

    # ---------- Root search ----------

    def _iterative_deepening(self, position, mask, valid_actions_mask):
        valid_moves = [i for i, v in enumerate(valid_actions_mask) if v == 1]

        if not valid_moves:
            return 0
        if len(valid_moves) == 1:
            return valid_moves[0]

        tac = self._tactical_override(position, mask, valid_moves)
        if tac is not None:
            return tac

        valid_moves.sort(key=lambda x: abs(x - 3))
        best_move = valid_moves[0]

        max_depth = 42

        for depth in range(1, max_depth + 1):
            if self._time_up():
                break
            try:
                score, move = self._negamax(position, mask, depth,
                                            -10**9, 10**9, valid_moves)
                if move is not None:
                    best_move = move

                if score >= 10**6:
                    return best_move

            except TimeoutError:
                break

        return best_move


    def _tactical_override(self, position, mask, valid_moves):
        for col in valid_moves:
            b = self._first_empty_bit(mask, col)
            if b is None:
                continue
            if self._check_win_bitboard(position | b):
                return col

        opp = mask ^ position
        for col in valid_moves:
            b = self._first_empty_bit(mask, col)
            if b is None:
                continue
            if self._check_win_bitboard(opp | b):
                return col

        return None


    def _negamax(self, position, mask, depth, alpha, beta, valid_moves):
        if self._time_up():
            raise TimeoutError()

        # Transposition table
        key = (position, mask, depth)
        if key in self.transposition_table:
            return self.transposition_table[key]

        if depth == 0:
            val = self._evaluate_heuristic(position, mask)
            res = (val, None)
            self.transposition_table[key] = res
            return res

        possible_moves = []
        for col in self.column_order:
            if col in valid_moves:
                # top occupied
                if (mask & (1 << (col * self.H + (self.ROWS - 1)))) == 0:
                    possible_moves.append(col)

        if not possible_moves:
            res = (0, None)
            self.transposition_table[key] = res
            return res

        best_score = -10**9
        best_move = possible_moves[0]

        for col in possible_moves:
            played_bit = self._first_empty_bit(mask, col)
            if played_bit is None:
                continue

            new_mask = mask | played_bit
            new_position = position | played_bit

            if self._check_win_bitboard(new_position):
                res = (10**7 + depth, col)
                self.transposition_table[key] = res
                return res

            opponent_position = new_mask ^ new_position

            score, _ = self._negamax(opponent_position, new_mask, depth - 1,
                                     -beta, -alpha, valid_moves)
            score = -score

            if score > best_score:
                best_score = score
                best_move = col

            alpha = max(alpha, score)
            if alpha >= beta:
                break

        res = (best_score, best_move)
        self.transposition_table[key] = res
        return res


    def _first_empty_bit(self, mask, col):
        height = 0
        base = col * self.H
        for r in range(self.ROWS):
            if (mask >> (base + r)) & 1:
                height += 1
            else:
                break
        if height >= self.ROWS:
            return None
        return 1 << (base + height)

    def _check_win_bitboard(self, pos):
        # Horizontal
        m = pos & (pos >> 7)
        if m & (m >> 14):
            return True

        # Diagonal \
        m = pos & (pos >> 6)
        if m & (m >> 12):
            return True

        # Diagonal /
        m = pos & (pos >> 8)
        if m & (m >> 16):
            return True

        # Vertical
        m = pos & (pos >> 1)
        if m & (m >> 2):
            return True

        return False


    def _evaluate_heuristic(self, position, mask):
        
        opp = mask ^ position
        grid = self._bitboards_to_grid(position, opp)

        score = 0

        center_col = 3
        center_count = 0
        for r in range(self.ROWS):
            if grid[r][center_col] == 1:
                center_count += 1
        score += center_count * 3

        score += self._score_windows(grid)

        return score

    def _bitboards_to_grid(self, me, opp):
        """
         grid[row][col]
      
        cell: 1 me, -1 opp, 0 empty
        """
        grid = [[0 for _ in range(self.COLS)] for _ in range(self.ROWS)]
        for col in range(self.COLS):
            base = col * self.H
            for row in range(self.ROWS):
                bit = 1 << (base + row)
                if me & bit:
                    grid[row][col] = 1
                elif opp & bit:
                    grid[row][col] = -1
        return grid

    def _score_window(self, window):
        me = window.count(1)
        opp = window.count(-1)
        empty = window.count(0)

        if me == 4:
            return 100000
        if me == 3 and empty == 1:
            return 50
        if me == 2 and empty == 2:
            return 10

        if opp == 4:
            return -100000
        if opp == 3 and empty == 1:
            return -80
        if opp == 2 and empty == 2:
            return -10

        return 0

    def _score_windows(self, grid):
        s = 0

        # Horizontal
        for r in range(self.ROWS):
            for c in range(self.COLS - 3):
                window = [grid[r][c+i] for i in range(4)]
                s += self._score_window(window)

        # Vertical
        for c in range(self.COLS):
            for r in range(self.ROWS - 3):
                window = [grid[r+i][c] for i in range(4)]
                s += self._score_window(window)

        # Diagonal /
        for r in range(self.ROWS - 3):
            for c in range(self.COLS - 3):
                window = [grid[r+i][c+i] for i in range(4)]
                s += self._score_window(window)

        # Diagonal \
        for r in range(3, self.ROWS):
            for c in range(self.COLS - 3):
                window = [grid[r-i][c+i] for i in range(4)]
                s += self._score_window(window)

        return s

    # ---------- Time ----------

    def _time_up(self):
        return (time.time() - self.start_time) > self.time_limit


