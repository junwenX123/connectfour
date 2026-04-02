import time
import random

class Agent:
    def __init__(self, env, player_name=None):
        self.env = env
        self.time_limit = 0.95  
        self.transposition_table = {}
        self.start_time = 0
        self.column_order = [3, 2, 4, 1, 5, 0, 6]

        
    def choose_action(self, observation, reward=0.0, terminated=False, truncated=False, info=None, action_mask=None):
        """
        Interface principale pour l'environnement.
        Convertit l'observation en bitboard et lance la recherche.
        """
        self.start_time = time.time()
        self.transposition_table = {} 
        
        # 1. Conversion de l'observation en Bitboards
        # channel 0 = current player, channel 1 = opponent
        obs = observation['observation'] if isinstance(observation, dict) else observation
        mask = action_mask if action_mask is not None else [1]*7
        
        position, mask_board = self._numpy_to_bitboard(obs)
        
        # 2. Recherche avec approfondissement itératif (Iterative Deepening)
        best_move = self._iterative_deepening(position, mask_board, mask)
        
        return best_move

    def _numpy_to_bitboard(self, obs):
        """
        Convertit la grille numpy (6, 7, 2) en deux entiers
        Connect 4 est 7 colonnes x 6 lignes.
        Bitboard format: 
        .  .  .  .  .  .  .
        5 12 19 26 33 40 47
        4 11 18 25 32 39 46
        ...
        0  7 14 21 28 35 42
        
        position: bitboard du joueur actuel
        mask_board: bitboard des deux joueurs combinés (cases occupées)
        """
        position = 0
        mask_board = 0
        
        # obs[:, :, 0] est le joueur actuel (1 là où il a des pièces)
        # obs[:, :, 1] est l'adversaire
        current_player_grid = obs[:, :, 0]
        opponent_grid = obs[:, :, 1]
       
        for col in range(7):
            for row in range(6):
                # Calcul de l'index dans le bitboard
                # Chaque colonne prend 7 bits (6 cases + 1 buffer pour éviter les débordements horizontaux)
                bit_index = col * 7 + row
                
                if current_player_grid[5-row][col] == 1: 
                    position |= (1 << bit_index)
                    mask_board |= (1 << bit_index)
                elif opponent_grid[5-row][col] == 1:
                    mask_board |= (1 << bit_index)
                    
        return position, mask_board

    def _iterative_deepening(self, position, mask, valid_actions_mask):
        """Recherche le meilleur coup en augmentant la profondeur tant qu'il reste du temps."""
        best_move = None
        
        # Obtenir les coups valides à partir du masque
        valid_moves = [i for i, v in enumerate(valid_actions_mask) if v == 1]
        if not valid_moves:
            return 0
        if len(valid_moves) == 1:
            return valid_moves[0]
            
        # Trier les coups pour prioriser le centre dès le début
        valid_moves.sort(key=lambda x: abs(x - 3))
        best_move = valid_moves[0]
        
        # Profondeur maximale théorique (42 cases)
        max_depth = 42 
        
        for depth in range(1, max_depth + 1):
            if time.time() - self.start_time > self.time_limit:
                break
                
            try:
                # Appel à Negamax
                # Scores attendus entre -42 et 42 (victoire rapide = score haut)
                score, move = self._negamax(position, mask, depth, -float('inf'), float('inf'), valid_moves)
                
                # Si on trouve une victoire forcée, on arrête et on joue
                if score >= 40: # Victoire quasi certaine
                    return move
                    
                if move is not None:
                    best_move = move
                    
            except TimeoutError:
                break
                
        return best_move

    def _negamax(self, position, mask, depth, alpha, beta, valid_moves):
        """
        Algorithme Negamax avec élagage Alpha-Beta et Bitboards.
        Retourne (score, meilleur_coup)
        """
        # Vérification du temps
        if self._check_timeout():
            raise TimeoutError()
            
        state_key = (position, mask)
        
        
        # Vérifier si le joueur précédent a gagné (nous sommes dans la vue de l'adversaire ici)
        # Si l'adversaire a joué et a formé un alignement, c'est une défaite pour nous.
        # On doit vérifier si le coup PRECEDENT a créé une victoire.
        
        # Vérification victoire immédiate (si nous venons de jouer un coup gagnant avant l'appel récursif, 
        
        if depth == 0:
            return self._evaluate_heuristic(position, mask), None

        # Générer les coups possibles via bitboards
        # Un coup est possible si la colonne n'est pas pleine (top bit non set)
        possible_moves = []
        for col in self.column_order:
            if col in valid_moves:
                # Vérifier si colonne jouable bitboard 
                # Mask indique les cases occupées. 
                # Si (mask & (1 << (col*7 + 5))) == 0, alors il y a de la place.
                if (mask & (1 << (col * 7 + 5))) == 0:
                    possible_moves.append(col)
        
        if not possible_moves:
            return 0, None # Match nul ou pas de coups

        best_score = -float('inf')
        best_move = possible_moves[0]
        
        for col in possible_moves:
            # Faire le coup
            # Le coup consiste à ajouter un bit au-dessus du top bit actuel de la colonne dans le mask
            # mask | (mask + (1 << (col*7)))
            # (mask + (1 << (col*7))) & partie_colonne
            move_bit = (mask + (1 << (col * 7))) & (0x3F << (col * 7)) 
            # (mask + bottom_mask(col)) & column_mask(col)
            
            height = 0
            for r in range(6):
                if (mask >> (col*7 + r)) & 1:
                    height += 1
                else:
                    break
            
            # Bit à jouer
            played_bit = 1 << (col*7 + height)
            
            new_position = position ^ played_bit # Le joueur actuel joue
            new_mask = mask | played_bit
            
            # Vérifier si ce coup gagne
            if self._check_win_bitboard(new_position):
                return 100 + depth, col # Victoire préférée si rapide (+depth)
            
            # Appel récursif
            # Negamax: score = -negamax(adversaire)
            # L'adversaire devient 'current', son bitboard est (new_mask ^ new_position)
            
            opponent_position = new_mask ^ new_position
            
            # Mettre à jour les coups valides pour l'enfant 
            # Pour l'instant on repasse la liste globale filtrée par la logique bitboard
            child_valid = valid_moves # Simplification
            
            score = -self._negamax(opponent_position, new_mask, depth - 1, -beta, -alpha, child_valid)[0]
            
            if score > best_score:
                best_score = score
                best_move = col
                
            alpha = max(alpha, score)
            if alpha >= beta:
                break # Élagage
                
        return best_score, best_move

    def _check_win_bitboard(self, pos):
        """Vérifie s'il y a 4 alignés dans le bitboard 'pos'."""
        # Horizontal
        m = pos & (pos >> 7)
        if m & (m >> 14): return True
        
        # Diagonal \
        m = pos & (pos >> 6)
        if m & (m >> 12): return True
        
        # Diagonal /
        m = pos & (pos >> 8)
        if m & (m >> 16): return True
        
        # Vertical
        m = pos & (pos >> 1)
        if m & (m >> 2): return True
        
        return False

    def _evaluate_heuristic(self, position, mask):
        """
        Évaluation heuristique simple pour les noeuds feuilles non terminaux.
        Compte les alignements partiels.
        """
        score = 0
        opponent = mask ^ position
        
        # Colonne centrale (3)
        center_col = (position >> (3*7)) & 0x3F
        score += bin(center_col).count('1') * 3
        
        # Un vrai solveur utiliserait ici des masques pour compter les 'menaces' (3 alignés + vide)
        return score

    def _check_timeout(self):

        return (time.time() - self.start_time) > self.time_limit

