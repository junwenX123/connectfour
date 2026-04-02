from __future__ import annotations

"""
Suite de tests pour les agents Connect Four.

- Tests fonctionnels (respect du masque d'action, coups légaux)
- Tests de scénarios stratégiques (victoire immédiate, blocage, centre)
- Tests d'intégration (taux de victoire contre RandomAgent)
"""

import time
import unittest
from typing import Dict, Tuple, List, Optional
import tracemalloc        
import numpy as np
from pettingzoo.classic import connect_four_v3

from random_agent import RandomAgent
from smart_agent import SmartAgent


# ----------------------------------------------------------------------
# Outils communs : jouer une partie entre deux agents
# ----------------------------------------------------------------------
def play_one_game_with_agents(
    env,
    agents: Dict[str, object],
    verbose: bool = False,
) -> Tuple[Optional[str], int]:
    """
    Joue une partie complète avec un dict {agent_name: agent}.

    Retourne
    --------
    winner : str ou None
        "player_0", "player_1" ou None (match nul).
    n_moves : int
        Nombre de coups joués dans la partie.
    """
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

        # Tour normal : on demande une action à l'agent courant
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
# Helpers pour les tests unitaires de SmartAgent
# ----------------------------------------------------------------------
def _make_dummy_agent() -> SmartAgent:
    """Crée un SmartAgent avec un environnement Connect Four vide."""
    env = connect_four_v3.env()
    env.reset(seed=0)
    return SmartAgent(env, player_name="test_smart")


# ----------------------------------------------------------------------
# 1. Tests fonctionnels
# ----------------------------------------------------------------------
class TestSmartAgentFunctional(unittest.TestCase):
    """Tests unitaires des aspects purement fonctionnels de SmartAgent."""

    def test_get_valid_actions_respects_mask(self):
        agent = _make_dummy_agent()
        mask = np.array([1, 0, 1, 0, 1, 0, 1], dtype=int)
        valid = agent._get_valid_actions(mask)
        self.assertEqual(valid, [0, 2, 4, 6])

    def test_choose_action_respects_mask(self):
        """L'action choisie doit être dans la liste des actions valides."""
        agent = _make_dummy_agent()
        board = np.zeros((6, 7, 2), dtype=int)
        mask = np.array([0, 1, 0, 1, 0, 1, 0], dtype=int)

        action = agent.choose_action(
            observation=board,
            action_mask=mask,
        )
        self.assertIn(action, [1, 3, 5])

    def test_does_not_play_in_full_column(self):
        """Même si le masque est faux, l'agent ne doit pas jouer dans une colonne pleine."""
        agent = _make_dummy_agent()
        board = np.zeros((6, 7, 2), dtype=int)

        # Remplir complètement la colonne 3
        col = 3
        board[:, col, 0] = 1

        # Masque dit à tort que la colonne 3 est légale
        mask = np.ones(7, dtype=int)

        action = agent.choose_action(
            observation=board,
            action_mask=mask,
        )
        self.assertNotEqual(action, col)


# ----------------------------------------------------------------------
# 2. Tests de scénarios stratégiques
# ----------------------------------------------------------------------
class TestSmartAgentScenarios(unittest.TestCase):
    """Scénarios décrits dans le plan de test (victoire, blocage, centre, etc.)."""

    def test_immediate_win_on_bottom_row(self):
        """Scénario 1 : 3 pions alignés, l'agent doit compléter la ligne."""
        agent = _make_dummy_agent()
        board = np.zeros((6, 7, 2), dtype=int)

        row = 5
        board[row, 0, 0] = 1
        board[row, 1, 0] = 1
        board[row, 2, 0] = 1

        mask = np.ones(7, dtype=int)

        action = agent.choose_action(
            observation=board,
            action_mask=mask,
        )
        self.assertEqual(action, 3)

    def test_block_opponent_immediate_win(self):
        """Scénario 2 : 3 pions adverses, l'agent doit bloquer."""
        agent = _make_dummy_agent()
        board = np.zeros((6, 7, 2), dtype=int)

        row = 5
        board[row, 0, 1] = 1
        board[row, 1, 1] = 1
        board[row, 2, 1] = 1

        mask = np.ones(7, dtype=int)

        action = agent.choose_action(
            observation=board,
            action_mask=mask,
        )
        self.assertEqual(action, 3)

    def test_prefer_center_on_empty_board(self):
        """Scénario 3 : plateau vide -> jouer centre."""
        agent = _make_dummy_agent()
        board = np.zeros((6, 7, 2), dtype=int)
        mask = np.ones(7, dtype=int)

        action = agent.choose_action(
            observation=board,
            action_mask=mask,
        )
        self.assertEqual(action, 3)

    def test_mask_forbids_center(self):
        """Scénario 4 : centre interdit par le masque."""
        agent = _make_dummy_agent()
        board = np.zeros((6, 7, 2), dtype=int)
        mask = np.ones(7, dtype=int)
        mask[3] = 0  # centre interdit

        action = agent.choose_action(
            observation=board,
            action_mask=mask,
        )
        self.assertNotEqual(action, 3)
        self.assertIn(action, [0, 1, 2, 4, 5, 6])


# ----------------------------------------------------------------------
# 3. Tests de performance & d'intégration
# ----------------------------------------------------------------------
class TestSmartAgentPerformance(unittest.TestCase):
    """Tests simples de performance et de 'force' contre RandomAgent."""

    def test_average_decision_time(self):
        """Mesure grossière du temps moyen par décision."""
        agent = _make_dummy_agent()

        n_trials = 200
        mask = np.ones(7, dtype=int)

        t0 = time.time()
        for _ in range(n_trials):
            # plateau aléatoire très grossier
            board = np.zeros((6, 7, 2), dtype=int)
            action = agent.choose_action(
                observation=board,
                action_mask=mask,
            )
            self.assertIn(action, range(7))
        t1 = time.time()

        avg_time = (t1 - t0) / n_trials
        # Seuil arbitraire : 0.01 seconde
        self.assertLess(avg_time, 0.01)
    def test_memory_usage_with_tracemalloc(self):
        """
        Utilise tracemalloc pour mesurer le pic de mémoire pendant plusieurs décisions.

        Critère (d'après test_plan.md) : pic < 10 Mo (valeur indicative).
        """
        agent = _make_dummy_agent()
        mask = np.ones(7, dtype=int)

        n_trials = 200

        tracemalloc.start()

        for _ in range(n_trials):
            board = np.zeros((6, 7, 2), dtype=int)
            action = agent.choose_action(
                observation=board,
                action_mask=mask,
            )
            self.assertIn(action, range(7))

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mb = peak / (1024 * 1024)  

        self.assertLess(
            peak_mb,
            10.0,
            msg=f"Peak memory too high: {peak_mb:.2f} MB",
        )

    def test_smart_agent_stronger_than_random(self):
        """Test d'intégration : SmartAgent doit gagner > 80% contre RandomAgent."""
        num_games = 50  # valeur modérée pour garder le test rapide

        wins_smart = 0
        wins_random = 0

        for i in range(num_games):
            env = connect_four_v3.env()
            env.reset(seed=i)

            agents = {
                "player_0": SmartAgent(env, player_name="smart"),
                "player_1": RandomAgent(env, player_name="random"),
            }

            winner, _ = play_one_game_with_agents(env, agents, verbose=False)

            if winner == "player_0":
                wins_smart += 1
            elif winner == "player_1":
                wins_random += 1

            env.close()

        win_rate_smart = wins_smart / num_games
        # Critère de succès : au moins 0.8
        self.assertGreaterEqual(win_rate_smart, 0.8)


if __name__ == "__main__":
    unittest.main()
