from __future__ import annotations

import random
from typing import Optional, Dict, Any, List

import numpy as np
from pettingzoo.classic import connect_four_v3


try:
    from loguru import logger
except ImportError:  
    logger = None


class RandomAgent:
    """
    A simple agent that plays randomly.
    """

    def __init__(self, env, player_name: Optional[str] = None):
        """
        Initialize the random agent.

        Parameters
        ----------
        env : PettingZoo environment
            The shared Connect Four environment.
        player_name : str, optional
            Optional display name for the agent.
        """
        self.env = env
        self.player_name = player_name or "RandomAgent"

        any_agent = env.possible_agents[0]
        self.action_space = env.action_space(any_agent)

    def reset(self) -> None:
        """Reset internal state before a new game (stateless agent)."""
        return

    # ------------------------------------------------------------------
    # Tâche 2.2 
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
        Choose a random valid action using the action mask.

        Parameters
        ----------
        observation : np.ndarray
            Current board state (6, 7, 2).
        reward : float
            Reward from previous action.
        terminated : bool
            True if the game is over (win/lose).
        truncated : bool
            True if the game is stopped early (time limit, etc.).
        info : dict, optional
            Additional information from the environment.
        action_mask : np.ndarray, shape (7,)
            Binary vector where 1 = legal move, 0 = illegal (full column).:contentReference[oaicite:0]{index=0}

        Returns
        -------
        int or None
            Column index (0–6) to play, or None if no action is possible.
        """
        if terminated or truncated:
            return None

        if action_mask is not None:
            action = int(self.action_space.sample(mask=action_mask))
        else:
            action = int(self.action_space.sample())

        if logger is not None:
            logger.debug(
                f"[{self.player_name}] choose_action -> column {action}, "
                f"mask={action_mask}"
            )

        return action

    # ------------------------------------------------------------------
    # Tâche 2.3 :
    # ------------------------------------------------------------------
    def choose_action_manual(
        self,
        observation: np.ndarray,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        info: Optional[Dict[str, Any]] = None,
        action_mask: Optional[np.ndarray] = None,
    ) -> Optional[int]:
        """
        Choose a random valid action *without* using action_space.sample(mask).

        This is only for learning what the action_mask does.
        """
        if terminated or truncated:
            return None

        if action_mask is None:
            return int(self.action_space.sample())

        valid_actions: List[int] = [i for i, m in enumerate(action_mask) if m == 1]

        if not valid_actions:
            if logger is not None:
                logger.warning(
                    f"[{self.player_name}] no valid actions although game not terminated"
                )
            return None

        action = int(random.choice(valid_actions))

        if logger is not None:
            logger.debug(
                f"[{self.player_name}] choose_action_manual -> column {action}, "
                f"valid={valid_actions}"
            )

        return action


# ----------------------------------------------------------------------
# Tâche 2.9 : 
# ----------------------------------------------------------------------
class WeightedRandomAgent(RandomAgent):
    """
    Random agent that prefers the center column (column 3).
    """

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
        Choose a random valid action but with higher probability for the center.
        """
        if terminated or truncated:
            return None

        if action_mask is None:
            return super().choose_action(
                observation, reward, terminated, truncated, info, action_mask
            )

        num_cols = len(action_mask)
        actions = list(range(num_cols))

        base_weights = [1.0] * num_cols
        center = num_cols // 2  # 7 -> 3
        base_weights[center] = 3.0

        filtered_actions: List[int] = []
        filtered_weights: List[float] = []
        for a, m, w in zip(actions, action_mask, base_weights):
            if m == 1:
                filtered_actions.append(a)
                filtered_weights.append(w)

        if not filtered_actions:
            return None

        chosen = random.choices(filtered_actions, weights=filtered_weights, k=1)[0]

        if logger is not None:
            logger.debug(
                f"[{self.player_name}] weighted choose_action -> column {chosen}, "
                f"actions={filtered_actions}, weights={filtered_weights}"
            )

        return int(chosen)


if __name__ == "__main__":
    env = connect_four_v3.env(render_mode="human")
    env.reset(seed=0)

    agents = {
        name: WeightedRandomAgent(env, player_name=name)
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
