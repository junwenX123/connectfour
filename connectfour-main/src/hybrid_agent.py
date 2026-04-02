# hybrid_agent.py
"""
Hybrid agent: quick tactics with shallow Minimax, strategy with MCTS.
"""

from minimax_agent import MinimaxAgent
from mcts_agent import MCTSAgent


class HybridAgent:
    """
    Combine Minimax + MCTS 
    """

    def __init__(self, env, player_name: str | None = None):
        self.env = env
        self.minimax = MinimaxAgent(env, depth=2, player_name="Hybrid-Minimax")
        self.mcts = MCTSAgent(env, time_limit=0.7, player_name="Hybrid-MCTS")
        self.player_name = player_name or "Hybrid"

    def choose_action(
        self,
        observation,
        reward: float = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        info=None,
        action_mask=None,
    ) -> int:
      
        action = self.minimax.choose_action(
            observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            action_mask=action_mask,
        )

       
        from minimax_agent import MinimaxAgent 

        tmp_agent = MinimaxAgent(self.env, depth=1)
        board_after = tmp_agent._simulate_move(observation, action, channel=0)
        if tmp_agent._check_win(board_after, 0):
            return action

        return self.mcts.choose_action(
            observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
            action_mask=action_mask,
        )

