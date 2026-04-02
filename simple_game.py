# -*- coding: utf-8 -*-
"""
Created on Wed Nov 26 23:47:04 2025

@author: User
"""

# simple_game.py
from pettingzoo.classic import connect_four_v3
import numpy as np


def simple_game():
    env = connect_four_v3.env(render_mode="human")  # ou "rgb_array" ou None
    env.reset(seed=42)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            # Partie terminée : on annonce le résultat
            if reward == 1:
                print(f"{agent} wins!")
            elif reward == -1:
                print("The other player wins!")  # suivant la convention de reward
            else:
                print("It's a draw!")
            break

        else:
            # TODO: Take a random valid action
            mask = observation["action_mask"]          # 0/1 pour chaque colonne
            action = env.action_space(agent).sample(mask=mask)
            print(f"{agent} plays column {action}")

        env.step(action)

    input("Press Enter to close...")
    env.close()


if __name__ == "__main__":
    simple_game()
