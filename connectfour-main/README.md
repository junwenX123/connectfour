# projet/
        ├── agent.py         Agent ML-Arena  
        └── src/             Code source  
                ├── random_agent.py  
                ├── smart_agent.py  
                ├── minimax_agent.py  
                └── mcts_agent.py  
        └── tests/          Tests  
        └── exos/ 
        


# Partie 1 – Règles du Puissance 4

## Tâche 1.1 – Analyse des règles du jeu


### 1. Dimensions du plateau

Un plateau standard de Puissance 4 contient **6 lignes et 7 colonnes** (6×7).

### 2. Comment un joueur gagne-t-il la partie ?

Un joueur gagne lorsqu’il **aligne 4 jetons de sa couleur consécutifs** :

- **horizontalement**
- **verticalement**
- **en diagonale** (deux directions possibles)

### 3. Que se passe-t-il si le plateau est complètement rempli sans gagnant ?

Si toutes les colonnes sont pleines et qu’aucun joueur n’a aligné 4 jetons,  
la partie se termine par un **match nul (égalité)**.

### 4. Jusqu’où peut-on placer un pion dans une colonne qui est déjà pleine ?

On **ne peut pas** placer un pion dans une colonne déjà pleine：


### 5. Quels sont les résultats possibles d’une partie ?

-  Victoire du joueur 1  
-  Victoire du joueur 2  
-  Match nul (égalité)
## Tâche 1.2 – Analyse des conditions de victoire

### 1. Les quatre motifs de victoire
Un joueur peut gagner de **4 façons**：

• Horizontale  
x x x x

• Verticale  
x  
x  
x  
x

• Diagonale descendante (\)  
x . . .  
. x . .  
. . x .  
. . . x

• Diagonale montante (/)  
. . . x  
. . x .  
. x . .  
x . . .


---

### 2. Pour une seule position jouée, combien de directions vérifier ?

- horizontale
- verticale
- diagonale \
- diagonale /

---

### 3. Pseudo-code d’algorithme pour vérifier l’alignement de 4 pions
pour chaque (dx, dy) dans directions :
    compteur = 1    # compte la case (row, col)

    # 1) aller dans le sens + (dx, dy)
    r = row + dx
    c = col + dy
    tant que (r, c) est dans le plateau ET board[r][c] == player :
        compteur += 1
        r += dx
        c += dy

    # 2) aller dans le sens - (dx, dy)
    r = row - dx
    c = col - dy
    tant que (r, c) est dans le plateau ET board[r][c] == player :
        compteur += 1
        r -= dx
        c -= dy

    si compteur >= 4 :
        retourner True

retourner False

## Tâche 2.1
## Réponses

- **Noms des deux agents**  
  Les deux agents sont nommés **`player_0`** et **`player_1`**. 

- **Que représente `action` ? Quel est son type ?**  
  `action` correspond à l’action choisie par l’agent actif : c’est l’indice de la colonne (dans le plateau à 7 colonnes) où l’agent décide de “laisser tomber” son jeton. 
  Son type est un entier (`int`), dans l’intervalle **0 à 6 inclus**, car l’espace d’actions est `Discrete(7)`. 

- **Que font `env.agent_iter()` et `env.step(action)` ?**  
  - `env.agent_iter()` crée un itérateur sur les agents, dans l’ordre de jeu (alterné), pour permettre à chaque agent de jouer à tour de rôle.   
  - `env.step(action)` applique l’action de l’agent courant : cela met à jour l’état du plateau en faisant “tomber” un jeton dans la colonne spécifiée, passe le tour au prochain agent, et gère la détection de victoire, de match nul ou de fin de partie.
- **Quelles informations sont retournées par `env.last()` ?**  
  Dans la boucle d’`agent_iter()`, `env.last()` retourne un tuple contenant notamment :  
  1. `observation` — l’observation actuelle de l’environnement pour l’agent courant. 
  2. `reward` — la récompense attribuée à l’agent (souvent 0 en cours de partie, +1 / –1 / 0 à la fin selon victoire, défaite ou match nul). 
  3. des indicateurs de fin : typiquement un booléen de terminaison (`termination` / “done”) selon si la partie est terminée. 
  4. `info` — un dictionnaire éventuellement vide avec des informations supplémentaires. 

- **Structure de l’observation retournée**  
  L’observation renvoyée est un **dictionnaire** avec au moins deux clés :  
  - `"observation"` : un tenseur de forme **(6, 7, 2)** — c’est une grille de 6 lignes × 7 colonnes, avec **2 “plans”** :  
    - un plan pour les jetons de l’agent courant,  
    - un plan pour les jetons de l’adversaire.  
    Dans chaque plan, 1 signifie qu’un jeton de l’agent concerné occupe la case, 0 sinon. 
  - `"action_mask"` : un vecteur binaire (type int ou int8) de longueur 7, indiquant pour chaque colonne si l’action correspondante (jouer dans cette colonne) est **légale (1)** ou **illégale (0)**. 

- **Qu’est-ce qu’un “action mask” et pourquoi est-il important ?**  
  - Un **action mask** (masque d’actions légales) est — dans cet environnement — un vecteur binaire qui encode, pour l’agent courant, quelles actions parmi les 7 possibles sont légales à ce moment (1 = légale, 0 = illégale). 
  - Il est important car il empêche l’agent de sélectionner des actions illégales (par exemple de “jouer” dans une colonne déjà pleine). Dans l’environnement, si un agent exécute une action illégale, la partie se termine immédiatement avec une pénalité (reward = –1 pour l’agent fautif). 
  - De plus, dans un cadre d’apprentissage par renforcement, l’`action_mask` permet de restreindre l’espace d’actions aux seules actions valides, ce qui évite à l’agent d’apprendre à éviter les pénalités plutôt qu’à optimiser des stratégies de victoire — et améliore donc l’efficacité de l’apprentissage. 
## Tâche 2.2
## Observation : forme, sens des dimensions, valeurs possibles

- **Quelle est la forme du tableau d’observation ?**  
  L’observation (champ `"observation"`) a la forme **(6, 7, 2)**. 

- **Que représente chaque dimension ?**  
  - Les **6** correspondent aux **lignes** du plateau. 
  - Les **7** correspondent aux **colonnes** du plateau.  
  - Le **3ᵉ** indice (de taille 2) correspond à deux “plans” (ou “channels”) :  
    1. Le **premier plan** encode les jetons de l’**agent courant**.
    2. Le **second plan** encode les jetons de l’**adversaire**. 

- **Quelles sont les valeurs possibles dans le tableau d’observation ?**  
  Dans chaque cellule de ces plans, la valeur est soit **0** soit **1** (dtype int / int8). 
  - `1` signifie qu’un jeton de l’agent concerné occupe cette case. 
  - `0` signifie que l’agent concerné **n’a pas** de jeton dans cette case — c’est-à-dire que soit la case est vide, soit l’adversaire y a un jeton
 
## Partie 3 – Décomposition de l’agent (squelette de réflexion)

###  3.1 : Décomposer l’implémentation de l’agent  

Un agent doit choisir quelle colonne jouer. On peut décomposer cela en sous-tâches :

1. **Analyse des entrées** — Quelles informations l’agent reçoit-il ?  
   - L’agent reçoit l’**observation** de l’état du jeu, typiquement un tableau (grid) représentant la position des jetons (le plateau) + un “action_mask” (masque des actions légales).  
   - Il peut également recevoir d’autres informations utiles (par exemple : le nom de l’agent courant, l’historique du jeu, le nombre de pions restants, éventuellement un “info” dict retourné par l’environnement).  
   - Sur cette base, l’agent doit “comprendre” quelles colonnes sont jouables, quelle est la configuration actuelle, etc.

2. **Détection des coups valides** — Comment déterminer quelles colonnes sont “jouables” (actions légales) ?  
   - Utiliser l’`action_mask` : c’est un vecteur binaire (taille = nombre de colonnes, ici 7) indiquant pour chaque colonne si jouer là est légal (1) ou non (0).  
   - Alternativement — si on ne fait pas confiance à `action_mask` — on peut examiner directement le plateau : vérifier pour chaque colonne si la case “haut” (ou la plus haute libre) n’est pas déjà pleine, ce qui rend la colonne invalide.  

3. **Sélection du coup** — Quel algorithme utiliser pour choisir un coup parmi les coups valides ?  
   - Selon le niveau de sophistication de l’agent :  
     - Agent très simple : choisir **au hasard** parmi les colonnes valides.  
     - Agent un peu plus malin : éviter les coups invalides, peut-être heuristique “éviter les pertes immédiates”.  
     - Agent intermédiaire : chercher des opportunités immédiates de gagner (aligner 4), ou bloquer l’adversaire.  
     - Agent plus avancé : stratégie défensive, positionnement stratégique (contrôle du centre, préparation de menaces), simulation (minimax, recherche adversariale), etc.

4. **Sortie** — Que doit retourner l’agent ?  
   - L’agent retourne un entier `action`, l’indice de la colonne choisie (entre 0 et 6).  
   - L’action doit être valide (colonne jouable). Idéalement, l’agent s’appuie sur l’`action_mask` pour s’assurer de la validité.  

---

###  3.2 : Progression des algorithmes (stratégies) — du plus simple au plus complexe  

Voici un exemple de **hiérarchie de niveaux** pour les stratégies d’agents, **Niveau 0 → Niveau 5+** :

| Niveau | Stratégie / Comportement attendu |
|--------|----------------------------------|
| **Niveau 0** | Agent aléatoire : choisit au hasard une colonne parmi celles valides. |
| **Niveau 1** | Agent naïf-minimale : évite systématiquement les coups invalides (utilise action_mask). |
| **Niveau 2** | Agent “opportuniste” : cherche des opportunités immédiates — par exemple, jouer là où il peut gagner tout de suite, ou empêcher que l’adversaire gagne au prochain coup. |
| **Niveau 3** | Agent défensif / blocage : priorise les coups qui bloquent l’adversaire, ou qui minimisent le risque. |
| **Niveau 4** | Agent “positionnel / stratégique” : pense plusieurs coups à l’avance, cherche à contrôler le centre, prépare des menaces, maximise les options futures. |
| **Niveau 5+** | Agent avancé (“expert”) : utilisation d’algorithmes adversariaux ou d’apprentissage — par exemple minimax, alpha-beta, Monte Carlo, heuristiques, peut-être apprentissage par renforcement, évaluation de l’état, etc. |

L’objectif est d’avoir une **progression de “aléatoire” → “expert”**, ce qui permet d’implémenter plusieurs agents de complexité croissante, et de tester / comparer leurs performances.  

---

###  3.3 : Squelette d’une classe `Agent` (interface + attributs / méthodes)  


```python
class Agent:
    def __init__(self, name: str):
        self.name = name  # par ex. "player_0" ou "player_1"
        # éventuellement d'autres attributs (historique, mémoire, paramètres stratégiques…)

    def select_action(self,
                      observation: dict,
                      valid_actions: list[int] = None) -> int:
        """
        Choisit une action (colonne) à jouer, à partir de l'observation actuelle.
        - observation : dict retourné par l'environnement, contient 'observation' (plateau) et 'action_mask'
        - valid_actions (optionnel) : liste des colonnes valides ; si None, on peut l’inférer via observation['action_mask']
        Retour : un entier (colonne entre 0 et 6), action choisie.
        """
        raise NotImplementedError("Méthode à implémenter dans les sous-classes")

    # Méthode utilitaires possibles :
    # - parse_board() : interpréter l'état du plateau dans une forme plus pratique (ex: matrice, liste de colonnes, etc.)
    # - get_valid_actions() : retourne la liste des colonnes valides (aide pour select_action)
    # - maybe evaluation / scoring functions, mémorisation, historique des coups, etc.

    def get_valid_actions(self, action_mask) -> list[int]:
        return [i for i, valid in enumerate(action_mask) if valid]

    # (D’autres méthodes selon complexité / stratégie)












