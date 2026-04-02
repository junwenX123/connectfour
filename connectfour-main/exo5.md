## Tâche 5.1  Minimax / Alpha-Beta

- On alterne **max / min** parce que le jeu est à deux joueurs adversaires :  
  - **MAX** = nous ➝ on cherche à maximiser notre gain.  
  - **MIN** = adversaire ➝ il cherche à minimiser notre gain (ou maximiser le sien).  
- `depth` contrôle la profondeur d’anticipation (nombre de coups regardés à l’avance).  
- Si la profondeur est trop grande → l’arbre de jeu explose → temps de calcul + mémoire deviennent énormes (impraticable).  
- L’**élagage alpha-bêta** permet de sauter les branches qui ne peuvent pas améliorer la meilleure solution connue → réduit l’espace de recherche sans changer le résultat.  

---
## Tâche 5.4 === Tournament between agents ===

--- Matchup SmartAgent (A) vs MinimaxAgent (B) ---
SmartAgent wins : 0 (0.0%)
MinimaxAgent wins : 10 (100.0%)
Draws       : 0
Average moves per game: 18.00

## Tâche 5.6 MCTS (Monte Carlo Tree Search) / UCT / UCB1

- UCB1 / UCT équilibre **exploitation** (choisir les coups déjà bons) vs **exploration** (tester les coups peu explorés).  
- La constante **C** dans UCB1 fixe l’équilibre :  
  - Grand **C** → plus d’exploration,  
  - Petit **C** → plus d’exploitation.  
- On utilise des **simulations aléatoires (rollouts)** pour évaluer une position sans heuristique spécialisée : utile quand l’arbre est vaste ou complexe.  
- Plus on donne de temps / de simulations, meilleure est la décision — MCTS est un algorithme **anytime**.  

## Tâche 5.8
=== Tournament between agents ===

--- Matchup SmartAgent (A) vs MinimaxAgent (B) ---
SmartAgent wins : 0 (0.0%)
MinimaxAgent wins : 10 (100.0%)
Draws       : 0
Average moves per game: 18.00

--- Matchup SmartAgent (A) vs MCTSAgent (B) ---
SmartAgent wins : 5 (50.0%)
MCTSAgent wins : 5 (50.0%)
Draws       : 0
Average moves per game: 14.30

--- Matchup MinimaxAgent (A) vs MCTSAgent (B) ---
MinimaxAgent wins : 10 (100.0%)
MCTSAgent wins : 0 (0.0%)
Draws       : 0
Average moves per game: 14.00

## Tâche 5.9 Expérimentations MCTS

### Exploration constant  
- `Fixed time_limit = 0.95`, rollout = random.  
- Tested **c** ∈ {0.5, 1.0, 1.41, 2.0}.  
- Observed best trade-off around **c = 2.0**.

### Time budget  
- Fixed **c = 2.0**, rollout = random.  
- Tested `time_limit` ∈ {0.2, 0.5, 0.95, 1.5}.  
- Win rate increases with time, with diminishing returns.

### Intelligent rollouts  
- Compared random vs heuristic policy.  
- Heuristic prioritizes immediate wins, blocks opponent threats, prefers center.

### Early termination  
- Introduced `rollout_depth` and optional heuristic evaluation at cutoff for speed.
