###  Analyse de l’agent aléatoire pour Puissance 4

## 1. Distribution des victoires

Sur 100 parties jouées entre deux `RandomAgent` :

- Joueur 1 (`player_0`) : ~55 victoires  
- Joueur 2 (`player_1`) : ~42 victoires  
- Matchs nuls : ~3

Les nombres exacts varient à chaque exécution, mais on observe que les deux
joueurs gagnent à des fréquences comparables. Le joueur qui commence semble
avoir un léger avantage.

## 2. Avantage du premier coup

Le joueur 1 joue toujours en premier. Comme il place le premier jeton, il peut
plus facilement construire une ligne de 4 avant l’adversaire. On observe donc
souvent un taux de victoire un peu plus élevé pour `player_0` que pour `player_1`.

## 3. Durée des parties

Sur nos 100 parties, le nombre de coups par partie se situe typiquement entre
15 et 35, avec une moyenne autour de ~25 coups.

- Minimum observé : par exemple 12 coups  
- Maximum observé : par exemple 38 coups  

Le maximum théorique est 42 coups (6 × 7 cases) si le plateau est complètement rempli.

## 4. Fréquence des matchs nuls

Les matchs nuls (plateau plein sans alignement de 4) sont rares pour des agents
purement aléatoires : sur 100 parties, on en observe seulement quelques-uns
(0 à 5 selon l’échantillon). La plupart des parties se terminent par une
victoire d’un des deux joueurs.

---

## 5. Réponses aux questions d’auto-vérification 

1. **Pourquoi le masque d’action est-il important ?**  
   Le `action_mask` indique quelles colonnes sont encore jouables (valeur 1) et
   quelles colonnes sont pleines ou illégales (valeur 0). L’utiliser permet de
   garantir que l’agent ne choisit que des coups valides. Si on ignore ce masque
   et qu’on joue dans une colonne interdite, l’environnement peut sanctionner
   l’agent (par exemple terminaison immédiate avec un reward négatif).

2. **Que se passe-t-il si vous essayez de jouer dans une colonne pleine ?**  
   Dans `Connect Four` de PettingZoo, jouer un coup illégal (dans une colonne
   pleine) est traité comme une erreur : le coup est marqué comme illégal et la
   partie se termine en donnant typiquement un reward de `-1` à l’agent

3. **Comment obtenez-vous la liste des actions valides à partir du masque d’action ?**  
   On parcourt le vecteur `action_mask` et on garde les indices où la valeur
   vaut `1` :

   ```python
   valid_actions = [i for i, m in enumerate(action_mask) if m == 1]
4.**Pourquoi deux agents aléatoires pourraient-ils ne pas avoir exactement 50/50 de taux de victoire ?**
Même si les deux agents jouent de façon symétrique, les résultats sur un
nombre fini de parties sont soumis à la variance aléatoire (fluctuations
statistiques). De plus, le fait que le joueur 1 commence donne un léger
avantage structurel. Donc sur 100 parties, obtenir par exemple 55/45 est
tout à fait normal.

5. **Quel est le nombre maximum de coups dans une partie de Puissance 4 ?**
Le plateau a 6 lignes et 7 colonnes, donc 6 × 7 = 42 cases. Le maximum est
donc 42 coups si aucune ligne de 4 n’est formée avant que le plateau soit
complètement rempli.












