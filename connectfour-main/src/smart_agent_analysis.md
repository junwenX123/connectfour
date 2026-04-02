# Analyse de l’agent intelligent (SmartAgent)

## 1. Taux de victoire : Intelligent vs Aléatoire (100 parties)

En lançant le tournoi `SmartAgent vs RandomAgent` (100 parties), on observe que :

- SmartAgent gagne **100** parties  
- RandomAgent gagne **0** parties  
- Matchs nuls : **0**  

En général, le taux de victoire de l’agent intelligent est nettement supérieur à 50 %, ce qui montre qu’il exploite effectivement des heuristiques plus fortes que le jeu aléatoire.

## 2. Efficacité de la stratégie

Les règles qui se déclenchent le plus souvent sont :

1. **Blocage de l’adversaire** – très fréquent car il y a souvent des menaces simples à un coup.
2. **Jeu au centre** – lorsque personne n’a de menace immédiate, l’agent préfère la colonne 3.
3. **Coup gagnant immédiat** – plus rare, mais décisif quand il se produit.

Cette hiérarchie de règles joue le rôle d’**évaluation heuristique** :  
au lieu d’explorer tout l’arbre de jeu comme dans un algorithme Minimax sur un game tree, l’agent applique directement des règles simples pour approximer une bonne décision.

## 3. Cas d’échec typiques

L’agent intelligent peut encore perdre dans plusieurs situations :

- Il ne regarde que **un coup à l’avance** : il ne voit pas certains sacrifices ou pièges à plus long terme.
- Il ne gère pas encore les **menaces doubles** complexes.
- Il n’a pas de vraie fonction d’évaluation globale de la position (il ne compte que la victoire immédiate, le blocage et le centre).

## 4. Idées d’amélioration

Quelques pistes pour le rendre plus fort :

1. **Détecter les menaces doubles** (double threat) :  
   jouer un coup qui crée deux possibilités de victoire au tour suivant.

2. **Évaluation plus fine du plateau** :  
   attribuer un score aux positions en comptant les lignes de 2 ou 3 pions alignés, pondérées par leur potentiel.

3. **Recherche limitée dans l’arbre de jeu** :  
   combiner cette fonction d’évaluation avec une recherche Minimax sur quelques niveaux de profondeur dans l’arbre des coups possibles.

4. **Symétrie / ouverture** :  
   préférer certaines structures d’ouverture (ex : contrôler le centre tôt dans la partie).

Ces améliorations rapprocheraient l’agent d’un véritable joueur “IA” utilisant une fonction d’évaluation et un parcours d’arbre de jeu.
