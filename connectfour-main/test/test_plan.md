# Plan de test pour les agents Connect Four

## 1.1 Que tester ?

Nous distinguons trois grandes catégories de tests.

### A. Tests fonctionnels

- **Sélection de coup valide**
  - L’agent ne doit jamais jouer dans une colonne pleine.
  - Le coup renvoyé doit toujours appartenir au `action_mask` (cases légales).
- **Respect du masque d’action**
  - Si une colonne est marquée 0 dans `action_mask`, l’agent ne doit pas la choisir.
- **Gestion de la fin de partie**
  - Si `terminated` ou `truncated` est vrai, l’agent doit renvoyer `None` (ou ne pas jouer).
- **Application des règles stratégiques**
  - Priorité au coup gagnant immédiat.
  - Sinon, blocage des menaces adverses évidentes.
  - Sinon, préférence pour les colonnes centrales.
  - Sinon, coup aléatoire parmi les coups légaux.
- (Optionnel) **Détection de double menace**
  - Si possible, jouer un coup qui crée deux menaces de victoire au tour suivant.

### B. Tests de performance

- **Temps par coup**
  - Mesurer le temps moyen d’appel à `choose_action` sur un ensemble d’états de plateau.
- **Utilisation de la mémoire**
  - Vérifier que l’agent n’alloue pas une quantité de mémoire anormalement élevée pendant la décision.

### C. Tests stratégiques

- **Taux de victoire**
  - Comparer l’agent intelligent (`SmartAgent`) à un agent aléatoire (`RandomAgent`) sur un grand nombre de parties.
- **Qualité des décisions**
  - Vérifier que l’agent joue :
    - le coup gagnant lorsqu’il existe,
    - le coup de blocage lorsqu’une victoire adverse est possible au prochain tour,
    - une colonne centrale en l’absence de menaces ou de coups gagnants.
- **Comportement contre différents styles**
  - Jouer contre plusieurs types d’agents (aléatoire, aléatoire pondéré, agent moins fort…).

---

## 1.2 Comment tester ?

### A. Tests fonctionnels

- Utiliser le module `unittest` de Python pour écrire des tests unitaires.  
- Construire des plateaux artificiels avec `numpy` (tableaux `(6, 7, 2)`).
- Fournir un `action_mask` explicite et vérifier :
  - que l’action retournée appartient à la liste des coups légaux,
  - que l’agent ne joue jamais dans une colonne pleine.
- Créer des scénarios simples où la stratégie attendue est unique :
  - trois pions alignés avec une case libre pour tester la victoire immédiate ;
  - trois pions adverses alignés pour tester le blocage ;
  - plateau sans menace pour tester la préférence du centre.

### B. Tests de performance

- Utiliser `time.time()` pour mesurer le temps de N appels à `choose_action`, puis diviser par N.
- Utiliser `tracemalloc` pour mesurer les allocations mémoire maximales pendant les tests.
- Exécuter les mesures plusieurs fois pour lisser les fluctuations.

### C. Tests stratégiques

- Écrire des **tests d’intégration** qui jouent des parties complètes dans l’environnement `connect_four_v3`.
- Réutiliser une fonction utilitaire du type `play_one_game_with_agents` pour faire jouer deux agents.
- Lancer un tournoi de N parties (par exemple 100) :
  - `SmartAgent` vs `RandomAgent`
  - (Optionnel) `SmartAgent` vs `WeightedRandomAgent`
- Calculer :
  - nombre de victoires / défaites / matchs nuls,
  - nombre moyen de coups par partie.

---

## 1.3 Critères de succès

### Critères fonctionnels

- 0 % de coups illégaux sur l’ensemble des tests unitaires.
- Tous les tests `unittest` doivent être **verts** (succès).

### Critères de performance

- Temps moyen par coup `< 0.01` seconde sur la machine de référence.
- Utilisation maximale de mémoire `< 10 Mo` pendant les tests (valeur indicative).

### Critères stratégiques

- Taux de victoire de `SmartAgent` > **80 %** contre `RandomAgent` sur au moins 100 parties.
- `SmartAgent` bloque systématiquement les menaces de victoire en 1 coup créées dans les scénarios de test.
- (Optionnel) Lorsque la détection de double menace est implémentée, l’agent doit choisir le coup créant la double menace dans les scénarios définis.

---

## 2. Cas de test détaillés (scénarios)

### Scénario 1 : Détecter une victoire immédiate **
```
État du plateau (vue simplifiée, `X` = agent courant, ligne du bas) :**
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
X X X . . . . 

Attendu : l’agent joue la **colonne 3** pour gagner immédiatement.

###  **
```




### Scénario 2 : Bloquer la victoire de l'adversaire **
```
État du plateau :** 
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
. . . . . . .
O O O . . . . 

Attendu : l’agent joue la **colonne 3** pour bloquer.
###  **
```

### Scénario 3 : Préférer le centre en l’absence de menace

Plateau vide, aucun pion joué, toutes les colonnes légales.

Attendu : l’agent joue la **colonne 3** (centre).

### Scénario 4 : Masque d’action partiel

Colonne 3 interdite (`action_mask[3] = 0`), toutes les autres colonnes libres, pas de menace.

Attendu : l’agent choisit une colonne **légale**, et **jamais** la colonne 3.

### Scénario 5 : Colonne pleine

Colonne 3 complètement remplie mais encore indiquée comme légale dans le masque (cas limite).

Attendu : l’agent **ne joue pas** la colonne 3 (il doit respecter l’état réel du plateau).

### Scénario 6 : Double menace

Construire un plateau où jouer dans une colonne donnée permet de créer deux possibilités de victoire au tour suivant.

Attendu : si aucun coup gagnant immédiat ou blocage urgent n’est disponible, l’agent choisit ce coup de **double menace**.


