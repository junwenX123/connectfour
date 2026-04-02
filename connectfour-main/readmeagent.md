## Agent IA pour Connect Four

Il s’agit d’un agent IA pour Connect Four, qui utilise une **représentation bitboard** du plateau + **recherche par approfondissement itératif** + **Negamax + élagage α-β** + une **évaluation heuristique simple** pour choisir un coup. Il est conçu pour des environnements utilisant une **grille d’observation numpy** .

### Caractéristiques

- Convertit le plateau au format numpy en bitboard pour une recherche efficace.  
- Utilise l’Approfondissement Itératif (Iterative Deepening) avec une limite de temps (par défaut ≈ 0,95 s), ce qui convient aux contraintes de temps d’un tournoi / d’une plateforme.  
- Emploie Negamax + élagage α-β pour la recherche en profondeur, et applique une évaluation heuristique au niveau des feuilles.  
- Inclut des optimisations comme la priorité aux colonnes centrales, un ordre préféré des colonnes à explorer, etc.
