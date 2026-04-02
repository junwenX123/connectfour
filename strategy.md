# Tâche 3.1 : Conception de la stratégie (Puissance 4)

## 1. Classement des priorités

Ordre des décisions que l'agent doit suivre à chaque coup :

1. **Si je peux gagner, jouer le coup gagnant.**  
   - Chercher un coup qui crée alignement de 4 pour moi (canal 0).

2. **Sinon, si l’adversaire peut gagner au prochain coup, le bloquer.**  
   - Chercher un coup qui empêche l’alignement de 4 de l’adversaire (canal 1).

3. **Sinon, préférer la colonne centrale.**  
   - Le centre donne souvent plus de possibilités d’alignements dans un arbre de jeu

4. **Sinon, jouer un coup valide aléatoire.**  
   - Pour ne pas être complètement prévisible quand aucune règle “forte” ne s’applique.

5. 
   **Créer des menaces doubles** (double threat) : un coup qui crée deux possibilités de victoire au coup suivant, ce qui est théoriquement imbattable car l’adversaire ne peut bloquer qu’une seule ligne.

---

## 2. Règles essentielles

Règles que l’agent doit absolument respecter :

1. **Règle Gagnante** :  
   > S’il existe un coup qui me fait gagner immédiatement, je le joue.

2. **Règle de Blocage** :  
   > S’il existe un coup qui empêche l’adversaire de gagner tout de suite, je le joue.

3. **Validité des coups** :  
   > Ne jamais jouer dans une colonne pleine (utiliser l’`action_mask` pour filtrer les actions valides).

---

## 3. Règles souhaitables (améliorations stratégiques)

Idées de règles supplémentaires :

1. **Préférer le centre** :  
   > Si la colonne centrale (3) est disponible, la privilégier.

2. **Préférer les coups qui créent plusieurs alignements potentiels** :  
   > Par exemple, un coup qui rallonge plusieurs lignes de 2 ou 3 pions.

3. **Créer des menaces doubles** :  
   > Jouer un coup qui donne au prochain tour au moins deux coups gagnants différents.

Ces règles supplémentaires peuvent être ajoutées au fur et à mesure sans casser la structure de base de l’algorithme.






