# Cours 1 Machine-Learning

## Session 1: Le machine-learning en pratique

### Introduction :

*a ajouter*

### Classes de modèles

*a ajouter*

### Focus sur les arbres de décisions

*a ajouter*

**Slide 16**

C4.5 : arbre de décision standards
(Attention plus personne utilise les arbres de décisions à part en tuto : en vrai c'est trop cool car c'est très lisible )

Random Forest : on fait pleins d'arbres chacun vote pour une classe et on prend celui le plus voté --> le random : prendre un sous ensemble de variables par arbres

Gradient boosting (XGBoost, cat Boost) : basé sur du random forest mais il y a un petit twits pour l'améliorer --> quand tu construits le nouvel arbre on essaye de corriger les erreurs qu'on fait tjs (on fait un tirage aléatoire biaisés sur les points sur lesquels on fait les erreurs pour pouvoir les corriger)

--> Attention tjs comparer avec le modèle linéaire pour comparer et si c'ets lamême chose (rasoir d'Ockam)


### Focus sur les SVM
définition SVM : 

**Slide 17-18**

Il faut absolument construire le noyau (gaussien, linéaire...) c'est nous qui faisons, on laisse pas la machine faire tte seule.


**Slide 22**

Classifieur le plus connu : régression logistique c'ets un classifieur binaire comme le SVM --> il faut passer de 2 classes à un mulktiples de classesv(pouvoir appliquer dans la vraie vie tt ça)

--> Seul contre tous : classe verte vs tt le monde (mettre en positif) / classe bleue vs tt le monde (mettre en positif) / classe rouge vs tt le monde (mettre en positif)

On aura donc 3 fonctions et quand on arrivera avec un nouveau point ya plusieurs cas de figure :

- positif pour 1 et négatif pour les 2 (super ! :) )
- négatif pour tt le monde (rejet en distance on ets trop éloigné) : souvent : on prend quand même le plus élevé (pas terrible mais déjà ça) sinon on le met nulle part et on se dit qu'un expert le fera à notre place (en plus ça permet de un peu augmenter notre score du coup c'est cool)
- 2 ou plus positif (rejet en ambiguité entre 2 classes) : solution fixer des seuils (compliqué)


### Focus sur les SVM

**Slide 26**

il faut bien sélectionner les données d'entrainements et les données d'évaluation. Plusieurs risques : 
- faire attention aux catégories : bien mélanger
- s'il y a un biais de représentativité : choisir de garder le même taux de chaque catégorie entre l'ensemble et modèle
- s'il est trop petit donc compliqué de diviser : on prend bcp de données en apprentissage et peu en test (super modèle mais mauvais test) --> solution on fait une validation croisée en faisant plusieurs modèles : on doit faire plusieurs modèles et faire la moyenne (attention coût de calcul)
  - cas extrème : LOO : Leavec one out on laisse 1 point donc il y a N modèle avec N le nombre de points

Attention il faut aussi évoter les fuites de données (des trucs d'apprentissages qui sont aussi dans le test donc on surestime le score)
--> Il faut donc adapter le modèle à notre sauce


### Revoir scikit-learn

Vachement puissant on adore tt faire en 1 ligne