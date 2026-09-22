# Cours 1 Machine-Learning

## Session 2: Le machine-learning sélection de modèles et cas avancés

### Evaluation(s) :
**Slide 2 :** 
Attention différentes métriques en fonction si c'est régressif ou classif 

MAPE : min absolute pourcentage error
Accuracy : taux de bonnes classifications --> Précision (en français) mais on pejut pas l'utiliser

**Slide 3 :**\
Matrice de confusion (Classe) : Vachement interessant avec plusieurs classes car on peut prédire quelles erreurs occurent\
*Exemple :* confusion entre les 4 et les 9

**Slide 4 :**\
histoire de classe minoritaire en gros dans la fraude bancaire vu qu'il y a pas bcp de fraude et qu'il faut pas se tromper et bas on applique ce cas là pour éviter au max les faux positifs

Il y a les données de précisison et rappel (attention ça se pose sur 1 seule classification) et vu qu'on aime pas 2 chiffres on peut utiliser F1.

Rappel/Recall : taux de bons positifs trouvé sur tous les positifs existants

Précision : taux de bons positifs trouvé sur tous les positifs trouvés

*C'est quoi la courbe recall / precision ?*



**Slide 5- :** Classification déséquilibrée\
On trouve en premier une pente puis on décale petit à petit on obtient une courbe de faux positifs et de vrais positifs

La courbe ROC meilleur indicateur de qualité des modèles et comme on aide pas les trucs trop compliqué : 

AUC : aire sous la courbe
Le méilleur AUC c'est 1 tous les True Positifs avant les Faux positifs
La "pire" courbe c'est la diagonale dès qu'on a un sur 2


**Slide 9 :** Processus d'évaluation\
cf ce qu'on a déà vu 

**Slide 11 :**
*Important à lire*
1 jeu de test et 1 jeu de protocle



### Selection de modèles :

Avec scikit learn c'est bien car on a déjà pas mal de modèles il faut donc fair : choisir quelques modèles les fair etournes, analyser les résultats et puis passer la suite : modifier les paramètres et les hyperparamètres


### Selection de caractéristiques :

**Slide 16**

**slide 17**
matrice de distance : *c'est quoi ?*
noir : distance faible et jaune loin

Quand on rajoute des dimensions avec du bruits --> les blocs disparaissent même si les 2 colonnes n'ont pas bougé.
Plus y a des colonnes plus c'est nul

Il ne faut pas simplement mettre le classifieur dans des données brutes: il faut donc enlever des dolonnes mais la question c'est c'est quoi les questions utiles et celles pas utiles (faut aller discuter avec des gens sur le terrain et aussi de façon algorithmique)

**Slide 18** : Bonne ou mauvaise variable

Anova : on peut en conserver certaines attention là on les a utilisé indépendemment Cas linéaire pas parfait


*A compléter*

**Slide 25** :  
ACP et PCA : *noter les différentes méthodes à connaitres pour le faire fonctionner

**Slide 27**
Résguralisation L2 : Formulation Ridge (on veut tjs minimiser le L)
C||w||² : terme de contrôle 

**Slide28**
Le lasso : au lieu de minimiser une norme 2 on minimise une norme 1 (ou l'inverse)
La courbe verte du nombre de variables que tu utilises augmente progressivement (c'est mieux quand même)
Astier type sherani

Par contre c'est absolument pas stable et les perf sont pas bonnes.

Solution : lasso pour sélectionner le svariables plus ridge pour avoir la performance sous le sous ensemble de variables avec le ridge


**Slide 29** Alternative : elastic Net
On combine les 2 pénlisation C1 et c2 donc on a efficacité et performance



