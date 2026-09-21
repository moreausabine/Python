# Cours 1 Machine-Learning

## Session 2: chaine de traitements

### Introduction :
les chaines de traitement : c'est pas aussi facile il va falloir bidouiller le prétraitement car c'est là où ça se joueù

### Valeurs manquantes
c'est chiant et de la merde

si on voit pas : erreur ou sinon si y a un ? alor sil bascule tte la colonne en chaine de caractère. Il faut donc forcer les valeurs numéqirques et passer en NAN tous les ?

Comment gérer : 
supprimer la ligne (génial) ou la colonne si y a que des points d'interrogation  
On suppose la valeur : moyenne ou médiane (valeurs stat)   
on fait un modèle pour générer les valeurs manquantes(attention il nous fiat bcp de données) : on prend toutes nos variables et on dit que la colonne qu'on veut compléter ets notrr Y. Attention :on peut être limité si plusieurs variables manquantes sur la même ligne (discuter avec le terrain poir connaître l'ordre)  

Vaut mieux commencer avec une moyenne/médiane pour pouvoir commencer à raisonner

### Feature engineering

On aime pas les valeurs discrètes : on peut passer en 0-1 si c'est binaire pour pouvoir traduire les infos

Si c'est pas binaire :  
exemple : jaune, vert, rouge: ne pas faire 0-1-2 car jaune est pas entre rouge et vert et ça va servir à pondérer des trucs 
--> OneHotEncoder() : ajouter une colonne par type de catégorie

> Habitude à avoir :  
> Faire pleins d'histogramme : regarder ce qu'on connaît pas

**Simplification des données :**  
il faut réfléchir à quel point on binarise (UPS : pixel allumé ou éteint)

Sur les CV (véhicules américains) : les puissances sont très diverses mais regroupées en différentes catégories : faut il laisser le CV ou on peut discréditer (faire 3 colonnes) : on va demander aux experts

**Transformation arbitraires**   
Par exemple tt passer au log parce que : ça va impacter notre truc  
Sinon dans le cas d'un cochon si on a longueur et largeur et qu'on pourrait avoir besoin le volume : on peut faire le calcul pour être sûr que ce soit considéré

--> On va donc en enlever et en rajouter pour avoir les informations les plus utiles dans notre pbm

**Target Encoding**
Ultra puissant mais on a tendace à faire des fuites de données train-test

on a une variable discrète souvent on fait Hone-Encoder maison peut encoder la target dans chacune des variables prédictives --> faire des boosts de performance  
Mais c'est dangereux --> on risque de tricher 


### Normalisation

On centre-réduit toutes nos données. Les colonnes deviennent comparable : 

Réseau de neuronne : on est obligé car on sait pas forcément faire autrement

Arbre de décision ou Gradient Boosting : on réfléchit en terme de coupure entre les données --> ça sert à rien du coup

-->ça dep du logiciel qu'on a derrière

Il existe pleins de types de normalisation :
- min max
- multinoùiale
- Normalisation des produits scalaires
  
On normalise à 90% les colonnes et en 10% les lignes : les textes (pour éviter impact de la longueur) et série temporelle (normalisation par individu, on normalise pour que la puissance d'un truc ne soit pas sur-représenté)