### <p style="text-align: center;">Livrable</p>

# <p style="text-align: center;">Simulation d'un écosystème proie prédateur</p>

---

## Fichiers du livrable

| Élément | Fichier |
|---|---|
| Code source | `2_TP_code_v2.py` |
| Diagramme UML | `2_TP_uml_diagramme.pdf` |
| Tests unitaires | `2_TP_test.py` |

---

## Questions finales

*Répondez en quelques lignes.*

**Q1 : Quelle différence faites-vous entre une classe et un objet ?**  
Une classe est un plan de construction qui définit les attributs et les méthodes.
Un objet est un exemple concret d'une classe avec des valeurs précises pour les attributs.

<br>

**Q2/Q3 : Pourquoi *Lapin* peut-il hériter de *Proie* ? Pourquoi *Loup* peut-il hériter de *Predateur* ?**  
Dans mon cas précis, je n'ai pas créé de classe *Proie* car je n'ai que deux animaux, loup et lapin. Mais si j'avais eu un autre type d'animal, *Mulot*, qui aurait été mangé par les loups, j'aurais pu définir une classe *Proie* intermédiaire entre *Animal* et *Lapin*/*Mulot* pour regrouper les fonctions partagées entre ces deux classes, comme par exemple fuir. *Lapin* et *Mulot* auraient été des cas particuliers de *Proie*, en partageant tout ce qui définit une proie tout en ayant des comportements différents.  
Même logique pour *Loup* et *Predateur* : un loup est un cas particulier de prédateur, qui partage le comportement générique de chasse tout en ayant ses propres réglages (vitesse, vision, gain d'énergie par capture).  
Dans mon cas, *Animal* joue ce rôle intermédiaire : *Lapin* et *Loup* héritent d'*Animal* parce qu'ils partagent tous deux vitesse, vision, déplacement et calcul de distance, malgré des comportements de recherche de cible opposés (fuir vs chasser).

<br>

**Q4 : Donnez un exemple d'héritage dans votre programme.**  
*Lapin* et *Loup* héritent d'*Animal*, qui hérite lui-même d'*Etre_vivant*. Ils héritent ainsi de `vitesse`, `vision`, `bouger`, `distance_avec`, sans avoir à les réécrire, et ajoutent chacun leurs méthodes propres (`fuir` pour *Lapin*, `chasser` pour *Loup*).

<br>

**Q5 : Donnez un exemple de composition.**  
Une composition est une relation de possession entre deux classes. Dans mon code par exemple, la *Simulation* contient l'*Environnement*, qui contient *Lapin*, *Loup* et *Carotte*.

<br>

**Q6 : Donnez un exemple de polymorphisme.**  
Le polymorphisme, c'est appeler la même méthode sur des objets de classes différentes, et obtenir un comportement différent selon la classe réelle de l'objet.  
Dans mon code, dans la méthode `simuler_tour` de *Environnement*, le même nom de méthode, `reproduire`, produit un comportement différent selon l'objet : le lapin crée un nouveau *Lapin*, le loup un nouveau *Loup*, la carotte se disperse à proximité.

<br>

**Q7 : Pourquoi *Animal* peut-il être abstrait ?**  
*Animal* définit des méthodes communes (`bouger`, `distance_avec`, `aller_vers`) mais n'a pas de comportement complet à lui seul. Rendre *Animal* abstraite (via `ABC`, héritée d'*Etre_vivant*) empêche de créer un `Animal()` directement — il faut passer par une sous-classe concrète (*Lapin* ou *Loup*) qui complète les comportements manquants. Ça formalise l'idée qu'« animal » est une catégorie, pas une chose qu'on instancie telle quelle.

<br>

**Q8 : Pourquoi est-il préférable de demander à un objet de modifier son énergie plutôt que de modifier directement son attribut depuis l'extérieur ?**  
Si un autre bout du code faisait `lapin.energie -= 200` directement, l'énergie pourrait devenir négative, ce qui n'a pas de sens et casserait potentiellement d'autres méthodes qui supposent `energie >= 0`. En passant par `perte_energie`, la classe garde le contrôle sur ses propres règles (ici, ne jamais descendre sous 0) partout où la méthode est appelée. Ça évite aussi de dupliquer cette règle (`max(..., 0)`) à chaque endroit du code qui modifie l'énergie — un seul endroit à corriger si la règle change.

<br>

**Q9 : Que se passerait-il si nous supprimions la classe *Environnement* et mettions toute la logique dans `main.py` ?**  
- Le code serait moins clair : le code de lancement et l'affichage seraient mélangés ensemble.
- Impossibilité de faire tourner deux simulations en parallèle, à moins de rajouter d'autres lignes de code.

<br>

**Q10 : Quel principe de conception avez-vous trouvé le plus difficile à comprendre ? Expliquez pourquoi.**  
Le principe que j'ai trouvé le plus difficile à comprendre est l'abstraction, en particulier la méthode abstraite `reproduire`. Je ne voyais pas vraiment l'intérêt de la déclarer dans *Etre_vivant* alors que j'aurais très bien pu mettre trois fonctions complètement différentes dans *Lapin*, *Loup* et *Carotte*, sans avoir besoin de l'annoncer dans une classe commune. Pour moi, ça semblait être une étape en plus pour rien, puisque le résultat final (chaque classe avec sa propre version de `reproduire`) aurait été le même sans ça. J'ai fini par comprendre que ça sert surtout à garantir que toutes les sous-classes définissent bien cette méthode, et à pouvoir appeler `reproduire` sur n'importe quel être vivant sans me soucier de son type exact.

<br>

---

## Rapport de conception

**Héritage**  
*Etre_vivant* est la classe de base, dont hérite *Animal*, elle-même héritée par *Lapin* et *Loup*. *Carotte* hérite directement d'*Etre_vivant*. En effet, tout être vivant a un âge et une énergie, mais seuls les animaux ont une vitesse, une vision et savent se déplacer. Cette hiérarchie évite de dupliquer le code commun (`vieillir`, `perte_energie`, `gain_energie`) dans chaque classe finale.

<br>

**Polymorphisme**  
La méthode `reproduire` se comporte différemment selon la classe : un lapin crée un nouveau lapin, un loup un nouveau loup, une carotte se disperse à proximité.  
Le même principe s'applique à `mort()` : chaque classe adapte la condition (énergie à 0, ou âge dépassé pour les animaux et les carottes).  

<br>

**Encapsulation**  
L'énergie n'est jamais modifiée directement depuis l'extérieur (`lapin.energie = ...`), mais toujours via `perte_energie` et `gain_energie`, qui garantissent qu'elle reste entre 0 et le maximum autorisé.    
Ça évite qu'un oubli ailleurs dans le code ne mette un objet dans un état incohérent (énergie négative, par exemple).

<br>

**Composition**  
*Environnement* contient des listes d'objets *Lapin*, *Loup* et *Carotte* (`proie`, `predateur`, `ressource`) : c'est une relation « a un », différente de l'héritage. *Simulation* compose à son tour un *Environnement*. Ces objets n'ont de sens que rattachés à leur conteneur : un lapin existe dans un environnement, il n'est pas un environnement.

<br>

**Répartition des responsabilités entre les classes**  
Chaque classe a un rôle précis et ne déborde pas sur celui des autres :  
- *Etre_vivant* / *Animal* / *Lapin* / *Loup* / *Carotte* : le comportement individuel de chaque être vivant (se déplacer, chasser, fuir, se reproduire).
- *Environnement* : les règles du monde partagé (qui contient quoi, comment un tour se déroule, suppression des morts).
- *Simulation* : le pilotage global (lancement, affichage, statistiques), sans connaître les détails de comportement des animaux.
- *Config* : tous les paramètres numériques, séparés du code logique, pour faciliter l'équilibrage sans toucher aux classes.  
Cette séparation permet de modifier un aspect (par exemple, l'affichage dans *Simulation*) sans risquer de casser la logique de simulation dans *Environnement*.
