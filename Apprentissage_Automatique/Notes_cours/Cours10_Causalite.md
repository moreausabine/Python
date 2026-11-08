# Cours 10 – Causalité : inférence et apprentissage

> Notes explicatives du cours d'Antoine Cornuéjols (AgroParisTech – INRAE MIA Paris-Saclay), *Causality: Inferencing and Learning* (152 diapositives).
> Le fil rouge de ces notes : **pour chaque concept, je donne d'abord l'intuition, puis la définition, puis l'exemple du cours.**

---

## Table des matières

1. [Pourquoi s'intéresser à la causalité ?](#1-pourquoi-sintéresser-à-la-causalité-)
2. [Corrélation ≠ causalité (et causalité ≠ corrélation linéaire)](#2-corrélation--causalité-et-causalité--corrélation-linéaire)
3. [Estimer des effets causaux](#3-estimer-des-effets-causaux)
4. [Diagrammes causaux et indépendance statistique](#4-diagrammes-causaux-et-indépendance-statistique)
5. [Le critère de la porte dérobée (back-door)](#5-le-critère-de-la-porte-dérobée-back-door)
6. [Impact sur l'interprétation des données observationnelles : le paradoxe de Lord](#6-impact-sur-linterprétation-des-données-observationnelles--le-paradoxe-de-lord)
7. [Le do-calcul](#7-le-do-calcul)
8. [La science de l'intervention et l'échelle de la causalité](#8-la-science-de-lintervention-et-léchelle-de-la-causalité)
9. [Conclusions](#9-conclusions)
10. [Glossaire et fiche de révision](#10-glossaire-et-fiche-de-révision)

---

## 1. Pourquoi s'intéresser à la causalité ?

### L'idée générale

Un algorithme d'apprentissage classique apprend des **associations** : « quand je vois X, je vois souvent Y ». C'est très utile pour **prédire**, mais insuffisant dès qu'on veut **comprendre**, **expliquer** ou **agir**.

### Les 5 raisons données dans le cours

| # | Raison | Ce que ça veut dire concrètement |
|---|--------|----------------------------------|
| 1 | Les règles de décision apprises peuvent être **non interprétables** | Un modèle peut marcher sans qu'on sache pourquoi. |
| 2 | Les **explications** que nous produisons sont naturellement **causales** | Quand on demande « pourquoi ? », on attend une réponse du type « parce que… », pas « parce que c'est corrélé ». |
| 3 | Les algorithmes eux-mêmes sont **difficiles à déchiffrer** | Ils sont forcément biaisés, mais ces biais peuvent être cachés. |
| 4 | Identifier des liens causaux permet d'**intervenir** | Résumer des données ≠ savoir ce qui se passera si j'agis. |
| 5 | Les relations causales sont **transférables** d'un domaine/contexte à un autre | Une corrélation dépend du contexte où elle a été mesurée ; un mécanisme causal, lui, reste valable ailleurs. |

### Exemple pour la raison 5

Imagine qu'on observe que dans une région, les gens qui portent des lunettes de soleil achètent plus de glaces. Cette corrélation ne se transportera pas en hiver ou en Norvège. En revanche, le mécanisme « il fait chaud → on achète des glaces » tient partout.

### Questions causales typiques (diapos 8-9)

- Un traitement est-il efficace pour prévenir une maladie ?
- Est-ce la nouvelle loi fiscale ou la campagne de publicité qui a fait monter nos ventes ?
- Quel est le coût de santé *attribuable* à l'obésité ?
- Des données d'embauche peuvent-elles prouver une discrimination sexuelle ?
- Le glyphosate augmente-t-il le risque de cancer ? Les néonicotinoïdes ont-ils un impact sur les abeilles ? Le statut socio-économique est-il une cause d'inégalités de santé ?

Toutes ces questions ont la même forme générale (diapo 10) :

```
   A  ──?──►  Y
 (événement,        (variable
  traitement)        d'intérêt)
```

où **A** est une variable binaire (traité / non traité) ou réelle (température), et **Y** est la variable d'intérêt (malade / pas malade, rendement…).

> **À retenir** : la causalité sert à répondre à des questions de type **« que se passe-t-il si… ? »** et **« pourquoi ? »**, que les statistiques classiques ne peuvent pas traiter seules.

---

## 2. Corrélation ≠ causalité (et causalité ≠ corrélation linéaire)

Cette partie démontre qu'il y a **deux erreurs symétriques** à éviter.

### 2.1 Causalité sans corrélation ?

**Intuition.** Si X cause Y, on pourrait penser que X et Y sont forcément corrélés. Pas du tout.

**Explication.** Les mesures usuelles (coefficient de **Pearson**, **Spearman**) ne détectent que les relations **linéaires** ou **monotones**. Si Y = X² avec X symétrique autour de 0, X cause Y de façon parfaitement déterministe, mais le coefficient de Pearson est ≈ 0 (les valeurs à gauche et à droite se compensent).

D'autres mesures savent détecter des liens non linéaires et non monotones :
- **MIC** (Maximal Information Coefficient),
- **HSIC** (Hilbert-Schmidt Independence Criterion).

> **Conclusion 1 : la causalité n'est pas de la corrélation (linéaire).** Une absence de corrélation de Pearson ne prouve pas l'absence de lien causal.

### 2.2 Corrélation sans causalité

C'est l'erreur la plus connue. Le cours en donne plusieurs variantes, chacune avec un **mécanisme** différent. Il faut bien les distinguer, car chacune correspond à une forme de graphe causal différente (qu'on formalisera en section 4).

#### a) Le problème de Monty Hall : l'illusion cognitive

**L'énoncé.**
1. Vous choisissez une porte parmi 3 (disons la n°1). Derrière l'une il y a une voiture, derrière les deux autres des chèvres.
2. Le présentateur, **qui sait où est la voiture**, ouvre une autre porte (disons la n°3) derrière laquelle il y a une chèvre.
3. Il vous demande : « Voulez-vous changer pour la porte n°2 ? »

**Réponse : oui, il faut changer** (vous gagnez 2 fois sur 3 en changeant).

**Pourquoi ? Le tableau de la diapo 17** (vous avez choisi la porte 1) :

| Voiture derrière | Si vous changez | Si vous restez |
|---|---|---|
| Porte 1 | Perdu | Gagné |
| Porte 2 | Gagné | Perdu |
| Porte 3 | Gagné | Perdu |

Rester ne gagne que si votre choix initial était le bon (probabilité 1/3). Changer gagne dans tous les autres cas (2/3).

**Le lien avec la causalité : le collider.** Le choix du présentateur dépend à la fois de **votre choix** et de **l'emplacement de la voiture** (il ne peut ouvrir ni votre porte, ni celle de la voiture) :

```
 Votre choix ──►  Porte ouverte  ◄── Position de la voiture
                  par l'animateur
```

Votre choix et la position de la voiture sont **indépendants** au départ. Mais dès que vous **observez** la porte ouverte (c'est-à-dire que vous **conditionnez** sur la variable du milieu), ils deviennent **dépendants** : l'information que donne l'animateur « fuite » vers la position de la voiture. C'est ce qu'on appelle l'**explaining away** ou **effet de collider** (plus de détails en 4.3).

**La contre-expérience (le « jeu de Simpson »).** Si l'animateur **ne sait pas** où est la voiture et ouvre une porte au hasard (qui se trouve être une chèvre), alors il n'y a plus de lien entre sa décision et la voiture. Changer ou non n'a plus d'importance (1/2 – 1/2). Le même comportement observé (une chèvre apparaît) a donc une signification radicalement différente selon le **mécanisme causal** qui l'a produite. C'est exactement le message du cours : **les données seules ne suffisent pas**.

> Le cours souligne que c'est **une illusion cognitive**, comme les illusions visuelles : on cherche toujours des explications causales, même quand le raisonnement est probabiliste.

#### b) L'alarme : explaining away (diapo 22)

```
 Séisme ──►  Alarme  ◄── Cambriolage
```

- *A priori*, séisme et cambriolage sont indépendants.
- Mais si on **sait que l'alarme a sonné**, ils deviennent dépendants : si on apprend qu'il y a eu un séisme, la probabilité d'un cambriolage **baisse** (le séisme « explique » l'alarme, donc on n'a plus besoin du cambriolage pour l'expliquer).

C'est le même mécanisme que Monty Hall : **conditionner sur un effet commun crée une dépendance entre ses causes**.

#### c) Le facteur de confusion : glaces et noyades (diapos 24-28)

- Variables : **Ice** (achat de glaces), **Acc.** (accident de baignade).
- On observe une forte corrélation linéaire entre les deux.
- Est-ce que manger des glaces cause des noyades ? Non.

**Explication.** Une troisième variable, la **température**, fait augmenter à la fois les achats de glaces et les baignades (donc les accidents) :

```
         Température
          ↙       ↘
       Ice         Acc.
```

Cette variable **Z** s'appelle un **facteur de confusion** (*confounder*). Quand on **conditionne** sur la température (on compare des jours à même température), la corrélation disparaît : « il n'y a plus d'effet de Ice sur Acc. ».

Dans un **fork** (fourche) `X ← Z → Y`, on a :
> Sachant Z, X et Y deviennent **indépendants**.

> **Problème majeur (diapo 27)** : la plupart du temps, on **ne peut pas distinguer** les relations *spurieuses* (fausses) des relations *non spurieuses* (vraies) **à partir des seules données statistiques**.

#### d) Le paradoxe de Simpson (diapos 29-49)

C'est l'exemple vedette du cours : une tendance qui **s'inverse** quand on regroupe les données.

**Les données (traitement de calculs rénaux) :**

| | Traitement A | Traitement B | Total |
|---|---|---|---|
| Petits calculs | **93 %** (81/87) | 87 % (234/270) | 357 |
| Gros calculs | **73 %** (192/263) | 69 % (55/80) | 343 |
| **Population** | 78 % (273/350) | **83 %** (289/350) | 700 |

**Le paradoxe.**
- Dans **chaque** sous-groupe (petits, gros calculs), A fait mieux que B.
- Mais dans la **population entière**, B fait mieux que A.

**Quelle est l'explication ?** La *condition* (taille des calculs) influence **à la fois** le traitement choisi et le succès :
- Les petits calculs sont **plus faciles à traiter** (succès plus élevé quel que soit le traitement).
- Le traitement **B est donné surtout aux petits calculs** : 270 / (270 + 80) = **77,1 %**.
- Le traitement **A est donné aux petits calculs seulement** : 87 / (87 + 263) = **23,1 %**.

Donc B récolte les cas faciles, ce qui gonfle artificiellement son taux global. La taille des calculs est un **facteur de confusion** :

```
        Condition (taille des calculs)
           ↙                  ↘
     Traitement  ────────►  Succès
```

**La bonne méthode : ajuster sur le facteur de confusion.** On calcule l'effet dans chaque strate, puis on fait la moyenne pondérée selon la **taille de la population** (357/700 et 343/700).

Vérification numérique (arrondis) :
- A : 0,93 × 0,51 + 0,73 × 0,49 ≈ **83 %**
- B : 0,87 × 0,51 + 0,69 × 0,49 ≈ **78 %**

→ **Le traitement A est meilleur.** (Le cours le conclut aussi : « Treatment A is better ».)

**Pourquoi c'est contre-intuitif ?** On croit spontanément que si `a/b > c/d` et `e/f > g/h`, alors `(a+e)/(b+f) > (c+g)/(d+h)`. C'est **faux** si les groupes n'ont pas la même taille dans chaque bras. Ici : 81/87 > 234/270 et 192/263 > 55/80, mais 273/350 < 289/350.

**La cause profonde du malentendu** : intuitivement, on suppose que le traitement est **indépendant** de la condition. Or ici ce n'est pas le cas.

**Autre version du paradoxe (diapos 38-39)** : l'exercice physique semble **bénéfique** (pente descendante) dans chaque tranche d'âge pour le cholestérol, mais **néfaste** (pente montante) dans la population globale. Ici **l'âge** est le facteur de confusion (plus on est âgé, plus on fait d'exercice *et* plus le cholestérol est élevé). Il faut **conditionner sur l'âge**.

#### e) Mais attention : il ne faut pas *toujours* ajuster ! (diapos 41-42)

**Exemple.** La pression artérielle est une cause possible de crise cardiaque. Le médicament B est censé réduire la pression. Réduit-il aussi les crises cardiaques ?

| | Contrôle (sans médicament) | Traitement (avec médicament) |
|---|---|---|
| Pression basse | 1 crise / 19 pas de crise | 3 crises / 37 pas de crise |
| Pression haute | 12 crises / 28 pas de crise | 8 crises / 12 pas de crise |
| **Total** | **13 / 47** (≈ 21,7 %) | **11 / 49** (≈ 18,3 %) |

- Par strate de pression : le médicament semble **moins bon** (5 % → 7,5 % chez les pressions basses ; 30 % → 40 % chez les pressions hautes).
- Globalement : le médicament semble **meilleur** (21,7 % → 18,3 %).

**Lequel croire ?** Ici le graphe est :

```
   Médicament ──► Pression artérielle ──► Crise cardiaque
        └───────────────────────────────────────┘
```

La pression est un **médiateur** (elle se trouve *sur le chemin causal* du médicament vers la crise), **pas un facteur de confusion**. Si on conditionne sur elle, on **bloque** une partie de l'effet qu'on veut mesurer (le médicament agit *via* la pression). On doit donc regarder les résultats **agrégés** : le médicament B est bon.

**Règle d'or :**
- **Confounder** (cause commune) → il faut **ajuster** (conditionner).
- **Mediator** (sur le chemin causal) → il ne faut **pas** ajuster si on veut l'effet total.

#### f) Simpson : la leçon « ça dépend » (COVID-27, diapos 43-49)

On a des données de mortalité selon le traitement (A ou B) et l'état du patient au moment du choix. On doit choisir le traitement pour tout un pays. Quelle décision ?

**Cela dépend du graphe causal !**

- **Scénario 1** : les médecins donnent le traitement A surtout aux patients en état léger. L'état du patient est une **cause commune** du traitement et de la mortalité (confounder). → Il faut regarder **par sous-population** : on choisit le **traitement B**.
- **Scénario 2** : le traitement B est si rare que les patients doivent attendre longtemps. Leur état empire pendant l'attente, ce qui augmente la mortalité. Ici l'état du patient est **causé par** le traitement (médiateur). Même si B est plus efficace *une fois administré*, il cause globalement de pires états. → Il faut regarder **l'agrégat** : on choisit le **traitement A**.

**Les mêmes chiffres, deux décisions opposées.** Leçon du cours :

> - Les relations causales apportent une information **absente des probabilités conditionnelles**.
> - **Les données seules ne suffisent pas.**
> - Connaître le graphe causal **change la décision**.

---

## 3. Estimer des effets causaux

### 3.1 Effet causal individuel (ITE)

**Intuition.** Pour savoir si un traitement a marché *pour moi*, je voudrais comparer ce qui s'est passé avec ce qui se serait passé **sans** le traitement.

**Notation (résultats potentiels).**
- `A` : variable de traitement (0 ou 1).
- `Y^{a=1}` : le résultat que l'individu **aurait** s'il était traité.
- `Y^{a=0}` : le résultat qu'il **aurait** s'il n'était pas traité.

**Définition.** `A` a un effet causal sur le résultat de l'individu si
```
Y^{a=1} ≠ Y^{a=0}
```
L'effet individuel est `τ_i = Y_i^{a=1} − Y_i^{a=0}` (ITE = *Individual Treatment Effect*).

Exemples du cours : l'exposition d'une abeille à un néonicotinoïde (A) sur sa survie (Y) ; la maladie (A) sur le rendement d'une parcelle de blé à Saclay en 2023 (Y).

### 3.2 Le problème fondamental de l'inférence causale

On ne peut **jamais** observer à la fois `Y^{a=1}` et `Y^{a=0}` pour le **même** individu : une parcelle est malade **ou** saine, une abeille est exposée **ou** non.

> C'est un **problème de données manquantes** très particulier : pour chaque individu, la moitié de l'information est *par construction* inobservable (on parle de **contrefactuel**).

Illustration (diapo 63) : tableau de 6 individus avec `T`, `Y`, `Y(1)`, `Y(0)` : pour chaque individu une seule des deux colonnes est remplie, l'autre contient « ? ».

### 3.3 Effet causal moyen (ATE)

Comme l'effet individuel est inaccessible, on vise l'**effet moyen** dans la population :
```
τ = E[Y^{a=1}] − E[Y^{a=0}]     (ATE = Average Treatment Effect)
```
Il y a un effet causal moyen si `E[Y^{a=1}] ≠ E[Y^{a=0}]`.

> **Notez** : par linéarité de l'espérance, `E[Y^{a=1}] − E[Y^{a=0}] = E[Y^{a=1} − Y^{a=0}]`. L'effet moyen est la moyenne des effets individuels, même si on ne connaît pas chaque effet individuel.

### 3.4 Pourquoi ne peut-on pas juste comparer les moyennes observées ?

**La tentation.** Calculer `E[Y | a=1] − E[Y | a=0]` : la différence entre la moyenne des traités et celle des non-traités. Sur l'exemple de la diapo 63, on trouve `2/3 − 1/3 = 1/3`.

**Le piège.** Ce n'est **pas** égal à `E[Y^{a=1}] − E[Y^{a=0}]` en général ! Pourquoi ?

**Si une variable X explique à la fois T et Y** (un confounder) :

```
        X
      ↙   ↘
     T ───► Y
```

alors les traités et les non-traités ne sont **pas comparables** : ils diffèrent déjà par X avant même le traitement. La différence observée mélange l'effet du traitement et l'effet de X.

**Exemple du cours (diapo 58-59)** : si on traite les gens *quand ils sont malades* (D = maladie), alors T et la maladie sont liées. L'effet de la maladie sur Y sera **sous-estimé** à cause du traitement (le traitement compense en partie).

### 3.5 Les deux questions fondamentales (diapo 60)

1. Une entité ne peut pas être à la fois traitée et non traitée : **comment estimer un effet causal ?**
2. Peut-on estimer des effets causaux **à partir de données observationnelles seules** ? Si oui, sous quelles conditions et comment ?

### 3.6 Première solution : l'essai contrôlé randomisé (RCT)

**Principe.** On **assigne les individus au traitement au hasard**. Le hasard casse le lien entre X (le confounder) et T :

```
        X                       X
      ↙   ↘        →             ↘
     T ───► Y                T ───► Y      (la flèche X → T est supprimée)
```

**Pourquoi ça marche ?** Avec une assignation aléatoire, les deux groupes sont **identiques en moyenne sur tous les aspects pertinents, sauf le traitement** (y compris ceux qu'on n'a pas mesurés). Toute différence de résultat est donc imputable au traitement.

**Limite (diapo 74)** : ce n'est pas toujours possible (éthique, coût…). Exemple : on ne peut pas assigner aléatoirement des gens à « fumer » pour étudier l'effet du tabac.

### 3.7 La question centrale

> **Comment contourner la confusion**, si possible, pour calculer les effets causaux à partir de données **observationnelles** (probabilités conditionnelles) seulement ?
> Et d'abord : **comment identifier les facteurs de confusion ?**

---

## 4. Diagrammes causaux et indépendance statistique

### 4.1 Trois grandes approches de la causalité (diapos 73-74)

| Approche | Auteur | Statut dans ce cours |
|---|---|---|
| 1. Essais contrôlés randomisés | – | Évoqués ; pas toujours possibles |
| 2. Cadre des résultats potentiels | Donald Rubin (1974) | **Non traité** |
| 3. **Graphes causaux** | **Judea Pearl** | **Approche principale du cours** |

*(Sur l'idée de Pearl, voir aussi la citation de la diapo 71-72 : l'IA traditionnelle « représente d'abord, acquiert ensuite », alors que la question est de savoir comment les machines peuvent acquérir une connaissance causale.)*

### 4.2 Diagrammes causaux / SCM

**Principe.** Un **diagramme causal** est un graphe orienté dont les nœuds sont des variables. Une flèche `X → Y` signifie : **il existe une règle qui dit comment Y changerait si X changeait**.

> Souvent, la **structure du graphe** suffit à répondre à toutes sortes de questions causales et contrefactuelles. Parfois, il faut en plus des probabilités.

#### L'exemple du peloton d'exécution (diapos 77-80)

```
                Ordre du tribunal (CO)
                         │
                         ▼
                  Capitaine (C)
                   ↙         ↘
        Soldat A (A)       Soldat B (B)
                   ↘         ↙
                  Mort (D)
```

Ce même graphe permet de poser **trois types de questions** de difficulté croissante :

| Type | Question | Raisonnement | Réponse |
|---|---|---|---|
| **Associationnelle** (*observer*) | Si le prisonnier est mort, l'ordre du tribunal a-t-il été donné ? | Les soldats n'auraient pas tiré sans le capitaine, qui n'aurait pas ordonné sans l'ordre du tribunal. | **Oui** |
| **Interventionnelle** (*agir*) | Et si le soldat A décidait de tirer **de lui-même**, sans attendre le capitaine ? | On **supprime les flèches qui entrent** dans le nœud sur lequel on intervient (A). | Le prisonnier est mort (**D = True**) |
| **Contrefactuelle** (*imaginer*) | Et si le soldat A **avait décidé de ne pas tirer** (alors qu'en réalité il a tiré) ? | On « rejoue » le monde : B aurait obéi au capitaine et tiré quand même. | Le prisonnier serait mort (**D = True**) |

**À retenir sur la méthode : une intervention = on coupe les flèches entrantes** du nœud sur lequel on agit. Ça traduit l'idée que « je décide de la valeur de A moi-même, ce qui ne dépend plus de ses causes habituelles ».

#### L'exemple de la vaccination contre la variole (diapos 81-84)

**Question** : la vaccination est-elle bénéfique ou néfaste ?

**Hypothèses** (sur 1 million d'enfants, 99 % vaccinés) :
- Enfant vacciné : 1 chance sur 100 de développer une réaction, qui est mortelle 1 fois sur 100 ; 0 chance d'avoir la variole.
- Enfant non vacciné : 0 chance de réaction, mais 1 chance sur 50 d'avoir la variole, mortelle 1 fois sur 5.

**Les données observées :**
- 990 000 vaccinés → 9 900 réactions → **99 morts** dues au vaccin.
- 10 000 non vaccinés → 200 cas de variole → **40 morts** dues à la maladie.

→ **Plus d'enfants meurent de la vaccination (99) que de la maladie (40) !** Conclusion naïve : « vacciner est dangereux ».

**La question contrefactuelle** : « Que se serait-il passé si on avait mis le taux de vaccination à **zéro** ? »
- 1 000 000 × 1/50 = **20 000** cas de variole.
- 20 000 × 1/5 = **4 000** morts.
- Soit **3 861 morts de plus** que la situation actuelle (4 000 − (99 + 40) = 3 861).

**Morale.** Les chiffres observés ne reflètent pas la situation d'un monde où personne n'est vacciné. La vaccination *protège* en réduisant la maladie, mais ce bénéfice est « invisible » dans les données : il se voit uniquement dans le scénario contrefactuel.

### 4.3 Graphes et indépendances statistiques : les trois briques de base

Tout graphe causal se décompose en **trois structures élémentaires** de 3 nœuds. Les comprendre, c'est comprendre toute la causalité de ce cours.

#### (a) La chaîne (*chain*) : `A → B → C`

```
 Feu ──► Fumée ──► Alarme
```

- A influence C **à travers** B.
- A et C sont **dépendants** (si je vois du feu, je m'attends à l'alarme).
- **Mais** si je **contrôle B** (je sais s'il y a de la fumée), A et C deviennent **indépendants** : `A ⊥ C | B`. Une fois qu'on connaît la fumée, savoir qu'il y a du feu n'apprend plus rien sur l'alarme.

**Image mentale** : un tuyau dans lequel circule l'information. Contrôler B revient à **boucher le tuyau**.

#### (b) La fourche (*fork*) : `A ← B → C`

```
        Âge
      ↙     ↘
Pointure   Capacité de lecture
```

- B est une **cause commune** (confounder) de A et C.
- A et C sont **dépendants** si on ne contrôle pas B (les enfants à grands pieds lisent mieux… parce qu'ils sont plus âgés !).
- Si on **contrôle B**, A et C deviennent **indépendants** : `A ⊥ C | B`.

**Image mentale** : l'information circule de A vers C en « remontant » à B puis en « redescendant ». Contrôler B **boucle le tuyau**.

#### (c) Le collider (*collider*) : `A → B ← C`

```
 Talent ──► Célébrité ◄── Beauté
```

- B est un **effet commun** de A et C.
- *A priori*, A et C sont **indépendants** : le talent et la beauté ne sont pas liés dans la population générale.
- **Mais si on contrôle B** (on ne regarde que les célébrités), A et C deviennent **dépendants** (`A ⊥̸ C | B`) : parmi les célébrités, quelqu'un de peu beau a probablement beaucoup de talent, et inversement.

**Image mentale** : par défaut le tuyau est **fermé**. Contrôler B l'**ouvre**. C'est le contraire des deux autres cas, et c'est la source des paradoxes de type Monty Hall, de l'alarme (explaining away) et des « biais de sélection ».

#### (d) Contrôler un descendant

Contrôler un **descendant** d'une variable, c'est la contrôler **partiellement**.
- Contrôler un descendant d'un **médiateur** (a) : ferme **partiellement** le tuyau.
- Contrôler un descendant d'un **collider** (c) : ouvre **partiellement** le tuyau.

#### Tableau récapitulatif

| Structure | Par défaut (sans rien contrôler) | Si on contrôle B |
|---|---|---|
| Chaîne `A → B → C` | A et C **dépendants** | **indépendants** (bloqué) |
| Fourche `A ← B → C` | A et C **dépendants** | **indépendants** (bloqué) |
| Collider `A → B ← C` | A et C **indépendants** | **dépendants** (ouvert !) |

> **Piège classique** : croire qu'« il vaut mieux contrôler le plus de variables possible ». C'est faux : contrôler un collider **crée** de la fausse corrélation.

---

## 5. Le critère de la porte dérobée (back-door)

### 5.1 L'objectif : éliminer le « do »

Dans cette partie, on cherche à passer de la quantité **causale** `P(Y | do(X))` à des quantités **observables** `P(Y | X, …)`.

**Les méthodes pour éliminer les variables contrefactuelles** (diapo 87) :
1. Essais randomisés (RCT)
2. **Critère back-door** ← vu ici
3. Critère front-door ← vu en 7
4. Do-calculus ← vu en 7
5. Variables instrumentales (IV) ← cité, non détaillé

### 5.2 L'opérateur `do` : `P(Y | do(X)) ≠ P(Y | X)`

**La différence entre « voir » et « faire » :**
- `P(Y | X = x)` : la probabilité de Y **parmi les individus pour lesquels on a observé** X = x. (Observation passive.)
- `P(Y | do(X = x))` : la probabilité de Y **si l'on force** X à valoir x. (Intervention.)

Ces deux quantités diffèrent dès qu'il y a de la confusion. **Les chemins non causaux sont la source de la confusion** : tout ce qui fait que `P(Y | do(X)) ≠ P(Y | X)`.

**L'opérateur do** « efface toutes les flèches qui entrent dans X » : il empêche l'information sur X de circuler dans la direction non causale. **La randomisation a exactement le même effet.**

### 5.3 Définition d'un chemin back-door

> Un **chemin back-door** de X à Y est tout chemin qui **commence par une flèche pointant vers X**.

**Pourquoi ces chemins posent-ils problème ?** Ils font circuler de l'information de X vers Y **par un autre chemin que l'effet causal direct** : c'est de la corrélation spurieuse.

**Critère.** X et Y sont **déconfondus** si on **bloque tous les chemins back-door**.

### 5.4 Quand un chemin est-il bloqué ?

En combinant les trois briques de la section 4.3 :
- Un chemin est bloqué si on contrôle un **médiateur** ou une **cause commune** (chaîne ou fourche) qui s'y trouve ;
- ou s'il contient un **collider** que l'on **ne contrôle pas** (ni ses descendants).

### 5.5 Les « jeux » d'entraînement (diapos 94-105)

Dans chaque jeu, on cherche les chemins back-door et on décide quoi contrôler.

| Jeu | Chemin(s) back-door | Que faire ? |
|---|---|---|
| **1** | Aucune flèche n'entre dans X → **aucun** back-door | Rien à contrôler |
| **2** | `X ← A → B ← D → E → Y` | **Déjà bloqué** par le collider en B. Rien à contrôler. |
| **3** | `X ← B → Y` | Seul moyen de bloquer : **contrôler B**. Si B n'est pas observable, **impossible** d'estimer l'effet sans expérience randomisée. |
| **4** | `X ← A → B ← C → Y` | **Déjà bloqué** par le collider en B. Rien à contrôler. |
| **5** | (1) `X ← A → B ← C → Y` et (2) `X ← B ← C → Y` | Pour bloquer (2), on contrôle B ; **mais** ça ouvre le collider du chemin (1) (B est un collider sur (1)). Il faut alors **aussi** contrôler A (ou C). |
| **6** | Le jeu 5 + 4 nouveaux confounders (D, E, F, G) | Voir ci-dessous. |

**Le jeu 5 est le plus instructif.** Il illustre le piège : en bloquant un chemin on peut **en ouvrir un autre**. La stratégie correcte demande de réfléchir sur **l'ensemble** des chemins.

**Le jeu 6 : tabagisme et asthme (diapo 105).** Variables :
- A : tabagisme des parents ; B : asthme dans l'enfance ; C : prédisposition à l'asthme (**non observable**) ; D : asthme des parents ; E : bronchite chronique ; F : sexe ; G : statut socio-économique.
- X : comportement tabagique ; Y : asthme à l'âge adulte.

Conclusions du cours :
- Il faut contrôler **A et B**, ou **A et C**. Mais C est inobservable, donc on prend **A et B**.
- Il faut aussi contrôler **E, F, G** (confounders).
- **Mais pas D** (le cours l'indique sans plus de détail ; dans ce type de configuration, contrôler D ouvrirait un chemin qu'un collider bloquait jusque-là, ou boucherait un chemin causal utile).

> **Méthode générale en 4 étapes :**
> 1. Lister tous les chemins entre X et Y qui **commencent par une flèche vers X**.
> 2. Pour chacun, regarder s'il est **déjà bloqué** par un collider non contrôlé.
> 3. Pour ceux qui ne le sont pas, choisir des variables **observables** à contrôler (chaîne/fourche).
> 4. Vérifier qu'on **n'ouvre pas** de nouveau chemin via un collider (ou un de ses descendants), et qu'on ne contrôle pas un **médiateur** de l'effet total.

---

## 6. Impact sur l'interprétation des données observationnelles : le paradoxe de Lord

**Intuition.** Les mêmes nuages de points peuvent mener à des conclusions opposées, selon les hypothèses causales qu'on fait. Le paradoxe de Lord (1967) le montre avec deux statisticiens qui ont tous les deux « raison ».

### 6.1 Première version : le régime alimentaire et le sexe (diapos 109-114)

**Le contexte.** Une école veut étudier si le régime de sa cantine a des effets **différents sur les filles et les garçons**. Le poids de chaque élève est mesuré en septembre (W_i) puis en juin (W_f).

**Les données** : deux ellipses (nuages de points), une pour les filles, une pour les garçons, le long de la diagonale `W_f = W_i`.

**Statisticien 1.** Les filles ont le même poids moyen en juin qu'en septembre (le nuage est symétrique autour de la diagonale) ; idem pour les garçons. → **Pas d'effet différencié** du régime selon le sexe.

**Statisticien 2.** Le poids final dépend fortement du poids initial. Il faut donc **stratifier selon le poids initial** (une coupe verticale à W₀). À poids initial égal, l'ellipse des garçons est plus haute que celle des filles. → **Il y a un effet différencié**.

**Qui a raison ?** Il faut regarder le **graphe causal** :

```
        W_i  (poids initial)
       ↗  │ ↘
     S    │   W_f
      (sexe)  │
         ↘    ▼
          Y = W_f − W_i  (gain de poids)
```

- Le sexe S est cause de W_i **et** de W_f.
- W_i influence W_f indépendamment du sexe.
- La variable d'intérêt est le **gain** : `Y = W_f − W_i`.

**Chemin back-door entre S et Y ?** Aucun. Donc **pas besoin de contrôler quoi que ce soit** (et contrôler W_i serait même une erreur, car W_i est un médiateur de S vers le gain).

→ **Le statisticien 1 a raison** : pas d'effet différencié du régime selon le sexe.

### 6.2 Seconde version : le régime lui-même (diapos 115-119)

**Nouvelle question** : on veut maintenant étudier l'effet **du régime (D) lui-même**, pas du sexe. Les élèves mangent dans l'une de deux cantines (A ou B) aux régimes différents. **Les élèves les plus lourds au départ ont tendance à manger à la cantine B**, les plus légers à la cantine A.

Les deux statisticiens refont **exactement le même raisonnement** qu'avant :
- Statisticien 1 (symétrie, gain `W_f − W_i` de même distribution dans les deux ellipses) → pas d'effet du régime.
- Statisticien 2 (compare les poids finaux à poids initial W₀ fixé) → il y a un effet.

**Qui a raison cette fois ?** Le graphe a changé : la variable causale est D (le régime), et **la flèche s'inverse** : le poids initial W_i **influence le régime D** (les lourds vont à B).

```
         W_i
       ↙  │  ↘
     D    │    W_f
       ↘  ▼  ↙
          Y (gain)
```

- W_i est maintenant un **confounder** de D et W_f (et non un médiateur).
- Il y a un **chemin back-door** entre D et Y.
- Il faut **contrôler W_i**.

→ **Le statisticien 2 a raison** : il y a un effet différencié du régime.

### 6.3 La conclusion de Lord

> - Les nuages de points, seuls, sont **ambigus**.
> - Il faut des **hypothèses sur le graphe causal** sous-jacent pour interpréter les résultats.

**Ce qu'il faut retenir.** C'est la **même structure statistique** (mêmes ellipses, mêmes raisonnements) mais **deux graphes causaux différents** donnant **deux conclusions opposées**. Le même mot « contrôler le poids initial » est **correct** dans un cas (W_i confounder) et **incorrect** dans l'autre (W_i médiateur).

---

## 7. Le do-calcul

### 7.1 Objectif

> Transformer une expression contenant l'opérateur **do** en une expression **sans do**, ne contenant que des variables **observables**, pour pouvoir calculer un **estimateur**.

**Intuition.** Le do-calcul est un **jeu de règles de réécriture** : on part de `P(y | do(x))`, qu'on ne sait pas calculer à partir des données, et on essaie, par applications successives de trois règles, d'arriver à une formule que l'on sait calculer.

### 7.2 Notation des graphes modifiés

- `G_{X̄}` (barre au-dessus) : le graphe G dans lequel on a **supprimé toutes les flèches entrant dans X**.
- `G_{X̲}` (barre en dessous) : le graphe G dans lequel on a **supprimé toutes les flèches sortant de X**.

### 7.3 Les trois règles (diapos 123-130)

#### Règle 1 : ajouter/supprimer une observation

> Quand on observe une variable W **non pertinente** pour Y (éventuellement conditionnée sur Z), la distribution de Y est inchangée :
> ```
> P(y | do(x), z, w) = P(y | do(x), z)
> ```
> à condition que Z bloque tous les chemins de W à Y après suppression des flèches entrant dans X.

*Exemple du cours* : Feu (W) → Fumée (Z) → Alarme (Y). Z bloque tous les chemins de W à Y, donc une fois qu'on connaît la fumée, savoir qu'il y a du feu n'apporte plus rien sur l'alarme.

**En une phrase** : on peut ignorer une variable observée si elle est indépendante de Y (compte tenu de ce qu'on sait déjà).

#### Règle 2 : échanger action et observation

> Si un ensemble Z bloque tous les chemins **back-door** de X à Y, alors **faire X** est équivalent à **voir X**, conditionné sur Z :
> ```
> P(y | do(x), z) = P(y | x, z)
> ```
> à condition que Z satisfasse le critère back-door.

**En une phrase** : c'est le critère back-door formulé comme une règle. Si j'ai correctement ajusté, alors observer revient à intervenir.

*(Formulation générale de Pearl : `P(y | do(x), do(z), w) = P(y | do(x), z, w)` si Y ⊥ Z | X, W dans le graphe où l'on supprime les flèches entrant dans X et sortant de Z.)*

#### Règle 3 : supprimer une action

> On peut retirer `do(x)` s'il n'y a **aucun chemin causal** de X à Y :
> ```
> P(y | do(x)) = P(y)
> ```
> à condition qu'il n'existe aucun chemin de X à Y composé uniquement de flèches allant « vers l'avant ».

**En une phrase** : si X ne peut pas influencer Y, forcer X ne change rien à la distribution de Y.

### 7.4 Application à quatre graphes simples (diapos 126-132)

#### (a) Le DAG le plus simple : `X → Y`

- Si on supprime les flèches sortant de X, il n'y a plus de flèche entre X et Y : ils sont indépendants dans `G_{X̲}`.
- On applique la règle 2 : `P(y | do(x)) = P(y | x)`.
- **Succès** : on a remplacé un do par une quantité observée.

*Remarque du cours* : cela signifie que, dans ce modèle, tous les chemins entre X et Y sont des chemins directs, ceux qu'on veut inclure dans l'estimation.

#### (b) Un DAG avec un confounder Z : `X ← Z → Y` et `X → Y`

La classique situation de confusion. On conditionne sur Z pour rendre X et Y indépendants dans `G_{X̲}` :

```
P(y | do(x)) = Σ_z P(y | do(x), z) P(z)
             = Σ_z P(y | x, z) P(z)         (règle 2 : Y ⊥ X | Z)
```

**C'est la célèbre « formule d'ajustement »** : on calcule l'effet de X sur Y dans chaque strate de Z, puis on moyenne **en pondérant par la distribution de Z dans la population**.

*(C'est exactement ce qu'on a fait avec le paradoxe de Simpson : moyenne pondérée par 357/700 et 343/700.)*

#### (c) Un DAG avec un collider Z : `X → Z ← Y` (et `X → Y`)

Z est un collider : il ne faut **pas** le contrôler. Pas besoin de le faire : après suppression des flèches sortant de X, X et Y sont indépendants, donc on peut simplement utiliser les données observées `P(y | x)`.

#### (d) Un DAG avec un confounder **non mesuré** U (la porte d'entrée / front-door)

```
         U  (non observé)
       ⇡   ⇣
  X ──► Z ──► Y
```

Ici U est un confounder **non mesuré** de X et Y. Le critère back-door **ne peut pas** s'appliquer (on ne peut pas contrôler U). Mais on a un **médiateur Z** entre X et Y, et le do-calcul permet quand même d'identifier l'effet de X sur Y !

La démarche (diapos 131-132) :

**Étape 1.** On conditionne sur Z :
```
P(y | do(x)) = Σ_z P(y | do(x), z) P(z | do(x))            (1)
```

**Étape 2.** Deuxième facteur : `P(z | do(x)) = P(z | x)` (règle 2, car il n'y a pas de back-door entre X et Z : le chemin passant par U est bloqué par le collider `U → Y ← Z`).

**Étape 3.** Premier facteur : `P(y | do(x), z)`. On enchaîne les règles 2 et 3 (on remplace do(x) par do(z), puis on utilise le fait qu'il n'y a pas de chemin causal …), pour obtenir :
```
P(y | do(x), z) = Σ_x' P(y | x', z) P(x')
```
Ici, on contrôle X (observable) pour bloquer le chemin back-door `Z ← X ← U → Y`.

**Étape 4.** On combine :
```
P(y | do(x)) = Σ_z P(z | x) Σ_x' P(y | x', z) P(x')
```

> C'est le **critère de la porte d'entrée (front-door)**, ici obtenu comme conséquence du do-calcul. Tous les termes sont **observables** : on a éliminé le do **malgré un confounder non mesuré**.

### 7.5 Exemple du GPS (diapo 133)

**Question** : l'usage du GPS diminue-t-il la mémoire spatiale ?

Variables :
- X : usage du GPS ;
- Z : volume de l'hippocampe (médiateur) ;
- Y : mémoire ;
- U : motivation (confounder **non observé**).

La démarche en deux temps :
1. **Effet de X sur Z.** Il y a un chemin back-door `X ← U → Y ← Z`, mais il est **déjà bloqué** par le collider `U → Y ← Z`. Donc : `P(z | do(x)) = P(z | x)`.
2. **Effet de Z sur Y.** Il y a un chemin back-door ouvert `Z ← X ← U → Y`. U est inobservé, donc on ne peut pas le contrôler ; **mais** on peut contrôler X ! Donc :
   `P(y | do(z)) = Σ_x P(y | Z=z, X=x) P(x)`.
3. **On combine** :
   `P(y | do(x)) = Σ_z P(z | x) Σ_x' P(y | x', z) P(x')`.

**Moralité.** Même avec une variable confondante **inobservable** (la motivation), on peut quand même identifier l'effet causal grâce au médiateur.

### 7.6 Le théorème de complétude

> **Théorème.** Chaque fois qu'un effet causal est estimable à partir des données, **une séquence d'étapes utilisant les trois règles du do-calcul** permet d'éliminer l'opérateur do. On peut donc estimer des effets causaux à partir de données **observationnelles seules**.

**Intérêt.** Le do-calcul est **complet** : si le do-calcul échoue, alors aucune méthode ne peut identifier l'effet sans informations supplémentaires (ex. expérimentation).

---

## 8. La science de l'intervention et l'échelle de la causalité

### 8.1 Science des données classique vs. questions d'intervention

La science des données classique est **associationnelle** : à partir de données, on cherche `P(y | x)` ou une relation `y = h(x)`.

*Exemple* : « Quelle est la probabilité qu'un client qui achète du dentifrice achète aussi du fil dentaire ? »

**Mais** et si la question est :

> **Que se passera-t-il pour les ventes de fil dentaire si on double le prix du dentifrice ?**

C'est une **intervention**. Elle appelle un **nouveau type de connaissance, absent des données**.

**Pourquoi ?** Même si, dans le passé, le prix du dentifrice a déjà été le double d'aujourd'hui, le **contexte** était peut-être différent : par exemple, à cause d'une pénurie, le prix avait doublé **partout**. Alors que maintenant on veut savoir ce qui se passe si on le double **dans notre magasin seulement**. Les données passées reflètent un mécanisme (« pénurie générale ») différent de celui qu'on veut tester (« décision locale »).

> **Question centrale** : sous quelles conditions des **observations passives** seules permettent-elles de répondre à des questions sur des **interventions** ?

### 8.2 Un cadre formel

Le cours rappelle qu'on cherche un cadre **formel** pour raisonner sur les relations de cause à effet. Des études semi-formelles existent depuis des siècles, mais ce n'est que depuis **trois décennies** qu'une compréhension formelle et profonde émerge.

### 8.3 La causalité n'est pas réductible aux probabilités

**La vision des philosophes** : « X cause Y si X **augmente la probabilité** de Y » (ex. conduite imprudente → accidents).

**Pourquoi c'est trompeur.** Dire « X augmente la probabilité de Y » **n'est pas la même chose** que `P(Y | X) > P(Y)` (qui signifie seulement : *si je vois X, la probabilité de Y augmente*). Cette augmentation peut avoir d'autres origines :
- Y est cause de X (causalité inversée) ;
- une autre variable Z est cause à la fois de X et de Y ;
- etc.

> **Citation de Pearl (diapo 146)** : « Alors que les probabilités encodent nos croyances sur un monde statique, la causalité nous dit **si et comment les probabilités changent quand le monde change**, que ce soit par intervention ou par acte d'imagination. »

### 8.4 Intervenir ≠ observer (retour sur les glaces)

On veut l'effet causal de ICE sur ACC (glaces → accidents), avec la température (TMP) comme confounder :

```
P(ACC = acc | do(ICE = ice)) = Σ_tmp P(ACC = acc | ICE = ice, TMP = tmp) · P(TMP = tmp)
```

**Forme générale** (formule d'ajustement sur les parents de X, `Pa`) :
```
P(Y = y | do(X = x)) = Σ_z P(Y = y | X = x, Pa = z) · P(Pa = z)
```

**Conséquence** :
```
P(ACC = acc | do(ICE = ice))  ≠  P(ACC = acc | ICE = ice)
```
On a donc deux façons de poser la question et deux réponses différentes :
- **Voir** que les gens achètent des glaces → les accidents augmentent (corrélation).
- **Faire** acheter des glaces (par exemple en distribuant des glaces gratuites) → rien ne change pour les accidents.

### 8.5 L'échelle de la causalité de Pearl (diapos 148-149)

Trois niveaux (ou « barreaux »), de plus en plus riches :

| Niveau | Nom | Activité | Notation | Question typique | Exemple | Lien avec l'apprentissage |
|---|---|---|---|---|---|---|
| **1** | **Associationnel** | *Voir* | `P(y | x)` | « Qu'est-ce que c'est ? Comment voir X change-t-il ma croyance sur Y ? » | Que nous dit un symptôme sur la maladie ? | Apprentissage supervisé / non supervisé |
| **2** | **Interventionnel** | *Faire* | `P(y | do(x), c)` | « Et si… ? Et si je fais X ? » | Si je prends de l'aspirine, mon mal de tête sera-t-il guéri ? | Apprentissage par renforcement |
| **3** | **Contrefactuel** | *Imaginer* | `P(y_x | x', y')` | « Pourquoi ? Et si j'avais agi autrement ? » | Et si je n'avais pas pris ce médicament ? | – |

**Idée-clé** : on **ne peut pas** répondre à une question de niveau supérieur avec uniquement des données (ou un modèle) du niveau inférieur. Pour monter, il faut **des hypothèses causales** (le graphe).

*Rappel de la notation du niveau 3* : `y_x` désigne « la valeur que Y aurait prise si X avait été fixé à x », alors qu'on a en fait observé `x'` et `y'`. On pose donc des questions sur un **monde différent du monde réel**.

---

## 9. Conclusions

1. **Inférer des relations causales demande des connaissances au-delà des statistiques.**
2. Une approche possible est celle de **Pearl** : **graphes causaux + do-calcul**.
3. Dans certaines situations, on peut **estimer des effets causaux à partir de données observationnelles seules**.
4. Il existe une **échelle de la causalité** à **trois niveaux**, qui culmine avec les **contrefactuels** (non traités dans ce cours).
5. Les techniques d'**apprentissage** des relations causales (découverte causale) sont encore **à leurs débuts** (non traitées dans ce cours).

---

## 10. Glossaire et fiche de révision

### 10.1 Glossaire

| Terme | Définition simple |
|---|---|
| **Corrélation** | Lien statistique entre deux variables (souvent mesuré de façon linéaire). Ne dit rien sur la direction ni la cause. |
| **Confounder / facteur de confusion** | Cause commune de X et Y. Crée une corrélation spurieuse. **À contrôler.** |
| **Mediator / médiateur** | Variable située sur le chemin causal X → M → Y. **À ne pas contrôler** si on veut l'effet total. |
| **Collider** | Effet commun de deux variables (`A → B ← C`). **À ne pas contrôler** : l'ouvrir crée de la fausse dépendance. |
| **Explaining away** | Quand on connaît un effet, les causes possibles deviennent dépendantes (apprendre une cause réduit la plausibilité des autres). |
| **Paradoxe de Simpson** | Une tendance observée dans chaque sous-groupe s'inverse (ou disparaît) dans la population agrégée. |
| **Paradoxe de Lord** | Deux analyses statistiques « raisonnables » aboutissent à des conclusions opposées ; seul le graphe causal tranche. |
| **Résultats potentiels** `Y^{a}` | Valeur que Y prendrait si le traitement valait a (une seule est observée par individu). |
| **ITE** | Effet causal individuel : `Y_i^{a=1} − Y_i^{a=0}`. Inobservable. |
| **ATE** | Effet causal moyen : `E[Y^{a=1}] − E[Y^{a=0}]`. |
| **RCT** | Essai contrôlé randomisé : l'assignation aléatoire du traitement élimine la confusion. |
| **DAG / diagramme causal** | Graphe orienté acyclique où une flèche X → Y représente « X cause Y ». |
| **Opérateur do** | `do(X = x)` : forcer X à x par intervention (supprime les flèches entrantes dans X). |
| **Chemin back-door** | Chemin de X à Y commençant par une flèche **entrant** dans X. Source de confusion. |
| **Critère back-door** | Si un ensemble Z bloque tous les chemins back-door (sans contenir de descendant de X), on peut ajuster sur Z. |
| **Formule d'ajustement** | `P(y | do(x)) = Σ_z P(y | x, z) P(z)`. |
| **Critère front-door** | Permet d'identifier l'effet même avec un confounder non mesuré, via un médiateur. |
| **Do-calcul** | Trois règles de réécriture permettant d'éliminer le do d'une expression. |
| **Contrefactuel** | Question « et si les choses s'étaient passées autrement ? » portant sur un monde non réalisé. |
| **Variables instrumentales (IV)** | Autre méthode d'identification (citée mais non détaillée). |
| **Échelle de Pearl** | Associationnel (voir) < Interventionnel (faire) < Contrefactuel (imaginer). |

### 10.2 Les 10 idées à retenir absolument

1. **Corrélation ≠ causalité**, mais aussi **causalité ≠ corrélation linéaire**.
2. **Voir ≠ faire** : `P(y | x) ≠ P(y | do(x))` en général.
3. Une même table de données peut mener à des **décisions opposées** selon le graphe causal (Simpson, Lord, COVID-27).
4. **Il faut un modèle causal** (hypothèses) : les données seules ne suffisent pas.
5. **Trois briques** : chaîne et fourche (contrôler → **bloque**), collider (contrôler → **ouvre**).
6. **Confounder** → ajuster. **Médiateur** → ne pas ajuster (effet total). **Collider** → ne pas ajuster.
7. **Back-door** : bloquer tous les chemins qui commencent par une flèche entrant dans X, sans en ouvrir de nouveaux.
8. **Formule d'ajustement** : `P(y | do(x)) = Σ_z P(y | x, z) P(z)`.
9. Le **do-calcul** (3 règles) permet de savoir si un effet est identifiable à partir de données observationnelles ; il est **complet**.
10. **Échelle de la causalité** : voir → faire → imaginer.

### 10.3 Questions pour s'auto-évaluer

1. Pourquoi une corrélation de Pearl nulle n'implique-t-elle pas l'absence de causalité ?
2. Expliquez le problème de Monty Hall avec le vocabulaire des colliders. Que se passe-t-il si le présentateur ne sait pas où est la voiture ?
3. Dans le paradoxe de Simpson des calculs rénaux, pourquoi le traitement B paraît-il meilleur globalement ? Quel est le facteur de confusion ?
4. Pourquoi, dans l'exemple du médicament et de la pression artérielle, regarde-t-on les résultats agrégés et non stratifiés ?
5. Quelle est la différence entre un confounder, un médiateur et un collider ? Pour chacun, faut-il conditionner ?
6. Qu'est-ce qu'un chemin back-door ? Donnez la condition pour qu'un ensemble Z satisfasse le critère back-door.
7. Dans le paradoxe de Lord, pourquoi change-t-on de conclusion quand on passe de « effet du sexe » à « effet du régime » ?
8. Écrivez la formule d'ajustement et expliquez chaque terme.
9. Comment peut-on identifier l'effet d'un traitement quand un confounder est inobservable ? (front-door, exemple du GPS)
10. Placez sur l'échelle de Pearl : « Quel est le prix moyen des maisons proches d'une école ? », « Que se passerait-il si on doublait le prix du dentifrice ? », « Aurais-je guéri si je n'avais pas pris ce médicament ? ».

---

*Notes rédigées d'après les diapositives du cours « Causality: Inferencing and Learning » (A. Cornuéjols). Les parties non traitées dans le cours (cadre de Rubin, variables instrumentales, contrefactuels, apprentissage causal) sont simplement signalées comme telles.*
