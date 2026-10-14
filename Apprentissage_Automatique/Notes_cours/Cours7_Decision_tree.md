# Les arbres de décision expliqués pas à pas

> Basé sur le cours « Learning Decision Trees » d'A. Cornuéjols (AgroParisTech – INRAE MIA-Paris-Saclay, 72 diapos).
> Les renvois **(diapo N)** te permettent de retrouver le passage dans ton PDF.
> Les paragraphes marqués **➕ Complément** ajoutent une précision qui n'est pas explicitement sur les diapos.
> Les encadrés **⚠️ Point d'attention** signalent des coquilles ou des diapos inachevées (j'ai vérifié les chiffres par le calcul).
> Les formules sont en LaTeX (`$...$`) : elles s'affichent dans Obsidian, Typora, VS Code, Jupyter, etc.

---

## Sommaire

1. [La grande image](#1-la-grande-image)
2. [Apprentissage supervisé : vocabulaire](#2-apprentissage-supervisé--vocabulaire)
3. [Qu'est-ce qu'un arbre de décision ?](#3-quest-ce-quun-arbre-de-décision-)
4. [Apprendre un arbre : le problème de l'espace de recherche](#4-apprendre-un-arbre--le-problème-de-lespace-de-recherche)
5. [Choisir le meilleur attribut : l'entropie et le gain d'information](#5-choisir-le-meilleur-attribut--lentropie-et-le-gain-dinformation)
6. [Exemple complet : la pastèque mûre](#6-exemple-complet--la-pastèque-mûre)
7. [Un autre critère : l'indice de Gini](#7-un-autre-critère--lindice-de-gini)
8. [Problèmes pratiques et solutions](#8-problèmes-pratiques-et-solutions)
9. [Le surapprentissage et l'élagage](#9-le-surapprentissage-et-lélagage)
10. [Biais, variance et arbres obliques](#10-biais-variance-et-arbres-obliques)
11. [Bilan : avantages et inconvénients](#11-bilan--avantages-et-inconvénients)
12. [Les arbres de régression](#12-les-arbres-de-régression)
13. [Conclusions](#13-conclusions)
14. [Fiche récapitulative](#14-fiche-récapitulative)
15. [Quiz d'auto-évaluation](#15-quiz-dauto-évaluation)

---

## 1. La grande image

Un arbre de décision, c'est **un jeu de « questions-réponses » enchaînées** qui conduit à une décision. C'est exactement le jeu des « 20 questions » : on pose une question (« est-ce un animal ? »), la réponse oriente la question suivante, et ainsi de suite jusqu'à pouvoir conclure.

Tout le cours tient en quatre idées :

1. Un **arbre** = une suite de tests sur les attributs, qui aboutit à une classe (ou, pour les arbres de régression, à une valeur).
2. **Apprendre** un arbre, c'est choisir *quelles questions poser et dans quel ordre*. Comme il y a un nombre gigantesque d'arbres possibles, on le construit **glouton­nement**, de haut en bas : à chaque étape, on pose la question la plus « informative ».
3. On mesure le caractère informatif d'une question par la **baisse d'impureté** qu'elle provoque (entropie → *gain d'information* ; ou indice de Gini).
4. Un arbre qui colle trop aux données **surapprend** : on le raccourcit par **élagage** (*pruning*). Et il existe une variante pour prédire des **nombres** : les arbres de régression.

Plan du cours (diapo 4) : arbres de décision → apprentissage → élagage → biais et arbres obliques → arbres de régression.

---

## 2. Apprentissage supervisé : vocabulaire

*(diapos 2 et 3)*

**Le problème (diapo 2).** On dispose :
- d'un **échantillon d'apprentissage** $S=\{(x_i,y_i)\}_{i=1..m}$ ;
- de connaissances préalables sur le type de dépendance entre $x$ et $y$.

On cherche une **fonction** $h$ qui prédit $y$ pour une **nouvelle** entrée $x$.

**Organisation des données (diapo 3).** Un tableau : chaque **ligne** est un **exemple** (ou *instance*), chaque **colonne** est un **attribut** (*feature*, descripteur), et la dernière colonne est l'**étiquette** (*label*, ici « À prospecter ? OUI / NON »).

| Identifieur | Genre | Âge | Niveau d'études | Marié ? | Nb enfants | Revenu | Profession | **À prospecter ?** |
|---|---|---|---|---|---|---|---|---|
| I_21 | M | 43 | Bac+5 | Oui | 3 | 55 000 | Architecte | **OUI** |
| I_34 | M | 25 | Bac+2 | Non | 0 | 21 000 | Infirmier | **NON** |

Les attributs peuvent être de **types différents** : nominaux (genre, profession), ordinaux (niveau d'études), numériques (âge, revenu). Retiens ce point : les arbres savent les traiter **tous ensemble** (section 11).

---

## 3. Qu'est-ce qu'un arbre de décision ?

*(diapos 6 à 12)*

### 3.1 La structure (diapos 6-7)

- Chaque **nœud interne** = un **test sur un attribut** (« la douleur est-elle au ventre, à la gorge, à la poitrine ? »).
- Chaque **branche** = une **valeur** possible de l'attribut testé.
- Chaque **feuille** = une **classe** (une décision : « appendicite », « grippe », « rhume »…).

> 💡 **Analogie.** C'est un organigramme de diagnostic, comme celui d'un médecin : *douleur au ventre → appendicite ; douleur à la gorge → fièvre ? → oui : grippe, non : angine*. Pour classer un nouvel exemple, on part de la **racine**, on suit les branches correspondant à ses valeurs d'attributs, et on lit la classe de la feuille atteinte.

### 3.2 Un arbre réel (diapo 9)

La diapo montre un arbre appris sur **1000 patients** (jeu de données médical, la cible est un risque obstétrical : probablement le risque de césarienne d'urgence, à confirmer d'après le contexte du cours). Sur chaque nœud, on lit les effectifs « +833 +167 » (nombre d'exemples de chaque classe) et les proportions correspondantes (0,83 / 0,17).

Lecture : le nœud racine contient 833 « + » et 167 « − ». En testant `fetal_presentation`, la population se scinde ; la branche `= 1` est plus « pure » (0,876 de +) que la racine, tandis que `= 2` ou `= 3` donne des nœuds majoritairement de l'autre classe, et se retrouve donc **étiquetée « 1 »** (dernière colonne).

> 💡 **À retenir.** Dans un arbre appris, chaque nœud contient les **effectifs des classes** des exemples qui y sont parvenus. Une feuille est « pure » si tous ses exemples ont la même classe ; sinon on prédit la **classe majoritaire**.

### 3.3 Ce qu'un arbre sait représenter (diapo 10)

- **Toute fonction booléenne** peut être représentée par un arbre de décision. Rappel (diapo 10) : avec 6 attributs booléens il y a $2^6=64$ exemples possibles et environ $2\cdot10^{19}$ fonctions booléennes ($2^{64}$) !
- Mais certaines fonctions exigent des **arbres exponentiellement grands** (la parité, la majorité), alors que d'autres tiennent en **un seul nœud**.
- Le formalisme est celui de la **logique propositionnelle** : on ne peut pas exprimer de relations entre objets (« X est plus grand que Y »).
- **Un arbre = une disjonction de règles** (un « OU » de « ET »). Chaque chemin racine → feuille est une règle :

```
(si plumes = non                      alors oiseau = non)
OU (si plumes = oui ET couleur = marron   alors oiseau = non)
OU (si plumes = oui ET couleur = noir&blanc  alors oiseau = oui)
OU (si plumes = oui ET couleur = jaune    alors oiseau = oui)
```

> 💡 **Pourquoi c'est important.** Cette lecture « en règles » est la raison pour laquelle les arbres sont réputés **interprétables** : on peut expliquer à un humain pourquoi une décision a été prise.

### 3.4 Exemples d'application (diapos 11-12)

La détection de **spam** (est-ce un courriel indésirable ?) et un travail sur la **différence d'appréciation des risques d'intoxication** des deux côtés de l'Atlantique (projet ANR Holyrisk), où l'on cherche à distinguer deux classes de textes à partir de leur contenu.

---

## 4. Apprendre un arbre : le problème de l'espace de recherche

*(diapos 14 à 18)*

### 4.1 Plusieurs arbres sont cohérents avec les données (diapo 15)

Prenons un mini jeu de données sur des oiseaux (couleur, ailes, plumes, sonar → oiseau ou pas). La diapo montre **quatre arbres différents** (DT1 à DT4) tous parfaitement cohérents avec les exemples.

C'est la même situation que pour le perceptron avec une infinité d'hyperplans séparateurs : **les données ne suffisent pas à désigner un arbre unique**. Il faut une **préférence**, un *biais* inductif. Pour les arbres, la préférence usuelle est : **les arbres les plus petits** (rasoir d'Ockham : l'explication la plus simple est la meilleure).

### 4.2 L'espace de recherche est gigantesque (diapos 16-17)

Le nombre d'arbres binaires à $n$ nœuds est le **nombre de Catalan** :

| $n$ | Nombre d'arbres binaires |
|---|---|
| 10 | **16 796** |
| 20 | **6,56 × 10⁹** (plus de 6 milliards) |

Et encore, c'est sans compter le choix de l'attribut testé à chaque nœud. **Explorer tous les arbres pour trouver le plus petit est impossible.**

### 4.3 La solution : une stratégie gloutonne descendante (diapo 18)

On construit l'arbre **de haut en bas** (*top-down*), de façon **gloutonne** (*greedy*) : à chaque étape, on prend la **meilleure décision locale**, sans jamais la remettre en cause.

1. **Choisir le meilleur attribut** à tester.
2. **Faire croître l'arbre** : créer une branche par valeur de cet attribut.
3. On se retrouve avec des **sous-ensembles** d'exemples aux feuilles.
4. **Recommencer** à l'étape 1 sur chaque sous-ensemble, jusqu'à un **critère d'arrêt** (par exemple : feuille « pure », plus d'attribut, plus assez d'exemples).

> 💡 **Analogie.** Tu cherches ton chemin dans une ville inconnue. Plutôt que de calculer tous les trajets possibles, à chaque carrefour tu prends la rue qui te rapproche le plus de ta destination. Rapide, mais tu peux rater le meilleur trajet global (c'est le côté **myope**, voir section 11).

**La question centrale devient donc : comment mesurer qu'un attribut est « bon » ?** → section 5.

---

## 5. Choisir le meilleur attribut : l'entropie et le gain d'information

*(diapos 19 à 25)*

### 5.1 L'intuition : on veut des sous-populations « pures » (diapos 19-21)

Un bon test est un test qui **sépare bien les classes** : après la question, chaque branche doit contenir des exemples de **plus en plus homogènes**. Idéalement, après le test, chaque branche est une feuille pure (« toutes les pastèques de cette branche sont mûres »).

Il nous faut donc une mesure de **désordre** (ou d'**impureté**) d'un ensemble d'exemples :
- **zéro** si tous les exemples sont de la même classe (ordre total) ;
- **maximale** si les classes sont **mélangées à parts égales** (désordre total).

### 5.2 L'entropie de Shannon (diapos 22-24)

Introduite par Shannon en 1949 (inspirée de l'entropie de Boltzmann en physique), elle mesure la **quantité d'information**, c'est-à-dire le nombre de bits nécessaires pour spécifier la distribution :

$$I(S)=-\sum_{i=1}^{C}p(c_i)\,\log_2 p(c_i)$$

où $p(c_i)$ est la proportion d'exemples de la classe $c_i$ dans $S$.

Propriétés (diapo 23) :
- **Nulle** s'il n'y a qu'une classe (pas d'incertitude) ;
- **croissante** quand les classes deviennent plus équiprobables ;
- **égale à $\log_2 k$** quand les $k$ classes sont équiprobables ;
- l'unité est le **bit**.

**Cas de deux classes (diapo 24).** Avec $P=p/(p+n)$ la proportion de « + » :

$$I(S)=-P\log_2 P-(1-P)\log_2(1-P)$$

C'est une **courbe en cloche** : 0 pour $P=0$ ou $P=1$, **maximale (1 bit) pour $P=0{,}5$**.

| Composition | Entropie | Interprétation |
|---|---|---|
| 8 « + », 0 « − » | 0 | Pur, aucune incertitude |
| 6 « + », 2 « − » | 0,811 | Plutôt ordonné |
| 4 « + », 4 « − » | **1** | Désordre maximal |

> 💡 **Analogie du jeu de devinette.** Si tu sais que l'urne contient 100 % de boules rouges, tu n'as aucune surprise en tirant : entropie 0. Si elle contient moitié rouge, moitié bleu, chaque tirage est une vraie surprise : entropie 1 bit (une question oui/non pour lever l'incertitude).

### 5.3 Le gain d'information (diapo 25)

Pour évaluer un attribut $A$, on compare le désordre **avant** et **après** l'avoir testé :

$$\boxed{Gain(S,A)=I(S)-\sum_{v\in valeurs(A)}\frac{|S_v|}{|S|}\;I(S_v)}$$

- $I(S)$ : l'entropie de la population de départ ;
- $S_v$ : le sous-ensemble des exemples pour lesquels $A=v$ ; $|S_v|/|S|$ est son **poids** ;
- la somme est donc l'**entropie moyenne après la séparation**, pondérée par la taille des branches.

**Lecture.** Le gain mesure *dans quelle mesure connaître la valeur de $A$ apporte de l'information sur la classe d'un exemple* (diapo 25). Gain élevé = la question réduit fortement l'incertitude → **on choisit l'attribut de gain maximal**.

> 💡 **Pourquoi pondérer par $|S_v|/|S|$ ?** Une branche de 2 exemples pèse peu : qu'elle soit pure ne prouve pas grand-chose. Une branche de 100 exemples pure, c'est une vraie réussite. La pondération donne à chaque branche l'importance qu'elle mérite.

---

## 6. Exemple complet : la pastèque mûre

*(diapos 14, 26 à 31)*

### 6.1 Les données

17 pastèques décrites par 6 attributs, étiquetées « mûre » (true) ou non (false) : **8 mûres, 9 non mûres**.

| ID | color | root | sound | texture | umbilicus | surface | **ripe** |
|---|---|---|---|---|---|---|---|
| 1 | green | curly | muffled | clear | hollow | hard | **true** |
| 2 | dark | curly | dull | clear | hollow | hard | true |
| 3 | dark | curly | muffled | clear | hollow | hard | true |
| 4 | green | curly | dull | clear | hollow | hard | true |
| 5 | light | curly | muffled | clear | hollow | hard | true |
| 6 | green | slightly curly | muffled | clear | slightly hollow | soft | true |
| 7 | dark | slightly curly | muffled | slightly blurry | slightly hollow | soft | true |
| 8 | dark | slightly curly | muffled | clear | slightly hollow | hard | true |
| 9 | dark | slightly curly | dull | slightly blurry | slightly hollow | hard | **false** |
| 10 | green | straight | crisp | clear | flat | soft | false |
| 11 | light | straight | crisp | blurry | flat | hard | false |
| 12 | light | curly | muffled | blurry | flat | soft | false |
| 13 | green | slightly curly | muffled | slightly blurry | hollow | hard | false |
| 14 | light | slightly curly | dull | slightly blurry | hollow | hard | false |
| 15 | dark | slightly curly | muffled | clear | slightly hollow | soft | false |
| 16 | light | curly | muffled | blurry | flat | hard | false |
| 17 | green | curly | dull | slightly blurry | slightly hollow | hard | false |

### 6.2 Étape 1 : entropie de départ (diapo 27)

$$I(D)=-\Big(\tfrac8{17}\log_2\tfrac8{17}+\tfrac9{17}\log_2\tfrac9{17}\Big)=\mathbf{0{,}998\ bit}$$

Presque 1 : les deux classes sont presque à égalité, le désordre est maximal.

### 6.3 Étape 2 : gain pour l'attribut « color » (diapo 28)

« color » a 3 valeurs, donc 3 sous-ensembles :

| Sous-ensemble | Composition | Effectif | Entropie |
|---|---|---|---|
| $D^1$ (green) | 3 mûres, 3 non mûres | 6 | $-\big(\tfrac36\log_2\tfrac36+\tfrac36\log_2\tfrac36\big)=1{,}000$ |
| $D^2$ (dark) | 4 mûres, 2 non mûres | 6 | 0,918 |
| $D^3$ (light) | 1 mûre, 4 non mûres | 5 | 0,722 |

$$Gain(D,color)=0{,}998-\Big(\tfrac6{17}\times1{,}000+\tfrac6{17}\times0{,}918+\tfrac5{17}\times0{,}722\Big)=0{,}998-0{,}889\approx\mathbf{0{,}109}$$

*(Avec les valeurs non arrondies on trouve 0,108 ; la différence vient simplement des arrondis intermédiaires de la diapo.)*

### 6.4 Étape 3 : gains des autres attributs (diapo 29)

| Attribut | Gain |
|---|---|
| color | 0,109 |
| root | 0,143 |
| sound | 0,141 |
| **texture** | **0,381** ← le meilleur |
| umbilicus | 0,289 |
| surface | 0,006 |

**Texture** gagne largement : elle devient la **racine**. Trois branches : *clear* {1,2,3,4,5,6,8,10,15} (9 exemples), *slightly blurry* {7,9,13,14,17} (5), *blurry* {11,12,16} (3). La branche *blurry* est déjà pure (toutes non mûres), c'est une feuille.

> 💡 **Interprétation.** Si une pastèque a une texture « blurry », on est sûr à 100 % qu'elle n'est pas mûre : voilà un test très informatif. À l'inverse, « surface » (gain 0,006) ne sépare presque rien : poser cette question en premier ferait perdre du temps.

### 6.5 Étape 4 : on recommence dans chaque branche (diapo 30)

On répète le procédé sur la branche *clear* ($D^1$ : 7 mûres, 2 non mûres). Gains calculés **uniquement sur ces 9 exemples** :

| Attribut | color | root | sound | umbilicus | surface |
|---|---|---|---|---|---|
| Gain($D^1$, ·) | 0,043 | **0,458** | 0,331 | **0,458** | **0,458** |

Trois attributs sont ex æquo : **root, surface et umbilicus**. « N'importe lequel peut être choisi » (diapo). Le résultat dépend donc du choix de départage : **il n'y a pas un unique arbre « appris »**.

> ⚠️ **Point d'attention.** Sur la diapo 30, la liste de $D^1$ indique « (8, −) ». C'est une coquille : d'après le tableau, la pastèque 8 est **mûre** (+). Les gains affichés (0,458…) ne sont corrects que si 8 est « + » ; je l'ai vérifié par le calcul.

### 6.6 Étape 5 : l'arbre final (diapo 31)

On obtient un arbre à plusieurs niveaux : racine **texture** ; sous *clear* on teste **root** (curly → mûre, straight → non mûre, slightly curly → on teste **color**, puis **surface** pour *dark*) ; sous *slightly blurry* on teste **surface** (hard → non mûre, soft → mûre) ; *blurry* → non mûre.

Chaque feuille est une règle lisible, par exemple : « *texture = clear ET root = curly → mûre* ».

> 💡 **Remarque.** L'arbre est complet : il classe **parfaitement** les 17 exemples d'apprentissage. Est-ce une bonne chose ? Pas forcément… c'est le sujet de la section 9.

---

## 7. Un autre critère : l'indice de Gini

*(diapo 32)*

Même objectif que l'entropie (mesurer l'impureté), formule plus simple, proposée par Breiman et al. (1984) :

$$Gini(D)=1-\sum_{j=1}^{k}p_j^2$$

Les exigences sont les mêmes (diapo 32) :
- **zéro** si la sous-population est homogène (une seule classe : $1-1^2=0$) ;
- **maximal** si les classes sont mélangées à parts égales (pour 2 classes, $1-0{,}5^2-0{,}5^2=0{,}5$).

**Exemple sur la pastèque.** $Gini=1-(\tfrac8{17})^2-(\tfrac9{17})^2\approx\mathbf{0{,}498}$.

| Critère | Formule | Valeur max (2 classes) | Calcul |
|---|---|---|---|
| Entropie | $-\sum p\log_2p$ | 1 | Utilise un logarithme |
| Gini | $1-\sum p^2$ | 0,5 | Seulement des carrés (plus rapide) |

➕ **Complément.** En pratique, entropie et Gini donnent le plus souvent des arbres très semblables ; Gini est un peu moins coûteux à calculer. De la même façon qu'avec l'entropie, on choisit l'attribut qui **réduit le plus** l'impureté moyenne pondérée.

---

## 8. Problèmes pratiques et solutions

*(diapos 33 à 36)*

### 8.1 Les attributs continus : on les discrétise (diapo 34)

Comment tester un attribut numérique (température, âge, revenu) alors qu'il y a une infinité de valeurs ? On le **coupe en deux** par un **seuil** : le test devient « $Temp>s$ ? ».

**Quels seuils essayer ?** Seulement ceux situés **entre deux exemples consécutifs de classes différentes** (c'est là qu'une coupure peut améliorer la pureté). Exemple de la diapo (jouer au golf ?) :

| Temp. | 6°C | 8°C | 14°C | 18°C | 20°C | 28°C | 32°C |
|---|---|---|---|---|---|---|---|
| Golf ? | Non | Non | Non | Oui | Oui | Oui | Non |

Seuls deux seuils sont candidats : **16°C** (entre 14 « Non » et 18 « Oui ») et **30°C** (entre 28 « Oui » et 32 « Non »).

| Seuil | Coupe | Gain |
|---|---|---|
| **Temp > 16°C** | {6,8,14} pur (Non) / {18,20,28,32} : 3 Oui, 1 Non | **0,522** |
| Temp > 30°C | {6…28} : 3 Oui, 3 Non / {32} pur | 0,128 |

→ **Temp > 16°C** est le plus informatif, il est choisi (conformément à la diapo). *(Calcul : $I(S)=0{,}985$ ; après le seuil 16°C : $\frac47\times0{,}811=0{,}463$ ; gain = 0,985 − 0,463 = 0,522.)*

> 💡 **Pourquoi seulement ces seuils ?** Couper entre deux exemples de la même classe ne peut jamais aider : on sépare des exemples qui vont dans la même feuille de toute façon.

### 8.2 Les attributs à nombreuses valeurs : un biais du gain (diapo 35)

**Le problème :** le gain d'information **favorise indûment les attributs qui ont beaucoup de valeurs**.

> 💡 **Exemple extrême.** Ajoute à la pastèque un attribut « identifiant » (1, 2, …, 17). Il crée 17 branches d'un seul exemple chacune, toutes **parfaitement pures**. Son gain est donc maximal ($0{,}998$). Pourtant c'est un attribut **parfaitement inutile** pour prédire une nouvelle pastèque !

**Deux solutions (diapo 35) :**
1. **Binariser tous les attributs** (chaque test n'a que 2 branches). Défaut : l'arbre perd en **interprétabilité**.
2. **Normaliser le gain** par un facteur de correction (le « rapport de gain » du C4.5) :

$$Gain\_norm(S,A)=\frac{Gain(S,A)}{-\sum_{i=1}^{nb\ valeurs}\frac{|S_i|}{|S|}\log_2\frac{|S_i|}{|S|}}$$

Le dénominateur est l'**entropie de la partition** elle-même : elle est **grande** quand l'attribut éclate les données en beaucoup de petites branches, ce qui **pénalise** ces attributs.

➕ **Sur l'exemple de l'identifiant :** avant normalisation, l'identifiant (gain 0,998) écrase la texture (gain 0,381). Après normalisation :
- identifiant : dénominateur $=\log_2 17\approx4{,}09$, donc $0{,}998/4{,}09\approx\mathbf{0{,}24}$ ;
- texture (branches de 9, 5 et 3 exemples) : dénominateur $\approx1{,}45$, donc $0{,}381/1{,}45\approx\mathbf{0{,}26}$.

La texture repasse devant l'identifiant : la normalisation a bien corrigé le biais.

> ⚠️ **Point d'attention.** La formule de la diapo 35 n'affiche pas le signe « − » devant la somme du dénominateur ; sans lui, le dénominateur serait négatif. J'ai rétabli le signe.

### 8.3 Les valeurs manquantes (diapo 36)

Que faire si un exemple n'a pas de valeur pour l'attribut $A$ ? Trois pistes :

1. Prendre la **valeur la plus fréquente** de $A$ dans $S$ ;
2. Prendre la **valeur la plus fréquente dans le nœud** ;
3. **Répartir l'exemple** en exemples fictifs, un par valeur possible de $A$, **pondérés par leurs fréquences**. Exemple de la diapo : si 6 exemples du nœud ont $A=a_1$ et 4 ont $A=a_2$, alors l'exemple manquant devient « $a_1$ avec probabilité 0,6 » et « $a_2$ avec probabilité 0,4 ». En prédiction, on donne l'étiquette de la **feuille la plus probable**.

---

## 9. Le surapprentissage et l'élagage

*(diapos 38 à 47)*

### 9.1 Évaluer la généralisation (diapo 38)

Avoir appris un arbre qui classe bien les exemples d'apprentissage ne dit rien de sa **qualité sur de nouvelles données**. On évalue donc la généralisation par :
- un **ensemble test** séparé ;
- la **validation croisée** ;
- le « **leave-one-out** » (validation croisée à $m$ plis, un exemple laissé de côté à chaque fois).

On suit aussi la **courbe d'apprentissage** (la performance en fonction du nombre d'exemples).

### 9.2 Le surapprentissage (diapos 39 à 41)

**Les causes : le bruit.** Types de bruit :
- **erreurs de description** (un attribut mal mesuré) ;
- **erreurs de classification** (une étiquette fausse) ;
- **« clashes »** : deux exemples avec la **même description mais des classes différentes** ;
- **valeurs manquantes**.

**L'effet :** un arbre **trop développé** (« touffu », trop profond), qui crée des feuilles pour expliquer jusqu'au bruit.

La courbe de la diapo 40 (Mitchell) l'illustre : quand le nombre de nœuds augmente, l'erreur sur l'**ensemble d'apprentissage diminue sans cesse**, mais l'erreur sur l'**ensemble test remonte** après un certain point.

> 💡 **Analogie.** Un élève qui apprend par cœur les corrigés (y compris les fautes de frappe du prof) a 20/20 aux exercices connus, mais échoue dès que la question change. L'arbre trop profond est cet élève : il a mémorisé le bruit.

**Les deux stratégies d'élagage (diapo 42) :**
- **Pré-élagage** (*pre-pruning*) : on **arrête** la croissance de l'arbre **avant** qu'il soit complet ;
- **Post-élagage** (*post-pruning*) : on **laisse l'arbre croître en entier**, puis on le **coupe** ensuite.

### 9.3 Mise en place : un jeu d'apprentissage et un jeu de validation (diapo 43)

On divise la base de la pastèque en deux :
- **Apprentissage** : exemples {1, 2, 3, 6, 7, 10, 14, 15, 16, 17} (5 mûres, 5 non mûres) ;
- **Validation** : exemples {4, 5, 8, 9, 11, 12, 13} (3 mûres, 4 non mûres).

**Principe :** on construit l'arbre avec l'ensemble d'apprentissage, et on **décide** de garder ou non un test selon la **précision sur l'ensemble de validation** (données que l'arbre n'a pas vues).

### 9.4 Le pré-élagage (diapos 44 à 46)

**Question.** Au nœud racine, le meilleur attribut est « umbilicus ». Faut-il vraiment faire cette séparation ?

**Avant la séparation** : on n'a qu'une feuille. Sa classe majoritaire sur l'apprentissage est « mûre » (égalité 5/5, départagée en faveur de « mûre »). Sur la validation, on prédit « mûre » pour tous : {4, 5, 8} sont bien classés, {9, 11, 12, 13} mal classés → précision $\tfrac37\approx\mathbf{42{,}9\%}$.

**Après la séparation** (3 branches) :

| Branche | Apprentissage | Majorité | Validation | Prédiction |
|---|---|---|---|---|
| hollow | {1,2,3,14} | mûre | {4,5,13} | 4 ✔, 5 ✔, 13 ✘ |
| slightly hollow | {6,7,15,17} | mûre (égalité) | {8,9} | 8 ✔, 9 ✘ |
| flat | {10,16} | non mûre | {11,12} | 11 ✔, 12 ✔ |

Bien classés : {4, 5, 8, 11, 12} → $\tfrac57\approx\mathbf{71{,}4\%}$ **> 42,9 %**. La séparation améliore la validation → **on la garde**.

On teste ensuite de **séparer les feuilles** suivantes (par « color » puis « root ») : la précision de validation passe de 71,4 % à 57,1 %, donc **on ne sépare pas** (diapo 46). L'arbre final pré-élagué est une **« souche de décision »** (*decision stump*) : **un seul nœud de test**.

| | Avantage | Inconvénient |
|---|---|---|
| **Pré-élagage** | Rapide : on évite de construire des sous-arbres inutiles | **Risque de sous-apprentissage** : une séparation qui n'améliore rien *maintenant* peut devenir utile *après* d'autres tests (même problème de myopie que le XOR) |

### 9.5 Le post-élagage (diapo 47)

On fait d'abord croître l'**arbre complet** (ici, sa précision de validation est de $\mathbf{42{,}9\%}$), puis on examine les nœuds de test **en partant des plus profonds** (ici le nœud 6).

**Principe.** Pour chaque nœud, on se demande : *et si je remplaçais ce sous-arbre par une feuille* (de classe majoritaire) *?* Si la **précision de validation augmente** (ou ne baisse pas), on **élague**.

Ici, élaguer le sous-arbre du nœud 6 le remplace par une feuille « mûre » (il contenait {7, 15}). La précision de validation passe de 42,9 % à **57,1 %** : on élague.

| | Avantage | Inconvénient |
|---|---|---|
| **Post-élagage** | Plus d'information disponible (on a vu l'arbre complet), donc **moins de risque de sous-apprentissage** ; généralement de meilleures performances | **Plus coûteux** : il faut d'abord construire l'arbre complet |

> 💡 **Fil rouge.** L'élagage est de la **régularisation** (comme dans le cours sur les SVM) : on paie le prix de la complexité (taille de l'arbre) pour mieux généraliser.

---

## 10. Biais, variance et arbres obliques

*(diapos 49 à 57)*

### 10.1 Un cas test : deux triangles (diapos 49 à 51)

Un problème en 2D : la classe vaut « bleu » si $x_2>2x_1$ (pour $x_1<0{,}5$) ou si $x_2>2-2x_1$ (pour $x_1\ge0{,}5$), « rouge » sinon. Géométriquement, les rouges sont **à l'intérieur d'un triangle**, les bleus à l'extérieur.

On apprend un arbre sur un petit échantillon d'apprentissage (100 points environ). L'arbre appris a 5 feuilles et découpe l'espace en **zones rectangulaires** (les coupures d'un arbre sont toujours **parallèles aux axes**). Sur le grand jeu de test, l'erreur est $\varepsilon=\mathbf{0{,}1632}$ : l'arbre approche le triangle par des rectangles, mais n'y parvient pas parfaitement.

> ⚠️ **Point d'attention.** Les diapos 50 et 51 contiennent des « toto » : ce sont des diapos de travail non terminées. L'idée à retenir est la légende de la diapo : **« Arbre profond : biais faible, variance forte ; arbre court : biais fort, variance faible. »**

### 10.2 Les deux composantes de l'erreur (diapo 52)

| Composante | Définition | Exemple de la diapo |
|---|---|---|
| **Biais** | L'**incapacité du modèle** à représenter le concept (la « vraie » fonction qui relie $Y$ à $X$) | Un classifieur **linéaire** ne peut pas séparer les points du triangle par une droite : même avec une infinité de données, il se trompera |
| **Variance** | La **sensibilité aux fluctuations d'échantillonnage** | Avec un faible échantillon, impossible de trouver exactement les « bonnes » frontières : un autre échantillon donnerait un autre arbre |

> 💡 **Analogie du tir à l'arc.** Le **biais**, c'est quand toutes tes flèches tombent loin du centre, toutes au même endroit (tu vises mal de façon systématique). La **variance**, c'est quand tes flèches sont éparpillées (tu es instable d'un tir à l'autre). Un bon modèle a peu des deux.

**Le compromis, pour les arbres (diapo 50) :**

| | Biais | Variance | Risque |
|---|---|---|---|
| **Arbre profond** (très détaillé) | Faible | **Forte** | Surapprentissage |
| **Arbre court** | **Fort** | Faible | Sous-apprentissage |

L'**élagage** sert précisément à régler ce curseur.

### 10.3 Les arbres obliques (diapos 53 à 57)

**Le problème.** Un arbre classique ne teste qu'**un attribut à la fois** : ses frontières sont **parallèles aux axes**. Pour approcher une frontière **diagonale**, il lui faut une « **escalier** » de nombreux petits tests (diapo 54 : plusieurs tests sur $x_1$ et $x_2$ pour suivre une diagonale).

**La solution : les arbres obliques.** Un nœud peut tester une **combinaison linéaire** des attributs, par exemple $1{,}1\,x_1+x_2<0{,}2$. La frontière est alors **un hyperplan oblique** (comme dans un perceptron !) au lieu d'une coupure alignée sur un axe.

**Exemple sur la pastèque avec deux attributs (densité, sucre) (diapos 55-57) :**
- L'**arbre classique** a besoin de **4 tests** successifs (sugar ≤ 0,126 ? ; density ≤ 0,381 ? ; sugar ≤ 0,205 ? ; density ≤ 0,560 ?) et sa frontière est un **escalier** dans le plan ;
- L'**arbre oblique** n'en a besoin que de **2**, du type $-0{,}800\times density-0{,}044\times sugar\le-0{,}313$, et sa frontière est faite de **droites inclinées**, bien plus naturelles ici.

| | Arbre classique | Arbre oblique |
|---|---|---|
| Frontières | Parallèles aux axes | Obliques (hyperplans quelconques) |
| Taille de l'arbre | Plus grand | Plus petit |
| Lisibilité des tests | **Très simple** (« âge > 30 ») | Moins lisible (combinaison de variables) |
| Coût d'apprentissage | Faible | **Plus élevé** (il faut apprendre les coefficients) |

---

## 11. Bilan : avantages et inconvénients

*(diapo 58)*

### Avantages
- **Interprétables** : les branches se lisent comme des **règles** ; la **sélection de variables « pertinentes »** est **automatique** (les attributs inutiles ne sont jamais choisis).
- **Non paramétriques** :
  - traitement **indifférencié** des différents types de variables (nominales, ordinales, numériques) ;
  - **robustes** face aux **données aberrantes** (un point extrême change juste de côté d'une coupure) ;
  - existence de **solutions pour les données manquantes** (section 8.3).
- **Complexité calculatoire faible** : $O(A^2\cdot m)$ avec $A$ attributs et $m$ exemples (diapo 71).

### Inconvénients
- **Instabilité sur les petites bases** : avec peu d'exemples, les feuilles ont de très petits effectifs ; une légère variation de l'échantillon peut donner un arbre très différent (variance forte).
- **Méthode gloutonne et myope** : le choix à chaque nœud ne regarde que le gain **immédiat**. Il est donc mal armé pour repérer des **interactions** entre variables, comme le **XOR**.

> 💡 **Le piège du XOR.** Soit $y=x_1\oplus x_2$ (vrai si exactement un des deux est vrai) avec les 4 exemples : $(0,0)\to0$, $(0,1)\to1$, $(1,0)\to1$, $(1,1)\to0$. Calcule le gain de $x_1$ : après le test, chaque branche contient un « 0 » et un « 1 » donc entropie 1, soit **gain = 0**. Idem pour $x_2$. Aucun attribut n'est informatif **pris seul**, alors que les deux **ensemble** déterminent totalement la classe ! Un algorithme glouton ne voit aucune raison de préférer un attribut à l'autre.

---

## 12. Les arbres de régression

*(diapos 60 à 69)*

### 12.1 Pourquoi ne pas se contenter de la régression classique ? (diapos 60-61)

Les méthodes classiques (régression linéaire simple, régression multiple, régression non linéaire) définissent **un modèle global** sur *tout* l'espace. Pour une relation non linéaire en grande dimension, il faudrait explorer **toutes les combinaisons** d'attributs : un espace de recherche énorme.

L'idée des arbres de régression : **partitionner l'espace** en régions et utiliser un **modèle local simple** dans chacune (diapo 61).

> 💡 **Analogie.** Plutôt que de dessiner une seule courbe compliquée pour toute la France, on découpe le pays en régions et on donne dans chacune un prix moyen simple. L'ensemble approxime bien la courbe complète.

### 12.2 Particularités (diapo 64)

- Les attributs **et la cible** peuvent être à **valeur continue** ;
- à chaque **région** $R_k$ de l'espace $X$, on associe une **valeur constante** $c_k$ (la prédiction de la feuille) ;
- on cherche à **minimiser l'erreur quadratique** :

$$MSE=\sum_{i=1}^{m}\sum_{k=1}^{K}\mathcal{I}(x_i\in R_k)\,(y_i-c_k)^2$$

où $\mathcal I(\cdot)$ vaut 1 si l'exemple $i$ tombe dans la région $R_k$, 0 sinon. Pour une région donnée, la constante qui minimise l'erreur quadratique est **la moyenne** des $y_i$ de cette région. **La feuille prédit donc la moyenne des cibles de ses exemples.**

### 12.3 Construction de l'arbre (diapos 65-66)

C'est la **même stratégie gloutonne** que pour la classification, avec un autre critère :
- **Choix** de l'**attribut** et du **point de division** qui minimisent la **somme des écarts quadratiques à la moyenne** dans chacune des régions créées ;
- **Arrêt** quand : il n'y a **plus assez de points** par région, ou quand la **différence des moyennes** entre régions est **sous un seuil fixé**.

**La diapo 66 donne la forme générale** : on **maximise** la réduction d'erreur

$$Err(S)-\sum_{i=1}^{2}\frac{|S_i|}{|S|}\,Err(S_i),\qquad Err(S)=\sum_{j\in S}\big(y_j-y(x_j)\big)^2$$

où $y(x_j)$ est la prédiction du modèle de la feuille : **soit une simple constante** (la moyenne), **soit un modèle linéaire** $y=ax+b$ (on parle alors d'**arbres de modèles** — *model trees*). On reconnaît exactement la structure du **gain d'information**, où l'entropie est remplacée par l'**erreur quadratique** !

| | Arbre de classification | Arbre de régression |
|---|---|---|
| Cible | Classe | Nombre réel |
| Mesure d'impureté | Entropie / Gini | **Erreur quadratique** (variance) |
| Valeur d'une feuille | Classe majoritaire | **Moyenne** des cibles (ou modèle linéaire) |
| Critère de choix | Maximiser le gain | **Maximiser la baisse d'erreur** |

### 12.4 Exemple chiffré (➕ inventé pour t'entraîner)

6 points $(x,y)$ : $(1;1)$, $(2;3)$, $(3;2)$, $(7;10)$, $(8;12)$, $(9;11)$.

- Avec **une seule feuille** (moyenne $=6{,}5$) : erreur quadratique $=125{,}5$.
- Avec la **coupure $x\le5$** : feuille de gauche $\{1,3,2\}$ (moyenne 2) : erreur $2$ ; feuille de droite $\{10,12,11\}$ (moyenne 11) : erreur $2$ ; total $=\mathbf4$.
- Gain : $125{,}5-4=121{,}5$ → la coupure est excellente.

**Lecture graphique (diapo 66) :** la courbe de régression est approchée par des **marches d'escalier** (constantes par morceaux) ou par plusieurs **petits segments linéaires** (si les feuilles sont des modèles linéaires).

### 12.5 Quand utiliser des arbres de régression ? (diapo 69)

- La régression classique **ne marche pas** (par exemple : **dimension d'entrée élevée**) ;
- l'**interprétabilité** du modèle est importante ;
- le problème **se prête bien à une division selon les axes**.

Exemples de la diapo 68 : modélisation non paramétrique de **séries temporelles** (Badel, Michel, Hero, 1996).

---

## 13. Conclusions

*(diapos 70-71)*

- Les arbres sont **bons** quand :
  - l'espace d'entrée est **vectoriel** ;
  - le concept cible correspond à des **boîtes récursives dont les frontières sont parallèles aux axes**.
- Ils sont **très simples**.
- **Complexité d'apprentissage** en $O(A^2\cdot m)$ ($A$ attributs, $m$ exemples).

**Mise en regard avec le cours SVM :** là où le SVM cherche un séparateur **linéaire global** (éventuellement dans un espace redécrit par un noyau), l'arbre construit des **règles locales alignées sur les axes**. L'un est une « boîte noire » puissante, l'autre est transparent mais plus rigide et plus instable.

---

## 14. Fiche récapitulative

| Concept | En une phrase |
|---|---|
| **Arbre de décision** | Suite de tests sur les attributs menant à une classe ; nœud = test, branche = valeur, feuille = classe |
| **Arbre = règles** | Chaque chemin racine → feuille est une règle ; l'arbre est une disjonction de règles |
| **Espace de recherche** | Nombre de Catalan : 16 796 arbres (10 nœuds), 6,56 × 10⁹ (20 nœuds) → impossible à explorer |
| **Stratégie** | Gloutonne, descendante : choisir le meilleur attribut, scinder, recommencer |
| **Entropie** | $-\sum p\log_2p$ ; 0 si pur, maximale (1 bit à 2 classes) si 50/50 |
| **Gain d'information** | $I(S)-\sum\frac{\|S_v\|}{\|S\|}I(S_v)$ : baisse d'entropie obtenue par le test |
| **Indice de Gini** | $1-\sum p_j^2$ ; 0 si pur, 0,5 max à 2 classes |
| **Attribut continu** | On cherche un seuil entre deux exemples consécutifs de classes différentes |
| **Biais du gain** | Favorise les attributs à beaucoup de valeurs → rapport de gain ou binarisation |
| **Valeurs manquantes** | Valeur la plus fréquente, ou répartition pondérée en exemples fictifs |
| **Surapprentissage** | Arbre trop profond qui mémorise le bruit (erreurs, « clashes », valeurs manquantes) |
| **Pré-élagage** | Arrêter la croissance si le test n'améliore pas la validation |
| **Post-élagage** | Construire l'arbre complet puis couper les sous-arbres inutiles (du bas vers le haut) |
| **Biais / variance** | Arbre profond : biais faible, variance forte ; arbre court : l'inverse |
| **Arbre oblique** | Tests sur des combinaisons linéaires d'attributs → frontières inclinées, arbre plus petit |
| **Atouts** | Interprétable, sélection automatique de variables, robuste aux aberrants, rapide |
| **Faiblesses** | Instable sur petites bases ; glouton et myope (XOR) |
| **Arbre de régression** | Feuille = moyenne ; critère = erreur quadratique ; mêmes principes que la classification |
| **Complexité** | $O(A^2\cdot m)$ |

---

## 15. Quiz d'auto-évaluation

> **Mode d'emploi.** Réponds d'abord sans regarder, puis clique sur « Voir la réponse ». Compte un point par bonne réponse. Barème indicatif à la fin.

### Partie A : Fondations et structure

**Q1.** Dans un arbre de décision, que représentent un nœud interne, une branche et une feuille ?
<details><summary>Voir la réponse</summary>

Un **nœud interne** = un **test sur un attribut** ; une **branche** = une **valeur** possible de l'attribut testé ; une **feuille** = une **classe** (la décision).
</details>

**Q2.** Comment utilise-t-on un arbre pour classer un nouvel exemple ?
<details><summary>Voir la réponse</summary>

On part de la **racine**, on suit à chaque nœud la branche correspondant à la valeur de l'attribut testé pour cet exemple, jusqu'à atteindre une **feuille**, dont on lit la classe.
</details>

**Q3.** Un arbre de décision est équivalent à quoi, en termes de logique ? Quelle en est la conséquence pratique ?
<details><summary>Voir la réponse</summary>

À une **disjonction de règles** (un « OU » de « ET »), chaque chemin racine → feuille étant une règle. Conséquence : les arbres sont **interprétables**.
</details>

**Q4.** Vrai ou faux : tout arbre de décision peut représenter n'importe quelle fonction booléenne avec un nombre raisonnable de nœuds.
<details><summary>Voir la réponse</summary>

**Faux.** Toute fonction booléenne *peut* être représentée, mais certaines (comme la **parité** ou la **majorité**) peuvent exiger des arbres de taille **exponentielle**.
</details>

**Q5.** Quelle limite de représentation a un arbre de décision par rapport à la logique du premier ordre ?
<details><summary>Voir la réponse</summary>

Il est limité à la **logique propositionnelle** : il ne peut pas exprimer de **relations entre objets**.
</details>

### Partie B : L'apprentissage glouton

**Q6.** Pourquoi plusieurs arbres peuvent-ils être cohérents avec les mêmes données ? Quelle préférence adopte-t-on en général ?
<details><summary>Voir la réponse</summary>

Parce que les exemples ne contraignent pas suffisamment la structure de l'arbre (diapo 15 : 4 arbres cohérents pour un petit jeu). On préfère **le plus petit arbre** (rasoir d'Ockham), qui généralise en principe mieux.
</details>

**Q7.** Pourquoi ne peut-on pas simplement tester tous les arbres possibles pour trouver le meilleur ?
<details><summary>Voir la réponse</summary>

Parce que l'espace est **gigantesque** : le nombre d'arbres binaires est un **nombre de Catalan** (16 796 pour 10 nœuds, 6,56 × 10⁹ pour 20 nœuds), sans compter le choix des attributs.
</details>

**Q8.** Décris la stratégie d'apprentissage des arbres en quatre étapes. Pourquoi la qualifie-t-on de « gloutonne » ?
<details><summary>Voir la réponse</summary>

1. Choisir le meilleur attribut ; 2. faire croître l'arbre selon ce choix ; 3. on obtient des sous-ensembles aux feuilles ; 4. recommencer jusqu'à un critère d'arrêt. **Gloutonne** car à chaque étape on prend la meilleure décision **locale**, sans jamais la remettre en cause.
</details>

### Partie C : Entropie, gain, Gini

**Q9.** Quelle est l'entropie d'un ensemble contenant 10 exemples tous de la même classe ? Et d'un ensemble 5 « + » / 5 « − » ?
<details><summary>Voir la réponse</summary>

**0** pour l'ensemble pur ; **1 bit** pour 5/5 (deux classes équiprobables : $\log_2 2=1$).
</details>

**Q10.** Calcule l'entropie d'un ensemble de 6 « + » et 2 « − ».
<details><summary>Voir la réponse</summary>

$-\big(\tfrac68\log_2\tfrac68+\tfrac28\log_2\tfrac28\big)=-(0{,}75\times(-0{,}415)+0{,}25\times(-2))=0{,}311+0{,}5=\mathbf{0{,}811}$ bit.
</details>

**Q11.** Écris la formule du gain d'information et explique ce qu'elle mesure.
<details><summary>Voir la réponse</summary>

$Gain(S,A)=I(S)-\sum_v\frac{|S_v|}{|S|}I(S_v)$. Elle mesure la **réduction moyenne d'entropie** obtenue en connaissant la valeur de $A$, c'est-à-dire **l'information que $A$ apporte sur la classe**.
</details>

**Q12.** Pourquoi pondère-t-on l'entropie de chaque branche par $|S_v|/|S|$ ?
<details><summary>Voir la réponse</summary>

Pour que chaque branche compte **proportionnellement à son poids** : une branche pure de 2 exemples prouve moins qu'une branche pure de 100 exemples.
</details>

**Q13.** Un ensemble a 6 « + » et 4 « − ». L'attribut $A$ le sépare en une branche (4 « + », 0 « − ») et une branche (2 « + », 4 « − »). Calcule le gain de $A$ (on donne $I(6,4)=0{,}971$ et $I(2,4)=0{,}918$).
<details><summary>Voir la réponse</summary>

$Gain=0{,}971-\big(\tfrac4{10}\times0+\tfrac6{10}\times0{,}918\big)=0{,}971-0{,}551=\mathbf{0{,}420}$ bit.
</details>

**Q14.** Dans l'exemple de la pastèque, quel est l'attribut racine et pourquoi ?
<details><summary>Voir la réponse</summary>

**texture**, de gain **0,381**, le plus élevé (devant umbilicus 0,289, root 0,143, sound 0,141, color 0,109, surface 0,006). Elle réduit le plus l'incertitude sur la maturité.
</details>

**Q15.** Dans la branche « texture = clear » de la pastèque, trois attributs ont un gain de 0,458. Que fait-on ?
<details><summary>Voir la réponse</summary>

On peut choisir **n'importe lequel** (root, surface ou umbilicus) : il y a égalité. L'arbre appris dépend donc de la règle de départage ; il n'est **pas unique**.
</details>

**Q16.** Donne la formule de l'indice de Gini. Quelle est sa valeur pour un ensemble pur ? pour 50/50 à deux classes ?
<details><summary>Voir la réponse</summary>

$Gini(D)=1-\sum_jp_j^2$. **0** pour un ensemble pur ; **0,5** pour 50/50 à deux classes.
</details>

**Q17.** Calcule l'indice de Gini d'un ensemble de 3 « + » et 1 « − ».
<details><summary>Voir la réponse</summary>

$1-\big((0{,}75)^2+(0{,}25)^2\big)=1-(0{,}5625+0{,}0625)=\mathbf{0{,}375}$.
</details>

### Partie D : Problèmes pratiques

**Q18.** Comment traite-t-on un attribut numérique dans un arbre ? Quels seuils teste-t-on ?
<details><summary>Voir la réponse</summary>

On le **discrétise** par un test de seuil « $A>s$ ? ». On ne teste que les seuils situés **entre deux exemples consécutifs (triés) de classes différentes**, car c'est là que la pureté peut s'améliorer.
</details>

**Q19.** Pour les températures 6, 8, 14 (Non), 18, 20, 28 (Oui), 32 (Non), quels sont les seuils candidats ?
<details><summary>Voir la réponse</summary>

**16°C** (entre 14 et 18) et **30°C** (entre 28 et 32). Le seuil 16°C est le plus informatif (gain ≈ 0,52 contre ≈ 0,13).
</details>

**Q20.** Quel biais a le gain d'information ? Donne un exemple extrême.
<details><summary>Voir la réponse</summary>

Il **favorise les attributs à nombreuses valeurs**. Exemple extrême : un **identifiant unique** crée une branche par exemple, toutes pures, donc un gain maximal, alors qu'il est inutile pour prédire.
</details>

**Q21.** Quelles sont les deux solutions au biais précédent ? Quel est le défaut de chacune ?
<details><summary>Voir la réponse</summary>

1. **Binariser** tous les attributs (défaut : l'arbre perd en interprétabilité) ; 2. **normaliser** le gain en le divisant par l'entropie de la partition (le « rapport de gain »), ce qui pénalise les attributs touffus.
</details>

**Q22.** Que peut-on faire pour un exemple dont une valeur d'attribut est manquante ?
<details><summary>Voir la réponse</summary>

(1) Prendre la valeur la plus fréquente dans $S$ ; (2) la plus fréquente dans le nœud ; (3) répartir l'exemple en **exemples fictifs pondérés par les fréquences** des valeurs (ex. 0,6 / 0,4), puis prédire avec la feuille la plus probable.
</details>

### Partie E : Surapprentissage et élagage

**Q23.** Cite des sources de bruit qui causent le surapprentissage. Que signifie « clash » ?
<details><summary>Voir la réponse</summary>

Erreurs de description, erreurs de classification, valeurs manquantes, « clashes ». Un **clash** : deux exemples ont la **même description mais des classes différentes**.
</details>

**Q24.** Que se passe-t-il, quand l'arbre grossit, pour l'erreur d'apprentissage et pour l'erreur de test ?
<details><summary>Voir la réponse</summary>

L'erreur d'**apprentissage diminue sans cesse** ; l'erreur de **test diminue puis remonte** : l'arbre commence à mémoriser le bruit. C'est le **surapprentissage**.
</details>

**Q25.** Quelle est la différence entre pré-élagage et post-élagage ?
<details><summary>Voir la réponse</summary>

**Pré-élagage** : on arrête la croissance **avant** que l'arbre soit complet. **Post-élagage** : on construit l'arbre **complet** puis on le coupe, en partant des nœuds les plus profonds.
</details>

**Q26.** Quel est le principal risque du pré-élagage ? Du post-élagage ?
<details><summary>Voir la réponse</summary>

Pré-élagage : **sous-apprentissage** (un test peu utile seul peut le devenir avec les suivants). Post-élagage : **coût de calcul** (il faut construire l'arbre complet d'abord), mais en général de meilleures performances.
</details>

**Q27.** Dans l'exemple de la diapo 44, la précision de validation avant la séparation par « umbilicus » est de 3/7. Pourquoi ?
<details><summary>Voir la réponse</summary>

Avant la séparation, on prédit la classe majoritaire de l'apprentissage, « mûre », pour tous les exemples de validation {4, 5, 8, 9, 11, 12, 13}. Seuls {4, 5, 8} sont mûrs : **3 sur 7 ≈ 42,9 %**.
</details>

**Q28.** Après la séparation par « umbilicus », quelle est la précision de validation et que décide-t-on ?
<details><summary>Voir la réponse</summary>

**5/7 ≈ 71,4 %** (bien classés : 4, 5, 8, 11, 12). C'est **mieux** que 42,9 % → la séparation est **conservée**.
</details>

**Q29.** Pourquoi utilise-t-on un ensemble de validation pour élaguer plutôt que l'ensemble d'apprentissage ?
<details><summary>Voir la réponse</summary>

Sur l'apprentissage, un arbre plus développé fait toujours aussi bien ou mieux : on ne couperait jamais rien. L'ensemble de validation, **non vu pendant la construction**, donne une estimation honnête de la généralisation.
</details>

### Partie F : Biais, variance, obliques

**Q30.** Définis le biais et la variance. Pour un arbre, quel est l'effet de la profondeur ?
<details><summary>Voir la réponse</summary>

**Biais** : incapacité du modèle à représenter le vrai concept. **Variance** : sensibilité aux fluctuations de l'échantillon. **Arbre profond : biais faible, variance forte** ; **arbre court : biais fort, variance faible**.
</details>

**Q31.** Pourquoi un classifieur linéaire ne peut-il pas résoudre le problème du triangle (diapo 52) ? De quelle composante de l'erreur s'agit-il ?
<details><summary>Voir la réponse</summary>

Aucune droite ne sépare les rouges (dans le triangle) des bleus (autour). C'est un problème de **biais** : le modèle ne peut pas représenter le concept, quelle que soit la quantité de données.
</details>

**Q32.** Quel est le défaut des frontières d'un arbre classique et comment un arbre oblique y remédie-t-il ?
<details><summary>Voir la réponse</summary>

Les frontières sont **parallèles aux axes** : une diagonale exige un escalier de nombreux tests. L'**arbre oblique** teste une **combinaison linéaire** d'attributs (ex. $1{,}1x_1+x_2<0{,}2$), ce qui donne des frontières inclinées et un **arbre plus petit** (mais moins lisible et plus coûteux à apprendre).
</details>

### Partie G : Bilan et régression

**Q33.** Cite trois avantages et deux inconvénients des arbres de décision.
<details><summary>Voir la réponse</summary>

**Avantages** : interprétables (règles) ; sélection automatique des variables ; traitent tous types de variables ; robustes aux aberrants ; gèrent les valeurs manquantes ; complexité faible. **Inconvénients** : instables sur de petites bases ; gloutons et myopes (interactions comme le XOR).
</details>

**Q34.** Pourquoi un algorithme glouton échoue-t-il sur le XOR ?
<details><summary>Voir la réponse</summary>

Pris **isolément**, aucun des deux attributs n'apporte d'information sur la classe : le gain de chacun est **nul**. Seule leur **combinaison** détermine la classe, mais le critère local ne peut pas le voir.
</details>

**Q35.** Que prédit une feuille d'un arbre de régression ? Quel critère minimise-t-on ?
<details><summary>Voir la réponse</summary>

Une **constante**, la **moyenne** des $y$ des exemples de la région (ou un modèle linéaire dans les « arbres de modèles »). On minimise l'**erreur quadratique** (somme des carrés des écarts à la prédiction).
</details>

**Q36.** Compare l'arbre de classification et l'arbre de régression : quels éléments changent, lesquels restent ?
<details><summary>Voir la réponse</summary>

**Restent** : la structure en arbre, la stratégie gloutonne descendante, le choix de la meilleure coupure, les critères d'arrêt et l'élagage. **Changent** : la cible (nombre réel), l'impureté (**erreur quadratique** au lieu d'entropie/Gini) et la valeur d'une feuille (**moyenne** au lieu de la classe majoritaire).
</details>

**Q37.** Points $(x;y)$ : $(1;1),(2;3),(3;2),(7;10),(8;12),(9;11)$. Quelle est la réduction d'erreur quadratique par la coupure $x\le5$ ?
<details><summary>Voir la réponse</summary>

Avant : moyenne 6,5, erreur 125,5. Après : gauche (moyenne 2, erreur 2), droite (moyenne 11, erreur 2) → total 4. **Réduction = 121,5.**
</details>

**Q38.** Dans quels cas les arbres de régression sont-ils pertinents ?
<details><summary>Voir la réponse</summary>

Quand la **régression classique ne marche pas** (par ex. dimension d'entrée élevée), quand l'**interprétabilité** compte, et quand le problème se prête bien à une **division selon les axes**.
</details>

**Q39 (synthèse).** Explique à un ami, sans formule, comment on construit un arbre de décision et pourquoi on l'élague.
<details><summary>Voir la réponse</summary>

On construit l'arbre **question après question** : à chaque étape on choisit la question qui **sépare le mieux** les classes (celle qui réduit le plus le désordre, mesuré par l'entropie ou Gini), puis on recommence dans chaque branche, jusqu'à obtenir des groupes assez homogènes. Comme on ne regarde que le meilleur choix immédiat, c'est rapide mais **myope**. En laissant l'arbre grandir, il finit par mémoriser le **bruit** des données (surapprentissage) : on le **raccourcit** (élagage), en vérifiant sur des données de validation que les coupures supprimées n'amélioraient pas vraiment les prédictions. Ainsi l'arbre reste simple, lisible et généralise mieux.
</details>

---

### Barème indicatif (39 questions)

| Score | Niveau |
|---|---|
| 0 – 16 | À retravailler : relis les sections 3, 4 et 5 |
| 17 – 27 | Bases acquises : approfondis l'exemple de la pastèque (6) et l'élagage (9) |
| 28 – 35 | Bonne maîtrise |
| 36 – 39 | Excellent ! Tu peux expliquer les arbres à quelqu'un d'autre |

> 💡 **Fil rouge de révision.** *Arbre = suite de tests → trop d'arbres possibles → construction gloutonne → on choisit le test qui réduit le plus l'impureté (gain d'entropie, Gini) → trop de détails = surapprentissage → élagage (valider sur des données non vues) → mêmes idées pour la régression, avec l'erreur quadratique.*
