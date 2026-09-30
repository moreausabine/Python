# La Conformal Prediction expliquée pas à pas

> Explication du cours d'Antoine Cornuéjols (AgroParisTech – INRAE, MIA Paris-Saclay).
> Objectif : comprendre **l'idée derrière chaque concept**, pas seulement la formule.

---

## Sommaire

1. [Le problème : pourquoi 90 % de précision ne suffit pas](#1-le-problème)
2. [Pourquoi le softmax n'est pas une vraie confiance](#2-le-softmax-nest-pas-une-vraie-confiance)
3. [Les approches existantes (deep ensembles) et leurs limites](#3-les-approches-existantes)
4. [Ce que l'on voudrait : la garantie de couverture](#4-ce-que-lon-voudrait--la-garantie-de-couverture)
5. [L'idée centrale : regarder le passé](#5-lidée-centrale--regarder-le-passé)
6. [Le recette en 3 étapes](#6-la-recette-en-3-étapes)
7. [Pourquoi `(n+1)` et pas `n` ? L'intuition de la garantie](#7-pourquoi-n1-et-pas-n-)
8. [Exemple complet pas à pas (chien / tigre / chat)](#8-exemple-complet-chien--tigre--chat)
9. [Le code Python ligne par ligne](#9-le-code-python-ligne-par-ligne)
10. [Message à retenir et hypothèse d'échangeabilité](#10-message-à-retenir)
11. [Petites erreurs / subtilités dans les slides](#11-subtilités-et-petites-erreurs-dans-les-slides)
12. [Mini-quiz d'auto-évaluation](#12-mini-quiz)

---

## 1. Le problème

### La situation classique

On veut apprendre un classifieur qui, à partir d'une **IRM** (`x`), prédit `y ∈ {normal, cancer}`.

Démarche habituelle :

1. Récupérer un **jeu d'entraînement** d'IRM annotées.
2. **Apprendre** un classifieur (par exemple un réseau de neurones).
3. **Évaluer** sa précision sur un jeu de test → disons **90 %**.

### Pourquoi ce n'est pas suffisant

Une précision de 90 % est une **statistique moyenne sur beaucoup d'images**. Elle te dit : « en moyenne, 9 fois sur 10 le modèle a raison ».

Mais le médecin, lui, a **une** IRM devant lui, **celle de son patient**. Il veut savoir :

> « Pour **cette** image-là, puis-je faire confiance à la prédiction ? »

Or le 90 % **ne dit rien sur la fiabilité pour l'entrée courante** (jamais vue auparavant).

**Analogie** : imagine un pilote de ligne qui te dit « mon taux de réussite à l'atterrissage est de 90 % ». C'est rassurant en moyenne, mais si ce soir il y a un brouillard épais, tu voudrais surtout savoir à quel point **cet atterrissage-ci** est risqué.

Le médecin a donc besoin :
- de connaître la **vraisemblance des différentes issues possibles**,
- ou au moins de pouvoir **écarter les issues très improbables**.

**Conclusion** : une prédiction ponctuelle (« cancer ») ne suffit pas. Il faut une **mesure d'incertitude associée à chaque prédiction**.

---

## 2. Le softmax n'est pas une vraie confiance

### Rappel : le softmax

Un réseau de neurones renvoie pour chaque classe `j` un score brut `z_j`. Le softmax les transforme en nombres qui somment à 1 :

```
softmax(z)_j = exp(z_j) / Σ_k exp(z_k)       pour j = 1, ..., K
```

Ça ressemble à des probabilités, donc **la tentation est grande** de dire : « le réseau donne 0,93 pour "cancer", donc il y a 93 % de chances que ce soit un cancer ».

### Pourquoi c'est faux (ou du moins non garanti)

Le cours montre l'exemple célèbre de **Goodfellow et al. (2015)** :

| Image | Prédiction | Confiance softmax |
|---|---|---|
| Photo de panda | « panda » | 57,7 % |
| Panda + un bruit imperceptible (× 0,007) | « gibbon » | **99,3 %** |

L'image est visuellement identique pour un humain, et pourtant le réseau devient **quasi certain d'une réponse absurde**.

**Idée à retenir** : le softmax est **mal calibré**. Il reflète la façon dont le réseau *répartit ses scores*, pas la **vraie probabilité** que la classification soit correcte pour cette entrée. Un réseau peut être très « sûr de lui » et avoir complètement tort.

**Analogie** : un élève peut répondre à une question avec un aplomb total... et se tromper. L'aplomb (le softmax) n'est pas la même chose que la probabilité d'avoir raison.

---

## 3. Les approches existantes

### Les *deep ensembles* (Lakshminarayanan et al., NeurIPS 2017)

Principe :
- On entraîne **plusieurs réseaux** (sur des sous-échantillons aléatoires, ou sur les mêmes données mais avec des poids initiaux aléatoires différents).
- Pour une entrée, on regarde la **distribution des prédictions** de tous ces réseaux : si tous s'accordent → confiance élevée ; s'ils divergent → incertitude.

**Intuition** : c'est comme demander l'avis de 10 médecins. Si tous disent « cancer », on est plus confiant que si 5 disent oui et 5 disent non.

### Leurs limites (à connaître)

- ❌ **Aucune garantie de correction** : rien ne prouve que « 80 % d'accord » signifie « 80 % de chances d'avoir raison ».
- ❌ **Coûteux en calcul** : il faut entraîner et exécuter plusieurs réseaux.

C'est ici que la **conformal prediction** apporte quelque chose de nouveau : **une garantie mathématique, à faible coût**.

---

## 4. Ce que l'on voudrait : la garantie de couverture

### Changer de type de sortie : un *ensemble* de classes

Au lieu de sortir **une** classe, on sort un **ensemble de prédiction** `C(x_test)` : l'ensemble des classes **plausibles**.

Ce que l'on exige :

```
P( y_test ∈ C(x_test) )  ≥  1 − α
```

Lecture : « la vraie classe est dans l'ensemble avec une probabilité **au moins** `1 − α` ».

- `α` est le **taux d'erreur toléré** choisi par l'utilisateur (ex. `α = 0,1` → on veut 90 % de couverture).
- On appelle cette propriété la **couverture** (*coverage*).

### Les exemples ImageNet de la slide 7

Trois images de « fox squirrel » (écureuil roux) de difficulté croissante :

| Difficulté | Ensemble de prédiction |
|---|---|
| Facile (image nette) | `{fox squirrel}` (0,99) → **1 seul élément** |
| Moyenne | `{fox squirrel, gray fox, bucket, rain barrel}` |
| Difficile | `{marmot, fox squirrel, mink, weasel, beaver, polecat}` |

**Le point crucial** : la **taille de l'ensemble s'adapte à la difficulté**.
- Image facile → petit ensemble (le modèle est sûr).
- Image difficile → grand ensemble (le modèle avoue son incertitude).

Et **dans tous les cas**, la garantie `≥ 1 − α` est respectée.

**Analogie** : un GPS qui te dit « tu arriveras entre 14h10 et 14h20 » (autoroute dégagée) ou « entre 13h30 et 15h30 » (centre-ville en heure de pointe). L'intervalle est plus large quand c'est plus incertain, mais il est calibré pour contenir la vraie heure d'arrivée dans 90 % des cas.

---

## 5. L'idée centrale : regarder le passé

> **Utiliser la confiance dans les classifications faites par le système *dans le passé* pour évaluer la confiance dans la prédiction pour *l'entrée courante*.**

Autrement dit : plutôt que de croire aveuglément le softmax, on va **mesurer empiriquement** à quel point le modèle est « confiant » **quand il a affaire à la vraie classe**, sur des exemples dont on connaît la réponse. Cela donne une **échelle de référence**. Ensuite, pour un nouvel exemple, on compare.

**Analogie** : avant de juger si une note de 14/20 est bonne, tu regardes la distribution des notes de la classe. Ici, on regarde la distribution des « confiances dans la bonne réponse » sur des cas passés pour décider à partir de quel niveau de confiance une classe mérite de rester dans l'ensemble.

---

## 6. La recette en 3 étapes

### Étape 0 : avoir un modèle déjà entraîné

Un classifieur `h` (réseau de neurones, forêt aléatoire...). On n'y touche pas : la conformal prediction est un **habillage post-hoc**. Le modèle doit renvoyer, pour chaque classe `y`, un score `h_y(x)` (par ex. le softmax).

### Étape 1 : définir une fonction de confiance `c(x, y)`

Pour la classification, on prend simplement :

```
c(x, y) = h_y(x)
```

C'est « la vraisemblance prédite par le modèle pour la classe `y` sachant `x` » (ex. softmax).

> Dans l'article d'Angelopoulos & Bates, on utilise plutôt le **score de non-conformité** `s = 1 − c(x, y)` : un score **élevé = mauvais** (le modèle s'est trompé). C'est juste l'envers de la confiance. Les deux points de vue sont équivalents.

### Étape 2 : constituer un jeu de calibration

On tire `n` exemples étiquetés :

```
Z = {(x₁, y₁), ..., (xₙ, yₙ)}
```

où `yᵢ` est la **vraie classe** de `xᵢ`.

⚠️ Ces exemples **ne doivent pas avoir servi à l'entraînement**. Sinon le modèle serait trop optimiste sur eux (il les a « appris par cœur ») et la calibration serait faussée.

### Étape 3 : calculer les confiances de calibration

Pour chaque exemple de calibration, on calcule **la confiance du modèle dans la vraie classe** :

```
c(x₁, y₁), ..., c(xₙ, yₙ)
```

**Point clé** : on ne regarde **que** la probabilité attribuée à la **vraie** classe, même si le modèle a prédit autre chose. Si le modèle se trompe, cette valeur sera faible, et c'est précisément l'information qu'on veut capter.

On obtient une **distribution** (l'histogramme des slides 10 et 23) : en abscisse le niveau de confiance (de 1 à 0), en ordonnée la proportion d'exemples.

### Étape 4 : trouver le seuil (le quantile `q̂`)

On choisit `α` (l'erreur tolérée). On cherche le seuil de confiance tel que **la vraie classe soit au-dessus du seuil dans au moins `1 − α` des cas**.

Formellement, on prend le rang :

```
k = ⌈ (1 − α) × (n + 1) ⌉
```

(`⌈ ⌉` = arrondi au supérieur.) Le seuil est alors **la k-ième valeur** dans l'ordre des scores de non-conformité (`s = 1 − c`) triés du plus petit au plus grand. De façon équivalente : c'est le **k-ième niveau de confiance en partant du plus élevé**.

> La formule de la slide 10 `q̂_α = ⌈(1−α)(n+1)⌉ / n` donne en réalité le **niveau du quantile** (une proportion entre 0 et 1), alors que la slide 11 donne le **rang** `⌈(1−α)(n+1)⌉`. C'est la même idée vue de deux façons : rang ÷ n = niveau de quantile.

### Étape 5 : construire l'ensemble de prédiction pour un nouvel `x_test`

On calcule `h(x_test)` = les confiances pour chaque classe, puis on **garde toutes les classes dont la confiance atteint le seuil** :

```
C(x_test) = { y : h_y(x_test) ≥ seuil }
```

C'est tout. Aucun réentraînement, seulement un tri, un quantile, une comparaison.

---

## 7. Pourquoi `(n+1)` et pas `n` ?

C'est la partie la plus subtile, et la plus belle. Voici l'intuition.

### L'argument de symétrie (échangeabilité)

Imagine que le nouvel exemple de test **n'est pas différent** des `n` exemples de calibration : ils viennent tous de la même source, dans un ordre qui n'a pas d'importance (c'est l'**échangeabilité**).

Alors tu as **n + 1 scores** au total : les `n` de calibration + celui du point de test (qu'on ne connaît pas encore). Si tu les triais, **le score du point de test a autant de chances d'occuper n'importe quelle des `n + 1` positions**. Aucune place n'est privilégiée.

Donc :

- La probabilité que le score de test soit **parmi les `k` plus petits** est `k / (n + 1)`.
- Si on prend `k = ⌈(1 − α)(n + 1)⌉`, alors `k / (n + 1) ≥ 1 − α`. 

Le point de test tombera donc dans la zone « acceptée » avec une probabilité **au moins** `1 − α`. C'est exactement la garantie de couverture. 🎉

### Pourquoi le « +1 » ?

Le `+1` compte le **point de test lui-même** dans le classement. C'est une petite correction de **taille finie** : sans elle, la garantie serait légèrement trop optimiste quand `n` est petit. Quand `n` est grand, `n+1 ≈ n` et l'effet disparaît.

### Une précision importante

La garantie est en réalité **encadrée** : la couverture est ≥ `1 − α` et ≤ `1 − α + 1/(n+1)`. D'où la phrase « *presque exactement* `1 − α` » dans l'article. Elle est **marginale** : elle vaut **en moyenne** sur les tirages du jeu de calibration et du point de test, pas pour chaque image individuellement, ni pour chaque classe séparément.

---

## 8. Exemple complet : chien / tigre / chat

On a **3 classes** `{dog, tiger, cat}` et **n = 10** exemples de calibration. Le modèle donne, pour un nouvel exemple, `h(x) = (0,05 ; 0,60 ; 0,35)` (dog, tiger, cat).

### 8.1 Premier jeu de calibration (slides 11 à 14)

Pour chaque image de calibration, le tableau donne les probabilités du modèle. Les cases **encadrées en vert** sont celles de la **vraie classe**. On extrait donc la ligne « confiance dans la vraie classe » :

```
Confiance vraie classe : 0.95  0.90  0.40  0.35  0.30  0.25  0.20  0.15  0.10  0.05
```

(Elle est déjà triée du plus grand au plus petit.)

Remarque : certaines images ont été mal classées par le modèle (cases rouges) : par exemple, une image de chat pour laquelle le modèle donne 0,50 à « tiger » et seulement 0,40 à « cat ». Elles comptent quand même : leur confiance dans la **vraie** classe (0,40) est simplement faible.

#### Cas α = 0,1 (on veut ≥ 90 % de couverture)

- Rang : `⌈0,9 × 11⌉ = ⌈9,9⌉ = 10` → on prend la **10ᵉ** valeur → seuil = **0,05**.
- On compare `h(x) = (0,05 ; 0,60 ; 0,35)` au seuil 0,05 :
  - dog : 0,05 ≥ 0,05 ✔
  - tiger : 0,60 ≥ 0,05 ✔
  - cat : 0,35 ≥ 0,05 ✔
- **`C(x) = {dog, tiger, cat}`**, avec couverture ≥ 0,9.

**Lecture** : pour être *très sûr* (90 %) de ne pas rater la vraie classe, il faut être très large. Ce modèle s'est parfois trompé de façon flagrante (confiance 0,05 sur la vraie classe !), donc le seuil doit descendre très bas pour couvrir ces cas. On garde tout le monde.

#### Cas α = 0,5 (on veut ≥ 50 % de couverture)

- Rang : `⌈0,5 × 11⌉ = ⌈5,5⌉ = 6`.
- Le seuil est de l'ordre de 0,25 – 0,30 selon l'arrondi (voir section 11).
- dog (0,05) ✘ ; tiger (0,60) ✔ ; cat (0,35) ✔.
- **`C(x) = {tiger, cat}`**, avec couverture ≥ 0,5.

**Lecture** : on accepte de se tromper 1 fois sur 2, donc on peut être plus sélectif → l'ensemble rétrécit.

#### Cas α = 0,9 (on veut seulement ≥ 10 % de couverture)

- Rang : `⌈0,1 × 11⌉ = ⌈1,1⌉ = 2` → 2ᵉ valeur → seuil = **0,90**.
- Aucune classe n'atteint 0,90 (le max est 0,60).
- **`C(x) = ∅`** (ensemble vide), avec couverture ≥ 0,1.

**Lecture** : on tolère 90 % d'erreur, donc on est ultra-exigeant, et **aucune classe** ne passe le seuil. Un ensemble vide est parfaitement **légitime** : ça arrive si `α` est grand. Dans ce cas, la garantie de 10 % est tenue par le fait que, sur 10 % des exemples de calibration, la vraie classe avait une confiance ≥ 0,90 : c'est le cas de « très faciles ».

**Le principe général** : *plus on veut de couverture (α petit), plus l'ensemble est grand.* C'est un **compromis** entre garantie de ne pas se tromper et précision (petits ensembles).

### 8.2 Deuxième jeu de calibration : un modèle peu sûr (slides 15 et 16)

Ici, les confiances sur la vraie classe sont **basses et serrées** :

```
0.52  0.50  0.45  0.42  0.40  0.39  0.38  0.37  0.36  0.35
```

(le modèle est hésitant : ses probabilités sont proches de 1/3 partout).

- Pour α = 0,8 : rang `⌈0,2 × 11⌉ = 3` → seuil **0,45** → tiger (0,60) ✔ ; cat (0,35) ✘ → `C(x) = {tiger}`, couverture ≥ 0,2.

### 8.3 Troisième jeu de calibration : un excellent modèle (slide 17)

Ici, les confiances sur la vraie classe sont **très élevées** :

```
0.92  0.90  0.88  0.87  0.86  0.84  0.82  0.80  0.78  0.75
```

- Pour α = 0,1 : rang 10 → seuil **0,75**.
- Notre `x` a un max de 0,60 < 0,75 → **aucune classe ne passe** → `C(x) = ∅`.

**Lecture, et c'est très intéressant** : le modèle est en général très sûr de lui (sur la vraie classe, il est toujours ≥ 0,75). Pour notre `x`, il ne dépasse pas 0,60. Cela signifie que **cet `x` est atypique** par rapport à ce que le modèle voit habituellement : il est *plus incertain que jamais* sur la bonne réponse. L'ensemble vide est un **signal d'alarme** : « cet exemple ne ressemble pas à ceux sur lesquels je suis fiable ».

### 8.4 Quatrième jeu : calibration organisée par classe (slides 18 à 21)

Ces slides regroupent les exemples de calibration par vraie classe (3 chiens, 4 tigres, 3 chats), pour illustrer que l'on prend bien **la confiance dans la vraie classe** de chaque image.

- Slide 19 (α = 0,1) : seuil 0,35 → `{tiger, cat}`.
- Slide 20 (α = 0,5) : `{tiger}`.
- Slide 21 (autre jeu de confiances, α = 0,1) : seuil 0,55 → `{tiger}`.

### 8.5 Ce que montrent tous ces exemples

| Situation | Effet sur l'ensemble de prédiction |
|---|---|
| Je veux plus de garantie (α ↓) | Ensemble **plus grand** |
| Je tolère plus d'erreur (α ↑) | Ensemble **plus petit** (voire vide) |
| Le modèle est bon sur la calibration | Seuil **élevé** → ensembles petits pour les cas faciles, **vides** pour les cas atypiques |
| Le modèle est médiocre sur la calibration | Seuil **bas** → ensembles plus grands |

**Idée fondamentale** : la garantie de couverture est **toujours vraie**, quelle que soit la qualité du modèle. Un mauvais modèle donne simplement des ensembles **plus grands** (moins informatifs), mais **jamais des ensembles faux**. La qualité du modèle influence l'**utilité**, pas la **validité**.

---

## 9. Le code Python ligne par ligne

Voici le code de la slide 22 :

```python
# 1: get conformal scores. n = calib_Y.shape[0]
cal_smx = model(calib_X).softmax(dim=1).numpy()
cal_scores = 1 - cal_smx[np.arange(n), cal_labels]

# 2: get adjusted quantile
q_level = np.ceil((n+1)*(1-alpha))/n
qhat = np.quantile(cal_scores, q_level, method='higher')

val_smx = model(val_X).softmax(dim=1).numpy()
prediction_sets = val_smx >= (1 - qhat)   # 3: form prediction sets
```

### Étape 1 : les scores de calibration

- `cal_smx = model(calib_X).softmax(dim=1).numpy()` : on passe les `n` images de calibration dans le modèle et on obtient, pour chacune, le vecteur softmax (une ligne = une image, une colonne = une classe).
- `cal_smx[np.arange(n), cal_labels]` : pour chaque ligne `i`, on va chercher **la colonne correspondant à la vraie classe** `cal_labels[i]`. On récupère ainsi la **confiance dans la vraie classe** (les cases vertes des slides).
- `1 - ...` : on la transforme en **score de non-conformité** `sᵢ`. Score **petit** = modèle confiant dans la bonne réponse (bien) ; score **grand** = modèle peu confiant dans la bonne réponse (mauvais).

### Étape 2 : le quantile ajusté

- `q_level = np.ceil((n+1)*(1-alpha))/n` : le niveau de quantile **corrigé** avec le `(n+1)` (section 7).
- `np.quantile(..., method='higher')` : renvoie une **vraie valeur** de la liste (pas d'interpolation) et arrondit **vers le haut**, ce qui est le choix conservateur qui préserve la garantie.

### Étape 3 : former les ensembles

- `val_smx >= (1 - qhat)` : pour chaque image de test et chaque classe, on garde la classe si sa probabilité softmax dépasse `1 − q̂`.
  - Pourquoi `1 − q̂` ? Parce que l'on est passé des confiances aux scores (`s = 1 − c`). Dire « `s ≤ q̂` » revient à dire « `c ≥ 1 − q̂` ».
- Le résultat est un tableau de booléens : pour chaque image, quelles classes sont dans l'ensemble.

### Le schéma des trois panneaux (slide 22)

1. **Compute scores on holdout data** : pour chaque image de calibration, on regarde la hauteur de la barre de la vraie classe (softmax) → score `1 − sᵢ`.
2. **Get quantile** : on trace l'histogramme des scores et on repère `q̂`.
3. **Construct prediction set** : sur une nouvelle image, on trace une ligne horizontale à la hauteur `1 − q̂` ; toutes les barres qui la dépassent sont dans l'ensemble (ici `{1, 4}`).

---

## 10. Message à retenir

Le cours résume la conformal prediction ainsi :

- ✅ Elle donne une **mesure d'incertitude pour l'entrée courante** (l'ensemble de prédiction s'adapte à chaque `x`).
- ✅ Elle fonctionne :
  - avec **n'importe quelle distribution** des données (*distribution-free*),
  - avec **n'importe quelle taille** de jeu de calibration `n` (garantie en échantillon fini, pas seulement asymptotique),
  - avec **n'importe quel modèle** (classification ou régression) qui fournit une estimation de confiance.
- ⚠️ **Sous l'hypothèse que `Z ∪ {(x_test, y_test)}` est échangeable.**

### L'échangeabilité, c'est quoi ?

Un ensemble de variables aléatoires est **échangeable** si leur loi jointe ne change pas quand on permute leur ordre. Aucune ne joue un rôle spécial. C'est un peu plus faible que « i.i.d. » (indépendantes et identiquement distribuées) : tout ensemble i.i.d. est échangeable, mais pas l'inverse.

**En pratique**, cela veut dire : *les données de calibration et les données de test doivent venir de la même source*.

Exemples où l'hypothèse est **violée** (donc la garantie tombe) :
- Calibration faite sur des IRM d'un hôpital A, test sur un hôpital B avec un autre scanner.
- Séries temporelles avec dérive au cours du temps.
- Données d'été pour la calibration, données d'hiver pour le test.

### Ce que la conformal prediction n'est PAS

| Idée reçue | Réalité |
|---|---|
| « C'est une probabilité que ma prédiction soit juste » | C'est une garantie sur un **ensemble** : `P(y ∈ C(x)) ≥ 1 − α`, en moyenne |
| « Ça améliore mon modèle » | Non : le modèle n'est pas modifié. On l'habille avec des ensembles calibrés |
| « Ça marche pour chaque image individuellement » | La garantie est **marginale** (moyenne), pas conditionnelle à `x` ni à la classe |
| « Un bon modèle est nécessaire pour que ça marche » | Non pour la **validité** ; oui pour l'**utilité** (des ensembles petits) |

---

## 11. Subtilités et petites erreurs dans les slides

En vérifiant les calculs, j'ai remarqué quelques points qui pourraient te perturber :

1. **Slides 13 et 20 (α = 0,5)** : avec `n = 10`, le rang est `⌈5,5⌉ = 6`. La 6ᵉ valeur de la liste triée est 0,25 (slide 13) et 0,50 (slide 20), alors que les slides indiquent `Q = 0,30` et `Q = 0,55`, ce qui correspond à la **5ᵉ** valeur. Petit décalage d'un cran. **Cela ne change pas les ensembles trouvés** dans ces exemples, mais si tu refais le calcul à la main à l'examen, applique bien la règle « k-ième valeur avec `k = ⌈(1−α)(n+1)⌉` ».
   - Pour la slide 20 précisément, avec la 6ᵉ valeur (0,50), on aurait `tiger` (0,60 ≥ 0,50) ✔ et `cat` (0,35) ✘, donc toujours `{tiger}`.

2. **Slide 10** : la formule affichée `⌈(1−α)(n+1)⌉ / n` est un **niveau de quantile** (une fraction), alors que sur la slide 11 la même quantité sans le `/n` est un **rang** (un entier). C'est cohérent, mais la notation `q̂` désigne deux choses différentes selon la slide.

3. **Slide 14 (`α = 0,9`)** : la notation `Q₀.₉` renvoie à un niveau de quantile, tandis que la ligne du dessus écrit `q̂₀.₁`. Le rang calculé (`⌈0,1 × 11⌉ = 2`) correspond bien à `1 − α = 0,1`. Rien de faux, mais deux conventions se croisent.

4. **Slide 23 « Possible ? »** : l'histogramme contient des exemples avec des confiances très basses sur la vraie classe (0 à 0,2). La question est : « est-ce possible qu'un modèle soit si peu confiant dans la bonne réponse ? ». Oui, tout à fait : c'est justement ce qu'un modèle mal calibré ou un cas difficile produit, et la méthode gère cela sans hypothèse sur la forme de la distribution.

---

## 12. Mini-quiz

Pour t'auto-évaluer (réponses en dessous).

**Q1.** Pourquoi une précision de 90 % sur le jeu de test ne suffit-elle pas au médecin ?

**Q2.** Que signifie « le softmax est mal calibré » ?

**Q3.** Dans la conformal prediction, quelle valeur de confiance regarde-t-on sur les exemples de calibration : celle de la classe prédite, ou celle de la vraie classe ?

**Q4.** Si je diminue `α` (de 0,2 à 0,05), l'ensemble de prédiction devient-il plus grand ou plus petit ? Pourquoi ?

**Q5.** Pourquoi utilise-t-on `(n + 1)` dans le calcul du rang ?

**Q6.** Peut-on avoir un ensemble de prédiction vide ? Que cela signifie-t-il ?

**Q7.** Que se passe-t-il pour la garantie si le jeu de test provient d'une distribution différente de celle de la calibration ?

**Q8.** Le modèle est mauvais. La garantie de couverture est-elle toujours vraie ? Qu'est-ce qui change ?

### Réponses

**R1.** Parce que c'est une moyenne sur beaucoup d'images : elle ne renseigne pas sur la fiabilité de la prédiction pour **cette** image précise.

**R2.** Les valeurs du softmax ne correspondent pas à la vraie probabilité que la prédiction soit correcte. Un réseau peut afficher 99 % de confiance et se tromper (exemple panda → gibbon).

**R3.** La confiance dans la **vraie** classe `yᵢ`, même si le modèle a prédit autre chose.

**R4.** Plus **grand** : on exige une couverture plus forte (95 % au lieu de 80 %), donc le seuil de confiance baisse, donc plus de classes le franchissent.

**R5.** Il tient compte du point de test lui-même dans le classement de `n + 1` scores échangeables. C'est ce qui donne la garantie exacte en échantillon fini : `k/(n+1) ≥ 1 − α`.

**R6.** Oui, quand `α` est grand ou quand `x` est atypique (plus incertain que ce que le modèle a montré en calibration). C'est un signal : « je ne sais pas / cet exemple sort de mon domaine de fiabilité ».

**R7.** L'échangeabilité est brisée : la garantie n'est plus assurée. C'est la principale limite pratique de la méthode (changement de domaine, dérive temporelle...).

**R8.** Oui, la garantie reste vraie (elle ne dépend pas du modèle). Ce qui change, c'est l'**utilité** : les ensembles de prédiction seront plus grands, donc moins informatifs.

---

## Pour aller plus loin

- Angelopoulos & Bates, *Conformal Prediction: A Gentle Introduction*, Foundations and Trends in Machine Learning, 2023 (référence citée dans le cours, très pédagogique).
- Le cours mentionne aussi la régression : l'idée est identique, mais l'ensemble de prédiction devient un **intervalle** `[ŷ − q̂, ŷ + q̂]` autour de la prédiction, où le score de non-conformité est typiquement l'erreur absolue `|y − ŷ|`.