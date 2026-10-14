# Cours 0 : Structure de données PyTorch

> Module 1 – Agro–IODAA – Semestre 1 (Vincent Guigue, AgroParisTech / Paris-Saclay)
> Notes pédagogiques : le *pourquoi* derrière chaque concept, pas seulement le *comment*.

---

## Plan

1. [Calcul matriciel](#1-calcul-matriciel)
2. [Gradient et autograd](#2-gradient-et-autograd)
3. [Application : la régression linéaire](#3-application--la-régression-linéaire)
4. [Conclusion : PyTorch dans le paysage](#4-conclusion--pytorch-dans-le-paysage)
5. [Cheatsheet PyTorch](#5-cheatsheet-pytorch)

**Fil rouge du cours :** PyTorch est une boîte à outils qui fait deux choses.
1. Il manipule des tableaux de nombres (*tensors*) très efficacement, sur CPU ou GPU.
2. Il calcule automatiquement des dérivées (*gradients*), ce qui permet d'entraîner des modèles.

Tout le reste (réseaux de neurones, optimiseurs…) se construit sur ces deux briques.

---

## 1. Calcul matriciel

### 1.1 Pourquoi des matrices ? La philosophie générale

**L'idée.** Quand on fait de la data science, on a presque toujours un tableau de données :

| | surface | nb pièces | prix |
|---|---|---|---|
| maison 1 | 80 | 3 | 250 |
| maison 2 | 120 | 5 | 400 |

- une **ligne = un individu** (une maison, une plante, un patient…)
- une **colonne = un descripteur** (une caractéristique mesurée)

Une matrice est la structure naturelle pour stocker ça.

**Trois points clés du cours :**

**a) Les matrices sont typées.**
Chaque case contient un nombre d'un type précis : `torch.float32`, `torch.int64`, `torch.int16`, etc. Un `int16` prend 2 octets par valeur, un `float32` en prend 4, un `float64` en prend 8. Sur des millions de valeurs, le choix du type change énormément l'occupation mémoire. D'où la phrase du cours : *« le plus souvent, on cherche à gagner de la place »*.

> Remarque : sur la slide, `torch.int16` est écrit deux fois, c'est une coquille. Pensez à des exemples comme `torch.int8`, `torch.int16`, `torch.float16`, `torch.float32`.

**b) Philosophie : opérateurs de haut niveau, pas de boucles.**
En Python pur, pour additionner deux listes élément par élément, on écrirait une boucle `for`. Avec PyTorch, on écrit `a + b` et c'est tout.
- C'est plus **lisible** (une ligne au lieu de cinq).
- C'est beaucoup plus **rapide** : la boucle est exécutée en code C/C++ optimisé (voire en parallèle sur GPU), pas par l'interpréteur Python.
- Le cours le résume ainsi : *penser en « motifs » (opérations sur des blocs de données) plutôt qu'en « boucles » (case par case)*.

**c) Tout est `torch`.**
On n'utilise pas des tableaux NumPy ou des listes Python pour calculer, mais uniquement des `torch.Tensor`. Deux raisons :
- le **passage sur GPU** est simple (on déplace le tensor, et les calculs suivent) ;
- le **calcul de gradient** est possible (c'est la fonctionnalité qui rend PyTorch si utile, voir partie 2).

### 1.2 Créer des matrices

```python
import torch

# Matrice de 4 lignes et 3 colonnes à partir de valeurs écrites à la main
cst   = torch.tensor([[3.1, 2.7, 1.9], [1.6, 0.0, 9.8], [1.7, 8.1, 5.0], [...]])
zeros = torch.zeros(4, 3)   # que des 0
ones  = torch.ones(4, 3)    # que des 1
x     = torch.rand(4, 3)    # valeurs aléatoires uniformes dans [0, 1[

zeros_like_x = torch.zeros_like(x)   # matrice de 0 ayant la même forme (et type) que x
```

**À quoi ça sert ?**
- `zeros` / `ones` : initialiser un accumulateur, un masque, etc.
- `rand` : initialiser aléatoirement (très utile pour les poids d'un modèle).
- `zeros_like(x)` : « fais-moi un tableau de la même taille que `x` » sans avoir à recopier ses dimensions.

### 1.3 Comment un tensor est stocké en mémoire (le schéma de la slide 3)

C'est l'un des schémas les plus instructifs du cours. Prenons :

```
x = | 0   1   2 |
    | 3   4   5 |
    | 6   7   8 |
    | 9  10  11 |
```

Même si on la voit comme un tableau 2D, la machine stocke **une seule longue suite de nombres** en mémoire :

```
data : [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
```

Le tensor est en fait un petit « dossier » avec 4 champs :

| Champ | Valeur dans l'exemple | Signification |
|---|---|---|
| `data` | pointeur vers la suite 0…11 | où sont les nombres |
| `data type` | entier sur 8 octets | comment lire chaque case |
| `shape` | `(4, 3)` | 4 lignes, 3 colonnes |
| `strides` | `(24, 8)` | de combien d'octets avancer pour passer à la case suivante |

**Comprendre les strides :**
- Pour aller à la **colonne suivante** : on avance de 1 élément, soit 8 octets.
- Pour aller à la **ligne suivante** : on saute 3 éléments (une ligne entière), soit 3 × 8 = **24 octets**.

**Pourquoi c'est important ?** Parce que beaucoup d'opérations (transposer, extraire une colonne, changer la forme) ne **copient pas** les données : elles changent juste `shape` et `strides`. C'est ce qui rend PyTorch rapide et économe en mémoire. Par contre, ça veut aussi dire que deux variables peuvent partager les mêmes données (une modification de l'une se voit dans l'autre).

### 1.4 Accéder aux valeurs (indexation et slicing)

Le message de la slide : **« penser aux motifs plutôt qu'aux boucles »**.

```python
x = torch.tensor([[2, 4, 6],
                  [3, 6, 9]])

x[0][0]        # tensor(2)   ┐ deux écritures équivalentes
x[0, 0]        # tensor(2)   ┘ (la 2e est préférable : plus rapide, plus claire)

x[:, 0]        # toutes les lignes, colonne 0   -> tensor([2, 3])
x[0, [0, 2]]   # ligne 0, colonnes 0 et 2       -> tensor([2, 6])
```

> Même une case seule est un tensor (de dimension 0). Pour récupérer le nombre Python, utiliser `.item()`.

**La syntaxe générale du slicing : `début:fin:pas`**
- le début est **inclus**, la fin est **exclue** ;
- chaque élément est optionnel.

```python
x[0, 0:2]    # = x[0, :2]    -> colonnes 0 et 1   (borne haute)
x[0, 1:3]    # = x[0, 1:]    -> colonnes 1 et 2   (borne basse)
x[0, 0:3:2]  # = x[0, ::2]   -> colonnes 0 et 2   (pas de 2)
```

**Moyen mnémotechnique :** `:` seul veut dire « tout ». `x[:, 0]` = « toutes les lignes, colonne 0 ». C'est exactement le motif « colonne » d'un tableau de données (par exemple, tous les individus pour un seul descripteur).

### 1.5 Calcul entre matrices

Opérateurs disponibles : `+`, `-`, `*`, `/`, `**` (puissance).

```python
# Avec un scalaire : l'opération est appliquée à CHAQUE case
x = torch.tensor([[2, 4, 6], [3, 6, 9]])
y = x + 2          # [[4, 6, 8], [5, 8, 11]]

# Entre deux matrices de même forme : TERME À TERME
z = x * y          # [[8, 24, 48], [15, 48, 99]]    (2*4, 4*6, 6*8 ...)

# Produit matriciel (au sens mathématique) : l'opérateur @
a = x @ torch.tensor([[1], [2], [3]])    # [[28], [42]]
```

**Distinction fondamentale à retenir :**

| Opérateur | Nom | Exemple |
|---|---|---|
| `*` | produit **terme à terme** (Hadamard) | `[[2,4]] * [[3,5]] = [[6,20]]` |
| `@` | produit **matriciel** (lignes × colonnes) | forme `(n,p) @ (p,q) -> (n,q)` |

Pour `@`, le nombre de colonnes de la première matrice doit être égal au nombre de lignes de la seconde. Ici : `(2,3) @ (3,1) -> (2,1)`. On a calculé 2·1 + 4·2 + 6·3 = 28 et 3·1 + 6·2 + 9·3 = 42.

C'est ce produit qui sert à appliquer un modèle linéaire à **tous les individus en une seule opération**.

### 1.5 bis Le *broadcasting* (à connaître)

Quand on écrit `x + 2`, PyTorch « étire » virtuellement le 2 pour qu'il ait la même forme que `x`. Cette règle s'appelle le **broadcasting** et marche aussi avec des tensors de formes différentes mais compatibles :

```python
x = torch.ones(4, 3)
b = torch.tensor([10., 20., 30.])   # forme (3,)
x + b                               # b est ajouté à chaque ligne de x
```

On s'en servira pour ajouter le biais `b` dans la régression linéaire (partie 3).

### 1.6 Opérations *in place* ou non

**Deux manières de faire un calcul :**

1. **Classique** : le résultat est stocké dans un **nouvel** objet. L'original n'est pas touché.
```python
b = torch.sin(a)    # b est nouveau, a inchangé
c = a + b           # c est nouveau
```

2. **In place** (« sur place ») : le résultat **écrase** l'objet d'origine. Convention : un **underscore `_`** à la fin du nom.
```python
torch.sin_(a)       # a est modifié (a = sin(a))
a.add_(b)           # a est modifié (a = a + b)
```

**Pourquoi deux versions ?**
- In place : on **économise de la mémoire** (pas de nouvelle matrice allouée). Important avec de grosses données.
- Classique : plus **sûr**. Si vous modifiez `a` sur place, vous perdez sa valeur d'origine, ce qui peut poser problème (notamment pour le calcul de gradient, où PyTorch a parfois besoin des valeurs d'origine).

**Règle pratique :** par défaut, utilisez la version classique ; passez en in place seulement si la mémoire est un vrai souci ou quand vous mettez à jour des paramètres (voir 2.4).

### 1.7 Calculs agrégatifs (sommes, moyennes…)

Un calcul **agrégatif** résume plusieurs valeurs en une seule : somme, moyenne, minimum, maximum…

Tout se joue sur l'argument **`axis`** (ou `dim`) : *c'est la dimension qui disparaît*.

```python
x = torch.tensor([[2, 4, 6],
                  [3, 6, 9]])

li  = x.sum(1)    # ou x.sum(axis=1) -> [12, 18]     : somme de chaque LIGNE
co  = x.sum(0)    #                   -> [5, 10, 15] : somme de chaque COLONNE
tot = x.sum()     #                   -> 30          : somme de TOUT

# Somme des colonnes 0 et 2 seulement, pour chaque ligne
li = x[:, ::2].sum(1)   # -> [8, 12]
```

**Comment ne pas se tromper :**
- `axis=1` : on **parcourt les colonnes** (flèche horizontale sur le schéma) et on les écrase → il reste une valeur **par ligne**.
- `axis=0` : on **parcourt les lignes** (flèche verticale) et on les écrase → il reste une valeur **par colonne**.
- Sans argument : tout est écrasé en un seul nombre.

Application concrète : si ligne = individu et colonne = descripteur, alors `x.mean(0)` donne **la moyenne de chaque descripteur** (utile pour normaliser), et `x.sum(1)` donne un total **par individu**.

Même logique pour `mean`, `min`, `max`, `std`, `prod`…

### 1.8 `squeeze` / `unsqueeze` : gérer les dimensions

**Rappel des dimensions :**
- une ligne de données = 1D (vecteur)
- une image = 2D (par canal)
- un corpus d'images = 3D ou plus

**Le problème.** Les modèles PyTorch travaillent par **batch** (lot) : ils attendent toujours une dimension supplémentaire en premier, qui indexe les individus. Même pour une seule image, il faut dire « c'est un batch de 1 image ».

- **`unsqueeze(d)`** : **ajoute** une dimension de taille 1 à la position `d`. Utile pour donner **une image** à un système qui attend un **batch**.
- **`squeeze()`** : **supprime** les dimensions de taille 1. Utile pour récupérer une sortie intermédiaire (batch de 1) et l'analyser comme un **individu** seul.

```python
a = torch.rand(3, 226, 226)    # une image : 3 canaux (RGB), 226 x 226 pixels
b = a.unsqueeze(0)             # on ajoute une dimension de batch en position 0

print(a.shape)   # torch.Size([3, 226, 226])
print(b.shape)   # torch.Size([1, 3, 226, 226])   <- batch de 1 image
```

**Analogie :** `a` est une feuille de papier. `b` est la même feuille, mais rangée dans un classeur qui peut contenir plusieurs feuilles. `squeeze` sort la feuille du classeur.

---

## 2. Gradient et autograd

### 2.1 Pourquoi un gradient ?

Entraîner un modèle, c'est **trouver les paramètres** qui rendent une **fonction de coût** (l'erreur) la plus petite possible.

Le **gradient** d'une fonction indique, pour chaque paramètre, **de combien le coût change si on bouge ce paramètre un tout petit peu**. C'est la pente.
- pente positive → si on augmente le paramètre, le coût monte → il faut **le diminuer** ;
- pente négative → il faut **l'augmenter**.

On avance donc dans la direction **opposée au gradient** : c'est la **descente de gradient**.

> Image : vous êtes dans le brouillard sur une colline et vous voulez descendre. Vous tâtez le sol autour de vous (le gradient), puis vous faites un pas dans la direction qui descend le plus.

Calculer ces dérivées à la main est pénible et source d'erreurs. PyTorch le fait **automatiquement** : c'est l'**autograd** (différentiation automatique).

### 2.2 Activer le gradient sur un tensor

Un `tensor` contient ses **données** et, en option, un champ **`grad`** pour stocker son gradient.

```python
a = torch.tensor(1.)
a.requires_grad = True                       # activation du suivi du gradient
b = torch.tensor(2., requires_grad=True)     # ou directement à la création

z = 2*a + b                                  # z dépend de a et b
z.backward()                                 # calcul des dérivées de z

print("dz/da :", a.grad.item(), " dz/db :", b.grad.item())
# dz/da : 2.0   dz/db : 1.0
```

**Ce qui se passe, étape par étape :**
1. `requires_grad=True` dit : « surveille tout calcul qui implique ce tensor ».
2. Quand on écrit `z = 2*a + b`, PyTorch ne se contente pas de calculer la valeur de `z` : il **mémorise la recette** (une multiplication puis une addition). C'est le **graphe de calcul**.
3. `z.backward()` remonte cette recette à l'envers et applique la règle de dérivation en chaîne.
4. Les résultats sont déposés dans `a.grad` et `b.grad`.

**Vérification à la main :** z = 2a + b, donc ∂z/∂a = 2 et ∂z/∂b = 1. ✔

**Lien avec l'apprentissage :** ici `z` joue le rôle de la **fonction de coût**, et `a`, `b` celui des **paramètres du modèle**. Le gradient nous donne la direction pour minimiser le coût.

### 2.3 Gradient sur un vecteur / une matrice

En pratique, les paramètres sont des vecteurs ou des matrices (et non de simples nombres).

**Règle essentielle :** le gradient a **exactement la même forme** que le tensor d'origine.

Si
$$A = \begin{pmatrix} a_{11} & a_{12} & \dots \\ a_{21} & a_{22} & \dots \end{pmatrix}, \qquad L = f(A)$$
alors
$$A.\text{grad} = \begin{pmatrix} \frac{\partial L}{\partial a_{11}} & \frac{\partial L}{\partial a_{12}} & \dots \\ \frac{\partial L}{\partial a_{21}} & \frac{\partial L}{\partial a_{22}} & \dots \end{pmatrix}$$

Chaque case du gradient dit : « si je bouge **cette** case de `A`, voilà comment `L` varie ».

**Conditions :**
- **`L` doit être un scalaire** (un seul nombre). C'est logique : on minimise **un** coût. Si on avait un vecteur de coûts, la question « dans quel sens descendre ? » n'aurait pas de réponse unique. En pratique, on calcule une erreur par individu, puis on prend la **moyenne** (ou la somme).
- On déclenche le calcul avec `L.backward()`.

> Le cours précise que les `tensor` sont en général des vecteurs ou des matrices, « mais pas des tenseurs » au sens mathématique strict (ordre ≥ 3). C'est une subtilité de vocabulaire : le mot *tensor* de PyTorch désigne simplement un tableau à n dimensions.

### 2.4 Pièges de syntaxe une fois le gradient activé

**Lire les valeurs sans le graphe :**
```python
b = torch.tensor(2., requires_grad=True)
b.data          # ok : accède à la valeur brute, sans le graphe
b.grad.data     # ok : idem pour le gradient
b.item()        # ok pour un tensor à un seul élément
```
La slide indique que `print(b)` provoque une erreur « ambiguë ». Dans les versions récentes de PyTorch, `print(b)` fonctionne et affiche `tensor(2., requires_grad=True)`. Retenez surtout qu'un tensor avec gradient n'est plus un simple tableau de nombres : il est relié à un graphe.

**Mise à jour des paramètres : couper le graphe avec `torch.no_grad()`.**

```python
# [...] on a calculé la perte et appelé backward()

with torch.no_grad():
    w += -EPS * w.grad     # w <- w - EPS * gradient
    b += -EPS * b.grad
```

**Pourquoi `no_grad` ?**
- La mise à jour des paramètres n'est **pas** une opération qu'on veut dériver. On ne veut pas que PyTorch l'ajoute au graphe.
- Sans `no_grad`, il mémoriserait inutilement ces opérations (gaspillage de mémoire et de temps), et PyTorch refuse d'ailleurs de modifier sur place un paramètre avec `requires_grad=True` en dehors de ce bloc.
- `EPS` (epsilon) est le **pas d'apprentissage** (*learning rate*) : la taille du pas qu'on fait dans la direction opposée au gradient.
  - trop grand : on « saute par-dessus » le minimum et ça diverge ;
  - trop petit : on avance, mais très lentement.

### 2.5 Accumulation du gradient

**Comportement surprenant mais voulu :** chaque appel à `backward()` **ajoute** le nouveau gradient à celui déjà stocké dans `.grad`, au lieu de le remplacer.

```python
a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)

z = 2*a + b
z.backward()          # a.grad = 2, b.grad = 1

z = 2*a + b           # il faut RECONSTRUIRE z (le graphe a été libéré)
z.backward()          # a.grad = 4, b.grad = 2   <- accumulation !
```

**Pourquoi ce choix de conception ?**
- On peut avoir **plusieurs fonctions de coût** (ou plusieurs morceaux de données) et vouloir **additionner** leurs gradients : l'accumulation permet de le faire naturellement.
- Après `backward()`, PyTorch **libère le graphe** pour économiser la mémoire ; c'est pourquoi il faut redéfinir `z` avant de rappeler `backward()`.

**Conséquence pratique (à ne jamais oublier) :** dans une boucle d'apprentissage, il faut **remettre les gradients à zéro** après chaque mise à jour, sinon les gradients de toutes les itérations s'additionnent et l'apprentissage est faussé :
```python
w.grad.zero_()
b.grad.zero_()
```

---

## 3. Application : la régression linéaire

La régression linéaire sert ici de **premier exemple complet** qui assemble tout ce qui précède. Le cours identifie 5 « verrous » (étapes à franchir) :

1. Lire les données (via `scikit-learn`)
2. Construire l'estimateur linéaire
3. Appliquer la fonction de coût
4. Trouver un critère d'arrêt pour la descente de gradient
5. Évaluer les performances

### 3.1 Accès aux données

On récupère un jeu de données (prix de l'immobilier en Californie) avec `scikit-learn`, puis on le convertit en `tensor`.

```python
from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()          # chargement des données

# Penser à typer les données pour éliminer les incertitudes
housing_x = torch.tensor(housing['data'],   dtype=torch.float)
housing_y = torch.tensor(housing['target'], dtype=torch.float)
```

**Pourquoi typer explicitement ?** Les données arrivent en NumPy (souvent en `float64`). PyTorch travaille par défaut en `float32`, et mélanger des types différents provoque des erreurs. En imposant `dtype=torch.float`, on est sûr de ce qu'on manipule.

**À vérifier systématiquement : les dimensions** (`housing_x.shape`, `housing_y.shape`). La majorité des bugs en PyTorch sont des erreurs de forme.

### 3.2 Estimateur linéaire

**Le modèle :** pour un individu décrit par un vecteur **x**, la prédiction est
$$\hat{y} = \langle \mathbf{w}, \mathbf{x} \rangle + b = w_1 x_1 + w_2 x_2 + \dots + b$$
- **w** : un poids par descripteur (l'importance de chaque caractéristique)
- **b** : le biais (valeur de base quand tout est nul)

**Déclarer les paramètres avec gradient activé :**
```python
w = torch.randn(1, housing_x.size(1), requires_grad=True)   # forme (1, nb_descripteurs)
b = torch.randn(1, 1,               requires_grad=True)     # forme (1, 1)
```
- `randn` : tirage aléatoire selon une loi normale (valeurs de départ).
- `housing_x.size(1)` : nombre de colonnes = nombre de descripteurs.
- `requires_grad=True` : ces paramètres sont ceux qu'on va **apprendre**.

**Appliquer le modèle à tous les individus d'un coup :**
```python
y_hat = (x @ w.T) + b
```
Décortiquons les formes, pour `n` individus et `d` descripteurs :
- `x` : `(n, d)`
- `w.T` : `(d, 1)` (la transposée de `(1, d)`)
- `x @ w.T` : `(n, 1)` → une prédiction par individu
- `+ b` : `b` de forme `(1, 1)` est « étiré » (broadcasting) sur les `n` lignes

> **Piège de forme à connaître !** Ici, `housing_y` a la forme `(n,)` alors que `y_hat` a la forme `(n, 1)`. Si on calcule `y - y_hat` tel quel, le broadcasting produit une matrice `(n, n)` au lieu de `(n, 1)` et le coût sera faux, **sans message d'erreur**. Il faut donc remettre `y` en `(n, 1)` : `housing_y.unsqueeze(1)` (c'est là qu'`unsqueeze` est utile).

### 3.3 Fonction de coût

On mesure l'erreur par les **moindres carrés** : pour chaque individu, on prend la différence entre la vérité `y` et la prédiction `ŷ`, on l'élève au carré, puis on moyenne.
$$\mathcal{L} = \frac{1}{n}\sum_i \left( y_i - \hat{y}_i \right)^2$$

Le carré permet d'avoir une erreur toujours positive et de pénaliser davantage les grosses erreurs. Le résultat est bien un **scalaire**, donc on peut appeler `.backward()`.

### 3.4 Autograd et graphe de calcul (slide 16)

Le schéma de la slide montre le **graphe de calcul** de l'erreur d'un seul individu :

```
x, w ──► s = <w, x> ──► ŷ = s + b ──► d = y − ŷ ──► e = d²
                           ▲                ▲
                           b                y
```

- **Passe forward** (flèches noires) : on calcule de gauche à droite, des entrées vers l'erreur `e`.
- **Passe backward** (flèches rouges) : on remonte de droite à gauche en appliquant la **règle de dérivation en chaîne**.

**Calcul à la main, étape par étape (pour comprendre ce que fait `backward()`) :**

| Étape | Dérivée locale | Valeur |
|---|---|---|
| `e = d²` | ∂e/∂d | **2d** |
| `d = y − ŷ` | ∂d/∂ŷ | **−1** |
| `ŷ = s + b` | ∂ŷ/∂s et ∂ŷ/∂b | **1** et **1** |
| `s = <w, x>` | ∇ₓs et ∇_w s | **w** et **x** |

On **multiplie** les dérivées locales le long du chemin :
- ∂e/∂b = (∂e/∂d)(∂d/∂ŷ)(∂ŷ/∂b) = 2d · (−1) · 1 = **−2d**
- ∇_w e = 2d · (−1) · 1 · **x** = **−2d·x**
- ∇ₓ e = **−2d·w**

Ces trois formules correspondent à celles écrites en rouge sur la slide.

**Ce qu'il faut retenir (encadré de la slide) :**
- Un **graphe de calcul** est un **graphe orienté** qui décrit l'enchaînement des opérations.
- Les **sources** sont les variables d'entrée ; le résultat final est le nœud de sortie.
- Chaque opération élémentaire connaît sa propre dérivée. Le graphe permet donc de **calculer le gradient de la sortie par rapport à chaque variable d'entrée**, en combinant ces dérivées. C'est exactement ce que PyTorch fait pour nous, même pour des modèles avec des millions de paramètres.

### 3.5 Les autres verrous

- **Critère d'arrêt :** pour l'instant, on fixe simplement un nombre d'itérations (par exemple 5000). Plus tard, on pourra s'arrêter quand le coût ne diminue plus.
- **Performance :** il faut calculer une mesure de qualité du modèle (par exemple l'erreur moyenne), idéalement sur des données **non utilisées pour l'entraînement**.
- **Remettre le gradient à 0** après chaque mise à jour, pour éviter l'accumulation (voir 2.5).

### 3.6 Exemple complet assemblé

> Ce code est ma synthèse des éléments du cours ; ce n'est pas une slide. J'y ajoute une **normalisation** des descripteurs, qui n'est pas dans le cours mais qui aide beaucoup la descente de gradient (descripteurs d'échelles très différentes → apprentissage instable).

```python
import torch
from sklearn.datasets import fetch_california_housing

# 1) Données
housing = fetch_california_housing()
x = torch.tensor(housing['data'],   dtype=torch.float)                  # (n, d)
y = torch.tensor(housing['target'], dtype=torch.float).unsqueeze(1)     # (n, 1)  <- important
print(x.shape, y.shape)

# (ajout) normalisation : moyenne 0, écart-type 1 pour chaque descripteur
x = (x - x.mean(0)) / x.std(0)

# 2) Paramètres à apprendre
w = torch.randn(1, x.size(1), requires_grad=True)
b = torch.randn(1, 1,         requires_grad=True)

EPS   = 1e-2     # pas d'apprentissage
N_IT  = 5000     # critère d'arrêt simple : nombre d'itérations fixe

for it in range(N_IT):
    # 3) Passe forward : prédiction + coût
    y_hat = x @ w.T + b
    loss  = ((y - y_hat) ** 2).mean()

    # 4) Passe backward : calcul des gradients
    loss.backward()

    # 5) Mise à jour des paramètres (hors graphe)
    with torch.no_grad():
        w -= EPS * w.grad
        b -= EPS * b.grad

    # 6) Remise à zéro des gradients (sinon accumulation !)
    w.grad.zero_()
    b.grad.zero_()

    if it % 500 == 0:
        print(f"it {it:5d}  loss = {loss.item():.4f}")
```

**Le « rythme » à retenir** (on le retrouvera dans tous les modèles PyTorch) :

```
forward  →  loss  →  backward  →  update (no_grad)  →  zero_grad
```

---

## 4. Conclusion : PyTorch dans le paysage

### PyTorch, c'est…
- un **framework** de développement et d'apprentissage de réseaux profonds sur **CPU et GPU** ;
- une **architecture modulaire** (des « contenants » et des « conteneurs ») qui permet de construire des architectures **flexibles** ;
- la **différentiation automatique** : **Autograd** (cf. partie 2) ;
- une **couche d'abstraction pour l'optimisation** : une variété de descentes de gradient déjà programmées (on n'a plus à écrire `w -= EPS*w.grad` à la main) ;
- une **gestion simplifiée des données** pour constituer des **mini-batchs** (sous-ensembles de données utilisés à chaque pas de la descente).

### PyTorch vs TensorFlow

| | PyTorch | TensorFlow |
|---|---|---|
| Ancienneté / industrie | plus récent, moins intégré dans l'industrie | plus ancien, très intégré |
| Déploiement, rapidité, processus industriel | | ✔ |
| Flexibilité, prototypage, simplicité | ✔ | |

Les deux frameworks ont tendance à **se rapprocher** en termes de fonctionnalités ces derniers temps.

---

## 5. Cheatsheet PyTorch

### Création de tensors

| Commande | Effet |
|---|---|
| `torch.tensor([[1,2],[3,4]])` | tensor à partir de listes Python |
| `torch.tensor(data, dtype=torch.float)` | idem avec type imposé |
| `torch.zeros(n, m)` | matrice de 0 |
| `torch.ones(n, m)` | matrice de 1 |
| `torch.rand(n, m)` | uniforme sur [0, 1[ |
| `torch.randn(n, m)` | loi normale (moyenne 0, écart-type 1) |
| `torch.zeros_like(x)` / `torch.ones_like(x)` | même forme que `x` |
| `torch.arange(0, 10, 2)` | `[0, 2, 4, 6, 8]` |
| `torch.linspace(0, 1, 5)` | 5 valeurs régulièrement espacées de 0 à 1 |
| `torch.eye(n)` | matrice identité |

### Inspecter un tensor

| Commande | Effet |
|---|---|
| `x.shape` ou `x.size()` | forme |
| `x.size(1)` | taille de la dimension 1 |
| `x.dtype` | type des éléments |
| `x.ndim` | nombre de dimensions |
| `x.numel()` | nombre total d'éléments |
| `x.item()` | valeur Python (tensor à 1 élément) |
| `x.device` | où il est stocké (`cpu` / `cuda`) |

### Types courants

`torch.int8`, `torch.int16`, `torch.int32`, `torch.int64`, `torch.float16`, `torch.float32` (= `torch.float`), `torch.float64`, `torch.bool`

```python
x.float()                 # convertir en float32
x.to(torch.int16)         # convertir vers un type donné
```

### Indexation et slicing

| Commande | Effet |
|---|---|
| `x[i, j]` | élément (ligne `i`, colonne `j`) |
| `x[i]` ou `x[i, :]` | ligne `i` |
| `x[:, j]` | colonne `j` |
| `x[:, [0, 2]]` | colonnes 0 et 2 |
| `x[a:b]` | lignes de `a` à `b-1` |
| `x[::2]` | une ligne sur deux |
| `x[x > 0]` | tous les éléments > 0 (masque booléen) |

Syntaxe : `début:fin(exclue):pas`

### Opérations

| Commande | Effet |
|---|---|
| `x + y`, `x - y`, `x * y`, `x / y`, `x ** 2` | terme à terme (avec broadcasting) |
| `x @ y` ou `torch.matmul(x, y)` | produit matriciel |
| `x.T` | transposée |
| `torch.sin(x)`, `torch.exp(x)`, `torch.log(x)`, `torch.sqrt(x)` | fonctions usuelles |
| `torch.abs(x)`, `torch.clamp(x, min, max)` | valeur absolue, bornage |

### Opérations in place (suffixe `_`)

```python
x.add_(y)       # x = x + y
x.mul_(2)       # x = x * 2
x.zero_()       # x = 0
torch.sin_(x)   # x = sin(x)
```

### Agrégations

| Commande | Effet |
|---|---|
| `x.sum()` | somme totale |
| `x.sum(0)` | somme par colonne (dimension 0 écrasée) |
| `x.sum(1)` | somme par ligne (dimension 1 écrasée) |
| `x.mean()`, `x.std()`, `x.min()`, `x.max()` | idem avec d'autres fonctions |
| `x.argmax(1)` | indice du max sur chaque ligne |
| `x.sum(1, keepdim=True)` | garde la dimension (forme `(n, 1)` et non `(n,)`) |

### Changer la forme

| Commande | Effet |
|---|---|
| `x.unsqueeze(0)` | ajoute une dimension de taille 1 en position 0 |
| `x.squeeze()` | supprime les dimensions de taille 1 |
| `x.reshape(2, 6)` ou `x.view(2, 6)` | change la forme (même nombre d'éléments) |
| `x.reshape(-1)` | aplatit en vecteur |
| `x.reshape(n, -1)` | `-1` = « calcule pour moi » |
| `torch.cat([a, b], dim=0)` | concatène le long d'une dimension existante |
| `torch.stack([a, b])` | empile le long d'une **nouvelle** dimension |

### Gradient et autograd

| Commande | Effet |
|---|---|
| `torch.tensor(1., requires_grad=True)` | active le suivi du gradient |
| `a.requires_grad = True` | idem sur un tensor existant |
| `L.backward()` | calcule les gradients (`L` doit être un scalaire) |
| `a.grad` | gradient accumulé (même forme que `a`) |
| `a.grad.zero_()` | remet le gradient à 0 |
| `a.data` / `a.detach()` | valeur sans le graphe |
| `with torch.no_grad(): ...` | coupe le suivi du gradient (mises à jour, évaluation) |

### Passage entre NumPy et PyTorch

```python
t = torch.from_numpy(arr)     # NumPy -> torch (partage la mémoire)
a = t.numpy()                 # torch -> NumPy (sans gradient)
a = t.detach().numpy()        # si t a requires_grad=True
```

### GPU

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
x = x.to(device)              # déplace le tensor sur le GPU (si disponible)
```

### Squelette d'une boucle d'apprentissage (modèle écrit à la main)

```python
w = torch.randn(1, d, requires_grad=True)
b = torch.randn(1, 1, requires_grad=True)

for it in range(N_IT):
    y_hat = x @ w.T + b                    # 1. forward
    loss  = ((y - y_hat) ** 2).mean()      # 2. coût (scalaire)
    loss.backward()                        # 3. gradients
    with torch.no_grad():                  # 4. mise à jour
        w -= EPS * w.grad
        b -= EPS * b.grad
    w.grad.zero_()                         # 5. remise à zéro
    b.grad.zero_()
```

### Les erreurs classiques

| Symptôme | Cause probable |
|---|---|
| Résultat de coût absurde, sans erreur | broadcasting involontaire : vérifier `shape` (`(n,)` vs `(n,1)`) |
| `RuntimeError: ... expected ... float ... got ... double` | mélange de types : imposer `dtype=torch.float` |
| `grad can be implicitly created only for scalar outputs` | `backward()` appelé sur un non-scalaire : faire `.mean()` ou `.sum()` |
| `Trying to backward through the graph a second time` | graphe déjà libéré : recalculer le forward avant un nouveau `backward()` |
| Modification d'un paramètre refusée | oubli de `with torch.no_grad():` lors de la mise à jour |
| Apprentissage qui diverge ou stagne | `EPS` mal choisi, gradients non remis à zéro, données non normalisées |

---

*Fin du cours 0 : la suite (réseaux de neurones, optimiseurs, mini-batchs) s'appuie directement sur ces briques.*
