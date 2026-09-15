# Les commandes à connaître (NumPy & Scikit-Learn)

## Présentation des bibliothèques

* **NumPy** : Bibliothèque fondamentale pour le calcul scientifique en Python. Elle offre des structures de tableaux multidimensionnels (`ndarray`) ultra-rapides et des fonctions mathématiques vectorisées.
* **Scikit-Learn** : Bibliothèque de référence pour l'apprentissage automatique (*machine learning*) en Python. Elle propose des outils clés en main pour la prédiction, la classification, la régression et le prétraitement des données.

---

## Les classiques (NumPy)

Soit `A` une matrice NumPy de dimension 4 lignes et 3 colonnes (`shape = (4, 3)`).

### `min` / `max`
* **Explication :** Renvoie la valeur minimale ou maximale globale de la matrice, ou le long d'un axe spécifique.
* **Syntaxe :**
  ```python
  np.min(A)  # ou A.min()
  np.max(A)  # ou A.max()
  ```
* **Min/Max par ligne (`axis=1`) :**
  ```python
  np.max(A, axis=1)  # ou np.max(A, 1)
  ```
  Parcourt chaque ligne et extrait le maximum $\rightarrow$ renvoie un vecteur 1D de 4 éléments (`shape = (4,)`).
* **Min/Max par colonne (`axis=0`) :**
  ```python
  np.max(A, axis=0)  # ou np.max(A, 0)
  ```
  Parcourt chaque colonne et extrait le maximum $\rightarrow$ renvoie un vecteur 1D de 3 éléments (`shape = (3,)`).
* **Forme du résultat & `reshape(-1, 1)` :**
  ```python
  maxA = np.max(A, axis=1).reshape(4, 1)
  # Alternative equivalente :
  maxA = np.max(A, axis=1).reshape(-1, 1)
  ```
  `reshape(-1, 1)` transforme le vecteur 1D `(4,)` en une matrice 2D colonne `(4, 1)`. Le `-1` indique à NumPy de calculer automatiquement le nombre de lignes en fonction du nombre d'éléments.

---

### `argmin` / `argmax`
* **Explication :** Renvoie l'**indice** (la position) de la valeur minimale ou maximale au lieu de la valeur elle-même.
* **Syntaxe :**
  ```python
  np.argmin(A)
  np.argmax(A, axis=1)
  ```
* **Spécificités :** Sans le paramètre `axis`, la matrice est "aplatie" (considérée comme un seul grand vecteur) et renvoie un seul indice entier.

---

### `mean`
* **Explication :** Calcule la moyenne arithmétique globale ou le long d'un axe.
* **Syntaxe :**
  ```python
  np.mean(A)        # Moyenne globale
  A.mean(axis=0)    # Moyenne par colonne
  ```
* **Spécificité :** Très sensible aux valeurs extrêmes (*outliers*).

---

### `std`
* **Explication :** Calcule l'écart-type (*standard deviation*), mesurant la dispersion des données autour de la moyenne.
* **Syntaxe :**
  ```python
  np.std(A)               # Écart-type de la population (ddof=0)
  np.std(A, ddof=1)       # Écart-type échantillonnel corrigé
  ```
* **Spécificité :** Par défaut dans NumPy, `ddof=0`. Pour avoir l'estimation non biaisée (comme dans R ou Pandas), utiliser `ddof=1`.

---

## Fonctions fondamentales de NumPy

### `np.random`
* **Explication :** Module pour la génération de tableaux de nombres aléatoires.
* **Syntaxe & Spécificités :**
  ```python
  # Loi uniforme sur [0, 1[
  B = np.random.rand(4, 6)

  # Loi normale centrée réduite N(0, 1)
  C = np.random.randn(4, 6)

  # Entiers aléatoires bornés entre low (inclus) et high (exclus)
  D = np.random.randint(low=0, high=10, size=(4, 3))
  ```

---

### `np.where`
* **Explication :** Recherche les indices des éléments vérifiant une condition logique.
* **Syntaxe :**
  ```python
  I, J = np.where(B > 0.5)
  ```
* **Spécificités :**
  * Pour une matrice 2D, `np.where` renvoie un tuple de 2 tableaux : `I` contient les indices des lignes, et `J` les indices des colonnes correspondantes.
  * Forme alternative (remplacement conditionnel) :
    ```python
    A_mod = np.where(A > 0, A, 0)  # Si A > 0 garder A, sinon mettre 0
    ```

---

### `zip`
* **Explication :** Fonction native Python (hors NumPy) qui associe les éléments de deux itérables index par index sous forme de tuples.
* **Syntaxe & Utilisation :**
  ```python
  for i, j in zip(I, J):
      print(B[i, j])
  ```
* **Spécificité :** Ne pas utiliser `range(zip(...))`. `zip` est déjà un itérateur.

---

### `np.sum`
* **Explication :** Effectue la somme des éléments d'un tableau ou compte les éléments vérifiant une condition.
* **Syntaxe :**
  ```python
  np.sum(B)           # Somme de tous les éléments
  np.sum(B > 0.5)     # Compte le nombre d'éléments > 0.5
  ```
* **Spécificité :** Appliqué à un masque booléen, `True` vaut `1` et `False` vaut `0`.

---

### Sélection conditionnelle & Indexation
* **Explication :** Filtrer des lignes ou des colonnes selon des critères complexes.
* **Syntaxe :**
  ```python
  # Sélection des lignes dont le max est > 7
  i = np.where(A.max(axis=1) > 7)[0]
  A2 = A[i, :]
  ```
* **Spécificités :**
  * Le symbole `:` indique qu'on conserve toutes les colonnes.
  * `A.max(axis=1)` renvoie un vecteur 1D, donc `np.where` renvoie un tuple avec un seul tableau d'indices `[0]`.

---

### `np.concatenate`
* **Explication :** Assemble plusieurs matrices selon un axe donné.
* **Syntaxe :**
  ```python
  np.concatenate((A, B), axis=0)  # Empilement vertical (lignes)
  np.concatenate((A, C), axis=1)  # Empilement horizontal (colonnes)
  ```
* **Spécificité :** Les matrices doivent obligatoirement avoir les mêmes dimensions sur tous les axes sauf celui du collage.

---

### Déterminer les dimensions d'un tableau
* **Propriétés principales :**
  ```python
  A.shape  # Tuple indiquant la taille (ex: (4, 3))
  A.ndim   # Nombre de dimensions (ex: 2)
  A.size   # Nombre total d'éléments (ex: 12)
  ```

---

### Autres fonctions utiles de NumPy
```python
np.zeros((4, 3))               # Matrice remplie de 0
np.ones((4, 3))                # Matrice remplie de 1
np.arange(0, 10, 2)            # Vecteur [0, 2, 4, 6, 8]
np.linspace(0, 1, 5)           # 5 points équidistants entre 0 et 1
A @ B                          # Produit matriciel (ou np.dot(A, B))
```

---

## Modèles Scikit-Learn

Tous les algorithmes dans Scikit-Learn utilisent l'interface unifiée `Estimator`.

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
```

### `.fit`
* **Explication :** Entraîne le modèle à partir des données transmises (calcule les paramètres interne du modèle).
* **Syntaxe :**
  ```python
  model.fit(X_train, y_train)
  ```
* **Spécificité :** `X_train` doit être une matrice 2D `(n_échantillons, n_caractéristiques)` et `y_train` un vecteur 1D.

---

### `.predict`
* **Explication :** Effectue des prédictions brutes sur de nouvelles données.
* **Syntaxe :**
  ```python
  y_pred = model.predict(X_test)
  ```
* **Spécificité :** Renvoie la classe prédite (classification) ou la valeur estimée (régression).

---

### `.predict_proba`
* **Explication :** Calcule les probabilités d'appartenance à chaque classe (uniquement en classification).
* **Syntaxe :**
  ```python
  proba = model.predict_proba(X_test)
  ```
* **Spécificité :** Renvoie une matrice `(n_échantillons, n_classes)` dont la somme de chaque ligne est égale à 1.