# Fiche de commandes : sélection de caractéristiques

Contenu de la fiche :

1. [Filtre naïf (corrélation variable / y)](#1-filtre-naïf-corrélation-variable--y) — B.1
2. [`SequentialFeatureSelector` (approche enveloppe)](#2-sequentialfeatureselector-approche-enveloppe) — B.2
3. [ACP : `PCA`](#3-acp--pca) — C
4. [Régularisation : Ridge, LASSO, Elastic Net](#4-régularisation--ridge-lasso-elastic-net) — D

---

## 1. Filtre naïf (corrélation variable / y)

**Package** : `numpy`

**Résumé** : le filtre naïf note chaque variable isolément par son lien avec `y` (une variable utile doit varier avec l'étiquette), classe les variables selon ce score et garde les meilleures. Il n'utilise aucun modèle et coûte seulement $d$ calculs élémentaires (c'est une approche **filtre**).

### Les 3 méthodes de calcul

```python
corr_brut = np.abs(X_train.T @ y_train)                         # produit scalaire brut, y dans {0,1}
corr_cent = np.abs(X_train.T @ (y_train*2-1)) / len(y_train)    # y recentré sur {-1,+1}
corr_pear = np.abs([np.corrcoef(X_train[:,j], y_train)[0,1]     # vraie corrélation de Pearson
                    for j in range(X_train.shape[1])])
```

| Méthode | Formule (variable $j$, $n$ exemples) | Remarque |
|---|---|---|
| Produit scalaire brut | $\lvert X_j^\top y\rvert = \left\lvert\sum_i x_{ij}\,y_i\right\rvert$, avec $y\in\{0,1\}$ | Ne somme que les exemples de la classe 1 et dépend de la moyenne de la variable : ce n'est **pas** une corrélation |
| $y$ recentré | $\dfrac{\lvert X_j^\top (2y-1)\rvert}{n}$, avec $2y-1\in\{-1,+1\}$ | Devient une vraie différence entre classes |
| Pearson | $\lvert r_j\rvert=\dfrac{\lvert\sum_i (x_{ij}-\bar x_j)(y_i-\bar y)\rvert}{\sqrt{\sum_i (x_{ij}-\bar x_j)^2}\,\sqrt{\sum_i (y_i-\bar y)^2}}$ | Borné entre 0 et 1 en valeur absolue, donc **comparable d'une variable à l'autre** : c'est la version propre |

### Les deux angles morts (détaillés en E.1 et E.2)

- **E.1 : interaction (XOR)**. Une variable utile *uniquement en interaction* avec une autre est invisible pour le filtre. Dans le XOR, chaque variable prise seule est indépendante de `y`, donc elle a un mauvais score, alors que le problème est bien séparable avec les deux ensemble. `f_classif` ne détecte que les liaisons linéaires ou monotones. Une approche enveloppe (sélection séquentielle) retrouve les bonnes variables.
- **E.2 : redondance**. Le filtre ne voit pas que des variables mesurent la même chose. Des quasi-copies d'une même variable utile reçoivent toutes le même bon score, et `SelectKBest(k=2)` risque de garder deux copies de `v0` au lieu de `v0` et `v1`. Ce défaut est structurel : chaque variable est notée sans tenir compte des autres. L'enveloppe l'évite, mais à un coût de calcul bien plus élevé.

### Attention : ce n'est pas toujours aussi simple que sur notre cas

Le critère marche très bien sur l'exemple jouet, où les deux variables utiles sortent largement du lot. Sur des données réelles, il faut le tester (« on ne sait jamais »), car les deux angles morts ci-dessus y sont fréquents, la redondance surtout (capteurs voisins, cm et pouces, chiffre d'affaires TTC et HT).

*Source : partie B.1 du notebook (réserve et angles morts annoncés), détails en E.1 et E.2.*

---

## 2. `SequentialFeatureSelector` (approche enveloppe)

**Package** : `sklearn.feature_selection`

**À quoi ça sert** : sélectionner un sous-ensemble de variables en évaluant des sous-ensembles avec un modèle. Soit on ajoute les variables une à une (*forward*), soit on retire les moins utiles une à une (*backward*).

### Hyperparamètres importants

- `estimator` : le **modèle** (n'importe quel classifieur ou régresseur scikit-learn, ex. `svm.SVC(kernel="linear")`) dont on utilise la performance comme critère pour comparer les sous-ensembles.
- `n_features_to_select` : le nombre de variables à garder. Le notebook le fixe à la main, ce qui n'est « pas très réaliste : il faut le tester » (E.5 propose une version qui le choisit automatiquement).
- `direction` : `"forward"` (par défaut, on part de 0 variable et on en ajoute une à la fois) ou `"backward"` (on part de toutes les variables et on en retire une à la fois).
- Autres, moins centraux : `cv` (validation croisée utilisée pour évaluer chaque sous-ensemble) et `scoring` (la métrique).

### Exemple (issu du notebook)

```python
from sklearn.feature_selection import SequentialFeatureSelector

estimator = svm.SVC(kernel="linear")
selector  = SequentialFeatureSelector(estimator, n_features_to_select=2)
selector  = selector.fit(X_train, y_train)     # trouve les variables à garder
print(np.where(selector.get_support())[0])     # indices retenus
Xnew = selector.transform(X_train)             # supprime les colonnes non retenues
```

**Logique scikit-learn** : `fit` fait tous les calculs, `transform` applique la sélection à `X` et à `X_test`.

### Pourquoi c'est une approche enveloppe (*wrapper*)

- **Définition** : une approche enveloppe essaie des sous-ensembles de variables et les évalue *pour un modèle donné*. Le modèle est « enveloppé » dans la procédure de sélection : il sert de juge.
- **Pourquoi ici** : le critère de choix est la performance de l'`estimator` sur chaque sous-ensemble candidat, ce qui implique de ré-apprendre le modèle à chaque essai. La sélection dépend donc du modèle choisi, contrairement au filtre naïf qui note chaque variable sans modèle.
- **Coût** : élevé, car chaque candidat demande des apprentissages. Il faut surveiller `n_features_to_select`, `direction` et `cv`.
- **Avantage** : contrairement au filtre, l'enveloppe évalue des variables *ensemble*, donc elle peut voir les interactions (cas du XOR) et éviter les redondances (E.1 et E.2).

---

## 3. ACP : `PCA`

**Package** : `sklearn.decomposition`

### Explication

L'ACP (Analyse en Composantes Principales) est une méthode **non supervisée** de réduction de dimension. Elle ne choisit pas de variables existantes : elle **construit de nouvelles variables**, combinaisons linéaires des anciennes. Elle cherche les directions de plus grande variance :

1. l'axe 1 est la direction dans laquelle les données sont le plus étalées ;
2. l'axe 2 est la direction de plus grande variance parmi celles **perpendiculaires** à l'axe 1 ;
3. et ainsi de suite.

### Les vecteurs propres et les axes perpendiculaires

*(complément de cours, absent du notebook)*

- Les axes sont les **vecteurs propres de la matrice de covariance** des données. Cette matrice est symétrique, donc ses vecteurs propres sont **orthogonaux** entre eux et de norme 1. Chaque valeur propre est la variance des données le long de l'axe correspondant.
- Un vecteur propre contient **un coefficient (un poids) par variable d'origine**. La coordonnée d'un exemple sur un axe est la somme pondérée de ses variables (centrées) avec les poids de cet axe.
- L'orthogonalité garantit que chaque nouvel axe capte de la variance **non déjà captée** par les précédents : les nouvelles variables sont **décorrélées**, donc sans redondance entre elles.
- Garder **tous** les axes revient à une simple **rotation** du repère : aucune information n'est perdue. C'est pourquoi, en gardant tous les axes, on retrouve la performance de la partie A.
- Les axes sont triés par variance décroissante ; garder les $k$ premiers, c'est projeter sur les $k$ directions les plus étalées.
- La variance expliquée par un axe est la variance des données **projetées sur cet axe** (la valeur propre associée). Le ratio (`explained_variance_ratio_`) est cette valeur propre divisée par la somme de toutes les valeurs propres.

### Limites à retenir

- Le critère est non supervisé : il n'utilise pas $y$. La direction la plus étalée n'est pas forcément celle qui sépare les classes (E.4).
- On perd l'interprétabilité : un axe est un mélange de toutes les variables.
- « **La variance n'est pas l'information** » : le critère « 90 % de variance » est mauvais quand le bruit occupe beaucoup de dimensions. On préfère le graphique de valeurs propres et on garde les axes situés avant la cassure (*elbow* / éboulis).
- L'ACP maximise la variance : la question de normaliser (`StandardScaler`) se pose quand les variables n'ont pas la même unité.

### Paramètres importants

- `n_components` : le nombre d'axes à conserver ou à calculer. Un entier donne un nombre d'axes, un flottant entre 0 et 1 donne la part de variance à conserver. Par défaut, tous les axes sont conservés.
- `whiten` (met les axes à variance 1) et `svd_solver` (algorithme de calcul) existent aussi mais sont moins centraux ici.

### Attributs utiles

- `components_` : les axes (un vecteur par axe, un coefficient par variable d'origine). Un axe peut être remis en forme pour l'interpréter (par exemple `pca.components_[k].reshape(16,16)` pour afficher un axe USPS comme une image).
- `explained_variance_ratio_` : la part de variance de chaque axe.
- `singular_values_` : les valeurs singulières.

### Méthodes : `fit` puis `transform`

- `pca.fit(X_train)` : calcule les axes et les valeurs propres (sans utiliser `y`).
- `pca.transform(X)` : **projette** des données sur les axes appris. On obtient les nouvelles variables, triées par variance décroissante. On garde ensuite les `k` premières colonnes pour ne conserver que `k` axes (`Xnew[:, :k]`), puis on apprend le modèle sur ces colonnes.
- **Règle importante** : la PCA est apprise **uniquement sur le train**, puis `transform` est appliqué au train **et** au test avec cette même PCA. On ne refait jamais un `fit` sur le test.

### Exemple (issu du notebook)

```python
from sklearn.decomposition import PCA

pca   = PCA()                        # ou PCA(n_components=10)
pca.fit(X_train)                     # y n'est PAS utilisé
Xnew  = pca.transform(X_train)       # projection du train
XnewT = pca.transform(X_test)        # projection du test avec la MÊME pca
Xd    = Xnew[:, :k]                  # on garde les k premiers axes
```

---

## 4. Régularisation : Ridge, LASSO, Elastic Net

**Package** : `sklearn.linear_model`

**Principe** : approche **intégrée** (*embedded*). On ajoute à la fonction de perte une pénalité sur les poids $w$. La sélection de variables se fait **pendant l'apprentissage** : un poids nul signifie que la variable est éliminée. Le paramètre $C$ des formules du notebook règle la force de la pénalité.

> **Attention au piège** : dans scikit-learn, ce multiplicateur s'appelle **`alpha`** pour `Ridge`, `Lasso` et `ElasticNet`. Plus `alpha` est grand, plus la régularisation est forte. Ce n'est pas le `C` de `SVC` ou de `LogisticRegression`, qui fonctionne à l'inverse (plus `C` est grand, plus la régularisation est faible).

### Les trois commandes

| | Ridge (L2) | LASSO (L1) | Elastic Net |
|---|---|---|---|
| **Commande** | `RidgeClassifier(alpha=a)` | `Lasso(alpha=a)` ou `lasso_path(X, y)` | `ElasticNet(alpha=a, l1_ratio=0.5)` |
| **Pénalité ajoutée** | $C\,\lVert w\rVert^2 = C\sum_j w_j^2$ | $C\sum_j \lvert w_j\rvert$ | $C\left(\rho\sum_j \lvert w_j\rvert+\frac{1-\rho}{2}\lVert w\rVert^2\right)$ |
| **Hyperparamètres clés** | `alpha` | `alpha` | `alpha` et `l1_ratio` ($=\rho$, part de L1 dans le mélange) |
| **Type de modèle** | Classifieur (`.predict` donne des classes ; `coef_[0]` pour compter les poids) | Régression : il faut seuiller à la main (`> 0.5`) pour obtenir des classes | Régression : même seuillage, et `coef_` est un simple vecteur (pas de `[0]`) |

### Exemples (issus du notebook)

```python
from sklearn.linear_model import RidgeClassifier, lasso_path, ElasticNet

# Ridge
mod = RidgeClassifier(alpha=a).fit(Xn_train, yn_train)
parcimonie = np.where(np.abs(mod.coef_[0]) > 1e-5, 1, 0).mean()

# LASSO : tout le chemin de régularisation d'un coup
alphas, coefs, dual_gaps = lasso_path(Xn_train - X_moy, yn_train - y_moy)
yhat = np.where((X - X_moy).dot(coefs[:, i]) + y_moy > 0.5, 1, 0)

# Elastic Net
mod = ElasticNet(alpha=a, l1_ratio=0.5).fit(Xn_train, yn_train)
yhat = np.where(mod.predict(Xn_train) > 0.5, 1, 0)
```

### Pièges à retenir

- `lasso_path` **n'a pas d'intercept**. Il faut centrer `X` et `y` avant l'appel, puis rajouter la moyenne de `y` à la prédiction. Sinon les performances plafonnent.
- `Lasso` et `ElasticNet` sont des modèles de **régression** : c'est à toi de fabriquer les étiquettes `{0,1}` en seuillant les prédictions.
- Pour mesurer la parcimonie, on compte les poids non nuls avec un seuil (`> 1e-5`) : $\text{Parcimonie}=\frac{\sum_j I_{w_j\neq 0}}{d}$.
- Les formules du notebook sont une version conceptuelle : scikit-learn met un facteur de normalisation légèrement différent sur le terme d'erreur pour `Lasso` et `ElasticNet`.

### Différences entre les trois

| | Ridge (L2) | LASSO (L1) | Elastic Net |
|---|---|---|---|
| **Effet sur les poids** | Les écrase vers 0, **sans jamais les annuler exactement** | **Annule exactement** certains poids | Écrase et annule, selon `l1_ratio` |
| **Sélectionne des variables ?** | Non | Oui | Oui, moins agressivement |
| **Pourquoi** | La dérivée de $w_j^2$ vaut $2w_j$ et disparaît près de 0 | La dérivée de $\lvert w_j\rvert$ vaut $\pm1$ même près de 0 : la pénalité pousse toujours avec la même force et maintient le poids à 0 | Combinaison des deux |
| **Variables corrélées** | Répartit le poids entre elles | Tend à en garder une seule, de façon parfois instable | Plus stable : peut garder le groupe (E.6 compare LASSO et Elastic Net) |

Cas limites de `l1_ratio` : `1` donne le LASSO pur, et une valeur proche de `0` donne quasiment du Ridge.

### Quand utiliser laquelle ?

*(conseils de cours et de pratique usuelle, pas directement du notebook)*

- **Ridge** : quand on veut surtout **limiter le sur-apprentissage** et que l'on pense que la plupart des variables contribuent un peu. Elle ne fait pas de sélection.
- **LASSO** : quand on soupçonne que **beaucoup de variables sont inutiles** et que l'on veut un modèle parcimonieux et interprétable.
- **Elastic Net** : quand on veut de la parcimonie mais que les variables sont **corrélées entre elles**. L2 stabilise et L1 sélectionne. C'est la formule du notebook : « jouer principalement sur la L2 et mettre un peu de L1 ».

**Bonnes pratiques générales** : choisir `alpha` (et `l1_ratio`) par validation croisée (il existe `RidgeCV`, `LassoCV`, `ElasticNetCV`), et normaliser les variables avant, car la pénalité dépend de l'échelle des variables.
