# Sélection de caractéristiques – cheatsheet

*Synthèse du notebook `3-notebook-selfeat` (parties A à E)*

## Les 4 familles

| Famille | Principe | Coût | Outil sklearn |
|---|---|---|---|
| **Filtre** | note chaque variable seule, indépendamment du modèle | très faible | `SelectKBest`, `SelectPercentile` |
| **Enveloppe** | teste des sous-ensembles avec un modèle donné | élevé | `SequentialFeatureSelector`, `RFE`, `RFECV` |
| **Intégrée** | la sélection est un effet de l'apprentissage | faible | `Lasso`, `ElasticNet` |
| **Extraction** | recombine les variables | faible | `PCA` (supervisé : `LDA`) |

---

## A. Fléau de la dimensionnalité

- Plus de variables ≠ plus d'information : en grande dimension, les points deviennent quasi **équidistants**.
- Les modèles à distance (kNN, SVM gaussien) apprennent par cœur : **le train reste haut, le test s'effondre**.
- Expérience : 2 dimensions utiles + 20 de bruit pur, `gamma` fixé.

```python
Noise = np.random.randn(n, 20) * std
Xn = np.concatenate((X, Noise), axis=1)
for ndim in range(2, 23, 2):
    mod.fit(X_train[:, :ndim], y_train)
```

> ⚠️ Fixer `np.random.seed` pour comparer les résultats.

---

## B. Sélectionner un sous-ensemble

### B.1 Filtre : corrélation avec y

- Score par variable, on garde les meilleures : seulement *d* calculs.
- Le produit scalaire brut `|Xᵀy|` n'est **pas** une corrélation (avec y∈{0,1} il dépend de la moyenne).
- Utiliser **Pearson** (borné [−1, 1]) ou recentrer y sur {−1, +1}.

```python
np.abs([np.corrcoef(X[:, j], y)[0, 1] for j in range(X.shape[1])])
```

> ⚠️ **Deux angles morts** : les interactions (XOR) et la redondance (des copies d'une même variable sont toutes bien notées).

### B.2 Enveloppe : sélection séquentielle

```python
from sklearn.feature_selection import SequentialFeatureSelector
sel = SequentialFeatureSelector(svm.SVC(kernel="linear"),
        n_features_to_select=2, direction="forward")   # ou "backward"
sel.fit(X_train, y_train)
np.where(sel.get_support())[0]    # variables retenues
Xnew = sel.transform(X_train)
```

- `fit` trouve les variables, `transform` supprime les colonnes.
- **Forward** : part de 0 variable et en ajoute. **Backward** : part de *d* et en retire (22 → 2 : environ 20 itérations, donc plus lent ici).
- Très coûteux (des centaines d'apprentissages). Le nombre de variables doit se tester → RFECV.

---

## C. ACP (PCA)

- **Construit** de nouvelles variables (combinaisons linéaires) qui maximisent la **variance**.
- **Non supervisée** (n'utilise pas y) ; perte d'interprétabilité.
- Nombre d'axes : diagramme d'**éboulis** (garder les axes avant la cassure). Le critère « 90 % de variance » est un mauvais conseiller : *la variance n'est pas l'information*.
- Garder tous les axes = retrouver la performance dégradée de la partie A.

```python
pca = PCA(n_components=10).fit(X_train)    # fit sur le TRAIN uniquement
Z  = pca.transform(X_train)
Zt = pca.transform(X_test)
pca.explained_variance_ratio_ ; pca.components_
```

**USPS** : `components_[k].reshape(16, 16)` affiche un axe comme une image.

---

## D. Régularisation (sélection pendant l'apprentissage)

| | Pénalité | Effet |
|---|---|---|
| **Ridge (L2)** | $C\lVert w\rVert^2$ | écrase les poids, **jamais exactement 0** (dérivée $2w \to 0$) |
| **LASSO (L1)** | $C\sum_j \lvert w_j\rvert$ | annule des poids (dérivée ±1) → **parcimonie** |
| **Elastic Net** | $C\left(\rho\sum_j \lvert w_j\rvert + \frac{1-\rho}{2}\lVert w\rVert^2\right)$ | compromis ; `l1_ratio` = ρ |

- Parcimonie = (nombre de $w_j \neq 0$) / *d*.
- Forte régularisation : modèle simple mais peu performant. Faible régularisation : sur-apprentissage.

```python
RidgeClassifier(alpha=a)               # coef_[0]
ElasticNet(alpha=a, l1_ratio=.5)       # coef_ (sans [0])
np.abs(coef) > 1e-5                    # seuiller pour compter les poids non nuls
```

> ⚠️ `Lasso` / `ElasticNet` sont des modèles de **régression** : seuiller `predict > 0.5` pour obtenir des classes. `lasso_path` n'a **pas d'intercept** : centrer X et y avant.

---

## E. Pièges et compléments

### E.1–E.2 Pièges des filtres

- `f_classif` (ANOVA) : liaisons linéaires ou monotones seulement. `mutual_info_classif` : peut voir le non linéaire.
- **XOR** : chaque variable seule est indépendante de y → le filtre échoue ; une enveloppe retrouve les bonnes variables.
- **Redondance** : le filtre garde 2 copies de la même variable.
- Bon usage : un premier tri bon marché, pas un verdict.

```python
sel = SelectKBest(f_classif, k=2).fit(X, y)
sel.scores_ ; sel.pvalues_
```

### E.3 Fuite de données (*data leakage*)

Sélectionner sur **tout X** puis valider par CV : les étiquettes des plis de test ont déjà servi. Sur du bruit pur, on obtient une performance > 0.5 (artefact), d'autant plus que *d* ≫ *n*.

```python
pipe = Pipeline([("sel", SelectKBest(f_classif, k=20)),
                 ("clf", svm.SVC(kernel="linear"))])
cross_val_score(pipe, X, y, cv=5)    # sélection refaite dans chaque pli
```

Même règle pour la normalisation, l'imputation, la PCA, le target encoding : **tout dans le pipeline**.

### E.4 Pièges de l'ACP

- **Échelle** : une variable en grande unité capte l'axe 1 → `StandardScaler` avant PCA (sauf si toutes les variables ont la même unité, ex. pixels).
- **Pas d'étiquettes** : l'axe le plus étalé n'est pas forcément discriminant.
- **LDA** : supervisée, maximise la variance inter-classes sur l'intra-classes. Limite : au plus **K−1 axes** pour K classes.

```python
LinearDiscriminantAnalysis(n_components=1).fit(X, y)
```

### E.5 RFE / RFECV

- **RFE** : un seul modèle par tour, retire les variables de plus petit poids ; bien moins cher que le séquentiel.
- Suppose que le modèle expose `coef_` ou `feature_importances_`. `ranking_` : 1 = retenue, puis ordre d'élimination.
- **RFECV** choisit le nombre de variables par validation croisée.
- Un SVM linéaire est bien moins sensible au bruit qu'un noyau gaussien à `gamma` fixe.

```python
rfecv = RFECV(svm.SVC(kernel="linear"), cv=5).fit(X, y)
rfecv.n_features_
rfecv.cv_results_["mean_test_score"]
```

### E.6 Stabilité : LASSO vs Elastic Net

- Variables corrélées : LASSO en garde **une, arbitrairement** (elle change selon l'échantillon).
- Elastic Net : effet de groupe, retient ou élimine les variables corrélées ensemble.
- Mesure : bootstrap, fréquence de sélection (*stability selection*).
- Comparaison équitable : `alpha_EN × l1_ratio = alpha_LASSO`.
- LASSO pour prédire avec peu de variables ; Elastic Net pour interpréter.

---

## Que choisir ?

- **Beaucoup de variables, peu de temps** → filtre (puis vérifier interactions et redondance).
- **Modèle fixé, budget de calcul** → RFE / RFECV plutôt que séquentiel.
- **Sélection pendant l'apprentissage** → LASSO / Elastic Net.
- **Visualiser ou compresser sans y** → PCA (standardiser si unités différentes) ; avec y → LDA.
- Toujours : **sélection dans le Pipeline**, jamais avant la validation croisée.