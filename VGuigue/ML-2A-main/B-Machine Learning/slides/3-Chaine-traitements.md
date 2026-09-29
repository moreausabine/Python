# Cours Machine Learning – Session 3 : chaîne de traitements

*Vincent Guigue – AgroParisTech / INRAE / Paris-Saclay*

## Introduction

**Idée centrale** : ce n'est pas aussi facile qu'un simple `fit()`. Il faut « bidouiller » le **prétraitement**, car c'est là que se joue une grande partie de la performance.

**Ce que la chaîne de traitements doit intégrer :**
- **Connaissances expert/métier** : construction de caractéristiques, sélection de caractéristiques, choix des métriques
- **Stabilisation numérique** : normalisation

**Schéma de la chaîne (exemple : pommes vertes/rouges)**
- Jeu de données (images de pommes) + supervision (Verte / Rouge)
- → **Pré-traitements** (ex. de l'expert : « regarder les niveaux moyens et max de rouge par zone »)
- → **Apprentissage** → **Classifieur** → **Évaluation**
- À l'**inférence**, une nouvelle image passe par **les mêmes pré-traitements** avant d'arriver au classifieur (→ Verte / Rouge)

> ⚠️ Les pré-traitements (normalisation, imputation…) sont **appris sur le train** puis **réappliqués tels quels** au test/à l'inférence.

**Objectifs de la session**
- Comprendre les principaux leviers de performance en machine learning (encore !)
- Savoir les mettre en œuvre dans **scikit-learn**

---

## 1. Valeurs manquantes

Pénible, mais incontournable.

**Comment ça se manifeste**
- Si on ne les voit pas : erreur au moment de l'apprentissage.
- Si elles sont codées par un `?` (ex. dataset *auto-mpg*, colonne `horsepower`) : pandas **bascule toute la colonne en chaîne de caractères**.
- → Il faut **forcer le type numérique** et passer tous les `?` en `NaN`.

```python
df["horsepower"] = pd.to_numeric(df["horsepower"], errors="coerce")
```

Les valeurs manquantes sont souvent **localisées sur une (ou quelques) colonne(s)**.

### 1.1 Variables continues

| Méthode | Remarque |
|---|---|
| **Supprimer la ligne** (« génial » 😅) | Perte de données ; OK si peu de lignes touchées. Supprimer la **colonne** si elle n'est remplie que de `?`. |
| **Valeur arbitraire** : moyenne / médiane | Simple, robuste. La médiane est moins sensible aux outliers. |
| **Plus proche voisin** | Copie la valeur du voisin le plus proche (sur les autres variables). |
| **Interpolation** | Estimation simple des données manquantes (surtout séries ordonnées). |
| **EM** (Expectation-Maximization) | Estimation statistique des données manquantes. |
| **Construire un modèle prédictif** | Toutes les autres variables = X, la colonne à compléter = Y. |

**Attention avec le modèle prédictif :**
- Il faut **beaucoup de données**.
- On est limité si **plusieurs variables manquent sur la même ligne** → discuter avec le terrain pour connaître **l'ordre** dans lequel les compléter.

**Conseil pratique** : commencer avec une **moyenne/médiane** pour pouvoir avancer et raisonner, raffiner ensuite.

```python
from sklearn.impute import SimpleImputer, KNNImputer
SimpleImputer(strategy="median")   # ou "mean"
KNNImputer(n_neighbors=5)
```

### 1.2 Variables discrètes

- **Valeur la plus fréquente** (`SimpleImputer(strategy="most_frequent")`)
- **Échantillonnage** (multinomial) selon la distribution observée
- **Plus proche voisin** (sur les autres caractéristiques)
- **Construire un prédicteur** (classifieur)

---

## 2. Feature engineering (construction de caractéristiques)

### 2.1 Variables discrètes

On n'aime pas les valeurs discrètes : les modèles veulent des nombres.

- **Cas binaire** → passer en 0/1 (ex. Gender : Female = 1, Male = 0).
- **Cas n-aire** (non binaire), ex. Color : jaune / vert / rouge
  - ❌ Ne **pas** faire 0-1-2 : cela impose un **ordre** et des distances (jaune serait « entre » rouge et vert), et le modèle s'en servira pour pondérer.
  - ✅ **`OneHotEncoder()`** : on ajoute **une colonne par catégorie** (Red / Yellow / Green, avec un seul 1 par ligne).

```python
from sklearn.preprocessing import OneHotEncoder
OneHotEncoder(min_frequency=6, sparse_output=False)
# (anciennement sparse=False, renommé sparse_output depuis sklearn 1.2)
```

- **Option intéressante** : `min_frequency` → **regrouper les catégories peu fréquentes** dans une catégorie « autres ».
- **Pour aller plus loin** : **ECoC** (Error-Correcting Output Codes), **Embeddings**.

> 💡 **Habitude à avoir** : faire plein d'**histogrammes** pour regarder ce qu'on ne connaît pas.

### 2.2 Ouverture vers le deep learning

*C'est quoi le deep learning ? Réponse assez ouverte… mais l'idée clé est* **apprendre la meilleure représentation** (conférence phare : **ICLR**, *International Conference on Learning Representations*).

- **Apprendre des représentations (et des distances) entre éléments discrets** : distance sémantique/grammaticale entre mots, entre profils utilisateurs (recommandation), entre graphes ou nœuds d'un graphe.
- **Apprendre des représentations d'objets complexes** : ex. en vision, projeter les images dans un espace de faible dimension sémantique.
- **Architectures génératives** : GPT, DALL·E… transférables d'une application à l'autre.
- **Architectures complexes** : différents objectifs / modalités de données → modélisation directe des contraintes métiers.

### 2.3 Simplification des données

Il faut réfléchir à **quel point on simplifie**.

**Binarisation**
- Ex. **USPS** (chiffres manuscrits) : les histogrammes de pixels montrent que beaucoup de pixels sont quasi toujours à 0 (ex. pixels 0, 18, 128) ; d'autres ont une distribution bimodale (pixel 37). → **pixel allumé ou éteint**.

**Discrétisation** (par quantiles ou intervalles linéaires)
- Ex. **CV (puissance fiscale)** des véhicules américains (*auto-mpg*) : les valeurs sont très diverses mais se **regroupent en catégories**.
- Question : garder la valeur continue ou **discrétiser** (ex. 3 colonnes / 3 classes) ? → **demander aux experts**.

```python
from sklearn.preprocessing import KBinsDiscretizer, Binarizer
KBinsDiscretizer(n_bins=3, strategy="quantile")   # ou "uniform"
```

### 2.4 Transformations arbitraires / métiers

- Passer tout au **log** (utile si la distribution est très asymétrique / pour réduire l'impact des grandes valeurs).
- **Créer des variables dérivées** : ex. un cochon avec longueur et largeur → si on a besoin du **volume**, on le calcule pour être sûr qu'il soit considéré (le modèle ne le devinera pas forcément).
- **Question du cours** : *comment gérer la dernière colonne (`car name`) d'*auto-mpg* ?* → variable textuelle à forte cardinalité : on peut extraire la **marque** (premier mot), puis l'encoder (one-hot avec `min_frequency`), ou l'ignorer si peu utile.

→ On **enlève** et on **rajoute** des colonnes pour avoir les informations **les plus utiles** au problème.

### 2.5 Exemple « checkers » (damier)

- Données : deux classes disposées en **damier** (alternance pair/impair sur les deux dimensions).
- **Classifieur linéaire par défaut** : scores CV ≈ **[0.5, 0.5, 0.3, 0.5, 0.4]** (hasard, il ne peut pas séparer un damier).
- **Ajout de colonnes** : intervalles binaires (pair/impair) sur les deux dimensions → scores CV ≈ **[0.75, 0.85, 0.9, 0.8, 0.85]**.
- ➡️ **Le feature engineering permet à un modèle simple de résoudre un problème non linéaire.**

### 2.6 Target encoding

- **Ultra puissant** mais **dangereux** : gros risque de **fuite de données train/test** (*data leakage*).
- Principe : pour une variable discrète, au lieu du OneHotEncoder, on encode chaque catégorie par la **moyenne de la target** dans cette catégorie.
  - Ex. workclass : State-gov → 0 ; Self-emp-not-inc → 1 ; Private → 1/3 (1 positif sur 3 exemples).
- Peut donner de gros **boosts de performance**, mais on **risque de tricher** : la target (Y) se retrouve dans les features.
- **Précautions** : calculer l'encodage **uniquement sur le train**, idéalement par **validation croisée interne** (*cross-fitting*) et/ou avec **lissage** (`sklearn.preprocessing.TargetEncoder`, depuis sklearn 1.3).

---

## 3. Normalisation

### 3.1 Pourquoi normaliser ? (par ordre d'usage)

1. **Améliorer les performances** : tirer parti d'informations à différentes échelles.
2. **Faciliter le réglage des hyper-paramètres** : mêmes ordres de grandeur ⇒ mêmes réglages.
3. **Respecter les propriétés du modèle utilisé** : ex. hypothèse multinomiale ; modèles sans biais (normalisation des *y*).

### 3.2 Normalisation gaussienne (par colonne)

On **centre-réduit** chaque variable :

$$Z_j = \frac{X_j - \mu_j}{\sigma_j}$$

- Les colonnes deviennent **comparables**.
- = **normalisation standard** (`StandardScaler` dans scikit-learn). **Un test à faire systématiquement** (implémentée dans tous les systèmes).

**Cela dépend du modèle derrière :**
- **Réseaux de neurones** : normalisation **obligatoire** (on ne sait pas faire autrement, optimisation sensible aux échelles).
- **Arbres de décision / Gradient Boosting** : on raisonne en termes de **coupures** sur les données → la normalisation ne sert **à rien** (invariants aux transformations monotones).
- Modèles à distances / linéaires régularisés (kNN, SVM, régression ridge/lasso) : très sensibles à l'échelle.

### 3.3 Normalisation par individu (par ligne)

⚠️ **Attention : lié à la nature des données.** On normalise **~90 % du temps les colonnes** et **~10 % les lignes**.

Utile pour : les **textes** (éviter l'impact de la longueur du document) et les **séries temporelles / signaux** (normalisation par individu pour que la « puissance » d'un signal ne soit pas sur-représentée).

| Type | Formule | Propriété |
|---|---|---|
| **min-max** | $\tilde{x}_i = \dfrac{x_i - \min(x_i)}{\max(x_i) - \min(x_i)}$ | valeurs dans [0, 1] |
| **multinomiale** | $\tilde{x}_i = \dfrac{x_i}{\sum_j x_{ij}}$ | $\sum_j \tilde{x}_{ij} = 1$ |
| **produits scalaires** (matrices de Gram) | $\tilde{x}_i = \dfrac{x_i}{\sqrt{\sum_j x_{ij}^2}}$ | $\lVert x_i \rVert^2 = 1$ |

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
StandardScaler()          # par colonne
MinMaxScaler()            # par colonne
Normalizer(norm="l1")     # par ligne (somme à 1) ; "l2" pour norme euclidienne = 1
```

*(Note : la formule du cours pour la version « produits scalaires » omet la racine ; pour avoir $\lVert x_i\rVert^2=1$ il faut bien diviser par $\sqrt{\sum_j x_{ij}^2}$.)*

---

## 4. Conclusion

### Exemple : classification de signaux

Que veut-on comme propriétés ?
- **Détection de motifs indépendamment de l'échelle** : normalisation par individu (min-max) ; somme à 1 (signaux positifs), somme des carrés à 1, somme à 0…
- **Invariance en translation** (ou non, selon la nature des informations discriminantes) : calcul de moyennes, écarts-types ; **FFT**, **PSD** (densité spectrale de puissance).

### Approche standard en ML

1. **Traitements = interprétation des informations métier/expert.**
2. **Batteries de tests classiques** pour estimer les performances attendues :
   - normalisation standard
   - modèles linéaires + forêts
   - validation croisée (ou autre selon les cas)
3. **Optimisation** (idéalement automatisée ou semi-automatisée) :
   - feature engineering
   - grid-search
   - ensembling

**À retenir** : beaucoup d'outils existent dans sklearn ⇒ **maîtrise + documentation**. Lorsqu'il manque un outil ⇒ **penser intégration** (écrire son propre transformer compatible `Pipeline`).

### Pour tout intégrer proprement : `Pipeline` + `ColumnTransformer`

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

num = Pipeline([("imp", SimpleImputer(strategy="median")),
                ("scale", StandardScaler())])
cat = Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                ("ohe", OneHotEncoder(min_frequency=6, handle_unknown="infrequent_if_exist"))])

prep = ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])
model = Pipeline([("prep", prep), ("clf", RandomForestClassifier())])
# cross_val_score(model, X, y) → le prétraitement est refait dans chaque fold : pas de fuite
```