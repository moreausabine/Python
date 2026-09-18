# Cheat Sheet — Notebook d'introduction à Scikit-learn

*Basé sur le notebook `1-notebook-intro.ipynb`*

---

## Partie 1 — Fonctions essentielles (référence rapide)

### Imports de base

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets, svm, linear_model, naive_bayes, tree
```

### A. Charger / fabriquer des données

| Fonction | Rôle | Exemple |
|---|---|---|
| `datasets.load_iris()` | Charge un jeu de données classique déjà en mémoire (retourne un dictionnaire-like) | `iris = datasets.load_iris()` puis `iris.data`, `iris.target` |
| `make_blobs(n_samples=, centers=, cluster_std=, n_features=, random_state=)` | Génère des données jouet en nuages gaussiens, très pratique pour visualiser en 2D | `X, y = make_blobs(n_samples=100, centers=[[0,0],[1.5,1.5]], cluster_std=[1.5,1.5], n_features=2)` |
| `train_test_split(X, y, test_size=, random_state=)` | Découpe propre en train/test (voir aussi la version manuelle ci-dessous) | `X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)` |

**Découpage manuel (numpy pur, à savoir refaire sans sklearn) :**
```python
pcapp   = 0.7
ind     = np.random.permutation(len(X))        # mélange des indices
indapp  = ind[:int(pcapp*len(X))]
indtest = ind[int(pcapp*len(X)):]
Xapp, Yapp   = X[indapp],  Y[indapp]
Xtest, Ytest = X[indtest], Y[indtest]
```

### B. Le triptyque universel de sklearn : `fit` / `predict` / `predict_proba`

**Tous** les modèles sklearn (quel que soit le type : bayésien, SVM, arbre, forêt, boosting...) suivent exactement le même protocole :

```python
mod = naive_bayes.GaussianNB()   # 1. création du modèle + choix des hyperparamètres
mod.fit(Xapp, Yapp)              # 2. apprentissage sur les données d'apprentissage
yhat  = mod.predict(Xtest)       # 3a. prédiction d'une CLASSE (attend un ensemble de points, même pour un seul point : [Xtest[0]])
proba = mod.predict_proba(Xtest) # 3b. prédiction d'un SCORE/probabilité par classe (matrice n × nb_classes)
```

**Piège classique :** `predict` attend toujours un **tableau 2D** de points, jamais un seul point isolé — pour prédire sur un unique exemple, il faut l'entourer de crochets : `mod.predict([Xtest[0]])`.

| Méthode | Rôle | Disponible sur |
|---|---|---|
| `.fit(X, y)` | Entraîne le modèle | tous les modèles |
| `.predict(X)` | Renvoie la classe prédite | tous les modèles |
| `.predict_proba(X)` | Renvoie une probabilité par classe (matrice n×k) | la plupart, à condition d'avoir activé la bonne option à la création (ex. `probability=True` pour un SVM) |
| `.decision_function(X)` | Score continu signé (pas une probabilité) | modèles linéaires / SVM |
| `.score(X, y)` | Raccourci pratique = accuracy directe | tous les modèles |

**Reconstruire une classe à partir d'un `predict_proba` (à savoir faire à la main) :**
```python
yhat_prob   = mod.predict_proba(Xtest)
yhat_deduit = np.argmax(yhat_prob, axis=1)      # classe = celle qui a la proba la plus haute
taux        = np.mean(yhat_deduit == Ytest)
```

**Technique de rejet des points ambigus** (utile en contexte opérationnel : ne classer que ce dont on est sûr) :
```python
# écart entre les deux probabilités = mesure de confiance (petit écart = ambigu)
confiance = np.abs(yhat_prob[:, 0] - yhat_prob[:, 1])
ordre     = np.argsort(confiance)               # du plus ambigu au plus sûr
seuil     = round(len(yhat_prob) * 0.05)         # on rejette les 5% les plus ambigus
ordre_95  = ordre[seuil:]                        # indices des 95% restants (les plus sûrs)
taux_95   = np.mean(np.argmax(yhat_prob[ordre_95], axis=1) == Ytest[ordre_95])
```

### C. Comparer des modèles / hyperparamètres (version simple, sans validation croisée)

```python
mod1, mod2 = naive_bayes.GaussianNB(), svm.SVC()
mod1.fit(Xapp, Yapp); mod2.fit(Xapp, Yapp)
yhat1, yhat2 = mod1.predict(Xtest), mod2.predict(Xtest)

print("perf modèle 1", np.where(yhat1 == Ytest, 1, 0).mean())   # = accuracy à la main
print("perf modèle 2", np.where(yhat2 == Ytest, 1, 0).mean())
```
Même logique pour comparer deux **jeux d'hyperparamètres** du même modèle (ex. noyau `linear` vs `gamma=10` sur un SVM) : la comparaison se fait toujours en rejouant fit/predict/score sur chaque candidat.

### D. Visualiser un classifieur en 2D (outils fournis dans `outils/frontiere.py`)

```python
from outils.frontiere import plot_frontiere, plot_mesh
%load_ext autoreload
%autoreload 2          # recharge automatiquement le module si le fichier .py est modifié
```

| Fonction | Rôle | Contrainte |
|---|---|---|
| `plot_frontiere(X, y, mod)` | Trace la frontière de décision 2D + les points | fonctionne avec n'importe quel modèle ayant `.predict` |
| `plot_mesh(X, y, mod)` | Trace la fonction de décision en 3D (surface) | nécessite `predict_proba` **et** une figure ouverte en 3D (`fig.add_subplot(projection='3d')`) — ne marche qu'avec les SVM dans ce notebook |

```python
plt.figure(figsize=(12,4), facecolor='white')
plt.subplot(1,3,1)
plot_frontiere(Xapp, Yapp, mod1)
plt.scatter(Xapp[:,0], Xapp[:,1], c=Yapp)
plt.title('Naive Bayes')
```

### E. Introspection : aller chercher ce que le modèle a appris

**Chaque famille de modèle stocke ses paramètres appris dans des attributs différents — il faut lire la documentation à chaque fois.**

| Modèle | Attribut(s) à explorer | Signification |
|---|---|---|
| `naive_bayes.GaussianNB()` | `.theta_` (moyennes), `.var_` (variances, `.sigma_` dans les vieilles versions) | une gaussienne par variable, par classe |
| `svm.SVC(kernel='linear')` | `.support_vectors_`, `.support_` (indices), `.coef_` (uniquement noyau linéaire) | les points qui définissent la frontière (vecteurs supports) |
| `linear_model.LogisticRegression()` | `.coef_` (poids par variable), `.intercept_` (biais) | importance/sens de chaque variable |
| `tree.DecisionTreeClassifier()` | `tree.plot_tree(mod)` | dessine l'arbre de décision complet, lisible directement |
| `RandomForestClassifier()` | `.n_estimators`, `.max_depth`, `.estimators_[i]` (accès à chaque arbre individuel), `.feature_importances_` | nombre/profondeur des arbres, importance des variables |
| `xgb.XGBClassifier()` | `.feature_importances_`, `xgb.plot_importance(bst)` | importance des variables (attention aux pièges d'interprétation, cf. lien dans le notebook) |

```python
# Naive Bayes : une gaussienne par variable et par classe
print(mod1.theta_)   # moyennes
print(mod1.var_)     # variances

# Modèle linéaire : importance des variables = valeur (absolue) des coefficients
poids = mod1.coef_[0]
plt.bar(range(len(poids)), poids)   # ou np.abs(poids), les deux se défendent

# Arbre de décision : visualiser directement l'algorithme appris
plt.figure(facecolor='white')
tree.plot_tree(mod, filled=True)    # filled=True colore les branches selon la classe majoritaire
```

### F. Focus arbre de décision

```python
from sklearn import tree
mod = tree.DecisionTreeClassifier()          # profondeur illimitée par défaut => risque de sur-apprentissage
mod = tree.DecisionTreeClassifier(max_depth=2)  # profondeur limitée = modèle plus simple, plus lisible
mod.fit(Xapp, Yapp)
plot_frontiere(Xapp, Yapp, mod)
plt.figure(); tree.plot_tree(mod, filled=True)
```
**Pourquoi l'arbre est-il plus « explicable » que les autres modèles ?** Parce que la décision se lit directement comme une suite de tests sur les variables (if/else), sans aucune boîte noire — contrairement aux poids d'un SVM à noyau gaussien ou aux centaines d'arbres d'une forêt.

### G. Focus SVM (linéaire vs noyau gaussien)

```python
mod_SVM1 = svm.SVC(kernel="linear", probability=True)   # frontière = droite/hyperplan
mod_SVM2 = svm.SVC(gamma=10, probability=True)           # noyau gaussien (rbf) par défaut, gamma = largeur de bande
```
**Effet de `gamma` :** plus `gamma` est grand, plus chaque point d'apprentissage n'influence qu'une toute petite zone autour de lui → frontière très découpée, sur-apprentissage quasi immédiat. Plus `gamma` est petit, plus la frontière reste lisse (proche du linéaire).

### H. Modèles ensemblistes (état de l'art)

```python
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb   # pip install xgboost si besoin

mod = RandomForestClassifier()                       # bagging : arbres indépendants + vote majoritaire
mod = RandomForestClassifier(max_depth=1, n_estimators=5)    # très simple (sous-apprentissage probable)
mod = RandomForestClassifier(max_depth=10, n_estimators=100) # complexe (risque de sur-apprentissage)

bst = xgb.XGBClassifier().fit(X_train, y_train)       # boosting : arbres ajoutés itérativement pour corriger les erreurs
```
Les deux s'utilisent **exactement** comme n'importe quel autre modèle sklearn (`.fit`, `.predict`) — c'est tout l'intérêt de l'interface unifiée.

### I. Créer son propre classifieur (extension de scikit-learn)

Pour qu'un modèle « maison » soit compatible avec tous les outils sklearn (validation croisée, grid search...), il suffit d'hériter de deux classes et d'implémenter `fit`/`predict` :

```python
from sklearn.base import BaseEstimator, ClassifierMixin

class LinearFixClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, data_dim=2):
        self.data_dim = data_dim          # stocker TOUS les attributs (nécessaire pour la sérialisation)
        self.w = np.random.randn(data_dim)

    def fit(self, X, y):
        self.classes_ = np.unique(y)      # convention sklearn : stocker les classes vues
        self.X_ = X
        self.y_ = y
        return self                       # fit doit toujours retourner self

    def predict(self, X):
        return np.where(X @ self.w > 0, self.y_[0], self.y_[1])
```
Une fois ces deux méthodes définies, le modèle fonctionne directement avec `cross_val_score`, `GridSearchCV`, etc., sans rien coder de plus :
```python
from sklearn.model_selection import cross_val_score
scores = cross_val_score(mod, X, y, cv=5, scoring='accuracy')
```

---

## Partie 2 — Fiche décision : quel outil pour quel cas ?

### 1. « Je veux juste essayer un modèle rapidement »
**Utiliser :** le triptyque `mod = Modele(...)` → `mod.fit(X, y)` → `mod.predict(X)`
**Pourquoi :** interface strictement identique quel que soit le modèle (bayésien, SVM, arbre, forêt, boosting...) — c'est tout l'intérêt de sklearn.
**Exemple :** passer d'un `GaussianNB()` à un `SVC()` ne change qu'une seule ligne de code.

### 2. « Je veux la classe ET la confiance associée »
**Utiliser :** `predict` pour la classe, `predict_proba` pour la confiance (ou `decision_function` sur les modèles linéaires/SVM si `predict_proba` n'est pas activé).
**Pourquoi :** `predict` seul ne dit rien sur la certitude du modèle — deux prédictions correctes peuvent être une certitude à 99% ou un quasi pile-ou-face à 51%.
**Exemple :** rejeter les 5% de points les plus ambigus pour améliorer le taux de bonne classification sur le reste (voir code Partie 1.B).

### 3. « Je veux comparer deux modèles (ou deux hyperparamètres) sur un même jeu de données »
**Utiliser :** rejouer fit/predict pour chaque candidat, puis comparer `np.where(yhat == y_test, 1, 0).mean()` (ou `.score(X_test, y_test)`).
**Pourquoi :** un hyperparamètre mal choisi peut faire chuter drastiquement la performance (ex. `gamma` trop grand dans un SVM).
**Exemple :** `SVC(kernel="linear")` vs `SVC(gamma=10)` sur les mêmes données jouet.

### 4. « Je veux comprendre visuellement pourquoi un modèle se trompe »
**Utiliser :** `plot_frontiere` (2D) ou `plot_mesh` (3D, si `predict_proba` disponible).
**Pourquoi :** en 2D, on peut littéralement voir la forme de la frontière apprise — utile pour comprendre le sur/sous-apprentissage d'un coup d'œil.
**Exemple :** comparer visuellement la frontière (toujours une droite) d'un SVM linéaire à celle (très découpée) d'un SVM à noyau gaussien avec un `gamma` trop grand.

### 5. « Je veux savoir ce que le modèle a réellement appris (pas juste sa performance) »
**Utiliser :** les attributs spécifiques du modèle (voir tableau Partie 1.E) — jamais génériques, toujours à chercher dans la doc.
**Pourquoi :** un score de performance ne dit rien sur le "pourquoi" ; l'introspection permet de vérifier que le modèle a appris quelque chose de sensé (et pas un artefact).
**Exemple :** un `GaussianNB` a appris une moyenne/variance par variable et par classe (`theta_`, `var_`) → on peut vérifier que ça correspond à l'intuition qu'on a des données.

### 6. « Je veux savoir quelles variables sont importantes pour le modèle »
**Utiliser :** `.coef_` pour un modèle linéaire (régression logistique...), `.feature_importances_` pour une approche ensembliste (Random Forest, XGBoost).
**Pourquoi :** un modèle linéaire donne une importance signée et directement interprétable (poids négatif = pousse vers l'autre classe) ; une approche ensembliste donne une importance non signée, à interpréter avec prudence (elle peut favoriser les variables à beaucoup de valeurs distinctes).
**Exemple :** sur un jeu à 2 vraies variables + 20 variables de bruit gaussien, un bon modèle doit donner un poids quasi nul aux 20 colonnes de bruit.

### 7. « J'ai un arbre de décision, je veux l'expliquer à quelqu'un »
**Utiliser :** `tree.plot_tree(mod, filled=True)`
**Pourquoi :** contrairement aux autres modèles, l'arbre de décision se lit directement comme une suite de règles (if/else) — c'est LE modèle le plus interprétable par construction, à condition de limiter sa profondeur (`max_depth`).
**Exemple :** un arbre de profondeur 1 ou 2 donne une règle de décision qu'on peut écrire à la main sur une feuille.

### 8. « Je soupçonne du sur-apprentissage »
**Utiliser :** comparer les performances en apprentissage vs en test, et/ou visualiser la frontière avec `plot_frontiere`.
**Pourquoi :** une frontière très découpée/tourmentée qui colle parfaitement aux points d'apprentissage est un signal direct de sur-apprentissage.
**Exemple :** un SVM avec `gamma` trop grand, ou une forêt aléatoire avec `max_depth=10` et beaucoup d'arbres sur peu de données, sur-apprennent très vite — comparer avec `max_depth=1, n_estimators=5` (très simple, plutôt sous-apprentissage).

### 9. « Je veux utiliser plusieurs modèles ensemblistes sans réapprendre l'interface »
**Utiliser :** `RandomForestClassifier` et `xgb.XGBClassifier()` s'utilisent **exactement** comme tout autre modèle sklearn (interface `.fit`/`.predict` identique).
**Pourquoi :** c'est l'intérêt principal de l'écosystème sklearn — même les librairies externes (XGBoost) s'y conforment via un wrapper.
**Différence conceptuelle à retenir :**
- **Random Forest (bagging)** : des arbres **indépendants**, chacun sur un sous-ensemble aléatoire de variables/exemples, combinés par vote majoritaire → réduit la variance.
- **Gradient Boosting / XGBoost** : des arbres ajoutés **itérativement**, chacun corrigeant les erreurs de l'ensemble précédent → réduit le biais, souvent plus performant mais plus sensible au sur-apprentissage si mal réglé.

### 10. « Je veux créer mon propre modèle compatible avec tout l'écosystème sklearn »
**Utiliser :** hériter de `BaseEstimator` et `ClassifierMixin`, implémenter `__init__`, `fit`, `predict`.
**Pourquoi :** dès que ces méthodes existent, TOUS les outils génériques de sklearn (validation croisée, grid search, pipelines...) fonctionnent automatiquement sur le modèle maison, sans rien recoder.
**Point de vigilance :** stocker tous les hyperparamètres tels quels dans `__init__` (nécessaire pour la sérialisation/le clonage utilisé en interne par sklearn, notamment dans `cross_val_score` et `GridSearchCV`), et toujours faire retourner `self` par `fit`.
**Exemple :** un classifieur linéaire à poids fixes (jouet) devient directement utilisable avec `cross_val_score(mod, X, y, cv=5)` sans aucun code supplémentaire.

---

## Tableau récapitulatif : quel attribut pour quel modèle ?

| Je veux… | Modèle linéaire | Naive Bayes | SVM (linéaire) | Arbre | Forêt/XGBoost |
|---|---|---|---|---|---|
| Le score de confiance | `predict_proba` / `decision_function` | `predict_proba` | `predict_proba` (si `probability=True`) | `predict_proba` | `predict_proba` |
| Les paramètres appris | `.coef_`, `.intercept_` | `.theta_`, `.var_` | `.support_vectors_`, `.coef_` | `tree.plot_tree()` | `.estimators_`, `.feature_importances_` |
| L'importance des variables | `.coef_` (signé) | — | `.coef_` (noyau linéaire seulement) | via la structure de l'arbre | `.feature_importances_` (non signé) |
