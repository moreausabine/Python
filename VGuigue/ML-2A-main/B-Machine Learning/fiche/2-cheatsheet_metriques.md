# Cheat Sheet — Métriques, données déséquilibrées, sélection de modèles

*Basé sur le notebook `2-notebook-metrique.ipynb`*

---

## Partie 1 — Fonctions essentielles (référence rapide)

### Imports de base

```python
import sklearn.metrics as met
from sklearn.model_selection import (train_test_split, StratifiedKFold, KFold,
                                      cross_val_score, GridSearchCV)
```

**Règle d'or à retenir partout :** toutes les métriques sklearn ont la signature `metrique(y_vrai, y_predit)` — **jamais l'inverse**.

### A. Métriques de base (classification)

| Fonction | Rôle | Exemple |
|---|---|---|
| `met.accuracy_score(y_true, y_pred)` | Taux de bonne classification global | `met.accuracy_score(y_test, mod.predict(X_test))` |
| `met.precision_score(y_true, y_pred, pos_label=1)` | Parmi les prédits positifs, combien le sont vraiment (VP/(VP+FP)) | `met.precision_score(y_test, y_hat, pos_label=1)` |
| `met.recall_score(y_true, y_pred, pos_label=1)` | Parmi les vrais positifs, combien sont détectés (VP/(VP+FN)) | `met.recall_score(y_test, y_hat, pos_label=1)` |
| `met.f1_score(y_true, y_pred)` | Moyenne harmonique précision/rappel | `met.f1_score(y_test, y_hat)` |
| `met.confusion_matrix(y_true, y_pred, labels=[...])` | Tableau VP/FP/VN/FN complet | `met.confusion_matrix(y_test, y_hat, labels=[-1,1])` |
| `met.ConfusionMatrixDisplay(cm, display_labels=[...]).plot()` | Affichage graphique de la matrice | voir A.1 ci-dessous |
| `met.classification_report(y_true, y_pred)` | Tableau complet (précision/rappel/f1 par classe + moyennes) | `print(met.classification_report(Yu_test, yhat))` |

**Convention scikit-learn pour la matrice de confusion :** lignes = classe réelle, colonnes = classe prédite, classes triées par ordre croissant.

```python
Conf_mat = met.confusion_matrix(y_test, y_hat, labels=[-1, 1])
disp = met.ConfusionMatrixDisplay(Conf_mat, display_labels=[-1, 1])
disp.plot()
plt.show()
```

### B. Scores continus (avant seuillage)

Un classifieur ne rend pas qu'une classe : il calcule d'abord un score continu, seuillé ensuite pour décider.

| Méthode | Pour quels modèles | Exemple |
|---|---|---|
| `mod.decision_function(X)` | Modèles linéaires / SVM (score signé, 0 = frontière) | `yhat_score = mod.decision_function(X_test)` |
| `mod.predict_proba(X)[:, 1]` | Modèles probabilistes (Naive Bayes, LogisticRegression, RandomForest...) | `proba = mod.predict_proba(X_test)[:, 1]` |

**Piège à connaître :** `decision_function` n'existe pas sur tous les modèles (ex. RandomForest) ; `predict_proba` n'existe pas sur `LinearSVC`. Vérifier avant de coder.

### C. Courbes de compromis et indicateurs agrégés

| Fonction | Rôle | Exemple |
|---|---|---|
| `met.PrecisionRecallDisplay.from_estimator(mod, X_test, y_test, pos_label=1)` | Trace la courbe précision/rappel en faisant varier le seuil | `disp = met.PrecisionRecallDisplay.from_estimator(mod, X_test, y_test, pos_label=1)` |
| `met.precision_recall_curve(y_true, scores, pos_label=1)` | Version « brute » (retourne les tableaux precision, recall, thresholds) | `p, r, th = met.precision_recall_curve(y_test, yhat, pos_label=1)` |
| `met.roc_curve(y_true, scores, pos_label=1)` | Retourne fpr, tpr, thresholds pour tracer la courbe ROC | `fpr, tpr, th = met.roc_curve(y_test, yhat, pos_label=1)` |
| `met.RocCurveDisplay.from_estimator(mod, X_test, y_test)` | Trace directement la courbe ROC | — |
| `met.auc(fpr, tpr)` | Aire sous une courbe déjà calculée (générique) | `met.auc(fpr, tpr)` |
| `met.roc_auc_score(y_true, scores)` | AUC calculée directement à partir des scores (raccourci) | `met.roc_auc_score(y_test, yhat)` |
| `met.average_precision_score(y_true, scores)` | Aire sous la courbe précision/rappel (alternative à l'AUC pour le fort déséquilibre) | `met.average_precision_score(y_test, yhat)` |

```python
yhat = mod.decision_function(X_test)
fpr, tpr, thresholds = met.roc_curve(y_test, yhat, pos_label=1)

plt.plot(fpr, tpr)
plt.plot([0, 1], [0, 1], 'k--', lw=1)   # diagonale = classifieur aléatoire
```

### D. Multi-classes : agréger plusieurs scores

L'argument `average` de `precision_score` / `recall_score` / `f1_score` :

| Valeur | Comportement |
|---|---|
| `average=None` | Aucune agrégation, un score par classe |
| `average='micro'` | Somme des VP/FP/FN de toutes les classes **avant** de calculer le ratio (= égal à l'accuracy en multi-classes à étiquette unique) |
| `average='macro'` | Moyenne **non pondérée** (chaque classe compte pour 1, même si elle est rare) |
| `average='weighted'` | Moyenne pondérée par l'effectif de chaque classe |

```python
met.recall_score(Yu_test, yhat, average=None)     # un score par chiffre 0-9
met.recall_score(Yu_test, yhat, average='macro')  # toutes les classes comptent pareil
```

### E. Validation croisée

| Fonction | Rôle | Exemple |
|---|---|---|
| `train_test_split(X, y, test_size=0.33, random_state=0)` | Découpage simple app/test | — |
| `StratifiedKFold(n_splits=5)` puis `.split(X, y)` | Découpage en K plis en gardant les proportions de classes ; donne les **indices** | boucle explicite, voir B.1 |
| `KFold(n_splits=5)` | Comme StratifiedKFold mais sans stratification (ok en régression) | — |
| `cross_val_score(mod, X, y, cv=5, scoring='accuracy')` | Fait tout d'un coup : découpe, entraîne, évalue, K fois | `scores.mean(), scores.std()` |

```python
skf = StratifiedKFold(n_splits=5)
allp = []
for train, test in skf.split(X, y):
    mod = SVC()                       # ATTENTION : un modèle NEUF à chaque fold
    mod.fit(X[train], y[train])
    allp.append(accuracy_score(y[test], mod.predict(X[test])))

# équivalent, en une ligne :
scores = cross_val_score(SVC(), X, y, cv=5, scoring='accuracy')
print('{:.3f} (+/- {:.3f})'.format(scores.mean(), scores.std()))
```

### F. Comparaison significative de modèles

```python
from mlxtend.evaluate import paired_ttest_kfold_cv
t, p = paired_ttest_kfold_cv(estimator1=mod1, estimator2=mod2, X=X, y=y, cv=5)
```
Test **apparié** : les deux modèles sont comparés sur exactement les mêmes découpages (folds), ce qui élimine la variabilité due au tirage et rend le test plus sensible.

### G. Sélection d'hyperparamètres

```python
from sklearn.model_selection import GridSearchCV

mod = SVC()
parameters = {'gamma': [0.01, 0.05, 0.1, 0.5, 1, 5], 'C': [0.1, 1, 10]}
meta_mod = GridSearchCV(estimator=mod, param_grid=parameters, cv=3)
meta_mod.fit(X_train, y_train)

meta_mod.best_params_      # meilleure combinaison
meta_mod.best_score_       # score en validation croisée (optimiste, voir Partie 2)
meta_mod.cv_results_       # dictionnaire complet (toutes les combinaisons testées)
```
Astuce pour retrouver la grille à partir de `cv_results_['mean_test_score']` (vecteur à plat) : les combinaisons sont énumérées dans l'ordre du produit cartésien des clés **triées alphabétiquement**, la dernière variant le plus vite → `reshape(len(C), len(gamma))`.

### H. Gestion du déséquilibre

| Outil | Rôle |
|---|---|
| `class_weight='balanced'` (argument de la plupart des modèles) | Pondère chaque classe inversement à son effectif dans la fonction de coût |
| Sous-échantillonnage (*undersampling*) | On jette des exemples de la classe majoritaire — **seulement sur l'apprentissage** |
| Sur-échantillonnage (*oversampling*, ex. SMOTE) | On duplique/interpole des exemples de la classe minoritaire — **seulement sur l'apprentissage** |
| `met.balanced_accuracy_score(y_true, y_pred)` | Moyenne des rappels par classe (= accuracy qui ne se laisse pas tromper par le déséquilibre) |

### I. Métriques de régression

| Fonction | Rôle |
|---|---|
| `met.mean_absolute_error(y_true, y_pred)` (MAE) | Erreur moyenne en valeur absolue, dans l'unité de la cible — robuste aux valeurs aberrantes |
| `met.mean_squared_error(y_true, y_pred)` (→ racine = RMSE) | Pénalise fortement les grosses erreurs (au carré) — sensible aux outliers |
| `met.r2_score(y_true, y_pred)` (R²) | Part de variance expliquée : 1 = parfait, 0 = aussi bon que prédire la moyenne, négatif = pire |

---

## Partie 2 — Fiche décision : quel outil pour quel cas ?

### 1. « Je veux un score rapide et global »
**Utiliser :** `accuracy_score`
**Pourquoi :** un seul chiffre, facile à lire.
**Piège :** trompeur dès que les classes sont déséquilibrées — un classifieur qui prédit toujours la classe majoritaire peut afficher 95%+ d'accuracy sans être utile.
**Exemple :** 200 points normaux vs 15 événements → un modèle qui ne détecte jamais rien a quand même ~93% d'accuracy.

### 2. « Je veux savoir quelle classe pose problème »
**Utiliser :** `precision_score` et `recall_score` avec `average=None` (ou `pos_label=` en binaire)
**Pourquoi :** l'accuracy masque les erreurs asymétriques entre classes ; précision et rappel sont **complémentaires** (on peut trivialement maximiser l'un en sacrifiant l'autre).
**Exemple :** un classifieur qui prédit toujours « +1 » obtient un rappel de 100% sur la classe +1, mais une précision qui s'effondre.

### 3. « Je veux un seul chiffre qui résume précision ET rappel »
**Utiliser :** `f1_score`
**Pourquoi :** moyenne harmonique — pénalise fortement si l'une des deux valeurs est basse (contrairement à la moyenne arithmétique).

### 4. « Je veux voir le détail de toutes les erreurs (VP/FP/VN/FN) »
**Utiliser :** `confusion_matrix` + `ConfusionMatrixDisplay`
**Pourquoi :** vue complète, utile pour du multi-classes (voir quelles classes sont confondues entre elles).
**Exemple :** sur USPS (chiffres manuscrits), la matrice montre par exemple que le 4 et le 9 sont souvent confondus.

### 5. « Mon classifieur a un seuil ajustable, je veux voir tous les compromis possibles »
**Utiliser :** `PrecisionRecallDisplay.from_estimator` (ou `precision_recall_curve`)
**Pourquoi :** un classifieur produit un score continu (`decision_function`/`predict_proba`) ; changer le seuil ne change pas le modèle mais change le compromis précision/rappel obtenu.
**Exemple :** en détection d'alarme, si je ne tolère aucune fausse alerte, quelle couverture (rappel) puis-je espérer ? La courbe répond directement à cette question.

### 6. « Cas déséquilibré : je veux étudier détection vs fausse alerte »
**Utiliser :** `roc_curve` / `RocCurveDisplay`
**Pourquoi :** trace le taux de vrais positifs (TPR = rappel) contre le taux de faux positifs (FPR), pour tous les seuils possibles.
**Exemple :** détection de fraude — la courbe ROC montre le compromis entre fraudes détectées et fausses alertes générées.

### 7. « Je veux un indicateur unique pour résumer une courbe ROC »
**Utiliser :** `auc(fpr, tpr)` ou `roc_auc_score(y_true, scores)`
**Pourquoi :** évite de comparer des courbes entières entre plusieurs modèles.
**Interprétation utile pour un non-expert :** l'AUC = probabilité qu'un événement tiré au hasard reçoive un score plus élevé qu'un non-événement tiré au hasard (0.5 = hasard pur, 1 = parfait).

### 8. « Déséquilibre TRÈS fort (événement rare, <1% de positifs) »
**Utiliser :** `average_precision_score` plutôt que l'AUC seule, et regarder la courbe précision/rappel plutôt que la ROC.
**Pourquoi :** le FPR (dénominateur = tous les négatifs) reste quasi inchangé même avec beaucoup de fausses alertes quand les négatifs sont très nombreux → l'AUC peut sembler excellente (ex. 0.95) alors que la précision s'effondre en pratique. La précision, elle, ne regarde jamais les vrais négatifs et réagit donc immédiatement au moindre bruit.
**Exemple :** 0.1% de positifs, AUC = 0.95 → peut quand même être inutilisable en production (à vérifier avec la précision).

### 9. « Peu de données, le score dépend trop du tirage train/test »
**Utiliser :** validation croisée — `cross_val_score` (rapide) ou boucle `StratifiedKFold` (si besoin de contrôler chaque fold en détail)
**Pourquoi :** moyenne le score sur plusieurs découpages + fournit un écart-type (mesure de confiance).
**Exemple :** avec `random_state` différent sur un simple `train_test_split`, le score peut varier de plusieurs points — la validation croisée lisse cet effet.

### 10. « Je veux comparer deux modèles et savoir si la différence est significative »
**Utiliser :** `paired_ttest_kfold_cv` (package `mlxtend`)
**Pourquoi :** un test apparié compare les modèles sur les mêmes folds — beaucoup plus sensible qu'une simple comparaison de moyennes.
**Repère grossier :** <50 exemples → approche statistique poussée nécessaire ; >10 000 exemples → la moindre amélioration est souvent significative ; cas intermédiaire → `paired_ttest_kfold_cv`.

### 11. « Je veux régler un hyperparamètre (ex. gamma d'un SVM) »
**Utiliser :** `GridSearchCV`
**Pourquoi :** automatise la boucle « essayer chaque valeur + évaluer en validation croisée », sans toucher au jeu de test.
**Exemple :** chercher le meilleur `gamma` ET le meilleur `C` d'un SVM à noyau gaussien simultanément (grille 2D).

### 12. « Je veux un score final honnête après avoir sélectionné des hyperparamètres »
**Utiliser :** validation croisée **imbriquée** (*nested cross-validation*) — englober le `GridSearchCV` dans un `cross_val_score` externe.
**Pourquoi :** `best_score_` d'un `GridSearchCV` est optimiste : c'est le maximum d'un ensemble de mesures bruitées, donc systématiquement au-dessus de la vraie performance. Une boucle externe qui refait sa propre recherche d'hyperparamètres sur chaque fold donne une estimation non biaisée.
**Exemple :**
```python
grid = GridSearchCV(SVC(), param_grid={'C':[...], 'gamma':[...]}, cv=3)
scores = cross_val_score(grid, X, y, cv=5)   # boucle externe honnête
```

### 13. « Problème multi-classes, je veux un score global »
**Utiliser :** `average='micro'`, `'macro'` ou `'weighted'` selon le besoin.
**Comment choisir :**
- `micro` = équivalent de l'accuracy globale (à étiquette unique).
- `macro` = chaque classe compte pareil, **à privilégier si les classes rares sont importantes** (ex. maladies rares) — une classe massacrée pèse autant qu'une classe bien traitée.
- `weighted` = pondéré par la taille des classes, plus proche du ressenti global mais peut masquer une classe rare mal traitée.

### 14. « Coût métier asymétrique : une erreur coûte plus cher qu'une autre »
**Utiliser :** `predict_proba` + boucle sur les seuils pour minimiser un coût total, plutôt que le seuil par défaut (0.5).
**Pourquoi :** le seuil optimal n'est pas dans les données, il dépend du **coût des erreurs**.
**Exemple :** panne non détectée = coût 10 (arrêt chaîne), fausse alerte = coût 1 (déplacement inutile) → on cherche le seuil qui minimise `10×FN + 1×FP` sur le jeu de validation. Plus le coût d'un faux négatif augmente, plus le seuil optimal baisse (on devient plus sensible, quitte à multiplier les fausses alertes).

### 15. « Le déséquilibre pose problème dès l'apprentissage (pas juste à l'évaluation) »
**Utiliser :** `class_weight='balanced'`, ou sous/sur-échantillonnage sur le jeu d'apprentissage **uniquement**.
**Pourquoi :** ces trois stratégies sont conceptuellement proches d'un simple déplacement de seuil (elles biaisent la frontière de décision), mais agissent **pendant** l'optimisation plutôt qu'après coup.
**Piège à vérifier :** l'AUC bouge très peu avec ces stratégies (le classement des scores ne change pas beaucoup), alors que la précision/rappel à seuil fixe bougent beaucoup plus.

### 16. « Je suis en régression, pas en classification »
**Utiliser :** MAE, RMSE et R² ensemble (jamais un seul indicateur).
**Comment choisir en fonction du contexte :**
- Une valeur aberrante dans le jeu de **test** fait exploser le RMSE (erreur au carré) bien plus que le MAE → si le métier tolère mal les grosses erreurs ponctuelles, optimiser/surveiller plutôt le RMSE ; si les données contiennent des outliers connus et qu'on veut une métrique robuste, préférer le MAE.
- Une valeur aberrante dans le jeu d'**apprentissage** dégrade le modèle lui-même (pas seulement la métrique) — effet généralement plus large et plus difficile à corriger après coup.

---

## Formules à connaître par cœur

| Nom | Formule |
|---|---|
| Précision | VP / (VP + FP) |
| Rappel (= sensibilité = TPR) | VP / (VP + FN) |
| F1-score | 2 × (Précision × Rappel) / (Précision + Rappel) |
| Accuracy | (VP + VN) / (VP + FP + VN + FN) |
| Taux de faux positifs (FPR) | FP / (FP + VN) |
| Balanced accuracy | moyenne des rappels de chaque classe |

**Rappel de vocabulaire :** VP = Vrai Positif, FP = Faux Positif, VN = Vrai Négatif, FN = Faux Négatif.
**Cout vs métrique :** la fonction de coût/loss (optimisée pendant l'apprentissage) doit être dérivable/convexe ; la métrique (présentée à un expert métier) n'a aucune contrainte mathématique, juste besoin d'avoir du sens pour le problème.
