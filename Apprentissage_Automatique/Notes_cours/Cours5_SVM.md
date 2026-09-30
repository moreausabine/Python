# Les SVM expliqués pas à pas

> Basé sur le cours « SVM and kernel methods » d'A. Cornuéjols (AgroParisTech – INRAE MIA 518, 138 diapos).
> Les renvois **(diapo N)** te permettent de retrouver le passage correspondant dans ton PDF.
> Les paragraphes marqués **➕ Complément** ajoutent une précision qui n'est pas explicitement sur les diapos.
> Les formules sont écrites en LaTeX (`$...$`) ; elles s'affichent correctement dans Obsidian, Typora, VS Code, Jupyter, etc.

---

## Sommaire

1. [La grande image : de quoi parle ce cours ?](#1-la-grande-image--de-quoi-parle-ce-cours-)
2. [Rappel : apprendre, c'est choisir une hypothèse](#2-rappel--apprendre-cest-choisir-une-hypothèse)
3. [Le perceptron et l'hyperplan séparateur](#3-le-perceptron-et-lhyperplan-séparateur)
4. [La marge maximale](#4-la-marge-maximale)
5. [La forme duale et les vecteurs supports](#5-la-forme-duale-et-les-vecteurs-supports)
6. [Marges souples (soft margin)](#6-marges-souples-soft-margin)
7. [Pourquoi la marge est-elle si importante ? (théorie)](#7-pourquoi-la-marge-est-elle-si-importante--théorie)
8. [Le passage au non linéaire : redescription et théorème de Cover](#8-le-passage-au-non-linéaire--redescription-et-théorème-de-cover)
9. [L'astuce du noyau (kernel trick)](#9-lastuce-du-noyau-kernel-trick)
10. [Les noyaux usuels et comment en fabriquer](#10-les-noyaux-usuels-et-comment-en-fabriquer)
11. [Exemple chiffré complet](#11-exemple-chiffré-complet)
12. [SVM en pratique](#12-svm-en-pratique)
13. [Au-delà de la classification](#13-au-delà-de-la-classification)
14. [Bilan : forces et limites](#14-bilan--forces-et-limites)
15. [Fiche récapitulative](#15-fiche-récapitulative)
16. [Quiz d'auto-évaluation](#16-quiz-dauto-évaluation)

---

## 1. La grande image : de quoi parle ce cours ?

Avant d'entrer dans le détail, voici l'histoire complète en cinq phrases. Si tu la gardes en tête, chaque section suivante trouvera sa place.

1. On veut **classer** des données en deux catégories (+1 / −1) à l'aide d'une **frontière**.
2. Si les classes sont séparables par une droite (un **hyperplan**), il y a en général **une infinité** de bonnes droites. Laquelle choisir ? → celle qui est **la plus éloignée des deux classes** : c'est la **marge maximale**.
3. Cette droite ne dépend que de quelques points, ceux qui sont au bord : les **vecteurs supports**.
4. Si les données ne sont pas séparables par une droite, on les **projette dans un espace de très grande dimension** où elles le deviennent. Et grâce à l'**astuce du noyau**, on n'a jamais besoin de calculer explicitement cette projection.
5. La marge joue aussi le rôle de **régularisation** : elle protège contre le surapprentissage, même en très grande dimension.

Les diapos suivent ce plan : rappel de théorie de l'apprentissage → perceptron à marge maximale → noyaux → pratique → ingénierie des noyaux → conclusion (diapo 2).

---

## 2. Rappel : apprendre, c'est choisir une hypothèse

*(diapos 3 à 30)*

### 2.1 Le problème de départ

En apprentissage supervisé, on cherche une fonction $h : X \to Y$ à partir d'exemples étiquetés $S = \{(x_i, y_i)\}_{i=1..m}$.

- **Classification** : $Y$ est un ensemble de classes (molécule → actif / inactif, microarray → cancer / pas cancer…).
- **Régression** : $Y$ est un nombre (prix, date de panne…).

### 2.2 Le vrai objectif : bien généraliser

**Le piège.** On ne mesure la performance que sur les données d'apprentissage, mais ce qui nous intéresse, c'est la performance sur des données **futures** (diapo 6). Parmi toutes les hypothèses qui se comportent bien sur $S$, laquelle favoriser ?

Deux notions à bien distinguer (diapos 10 et 11) :

| Notion | Définition | Peut-on la calculer ? |
|---|---|---|
| **Risque empirique** $R_{emp}(h)$ | Coût moyen des erreurs de $h$ **sur l'échantillon** $S$ | ✅ Oui |
| **Risque réel** $R(h)$ | **Espérance** du coût des erreurs sur toute la distribution des données | ❌ Non (distribution inconnue) |

La **fonction de perte** $\ell$ traduit le coût d'une erreur de prédiction (par exemple : 1 si on se trompe, 0 sinon).

> 💡 **Analogie.** Le risque empirique, c'est ta note aux exercices que tu as déjà vus en TD. Le risque réel, c'est ta note à l'examen sur des exercices nouveaux. Apprendre par cœur les corrigés donne 20/20 en TD… mais pas forcément à l'examen.

### 2.3 L'ERM et pourquoi il est dangereux tout seul

Le critère le plus naturel est la **minimisation du risque empirique (ERM)** : choisir $h$ qui a le plus petit $R_{emp}$.

Mais la théorie « PAC learning » de Vapnik montre (diapo 15) que :

> **L'ERM n'est un principe sain que s'il y a des contraintes sur l'espace des hypothèses $H$.**

Pourquoi ? Si $H$ est trop riche (ex. : « n'importe quelle fonction »), on trouvera toujours une hypothèse qui colle parfaitement aux données, y compris à leur bruit. Le lien entre $R_{emp}$ et $R$ n'existe plus. Plus $H$ est limité, plus ce lien est serré.

### 2.4 Mesurer la « richesse » d'un espace d'hypothèses

- **$H$ fini** (diapos 16-17) : l'écart entre $R$ et $R_{emp}$ dépend de $\log|H|$ et de $m$. Cas *réalisable* (la bonne hypothèse est dans $H$) : l'écart décroît **plus vite**, ce qui donnera plus tard l'idée de faire en sorte que $f \in H$ (diapo 26).
- **$H$ infini** (diapos 18-19) : on utilise la **dimension de Vapnik-Chervonenkis** $d_{VC}$ = taille du plus grand ensemble de points que $H$ peut étiqueter **de toutes les façons possibles** (on dit que $H$ « pulvérise » cet ensemble). C'est une mesure purement combinatoire, indépendante du nombre d'exemples.
  - Pour les séparateurs linéaires en dimension $d$ : $d_{VC} = d+1$ (diapo 68).
  - **➕ Complément** : pour les rectangles alignés sur les axes dans le plan, $d_{VC}=4$.

### 2.5 La solution : le risque régularisé

Vapnik propose (diapos 22-23) de **ne pas choisir $H$ au hasard**, mais d'optimiser un compromis :

$$\text{Critère} = \underbrace{\text{Risque empirique}}_{\text{coller aux données}} + \underbrace{\text{Terme de régularisation}}_{\text{limiter la capacité de } H}$$

C'est le **risque empirique régularisé**, avec deux objectifs :
1. **satisfaire les contraintes posées par les exemples** ;
2. **choisir le meilleur espace d'hypothèses** (contrôler la « capacité » de $H$).

La **SRM** (*Structural Risk Minimization*, diapo 22) formalise cela en emboîtant des espaces d'hypothèses de complexité croissante, fixés **a priori** (par exemple selon la $d_{VC}$).

**La recette générale** pour concevoir un algorithme d'apprentissage (diapo 24) :
1. Définir un critère inductif régularisé : (a) une fonction de perte, (b) un terme de régularisation qui exprime nos attentes sur les régularités du monde, (c) si possible, un problème d'optimisation **convexe**.
2. Utiliser un algorithme d'optimisation efficace.

> Tu connais déjà des exemples de cette recette (diapo 25) : la **ridge regression** (pénalise la norme $L_2$ des coefficients) et le **lasso** (pénalise la norme $L_1$, qui favorise les coefficients nuls, donc des solutions *parcimonieuses*). **Les SVM suivent exactement la même recette**, avec une perte et une régularisation particulières que l'on va découvrir.

### ✅ À retenir de cette section
- Ce qui compte : $R$ (inconnu), pas $R_{emp}$ (connu).
- Minimiser uniquement $R_{emp}$ est dangereux si $H$ est trop riche.
- Il faut **régulariser** : payer un prix pour la complexité.

---

## 3. Le perceptron et l'hyperplan séparateur

*(diapos 32 à 35)*

### 3.1 L'idée

Un **perceptron** est le classifieur linéaire le plus simple : il trace une **droite** (en 2D), un **plan** (en 3D), ou plus généralement un **hyperplan** (en dimension $d$, un objet de dimension $d-1$) qui coupe l'espace en deux moitiés. Un côté = classe +1, l'autre = classe −1.

### 3.2 L'algèbre

Un hyperplan est l'ensemble des points $x$ tels que :

$$w \cdot x + w_0 = 0$$

- $w$ est le vecteur **normal** à l'hyperplan (il indique sa « direction perpendiculaire »).
- $w_0$ est un **décalage** (le biais, ou seuil) qui règle la position de l'hyperplan.

La règle de décision est :

$$h(x) = \text{sign}(w \cdot x + w_0)$$

> 💡 **Lecture géométrique.** $w\cdot x + w_0$ est proportionnel à la **distance signée** de $x$ à l'hyperplan. Positive d'un côté, négative de l'autre, nulle sur la frontière. On ne garde que son signe pour décider.

### 3.3 Le problème (diapo 35)

Quand les données sont linéairement séparables, **il existe tout un ensemble de solutions** : on peut incliner ou déplacer légèrement l'hyperplan sans faire d'erreur sur $S$. Toutes ont $R_{emp} = 0$. **Laquelle choisir ?** Le perceptron classique n'a pas de raison particulière de préférer l'une à l'autre.

---

## 4. La marge maximale

*(diapos 36 à 45)*

### 4.1 L'intuition

> 💡 **Analogie de la route.** Deux villages, l'un de « + », l'autre de « − ». Il faut tracer une route entre eux. Vous pourriez la faire passer à 1 m des maisons d'un village, ou bien **au milieu**, le plus loin possible des deux. Si les maisons bougent un peu (nouvelles données, bruit), la route centrée a le plus de chances de rester valable.

La **marge**, c'est la largeur de cette « route » : la distance entre l'hyperplan et les exemples les plus proches. Le classifieur à **marge maximale** est, d'après la diapo 37, « le classifieur linéaire qui satisfait les contraintes des exemples et qui est **le plus robuste aux variations de $S$** ». C'est l'idée de Vapnik (1992) qui donne les **SVM** (*Séparateurs à Vastes Marges*).

### 4.2 Mise en équation, étape par étape

**Étape 1 : coder les classes.** On note $y_i = +1$ pour un « + » et $y_i = -1$ pour un « − ». Astuce de codage : un point est **bien classé** si et seulement si $y_i\,(w\cdot x_i + w_0) > 0$. (Signe de la sortie = signe de la classe → le produit est positif.)

**Étape 2 : fixer l'échelle.** Le couple $(w, w_0)$ est défini à un facteur multiplicatif près : $(2w, 2w_0)$ donne le même hyperplan. On lève cette ambiguïté en **imposant** que les points les plus proches de la frontière vérifient exactement $y_i(w\cdot x_i + w_0) = 1$. Les contraintes deviennent donc :

$$y_i\,(w\cdot x_i + w_0) \;\ge\; 1 \quad \text{pour tout } i$$

Les deux hyperplans $w\cdot x + w_0 = +1$ et $w\cdot x + w_0 = -1$ bordent la « route », l'hyperplan de décision passe au milieu.

**Étape 3 : calculer la largeur de la route.** La distance d'un point $x$ à l'hyperplan vaut $|w\cdot x + w_0| / \|w\|$. Pour un point sur le bord, le numérateur vaut 1, donc la distance vaut $1/\|w\|$ de chaque côté :

$$\text{marge} = \frac{2}{\|w\|}$$

*(diapo 44, « normalized margin »)*.

**Étape 4 : transformer le problème.** Maximiser $2/\|w\|$ revient à **minimiser $\|w\|$**, donc à minimiser $\tfrac12\|w\|^2$ (le carré et le ½ sont là pour que ce soit dérivable et plus joli). On obtient la **forme primale** :

$$\boxed{\begin{cases}\displaystyle \min_{w, w_0}\ \tfrac12 \|w\|^2 \\[4pt] \text{sous les contraintes } y_i\,(w\cdot x_i + w_0)\ge 1,\quad i=1..m\end{cases}}$$

C'est un **problème d'optimisation quadratique avec contraintes linéaires** : il est **convexe**, donc il admet **un unique optimum global** et il existe des algorithmes efficaces pour le résoudre (diapo 44).

### ✅ À retenir
- Marge $= 2/\|w\|$ : **petite norme de $w$ = grande marge**.
- Le SVM linéaire est un problème **convexe** : pas de minimum local piégeant, contrairement à un réseau de neurones.

---

## 5. La forme duale et les vecteurs supports

*(diapos 46 à 56)*

### 5.1 Pourquoi passer au dual ?

La forme primale marche, mais la forme **duale** a deux avantages capitaux :
1. Elle fait apparaître les données **uniquement via des produits scalaires** $\langle x_i, x_j\rangle$ → c'est la porte d'entrée de l'**astuce du noyau** (section 9).
2. Elle révèle que la solution ne dépend que de **quelques points** (les vecteurs supports).

### 5.2 Le Lagrangien (diapos 46-50)

Pour traiter un problème avec contraintes, on introduit un **multiplicateur de Lagrange** $\alpha_i \ge 0$ par contrainte, et on construit le **Lagrangien** :

$$L(w, w_0, \alpha) = \tfrac12\|w\|^2 - \sum_{i=1}^m \alpha_i\Big(y_i(w\cdot x_i + w_0) - 1\Big)$$

> 💡 **Intuition des $\alpha_i$.** Chaque $\alpha_i$ est le « prix » que l'on paie si la contrainte de l'exemple $i$ est violée. La solution est un **point selle** du Lagrangien : minimum par rapport à $(w, w_0)$, maximum par rapport à $\alpha$.

On annule les dérivées par rapport aux variables primales :

$$\frac{\partial L}{\partial w} = w - \sum_i \alpha_i y_i x_i = 0 \;\Rightarrow\; \boxed{w^* = \sum_{i=1}^m \alpha_i y_i x_i}$$

$$\frac{\partial L}{\partial w_0} = -\sum_i \alpha_i y_i = 0 \;\Rightarrow\; \boxed{\sum_{i=1}^m \alpha_i y_i = 0}$$

La première relation est le **théorème du représentant** (*representer theorem*, diapo 50). Elle dit une chose profonde :

> **Le vecteur $w$ optimal est une combinaison linéaire des exemples d'apprentissage.**

### 5.3 Le problème dual (diapo 51)

En réinjectant ces relations dans le Lagrangien, $w$ et $w_0$ disparaissent et il ne reste qu'un problème en $\alpha$ :

$$\max_\alpha \;\Big[\sum_{i=1}^m \alpha_i \;-\; \tfrac12\sum_{i,j=1}^m \alpha_i\alpha_j\, y_i y_j\,\langle x_i, x_j\rangle\Big] \quad\text{avec}\quad \alpha_i\ge 0,\;\; \sum_i \alpha_i y_i = 0$$

C'est de nouveau un problème quadratique convexe. Une fois les $\alpha_i^*$ trouvés, la **fonction de décision** s'écrit :

$$h^*(x) = \text{sign}\Big(\sum_{i\in \mathcal{P}_S} \alpha_i^*\, y_i\, \langle x_i, x\rangle + w_0^*\Big)$$

où $\mathcal{P}_S$ est l'ensemble des **vecteurs supports**.

### 5.4 Les vecteurs supports et la parcimonie (diapos 52-53)

Les conditions de **Karush-Kuhn-Tucker (KKT)** imposent, pour chaque exemple :

$$\alpha_i\,\big[y_i(w\cdot x_i + w_0) - 1\big] = 0$$

Autrement dit, pour chaque point, **l'un des deux facteurs est nul** :

| Cas | Ce que ça signifie | Rôle dans la solution |
|---|---|---|
| $\alpha_i = 0$ | Le point est **strictement à l'extérieur** de la marge (contrainte non serrée) | **Aucun.** On pourrait le supprimer sans rien changer |
| $\alpha_i > 0$ | Le point est **exactement sur le bord** de la marge : $y_i(w\cdot x_i + w_0) = 1$ | C'est un **vecteur support** : il « tient » l'hyperplan |

> 💡 **Analogie.** Imagine que l'hyperplan est une planche coincée entre deux nuages de billes. Seules les billes qui **touchent** la planche la retiennent. Retirer ou déplacer une bille loin de la planche ne change rien. La solution est donc **creuse** (*sparse*) : peu de $\alpha_i$ sont non nuls.

### 5.5 Calculer le seuil $w_0$ (diapo 55)

$w$ se reconstruit avec $\sum \alpha_i y_i x_i$. Pour $w_0$, on prend n'importe quel vecteur support $(x_c, y_c)$ : puisqu'il est sur la marge, $y_c(w\cdot x_c + w_0) = 1$, d'où $w_0 = y_c - w\cdot x_c$.

### 5.6 Une autre lecture : proche d'un k-plus-proches-voisins ! (diapo 104)

Le cours propose un parallèle éclairant avec les $k$-NN :

| $k$-NN | SVM |
|---|---|
| Quels voisins sont pertinents ? | Les **vecteurs supports** |
| Comment les pondérer ? | Par les coefficients $\alpha_i$ |
| Quelle distance / similarité ? | Le **produit scalaire** (puis le noyau) |

Le SVM est donc un « $k$-NN intelligent » : il **apprend** quels exemples garder et combien ils comptent. Toute la solution est déterminée par la **matrice de Gram** $G_{ij} = \langle x_i, x_j\rangle$ (les similarités deux à deux).

### 5.7 Les trois idées à retenir sur la forme duale (diapo 73)

1. L'hypothèse est une **combinaison linéaire** ;
2. qui dépend **directement des exemples** (les exemples supports) ;
3. et qui minimise un **risque régularisé** dans lequel la marge mesure la « versatilité » de l'hypothèse.

---

## 6. Marges souples (soft margin)

*(diapos 57 à 66)*

### 6.1 Le problème : données bruitées

La marge maximale « dure » exige que **toutes** les contraintes soient respectées. Mais :
- si les classes **ne sont pas séparables** (chevauchement, erreurs d'étiquetage), il n'existe **aucune** solution ;
- même si elles le sont, **un seul point aberrant** peut imposer une marge minuscule et un hyperplan absurde (diapos 58-59).

### 6.2 L'idée : tolérer des violations, mais les payer

On accepte que certains exemples violent la contrainte (être dans la marge, voire du mauvais côté), **à condition de payer une pénalité**. Ce niveau de violation s'appelle la **variable d'écart** (*slack*, diapo 61).

On mesure cette violation par la **fonction de perte charnière (hinge loss)** :

$$\ell_{hinge}\big(y_i, h(x_i)\big) = \big|\,1 - y_i\,h(x_i)\,\big|_+ = \max\big(0,\; 1 - y_i\,h(x_i)\big)$$

Lecture :
- $y_i h(x_i) \ge 1$ : bien classé **et** hors de la marge → perte **0**.
- $0 < y_i h(x_i) < 1$ : bien classé mais **dans la marge** → petite perte.
- $y_i h(x_i) < 0$ : **mal classé** → perte qui croît linéairement avec l'erreur.

### 6.3 Le critère complet du SVM (diapo 62)

$$w^\star = \arg\min_w \Big[\underbrace{\sum_{i=1}^m |1 - y_i h(x_i)|_+}_{\text{risque empirique (erreurs)}} \;+\; \underbrace{\tfrac12 w^\top w}_{\text{terme de marge}}\Big]$$

**C'est exactement la recette de la section 2.5** : perte + régularisation. Le terme $\tfrac12 w^\top w$ est la régularisation, et il **est** la marge (petite norme = grande marge).

➕ **Complément (formulation usuelle avec la constante $C$).** En pratique on pondère les deux termes avec une constante $C$ :

$$\min_{w,w_0,\xi}\ \tfrac12\|w\|^2 + C\sum_i \xi_i \quad\text{sous}\quad y_i(w\cdot x_i+w_0)\ge 1-\xi_i,\ \ \xi_i\ge 0$$

Ici $\xi_i$ est le *slack* de l'exemple $i$. Dans le dual, la seule différence est que les multiplicateurs sont bornés : $0\le\alpha_i\le C$. Trois types de points apparaissent alors :
- $\alpha_i = 0$ : bien classé, hors marge ;
- $0<\alpha_i<C$ : pile sur le bord de la marge ;
- $\alpha_i = C$ : dans la marge ou mal classé (le point « tire » au maximum).

### 6.4 Pourquoi une « perte de substitution » ? (diapos 61-64)

Le vrai coût qui nous intéresse est la perte **0-1** (1 si erreur, 0 sinon), mais elle est **non convexe** et impossible à optimiser efficacement. On la remplace par une perte **convexe** qui la majore : une **perte de substitution** (*surrogate loss*). Le graphique de la diapo 61 compare :
- la perte 0-1 (marche brutale),
- la **charnière** (hinge, la plus utilisée « pour de bonnes raisons théoriques »),
- la perte quadratique,
- la perte logistique.

> Diapo 64 : « c'est la **forme de la fonction de coût** qui induit la régularisation ». Autrement dit, la charnière, en étant plate à zéro après la marge, se désintéresse des points « faciles » : c'est ce qui rend la solution parcimonieuse (seuls les points près de la frontière comptent).

### 6.5 Le rôle de $C$ : le compromis biais-variance (diapo 66)

$C$ règle « à quel point on veut coller aux données » :

| | **$C$ grand** | **$C$ petit** |
|---|---|---|
| Attitude | Les erreurs coûtent cher : on ne les tolère presque pas | On est tolérant aux erreurs |
| Marge (formulation usuelle avec $C$ devant les erreurs) | Plus **étroite** | Plus **large** |
| Risque | Surapprentissage (sensible au bruit) | Sous-apprentissage (modèle trop rigide) |

> ⚠️ **Attention, point à vérifier avec ton enseignant.** La diapo 66 associe « $C$ grand » à « *low variance* / *stronger bias* / on veut de grandes marges ». Cette formulation est l'inverse de la convention habituelle décrite ci-dessus (où $C$ multiplie les erreurs, donc $C$ grand ⇒ marge plus étroite ⇒ biais faible, variance élevée). Cela vient sans doute d'une définition différente du rôle de $C$ sur cette diapo. Dans tous les cas, retiens le principe : **il existe un compromis entre marge large (régularisation forte, biais) et faible erreur d'entraînement (variance)**, et $C$ se règle par **validation croisée**.

### ✅ À retenir de cette section
- Soft margin = on tolère des erreurs contre une pénalité (hinge loss).
- Le SVM = **perte charnière + régularisation par la marge**.
- $C$ est un hyperparamètre de compromis, choisi par validation croisée.

---

## 7. Pourquoi la marge est-elle si importante ? (théorie)

*(diapos 67 à 72 — « Very briefly »)*

### 7.1 Le problème de la dimension

Avec la dimension VC classique ($d+1$ pour un séparateur linéaire en dimension $d$), la borne sur l'erreur de généralisation est **inutile quand $d$ est grand devant $m$** (diapo 68). C'est fâcheux : les SVM travaillent justement en très grande dimension !

### 7.2 Le résultat clé

Vapnik montre qu'on peut à la place borner l'erreur de généralisation en fonction du rapport entre :
- **$R$** : le rayon de la plus petite sphère contenant tous les points d'apprentissage ;
- **la marge** $\gamma$.

L'erreur ne dépend alors **pas de $d$** (diapo 69 : « which does not depend on $d$! »). Intuitivement, ce qui compte, c'est la **taille de la marge relative à l'étalement des données**, pas le nombre de coordonnées.

### 7.3 Ce que cela change (diapos 71-72)

- La marge est une **mesure de capacité indépendante de la dimension**, même pour un séparateur linéaire.
- On peut donc l'appliquer dans des espaces de **dimension infinie** : c'est **central pour les noyaux**.
- Contrairement à la SRM classique (où la structure des hypothèses est fixée **a priori**), la marge contrôle automatiquement la capacité **après avoir vu les exemples**.
- La perte charnière agit comme un **régulariseur**.
- La marge contrôle aussi la **vitesse de convergence**.

> 💡 **En clair.** Pourquoi les SVM ne surapprennent pas malgré des milliers (voire une infinité) de dimensions ? Parce qu'ils ne cherchent pas « n'importe quel séparateur », mais celui de **marge maximale**, qui est très contraint. C'est la marge, pas la dimension, qui mesure la complexité.

---

## 8. Le passage au non linéaire : redescription et théorème de Cover

*(diapos 26-27, 76 à 83)*

### 8.1 Le problème

Un séparateur linéaire ne suffit pas toujours. Exemple : des « + » au centre entourés de « − » en anneau : aucune droite ne les sépare.

### 8.2 L'idée : changer de point de vue

> 💡 **Analogie.** Des billes rouges et bleues sont mélangées sur une table. Aucune ligne droite ne les sépare. Mais si on **lance** les billes rouges en l'air (on ajoute une dimension « hauteur »), un simple plan horizontal les sépare. Le problème n'était pas difficile en soi : on le regardait dans le mauvais espace.

Formellement (diapo 77) : **augmenter la dimension** de l'espace d'entrée en générant de nouvelles coordonnées, obtenues par combinaisons et fonctions des descripteurs initiaux, puis chercher un séparateur **linéaire** dans ce nouvel espace, dit **espace de redescription** $F$ (*feature space*), grâce à une application $\Phi : X \to F$ (*feature expansion map*).

Un séparateur linéaire dans $F$ = une frontière **non linéaire** dans $X$.

### 8.3 Pourquoi ça marche : le théorème de Cover (diapo 27)

Le théorème de Cover répond à la question « quelle est la probabilité que $m$ points, étiquetés au hasard, soient linéairement séparables en dimension $d$ ? ». Le cours en retient deux messages :

- Si $m > 2(d+1)$ : la probabilité de séparabilité linéaire devient **faible**. Trop de points pour trop peu de dimensions.
- Pour $d$ grand et $m < 2(d+1)$ : la probabilité que **n'importe quel étiquetage** soit linéairement séparable tend vers **1**.

**Conclusion : projeter dans un espace de grande dimension rend la séparabilité linéaire beaucoup plus probable** (diapo 105 : « Is it more likely that the training set be linearly separable when projected in a high dimensional space? — Yes »).

### 8.4 Ce qu'on cherche (diapo 83)

Une redescription telle que :
- une **fonction de décision linéaire** existe entre les classes ;
- elle respecte la **vraie « similarité »** entre les données.

Deux difficultés : comment trouver $\Phi$ ? Et comment garder des **calculs faisables** dans un espace de très grande dimension (voire infinie) ? C'est la réponse à cette seconde question qui est géniale : le noyau.

---

## 9. L'astuce du noyau (kernel trick)

*(diapos 84 à 92, 100, 105)*

### 9.1 L'observation clé

Regarde le problème dual (section 5.3). Les données $x_i$ n'y apparaissent **que dans des produits scalaires** $\langle x_i, x_j\rangle$. Il en va de même dans la fonction de décision, avec $\langle x_i, x\rangle$.

Donc, si l'on travaille dans l'espace $F$, il suffit de remplacer $\langle x_i, x_j\rangle$ par $\langle \Phi(x_i), \Phi(x_j)\rangle$.

### 9.2 L'approche naïve… et l'astuce

- **Approche naïve** : calculer explicitement $\Phi(x)$ pour chaque point, puis résoudre dans $F$. **Coûteux, voire impossible** si $F$ est de dimension infinie.
- **Astuce du noyau** : trouver une fonction $K$ qui donne directement le produit scalaire dans $F$, **sans jamais calculer $\Phi$** :

$$K(x, x') = \langle \Phi(x), \Phi(x')\rangle$$

On écrit l'algorithme **uniquement à l'aide de produits scalaires**, puis on remplace chaque $\langle x, x'\rangle$ par $K(x,x')$ (diapo 87).

> 💡 **Analogie.** Tu veux savoir si deux personnes sont « proches » dans une immense base de données de 10 000 caractéristiques. Plutôt que de construire les 10 000 colonnes pour chacun puis de les comparer, tu as une **formule raccourcie** qui donne directement le score de ressemblance. Le résultat est le même, sans la dépense.

Le problème dual devient :

$$\max_\alpha \Big[\sum_i\alpha_i - \tfrac12\sum_{i,j}\alpha_i\alpha_j y_i y_j\, K(x_i,x_j)\Big],\qquad h(x)=\text{sign}\Big(\sum_{i\in\mathcal{P}_S}\alpha_i y_i K(x_i,x)+w_0\Big)$$

### 9.3 Exemple minimal pour comprendre (diapo 93)

Prenons $x=(x_1,x_2)\in\mathbb{R}^2$ et l'application vers $F=\mathbb{R}^3$ :

$$\Phi(x) = \big(x_1^2,\; x_2^2,\; \sqrt{2}\,x_1x_2\big)$$

Calculons le produit scalaire de deux images $\Phi(x)$ et $\Phi(z)$ :

$$\langle\Phi(x),\Phi(z)\rangle = x_1^2z_1^2 + x_2^2z_2^2 + 2x_1x_2z_1z_2 = (x_1z_1 + x_2z_2)^2 = \langle x,z\rangle^2$$

Donc $K(x,z)=\langle x,z\rangle^2$ **est** un noyau : on obtient le produit scalaire dans $\mathbb{R}^3$ en ne calculant qu'un produit scalaire dans $\mathbb{R}^2$ puis en l'élevant au carré. Et l'hypothèse $h(x)=w_{11}x_1^2 + w_{22}x_2^2 + w_{12}\sqrt2 x_1x_2$ est une **conique** dans le plan d'origine, alors qu'elle est linéaire dans $F$.

**Remarque importante** (diapo 93) : l'espace $F$ associé à un noyau **n'est pas unique**. La transformation $(x_1^2, x_2^2, x_1x_2, x_2x_1)\in\mathbb{R}^4$ donne exactement le même noyau.

Le gain devient spectaculaire en grande dimension : un noyau polynomial de degré $d$ sur $n$ variables correspond à un espace de dimension de l'ordre de $\binom{n+d-1}{d}$ (énorme), alors que le calcul de $K$ ne coûte que $O(n)$.

### 9.4 Condition de validité

Toute fonction n'est pas un noyau : il faut que la **matrice de Gram** $K_{ij}=K(x_i,x_j)$ soit **semi-définie positive** (diapo 87). C'est ce qui garantit que $K$ correspond bien à un produit scalaire dans un certain espace, et que le problème d'optimisation reste convexe.

### 9.5 Les quatre questions et leurs réponses (diapo 105)

| Question | Réponse du cours | Pourquoi |
|---|---|---|
| Est-il plus probable que les données soient séparables en grande dimension ? | **Oui** | Théorème de Cover |
| Comment choisir les nouvelles dimensions ? | **Automatiquement** | C'est le noyau qui les définit implicitement |
| Comment éviter le surapprentissage ? | **Simple** | La marge maximale régularise |
| Comment garder des calculs tractables ? | **Ce n'est même pas un problème** | Le *kernel trick* |

> C'est le cœur de l'élégance des SVM : **un algorithme linéaire simple + un noyau = un classifieur non linéaire puissant, sans coût de calcul explosif.**

---

## 10. Les noyaux usuels et comment en fabriquer

*(diapos 93 à 96, 122 à 132)*

### 10.1 Noyaux pour vecteurs (diapos 94-95)

| Noyau | Formule | Ce qu'il capture |
|---|---|---|
| **Polynomial (exact)** | $k(x,z)=(x^\top z)^d$ | Tous les produits d'**exactement** $d$ variables |
| **Polynomial (inhomogène)** | $k(x,z)=(x^\top z + c)^d$ | Tous les produits d'**au plus** $d$ variables |
| **Gaussien / RBF** | $k(x,z)=\exp\!\big(-\gamma\|x-z\|^2\big)$ (ou $\exp(-d(x,z)^2/2\sigma^2)$) | Similarité **locale** : proche = valeur ≈ 1, loin = ≈ 0 |
| **Sigmoïde** | $k(x,z)=\tanh(\kappa\,x^\top z+\theta)$ | Proche des réseaux de neurones. **Pas** semi-défini positif : ce n'est donc pas un « vrai » noyau |

**Le noyau gaussien (RBF)** mérite qu'on s'y arrête (diapo 95) :
- il n'a **pas de $\Phi$ explicite** sous forme fermée ;
- l'espace $F$ correspondant est de **dimension infinie** ;
- son paramètre $\gamma$ (ou $\sigma$) fixe la « portée » de l'influence de chaque point : **$\gamma$ grand / $\sigma$ petit** → influence très locale, frontière très flexible (risque de surapprentissage) ; **$\gamma$ petit / $\sigma$ grand** → frontière lisse.

### 10.2 Fabriquer de nouveaux noyaux : règles de composition (diapo 96)

Pas besoin de tout redémontrer à chaque fois : à partir de noyaux valides, on peut en construire d'autres. Parmi les règles listées, on peut, à partir de noyaux $\kappa_1,\kappa_2$ valides :
- multiplier par une constante positive : $c\,\kappa_1$ ;
- **sommer** : $\kappa_1+\kappa_2$ ;
- **multiplier** : $\kappa_1\,\kappa_2$ ;
- prendre l'exponentielle : $\exp(\kappa_1)$ ;
- appliquer un polynôme à coefficients positifs : $\text{poly}(\kappa_1)$ ;
- composer avec une application $\Phi$ : $\kappa_3(\Phi(x),\Phi(x'))$ ;
- combiner des noyaux portant sur des **sous-ensembles de variables** différents ($x_a$ et $x_b$) : très utile pour des données hétérogènes.

### 10.3 Noyaux pour objets non vectoriels (diapos 122 à 129)

Un point fort des méthodes à noyaux : **la modularité**. On **découple** :
- l'**algorithme** (linéaire, générique) ;
- la **description des données** (le noyau).

Il suffit de savoir mesurer la **similarité** entre deux objets, même si ces objets ne sont pas des vecteurs. Il existe donc des noyaux pour des **génomes, textes, images, vidéos, graphes sociaux**… Exemples cités : *string kernels* (comparer des chaînes de caractères) et noyaux de graphes basés sur des **parcours** (« walks »).

> 💡 **Pourquoi c'est puissant.** Le même SVM peut classer des protéines, des textes ou des graphes moléculaires : il suffit de changer le noyau. C'est aussi *le* seul endroit où l'on peut injecter des connaissances a priori (diapo 137).

---

## 11. Exemple chiffré complet

*(diapos 97 à 99)*

### 11.1 Les données

Cinq points sur une droite (1 dimension) :

| $x$ | 1 | 2 | 4 | 5 | 6 |
|---|---|---|---|---|---|
| $u$ (classe) | +1 | +1 | −1 | −1 | +1 |

Sur la droite, on a « + + − − + » : **aucun seuil unique** ne sépare les classes. Les données **ne sont pas linéairement séparables** en 1D.

*(Sur la diapo, $u_2$ est écrit « 2 » : c'est très probablement une coquille pour +1, ce que confirment les calculs plus bas et la figure « classe 1 / classe 2 / classe 1 ».)*

### 11.2 Le choix du noyau

Noyau polynomial de degré 2 : $k(x_i,x_j)=(x_ix_j+1)^2$, avec $C=100$. Ce noyau revient à travailler avec les monômes $1,\ x,\ x^2$ : on donne au SVM la possibilité de tracer une **parabole**.

### 11.3 Le résultat du solveur

Un algorithme d'optimisation quadratique trouve :

$$\alpha_1=0,\quad \alpha_2=2.5,\quad \alpha_3=0,\quad \alpha_4=7.333,\quad \alpha_5=4.833$$

Les **vecteurs supports** sont donc $\{x_2=2,\ x_4=5,\ x_5=6\}$ (ceux dont $\alpha\neq0$). Les points $x=1$ et $x=4$ n'interviennent pas.

**Vérification de la contrainte dual** : $\sum_i\alpha_i u_i = 2.5\times(+1) + 7.333\times(-1) + 4.833\times(+1) = 0$ ✔.

### 11.4 La fonction de décision

$$h(x)=\sum_i \alpha_i u_i (x_ix+1)^2 + b = 2.5(2x+1)^2 - 7.333(5x+1)^2 + 4.833(6x+1)^2 + b$$

En développant, le terme en $x^2$, le terme en $x$ et le terme constant donnent :

$$h(x)=0.6667\,x^2 - 5.333\,x + b$$

Pour trouver $b$, on utilise qu'un vecteur support est **exactement sur la marge** : $h(2)=+1$, soit $2.667-10.667+b=1$, donc $b=9$ :

$$\boxed{h(x)=0.6667\,x^2-5.333\,x+9}$$

### 11.5 Vérification

| $x$ | $h(x)$ | Signe | Classe attendue |
|---|---|---|---|
| 1 | 4.33 | + | + ✔ |
| 2 | 1.00 | + | + ✔ (sur la marge) |
| 4 | −1.67 | − | − ✔ |
| 5 | −1.00 | − | − ✔ (sur la marge) |
| 6 | 1.00 | + | + ✔ (sur la marge) |

La parabole est positive pour $x$ petit, négative entre les deux racines (environ $x\approx 2.4$ et $x\approx 5.6$), puis repositive : elle capture bien « + + − − + ». **Une frontière courbe en 1D = une frontière linéaire dans l'espace $(1, x, x^2)$.**

*(Note : dans la ligne de calcul de la diapo, le terme de $x_4$ apparaît sans signe moins ; le résultat final $0.6667x^2 - 5.333x + 9$ n'est cohérent que si $u_4=-1$ est bien pris en compte, comme ci-dessus.)*

---

## 12. SVM en pratique

*(diapos 107 à 118)*

### 12.1 Ce qu'il faut choisir (diapo 108)

1. **Le type de noyau** $k$ (sa forme) et **ses paramètres** (degré $d$, largeur $\sigma$ ou $\gamma$…) ;
2. **La constante $C$**.

Ces **méta-paramètres** se choisissent généralement par **validation croisée**.

### 12.2 Effet des paramètres (diapos 109 à 115)

Les diapos illustrent avec des démonstrations interactives :
- **Damier + noyau gaussien** (diapo 111) : deux valeurs de $\sigma$, dans les deux cas $R_{emp}=0$, mais avec un **petit $\sigma$ on a plus de vecteurs supports** que pour un grand $\sigma$. Un petit $\sigma$ rend la frontière très « accrochée » à chaque point : plus de points sont nécessaires pour la définir.
- **47 exemples (22 +, 25 −)** avec noyaux polynomiaux de degrés 2, 5, 8 ou gaussiens ($\sigma$ = 2, 5, 10) et $C=10000$ (diapos 112-113) : la forme de la frontière et le nombre de vecteurs supports changent avec le noyau.
- **Ajout de quelques points** (diapo 114) : la solution est réorganisée, avec plus de vecteurs supports.
- **Chemin de régularisation** (diapo 115, Hastie et al. 2004) : on peut suivre l'évolution de la solution quand $C$ varie.

**Leçon** : le noyau et $C$ ont un effet fort sur la frontière ; il faut les régler.

### 12.3 Estimer la performance en généralisation (diapos 116 à 118)

**a) Empiriquement** : par validation croisée (la méthode de référence).

**b) Heuristiquement (mais avec une base théorique solide)** :

*Le nombre de vecteurs supports* (diapo 116). Pour l'hyperplan à marge maximale, avec probabilité $1-\delta$ :

$$R(h)\ \le\ \frac{1}{m-\#sv}\Big(\#sv\,\log\frac{e\,m}{\#sv} + \log\frac{m}{\delta}\Big)$$

Plus **#sv est petit**, plus la borne est bonne. Intuition : une solution qui n'a besoin que de peu d'exemples est « simple », donc probablement généralisable (c'est un argument de compression).

*La matrice de Gram $G$* (diapo 118). Si $G$ n'a aucune structure, aucune régularité ne peut être trouvée dans les données :
- **si les termes hors diagonale sont très petits** (chaque point n'est similaire qu'à lui-même) → **surapprentissage** (par ex. $\sigma$ trop petit avec un RBF) ;
- **si la matrice est uniforme** (tous les points se ressemblent) → **sous-apprentissage** (tout est mis dans la même classe).

---

## 13. Au-delà de la classification

*(diapos 119, 120, 132)*

Le principe « marge / perte charnière + noyau » s'étend :

| Extension | Idée | Diapo |
|---|---|---|
| **Régression (SVR)** | Perte **$\varepsilon$-insensible** $|y-f(x)|_\varepsilon=\max\{0,|y-f(x)|-\varepsilon\}$ : on ne pénalise pas les erreurs inférieures à $\varepsilon$ (un « tube » autour de la fonction). On minimise $\tfrac12\|w\|^2 + C\sum_i|y_i-f(x_i)|_\varepsilon$, et la solution s'écrit $f(x)=\sum_i(\alpha_i^*-\alpha_i)K(x_i,x)+w_0$ | 119 |
| **Détection de nouveautés (non supervisé)** | On cherche à **séparer au maximum le nuage de points de l'origine** : ce qui tombe du mauvais côté est « nouveau » ou anormal | 120 |
| **Multi-classes, ACP non linéaire** | Les fonctions noyau s'appliquent à de nombreuses tâches | 132 |

**Applications citées** (diapo 131) : catégorisation de textes, reconnaissance de caractères manuscrits (ex. distinguer 4 et 7), détection de visages, diagnostic (cancer du sein), classification de protéines, prévision de consommation électrique, indexation de vidéos…

---

## 14. Bilan : forces et limites

*(diapos 135 à 137)*

### Points forts
- **Optimum global** : recherche d'un séparateur linéaire dans un problème convexe, pas de minima locaux.
- **Changement d'espace de description automatique** (via le noyau).
- **Le séparateur est défini par un ensemble d'exemples**… mais **seulement ceux sur la marge** (contrairement au $k$-NN qui garde tout).
- Combine les avantages de deux familles :
  - **non paramétrique** : s'ajuste automatiquement aux données ;
  - **paramétrique** : résiste bien au surapprentissage.
- Conceptuellement : introduit la **marge comme mesure de capacité**, et le *kernel trick* ouvre les problèmes non linéaires à des méthodes linéaires bien maîtrisées grâce à des redescriptions virtuelles.

### Limites
- **Boîte noire** : difficile d'interpréter ce qui a été appris (vecteurs supports et poids dans un espace de description inconnu).
- Les **connaissances a priori** ne s'expriment que par le **choix du noyau**.
- Mal adapté aux problèmes à **dépendances « à longue distance »** : un seul niveau de non-linéarité (contrairement aux réseaux profonds).
- **Recherche actuelle** : apprentissage de noyaux, combinaison de noyaux.

---

## 15. Fiche récapitulative

| Concept | En une phrase |
|---|---|
| **Risque empirique / réel** | Erreur sur l'échantillon (connue) vs erreur future attendue (inconnue) |
| **Régularisation** | Payer un prix pour la complexité afin de mieux généraliser |
| **Hyperplan** | $w\cdot x+w_0=0$ ; décision $\text{sign}(w\cdot x+w_0)$ |
| **Marge** | Distance de l'hyperplan aux points les plus proches ; vaut $2/\|w\|$ |
| **SVM primal** | $\min \tfrac12\|w\|^2$ sous $y_i(w\cdot x_i+w_0)\ge1$ |
| **Représentant** | $w=\sum_i\alpha_iy_ix_i$ : $w$ est une combinaison des exemples |
| **Forme duale** | Problème en $\alpha$ ne faisant intervenir que $\langle x_i,x_j\rangle$ |
| **KKT / vecteurs supports** | Seuls les points avec $\alpha_i>0$, sur la marge, comptent : solution parcimonieuse |
| **Perte charnière** | $\max(0,1-y\,h(x))$ : perte convexe qui remplace la perte 0-1 |
| **Constante $C$** | Compromis entre largeur de marge et tolérance aux erreurs ; choisie par CV |
| **Théorème de Cover** | Les données sont bien plus souvent séparables en grande dimension |
| **Redescription $\Phi$** | Projection vers un espace où le problème devient linéaire |
| **Kernel trick** | $K(x,x')=\langle\Phi(x),\Phi(x')\rangle$ : on évite de calculer $\Phi$ |
| **Noyau valide** | Matrice de Gram semi-définie positive |
| **RBF** | $\exp(-\gamma\|x-x'\|^2)$ : espace de dimension infinie ; $\gamma$ règle la portée |
| **Bornes** | Dépendent de la marge (et de #sv), pas de la dimension |
| **Extensions** | Régression ($\varepsilon$-insensible), détection de nouveautés, noyaux non vectoriels |
| **Limites** | Boîte noire, a priori uniquement via le noyau |

---

## 16. Quiz d'auto-évaluation

> **Mode d'emploi.** Réponds d'abord sans regarder, puis clique sur « Voir la réponse » pour vérifier. Compte un point par bonne réponse. Barème indicatif à la fin.

### Partie A : Fondations

**Q1.** Quelle est la différence entre le risque empirique et le risque réel ?
- A. Le risque réel est calculé sur l'échantillon d'apprentissage, le risque empirique sur de nouvelles données
- B. Le risque empirique est calculé sur l'échantillon d'apprentissage, le risque réel est l'espérance du coût sur toute la distribution
- C. Ils sont toujours égaux
- D. Le risque empirique est toujours plus grand que le risque réel

<details><summary>Voir la réponse</summary>

**B.** Le risque empirique est mesurable sur $S$ ; le risque réel est l'espérance de la perte sur la distribution inconnue. Le premier est un estimateur optimiste du second.
</details>

**Q2.** Pourquoi minimiser uniquement le risque empirique (ERM) est-il dangereux ?
- A. Parce que c'est trop long à calculer
- B. Parce que si l'espace d'hypothèses est trop riche, on peut coller parfaitement aux données (bruit compris) sans bien généraliser
- C. Parce que la fonction de perte n'est pas définie
- D. Parce qu'il n'y a aucune solution

<details><summary>Voir la réponse</summary>

**B.** Sans contrainte sur $H$, le lien entre $R_{emp}$ et $R$ disparaît : c'est le surapprentissage. Il faut régulariser.
</details>

**Q3.** Que mesure la dimension de Vapnik-Chervonenkis d'un espace d'hypothèses $H$ ?
<details><summary>Voir la réponse</summary>

La taille du plus grand ensemble de points que $H$ peut étiqueter de **toutes** les façons possibles (« pulvériser »). C'est une mesure combinatoire de la **capacité** de $H$. Pour les séparateurs linéaires en dimension $d$, $d_{VC}=d+1$.
</details>

**Q4.** Quelle est la « recette » générale pour concevoir un algorithme d'apprentissage vue dans le cours ?
<details><summary>Voir la réponse</summary>

1. Définir un critère inductif régularisé (fonction de perte + terme de régularisation, idéalement convexe) ; 2. utiliser un algorithme d'optimisation efficace. Les SVM = perte charnière + régularisation par la marge.
</details>

### Partie B : Marge maximale et dualité

**Q5.** Dans un problème linéairement séparable, pourquoi préférer l'hyperplan de marge maximale à un autre hyperplan sans erreur ?
<details><summary>Voir la réponse</summary>

Parce qu'il est **le plus robuste aux variations de l'échantillon** : tous les hyperplans sans erreur ont le même risque empirique (nul), mais celui qui est le plus éloigné des deux classes a le plus de chances de bien classer de nouveaux points. Il a aussi une capacité contrôlée (borne de généralisation).
</details>

**Q6.** Si l'on impose $y_i(w\cdot x_i+w_0)\ge 1$, quelle est la largeur de la marge ?
- A. $\|w\|$
- B. $1/\|w\|$
- C. $2/\|w\|$
- D. $\|w\|^2$

<details><summary>Voir la réponse</summary>

**C.** Chaque côté est à distance $1/\|w\|$ de l'hyperplan, soit $2/\|w\|$ au total. Maximiser la marge revient à minimiser $\tfrac12\|w\|^2$.
</details>

**Q7.** Pourquoi dit-on que le SVM linéaire est un problème « facile » à optimiser ?
<details><summary>Voir la réponse</summary>

C'est un problème d'optimisation **quadratique convexe avec contraintes linéaires** : il a un **optimum global unique** et il existe des algorithmes efficaces (pas de minima locaux).
</details>

**Q8.** Que dit le théorème du représentant appliqué aux SVM ?
<details><summary>Voir la réponse</summary>

Le vecteur optimal s'écrit $w^*=\sum_i\alpha_i y_i x_i$ : c'est une **combinaison linéaire des exemples d'apprentissage**.
</details>

**Q9.** Qu'est-ce qu'un vecteur support ? Que vaut son coefficient $\alpha_i$ ? Et celui d'un point loin de la frontière ?
<details><summary>Voir la réponse</summary>

Un vecteur support est un exemple situé **exactement sur le bord de la marge** ($y_i(w\cdot x_i+w_0)=1$). Son $\alpha_i>0$. Un point strictement hors de la marge a $\alpha_i=0$ (conditions KKT) : il n'a aucune influence sur la solution, qui est donc **parcimonieuse**.
</details>

**Q10.** Pourquoi la forme duale est-elle si importante pour la suite du cours ?
<details><summary>Voir la réponse</summary>

Parce que les données n'y apparaissent que par des **produits scalaires** $\langle x_i,x_j\rangle$. On peut donc les remplacer par un noyau $K(x_i,x_j)$ et travailler implicitement dans un espace de grande dimension : c'est le *kernel trick*.
</details>

**Q11.** Vrai ou faux : si l'on supprime un exemple qui n'est pas un vecteur support et qu'on ré-entraîne un SVM à marge dure, l'hyperplan change.
<details><summary>Voir la réponse</summary>

**Faux.** Son $\alpha_i$ est nul, il n'intervient ni dans $w$ ni dans $w_0$. Supprimer un vecteur support, en revanche, peut modifier l'hyperplan.
</details>

**Q12.** Par analogie avec le $k$-NN, que jouent les vecteurs supports, les $\alpha_i$ et le produit scalaire (ou noyau) dans un SVM ?
<details><summary>Voir la réponse</summary>

Les vecteurs supports sont les **voisins pertinents** ; les $\alpha_i$ sont les **poids** de ces voisins ; le produit scalaire (noyau) est la **mesure de similarité**.
</details>

### Partie C : Marges souples

**Q13.** Pourquoi introduit-on les marges souples ?
<details><summary>Voir la réponse</summary>

Pour gérer les données **bruitées** ou **non séparables**, et éviter qu'un seul point aberrant n'impose une marge minuscule. On tolère des violations des contraintes, à condition de les pénaliser.
</details>

**Q14.** Écris la perte charnière et donne sa valeur pour un point bien classé hors marge, dans la marge, et mal classé.
<details><summary>Voir la réponse</summary>

$\ell=\max(0,\,1-y\,h(x))$. Bien classé hors marge ($yh\ge1$) → **0**. Bien classé mais dans la marge ($0<yh<1$) → entre 0 et 1. Mal classé ($yh<0$) → supérieure à 1, croissante.
</details>

**Q15.** Pourquoi utilise-t-on une « perte de substitution » (surrogate) plutôt que la perte 0-1 ?
<details><summary>Voir la réponse</summary>

Parce que la perte 0-1 est **non convexe** et difficile à optimiser. La charnière est **convexe** et majore la perte 0-1, ce qui garde le problème traitable.
</details>

**Q16.** Que contrôle la constante $C$ ? Comment la règle-t-on ?
<details><summary>Voir la réponse</summary>

Le **compromis** entre largeur de la marge et tolérance aux erreurs d'apprentissage (compromis biais-variance). Dans la formulation usuelle, $C$ grand = erreurs coûteuses = marge étroite = risque de surapprentissage ; $C$ petit = tolérant = marge large = risque de sous-apprentissage. On la règle par **validation croisée**.
</details>

### Partie D : Théorie de la marge

**Q17.** Pourquoi la borne de généralisation basée sur la dimension VC est-elle inutile pour les SVM à noyaux ?
<details><summary>Voir la réponse</summary>

Parce qu'en grande dimension (voire infinie), $d_{VC}=d+1$ est immense devant $m$ et la borne devient vide. La borne basée sur la **marge** (rapport rayon des données / marge) est **indépendante de la dimension** et reste informative.
</details>

**Q18.** Quelle différence entre la SRM classique et la façon dont la marge contrôle la capacité ?
<details><summary>Voir la réponse</summary>

Avec la SRM, la hiérarchie d'espaces d'hypothèses est fixée **a priori**, indépendamment des données. Avec la marge, la capacité est contrôlée **automatiquement, après avoir observé les exemples**.
</details>

### Partie E : Noyaux

**Q19.** Que dit le théorème de Cover et pourquoi est-il utile pour les SVM ?
<details><summary>Voir la réponse</summary>

Il dit que la probabilité qu'un jeu de points soit linéairement séparable augmente fortement avec la dimension (elle tend vers 1 pour $d$ grand et $m<2(d+1)$). Il justifie l'idée de **projeter dans un espace de grande dimension** pour rendre le problème linéairement séparable.
</details>

**Q20.** Explique l'astuce du noyau en deux phrases.
<details><summary>Voir la réponse</summary>

Comme l'algorithme n'utilise que des produits scalaires, on remplace $\langle x,x'\rangle$ par $K(x,x')=\langle\Phi(x),\Phi(x')\rangle$. On obtient le produit scalaire dans l'espace de redescription **sans jamais calculer $\Phi$**, ce qui rend les calculs faisables même en dimension très grande ou infinie.
</details>

**Q21.** Pour $x=(x_1,x_2)$ et $\Phi(x)=(x_1^2,\,x_2^2,\,\sqrt2x_1x_2)$, quel noyau obtient-on ?
- A. $\langle x,z\rangle$
- B. $\langle x,z\rangle^2$
- C. $\langle x,z\rangle^3$
- D. $\exp(-\|x-z\|^2)$

<details><summary>Voir la réponse</summary>

**B.** $\langle\Phi(x),\Phi(z)\rangle=x_1^2z_1^2+x_2^2z_2^2+2x_1x_2z_1z_2=(x_1z_1+x_2z_2)^2=\langle x,z\rangle^2$.
</details>

**Q22.** Vrai ou faux : l'espace de redescription associé à un noyau est unique.
<details><summary>Voir la réponse</summary>

**Faux.** Par exemple, $(x_1^2,x_2^2,x_1x_2,x_2x_1)\in\mathbb{R}^4$ donne le même noyau $\langle x,z\rangle^2$ que la projection dans $\mathbb{R}^3$.
</details>

**Q23.** Quelle condition doit vérifier une fonction $K$ pour être un noyau valide ? Le noyau sigmoïde la respecte-t-il ?
<details><summary>Voir la réponse</summary>

La matrice de Gram doit être **semi-définie positive**. Le noyau **sigmoïde** ne la respecte **pas** en général : ce n'est pas un noyau au sens strict.
</details>

**Q24.** Quelle est la différence entre $(x^\top z)^d$ et $(x^\top z+c)^d$ ?
<details><summary>Voir la réponse</summary>

Le premier contient tous les produits d'**exactement** $d$ variables ; le second, tous les produits d'**au plus** $d$ variables.
</details>

**Q25.** Quelles sont les propriétés du noyau gaussien (RBF) ? Que fait $\gamma$ ?
<details><summary>Voir la réponse</summary>

Espace de redescription de **dimension infinie**, pas de $\Phi$ explicite fermé. $\gamma$ règle la portée de l'influence de chaque point : $\gamma$ grand (petit $\sigma$) → frontière très locale et flexible (plus de vecteurs supports, risque de surapprentissage) ; $\gamma$ petit → frontière plus lisse.
</details>

**Q26.** Comment peut-on fabriquer un nouveau noyau à partir de noyaux existants ?
<details><summary>Voir la réponse</summary>

Par des **règles de composition** : somme, produit, multiplication par une constante positive, exponentielle, polynôme à coefficients positifs, composition avec une application $\Phi$, combinaison de noyaux sur des sous-ensembles de variables…
</details>

**Q27.** Pourquoi dit-on que les méthodes à noyaux sont « modulaires » ?
<details><summary>Voir la réponse</summary>

Parce qu'elles **découplent** l'algorithme (linéaire, générique) de la **description des données** (le noyau). On peut appliquer le même SVM à des textes, des génomes ou des graphes en changeant seulement le noyau.
</details>

### Partie F : Exercice de calcul

**Q28.** Avec $h(x)=0.6667\,x^2-5.333\,x+9$ (exemple de la section 11), quelle est la classe prédite pour $x=3$ ? Pour $x=7$ ?

<details><summary>Voir la réponse</summary>

$h(3)=0.6667\times9-5.333\times3+9=6-16+9=-1$ → classe **−1**.
$h(7)=0.6667\times49-5.333\times7+9=32.67-37.33+9=4.33$ → classe **+1**.
</details>

**Q29.** Dans cet exemple, quels sont les vecteurs supports et pourquoi $x=1$ et $x=4$ n'en sont-ils pas ?
<details><summary>Voir la réponse</summary>

Les vecteurs supports sont $x=2$, $x=5$ et $x=6$ (leurs $\alpha_i\neq0$ et $|h(x)|=1$). $x=1$ ($h=4.33$) et $x=4$ ($h=-1.67$) sont **strictement hors de la marge** ($|h|>1$), donc leur $\alpha$ est nul.
</details>

### Partie G : En pratique et bilan

**Q30.** Quels hyperparamètres doit-on choisir pour un SVM à noyau et comment ?
<details><summary>Voir la réponse</summary>

Le **type de noyau** et ses paramètres (degré, $\sigma$/$\gamma$…) et la constante **$C$**. On les choisit généralement par **validation croisée**.
</details>

**Q31.** Comment le nombre de vecteurs supports renseigne-t-il sur la capacité de généralisation ?
<details><summary>Voir la réponse</summary>

La borne de généralisation croît avec #sv : **moins il y a de vecteurs supports, mieux c'est**. Une solution qui repose sur peu d'exemples est plus « simple » et généralise mieux.
</details>

**Q32.** Que révèlent les valeurs de la matrice de Gram sur le risque de sur- ou sous-apprentissage ?
<details><summary>Voir la réponse</summary>

Termes hors diagonale **très petits** (chaque point ne ressemble qu'à lui-même) → **surapprentissage**. Matrice **uniforme** (tout se ressemble) → **sous-apprentissage** (tous les points dans la même classe). Si $G$ n'a aucune structure, aucune régularité ne peut être apprise.
</details>

**Q33.** Comment adapte-t-on les SVM à la régression ? Et à la détection de nouveautés ?
<details><summary>Voir la réponse</summary>

**Régression** : perte $\varepsilon$-insensible $\max(0,|y-f(x)|-\varepsilon)$ (on ignore les erreurs dans un « tube » de largeur $\varepsilon$). **Détection de nouveautés** : on sépare au maximum le nuage de points de l'origine.
</details>

**Q34.** Cite trois forces et trois limites des SVM.
<details><summary>Voir la réponse</summary>

**Forces** : optimum global (convexité) ; changement d'espace automatique via le noyau ; solution parcimonieuse (peu d'exemples) ; bonne résistance au surapprentissage ; noyaux pour objets non vectoriels.
**Limites** : boîte noire peu interprétable ; a priori uniquement via le noyau ; un seul niveau de non-linéarité (mal adapté aux dépendances à longue distance).
</details>

**Q35 (synthèse).** Explique en quelques phrases, sans formule, comment un SVM à noyau gaussien parvient à classer des données non séparables linéairement sans surapprendre.
<details><summary>Voir la réponse</summary>

Le noyau gaussien projette **implicitement** les données dans un espace de dimension infinie où elles deviennent séparables linéairement (Cover), sans jamais calculer cette projection (*kernel trick*). Dans cet espace, le SVM cherche l'hyperplan de **marge maximale**, dont la capacité est contrôlée par la marge (indépendante de la dimension) et non par le nombre de coordonnées. La perte charnière avec $C$ tolère quelques erreurs. La solution ne dépend que de quelques **vecteurs supports**, ce qui la rend simple et parcimonieuse. $C$, $\gamma$ et le noyau se règlent par validation croisée.
</details>

---

### Barème indicatif (35 questions)

| Score | Niveau |
|---|---|
| 0 – 14 | À retravailler : relis les sections 4, 5 et 9 en priorité |
| 15 – 24 | Bases acquises : approfondis la soft margin (6) et les noyaux (10) |
| 25 – 31 | Bonne maîtrise |
| 32 – 35 | Excellent ! Tu peux expliquer les SVM à quelqu'un d'autre |

> 💡 **Conseil de révision.** Si tu ne dois retenir qu'un fil rouge : *marge maximale → vecteurs supports → forme duale (produits scalaires) → kernel trick → non linéaire sans surapprentissage.*