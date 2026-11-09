 

# SVM et méthodes à noyaux : questions type partiel corrigées

> Source : ton cours « Les SVM expliqués pas à pas » (diapos d'A. Cornuéjols). Les passages marqués *(complément)* sont des précisions de ma part, absentes de to   n cours.
> Refais les calculs toi-même avant de regarder la correction.

## Points de vigilance dans ton cours (à vérifier avec ton prof)

1. **Convention de $C$ (diapo 66) :** la diapo associe « $C$ grand » à « *low variance* / *stronger bias* / grandes marges », ce qui est l'inverse de la convention usuelle (où $C$ multiplie les erreurs : $C$ grand ⇒ erreurs coûteuses ⇒ marge plus étroite ⇒ surapprentissage). Le critère de la diapo 62 n'a d'ailleurs pas de $C$. Apprends **le principe** (compromis marge large ↔ peu d'erreurs, réglé par validation croisée) et demande quelle convention est attendue au partiel.
2. **Exemple chiffré (diapos 97-99) :** $u_2$ est écrit « 2 » sur la diapo : c'est une coquille pour $+1$. De plus, le terme de $x_4$ apparaît sans signe moins dans une ligne de calcul ; le résultat final n'est cohérent que si $u_4=-1$.

---

## A. Fondations : théorie de l'apprentissage

### Q1. Distinguer risque empirique et risque réel. Pourquoi minimiser uniquement le risque empirique (ERM) est-il dangereux ? Que devient le lien entre les deux quand l'espace d'hypothèses $H$ est très riche ?

- **Risque empirique $R_{emp}(h)$ :** coût moyen des erreurs de $h$ **sur l'échantillon** $S$ ; calculable.
- **Risque réel $R(h)$ :** **espérance** du coût sur toute la distribution des données ; inaccessible (distribution inconnue).
- La **fonction de perte** $\ell$ traduit le coût d'une erreur (ex. 1 si erreur, 0 sinon).

**Danger de l'ERM seul :** si $H$ est trop riche (« n'importe quelle fonction »), on trouve toujours une hypothèse qui colle parfaitement à $S$, bruit compris : $R_{emp}$ n'est plus un bon indicateur de $R$ (surapprentissage). La théorie PAC de Vapnik dit que l'ERM n'est sain que si $H$ est **contraint** ; plus $H$ est limité, plus le lien $R_{emp}\leftrightarrow R$ est serré. *Analogie :* la note aux TD déjà corrigés (risque empirique) vs la note à l'examen sur des exercices nouveaux (risque réel).

### Q2. Définir la dimension de Vapnik-Chervonenkis. Quelle est celle des séparateurs linéaires en dimension $d$ ? Vérifier en dimension 2 qu'elle vaut 3.

**Définition :** $d_{VC}$ = taille du plus grand ensemble de points que $H$ peut étiqueter de **toutes** les façons possibles (« pulvériser »). Mesure purement **combinatoire** de la capacité de $H$, indépendante du nombre d'exemples. Pour les séparateurs linéaires en dimension $d$ : $d_{VC}=d+1$. *(Complément : pour les rectangles alignés sur les axes dans le plan, $d_{VC}=4$.)*

**En dimension 2 :** 3 points non alignés peuvent être étiquetés des $2^3=8$ façons par une droite (chaque étiquetage est séparable). Avec 4 points, la configuration « XOR » (sommets d'un carré, étiquettes alternées sur les diagonales) n'est séparable par aucune droite. Donc $d_{VC}=3=2+1$.

### Q3. Qu'est-ce que le risque empirique régularisé ? Énoncer la « recette » générale pour concevoir un algorithme d'apprentissage et situer ridge, lasso et SVM dans cette recette. Qu'est-ce que la SRM ?

$$
\text{Critère}=\underbrace{\text{Risque empirique}}_{\text{coller aux données}}+\underbrace{\text{Régularisation}}_{\text{limiter la capacité de }H}
$$

**Recette (diapo 24) :** (1) définir un critère inductif régularisé : une **fonction de perte**, un **terme de régularisation** exprimant nos attentes sur les régularités du monde, et si possible un problème d'optimisation **convexe** ; (2) utiliser un algorithme d'optimisation efficace.

- **Ridge :** pénalise la norme $L_2$ des coefficients. **Lasso :** pénalise la norme $L_1$, qui favorise les coefficients nuls (solutions **parcimonieuses**).
- **SVM :** même recette avec une perte particulière (**charnière**) et une régularisation particulière ($\tfrac12\|w\|^2$, c'est-à-dire **la marge**).

**SRM (*Structural Risk Minimization*) :** on emboîte des espaces d'hypothèses de complexité croissante (par exemple selon $d_{VC}$), fixés **a priori**, et on choisit le meilleur compromis.

---

## B. Perceptron et hyperplan séparateur

### Q4. Définir un hyperplan séparateur et la règle de décision d'un classifieur linéaire. Que représentent $w$ et $w_0$ ? Quel problème pose le perceptron classique sur des données séparables ?

Hyperplan : $w\cdot x+w_0=0$ (dimension $d-1$ en dimension $d$). Décision : $h(x)=\mathrm{sign}(w\cdot x+w_0)$.

- $w$ : vecteur **normal** à l'hyperplan (sa direction perpendiculaire) ; $w_0$ : décalage (biais/seuil) qui règle sa position.
- $w\cdot x+w_0$ est proportionnel à la **distance signée** de $x$ à l'hyperplan : positive d'un côté, négative de l'autre, nulle sur la frontière.

**Problème :** si les données sont séparables, il existe **tout un ensemble** d'hyperplans sans erreur (tous avec $R_{emp}=0$). Le perceptron classique n'a aucune raison de préférer l'un à l'autre : lequel choisir ?

### Q5. (Calcul) Soit $w=(3,4)$ et $w_0=-5$. Calculer la distance de $x=(3,4)$ à l'hyperplan et sa classe. Pourquoi le couple $(w,w_0)$ n'est-il défini qu'à un facteur près ?

$w\cdot x+w_0=9+16-5=20$ ; $\|w\|=\sqrt{9+16}=5$ ; distance $=|20|/5=4$ ; signe positif → classe $+1$.

$(2w,2w_0)$ définit le **même** hyperplan (l'équation $w\cdot x+w_0=0$ ne change pas quand on multiplie par une constante). Cette ambiguïté est levée dans le SVM en imposant $y_i(w\cdot x_i+w_0)=1$ pour les points les plus proches (forme canonique).

---

## C. Marge maximale (forme primale)

### Q6. Expliquer l'intuition de la marge maximale (analogie de la route). Pourquoi cet hyperplan est-il préférable aux autres hyperplans sans erreur ?

La **marge** est la largeur de la « route » entre les deux classes : la distance de l'hyperplan aux exemples les plus proches. Le classifieur à marge maximale est celui qui satisfait les contraintes des exemples et qui est **le plus robuste aux variations de $S$** : si les points bougent un peu (nouvelles données, bruit), la route centrée a le plus de chances de rester valable. Tous les hyperplans sans erreur ont le même risque empirique (nul), mais celui de marge maximale a la capacité la plus contrôlée (borne de généralisation, voir Q17). C'est l'idée de Vapnik (1992) qui donne les **SVM** (Séparateurs à Vastes Marges).

### Q7. Mettre en équation le SVM à marge dure : codage des classes, contraintes, largeur de la marge, forme primale. Pourquoi minimise-t-on $\tfrac12\|w\|^2$ ? Pourquoi dit-on que le problème est « facile » ?

1. **Codage :** $y_i=\pm1$ ; un point est bien classé ssi $y_i(w\cdot x_i+w_0)>0$.
2. **Échelle :** on impose $y_i(w\cdot x_i+w_0)\ge1$ pour tout $i$ (égalité pour les points les plus proches). Les hyperplans $w\cdot x+w_0=\pm1$ bordent la « route ».
3. **Largeur :** distance d'un point au bord $=1/\|w\|$ de chaque côté, donc **marge $=2/\|w\|$**.
4. **Problème :** maximiser $2/\|w\|$ ⇔ minimiser $\|w\|$ ⇔ minimiser $\tfrac12\|w\|^2$ (le carré et le $\tfrac12$ rendent le problème dérivable et plus simple).

$$
\min_{w,w_0}\ \tfrac12\|w\|^2\quad\text{sous}\quad y_i(w\cdot x_i+w_0)\ge1,\ i=1..m
$$

C'est un problème **quadratique convexe à contraintes linéaires** : **optimum global unique**, pas de minima locaux, algorithmes efficaces (contrairement à un réseau de neurones).

### Q8. (Calcul) Quatre points dans le plan : $(0,0)$ et $(-1,-1)$ de classe $-1$ ; $(2,2)$ et $(3,3)$ de classe $+1$. On admet que les points les plus proches des deux classes sont $(0,0)$ et $(2,2)$. Déterminer $w$, $w_0$, la marge, les vecteurs supports. Classer $(1,0)$ et $(1.5,1.5)$.

Par symétrie, $w$ est colinéaire à $(1,1)$ : $w=(a,a)$. Contraintes saturées sur les deux points les plus proches :

- $(0,0)$, $y=-1$ : $-(w_0)=1\Rightarrow w_0=-1$.
- $(2,2)$, $y=+1$ : $4a+w_0=1\Rightarrow 4a=2\Rightarrow a=0.5$.

Donc $w=(0.5,0.5)$, $w_0=-1$ ; $\|w\|=0.5\sqrt2\approx0.707$ ; **marge $=2/\|w\|\approx2.83$** (égale à la distance entre $(0,0)$ et $(2,2)$ : $2\sqrt2$ ✓).
Vérification des autres points : $(3,3)$ : $h=1.5+1.5-1=2$, $y\,h=2\ge1$ ✓ (hors marge) ; $(-1,-1)$ : $h=-2$, $y\,h=2\ge1$ ✓.
**Vecteurs supports :** $(0,0)$ et $(2,2)$.
$(1,0)$ : $h=0.5-1=-0.5\to$ classe $-1$ ; $(1.5,1.5)$ : $h=0.75+0.75-1=0.5\to$ classe $+1$ (mais $|h|<1$ : à l'intérieur de la marge).

---

## D. Forme duale et vecteurs supports

### Q9. Écrire le Lagrangien du SVM et les relations obtenues en annulant les dérivées. Qu'énonce le théorème du représentant ?

$$
L(w,w_0,\alpha)=\tfrac12\|w\|^2-\sum_i\alpha_i\big(y_i(w\cdot x_i+w_0)-1\big),\quad\alpha_i\ge0
$$

Chaque $\alpha_i$ est le « prix » de la contrainte $i$. La solution est un **point selle** : minimum en $(w,w_0)$, maximum en $\alpha$.

$$
\frac{\partial L}{\partial w}=0\Rightarrow w^*=\sum_i\alpha_iy_ix_i\qquad\frac{\partial L}{\partial w_0}=0\Rightarrow\sum_i\alpha_iy_i=0
$$

**Théorème du représentant :** le vecteur optimal $w$ est une **combinaison linéaire des exemples d'apprentissage**.

### Q10. Écrire le problème dual et la fonction de décision. Pourquoi la forme duale est-elle capitale pour la suite du cours ?

$$
\max_\alpha\ \sum_i\alpha_i-\tfrac12\sum_{i,j}\alpha_i\alpha_jy_iy_j\langle x_i,x_j\rangle\quad\text{avec }\alpha_i\ge0,\ \sum_i\alpha_iy_i=0
$$

$$
h^*(x)=\mathrm{sign}\Big(\sum_{i\in\mathcal P_S}\alpha_i^*y_i\langle x_i,x\rangle+w_0^*\Big)
$$

Deux raisons : (1) les données n'y apparaissent **que par des produits scalaires** $\langle x_i,x_j\rangle$ → porte d'entrée de l'**astuce du noyau** ; (2) elle révèle que la solution ne dépend que de **quelques points** (les vecteurs supports).

### Q11. Énoncer les conditions KKT. Que valent $\alpha_i$ pour un point hors marge et pour un point sur le bord ? Qu'est-ce qu'un vecteur support ? Comment calcule-t-on $w_0$ ? Quel parallèle avec le $k$-NN ?

**KKT :** $\alpha_i\,[y_i(w\cdot x_i+w_0)-1]=0$ : pour chaque point, l'un des deux facteurs est nul.

- $\alpha_i=0$ : point **strictement hors de la marge** → aucun rôle, on pourrait le supprimer sans rien changer.
- $\alpha_i>0$ : point **exactement sur le bord** ($y_i(w\cdot x_i+w_0)=1$) → **vecteur support**, il « tient » l'hyperplan (billes qui touchent la planche).
  La solution est donc **creuse** (parcimonieuse).

**$w_0$ :** avec un vecteur support $(x_c,y_c)$ : $y_c(w\cdot x_c+w_0)=1\Rightarrow w_0=y_c-w\cdot x_c$.

**Parallèle $k$-NN :** vecteurs supports = voisins pertinents ; $\alpha_i$ = poids des voisins ; produit scalaire (puis noyau) = mesure de similarité. Le SVM est un « $k$-NN intelligent » qui **apprend** quels exemples garder ; toute la solution est déterminée par la **matrice de Gram** $G_{ij}=\langle x_i,x_j\rangle$.

### Q12. (Calcul) Sur l'exemple de Q8, trouver les $\alpha_i$, vérifier $\sum\alpha_iy_i=0$, retrouver $w$ et $w_0$ par la forme duale, et vérifier que la valeur du dual égale $\tfrac12\|w\|^2$.

Seuls les deux vecteurs supports ont $\alpha\neq0$ ; $\sum\alpha_iy_i=0\Rightarrow\alpha_{(2,2)}=\alpha_{(0,0)}=\alpha$. Alors $w=\alpha\cdot(+1)\cdot(2,2)+\alpha\cdot(-1)\cdot(0,0)=(2\alpha,2\alpha)$. Avec $w=(0.5,0.5)$ : $\alpha=0.25$.
Points non supports : $\alpha=0$ pour $(3,3)$ et $(-1,-1)$.
$w_0=y_c-w\cdot x_c=1-(0.5\cdot2+0.5\cdot2)=-1$ ✓ (avec $x_c=(2,2)$).
**Dual :** $\sum\alpha_i=0.5$ ; le terme quadratique vaut $\tfrac12\|w\|^2=\tfrac12\cdot0.5=0.25$ ; valeur du dual $=0.5-0.25=0.25=\tfrac12\|w\|^2$ (dualité forte ✓).

---

## E. Marges souples (soft margin)

### Q13. Pourquoi introduit-on les marges souples ? Qu'est-ce que la variable d'écart (slack) ? Écrire la perte charnière et le critère complet du SVM.

**Pourquoi :** si les classes ne sont pas séparables (chevauchement, erreurs d'étiquetage), la marge dure n'a **aucune** solution ; même séparables, **un seul point aberrant** peut imposer une marge minuscule et un hyperplan absurde.
**Idée :** tolérer des violations de contraintes à condition de les **payer**. Le niveau de violation est la variable d'écart (*slack*).

$$
\ell_{hinge}(y,h(x))=\max(0,1-y\,h(x))
$$

- $y\,h\ge1$ : bien classé **et** hors marge → perte 0 ;
- $0<y\,h<1$ : bien classé mais **dans la marge** → petite perte ;
- $y\,h<0$ : **mal classé** → perte $>1$ croissant linéairement.

$$
w^\star=\arg\min_w\Big[\sum_i|1-y_ih(x_i)|_++\tfrac12w^\top w\Big]
$$

Risque empirique (pertes) + régularisation (le terme de marge) : exactement la recette de Q3.

### Q14. (Calcul) Avec l'hyperplan de Q8, calculer la perte charnière de chaque point du jeu d'entraînement, plus celle d'un point ajouté $(1.5,1.5)$ de classe $-1$ et d'un point $(1,0)$ de classe $+1$. Calculer le critère de la diapo 62 sur ces deux derniers seuls (en plus du terme $\tfrac12\|w\|^2$).

Valeurs de $h$ : $(0,0)$ : $-1$ ; $(2,2)$ : $+1$ ; $(3,3)$ : $+2$ ; $(-1,-1)$ : $-2$. Les $y\,h$ valent $1,1,2,2$ → pertes **0, 0, 0, 0**.
Point ajouté $(1.5,1.5)$, $y=-1$ : $h=0.5$, $y\,h=-0.5$ → perte $1-(-0.5)=1.5$ (mal classé).
Point $(1,0)$, $y=+1$ : $h=-0.5$, $y\,h=-0.5$ → perte $1.5$ (mal classé).
Somme des pertes de ces deux points $=3.0$ ; $\tfrac12\|w\|^2=0.25$ ; critère $=3.25$.

### Q15. Pourquoi remplacer la perte 0-1 par une « perte de substitution » ? Comparer 0-1, charnière, logistique et quadratique. Que signifie « c'est la forme de la fonction de coût qui induit la régularisation » ?

La perte 0-1 (vraie erreur) est **non convexe** et impossible à optimiser efficacement. On la remplace par une perte **convexe qui la majore** (*surrogate loss*) : la **charnière** (la plus utilisée, « pour de bonnes raisons théoriques »), la logistique ou la quadratique. La charnière est **plate (nulle) après la marge** : elle se désintéresse des points « faciles » et ne s'intéresse qu'à ceux proches de la frontière, d'où la **parcimonie** (seuls les vecteurs supports comptent). C'est le sens de la diapo 64 : la forme du coût induit un effet de régularisation.

### Q16. Que contrôle la constante $C$ ? Écrire la formulation avec variables d'écart $\xi_i$ et les trois types de points selon $\alpha_i$ (avec $\alpha_i\le C$). **(Calcul)** Avec $C=100$, interpréter $\alpha=0$, $\alpha=40$, $\alpha=100$. Quelle réserve sur la diapo 66 ?

$$
\min_{w,w_0,\xi}\ \tfrac12\|w\|^2+C\sum_i\xi_i\quad\text{sous}\quad y_i(w\cdot x_i+w_0)\ge1-\xi_i,\ \xi_i\ge0
$$

Dans le dual, les multiplicateurs sont bornés : $0\le\alpha_i\le C$.

- $\alpha_i=0$ : bien classé, hors marge ;
- $0<\alpha_i<C$ : **pile sur le bord** de la marge ;
- $\alpha_i=C$ : **dans la marge ou mal classé** (le point « tire » au maximum).

Avec $C=100$ : $\alpha=0$ → hors marge ; $\alpha=40$ → sur le bord (vecteur support « libre ») ; $\alpha=100$ → dans la marge ou mal classé (violation).

**Rôle de $C$ :** compromis entre largeur de marge et tolérance aux erreurs (compromis biais-variance). Convention usuelle : $C$ grand → erreurs coûteuses → marge étroite → surapprentissage ; $C$ petit → tolérant → marge large → sous-apprentissage. Se règle par **validation croisée**. *(Réserve : la diapo 66 semble inverser cette convention, voir l'en-tête.)*

---

## F. Pourquoi la marge est si importante (théorie)

### Q17. Pourquoi les bornes classiques fondées sur la dimension VC sont-elles inutiles pour les SVM ? Quel résultat de Vapnik les remplace ? Quelles sont les conséquences (SRM vs marge, noyaux) ?

Avec $d_{VC}=d+1$, la borne sur l'erreur de généralisation est **vide quand $d$ est grand devant $m$** ; or les SVM travaillent justement en très grande dimension (infinie avec le noyau gaussien).

**Résultat de Vapnik :** l'erreur de généralisation peut être bornée en fonction du rapport entre le **rayon $R$ de la plus petite sphère contenant les données** et la **marge $\gamma$**, **indépendamment de $d$**.

**Conséquences :** la marge est une **mesure de capacité indépendante de la dimension**, donc applicable en dimension infinie (central pour les noyaux). Contrairement à la SRM classique (structure fixée **a priori**), la marge contrôle automatiquement la capacité **après avoir vu les exemples**. La charnière agit comme régulariseur, et la marge contrôle la vitesse de convergence.
**En clair :** un SVM ne surapprend pas malgré un nombre immense de dimensions car il cherche le séparateur de marge maximale, très contraint ; c'est la marge, pas la dimension, qui mesure la complexité.

### Q18. (Calcul) La borne $R(h)\le\frac1{m-\#sv}\Big(\#sv\,\ln\frac{e\,m}{\#sv}+\ln\frac m\delta\Big)$ (avec probabilité $1-\delta$). Calculer-la pour $m=1000$, $\delta=0.05$ avec $\#sv=50$, puis $\#sv=200$. Interpréter.

- $\#sv=50$ : $\ln(e\cdot1000/50)=\ln54.37\approx3.996$ ; $50\times3.996\approx199.8$ ; $\ln(1000/0.05)=\ln20000\approx9.90$ ; somme $\approx209.7$ ; divisé par $950$ → **≈ 0.22**.
- $\#sv=200$ : $\ln(e\cdot1000/200)=\ln13.59\approx2.61$ ; $200\times2.61\approx521.9$ ; $+9.90\approx531.8$ ; divisé par $800$ → **≈ 0.66**.

**Interprétation :** plus $\#sv$ est **petit**, meilleure est la borne (argument de compression : une solution qui n'a besoin que de peu d'exemples est « simple », donc probablement généralisable).

---

## G. Non-linéaire : redescription, Cover, noyaux

### Q19. Pourquoi une redescription ? Expliquer l'idée (analogie des billes), la notation $\Phi:X\to F$, et ce que devient un séparateur linéaire de $F$ dans $X$. Quelles deux difficultés se posent ?

Un séparateur linéaire ne suffit pas toujours (ex. des « + » au centre entourés d'un anneau de « − »). On **augmente la dimension** en générant de nouvelles coordonnées (combinaisons, fonctions des descripteurs), puis on cherche un séparateur **linéaire** dans le nouvel espace $F$ (espace de redescription, *feature space*) via $\Phi$ (*feature expansion map*). *Analogie :* des billes mélangées sur une table ; en lançant les rouges en l'air, un plan horizontal les sépare. Un séparateur linéaire dans $F$ = une **frontière non linéaire dans $X$**.

**Difficultés :** (1) comment trouver $\Phi$, qui respecte la vraie similarité entre données ? (2) comment garder des **calculs faisables** en dimension très grande ou infinie ? La seconde trouve sa réponse dans le noyau.

### Q20. Énoncer le théorème de Cover. **(Calcul)** Appliquer-le aux 5 points de l'exemple 1D du cours puis après projection $x\mapsto(x,x^2)$. Pourquoi est-il utile pour les SVM ?

**Cover :** quelle est la probabilité que $m$ points, étiquetés au hasard, soient linéairement séparables en dimension $d$ ?

- Si $m>2(d+1)$ : probabilité **faible**.
- Pour $d$ grand et $m<2(d+1)$ : la probabilité que **n'importe quel étiquetage** soit séparable tend vers **1**.
  Donc projeter en grande dimension rend la séparabilité beaucoup plus probable.

**Calcul :** en 1D, $d=1$, $2(d+1)=4$ ; avec $m=5>4$, la séparabilité est improbable ✓ (et l'étiquetage « + + − − + » ne l'est pas). Après projection en $(x,x^2)$, $d=2$, $2(d+1)=6>5$ : séparable avec forte probabilité, et c'est le cas (une parabole sépare).
Utile car il justifie de **projeter en grande dimension** pour linéariser le problème.

### Q21. Expliquer l'astuce du noyau (kernel trick) : observation de départ, approche naïve, approche astucieuse. Écrire le dual et la fonction de décision avec un noyau.

**Observation :** dans le dual (et dans la décision), les données n'apparaissent que par des produits scalaires $\langle x_i,x_j\rangle$. Dans $F$ il suffit de les remplacer par $\langle\Phi(x_i),\Phi(x_j)\rangle$.
**Naïf :** calculer explicitement $\Phi(x)$ pour chaque point : coûteux, voire impossible si $\dim F=\infty$.
**Astuce :** trouver $K(x,x')=\langle\Phi(x),\Phi(x')\rangle$ qui donne directement le produit scalaire dans $F$ **sans jamais calculer $\Phi$** ; on écrit l'algorithme uniquement avec des produits scalaires et on remplace chaque $\langle x,x'\rangle$ par $K(x,x')$.

$$
\max_\alpha\Big[\sum_i\alpha_i-\tfrac12\sum_{i,j}\alpha_i\alpha_jy_iy_jK(x_i,x_j)\Big]\qquad h(x)=\mathrm{sign}\Big(\sum_{i\in\mathcal P_S}\alpha_iy_iK(x_i,x)+w_0\Big)
$$

*Analogie :* une formule raccourcie donne directement le score de ressemblance sans construire les 10 000 colonnes.

### Q22. (Calcul) Pour $x=(x_1,x_2)$ et $\Phi(x)=(x_1^2,x_2^2,\sqrt2x_1x_2)$ : montrer que $\langle\Phi(x),\Phi(z)\rangle=\langle x,z\rangle^2$. Vérifier numériquement avec $x=(1,2)$ et $z=(3,1)$. L'espace $F$ associé à un noyau est-il unique ? Quelle frontière obtient-on dans $X$ ?

$\langle\Phi(x),\Phi(z)\rangle=x_1^2z_1^2+x_2^2z_2^2+2x_1x_2z_1z_2=(x_1z_1+x_2z_2)^2=\langle x,z\rangle^2$ : $K(x,z)=\langle x,z\rangle^2$ est un noyau.

**Numérique :** $\langle x,z\rangle=3+2=5$, donc $K=25$. Via $\Phi$ : $\Phi(x)=(1,4,2\sqrt2)$, $\Phi(z)=(9,1,3\sqrt2)$ ; produit $=9+4+2\sqrt2\cdot3\sqrt2=9+4+12=25$ ✓.

**Unicité :** non. $(x_1^2,x_2^2,x_1x_2,x_2x_1)\in\mathbb R^4$ donne le même noyau.
**Frontière :** $h(x)=w_{11}x_1^2+w_{22}x_2^2+w_{12}\sqrt2x_1x_2+\dots$ est une **conique** dans le plan d'origine, linéaire dans $F$.

### Q23. (Calcul) **(a)** Pour $x\in\mathbb R$, trouver une application $\Phi$ telle que $\langle\Phi(x),\Phi(z)\rangle=(xz+1)^2$. **(b)** Quelle est la dimension de l'espace des monômes de degré exactement 3 pour $n=100$ variables ? Comparer au coût de $K(x,z)=(x^\top z)^3$.

**(a)** $\Phi(x)=(1,\sqrt2\,x,\,x^2)$ : $\langle\Phi(x),\Phi(z)\rangle=1+2xz+x^2z^2=(xz+1)^2$ ✓. C'est le noyau de l'exemple chiffré : le SVM peut tracer une **parabole** (monômes $1$, $x$, $x^2$).
**(b)** Dimension $\binom{n+d-1}{d}=\binom{102}{3}=\frac{102\cdot101\cdot100}{6}=171\,700$. Le calcul de $K$ ne coûte que $O(n)$ : un produit scalaire de dimension 100 puis une puissance. Gain spectaculaire en grande dimension.

### Q24. Quelle condition une fonction doit-elle vérifier pour être un noyau valide ? **(Calcul)** Les matrices $\begin{pmatrix}1&0.5\\0.5&1\end{pmatrix}$ et $\begin{pmatrix}1&2\\2&1\end{pmatrix}$ peuvent-elles être des matrices de Gram ? Le noyau sigmoïde est-il valide ?

La **matrice de Gram** $K_{ij}=K(x_i,x_j)$ doit être **semi-définie positive** : cela garantit que $K$ est un produit scalaire dans un certain espace et que l'optimisation reste convexe.

- $\begin{pmatrix}1&0.5\\0.5&1\end{pmatrix}$ : valeurs propres $1\pm0.5=1.5,\ 0.5$, toutes $\ge0$ → **valide**.
- $\begin{pmatrix}1&2\\2&1\end{pmatrix}$ : valeurs propres $1\pm2=3,\ -1$ → une valeur propre négative → **pas une matrice de Gram**, ce ne peut pas être un noyau.
  Le noyau **sigmoïde** $\tanh(\kappa x^\top z+\theta)$ n'est **pas** semi-défini positif en général : ce n'est donc pas un « vrai » noyau (il rappelle les réseaux de neurones).

### Q25. Présenter les noyaux usuels : polynomial exact, polynomial inhomogène, gaussien (RBF), sigmoïde. Que fait $\gamma$ (ou $\sigma$) ? **(Calcul)** Calculer $K(x,z)$ pour le RBF avec $x=(1,0)$, $z=(0,1)$ et $\sigma=1$ ; relier $\gamma$ et $\sigma$.

| Noyau                  | Formule                           | Capture                                               |
| ---------------------- | --------------------------------- | ----------------------------------------------------- |
| Polynomial exact       | $(x^\top z)^d$                  | produits d'**exactement** $d$ variables       |
| Polynomial inhomogène | $(x^\top z+c)^d$                | produits d'**au plus** $d$ variables          |
| Gaussien / RBF         | $\exp(-\gamma\|x-z\|^2)$        | similarité **locale** (proche ≈ 1, loin ≈ 0) |
| Sigmoïde              | $\tanh(\kappa x^\top z+\theta)$ | proche des réseaux de neurones ; **pas** PSD   |

**RBF :** pas de $\Phi$ explicite fermé, espace $F$ de **dimension infinie**. $\gamma$ grand / $\sigma$ petit → influence très locale, frontière très flexible, plus de vecteurs supports, **risque de surapprentissage** ; $\gamma$ petit / $\sigma$ grand → frontière lisse.

**Calcul :** $\|x-z\|^2=1+1=2$ ; avec $\gamma=1/(2\sigma^2)=0.5$ : $K=\exp(-0.5\times2)=e^{-1}\approx0.368$.

### Q26. Comment fabriquer de nouveaux noyaux ? Pourquoi dit-on que les méthodes à noyaux sont « modulaires » ? Donner des exemples de noyaux pour objets non vectoriels.

**Règles de composition** (à partir de noyaux valides $\kappa_1,\kappa_2$) : constante positive $c\kappa_1$ ; somme $\kappa_1+\kappa_2$ ; produit $\kappa_1\kappa_2$ ; exponentielle $\exp(\kappa_1)$ ; polynôme à coefficients positifs ; composition avec une application $\Phi$ ; combinaison de noyaux portant sur des **sous-ensembles de variables** différents (utile pour des données hétérogènes).

**Modularité :** on **découple** l'algorithme (linéaire, générique) et la **description des données** (le noyau). Il suffit de savoir mesurer une **similarité** entre objets, même non vectoriels : génomes, textes, images, vidéos, graphes sociaux. Exemples : *string kernels* (chaînes de caractères), noyaux de graphes fondés sur des **parcours** (*walks*). Le même SVM classe des protéines, des textes ou des graphes moléculaires en changeant le noyau ; c'est aussi *le* seul endroit où injecter de la connaissance a priori.

---

## H. L'exemple chiffré complet (1D)

### Q27. Données : $x=(1,2,4,5,6)$, classes $(+1,+1,-1,-1,+1)$, noyau $k(x_i,x_j)=(x_ix_j+1)^2$, $C=100$. Le solveur donne $\alpha=(0,\,2.5,\,0,\,7.333,\,4.833)$.

- **(a)** Pourquoi ces données ne sont-elles pas linéairement séparables en 1D ?
- **(b)** Vérifier $\sum\alpha_iy_i=0$ ; quels sont les vecteurs supports ?
- **(c)** Retrouver $h(x)=0.6667x^2-5.333x+b$ et calculer $b$.
- **(d)** Calculer $h(x)$ pour $x=1,2,4,5,6$ puis la classe prédite pour $x=3$ et $x=7$. Où se situent les racines ?
- **(e)** Interpréter.

**(a)** Sur la droite : « + + − − + » : aucun seuil unique ne sépare les classes (cf. Q20 : $m=5>2(d+1)=4$).

**(b)** $2.5\cdot(+1)+7.333\cdot(-1)+4.833\cdot(+1)=2.5-7.333+4.833=0$ ✓. Vecteurs supports : $x=2,5,6$ ($\alpha\neq0$). $x=1$ et $x=4$ n'interviennent pas.

**(c)** $h(x)=2.5(2x+1)^2-7.333(5x+1)^2+4.833(6x+1)^2+b$.

- coefficient de $x^2$ : $2.5\cdot4-7.333\cdot25+4.833\cdot36=10-183.33+174=0.667$ ;
- coefficient de $x$ : $2.5\cdot4-7.333\cdot10+4.833\cdot12=10-73.33+58=-5.333$ ;
- constante : $2.5-7.333+4.833=0$, donc il reste $b$.

$x=2$ est sur la marge : $h(2)=+1\Rightarrow0.6667\cdot4-5.333\cdot2+b=1\Rightarrow2.667-10.667+b=1\Rightarrow b=9$.

$$
h(x)=0.6667x^2-5.333x+9
$$

**(d)** $h(1)=4.33$ (+) ; $h(2)=1.00$ (+, marge) ; $h(4)=-1.67$ (−) ; $h(5)=-1.00$ (−, marge) ; $h(6)=1.00$ (+, marge). Tout est bien classé. $h(3)=6-16+9=-1\to$ classe $-1$ ; $h(7)=32.67-37.33+9=4.33\to$ classe $+1$. Racines : $x\approx4\pm1.58$, soit $\approx2.42$ et $\approx5.58$ ; $h$ est négative entre elles.

**(e)** La parabole est positive pour $x$ petit, négative entre les deux racines, puis repositive : elle capture « + + − − + ». **Une frontière courbe en 1D = une frontière linéaire dans l'espace $(1,x,x^2)$.**

---

## I. SVM en pratique et extensions

### Q28. Quels hyperparamètres choisir pour un SVM à noyau et comment ? Décrire l'effet de $\sigma$ sur le nombre de vecteurs supports (exemple du damier).

À choisir : le **type de noyau**, ses **paramètres** (degré $d$, largeur $\sigma$ ou $\gamma$…) et la constante **$C$**. Ces méta-paramètres se règlent généralement par **validation croisée** (voir le cours sur l'évaluation : ils se règlent sur la validation, pas sur le test).

**Damier + noyau gaussien :** avec deux valeurs de $\sigma$, on peut avoir $R_{emp}=0$ dans les deux cas, mais un **petit $\sigma$ donne plus de vecteurs supports** : la frontière est très « accrochée » à chaque point, il faut plus de points pour la définir. Le noyau et $C$ ont un effet fort sur la frontière ; il faut les régler. *(Chemin de régularisation, Hastie et al. 2004 : on peut suivre la solution quand $C$ varie.)*

### Q29. Comment estimer la performance en généralisation d'un SVM ? Que révèle la matrice de Gram sur le risque de sur- ou sous-apprentissage ?

- **Empiriquement :** validation croisée (méthode de référence).
- **Nombre de vecteurs supports :** plus il est petit, meilleure est la borne (Q18).
- **Matrice de Gram $G$ :** si elle n'a aucune structure, aucune régularité ne peut être apprise. Termes hors diagonale **très petits** (chaque point n'est similaire qu'à lui-même, par ex. $\sigma$ trop petit) → **surapprentissage** ; matrice **uniforme** (tout se ressemble) → **sous-apprentissage** (tout dans la même classe).

### Q30. Comment étend-on le principe « marge / perte charnière + noyau » à la régression et à la détection de nouveautés ? **(Calcul)** Perte $\varepsilon$-insensible avec $\varepsilon=0.5$ pour $y=3$ et $f(x)=2.2$, puis $f(x)=2.7$.

**Régression (SVR) :** perte $\varepsilon$-insensible $|y-f(x)|_\varepsilon=\max\{0,|y-f(x)|-\varepsilon\}$ : on ne pénalise pas les erreurs inférieures à $\varepsilon$ (un « tube » autour de la fonction). On minimise $\tfrac12\|w\|^2+C\sum_i|y_i-f(x_i)|_\varepsilon$ ; la solution s'écrit $f(x)=\sum_i(\alpha_i^*-\alpha_i)K(x_i,x)+w_0$.
**Détection de nouveautés (non supervisé) :** on cherche à **séparer au maximum le nuage de points de l'origine** ; ce qui tombe du mauvais côté est « nouveau » ou anormal.
**Autres :** multi-classes, ACP non linéaire, noyaux pour objets non vectoriels.

**Calcul :** $f=2.2$ : $|3-2.2|=0.8$, perte $=0.8-0.5=0.3$ ; $f=2.7$ : $|3-2.7|=0.3<0.5$ → perte $=0$ (dans le tube).

---

## J. Synthèse

### Q31. Citer et justifier trois forces et trois limites des SVM.

**Forces :** (1) **optimum global** (problème convexe, pas de minima locaux) ; (2) **changement d'espace automatique** via le noyau, et noyaux pour objets non vectoriels ; (3) séparateur défini par **peu d'exemples** (seulement ceux sur la marge, contrairement au $k$-NN qui garde tout), **parcimonie** ; (4) mélange des avantages du **non paramétrique** (s'ajuste aux données) et du **paramétrique** (résiste au surapprentissage) ; (5) introduit la marge comme mesure de capacité.
**Limites :** (1) **boîte noire** (vecteurs supports et poids dans un espace inconnu) ; (2) connaissances a priori **seulement** via le choix du noyau ; (3) **un seul niveau** de non-linéarité : mal adapté aux dépendances à longue distance (contrairement aux réseaux profonds). Recherche actuelle : apprentissage et combinaison de noyaux.

### Q32. Comparer SVM, perceptron, réseaux de neurones et $k$-NN.

- **SVM vs perceptron :** tous deux apprennent un séparateur linéaire ; le perceptron s'arrête sur un séparateur **quelconque** (l'un des infinité d'hyperplans sans erreur), le SVM choisit celui de **marge maximale**.
- **SVM vs réseaux de neurones :** SVM = problème **convexe** (optimum global) alors que l'entraînement d'un réseau peut se coincer dans des minima locaux. Le réseau **apprend** sa redescription (couches cachées, hiérarchie d'abstractions, plusieurs niveaux de non-linéarité) ; le SVM la fixe **implicitement par le choix du noyau** (un seul niveau). Les deux sont des boîtes noires peu interprétables.
- **SVM vs $k$-NN :** dans les deux cas la prédiction s'appuie sur des exemples pondérés par une similarité ; le SVM n'en garde que quelques-uns (vecteurs supports, poids $\alpha_i$), le $k$-NN garde toutes les données.

### Q33. (Synthèse sans formule) Expliquer comment un SVM à noyau gaussien classe des données non linéairement séparables sans surapprendre.

Le noyau gaussien projette **implicitement** les données dans un espace de dimension infinie où elles deviennent séparables linéairement (Cover), sans jamais calculer cette projection (*kernel trick*). Dans cet espace, le SVM cherche l'hyperplan de **marge maximale**, dont la capacité est contrôlée par la marge (indépendante de la dimension) et non par le nombre de coordonnées. La perte charnière avec $C$ tolère quelques erreurs. La solution ne dépend que de quelques **vecteurs supports**, ce qui la rend simple et parcimonieuse. $C$, $\gamma$ et le noyau se règlent par **validation croisée**.

### Q34. Relier ce cours aux cours précédents (introduction, réseaux de neurones, évaluation).

- **Cours 1 :** le SVM applique la **recette du risque empirique régularisé** (ERM contraint par un biais) ; la marge est une régularisation / un biais inductif ; la dimension VC et la théorie PAC fournissent le cadre (mais la borne VC est inutile en grande dimension, d'où la borne par la marge) ; le choix du noyau est le biais inductif (il doit être aligné avec la structure du problème) ; NFL : pas de noyau universel.
- **Cours 2 :** perceptron (même hyperplan) ; réseaux de neurones vs noyaux (redescription apprise vs fixée) ; minima locaux vs convexité ; sur-apprentissage.
- **Cours 3 :** $C$, le noyau et ses paramètres sont des **hyperparamètres** réglés par validation croisée ; matrice de confusion/ROC pour évaluer un SVM ; ne pas régler sur le test.

### Q35. Vrai / Faux (justifier)

- **(a)** « Sur des données séparables, il existe un unique hyperplan sans erreur. » → **Faux.** Il en existe une infinité ; le SVM choisit celui de marge maximale.
- **(b)** « La largeur totale de la marge vaut $1/\|w\|$. » → **Faux.** $2/\|w\|$ ($1/\|w\|$ de chaque côté).
- **(c)** « Supprimer un exemple qui n'est pas un vecteur support change l'hyperplan (marge dure). » → **Faux.** Son $\alpha_i=0$ ; supprimer un vecteur support peut en revanche le changer.
- **(d)** « Le SVM linéaire peut rester coincé dans un minimum local. » → **Faux.** Problème convexe, optimum global unique.
- **(e)** « Le noyau sigmoïde est toujours un noyau valide. » → **Faux.** Il n'est pas semi-défini positif en général.
- **(f)** « Le kernel trick demande de calculer $\Phi(x)$. » → **Faux.** On calcule directement $K(x,x')$.
- **(g)** « La perte 0-1 est convexe. » → **Faux.** Non convexe ; on utilise une perte de substitution convexe (charnière).
- **(h)** « L'espace de redescription du noyau gaussien est de dimension infinie. » → **Vrai.**
- **(i)** « Un $\gamma$ très grand dans un RBF favorise le surapprentissage. » → **Vrai.** Influence très locale, matrice de Gram proche de l'identité.
- **(j)** « Une marge dure admet toujours une solution. » → **Faux.** Pas si les classes ne sont pas séparables (d'où les marges souples).
- **(k)** « La borne de généralisation par la marge dépend de la dimension $d$. » → **Faux.** Elle dépend du rapport rayon / marge, pas de $d$.
- **(l)** « Plus il y a de vecteurs supports, meilleure est la borne. » → **Faux.** Plus $\#sv$ est petit, meilleure est la borne.

### Q36. (Mini-cas) On veut classer 500 séquences de protéines (fonction A / fonction B) avec un SVM. Les classes sont déséquilibrées (10 % de A). Proposer un protocole complet : noyau, hyperparamètres, évaluation, limites.

**Plan de réponse :**

1. **Données non vectorielles :** utiliser un **noyau pour objets non vectoriels** (*string kernel* sur les séquences), car il suffit de mesurer une similarité ; le même SVM s'applique en changeant le noyau. Vérifier que le noyau est valide (Gram PSD) ; c'est ici qu'on injecte la connaissance a priori du domaine.
2. **Marges souples :** les données biologiques sont bruitées ; utiliser la perte charnière avec $C$ pour tolérer des erreurs.
3. **Hyperparamètres :** noyau, ses paramètres et $C$ réglés par **validation croisée** (avec peu de données, 500 exemples) ; garder un ensemble de test jamais utilisé pour le réglage ; surveiller la matrice de Gram (surapprentissage si trop proche de l'identité).
4. **Évaluation :** l'accuracy seule est trompeuse (90 % en répondant « B ») : regarder **matrice de confusion, sensibilité, précision, $F_1$, ROC/AUC** ; accompagner d'un **intervalle de confiance** ; comparer les modèles avec un test statistique adapté.
5. **Déséquilibre :** pondérer les coûts d'erreur par classe (ou rééchantillonner) pour que le SVM ne soit pas biaisé vers la classe majoritaire (comme le SVM vs Naive Bayes du cours 3).
6. **Contrôle de la généralisation :** regarder le nombre de vecteurs supports (peu = bonne borne) ; la marge régularise même en dimension infinie.
7. **Limites :** boîte noire, peu interprétable ; a priori seulement via le noyau ; un seul niveau de non-linéarité.

---

---

# Cours 6 : Comparaison d'algorithmes de classification supervisée

---

## Partie A : Problématique et fil rouge

**Q1.** Un classifieur A obtient 82 % de précision, un classifieur B 80 %. Un camarade conclut « A est meilleur ». Pourquoi cette conclusion n'est-elle pas valide en l'état ? À quelle question générale le cours cherche-t-il à répondre ?

**Réponse.**
Une précision mesurée est une **estimation bruitée** de la vraie performance : elle dépend de l'échantillon de test, du découpage des données, de l'ensemble d'entraînement. Un écart de 2 points peut venir d'un vrai avantage de A **ou** du simple hasard.
La question du cours : **comment décider qu'une différence de performance observée est « réelle » et non due au hasard ?** Réponse : par des **tests de significativité statistique**, dont le choix dépend de la situation.

---

**Q2.** La diapo 2 cite quatre qualités à évaluer au-delà de la « précision moyenne » : consistance, robustesse, calibration, sûreté opérationnelle. Définis chacune en une phrase. Pourquoi une seule métrique ne suffit-elle pas ?

**Réponse.**

- **Consistance** : réussir **à chaque fois**, de façon prévisible, pas seulement en moyenne.
- **Robustesse** : continuer à fonctionner quand les **conditions sont imparfaites** (bruit, données légèrement différentes de l'entraînement).
- **Calibration** : **savoir quand on a réussi ou échoué** (confiance annoncée cohérente avec la fréquence réelle de succès : « sûr à 99 % » doit être juste 99 % du temps).
- **Sûreté opérationnelle** : **éviter les erreurs catastrophiques** (dix petites erreurs valent mieux qu'une seule fatale).

Une moyenne unique cache ces dimensions : deux modèles à 90 % peuvent avoir des comportements très différents (régularité, erreurs graves, fiabilité de leur confiance). Certaines dimensions (ex. la créativité, « archétype d'une tâche invérifiable ») échappent même aux tests quantitatifs.

---

**Q3.** Qu'est-ce qu'un benchmark ? Quel est son avantage, son inconvénient, et qu'illustre la métaphore du réverbère (diapos 3-4) ?

**Réponse.**
Un **benchmark** est un jeu de données standard sur lequel toute la communauté compare ses algorithmes.

- **Avantage** : l'**efficacité**. Si un modèle bat l'état de l'art sur le benchmark, c'est probablement un bon travail, sans avoir à relire des années de littérature.
- **Inconvénient** : il **rétrécit la vision collective** de la communauté.
- **Réverbère** : l'homme cherche ses clés sous le lampadaire « parce qu'il y a de la lumière », pas parce qu'il les a perdues là. De même, on cherche ce qui est **facile à mesurer** (benchmarks), pas forcément ce qui est **important**.

Conséquence : bien évaluer la qualité d'un travail deviendra plus coûteux. Et même un test statistique irréprochable ne valide que ce que la métrique et le benchmark mesurent.

---

**Q4.** Complète le tableau qui structure le cours (diapo 5) et explique en une phrase ce que représente la case « non traité ».

|                                      | 2 algorithmes | $k$ algorithmes |
| ------------------------------------ | ------------- | ----------------- |
| **1 seul jeu de données**     | ?             | ?                 |
| **Plusieurs jeux de données** | ?             | ?                 |

**Réponse.**

|                                      | 2 algorithmes                     | $k$ algorithmes  |
| ------------------------------------ | --------------------------------- | ------------------ |
| **1 seul jeu de données**     | t-test apparié, McNemar, 5×2 cv | *non traité*    |
| **Plusieurs jeux de données** | Wilcoxon (rang signé)            | Friedman + Nemenyi |

La case « non traité » signifie que le cours ne propose pas de procédure dédiée pour comparer $k>2$ algorithmes sur un unique jeu de données. On ne peut pas pour autant enchaîner des tests deux à deux sans précaution (voir comparaisons multiples, Q22).

---

## Partie B : Test d'hypothèse, p-value, erreurs

**Q5.** Définis l'hypothèse nulle $H_0$, la p-value et le seuil $\alpha$. Décris en trois étapes le raisonnement général d'un test (diapo 12).

**Réponse.**

- **$H_0$** : « il n'y a **aucune différence** » entre les algorithmes.
- **p-value** : probabilité, **sous $H_0$**, d'obtenir un résultat **au moins aussi extrême** que celui observé.
- **$\alpha$** : seuil fixé **à l'avance** (typiquement 0,05) en dessous duquel on rejette $H_0$.

Raisonnement (par l'absurde probabiliste) :

1. On suppose $H_0$ vraie.
2. On caractérise la distribution des différences que produirait le **hasard seul** sous $H_0$.
3. On regarde la différence observée : si elle est **très improbable** sous $H_0$ ($p<\alpha$), on **rejette** $H_0$ et on conclut à une vraie différence.

*Analogie du tribunal* : on ne condamne que si les preuves seraient très improbables pour un innocent ; si elles sont ambiguës, on ne condamne pas, ce qui ne prouve pas l'innocence.

---

**Q6.** Vrai ou faux ? Justifie chaque réponse.
(a) « $p=0{,}03$ signifie qu'il y a 3 % de chances que $H_0$ soit vraie. »
(b) « $p=0{,}30$ prouve que les deux algorithmes sont équivalents. »
(c) « Plus $p$ est petit, plus la différence entre les algorithmes est grande. »
(d) « Avec $\alpha=0{,}05$, si $H_0$ est vraie, on a 5 % de chances de la rejeter à tort. »

**Réponse.**
(a) **Faux.** La p-value est la probabilité des **données (ou pire) sachant $H_0$**, pas la probabilité de $H_0$ sachant les données.
(b) **Faux.** $p>\alpha$ signifie qu'on **ne peut pas rejeter** $H_0$ : manque de preuves, pas preuve d'égalité (peut-être manque-t-on de données).
(c) **Faux.** $p$ petit signifie « peu probable par hasard », pas « grosse différence ». Avec beaucoup de données, une différence minuscule peut être très significative.
(d) **Vrai.** $\alpha$ est la probabilité de l'erreur de type I (rejeter $H_0$ alors qu'elle est vraie). Attention : cela concerne les cas où $H_0$ est vraie, pas 5 % de *toutes* les conclusions.

---

**Q7.** Définis les erreurs de type I et de type II. Laquelle est contrôlée directement par $\alpha$ ? Donne une conséquence concrète de chacune quand on compare des algorithmes.

**Réponse.**

- **Type I (faux positif)** : rejeter $H_0$ alors qu'elle est vraie. Sa probabilité est **$\alpha$** (contrôlée directement).
- **Type II (faux négatif)** : ne pas rejeter $H_0$ alors qu'elle est fausse.

En pratique :

- Type I : déclarer à tort qu'une nouvelle méthode est meilleure (on « publie du bruit »).
- Type II : passer à côté d'une vraie amélioration, typiquement quand on a trop peu de plis ou de jeux de données (le test manque de **puissance**).

À taille d'échantillon fixée, diminuer $\alpha$ rend le test plus exigeant et augmente donc le risque de type II.

---

**Q8.** (a) Justifie que l'écart-type de l'estimation d'une précision $\hat a$ sur $n$ exemples est $\sqrt{\hat a(1-\hat a)/n}$. (b) Calcule-le pour $\hat a=0{,}9$ et $n=400$. (c) Que devient-il si $n$ est multiplié par 4 ? (d) Pour quelle valeur de $a$ l'estimation est-elle la plus « floue » ?

**Réponse.**
(a) Chaque exemple est bien classé (1) ou mal classé (0) : variable de **Bernoulli** de paramètre $a$, de variance $a(1-a)$. La moyenne de $n$ telles variables a une variance $n$ fois plus petite, d'où $\sigma_{\hat a}=\sqrt{\hat a(1-\hat a)/n}$.
(b) $\sqrt{0{,}9\times0{,}1/400}=\sqrt{0{,}000225}=\mathbf{0{,}015}$.
(c) Il est **divisé par 2** (racine de $n$) : $0{,}0075$.
(d) Quand $a$ est proche de **0,5**, car $a(1-a)$ y est maximal.

---

**Q9.** On teste $H_0$ : « la vraie précision vaut 0,5 » (le modèle ne fait pas mieux que le hasard). On observe $\hat a=0{,}58$. Calcule le nombre d'écarts-types qui séparent 0,58 de 0,5 dans deux cas : $n=100$, puis $n=400$. Conclus à $\alpha=0{,}05$ (seuil bilatéral $\approx1{,}96$). Quel enseignement en tires-tu ?

**Réponse.**

- $n=100$ : écart-type sous $H_0$ $=\sqrt{0{,}25/100}=0{,}05$ ; $z=(0{,}58-0{,}5)/0{,}05=\mathbf{1{,}6}<1{,}96$ → on **ne rejette pas** $H_0$ ($p\approx0{,}11$).
- $n=400$ : écart-type $=\sqrt{0{,}25/400}=0{,}025$ ; $z=0{,}08/0{,}025=\mathbf{3{,}2}>1{,}96$ → on **rejette** $H_0$ ($p\approx0{,}001$).

Enseignement : à écart observé identique, la **significativité dépend de la quantité de données**. Un résultat « non significatif » peut simplement venir d'un $n$ trop petit, d'où la prudence sur « $p>\alpha$ ≠ équivalence ».

---

## Partie C : t-test, t-test apparié, 5×2 cv

**Q10.** Pourquoi la statistique du t-test suit-elle une loi de Student et non une loi normale ? Que se passe-t-il quand le nombre de plis $k$ augmente ? Comment évolue la valeur critique quand $\alpha$ diminue ?

**Réponse.**
Le **vrai écart-type est inconnu** : on l'**estime** à partir de seulement $k$ plis, ce qui ajoute de l'incertitude. La loi de Student a des **queues plus lourdes** que la normale, c'est une « marge de prudence ».
Quand $k$ augmente, l'écart-type est mieux estimé, la loi de Student **tend vers la normale** et la valeur critique **diminue** (elle tend vers 1,96 à $\alpha=0{,}05$).
Quand $\alpha$ diminue (on veut être plus sûr), la valeur critique **augmente** : on est plus exigeant avant de rejeter.

---

**Q11.** On évalue un classifieur en 10-CV : erreur moyenne $\mu=0{,}23$, écart-type $\sigma=0{,}06$. On veut tester $H_0$ : « l'erreur vaut $\epsilon_0=0{,}20$ ». (a) Écris la statistique $\tau_t$ et explique le rôle de $\sigma/\sqrt k$. (b) Calcule-la. (c) Conclus sachant que la valeur critique bilatérale à $\alpha=0{,}05$ pour 10-CV est 2,262.

**Réponse.**
(a) $\tau_t=\dfrac{\sqrt k\,(\mu-\epsilon_0)}{\sigma}=\dfrac{\mu-\epsilon_0}{\sigma/\sqrt k}$, qui suit une loi $t$ à $k-1$ degrés de liberté. Le dénominateur $\sigma/\sqrt k$ est l'**erreur standard de la moyenne** (l'incertitude sur $\mu$). $\tau_t$ répond à : « mon écart vaut combien de fois l'incertitude de ma moyenne ? ».
(b) $\tau_t=\dfrac{\sqrt{10}\times0{,}03}{0{,}06}=\sqrt{10}\times0{,}5\approx\mathbf{1{,}58}$.
(c) $|1{,}58|<2{,}262$ → on **ne peut pas rejeter** $H_0$ : l'écart de 3 points n'est pas significatif.

---

**Q12.** Qu'est-ce qu'un test « apparié » et pourquoi est-il plus sensible ? Applique-le : sur 5 plis, les différences d'erreur $\Delta_i=\epsilon^A_i-\epsilon^B_i$ valent $0{,}02\;;\;0{,}04\;;\;0{,}01\;;\;0{,}03\;;\;0{,}05$. Calcule $\tau_t$ et conclus (valeur critique bilatérale à $\alpha=0{,}05$ pour $k=5$ : 2,776).

**Réponse.**
**Apparié** : A et B sont évalués sur **les mêmes plis** et on travaille sur les **différences par pli** $\Delta_i$, avec $H_0$ : « moyenne des $\Delta_i$ = 0 ». Certains plis sont faciles, d'autres difficiles **pour tous** les algorithmes : la différence par pli annule cette difficulté commune, seul reste l'effet propre aux algorithmes (comme comparer chaque personne avant/après un régime plutôt que deux groupes différents).

Calcul :

- $\mu=\dfrac{0{,}02+0{,}04+0{,}01+0{,}03+0{,}05}5=0{,}03$.
- Écarts à la moyenne : $-0{,}01;\,+0{,}01;\,-0{,}02;\,0;\,+0{,}02$ ; carrés : $0{,}0001+0{,}0001+0{,}0004+0+0{,}0004=0{,}001$.
- $\sigma^2=\dfrac{0{,}001}{k-1}=\dfrac{0{,}001}4=0{,}00025$, donc $\sigma\approx0{,}0158$.
- $\tau_t=\dfrac{\sqrt5\times0{,}03}{0{,}0158}\approx\mathbf{4{,}24}$.

$4{,}24>2{,}776$ → on **rejette** $H_0$ : différence significative (A et B n'ont pas la même performance).

---

**Q13.** Deux études comparent A et B en 10-CV (valeur critique 2,262). Dans les deux, l'écart moyen des différences par pli est de 3 points. Dans l'étude 1, l'écart-type des différences est de 2 points ; dans l'étude 2, il est de 12 points. Calcule $\tau_t$ dans chaque cas et conclus. Quelle « morale » en tires-tu ?

**Réponse.**

- Étude 1 : $\tau_t=\sqrt{10}\times3/2\approx\mathbf{4{,}74}>2{,}262$ → **significatif**.
- Étude 2 : $\tau_t=\sqrt{10}\times3/12\approx\mathbf{0{,}79}<2{,}262$ → **non significatif**.

Morale : ce n'est pas la **taille de l'écart de moyennes** qui compte, mais son **rapport à la variabilité** (à l'erreur standard). Un écart de 3 points est une preuve forte si les plis sont très réguliers, et ne prouve rien s'ils sont très dispersés. (Même logique que dans l'exemple NB/DT/NN de la diapo 17.)

---

**Q14.** Pourquoi dit-on que le t-test apparié en validation croisée n'est pas parfaitement rigoureux ? Quel est l'effet sur les conclusions ? Quelle parade propose Dietterich (1998) et sur quelle idée repose-t-elle ?

**Réponse.**
Le test suppose que les $\Delta_i$ sont **i.i.d.** (diapo 15), ce qui est faux : en 10-CV, deux ensembles d'entraînement partagent environ 80 % des **données** (≈ 89 % de chacun), donc les modèles sont très corrélés et leurs erreurs aussi.

Effet : la variance est **sous-estimée**, le test est **trop optimiste** et déclare trop souvent des différences « significatives » qui n'existent pas (**trop de faux positifs**).

Parade : la **validation croisée 5×2** (5 répétitions d'une CV à 2 plis). Dans chaque répétition, les deux ensembles d'entraînement sont **disjoints**, ce qui limite la dépendance ; on obtient 10 différences et une statistique approchée par une loi $t$ à 5 degrés de liberté. (À retenir : le **problème** et la **parade**, la formule n'est pas détaillée dans le cours.)

---

## Partie D : McNemar

**Q15.** Dessine le tableau de contingence du test de McNemar. Quels effectifs contiennent l'information sur « qui est meilleur » et pourquoi ? Que doit-on observer si A et B sont équivalents ?

**Réponse.**

|                       | A correct  | A incorrect |
| --------------------- | ---------- | ----------- |
| **B correct**   | $e_{00}$ | $e_{01}$  |
| **B incorrect** | $e_{10}$ | $e_{11}$  |

Seuls les **cas discordants** $e_{01}$ (B a raison, A a tort) et $e_{10}$ (A a raison, B a tort) comptent. Quand les deux ont raison ($e_{00}$) ou les deux ont tort ($e_{11}$), on n'apprend rien sur lequel est meilleur (comme deux correcteurs d'accord sur une copie).
Si A et B sont équivalents, quand ils divergent chacun « gagne » autant de fois : $e_{01}\approx e_{10}$.

---

**Q16.** Sur 300 exemples de test : $e_{00}=200$, $e_{11}=40$, $e_{01}=20$, $e_{10}=40$. Calcule la statistique de McNemar. Conclus à $\alpha=0{,}05$ (seuil $\chi^2=3{,}84$). Quel algorithme est le meilleur ?

**Réponse.**

$$
\tau_{\chi^2}=\frac{(|e_{01}-e_{10}|-1)^2}{e_{01}+e_{10}}=\frac{(|20-40|-1)^2}{60}=\frac{19^2}{60}=\frac{361}{60}\approx\mathbf{6{,}02}
$$

$6{,}02>3{,}84$ → on **rejette** $H_0$ : différence significative.
$e_{10}=40>e_{01}=20$ : A a raison là où B a tort deux fois plus souvent que l'inverse, donc **A est meilleur**. (Le « −1 » est une correction de continuité ; $e_{00}$ et $e_{11}$ n'interviennent pas dans le calcul.)

---

**Q17.** Compare McNemar et le t-test apparié en validation croisée : dans quelle situation préfère-t-on McNemar, et quelle est sa limite ?

**Réponse.**
McNemar n'utilise qu'**un seul découpage** entraînement/test, donc **un seul entraînement par algorithme**. Il est adapté quand **entraîner coûte très cher** (on ne peut pas le répéter $k$ fois).
Limite : il **ne capture pas la variabilité due au choix de l'ensemble d'entraînement** (un autre découpage aurait pu donner un autre résultat). Le t-test apparié en CV, lui, tient compte de cette variabilité mais souffre du problème de dépendance entre plis (Q14).

---

## Partie E : Wilcoxon (plusieurs jeux de données)

**Q18.** Pourquoi ne peut-on pas utiliser le t-test quand on passe de un à plusieurs jeux de données ? Donne les deux raisons de la diapo 20, définis un test non paramétrique et indique ses avantages et son prix.

**Réponse.**
Deux raisons :

1. La performance dépend de l'**adéquation algorithme/jeu de données** (un algorithme excellent sur l'un peut être médiocre sur l'autre, idée du « no free lunch »).
2. On ne peut plus supposer que les taux d'erreur suivent une **loi normale** : les jeux ne sont pas des tirages d'une même population et leurs échelles diffèrent (un jeu « facile » à 99 %, un « difficile » à 60 %).

Conclusion : il faut des tests **non paramétriques** : ils **ne supposent aucune forme de loi**, et s'appuient souvent sur les **rangs**. Avantage : **robustesse** (valeurs aberrantes, échelles différentes). Prix : un peu **moins de puissance** quand les hypothèses paramétriques seraient en fait vraies.

---

**Q19.** Sur 7 jeux de données, les différences de précision $d_i=\text{Perf}(h_2)-\text{Perf}(h_1)$ valent : $+0{,}04\;;\;-0{,}01\;;\;+0{,}08\;;\;+0{,}02\;;\;-0{,}03\;;\;0\;;\;+0{,}06$. Calcule $W_{s1}$, $W_{s2}$ et $T_{wilcox}$. Conclus sachant que la valeur critique de la table (bilatéral, 0,05) pour 6 jeux est 0.

**Réponse.**

1. On **écarte** le jeu à $d_i=0$ → $n=6$.
2. Rangs des $|d_i|$ : $0{,}01\to1$ ; $0{,}02\to2$ ; $0{,}03\to3$ ; $0{,}04\to4$ ; $0{,}06\to5$ ; $0{,}08\to6$.
3. Différences positives ($0{,}04;\,0{,}08;\,0{,}02;\,0{,}06$) : $W_{s1}=4+6+2+5=\mathbf{17}$.
4. Différences négatives ($-0{,}01;\,-0{,}03$) : $W_{s2}=1+3=\mathbf{4}$.
5. $T=\min(17,4)=\mathbf{4}$. (Contrôle : $17+4=21=6\times7/2$.)

$4>0$ → on **ne peut pas rejeter** $H_0$. Alors que $h_2$ gagne 4 fois sur 6, avec seulement 6 jeux utilisables le test manque de puissance : il faudrait une victoire quasi totale pour conclure.

---

**Q20.** (a) Pourquoi un **petit** $T$ indique-t-il une différence significative ? (b) Pour $n=9$ jeux, quelle est la valeur maximale de $T$ et que signifie-t-elle ? (c) Pourquoi Wilcoxon est-il préférable au simple décompte des victoires ?

**Réponse.**
(a) Si A et B sont équivalents, les victoires de chacun sont réparties au hasard : les deux sommes de rangs sont **proches** et $T=\min(W_{s1},W_{s2})$ est **grand**. Si un algorithme gagne presque toujours et largement, l'autre somme est **minuscule**, donc $T$ est **petit**.
(b) Le total des rangs est $n(n+1)/2=45$. $T$ est maximal quand les deux sommes sont égales : $T_{max}\approx22{,}5$ (22 s'il n'y a pas d'ex æquo), c'est-à-dire $\mu_T=n(n+1)/4$. Cela correspond à une situation **parfaitement équilibrée**, aucune différence.
(c) Le décompte des victoires ignore l'**ampleur** des écarts. Les rangs en tiennent compte : gagner de 30 points pèse plus que gagner de 0,1 point, sans être sensible aux valeurs aberrantes ni aux échelles différentes des jeux.

---

**Q21.** À partir de quelle taille $n$ approxime-t-on la loi de $T$ par une loi normale ? Donne $\mu_T$ et $\sigma_T$. Application : $n=30$ et $T=120$. Conclus à $\alpha=0{,}05$ (seuil bilatéral 1,96).

**Réponse.**
À partir de **$n\ge25$** (en dessous, on lit une table).

$$
\mu_T=\frac{n(n+1)}4,\qquad\sigma_T=\sqrt{\frac{n(n+1)(2n+1)}{24}},\qquad z=\frac{T-\mu_T}{\sigma_T}
$$

$n=30$ : $\mu_T=232{,}5$ ; $\sigma_T=\sqrt{30\times31\times61/24}\approx48{,}6$.
$z=(120-232{,}5)/48{,}6\approx\mathbf{-2{,}31}$. $|z|>1{,}96$ → on **rejette** $H_0$ : différence significative.

---

## Partie F : Friedman et Nemenyi ($k$ algorithmes, plusieurs jeux)

**Q22.** Pourquoi ne pas faire simplement tous les tests de Wilcoxon deux à deux entre $k$ algorithmes ? Calcule la probabilité d'au moins un faux positif pour $k=4$ algorithmes à $\alpha=0{,}05$ (tests supposés indépendants). Quelle démarche le cours préconise-t-il ? Quel lien le cours fait-il avec le cours sur les SVM (diapo 30) ?

**Réponse.**
Problème des **comparaisons multiples** : plus on fait de tests, plus la probabilité qu'**au moins un** soit un faux positif augmente, et le niveau de confiance s'effondre.
Pour $k=4$ : $\binom42=6$ paires ; $P(\ge1\text{ faux positif})=1-0{,}95^6\approx1-0{,}735=\mathbf{26\ \%}$ (au lieu de 5 %).

Démarche en **deux temps** :

1. un test **global** (omnibus) : « y a-t-il au moins une différence ? » → **Friedman** ;
2. si oui, un test **post hoc** pour localiser les paires qui diffèrent → **Nemenyi**.

Lien avec les SVM : c'est le rapport **risque empirique / risque réel**. Sélectionner le meilleur parmi beaucoup de candidats sur la base d'une mesure bruitée revient à faire du **surapprentissage sur le test** : « un algorithme peut sembler meilleur par le hasard des tests ».

---

**Q23.** Décris le principe du test de Friedman (comment obtient-on les rangs, ce qu'est le rang moyen). Pourquoi, sous $H_0$, le rang moyen attendu est-il $\frac{k+1}2$ ? Donne l'espérance et la variance du rang moyen d'un algorithme pour $k=4$ et $N=10$.

**Réponse.**
Test **non paramétrique** basé sur les rangs :

1. chaque algorithme est évalué sur chaque jeu ;
2. **pour chaque jeu**, on classe les algorithmes du meilleur (rang 1) au moins bon (rang $k$), avec le rang moyen en cas d'ex æquo ;
3. on calcule le **rang moyen** $r_i$ de chaque algorithme sur les $N$ jeux.

Sous $H_0$, aucun algorithme n'est systématiquement bien classé : par symétrie, chaque algorithme obtient en moyenne le rang du milieu, $\frac{1+\dots+k}{k}=\frac{k+1}2$. Un rang moyen **très éloigné** de cette valeur est suspect.

$E[r_i]=\dfrac{k+1}2=\mathbf{2{,}5}$ ; $\text{Var}(r_i)=\dfrac{k^2-1}{12N}=\dfrac{15}{120}=\mathbf{0{,}125}$ (écart-type $\approx0{,}35$).

---

**Q24.** Trois algorithmes A, B, C sont classés (1 = meilleur) sur 5 jeux :

| Jeu     | A | B | C |
| ------- | - | - | - |
| $D_1$ | 1 | 2 | 3 |
| $D_2$ | 1 | 3 | 2 |
| $D_3$ | 2 | 1 | 3 |
| $D_4$ | 1 | 2 | 3 |
| $D_5$ | 1 | 2 | 3 |

(a) Calcule les rangs moyens. (b) Calcule $\tau_{\chi^2}=\dfrac{12N}{k(k+1)}\Big(\sum r_i^2-\dfrac{k(k+1)^2}4\Big)$. (c) Calcule la statistique améliorée $\tau_F=\dfrac{(N-1)\tau_{\chi^2}}{N(k-1)-\tau_{\chi^2}}$ et conclus avec la valeur critique 4,459 ($\alpha=0{,}05$). (d) Que reste-t-il à faire ?

**Réponse.**
(a) $r_A=6/5=\mathbf{1{,}2}$ ; $r_B=10/5=\mathbf{2{,}0}$ ; $r_C=14/5=\mathbf{2{,}8}$. (Contrôle : somme $=6=k(k+1)/2$.)
(b) $\sum r_i^2=1{,}44+4+7{,}84=13{,}28$ ; $\dfrac{k(k+1)^2}4=12$ ; $\tau_{\chi^2}=\dfrac{12\times5}{12}\times(13{,}28-12)=5\times1{,}28=\mathbf{6{,}4}$.
(c) $\tau_F=\dfrac{4\times6{,}4}{5\times2-6{,}4}=\dfrac{25{,}6}{3{,}6}\approx\mathbf{7{,}11}$. Règle : $\tau_F$ **plus grand** que la valeur critique → rejet. $7{,}11>4{,}459$ → on **rejette** $H_0$ : au moins deux algorithmes diffèrent. (La version Iman-Davenport est valable aussi pour de petits $k$ et $N$.)
(d) Friedman ne dit **pas lesquels** : il faut un test **post hoc** (Nemenyi).

---

**Q25.** Après un Friedman significatif avec $k=4$ algorithmes et $N=10$ jeux, on obtient les rangs moyens $A:1{,}2$ ; $B:2{,}1$ ; $C:3{,}0$ ; $D:3{,}7$. (a) Donne la formule de la différence critique CD et calcule-la pour $\alpha=0{,}05$ ($q_\alpha=2{,}569$ pour $k=4$). (b) Quelles paires sont significativement différentes ? (c) Que mesure le dénominateur $\sqrt{k(k+1)/(6N)}$ ?

**Réponse.**
(a) $CD=q_\alpha\sqrt{\dfrac{k(k+1)}{6N}}=2{,}569\times\sqrt{\dfrac{20}{60}}=2{,}569\times0{,}577\approx\mathbf{1{,}48}$.
(b) Écarts de rangs moyens :

| Paire | Écart | > 1,48 ?      |
| ----- | ------ | ------------- |
| A–B  | 0,9    | non           |
| A–C  | 1,8    | **oui** |
| A–D  | 2,5    | **oui** |
| B–C  | 0,9    | non           |
| B–D  | 1,6    | **oui** |
| C–D  | 0,7    | non           |

Significatives : **A–C, A–D, B–D**.
(c) C'est l'**écart-type de la différence de deux rangs moyens sous $H_0$** : le test mesure de combien d'écarts-types deux rangs moyens sont éloignés ($q=\frac{\overline R_{j_1}-\overline R_{j_2}}{\sqrt{k(k+1)/(6N)}}$, on rejette l'égalité si $|q|>q_\alpha$).

---

**Q26.** Dans un diagramme de différence critique, A et B sont reliés par une barre, B et C aussi, mais A et C ne le sont pas. Comment l'interprètes-tu ? Est-ce contradictoire ? Comment lit-on un tel diagramme en général (diapo 35) ?

**Réponse.**
Lecture : A et B ne sont **pas significativement différents**, B et C non plus, mais **A et C le sont**. Ce n'est **pas contradictoire** : la relation « significativement différent » **n'est pas transitive**. B est « entre » A et C, trop proche des deux pour être séparé, alors que A et C sont distants de plus que CD. Un test peut manquer de puissance sur des écarts modérés (surtout avec peu de jeux).

Lecture générale : les méthodes sont **triées par rang moyen** (meilleure à gauche, rang 1 = meilleur). Les méthodes **reliées par une barre épaisse** ne sont pas significativement différentes ; deux méthodes **non reliées** le sont. Le segment « Critical Difference » donne l'échelle : deux rangs moyens plus éloignés que CD sont significativement différents.

---

**Q27.** Un article affirme : « notre méthode est significativement mieux classée que toutes les autres (Friedman + Nemenyi, 30 jeux, $p<0{,}05$) ». Que peut-on en déduire ? Que ne peut-on pas en déduire ? Quelles informations complémentaires demandes-tu ?

**Réponse.**
On peut en déduire que la méthode est **plus régulièrement bien classée** que les autres sur ces jeux, de façon peu probable par hasard.
On **ne peut pas** en déduire qu'elle est « bien meilleure » : un **rang est un ordre, pas une quantité** (la comparaison des rangs « ne dit rien sur les performances », diapo 36). Une méthode à 85,02 % qui bat systématiquement une méthode à 85,01 % est « rang 1 » à chaque fois, et le test peut être significatif alors que l'écart est **négligeable en pratique**.
À demander : les **valeurs de performance** (ampleur de l'écart, taille d'effet), pas seulement les rangs, et se rappeler que le résultat ne vaut que pour les **jeux de données** et la **métrique** choisis. **Significativité statistique ≠ importance pratique.**

---

## Partie G : Synthèse et esprit critique

**Q28.** Pour chaque situation, indique le test adapté et justifie brièvement.
(a) Un algorithme, 10-CV, on veut comparer son erreur à une valeur de référence $\epsilon_0=15\%$.
(b) Deux algorithmes, un jeu de données, 10-CV.
(c) Deux algorithmes, un jeu de données, un entraînement de plusieurs jours (un seul découpage possible).
(d) Deux algorithmes, 15 jeux de données.
(e) Deux algorithmes, 40 jeux de données.
(f) Cinq algorithmes, 20 jeux de données.
(g) Trois algorithmes, un seul jeu de données.

**Réponse.**
(a) **t-test de Student** (bilatéral) sur les $k$ erreurs de CV, $k-1$ degrés de liberté.
(b) **t-test apparié** sur les différences par pli ; à nuancer par le problème de dépendance des plis (→ **5×2 cv**).
(c) **McNemar** : un seul entraînement, tableau de contingence 2×2, on exploite les cas discordants.
(d) **Wilcoxon (rang signé)** : plusieurs jeux, non paramétrique ; $n<25$ → lecture dans une **table**.
(e) **Wilcoxon** encore, mais $n\ge25$ → **approximation normale** (statistique $z$).
(f) **Friedman** (global, version Iman-Davenport) puis **Nemenyi** (post hoc) si Friedman est significatif.
(g) Cas **non traité** dans le cours (une seule base, $k>2$ algorithmes) : il faut en tout cas éviter d'enchaîner des tests appariés sans tenir compte des **comparaisons multiples**.

---

**Q29.** (Question de rédaction.) Retrace la **logique d'enchaînement** du cours : pour chaque test, quel problème du précédent cherche-t-il à résoudre ? Termine par les limites de tout l'édifice.

**Réponse.**

1. **Point de départ** : une différence de précision observée n'est pas une différence prouvée → il faut tester $H_0$ (« pas de différence »).
2. **t-test** : un algorithme, plis de CV, comparé à une valeur $\epsilon_0$ ; Student car l'écart-type est estimé.
3. **t-test apparié** : pour comparer **deux** algorithmes, on travaille sur les **différences par pli** (annule la difficulté propre à chaque pli).
4. **Problème** : les plis se recouvrent, les $\Delta_i$ ne sont pas i.i.d. → trop de faux positifs. **Parades** : **5×2 cv** (entraînements disjoints) et **McNemar** (un seul découpage, utile si l'entraînement est coûteux, mais ne capture pas la variabilité de l'entraînement).
5. **Plusieurs jeux de données** : la performance dépend de l'adéquation algorithme/jeu et la normalité n'est plus plausible → tests **non paramétriques** sur les rangs : **Wilcoxon** (2 algorithmes).
6. **$k$ algorithmes** : enchaîner des tests deux à deux fait exploser les faux positifs (**comparaisons multiples**) → test **global Friedman**, puis **post hoc Nemenyi**.
7. **Limites** : les rangs disent l'ordre, pas l'ampleur (significativité ≠ importance pratique) ; un test ne valide que ce que la **métrique** et les **benchmarks** mesurent (le piège du réverbère) ; l'évaluation sérieuse devient plus coûteuse.

---

**Q30.** Un doctorant compare 6 algorithmes sur un **seul** jeu de données avec une 10-CV. Il réalise tous les t-tests appariés deux à deux, trouve que A bat F avec $p=0{,}03$ et conclut « A est le meilleur algorithme ». Critique cette démarche en citant au moins quatre problèmes.

**Réponse.**

1. **Comparaisons multiples** : 6 algorithmes → 15 paires ; $P(\ge1\text{ faux positif})=1-0{,}95^{15}\approx54\ \%$. Un « $p=0{,}03$ » isolé parmi 15 tests n'a plus du tout la même valeur.
2. **Sélection sur mesure bruitée** : choisir « le meilleur » parmi beaucoup de candidats sur un jeu bruité, c'est du surapprentissage sur le test (lien risque empirique / risque réel).
3. **Dépendance des plis** : chaque t-test apparié en CV viole l'hypothèse i.i.d. (recouvrement des ensembles d'entraînement) → variance sous-estimée, faux positifs supplémentaires (→ 5×2 cv).
4. **Un seul jeu de données** : la performance dépend de l'adéquation algorithme/jeu ; on ne peut pas généraliser (« A est le meilleur » en général). La procédure Friedman + Nemenyi du cours suppose **plusieurs jeux**.
5. **Un seul seuil $p$ ≠ ampleur** : $p=0{,}03$ ne dit pas si l'écart A–F est important en pratique ; il faut regarder les performances brutes.
6. **Conclusion abusive** : « $A$ bat $F$ » ≠ « $A$ est le meilleur de tous » (non-transitivité, pas de comparaison avec B, C, D, E contrôlée).

---

**Q31.** (Question de pont avec les cours précédents ; à adapter aux définitions de ton cours.) Relie ce cours à (a) l'évaluation de l'apprentissage (risque empirique/réel, validation croisée), (b) les SVM, (c) la conformal prediction. En quoi chacun éclaire-t-il l'idée « une précision mesurée est une estimation incertaine » ?

**Réponse.**
(a) **Évaluation** : l'erreur mesurée sur un échantillon fini (test ou plis de CV) est une **estimation du risque réel** ; sa variance, $\hat a(1-\hat a)/n$, est précisément ce que les tests prennent en compte. La validation croisée fournit les $k$ valeurs utilisées par le t-test, mais ses plis ne sont pas indépendants (Q14).
(b) **SVM** : le lien est fait explicitement à la diapo 30 : un écart apparent entre algorithmes peut n'être que du risque empirique favorable par hasard ; choisir le meilleur parmi beaucoup de candidats sur une mesure bruitée, c'est ne pas contrôler le risque réel.
(c) **Conformal prediction** : toujours une logique de **quantification de l'incertitude**, mais à un niveau différent. La conformal prediction quantifie l'incertitude **d'une prédiction** (ensembles de prédiction avec garantie de couverture), alors que les tests de ce cours quantifient l'incertitude **sur la comparaison de performances globales** de deux ou plusieurs algorithmes. Dans les deux cas, on refuse de traiter un chiffre ponctuel comme une vérité.

---

## Pièges fréquents à vérifier avant le partiel

| Piège                                                            | À retenir                                                                                                                                                          |
| ----------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Confondre « ne pas rejeter $H_0$ » et « $H_0$ est vraie » | Absence de preuve ≠ preuve d'absence                                                                                                                               |
| Lire la p-value comme $P(H_0)$                                  | C'est $P(\text{données}\mid H_0)$                                                                                                                                |
| Sens des décisions                                               | t-test : on rejette si $|\tau_t|\ge$ valeur critique. Wilcoxon : on rejette si $T$ est **petit**. Friedman : on rejette si $\tau_F$ est **grand** |
| Rangs (Friedman)                                                  | **Rang 1 = meilleur** ; « $f_B>f_A$ » en rangs = $f_B$ est **moins bon**                                                                          |
| Degrés de liberté                                               | t-test : $k-1$ ; McNemar : 1 ; Friedman : $\chi^2$ à $k-1$                                                                                                   |
| McNemar                                                           | Seuls $e_{01}$ et $e_{10}$ comptent ; seuil 3,84 à $\alpha=0{,}05$                                                                                           |
| Wilcoxon                                                          | Retirer les $d_i=0$ avant de ranger ; $n\ge25$ → approx. normale                                                                                               |
| Nemenyi                                                           | Non transitif ; CD dépend de $k$, $N$, $q_\alpha$                                                                                                            |
| Conclusion finale                                                 | Toujours ajouter : significatif ≠ important en pratique                                                                                                            |

**Rappel des coquilles de diapos** (déjà signalées dans ta fiche) : diapo 8 ($\sqrt{0{,}5(1-0{,}5)/100}$ et non « 9.5 »), diapo 11 (intervalle de non-rejet $[-t_{\alpha/2},t_{\alpha/2}]$ et sens de la phrase), diapo 22 (n = 9 jeux utilisés), diapos 32-33 (sommes vs moyennes de rangs). Si un point de ce type tombe à l'examen, vérifie la convention avec ton enseignant.

---

---

# Cours 7 : Les arbres de décision

## Questions type partiel : version corrigée

> Questions centrées sur les **définitions**, les **concepts** et la **logique d'enchaînement** du cours (peu de calcul, et seulement des calculs faisables à la main).
> Les renvois **(diapo N)** correspondent au cours d'A. Cornuéjols tel que résumé dans ta fiche.
> Les formules sont en LaTeX (`$...$`).
> Les questions sont **formulées différemment** du quiz de ta fiche (certains thèmes se recoupent forcément) : tu peux faire les deux.
> Valeurs utiles : $\log_2 3\approx1{,}585$ ; $I(\tfrac45,\tfrac15)=0{,}722$ ; $I(\tfrac25,\tfrac35)=0{,}971$ ; $I(\tfrac13,\tfrac23)=0{,}918$ ; $I(\tfrac34,\tfrac14)=0{,}811$.

---

## Partie A : Structure, représentation et espace de recherche

**Q1.** Voici un arbre de décision (« Faire du tennis ? ») :

```
Météo ?
├─ ensoleillé → Humidité ?
│                 ├─ haute    → NON
│                 └─ normale  → OUI
├─ couvert    → OUI
└─ pluvieux   → Vent ?
                  ├─ fort     → NON
                  └─ faible   → OUI
```

(a) Identifie les nœuds internes, les branches et les feuilles. (b) Écris l'arbre sous forme de disjonction de règles. (c) Classe l'exemple (ensoleillé, humidité normale, vent faible). (d) Pourquoi cette lecture en règles est-elle importante ?

**Réponse.**
(a) **Nœuds internes** (tests sur un attribut) : Météo, Humidité, Vent (3). **Branches** (une par valeur de l'attribut testé) : 7. **Feuilles** (classes) : 5.
(b) Un arbre est une **disjonction de règles** (un « OU » de « ET »), chaque chemin racine → feuille étant une règle :

- (Météo = ensoleillé ET Humidité = haute → NON)
- OU (Météo = ensoleillé ET Humidité = normale → OUI)
- OU (Météo = couvert → OUI)
- OU (Météo = pluvieux ET Vent = fort → NON)
- OU (Météo = pluvieux ET Vent = faible → OUI)

(c) On part de la racine : ensoleillé → test Humidité → normale → feuille **OUI**. (Le vent n'est jamais consulté : seul le chemin suivi compte.)
(d) Cette lecture explique que les arbres soient **interprétables** : on peut expliquer à un humain pourquoi une décision a été prise.

---

**Q2.** (a) Avec 6 attributs booléens, combien d'exemples différents peut-on décrire et combien de fonctions booléennes existe-t-il ? (b) Que veut dire « un arbre peut représenter toute fonction booléenne » ? Pourquoi cela ne suffit-il pas ? (c) Quelle est la limite de formalisme d'un arbre ?

**Réponse.**
(a) $2^6=\mathbf{64}$ exemples possibles ; chacun peut être de classe 0 ou 1, donc $2^{64}\approx1{,}8\times10^{19}$ fonctions (« environ $2\cdot10^{19}$ » dans le cours).
(b) Il existe pour toute fonction booléenne un arbre qui la calcule (le pouvoir d'expression est total). Mais **certaines fonctions exigent des arbres exponentiellement grands** (parité, majorité), alors que d'autres tiennent en un seul nœud : « pouvoir tout représenter » ne garantit ni un arbre petit, ni un arbre appris efficacement.
(c) Le formalisme est celui de la **logique propositionnelle** : l'arbre ne peut pas exprimer de **relations entre objets** (« X est plus grand que Y »), contrairement à la logique du premier ordre.

---

**Q3.** Quatre arbres différents (DT1 à DT4) peuvent être tous parfaitement cohérents avec un petit jeu de données (diapo 15). Qu'en conclus-tu sur le problème d'apprentissage ? Quel biais inductif adopte-t-on et comment le justifier ? Avec quel autre cours peux-tu faire un parallèle ?

**Réponse.**
Les données **ne désignent pas un arbre unique** : il faut une **préférence** (un **biais inductif**) pour choisir parmi les arbres cohérents.
Préférence usuelle : le **plus petit arbre**, par le **rasoir d'Ockham** (l'explication la plus simple est préférable, et en principe elle généralise mieux).
Parallèle : le **perceptron**, pour lequel il existe une infinité d'hyperplans séparateurs cohérents avec les données (et, pour les SVM, on a aussi dû choisir un critère de préférence : la marge).

---

**Q4.** (a) Combien d'arbres binaires existe-t-il pour $n=10$ et $n=20$ nœuds ? Que conclut-on sur la recherche du plus petit arbre ? (b) Décris la stratégie d'apprentissage adoptée en quatre étapes. (c) Quelle analogie l'illustre, et quelle limite en découle ?

**Réponse.**
(a) Nombres de **Catalan** : **16 796** pour $n=10$ ; **$6{,}56\times10^9$** pour $n=20$ (et on n'a pas encore compté le choix de l'attribut à chaque nœud). **Explorer tous les arbres pour trouver le plus petit est impossible.**
(b) Stratégie **gloutonne, descendante** (*top-down*) :

1. choisir le meilleur attribut à tester ;
2. créer une branche par valeur de cet attribut ;
3. répartir les exemples en sous-ensembles ;
4. recommencer sur chaque sous-ensemble jusqu'à un **critère d'arrêt** (feuille pure, plus d'attribut, plus assez d'exemples).

(c) À chaque carrefour on prend la rue qui rapproche le plus de la destination, sans calculer tous les trajets : rapide, mais **myope** (on peut rater le meilleur trajet global ; voir le XOR, Q22). La question centrale devient : comment mesurer qu'un attribut est « bon » ?

---

## Partie B : Entropie, gain d'information, Gini

**Q5.** (a) Donne la formule de l'entropie et ses propriétés (valeur minimale, valeur maximale, unité). (b) Calcule l'entropie d'un ensemble de 12 exemples contenant 6 « + » et 6 « − », puis de $\{9+,3-\}$ et de $\{6+,2-\}$. Que remarques-tu ? (c) Quelle est l'entropie pour 3 classes équiprobables ? 4 classes ?

**Réponse.**
(a) $I(S)=-\sum_{i=1}^{C}p(c_i)\log_2p(c_i)$. Elle est **nulle** pour une seule classe (pas d'incertitude), **croissante** quand les classes deviennent plus équiprobables, **égale à $\log_2 k$** pour $k$ classes équiprobables ; unité : le **bit**. Pour deux classes, c'est une courbe en cloche, maximale (1 bit) pour $P=0{,}5$.
(b) $\{6+,6-\}$ : $\mathbf1$ bit. $\{9+,3-\}$ : $P=0{,}75$ → $-0{,}75\log_20{,}75-0{,}25\log_20{,}25\approx0{,}311+0{,}5=\mathbf{0{,}811}$. $\{6+,2-\}$ : $P=0{,}75$ aussi → $\mathbf{0{,}811}$.
Remarque : l'entropie dépend des **proportions**, pas des effectifs bruts.
(c) $\log_23\approx\mathbf{1{,}585}$ bit ; $\log_24=\mathbf2$ bits.

---

**Q6.** Un ensemble $S$ de 10 exemples contient 6 « OUI » et 4 « NON ». Deux attributs sont candidats :

- **A** : $A=a_1$ → 5 exemples (4 OUI, 1 NON) ; $A=a_2$ → 5 exemples (2 OUI, 3 NON).
- **B** : $B=b_1$ → 4 exemples (4 OUI, 0 NON) ; $B=b_2$ → 6 exemples (2 OUI, 4 NON).

Calcule $I(S)$, $Gain(S,A)$ et $Gain(S,B)$. Quel attribut choisit-on à la racine ?

**Réponse.**

- $I(S)=-0{,}6\log_20{,}6-0{,}4\log_20{,}4\approx\mathbf{0{,}971}$.
- **A** : $I(a_1)=I(\tfrac45,\tfrac15)=0{,}722$ ; $I(a_2)=I(\tfrac25,\tfrac35)=0{,}971$. Entropie moyenne $=0{,}5\times0{,}722+0{,}5\times0{,}971=0{,}846$. $Gain(S,A)\approx\mathbf{0{,}125}$.
- **B** : $I(b_1)=0$ (pur) ; $I(b_2)=I(\tfrac13,\tfrac23)=0{,}918$. Entropie moyenne $=0{,}4\times0+0{,}6\times0{,}918=0{,}551$. $Gain(S,B)\approx\mathbf{0{,}420}$.

On choisit **B** (gain maximal) : il produit une feuille pure et une branche moins désordonnée.

---

**Q7.** Dans $Gain(S,A)=I(S)-\sum_v\frac{|S_v|}{|S|}I(S_v)$ : (a) pourquoi pondère-t-on par $|S_v|/|S|$ ? (b) Que signifie un gain nul ? (c) Quelle est la valeur maximale possible du gain ? (d) Comment interprète-t-on le gain en termes d'information ?

**Réponse.**
(a) Une branche de 2 exemples pèse peu : qu'elle soit pure ne prouve pas grand-chose ; une branche de 100 exemples pure est une vraie réussite. La pondération donne à chaque branche l'importance qu'elle mérite.
(b) Gain nul : connaître la valeur de $A$ **n'apporte aucune information** sur la classe (l'entropie moyenne après coupure est la même qu'avant). C'est ce qui se passe pour chaque attribut pris isolément dans le XOR.
(c) Le gain est au plus $I(S)$, atteint quand **toutes les branches sont pures** (entropie moyenne après coupure nulle). ➕ Le gain est aussi toujours $\ge0$.
(d) Il mesure dans quelle mesure connaître la valeur de $A$ apporte de l'information sur la classe d'un exemple : **baisse d'incertitude** obtenue par le test. On choisit l'attribut de **gain maximal**.

---

**Q8.** (a) Donne la formule de l'indice de Gini, sa valeur minimale et sa valeur maximale à 2 classes. (b) Reprends les données de la Q6 : calcule la baisse de Gini pour A et pour B. Les deux critères désignent-ils le même attribut ? (c) Compare entropie et Gini.

**Réponse.**
(a) $Gini(D)=1-\sum_jp_j^2$ (Breiman et al., 1984) ; **0** si la population est homogène ; **0,5** maximum à 2 classes (parts égales).
(b) $Gini(S)=1-0{,}6^2-0{,}4^2=\mathbf{0{,}48}$.

- **A** : $Gini(a_1)=1-0{,}8^2-0{,}2^2=0{,}32$ ; $Gini(a_2)=1-0{,}4^2-0{,}6^2=0{,}48$ ; moyenne $=0{,}40$ ; baisse $=\mathbf{0{,}08}$.
- **B** : $Gini(b_1)=0$ ; $Gini(b_2)=1-(\tfrac13)^2-(\tfrac23)^2=\tfrac49\approx0{,}444$ ; moyenne $=0{,}6\times0{,}444=0{,}267$ ; baisse $=\mathbf{0{,}213}$.

Les deux critères désignent **B**.
(c) Même objectif (mesurer l'impureté, nulle si pur, maximale si parts égales). Entropie : $-\sum p\log_2p$, maximum 1 à 2 classes, utilise un logarithme. Gini : $1-\sum p^2$, maximum 0,5, seulement des carrés (**plus rapide à calculer**). En pratique, ils donnent le plus souvent des arbres très semblables.

---

## Partie C : Problèmes pratiques

**Q9.** Attribut continu « Âge » avec classe (O/N) : 20→N ; 25→N ; 30→O ; 35→O ; 40→N ; 45→O. (a) Pourquoi discrétise-t-on et comment (quel test) ? (b) Quels seuils candidats et pourquoi seulement ceux-là ? (c) Calcule le gain de chacun. Lequel retient-on ?

**Réponse.**
(a) Un attribut numérique prend une infinité de valeurs : on le **coupe en deux** par un seuil, le test devient « $\text{Âge}>s$ ? ».
(b) On ne teste que les seuils **entre deux exemples consécutifs de classes différentes** : **27,5** (25 N / 30 O), **37,5** (35 O / 40 N), **42,5** (40 N / 45 O). Couper entre deux exemples de même classe ne peut jamais améliorer la pureté.
(c) $I(S)=1$ (3 O, 3 N).

- **27,5** : gauche $\{N,N\}$ pur ; droite $\{O,O,N,O\}$ : 0,811. Entropie moyenne $=\frac46\times0{,}811=0{,}541$ ; **gain $\approx0{,}459$**.
- **37,5** : gauche $\{N,N,O,O\}$ : 1 ; droite $\{N,O\}$ : 1 ; **gain $=0$**.
- **42,5** : gauche $\{N,N,O,O,N\}$ : 0,971 ; droite $\{O\}$ pur. Entropie moyenne $=\frac56\times0{,}971=0{,}809$ ; **gain $\approx0{,}191$**.

On retient le seuil **27,5** (Âge > 27,5).

---

**Q10.** (a) Quel biais présente le gain d'information ? Illustre avec un attribut « identifiant ». (b) Quelles sont les deux solutions du cours ? (c) Application : $S$ contient 8 exemples (4+, 4−). L'attribut **ID** (8 valeurs distinctes) a un gain de 1. L'attribut **C** donne $c_1$ : (4+,1−) et $c_2$ : (0+,3−), soit un gain de 0,549. Calcule le rapport de gain de chacun (dénominateur $=-\sum\frac{|S_i|}{|S|}\log_2\frac{|S_i|}{|S|}$) et conclus.

**Réponse.**
(a) Le gain **favorise indûment les attributs à beaucoup de valeurs**. Un attribut identifiant crée une branche d'un seul exemple par valeur, toutes parfaitement pures : son gain est **maximal** alors qu'il est **inutile** pour prédire un nouvel exemple.
(b) (1) **Binariser** tous les attributs (inconvénient : perte d'**interprétabilité**) ; (2) **normaliser** le gain par l'entropie de la partition elle-même (rapport de gain de C4.5), grande quand l'attribut éclate les données en beaucoup de petites branches, ce qui le pénalise.
(c) - **ID** : dénominateur $=\log_28=3$ ; rapport $=1/3\approx\mathbf{0{,}333}$.

- **C** : dénominateur $=-(\tfrac58\log_2\tfrac58+\tfrac38\log_2\tfrac38)\approx0{,}954$ ; rapport $=0{,}549/0{,}954\approx\mathbf{0{,}575}$.

Avant normalisation, ID gagne (1 contre 0,549) ; après, **C** passe devant (0,575 contre 0,333) : la correction fonctionne. (Dans le cours, elle fonctionne aussi mais de justesse : texture $\approx0{,}26$ contre identifiant $\approx0{,}24$.)

---

**Q11.** Quelles sont les trois façons de traiter une valeur manquante pour un attribut $A$ ? Application : dans le nœud, 9 exemples ont $A=$ rouge et 3 ont $A=$ bleu ; un exemple n'a pas de valeur. Que devient-il avec la troisième méthode, et comment prédit-on ensuite ?

**Réponse.**

1. Prendre la **valeur la plus fréquente** de $A$ dans $S$ ;
2. prendre la **valeur la plus fréquente dans le nœud** ;
3. **répartir** l'exemple en exemples fictifs, un par valeur, **pondérés par les fréquences**.

Application : « rouge » avec poids $9/12=\mathbf{0{,}75}$ et « bleu » avec poids $3/12=\mathbf{0{,}25}$. En prédiction, on donne l'étiquette de la **feuille la plus probable**.

---

## Partie D : Surapprentissage et élagage

**Q12.** (a) Qu'est-ce que le surapprentissage pour un arbre ? (b) Cite les quatre types de bruit du cours et définis un « clash ». (c) Que montre la courbe de la diapo 40 (Mitchell) ? (d) Comment estime-t-on la généralisation ?

**Réponse.**
(a) Un arbre **trop développé** (touffu, trop profond) qui crée des feuilles pour expliquer jusqu'au **bruit** de l'échantillon d'apprentissage ; il généralise mal.
(b) **Erreurs de description** (attribut mal mesuré) ; **erreurs de classification** (étiquette fausse) ; **clashes** (deux exemples de **même description mais de classes différentes**) ; **valeurs manquantes**.
(c) Quand le nombre de nœuds augmente, l'erreur sur l'**apprentissage diminue sans cesse**, mais l'erreur sur le **test remonte** après un certain point.
(d) Par un **ensemble test** séparé, la **validation croisée**, ou le **leave-one-out** (CV à $m$ plis) ; on peut aussi suivre la **courbe d'apprentissage**.

---

**Q13.** (a) Un arbre classe parfaitement les 17 exemples d'apprentissage : est-ce nécessairement bon signe ? (b) Un jeu d'apprentissage contient un « clash ». Peut-on obtenir une erreur d'apprentissage nulle ? Que prédit alors la feuille concernée ?

**Réponse.**
(a) **Non.** Une bonne performance sur l'apprentissage ne dit rien de la qualité sur de **nouvelles données** ; un arbre complet peut avoir mémorisé le bruit (« l'élève qui apprend par cœur les corrigés, fautes de frappe du prof comprises »).
(b) **Non.** Deux exemples de même description suivent le même chemin et atterrissent dans la **même feuille** : au mieux un seul est bien classé. La feuille prédit la **classe majoritaire** de ses exemples.

---

**Q14.** Définis pré-élagage et post-élagage. Donne l'avantage et l'inconvénient de chacun, et fais le lien avec le problème du XOR.

**Réponse.**

- **Pré-élagage** : on **arrête** la croissance **avant** que l'arbre soit complet (on ne scinde pas un nœud si cela n'améliore pas la validation). **Avantage** : rapide, on évite de construire des sous-arbres inutiles. **Inconvénient** : risque de **sous-apprentissage** : une scission inutile *maintenant* peut devenir utile *après* d'autres tests.
- **Post-élagage** : on laisse l'arbre **croître en entier**, puis on le **coupe**, en examinant les nœuds **du plus profond vers la racine**. **Avantage** : plus d'information disponible, donc moins de risque de sous-apprentissage, en général de meilleures performances. **Inconvénient** : **plus coûteux** (arbre complet à construire d'abord).

Lien avec le XOR : le pré-élagage souffre de la même myopie qu'un algorithme glouton : un test au gain nul (ou sans amélioration sur la validation) peut pourtant être nécessaire pour que le test suivant soit très informatif.

---

**Q15.** Pour le post-élagage, on dispose d'un ensemble de validation de 10 exemples. L'arbre complet en classe bien **6 sur 10**. Deux sous-arbres sont candidats :

- le sous-arbre **N** concerne 4 exemples de validation, dont il classe bien **1** ; remplacé par une feuille (classe majoritaire), il en classerait bien **3** ;
- le sous-arbre **M** concerne 3 exemples de validation, dont il classe bien **3** ; remplacé par une feuille, il n'en classerait bien que **1**.

Applique la règle de décision. Que fait-on pour N ? pour M ?

**Réponse.**
Règle : on remplace un sous-arbre par une feuille de classe majoritaire si la **précision de validation augmente** (ou ne baisse pas).

- **N** : la précision passe de $6/10$ à $6-1+3=8$, soit **$8/10$ = 80 %** > 60 % → on **élague** N.
- **M** : elle passerait de $6/10$ à $6-3+1=4$, soit **$4/10$ = 40 %** < 60 % → on **conserve** M.

---

**Q16.** (a) Pourquoi élague-t-on avec un ensemble de validation et pas avec l'ensemble d'apprentissage ? (b) Après avoir élagué avec cet ensemble de validation, peut-on utiliser la précision obtenue sur ce même ensemble comme estimation de la performance finale ? (c) Fais le lien entre élagage et régularisation (cours SVM).

**Réponse.**
(a) Sur l'apprentissage, un arbre plus développé fait **toujours aussi bien ou mieux** : on ne couperait jamais rien. La validation, **non vue pendant la construction**, donne une estimation honnête de la généralisation.
(b) ➕ **Non**, ou alors de façon trop optimiste : cet ensemble a servi à **prendre des décisions** (quoi couper), donc il n'est plus indépendant du modèle final. Pour estimer la performance finale, il faut un **ensemble test** indépendant (ou une validation croisée emboîtée), comme dans le cours d'évaluation.
(c) L'élagage est de la **régularisation** : on pénalise la complexité (taille de l'arbre) pour mieux généraliser, comme on contrôlait la complexité du modèle dans les SVM (compromis entre ajustement aux données et simplicité).

---

## Partie E : Biais, variance, arbres obliques

**Q17.** Définis biais et variance dans le cours. Que se passe-t-il pour un arbre profond et pour un arbre court ? Quel rôle joue l'élagage ? Comment l'analogie du tir à l'arc l'illustre-t-elle ?

**Réponse.**

- **Biais** : **incapacité du modèle à représenter le concept** (la « vraie » fonction entre $Y$ et $X$), indépendamment de la quantité de données.
- **Variance** : **sensibilité aux fluctuations d'échantillonnage** (un autre échantillon donnerait un autre arbre).

|               | Biais          | Variance        | Risque             |
| ------------- | -------------- | --------------- | ------------------ |
| Arbre profond | faible         | **forte** | surapprentissage   |
| Arbre court   | **fort** | faible          | sous-apprentissage |

L'**élagage** règle ce curseur (il cherche le bon compromis).
Tir à l'arc : biais = toutes les flèches loin du centre, groupées au même endroit ; variance = flèches éparpillées. Un bon modèle a peu des deux.

---

**Q18.** Dans le problème des deux triangles (diapos 49 à 52), la classe « rouge » correspond à l'intérieur d'un triangle. (a) Pourquoi un classifieur linéaire échoue-t-il, et de quelle composante de l'erreur s'agit-il ? (b) Pourquoi l'arbre appris sur ~100 points (5 feuilles) a-t-il une erreur de 0,1632 non nulle ? Quelles composantes sont en jeu ?

**Réponse.**
(a) Aucune droite ne sépare les rouges (dans le triangle) des bleus (autour). C'est un problème de **biais** : le modèle ne peut pas représenter le concept, **même avec une infinité de données**.
(b) L'arbre découpe l'espace en **rectangles** (coupures parallèles aux axes) et approche donc les frontières obliques du triangle par un « escalier » : c'est une part de **biais** (les rectangles ne peuvent représenter exactement une diagonale). Avec un **petit échantillon**, il ne trouve pas non plus exactement les « bonnes » frontières, et un autre échantillon donnerait un autre arbre : c'est la **variance**.

---

**Q19.** Qu'est-ce qu'un arbre oblique ? Quel défaut des arbres classiques corrige-t-il ? Donne l'exemple de la pastèque (densité, sucre) et les avantages et inconvénients.

**Réponse.**
Un nœud d'arbre oblique teste une **combinaison linéaire** d'attributs (ex. $1{,}1x_1+x_2<0{,}2$) : la frontière est un **hyperplan oblique** (comme un perceptron), au lieu d'une coupure alignée sur un axe.
Il corrige le défaut des arbres classiques dont les frontières sont **parallèles aux axes** : approcher une diagonale demande un « escalier » de nombreux tests.
Exemple pastèque : l'arbre classique utilise **4 tests** successifs (frontière en escalier), l'oblique seulement **2** (droites inclinées).
Avantages : arbre **plus petit**, frontières plus naturelles. Inconvénients : tests **moins lisibles** (combinaison de variables) et apprentissage **plus coûteux** (il faut apprendre les coefficients).

---

## Partie F : Bilan, XOR, régression

**Q20.** Cite trois avantages et deux inconvénients des arbres de décision. Pourquoi sont-ils « robustes aux données aberrantes » ?

**Réponse.**
**Avantages** (au choix) : **interprétables** (règles) ; **sélection automatique des variables** pertinentes ; traitement indifférencié des types de variables (nominales, ordinales, numériques) ; **robustes aux aberrants** ; solutions pour les **valeurs manquantes** ; **complexité faible**.
**Inconvénients** : **instabilité** sur les petites bases (feuilles de très petits effectifs, variance forte) ; **glouton et myope** (interactions comme le XOR).
Robustesse aux aberrants : un point extrême ne fait que **changer de côté d'une coupure** ; il n'influence pas les seuils comme il le ferait sur une moyenne.

---

**Q21.** Soit $y=x_1\oplus x_2$ avec les exemples $(0,0)\to0$, $(0,1)\to1$, $(1,0)\to1$, $(1,1)\to0$. Calcule le gain de $x_1$ et de $x_2$. Conclus sur le comportement d'un algorithme glouton. Que se passe-t-il si on le laisse tout de même croître ?

**Réponse.**
$I(S)=1$ (2 exemples de chaque classe).

- $x_1=0$ : $\{(0,0)\to0,(0,1)\to1\}$ → entropie 1 ; $x_1=1$ : $\{(1,0)\to1,(1,1)\to0\}$ → entropie 1. Entropie moyenne $=1$, donc **gain $=0$**.
- Même calcul pour $x_2$ : **gain $=0$**.

Aucun attribut n'est informatif **pris seul**, alors que les deux **ensemble** déterminent la classe. Un critère local ne voit aucune raison de préférer l'un à l'autre.
Si on force la croissance, un attribut est choisi arbitrairement et le **second niveau** (gain 1) trouve la solution sur ce jeu minuscule. Le vrai danger apparaît avec d'autres attributs un peu informatifs (qui seront préférés à tort) ou avec un **critère d'arrêt/pré-élagage** qui stoppe parce que le gain est nul.

---

**Q22.** Que prédit une feuille d'un arbre de régression ? Pourquoi la moyenne ? Quel critère minimise-t-on ? Complète le tableau de comparaison avec l'arbre de classification.

|                      | Classification | Régression |
| -------------------- | -------------- | ----------- |
| Cible                | ?              | ?           |
| Mesure d'impureté   | ?              | ?           |
| Valeur d'une feuille | ?              | ?           |
| Critère de choix    | ?              | ?           |

**Réponse.**
Une feuille prédit une **constante** : la **moyenne** des $y$ des exemples de sa région (ou un modèle linéaire dans les **arbres de modèles**). La moyenne est la constante qui **minimise l'erreur quadratique** sur la région. On minimise $MSE=\sum_i\sum_k\mathcal I(x_i\in R_k)(y_i-c_k)^2$.

|                      | Classification              | Régression                                         |
| -------------------- | --------------------------- | --------------------------------------------------- |
| Cible                | classe                      | nombre réel                                        |
| Mesure d'impureté   | entropie / Gini             | **erreur quadratique** (variance)             |
| Valeur d'une feuille | classe majoritaire          | **moyenne** des cibles (ou modèle linéaire) |
| Critère de choix    | maximiser le **gain** | maximiser la **baisse d'erreur**              |

Ce qui ne change pas : structure en arbre, stratégie gloutonne descendante, choix de la meilleure coupure, critères d'arrêt, élagage.

---

**Q23.** Points $(x;y)$ : $(1;4),(2;6),(3;5),(4;20),(5;22),(6;21)$. Calcule l'erreur quadratique avec une seule feuille, puis la réduction d'erreur pour la coupure $x\le3$ et pour la coupure $x\le2$. Quelle coupure retient-on et que prédit l'arbre pour $x=2{,}5$ ?

**Réponse.**

- **Une feuille** : moyenne $=78/6=13$ ; erreur $=81+49+64+49+81+64=\mathbf{388}$.
- **$x\le3$** : gauche $\{4,6,5\}$, moyenne 5, erreur $1+1+0=2$ ; droite $\{20,22,21\}$, moyenne 21, erreur $2$. Total $=4$ → **réduction $=384$**.
- **$x\le2$** : gauche $\{4,6\}$, moyenne 5, erreur 2 ; droite $\{5,20,22,21\}$, moyenne 17, erreur $144+9+25+16=194$. Total $=196$ → **réduction $=192$**.

On retient **$x\le3$**. Pour $x=2{,}5$ (région gauche), la prédiction est **5** (la moyenne de la feuille). (Si on pondère les erreurs par $|S_i|/|S|$ comme dans la formule de la diapo 66, le classement des coupures est le même.)

---

**Q24.** Dans quels cas les arbres de régression sont-ils pertinents ? Quelles sont leurs limites visibles sur la courbe approchée ?

**Réponse.**
Pertinents quand : la régression classique **ne marche pas** (par ex. entrée en **grande dimension**) ; l'**interprétabilité** compte ; le problème **se prête bien à une division selon les axes**.
Idée : au lieu d'un modèle global compliqué, on **partitionne l'espace** en régions et on utilise un **modèle local simple** dans chacune.
Limite : la courbe est approchée par des **marches d'escalier** (constantes par morceaux), sauf avec des feuilles linéaires (arbres de modèles) qui donnent de petits segments.

---

**Q25.** Quand les arbres sont-ils « bons » selon les conclusions du cours ? Quelle est leur complexité d'apprentissage et que représentent les termes ?

**Réponse.**
Bons quand l'**espace d'entrée est vectoriel** et que le concept cible correspond à des **boîtes récursives à frontières parallèles aux axes**. Ils sont **très simples**.
Complexité : $O(A^2\cdot m)$, avec $A$ le nombre d'**attributs** et $m$ le nombre d'**exemples**.

---

## Partie G : Synthèse et esprit critique

**Q26.** Pour chaque symptôme, donne la cause probable et le remède du cours.
(a) L'attribut « numéro de client » est choisi à la racine.
(b) Arbre à 100 % sur l'apprentissage, 62 % en test.
(c) La frontière réelle est une diagonale ; l'arbre est un escalier énorme.
(d) Deux attributs inutiles seuls mais parfaits ensemble ne sont jamais choisis.
(e) Avec 30 exemples, un petit changement de l'échantillon donne un arbre très différent.
(f) La cible est un nombre réel (prix).

**Réponse.**
(a) **Biais du gain** pour les attributs à beaucoup de valeurs → **rapport de gain** (C4.5) ou **binarisation**.
(b) **Surapprentissage** (bruit mémorisé) → **élagage** (pré ou post) validé sur des données non vues.
(c) Frontières parallèles aux axes → **arbre oblique** (combinaisons linéaires d'attributs).
(d) **Myopie du glouton** (type XOR) → c'est une faiblesse structurelle ; le pré-élagage l'aggrave, le post-élagage la limite.
(e) **Instabilité / variance forte** sur petites bases → simplifier (élaguer) ; reconnaître la limite des arbres.
(f) **Arbre de régression** (feuille = moyenne, critère = erreur quadratique).

---

**Q27.** (Rédaction.) Retrace la logique d'enchaînement du cours, de « un arbre est une suite de questions » jusqu'aux arbres de régression, en expliquant à chaque étape le problème rencontré et sa solution.

**Réponse.**

1. **Un arbre** = suite de tests (nœud = test, branche = valeur, feuille = classe) ; équivalent à une disjonction de règles → interprétable.
2. **Problème** : beaucoup d'arbres cohérents avec les données, et un espace de recherche gigantesque (Catalan) → il faut un **biais** (petits arbres) et une **recherche gloutonne descendante**.
3. **Problème** : comment choisir le meilleur test ? → mesurer l'**impureté** : **entropie** (gain d'information) ou **Gini**.
4. **Problèmes pratiques** : attributs continus (seuils entre classes différentes), biais du gain pour les attributs à nombreuses valeurs (rapport de gain), valeurs manquantes.
5. **Problème** : l'arbre complet colle au bruit → **surapprentissage** → **élagage** (pré/post) validé sur des données non vues.
6. **Lecture biais/variance** : arbre profond = biais faible, variance forte ; arbre court = l'inverse ; l'élagage règle le compromis. Limite des frontières parallèles aux axes → **arbres obliques**.
7. **Bilan** : interprétable, rapide, robuste, mais instable et myope (XOR).
8. **Extension** : mêmes idées pour prédire un nombre → **arbres de régression** (impureté = erreur quadratique, feuille = moyenne).

---

**Q28.** Compare arbre de décision et SVM (cours précédent) sur : type de frontière, interprétabilité, biais/variance, types de variables, robustesse aux aberrants, nature du modèle.

**Réponse.**

|                             | Arbre de décision                                                               | SVM                                                                                                     |
| --------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| Frontière                  | **règles locales parallèles aux axes** (récursives)                     | séparateur **linéaire global**, éventuellement dans un espace redécrit par un **noyau** |
| Interprétabilité          | **transparent** (règles lisibles)                                         | plutôt « boîte noire » puissante                                                                    |
| Biais / variance            | souple mais **instable** (variance forte si profond ou petit échantillon) | contrôle de la complexité par la marge/régularisation                                                |
| Types de variables          | **tous types** traités sans transformation                                | demande des attributs numériques et une métrique/noyau                                                |
| Aberrants                   | **robuste** (changement de côté d'une coupure)                           | sensible aux points proches de la frontière (vecteurs de support)                                      |
| Principe de régularisation | **élagage**                                                               | paramètre de régularisation / marge                                                                   |

*(Les lignes « Types de variables » et « Aberrants » du SVM sont des repères généraux ; à ajuster selon ce que ton cours SVM a insisté.)* Fil commun : régularisation pour mieux généraliser.

---

**Q29.** (Pont avec le cours 6.) Tu veux comparer un arbre élagué à un SVM : (a) sur 20 jeux de données ; (b) parmi 5 algorithmes (arbre, arbre oblique, SVM, Naïve Bayes, kNN) sur 25 jeux ; (c) sur un seul jeu avec une 10-CV. Quel test et pourquoi ? Que dire d'une conclusion du type « notre arbre est meilleur de 0,5 point » ?

**Réponse.**
(a) **Wilcoxon (rang signé)** : 2 algorithmes, plusieurs jeux, non paramétrique ($n=20<25$ : table).
(b) **Friedman** (global), puis **Nemenyi** si significatif ; surtout pas tous les tests deux à deux (comparaisons multiples).
(c) **t-test apparié** sur les différences par pli (avec la réserve de dépendance des plis → 5×2 cv), ou McNemar si un seul entraînement.
Une différence de 0,5 point peut venir du **hasard** : une précision est une estimation bruitée ; on teste $H_0$ (« pas de différence »). Et même significative, une différence peut être **négligeable en pratique** (significativité ≠ importance). ➕ Attention aussi à ne pas réutiliser le jeu de validation de l'élagage comme jeu de comparaison.

---

## Pièges fréquents à vérifier avant le partiel

| Piège                 | À retenir                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------ |
| Entropie vs effectifs  | Elle dépend des **proportions** ; 0 si pur, $\log_2k$ si $k$ classes équiprobables   |
| Choix de l'attribut    | On maximise le **gain** (ou la baisse de Gini), pas l'entropie                             |
| Pondération           | L'entropie moyenne est **pondérée par $\|S_v\|/\|S\|$**                                |
| Attributs continus     | Seuils seulement **entre exemples consécutifs de classes différentes**                   |
| Biais du gain          | Favorise les attributs à nombreuses valeurs ; rapport de gain = gain / entropie de la partition |
| Pré- vs post-élagage | Pré : risque de **sous-apprentissage** ; post : **coût** de l'arbre complet        |
| Élagage               | Se décide sur un ensemble **non vu** (validation), pas sur l'apprentissage                |
| Biais/variance         | Profond : biais **faible**, variance **forte** ; court : l'inverse                   |
| XOR                    | Gain nul pour chaque attribut isolé : le glouton est **myope**                            |
| Régression            | Feuille = **moyenne**, critère = **erreur quadratique**                             |

---

---

# Cours 8 : Les méthodes d'ensemble (boosting, bagging, gradient boosting)

## Questions type partiel : version corrigée

---

## Partie A : Pourquoi combiner ?

**Q1.** Décris l'expérience de Galton (1906). Quelles conditions sur les estimations individuelles font que la moyenne fonctionne ? Que se passe-t-il si l'une de ces conditions n'est pas respectée ?

**Réponse.**
Galton fait estimer à **787 participants** le poids d'un objet. **Le meilleur** participant se trompe de **plus d'un centième** ; **la moyenne** de tous se trompe de **moins d'un millième**. Une foule d'estimateurs individuellement médiocres bat le meilleur individu.

Conditions (diapo 8) : estimations **bruitées mais non biaisées** (pas d'erreur systématique), **symétriques** (autant d'erreurs par excès que par défaut), **indépendantes**.

Si une condition manque :

- **biais** : la moyenne conserve l'erreur systématique (tout le monde se trompe dans le même sens) ;
- **asymétrie** : les erreurs ne s'annulent pas ;
- **dépendance** : les erreurs se corrèlent et la moyenne n'est pas meilleure qu'un seul estimateur (voir Q2).

*Analogie : 787 personnes lancent une fléchette avec une main tremblante, sans viser systématiquement trop à gauche ; le centre de gravité des impacts tombe presque pile sur le centre.*

---

**Q2.** On moyenne $k=25$ estimateurs, chacun de variance $\sigma^2=4$. (a) Quelle est la variance de la moyenne s'ils sont indépendants ? (b) Et s'ils ont une corrélation $\rho=0{,}5$, sachant que la variance vaut alors $\rho\sigma^2+\frac{1-\rho}{k}\sigma^2$ ? (c) Quelle limite quand $k\to\infty$ ? Quelle conclusion pour la conception d'un ensemble ?

**Réponse.** (➕ formule marquée complément dans ta fiche)
(a) $\sigma^2/k=4/25=\mathbf{0{,}16}$ (écart-type 0,4 au lieu de 2).
(b) $0{,}5\times4+\dfrac{0{,}5}{25}\times4=2+0{,}08=\mathbf{2{,}08}$.
(c) La variance tend vers $\rho\sigma^2=\mathbf2$ : **on ne descend jamais sous $\rho\sigma^2$**, quel que soit le nombre d'estimateurs.
Conclusion : multiplier des modèles **corrélés** ne sert presque à rien ; il faut des apprenants faibles **le plus décorrélés possible**. C'est le fil rouge du cours : « comment engendrer des apprenants décorrélés ? » (diapo 30).

---

**Q3.** Le cours cite Dietterich (2000) pour justifier qu'un ensemble peut faire mieux qu'un seul modèle. Quelles sont les trois raisons classiques ?

**Réponse.** (➕ trois raisons de l'article, non détaillées sur la diapo 12)

1. **Statistique** : avec peu de données, plusieurs hypothèses sont également bonnes sur l'apprentissage ; en les moyennant, on réduit le risque de choisir la mauvaise.
2. **Calculatoire** : une recherche locale peut se coincer dans un mauvais optimum ; plusieurs recherches approximent mieux la vraie solution.
3. **Représentationnelle** : la vraie fonction n'est dans l'espace d'hypothèses d'aucun modèle individuel, mais une **combinaison** peut s'en approcher.

---

**Q4.** Le cours pose cinq questions qui structurent l'étude des méthodes d'ensemble (diapo 5). Énonce-les et donne la réponse du boosting à chacune.

**Réponse.**

| # | Question                                                       | Réponse du boosting                                                                  |
| - | -------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| 1 | **Quels agents** (quels apprenants) ?                    | Des apprenants **faibles** (arbres, souches…)                                  |
| 2 | **Que s'échangent-ils** si le processus est itératif ? | Des **poids sur les exemples** ($D_t$)                                        |
| 3 | **Comment combiner** leurs avis ?                        | **Vote pondéré**                                                              |
| 4 | **Quelles conditions de convergence ?**                  | Chaque apprenant doit faire **mieux que le hasard** ($\varepsilon_t<\frac12$) |
| 5 | **Si convergence, vers quoi ?**                          | Vers un classifieur de **grande marge / faible erreur**                         |

(Pour le bagging : les apprenants ne s'échangent rien, ils sont indépendants ; la combinaison est un vote simple.)

---

**Q5.** Cite deux succès applicatifs des méthodes d'ensemble (diapo 13). Que montrent-ils ?

**Réponse.**

- **KDD-Cup** : sur 2009-2011, tous les 1ers et 2es prix sont allés à des méthodes d'ensemble (le cours donne aussi une liste de tâches : détection d'intrusion, bioactivité, relation client, recommandation musicale…).
- **Netflix Prize** : 1 000 000 $ pour améliorer de plus de 10 % l'algorithme de recommandation ; **gagné en 2009 par une méthode d'ensemble**.

Ils montrent que les méthodes d'ensemble sont, en pratique, parmi les plus performantes sur des problèmes réels.

---

## Partie B : Le boosting : principe

**Q6.** Distingue apprentissage « fort » et apprentissage « $\gamma$-faible ». Que démontre Schapire (1990) ? Donne la définition du boosting.

**Réponse.**

- **Fort** (PAC) : pour **toute distribution** et **toute fonction cible**, on atteint une erreur **arbitrairement petite** avec une probabilité arbitrairement grande.
- **$\gamma$-faible** : on exige seulement une erreur **un peu meilleure que le hasard**, au plus $\frac12-\gamma$ avec $\gamma>0$.

Schapire (*The strength of weak learnability*) montre qu'ils sont **équivalents** : on peut **convertir** un apprenant faible en apprenant fort.
**Boosting** = méthode générale pour **convertir des règles de prédiction peu performantes en une règle (très) performante**, avec une garantie **prouvée** : si l'apprenant faible retourne toujours une hypothèse d'erreur $\le\frac12-\gamma$, on peut construire une hypothèse d'erreur $\le\varepsilon$ aussi petite qu'on veut.

---

**Q7.** Explique le principe du boosting avec l'analogie du parieur (diapo 34) : quelles sont les deux questions à résoudre et comment le boosting y répond-il ? Que reste-t-il à préciser dans la « version générale » (diapo 31) ?

**Réponse.**
Le parieur demande à un expert des règles empiriques, chacune faible. Deux questions :

1. **Comment choisir les courses à chaque étape ?** → se concentrer sur les plus **difficiles** (celles où les règles précédentes se trompent le plus) : c'est le choix de la **distribution $D_t$** sur les exemples.
2. **Comment combiner les règles ?** → par un **vote majoritaire pondéré**.

Version générale : à chaque étape on modifie l'échantillon en **diminuant** l'importance des exemples **bien classés** et en **augmentant** celle des **mal classés**. Reste la question : **de combien ?** C'est ce que règle AdaBoost.

---

**Q8.** Dans l'expérience en dimension 10 (diapos 16-18), on compare un arbre à un nœud, un arbre à 400 nœuds, et des arbres à un nœud boostés. Quel est le message ?

**Réponse.** (message classique de cette expérience, ➕ à vérifier sur ta diapo 17)
Un arbre à un nœud (**souche**, très faible) a une erreur élevée ; un arbre à 400 nœuds est meilleur mais limité ; **des souches combinées par boosting** descendent **bien plus bas**, alors que chaque brique reste minuscule.
Message : on peut atteindre une grande précision à partir de briques **extrêmement faibles**, à condition de les **combiner intelligemment**.

---

## Partie C : AdaBoost : l'algorithme

**Q9.** (a) Écris les formules d'AdaBoost : $\alpha_t$, la mise à jour des poids, le rôle de $Z_t$, l'hypothèse finale. (b) Calcule $\alpha_t$ pour $\varepsilon_t=0{,}1$ ; $0{,}4$ ; $0{,}5$ ; $0{,}6$ et interprète. (c) Pourquoi exige-t-on $\varepsilon_t<\frac12$ ? (d) Que faire si l'apprenant faible ne sait pas utiliser de poids ?

**Réponse.** (➕ formules standard, images dans tes diapos 38-39)
(a) $\alpha_t=\dfrac12\ln\dfrac{1-\varepsilon_t}{\varepsilon_t}$ ; $D_1(i)=\dfrac1m$ ; $D_{t+1}(i)=\dfrac{D_t(i)}{Z_t}\times\begin{cases}e^{-\alpha_t}&\text{bien classé}\\e^{+\alpha_t}&\text{mal classé}\end{cases}$ ; $Z_t$ = constante de **normalisation** (pour que $D_{t+1}$ somme à 1) ; $H(x)=\text{sign}\big(\sum_t\alpha_th_t(x)\big)$. L'erreur $\varepsilon_t$ est mesurée **sur la distribution $D_t$** (pondérée).
(b) - $\varepsilon=0{,}1$ : $\frac12\ln9=\ln3\approx\mathbf{1{,}10}$ (grande voix) ;

- $\varepsilon=0{,}4$ : $\frac12\ln1{,}5\approx\mathbf{0{,}20}$ ;
- $\varepsilon=0{,}5$ : $\mathbf0$ (au niveau du hasard : aucune voix) ;
- $\varepsilon=0{,}6$ : $\frac12\ln\frac{0{,}4}{0{,}6}\approx\mathbf{-0{,}20}$ (pire que le hasard : on inverse son avis).

Plus un classifieur est précis, plus son vote pèse.
(c) Sans cela le classifieur n'a pas de voix utile ($\alpha_t\le0$) et la combinaison ne progresse pas : c'est la **condition de convergence**.
(d) Le **rééchantillonner** : on tire les exemples avec probabilité proportionnelle à $D_t$.

---

**Q10.** Un jeu de 10 exemples ; le classifieur $h_1$ en classe mal 2. (a) Calcule $\varepsilon_1$, $\alpha_1$ et les facteurs $e^{\mp\alpha_1}$. (b) Calcule $Z_1$ et les nouveaux poids des exemples bien et mal classés. (c) Vérifie que les mal classés pèsent ensemble la moitié du poids total. Qu'en déduis-tu pour l'apprenant suivant ?

**Réponse.**
(a) $\varepsilon_1=2/10=0{,}2$ ; $\alpha_1=\frac12\ln\frac{0{,}8}{0{,}2}=\frac12\ln4=\ln2\approx\mathbf{0{,}693}$ ; $e^{-\alpha_1}=\mathbf{0{,}5}$ ; $e^{+\alpha_1}=\mathbf{2}$. Poids initial : $0{,}1$.
(b) $Z_1=8\times0{,}1\times0{,}5+2\times0{,}1\times2=0{,}4+0{,}4=\mathbf{0{,}8}$.

- bien classés : $0{,}1\times0{,}5/0{,}8=\mathbf{0{,}0625}$ (=1/16) ;
- mal classés : $0{,}1\times2/0{,}8=\mathbf{0{,}25}$ (=1/4).

(c) Total des mal classés : $2\times0{,}25=0{,}5$ ; total des bien classés : $8\times0{,}0625=0{,}5$. ✔
Sous $D_2$, $h_1$ est donc **exactement au niveau du hasard** (erreur 0,5), quelle que soit l'erreur initiale. L'apprenant suivant **ne peut pas recopier $h_1$** : il est forcé d'apporter une information nouvelle, ce qui **décorrèle** les apprenants.

---

**Q11.** Trois classifieurs ont pour poids $\alpha_1=1{,}0$, $\alpha_2=0{,}3$, $\alpha_3=0{,}4$. Pour un exemple $x$, ils votent $h_1(x)=+1$, $h_2(x)=-1$, $h_3(x)=-1$. (a) Quelle est la prédiction de $H$ ? (b) Que donnerait un vote majoritaire simple ? (c) Si la vraie classe est $+1$, quelle est la marge normalisée de $x$ ? Que signifie-t-elle ?

**Réponse.**
(a) $\sum\alpha_th_t=1{,}0-0{,}3-0{,}4=+0{,}3$ → $H(x)=\mathbf{+1}$.
(b) Majorité simple : 2 votes « −1 » contre 1 « +1 » → **−1**. Le vote **pondéré** donne un avis différent parce que $h_1$, précis, a une grosse voix.
(c) Marge $=\dfrac{\text{poids des votes corrects}-\text{poids des votes erronés}}{\sum\alpha_t}=\dfrac{1{,}0-0{,}7}{1{,}7}\approx\mathbf{0{,}18}$. Positive (bien classé) mais **proche de 0** : vote serré, prédiction peu confiante. Une marge proche de $+1$ correspondrait à une prédiction très confiante ; une marge négative à un exemple mal classé.

---

## Partie D : Pourquoi le boosting marche

**Q12.** (a) Que dit le premier résultat théorique sur l'erreur d'apprentissage de $H$ ? (b) Application : $\gamma=0{,}1$ pour toutes les étapes ; quelle borne pour $T=200$ ? (c) Avec $m=1000$ exemples d'apprentissage, à partir de quelle valeur de $T$ la borne garantit-elle une erreur d'apprentissage nulle ?

**Réponse.** (➕ formule standard)
(a) L'erreur d'apprentissage **décroît exponentiellement** avec le nombre d'étapes :
$\text{err}_S(H)\le\prod_tZ_t=\prod_t2\sqrt{\varepsilon_t(1-\varepsilon_t)}\le\exp\big(-2\sum_t\gamma_t^2\big)$, avec $\gamma_t=\frac12-\varepsilon_t$. Tant que chaque apprenant bat le hasard d'une marge $\gamma>0$, l'erreur tend vers 0.
(b) $e^{-2\gamma^2T}=e^{-2\times0{,}01\times200}=e^{-4}\approx\mathbf{0{,}018}$ (moins de 2 %).
(c) Une erreur d'apprentissage est un multiple de $1/m$ ; elle est donc nulle dès que la borne est $<1/1000$ : $e^{-0{,}02T}<10^{-3}$ ⟺ $T>\dfrac{\ln1000}{0{,}02}\approx345{,}4$, soit **$T\ge346$**.

---

**Q13.** (a) Quelle est « l'énigme de l'erreur de test » ? Pourquoi est-elle contraire à l'intuition (lien avec le cours sur les arbres) ? (b) Définis la marge d'un exemple. (c) Comment la marge explique-t-elle le phénomène ? (d) Quelle nuance le cours apporte-t-il ?

**Réponse.**
(a) En bouclant sur des centaines d'étapes, l'erreur d'apprentissage atteint **zéro**, et pourtant l'erreur de test **continue de baisser ou ne remonte pas** (C4.5 boosté sur le jeu *letter* : pas de remontée même après 1000 étapes, soit $2\cdot10^6$ nœuds de test). C'est contraire à l'intuition : dans le cours sur les arbres, plus de nœuds → surapprentissage → l'erreur de test remonte. C'est « *the unreasonable power of boosting* ».
(b) La **marge** = poids des classifieurs ayant voté correctement − poids de ceux en erreur (normalisée entre −1 et +1) : c'est la **confiance** de la prédiction. L'erreur d'apprentissage ne regarde que *si* la prédiction est juste.
(c) À chaque étape, AdaBoost met le plus de poids sur les exemples de **plus faible marge** tout en continuant d'améliorer celle des autres : même si l'hypothèse finale est complexe, elle est proche d'une hypothèse simple d'erreur d'apprentissage proche, donc de bonne généralisation. Les bornes de généralisation sont exprimées **en fonction de la marge**, pas du nombre de classifieurs. *Analogie électorale : à 80 % des voix, un petit sondage suffit à prédire le résultat ; à 51 %, il en faut énormément ; plus la marge est grande, plus la décision est robuste.*
(d) L'explication par la marge est **belle mais pas parfaite** : les diapos « Mais… » (79-81) indiquent des questions ouvertes. On ne la présente pas comme une preuve définitive.

---

**Q14.** Fais le parallèle entre boosting et SVM sur deux points : (a) le rôle de la marge, (b) l'espace de redescription. Quelle différence principale ?

**Réponse.**
(a) Même philosophie : dans les SVM, on **maximise une marge** pour contrôler la capacité ; dans le boosting, on **augmente la marge** de la combinaison. Dans les deux cas, **ce n'est pas le nombre de paramètres qui mesure la complexité, mais la marge**.
(b) Chaque $h_t$ est une **nouvelle caractéristique** (une coordonnée) ; $\sum\alpha_th_t(x)$ est un **séparateur linéaire dans un espace de redescription**, comme avec les noyaux.
Différence : pour le boosting, l'espace est **construit pas à pas à partir des données** ; pour les SVM, il est **défini a priori par le noyau**.

---

**Q15.** Quels sont les autres points de vue sur le boosting (diapo 82) ? Explique celui de la perte exponentielle et sa conséquence pratique.

**Réponse.**

- **Théorie des jeux** : l'inspiration originale (un jeu entre l'apprenant et un adversaire qui choisit les exemples difficiles).
- **Minimisation d'une perte surrogée** : en minimisant étape par étape la perte exponentielle $\ell(y,H(x))=e^{-yH(x)}$, on retrouve **exactement** les formules d'AdaBoost (choix de $\alpha_t$ et mise à jour des poids).
- **Géométrie de l'information**.

Conséquence pratique : la perte exponentielle **explose** pour les grosses erreurs, ce qui donne un poids exorbitant aux aberrants (d'où la sensibilité au bruit et les variantes comme LogitBoost, Q18).

---

## Partie E : Bilan, limites, variantes, application

**Q16.** Cite au moins quatre avantages et quatre difficultés d'AdaBoost (diapos 85 et 87).

**Réponse.**
**Avantages** : **méta-algorithme** (n'importe quel apprenant faible) ; **(très) rapide** et facile à programmer ; **un seul paramètre** ($T$) ; **performances théoriques garanties** ; pas de sur-spécialisation grâce à la maximisation des marges ; adaptable à $h_t:X\to\mathbb R$ (signe = classe, $|h_t(x)|$ = confiance), au multi-classes et multi-étiquettes ; permet de **détecter les aberrants** (ceux dont le poids ne cesse de grimper).
**Difficultés** : difficile d'incorporer des **connaissances a priori** ; difficile de savoir **comment régulariser** ; le **meilleur choix d'apprenant faible** n'est pas évident ; avec des arbres alignés sur les axes, les **frontières sont très irrégulières** (non interprétables).

---

**Q17.** Dans quels cas le boosting ne marche-t-il pas bien (diapo 86) ? Explique en particulier pourquoi les données bruitées lui posent problème. Quel théorème rappelle-t-on ?

**Réponse.**
Le **no free lunch** : aucun algorithme n'est meilleur partout. Le boosting est inadapté quand :

- **pas assez de données** ;
- **comité trop restreint** (apprenants trop peu variés) ;
- **apprenants trop stables** (même hypothèse malgré les poids) ;
- **apprenants trop forts** (ils surapprennent déjà seuls) ;
- **données bruitées**.

Bruit : en sur-pondérant les exemples difficiles, AdaBoost **s'acharne sur les exemples mal étiquetés** (les erreurs d'étiquetage deviennent la cible principale) ; la perte exponentielle donne un poids exorbitant aux aberrants. (Retourné en atout : les poids qui grimpent signalent justement les aberrants.)

---

**Q18.** Qu'est-ce que LogitBoost ? Quels avantages par rapport à AdaBoost ?

**Réponse.**
Variante qui remplace la perte exponentielle par la **perte logistique (logloss)**. Avantages : (1) **pas de poids exorbitant** sur les aberrants, la perte logistique croissant seulement **linéairement** pour de grosses erreurs là où l'exponentielle explose ; (2) possibilité de calculer des **probabilités** associées aux étiquettes.

---

**Q19.** Détection de visages de Viola et Jones : quels étaient les défis ? Quel est le rôle d'AdaBoost ? Pourquoi une cascade ?

**Réponse.**
Défis : identifier des **descripteurs** pertinents, les **calculer rapidement**, les **organiser** en un système avec peu de **faux négatifs** (visages ratés) et peu de **faux positifs** (fausses alarmes), en **temps réel** (15 images/s sur un smartphone).
Il y a plus de **3 milliards** de descripteurs possibles. Chacun est associé à une **souche de décision** ; **AdaBoost sélectionne** une séquence d'environ **200** descripteurs utiles.
Mais évaluer ces 200 sur chaque fenêtre est trop coûteux (≈ 0,7 s par image) : on les organise en **cascade**. Les premiers étages, très simples et rapides, **éliminent le plus tôt possible les négatifs** (l'immense majorité des fenêtres) ; seules les fenêtres survivantes passent aux étages suivants plus coûteux. *(Comme un contrôle de sécurité en plusieurs étapes : un coup d'œil rapide écarte 90 % des gens.)*

---

## Partie F : Gradient boosting

**Q20.** Compare gradient boosting et AdaBoost sur la combinaison, ce que le suivant apprend, et la fonction de perte. Quelles sont les forces et faiblesses pratiques du gradient boosting ?

**Réponse.**

|                           | AdaBoost                                                    | Gradient boosting                                                        |
| ------------------------- | ----------------------------------------------------------- | ------------------------------------------------------------------------ |
| Combinaison               | vote **pondéré**                                    | **somme non pondérée** (le pas est dans le taux d'apprentissage) |
| Ce que le suivant apprend | les mêmes cibles, avec des **exemples repondérés** | les **résidus** (écart entre valeur désirée et prédiction)    |
| Perte                     | **exponentielle**                                     | **toute fonction différentiable**                                 |

Forces : **gère mieux** les données complexes (images, texte) ou **bruitées**. Faiblesses : **réglage fin de plusieurs hyperparamètres** nécessaire et **plus susceptible de surapprentissage**.

---

**Q21.** Version plus petite que l'exemple du cours : 4 exemples, $y=(1,1,1,0)$. (a) Donne la prédiction initiale constante $h_0$ et les résidus. (b) Un arbre a deux feuilles : $\{1,2\}$ et $\{3,4\}$ ; donne la prédiction de chaque feuille. (c) Avec $\eta=0{,}5$, calcule $h_1$ et les nouveaux résidus. (d) Compare l'erreur quadratique avant et après.

**Réponse.**
(a) $h_0=3/4=\mathbf{0{,}75}$ ; résidus $y-h_0$ : $+0{,}25\,;\,+0{,}25\,;\,+0{,}25\,;\,-0{,}75$.
(b) Chaque feuille prédit la **moyenne des résidus** de ses exemples : feuille $\{1,2\}$ : $\frac{0{,}25+0{,}25}2=\mathbf{0{,}25}$ ; feuille $\{3,4\}$ : $\frac{0{,}25-0{,}75}2=\mathbf{-0{,}25}$.
(c) $h_1=h_0+\eta f$ : exemples 1 et 2 : $0{,}75+0{,}5\times0{,}25=\mathbf{0{,}875}$ ; exemples 3 et 4 : $0{,}75+0{,}5\times(-0{,}25)=\mathbf{0{,}625}$.
Nouveaux résidus : $+0{,}125\,;\,+0{,}125\,;\,+0{,}375\,;\,-0{,}625$.
(d) Avant : $3\times0{,}0625+0{,}5625=\mathbf{0{,}75}$. Après : $2\times0{,}0156+0{,}1406+0{,}3906=\mathbf{0{,}5625}$. L'erreur **diminue** : chaque arbre est un « coup de correction ». Le mécanisme est celui de la fiche : résidus → arbre → correction pondérée par $\eta$ → nouveaux résidus plus petits. (Le résidu de l'exemple 4 reste grand : d'autres arbres suivront.)

---

**Q22.** (a) Écris la mise à jour du gradient boosting et le rôle du taux d'apprentissage $\eta$. (b) Pourquoi parle-t-on de « descente de gradient dans l'espace des fonctions » ? (c) Cite les quatre hyperparamètres du cours et l'effet de chacun. (d) Quel lien avec le bagging ?

**Réponse.**
(a) $h_{t+1}(x)=h_t(x)+\eta f_t(x)$, où $f_t$ est entraîné à prédire le résidu $r_i=y_i-h_t(x_i)$. $\eta$ règle l'**amplitude de la correction** (golfeur : petits coups prudents ou grands coups risqués).
(b) Pour la perte quadratique, le résidu est (à un facteur près) l'**opposé du gradient** de la perte par rapport à la prédiction : chaque arbre fait un pas dans la direction de la descente.
(c) 1. **Taux d'apprentissage** : petit = lent mais plus sûr (demande plus d'arbres) ; 2. **nombre d'arbres** ; 3. **profondeur des arbres** : peu profonds = apprenants faibles qui captent des interactions limitées ; 4. **taux de sous-échantillonnage** : fraction des données par arbre, introduit du hasard et **réduit la variance**.
(d) Le sous-échantillonnage est une idée **empruntée au bagging**.

---

## Partie G : Le bagging

**Q23.** (a) Décris les trois étapes du bagging et ce que signifie son nom. (b) Quelle est la probabilité qu'un exemple donné ne soit pas tiré dans un échantillon bootstrap de taille $m=5$ ? de taille $m=10$ ? quand $m\to\infty$ ? (c) Quelle proportion d'exemples distincts contient un échantillon, et à quoi servent les autres ?

**Réponse.**
(a) **Bootstrap AGGregatING** (Breiman, 1996) : 1. générer $k$ échantillons par **tirage avec remise** dans $S_m$ ; 2. entraîner un classifieur sur chacun avec **le même algorithme** ; 3. **vote simple** (ou moyenne en régression).
(b) $(1-\frac1m)^m$ : $m=5$ : $0{,}8^5\approx\mathbf{0{,}328}$ ; $m=10$ : $0{,}9^{10}\approx\mathbf{0{,}349}$ ; $m\to\infty$ : $e^{-1}\approx\mathbf{0{,}368}$.
(c) Environ **63 %** d'exemples distincts (avec des doublons) ; les ~37 % restants sont « **hors sac** » : ➕ on peut les utiliser pour estimer l'erreur « gratuitement ».

---

**Q24.** Que réduit principalement le bagging : le biais ou la variance ? Explique pourquoi. Quel type d'apprenant lui convient ? Que dit le cours sur la compréhension de ses mécanismes ?

**Réponse.**
Le bagging **réduit la variance en laissant le biais inchangé**. En moyennant des modèles entraînés sur des échantillons différents, on lisse leurs fluctuations ; mais la moyenne de modèles qui font tous la même erreur systématique conserve cette erreur.
Il convient à des apprenants **instables et à faible biais** : typiquement des **arbres profonds** (biais faible, variance forte, cf. cours sur les arbres). Avec des **souches** (biais fort, très stables), toutes les souches se ressemblent et le bagging n'apporte presque rien : le bagging a besoin d'apprenants **instables**.
Le cours précise que ce n'est **pas encore complètement compris** (Grandvalet 2004 : le bagging **égalise l'influence** des exemples).

---

**Q25.** Complète le tableau de comparaison bagging / boosting.

|                              | Bagging | Boosting |
| ---------------------------- | ------- | -------- |
| Entraînement                | ?       | ?        |
| Données de chaque apprenant | ?       | ?        |
| Vote                         | ?       | ?        |
| Cible principale             | ?       | ?        |
| Apprenant idéal             | ?       | ?        |
| Sensibilité au bruit        | ?       | ?        |

**Réponse.**

|                              | Bagging                                            | Boosting                                                        |
| ---------------------------- | -------------------------------------------------- | --------------------------------------------------------------- |
| Entraînement                | **parallèle**, indépendant                 | **séquentiel**, dépendant                               |
| Données de chaque apprenant | **rééchantillonnage** uniforme avec remise | **repondération** (ou rééchantillonnage selon $D_t$) |
| Vote                         | **simple**                                   | **pondéré** par la qualité                             |
| Cible principale             | **variance**                                 | **biais** (et marge)                                      |
| Apprenant idéal             | **instable et fort** (arbres profonds)       | **faible** (souches)                                      |
| Sensibilité au bruit        | faible                                             | **forte**                                                 |

---

**Q26.** Tu disposes (i) d'arbres de décision profonds, (ii) de souches de décision. Quelle méthode d'ensemble utilises-tu pour chacun et pourquoi ? Que se passe-t-il si tu inverses ?

**Réponse.**
(i) **Arbres profonds** : biais faible, variance forte, instables → **bagging** (la moyenne réduit la variance sans toucher au biais déjà faible).
(ii) **Souches** : biais fort, très stables → **boosting** (il réduit le biais en construisant, par repondération, une combinaison riche à partir de briques très faibles).
Si on inverse : le **bagging de souches** n'aide presque pas (toutes se ressemblent) ; le **boosting d'apprenants trop forts** est peu utile (ils surapprennent déjà seuls ; erreur d'apprentissage nulle dès la première étape, il n'y a plus rien à repondérer).

---

## Partie H : Synthèse et ponts avec les autres cours

**Q27.** Pour chaque symptôme, donne la cause probable et le remède du cours.
(a) AdaBoost dégrade ses performances quand 30 % des étiquettes sont fausses.
(b) Un comité de 100 arbres profonds entraînés sur le même jeu ne gagne presque rien sur un seul arbre.
(c) L'erreur d'apprentissage d'AdaBoost est nulle mais l'erreur de test continue de baisser.
(d) Avec seulement 40 exemples, le boosting ne donne rien de mieux.
(e) Un gradient boosting surapprend.
(f) Tu boostes un apprenant très puissant : erreur d'apprentissage nulle dès le premier tour.

**Réponse.**
(a) **Bruit** : AdaBoost s'acharne sur les exemples mal étiquetés (perte exponentielle) → **LogitBoost** (perte logistique), ou utiliser les poids élevés pour **repérer les aberrants** ; le bagging est peu sensible au bruit.
(b) Arbres **corrélés** (même jeu, même algorithme) : la variance reste $\ge\rho\sigma^2$ → **décorréler** (rééchantillonnage bootstrap du bagging ; ➕ sous-ensemble d'attributs : forêts aléatoires, hors fichier).
(c) Pas de surapprentissage : le boosting **continue d'augmenter la marge** (confiance), ce qui améliore la généralisation.
(d) **Pas assez de données** : un des cas où le boosting est inadapté (no free lunch).
(e) Réduire $\eta$ (et augmenter le nombre d'arbres), utiliser des **arbres moins profonds**, **sous-échantillonner** les données, limiter le nombre d'arbres.
(f) **Apprenant trop fort** : plus rien à repondérer → prendre des apprenants faibles (souches, arbres peu profonds).

---

**Q28.** (Rédaction.) Retrace la logique d'enchaînement du cours : de la sagesse des foules au gradient boosting, avec à chaque étape le problème et la solution.

**Réponse.**

1. **Sagesse des foules** : des estimateurs bruités mais non biaisés, symétriques et **indépendants**, une moyenne les annule ; la variance baisse comme $\sigma^2/k$ mais **pas sous $\rho\sigma^2$ s'ils sont corrélés** → il faut des apprenants **décorrélés**.
2. **Deux questions** : comment engendrer des apprenants faibles décorrélés ? comment combiner leurs prédictions ?
3. **Bagging** : on décorrèle en **rééchantillonnant** (bootstrap), vote simple ; réduit la **variance**, convient aux apprenants instables.
4. **Boosting** : on décorrèle en **repondérant** les exemples, séquentiellement ; vote pondéré ; part de la question théorique « faible = fort ? » (Schapire : équivalents).
5. **AdaBoost** : donne le « de combien » ($\alpha_t$, mise à jour des poids) ; après la mise à jour, $h_t$ est au niveau du hasard sous $D_{t+1}$.
6. **Pourquoi ça marche** : erreur d'apprentissage qui décroît exponentiellement ; énigme de l'erreur de test → explication par la **marge** (parallèle avec les SVM), avec ses limites.
7. **Limites et variantes** : bruit, peu de données, apprenants trop forts ou stables → **LogitBoost** ; application : Viola-Jones (sélection de descripteurs + cascade).
8. **Gradient boosting** : au lieu de repondérer, chaque arbre apprend les **résidus** ; perte différentiable quelconque ; plus puissant mais plus de réglages et plus de risque de surapprentissage.

---

**Q29.** (Pont avec le cours 7.) Pourquoi les arbres de décision sont-ils l'apprenant de prédilection des méthodes d'ensemble ? Quelle est la contrepartie ?

**Réponse.**
Ils sont **adaptables aux deux familles** : un arbre profond a un biais faible et une variance forte (**instable**) → idéal pour le **bagging** ; une **souche** ou un arbre peu profond est un apprenant **faible** → idéal pour le **boosting** (et pour le gradient boosting, avec la profondeur comme hyperparamètre). Ils sont aussi **rapides** et traitent tous types de variables.
Contrepartie : on perd l'**interprétabilité** qui faisait leur force (un comité d'arbres n'est plus lisible en règles) ; avec des arbres alignés sur les axes, les **frontières sont très irrégulières** (diapo 87).

---

**Q30.** (Pont avec le cours 6.) Tu compares un arbre, un bagging d'arbres, AdaBoost, un gradient boosting et un SVM. (a) Sur 30 jeux de données, quel test global et post hoc ? (b) Pour comparer seulement AdaBoost et le SVM sur 30 jeux ? (c) Peut-on comparer sur le jeu d'apprentissage ? (d) Si AdaBoost est « significativement mieux classé » ?

**Réponse.**
(a) **Friedman** (global), puis **Nemenyi** (post hoc) si Friedman est significatif ; surtout pas tous les tests deux à deux (comparaisons multiples).
(b) **Wilcoxon (rang signé)** ; avec $n=30\ge25$, approximation normale (statistique $z$).
(c) **Non** : l'erreur d'apprentissage d'AdaBoost tend vers 0, elle n'est pas informative sur la généralisation. Il faut un ensemble test ou une validation croisée.
(d) Les rangs indiquent un **ordre**, pas l'ampleur : il faut aussi regarder les **valeurs de performance** (significativité statistique ≠ importance pratique).

---

## Partie facultative (hors fichier fourni)

**Q31.** (Si traité dans ton cours.) Quelle astuce distingue une forêt aléatoire d'un simple bagging d'arbres, et pourquoi peut-elle être meilleure ?

**Réponse.** (➕ complément de ta fiche, section 12)
À **chaque nœud**, on ne considère qu'un **sous-ensemble aléatoire d'attributs** pour choisir la coupure : les arbres sont **moins corrélés**, donc la variance de la moyenne baisse davantage (cf. $\rho\sigma^2+\frac{1-\rho}k\sigma^2$ : on diminue $\rho$). Vote majoritaire final.

**Q32.** (Si traité dans ton cours.) À quoi correspondent XGBoost et le stacking dans la famille des méthodes d'ensemble ?

**Réponse.** (➕ complément de ta fiche, section 12)
**XGBoost** : implémentation très optimisée du **gradient boosting d'arbres**, avec **régularisation** explicite de la complexité des arbres. **Stacking** : un **méta-modèle** apprend à combiner les prédictions d'autres modèles.

---

## Pièges fréquents à vérifier avant le partiel

| Piège                                    | À retenir                                                                                                            |
| ----------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| « Plus d'estimateurs = toujours mieux » | Faux s'ils sont **corrélés** : plancher $\rho\sigma^2$                                                      |
| Boosting ≠ bagging                       | Boosting : séquentiel, repondération, vote pondéré, biais. Bagging : parallèle, bootstrap, vote simple, variance |
| Poids $\alpha_t$                        | $\frac12\ln\frac{1-\varepsilon_t}{\varepsilon_t}$ : nul à $\varepsilon=0{,}5$, négatif au-delà                 |
| Mise à jour                              | Bien classé $\times e^{-\alpha_t}$ (↓), mal classé $\times e^{+\alpha_t}$ (↑), puis divisé par $Z_t$       |
| Après un tour                            | Les mal classés pèsent **0,5** du total : $h_t$ est au niveau du hasard sous $D_{t+1}$                    |
| Erreur d'apprentissage nulle              | Ne signifie pas surapprentissage : regarder la **marge**                                                        |
| Marge                                     | Poids des votes corrects − poids des votes erronés (normalisée dans $[-1,+1]$)                                   |
| Bruit                                     | AdaBoost y est **sensible** (perte exponentielle) ; LogitBoost atténue                                         |
| Gradient boosting                         | Résidus, somme **non pondérée**, $\eta$ ; plus de réglages, plus de risque de surapprentissage            |
| Bootstrap                                 | $\approx63\,\%$ d'exemples distincts, $\approx37\,\%$ hors sac                                                    |
| Bagging et apprenants                     | Instables et forts (arbres profonds) ; presque inutile sur des souches                                                |

**Rappel des incohérences signalées dans ta fiche** : diapo 105 (« $\log\frac32=0{,}176$ » est un log décimal ; le népérien vaut $\ln1{,}5\approx0{,}405$) ; diapo 106 (la moyenne de $-0{,}6$ et $+0{,}4$ vaut $-0{,}1$) ; diapo 107 (valeurs de $h_1$ non reconstituées : regarde la diapo). Si ces chiffres tombent à l'examen, retiens le **mécanisme** et vérifie avec ton enseignant.

---

---
