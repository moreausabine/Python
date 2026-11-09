# Questions type partiel corrigées : introduction au ML

---

## Partie A : Trois grandes catégories de tâches

### Q1. Citer les trois grandes catégories de tâches et donner un exemple pour chacune.

1. **Reconnaissance / prédiction** : associer une étiquette ou une valeur à une donnée. Exemples : reconnaître un objet (base des CAPTCHA), identifier des molécules bio-actives contre le cancer, prédire une série temporelle.
2. **IA générative** : produire de nouveaux objets. Exemples : AlphaFold (structure 3D d'une protéine à partir de sa séquence), ITHACA (compléter des textes anciens lacunaires).
3. **IA agentique** : agents partiellement autonomes qui agissent sur leur environnement. Exemples : aide au code / débogage, découverte scientifique. Enjeu actuel : apprendre à *raisonner*.

### Q2. Dans quelle catégorie classer AlphaFold ? ITHACA ? Justifier.

Les deux relèvent de l'**IA générative** : AlphaFold *produit* une structure 3D à partir d'une séquence, ITHACA *produit* le texte manquant d'une inscription lacunaire. Dans les deux cas la sortie est un nouvel objet structuré, pas une simple étiquette.

*Nuance acceptable :* AlphaFold peut aussi être vu comme une prédiction (séquence → structure). La catégorie dépend de ce qu'on met en avant : la sortie est un objet complexe produit par le modèle.

### Q3. Qu'est-ce qui distingue l'IA agentique de la reconnaissance/prédiction ?

- La **reconnaissance/prédiction** est une fonction statique : une donnée en entrée, une étiquette ou une valeur en sortie. Le modèle ne modifie pas le monde.
- L'**IA agentique** met en jeu un agent **partiellement autonome qui agit sur son environnement** : il enchaîne des actions, observe leurs effets, et doit planifier / raisonner sur plusieurs étapes.

---

## Partie B : Les grands types d'apprentissage

### Q4. Définir apprentissage supervisé, non supervisé et par renforcement. Préciser pour chacun le type de signal dont dispose l'algorithme.

| Type                       | Principe                                                                          | Signal disponible                                        |
| -------------------------- | --------------------------------------------------------------------------------- | -------------------------------------------------------- |
| **Supervisé**       | Apprendre une fonction $f : X \to Y$ à partir d'exemples $(x_i, y_i)$        | Étiquettes $y_i$ fournies                             |
| **Non supervisé**   | Découvrir une structure dans les données (groupes, dimensions utiles, densité) | Aucune étiquette, seulement les $x_i$                 |
| **Par renforcement** | Un agent apprend une stratégie en interagissant avec un environnement            | Récompenses (ou pénalités) reçues après ses actions |

### Q6. Différence entre classification et régression ?

Les deux sont de l'apprentissage supervisé. En **classification**, $Y$ est un ensemble fini de classes (spam / non spam). En **régression**, $Y$ est une valeur numérique continue (un prix, une température). Les pertes usuelles diffèrent : perte 0-1 ou entropie croisée pour la classification, perte quadratique (MSE) pour la régression.

---

## Partie C : Le biais : pourquoi, et quels types

### Q7. C'est quoi un biais (différents types)

**Un biais** est ce qui restreint l'univers des hypothèses envisagées, c'est-à-dire ce qui fait préférer certaines hypothèses à d'autres. Un biais est **nécessaire** : sans lui, aucune généralisation n'est possible

| Type                                       | Principe                                      | Signal disponible                                                                                                                                                                                    |
| ------------------------------------------ | --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **De représentation** (déclaratif) | Restreint l'espace $\mathcal H$             | On choisit un sous-ensemble de l'univers des hypothèses, en admettant que les hypothèses exclues sont impossibles ou très improbables                                                             |
| **De recherche** (procédural)       | Restreint l'*exploration* de $\mathcal H$ | On cherche (ex. par une méthode de type gradient) une hypothèse de performance optimale dans l'espace autorisé — sans garantie d'unicité de la solution trouvée                                |
| **De cursus**                        | Restreint l'*ordre* de présentation        | Biais dû à l'ordre dans lequel les exemples sont vus. Analogie scolaire : apprendre d'abord l'addition puis la soustraction façonne durablement notre façon de penser les mathématiques ensuite |

### Q8. Les trois ingrédients de l'apprentissage

* **L'espace des hypothèses $\mathcal H$** — quelle famille de fonctions est admise ?
* **Le critère inductif** — comment évaluer chaque $h$ à partir de l'échantillon $S$ ? (§10)
* **La méthode d'exploration de $\mathcal H$** — comment trouver une bonne hypothèse ?

---

## Partie D : Induction, biais inductif, NFL

### Q9. Expliquer le problème de l'induction. Pourquoi évalue-t-on un modèle sur un individu jamais observé ?

**Induire** c'est passer du particulier au général : à partir d'un nombre fini d'exemples, on veut une règle valable pour des cas non vus. Aucune quantité finie d'exemples ne prouve logiquement une règle générale (problème de Hume : le 1001ᵉ cygne peut être noir).

En ML supervisé, on dispose d'un jeu fini $\{(x_i,y_i)\}$ mais on veut que $f$ marche sur de nouveaux $x$. Tester sur un individu jamais observé mesure donc la **robustesse / généralisation** du modèle, ce qui est le vrai objectif.

### Q10. (Calcul) On a 4 variables binaires en entrée et une sortie binaire.

**(a)** Combien d'entrées possibles ? **(b)** Combien de fonctions possibles ? **(c)** On observe 10 entrées avec leurs sorties : combien de fonctions restent compatibles ? **(d)** Que conclure ?

- (a) $2^4 = 16$ entrées.
- (b) $2^{16} = 65\,536$ fonctions.
- (c) Il reste $6$ entrées non observées, chacune pouvant avoir sortie 0 ou 1 : $2^6 = 64$ fonctions compatibles.
- (d) Les données ne permettent pas de départager ces 64 fonctions qui diffèrent sur les points non vus. Sans hypothèse supplémentaire (biais inductif), on ne peut pas généraliser. Avec $n$ variables, le nombre de fonctions est $2^{2^n}$ (doublement exponentiel), et l'immense majorité est incompressible en une règle courte.

### Q11. Qu'est-ce qu'un biais inductif ? Donner un exemple et expliquer pourquoi il doit être « aligné avec la vérité ».

Un **biais inductif** est une restriction de l'espace des hypothèses $\mathcal{H}$ décidée à l'avance (« œillères ») : fonction linéaire, arbre de profondeur ≤ 3, gène de longueur ≤ $x$ paires de bases, couleur des yeux dépendant seulement des chromosomes 2, 7 et 9.

Il rend l'apprentissage possible avec peu de données et peu de temps. Mais s'il est faux (on suppose un gène de 150 paires de bases alors qu'il en a 200), la vraie fonction n'est pas dans $\mathcal{H}$ : on ne la trouvera **jamais**, quelle que soit la quantité de données. Restreindre est un pari.

### Q12. Énoncer le théorème du no free lunch et expliquer son lien avec le surapprentissage.

**NFL :** moyenné sur tous les problèmes possibles (toutes les fonctions cibles également probables), aucun algorithme n'est meilleur qu'un autre ; sur les points non vus, la sortie vraie est 0 ou 1 avec probabilité 1/2 quoi qu'on ait observé.

**Lien :** le surapprentissage consiste à traiter l'ajustement aux données comme une preuve de généralisation. NFL montre que c'est faux : coller à l'entraînement ne dit rien sur les points non vus sans biais inductif. Pour choisir parmi les hypothèses compatibles, il faut un pari sur le monde (restriction de $\mathcal{H}$, régularisation). Un algorithme n'est bon que *relativement à une classe de problèmes*.

---

## Partie E : Critère inductif

### Q13. Différence entre fonction de perte, MSE et risque ? La MSE est-elle une fonction de perte ?

- **Perte** $\ell(f(x),y)$ : erreur sur **un seul** exemple (ex. perte quadratique $(f(x)-y)^2$, perte 0-1, entropie croisée).
- **MSE** : moyenne de la perte quadratique sur un ensemble d'exemples, $\frac1n\sum_i (f(x_i)-y_i)^2$. C'est donc un **risque empirique** avec perte quadratique.
- Par abus de langage on dit souvent « loss = MSE », mais rigoureusement la perte est par exemple et le risque est la moyenne.

### Q14. Définir risque réel et risque empirique. Pourquoi le premier est-il inaccessible ? Écrire le principe ERM.

$$
R(f) = \mathbb{E}_{(x,y)\sim P}[\ell(f(x),y)] \qquad \hat R_n(f) = \frac1n\sum_{i=1}^n \ell(f(x_i),y_i)
$$

Le risque réel est l'espérance de la perte sur toute la distribution $P$ ; $P$ étant **inconnue**, on ne peut pas le calculer. Le risque empirique est la moyenne sur l'échantillon, calculable.

**ERM :** $\hat f = \arg\min_{f\in\mathcal{H}} \hat R_n(f)$ : on minimise le substitut calculable en espérant que cela minimise le risque réel.

### Q15. Exprimer le surapprentissage et le sous-apprentissage à l'aide des risques et de $\mathcal{H}$.

- **Surapprentissage :** $\hat R_n(f)$ très petit mais $R(f)$ grand ; écart de généralisation $R(f)-\hat R_n(f)$ important. Typique d'un $\mathcal{H}$ trop riche par rapport à $n$.
- **Sous-apprentissage :** $\hat R_n$ et $R$ tous deux élevés ; la bonne fonction n'est pas dans $\mathcal{H}$ (biais trop fort ou mal choisi).

### Q16. (PAC) **(a)** Définir un algorithme PAC (rôle de $\varepsilon$ et $\delta$). **(b)** Pour $\mathcal{H}$ fini en cas réalisable, $n \ge \frac1\varepsilon\left(\ln|\mathcal{H}|+\ln\frac1\delta\right)$. Calculer $n$ pour $\varepsilon=0.1$, $\delta=0.05$ avec $|\mathcal{H}|=2^{10}$ puis $2^{20}$. **(c)** Interpréter.

**(a)** Pour tous $\varepsilon,\delta\in(0,1)$, avec au moins $n(\varepsilon,\delta)$ exemples, l'algorithme renvoie $\hat f$ telle que $\Pr[R(\hat f)\le\varepsilon]\ge 1-\delta$.

- $\varepsilon$ : erreur réelle tolérée (*approximately correct*).
- $\delta$ : probabilité d'échec, par malchance sur l'échantillon (*probably*).

**(b)**

- $|\mathcal{H}|=2^{10}$ : $\ln|\mathcal{H}| = 10\ln 2 \approx 6{,}93$ ; $\ln(1/0{,}05)\approx 3{,}00$. Donc $n \ge 10\times(6{,}93+3{,}00) \approx 99{,}3$, soit **$n = 100$**.
- $|\mathcal{H}|=2^{20}$ : $\ln|\mathcal{H}|\approx 13{,}86$. Donc $n \ge 10\times(13{,}86+3{,}00)\approx 168{,}6$, soit **$n = 169$**.

**(c)** Doubler le *logarithme* de la taille de $\mathcal{H}$ (de $2^{10}$ à $2^{20}$ hypothèses, soit $1024$ fois plus d'hypothèses) n'augmente $n$ que d'environ 70 %. $n$ croît avec $\ln|\mathcal{H}|$, donc un espace plus riche exige plus de données : c'est la version quantitative de « restreindre $\mathcal{H}$ rend l'apprentissage possible avec peu de données ». Pour $\mathcal{H}$ infini, on remplace $\ln|\mathcal{H}|$ par la dimension VC.

---

## Partie F : Vrai / Faux (justifier)

**(a)** « La MSE est la perte calculée sur un seul exemple. » → **Faux.** C'est une moyenne sur plusieurs exemples (risque empirique avec perte quadratique).

**(b)** « L'ERM garantit un faible risque réel quelle que soit la taille de $\mathcal{H}$. » → **Faux.** Si $\mathcal{H}$ est trop riche par rapport à $n$, on peut avoir $\hat R_n=0$ avec $R$ grand (surapprentissage). Il faut un $\mathcal{H}$ contraint et assez de données (convergence uniforme).

**(c)** « Agrandir $\mathcal{H}$ réduit toujours le risque réel. » → **Faux.** Cela réduit le biais mais augmente le risque de surapprentissage ; il y a un compromis (biais-variance).

**(d)** « NFL implique qu'aucun algorithme n'est utile en pratique. » → **Faux.** NFL parle de la moyenne sur *tous* les problèmes. Les problèmes réels ont de la structure (régularité, parcimonie) qui permet à nos biais de fonctionner.

**(e)** « Un jeu de test peut être réutilisé pour choisir les hyperparamètres sans conséquence. » → **Faux.** Il faut un jeu de validation distinct ; sinon l'estimation du risque réel devient optimiste.

---

## Partie G : Question de synthèse (type dissertation courte)

### Q17. Expliquer comment le problème de l'induction conduit au principe ERM et à la théorie PAC, en vous appuyant sur NFL et le surapprentissage.

**Plan de réponse :**

1. **Problème :** induire à partir d'exemples finis n'a aucune garantie logique ; l'objectif est le risque réel $R(f)$, inaccessible car $P$ est inconnue.
2. **NFL :** sans hypothèse, les données d'entraînement ne renseignent pas sur les points non vus (il y a $2^{2^n}$ fonctions, presque toutes incompressibles). Il faut un **biais inductif** = restriction de $\mathcal{H}$.
3. **ERM :** on remplace $R$ par $\hat R_n$ et on minimise sur $\mathcal{H}$ ; c'est le critère inductif.
4. **Risque :** si $\mathcal{H}$ est trop riche, surapprentissage ($\hat R_n$ petit, $R$ grand) ; si le biais est faux, sous-apprentissage.
5. **PAC / VC :** quantifie quand $\hat R_n\approx R$ : $n$ doit croître avec la complexité de $\mathcal{H}$ ($\ln|\mathcal{H}|$ ou dimension VC), avec garantie « probablement approximativement correcte ».
6. **Conclusion :** apprendre exige toujours un pari (biais) *aligné avec la vérité* ; il n'y a pas de déjeuner gratuit.

---

# Réseaux de neurones : questions type partiel corrigées

---

## A. Du neurone biologique au neurone formel

### Q1. Quels sont les attraits et les défauts des réseaux de neurones artificiels ? Donner des ordres de grandeur.

**Attraits des réseaux artificiels :** calculs parallélisables, robustesse (information distribuée sur de nombreuses petites unités), algorithmes d'apprentissage simples, emploi très général.

**Défaut principal :** **opacité** du raisonnement et de l'hypothèse produite (problème d'explicabilité).

### Q2. Définir le neurone formel. Faire la correspondance avec le neurone biologique. Quel est le rôle du biais $w_0$ ?

Un neurone formel calcule une somme pondérée de ses entrées puis applique une fonction d'activation :

$$
y = g\Big(\sum_j w_j x_j\Big)
$$

Correspondance : dendrites = entrées $x_j$ ; synapses = poids $w_j$ (positif = excitateur, négatif = inhibiteur) ; corps cellulaire + potentiel d'action = somme puis fonction d'activation $g$ ; axone = sortie $y$.

Le **biais** $w_0$ est un décalage constant, vu comme le poids d'une entrée fictive $x_0 = 1$. Il permet à la frontière de décision de ne pas passer par l'origine.

### Q3. Retracer les grandes étapes historiques. Pourquoi y a-t-il eu un blocage, puis une relance ?

- 1943 McCulloch & Pitts : premier neurone formel.
- 1949 Hebb : règle d'apprentissage (« fire together, wire together »).
- 1958-62 Rosenblatt : le perceptron, premier neurone effectivement entraînable ; 1960 ADALINE (Widrow-Hoff).
- 1969 Minsky & Papert : limites du perceptron (XOR) → **blocage** de la recherche pendant 15 à 20 ans selon les versions du cours.
- 1982 Hopfield (réseau bouclé) ; **1985 perceptron multi-couches et rétropropagation** → relance ; 2003 reservoir computing ; puis réseaux profonds et réseaux impulsionnels.

Blocage : un neurone seul ne sépare que linéairement. Relance : on sait empiler des couches (non-linéarités) **et** on dispose d'un algorithme pour entraîner les couches cachées (rétropropagation).

---

## B. Le perceptron

### Q4. Donner l'interprétation géométrique du perceptron. Que sont les coordonnées homogènes ?

L'équation $\sum_j w_j x_j + w_0 = 0$ est celle d'un **hyperplan** (droite en 2D, plan en 3D, dimension $d-1$ en dimension $d$). Il coupe l'espace en deux régions : $h(x) \ge 0$ d'un côté, $h(x) < 0$ de l'autre. Le perceptron classe $x$ selon le **signe** de $h(x)$, donc selon le côté de l'hyperplan où il tombe. Le vecteur $w$ est normal à l'hyperplan.

**Coordonnées homogènes :** on ajoute $x_0 = 1$ à chaque entrée, ce qui absorbe le biais : $h(x) = w\cdot x$.

### Q5. Énoncer la règle d'apprentissage du perceptron et justifier qu'elle « va dans le bon sens ».

$$
\text{si } y_i\,(w\cdot x_i) \le 0 \text{ (exemple mal classé) alors } w \leftarrow w + \eta\, y_i\, x_i
$$

On n'apprend **qu'en cas d'échec**. Algorithme : $w$ initialisé à 0, on parcourt les exemples jusqu'à ce qu'une passe complète ne modifie plus rien.

**Justification :** si $y=+1$ mais $w\cdot x<0$, on veut augmenter $w\cdot x$. Avec $w' = w+\eta x$ : $w'\cdot x = w\cdot x + \eta\,x\cdot x = w\cdot x + \eta\|x\|^2 > w\cdot x$. Symétrique si $y=-1$. Géométriquement, on fait pivoter la frontière vers le bon côté.

### Q6. (Calcul) En coordonnées homogènes $x=(1,x_1,x_2)$, $w$ initialisé à $(0,0,0)$, $\eta=1$. Exemples dans l'ordre :

- $x^{(1)}=(1,2,-1)$, $y=+1$ ; $x^{(2)}=(1,-1,1)$, $y=-1$ ; $x^{(3)}=(1,1,1)$, $y=-1$.

**Dérouler l'algorithme. Combien de mises à jour ? Que conclure ?**

Passe 1 :

- $x^{(1)}$ : $y\,w\cdot x = 0 \le 0$ → mise à jour : $w=(0,0,0)+1\cdot(1,2,-1)=(1,2,-1)$.
- $x^{(2)}$ : $w\cdot x = 1-2-1=-2$ ; $y\,w\cdot x = (-1)(-2)=2>0$ → bien classé, rien.
- $x^{(3)}$ : $w\cdot x = 1+2-1=2$ ; $y\,w\cdot x=-2\le 0$ → mise à jour : $w=(1,2,-1)-(1,1,1)=(0,1,-2)$.

Passe 2 : $x^{(1)}$ : $0+2+2=4$, $y=+1$ ✓ ; $x^{(2)}$ : $0-1-2=-3$, $y=-1$ ✓ ; $x^{(3)}$ : $0+1-2=-1$, $y=-1$ ✓. Aucune correction : arrêt.

**Résultat : 2 mises à jour, $w=(0,1,-2)$**, soit la frontière $x_1 - 2x_2 = 0$. Les données sont linéairement séparables, donc la convergence est cohérente avec le théorème.

### Q7. Énoncer le théorème de convergence du perceptron. Quelles sont son hypothèse et ses limites ?

**Énoncé :** s'il existe au moins une séparatrice linéaire de l'échantillon, l'algorithme converge en un **nombre fini d'étapes**, indépendamment du nombre d'exemples, de leur distribution et de la dimension de l'espace d'entrée.

**Hypothèse cruciale :** séparabilité linéaire. Sinon (ex. XOR), l'algorithme ne s'arrête jamais.

**Limite :** le théorème porte sur l'**échantillon d'apprentissage** (risque empirique nul) ; il ne dit **rien** sur les exemples à venir (risque réel). On retrouve le problème de l'induction.

### Q8. Pourquoi le XOR n'est-il pas linéairement séparable ? Qu'ont montré Minsky et Papert, et comment le contourner ?

Points de classe 1 : $(0,1)$ et $(1,0)$ ; classe 0 : $(0,0)$ et $(1,1)$. Les segments qui relient les deux points de chaque classe se coupent en $(0.5,0.5)$ : aucune droite ne peut avoir les deux points de classe 1 d'un côté et les deux de classe 0 de l'autre. Contrairement à AND ou OR, qui sont séparables.

Minsky & Papert (1969) ont montré qu'un neurone seul ne peut pas représenter ce type de fonction, ce qui a freiné le domaine. Ils nuancent : en **changeant d'espace d'entrée** avec de bonnes fonctions de base (détecteurs locaux), un perceptron peut y arriver, mais ce prétraitement est difficile à concevoir. D'où l'idée : **apprendre aussi les fonctions de base**, dans les couches cachées.

### Q9. Expliquer le lien entre la règle du perceptron et la descente de gradient sur le risque empirique.

$$
R_{Emp}(w) = -\sum_{x_j\in\mathcal M} w^\top x_j\,u_j
$$

où $\mathcal M$ est l'ensemble des exemples mal classés et $u_j=\pm1$. Chaque terme est positif (car mal classé : $u_j\,w^\top x_j<0$), donc $R_{Emp}\ge0$ et vaut 0 si plus aucun exemple n'est mal classé. Le gradient par rapport à $w$ est $-\sum_{\mathcal M} u_j x_j$. Un pas de descente sur un seul exemple mal classé donne $w \leftarrow w+\eta\,u_j x_j$ : c'est exactement la règle du perceptron, une **descente de gradient en ligne** sur le risque empirique. Les exemples bien classés ne contribuent pas à l'erreur, donc ne modifient pas $w$.

---

## C. Le perceptron multi-couches (PMC)

### Q10. Décrire la structure d'un PMC et écrire la propagation avant.

Une couche d'entrée (les données, pas de vrais neurones), une ou plusieurs **couches cachées**, une couche de sortie. L'information va de l'entrée vers la sortie (feedforward). Pour chaque cellule $k$ :

$$
a_k=\sum_{j=0}^{d} w_{jk}\,z_j \qquad z_k=g(a_k)
$$

$w_{jk}$ : poids de la cellule $j$ vers $k$ ; $a_k$ : activation ; $g$ : fonction d'activation ; $j=0$ : biais.

### Q11. Pourquoi une fonction d'activation non linéaire est-elle indispensable ? Citer plusieurs fonctions d'activation avec leur intervalle de sortie. Pourquoi Heaviside est-elle mal adaptée à l'apprentissage ?

**Indispensable :** une composition de transformations linéaires reste une transformation linéaire (*complément* : cours). Sans non-linéarité, empiler des couches n'apporte rien et on retombe sur un séparateur linéaire. Avec non-linéarité, le réseau peut représenter des frontières courbes.

| Fonction  | Formule                         | Sortie         |
| --------- | ------------------------------- | -------------- |
| Heaviside | 0 si $x<0$, 1 sinon           | $\{0,1\}$    |
| Sigmoïde | $1/(1+e^{-a})$, $g'=g(1-g)$ | $[0,1]$      |
| tanh      | $(e^x-e^{-x})/(e^x+e^{-x})$   | $[-1,1]$     |
| ReLU      | $\max(0,x)$                   | $[0,\infty[$ |

Heaviside **n'est pas dérivable** (et sa dérivée vaut 0 partout ailleurs) : inutilisable avec la descente de gradient. La sigmoïde est dérivable avec $g'(a)=g(a)(1-g(a))$ (on réutilise $g(a)$ déjà calculé). ReLU est très utilisée dans les réseaux profonds.

### Q12. Comment choisir la fonction d'activation de la couche de sortie ? Quel piège éviter ?

- **Classification :** sigmoïde (une classe, sortie dans $[0,1]$ interprétable comme une probabilité) ou softmax (plusieurs classes).
- **Régression :** fonction linéaire.
- **Piège :** ne pas mettre de sigmoïde sur une cible qui sort de $[0,1]$ (le réseau ne pourra jamais l'atteindre).

### Q13. Quel est le rôle profond des couches cachées ? Illustrer avec le codage one-hot, et faire le lien avec le deep learning.

Chaque couche **transforme l'espace des données en une nouvelle représentation** (variables latentes apprises). Le but de la succession de transformations est de rapprocher la distribution des sorties de la distribution cible. Avec un codage one-hot (classe bleue → (1,0), classe rouge → (0,1)), les couches successives rendent les classes de plus en plus séparables jusqu'à ce que la dernière couche les distingue trivialement.

Une couche linéaire déforme l'espace de façon affine ; ReLU « écrase » une partie de l'espace vers 0 ; la norme L2 projette sur une sphère. Avec une activation non linéaire l'espace se « plie ». **Deep learning** = pousser l'idée à l'extrême : beaucoup de couches, chacune apprenant une abstraction un peu plus haute que la précédente, sans les spécifier à la main.

### Q14. Expliquer comment un réseau 2-2-1 résout le XOR et proposer 9 poids qui fonctionnent. (Activation rampe : 0 si somme ≤ −1, 1 si somme ≥ 1, linéaire entre les deux.)

**Idée :** chaque neurone caché apprend une droite de séparation partielle ; la sortie les combine. Ici $h_1\approx$ OR, $h_2\approx$ AND, et XOR = OR et non AND.

Une solution possible (pas forcément celle du cours) :

- $h_1 = g(-1 + 2x_1 + 2x_2)$ : vaut 0 en (0,0), 1 sinon → OR.
- $h_2 = g(-3 + 2x_1 + 2x_2)$ : vaut 1 seulement en (1,1) → AND.
- $y = g(-1 + 2h_1 - 2h_2)$.

Vérification : (0,0) → $h=(0,0)$, $y=g(-1)=0$ ✓ ; (0,1) et (1,0) → $h=(1,0)$, $y=g(1)=1$ ✓ ; (1,1) → $h=(1,1)$, $y=g(-1)=0$ ✓. Total : $3+3+3=9$ poids.

---

## D. Apprendre les poids : gradient et rétropropagation

### Q15. Poser le problème d'optimisation, écrire la descente de gradient, et comparer les modes hors ligne (batch) et en ligne.

$$
w^*=\arg\min_w \frac1m\sum_{l=1}^m \big[y(x_l;w)-u(x_l)\big]^2 \qquad w^{(t)}=w^{(t-1)}-\eta\,\nabla_w E
$$

On va dans la direction opposée au gradient, avec un pas $\eta$ (taux d'apprentissage).

|              | Hors ligne (batch / total)                       | En ligne (stochastique)                                           |
| ------------ | ------------------------------------------------ | ----------------------------------------------------------------- |
| Mise à jour | Après tous les exemples, avec le gradient moyen | Après **chaque** exemple                                   |
| Gradient     | Exact (moyenne)                                  | Bruité                                                           |
| Avantage     | Direction plus stable                            | Plus de mises à jour, bruit qui aide à sortir des minima locaux |

### Q16. Quel problème la rétropropagation résout-elle ? Écrire les formules des $\delta$ et de la mise à jour, et donner l'intuition. Lister les 7 étapes d'une itération.

**Problème :** l'erreur n'est mesurée qu'en sortie ; un neurone caché n'a pas de « cible ». Comment savoir de combien il a contribué à l'erreur ?

**Solution :** règle de dérivation en chaîne. Avec $\delta_j=\partial E/\partial a_j$ :

$$
\frac{\partial E}{\partial w_{ij}}=\delta_j z_i \quad;\quad \delta_k=(y_k-u_k)\,g'(a_k)\ \text{(sortie)} \quad;\quad \delta_j=g'(a_j)\sum_k w_{jk}\delta_k\ \text{(cachée)}
$$

$$
\Delta w_{ij}=-\eta\,\delta_j z_i
$$

**Intuition :** un neurone caché est « responsable » de l'erreur proportionnellement à son influence sur les neurones qu'il alimente. On fait donc remonter les $\delta$ de la couche suivante, pondérés par les poids, d'où « rétro-propagation ».

**7 étapes :** (1) présenter un exemple ; (2) propagation avant ; (3) calcul de l'erreur ; (4) calcul des gradients par rétropropagation ; (5) modification des poids ; (6) test du critère d'arrêt ; (7) retour en 1.

### Q17. (Calcul) **(a)** Un neurone de sortie sigmoïde reçoit $z=1$ avec un poids $w=0.5$ (pas de biais). Cible $u=1$, $E=\tfrac12(y-u)^2$, $\eta=1$. Calculer $y$, $\delta$, $\partial E/\partial w$, le nouveau poids, et vérifier que $E$ diminue. **(b)** Un neurone caché sigmoïde a pour sortie $z_j=0.5$ ; il alimente un seul neurone de sortie par un poids $w_{jk}=2$ avec $\delta_k=-0.1$. Calculer $\delta_j$.

**(a)**

- $a=0.5$ ; $y=g(0.5)=1/(1+e^{-0.5})\approx0.6225$.
- $g'(a)=y(1-y)\approx0.6225\times0.3775\approx0.2350$.
- $\delta=(y-u)g'(a)\approx(-0.3775)(0.2350)\approx-0.0887$.
- $\partial E/\partial w=\delta z\approx-0.0887$.
- $w'=w-\eta\,\partial E/\partial w\approx0.5+0.0887=0.5887$.
- Vérification : $E$ avant $=\tfrac12(0.3775)^2\approx0.0713$ ; après : $y'=g(0.5887)\approx0.6431$, $E'=\tfrac12(0.3569)^2\approx0.0637$. $E$ a baissé ✓.

**(b)** $g'(a_j)=z_j(1-z_j)=0.5\times0.5=0.25$ (on réutilise la sortie, pas l'activation). $\delta_j=g'(a_j)\,w_{jk}\,\delta_k=0.25\times2\times(-0.1)=-0.05$.

### Q18. Discuter les limites pratiques de la rétropropagation : minima locaux, coût, plausibilité biologique, piège du signe. Comparer avec la règle de Hebb.

- **Minima locaux :** la fonction d'erreur dans l'espace des poids est un paysage montagneux ; la descente peut rester coincée dans un creux non optimal. Remèdes : apprentissage stochastique (en ligne), plusieurs initialisations (plusieurs dizaines de relances).
- **Coût :** $O(w)$ par passe ($w$ = nombre de poids), plusieurs centaines de passes ; en revanche la reconnaissance est en temps réel.
- **Plausibilité biologique :** la rétropropagation est une règle **top-down** globale (une erreur renvoyée couche par couche), peu plausible. La règle de Hebb est **bottom-up et locale** (chaque connexion dépend de l'activité de ses deux neurones), plus plausible, mais elle ne minimise pas explicitement un coût ; elle apprend des représentations « info-max ».
- **Piège du signe :** selon que $\delta$ est défini avec $(u-y)$ ou $(y-u)$, les diapos écrivent $+\eta\delta$ ou $-\eta\delta$. La règle de fond reste $w\leftarrow w-\eta\,\partial E/\partial w$.

---

## E. Généralisation, architecture et codage de la sortie

### Q19. Définir le sur-apprentissage à partir des courbes d'erreur. Qu'est-ce que l'arrêt précoce ? Que deviennent les courbes avec 2 000 exemples au lieu de 1 000 ? Reformuler avec les risques empirique et réel.

**Sur-apprentissage :** l'erreur sur l'ensemble d'apprentissage continue de baisser alors que l'erreur sur l'ensemble de **validation** remonte après un minimum.
**Arrêt précoce :** on arrête l'apprentissage quand l'erreur de validation cesse de baisser.
**Avec 2 000 exemples :** l'écart entre les deux courbes se réduit, le minimum de validation est plus bas et plus tardif.
**Avec les risques :** $\hat R_n(f)$ continue de baisser mais $R(f)$ (estimé par la validation) remonte : l'écart de généralisation grandit. Plus de données le réduit (cohérent avec PAC : $n$ doit croître avec la complexité de $\mathcal H$).

### Q20. Définir erreur d'approximation et erreur d'estimation. Comment le choix de l'architecture les contrôle-t-il ? Faire le lien avec le biais inductif.

- **Approximation (biais) :** distance entre la meilleure hypothèse de $\mathcal H$ et la fonction cible $f$.
- **Estimation (variance) :** distance entre l'hypothèse **apprise** (échantillon limité et bruité) et la meilleure de $\mathcal H$.

Trop peu de neurones : $\mathcal H$ trop pauvre → biais élevé → **sous-apprentissage**. Trop de neurones/paramètres par rapport aux exemples : variance élevée → **sur-apprentissage** (frontières trop découpées, courbe qui oscille entre les points en régression). On ne peut généralement pas réduire les deux en même temps.

**Lien avec le cours 1 :** l'architecture est notre **biais inductif** (restriction de $\mathcal H$). Si la vraie fonction n'est pas dans $\mathcal H$ → erreur d'approximation irréductible (le gène de 200 paires de bases avec $\mathcal H$ limité à 150) ; si $\mathcal H$ est trop riche pour $n$ → erreur d'estimation (borne PAC en $\ln|\mathcal H|$).

### Q21. Écrire l'entropie croisée pour une sortie sigmoïde. Pourquoi l'utiliser plutôt que l'erreur quadratique en classification ? Calculer la perte pour $y=1$ avec $\hat y=0.8$ puis $\hat y=0.1$ (log naturel).

$$
\ell(y,\hat y)=-\big[y\log\hat y+(1-y)\log(1-\hat y)\big],\quad \hat y=\frac1{1+\exp(-w^\top x)},\ y\in\{0,1\}
$$

Le cours : ce critère entropique **« calibre »** la sortie pour qu'elle soit interprétable comme une probabilité d'appartenance à la classe. Avec plusieurs classes : un neurone par classe, sortie softmax, même critère. *(Complément :)* la perte tend vers $+\infty$ quand une prédiction confiante est fausse, ce qui pénalise fortement ces erreurs.

Calcul : $y=1$, $\hat y=0.8$ → $-\ln0.8\approx0.223$ ; $\hat y=0.1$ → $-\ln0.1\approx2.303$. La perte est ~10 fois plus grande quand le modèle est confiant dans la mauvaise direction.

### Q22. Multi-classes : citer les trois codages possibles. Expliquer le principe des codes correcteurs d'erreur (ECOC) et pourquoi 10 bits pour 26 lettres n'est pas du gaspillage. **(Calcul)** Codes sur 6 bits : A = 000000, B = 111000, C = 000111. Les 6 classifieurs binaires rendent 100100. Décoder. Combien d'erreurs ce code corrige-t-il à coup sûr ?

**Trois codages :** (1) $c-1$ problèmes de discrimination (un classifieur binaire par frontière) ; (2) **un seul neurone** de sortie (dans $\{0,\dots,c\}$ ou $[0,1]$), fragile car il impose un ordre entre classes ; (3) $n\le c$ neurones : un par classe (one-hot) ou code correcteur d'erreur.

**ECOC :** chaque classe reçoit un mot-code binaire ; on apprend un classifieur binaire par bit, chacun séparant les classes en deux groupes ; les codes sont choisis pour **maximiser la distance de Hamming** entre classes. À la prédiction, on calcule tous les bits puis on décode par le **plus proche voisin** parmi les codes connus. 5 bits suffiraient pour 26 lettres ($2^5=32$), mais 10 bits ajoutent de la **redondance** qui corrige les erreurs de certains classifieurs, comme un code correcteur en télécommunication.

**Calcul :** distances de Hamming de 100100 : à A (000000) : 2 ; à B (111000) : 3 ; à C (000111) : 3. Le plus proche est **A** (distance 2).
La distance minimale entre codes ici est 3 (A-B et A-C) ; un code de distance minimale $d$ corrige à coup sûr $\lfloor (d-1)/2\rfloor$ erreurs, soit **1 erreur** *(complément)*.

---

## F. Réseaux profonds

### Q23. Définir la convolution (filtre, champ réceptif, carte de traits). Pourquoi le partage de poids est-il puissant ? **(Calcul)** Patch $\begin{pmatrix}1&0&2\\3&1&0\\0&2&1\end{pmatrix}$, filtre $\begin{pmatrix}1&0&-1\\1&0&-1\\1&0&-1\end{pmatrix}$ : calculer la sortie.

**Convolution :** on fait glisser un petit filtre (grille de poids, ex. 3×3 ou 11×11) sur toute l'image et on calcule, à chaque position, la somme pondérée des pixels recouverts. C'est le calcul d'un neurone (somme pondérée + activation), mais :

- il ne regarde qu'une petite région locale (**champ réceptif** / fenêtre) ;
- le **même** filtre (mêmes poids) est réutilisé à toutes les positions : **partage de poids**.

Chaque valeur calculée devient un pixel de la **carte de traits** (feature map), qui indique où le motif cherché (bord, courbure, texture) est présent.

**Pourquoi c'est puissant :** (1) bien moins de paramètres à apprendre qu'un réseau totalement connecté ; (2) un détecteur appris dans un coin sert partout : **invariance par translation**.

**Calcul :** ligne 1 : $1\cdot1+0\cdot0+2\cdot(-1)=-1$ ; ligne 2 : $3+0+0=3$ ; ligne 3 : $0+0-1=-1$. Total : **1**.

### Q24. Définir le max-pooling et son intérêt. **(Calcul)** Appliquer un max-pooling 2×2 (sans recouvrement) puis un average-pooling 2×2 à $\begin{pmatrix}2&7&1&0\\3&4&6&2\\0&1&9&8\\5&2&3&4\end{pmatrix}$.

**Max-pooling :** on découpe la carte en blocs (ex. 2×2) et on ne garde que la valeur maximale de chaque bloc. Intérêts : (1) réduit la taille des données donc le coût des couches suivantes ; (2) rend la reconnaissance robuste à de petits décalages ou déformations (seule compte la présence du signal fort dans la zone). Il n'a **aucun paramètre appris** *(complément)*.

**Calcul :**

- Max : bloc haut-gauche $\max(2,7,3,4)=7$ ; haut-droite $\max(1,0,6,2)=6$ ; bas-gauche $\max(0,1,5,2)=5$ ; bas-droite $\max(9,8,3,4)=9$ → $\begin{pmatrix}7&6\\5&9\end{pmatrix}$.
- Moyenne : $(4,\ 2.25;\ 2,\ 6)$ → $\begin{pmatrix}4&2.25\\2&6\end{pmatrix}$.

### Q25. Décrire l'architecture générale d'un CNN. Comment les filtres sont-ils obtenus ? Situer LeNet-5, ImageNet/AlexNet, U-Net et GoogleNet.

**CNN :** (1) convolution (filtres qui détectent des motifs locaux) ; (2) pooling (réduction de résolution) ; (3) nouvelle convolution sur les cartes simplifiées (motifs plus complexes et abstraits) ; on alterne ; (4) classifieur final : couches **complètement connectées** (un PMC) qui prédisent la classe à partir des descripteurs générés automatiquement.

**Les filtres sont appris par rétropropagation**, comme les poids d'un PMC ; la seule différence est la contrainte de structure (connexions locales, poids partagés). On ne dit pas au réseau quels motifs chercher.

- **Ancêtre (LeCun 1989) :** codes postaux, ≈ 10 000 chiffres 16×16, poids partagés, *optimal brain damage* (élagage), 99 % de reconnaissance correcte sur l'apprentissage et 9 % de rejet.
- **LeNet-5 :** convolutions + agrégations + couches complètement connectées, sortie softmax.
- **ImageNet :** > 15 millions d'images étiquetées, ≈ 22 000 catégories, étiquetées via Amazon Mechanical Turk. **AlexNet (2012)** y a dominé.
- **U-Net :** **segmentation** (une classe par pixel). Architecture en U, entièrement convolutive et symétrique : la descente réduit la taille spatiale et augmente le nombre de filtres ; la montée reconstruit une carte à la résolution d'origine ; des connexions directes (*copy and crop*) conservent les détails fins.
- **GoogleNet :** « mécano » de blocs convolutifs combinés, pas une simple pile. Système Google/Stanford : 9 couches, $10^9$ connexions, 10 millions d'images, 3 jours sur 16 000 processeurs, non supervisé.

### Q26. (Calcul) Première couche d'AlexNet : entrée 227×227×3, filtres 11×11, stride 4, 96 filtres. Calculer la taille de la carte de sortie, le nombre de paramètres **sans** partage de poids, puis **avec** partage. Interpréter.

- Taille : $(227-11)/4+1=55$ par dimension → sortie $55\times55\times96$.
- Poids par neurone : $11\times11\times3=363$ + 1 biais = 364.
- **Sans partage :** $55\times55\times96\times364=105\,705\,600$ paramètres (le chiffre du cours).
- **Avec partage :** un seul jeu de poids par filtre : $96\times364=34\,944$ paramètres.
- Rapport : $55\times55=3\,025$ fois moins. C'est exactement l'intérêt de la convolution. Même un réseau peu profond a énormément de paramètres sans partage, d'où le besoin de beaucoup de données, de calcul et de régularisation.

### Q27. Que représentent les neurones cachés d'un CNN ? Expliquer la hiérarchie de représentations, la *feature visualization*, et la phrase « l'apprentissage est facile quand on sait ce qu'il faut chercher ».

**Hiérarchie :** couche 1 : bords orientés, contrastes ; couche 2 : parties d'objets (œil, nez, oreille) ; couche 3 : objets entiers (visage). Même phénomène pour la parole et le texte : les couches basses captent des régularités locales, les hautes des représentations sémantiques. C'est cela qui caractérise le deep learning : des abstractions successives apprises automatiquement.

**Feature visualization :** grâce à une symétrie de la rétropropagation entre poids et exemples, on peut optimiser l'**entrée** (au lieu des poids) pour qu'elle active au maximum un neurone caché donné, ce qui montre à quoi il « répond » (une façon d'atténuer l'opacité).

**Représentation des entrées :** un bon codage peut rendre l'apprentissage presque trivial, mais comment connaître a priori le bon ? Les réseaux profonds réduisent cette dépendance : au lieu de fabriquer des descripteurs à la main, on laisse le réseau les apprendre à partir des pixels bruts.

### Q28. Que faire quand on manque de données étiquetées ? Comparer augmentation de données, auto-encodage et auto-apprentissage.

- **Augmentation de données :** générer de nouveaux exemples à partir des exemples étiquetés, en conservant l'étiquette (un chat pivoté reste un chat). Soit transformations simples (bruitage, flou, contraste, luminosité, rotation, dilatation, translation), soit **GAN** (puissant mais « pas évident du tout à utiliser »).
- **Auto-encodage :** un encodeur compresse $x$ en une représentation réduite (l'**embedding**) à travers un **goulot d'étranglement**, un décodeur reconstruit $\hat x\approx x$. Pour y arriver, l'encodeur apprend une représentation compacte et informative. Mécaniquement supervisé (la cible est $x$) mais **sans étiquette humaine**.
- **Auto-apprentissage (self-supervised) :** non supervisé (aucune étiquette humaine) mais ramené à un apprentissage supervisé en fabriquant automatiquement des paires (entrée, étiquette) via une **tâche prétexte**, ex. prédire la position relative de deux morceaux d'une image, ou dire si une séquence est valide / prédire la suite. Le réseau acquiert des connaissances réutilisables pour la vraie tâche.

### Q29. Expliquer BatchNorm, LayerNorm et L2-norm : que normalise-t-on, sur quoi calcule-t-on les statistiques, à quoi ça sert ? **(Calcul)** **(a)** Un descripteur prend les valeurs 2, 4, 6 sur un batch de 3 exemples : BatchNorm (γ=1, β=0, on ignore ε). **(b)** Un exemple a pour descripteurs (1, 3, 5, 7) : LayerNorm (γ=1, β=0). **(c)** L2-norm de (3, 4).

**But :** empêcher les valeurs qui circulent dans le réseau d'être trop grandes, trop petites ou trop disparates d'une couche à l'autre, ce qui stabilise et accélère la descente de gradient.

|                     | Normalise                          | Statistiques calculées sur                    |
| ------------------- | ---------------------------------- | ---------------------------------------------- |
| **BatchNorm** | chaque descripteur indépendamment | tous les exemples du **batch**           |
| **LayerNorm** | chaque exemple indépendamment     | tous les **descripteurs** de cet exemple |
| **L2-norm**   | $x/\|x\|_2$                      | projection sur une sphère de norme constante  |

$$
\text{BatchNorm}:\ \gamma\frac{x-\mathbb E[x]}{\sqrt{\mathrm{Var}[x]}}+\beta \qquad \text{LayerNorm}:\ \gamma\frac{x-\mu}{\sigma}+\beta
$$

**(a)** moyenne 4, variance $(4+0+4)/3=8/3$, écart-type ≈ 1,633 → $(-1.225,\ 0,\ 1.225)$.
**(b)** moyenne 4, variance $(9+1+1+9)/4=5$, écart-type ≈ 2,236 → $(-1.342,\ -0.447,\ 0.447,\ 1.342)$.
**(c)** $\|(3,4)\|_2=5$ → $(0.6,\ 0.8)$.

### Q30. Expliquer le dropout. Pourquoi est-ce une régularisation ? Que se passe-t-il au moment du test ?

Pendant l'apprentissage, à chaque itération, on **éteint aléatoirement une fraction des neurones** (et leurs connexions). Intuition : on empêche la **co-adaptation** (neurones qui comptent trop les uns sur les autres) ; le réseau doit apprendre des représentations plus redondantes et robustes, qui ne dépendent d'aucun neurone en particulier (comme une équipe où n'importe quel coéquipier peut être absent). À chaque exemple, c'est un réseau différent. Cela limite le sur-apprentissage. **Au test, tous les neurones sont réactivés.**

### Q32. Citer des limites des réseaux profonds évoquées dans le cours.

- **Exemples adversariaux :** une perturbation imperceptible change la prédiction.
- **Explicabilité :** cartes Grad-CAM (Selvaraju et al. 2017) : difficulté à expliquer, même sur des illusions d'optique.
- **Annotation automatique d'images :** pas toujours fiable (exemple médical du cours).
- **« Voiture dans la piscine » (Bottou 2015) :** le réseau utilise des **corrélations de contexte** plutôt que l'objet.
- Plus : opacité générale, difficulté d'injecter des connaissances a priori, choix d'architecture empirique, apprentissage lent et sujet aux minima locaux.

---

## G. Autres architectures

### Q33. Expliquer le reservoir computing : que fixe-t-on, qu'entraîne-t-on, quel est l'avantage ?

Un grand **réseau récurrent** (boucles internes, donc mémoire) dont les **poids sont fixés aléatoirement une fois pour toutes** : le « réservoir ». Une entrée y génère des dynamiques riches (« échos » du signal qui se propagent et se combinent dans le temps). **Seule la couche de sortie (readout) est entraînée**, par simple régression linéaire, pour combiner ces états internes.

**Avantages :** réintroduit une dynamique temporelle (utile pour les séries temporelles) et évite les difficultés d'entraînement par rétropropagation des réseaux récurrents profonds, notoirement instables sur de longues séquences.

### Q34. Expliquer les réseaux impulsionnels (spiking). Pourquoi sont-ils difficiles à entraîner ?

Plus proches du neurone biologique : au lieu d'une valeur continue, le neurone communique par de brèves **impulsions (spikes)** discrètes. Chaque impulsion présynaptique crée un potentiel postsynaptique **excitateur (EPSP)** ou **inhibiteur (IPSP)** selon le signe du poids ; ces potentiels s'additionnent dans le temps ; quand le potentiel cumulé $u_j(t)$ atteint un **seuil**, le neurone émet une impulsion puis son potentiel retombe.

Difficiles à entraîner : l'émission d'une impulsion est un événement **discret, non dérivable** au sens usuel, donc la descente de gradient classique ne s'applique pas directement.

---

## H. Questions pratiques et prétraitement

### Q35. Quelles sont les principales décisions pratiques d'entraînement ? Pour chacune, donner les options du cours et leurs risques.

- **Batch ou en ligne :** total (un pas par passe complète) vs en ligne (un pas par exemple, plus bruité, aide à sortir des minima locaux).
- **Pas $\eta$ :** fixe ; variable ; méthode du **moment** ; méthodes de 2ᵉ ordre (Hessien). Pas trop grand → divergence ; pas trop petit → lenteur.
- **Critère d'arrêt :** nombre d'époques fixé ; arrêt de la baisse de l'erreur empirique ; **arrêt précoce** (arrêt de la baisse de l'erreur de validation).
- **Codage de la sortie :** entropie croisée, softmax, ECOC.
- **Architecture :** nombre de couches et de neurones, contrôle la généralisation.

### Q36. Distinguer variables nominales, ordinales et numériques. Pourquoi coder une marque de voiture par un entier est-il problématique ? Qu'est-ce que la discrétisation ?

| Type                 | Exemples                         | Propriétés                                                      |
| -------------------- | -------------------------------- | ----------------------------------------------------------------- |
| Nominal              | sexe, profession, marque         | valeurs dénombrables, **aucun ordre**, pas d'arithmétique |
| Ordinal              | taille (petite, moyenne, grande) | modalités **ordonnées**, calculs sur des rangs            |
| Numérique / continu | âge, poids                      | ordre, distance, écart, arithmétique                            |

Coder « marque » par 1, 2, 3 introduit un **ordre et des distances absurdes** (la marque 3 serait « plus proche » de 2 que de 1). Solution : **one-hot encoding** (un bit par modalité).

**Discrétisation :** passer du numérique au discret en découpant en intervalles (de **même largeur**, ou de **même fréquence**). Le choix compte pour un attribut asymétrique.

### Q37. Quels types de bruit et de valeurs manquantes peut-on rencontrer ? Quelles stratégies pour les valeurs manquantes, et quelles limites ?

**Bruit :** aléatoire, fonction de la classe, fonction de la source ; données fiables ou non ; données censurées.

**Valeurs manquantes :** examens non réalisés, question non posée, acheteurs qui n'achètent qu'une infime fraction des produits. Beaucoup de systèmes ne gèrent pas les données incomplètes. **Imputation :** valeur moyenne ; valeur la plus fréquente ; valeur moyenne **de la classe** ; fonction de prédiction (suppose une interdépendance entre variables, rappelle les systèmes de recommandation et pose la question de l'utilité de la variable). *(Limite, complément :)* l'imputation fabrique de l'information et peut biaiser, notamment si l'absence est elle-même informative.

### Q38. Pourquoi normaliser les variables ? Donner un exemple où la normalisation change le plus proche voisin. Pourquoi les statistiques de normalisation ne doivent-elles être calculées que sur l'apprentissage ?

Quand les domaines de variation sont très différents, les régularités trouvées sont faussées. La plus courante : normalisation **centrée réduite** (centrée autour de 0, divisée par l'écart-type) ; autres : tout dans $[0,1]$, discrétisation en un même nombre de valeurs. Ce n'est **pas forcément** une bonne idée.

**Exemple :** âge (en années, écart typique de quelques dizaines) et revenu (en euros, écart de plusieurs milliers) : dans la distance euclidienne le revenu écrase l'âge, donc le plus proche voisin est quasi uniquement choisi sur le revenu. Après normalisation, les deux variables pèsent comparablement et le plus proche voisin peut changer.

**Apprentissage uniquement :** le jeu de test représente des données jamais vues ; calculer moyenne/écart-type avec lui ferait fuiter de l'information du test dans l'apprentissage et fausserait l'estimation du risque réel *(complément)*.

### Q39. Classes déséquilibrées : pourquoi l'exactitude est-elle un piège ? Quelles solutions ?

Un classifieur qui répond toujours « classe majoritaire » obtient une **excellente exactitude** (ex. 99 % si la classe minoritaire représente 1 %) sans rien détecter. Solutions du cours : (1) **sous-échantillonner** la classe majoritaire (inadapté si peu de données) ; (2) créer des exemples **fictifs** de la classe minoritaire (ex. par bruitage) ; (3) **modifier les coûts** de mauvaise classification (ex. coût(faux positif) ≫ coût(faux négatif) selon le contexte).

### Q40. Qu'est-ce qu'un point aberrant ? Comment les traiter ? Quelles méthodes y sont sensibles ?

Valeur erronée ou exceptionnelle (salaire de diplômés à la sortie, fraudes, attaques sur un réseau). Méthodes : **expertise**, **contraintes d'intégrité**, **détection par apprentissage** (one-class SVM, boosting…). Une aberration peut être une erreur à corriger ou la cible même de l'étude (fraude). L'**ACP** y est très sensible (critère quadratique).

### Q41. Définir la malédiction de la dimensionnalité. Quelles en sont deux manifestations ? Que devient l'histogramme des distances entre points aléatoires quand $d$ passe de 2 à 10 000 ?

- **Statistique :** il faut un nombre de données **exponentiel en la dimension** pour approcher une distribution.
- **Apprentissage fondé sur les distances :** les distances se **concentrent**.

Quand $d$ augmente, l'histogramme des distances entre paires de points du cube unité devient de plus en plus **étroit et centré** : toutes les paires sont presque à la même distance, donc « plus proche » et « plus lointain » ne se distinguent plus (les méthodes de plus proche voisin perdent leur sens). D'où l'intérêt de réduire l'espace de description : attributs redondants, non pertinents ou peu informatifs.

### Q42. Comparer les trois familles de sélection de variables (intégrées, symbiose/wrapper, filtre) avec des exemples. Pourquoi le wrapper est-il coûteux ? Interpréter le rapport $F$ de l'ANOVA.

| Famille                         | Principe                                                                   | Exemples                                                  |
| ------------------------------- | -------------------------------------------------------------------------- | --------------------------------------------------------- |
| **Intégrée** (embedded) | l'apprentissage sélectionne lui-même                                     | arbres de décision ; Ridge, LASSO, Elastic Net           |
| **Symbiose** (wrapper)    | on explore des sous-ensembles en observant la performance d'un classifieur | sélection ascendante (forward) ou descendante (backward) |
| **Filtre**                | variables évaluées indépendamment du classifieur                        | ANOVA (paramétrique), RELIEF (non paramétrique)         |

Le wrapper est bien optimisé pour le classifieur mais l'espace des sous-ensembles est **gigantesque** ($2^d$), donc exploration heuristique et coûteuse.

**ANOVA :** $F=S^2_{ent}/S^2_{int}$ (variance **entre** classes sur variance **intra** classe). Plus $F$ est grand, moins la variance entre moyennes s'explique par le seul échantillonnage, plus l'attribut est **indicatif d'un effet de classe**. RELIEF a un principe voisin sans hypothèse de loi.

### Q43. Comparer ACP, ISOMAP et t-SNE.

- **ACP :** méthode **linéaire**, identifie les axes principaux d'inertie en optimisant un critère quadratique → **très sensible aux aberrants** ; sur 60 000 images de chiffres projetées en 2D, les classes se chevauchent (limite d'une méthode linéaire).
- **ISOMAP :** **non linéaire**, préserve les propriétés de la variété sous-jacente en repérant les paires de points proches et en reportant leur distance.
- **t-SNE :** **non linéaire**, à partir de probabilités conditionnelles, projette en dimension 2 ou 3 ; complexité en $m^2$ ($m$ = nombre d'exemples), d'où des optimisations. Sépare bien les chiffres 8×8 ($d=64$) en 2D.

Ce sont des méthodes de **changement d'espace**, à distinguer de la sélection de variables (qui garde un sous-ensemble des variables d'origine).

---

## I. Questions de synthèse

### Q44. Raconter le fil logique du perceptron au deep learning : à chaque étape, quel problème est résolu et quel nouveau problème apparaît ?

1. **Perceptron :** apprend un séparateur linéaire avec garantie de convergence si séparable → mais limite linéaire (XOR).
2. **PMC :** couches cachées + non-linéarités → frontières non linéaires et changement de représentation ; il faut un moyen d'entraîner les couches cachées → **rétropropagation** (descente de gradient + dérivation en chaîne).
3. Nouveaux problèmes : minima locaux, **sur-apprentissage**, choix de l'architecture (compromis biais-variance) → arrêt précoce, validation.
4. **Réseaux profonds / CNN :** pour les images, contraintes de structure (connexions locales, poids partagés) qui réduisent les paramètres et apprennent une **hiérarchie de représentations** → mais besoin d'énormément de données et de calcul.
5. **Peu d'étiquettes :** augmentation, auto-encodage, auto-apprentissage. **Stabilité :** normalisation, dropout.
6. **Paradoxe :** ces réseaux mémorisent n'importe quoi et généralisent pourtant → la théorie classique ne l'explique pas. **Autres architectures** : reservoir, impulsionnels.

### Q45. Relier ce cours aux notions du cours d'introduction : induction, biais inductif, perte, risque empirique/réel, ERM, NFL, PAC.

- **Fonction de perte / ERM :** minimiser l'erreur quadratique moyenne $\frac1m\sum(y-u)^2$ (ou l'entropie croisée) par descente de gradient, c'est **faire de l'ERM** : minimiser le risque empirique sur $\mathcal H$ (la famille de fonctions que le réseau peut représenter).
- **Risque empirique vs réel :** sur-apprentissage = $\hat R_n$ qui baisse pendant que $R$ (validation) remonte ; l'arrêt précoce utilise la validation pour estimer $R$.
- **Biais inductif :** le choix de l'architecture restreint $\mathcal H$. Les CNN encodent un biais fort (localité, partage de poids = invariance par translation) *aligné avec la vérité* pour les images ; mal aligné, il causerait du sous-apprentissage.
- **Régularisation :** dropout, arrêt précoce, augmentation de données, élagage préfèrent certaines hypothèses (même rôle que restreindre $\mathcal H$).
- **Approximateurs universels :** $\mathcal H$ très riche → pose directement le problème de l'induction : comment généraliser avec tant de flexibilité ?
- **NFL / PAC / VC :** sans biais, les données ne disent rien sur les points non vus ; la borne PAC croît avec la complexité de $\mathcal H$. L'expérience de Zhang et al. montre que cette théorie ne suffit pas à expliquer le succès des réseaux profonds.

### Q46. Vrai / Faux (justifier)

- **(a)** « Le perceptron converge toujours. » → **Faux.** Seulement si l'échantillon est linéairement séparable (sinon boucle, ex. XOR).
- **(b)** « Empiler des couches linéaires augmente la capacité du réseau. » → **Faux.** Une composition de transformations linéaires reste linéaire.
- **(c)** « Le $\delta$ d'un neurone caché se calcule à partir des $\delta$ de la couche suivante. » → **Vrai.** $\delta_j=g'(a_j)\sum_k w_{jk}\delta_k$.
- **(d)** « 100 % d'exactitude d'apprentissage garantit une bonne généralisation. » → **Faux.** Zhang et al. : 100 % même avec des étiquettes aléatoires (test ≈ 10 %).
- **(e)** « BatchNorm normalise chaque exemple sur l'ensemble de ses descripteurs. » → **Faux.** C'est LayerNorm. BatchNorm normalise chaque descripteur sur le batch.
- **(f)** « Le dropout éteint aussi des neurones au moment du test. » → **Faux.** Tous les neurones sont réactivés.
- **(g)** « Utiliser 10 bits pour coder 26 lettres en ECOC est un gaspillage. » → **Faux.** La redondance permet de corriger des erreurs.
- **(h)** « Dans une classification à 99 % / 1 %, 99 % d'exactitude prouve que le modèle est bon. » → **Faux.** Il peut toujours répondre « majoritaire ».
- **(i)** « Dans le reservoir computing, tous les poids sont entraînés. » → **Faux.** Seule la couche de sortie l'est (régression linéaire).
- **(j)** « Les statistiques de normalisation peuvent être calculées sur tout le jeu de données, test compris. » → **Faux.** Uniquement sur l'apprentissage.
- **(k)** « L'ACP est robuste aux points aberrants. » → **Faux.** Critère quadratique, très sensible.

### Q47. (Mini-cas) On dispose de 5 000 images médicales étiquetées et de 200 000 non étiquetées, avec 2 % de cas positifs. On veut un classifieur par réseau profond. Proposer une stratégie complète en justifiant chaque choix.

**Plan de réponse :**

1. **Tâche :** classification supervisée d'images → CNN (filtres appris, poids partagés, hiérarchie de représentations), classifieur final complètement connecté, sortie sigmoïde (binaire) ou softmax, perte **entropie croisée**.
2. **Peu d'étiquettes :** pré-entraîner sur les 200 000 images non étiquetées par **auto-encodage** ou **auto-apprentissage** (tâche prétexte), puis affiner sur les 5 000 étiquetées ; **augmentation de données** (rotation, contraste, bruit…) en conservant l'étiquette.
3. **Classes déséquilibrées :** ne pas juger sur l'exactitude (98 % en répondant toujours « négatif ») ; sous-échantillonner, générer des positifs fictifs, ou **pondérer les coûts** (un faux négatif médical est souvent bien plus grave).
4. **Sur-apprentissage :** séparer apprentissage / validation / test ; **arrêt précoce** sur la validation ; **dropout** ; normalisation (BatchNorm) pour stabiliser.
5. **Prétraitement :** normaliser avec les statistiques de l'apprentissage uniquement ; traiter les images aberrantes ; éventuellement revoir le bruit d'étiquetage.
6. **Entraînement :** mode en ligne / mini-batch, pas $\eta$ adapté (moment), plusieurs initialisations contre les minima locaux.
7. **Garde-fous :** 100 % en apprentissage ne prouve rien (Zhang et al.) ; exemples adversariaux et corrélations de contexte (voiture dans la piscine) ; **explicabilité** (Grad-CAM, feature visualization) indispensable en médecine, et prévoir un **rejet** des cas incertains vers un humain (comme les 9 % de rejet des codes postaux).

---

---

# Évaluation de l'apprentissage : questions type partiel corrigées

---

## A. Pourquoi évaluer, cadre général

### Q1. Pourquoi faut-il évaluer un système d'apprentissage ? Quelles trois questions motivent le chapitre ? Quelles deux précautions faut-il garder en tête ?

**Pourquoi :** l'induction est une inférence **faillible** : contrairement à une déduction, une généralisation apprise sur des exemples peut être fausse sur de nouveaux cas. Il faut donc mesurer la qualité de ce qui a été appris.

**Trois questions :** (1) quelle est la performance d'un système sur une tâche ? (2) mon système est-il meilleur qu'un autre ? (3) comment régler ses paramètres (hyperparamètres) ? Ce sont trois problèmes liés mais distincts, avec chacun ses précautions.

**Deux précautions :**

- Les résultats ne mesurent **pas une performance absolue** : ils dépendent du jeu de données (théorème du *no free lunch* : pas de meilleur algorithme dans l'absolu).
- Le **taux d'erreur n'est qu'un critère parmi d'autres** : s'y limiter peut « aveugler » (Bottou, Dessalles) ; intelligibilité et coûts comptent aussi.

### Q2. Distinguer facteurs contrôlables et incontrôlables. Quelle conséquence méthodologique en tirer ?

- **Contrôlables :** l'algorithme, ses méta-paramètres (architecture d'un réseau, $k$ et distance d'un $k$-ppv, prétraitements).
- **Incontrôlables :** le bruit dans les données, le **tirage aléatoire** des données (quels exemples en apprentissage/validation/test), le caractère aléatoire de l'algorithme (initialisation des poids).

**Conséquence :** ces facteurs introduisent de la variabilité, donc il faut **répéter les expériences** pour obtenir une distribution des performances plutôt que de se fier à une mesure isolée (qui peut être due à un tirage ou une initialisation favorable).

### Q3. Comment optimiser les hyperparamètres ? Quel piège menace cette optimisation et comment s'en protéger ?

**Méthodes :** recherche exhaustive sur grille (*grid search*) ; optimisation bayésienne (on modélise la performance en fonction des hyperparamètres et on choisit intelligemment les points à tester) ; optimisation par gradient (si la performance est différentiable par rapport aux hyperparamètres) ; algorithmes évolutionnaires.

**Piège :** l'optimisation se fait sur l'ensemble de **validation**, donc elle est elle-même sujette au **sur-ajustement** : en essayant beaucoup de combinaisons, on en trouve une qui « colle » à la validation par hasard.

**Protection :** garder un **ensemble de test séparé**, jamais utilisé pour le réglage, pour l'estimation finale. *(Complément : la validation croisée imbriquée applique le même principe quand on a peu de données.)*

### Q4. Pourquoi ne pas se limiter au taux d'erreur ? Citer les autres critères d'évaluation et la critique des benchmarks.

**Autres critères :**

- **Intelligibilité** des hypothèses produites : un arbre de décision, qu'on peut expliquer, peut être préféré à un modèle plus performant mais opaque (« exit les réseaux de neurones » quand l'interprétabilité prime) ; or l'intelligibilité et la généralisation ne vont pas toujours ensemble.
- **Coûts :** préparation des données, calcul (coût d'une passe et nombre de passes), expertise en apprentissage automatique, expertise du domaine pour interpréter.

**Critique des benchmarks (fiche) :** l'évaluation moderne (LLM) repose trop sur eux. Ils filtrent bien l'état de l'art mais restreignent la vision globale (on cherche « sous le lampadaire » ce qui est facile à mesurer) et évaluent mal la créativité ou l'amélioration continue des représentations internes.

---

## B. Erreurs, découpage des données, courbes

### Q5. Définir erreur vraie et erreur de test. Pourquoi la première est-elle inaccessible ? Quel est l'enjeu central du chapitre ?

$$
e_D=\int_D |y-f(x,\theta)|\,p(x,y)\,dx\,dy \qquad\quad \hat e_S=\frac1m\sum_{(x,y)\in T}|y-f(x,\theta)|
$$

- **Erreur vraie (risque réel) $e_D$ :** intégrale sur **toutes** les données possibles, pondérée par $p(x,y)$. Inaccessible : on n'a jamais « toutes les données ».
- **Erreur de test $\hat e_S$ :** moyenne sur un échantillon fini $T$ de $m$ exemples de test. Seule quantité calculable ; c'est un **estimateur imparfait** de $e_D$.

**Attention :** ne pas confondre avec l'erreur **d'entraînement** (sur les exemples utilisés pour apprendre), qui est optimiste.

**Enjeu :** toute la méthodologie vise à estimer à quel point $\hat e_S$ est représentative de $e_D$, et avec quelle marge d'incertitude.

### Q6. Donner le rôle de chaque ensemble (apprentissage, validation, test). Énoncer la règle d'or. Qu'est-ce que la fuite de données ?

- **Apprentissage :** ajuster les **paramètres** internes (poids).
- **Validation :** régler les **hyperparamètres** et décider quand arrêter l'apprentissage.
- **Test :** estimer, **une seule fois, à la toute fin**, la performance réelle du modèle final.

**Règle d'or :** chaque ensemble ne sert qu'à un seul rôle. Utiliser le test pour régler quoi que ce soit, même indirectement, invalide son rôle d'estimateur non biaisé (estimation **optimiste**). C'est la **fuite de données** (*data leakage*) : de l'information du test influence le modèle.

### Q7. Décrire les deux comportements des courbes d'erreur : (a) le cas asymptotique avec beaucoup de données ; (b) le sur-apprentissage au fil de l'entraînement. Qu'est-ce que l'arrêt précoce ? Que changerait-on avec 2 000 exemples au lieu de 1 000 ?

**(a) Asymptotique :** quand le nombre d'exemples $n$ croît, $E_{test}$ diminue et converge vers une limite $a$, tandis que $E_{train}$ (très basse au départ car le modèle « connaît » ses exemples) **remonte** légèrement vers la même limite $a$. Les deux courbes se rejoignent : le modèle généralise aussi bien qu'il mémorise.

**(b) Sur-apprentissage :** au fil de l'entraînement, $E_{train}$ diminue continûment, mais $E_{test}$/$E_{val}$ diminue, atteint un **minimum**, puis remonte.

**Arrêt précoce :** on arrête quand l'erreur de **validation** est minimale (pas celle d'apprentissage).

**Avec 2 000 exemples :** plus de données retarde le sur-apprentissage : l'écart entre les courbes se réduit, le minimum de validation est plus bas et plus tardif.

---

## C. Matrice de confusion et indicateurs

### Q8. Écrire la matrice de confusion binaire et définir VP, FP, FN, VN. Donner les formules de sensibilité, spécificité, précision, taux d'erreur, accuracy, FP-rate, FN-rate. **(Calcul)** Sur 200 patients : VP=30, FN=10, FP=20, VN=140. Calculer tous les indicateurs et le F1. Que vaut l'accuracy d'un classifieur qui répond toujours « négatif » ?

- **VP :** prédit + et réel + ; **FP :** prédit + mais réel − ; **FN :** prédit − mais réel + ; **VN :** prédit − et réel −.
- Sensibilité (rappel, TPR) $=VP/(VP+FN)$ ; spécificité (TNR) $=VN/(VN+FP)$ ; précision $=VP/(VP+FP)$ ; taux d'erreur $=(FP+FN)/m$ ; accuracy $=1-$ taux d'erreur $=(VP+VN)/m$ ; FP-rate $=FP/(FP+VN)=1-$ spécificité ; FN-rate $=FN/(VP+FN)=1-$ sensibilité.

**Calcul :**

- Accuracy $=(30+140)/200=0.85$ ; taux d'erreur $=0.15$.
- Sensibilité $=30/40=0.75$ ; spécificité $=140/160=0.875$ ; FP-rate $=20/160=0.125$ ; FN-rate $=0.25$.
- Précision $=30/50=0.60$.
- $F_1=\dfrac{2VP}{2VP+FP+FN}=\dfrac{60}{90}\approx0.667$ (même résultat avec $2PR/(P+R)=2\cdot0.6\cdot0.75/1.35$).
- Classifieur « toujours négatif » : accuracy $=160/200=0.80$, mais sensibilité $=0$ : il ne détecte **aucun** malade. L'accuracy est donc trompeuse sur classes déséquilibrées.

### Q9. (SVM vs Naive Bayes) Sur un test de 165 exemples (105 « good », 60 « bad ») : SVM : VP=94, FP=37, FN=11, VN=23 ; Naive Bayes : VP=72, FP=29, FN=33, VN=31 (« good » = positif). **(a)** Recalculer l'accuracy, la sensibilité, la spécificité, la précision et le F1 de chacun. **(b)** Quel est le meilleur classifieur ? Discuter.

**(a)**

|                     | SVM                                                                   | Naive Bayes            |
| ------------------- | --------------------------------------------------------------------- | ---------------------- |
| Accuracy            | $117/165=70.9\,\%$ *(le cours indique 70.1 %, coquille probable)* | $103/165=62.4\,\%$   |
| Sensibilité (good) | $94/105=89.5\,\%$                                                   | $72/105=68.6\,\%$    |
| Spécificité (bad) | $23/60=38.3\,\%$                                                    | $31/60=51.7\,\%$     |
| Précision (good)   | $94/131=71.8\,\%$                                                   | $72/101=71.3\,\%$    |
| $F_1$ (good)      | $188/236\approx0.80$                                                | $144/206\approx0.70$ |

**(b)** Le SVM « gagne » en accuracy (l'« apparent best »), mais il classe **131** exemples en « good » contre seulement 34 en « bad », alors que le test en contient 105 / 60 : il est **biaisé vers la classe majoritaire** et son accuracy est gonflée. Naive Bayes répartit mieux (101 / 64). Si la classe importante est « bad » (détecter un défaut de paiement par exemple), la spécificité compte : 51.7 % pour Naive Bayes contre 38.3 % pour le SVM, et Naive Bayes peut être le meilleur choix malgré une accuracy plus faible.
**Leçon :** l'accuracy globale masque la répartition des erreurs entre classes ; il faut toujours regarder la matrice complète et choisir la métrique selon l'objectif.

### Q10. Que mesurent le rappel et la précision ? Pourquoi les combiner par une moyenne harmonique ? Écrire la F-mesure générale. **(Calcul)** Précision = 100 %, rappel = 1 % : calculer le F1 et le comparer à la moyenne arithmétique.

- **Rappel :** parmi tous les exemples réellement positifs, quelle proportion a été retrouvée ? **Précision :** parmi ceux annoncés positifs, quelle proportion l'était vraiment ? Un modèle peut avoir un excellent rappel et une mauvaise précision (beaucoup de fausses alertes), et réciproquement. Ils évoluent souvent en sens opposé.
- $F=\dfrac{(\beta^2+1)\,P\,R}{\beta^2P+R}$ ; $F_1$ ($\beta=1$) : $\dfrac{2PR}{P+R}$.
- La **moyenne harmonique pénalise fortement** le cas où l'un des deux est très faible, ce que ne fait pas la moyenne arithmétique.
- Calcul : $F_1=2\times1\times0.01/1.01\approx0.0198\approx2\,\%$, contre $50\,\%$ en moyenne arithmétique. Le $F_1$ reflète mieux l'inutilité pratique d'un tel modèle.

### Q11. Un camarade dit : « pour limiter les fausses alertes (FPR), je maximise la précision ». Que dire ? Quelle différence entre FPR et précision ? **(Calcul)** Un classifieur a TPR = 0.9 et FPR = 0.1. Calculer sa précision avec 100 positifs / 900 négatifs, puis avec 100 positifs / 9 900 négatifs.

- Le **FPR** $=FP/(FP+VN)=1-$ spécificité ne dépend que des **négatifs réels**. C'est la métrique qui cible directement les fausses alertes (à minimiser, ou spécificité à maximiser).
- La **précision** $=VP/(VP+FP)$ dépend aussi de la **proportion de positifs** dans les données. Maximiser la précision réduit bien les FP en pratique, mais ce n'est pas équivalent : la précision peut s'effondrer sans que le FPR bouge.

**Calcul :**

- 100 / 900 : VP$=90$, FP$=0.1\times900=90$ → précision $=90/180=0.50$.
- 100 / 9 900 : VP$=90$, FP$=990$ → précision $=90/1080\approx0.083$.

Même classifieur (mêmes TPR et FPR), précision divisée par 6. (La fiche répond « Précision » à cette question ; c'est acceptable comme réponse pratique, mais le FPR/la spécificité est la mesure directe.)

### Q12. Comment la matrice de confusion se généralise-t-elle à plusieurs classes ? À quoi sert-elle ?

Chaque ligne = classe **réelle**, chaque colonne = classe **prédite**, la cellule $(i,j)$ = proportion d'exemples de la classe réelle $i$ classés en $j$. La **diagonale** donne les bonnes classifications ; le reste indique les **confusions** (ex. du cours : 14 % des papillons pris pour des poissons).
Utilité : voir quelles classes sont confondues et où concentrer les efforts (plus de données pour ces classes, meilleurs descripteurs).

---

## D. La courbe ROC

### Q13. Distinguer erreurs de type 1 et de type 2. Comment le seuil de décision arbitre-t-il entre les deux ? Pourquoi aucun seuil ne minimise les deux à la fois ?

- **Type 1 ($\alpha$) :** faux positifs (accepter l'hypothèse alors qu'elle est fausse).
- **Type 2 ($\beta$) :** faux négatifs (rejeter l'hypothèse alors qu'elle est vraie).

Un classifieur produit un **score** ; le seuil $T$ décide de la classe. Les distributions de scores des deux classes se **recouvrent**. Seuil **sévère** ($T$ élevé) : moins de FP mais plus de FN. Seuil **laxiste** ($T$ bas) : l'inverse. Déplacer $T$ change les deux erreurs en sens opposé, donc aucun seuil ne minimise les deux. Le bon choix dépend du **coût relatif** des erreurs (en diagnostic médical, un faux négatif est souvent bien plus coûteux).

### Q14. Définir la courbe ROC et expliquer sa construction. **(Calcul)** Huit exemples triés par score décroissant, avec leur vraie classe : $+,+,-,+,-,+,-,-$. Tracer les points de la courbe et calculer l'AUC.

**Définition :** ROC = *Receiver Operating Characteristic*. On trace le **TPR** (ordonnée) en fonction du **FPR** (abscisse) pour toutes les valeurs du seuil $T$.
**Construction :** trier les exemples par score décroissant, abaisser le seuil de la valeur maximale à la minimale ; un vrai positif fait monter la courbe d'un cran, un faux positif la fait aller à droite. Elle part de $(0,0)$ (seuil maximal, aucune prédiction positive) et arrive en $(1,1)$ (toutes prédites positives). C'est une courbe en escalier.

**Calcul :** $P=4$, $N=4$ (pas de $1/4$ sur chaque axe). Points : $(0,0)\to(0,0.25)\to(0,0.5)\to(0.25,0.5)\to(0.25,0.75)\to(0.5,0.75)\to(0.5,1)\to(0.75,1)\to(1,1)$.
**AUC** (aire sous la courbe) : $0.25\times0.5+0.25\times0.75+0.25\times1+0.25\times1=0.8125$.
Vérification : sur les $4\times4=16$ paires (positif, négatif), 13 sont bien ordonnées (le positif a un score supérieur au négatif) → $13/16=0.8125$ *(complément : c'est l'interprétation probabiliste de l'AUC)*.

### Q15. Comment interpréter une courbe ROC ? Que représentent la diagonale, le point (0,1), un point isolé sur le graphe, et l'AUC ? Comment comparer deux classifieurs ?

- **Diagonale :** classifieur au niveau du hasard (TPR = FPR), AUC $=0.5$.
- **Point $(0,1)$ :** classifieur **parfait** (tous les positifs détectés, aucune fausse alerte). Le meilleur modèle a une forme de « rectangle ».
- Plus la courbe s'éloigne de la diagonale vers le coin supérieur gauche, meilleur est le classifieur.
- **Un point isolé** = performance pour **un seuil fixé** ; **la courbe entière** = toutes les performances possibles en faisant varier le seuil.
- **AUC :** aire sous la courbe, **performance globale indépendante du seuil**, entre 0.5 et 1.
- **Comparaison :** un classifieur dont la courbe **domine entièrement** l'autre (toujours au-dessus) est meilleur quel que soit le compromis FP/FN voulu. Si les courbes se croisent, le meilleur dépend de la zone de fonctionnement (*complément*) ; l'AUC seule peut alors masquer une différence.

---

## E. Intervalles de confiance

### Q16. Définir un intervalle de confiance. Sur quel modèle probabiliste repose le calcul pour l'erreur ? Quelle approximation et quelle condition ? Écrire la formule.

**Définition :** un intervalle de confiance à $N\,\%$ pour une quantité est l'intervalle dans lequel sa valeur est attendue avec une probabilité de $N\,\%$.

**Modèle :** chaque exemple de test est un tirage indépendant avec probabilité $p$ de se tromper ; le nombre d'erreurs $r$ sur $n$ exemples suit une **loi binomiale** : $P(r)=\binom nr p^r(1-p)^{n-r}$, d'espérance $np$ et variance $np(1-p)$.

**Approximation normale :** si $np(1-p)\ge5$, la binomiale est approchée par une loi normale de même moyenne et variance, d'où $\sigma_T\approx\sqrt{\hat e(1-\hat e)/n}$.

**Formule :**

$$
e_D\in\Big[\hat e_S\pm z_N\sqrt{\tfrac{\hat e_S(1-\hat e_S)}{m}}\Big]
$$

$z_N$ : 50 % → 0.67 ; 68 % → 1.0 ; 80 % → 1.28 ; 90 % → 1.64 ; **95 % → 1.96** ; 98 % → 2.33 ; 99 % → 2.58. Condition supplémentaire : $m\ge30$ exemples de test tirés indépendamment.

### Q17. (Calcul) Un modèle a une erreur de test de 0.2 sur $m=100$ exemples. **(a)** Intervalle de confiance à 95 %. **(b)** Idem avec $m=400$ (même erreur). **(c)** Interpréter. **(d)** Pourquoi un petit échantillon de test est-il problématique ? (Exemple du cours : 12 erreurs sur 40.)

**(a)** Condition : $100\times0.2\times0.8=16\ge5$ ✓. $\sigma=\sqrt{0.16/100}=0.04$ ; $1.96\times0.04\approx0.078$ → $e_D\in[0.122\,;\,0.278]$.
**(b)** $\sigma=\sqrt{0.16/400}=0.02$ ; $1.96\times0.02\approx0.039$ → $e_D\in[0.161\,;\,0.239]$.
**(c)** Multiplier $m$ par 4 **divise la largeur par 2** (en $1/\sqrt m$). Pour gagner une décimale de précision, il faut beaucoup plus de données de test.
**(d)** Exemple du cours : $12/40=0.3$ ; $1.96\sqrt{0.3\times0.7/40}\approx0.14$ → $[0.16\,;\,0.44]$ : un taux d'erreur mesuré de 30 % peut correspondre à une erreur réelle de 16 % à 44 %. Une erreur sans intervalle de confiance ne veut pas dire grand-chose.

### Q18. Comment comparer deux modèles avec un intervalle de confiance ? **(Calcul)** $\hat e_S(\theta_1)=0.20$ et $\hat e_S(\theta_2)=0.30$, chacun sur $m=100$ exemples. Peut-on dire que $\theta_1$ est meilleur ? Quelle réserve le cours fait-il ?

On estime $d=e_D(\theta_1)-e_D(\theta_2)$ par $\hat d=\hat e_S(\theta_1)-\hat e_S(\theta_2)$ ; la différence de deux lois normales est normale, d'où

$$
\hat d\pm1.96\sqrt{\frac{\hat e_1(1-\hat e_1)}{m_1}+\frac{\hat e_2(1-\hat e_2)}{m_2}}
$$

**Calcul :** $\hat d=-0.10$ ; $\sqrt{0.0016+0.0021}=\sqrt{0.0037}\approx0.0608$ ; $1.96\times0.0608\approx0.119$ → $d\in[-0.219\,;\,0.019]$.
L'intervalle **contient 0** : on ne peut pas conclure que $\theta_1$ est significativement meilleur, malgré 10 points d'écart.

**Réserve :** il ne faudrait normalement pas tester les deux modèles sur le même ensemble de test (l'indépendance supposée serait violée). En pratique, c'est pourtant courant et donne une meilleure estimation de la variance : on utilise alors des **tests appariés** (*paired t-tests*).
*(Fiche :)* une différence de 1 % d'accuracy n'a aucun sens si l'intervalle de confiance est de ±5 %.

---

## F. Quand les données sont rares

### Q19. Avec 200 exemples seulement, discuter les découpages 50/50 et 80/20. Quel est le dilemme ?

- **50/50 :** la moitié des données est réservée au test, le modèle entraîné sur 100 exemples est moins bon que celui qu'on aurait avec les 200 : on **sous-estime** l'exactitude de généralisation (on mesure un modèle sous-entraîné).
- **80/20 :** test de seulement 40 exemples : intervalle de confiance **large**, et des tirages différents du test (*hold-out sets*) peuvent donner des résultats très différents (**haute variance**).

**Dilemme :** plus on garde de données pour l'apprentissage, moins l'estimation de l'erreur est fiable ; plus on en garde pour le test, moins le modèle est représentatif du modèle final. La validation croisée, le leave-one-out et le bootstrap visent à le résoudre.

### Q20. Décrire la validation croisée à $k$ plis. **(Calcul)** $m=200$, $k=10$ : tailles d'apprentissage et de test, nombre d'entraînements, nombre de fois où chaque exemple est testé. Quels sont ses atouts et ses limites ?

On divise les données en $k$ plis de taille égale ; on répète $k$ fois : entraîner sur $k-1$ plis, tester sur le pli restant ; l'erreur finale est $\frac1k\sum_i\text{erreur}_i$. Chaque pli sert **une fois et une seule** de test, donc toutes les données servent à l'apprentissage **et** au test (à des moments différents) sans mélanger les rôles pour un même exemple à une même itération.

**Calcul :** apprentissage $=180$, test $=20$ ; 10 entraînements ; chaque exemple est testé exactement 1 fois (200 prédictions de test au total).

**Limites :** coût ($k$ entraînements) ; biais légèrement pessimiste car les modèles voient $(k-1)/k$ des données *(complément)* ; les erreurs des plis ne sont pas indépendantes (voir Q23).

### Q21. Définir le leave-one-out (LOO). Pourquoi a-t-il un faible biais mais une haute variance ? Quand sous-estime-t-il l'erreur ?

Cas particulier de CV avec $k=m$ : à chaque itération, un seul exemple sert de test.

- **Faible biais :** on entraîne sur $m-1$ exemples, donc très proche du modèle final.
- **Haute variance :** chaque test ne porte que sur un exemple (erreur 0 ou 1), très instable.
- **Coût :** $m$ entraînements, très coûteux.
- **Sous-estime l'erreur** si les données ne sont pas vraiment **i.i.d.** (exemples très similaires ou corrélés présents plusieurs fois : l'exemple testé a un jumeau dans l'apprentissage) (Guyon & Elisseeff 2003).

### Q22. Décrire le bootstrap. Montrer que l'ensemble d'apprentissage contient environ 63.2 % d'exemples distincts. Pourquoi l'estimation est-elle biaisée ? Écrire l'estimateur corrigé.

**Principe :** on tire **avec remise** $m$ exemples pour former l'ensemble d'apprentissage ; les exemples non tirés forment le test ; on répète et on moyenne. Certains exemples apparaissent plusieurs fois, d'autres jamais. *(Fiche : on crée $k$ ensembles d'apprentissage de taille $m$.)*

**63.2 % :** la probabilité qu'un exemple donné ne soit **jamais** tiré en $m$ tirages vaut $(1-1/m)^m\to e^{-1}\approx0.368$. Donc la fraction d'exemples distincts tirés est $1-0.368\approx0.632$.

**Biais :** le modèle est entraîné sur moins de diversité réelle (63.2 % des exemples distincts) donc moins bon : l'erreur sur les exemples non tirés est **surestimée** (pessimiste).

**Correction (estimateur .632) :**

$$
\hat R_{réel}(h)=0.632\times\bar P_1+0.368\times\bar P_2
$$

$\bar P_1$ : erreur moyenne sur les échantillons de test (exemples non tirés), pessimiste ; $\bar P_2$ : erreur du modèle final réévalué sur **l'ensemble complet**, optimiste. Le mélange compense les deux biais. *(Le PDF écrit 0.636, coquille : les poids doivent sommer à 1.)*

### Q23. Pourquoi les intervalles de confiance calculés « naïvement » après CV, LOO ou bootstrap sont-ils trop optimistes ? Que faire du modèle après une validation croisée ?

Le calcul classique suppose des estimations **indépendantes**. Or dans k-fold, LOO et bootstrap, les ensembles d'apprentissage successifs partagent en grande partie les mêmes données (en 10-CV, environ 89 % de données communes entre deux entraînements) : les erreurs sont **statistiquement dépendantes**, donc la variance est sous-estimée et les intervalles sont **trop étroits**.

**Après la CV (fiche) :** soit on fait **voter** les modèles des $k$ plis, soit on **ré-entraîne sur toutes les données** avec les hyperparamètres trouvés (la norme). Risques : l'entraînement final peut tomber dans un minimum local différent ; et surtout on n'a plus de jeu de test indépendant pour évaluer *ce* modèle, on suppose que sa performance est au moins celle de la moyenne de la CV.

---

## G. Tests statistiques de comparaison

### Q24. Expliquer le principe d'un test statistique de comparaison : hypothèse nulle, p-value, seuil $\alpha$. Quelle erreur le seuil $\alpha$ contrôle-t-il ?

On veut savoir si A est **vraiment** meilleur que B ou si l'écart est dû au hasard. **Hypothèse nulle $H_0$ :** « pas de différence de performance ». On calcule une statistique de test et une **p-value** (*complément :* probabilité d'observer un écart au moins aussi grand si $H_0$ était vraie). Si p-value $<\alpha$ (souvent 0.05), on rejette $H_0$ : la différence est significative. $\alpha$ contrôle le risque d'erreur de type 1 : déclarer une différence qui n'existe pas.

### Q25. Le t-test couplé : écrire la statistique, les hypothèses et la règle de décision. Quel est le piège dans une 10-CV ? Que recommande Dietterich ? **(Calcul)** Sur $k=5$ plis, les différences d'erreur $\Delta_i=e_i^A-e_i^B$ valent $0.02,\ 0.04,\ 0.01,\ 0.03,\ 0.05$. Conclure à $\alpha=0.05$ ($t_{0.025,4}=2.776$).

On calcule $\Delta_i$ sur chaque pli, avec $\mu$ la moyenne et $\sigma$ l'écart-type des $\Delta_i$ :

$$
\tau_t=\frac{\sqrt k\,\mu}{\sigma}
$$

Hypothèse : la distribution des différences moyennes est approximativement normale. On rejette $H_0$ si $\tau_t\notin[-t_{\alpha/2,k-1},\,t_{\alpha/2,k-1}]$.

**Piège :** les $\Delta_i$ d'une 10-CV ne sont **pas i.i.d.** (les entraînements se chevauchent) → le t-test est mal fondé. **Dietterich (1998)** recommande une **$5\times2$ CV couplée**.

**Calcul :** $\mu=0.03$ ; écarts : $-0.01,0.01,-0.02,0,0.02$ ; somme des carrés $=0.001$ ; $\sigma=\sqrt{0.001/4}\approx0.0158$ (écart-type empirique, diviseur $k-1$). $\tau_t=\sqrt5\times0.03/0.0158\approx4.24$. Comme $4.24>2.776$ : on **rejette $H_0$**, A et B sont significativement différents.

### Q26. Le test de McNemar : principe, formule, statistique. **(Calcul)** A se trompe et B a bon sur 25 exemples ; A a bon et B se trompe sur 10. Conclure à 5 % ($\chi^2_{0.05,1}=3.84$).

**Principe :** on compare les classifications de A et B sur le **même** jeu de test, en ne regardant que les exemples où ils sont **en désaccord**. $e_{01}$ : A faux / B bon ; $e_{10}$ : A bon / B faux.

$$
\tau_{\chi^2}=\frac{(|e_{01}-e_{10}|-1)^2}{e_{01}+e_{10}}
$$

Il suit une loi du $\chi^2$ ; si $\tau_{\chi^2}>\chi^2_\alpha$ on rejette $H_0$. Intuition : sous $H_0$ chaque désaccord est équiprobable entre « A gagne » et « B gagne ».

**Calcul :** $e_{01}=25$, $e_{10}=10$ : $\tau=(|25-10|-1)^2/35=14^2/35=196/35=5.6>3.84$ → **on rejette $H_0$** : les modèles diffèrent significativement (B meilleur ici).

### Q27. Pourquoi ne peut-on pas utiliser le t-test pour comparer deux algorithmes sur plusieurs jeux de données ? Décrire le test de Wilcoxon. **(Calcul)** Différences de performance $d_i=\text{Perf}(A)-\text{Perf}(B)$ sur 6 jeux : $+0.02,\ -0.01,\ +0.03,\ +0.05,\ -0.04,\ +0.06$. Calculer $T_{wilcox}$.

Sur plusieurs jeux de données de natures différentes, les taux d'erreur **ne suivent plus une loi normale** : il faut un test **non paramétrique**, fondé sur les **rangs** et non sur les valeurs.

**Wilcoxon (rang signé) :** (1) calculer $d_i$ ; (2) ranger les $|d_i|$ ; (3) sommer les rangs des $d_i>0$ ($W_{s1}$) et des $d_i<0$ ($W_{s2}$) ; (4) $T_{wilcox}=\min(W_{s1},W_{s2})$, comparé à une table critique (approximation normale si $n\ge25$).

**Calcul :** rangs des $|d_i|$ : $0.01\to1$, $0.02\to2$, $0.03\to3$, $0.04\to4$, $0.05\to5$, $0.06\to6$. $W_{s1}=2+3+5+6=16$ ; $W_{s2}=1+4=5$ ; $T=\min=5$. Pour $n=6$ en bilatéral à 5 %, la valeur critique de la table usuelle est 0 *(à vérifier sur la table de ton cours)* : $5>0$, donc on **ne rejette pas $H_0$** : avec seulement 6 jeux, le test a très peu de puissance.

### Q28. Pourquoi comparer $N$ algorithmes deux à deux pose-t-il problème ? Décrire le test de Nemenyi. **(Calcul)** $k=3$ algorithmes, $n=10$ jeux, rangs moyens A=1.5, B=2.0, C=2.5, $q_{0.05}=2.343$ pour $k=3$. Quelle différence critique ? Quelles paires diffèrent ? Quel piège dans l'interprétation du rang moyen ?

**Problème des tests multiples :** faire de nombreuses comparaisons deux à deux gonfle la probabilité de trouver au moins une « différence » par hasard. On utilise un test **post hoc** appliqué après qu'un test global a détecté une différence *(complément : test de Friedman)*.

**Nemenyi :** (1) rang moyen $\bar R_j$ de chaque algorithme sur tous les jeux ; (2) statistique pour deux algorithmes :

$$
q=\frac{\bar R_{j_1}-\bar R_{j_2}}{\sqrt{k(k+1)/(6n)}}
$$

(3) on rejette $H_0$ si $q>q_\alpha$. On définit la **différence critique** $CD=q_\alpha\sqrt{k(k+1)/(6n)}$ : si l'écart de rang moyen dépasse $CD$, les algos diffèrent significativement (visualisé par des barres reliant les algos équivalents).

**Calcul :** $\sqrt{3\times4/(6\times10)}=\sqrt{0.2}\approx0.447$ ; $CD=2.343\times0.447\approx1.05$. Les écarts de rang valent 0.5 (A-B), 0.5 (B-C), 1.0 (A-C), tous $<1.05$ : **aucune paire n'est significativement différente**, même A vs C (q$\approx2.24<2.343$).

**Piège :** le rang moyen ne donne **aucune information sur l'écart de performance absolu** : un classé 2ᵉ peut être à 0.001 % du 1ᵉʳ, ou très loin.

### Q29. (Choix du test) Quel outil utiliser dans chaque cas, et pourquoi ?

- **(a)** Estimer l'erreur vraie d'un seul algorithme avec un test de 200 exemples.
- **(b)** Comparer A et B sur un jeu de données, avec une validation croisée.
- **(c)** Comparer A et B sur un même ensemble de test, en regardant leurs désaccords.
- **(d)** Comparer A et B sur 15 jeux de données.
- **(e)** Comparer 5 algorithmes sur 20 jeux de données.
- **(a)** **Intervalle de confiance** sur l'erreur de test (binomiale → normale, $m\ge30$).
- **(b)** **T-test couplé** sur les différences par pli (idéalement $5\times2$ CV, car la 10-CV viole l'indépendance).
- **(c)** **Test de McNemar** (ne considère que les exemples où A et B diffèrent).
- **(d)** **Wilcoxon** (non paramétrique : pas de normalité sur des jeux hétérogènes ; rangs des différences).
- **(e)** Test global puis **Nemenyi** (tests multiples : comparer 2 à 2 fausserait les probabilités).

---

## H. Synthèse

### Q30. Expliquer ces pièges classiques : (a) mélanger validation et test ; (b) oublier les intervalles de confiance ; (c) se fier au rang moyen ; (d) mal choisir sa métrique.

- **(a)** Régler les hyperparamètres sur le test est la pire erreur (fuite de données) : l'estimation devient optimiste. Le test ne doit être vu qu'**une seule fois, à la fin**.
- **(b)** Une différence de 1 % d'accuracy n'a aucun sens si l'incertitude est de ±5 % : toujours accompagner une erreur d'un intervalle de confiance (Q17-18).
- **(c)** Le rang moyen (Nemenyi) ignore l'écart de performance absolue.
- **(d)** Utiliser l'accuracy sur un jeu déséquilibré (99 % / 1 %) : le modèle qui répond toujours « majoritaire » a 99 % d'accuracy mais est inutile ; regarder sensibilité, précision, F-mesure, ROC.

### Q31. Raconter le fil logique du chapitre : de « pourquoi évaluer » à « comparer des algorithmes ». À chaque étape, quel problème est résolu et quel nouveau problème apparaît ?

1. **Induction faillible** → il faut évaluer ; $e_D$ est inaccessible → on estime avec $\hat e_S$ sur un échantillon de **test**.
2. Éviter le sur-apprentissage et la fuite : découper en **apprentissage / validation / test**.
3. L'erreur seule trompe (déséquilibre) → **matrice de confusion**, sensibilité/précision/F-mesure ; un seuil arbitre FP/FN → **ROC/AUC**.
4. $\hat e_S$ est incertaine → **intervalles de confiance** (binomiale → normale) → il faut beaucoup de données de test.
5. Données rares → **CV, LOO, bootstrap**, mais estimations dépendantes → IC naïfs trop étroits.
6. Comparer des algorithmes : l'écart est-il dû au hasard ? → **tests** (t-test couplé / McNemar sur un jeu ; Wilcoxon sur plusieurs jeux ; Nemenyi pour $N$ algos), chacun avec ses hypothèses.
7. Garder l'esprit critique : critères autres que l'erreur, benchmarks limités.

### Q32. Relier ce chapitre aux cours précédents (introduction, réseaux de neurones).

- **Cours 1 :** $e_D$ est le **risque réel**, $\hat e_S$ un estimateur sur échantillon indépendant ; l'erreur d'entraînement (risque empirique minimisé par l'ERM) est optimiste → d'où la nécessité du jeu de test. L'**induction faillible** et le **NFL** justifient l'évaluation et la mise en garde « pas de meilleur algorithme dans l'absolu ». La borne PAC donne une garantie théorique ; l'évaluation donne une estimation empirique avec intervalle de confiance.
- **Cours 2 :** sur-apprentissage et **arrêt précoce** (validation) ; les **hyperparamètres** (architecture, pas $\eta$, dropout) se règlent sur la validation ; l'**initialisation aléatoire** et les minima locaux sont des facteurs incontrôlables → répéter les expériences ; l'exemple de Zhang et al. (100 % en apprentissage sans généralisation) rappelle qu'une erreur d'entraînement ne prouve rien ; classes déséquilibrées et piège de l'accuracy déjà vus en prétraitement.

### Q33. Vrai / Faux (justifier)

- **(a)** « On peut régler les hyperparamètres sur l'ensemble de test. » → **Faux.** C'est le rôle de la validation ; le test sert à la fin.
- **(b)** « La précision vaut $VP/(VP+FN)$. » → **Faux.** C'est le rappel ; la précision est $VP/(VP+FP)$.
- **(c)** « Le $F_1$ est la moyenne arithmétique de la précision et du rappel. » → **Faux.** Moyenne harmonique, qui pénalise un score très faible.
- **(d)** « Une courbe ROC sur la diagonale correspond à un classifieur au niveau du hasard. » → **Vrai.** AUC $=0.5$.
- **(e)** « Passer de 100 à 400 exemples de test divise par 2 la largeur de l'intervalle de confiance. » → **Vrai.** En $1/\sqrt m$.
- **(f)** « Le leave-one-out a un faible biais mais une variance élevée. » → **Vrai.**
- **(g)** « Le bootstrap tire les exemples sans remise. » → **Faux.** Avec remise.
- **(h)** « Le t-test couplé sur une 10-CV respecte parfaitement l'indépendance. » → **Faux.** Les entraînements se chevauchent ; on préfère la $5\times2$ CV.
- **(i)** « Wilcoxon suppose que les erreurs suivent une loi normale. » → **Faux.** Test non paramétrique sur les rangs.
- **(j)** « Le rang moyen de Nemenyi renseigne sur l'écart absolu de performance. » → **Faux.**
- **(k)** « 99 % d'accuracy sur un jeu 99/1 prouve que le modèle est bon. » → **Faux.** Il peut toujours répondre « majoritaire ».
- **(l)** « Les intervalles de confiance issus d'une k-fold CV sont généralement trop étroits. » → **Vrai.** Estimations dépendantes.

### Q34. (Mini-cas) On dispose de 300 dossiers patients, dont 10 % de positifs. On veut comparer trois algorithmes (arbre, $k$-ppv, réseau de neurones), régler leurs hyperparamètres, puis annoncer une performance fiable. Proposer un protocole complet en justifiant chaque choix.

**Plan de réponse :**

1. **Données rares :** un simple 50/50 ou 80/20 est instable (Q19) ; utiliser une **validation croisée** (k-fold, ou $5\times2$ si on veut un test d'hypothèse) pour estimer les performances.
2. **Hyperparamètres :** recherche sur grille (ou bayésienne) sur la **validation** ; garder un **ensemble de test** (ou une CV imbriquée *(complément)*) jamais utilisé pour le réglage (sinon estimation optimiste).
3. **Facteurs incontrôlables :** répéter avec plusieurs tirages et initialisations (réseau) et regarder la distribution, pas une seule mesure.
4. **Métriques :** pas l'accuracy seule (90 % en répondant « négatif ») ; matrice de confusion, **sensibilité**, spécificité, précision, **$F_1$**, **ROC/AUC** pour comparer indépendamment du seuil ; choisir le seuil selon le coût relatif FP/FN (en médecine, un faux négatif est souvent plus grave).
5. **Incertitude :** donner un **intervalle de confiance** (avec peu de données de test il sera large, Q17) ; rappeler que les IC issus de CV sont trop optimistes (Q23).
6. **Comparer les trois algorithmes :** sur un seul jeu → t-test couplé ($5\times2$ CV) ou McNemar pour deux à deux ; ne pas multiplier les comparaisons sans correction (Nemenyi nécessite plusieurs jeux de données).
7. **Au-delà de l'erreur :** intelligibilité (l'arbre est explicable), coût de calcul, coût d'expertise ; critiquer les résultats « trop beaux ».
8. **Modèle final :** ré-entraîner sur toutes les données avec les hyperparamètres choisis, en sachant qu'on n'a plus de test indépendant pour lui.

---

---

# Correction – Questions type partiel sur la Conformal Prediction

---

## A. Définitions

### 1. Définis la **couverture** ( *coverage* ) et écris la garantie de la conformal prediction. Que représente α, et qui le choisit ?

La **couverture** (*coverage*) est la probabilité que la vraie réponse appartienne à l'ensemble de prédiction construit par la méthode.

Garantie de la conformal prediction :

```
P( y_test ∈ C(x_test) ) ≥ 1 − α
```

- **α** est le **taux d'erreur toléré**, choisi par l'**utilisateur** (ex. α = 0,1 → couverture ≥ 90 %).
- La probabilité porte sur le tirage aléatoire du jeu de calibration **et** du point de test.

### 2. Qu’est-ce qu’un **ensemble de prédiction** ? En quoi diffère-t-il de la sortie d’un classifieur classique ?

C'est l'ensemble `C(x)` des classes (ou, en régression, un intervalle de valeurs) jugées **plausibles** pour l'entrée `x`.

Différence avec un classifieur classique : celui-ci renvoie **une seule** classe (prédiction ponctuelle) sans garantie sur sa fiabilité pour cette entrée. La conformal prediction renvoie un **ensemble dont la taille s'adapte à l'incertitude** (petit si le modèle est sûr, grand s'il hésite), avec une garantie de couverture.

### 3. Définis le **score de non-conformité** et la  **fonction de confiance** . Quel est le lien entre les deux ?

- **Fonction de confiance** `c(x, y)` : degré de confiance du modèle dans l'étiquette `y` pour l'entrée `x`. En classification, `c(x, y) = h_y(x)` (par ex. le softmax). Valeur **élevée = bon**.
- **Score de non-conformité** `s(x, y) = 1 − c(x, y)` : mesure à quel point la paire `(x, y)` est « étrange » pour le modèle. Valeur **élevée = mauvais**.

Les deux sont l'envers l'un de l'autre : « `s ≤ q̂` » équivaut à « `c ≥ 1 − q̂` ». Les deux points de vue sont équivalents.

### 4. Définis l’ **échangeabilité** . Quelle est sa différence avec l’hypothèse i.i.d. ? Donne un exemple où elle est violée

Des variables aléatoires sont **échangeables** si leur loi jointe est **invariante par permutation** de leur ordre : aucune ne joue un rôle spécial.

- i.i.d. ⇒ échangeable, mais **pas l'inverse** : l'échangeabilité est une hypothèse plus faible.
- En pratique : calibration et test doivent provenir de **la même source**.
- **Exemples de violation** : calibration sur des IRM de l'hôpital A et test sur l'hôpital B (autre scanner) ; série temporelle avec dérive ; calibration en été et test en hiver.

### 5. Que signifie « **mal calibré** » à propos du softmax ? Cite un exemple du cours.

Les sorties du softmax ressemblent à des probabilités (positives, somme = 1), mais elles ne correspondent **pas** à la vraie probabilité que la prédiction soit correcte : elles reflètent la façon dont le réseau répartit ses scores bruts.

Exemple du cours (Goodfellow et al., 2015) : une photo de panda est classée « panda » avec 57,7 % de confiance ; ajoutée d'un bruit imperceptible, elle est classée « gibbon » avec **99,3 %** de confiance. Le réseau est quasi certain d'une réponse absurde.

### 6. Qu’appelle-t-on **jeu de calibration** ? Pourquoi doit-il être distinct du jeu d’entraînement ?

C'est un jeu de `n` exemples **étiquetés** `{(x₁, y₁), …, (xₙ, yₙ)}`, utilisé pour mesurer empiriquement la confiance du modèle sur la **vraie** classe et en déduire le seuil `q̂`.

Il doit être **distinct de l'entraînement** : sur les données d'entraînement, le modèle est trop optimiste (il les a « appris »), donc les scores seraient trop bons, le seuil mal placé, et la garantie fausse. Il doit aussi être échangeable avec les données de test.

### 7. Que veut dire **model-agnostic** et **distribution-free** ? Pourquoi ces deux propriétés sont-elles importantes ?

- **Model-agnostic** : la méthode s'applique à **n'importe quel modèle** (réseau, forêt aléatoire…), classification ou régression, pourvu qu'il fournisse une mesure de confiance. Le modèle n'est pas modifié : c'est un habillage *post-hoc*.
- **Distribution-free** : la garantie ne suppose **aucune forme particulière** pour la loi des données (pas de gaussienne, etc.), seulement l'échangeabilité. Elle est valable en **échantillon fini**, pas seulement asymptotiquement.

Importance : les hypothèses paramétriques sont rarement vérifiées en pratique, et on ne veut pas dépendre de l'architecture utilisée.

### 8. Quelle est la différence entre une couverture **marginale** et une couverture **conditionnelle** ?

- **Marginale** : `P(y ∈ C(x))` moyennée sur **tous** les `x` (et sur le tirage de la calibration). C'est ce que garantit la méthode.
- **Conditionnelle** : `P(y ∈ C(x) | X = x)` pour **un `x` donné** (ou pour une classe/sous-groupe donné). **Non garantie** : un sous-groupe difficile peut être moins bien couvert que 1 − α, compensé par des cas faciles sur-couverts.

---

## B. Logique et enchaînement des concepts

### 9. Pourquoi une précision de 90 % sur le jeu de test ne répond-elle pas au besoin du médecin face à une IRM précise ?

La précision est une **moyenne sur de nombreuses images**. Le médecin a **une** IRM, celle de son patient, et veut savoir si la prédiction est fiable **pour cette entrée**. Or le 90 % ne dit rien sur la fiabilité pour une entrée particulière jamais vue. Il faut une **mesure d'incertitude associée à chaque prédiction** (vraisemblance des issues possibles, ou au moins exclusion des issues très improbables).

### 10. Pourquoi ne peut-on pas simplement lire le softmax comme une probabilité ? Quel besoin cela crée-t-il ?

Car il est mal calibré (cf. panda → gibbon à 99,3 %) : une confiance élevée n'implique pas une probabilité élevée d'avoir raison. On ne peut donc pas s'en servir directement pour quantifier l'incertitude, d'où le **besoin d'une méthode fournissant une mesure d'incertitude avec garantie**.

### 11. Quelles sont les limites des *deep ensembles* ? En quoi la conformal prediction apporte-t-elle quelque chose de différent ?

*Deep ensembles* : on entraîne plusieurs réseaux et on regarde la dispersion de leurs prédictions (accord = confiance).

Limites :

- **aucune garantie de correction** (80 % d'accord ≠ 80 % de chances d'avoir raison) ;
- **coûteux** (entraîner et exécuter plusieurs réseaux).

La conformal prediction apporte une **garantie mathématique de couverture** pour un **faible coût** (un seul modèle, un tri, un quantile, une comparaison), sans réentraînement.

### 12. Explique l’idée centrale « utiliser la confiance dans les classifications passées pour évaluer la confiance dans l’entrée courante ». Comment cette idée se traduit-elle concrètement en algorithme ?

> Utiliser la confiance dans les classifications faites **dans le passé** pour évaluer la confiance dans la prédiction pour **l'entrée courante**.

Plutôt que de croire le softmax, on **mesure empiriquement**, sur des exemples dont on connaît la réponse, à quel niveau de confiance se situe le modèle **quand il traite la vraie classe**. On obtient une échelle de référence.

Traduction algorithmique :

1. calculer les scores de calibration (confiance dans la vraie classe) ;
2. en extraire un **quantile** `q̂` correspondant au niveau `1 − α` ;
3. pour un nouvel `x`, garder les classes dont la confiance atteint le seuil.

Analogie : situer une note de 14/20 par rapport à la distribution des notes de la classe.

### 13. Sur le jeu de calibration, pourquoi regarde-t-on la confiance dans la **vraie** classe et non dans la classe prédite ? Que se passe-t-il quand le modèle se trompe ?

Ce qu'on veut contrôler, c'est la probabilité que **la vraie classe soit dans l'ensemble**. Il faut donc savoir **à quel niveau de confiance le modèle place la bonne réponse**, y compris quand il se trompe.

Si le modèle se trompe, la confiance qu'il accorde à la vraie classe est **faible** : c'est exactement l'information à capter. Elle tire le seuil vers le bas et élargit les ensembles pour couvrir ces cas. Regarder seulement la classe prédite masquerait les erreurs.

### 14. Reconstitue les étapes de la procédure, de la calibration à la construction de l’ensemble pour un nouvel x. Justifie le rôle de chaque étape.

0. **Modèle déjà entraîné** `h` (non modifié), qui renvoie un score par classe.
1. **Fonction de confiance** `c(x, y) = h_y(x)` (ou score `s = 1 − c`) : donne une mesure comparable entre exemples.
2. **Jeu de calibration** de `n` exemples étiquetés, **non utilisés à l'entraînement** : sert de référence honnête.
3. **Confiances de calibration** `c(xᵢ, yᵢ)` sur la **vraie** classe : donne la distribution de référence.
4. **Seuil** : rang `k = ⌈(1 − α)(n + 1)⌉`, seuil = k-ième plus grand niveau de confiance (équivalent : k-ième plus petit score `s`). Garantit la couverture `≥ 1 − α`.
5. **Ensemble pour `x_test`** : `C(x_test) = { y : h_y(x_test) ≥ seuil }`, c'est-à-dire `s ≤ q̂`.

### 15. Explique pourquoi on utilise `⌈(n+1)(1−α)⌉` et non `⌈n(1−α)⌉`. Appuie-toi sur l’argument d’échangeabilité.

Sous échangeabilité, les `n` scores de calibration et celui du point de test forment **`n + 1` scores échangeables**. Le score du test a donc autant de chances d'occuper **n'importe laquelle** des `n + 1` positions du classement.

Ainsi : `P(score test parmi les k plus petits) = k / (n + 1)`. Avec `k = ⌈(1 − α)(n + 1)⌉`, on a `k / (n + 1) ≥ 1 − α`. La couverture est donc ≥ 1 − α.

Avec `n` au lieu de `n + 1`, on ignorerait que le point de test fait partie du classement : la garantie serait légèrement trop optimiste, surtout pour `n` petit. C'est une **correction de taille finie** ; quand `n` est grand, `n + 1 ≈ n` et l'effet disparaît.

### 16. Pourquoi la couverture est encadrée par 1 − α + 1/(n+1)

Borne inférieure : `k / (n + 1) ≥ 1 − α` par définition du plafond.

Borne supérieure : le plafond ajoute au plus `1` à `(1 − α)(n + 1)`, donc `k ≤ (1 − α)(n + 1) + 1` et `k / (n + 1) ≤ 1 − α + 1/(n+1)`. (Cette borne supérieure suppose des scores sans ex æquo.)

D'où « couverture *presque exactement* 1 − α » : la méthode n'est ni trop conservatrice, ni trop optimiste.

---

## C. Compromis et cas limites

### 18. Que se passe-t-il sur la taille de l’ensemble quand α diminue ? Justifie à partir du seuil.

Quand α **diminue**, on exige une couverture plus forte. Il faut alors un rang `k` plus grand, donc un **seuil de confiance plus bas** (on descend plus loin dans les scores de calibration pour couvrir presque tous les cas, y compris les erreurs). Plus de classes franchissent le seuil : l'ensemble est **plus grand**.

Compromis : **garantie ↔ précision (petits ensembles)**.

Exemple du cours (n = 10) : α = 0,1 → seuil 0,05 → `{dog, tiger, cat}` ; α = 0,5 → seuil ≈ 0,25–0,30 → `{tiger, cat}` ; α = 0,9 → seuil 0,90 → `∅`.

### 19. Un ensemble de prédiction peut-il être vide ? Dans quelles situations, et comment l’interpréter ?

Oui, c'est possible. Cas typiques :

- **α grand** : on est très exigeant, aucune classe n'atteint le seuil (exemple : α = 0,9, seuil 0,90, max = 0,60) ;
- **`x` atypique** : le modèle est en général très sûr sur la vraie classe (seuil élevé, ex. 0,75) mais ne dépasse pas ce seuil pour cet `x` (ex. max = 0,60).

Interprétation : un ensemble vide n'est pas une erreur, c'est un **signal d'alarme** : « cet exemple ne ressemble pas à ceux sur lesquels je suis fiable ». La garantie reste tenue.

### 20. Si le modèle est mauvais, la garantie de couverture tient-elle encore ? Qu’est-ce qui est dégradé ?

La **garantie de couverture reste vraie** : elle ne dépend pas du modèle, seulement de l'échangeabilité. Ce qui est dégradé, c'est l'**utilité** : un mauvais modèle donne des scores de calibration bas, donc un seuil bas, donc des ensembles **plus grands et moins informatifs**, mais jamais des ensembles faux. La qualité du modèle influe sur l'**efficacité**, pas sur la **validité**.

### 21. Pourquoi la taille de l’ensemble s’adapte-t-elle à la difficulté de l’image alors que la garantie est marginale ?Adaptation à la difficulté malgré une garantie margina

Le **seuil est unique** (calculé sur la calibration), mais le **vecteur de confiances `h(x)` varie avec `x`**. Image facile : une classe domine avec une confiance élevée, donc seule celle-là passe le seuil → petit ensemble. Image difficile : les confiances sont réparties, plusieurs classes passent le seuil → grand ensemble.

La garantie, elle, porte sur la **moyenne** sur tous les `x`. L'adaptation à `x` est un effet de la construction, mais ne garantit pas une couverture de 1 − α **pour chaque `x`** (voir Q8).

### 22. Que devient la garantie si la calibration est faite sur les données de l’hôpital A et le test sur l’hôpital B ? Justifie.

L'**échangeabilité est violée** : calibration et test ne viennent pas de la même distribution (autre scanner, autre population). Le score du point de test n'est plus échangeable avec ceux de calibration, donc l'argument sur le rang uniforme tombe et la **garantie n'est plus assurée**. La couverture réelle peut être inférieure à 1 − α. C'est la principale limite pratique de la méthode (changement de domaine, dérive temporelle).

### 23. Un camarade dit : « la conformal prediction améliore la précision du modèle. » Réponds-lui.

Faux. Le modèle n'est **pas modifié** : la méthode l'habille d'une mesure d'incertitude sous forme d'ensembles de prédiction calibrés. Elle ne change ni ses prédictions ni sa précision ; elle dit **dans quelle mesure on peut s'y fier**.

### 24.Un autre dit : « avec α = 0,1, il y a 90 % de chances que la prédiction pour cette image soit correcte. » Qu’est-ce qui est inexact dans cette phrase ?

Trois inexactitudes :

1. La garantie porte sur un **ensemble** : `P(y ∈ C(x)) ≥ 1 − α`, et non sur une prédiction ponctuelle.
2. Elle est **marginale** (en moyenne sur les données et le tirage de la calibration), pas conditionnelle à **cette** image.
3. C'est une borne inférieure (« au moins 90 % »), valable **sous l'hypothèse d'échangeabilité**.

Formulation correcte : « en moyenne, dans au moins 90 % des cas, la vraie classe appartiendra à l'ensemble de prédiction ».

---

## Fiche de synthèse

| Notion                      | À retenir                                                      |
| --------------------------- | --------------------------------------------------------------- |
| Garantie                    | `P(y ∈ C(x)) ≥ 1 − α`, marginale, en échantillon fini    |
| Hypothèse                  | Échangeabilité calibration / test                             |
| Seuil                       | k-ième valeur avec `k = ⌈(1 − α)(n + 1)⌉`                |
| Le « +1 »                 | Compte le point de test dans les `n + 1` scores échangeables |
| α ↓                       | Ensemble plus grand                                             |
| Ensemble vide               | Possible : α grand ou `x` atypique                           |
| Mauvais modèle             | Validité intacte, utilité dégradée                          |
| Distribution shift          | Garantie perdue                                                 |
| Marginale ≠ conditionnelle | Pas de garantie par `x` ni par classe                         |
