# Cours 8 : Les méthodes d'ensemble (boosting, bagging…) expliquées pas à pas

> Basé sur le cours « Ensemble methods: boosting, bagging, random forests… » d'A. Cornuéjols (AgroParisTech).
> Les renvois **(diapo N)** correspondent à la numérotation du fichier fourni.
>
> **⚠️ Limites de ce que j'ai pu lire.**
> 1. Le fichier fourni est une **extraction texte** : les **formules, schémas et exemples chiffrés (images)** n'y figurent pas. Quand une formule n'était pas lisible, je la donne comme **➕ Complément** (théorie standard d'AdaBoost), pas comme contenu de tes diapos.
> 2. Le texte **s'arrête à la diapo 116 (le bagging)**. Les parties 5 (forêts aléatoires), 6 (XGBoost) et 7 (autres méthodes) annoncées au plan **ne sont pas dans le fichier**. J'en donne une courte présentation en section 12, clairement marquée comme complément hors cours ; envoie-moi la suite si tu veux la version fidèle à tes diapos.
>
> Les paragraphes **➕ Complément** ajoutent une précision qui n'est pas dans le texte fourni. Les encadrés **⚠️ Point d'attention** signalent une incohérence repérée (vérifiée par le calcul).
> Les formules sont en LaTeX (`$...$`) : elles s'affichent dans Obsidian, Typora, VS Code, Jupyter, etc.

---

## Sommaire

1. [La grande image](#1-la-grande-image)
2. [Pourquoi combiner ? La sagesse des foules](#2-pourquoi-combiner--la-sagesse-des-foules)
3. [Le schéma général d'une méthode d'ensemble](#3-le-schéma-général-dune-méthode-densemble)
4. [Le boosting : le principe](#4-le-boosting--le-principe)
5. [AdaBoost : l'algorithme](#5-adaboost--lalgorithme)
6. [Exemple chiffré : un tour d'AdaBoost](#6-exemple-chiffré--un-tour-dadaboost)
7. [Le boosting : pourquoi ça marche](#7-le-boosting--pourquoi-ça-marche)
8. [Bilan pratique, limites et variantes](#8-bilan-pratique-limites-et-variantes)
9. [Applications : la détection de visages](#9-applications--la-détection-de-visages)
10. [Gradient boosting](#10-gradient-boosting)
11. [Le bagging](#11-le-bagging)
12. [Pour la suite du cours (hors fichier fourni)](#12-pour-la-suite-du-cours-hors-fichier-fourni)
13. [Fiche récapitulative](#13-fiche-récapitulative)
14. [Quiz d'auto-évaluation](#14-quiz-dauto-évaluation)

---

## 1. La grande image

Un seul classifieur, même bien choisi, a ses limites. Si on en **combine plusieurs**, le comité fait souvent mieux que le meilleur de ses membres. Tout le cours répond à une idée :

> **Trouver une meilleure solution en combinant des solutions « faibles »** (diapo 4).

Le cours pose cinq questions (diapo 5) qui structurent tout :

| # | Question | Réponse selon la méthode |
|---|---|---|
| 1 | **Quels agents ?** (quels apprenants ?) | Des apprenants « faibles » : arbres, souches… |
| 2 | **Que s'échangent-ils** si le processus est itératif ? | Boosting : des **poids sur les exemples** |
| 3 | **Comment combiner** leurs avis ? | Vote (pondéré ou non), moyenne |
| 4 | **Quelles conditions de convergence ?** | Chaque apprenant doit faire mieux que le hasard |
| 5 | **Si convergence, vers quoi ?** | Vers un classifieur de grande marge / de faible erreur |

Deux grandes familles se dégagent :

| | **Boosting** | **Bagging** |
|---|---|---|
| Construction | **Séquentielle** : chaque classifieur dépend des précédents | **Parallèle** : classifieurs indépendants |
| Données vues | Les mêmes, mais **repondérées** | Des **échantillons bootstrap** différents |
| Vote | **Pondéré** | **Simple** |
| Effet principal | Réduit surtout le **biais** (faibles → fort) | Réduit la **variance** |

---

## 2. Pourquoi combiner ? La sagesse des foules

*(diapos 6 à 8, 12-13)*

### 2.1 L'expérience de Galton (diapos 6-7)

*« The wisdom of crowd »* (J. Surowiecki, 2004). En **1906**, Francis Galton (85 ans, à la fois anthropologue, explorateur, statisticien, météorologue…) fait estimer à **787 participants** le poids d'un objet sur un marché. Résultat :
- **le meilleur** participant se trompe de **plus d'un centième** (>1 %) ;
- **la moyenne** de tous se trompe de **moins d'un millième** (<0,1 %).

La foule, composée de gens **individuellement médiocres**, bat le meilleur individu.

### 2.2 Pourquoi ça marche (diapo 8)

Les conditions listées par le cours : des « experts faibles » dont les estimations sont **bruitées**, mais :
- **non biaisées** (pas d'erreur systématique dans un sens) ;
- **symétriques** (autant d'erreurs par excès que par défaut) ;
- **indépendantes** (l'erreur de l'un n'influence pas celle de l'autre).

Alors une **combinaison simple, la moyenne**, annule les erreurs.

> 💡 **Analogie.** 787 personnes lancent chacune une fléchette vers le centre d'une cible, avec une main tremblante, sans viser systématiquement trop à gauche ou trop haut. Les impacts sont éparpillés, mais **leur centre de gravité tombe presque pile sur le centre**. Chaque erreur individuelle est compensée par les autres.

➕ **Le chiffre derrière.** Si l'on moyenne $k$ estimations **indépendantes**, chacune de variance $\sigma^2$, la variance de la moyenne est $\sigma^2/k$ : l'erreur typique diminue comme $1/\sqrt k$. Si elles sont **corrélées** (coefficient $\rho$), la variance vaut $\rho\sigma^2+\frac{1-\rho}{k}\sigma^2$ : on ne descend jamais sous $\rho\sigma^2$. **D'où l'importance d'avoir des apprenants le plus décorrélés possible**, c'est le fil rouge de tout le cours.

### 2.3 Justifications intuitives (diapo 12)

Le cours cite Dietterich (2000, *Ensemble Methods in Machine Learning*) pour justifier pourquoi un ensemble peut mieux faire qu'un seul modèle. ➕ Les trois raisons classiques de cet article :
1. **Statistique** : avec peu de données, plusieurs hypothèses sont également bonnes sur l'apprentissage ; en les moyennant, on réduit le risque de choisir la mauvaise ;
2. **Calculatoire** : un algorithme de recherche locale peut se coincer dans un mauvais optimum ; en lançant plusieurs recherches, on approxime mieux la vraie solution ;
3. **Représentationnelle** : la vraie fonction n'est peut-être dans l'espace d'hypothèses d'aucun modèle individuel, mais une **combinaison** peut s'en approcher.

### 2.4 Des succès applicatifs (diapo 13)

- **KDD-Cup** (concours de fouille de données) : détection d'intrusion réseau (1999), bioactivité moléculaire (2001), détection d'embolie pulmonaire (2006), gestion de la relation client (2009), données éducatives (2010), recommandation musicale (2011)… Sur **2009-2011**, tous les **1ers et 2es prix** sont allés à des **méthodes d'ensemble** (la diapo met des points d'interrogation sur les dates exactes).
- **Netflix Prize** : 1 000 000 $ à qui améliorerait de **plus de 10 %** l'algorithme de recommandation de Netflix. **Gagné en 2009 par une méthode d'ensemble.**

---

## 3. Le schéma général d'une méthode d'ensemble

*(diapos 11, 33, 36)*

**Apprentissage.** À partir de l'échantillon d'apprentissage $S$, on construit **$T$ apprenants** $h_1,\dots,h_T$ (en modifiant les données, l'algorithme, ou les deux).

**Prédiction.** Pour un nouvel $x$, on interroge tous les apprenants et on **agrège** leurs réponses :

$$H(x)=\text{sign}\Big(\sum_{t=1}^{T}\alpha_t\,h_t(x)\Big)\qquad(\text{classification binaire, }h_t(x)\in\{-1,+1\})$$

Pour tout ce cours, deux questions reviennent (diapo 30) :
1. **Comment engendrer des apprenants faibles décorrélés ?**
2. **Comment combiner leurs prédictions ?**

---

## 4. Le boosting : le principe

*(diapos 16 à 38)*

### 4.1 Une illustration frappante (diapos 16-18)

Problème : espace à **10 dimensions** (attributs indépendants, gaussiens), concept cible défini à partir de ces attributs, 2000 exemples d'apprentissage (1000 + / 1000 −), 10 000 de test, apprentissage d'**arbres de décision**. La diapo 17 trace l'**erreur de test** (axe vertical de 0 à 0,5) en fonction du nombre d'étapes (0 à 400) pour trois systèmes : un arbre à un nœud, un arbre à 400 nœuds, et des arbres à un nœud avec boosting. Les courbes sont dans une image, mais le message classique de cette expérience est le suivant (➕ à vérifier sur ta diapo) :
- **un arbre à un nœud** (une « souche » : très faible) → erreur élevée ;
- **un arbre à 400 nœuds** → mieux, mais limité ;
- **des souches avec boosting** → l'erreur descend **bien plus bas**, alors que chaque brique reste minuscule.

**Message :** on peut atteindre une grande précision à partir de briques **extrêmement faibles**, à condition de les combiner intelligemment.

### 4.2 Un exemple simple en 2D (diapos 19 à 24)

On cherche le **meilleur séparateur linéaire** sur 20 points. Le meilleur $h_1$ fait **5 erreurs sur 20** : taux d'erreur $=5/20=0{,}25$.

> *« Et si je pouvais combiner avec un autre séparateur linéaire ? Ou même plusieurs autres ! »*

On utilise un **vote pondéré** (diapo 23) :

$$H(x)=\text{sign}\{0{,}549\,h_1(x)+0{,}347\,h_2(x)+0{,}310\,h_3(x)+0{,}406\,h_4(x)+0{,}503\,h_5(x)\}$$

Chaque $h_t$ est un simple séparateur linéaire, mais leur **combinaison pondérée** trace une frontière beaucoup plus riche (non linéaire).

> **Question (diapo 24) :** *comment arriver à ce genre de combinaison ?* → l'**algorithme du boosting**.

➕ **Un détail instructif.** Ces poids ne sont pas arbitraires : ils sont liés à l'erreur de chaque $h_t$ (cf. section 5). Par exemple $0{,}549=\tfrac12\ln\frac{1-0{,}25}{0{,}25}$, ce qui correspond exactement à l'erreur $0{,}25$ de $h_1$. Les poids $0{,}347$ ; $0{,}310$ ; $0{,}406$ ; $0{,}503$ correspondent à des erreurs pondérées d'environ $0{,}33$ ; $0{,}35$ ; $0{,}31$ ; $0{,}27$ pour $h_2$ à $h_5$ : **plus un classifieur est précis, plus son vote pèse.**

### 4.3 Une question théorique : fort ou faible ? (diapo 26)

- **Apprentissage « fort »** (PAC) : une classe de fonctions est apprenable si un algorithme peut, pour **toute distribution** et **toute fonction cible**, atteindre une erreur **arbitrairement petite** avec une probabilité arbitrairement grande.
- **Apprentissage « $\gamma$-faible »** : on exige seulement une erreur **un peu meilleure que le hasard** (au plus $\frac12-\gamma$, avec $\gamma>0$).

> **« Sont-ils de nature différente ? »** On pourrait le croire : le « fort » est bien plus exigeant. **Schapire (1990, *The strength of weak learnability*) a montré que non : ils sont équivalents.** On peut *convertir* un apprenant faible en apprenant fort. C'est précisément le boosting.

### 4.4 La définition (diapos 32, 34)

> **Boosting = méthode générale pour convertir des règles de prédiction peu performantes en une règle de prédiction (très) performante.**

Plus précisément : étant donné un algorithme « faible » capable de toujours retourner une hypothèse d'erreur $\le\frac12-\gamma$, un algorithme de boosting peut construire (**de manière prouvée**) une hypothèse d'erreur $\le\varepsilon$, aussi petite qu'on veut.

**Analogie des « courses » (diapo 34).** Un parieur veut battre les bookmakers. Il demande à un expert des **règles empiriques** (« mise sur le cheval qui a gagné la dernière fois »). Chaque règle est faible. Deux questions :
1. **Comment choisir les courses à chaque étape ?** → se concentrer sur les courses les plus **difficiles** (celles où les règles précédentes se trompent le plus).
2. **Comment combiner les règles ?** → prendre un **vote majoritaire pondéré**.

### 4.5 Comment engendrer les apprenants (diapos 27 à 31)

**Recette historique (Schapire, 1990).** Elle enchaîne trois apprenants (diapos 27-29). ➕ L'idée (diapos en images) : $h_1$ est appris sur $S$ ; $h_2$ sur un échantillon où **la moitié** des exemples sont mal classés par $h_1$ ; $h_3$ sur les exemples où $h_1$ et $h_2$ **divergent** ; vote majoritaire final.

**Version générale (diapo 31).** On **modifie l'échantillon d'apprentissage à chaque étape** :
- en **diminuant** l'importance des exemples **bien classés** ;
- en **augmentant** celle des exemples **mal classés**.

La question restante : **de combien ?** C'est ce que règle AdaBoost.

---

## 5. AdaBoost : l'algorithme

*(diapos 35 à 39)*

### 5.1 Vue formelle (diapo 35)

Entrée : $S=\{(x_1,y_1),\dots,(x_m,y_m)\}$ avec $y_i\in\{-1,+1\}$.

Pour $t=1,\dots,T$ :
1. **Construire une distribution $D_t$** sur $\{1,\dots,m\}$ (les « poids » des exemples) ;
2. **Trouver l'hypothèse faible** $h_t:X\to\{-1,+1\}$ avec **une petite erreur** $\varepsilon_t$ **sur $D_t$** :
$$\varepsilon_t=\Pr_{i\sim D_t}\big[h_t(x_i)\ne y_i\big]$$

Retourner l'hypothèse finale $h_{final}$ (vote pondéré).

**Important :** l'erreur $\varepsilon_t$ est mesurée **en tenant compte des poids** $D_t$ : une erreur sur un exemple de poids élevé coûte plus cher.

### 5.2 Les formules d'AdaBoost (Freund & Schapire, 1996-97)

Les formules sont des images dans tes diapos 38-39. ➕ Version standard :

$$D_1(i)=\frac1m\qquad\qquad D_{t+1}(i)=\frac{D_t(i)}{Z_t}\times\begin{cases}e^{-\alpha_t}&\text{si }h_t(x_i)=y_i\ \ (\text{bien classé})\\ e^{+\alpha_t}&\text{si }h_t(x_i)\neq y_i\ \ (\text{mal classé})\end{cases}$$

$$\alpha_t=\frac12\ln\frac{1-\varepsilon_t}{\varepsilon_t}\qquad\qquad H(x)=\text{sign}\Big(\sum_{t=1}^T\alpha_t\,h_t(x)\Big)$$

où $Z_t$ est une **constante de normalisation** (pour que $D_{t+1}$ somme à 1), comme l'indique la diapo 38. On écrit parfois $D_{t+1}(i)=\frac{D_t(i)}{Z_t}e^{-\alpha_t y_ih_t(x_i)}$.

### 5.3 Comprendre chaque morceau

**Le poids $\alpha_t$ du vote** (réponse à « comment calculer la pondération $\alpha_t$ ? », diapo 37) :

| Erreur $\varepsilon_t$ | $\alpha_t=\frac12\ln\frac{1-\varepsilon_t}{\varepsilon_t}$ | Interprétation |
|---|---|---|
| Proche de 0 (très bon) | grand positif | Il a une **grande voix** |
| 0,25 | $0{,}549$ | Voix moyenne |
| 0,5 (hasard) | **0** | Aucune voix : un classifieur au niveau du hasard n'apporte rien |
| >0,5 | négatif | (Pire que le hasard : on inverse son avis.) |

C'est pourquoi on exige que l'apprenant soit **meilleur que le hasard** ($\varepsilon_t<\frac12$) : sans cela, il n'a aucun poids utile.

**La mise à jour des poids** (réponse à « comment passer de $D_t$ à $D_{t+1}$ ? ») :
- exemple **bien classé** → son poids est **multiplié par $e^{-\alpha_t}<1$** (il baisse) ;
- exemple **mal classé** → son poids est **multiplié par $e^{+\alpha_t}>1$** (il monte).

> 💡 **L'intuition en une phrase.** AdaBoost dit à l'apprenant suivant : « *ne perds pas ton temps sur ce que les autres savent déjà : concentre-toi sur ce qui leur échappe.* » C'est l'élève qui, après un contrôle, **retravaille uniquement ses fautes**.

**Cas particulier :** un apprenant qui ne sait pas travailler avec des poids peut le simuler en **ré-échantillonnant** les exemples avec probabilité proportionnelle à $D_t$.

---

## 6. Exemple chiffré : un tour d'AdaBoost

*(diapos 40 à 46 en images ; calcul ➕ refait ici)*

Reprenons les 20 points, où $h_1$ fait **5 erreurs** : $\varepsilon_1=5/20=0{,}25$. Au départ, chaque exemple a un poids de $1/20=0{,}05$.

1. **Poids du classifieur** : $\alpha_1=\frac12\ln\frac{0{,}75}{0{,}25}=\frac12\ln3\approx\mathbf{0{,}549}$ ✔ (c'est bien le premier coefficient du vote de la diapo 23).
2. **Facteurs de mise à jour** : bien classés $e^{-0{,}549}\approx0{,}577$ ; mal classés $e^{+0{,}549}\approx1{,}732$.
3. **Normalisation** : $Z_1=15\times0{,}05\times0{,}577+5\times0{,}05\times1{,}732=0{,}433+0{,}433=0{,}866$.
4. **Nouveaux poids** :

| Exemples | Poids avant | Poids après | Poids total du groupe |
|---|---|---|---|
| 15 bien classés | 0,05 | $0{,}05\times0{,}577/0{,}866=\mathbf{1/30}\approx0{,}033$ | **0,5** |
| 5 mal classés | 0,05 | $0{,}05\times1{,}732/0{,}866=\mathbf{1/10}=0{,}1$ | **0,5** |

> 💡 **À remarquer, c'est très élégant.** Après la mise à jour, les **mal classés pèsent exactement la moitié** du poids total, quelle que soit l'erreur initiale. Du point de vue de $D_2$, **$h_1$ est donc exactement au niveau du hasard** (erreur 0,5). L'apprenant suivant **ne peut pas** se contenter de recopier $h_1$ : il est forcé d'apporter une information nouvelle. C'est ce qui « décorrèle » les apprenants.

Et on recommence : $h_2$ est appris sur $D_2$, etc. Les diapos 47 à 51 déroulent un **exemple jouet** en trois étapes avec **l'hypothèse finale** qui combine les trois. Les diapos 52-53 le refont avec des **arbres de décision / des souches de décision** (« decision stump », d'après Zhou, *Machine Learning*, 2021) sur un jeu de données.

**Effets de deux réglages (diapos 54-55)**, sur le jeu *Iris* :
- **Nombre d'itérations** $T$ (Setosa vs Versicolor) : plus on boucle, plus la frontière se précise ;
- **Profondeur des arbres** (Versicolor vs Virginica) : des arbres plus profonds donnent des apprenants plus puissants, donc une frontière plus complexe.

---

## 7. Le boosting : pourquoi ça marche

*(diapos 60 à 82)*

### 7.1 Borne sur l'erreur d'apprentissage (diapos 64-65)

Premier résultat : **l'erreur d'apprentissage de $H$ décroît exponentiellement** avec le nombre d'étapes. ➕ Formule standard (diapo en image) :

$$\text{err}_{S}(H)\ \le\ \prod_{t=1}^T Z_t=\prod_{t=1}^T2\sqrt{\varepsilon_t(1-\varepsilon_t)}\ \le\ \exp\Big(-2\sum_{t=1}^T\gamma_t^2\Big),\qquad\gamma_t=\tfrac12-\varepsilon_t$$

Tant que chaque apprenant bat le hasard d'une marge $\gamma_t\ge\gamma>0$, l'erreur d'apprentissage est $\le e^{-2\gamma^2T}$ : elle **tend vers 0**.

*(Sur l'exemple à cinq classifieurs de la section 4.2, le produit des $Z_t$ vaut déjà $\approx0{,}64$ ; avec des centaines d'étapes, on descend très bas.)*

### 7.2 Une re-dérivation par la perte exponentielle (diapos 61-62)

On peut retrouver l'algorithme **sans le deviner** : en choisissant une **fonction de perte surrogée de forme exponentielle**, $\ell(y,H(x))=e^{-yH(x)}$, et en la minimisant **étape par étape**, on obtient exactement les formules d'AdaBoost (le choix de $\alpha_t$ et la mise à jour des poids). C'est la « vue plus haute » du boosting (diapo 82).

### 7.3 L'énigme de l'erreur de test (diapos 66-67)

**Le phénomène étrange :** quand on boucle sur des centaines d'étapes, l'erreur d'apprentissage atteint **zéro** — et pourtant **l'erreur de test continue de baisser**, ou au moins **ne remonte pas** (diapo 67 : en boostant C4.5 sur le jeu *letter*, pas de remontée **même après 1000 étapes, soit $2\cdot10^6$ nœuds de test !**).

C'est **contraire à l'intuition** : on attendrait du surapprentissage (cf. cours sur les arbres : trop de nœuds → l'erreur de test remonte). C'est ce que la diapo 68 appelle *« the unreasonable power of boosting »*, le pouvoir déraisonnable du boosting. **Comment l'expliquer ?**

### 7.4 L'explication par la marge (diapos 71 à 75)

**L'idée :** le taux d'erreur d'apprentissage ne regarde que **si** la prédiction est juste. Il faut aussi regarder **la confiance** de la prédiction.

On mesure cette confiance par la **marge** d'un exemple :

$$\text{marge}(x_i)=\text{poids des classifieurs ayant voté correctement}-\text{poids des classifieurs en erreur}$$

(normalisée entre −1 et +1). Marge proche de **+1** : presque tous les votants (pondérés) sont d'accord et ont raison → **prédiction très confiante**. Marge proche de **0** : vote serré. Marge **négative** : exemple mal classé.

> 💡 **Analogie électorale (diapo 76).** Si un candidat gagne **80 % des voix**, un petit sondage aléatoire de quelques dizaines de personnes suffit à prédire le résultat ; s'il gagne **51 %**, il en faut énormément. **Plus la marge est grande, plus la décision est robuste**, donc plus il est possible de « simplifier » le comité sans changer le résultat.

**L'argument (diapo 73).** À chaque étape, AdaBoost place le plus de poids sur les exemples de **plus faible marge**, tout en continuant à améliorer la marge des autres. Résultat : même si l'hypothèse finale est **complexe** (beaucoup de classifieurs), elle est **proche d'une hypothèse simple d'erreur d'apprentissage proche**, donc d'**erreur de généralisation faible aussi**. Les bornes de généralisation (diapos 77-78) sont exprimées **en fonction de la marge**.

**Parallèle avec les SVM** (diapo 70 renvoie au cours « SVM and kernel methods ») : c'est **exactement la même philosophie**. Là-bas, on maximise une marge pour contrôler la capacité ; ici, le boosting augmente la marge de la combinaison. **Dans les deux cas, ce n'est pas le nombre de paramètres qui mesure la complexité, mais la marge.**

> ⚠️ **Nuance (diapos 79-80, « Mais… »).** L'explication par la marge est **belle mais pas parfaite** : les diapos 79-81 ("Mais ...") indiquent qu'elle laisse des questions ouvertes. (Le texte de ces diapos n'est pas dans le fichier.) Ne la présente donc pas comme une preuve définitive.

### 7.5 Le boosting comme construction d'un espace de redescription (diapo 69)

Autre lecture : chaque $h_t$ est une **nouvelle caractéristique** (une coordonnée). La combinaison $\sum\alpha_th_t(x)$ est un **séparateur linéaire dans cet espace de redescription construit itérativement**. On retrouve le thème des **noyaux** des SVM : projeter dans un espace plus riche et séparer linéairement. La différence : **ici l'espace est construit pas à pas à partir des données**, pas défini a priori par un noyau.

### 7.6 Autres points de vue (diapo 82)

- **Théorie des jeux** : l'**inspiration originale** du boosting (un jeu entre l'apprenant et un adversaire qui choisit les exemples difficiles) ;
- **Minimisation d'une perte surrogée** : le point de vue ci-dessus (« vue plus haut et plus loin ») ;
- **Géométrie de l'information**.

---

## 8. Bilan pratique, limites et variantes

*(diapos 84 à 92)*

### 8.1 Avantages pratiques d'AdaBoost (diapo 85)

- **(Très) rapide**, **simple et facile à programmer** ;
- **un seul paramètre à régler : le nombre d'étapes $T$** ;
- applicable à de nombreux domaines en **choisissant bien le classifieur faible** (réseau de neurones, C4.5…) ;
- **pas de sur-spécialisation** grâce à la maximisation des marges ;
- adaptable à des $h_t:X\to\mathbb R$ (le **signe** donne la classe, la **valeur absolue $|h_t(x)|$ donne la confiance**) ;
- adaptable aux problèmes **multi-classes** et **multi-étiquettes** ;
- permet de **détecter les exemples aberrants** (*outliers*) : ceux dont le poids ne cesse de grimper sont ceux que personne n'arrive à classer.

### 8.2 Quand ça ne marche pas (diapo 86)

Rappel du **théorème « no free lunch »** : aucun algorithme n'est meilleur partout. Le boosting est **inadapté** quand :
- **pas assez de données** ;
- **comité trop restreint** (apprenants faibles trop peu variés) ;
- **apprenants faibles trop stables** (ils donnent toujours la même hypothèse malgré les poids ; l'argument vaut surtout pour le bagging) ;
- **apprenants faibles trop forts** : ils peuvent déjà **surapprendre par eux-mêmes** ;
- **données bruitées** : AdaBoost, en sur-pondérant les exemples difficiles, **s'acharne sur les exemples mal étiquetés** (les erreurs d'étiquetage deviennent la « cible » principale). La perte exponentielle donne un poids exorbitant aux aberrants.

### 8.3 Avantages et difficultés (diapo 87)

| Avantages | Difficultés |
|---|---|
| **Méta-algorithme** : utilise n'importe quel apprenant faible | Difficile d'incorporer des **connaissances a priori** |
| En principe un seul paramètre ($T$) | Difficile de savoir **comment régulariser** |
| Facile à programmer | Le **meilleur choix d'apprenant faible n'est pas évident** |
| **Performances théoriques garanties** | Avec des arbres alignés sur les axes, les **frontières sont très irrégulières** (non interprétables) |

### 8.4 Résumé (diapo 88)

- La prédiction finale est un **vote pondéré** de plusieurs prédictions.
- Méthode **itérative** : **chaque classifieur dépend des précédents** (donc **non indépendants**, contrairement aux autres méthodes de vote).
- Les **exemples sont pondérés différemment**.
- Le poids d'un exemple **reflète la difficulté** qu'ont eue les classifieurs précédents à l'apprendre.

### 8.5 Variantes : d'autres fonctions de perte (diapos 89 à 91)

AdaBoost minimise une **perte exponentielle**. On peut en choisir d'autres :
- **LogitBoost** : utilise la **perte logistique (logloss)**. Avantages : **pas de poids exorbitant sur les points aberrants** (la perte logistique croît seulement linéairement pour de grosses erreurs, tandis que l'exponentielle explose), et **possibilité de calculer des probabilités** associées aux étiquettes.

---

## 9. Applications : la détection de visages

*(diapos 92 à 102)*

**Viola & Jones (2004)** : détection robuste de visages **en temps réel** (**15 images/seconde sur un smartphone**), images de 384×288 en niveaux de gris, visages à **toute échelle**. Défis (diapo 95) :
- identifier des **descripteurs** pertinents ;
- les **calculer rapidement** ;
- les **organiser** en un système performant : peu de **faux négatifs** (visages ratés) *et* peu de **faux positifs** (fausses alarmes).

**Les descripteurs.** Plus de **3 000 000 000** (toutes positions et échelles, avec un seuil à fixer pour chacun).

**Le rôle du boosting (diapo 99).** Chaque descripteur est associé à une **souche de décision** (un seul test de seuil). **AdaBoost sélectionne** une séquence de descripteurs utiles : **200** dans l'étude. **Mais ce serait trop coûteux** à évaluer (≈ 0,7 s par image).

**La cascade (diapo 100).** On organise les détecteurs **en cascade** : les premiers étages sont **très simples et rapides**, et **éliminent le plus tôt possible les négatifs** (l'immense majorité des fenêtres d'une image ne contiennent pas de visage). Seules les fenêtres qui survivent passent aux étages suivants, plus coûteux.

> 💡 **Analogie.** Un contrôle de sécurité en plusieurs étapes : un coup d'œil rapide écarte 90 % des gens, puis un contrôle plus poussé pour le reste. Le temps total est dominé par le premier filtre, bon marché.

**Autre application (diapo 102) :** reconnaissance du contour du rein.

---

## 10. Gradient boosting

*(diapos 103 à 109)*

### 10.1 Le principe (diapo 103)

Le gradient boosting apprend, comme AdaBoost, **séquentiellement** une combinaison de classifieurs faibles (en général des **arbres de décision**), mais :

| | **AdaBoost** | **Gradient boosting** |
|---|---|---|
| Combinaison | Vote **pondéré** | Somme **non pondérée** (le pas est dans le taux d'apprentissage) |
| Ce que le suivant apprend | Les **mêmes cibles**, avec des **exemples repondérés** | Les **résidus** (écarts entre valeur désirée et prédiction) |
| Fonction de perte | **Exponentielle** | **Toute fonction différentiable** (souvent entropique en classification) |

**Comparaison pratique (diapo 103).** Le gradient boosting :
- **gère mieux** les données **complexes** (images, texte) ou **bruitées** qu'AdaBoost ;
- mais **exige un réglage fin de plusieurs hyperparamètres** et est **davantage susceptible de surapprentissage**.

### 10.2 L'idée des résidus (diapo 104)

Soit le classifieur combiné à l'étape $t$, $h_t$. Le **résidu** sur l'exemple $i$ est $r_i=y_i-h_t(x_i)$. Le nouvel apprenant $f_{t}$ est entraîné à **prédire ce résidu**, puis on corrige :

$$h_{t+1}(x)=h_t(x)+\eta\,f_t(x)$$

où $\eta$ est le **taux d'apprentissage** (*learning rate*).

> 💡 **Analogie du golfeur.** Au premier coup, tu envoies la balle vers le trou, mais elle s'arrête à 20 m. Le **résidu**, c'est cette distance restante. Le coup suivant vise à corriger **cette distance**, pas à repartir de zéro. Chaque nouvel arbre est un **coup de correction**. Le taux $\eta$ règle l'amplitude : petits coups prudents ($\eta$ petit) ou grands coups ($\eta$ grand, risqué).

**Lien avec la descente de gradient :** pour la perte quadratique, le résidu $y-h(x)$ est (à un facteur près) l'**opposé du gradient** de la perte par rapport à la prédiction. Le gradient boosting fait donc une **descente de gradient dans l'espace des fonctions**, d'où son nom.

### 10.3 Exemple chiffré (diapos 105 à 107)

Données : 5 exemples, 3 attributs, classe $y\in\{0,1\}$ (trois « 1 » et deux « 0 »).

| $x_1$ | $x_2$ | $x_3$ | $y$ |
|---|---|---|---|
| 12 | 750 | 4 | 0 |
| 15 | 900 | 7 | 0 |
| 18 | 850 | 6 | 1 |
| 9 | 820 | 12 | 1 |
| 6 | 790 | 6 | 1 |

**Étape 0 : une prédiction initiale constante.** La proportion de « 1 » est $3/5=0{,}6$ : $h_0(x)=0{,}6$ pour tous. (Via la régression logistique : cote $P(y=1)/P(y=0)=3/2=1{,}5$, d'où $P(y=1)=\frac{1{,}5}{1+1{,}5}=0{,}6$.)

> ⚠️ **Point d'attention.** La diapo écrit « $\log\frac32=\log1{,}5=0{,}176$ ». Or $0{,}176$ est le logarithme **décimal** de $1{,}5$ ; avec le **logarithme népérien** (celui de la régression logistique, $P=\frac{e^V}{1+e^V}$), on a $V=\ln1{,}5\approx0{,}405$. C'est cette dernière valeur qui redonne bien $P=0{,}6$ avec la formule de la diapo.

**Étape 1 : les résidus** $y-h_0$ : $-0{,}6\ ;\ -0{,}6\ ;\ +0{,}4\ ;\ +0{,}4\ ;\ +0{,}4$ ✔ (vérifié).

**Étape 2 : un arbre prédit ces résidus.** L'arbre de la diapo 106 teste « $x_1>10$ ? » puis « $x_2>800$ ? » (à droite) ou « $x_3>8$ ? » (à gauche). Le point central : **chaque feuille prédit la moyenne des résidus des exemples qu'elle contient.** Par exemple, une feuille contenant les résidus $-0{,}6$ et $+0{,}4$ prédit $\frac{-0{,}6+0{,}4}{2}=-0{,}1$ (la diapo écrit « 0,6+0,4 / 2 = 0,1 » ; c'est le signe « − » qui a sauté à l'extraction : la moyenne de $-0{,}6$ et $0{,}4$ est bien $-0{,}1$).

**Étape 3 : mise à jour** $h_1=h_0+\eta f$ avec $\eta=0{,}5$ (diapo 107), puis **nouveaux résidus** $y-h_1$. Sur la diapo, les nouveaux résidus (valeurs absolues) sont plus petits qu'avant : $-0{,}45\ ;\ -0{,}70\ ;\ +0{,}20\ ;\ +0{,}05\ ;\ +0{,}05$.

> ⚠️ **Je n'ai pas pu reconstituer entièrement ces dernières valeurs.** Les colonnes $h_1(x)$ de la diapo (0,45 ; 0,70 ; 0,80 ; 0,95 ; 0,95) ne se retrouvent pas simplement à partir de $h_0=0{,}6$ et des feuilles lisibles dans le texte extrait. Retiens le **mécanisme** (résidus → arbre → correction pondérée par $\eta$ → nouveaux résidus plus petits), et regarde la diapo 107 pour le détail des chiffres.

**Ce qu'on voit :** les résidus **diminuent** d'étape en étape pour les exemples bien traités ; on répète jusqu'à ce qu'ils soient négligeables ou que le nombre d'arbres soit atteint.

### 10.4 Les hyperparamètres (diapo 108)

1. **Taux d'apprentissage** $\eta$ : petit = apprentissage lent mais plus sûr (demande plus d'arbres) ;
2. **Nombre d'apprenants faibles** (arbres) ;
3. **Profondeur des arbres** : arbres peu profonds = apprenants faibles, qui captent des interactions limitées ;
4. **Taux de sous-échantillonnage** (*subsampling*) : fraction des données utilisée pour chaque arbre ; introduit du hasard, **réduit la variance** (idée empruntée au bagging).

Démo en ligne indiquée en diapo 109 (*Gradient Boosting playground*, A. Rogozhnikov).

---

## 11. Le bagging

*(diapos 111 à 116)*

### 11.1 Le principe (diapo 112)

**Bagging = Bootstrap AGGregatING** (Breiman, 1996).

1. **Génération de $k$ échantillons « indépendants »** par **tirage avec remise** dans l'échantillon $S_m$ de taille $m$ (ce sont des **échantillons bootstrap**) ;
2. Pour chaque échantillon, **apprentissage d'un classifieur** avec **le même algorithme** ;
3. Prédiction finale pour un nouvel exemple : **vote simple** des classifieurs (ou moyenne en régression).

> 💡 **Analogie.** Pour estimer un résultat incertain, tu demandes à $k$ analystes qui n'ont chacun vu qu'un **échantillon différent** (et un peu redondant) du dossier. Leurs avis diffèrent, mais **la moyenne des avis est plus stable** que n'importe lequel d'entre eux.

**Tirage avec remise : ce que ça donne.** Comme chaque tirage est avec remise, un échantillon bootstrap de taille $m$ contient **des doublons** et laisse de côté une partie des exemples. ➕ La probabilité qu'un exemple donné **ne soit pas tiré** vaut $(1-\frac1m)^m\to e^{-1}\approx0{,}368$ : **environ 63 % des exemples distincts** sont présents dans chaque échantillon, et ~37 % sont « hors sac » (utilisables pour estimer l'erreur gratuitement).

### 11.2 Boosting contre bagging

| | Bagging | Boosting |
|---|---|---|
| Entraînement | **Parallèle**, indépendant | **Séquentiel**, dépendant |
| Données de chaque apprenant | **Rééchantillonnage** uniforme avec remise | **Repondération** (ou rééchantillonnage selon $D_t$) |
| Vote | **Simple** | **Pondéré** par la qualité |
| Cible principale | **Variance** | **Biais** (et marge) |
| Apprenant idéal | **Instable et fort** (arbres profonds) | **Faible** (souches) |
| Sensibilité au bruit | Faible | **Forte** |

Les diapos 114-115 comparent les deux, avec des **arbres de décision** puis des **souches** (« decision stump »), sur un jeu de données (Zhou, 2021). Avec des **souches** (biais fort, très stables), le bagging n'aide presque pas (toutes les souches se ressemblent) : **le bagging a besoin d'apprenants instables**, c'est l'argument de la diapo 86.

### 11.3 Pourquoi ça marche (diapo 116)

Ce que l'on dit souvent : **le bagging réduit la variance en laissant le biais inchangé.**

> 💡 **Lecture.** En moyennant des modèles entraînés sur des échantillons différents, on **lisse leurs fluctuations** (variance ↓), mais la **moyenne de modèles qui font tous la même erreur systématique conserve cette erreur** (biais =). C'est pour ça qu'on l'utilise avec des **arbres profonds** (biais faible, variance forte, cf. cours sur les arbres) : on ne garde que le bon côté.

La diapo ajoute que ce n'est **pas encore complètement compris** (référence : Y. Grandvalet, *« Bagging equalizes influence »*, Machine Learning, 2004 : le bagging **égalise l'influence** des exemples).

---

## 12. Pour la suite du cours (hors fichier fourni)

> Cette section est un **complément de ma part** : le fichier fourni s'arrête avant ces parties. Envoie-moi les diapos 117+ si tu veux une version fidèle à ton cours.

Le plan annonce (diapo 10) :

**Forêts aléatoires (*random forests*).** ➕ Idée : un **bagging d'arbres de décision**, avec une astuce de plus. À **chaque nœud**, on ne considère qu'un **sous-ensemble aléatoire d'attributs** pour choisir la meilleure coupure. Cela **décorrèle** davantage les arbres (rappel de §2.2 : moyenner des modèles décorrélés réduit mieux la variance). Vote majoritaire final.

**XGBoost : le boosting d'arbres.** ➕ Une implémentation très optimisée du **gradient boosting d'arbres**, avec de la **régularisation** explicite sur la complexité des arbres et un calcul efficace. Très utilisé dans les concours de données tabulaires.

**Autres méthodes d'ensemble.** ➕ Par exemple le *stacking* (un méta-modèle apprend à combiner les prédictions d'autres modèles).

---

## 13. Fiche récapitulative

| Concept | En une phrase |
|---|---|
| **Méthode d'ensemble** | Combiner plusieurs apprenants faibles pour faire mieux que chacun |
| **Sagesse des foules** | Estimations bruitées, non biaisées, symétriques, indépendantes → la moyenne annule les erreurs |
| **Décorrélation** | Plus les apprenants sont indépendants, plus la variance de l'ensemble diminue |
| **Apprenant faible** | Erreur $\le\frac12-\gamma$ : juste un peu mieux que le hasard |
| **Boosting** | Convertir des règles faibles en une règle forte (prouvé, Schapire 1990) |
| **AdaBoost** | Séquentiel : repondère les exemples (↑ mal classés, ↓ bien classés), vote pondéré par $\alpha_t$ |
| **$\alpha_t$** | $\frac12\ln\frac{1-\varepsilon_t}{\varepsilon_t}$ : grand si $h_t$ précis, nul si au niveau du hasard |
| **Poids $D_t$** | Reflète la difficulté qu'ont eue les apprenants précédents sur l'exemple |
| **Erreur d'apprentissage** | Décroît exponentiellement avec $T$ si chaque $h_t$ bat le hasard |
| **Énigme** | L'erreur de test ne remonte pas même quand l'apprentissage est à 0 |
| **Marge** | Poids des votes corrects − poids des votes erronés : confiance de la prédiction |
| **Explication par la marge** | Boosting augmente la marge → robustesse → bonne généralisation (comme les SVM) |
| **Limites** | Données bruitées, peu de données, apprenants trop forts ou trop stables |
| **LogitBoost** | Perte logistique : moins sensible aux aberrants, donne des probabilités |
| **Viola-Jones** | AdaBoost sélectionne les descripteurs ; cascade pour éliminer vite les négatifs |
| **Gradient boosting** | Chaque arbre prédit les **résidus** ; $h_{t+1}=h_t+\eta f_t$ ; perte différentiable quelconque |
| **Bagging** | Bootstrap + vote simple ; réduit la variance sans changer le biais |
| **Bootstrap** | Tirage avec remise ; ≈ 63 % d'exemples distincts par échantillon |
| **Bagging vs boosting** | Parallèle/indépendant/variance vs séquentiel/dépendant/biais |

---

## 14. Quiz d'auto-évaluation

> **Mode d'emploi.** Réponds d'abord sans regarder, puis clique sur « Voir la réponse ». Compte un point par bonne réponse. Barème indicatif à la fin. Les questions Q36 à Q38 portent sur la section 12 (hors fichier) : elles sont facultatives.

### Partie A : Pourquoi combiner

**Q1.** Que montre l'expérience de Galton (1906) ?
<details><summary>Voir la réponse</summary>

Que la **moyenne des estimations** de 787 participants est bien plus précise (erreur < 1 millième) que **le meilleur participant** (erreur > 1 centième) : des estimateurs individuellement médiocres, combinés, donnent un très bon résultat.
</details>

**Q2.** Quelles sont les trois conditions sur les estimations individuelles pour que la moyenne fonctionne ?
<details><summary>Voir la réponse</summary>

Estimations **non biaisées**, **symétriques**, et **indépendantes**.
</details>

**Q3.** Si l'on moyenne 100 estimateurs indépendants de variance $\sigma^2$ chacun, que vaut la variance de la moyenne ? Que se passe-t-il s'ils sont corrélés ?
<details><summary>Voir la réponse</summary>

$\sigma^2/100$. S'ils sont corrélés (coefficient $\rho$), la variance vaut $\rho\sigma^2+\frac{1-\rho}{100}\sigma^2$ : elle ne descend jamais sous $\rho\sigma^2$. **La décorrélation est essentielle.**
</details>

**Q4.** Cite les deux grandes différences de construction entre bagging et boosting.
<details><summary>Voir la réponse</summary>

Bagging : classifieurs **indépendants/parallèles**, **vote simple**, échantillons bootstrap. Boosting : **séquentiel**, chaque classifieur **dépend des précédents**, exemples **repondérés**, **vote pondéré**.
</details>

### Partie B : Le boosting

**Q5.** Qu'est-ce qu'un apprenant « faible » ? Pourquoi exige-t-on qu'il batte le hasard ?
<details><summary>Voir la réponse</summary>

Un apprenant d'erreur $\le\frac12-\gamma$ avec $\gamma>0$ : **juste un peu meilleur que le hasard**. Sinon il n'apporte aucune information utile (poids $\alpha_t\le0$) et la combinaison ne progresse pas.
</details>

**Q6.** Que dit Schapire (1990) sur l'apprentissage fort et faible ?
<details><summary>Voir la réponse</summary>

Qu'ils sont **équivalents** : tout apprenant faible peut être **converti en apprenant fort**. Le boosting est la méthode qui le fait.
</details>

**Q7.** Décris le principe du boosting en deux phrases.
<details><summary>Voir la réponse</summary>

On entraîne des apprenants **les uns après les autres**, en **augmentant le poids des exemples mal classés** et en **diminuant celui des bien classés**, pour que chaque nouvel apprenant se concentre sur les cas difficiles. La prédiction finale est un **vote pondéré** des apprenants, les plus précis comptant le plus.
</details>

**Q8.** Dans l'analogie des courses de chevaux, à quoi correspondent le choix des courses et la combinaison des règles ?
<details><summary>Voir la réponse</summary>

Choix des courses = **choix de la distribution $D_t$ sur les exemples** (se concentrer sur les plus difficiles). Combinaison des règles = **vote majoritaire pondéré**.
</details>

**Q9.** Écris la formule du poids $\alpha_t$ d'AdaBoost et donne sa valeur pour $\varepsilon_t=0{,}5$ et $\varepsilon_t=0{,}25$.
<details><summary>Voir la réponse</summary>

$\alpha_t=\frac12\ln\frac{1-\varepsilon_t}{\varepsilon_t}$. Pour $\varepsilon_t=0{,}5$ : $\alpha_t=0$ (aucune voix). Pour $\varepsilon_t=0{,}25$ : $\frac12\ln3\approx0{,}549$.
</details>

**Q10.** Comment AdaBoost met-il à jour le poids d'un exemple bien classé ? d'un exemple mal classé ?
<details><summary>Voir la réponse</summary>

Bien classé : poids **multiplié par $e^{-\alpha_t}<1$** (baisse). Mal classé : **multiplié par $e^{+\alpha_t}>1$** (hausse). Puis division par la constante de normalisation $Z_t$.
</details>

**Q11.** Premier tour d'AdaBoost sur 20 exemples avec 5 erreurs. Quel est le poids de chaque exemple mal classé après mise à jour ? Et de chaque bien classé ?
<details><summary>Voir la réponse</summary>

$\alpha_1=\frac12\ln3\approx0{,}549$, $Z_1\approx0{,}866$. Mal classés : $0{,}05\times1{,}732/0{,}866=\mathbf{0{,}1}$. Bien classés : $0{,}05\times0{,}577/0{,}866\approx\mathbf{0{,}033}$. (Les 5 mal classés pèsent au total 0,5.)
</details>

**Q12.** Pourquoi dit-on qu'après la mise à jour, $h_t$ est « au niveau du hasard » pour $D_{t+1}$ ?
<details><summary>Voir la réponse</summary>

Parce que le poids total des exemples mal classés par $h_t$ devient **exactement 0,5** sous $D_{t+1}$. L'apprenant suivant doit donc apporter une information **nouvelle** : cela décorrèle les apprenants.
</details>

**Q13.** Pourquoi le vote final est-il pondéré et non simple ?
<details><summary>Voir la réponse</summary>

Parce que les apprenants n'ont **pas la même qualité** : un classifieur précis (erreur faible) doit peser davantage qu'un classifieur à peine meilleur que le hasard.
</details>

**Q14.** Que signifie « l'erreur est mesurée sur $D_t$ » ?
<details><summary>Voir la réponse</summary>

L'erreur $\varepsilon_t$ est la **somme des poids** des exemples mal classés, pas leur nombre brut : se tromper sur un exemple difficile (poids élevé) coûte plus cher.
</details>

### Partie C : Pourquoi ça marche

**Q15.** Que dit le résultat sur l'erreur d'apprentissage d'AdaBoost ?
<details><summary>Voir la réponse</summary>

Si chaque apprenant a une erreur $\le\frac12-\gamma$, l'erreur d'apprentissage de l'hypothèse finale est bornée par $e^{-2\gamma^2T}$ : elle **décroît exponentiellement** avec le nombre d'étapes $T$.
</details>

**Q16.** Quel phénomène « déraisonnable » observe-t-on sur l'erreur de test, et pourquoi est-il surprenant ?
<details><summary>Voir la réponse</summary>

L'erreur d'apprentissage atteint zéro mais l'**erreur de test continue de baisser ou ne remonte pas** (même après 1000 étapes et 2 millions de nœuds). C'est surprenant car on attendrait du **surapprentissage** quand le modèle devient très complexe.
</details>

**Q17.** Définis la marge d'un exemple dans le boosting. Que signifie une marge proche de +1 ?
<details><summary>Voir la réponse</summary>

Marge = **poids des classifieurs ayant voté correctement − poids de ceux en erreur** (normalisé). Proche de +1 : presque tous les votants (pondérés) ont raison → prédiction **très confiante**.
</details>

**Q18.** Résume l'explication du boosting par la maximisation des marges.
<details><summary>Voir la réponse</summary>

À chaque étape, AdaBoost place le plus de poids sur les exemples de **plus faible marge** tout en améliorant celle des autres. L'hypothèse finale, bien que complexe, est **proche d'une hypothèse simple**, de grande marge, donc a une **erreur de généralisation faible**, comme pour les SVM.
</details>

**Q19.** Quel est le parallèle entre le boosting et les SVM ?
<details><summary>Voir la réponse</summary>

Dans les deux cas, la **marge** (et non le nombre de paramètres ou de dimensions) contrôle la capacité et la généralisation. Le boosting construit aussi un **espace de redescription** itératif (chaque $h_t$ est une nouvelle caractéristique) où la séparation est linéaire.
</details>

**Q20.** Cite deux autres perspectives pour expliquer le boosting.
<details><summary>Voir la réponse</summary>

La **théorie des jeux** (inspiration originale), la **minimisation d'une perte surrogée** (exponentielle), la **géométrie de l'information**.
</details>

### Partie D : Bilan, variantes, applications

**Q21.** Donne quatre avantages pratiques d'AdaBoost.
<details><summary>Voir la réponse</summary>

Très rapide ; simple à programmer ; un seul paramètre ($T$) ; utilisable avec n'importe quel apprenant faible ; peu de surapprentissage grâce aux marges ; adaptable au multi-classes ; détection des aberrants.
</details>

**Q22.** Dans quels cas le boosting échoue-t-il ?
<details><summary>Voir la réponse</summary>

Pas assez de données ; comité trop restreint ; apprenants trop stables ; apprenants **trop forts** (surapprennent seuls) ; surtout **données bruitées** (AdaBoost s'acharne sur les exemples mal étiquetés).
</details>

**Q23.** Pourquoi AdaBoost est-il sensible aux données aberrantes ? Comment LogitBoost y remédie-t-il ?
<details><summary>Voir la réponse</summary>

AdaBoost minimise une perte **exponentielle** : les aberrants (toujours mal classés) voient leur poids **exploser**. **LogitBoost** utilise la **perte logistique**, qui croît seulement linéairement : pas de poids exorbitant, et il donne en plus des **probabilités**.
</details>

**Q24.** Dans la détection de visages de Viola-Jones, à quoi sert AdaBoost et à quoi sert la cascade ?
<details><summary>Voir la réponse</summary>

AdaBoost **sélectionne** parmi plus de 3 milliards de descripteurs une séquence de ~200 (chacun associé à une souche de décision). La **cascade** organise ces détecteurs pour **éliminer le plus tôt possible les négatifs** (fenêtres sans visage) avec des étages très rapides, afin de tenir le temps réel.
</details>

### Partie E : Gradient boosting

**Q25.** Quelles sont les trois différences entre gradient boosting et AdaBoost ?
<details><summary>Voir la réponse</summary>

1. Chaque modèle apprend les **résidus**, au lieu de changer la distribution des exemples ; 2. la combinaison est une **somme non pondérée** (le pas est dans le taux d'apprentissage) ; 3. la perte peut être **n'importe quelle fonction différentiable** (au lieu de l'exponentielle).
</details>

**Q26.** Dans l'exemple (3 « 1 » et 2 « 0 »), quelle est la prédiction initiale et quels sont les résidus ?
<details><summary>Voir la réponse</summary>

$h_0=3/5=0{,}6$. Résidus $y-h_0$ : $-0{,}6$ pour les « 0 » (×2) et $+0{,}4$ pour les « 1 » (×3).
</details>

**Q27.** Une feuille de l'arbre qui prédit les résidus contient les résidus $-0{,}6$ et $+0{,}4$. Que prédit-elle ?
<details><summary>Voir la réponse</summary>

Leur **moyenne** : $\frac{-0{,}6+0{,}4}{2}=\mathbf{-0{,}1}$.
</details>

**Q28.** Écris la mise à jour du gradient boosting et explique le rôle du taux d'apprentissage $\eta$.
<details><summary>Voir la réponse</summary>

$h_{t+1}(x)=h_t(x)+\eta f_t(x)$. $\eta$ règle l'**amplitude de la correction** : petit = apprentissage lent et prudent (plus d'arbres nécessaires, moins de surapprentissage) ; grand = rapide mais risqué.
</details>

**Q29.** Cite les quatre hyperparamètres du gradient boosting vus en cours.
<details><summary>Voir la réponse</summary>

Taux d'apprentissage ; nombre d'arbres ; profondeur des arbres ; taux de sous-échantillonnage.
</details>

### Partie F : Bagging

**Q30.** Décris les trois étapes du bagging.
<details><summary>Voir la réponse</summary>

1. Générer $k$ échantillons par **tirage avec remise** dans l'échantillon d'apprentissage ; 2. entraîner un classifieur sur chacun avec le **même algorithme** ; 3. prédire par **vote simple** (ou moyenne).
</details>

**Q31.** Quelle proportion d'exemples distincts contient en moyenne un échantillon bootstrap ?
<details><summary>Voir la réponse</summary>

Environ **63 %** : la probabilité qu'un exemple ne soit **pas** tiré est $(1-1/m)^m\to e^{-1}\approx0{,}37$.
</details>

**Q32.** Que réduit principalement le bagging : le biais ou la variance ? Quel type d'apprenant lui convient ?
<details><summary>Voir la réponse</summary>

La **variance**, en laissant le biais inchangé. Il convient à des apprenants **instables** à faible biais (par exemple des arbres profonds) ; avec des apprenants très stables (souches), il n'apporte presque rien.
</details>

**Q33.** Compare bagging et boosting sur cinq critères (construction, données, vote, cible, sensibilité au bruit).
<details><summary>Voir la réponse</summary>

Bagging : parallèle, rééchantillonnage bootstrap, vote simple, réduit la variance, peu sensible au bruit. Boosting : séquentiel, repondération des exemples, vote pondéré, réduit surtout le biais (et augmente la marge), **sensible au bruit**.
</details>

### Partie G : Synthèse

**Q34.** Pourquoi dit-on que le boosting peut utiliser « n'importe quel apprenant faible » ? Est-ce toujours une bonne idée de prendre un apprenant très puissant ?
<details><summary>Voir la réponse</summary>

C'est un **méta-algorithme** : il ne dépend que de la capacité de l'apprenant à produire une hypothèse meilleure que le hasard sur la distribution $D_t$. Mais un apprenant **trop fort** peut surapprendre par lui-même (erreur d'apprentissage nulle dès la première étape, plus rien à repondérer) : on préfère des apprenants faibles (souches, arbres peu profonds).
</details>

**Q35 (synthèse).** Explique à un ami, sans formule, pourquoi combiner plusieurs modèles médiocres peut battre un bon modèle, et la différence entre bagging et boosting.
<details><summary>Voir la réponse</summary>

Comme la foule de Galton : chaque modèle se trompe, mais **pas de la même façon** ; en combinant leurs avis, les erreurs se compensent. Dans le **bagging**, on entraîne beaucoup de modèles **indépendants** sur des versions rééchantillonnées des données et on **vote** : cela stabilise les prédictions (réduit la variance). Dans le **boosting**, on entraîne les modèles **à la suite** : chacun se concentre sur les exemples que les précédents ont ratés, et les modèles les plus fiables ont un vote plus fort. On transforme ainsi des règles à peine meilleures que le hasard en un classifieur très précis, avec une grande marge, ce qui explique sa bonne généralisation.
</details>

### Partie H : Facultatif, hors fichier fourni (section 12)

**Q36.** Quelle astuce supplémentaire distingue une forêt aléatoire d'un simple bagging d'arbres ?
<details><summary>Voir la réponse</summary>

À chaque nœud, on ne considère qu'un **sous-ensemble aléatoire d'attributs** pour choisir la coupure, ce qui **décorrèle** davantage les arbres.
</details>

**Q37.** À quelle méthode correspond XGBoost dans la famille des méthodes d'ensemble ?
<details><summary>Voir la réponse</summary>

À une implémentation très optimisée du **gradient boosting d'arbres**, avec une **régularisation** de la complexité des arbres.
</details>

**Q38.** Pourquoi une forêt aléatoire peut-elle être meilleure qu'un bagging d'arbres classique ?
<details><summary>Voir la réponse</summary>

Parce que les arbres sont **moins corrélés** entre eux : la variance de la moyenne baisse davantage (cf. $\rho\sigma^2+\frac{1-\rho}{k}\sigma^2$).
</details>

---

### Barème indicatif (35 questions obligatoires)

| Score | Niveau |
|---|---|
| 0 – 14 | À retravailler : relis les sections 2, 4 et 5 |
| 15 – 24 | Bases acquises : approfondis la marge (7) et le gradient boosting (10) |
| 25 – 31 | Bonne maîtrise |
| 32 – 35 | Excellent ! Tu peux expliquer les méthodes d'ensemble à quelqu'un d'autre |

> 💡 **Fil rouge de révision.** *Modèles faibles et décorrélés → on combine → bagging : parallèle, vote simple, réduit la variance → boosting : séquentiel, repondère les exemples difficiles, vote pondéré, augmente la marge → gradient boosting : chaque arbre corrige les résidus.*
