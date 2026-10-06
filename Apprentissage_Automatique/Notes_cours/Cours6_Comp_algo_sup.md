# Comparer des algorithmes de classification supervisée, expliqué pas à pas

> Basé sur le cours « Comparaison d'algorithmes de classification supervisée » d'A. Cornuéjols (AgroParisTech – INRAE, 38 diapos).
> Les renvois **(diapo N)** te permettent de retrouver le passage dans ton PDF.
> Les paragraphes marqués **➕ Complément** ajoutent une précision qui n'est pas explicitement sur les diapos.
> Les encadrés **⚠️ Point d'attention** signalent des coquilles ou ambiguïtés repérées dans les diapos (je les ai vérifiées par le calcul).
> Les formules sont en LaTeX (`$...$`) : elles s'affichent dans Obsidian, Typora, VS Code, Jupyter, etc.

---

## Sommaire

1. [La grande image](#1-la-grande-image)
2. [Pourquoi évaluer est difficile (introduction)](#2-pourquoi-évaluer-est-difficile-introduction)
3. [Les bases : test d'hypothèse, p-value, intervalle de confiance](#3-les-bases--test-dhypothèse-p-value-intervalle-de-confiance)
4. [Un algorithme, plusieurs tests : le t-test](#4-un-algorithme-plusieurs-tests--le-t-test)
5. [Comparer deux algorithmes sur le même jeu de données : t-test apparié](#5-comparer-deux-algorithmes-sur-le-même-jeu-de-données--t-test-apparié)
6. [Le problème de l'indépendance et la validation croisée 5×2](#6-le-problème-de-lindépendance-et-la-validation-croisée-52)
7. [Le test de McNemar](#7-le-test-de-mcnemar)
8. [Plusieurs jeux de données : le test de Wilcoxon](#8-plusieurs-jeux-de-données--le-test-de-wilcoxon)
9. [Plusieurs algorithmes sur plusieurs jeux : Friedman puis Nemenyi](#9-plusieurs-algorithmes-sur-plusieurs-jeux--friedman-puis-nemenyi)
10. [Lire les résultats avec prudence](#10-lire-les-résultats-avec-prudence)
11. [Quel test pour quelle situation ?](#11-quel-test-pour-quelle-situation-)
12. [Fiche récapitulative](#12-fiche-récapitulative)
13. [Quiz d'auto-évaluation](#13-quiz-dauto-évaluation)

---

## 1. La grande image

Tu as entraîné deux classifieurs, A et B. A obtient 82 % de précision, B obtient 80 %. **A est-il meilleur ?** La question paraît simple, mais la réponse honnête est : *peut-être, peut-être pas*. Ces 2 points d'écart peuvent venir d'un vrai avantage de A… ou simplement de la chance du découpage des données.

Tout le cours répond à une seule question :

> **Comment décider qu'une différence de performance observée est « réelle » et pas due au hasard ?**

La réponse passe par les **tests de significativité statistique**. Et comme le bon test dépend de la situation, le cours les range selon deux critères (diapo 5) :

| | **2 algorithmes** | **k algorithmes** |
|---|---|---|
| **Un seul jeu de données** | t-test apparié, McNemar, 5×2 cv | *(non traité)* |
| **Plusieurs jeux de données** | Wilcoxon (signed-rank) | Friedman + Nemenyi |

Si tu retiens ce tableau, tu as la carte du cours. Chaque section suivante remplit une case.

---

## 2. Pourquoi évaluer est difficile (introduction)

*(diapos 2 à 4)*

### 2.1 Ce que veut dire « bien évaluer » une IA

La diapo 2 liste quatre qualités qu'on voudrait mesurer, au-delà de la simple « précision moyenne » :

| Qualité | En une phrase | Exemple |
|---|---|---|
| **Consistance** (*consistency*) | Réussir **à chaque fois**, pas seulement parfois | Un modèle juste 90 % du temps mais de façon imprévisible n'est pas consistant |
| **Robustesse** | Fonctionner même quand les **conditions sont imparfaites** | Images bruitées, données légèrement différentes de l'entraînement |
| **Calibration** | **Savoir** quand on a réussi ou échoué | Un modèle qui dit « je suis sûr à 99 % » doit avoir raison 99 % du temps |
| **Sûreté opérationnelle** | **Éviter les erreurs catastrophiques** | Mieux vaut dix petites erreurs qu'une seule fatale |

La diapo ajoute que la **créativité** est « l'archétype d'une tâche invérifiable » : on ne sait même pas clairement la tester chez l'humain (sciences cognitives, neurosciences), donc encore moins chez une IA.

> 💡 **Message.** Une seule métrique ne dit pas tout. Et certaines dimensions de l'intelligence échappent complètement aux tests quantitatifs.

### 2.2 Le piège des benchmarks : « chercher sous le réverbère »

*(diapos 3-4, référence : « Position : There are futures that benchmark-driven AI cannot see », Lotfi, Iranmanesh et al., ICML 2026)*

Un **benchmark** est un jeu de données standard sur lequel tout le monde compare ses algorithmes. Son atout : l'**efficacité**. Pas besoin de passer des années à relire des articles : si un modèle bat l'état de l'art sur le benchmark, c'est probablement un bon article.

Son défaut : il **rétrécit la vision collective** de la communauté.

> 💡 **L'histoire du réverbère** (l'image de la diapo 3 : une personne sous un lampadaire, la rue est sombre autour). Un homme cherche ses clés la nuit sous un réverbère. « Tu les as perdues ici ? » — « Non, plus loin, mais ici il y a de la lumière. » La communauté fait pareil : elle cherche **ce qui est facile à mesurer** avec les benchmarks, pas forcément ce qui est important.

Conséquence annoncée par le cours : évaluer la qualité d'un travail deviendra **plus coûteux**, et c'est un prix à payer, alors que la communauté « se précipite dans la direction opposée » (diapo 4).

**Pourquoi c'est dans un cours de statistique ?** Parce que le reste du cours donne les outils pour comparer *proprement*. Mais il faut garder en tête que même une comparaison statistiquement irréprochable ne mesure que ce que la métrique et le benchmark capturent.

---

## 3. Les bases : test d'hypothèse, p-value, intervalle de confiance

*(diapos 7, 8, 12)*

### 3.1 Le principe général d'un test (diapo 12)

La logique est celle d'un **raisonnement par l'absurde probabiliste** :

1. On suppose l'**hypothèse nulle $H_0$** : « il n'y a en fait **aucune différence** entre les algorithmes ».
2. On **caractérise la distribution** des différences de performance qu'on observerait *si $H_0$ était vraie* (ce que le hasard seul produit).
3. On regarde la différence **réellement observée**. Si elle est **très improbable sous $H_0$**, on **décide** qu'il y a une vraie différence.

> 💡 **Analogie du tribunal.** $H_0$ = « l'accusé est innocent ». On ne condamne que si les preuves seraient très improbables pour un innocent. Si elles sont ambiguës, on ne condamne pas… ce qui ne prouve pas l'innocence.

### 3.2 Estimer une précision et son incertitude (diapo 7)

Quand on mesure la précision $\hat a$ d'un classifieur sur $n$ exemples de test, ce n'est qu'une **estimation** de sa vraie précision $a$. La diapo suppose que $\hat a$ suit une loi **normale** autour de la vraie valeur, avec un écart-type $\sigma$.

- Si on connaît ces paramètres, on peut calculer un **intervalle de confiance**.
- Sinon, avec $n$ exemples, la variance d'un exemple peut être estimée par $\hat a(1-\hat a)$.

**D'où vient cette formule ?** Chaque exemple est soit bien classé (1), soit mal classé (0) : c'est une variable de Bernoulli de paramètre $a$, de variance $a(1-a)$. La moyenne de $n$ telles variables a une variance $n$ fois plus petite :

$$\sigma_{\hat a} = \sqrt{\frac{\hat a\,(1-\hat a)}{n}}$$

> 💡 **Intuition.** Plus $n$ est grand, plus l'estimation est précise (racine de $n$). Et la précision est la plus « floue » quand $a$ est proche de 0,5 (maximum de $a(1-a)$).

### 3.3 Exemple chiffré : tester une hypothèse nulle (diapo 8)

*Situation.* On fait l'hypothèse nulle « la vraie précision vaut 0,5 » (le classifieur ne fait pas mieux qu'une pièce à pile ou face). On observe **0,80** sur $n=100$ exemples.

1. Écart-type sous $H_0$ : $\sqrt{0{,}5\times(1-0{,}5)/100}=0{,}05$.
2. À quelle distance est 0,80 de 0,5 ? $(0{,}80-0{,}50)/0{,}05=\mathbf{6}$ écarts-types. C'est énorme.
3. On calcule la **p-value** : la probabilité, sous $H_0$, d'observer un résultat **au moins aussi extrême**. Ici $p\approx 1{,}97\times10^{-9}$.
4. On compare à un seuil $\alpha$ fixé à l'avance, typiquement **$\alpha=0{,}05$** (confiance de 95 %). Comme $p<\alpha$, on **rejette $H_0$**.

> ⚠️ **Point d'attention.** La diapo écrit « $\sqrt{9.5\,(1-0.5)/100}$ » : c'est une coquille. Il faut lire $0{,}5\,(1-0{,}5)/100$, qui donne bien 0,05. De plus, $1{,}97\times10^{-9}$ est la p-value **bilatérale** (deux côtés) ; unilatérale on aurait la moitié, environ $10^{-9}$.

### 3.4 Bien comprendre la p-value (encadré de la diapo 17)

> **p-value = probabilité d'obtenir un résultat au moins aussi « extrême » que celui observé, sous l'hypothèse $H_0$.**

Ce qu'elle **n'est pas** (erreurs fréquentes) :

| ❌ Faux | ✅ Vrai |
|---|---|
| « La probabilité que $H_0$ soit vraie » | La probabilité des **données** (ou pire) **si** $H_0$ est vraie |
| « Si $p>0{,}05$, les algorithmes sont équivalents » | On **ne peut pas rejeter** $H_0$ : manque de preuves, pas preuve d'égalité |
| « $p$ petit = grosse différence » | $p$ petit = différence **peu probable par hasard**, pas forcément grande (avec beaucoup de données, une différence minuscule peut être significative) |

### 3.5 ➕ Complément : les deux façons de se tromper

| | $H_0$ vraie (pas de différence) | $H_0$ fausse (vraie différence) |
|---|---|---|
| On **rejette** $H_0$ | **Erreur de type I** (faux positif), probabilité $\alpha$ | Bonne décision |
| On **ne rejette pas** $H_0$ | Bonne décision | **Erreur de type II** (faux négatif) |

Choisir $\alpha=0{,}05$, c'est accepter de se tromper (faux positif) dans 5 % des cas où il n'y a rien. Ce chiffre sera crucial à la section 9.

---

## 4. Un algorithme, plusieurs tests : le t-test

*(diapos 9 à 11)*

### 4.1 Le contexte : la validation croisée

On évalue un algorithme en **validation croisée à $k$ plis** (par exemple 10-CV) : on obtient $k$ taux d'erreur $\hat\epsilon_1,\dots,\hat\epsilon_k$, un par pli. On veut savoir si l'erreur moyenne observée est compatible avec une valeur de référence $\epsilon_0$ (par exemple « 20 % »).

On calcule la **moyenne** et la **variance** empirique :

$$\mu=\frac1k\sum_{i=1}^k\hat\epsilon_i\qquad\qquad \sigma^2=\frac1{k-1}\sum_{i=1}^k(\hat\epsilon_i-\mu)^2$$

(le $k-1$ au lieu de $k$ corrige le biais de l'estimation de la variance à partir de l'échantillon.)

### 4.2 Pourquoi une loi de Student et pas une loi normale ?

Si on connaissait le **vrai** écart-type, la loi normale suffirait. Mais on ne le connaît pas : on l'**estime** à partir de seulement $k$ valeurs (diapo 9). Cette estimation ajoute de **l'incertitude**. La distribution qui en résulte est un peu **plus « à queues lourdes »** que la normale : c'est la **loi de Student** (loi $t$).

> 💡 **Intuition.** Avec peu de plis, ton estimation de la dispersion est elle-même bancale. La loi de Student est la loi normale « avec une marge de prudence » : plus $k$ est petit, plus elle est prudente (queues lourdes) ; quand $k$ grandit, elle tend vers la normale.

### 4.3 La statistique de test

En considérant les $k$ taux d'erreur comme des tirages **i.i.d.** (indépendants et identiquement distribués) de l'erreur de généralisation, la variable

$$\boxed{\tau_t=\frac{\sqrt k\,(\mu-\epsilon_0)}{\sigma}}$$

suit une **loi $t$ à $k-1$ degrés de liberté**.

**Lecture de la formule.** Le numérateur $(\mu-\epsilon_0)$ est l'écart entre ce qu'on observe et ce qu'on suppose. On le divise par $\sigma/\sqrt k$, qui est l'**erreur standard** de la moyenne (incertitude de $\mu$). Donc $\tau_t$ répond à : *« mon écart vaut combien de fois l'incertitude de ma moyenne ? »* Plus il est grand, plus l'écart est suspect.

### 4.4 La règle de décision (diapo 11)

On fixe $\alpha$ (par exemple 0,05) et on utilise un test **bilatéral** (*two-sided*) : on se méfie d'un écart dans les deux sens. On lit dans une table la valeur critique $t_{\alpha/2}$ (à $k-1$ degrés de liberté).

- Si $|\tau_t| > t_{\alpha/2}$ → l'écart est trop grand pour être dû au hasard → on **rejette** $H_0$ (« $\mu$ est significativement différent de $\epsilon_0$ »).
- Si $\tau_t\in[-t_{\alpha/2},\,t_{\alpha/2}]$ → on **ne peut pas rejeter** $H_0$.

Valeurs usuelles du t-test à deux côtés (diapo 11) :

| $\alpha$ | $k=2$ | $k=5$ | $k=10$ | $k=20$ | $k=30$ |
|---|---|---|---|---|---|
| 0,05 | 12,706 | 2,776 | 2,262 | 2,093 | 2,045 |
| 0,10 | 6,314 | 2,132 | 1,833 | 1,729 | 1,699 |

Observe que la valeur critique **diminue** quand $k$ augmente (plus de données → plus de confiance → moins besoin de marge) et **augmente** quand $\alpha$ diminue (on veut être plus sûr → exigence plus forte). Elle tend vers 1,96 pour $\alpha=0{,}05$ quand $k\to\infty$ (la normale).

> ⚠️ **Point d'attention.** Sur la diapo 11, la phrase « Si $\tau_t\in[\ldots]$ alors l'hypothèse … peut être rejetée » est écrite à l'envers par rapport à la diapo 15 : *dans* l'intervalle, on **ne peut pas** rejeter ; c'est *en dehors* qu'on rejette. L'intervalle est aussi mal écrit $[t_{\alpha/2},t_{\alpha/2}]$ : c'est $[-t_{\alpha/2},t_{\alpha/2}]$. Je donne ci-dessus la version cohérente.

---

## 5. Comparer deux algorithmes sur le même jeu de données : t-test apparié

*(diapos 13 à 17)*

### 5.1 Le principe (diapos 13-14)

On a deux algorithmes A et B, évalués **sur les mêmes $k$ plis** : erreurs $\epsilon^A_1,\dots,\epsilon^A_k$ et $\epsilon^B_1,\dots,\epsilon^B_k$.

**L'idée-clé : travailler sur les différences par paire.**

$$\Delta_i=\epsilon_i^A-\epsilon_i^B$$

Si A et B ont la même performance, la **moyenne** des $\Delta_i$ devrait être **proche de 0**. On réalise donc un t-test sur $\Delta_1,\dots,\Delta_k$, sous $H_0$ : « A et B ont la même performance » (c'est le cas particulier du §4 avec $\epsilon_0=0$).

> 💡 **Pourquoi « apparié » (*paired*) ?** Certains plis sont naturellement faciles, d'autres difficiles, pour *tous* les algorithmes. En comparant A et B **sur le même pli**, on annule cette difficulté commune : seul reste l'effet propre aux algorithmes. C'est comme comparer deux régimes en mesurant le poids de **chaque personne avant/après**, plutôt que de comparer deux groupes de personnes différentes. Cela rend le test beaucoup plus sensible.

### 5.2 La règle de décision (diapo 15)

On calcule $\mu$ et $\sigma^2$ des $\Delta_i$, puis :

$$\tau_t=\left|\frac{\sqrt k\,\mu}{\sigma}\right|\;<\;t_{\alpha/2,\,k-1}\quad\Longrightarrow\quad H_0\text{ ne peut pas être rejetée}$$

et à l'inverse, si $\tau_t\ge t_{\alpha/2,k-1}$, on conclut à une **différence significative**.

### 5.3 Exemple complet : Naïve Bayes vs arbre de décision vs plus proche voisin (diapos 16-17)

Précisions mesurées en 10-CV (source : P. Flach, 2012) :

| | Naïve Bayes (NB) | Arbre de décision (DT) | Plus proche voisin (NN) |
|---|---|---|---|
| **Moyenne** | 0,6937 | 0,7902 | 0,7606 |
| **Écart-type** | 0,0448 | 0,1014 | 0,1248 |

On calcule, pli par pli, les différences de précision, puis leur moyenne, leur écart-type et la p-value (t de Student à $k-1=9$ degrés de liberté) :

| Comparaison | Moyenne des différences | Écart-type | $\tau_t=\sqrt{10}\,\mu/\sigma$ | **p-value** | Verdict à $\alpha=0{,}05$ |
|---|---|---|---|---|---|
| NB − DT | −0,0965 | 0,1246 | −2,45 | **0,0369** | ✅ **Significatif** |
| NB − NN | −0,0669 | 0,1473 | −1,44 | 0,1848 | ❌ Non significatif |
| DT − NN | +0,0295 | 0,1278 | +0,73 | 0,4833 | ❌ Non significatif |

*(Les $\tau_t$ sont recalculés ici ; ils reproduisent bien les p-values de la diapo. La valeur critique est $t_{0{,}025;9}=2{,}262$ : seul $|{-2{,}45}|$ la dépasse.)*

**Lecture.** NB est en moyenne 9,65 points en dessous de DT, et c'est suffisamment systématique d'un pli à l'autre pour ne pas être du hasard. En revanche, NB vs NN : l'écart moyen (6,7 points) est noyé dans la variabilité (écart-type 14,7 points) → pas assez de preuves.

> 💡 **Morale.** Ce n'est pas la taille de l'écart de moyennes qui compte, mais son **rapport à la variabilité**. Un écart de 3 points avec une variabilité de 12 points ne prouve rien.

---

## 6. Le problème de l'indépendance et la validation croisée 5×2

*(diapo 15, remarque en bas)*

La diapo 15 signale une faille : **« les $\Delta_i$ ne sont pas i.i.d. »**, d'où la suggestion de la **validation croisée 5×2** de Tom Dietterich (1998).

### 6.1 Pourquoi les $\Delta_i$ ne sont pas indépendants ? (➕ Complément)

En 10-CV, chaque ensemble d'entraînement contient 90 % des données : **deux plis d'entraînement partagent environ 80 % de leurs exemples**. Les modèles appris sont donc très corrélés, et les erreurs mesurées aussi. Or le t-test suppose l'indépendance. En la violant, on **sous-estime la variance**, ce qui rend le test **trop optimiste** : il déclare trop souvent des différences « significatives » qui n'existent pas (trop de faux positifs).

### 6.2 L'idée du 5×2 cv (➕ Complément)

On répète **5 fois** une validation croisée à **2 plis** (on coupe les données en deux moitiés au hasard, on entraîne sur l'une et on teste sur l'autre, puis l'inverse). Les deux ensembles d'entraînement d'une répétition sont **disjoints**, ce qui limite la dépendance. On obtient 10 différences de performance et une statistique dont la loi est approchée par une **loi $t$ à 5 degrés de liberté**.

Le cours cite cette méthode (plan : « 5×2 cv tests couplés ») mais ne détaille pas la formule ; l'essentiel à retenir est **le problème** (dépendance entre plis) et **la parade** (plis d'entraînement disjoints).

---

## 7. Le test de McNemar

*(diapo 18)*

### 7.1 L'idée : regarder les exemples, pas les taux

Au lieu de comparer des **taux d'erreur** moyennés par pli, on regarde **exemple par exemple** si A et B ont raison ou tort. On construit un **tableau de contingence 2×2** :

| | **A correct** | **A incorrect** |
|---|---|---|
| **B correct** | $e_{00}$ | $e_{01}$ |
| **B incorrect** | $e_{10}$ | $e_{11}$ |

- $e_{00}$ : les deux ont raison. $e_{11}$ : les deux ont tort. Ces cas **ne départagent personne**.
- $e_{01}$ : B a raison et A a tort. $e_{10}$ : A a raison et B a tort. Ce sont les **cas discordants** : **eux seuls** contiennent l'information sur qui est meilleur.

> 💡 **Analogie.** Deux correcteurs notent les mêmes copies. Là où ils sont d'accord, on n'apprend rien sur lequel est le plus sévère. C'est dans les **désaccords** qu'on voit une asymétrie. Si A et B sont équivalents, quand ils divergent, chacun doit « gagner » à peu près autant de fois : **$e_{01}\approx e_{10}$**.

### 7.2 La statistique

$$\tau_{\chi^2}=\frac{\big(|e_{01}-e_{10}|-1\big)^2}{e_{01}+e_{10}}$$

Elle suit approximativement une **loi du $\chi^2$** (à 1 degré de liberté). Le « −1 » est une **correction de continuité** (on approxime une loi discrète par une loi continue).

**Règle (diapo 18)** : $H_0$ (même performance) **ne peut pas être rejetée** au niveau $\alpha$ si $\tau_{\chi^2}<\chi^2_\alpha$. Pour $\alpha=0{,}05$ et 1 degré de liberté, le seuil est **3,84**.

### 7.3 ➕ Exemple chiffré (inventé pour t'entraîner)

Sur 200 exemples de test : $e_{01}=25$ (B a raison, pas A) et $e_{10}=10$ (A a raison, pas B).

$$\tau_{\chi^2}=\frac{(|25-10|-1)^2}{25+10}=\frac{14^2}{35}=\frac{196}{35}=5{,}6$$

$5{,}6>3{,}84$ → on **rejette** $H_0$ : B est significativement meilleur que A ($p\approx0{,}018$). Notons que si l'on avait eu $e_{01}=18$ et $e_{10}=12$, on aurait $\tau=\frac{(6-1)^2}{30}\approx0{,}83<3{,}84$ : même impression « B gagne plus souvent », mais pas assez pour conclure.

### 7.4 ➕ Quand préférer McNemar ?

McNemar n'utilise qu'**un seul découpage entraînement/test** : un seul entraînement par algorithme. C'est utile quand **entraîner coûte cher**. En contrepartie, il ne capture pas la variabilité due au choix de l'ensemble d'entraînement.

---

## 8. Plusieurs jeux de données : le test de Wilcoxon

*(diapos 19 à 22)*

### 8.1 Pourquoi changer de test ? La remarque fondamentale (diapo 20)

Jusqu'ici, tous les tests comparaient A et B sur **un seul jeu de données** (avec ses plis). Maintenant on compare A et B sur **plusieurs jeux de données différents** (images, textes, données médicales…). Deux problèmes apparaissent :

1. **La performance dépend de l'adéquation algorithme/jeu de données** : un algorithme excellent sur l'un peut être médiocre sur l'autre (c'est le principe du « no free lunch »).
2. **On ne peut plus supposer que les taux d'erreur suivent une loi normale** : les jeux de données ne sont pas des tirages d'une même population, et les précisions sont sur des échelles très différentes (un jeu « facile » à 99 %, un autre « difficile » à 60 %).

> **Conséquence : il faut recourir à des tests non paramétriques.**

> 💡 **Qu'est-ce qu'un test non paramétrique ?** Un test qui **ne suppose pas de forme de loi** (pas de normalité). Il s'appuie souvent sur les **rangs** (l'ordre) plutôt que sur les valeurs brutes. Avantage : robuste. Prix : un peu moins puissant quand les hypothèses paramétriques seraient vraies.

### 8.2 Le test de rang signé de Wilcoxon (diapo 21)

**Idée.** Pour chaque jeu de données, on calcule la différence $d_i$ de performance moyenne entre les deux algorithmes. Plutôt que d'utiliser les $d_i$ eux-mêmes, on utilise :
- leur **signe** (qui gagne ?),
- le **rang** de leur valeur absolue (de combien ? classé du plus petit au plus grand).

**Procédure en 5 étapes :**

1. Calculer $d_i=\text{Perf}(h_2)-\text{Perf}(h_1)$ pour chaque jeu $i$.
2. **Écarter** les jeux où $d_i=0$ (aucune information).
3. **Classer** les $|d_i|$ du plus petit (rang 1) au plus grand ; en cas d'égalité, prendre la moyenne des rangs.
4. Additionner séparément les rangs des différences positives et ceux des négatives :
$$W_{s1}=\sum_i I(d_i>0)\,\text{rang}(d_i)\qquad W_{s2}=\sum_i I(d_i<0)\,\text{rang}(d_i)$$
5. Prendre le plus petit des deux : $\;T_{wilcox}=\min(W_{s1},W_{s2})$.

**Raisonnement.** Si A et B sont équivalents, les « victoires » de chacun sont réparties au hasard, donc les deux sommes de rangs sont **proches** et $T$ est **grand** (proche de la moitié du total). Si un algorithme gagne presque toujours et largement, l'autre somme est **minuscule** donc $T$ est **petit**. **Petit $T$ = différence significative.**

> 💡 **Pourquoi mieux que simplement compter les victoires ?** Compter les victoires ignore l'ampleur. Les rangs tiennent compte de l'**importance relative** des écarts (gagner de 30 points pèse plus que gagner de 0,1 point), sans être sensibles aux valeurs aberrantes ni aux échelles différentes.

### 8.3 La loi de $T$

- Pour **$n<25$** jeux de données : on lit la valeur critique dans une **table**.
- Pour **$n\ge25$** : $T$ est approximativement normale, d'où une statistique $z$ :

$$z_{wilcox}=\frac{T_{wilcox}-\mu_T}{\sigma_T},\qquad \mu_T=\frac{n(n+1)}4,\qquad \sigma_T=\sqrt{\frac{n(n+1)(2n+1)}{24}}$$

($\mu_T$ est la moyenne de $T$ sous $H_0$ : la moitié de la somme totale des rangs $n(n+1)/2$.) Exemple avec $n=30$ : $\mu_T=232{,}5$ et $\sigma_T\approx48{,}6$.

### 8.4 Application pas à pas : Naïve Bayes vs SVM sur 10 domaines (diapo 22)

| Domaine | NB | SVM | NB − SVM | \|NB − SVM\| | Rang | Rang signé |
|---|---|---|---|---|---|---|
| 1 | 0,9643 | 0,9944 | −0,0301 | 0,0301 | 3 | −3 |
| 2 | 0,7342 | 0,8134 | −0,0792 | 0,0792 | 6 | −6 |
| 3 | 0,7230 | 0,9151 | −0,1921 | 0,1921 | 8 | −8 |
| 4 | 0,7170 | 0,6616 | +0,0554 | 0,0554 | 5 | +5 |
| 5 | 0,7167 | 0,7167 | 0 | 0 | *retiré* | *retiré* |
| 6 | 0,7436 | 0,7708 | −0,0272 | 0,0272 | 2 | −2 |
| 7 | 0,7063 | 0,6221 | +0,0842 | 0,0842 | 7 | +7 |
| 8 | 0,8321 | 0,8063 | +0,0258 | 0,0258 | 1 | +1 |
| 9 | 0,9822 | 0,9358 | +0,0464 | 0,0464 | 4 | +4 |
| 10 | 0,6962 | 0,9990 | −0,3028 | 0,3028 | 9 | −9 |

- Somme des rangs des différences négatives (SVM meilleur) : $3+6+8+2+9=\mathbf{28}$.
- Somme des rangs des différences positives (NB meilleur) : $5+7+1+4=\mathbf{17}$.
- $T_{wilcox}=\min(28,17)=\mathbf{17}$.

La table au seuil $p=0{,}05$ exige $T<8$ (unilatéral) ou $T<5$ (bilatéral). **$17$ est bien au-dessus** → on **ne peut pas rejeter** l'hypothèse nulle d'égalité.

**Lecture.** Le SVM gagne 5 fois sur 9 et NB 4 fois sur 9 : pas de supériorité nette. Certes SVM gagne parfois très largement (domaines 3 et 10), mais NB gagne aussi, et le bilan global reste équilibré.

> ⚠️ **Point d'attention.** La diapo parle de « $n-1=9$ degrés de liberté » ; il s'agit plutôt de **$n=9$ jeux utilisés** (10 moins le jeu à différence nulle) qu'on lit dans la table de Wilcoxon. Par convention, on rejette si $T\le$ valeur critique (la diapo écrit « < »). Cela ne change pas la conclusion ici (17 est loin de 5 ou 8).

---

## 9. Plusieurs algorithmes sur plusieurs jeux : Friedman puis Nemenyi

*(diapos 23 à 35)*

### 9.1 Le problème des comparaisons multiples (diapos 24 et 30)

Avec $k$ algorithmes, la tentation est de faire **tous les tests deux à deux** (Wilcoxon sur chaque paire). Mauvaise idée :

> **Plus on fait de tests, plus la probabilité qu'au moins un soit un faux positif augmente.**

Chaque test à $\alpha=0{,}05$ a 5 % de chances de crier « différence ! » à tort. Avec 3 algorithmes il y a 3 paires ; avec 5, il y en a 10.

> ➕ **Calcul.** Pour 10 tests indépendants à $\alpha=0{,}05$ : $P(\text{au moins un faux positif})=1-0{,}95^{10}\approx\mathbf{40\ \%}$. Le niveau de confiance **s'effondre** à chaque comparaison supplémentaire.

La diapo 30 le dit ainsi : *« un algorithme peut sembler meilleur par le hasard des tests »*, avec un renvoi au lien entre **risque empirique et risque réel** vu dans le cours sur les SVM : sélectionner le meilleur parmi beaucoup de candidats sur la base d'une mesure bruitée, c'est faire du surapprentissage sur le test.

**La démarche correcte en deux temps :**

1. **Un test global (omnibus)** : « *y a-t-il au moins une différence quelque part ?* » → **Friedman**.
2. **Si oui, un test post hoc** pour localiser *quelles paires* diffèrent → **Nemenyi**.

### 9.2 Le test de Friedman : le principe (diapos 25-26)

Test **non paramétrique** (aucune loi supposée), basé sur les **rangs**.

1. Chaque algorithme est testé sur chaque jeu de données (par CV ou jeu de test à part).
2. **Pour chaque jeu de données**, on **classe** les algorithmes du meilleur (rang 1) au moins bon (rang $k$) ; en cas d'égalité, on prend la moyenne (ex. 2,5 pour deux ex æquo aux places 2 et 3).
3. On calcule le **rang moyen** $r_i$ de chaque algorithme sur les $N$ jeux.

**Raisonnement.** Si tous les algorithmes sont équivalents, aucun n'est systématiquement bien classé : chacun obtient un rang moyen proche de la valeur « neutre » $\frac{k+1}2$. Avec $k=3$, c'est 2. Sous $H_0$ le rang moyen $r_i$ a pour moyenne et variance :

$$E[r_i]=\frac{k+1}2\qquad\qquad \text{Var}(r_i)=\frac{k^2-1}{12\,N}$$

Si un algorithme a un rang moyen **très éloigné** de $\frac{k+1}2$, c'est suspect.

**Exemple de la diapo 26** (3 algorithmes, 4 jeux de données) :

| Jeu | A | B | C |
|---|---|---|---|
| $D_1$ | 1 | 2 | 3 |
| $D_2$ | 1 | 2,5 | 2,5 |
| $D_3$ | 1 | 2 | 3 |
| $D_4$ | 1 | 2 | 3 |
| **Rang moyen** | **1** | **2,125** | **2,875** |

A est toujours premier. $D_2$ illustre une égalité (B et C ex æquo → 2,5 chacun).

### 9.3 La statistique de Friedman (diapo 27)

$$\tau_{\chi^2}=\frac{k-1}{k}\cdot\frac{12N}{k^2-1}\sum_{i=1}^k\Big(r_i-\frac{k+1}2\Big)^2=\frac{12N}{k(k+1)}\Big(\sum_{i=1}^k r_i^2-\frac{k(k+1)^2}4\Big)$$

**Lecture.** C'est essentiellement la **somme des carrés des écarts** des rangs moyens à la valeur neutre, normalisée. Si les rangs moyens s'éloignent de $\frac{k+1}2$, la somme grossit. Elle suit une **loi du $\chi^2$ à $k-1$ degrés de liberté** quand $k$ et $N$ sont grands.

**Version améliorée (Iman-Davenport)**, valable aussi pour de **petites valeurs de $k$ et $N$** :

$$\tau_F=\frac{(N-1)\,\tau_{\chi^2}}{N(k-1)-\tau_{\chi^2}}$$

On compare $\tau_F$ à une **valeur critique** des tables (diapos 28-29, pour $\alpha=0{,}05$ et $\alpha=0{,}1$).

> ⚠️ **Règle de décision à ne pas inverser (diapo 27).** Si $\tau_F$ est **plus petit** que la valeur critique → **pas de différence significative**. Si $\tau_F$ est **plus grand** → on rejette $H_0$ (au moins deux algorithmes diffèrent).

### 9.4 ➕ Calcul complet sur l'exemple de la diapo 26

$k=3$, $N=4$ ; rangs moyens $(1;\,2{,}125;\,2{,}875)$.

1. $\sum r_i^2=1+4{,}516+8{,}266=13{,}781$ ; $\;\frac{k(k+1)^2}4=\frac{3\times16}4=12$.
2. $\tau_{\chi^2}=\frac{12\times4}{3\times4}(13{,}781-12)=4\times1{,}781=\mathbf{7{,}125}$.
3. $\tau_F=\frac{3\times7{,}125}{4\times2-7{,}125}=\frac{21{,}375}{0{,}875}\approx\mathbf{24{,}4}$.
4. Valeur critique (diapo 28, $\alpha=0{,}05$, $N=4$, $k=3$) : **5,143**.

$24{,}4>5{,}143$ → on **rejette** $H_0$ : il existe des différences significatives entre A, B et C. **Mais on ne sait pas encore lesquelles.** C'est le rôle du post hoc.

### 9.5 Le test post hoc de Nemenyi (diapos 30 à 34)

Friedman dit *« il y a une différence quelque part »*. Nemenyi dit *« entre ces deux-là précisément »*.

**Idée.** Deux algorithmes sont jugés **significativement différents** si leurs rangs moyens diffèrent **d'au moins une certaine quantité**, la **différence critique** (CD, *Critical Difference*) :

$$CD=q_\alpha\sqrt{\frac{k(k+1)}{6N}}$$

où $q_\alpha$ est une valeur tabulée (diapo 34). De façon équivalente, pour deux classifieurs $f_{j_1},f_{j_2}$ on calcule

$$q=\frac{\overline R_{\cdot j_1}-\overline R_{\cdot j_2}}{\sqrt{k(k+1)/(6N)}}$$

et on **rejette** l'égalité si $|q|$ dépasse $q_\alpha$. Le dénominateur est l'**écart-type** de la différence de deux rangs moyens sous $H_0$ : on mesure de combien d'écarts-types les deux rangs moyens sont éloignés.

Valeurs critiques $q_\alpha$ (diapo 34) :

| $\alpha$ | $k=2$ | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|
| 0,05 | 1,960 | 2,344 | 2,569 | 2,728 | 2,850 | 2,949 | 3,031 | 3,102 | 3,164 |
| 0,10 | 1,645 | 2,052 | 2,291 | 2,459 | 2,589 | 2,693 | 2,780 | 2,855 | 2,920 |

*(Pour $k=2$, on retrouve 1,960 : la valeur de la loi normale. C'est cohérent : pour 2 algorithmes il n'y a qu'une comparaison.)*

### 9.6 Exemple propre : 3 algorithmes, 4 jeux (diapo 34)

$k=3$, $N=4$, $\alpha=0{,}05$ :

$$CD=2{,}344\times\sqrt{\frac{3\times4}{6\times4}}=2{,}344\times0{,}707\approx\mathbf{1{,}66}$$

Différences de rangs moyens ($1$ ; $2{,}125$ ; $2{,}875$) :

| Paire | Écart | > 1,66 ? | Verdict |
|---|---|---|---|
| A vs B | 1,125 | non | pas de différence significative |
| B vs C | 0,750 | non | pas de différence significative |
| **A vs C** | **1,875** | **oui** | **A significativement meilleur que C** |

Cela correspond au schéma de la diapo 34 : chaque algorithme est représenté par son rang moyen entouré d'un segment de longueur CD ; A et C sont significativement différents quand leurs segments **ne se chevauchent pas**.

**Pourquoi A–B et B–C ne sont-ils pas significatifs alors que A–C l'est ?** C'est le comportement normal des tests : la « relation de différence significative » **n'est pas transitive**. B est « entre » A et C, trop proche des deux pour être séparé, alors que A et C sont assez éloignés. Avec seulement 4 jeux de données, le test manque de puissance : il faut un écart de rang d'environ 1,66 pour conclure.

### 9.7 Exemple de la diapo 32-33 (10 domaines) et pièges de lecture

Trois classifieurs $f_A,f_B,f_C$ sur 10 domaines. Les **sommes de rangs** sont $15{,}5$ ; $30$ ; $14{,}5$ (donc rangs moyens $1{,}55$ ; $3{,}00$ ; $1{,}45$). $f_B$ est toujours dernier (rang 3) : ses précisions sont autour de 75 % contre environ 86 % pour les deux autres.

La diapo conclut : *« $f_B>f_A$ et $f_B>f_C$, mais pas $f_A>f_C$ »*, c'est-à-dire que **$f_B$ est significativement à l'écart des deux autres**, alors qu'**A et C ne se distinguent pas**.

> ⚠️ **Deux points d'attention.**
> 1. **Sens du « > ».** Ici le rang 1 est le **meilleur**. « $f_B>f_A$ » signifie que $f_B$ a un **rang plus grand**, donc qu'il est **moins bon**. Les données le confirment (B a les plus faibles précisions). Ne lis donc pas « $>$ » comme « meilleur que ».
> 2. **Sommes vs moyennes.** La formule de $q$ est définie avec les **rangs moyens** $\overline R_{\cdot j}$, mais les calculs de la diapo 33 sont faits avec les **sommes** (15,5 ; 30 ; 14,5) : ce qui gonfle les valeurs de $q$ d'un facteur $N=10$ (−32,22 ; 2,22 ; 34,44). Avec les moyennes, on obtient $q_{12}\approx-3{,}24$, $q_{13}\approx0{,}22$, $q_{23}\approx3{,}46$, et $CD=2{,}344\times\sqrt{12/60}\approx1{,}05$ (je n'ai pas pu retrouver le « 2,55 » de la diapo). **La conclusion qualitative est identique** (B se détache nettement ; A et C sont indiscernables), mais si ce point tombe à l'examen, vérifie la convention avec ton enseignant.

### 9.8 Lire un diagramme de différence critique (diapo 35)

Exemple de la diapo (Pölsterl et al., 2017) : comparaison de méthodes de prédiction de survie (modèle de Cox avec ridge, forêt de survie aléatoire, boosting, SVM de survie…).

- Les méthodes sont **triées par rang moyen**, de la gauche (meilleur) vers la droite (moins bon).
- Les **barres épaisses** relient les groupes de méthodes **non significativement différentes** (p-value > 0,05).
- Deux méthodes **non reliées** par une barre sont significativement différentes.
- Le segment « Critical Difference » en haut donne l'échelle : deux méthodes dont les rangs moyens sont distants de plus que CD sont significativement différentes.

---

## 10. Lire les résultats avec prudence

*(diapos 36 à 37)*

Les dernières diapos comparent **11 algorithmes sur ~30 jeux de données** (le contexte : des algorithmes à compromis entre précision et délai de décision, paramétré par $\alpha$ de 0 à 1 ; « High delay cost » à gauche, « Low delay cost » à droite). On y trace le **rang moyen** de chaque algorithme en fonction de $\alpha$. Pour $\alpha=0{,}5$, on voit **4 algorithmes se détacher** (stopping_rule, proba_threshold, economy, calimera).

**Mise en garde centrale (diapo 36) :**

> ⚠️ **La comparaison des rangs ne dit rien sur les performances.** Peut-être qu'il y a très peu de différences !

Pourquoi ? Un rang est un **ordre**, pas une **quantité**. Si deux algorithmes ont 85,01 % et 85,02 % de précision, le second est « rang 1 » et le premier « rang 2 », avec la même régularité si la différence est systématique. Un test de rangs peut alors être « significatif » alors que l'**écart est négligeable en pratique**. On le voit sur la diapo 37 : les valeurs de métrique affichées pour les quatre algorithmes détachés sont très proches (de l'ordre de 0,23-0,24, d'après le schéma).

> 💡 **À retenir.**
> - **Significativité statistique ≠ importance pratique.** Il faut toujours regarder **aussi** l'ampleur de l'écart (la différence de performance), pas seulement le verdict du test.
> - Rapporter à la fois les **rangs** (pour la robustesse de la comparaison) et les **valeurs de performance** (pour la taille de l'effet).
> - Et rappelle-toi la section 2 : un test statistique, aussi rigoureux soit-il, ne valide que ce que la **métrique** et les **jeux de données** mesurent.

---

## 11. Quel test pour quelle situation ?

| Situation | Test | Nature | Idée en une phrase |
|---|---|---|---|
| 1 algorithme, comparer à une valeur $\epsilon_0$ | **t-test** (Student) | Paramétrique | La moyenne des erreurs en CV diffère-t-elle de $\epsilon_0$ ? |
| 2 algorithmes, **1 jeu**, plusieurs plis | **t-test apparié** | Paramétrique | La moyenne des différences par pli est-elle ≠ 0 ? |
| Idem, mais plis dépendants | **5×2 cv** (Dietterich) | Paramétrique | Plis d'entraînement disjoints pour limiter la dépendance |
| 2 algorithmes, **1 jeu**, **1 seul découpage** | **McNemar** | Sur tableau de contingence ($\chi^2$) | Dans les désaccords, un algorithme gagne-t-il plus souvent ? |
| 2 algorithmes, **plusieurs jeux** | **Wilcoxon** (rang signé) | Non paramétrique | Les rangs des différences penchent-ils d'un côté ? |
| $k$ algorithmes, **plusieurs jeux** | **Friedman** | Non paramétrique | Les rangs moyens sont-ils tous proches de $\frac{k+1}2$ ? |
| Après un Friedman significatif | **Nemenyi** (post hoc) | Non paramétrique | Quelles paires ont des rangs moyens éloignés de plus que CD ? |

**Arbre de décision rapide :**

1. Combien de jeux de données ? **Un** → t-test apparié / McNemar / 5×2 cv. **Plusieurs** → tests non paramétriques.
2. Combien d'algorithmes ? **Deux** → Wilcoxon. **Plus de deux** → Friedman, puis Nemenyi si Friedman est significatif.

---

## 12. Fiche récapitulative

| Concept | En une phrase |
|---|---|
| **Hypothèse nulle $H_0$** | « Il n'y a pas de vraie différence » |
| **p-value** | Probabilité, sous $H_0$, d'un résultat au moins aussi extrême que celui observé |
| **Niveau $\alpha$** | Seuil fixé à l'avance (souvent 0,05) ; on rejette $H_0$ si $p<\alpha$ |
| **Écart-type d'une précision** | $\sqrt{\hat a(1-\hat a)/n}$ |
| **Loi de Student** | Loi normale « prudente » quand on estime l'écart-type avec peu de données |
| **t-test** | $\tau_t=\sqrt k(\mu-\epsilon_0)/\sigma$, loi $t$ à $k-1$ degrés de liberté |
| **Test apparié** | Travailler sur les différences par pli pour annuler la difficulté commune |
| **Problème d'indépendance** | Les plis de CV partagent des données d'entraînement → t-test trop optimiste → 5×2 cv |
| **McNemar** | Compare uniquement les désaccords $e_{01}$ et $e_{10}$ ; $\tau=(|e_{01}-e_{10}|-1)^2/(e_{01}+e_{10})$ |
| **Test non paramétrique** | Aucune loi supposée ; s'appuie sur les rangs |
| **Wilcoxon** | $T=\min(W_{s1},W_{s2})$ ; petit $T$ = différence significative |
| **Comparaisons multiples** | Plus on teste, plus les faux positifs s'accumulent |
| **Friedman** | Test global sur les rangs moyens de $k$ algorithmes ($\chi^2$ à $k-1$ ddl) |
| **Nemenyi** | Post hoc : différence critique $CD=q_\alpha\sqrt{k(k+1)/(6N)}$ |
| **Rang moyen ≠ performance** | Un test de rangs ne dit rien sur la taille de l'écart |
| **Benchmark** | Efficace mais rétrécit la vision (« chercher sous le réverbère ») |

---

## 13. Quiz d'auto-évaluation

> **Mode d'emploi.** Réponds d'abord sans regarder, puis clique sur « Voir la réponse ». Compte un point par bonne réponse. Barème indicatif à la fin.

### Partie A : Contexte et bases

**Q1.** Cite les quatre qualités d'un système d'IA évoquées en introduction et explique-les en quelques mots.
<details><summary>Voir la réponse</summary>

**Consistance** (réussir à chaque fois, pas seulement parfois), **robustesse** (fonctionner même en conditions imparfaites), **calibration** (savoir quand on a réussi ou échoué), **sûreté opérationnelle** (éviter les erreurs catastrophiques).
</details>

**Q2.** Que signifie « chercher sous les réverbères » à propos des benchmarks ?
<details><summary>Voir la réponse</summary>

La communauté explore ce qui est **facile à mesurer** avec une évaluation par benchmarks, et non nécessairement ce qui est important. Les benchmarks sont **efficaces** (on juge vite un travail) mais **rétrécissent** la vision collective du domaine.
</details>

**Q3.** Que représente l'hypothèse nulle $H_0$ quand on compare deux algorithmes ?
- A. Que l'algorithme A est meilleur
- B. Qu'il n'existe aucune vraie différence de performance entre eux
- C. Que l'algorithme B est meilleur
- D. Que les données sont normales

<details><summary>Voir la réponse</summary>

**B.** On cherche à savoir si les données permettent de rejeter l'idée « pas de différence ».
</details>

**Q4.** Définis la p-value. Que ne signifie-t-elle pas ?
<details><summary>Voir la réponse</summary>

C'est la **probabilité d'obtenir, sous $H_0$, un résultat au moins aussi extrême que celui observé**. Ce n'est **pas** la probabilité que $H_0$ soit vraie, et $p>\alpha$ ne prouve **pas** que les algorithmes sont équivalents (on ne peut simplement pas rejeter $H_0$).
</details>

**Q5.** Vrai ou faux : si $p=0{,}20>0{,}05$, on peut affirmer que les deux algorithmes ont exactement la même performance.
<details><summary>Voir la réponse</summary>

**Faux.** On **ne peut pas rejeter $H_0$** : les données ne suffisent pas à conclure à une différence, mais cela ne démontre pas l'égalité (peut-être manque-t-on de données ou de puissance).
</details>

**Q6.** Un classifieur obtient une précision de 0,8 sur 100 exemples. Quel est l'écart-type de cette estimation (avec $\hat a=0{,}8$) ?
<details><summary>Voir la réponse</summary>

$\sqrt{0{,}8\times0{,}2/100}=\sqrt{0{,}0016}=\mathbf{0{,}04}$. *(Sous l'hypothèse nulle $a=0{,}5$ de la diapo 8, on utilise $\sqrt{0{,}25/100}=0{,}05$.)*
</details>

**Q7.** Dans l'exemple de la diapo 8 (vraie précision supposée 0,5, estimation 0,80, écart-type 0,05), de combien d'écarts-types l'estimation s'éloigne-t-elle de la valeur supposée ? Que conclut-on pour $\alpha=0{,}05$ ?
<details><summary>Voir la réponse</summary>

$(0{,}80-0{,}50)/0{,}05=\mathbf{6}$ écarts-types. La p-value ($\approx2\times10^{-9}$) est bien inférieure à 0,05 → on **rejette** $H_0$.
</details>

### Partie B : t-test

**Q8.** Pourquoi utilise-t-on la loi de Student plutôt que la loi normale pour le t-test ?
<details><summary>Voir la réponse</summary>

Parce que le **vrai écart-type est inconnu** et doit être **estimé** à partir des $k$ plis, ce qui ajoute de l'incertitude. La loi de Student a des **queues plus lourdes** que la normale pour en tenir compte (et tend vers la normale quand $k$ augmente).
</details>

**Q9.** Écris la statistique du t-test et donne son nombre de degrés de liberté pour une validation croisée à $k$ plis.
<details><summary>Voir la réponse</summary>

$\tau_t=\dfrac{\sqrt k\,(\mu-\epsilon_0)}{\sigma}$, qui suit une loi $t$ à **$k-1$ degrés de liberté**.
</details>

**Q10.** Pourquoi la variance utilise-t-elle $\frac1{k-1}$ et non $\frac1k$ ?
<details><summary>Voir la réponse</summary>

Pour obtenir un estimateur **non biaisé** de la variance : on estime déjà la moyenne $\mu$ à partir des mêmes données, ce qui « consomme » un degré de liberté.
</details>

**Q11.** Pour une validation croisée à 10 plis et $\alpha=0{,}05$ (test bilatéral), la valeur critique est 2,262. Tu obtiens $\tau_t=2{,}0$. Conclusion ?
<details><summary>Voir la réponse</summary>

$|2{,}0|<2{,}262$ → on **ne peut pas rejeter $H_0$** : l'écart n'est pas significatif au niveau 0,05.
</details>

**Q12.** Pourquoi la valeur critique diminue-t-elle quand $k$ augmente (par exemple 12,706 pour $k=2$ contre 2,045 pour $k=30$) ?
<details><summary>Voir la réponse</summary>

Avec plus de plis, l'écart-type est **mieux estimé** : l'incertitude supplémentaire diminue et la loi de Student se rapproche de la normale (queues moins lourdes). Il faut donc moins de « marge de prudence ».
</details>

**Q13.** Qu'est-ce qu'un test « apparié » (*paired*) et quel est son avantage ?
<details><summary>Voir la réponse</summary>

On compare les deux algorithmes **sur les mêmes plis** et on travaille sur les **différences $\Delta_i=\epsilon_i^A-\epsilon_i^B$**. Cela annule la variabilité due à la difficulté propre de chaque pli et rend le test plus sensible.
</details>

**Q14.** Sur la diapo 17, la p-value de NB − DT est 0,0369, celle de NB − NN est 0,1848. Qu'en conclut-on à $\alpha=0{,}05$ ?
<details><summary>Voir la réponse</summary>

NB et DT diffèrent **significativement** ($0{,}0369<0{,}05$). NB et NN ne diffèrent **pas** significativement ($0{,}1848>0{,}05$). Seule la différence NB/arbre de décision est jugée significative (DT − NN aussi non significative, $p=0{,}4833$).
</details>

**Q15.** Pourquoi dit-on que le t-test apparié en validation croisée n'est pas parfaitement rigoureux ? Quelle solution a été proposée ?
<details><summary>Voir la réponse</summary>

Les $\Delta_i$ ne sont **pas i.i.d.** : les ensembles d'entraînement des différents plis se **recouvrent fortement**, donc les résultats sont corrélés et la variance est sous-estimée (trop de faux positifs). Solution de Dietterich (1998) : la **validation croisée 5×2**.
</details>

### Partie C : McNemar

**Q16.** Dans le test de McNemar, quels sont les effectifs qui comptent vraiment et pourquoi ?
<details><summary>Voir la réponse</summary>

Les **cas discordants** $e_{01}$ et $e_{10}$ (un algorithme a raison, l'autre tort). Quand les deux ont raison ou les deux tort, on n'apprend rien sur leur différence. Si A et B sont équivalents, on attend $e_{01}\approx e_{10}$.
</details>

**Q17.** Calcule la statistique de McNemar pour $e_{01}=30$ et $e_{10}=10$. Est-ce significatif à $\alpha=0{,}05$ (seuil $\chi^2=3{,}84$) ?
<details><summary>Voir la réponse</summary>

$\tau=\dfrac{(|30-10|-1)^2}{30+10}=\dfrac{19^2}{40}=\dfrac{361}{40}\approx9{,}03>3{,}84$ → **significatif** : on rejette $H_0$.
</details>

**Q18.** Dans quelle situation pratique McNemar est-il particulièrement adapté ?
<details><summary>Voir la réponse</summary>

Quand on ne peut faire **qu'un seul entraînement** par algorithme (par exemple entraînement très coûteux) : le test se base sur **un seul découpage** entraînement/test.
</details>

### Partie D : Wilcoxon

**Q19.** Pourquoi ne peut-on pas utiliser un t-test classique pour comparer des algorithmes sur plusieurs jeux de données ?
<details><summary>Voir la réponse</summary>

Parce que la performance dépend de l'**adéquation algorithme/jeu de données**, et qu'on ne peut plus supposer que les taux d'erreur sont **normalement distribués** (jeux différents, échelles différentes). Il faut un test **non paramétrique**.
</details>

**Q20.** Qu'est-ce qu'un test non paramétrique ? Donne ses avantages.
<details><summary>Voir la réponse</summary>

Un test qui **ne suppose aucune forme de loi** (pas de normalité), souvent basé sur les **rangs**. Il est **robuste** aux valeurs aberrantes et aux échelles différentes, au prix d'une puissance parfois un peu moindre.
</details>

**Q21.** Décris les étapes du test de rang signé de Wilcoxon.
<details><summary>Voir la réponse</summary>

1. Calculer les différences $d_i$ par jeu de données ; 2. écarter celles égales à 0 ; 3. classer les $|d_i|$ (rang 1 = plus petit) ; 4. sommer les rangs des $d_i>0$ ($W_{s1}$) et des $d_i<0$ ($W_{s2}$) ; 5. $T=\min(W_{s1},W_{s2})$. Un **petit $T$** indique une différence significative.
</details>

**Q22.** Soient les différences $d=(+0{,}05;\,-0{,}02;\,+0{,}10;\,0;\,-0{,}01;\,+0{,}03)$. Calcule $W_{s1}$, $W_{s2}$ et $T$.
<details><summary>Voir la réponse</summary>

On retire le 0. Valeurs absolues : 0,05 ; 0,02 ; 0,10 ; 0,01 ; 0,03 → rangs : **4, 2, 5, 1, 3**.
Positifs : $+0{,}05$ (rang 4), $+0{,}10$ (rang 5), $+0{,}03$ (rang 3) → $W_{s1}=12$.
Négatifs : $-0{,}02$ (rang 2), $-0{,}01$ (rang 1) → $W_{s2}=3$.
$T=\min(12,3)=\mathbf{3}$.
</details>

**Q23.** Dans l'exemple NB vs SVM (diapo 22), on obtient $T=17$ avec 9 jeux utilisés et un seuil critique de 5 (bilatéral). Quelle est la conclusion ?
<details><summary>Voir la réponse</summary>

$17$ est largement supérieur à 5 → on **ne peut pas rejeter** l'hypothèse nulle : pas de différence significative entre NB et SVM à $p=0{,}05$.
</details>

**Q24.** Pour quelle taille $n$ peut-on approximer la loi de $T_{wilcox}$ par une loi normale ? Donne $\mu_T$.
<details><summary>Voir la réponse</summary>

Pour $n\ge25$. $\mu_T=\dfrac{n(n+1)}4$, $\sigma_T=\sqrt{\dfrac{n(n+1)(2n+1)}{24}}$. Sinon on utilise une table.
</details>

**Q25.** Pourquoi le test de Wilcoxon est-il préférable au simple décompte des victoires ?
<details><summary>Voir la réponse</summary>

Il tient compte de l'**ampleur relative** des écarts via les rangs (une grande victoire pèse plus qu'une victoire minime), sans être sensible aux valeurs aberrantes ni aux différences d'échelle entre jeux.
</details>

### Partie E : Friedman et Nemenyi

**Q26.** Pourquoi ne pas simplement faire tous les tests de Wilcoxon deux à deux entre $k$ algorithmes ?
<details><summary>Voir la réponse</summary>

Parce que plus on multiplie les tests, plus le risque d'**au moins un faux positif** augmente (par exemple $\approx40\%$ pour 10 tests à $\alpha=0{,}05$). Le niveau de confiance **s'effondre** : c'est le problème des **comparaisons multiples**.
</details>

**Q27.** Comment procède-t-on correctement pour comparer $k$ algorithmes sur plusieurs jeux de données ?
<details><summary>Voir la réponse</summary>

En deux temps : (1) un test **global** (**Friedman**) pour savoir s'il existe au moins une différence ; (2) si oui, un test **post hoc** (**Nemenyi**) pour identifier les paires différentes.
</details>

**Q28.** Quelle est la valeur attendue du rang moyen d'un algorithme sous $H_0$ pour $k=5$ algorithmes ?
<details><summary>Voir la réponse</summary>

$\dfrac{k+1}2=\dfrac{6}{2}=\mathbf{3}$.
</details>

**Q29.** Trois algorithmes sont comparés sur trois jeux ; les rangs (1 = meilleur) sont : $D_1:(1,2,3)$, $D_2:(1,3,2)$, $D_3:(2,1,3)$. Calcule les rangs moyens.
<details><summary>Voir la réponse</summary>

Algorithme A : $(1+1+2)/3=\mathbf{1{,}33}$. Algorithme B : $(2+3+1)/3=\mathbf{2{,}00}$. Algorithme C : $(3+2+3)/3=\mathbf{2{,}67}$.
</details>

**Q30.** Dans l'exemple de la diapo 26 (rangs moyens 1 ; 2,125 ; 2,875, $N=4$, $k=3$), on trouve $\tau_F\approx24{,}4$ et la valeur critique est 5,143. Que décide-t-on ? Que reste-t-il à faire ?
<details><summary>Voir la réponse</summary>

$24{,}4>5{,}143$ → on **rejette $H_0$** : au moins deux algorithmes diffèrent. Mais Friedman ne dit pas lesquels : il faut un **test post hoc** (Nemenyi).
</details>

**Q31.** Calcule la différence critique CD de Nemenyi pour $k=3$ algorithmes, $N=4$ jeux et $\alpha=0{,}05$ ($q_\alpha=2{,}344$).
<details><summary>Voir la réponse</summary>

$CD=2{,}344\times\sqrt{\dfrac{3\times4}{6\times4}}=2{,}344\times\sqrt{0{,}5}\approx2{,}344\times0{,}707=\mathbf{1{,}66}$.
</details>

**Q32.** Avec ce CD de 1,66 et les rangs moyens 1 ; 2,125 ; 2,875, quelles paires sont significativement différentes ?
<details><summary>Voir la réponse</summary>

Seule **A–C** : $2{,}875-1=1{,}875>1{,}66$. A–B ($1{,}125$) et B–C ($0{,}75$) restent sous le seuil. **A est significativement meilleur que C.**
</details>

**Q33.** Comment lit-on un diagramme de différence critique (type Nemenyi) ?
<details><summary>Voir la réponse</summary>

Les méthodes sont triées par **rang moyen** (meilleure à gauche). Les méthodes **reliées par une barre épaisse** ne sont **pas significativement différentes**. Deux méthodes non reliées le sont.
</details>

**Q34.** Dans les tests de rangs, le rang 1 correspond au meilleur algorithme. Que signifie alors « $f_B>f_A$ » à la diapo 33 pour les rangs ?
<details><summary>Voir la réponse</summary>

Que **$f_B$ a un rang plus élevé, donc est moins bon** que $f_A$. Dans l'exemple, $f_B$ est effectivement bien moins précis (≈ 75 % contre ≈ 86 %). Il faut lire le « $>$ » comme « rang plus grand », pas « meilleur ».
</details>

### Partie F : Synthèse et esprit critique

**Q35.** Un test de Friedman/Nemenyi montre que l'algorithme X est significativement mieux classé que Y sur 30 jeux de données. Peut-on en déduire que X est « bien meilleur » que Y ?
<details><summary>Voir la réponse</summary>

**Non.** Les rangs indiquent un **ordre**, pas la taille de l'écart de performance. X peut battre Y de façon très régulière **mais d'un cheveu** (par exemple 0,01 point). La significativité statistique n'implique pas l'importance pratique : il faut aussi regarder les **valeurs de performance**.
</details>

**Q36.** Pour chaque situation, quel test choisis-tu ?
- (a) 2 algorithmes, un seul jeu de données, 10-CV
- (b) 2 algorithmes, un seul entraînement très coûteux
- (c) 2 algorithmes, 20 jeux de données
- (d) 6 algorithmes, 25 jeux de données

<details><summary>Voir la réponse</summary>

(a) **t-test apparié** (ou 5×2 cv pour limiter le problème de dépendance) ; (b) **McNemar** ; (c) **Wilcoxon** (rang signé) ; (d) **Friedman**, puis **Nemenyi** si Friedman est significatif.
</details>

**Q37 (synthèse).** Explique en quelques phrases, sans formule, pourquoi comparer deux algorithmes en regardant simplement quelle précision moyenne est la plus haute est insuffisant, et comment les tests statistiques répondent à ce problème.
<details><summary>Voir la réponse</summary>

Une précision est une **estimation bruitée** : elle dépend du découpage des données et de l'échantillon. Une petite différence peut venir du hasard. Les tests statistiques supposent **qu'il n'y a pas de différence** ($H_0$), caractérisent ce que le hasard seul produirait, puis ne concluent à une vraie différence que si l'écart observé est **très improbable** sous $H_0$ (p-value inférieure à $\alpha$). Le choix du test dépend du contexte : un ou plusieurs jeux, deux ou plusieurs algorithmes, hypothèses de normalité, indépendance… et avec beaucoup de comparaisons il faut un test global puis un post hoc pour ne pas accumuler de faux positifs. Enfin, il faut toujours compléter le verdict par l'**ampleur** de la différence et rester conscient des limites des benchmarks.
</details>

---

### Barème indicatif (37 questions)

| Score | Niveau |
|---|---|
| 0 – 15 | À retravailler : relis les sections 3, 4 et 5 en priorité |
| 16 – 26 | Bases acquises : approfondis Wilcoxon (8) et Friedman/Nemenyi (9) |
| 27 – 33 | Bonne maîtrise |
| 34 – 37 | Excellent ! Tu peux expliquer ces tests à quelqu'un d'autre |

> 💡 **Fil rouge de révision.** *Une différence observée n'est pas une différence prouvée → on teste $H_0$ → un jeu de données : t-test apparié / McNemar → plusieurs jeux : tests de rangs → $k$ algorithmes : Friedman puis Nemenyi → toujours regarder aussi l'ampleur de l'écart.*
