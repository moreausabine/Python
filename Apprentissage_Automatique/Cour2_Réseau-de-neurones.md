# <p style="text-align: center;">Le réseau de neurone</p>

Cours d'Antoine Cornuéjols (AgroParisTech – INRAe MIA Paris-Saclay, EKINOCS research group)


## Plan du cours

1. Introduction et réseaux de neurones à une couche
2. Les perceptrons multi-couches
3. L'apprentissage dans les réseaux de neurones
4. Une introduction aux réseaux de neurones profonds
5. Autres architectures


---

## Introduction et réseaux de neurones à une couche

### Contexte rapide

**Ancrage biologique**

Pourquoi s'inspirer du cerveau ? (slide 6) Le cerveau est un modèle séduisant car il est capable d'apprentissage, robuste et tolérant aux fautes, capable de s'accommoder d'informations incomplètes/incertaines/imprécises, et massivement parallèle.

Quelques ordres de grandeur donnés en cours : environ 10¹¹ neurones dans le cerveau humain (contre 950×10³ chez l'abeille !), environ 10³ à 10⁴ connexions par neurone. Chaque neurone biologique fonctionne avec un potentiel d'action, une période réfractaire, des neuro-transmetteurs, et transmet des signaux soit excitateurs, soit inhibiteurs.

Attraits des réseaux de neurones artificiels : calculs parallélisables, robustes/tolérants aux fautes (car distribués sur plein de petites unités), algorithmes simples, d'emploi très général.
Défauts : opacité des raisonnements et de l'hypothèse produite (on retrouve ici le problème d'explicabilité déjà vu au cours 1 avec le connexionnisme).

---

### Le réseau à une couche

**Le perceptron :**
Un perceptron, c'est **un seul neurone**.

Il apprend à discriminer entre 2 classes (+ et -) en plaçant un **hyperplan** dans l'espace des entrées.

> ### *💡 Explication : qu'est-ce qu'un hyperplan ?*
>
> Un hyperplan, c'est simplement la généralisation d'une droite (en 2D) ou d'un plan (en 3D) à un espace de dimension quelconque d. Dans un espace à d dimensions, un hyperplan a une dimension d-1 — c'est l'objet qui **coupe l'espace en deux régions**.
>
> Concrètement, dans le plan (2 attributs x₁, x₂), un hyperplan est juste une droite. Son équation s'écrit :
> *1 + 2x₁ + 3x₂ = 0 (exemple donné en cours), ce qu'on peut réécrire x₂ = -⅔x₁ - ⅓ — c'est l'équation d'une droite, comme au collège (y = ax + b).*
>
> Cette droite sépare le plan en deux zones : C₁ (là où 1 + 2x₁ + 3x₂ ≥ 0) et C₂ (là où 1 + 2x₁ + 3x₂ ≤ 0). Pour un nouveau point x*, on regarde juste le signe de cette expression pour savoir de quel côté il tombe → c'est exactement la règle de décision du perceptron.
>
> En dimension d quelconque, la même idée se généralise : l'équation de l'hyperplan est 
> ∑ⱼ wⱼxⱼ + w₀ = 0 (somme de j=1 à d),
> et l'hypothèse (= la fonction que le perceptron a apprise) s'écrit :
> h(x) = ∑ⱼ wⱼxⱼ + w₀.
> Le signe de h(x) donne la classe prédite (+ ou -).
>
> → Les wⱼ sont donc les coefficients qui définissent l'orientation de l'hyperplan, et w₀ le décalage par rapport à l'origine (le "biais" du neurone, à ne pas confondre avec le biais inductif vu au cours 1 !).


Note : à ce stade du cours, le perceptron "de base" utilise une fonction seuil/signe — la vraie importance de la non-linéarité de la fonction d'activation deviendra centrale plus tard, avec le perceptron multi-couches, car sans elle, empiler des couches reviendrait à faire une seule combinaison linéaire, donc ça ne servirait à rien.

**Comment on décrit l'hyperplan ?**
On définit la **normale**, et c'est cette normale qui va varier pour ajuster et intégrer les points mal placés (les exemples arrivent les uns après les autres, et à chaque fois le perceptron regarde si l'exemple est mal classé ; si c'est le cas, il adapte).

> ### 💡 Explication : le vecteur de poids w EST la normale à l'hyperplan
>
> Dans l'équation ∑ wⱼxⱼ + w₀ = 0, le vecteur w = (w₁, …, w_d) est justement le vecteur **normal** (perpendiculaire) à l'hyperplan. Faire varier w, c'est donc littéralement faire pivoter/orienter l'hyperplan dans l'espace.
>
> L'intuition de l'algorithme d'apprentissage (slide 20) : on a un point mal classé xᵢ. On veut faire tourner w pour que l'hyperplan se rapproche de ce point et finisse par le classer du bon côté. La règle de mise à jour est :
>
> w' = w + η·yᵢ·xᵢ
>
> où η est le taux d'apprentissage (learning rate) et yᵢ est la vraie classe de l'exemple (+1 ou -1). Intuition géométrique : on ajoute au vecteur w un petit vecteur dans la direction de xᵢ (pondéré par le signe yᵢ), ce qui "tire" la normale vers ce point mal classé, jusqu'à ce qu'il finisse du bon côté.
>
> Algorithme complet (slide 23) :
> 1. Initialiser w = 0.
> 2. Tant que ce n'est pas convergé : pour chaque exemple (xᵢ, yᵢ), si yᵢ·(w·xᵢ) ≤ 0 (= mal classé), alors mettre à jour w ← w + η·yᵢ·xᵢ.
> 3. On s'arrête dès qu'un passage complet sur les données ne produit plus aucune correction.

à la fin l'hyperplan sera une somme pondérée des exemples.

→ Logique : puisqu'on ne fait qu'additionner des termes η·yᵢ·xᵢ à w à chaque correction, en partant de w = 0, le vecteur final w est bien une combinaison linéaire (une somme pondérée) des exemples d'apprentissage qui ont, à un moment, été mal classés. C'est ce qu'on appelle la "forme duale" du perceptron.

**Culture G :** c'est quoi un SVM ?

>💡 Un SVM (Support Vector Machine) reprend cette même idée de "l'hyperplan comme combinaison pondérée de certains exemples" mais va plus loin : au lieu de s'arrêter dès qu'il trouve N'IMPORTE QUELLE séparatrice, il cherche la séparatrice qui maximise la marge (la distance minimale entre l'hyperplan et les exemples les plus proches de chaque classe). Ces exemples les plus proches sont les "vecteurs de support" (d'où le nom). C'est une amélioration directe du perceptron : le perceptron s'arrête au premier séparateur trouvé, le SVM cherche le meilleur séparateur possible.

L'algorithme du perceptron permet de minimiser les erreurs.

Vu autrement, avec la formulation par descente de gradient — slide 24-25 : on peut aussi voir l'apprentissage du perceptron comme la minimisation d'un risque empirique sur les exemples mal classés M.  
R_Emp(w) = -∑ wᵀxⱼ·uⱼ  
Minimiser cette quantité revient exactement à la même règle de mise à jour que la règle "ad hoc" historique de Rosenblatt : "apprendre seulement en cas d'échec". Les deux points de vue — règle ad hoc / descente de gradient — mènent au même algorithme, ce qui est plutôt satisfaisant intellectuellement.

---

### Propriétés remarquables

**Convergence en un nombre fini d'étapes :**
– Indépendamment du nombre d'exemples
– Indépendamment de la distribution des exemples
– Indépendamment de la dimension de l'espace d'entrée

⚠️ Condition cruciale : cette garantie de convergence n'est valable que S'IL EXISTE au moins une séparatrice linéaire des exemples (= si les données sont linéairement séparables). Si ce n'est pas le cas, l'algorithme ne convergera jamais (il continuera à corriger indéfiniment).


> ### FUN FACT : débat sur la justesse de ce théorème entre statisticien et data scientist
> 
> pourquoi cette différence de point de vue est LE concept clé de cette section
>
> C'est en fait l'opposition classique entre deux grandes familles de méthodes en apprentissage automatique :
>
> - **Approche générative (le statisticien)** : on essaie d'estimer la distribution de probabilité complète p(x | classe) pour chaque classe (ex. en supposant une loi gaussienne), puis on en déduit la frontière optimale entre les classes.   
> Problème : plus l'espace a de dimensions, plus il faut de données pour estimer correctement cette distribution (c'est la "malédiction de la dimensionnalité" : le volume de l'espace croît exponentiellement avec le nombre de dimensions, donc les données deviennent "clairsemées" et l'estimation devient très difficile/coûteuse).
>
> - **Approche discriminative (le perceptron / le "data scientist")** : on ne cherche PAS à modéliser toute la distribution des données. On cherche directement la frontière de décision qui sépare les classes, sans se soucier de la forme exacte des nuages de points.  
> C'est pour ça que le perceptron est "économique" : il n'a besoin de connaître ni la distribution, ni le nombre d'exemples, ni même la dimension de l'espace pour garantir sa convergence — il n'essaie de résoudre qu'un problème plus simple (trouver UNE frontière) plutôt qu'un problème plus difficile (connaître toute la distribution).
> → Le prix à payer : en ne modélisant pas la distribution, on perd de l'information (on ne sait pas, par exemple, à quel point un point est "typique" d'une classe, ni estimer des probabilités). C'est exactement ce que dit la note : "on est économique, mais il nous manque des infos".
> L'algorithme s'arrête dès qu'on a une séparatrice : il pourrait être vraiment plus proche de l'un des nuages de points que de l'autre, et on s'en fiche si ce n'est pas au milieu.
>(Ce point rejoint directement l'intérêt du SVM évoqué plus haut : le SVM corrige justement ce défaut en cherchant la séparatrice la mieux centrée possible entre les deux nuages, alors que le perceptron s'arrête à la première séparatrice trouvée, même si elle "frôle" un des deux nuages.)

**Capacité expressive :**
Trouver un hyperplan, ça va beaucoup réduire l'espace des hypothèses. 
En effet cela rejoint directement la notion de biais de représentation vue au cours 1 : en décidant de ne chercher QUE des hyperplans, on restreint énormément l'espace des hypothèses considérées H — c'est un choix de biais fort, qui rend le problème traitable, mais qui limite aussi ce qu'on peut apprendre — voir "Limites" ci-dessous.

---

### Limites

Le Perceptron ne peut apprendre que des séparatrices **linéaires** (e.g. pas le XOR), mais du coup, c'est un seul neurone.

> ### 💡 Explication : pourquoi le XOR est le contre-exemple emblématique
>
> Le cours illustre ce point avec les 3 fonctions logiques à 2 entrées booléennes I₁, I₂ (slide 28) :
> - **I₁ AND I₂** : linéairement séparable (une droite sépare facilement le point (1,1) des trois autres).
> - **I₁ OR I₂** : linéairement séparable aussi (une droite sépare le point (0,0) des trois autres).
> - **I₁ XOR I₂** : **PAS linéairement séparable !** Les points "vrais" ((0,1) et (1,0)) et les points "faux" ((0,0) et (1,1)) sont disposés en damier — aucune droite ne peut les séparer proprement. C'est LE contre-exemple historique qui montre la limite fondamentale d'un perceptron à une seule couche.
>
> Nuance (slide 29) : ce n'est "pas tout à fait vrai" si on change d'espace d'entrée [Minsky et Papert, 1969] — c'est-à-dire qu'en transformant astucieusement les données au préalable (par un prétraitement), on peut parfois rendre séparable ce qui ne l'était pas dans l'espace d'origine. Mais ce prétraitement est difficile à trouver et n'est pas généralisable — d'où l'idée, développée juste après, d'empiler plusieurs couches de neurones pour que le réseau apprenne LUI-MÊME cette transformation, plutôt que de la faire à la main.

### **Le perceptron, le bouquin Rosenblatt (1958-1962) :** faudrait faire une succession, non ? Voir plusieurs couches.
Le souci, c'est apprendre les poids, car si on en change un, ça change tout.

> 💡 Explication : pourquoi passer à plusieurs couches pose un problème nouveau
>
> Avec un seul neurone (un seul perceptron), la règle d'apprentissage est simple : on sait directement si la sortie est fausse, et de quel côté il faut corriger w. Mais si on empile plusieurs neurones en couches successives (couche d'entrée → couche cachée → couche de sortie, voir plus bas), le problème devient : **comment sait-on quels poids, dans quelle couche, sont responsables de l'erreur finale ?**  
> Changer un seul poids dans une couche intermédiaire a des répercussions sur TOUS les neurones des couches suivantes (puisque leurs entrées en dépendent). On ne peut donc plus appliquer directement la règle simple du perceptron à chaque couche indépendamment — il faut un moyen de "répartir la responsabilité" de l'erreur entre toutes les couches. (Ce problème, appelé le problème de l'assignation du crédit, sera résolu par l'algorithme de rétropropagation du gradient)

**Ils ont tué le domaine en résolvant tous les problèmes résolvables en 6 mois, pour les Master.**  
Allusion historique : c'est une référence à l'analyse de Minsky et Papert (1969) qui, en démontrant formellement les limites du perceptron (comme le XOR), a douché l'enthousiasme de l'époque et précipité le premier "hiver de l'IA" pour le connexionnisme — plus personne ne finançait la recherche sur les réseaux de neurones pendant presque 15 ans, jusqu'au retour en grâce avec le Perceptron Multi-Couches en 1985. On retrouve ici exactement la 4ᵉ période de la frise historique vue au cours 1 : "1980-95 : renouveau du connexionnisme, Hopfield (1982), Perceptron Multi-Couches (1985)".

---

## Les perceptrons multi-couches

### Topologie typique d'un PMC

Le perceptron multi-couches (PMC) empile plusieurs couches de neurones :

- **Couche d'entrée** : reçoit le signal d'entrée xₖ (autant de neurones que d'attributs/descripteurs).
- **Couche(s) cachée(s)** : une ou plusieurs couches intermédiaires, entièrement connectées à la couche précédente (chaque neurone d'une couche reçoit les sorties de TOUS les neurones de la couche précédente).
- **Couche de sortie** : produit la sortie finale yₖ du réseau.

Le signal circule dans un seul sens, de la couche d'entrée vers la couche de sortie ("flot des signaux") — c'est pour cela qu'on parle de réseau "feedforward" (par opposition aux réseaux "bouclés" comme celui de Hopfield, mentionné dans l'historique).

Pendant l'apprentissage, on compare la sortie produite yₖ à la **sortie désirée** uₖ (la vraie étiquette attendue) pour calculer l'erreur — c'est cette comparaison qui permettra (au prochain cours) de corriger les poids de TOUTES les couches via la rétropropagation.

→ L'idée centrale à retenir : en ajoutant une couche cachée entre l'entrée et la sortie, le réseau peut désormais apprendre des séparations NON linéaires (comme le XOR), car la couche cachée peut recombiner les entrées de façon non triviale avant la décision finale de la couche de sortie. C'est exactement la réponse au problème du XOR soulevé plus haut : on n'a plus un seul hyperplan, mais une combinaison de plusieurs hyperplans (un par neurone caché), ce qui permet de découper l'espace en régions bien plus complexes.