# Cours 9 : Introduction à l'IA explicable

## Partie A : Motivation et contexte

**Q1.** Pourquoi les systèmes experts (MYCIN, années 1970) savaient-ils « s'expliquer » presque gratuitement ? Reprends l'exemple de la tétracycline. Distingue explication « de surface » et explication « profonde » : à qui s'adresse chacune ?

**Réponse.**
Un système expert fonctionne avec des **règles écrites par des humains** ; pour expliquer une réponse, il suffit de **rejouer la chaîne de règles** utilisée. L'explication est une conséquence directe de la construction du système, d'où « gratuite ».
Exemple (diapo 5), « pourquoi pas de tétracycline avant 8 ans ? » : le médicament se dépose sur les os en développement → noircissement définitif des dents → coloration socialement indésirable → ne pas administrer. L'explication fait appel à des **relations de causalité** et à la notion d'**effet secondaire indésirable**.

|                    | Contenu                                                                    | Pour qui ?                            |
| ------------------ | -------------------------------------------------------------------------- | ------------------------------------- |
| **Surface**  | *comment* le système fonctionne (quelles règles se sont déclenchées) | l'expert en IA                        |
| **Profonde** | *pourquoi* c'est vrai dans le domaine (raisons de fond, causalité)      | l'expert du domaine, les utilisateurs |

Expliquer « ce que la machine a calculé » n'est pas expliquer « pourquoi c'est la bonne décision » ; les explications profondes sont **difficiles à découvrir et à formuler**.

---

**Q2.** Pourquoi les réseaux de neurones profonds ont-ils rendu le problème de l'explicabilité aigu ? Cite les trois illustrations du cours (diapos 10-12) et le type de motivation qu'illustre chacune.

**Réponse.**
Les réseaux profonds ont de très nombreuses couches (parfois des centaines) et de **$10^7$ à $10^9$ paramètres** : impossible de les « lire » comme une liste de règles, ce sont des **boîtes noires**.

- **Illusions d'optique** (article Grad-CAM) : un réseau peut se tromper face à une image, on veut comprendre **sur quoi il s'est appuyé**.
- **Médecine** : un système d'aide au diagnostic doit **justifier ses résultats** auprès du médecin.
- **AlphaGo** : joue comme un « joueur extraterrestre » et a révolutionné le jeu de go ; l'enjeu est la **découverte scientifique** : si la machine a trouvé ce que les humains ignoraient, on veut **apprendre d'elle**, donc comprendre ce qu'elle a compris.

---

**Q3.** (a) Que veut dire « le ML ne crée pas d'information » ? Quel compromis en découle ? (b) Où se cachent les connaissances a priori ? Donne au moins trois endroits dans les données et deux exemples de biais de représentation. (c) Pourquoi est-ce important pour l'explicabilité ?

**Réponse.**
(a) Le ML **reformule** ce qu'on lui donne : c'est une sorte de **théorème de conservation**. Ce que le modèle sait vient soit des **données**, soit des **connaissances a priori**. Compromis : *peu de données + beaucoup d'a priori* ou *beaucoup de données + peu d'a priori*. (Analogie : élève avec un bon cours mais peu d'exercices, ou des milliers d'exercices corrigés sans cours.)
(b) **Dans les données** (diapos 16-17, « Biais sur les données ») : dispositif expérimental ; choix des descripteurs ; enrichissement par des ontologies ; normalisation ; traitement des valeurs manquantes ; exemples ajoutés (invariances). Chacun est aussi une source possible de biais.
**Dans l'algorithme** (**biais de représentation**) : hypothèses **linéaires parcimonieuses** (peu de coefficients non nuls, on favorise les modèles simples) ; **réseaux convolutifs** (on suppose l'information **locale** et un motif pouvant apparaître **n'importe où**).
(c) Comprendre un modèle passe aussi par **comprendre ce qu'on lui a injecté** (données + biais).

---

**Q4.** (a) Classe par ordre croissant de difficulté d'interprétation : réseaux de neurones, arbres de décision, forêts aléatoires, fonctions linéaires, SVM. (b) Quelle différence entre utiliser une fonction de décision seule ou l'intégrer dans un système d'inférence plus large ? (c) Qu'appelle-t-on « interprétabilité du processus d'apprentissage » ?

**Réponse.**
(a) Du plus facile au plus difficile : **arbres de décision > fonctions linéaires > forêts aléatoires > SVM, réseaux de neurones** (ces deux derniers « nécessitent une analyse difficile »).
(b) **Seule**, on peut se contenter de la prédiction. **Intégrée** dans un système d'inférence plus large (par exemple un raisonnement juridique), on doit pouvoir **questionner d'où vient la décision**.
(c) Comprendre comment l'apprentissage se déroule : **analyse de sensibilité** (« si je change cette entrée, que se passe-t-il en sortie ? ») et **Explanation-Based Learning** (apprentissage guidé par des explications).

---

## Partie B : Besoin d'interprétabilité et effet Clever Hans

**Q5.** Dans quels cas l'interprétabilité n'est-elle pas nécessaire ? Quelles sont les quatre raisons de la rendre nécessaire ? Pour chaque situation, indique la raison : (a) recommandation de films ; (b) reconnaissance de caractères ; (c) décision chirurgicale ; (d) comprendre un coup surprenant d'AlphaGo ; (e) comprendre pourquoi un système semble favoriser un genre ; (f) prêt accordé « si plus de 2 cartes de crédit ».

**Réponse.**
**Non nécessaire** : quand le **risque est faible** ; quand il existe de **bonnes garanties de performance**.
**Nécessaire** (4 raisons) : ① **décisions à haut risque** ; ② **satisfaire la curiosité** (but de la science) ; ③ **déboguer** ; ④ **exiger des systèmes de meilleure qualité** (non manipulables, fondés sur des facteurs causaux).
(a) **Non nécessaire** : risque faible. (b) **Non nécessaire** : bonnes garanties de performance. (c) ① **haut risque**. (d) ② **curiosité / découverte scientifique**. (e) ③ **débogage** (biais). (f) ④ **manipulation** : il suffit d'ouvrir 3 cartes ; le système devrait s'appuyer sur des **facteurs causaux** et non sur de simples corrélations.

---

**Q6.** Qu'est-ce qu'un raisonnement contrefactuel ? Illustre-le avec l'exemple du vélo. Pourquoi dit-on qu'un système interprétable peut être manipulé, et quelle parade suggère le cours ?

**Réponse.**
Raisonnement **contrefactuel** : « qu'est-ce qui aurait dû être différent pour que la décision change ? ». Exemple : si un vélo n'est reconnu que parce qu'il a **deux roues**, que se passe-t-il quand une roue est cachée derrière des sacoches ? Cela permet de **déboguer** (comprendre pourquoi une décision est fausse).
Manipulation : si l'on connaît la règle (« prêt si plus de 2 cartes »), on la contourne (ouvrir 3 cartes). Parade : s'appuyer sur des **facteurs causaux** et non sur de simples corrélations ; une **explication causale** est robuste, non manipulable et généralisable.

---

**Q7.** Raconte l'histoire de Clever Hans et explique le parallèle avec l'apprentissage supervisé. Quel exemple le cours donne-t-il (diapos 38-39) ? Quelle leçon en tires-tu sur le score de performance ?

**Réponse.**
**Hans le Malin** : cheval allemand réputé capable de calculer ; Pfungst a découvert qu'il n'avait **aucune** capacité de calcul, il **lisait des signaux corporels** émis inconsciemment par son maître ou par les personnes qui l'interrogeaient.
Parallèle : un modèle peut donner la **bonne réponse pour de mauvaises raisons**, en exploitant des **corrélations accidentelles (*spurious*) mais très prédictives** dans la base d'apprentissage.
Exemple : les 9963 images de chevaux de Pascal VOC ; ➕ beaucoup contenaient une mention de copyright dans un coin : le réseau détectait la mention, pas le cheval (il suffisait de la coller sur une voiture).
Leçon : un **bon score ne prouve pas** que le modèle a compris la tâche ; seule une explication permet de détecter ce genre de « triche » involontaire.

---

## Partie C : Taxonomie et dilemme central

**Q8.** Définis les couples : *by-design* / *post-hoc*, *model-agnostic* / *model-specific*, locale / globale. Place dans la taxonomie : régression linéaire, arbre de décision, LIME, valeurs de Shapley, Grad-CAM, SmoothGrad, Concept Bottleneck. Quelles sont les analogies de la recette et du plat ?

**Réponse.**

- **By-design (ante hoc)** : le modèle est interprétable **par nature** (son utilisation même donne l'importance des descripteurs). **Post-hoc** : on **découple** l'apprentissage et l'explication ; on explique **après coup** un modèle éventuellement très complexe.
- **Model-agnostic** : traite le modèle comme une boîte noire (entrées/sorties), marche avec n'importe quel modèle. **Model-specific** : exploite la structure interne d'une famille de modèles (plus riche, mais limitée).
- **Locale** : explique la prédiction pour **un exemple**. **Globale** : explique la **stratégie générale** du modèle.

| Méthode                     | Famille                                              |
| ---------------------------- | ---------------------------------------------------- |
| Régression linéaire, arbre | by-design                                            |
| LIME, Shapley                | post-hoc, agnostique (locales)                       |
| Grad-CAM, SmoothGrad         | post-hoc, spécifique (réseaux de neurones, vision) |
| Concept Bottleneck           | par conception (architecture), pour les réseaux     |

Analogies : by-design = **recette écrite par le cuisinier** (on la lit) ; post-hoc = **goûter un plat dont on ignore la recette** et deviner les ingrédients.

---

**Q9.** Quels sont les avantages et l'inconvénient des modèles par conception ? Quel est le dilemme central de l'IA explicable ? Quelle position défend Rudin (diapo 124) ?

**Réponse.**
✅ Plus facilement interprétables ; on interprète **le modèle entier**, pas seulement une prédiction. ⚠️ Souvent enclins au **sous-apprentissage** : trop simples pour capturer des relations complexes.
**Dilemme : interprétabilité ↔ performance** (les modèles simples sont lisibles mais moins puissants ; les modèles puissants sont opaques). Les méthodes post-hoc cherchent à le résoudre.
Rudin défend l'idée **inverse** : pour les **décisions à haut risque**, mieux vaut utiliser des **modèles interprétables** plutôt que d'expliquer des boîtes noires.

---

## Partie D : Modèles par conception

**Q10.** Écris le modèle de régression linéaire et explique comment on estime les coefficients. Cite les quatre hypothèses (diapo 48). Comment interprète-t-on un coefficient $b_j$ ? Donne deux avantages et deux inconvénients.

**Réponse.**
$\hat y=b_0+b_1x_1+\dots+b_dx_d$ ; les $b_j$ minimisent l'**écart quadratique** entre prédictions et valeurs réelles (**moindres carrés**).
Hypothèses : fonction cible (quasi) **linéaire** ; sortie **normale** sachant les descripteurs ; **homoscédasticité** (même dispersion partout) ; descripteurs **non corrélés**.
$b_j$ = de combien varie la sortie quand $x_j$ augmente d'**une unité**, **toutes choses égales par ailleurs** (ex. le prix d'une maison avec une pièce de plus). Le graphique des **effets individuels** donne déjà une explication **locale**.
Avantages : simple à mettre en œuvre ; **intervalles de confiance**. Inconvénients : suppose des descripteurs décorrélés ; ne rend pas bien compte des **dépendances non linéaires**.

---

**Q11.** Le tableau d'une régression donne pour un descripteur A : coefficient 12,5, erreur standard 5 ; pour un descripteur B : coefficient 3, erreur standard 4. (a) Calcule la t-statistique de chacun. (b) Donne un intervalle de confiance approximatif à 95 % (coefficient $\pm1{,}96\times$ erreur standard). (c) Conclus. (d) Que représente la p-valeur $P>|t|$ ?

**Réponse.** (➕ approximation normale du 1,96, valable pour un échantillon assez grand)
(a) $t=\dfrac{\text{coefficient}}{\text{erreur standard}}$ : A : $12{,}5/5=\mathbf{2{,}5}$ ; B : $3/4=\mathbf{0{,}75}$.
(b) A : $12{,}5\pm9{,}8=[\mathbf{2{,}7\,;\,22{,}3}]$ ; B : $3\pm7{,}84=[\mathbf{-4{,}8\,;\,10{,}8}]$.
(c) A : l'intervalle **ne contient pas 0**, $|t|$ grand ($p\approx0{,}01<0{,}05$) → effet **significatif**. B : l'intervalle **contient 0**, $p\approx0{,}45$ → l'effet **n'est pas clairement établi**.
(d) La probabilité d'observer un tel coefficient **si la variable n'avait en réalité aucun effet** ; $<0{,}05$ : le descripteur a un impact significatif.

---

**Q12.** Une régression linéaire donne un coefficient **négatif** à un descripteur pourtant positivement corrélé à la cible. Explique pourquoi avec un exemple. Quel outil permet de le détecter ? Quelle conséquence sur la lecture des coefficients ?

**Réponse.**
Si deux descripteurs sont **fortement corrélés** et que l'un reçoit un très fort coefficient positif, l'autre peut recevoir un coefficient **négatif** « pour corriger ». Exemple (➕ complément) : surface et nombre de pièces pour prédire le prix : gros coefficient sur la surface, petit coefficient négatif sur les pièces ; conclure que « plus de pièces fait baisser le prix » serait une erreur.
Outil : la **matrice de corrélation** (+1, −1, 0) ; une forte corrélation signale des descripteurs **redondants**.
Conséquence : le coefficient ne se lit plus comme « l'effet de cette variable » : il **dépend des autres variables du modèle**.

---

**Q13.** Donne deux avantages et trois inconvénients des arbres de décision comme modèles interprétables. Que signifie le titre de la diapo 56, « Lisibles… jusqu'à un certain point » ? Illustre avec les fins de parties aux échecs.

**Réponse.**
Avantages : capturent naturellement les **interactions** entre descripteurs ; montrent **comment se forment différents groupes** ; structure se prêtant à l'interprétation (jeu de 20 questions).
Inconvénients : peu adaptés aux **dépendances linéaires** (beaucoup de découpes pour approcher une droite → arbres obliques) ; **instables** (un petit changement des données change beaucoup l'arbre) ; peuvent devenir **opaques** si les descripteurs sont mal choisis.
Un arbre à 3 niveaux se lit facilement, un arbre à 300 nœuds non. Échecs : avec des descripteurs bruts (positions des pièces), on ne peut pas formuler de règle lisible pour expliquer les fins de parties.

---

## Partie E : LIME

**Q14.** Explique l'idée de LIME avec l'analogie de la carte plane. Décris les six étapes de l'algorithme et lis la figure de la diapo 62 (croix rouge, points pondérés, droite pointillée).

**Réponse.**
**LIME** (*Local Interpretable Model-agnostic Explanations*) : le modèle $f$ est peut-être très complexe, mais **de très près autour d'un point** il ressemble à quelque chose de simple. Comme la Terre est ronde mais une carte plane suffit à l'échelle d'un quartier.
Étapes : 1. choisir l'**exemple à expliquer** ; 2. générer des **exemples virtuels** en perturbant des valeurs d'attributs ; 3. leur donner un **poids** selon leur **distance** à l'exemple ; 4. **prédire** leur classe avec la boîte noire ; 5. entraîner un **modèle surrogé simple** (ex. linéaire) sur ces exemples étiquetés ; 6. **interpréter** le modèle surrogé (lire les coefficients).
Figure : $f$ a une frontière complexe (zone bleue / rose) ; la **croix rouge** est l'exemple à expliquer ; les points proches sont plus gros (plus de poids) ; la **droite en pointillés $g$** est le modèle simple qui approxime $f$ **localement**.

---

**Q15.** Pourquoi le choix de la fonction de distance est-il crucial dans LIME ? Quel noyau et quelle largeur utilisent Ribeiro et al., et que vaut cette largeur pour $d=16$ descripteurs ? Quelles sont les limites de LIME ?

**Réponse.**
La distance définit ce que veut dire « **proche** », donc la **taille du voisinage local** sur lequel le modèle simple doit imiter la boîte noire.
Ribeiro et al. : **noyau exponentiel** de largeur $0{,}75\times\sqrt{d}$ ; pour $d=16$ : $0{,}75\times4=\mathbf3$.
Limites (➕ complément) : le résultat dépend de la façon dont on génère les exemples virtuels et de la largeur du voisinage ; deux exécutions peuvent donner des explications un peu différentes. Pas de garantie d'additivité (contrairement à Shapley).

---

## Partie F : Valeurs de Shapley

**Q16.** Deux joueurs, 1 et 2, forment une coalition. Gains : $v(\emptyset)=0$, $v(\{1\})=6$, $v(\{2\})=2$, $v(\{1,2\})=10$. (a) Calcule les valeurs de Shapley des deux joueurs en considérant tous les ordres d'arrivée. (b) Vérifie l'efficacité. (c) Pourquoi moyenner sur tous les ordres ?

**Réponse.**
(a) Deux ordres :

- 1 → 2 : le joueur 1 apporte $6$, le joueur 2 apporte $10-6=4$ ;
- 2 → 1 : le joueur 2 apporte $2$, le joueur 1 apporte $10-2=8$.

$\varphi_1=\dfrac{6+8}2=\mathbf7$ ; $\varphi_2=\dfrac{4+2}2=\mathbf3$.
(b) $7+3=10=v(\{1,2\})$ : **efficacité** (tout le gain est réparti, rien n'est perdu ni inventé). Remarque : la synergie $10-(6+2)=2$ est partagée équitablement (+1 chacun par rapport au gain seul).
(c) La contribution d'un joueur **dépend de qui est déjà là** (dans l'exemple Kaggle, E3 n'apporte rien quand E1 et E2 sont présents, mais beaucoup s'il arrive en premier). Pour être équitable, on regarde toutes les façons de construire la coalition ($d!$ ordres).

---

**Q17.** Dans la formule $\varphi_j=\sum_{S\subseteq\{1..d\}\setminus\{j\}}\frac{|S|!\,(d-|S|-1)!}{d!}\big(val(S\cup\{j\})-val(S)\big)$ : (a) décode chaque terme ; (b) calcule les poids pour $d=3$ et vérifie qu'ils somment à 1 ; (c) que représente le poids ?

**Réponse.**
(a) $j$ : le descripteur étudié ; $S$ : une **coalition** ne contenant pas $j$ ; $d$ : nombre de descripteurs ; $val(S\cup\{j\})-val(S)$ : **contribution marginale** de $j$ ; $val_x(S)$ : prédiction attendue quand seuls les descripteurs de $S$ sont connus (les autres étant moyennés selon leur distribution), moins la prédiction moyenne.
(b) $d=3$, les coalitions sont des sous-ensembles des 2 autres descripteurs :

- $|S|=0$ (1 coalition) : $\frac{0!\,2!}{3!}=\frac13$ ;
- $|S|=1$ (2 coalitions) : $\frac{1!\,1!}{3!}=\frac16$ chacune ;
- $|S|=2$ (1 coalition) : $\frac{2!\,0!}{3!}=\frac13$.

Total : $\frac13+2\times\frac16+\frac13=\mathbf1$ ✔.
(c) La **fraction des ordres d'arrivée** dans lesquels $j$ arrive juste après $S$ : c'est ce qui réalise la « moyenne sur tous les ordres ». En une phrase : la valeur de Shapley de $j$ est la **moyenne pondérée** de tout ce que $j$ apporte à la prédiction, dans toutes les coalitions possibles.

---

**Q18.** (a) Donne la correspondance théorie des jeux / ML pour joueur, coalition, gain, gain de référence. (b) Que cherche-t-on à expliquer ? (c) Pour un modèle linéaire, $\varphi_j=b_j(x_j-\bar x_j)$. Application : $h(x)=3x_1-2x_2+10x_3$, moyennes $\bar x=(10,5,2)$, exemple $x=(12,8,1)$. Calcule les $\varphi_j$ et vérifie qu'ils somment à $h(x)-E[h]$.

**Réponse.**
(a) Joueur = **descripteur** ; coalition = sous-ensemble de descripteurs « **connus** » ; gain = **prédiction du modèle** ; gain de référence (personne ne joue) = **prédiction moyenne** $E[\hat h(X)]$.
(b) L'**écart entre la prédiction pour l'exemple et la prédiction moyenne**, réparti entre les descripteurs.
(c) ➕ (formule indiquée comme complément dans ta fiche)

- $\varphi_1=3\times(12-10)=\mathbf{+6}$ ;
- $\varphi_2=-2\times(8-5)=\mathbf{-6}$ ;
- $\varphi_3=10\times(1-2)=\mathbf{-10}$.

Somme $=-10$. Vérification : $h(x)=36-16+10=30$ ; $E[h]=30-10+20=40$ ; $30-40=-10$ ✔.

---

**Q19.** Quelles sont les deux difficultés pratiques du calcul des valeurs de Shapley ? Combien de coalitions pour $d=10$, $d=20$, $d=50$ ? Quelles solutions ?

**Réponse.**

1. **Distribution des attributs inconnue** : on l'estime en tirant au hasard, dans l'**ensemble d'apprentissage**, des valeurs pour les variables « inconnues » de la coalition (idéalement toutes les valeurs).
2. **Explosion combinatoire** : $2^d$ coalitions : $d=10$ → **1 024** ; $d=20$ → **plus d'un million** ($1\,048\,576$) ; $d=50$ → $\approx10^{15}$, **hors de portée**. ➕ Solution pratique : **échantillonnage de Monte-Carlo** (on tire au hasard des ordres d'arrivée au lieu de les énumérer tous).

---

**Q20.** Que signifie « SHAP mesure l'importance pour le modèle, pas pour le monde réel » (G. Zuin) ? Compare LIME et Shapley (idée, base théorique, additivité, coût).

**Réponse.**
Si le modèle a appris une corrélation absurde (**effet Clever Hans**), SHAP lui attribuera de l'importance : il explique **le modèle**, pas **la réalité**.

|                 | LIME                                           | Shapley (SHAP)                                          |
| --------------- | ---------------------------------------------- | ------------------------------------------------------- |
| Idée           | modèle simple local qui imite la boîte noire | partage équitable de la prédiction entre descripteurs |
| Base théorique | approximation locale                           | théorie des jeux (propriétés d'équité)             |
| Additivité     | non                                            | **oui** (somme = prédiction − moyenne)          |
| Coût           | faible                                         | élevé ($2^d$ coalitions)                            |

(Tableau marqué ➕ dans ta fiche.) Dans le cours : LIME donne l'**importance relative** de chaque attribut ; Shapley donne la contribution « **exacte** » de chaque valeur mesurée, avec la propriété que tout s'additionne. Exemple des appartements : −10 k€ expliqués par +30, +10, 0, −50.

---

## Partie G : Réseaux de neurones en vision

**Q21.** Qu'est-ce qu'une carte de saillance ? Explique le vanilla gradient : quelle quantité est dérivée par rapport à quoi, et pourquoi un gradient élevé signifie-t-il qu'un pixel est important ? Qu'est-ce qu'un logit ?

**Réponse.**
Une **carte de saillance** (*saliency map*, *heat map*) est une image où les zones importantes pour la décision sont « allumées ».
Vanilla gradient : on calcule $\dfrac{\partial y_c}{\partial\text{pixel}}$, le gradient du **logit** de la classe prédite $y_c$ par rapport à chaque pixel. Le **logit** est le score brut avant le softmax. Un gradient élevé signifie : « si je change un peu ce pixel, la sortie change beaucoup ». La carte est souvent **bruitée** (le gradient fluctue brusquement d'un pixel à l'autre).

---

**Q22.** Soit $y=0{,}5x_1+20x_2+10$, avec $x_1=400$ et $x_2=3$. (a) Calcule les gradients. Quelle variable paraît la plus importante ? (b) Calcule Input × Gradient. Que change la conclusion ? (c) Quelles limites de ces méthodes sur les images (diapo 85) ?

**Réponse.**
(a) $\partial y/\partial x_1=0{,}5$ ; $\partial y/\partial x_2=20$ : $x_2$ paraît **40 fois plus important**.
(b) $x_1\cdot0{,}5=\mathbf{200}$ ; $x_2\cdot20=\mathbf{60}$ : c'est $x_1$ qui contribue **le plus** (≈ 3,3 fois). Les échelles des variables sont très différentes : le gradient seul mesure la **sensibilité**, multiplier par l'entrée mesure la **contribution** réelle (dans le cours : $30\times100=3000$ contre $50000\times5=250000$).
(c) Dans une image, les valeurs de pixels n'ont **pas de relation linéaire directe** avec la sortie ; la carte met en évidence la **structure de l'image** (contours, formes), agréable à regarder mais **pas nécessairement ce que le système utilise réellement**.

---

**Q23.** Explique la Guided Backpropagation : quelle modification de la rétropropagation, pourquoi, et quelle limite principale (illustre avec la fourrure) ?

**Réponse.**
On ne laisse passer que les **gradients positifs**. Dans la rétropropagation standard, les gradients négatifs diminuent l'activation des éléments **non liés** à la classe cible ; pour une *visualisation*, on veut voir ce qui est **associé** à la classe, et supprimer les gradients négatifs **réduit le bruit** (visualisations plus nettes). Utile aussi pour regarder les couches intermédiaires (formes basiques puis descripteurs plus abstraits).
Limite : **ne distingue pas les classes**. La fourrure est importante pour reconnaître un chat ; si plusieurs animaux à fourrure sont sur l'image, la GBP éclaire **toutes** les zones de fourrure : elle montre ce qui contribue à la classe, pas ce qui lui est **spécifique**. De plus, un détecteur de contours peut contribuer positivement à une couche profonde même si le contour vient d'une région non pertinente.

---

**Q24.** Grad-CAM : (a) principe ; (b) pourquoi s'arrêter à la dernière couche convolutive ; (c) points forts et limites ; (d) que montrent les exemples de succès et d'échec (diapos 90-91) ?

**Réponse.**
(a) On propage le gradient non pas jusqu'aux pixels mais **jusqu'à la dernière couche convolutive**.
(b) Cette couche, proche de la sortie, contient à la fois des informations **sémantiques** (« il y a un chat ») et **spatiales** (« à cet endroit ») ; les gradients y sont les plus forts aux endroits des objets ayant contribué positivement au logit cible.
(c) **Forts** : **discrimine les classes**, rapide, applicable à **tout réseau convolutif**, *heat map* intuitive. **Limites** : **résolution faible** (couches profondes de petite taille) → carte floue, moins précise. D'où sa popularité « malgré ses défauts ».
(d) Succès : même avec un **bruit adverse** (le réseau voit un « avion » à plus de 0,9999), Grad-CAM localise encore les catégories **originales** (boxer, chat tigré) : robustesse. Échecs : image classée « lévrier à 35 % » ou **anguille à 70 %** avec une carte peu convaincante.

---

**Q25.** (a) Pourquoi combine-t-on Guided Backprop et Grad-CAM (Guided Grad-CAM) ? (b) Quel est le principe de SmoothGrad et son analogie ? (c) ➕ Quelle idée porte Integrated Gradients ?

**Réponse.**
(a) Elles sont **complémentaires** : GBP met en évidence des **détails fins** mais ne discrimine pas les classes ; Grad-CAM **discrimine les classes** mais est grossière. On **multiplie les deux cartes** : précise et spécifique à la classe.
(b) **« Ajouter du bruit pour enlever du bruit »** : on crée plusieurs **copies bruitées** de l'image, on calcule la carte de chacune puis on **moyenne** ; le bruit s'annule, la carte est plus lisse. Analogie : superposer plusieurs photos légèrement tremblées donne une image plus stable. Coût : plusieurs calculs.
(c) ➕ Au lieu d'un gradient en un seul point, on part d'une **image de référence neutre** (ex. noire) et on **somme les gradients** le long du chemin jusqu'à l'image réelle, ce qui évite les gradients « saturés » (un pixel très important peut avoir un gradient proche de 0 si le réseau est déjà sûr).

---

**Q26.** Quelles sont les trois limites générales des méthodes à gradient (diapo 99) ? Décris l'expérience de Kindermans et al. et sa conclusion.

**Réponse.**

1. **Difficile de savoir si l'explication est correcte** (réellement causale dans la prédiction).
2. **Biais de confirmation** : on croit une explication bonne parce qu'elle satisfait nos attentes (« c'est bien la tête du chat qui est éclairée »).
3. **Instabilité** : de petites variations de l'image peuvent changer beaucoup l'explication.

Kindermans et al. (2019) : on **ajoute une constante à tous les pixels** ; le neurone de biais compense, les poids restent identiques et **le modèle est le même** ; pourtant les « explications » changent. Conclusion : si deux modèles **strictement équivalents** donnent des explications différentes, l'explication **ne reflète pas toujours le fonctionnement du modèle**.

---

## Partie H : Concept Bottleneck Models

**Q27.** (a) Quel défaut des cartes de saillance les CBM corrigent-ils ? (b) Décris l'architecture $f(g(x))$ (dimensions de $g$ et de $f$, fonctions de coût). (c) Quelles données faut-il ? (d) Qu'est-ce que l'intervention et pourquoi est-elle un avantage ? Donne un exemple.

**Réponse.**
(a) Les cartes montrent **où** le réseau regarde, pas **quoi** il voit (regarder « la zone du genou » ne dit pas s'il a vu une « excroissance osseuse »).
(b) On force le passage par des **concepts humains** : **encodeur de concepts $g:\mathbb R^d\to\mathbb R^k$** ($k$ concepts) qui transforme les données brutes en concepts, puis **prédicteur $f:\mathbb R^k\to\mathbb R^n$** ($n$ classes) qui utilise **uniquement** les concepts prédits. **Deux fonctions de coût** : une pour bien prédire les concepts ($\hat g$), une pour bien prédire la classe à partir des concepts ($\hat f$).
(c) Des **triplets $(x_i,y_i,c_i)$** : entrée, étiquette **et concepts annotés** (souvent par des experts) : c'est le principal coût des CBM.
(d) Un humain peut **corriger un concept** (« non, pas d'excroissance osseuse ici ») et voir **comment la prédiction change**. Exemples : radiographie du genou (excroissances osseuses, sclérose), oiseaux (couleur des ailes, longueur du bec). Analogie : médecin qui dit « j'ai observé ces trois signes, donc je conclus » : on peut discuter chaque signe.

---

**Q28.** Quels sont les cinq critères d'évaluation d'un système à concepts (diapo 106) ?

**Réponse.**

1. **Performance de prédiction** (non détériorée par la contrainte d'interprétabilité) ;
2. **Performance de reconnaissance des concepts** intermédiaires ;
3. **Stabilité** des poids entre concepts et classes (répétitions, conditions initiales) ;
4. **Cohérence** : les concepts importants dans le **codage** le sont-ils aussi dans le **calcul de la sortie** ?
5. **Sensibilité** de la sortie aux interventions sur les poids des concepts.

---

## Partie I : Quel avenir ?

**Q29.** Décris l'expérience de l'effet spectateur (conditions, résultats, explications données). Quelle leçon (diapo 116) et quel lien avec les méthodes d'explication post-hoc ?

**Réponse.**
Participants accueillis individuellement pour une « étude de marché sur les jeux » ; dans la pièce voisine, un fracas et un cri de douleur. **Seuls** : **70 %** interviennent. Avec un **complice passif** : **7 %**.
Explications après coup : seuls, « il fallait au moins aller voir » ; avec le complice, « ce n'était pas grave », et **aucun ne mentionne l'autre personne**, affirmant qu'elle n'a eu **aucune influence**.
Leçon : les raisons sont données **après coup**, sont **manifestement fausses** (la vraie cause est ignorée) : ce sont des **justifications** (rationalisations), pas des comptes rendus fidèles du processus.
Lien : une explication post-hoc (LIME, Grad-CAM, *chains of thought*) peut être une **histoire plausible** qui ne reflète pas le calcul réel : une explication plausible n'est pas forcément fidèle.

---

**Q30.** Pourquoi l'explicabilité devient-elle cruciale dans les systèmes multi-agents ? Reprends l'accident Tesla (2016) : que se serait-il passé si chaque capteur avait justifié ses doutes ? Que montre l'exemple de la publicité adaptative ?

**Réponse.**
Les IA sont de plus en plus des **sociétés d'agents** qui communiquent et décident selon leurs interactions : chaque agent devrait pouvoir **expliquer les raisons** de ses sorties aux autres.
Tesla : le **radar** avait détecté la semi-remorque, mais les panneaux routiers ont une signature radar proche de celle d'une voiture ; la **caméra** était incertaine (ciel blanc laiteux, remorque blanche). Si chacun justifiait son doute (« ça ressemble à un panneau » / « couleur proche du ciel »), un **module de fusion** qui reçoit des **justifications** (et pas seulement des scores) comprendrait que les **deux erreurs ont des causes indépendantes et faibles** et déciderait plus prudemment.
Publicité adaptative : deux sous-systèmes (placement des liens / choix des publicités) apprennent à partir de **données de clics** qui dépendent de **l'autre système** et de facteurs non contrôlés : ils s'influencent mutuellement, ce qui rend l'**analyse causale très délicate**.

---

**Q31.** (a) Quelle « formulation incomplète » du problème d'apprentissage le cours dénonce-t-il, et quelle conclusion en tire-t-il ? (b) Cite quatre des questions ouvertes de la fin du cours et explique l'une d'elles.

**Réponse.**
(a) Le **focus exclusif sur la performance prédictive** : on veut aussi l'interprétabilité des résultats, celle du processus d'apprentissage, et **mieux comprendre le monde** (intégrer la fonction apprise dans une théorie existante). Conclusion : il faut **changer le critère inductif** : aujourd'hui on minimise une erreur de prédiction, demain on devrait aussi optimiser l'**intelligibilité**, par exemple dans la fonction de coût.
(b) Questions ouvertes (diapos 122-123) : interprétabilité vs explicabilité ; comment évaluer une explication ; explications et causalité ; une explication est-elle symbolique par nature ; explication *one-shot* ou par interactions (dialogue) ; expliquer permet-il de tromper le système ; pour qui (experts IA, experts du domaine, utilisateurs) ; les *chains of thought* sont-elles fiables ; place de la causalité (« le graal »).
Exemple : **interprétabilité vs explicabilité** : *interpréter* = comprendre le fonctionnement d'un modèle transparent ; *expliquer* = produire une justification (souvent après coup) d'un modèle opaque.

---

## Partie J : Synthèse et ponts avec les autres cours

**Q32.** Pour chaque situation, choisis la méthode la plus adaptée et justifie.
(a) Expliquer la prédiction d'une forêt aléatoire pour un patient, avec l'exigence que les contributions s'additionnent exactement à l'écart avec la prédiction moyenne.
(b) Obtenir rapidement une explication locale d'un modèle quelconque.
(c) Voir où un CNN « regarde » sur une radiographie, en distinguant les classes.
(d) Les cartes de gradient sont très bruitées.
(e) Un médecin veut corriger une décision en agissant sur un signe clinique.
(f) Décision à haut risque nécessitant une lisibilité totale.
(g) Modèle linéaire avec variables d'échelles très différentes : le gradient seul semble donner une mauvaise importance.

**Réponse.**
(a) **Shapley** (propriété d'additivité/efficacité ; coût élevé, approximé par Monte-Carlo).
(b) **LIME** (modèle surrogé local, coût faible, agnostique).
(c) **Grad-CAM** (discrimine les classes, applicable à tout CNN) ; + Guided Backprop → **Guided Grad-CAM** si on veut aussi du détail fin.
(d) **SmoothGrad** (moyenne sur copies bruitées).
(e) **Concept Bottleneck Model** (intervention sur un concept).
(f) **Modèle par conception** (arbre petit, régression) : position de Rudin.
(g) **Input × Gradient** (contribution plutôt que simple sensibilité).

---

## Pièges fréquents à vérifier avant le partiel

| Piège                               | À retenir                                                                                                              |
| ------------------------------------ | ----------------------------------------------------------------------------------------------------------------------- |
| Interprétabilité vs explicabilité | Interpréter = comprendre un modèle transparent ; expliquer = justifier (souvent après coup) un modèle opaque        |
| By-design vs post-hoc                | Par conception : la lecture du modèle suffit ; post-hoc : explication séparée de l'apprentissage                     |
| Agnostique vs spécifique            | Agnostique : entrées/sorties seulement (LIME, Shapley) ; spécifique : structure interne (gradients, CBM)              |
| Coefficient négatif en régression  | Peut venir de **variables corrélées**, pas d'un effet négatif réel                                            |
| Valeurs de Shapley                   | Somme = prédiction − prédiction moyenne ; $2^d$ coalitions ; explique le **modèle**, pas le **monde** |
| LIME                                 | Surrogé local pondéré par la proximité ; la **distance** est cruciale ; pas d'additivité                     |
| Gradient seul vs Input×Gradient     | Sensibilité vs contribution                                                                                            |
| GBP vs Grad-CAM                      | GBP : détail fin, ne discrimine pas les classes ; Grad-CAM : discrimine les classes, résolution faible                |
| Kindermans                           | Deux modèles équivalents, explications différentes → l'explication ne reflète pas toujours le modèle              |
| CBM                                  | Il faut des annotations de concepts $(x,y,c)$ ; avantage : l'**intervention**                                   |
| Effet spectateur                     | Les raisons sont des **justifications après coup**, pas la vraie cause                                           |
| Explication ≠ cause                 | Une explication plausible n'est pas forcément **fidèle** ni **causale**                                   |

---

---

# Cours 10 : Causalité, inférence et apprentissage

## Questions type partiel : version corrigée

---

## Partie A : Pourquoi la causalité ? Corrélation et causalité

**Q1.** (a) Quelles sont les cinq raisons de s'intéresser à la causalité ? (b) Illustre la raison 5 (transférabilité) avec ton propre exemple. (c) Quelle est la forme générale des questions causales (diapo 10) et quels types de questions la causalité permet-elle de traiter ?

**Réponse.**
(a) 1. Les règles de décision apprises peuvent être **non interprétables** ; 2. les **explications** que nous produisons sont naturellement **causales** (on attend « parce que… », pas « parce que c'est corrélé ») ; 3. les algorithmes sont **difficiles à déchiffrer** (biais cachés) ; 4. identifier des liens causaux permet d'**intervenir** (résumer des données ≠ savoir ce qui se passera si j'agis) ; 5. les relations causales sont **transférables** d'un contexte à un autre.
(b) Exemple du cours : lunettes de soleil et glaces dans une région chaude : cette corrélation ne se transporte pas en hiver ni en Norvège ; le mécanisme « il fait chaud → on achète des glaces » tient partout. (Un exemple personnel valide : la corrélation ne vaut que dans le contexte où elle a été mesurée, le mécanisme causal vaut ailleurs.)
(c) $A\overset{?}{\longrightarrow}Y$ où $A$ est le traitement/l'événement (binaire ou réel) et $Y$ la variable d'intérêt. La causalité répond aux questions **« que se passe-t-il si… ? »** et **« pourquoi ? »** que les statistiques classiques ne traitent pas seules.

---

**Q2.** (a) Pourquoi dit-on que « causalité $\neq$ corrélation linéaire » ? (b) Calcule la covariance, donc le coefficient de Pearson, de $X\in\{-2,-1,0,1,2\}$ avec $Y=X^2$. (c) Quelles mesures détectent ce type de lien ?

**Réponse.**
(a) Pearson et Spearman ne détectent que les liens **linéaires** ou **monotones**. Une absence de corrélation de Pearson **ne prouve pas** l'absence de lien causal.
(b) $\bar x=0$ ; $Y=(4,1,0,1,4)$ donc $\bar y=2$. $\sum(x_i-\bar x)(y_i-\bar y)=(-2)(2)+(-1)(-1)+0\times(-2)+(1)(-1)+(2)(2)=-4+1+0-1+4=\mathbf0$. La covariance est nulle, donc **Pearson $=0$**, alors que $X$ cause $Y$ de façon parfaitement déterministe (les valeurs à gauche et à droite se compensent).
(c) **MIC** (*Maximal Information Coefficient*) et **HSIC** (*Hilbert-Schmidt Independence Criterion*) détectent des liens non linéaires et non monotones.

---

## Partie B : Colliders : Monty Hall, alarme

**Q3.** (a) Énonce le problème de Monty Hall et la bonne stratégie. (b) Explique-le avec le vocabulaire des colliders : quel est le graphe, pourquoi observer la porte ouverte crée-t-il une dépendance ? (c) L'animateur ouvre une porte **au hasard** (qui se trouve être une chèvre). Calcule, après avoir choisi la porte 1 et vu s'ouvrir la porte 3 (chèvre), la probabilité que la voiture soit derrière la porte 1 et derrière la porte 2. Quelle leçon ?

**Réponse.**
(a) 3 portes, une voiture ; tu choisis la porte 1 ; l'animateur, **qui sait où est la voiture**, ouvre une autre porte avec une chèvre ; **il faut changer** : on gagne 2 fois sur 3 (rester ne gagne que si le choix initial était le bon, probabilité 1/3).
(b) Graphe : **Ton choix → Porte ouverte ← Position de la voiture**. Le choix et la position de la voiture sont **indépendants** au départ ; mais l'animateur ne peut ouvrir ni ta porte ni celle de la voiture : en **conditionnant sur la porte ouverte** (le collider), on rend ces deux variables **dépendantes** (explaining away / effet de collider). L'information donnée par l'animateur « fuit » vers la position de la voiture. Le cours parle d'**illusion cognitive** : on cherche des explications causales même dans un raisonnement probabiliste.
(c) Animateur ignorant, tu as choisi la porte 1. Cas où la porte 3 est ouverte **et** contient une chèvre :

- voiture en 1 (prob. 1/3) : l'animateur ouvre la porte 3 avec probabilité 1/2 → $1/6$ ;
- voiture en 2 (prob. 1/3) : il ouvre la porte 3 avec probabilité 1/2 → $1/6$ ;
- voiture en 3 : la porte 3 révélerait la voiture → exclu.

Donc $P(\text{voiture en }1\mid\text{chèvre en }3)=\dfrac{1/6}{1/6+1/6}=\mathbf{\tfrac12}$ et de même $\tfrac12$ pour la porte 2 : changer ou non n'a **aucune importance**. Leçon : la **même observation** (une chèvre apparaît) a une signification différente selon le **mécanisme causal** qui l'a produite ; **les données seules ne suffisent pas**.

---

**Q4.** (a) Explique « explaining away » avec l'exemple de l'alarme (séisme/cambriolage). (b) Explique l'exemple talent/beauté/célébrité. (c) Quel point commun avec les biais de sélection ?

**Réponse.**
(a) **Séisme → Alarme ← Cambriolage**. A priori séisme et cambriolage sont **indépendants**. Si l'on **sait que l'alarme a sonné**, ils deviennent **dépendants** : apprendre qu'il y a eu un séisme **fait baisser** la probabilité d'un cambriolage (le séisme « explique » l'alarme, on n'a plus besoin du cambriolage). **Conditionner sur un effet commun crée une dépendance entre ses causes.**
(b) **Talent → Célébrité ← Beauté**. Dans la population générale, talent et beauté sont indépendants ; **si on ne regarde que les célébrités** (on contrôle le collider), ils deviennent dépendants : quelqu'un de peu beau a probablement beaucoup de talent, et inversement.
(c) Restreindre l'échantillon à ceux qui ont une certaine propriété (célébrité) revient à **conditionner sur un collider** : c'est la source des biais de sélection.

---

## Partie C : Confusion, Simpson, médiateur

**Q5.** Glaces et accidents de baignade. (a) Quel est le graphe et que dit la corrélation observée ? (b) Tu disposes de : $P(\text{TMP}=\text{chaud})=0{,}5$ ; $P(\text{ACC}\mid\text{ICE},\text{chaud})=0{,}10$ et $P(\text{ACC}\mid\text{ICE},\text{froid})=0{,}02$ **quelle que soit** la valeur de ICE ; $P(\text{chaud}\mid\text{ICE}=1)=0{,}9$ ; $P(\text{chaud}\mid\text{ICE}=0)=0{,}1$. Calcule $P(\text{ACC}\mid\text{ICE}=1)$, $P(\text{ACC}\mid\text{ICE}=0)$, puis $P(\text{ACC}\mid do(\text{ICE}=1))$ et $P(\text{ACC}\mid do(\text{ICE}=0))$. Conclus.

**Réponse.**
(a) **ICE ← Température → ACC** : la température est un **facteur de confusion** (fourche) ; elle fait augmenter à la fois les achats de glaces et les baignades. Sachant la température, ICE et ACC sont **indépendants**.
(b) **Observation** :

- $P(\text{ACC}\mid\text{ICE}=1)=0{,}9\times0{,}10+0{,}1\times0{,}02=\mathbf{0{,}092}$ ;
- $P(\text{ACC}\mid\text{ICE}=0)=0{,}1\times0{,}10+0{,}9\times0{,}02=\mathbf{0{,}028}$.

**Intervention** (formule d'ajustement, on pondère par $P(\text{TMP})$, pas par $P(\text{TMP}\mid\text{ICE})$) :

- $P(\text{ACC}\mid do(\text{ICE}=1))=0{,}5\times0{,}10+0{,}5\times0{,}02=\mathbf{0{,}06}$ ;
- $P(\text{ACC}\mid do(\text{ICE}=0))=0{,}5\times0{,}10+0{,}5\times0{,}02=\mathbf{0{,}06}$.

Conclusion : en observant, les glaces semblent multiplier le risque par 3 ($0{,}092$ contre $0{,}028$) ; en intervenant, **aucun effet** (effet causal moyen $=0$). $P(\text{ACC}\mid do(\text{ICE}))\ne P(\text{ACC}\mid\text{ICE})$ : **voir ≠ faire**.

---

**Q6.** Un traitement A est donné à 10 patients à petits calculs (9 succès) et 100 patients à gros calculs (30 succès). Un traitement B est donné à 100 patients à petits calculs (80 succès) et 10 patients à gros calculs (2 succès). (a) Calcule les taux de succès par strate et globaux. (b) Quel est le paradoxe ? (c) Quel est le facteur de confusion, comment l'identifier ? (d) Calcule les taux ajustés et conclus. (e) Pourquoi le résultat est-il contre-intuitif ?

**Réponse.**
(a) Petits calculs : A $=9/10=\mathbf{90\,\%}$, B $=80/100=\mathbf{80\,\%}$. Gros calculs : A $=30/100=\mathbf{30\,\%}$, B $=2/10=\mathbf{20\,\%}$. **Global** : A $=39/110\approx\mathbf{35{,}5\,\%}$, B $=82/110\approx\mathbf{74{,}5\,\%}$.
(b) **Paradoxe de Simpson** : A fait mieux dans **chaque** strate, mais B fait mieux **globalement**.
(c) La **taille des calculs** influence à la fois le traitement choisi et le succès : **B est donné à 91 % à des petits calculs** (100/110), plus faciles à traiter, A à 91 % à des gros calculs (100/110). Graphe : Condition (taille) → Traitement et Condition → Succès (confounder).
(d) Chaque strate a le même poids ($110/220=0{,}5$) : A $=0{,}5\times0{,}9+0{,}5\times0{,}3=\mathbf{0{,}60}$ ; B $=0{,}5\times0{,}8+0{,}5\times0{,}2=\mathbf{0{,}50}$. **A est meilleur.**
(e) On croit spontanément que si $a/b>c/d$ et $e/f>g/h$ alors $(a+e)/(b+f)>(c+g)/(d+h)$ : **faux** quand les groupes n'ont pas la même taille dans chaque bras. La cause profonde : on suppose le traitement **indépendant** de la condition, ce qui est faux ici.

---

**Q7.** Un médicament B est censé réduire la pression artérielle, cause de crise cardiaque. Résultats : pression basse : contrôle 1 crise / 19 sans crise, traitement 3 crises / 37 sans crise ; pression haute : contrôle 12 / 28, traitement 8 / 12. (a) Calcule les taux de crise par strate et globaux. (b) Quel est le graphe ? (c) Faut-il stratifier ? Quelle est la « règle d'or » ?

**Réponse.**
(a) Pression basse : contrôle $1/20=5\,\%$, traitement $3/40=7{,}5\,\%$. Pression haute : contrôle $12/40=30\,\%$, traitement $8/20=40\,\%$. **Global** : contrôle $13/60\approx21{,}7\,\%$, traitement $11/60\approx18{,}3\,\%$. Par strate le médicament semble **moins bon** ; globalement, **meilleur**.
(b) **Médicament → Pression artérielle → Crise cardiaque** (et éventuellement Médicament → Crise). La pression est un **médiateur**, pas un facteur de confusion.
(c) Non : conditionner sur le médiateur **bloque** une partie de l'effet qu'on veut mesurer (le médicament agit *via* la pression). Il faut regarder les résultats **agrégés** : le médicament est bon.
**Règle d'or** : **confounder** (cause commune) → **ajuster** ; **médiateur** (sur le chemin causal) → **ne pas ajuster** si on veut l'effet total (et ne pas ajuster non plus un collider). Il est faux de croire qu'« il vaut mieux contrôler le plus de variables possible ».

---

**Q8.** On étudie l'effet de l'exercice physique ($X$) sur la santé cardiaque ($Y$). Variables : l'âge, qui influence $X$ et $Y$ ; l'IMC, causé par $X$ et qui influence $Y$ ; la fatigue ressentie, causée à la fois par $X$ et par $Y$ ; le nombre de consultations médicales, causé par la fatigue. Pour chaque variable, donne son rôle (confounder, médiateur, collider, descendant) et dis s'il faut ajuster pour estimer l'effet total de $X$ sur $Y$.

**Réponse.**

- **Âge** : $X\leftarrow$ Âge $\rightarrow Y$ : **confounder** (cause commune) → **ajuster** (bloque le chemin back-door).
- **IMC** : $X\rightarrow$ IMC $\rightarrow Y$ : **médiateur** → **ne pas ajuster** (on bloquerait une partie de l'effet total).
- **Fatigue** : $X\rightarrow$ Fatigue $\leftarrow Y$ : **collider** → **ne pas ajuster** (contrôler un collider **ouvre** le chemin et crée une fausse dépendance).
- **Consultations** : **descendant du collider** → ne pas ajuster non plus (le contrôler ouvre **partiellement** le tuyau).

Ensemble d'ajustement : $\{\text{Âge}\}$.

---

**Q9.** Des données de mortalité selon le traitement (A ou B) et l'état du patient sont identiques dans deux scénarios. Scénario 1 : les médecins réservent le traitement A aux patients en état léger. Scénario 2 : le traitement B est rare, les patients attendent longtemps et leur état empire. Quelle décision à l'échelle d'un pays dans chaque cas ? Quelle leçon ?

**Réponse.**

- **Scénario 1** : l'état du patient est une **cause commune** du traitement et de la mortalité (confounder) → il faut regarder **par sous-population** : on choisit le **traitement B**.
- **Scénario 2** : l'état du patient est **causé par** le traitement (via l'attente) : c'est un **médiateur** ; B, même plus efficace une fois administré, cause globalement de pires états → il faut regarder **l'agrégat** : on choisit le **traitement A**.

Les **mêmes chiffres** conduisent à deux décisions **opposées**. Leçon : les relations causales apportent une information **absente des probabilités conditionnelles** ; les données seules ne suffisent pas ; connaître le graphe causal **change la décision**.

---

## Partie D : Estimer des effets causaux

**Q10.** (a) Définis les résultats potentiels, l'effet individuel (ITE) et l'effet moyen (ATE). (b) Quel est le problème fondamental de l'inférence causale ? (c) Six individus ont les résultats potentiels suivants (non observables en pratique) : $Y(1)=(1,1,0,1,0,1)$ et $Y(0)=(0,1,0,0,0,1)$. Calcule les ITE et l'ATE.

**Réponse.**
(a) $Y^{a=1}$ (resp. $Y^{a=0}$) : le résultat que l'individu **aurait** s'il était traité (resp. non traité). $A$ a un effet causal sur l'individu si $Y^{a=1}\ne Y^{a=0}$ ; **ITE** : $\tau_i=Y_i^{a=1}-Y_i^{a=0}$ ; **ATE** : $\tau=E[Y^{a=1}]-E[Y^{a=0}]$.
(b) On ne peut **jamais observer à la fois** $Y^{a=1}$ et $Y^{a=0}$ pour le **même** individu : un problème de **données manquantes** très particulier (la moitié de l'information est par construction inobservable, c'est le **contrefactuel**). D'où les deux questions du cours (diapo 60) : comment estimer un effet causal ? peut-on le faire à partir de données observationnelles, et sous quelles conditions ?
(c) ITE $=(1,0,0,1,0,0)$ ; ATE $=2/6=\mathbf{1/3}$ (vérification : $E[Y(1)]=4/6$, $E[Y(0)]=2/6$).

---

**Q11.** Reprends la question 10 : seuls les individus 1, 2 et 4 reçoivent le traitement. (a) Quels résultats sont observés ? (b) Calcule $E[Y\mid a=1]-E[Y\mid a=0]$. Est-ce égal à l'ATE ? Pourquoi ? (c) Comment un essai contrôlé randomisé résout-il le problème, et quelle est sa limite ?

**Réponse.**
(a) Individus traités (1, 2, 4) : on observe $Y(1)=1,1,1$. Non traités (3, 5, 6) : on observe $Y(0)=0,0,1$.
(b) $E[Y\mid a=1]=1$ ; $E[Y\mid a=0]=1/3$ ; différence $=\mathbf{2/3}\ne$ ATE $=1/3$. Ce n'est **pas** égal en général : si une variable $X$ explique à la fois $T$ et $Y$ (**confounder**), traités et non-traités ne sont **pas comparables** ; la différence observée mélange l'effet du traitement et celui de $X$. Ici, ceux qui ont été traités sont justement des individus qui bénéficient du traitement (1 et 4).
(c) **RCT** : on **assigne le traitement au hasard**, ce qui **supprime la flèche $X\to T$** : les groupes sont identiques en moyenne sur tous les aspects pertinents sauf le traitement (même non mesurés). Toute différence est imputable au traitement. Limite : pas toujours possible (**éthique**, **coût**) : on ne peut pas assigner aléatoirement des gens à « fumer ».

---

**Q12.** (a) Quelles sont les trois approches de la causalité évoquées (diapos 73-74) et leur statut dans le cours ? (b) Qu'est-ce qu'un diagramme causal ? Que signifie une flèche $X\to Y$ ? (c) Que signifie « intervenir » sur un nœud ?

**Réponse.**
(a) 1. **Essais contrôlés randomisés** (évoqués, pas toujours possibles) ; 2. **cadre des résultats potentiels** (Rubin, 1974), **non traité** ; 3. **graphes causaux** (Pearl) : **approche principale** du cours.
(b) Un **graphe orienté** (acyclique) dont les nœuds sont des variables ; $X\to Y$ signifie qu'**il existe une règle qui dit comment $Y$ changerait si $X$ changeait**. Souvent la structure du graphe suffit à répondre à des questions causales et contrefactuelles ; parfois il faut aussi des probabilités.
(c) **Intervenir** sur un nœud = **couper les flèches qui entrent** dans ce nœud : on décide de sa valeur nous-mêmes, elle ne dépend plus de ses causes habituelles.

---

## Partie E : Graphes et indépendances

**Q13.** Complète le tableau des trois briques de base et donne l'« image mentale » du tuyau. Que se passe-t-il si on contrôle un descendant ?

| Structure                       | Sans contrôle | Si on contrôle $B$ |
| ------------------------------- | -------------- | --------------------- |
| Chaîne $A\to B\to C$         | ?              | ?                     |
| Fourche $A\leftarrow B\to C$  | ?              | ?                     |
| Collider $A\to B\leftarrow C$ | ?              | ?                     |

**Réponse.**

| Structure                                        | Sans contrôle                                     | Si on contrôle $B$                                                  |
| ------------------------------------------------ | -------------------------------------------------- | ---------------------------------------------------------------------- |
| Chaîne (ex. Feu → Fumée → Alarme)            | $A$, $C$ **dépendants**                 | **indépendants** ($A\perp C\mid B$) : tuyau **bouché** |
| Fourche (ex. Âge → pointure, lecture)          | **dépendants**                              | **indépendants** : tuyau **bouclé**                      |
| Collider (ex. Talent → Célébrité ← Beauté) | **indépendants** (tuyau fermé par défaut) | **dépendants** : tuyau **ouvert**                         |

Fourche : les enfants à grands pieds lisent mieux… parce qu'ils sont plus âgés. Contrôler un **descendant** revient à contrôler **partiellement** : ferme partiellement le tuyau pour un médiateur, **ouvre partiellement** pour un collider. Piège : croire qu'il faut contrôler le plus de variables possible.

---

**Q14.** Peloton d'exécution : Ordre du tribunal $\to$ Capitaine $\to$ Soldat A et Soldat B $\to$ Mort du prisonnier. (a) Si le prisonnier est mort, peut-on conclure que l'ordre a été donné ? (b) Compare « on **observe** que A a tiré » et « on **fait** tirer A de lui-même » quant à l'information sur B. (c) Que se passe-t-il si on intervient pour que le capitaine ne donne pas l'ordre ? (d) Que répondrait-on à : « et si A n'avait pas tiré (alors qu'il a tiré) ? » (e) À quel niveau de l'échelle de Pearl correspond chaque question ?

**Réponse.**
(a) **Oui** (niveau associationnel) : les soldats ne tirent pas sans le capitaine, qui n'ordonne pas sans l'ordre du tribunal.
(b) **Observer** que A a tiré informe sur le capitaine, donc sur B : il a probablement tiré aussi. **Faire** tirer A (on coupe la flèche Capitaine $\to$ A) : l'action de A ne dit plus rien sur le capitaine ni sur B ; le prisonnier est mort ($D=$ True) mais on ne peut rien en déduire sur l'ordre.
(c) $do(C=\text{pas d'ordre})$ : ni A ni B ne tirent, le prisonnier est **vivant** ($D=$ False).
(d) **Contrefactuel** : on « rejoue » le monde ; B aurait obéi au capitaine et tiré quand même : le prisonnier serait **mort**.
(e) (a) et (b, observation) : niveau 1 **associationnel** (*voir*) ; (b, intervention) et (c) : niveau 2 **interventionnel** (*faire*) ; (d) : niveau 3 **contrefactuel** (*imaginer*).

---

**Q15.** Vaccination contre la variole (1 million d'enfants, 99 % vaccinés). Un vacciné a 1 chance sur 100 de réaction, mortelle 1 fois sur 100, et 0 chance d'avoir la variole. Un non-vacciné a 1 chance sur 50 d'avoir la variole, mortelle 1 fois sur 5. (a) Calcule les morts observées dues au vaccin et à la maladie. (b) Quelle conclusion naïve ? (c) Calcule ce qui se passerait avec un taux de vaccination nul. (d) Quelle leçon sur la causalité ?

**Réponse.**
(a) $990\,000\times\frac1{100}=9\,900$ réactions → $9\,900\times\frac1{100}=\mathbf{99}$ morts dues au vaccin. $10\,000\times\frac1{50}=200$ cas de variole → $200\times\frac15=\mathbf{40}$ morts dues à la maladie.
(b) Plus de morts par la vaccination (99) que par la maladie (40) : « vacciner est dangereux ».
(c) Sans vaccination : $10^6\times\frac1{50}=20\,000$ cas → $20\,000\times\frac15=\mathbf{4\,000}$ morts, soit $4\,000-(99+40)=\mathbf{3\,861}$ morts de plus.
(d) Les chiffres observés ne reflètent pas un monde où personne n'est vaccinié : le bénéfice du vaccin est **invisible dans les données** et n'apparaît que dans le **scénario contrefactuel**. Il faut un modèle causal pour poser la bonne question.

---

## Partie F : Critère de la porte dérobée (back-door)

**Q16.** (a) Différence entre $P(Y\mid X=x)$ et $P(Y\mid do(X=x))$ ? (b) Que fait l'opérateur $do$ au graphe, et quel autre mécanisme a le même effet ? (c) Définis un chemin back-door. Pourquoi pose-t-il problème ? (d) Quand un chemin est-il bloqué ? (e) Énonce le critère back-door et la formule d'ajustement.

**Réponse.**
(a) $P(Y\mid X=x)$ : probabilité de $Y$ **parmi les individus pour lesquels on a observé** $X=x$ (observation passive). $P(Y\mid do(X=x))$ : probabilité de $Y$ **si l'on force** $X$ à $x$ (intervention). Elles diffèrent dès qu'il y a de la confusion.
(b) $do$ **efface toutes les flèches qui entrent dans $X$** ; la **randomisation** a exactement le même effet.
(c) **Chemin back-door** : tout chemin de $X$ à $Y$ qui **commence par une flèche pointant vers $X$**. Il fait circuler de l'information de $X$ vers $Y$ par un chemin **non causal** (corrélation spurieuse).
(d) Un chemin est bloqué si on contrôle un **médiateur ou une cause commune** (chaîne ou fourche) qu'il contient, ou s'il contient un **collider non contrôlé** (ni ses descendants).
(e) Si un ensemble $Z$ **bloque tous les chemins back-door** de $X$ à $Y$ (sans contenir de descendant de $X$), $X$ et $Y$ sont déconfondus et $P(y\mid do(x))=\sum_zP(y\mid x,z)P(z)$.

---

**Q17.** Graphe : $A\to X$, $A\to B$, $C\to B$, $C\to Y$, $D\to X$, $D\to Y$, $X\to M$, $M\to Y$. (a) Liste les chemins back-door de $X$ à $Y$. (b) Lesquels sont déjà bloqués ? (c) Donne un ensemble d'ajustement minimal. (d) Pourquoi ne faut-il pas contrôler $B$ ni $M$ ? (e) Si $D$ est inobservable, que faire ? Écris la formule.

**Réponse.**
(a) Deux chemins commençant par une flèche entrant dans $X$ : (1) $X\leftarrow D\rightarrow Y$ ; (2) $X\leftarrow A\rightarrow B\leftarrow C\rightarrow Y$.
(b) (2) est **déjà bloqué** par le **collider $B$** (non contrôlé) ; (1) est **ouvert** (fourche en $D$).
(c) $\{D\}$ (aussi valides : $\{D,A\}$, $\{D,C\}$).
(d) Contrôler **$B$** **ouvrirait** le chemin (2) (il faudrait alors contrôler aussi $A$ ou $C$) ; contrôler **$M$**, **médiateur** de $X\to M\to Y$, **bloquerait l'effet causal** qu'on veut mesurer.
(e) Le back-door est inapplicable (on ne peut pas contrôler $D$), mais $M$ est un médiateur qui intercepte tout chemin causal de $X$ à $Y$, et les chemins back-door de $X$ vers $M$ sont bloqués par des colliders : on applique le **critère front-door** :
$P(y\mid do(x))=\sum_mP(m\mid x)\sum_{x'}P(y\mid x',m)P(x')$.

---

**Q18.** (a) Jeu 5 de la fiche : chemins (1) $X\leftarrow A\rightarrow B\leftarrow C\rightarrow Y$ et (2) $X\leftarrow B\leftarrow C\rightarrow Y$. Quel est le piège ? Que faut-il contrôler ? (b) Jeu 6 (tabagisme/asthme) : que conclut le cours ? (c) Donne la méthode générale en quatre étapes.

**Réponse.**
(a) Pour bloquer (2) on contrôle $B$ ; mais $B$ est un **collider** sur (1), donc contrôler $B$ **ouvre** (1). Il faut alors **aussi contrôler $A$ (ou $C$)**. Le piège : en bloquant un chemin on peut **en ouvrir un autre** ; il faut raisonner sur **l'ensemble** des chemins.
(b) Il faut contrôler **A et B** (ou A et C, mais C est inobservable, donc A et B), ainsi que **E, F, G** (confounders) ; mais **pas D** (le cours l'indique sans détail).
(c) 1. Lister les chemins entre $X$ et $Y$ qui **commencent par une flèche vers $X$** ; 2. repérer ceux **déjà bloqués** par un collider non contrôlé ; 3. pour les autres, choisir des variables **observables** à contrôler (chaîne/fourche) ; 4. vérifier qu'on **n'ouvre pas** de chemin via un collider (ou un de ses descendants) et qu'on ne contrôle pas un **médiateur** de l'effet total.

---

## Partie G : Le paradoxe de Lord

**Q19.** Version 1 (régime et sexe). Une école mesure le poids en septembre ($W_i$) et en juin ($W_f$) et veut savoir si le régime a des effets différents sur filles et garçons. Quels sont les arguments des deux statisticiens ? Quel est le graphe causal ? Qui a raison et pourquoi ?

**Réponse.**

- **Statisticien 1** : filles comme garçons ont le même poids moyen en juin qu'en septembre (nuages symétriques autour de $W_f=W_i$) → **pas d'effet différencié**.
- **Statisticien 2** : $W_f$ dépend fortement de $W_i$ ; à poids initial fixé, l'ellipse des garçons est plus haute que celle des filles → **il y a un effet différencié**.

Graphe : le sexe $S$ est cause de $W_i$ **et** de $W_f$ ; $W_i$ influence $W_f$ ; la variable d'intérêt est le **gain** $Y=W_f-W_i$. **Aucun chemin back-door** entre $S$ et $Y$ : rien à contrôler, et contrôler $W_i$ serait même une erreur ($W_i$ est un **médiateur** de $S$ vers le gain). → **Le statisticien 1 a raison.**

---

**Q20.** Version 2 : on étudie l'effet du **régime $D$ lui-même** (cantines A et B), et les élèves les plus lourds au départ mangent à la cantine B. Qui a raison cette fois ? Que change-t-il dans le graphe ? Quelle conclusion générale de Lord ?

**Réponse.**
Les deux statisticiens refont **exactement le même raisonnement**, mais le graphe a changé : $W_i$ **influence le régime $D$** (les lourds vont en B). $W_i$ est maintenant un **confounder** de $D$ et $W_f$ (et plus un médiateur) : il y a un **chemin back-door** entre $D$ et $Y$, il faut **contrôler $W_i$**. → **Le statisticien 2 a raison.**
Conclusion : la **même structure statistique** (mêmes ellipses) avec **deux graphes causaux différents** donne **deux conclusions opposées**. « Contrôler le poids initial » est correct quand $W_i$ est un confounder, incorrect quand c'est un médiateur. Les nuages de points seuls sont **ambigus** ; il faut des **hypothèses sur le graphe**.

---

## Partie H : Le do-calcul

**Q21.** (a) Quel est l'objectif du do-calcul ? (b) Que désignent $G_{\overline X}$ et $G_{\underline X}$ ? (c) Énonce les trois règles en français, avec une phrase d'intuition.

**Réponse.**
(a) Transformer une expression contenant $do$ en une expression **sans $do$**, ne contenant que des variables **observables**, pour obtenir un estimateur calculable (jeu de règles de réécriture).
(b) $G_{\overline X}$ : $G$ sans les flèches **entrant** dans $X$ ; $G_{\underline X}$ : $G$ sans les flèches **sortant** de $X$.
(c) **Règle 1** (ajouter/supprimer une observation) : $P(y\mid do(x),z,w)=P(y\mid do(x),z)$ si $Z$ bloque tous les chemins de $W$ à $Y$ dans le graphe sans flèches entrant dans $X$ : on peut ignorer une variable observée indépendante de $Y$ compte tenu de ce qu'on sait (ex. Feu, Fumée, Alarme).
**Règle 2** (échanger action et observation) : si $Z$ satisfait le critère back-door, $P(y\mid do(x),z)=P(y\mid x,z)$ : bien ajusté, **observer revient à intervenir**.
**Règle 3** (supprimer une action) : s'il n'y a **aucun chemin causal** de $X$ à $Y$, $P(y\mid do(x))=P(y)$ : si $X$ ne peut pas influencer $Y$, forcer $X$ ne change rien.

---

**Q22.** Pour chaque graphe, donne l'expression de $P(y\mid do(x))$ sans $do$ ou explique ce qu'il faut faire : (a) $X\to Y$ ; (b) $X\leftarrow Z\rightarrow Y$ avec $X\to Y$ ; (c) $X\to Z\leftarrow Y$ avec $X\to Y$ ; (d) $X\leftarrow Z\rightarrow Y$ **sans** flèche $X\to Y$.

**Réponse.**
(a) Pas de chemin non causal : $P(y\mid do(x))=\mathbf{P(y\mid x)}$ (règle 2 dans $G_{\underline X}$).
(b) Confounder $Z$ : $P(y\mid do(x))=\sum_zP(y\mid do(x),z)P(z)=\mathbf{\sum_zP(y\mid x,z)P(z)}$ (règle 2) : **formule d'ajustement**, la même idée que la moyenne pondérée du paradoxe de Simpson.
(c) $Z$ est un **collider** : on ne le contrôle **pas** ; après suppression des flèches sortant de $X$, $X$ et $Y$ sont indépendants, donc $P(y\mid do(x))=\mathbf{P(y\mid x)}$.
(d) Aucun chemin causal de $X$ à $Y$ : règle 3, $P(y\mid do(x))=\mathbf{P(y)}$, alors que $P(y\mid x)\ne P(y)$ en général (corrélation purement spurieuse).

---

**Q23.** Front-door numérique. Variables binaires $X\to Z\to Y$ avec un confounder non observé de $X$ et $Y$. On a $P(x'=1)=P(x'=0)=0{,}5$ ; $P(Z=1\mid X=1)=0{,}9$, $P(Z=1\mid X=0)=0{,}1$ ; $P(Y=1\mid x',z)$ : $(0,0)\to0{,}1$ ; $(1,0)\to0{,}2$ ; $(0,1)\to0{,}5$ ; $(1,1)\to0{,}7$. Calcule $P(Y=1\mid do(X=1))$, $P(Y=1\mid do(X=0))$ et l'effet causal moyen.

**Réponse.**
Formule : $P(y\mid do(x))=\sum_zP(z\mid x)\sum_{x'}P(y\mid x',z)P(x')$.
Terme interne : $T(z)=\sum_{x'}P(Y=1\mid x',z)P(x')$ : $T(0)=0{,}5\times0{,}1+0{,}5\times0{,}2=\mathbf{0{,}15}$ ; $T(1)=0{,}5\times0{,}5+0{,}5\times0{,}7=\mathbf{0{,}60}$.

- $do(X=1)$ : $0{,}1\times0{,}15+0{,}9\times0{,}60=0{,}015+0{,}54=\mathbf{0{,}555}$.
- $do(X=0)$ : $0{,}9\times0{,}15+0{,}1\times0{,}60=0{,}135+0{,}06=\mathbf{0{,}195}$.

Effet causal moyen : $0{,}555-0{,}195=\mathbf{0{,}36}$.
Tous les termes sont **observables** : on a éliminé le $do$ **malgré** le confounder non mesuré.

---

**Q24.** Exemple du GPS : $X$ = usage du GPS, $Z$ = volume de l'hippocampe, $Y$ = mémoire spatiale, $U$ = motivation (non observée). Décris la démarche en deux temps et la formule finale. Pourquoi peut-on identifier l'effet malgré $U$ ?

**Réponse.**

1. **Effet de $X$ sur $Z$** : il existe un chemin back-door $X\leftarrow U\rightarrow Y\leftarrow Z$, mais il est **déjà bloqué par le collider** $U\to Y\leftarrow Z$ : $P(z\mid do(x))=P(z\mid x)$.
2. **Effet de $Z$ sur $Y$** : chemin back-door ouvert $Z\leftarrow X\leftarrow U\rightarrow Y$ ; $U$ inobservé, mais on peut **contrôler $X$** (observable) : $P(y\mid do(z))=\sum_xP(y\mid Z=z,X=x)P(x)$.
3. **On combine** : $P(y\mid do(x))=\sum_zP(z\mid x)\sum_{x'}P(y\mid x',z)P(x')$.

On peut identifier l'effet **grâce au médiateur $Z$** : l'effet causal est décomposé en deux effets chacun identifiable. (Critère **front-door**.)

---

**Q25.** Que dit le théorème de complétude du do-calcul ? Quel intérêt pratique ? Quelles sont les méthodes pour éliminer les $do$ listées dans le cours ?

**Réponse.**
Chaque fois qu'un effet causal est estimable à partir des données, **une séquence d'applications des trois règles** permet d'éliminer le $do$ : on peut estimer des effets causaux à partir de données **observationnelles seules** (quand c'est possible). Le do-calcul est **complet** : s'il échoue, **aucune méthode** ne peut identifier l'effet sans information supplémentaire (par exemple une expérimentation).
Méthodes (diapo 87) : essais randomisés ; **critère back-door** ; **critère front-door** ; **do-calcul** ; variables instrumentales (citées, non détaillées).

---

## Partie I : Science de l'intervention et échelle de Pearl

**Q26.** « Que se passera-t-il pour les ventes de fil dentaire si on double le prix du dentifrice ? » Pourquoi les données passées ne suffisent-elles pas, même si le prix a déjà doublé dans le passé ? Quelle est la question centrale qui en découle ?

**Réponse.**
C'est une **intervention**, qui appelle un **type de connaissance absent des données**. Dans le passé, le prix a pu doubler à cause d'une **pénurie générale** (partout), alors qu'on veut savoir l'effet d'une **décision locale** (dans notre magasin seulement) : les données reflètent un **mécanisme différent** de celui qu'on veut tester.
Question centrale : sous quelles conditions des **observations passives** seules permettent-elles de répondre à des questions sur des **interventions** ?

---

**Q27.** (a) La vision des philosophes dit « $X$ cause $Y$ si $X$ augmente la probabilité de $Y$ ». Pourquoi est-ce trompeur ? (b) Que dit la citation de Pearl (diapo 146) ? (c) Écris la formule d'ajustement sur les parents de $X$ et donne l'inégalité de la fin du cours pour les glaces/accidents.

**Réponse.**
(a) Dire que « $X$ augmente la probabilité de $Y$ » n'est **pas** la même chose que $P(Y\mid X)>P(Y)$ (qui signifie seulement : *si je vois $X$*, la probabilité de $Y$ augmente). Cette hausse peut venir d'une **causalité inversée** ($Y$ cause $X$) ou d'une **cause commune** $Z$.
(b) Les probabilités encodent nos croyances sur un monde **statique** ; la causalité dit **si et comment les probabilités changent quand le monde change**, par **intervention** ou par **acte d'imagination**.
(c) $P(Y=y\mid do(X=x))=\sum_zP(Y=y\mid X=x,Pa=z)\,P(Pa=z)$ où $Pa$ désigne les **parents de $X$**. Pour les glaces : $P(\text{ACC}\mid do(\text{ICE}))=\sum_{tmp}P(\text{ACC}\mid\text{ICE},\text{TMP}=tmp)P(\text{TMP}=tmp)\ne P(\text{ACC}\mid\text{ICE})$ : **voir** que les gens achètent des glaces → les accidents augmentent ; **faire** acheter des glaces (distribution gratuite) → rien ne change.

---

**Q28.** (a) Décris les trois niveaux de l'échelle de Pearl (activité, notation, question type, lien avec l'apprentissage). (b) Place chaque question : (i) « Les patients qui prennent ce médicament guérissent-ils plus souvent ? » ; (ii) « Si on impose ce médicament à tous, quel sera le taux de guérison ? » ; (iii) « Ce patient, mort après le traitement, serait-il vivant sans traitement ? » ; (iv) « Quel est le salaire moyen des diplômés ? » ; (v) « Que se passerait-il si on supprimait cette taxe ? ». (c) Quelle est l'idée-clé sur la montée d'un niveau ?

**Réponse.**
(a)

| Niveau | Nom                       | Activité    | Notation             | Question                                                                           | Lien avec l'apprentissage                 |
| ------ | ------------------------- | ------------ | -------------------- | ---------------------------------------------------------------------------------- | ----------------------------------------- |
| 1      | **Associationnel**  | *Voir*     | $P(y\mid x)$       | « Qu'est-ce que c'est ? Comment voir $X$ change-t-il ma croyance sur $Y$ ? » | apprentissage supervisé / non supervisé |
| 2      | **Interventionnel** | *Faire*    | $P(y\mid do(x),c)$ | « Et si je fais $X$ ? »                                                        | apprentissage par renforcement            |
| 3      | **Contrefactuel**   | *Imaginer* | $P(y_x\mid x',y')$ | « Pourquoi ? Et si j'avais agi autrement ? »                                     | –                                        |

$y_x$ : la valeur que $Y$ aurait prise si $X$ avait été fixé à $x$, alors qu'on a observé $x'$ et $y'$ : un monde **différent du monde réel**.
(b) (i) niveau **1** ; (ii) niveau **2** ; (iii) niveau **3** ; (iv) niveau **1** ; (v) niveau **2**.
(c) On **ne peut pas** répondre à une question d'un niveau supérieur avec uniquement des données (ou un modèle) du niveau inférieur ; pour monter, il faut des **hypothèses causales** (le graphe).

---

## Partie J : Synthèse et ponts avec les autres cours

**Q29.** Associe chaque exemple du cours à sa structure causale et dis s'il faut conditionner : (a) Monty Hall ; (b) alarme ; (c) glaces/noyades ; (d) Simpson (calculs rénaux) ; (e) médicament/pression artérielle ; (f) COVID scénario 1 ; (g) COVID scénario 2 ; (h) Lord (sexe) ; (i) Lord (régime).

**Réponse.**

| Exemple                  | Structure                                              | Conditionner ?                                                       |
| ------------------------ | ------------------------------------------------------ | -------------------------------------------------------------------- |
| (a) Monty Hall           | **collider** : choix → porte ouverte ← voiture | **Non** : conditionner crée la dépendance (c'est l'illusion) |
| (b) Alarme               | **collider** : séisme → alarme ← cambriolage  | Non (explaining away)                                                |
| (c) Glaces               | **fourche / confounder** : température          | **Oui**, sur la température                                   |
| (d) Simpson calculs      | **confounder** : taille des calculs              | **Oui** (ajuster par strate)                                   |
| (e) Pression artérielle | **médiateur**                                   | **Non** (agrégat)                                             |
| (f) COVID sc. 1          | **confounder** : état léger → traitement A    | **Oui** (par sous-population → B)                             |
| (g) COVID sc. 2          | **médiateur** : traitement → état             | **Non** (agrégat → A)                                        |
| (h) Lord (sexe)          | $W_i$ **médiateur** (pas de back-door)        | **Non** : statisticien 1                                       |
| (i) Lord (régime)       | $W_i$ **confounder**                           | **Oui** : statisticien 2                                       |

---

**Q30.** (Rédaction.) Retrace la logique d'enchaînement du cours : de la corrélation $\ne$ causalité à l'échelle de Pearl, en montrant à chaque étape le problème et la solution.

**Réponse.**

1. **Motivation** : prédire n'est pas expliquer ni agir ; cinq raisons de s'intéresser à la causalité.
2. **Corrélation $\ne$ causalité (et inversement)** : liens non linéaires non détectés par Pearson ; corrélations sans causalité : collider (Monty Hall, alarme), confounder (glaces), Simpson, médiateur.
3. **Problème fondamental** : on n'observe jamais $Y^{a=1}$ et $Y^{a=0}$ pour un même individu ; ATE ; la simple différence de moyennes est biaisée en présence de confounders ; **RCT** (mais pas toujours possible).
4. **Graphes causaux (Pearl)** : trois briques (chaîne, fourche, collider) pour lire les indépendances ; l'**intervention** coupe les flèches entrantes.
5. **Critère back-door** : bloquer tous les chemins back-door sans ouvrir de collider ni contrôler un médiateur ; **formule d'ajustement**.
6. **Lord** : mêmes données, deux graphes, deux conclusions : il faut des hypothèses causales.
7. **Do-calcul** : trois règles pour éliminer le $do$ ; **front-door** (même avec confounder non mesuré) ; **complétude**.
8. **Échelle de Pearl** : voir, faire, imaginer ; les contrefactuels et l'apprentissage de la structure causale sont **hors du cours**.

---

**Q31.** (Pont avec le cours 9 sur l'IA explicable.) (a) Quel lien entre l'effet Clever Hans et la causalité ? (b) Pourquoi SHAP, qui « explique le modèle et non le monde », ne donne-t-il pas d'information causale ? (c) Pourquoi la causalité est-elle présentée comme « le graal » de l'explicabilité (robustesse, manipulation, transférabilité) ?

**Réponse.**
(a) Clever Hans : un modèle exploite une **corrélation accidentelle (spurieuse)** très prédictive ; c'est une relation **non causale** qui disparaît quand le contexte change. La causalité fournit le vocabulaire (confounder, fourche) pour décrire pourquoi ce lien n'est pas fiable.
(b) Les valeurs de Shapley répartissent l'écart entre prédiction et moyenne **pour le modèle** ; si le modèle a appris une corrélation absurde, SHAP lui attribue de l'importance : c'est du **niveau 1 (associationnel)** appliqué au modèle, pas une réponse à « que se passe-t-il si j'agis sur cette variable ? ».
(c) Une explication **causale** est **robuste** (valable quand le contexte change : raison 5), **non manipulable** (un système fondé sur des facteurs causaux ne se trompe pas en ouvrant 3 cartes de crédit) et **généralisable**.

---

**Q32.** (Pont avec les cours précédents.) (a) À quel niveau de l'échelle de Pearl se situent l'apprentissage supervisé (régression, arbres, SVM, ensembles) et l'apprentissage par renforcement ? (b) Pourquoi un modèle très performant sur des données observationnelles peut-il être inutile pour décider d'une action ? (c) Pourquoi une p-value très petite ne prouve-t-elle pas une relation causale ?

**Réponse.**
(a) L'apprentissage supervisé et non supervisé : niveau 1 **associationnel** ($P(y\mid x)$) ; l'apprentissage par **renforcement** : niveau 2 **interventionnel**.
(b) Un modèle appris sur des observations capture $P(y\mid x)$ ; décider d'une action demande $P(y\mid do(x))$, qui diffère dès qu'il y a confusion (glaces/accidents, voir ≠ faire). Un bon score ne garantit pas que le modèle ait appris le bon mécanisme.
(c) Une p-value petite signifie qu'une différence est **peu probable par hasard**, pas qu'elle est **causale** : une corrélation spurieuse (confounder, collider, sélection) peut être très significative. Le test de significativité répond à « est-ce du bruit ? », pas à « pourquoi ? ». (Comme en cours 6 : significativité statistique ≠ importance pratique ; ici, significativité ≠ causalité.)

---

## Pièges fréquents à vérifier avant le partiel

| Piège                                   | À retenir                                                                                                                       |
| ---------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------- |
| Corrélation nulle                       | N'implique **pas** l'absence de causalité (Pearson = linéaire)                                                           |
| « Contrôler un maximum de variables » | Faux : contrôler un **collider** ouvre un chemin, contrôler un **médiateur** bloque l'effet total                 |
| Confounder / médiateur / collider       | Cause commune → **ajuster** ; sur le chemin causal → **ne pas ajuster** ; effet commun → **ne pas ajuster** |
| Chaîne, fourche, collider               | Chaîne et fourche : contrôler $B$ **bloque** ; collider : contrôler $B$ **ouvre**                             |
| $P(y\mid x)$ vs $P(y\mid do(x))$     | Différents dès qu'il y a confusion ; $do$ efface les flèches **entrantes**                                            |
| Chemin back-door                         | Commence par une flèche **vers** $X$                                                                                    |
| Formule d'ajustement                     | Pondérer par $P(z)$, **pas** par $P(z\mid x)$                                                                         |
| Simpson                                  | Même chiffres, décision selon le **graphe** (confounder → stratifier, médiateur → agréger)                           |
| Lord                                     | Même structure statistique, deux graphes : $W_i$ médiateur vs confounder                                                     |
| Do-calcul                                | Règle 1 : ignorer une observation ; règle 2 : action ↔ observation ; règle 3 : supprimer une action                          |
| Front-door                               | Marche avec un confounder **non mesuré**, via un médiateur                                                               |
| Échelle de Pearl                        | Voir (1) < faire (2) < imaginer (3) ; on ne monte qu'avec des hypothèses causales                                               |
| Hors cours                               | Rubin, variables instrumentales, contrefactuels, apprentissage causal : cités, non traités                                     |
