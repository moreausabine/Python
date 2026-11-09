# Cours 9 – Introduction à l'IA explicable

> Notes explicatives rédigées à partir du cours d'Antoine Cornuéjols (AgroParisTech – INRAE, « IA et explicabilité », 124 diapositives).
> Les numéros entre parenthèses (ex. *diapo 63*) renvoient aux diapositives du cours.
> Les passages marqués **Complément** ne figurent pas sur les diapos : je les ajoute pour rendre le raisonnement plus clair.

---

## Sommaire

1. [Motivation : pourquoi veut-on des explications ?](#1-motivation--pourquoi-veut-on-des-explications-)
2. [Les familles de méthodes](#2-les-familles-de-méthodes)
3. [Interprétabilité par conception (*by design*)](#3-interprétabilité-par-conception-by-design)
4. [Interprétation post-hoc](#4-interprétation-post-hoc)
   - 4.a [Méthodes indépendantes du modèle : LIME et Shapley](#4a-méthodes-indépendantes-du-modèle-model-agnostic)
   - 4.b [Méthodes dépendantes du modèle : réseaux de neurones en vision](#4b-méthodes-dépendantes-du-modèle-model-specific)
5. [Quel avenir pour l&#39;IA explicable ?](#5-quel-avenir-pour-lia-explicable-)
6. [Fiche de révision](#6-fiche-de-révision)
7. [Questions pour tester sa compréhension](#7-questions-pour-tester-sa-compréhension)
8. [Glossaire](#8-glossaire)

---

## 1. Motivation : pourquoi veut-on des explications ?

### 1.1 Un problème qui n'est pas nouveau : les systèmes experts (diapos 3-7)

L'explicabilité n'est pas née avec le deep learning. Dès les années 1970, les **systèmes experts** comme **MYCIN** (diagnostic et prescription d'antibiotiques) savaient déjà « s'expliquer ».

**Comment un système expert explique-t-il ses décisions ?**
Un système expert fonctionne avec des **règles écrites par des humains** (« SI infection de type X ET patient âgé de moins de 8 ans ALORS pas de tétracycline »). Pour expliquer sa réponse, il suffit de **rejouer la chaîne de règles** qu'il a utilisée. L'explication est donc « gratuite » : elle est une conséquence directe de la façon dont le système est construit.

Exemple de la diapo 5 : *« Pourquoi ne faut-il pas prescrire de tétracycline à un enfant de moins de 8 ans ? »*
La réponse s'appuie sur une chaîne de connaissances justificatives :

```
Le médicament se dépose sur les os en développement
        ↓
Noircissement définitif des dents
        ↓
Coloration socialement indésirable
        ↓
=> Ne pas administrer de tétracycline aux enfants de moins de 8 ans
```

Remarquez que cette explication fait appel à des **relations de causalité** et à la notion d'**effets secondaires indésirables**. C'est un point qui reviendra tout au long du cours : une bonne explication est souvent une explication **causale**.

### 1.2 Deux niveaux d'explication (diapo 7)

| Type d'explication         | Contenu                                                                              | Pour qui ?                                           |
| -------------------------- | ------------------------------------------------------------------------------------ | ---------------------------------------------------- |
| **De « surface »** | Décrit *comment le système fonctionne* (quelles règles se sont déclenchées)   | L'expert en IA, qui vérifie le fonctionnement       |
| **« Profonde »**   | Décrit *pourquoi c'est vrai dans le domaine* (les raisons de fond, la causalité) | L'expert du domaine (médecin…) et les utilisateurs |

**Idée à retenir :** expliquer « ce que la machine a calculé » n'est pas la même chose qu'expliquer « pourquoi c'est la bonne décision ». Les explications profondes sont **difficiles à découvrir et à formuler**.

### 1.3 L'arrivée des réseaux de neurones profonds (diapos 8-12)

Les **réseaux de neurones profonds** sont des réseaux avec un très grand nombre de couches (parfois plusieurs centaines) et de paramètres (de 10⁷ à 10⁹). Impossible de « lire » un tel modèle comme on lit une liste de règles : c'est la fameuse **boîte noire**.

Le cours donne plusieurs illustrations de ce malaise :

- **Illusions d'optique** (diapo 10, article Grad-CAM) : un réseau peut se tromper face à une image, et on voudrait comprendre *sur quoi* il s'est appuyé.
- **Médecine** (diapo 11) : un système qui aide au diagnostic doit pouvoir justifier ses résultats auprès du médecin.
- **AlphaGo** (diapo 12) : le programme joue comme un « joueur extraterrestre », avec un jeu stupéfiant qui a révolutionné la manière de jouer et créé une effervescence dans les écoles de go. Ici l'explicabilité a un autre intérêt : **la découverte scientifique**. Si la machine a trouvé quelque chose que les humains ignoraient, on voudrait *apprendre* d'elle, donc comprendre ce qu'elle a compris.

### 1.4 Un principe de base : le ML ne crée pas d'information (diapos 13-20)

> **Le Machine Learning « reformule » simplement ce qu'on lui a donné en entrée.**
> C'est une sorte de **théorème de conservation** : aucune information n'est « ajoutée ». Ce que le modèle sait vient soit des **données**, soit des **connaissances a priori**.

Il existe un **compromis** :

```
Peu de données   + beaucoup de connaissances a priori
Beaucoup de données + peu de connaissances a priori
```

**Analogie :** imaginez un élève qui prépare un examen. Soit il a peu d'exemples d'exercices mais un très bon cours (connaissances a priori), soit il a des milliers d'exercices corrigés mais pas de cours. Dans les deux cas, ce qu'il sait vient de ce qu'on lui a fourni.

**Où se cachent les connaissances a priori ?** Le cours en distingue deux endroits :

**a) Dans les données (diapos 16-17)**

- le dispositif expérimental (comment les données ont été recueillies) ;
- le choix des descripteurs (*features*) ;
- l'enrichissement avec des ontologies ;
- la normalisation des valeurs ;
- le traitement des valeurs manquantes ;
- éventuellement des exemples ajoutés (en pensant à des invariances).

Le titre de la diapo 16, « **Biais sur les données** », rappelle que chacun de ces choix est aussi une source possible de biais.

**b) Dans l'algorithme d'apprentissage (diapos 18-20)** : c'est le **biais de représentation**.

- Exemple 1 : chercher des hypothèses **linéaires parcimonieuses** (*sparse*), c'est-à-dire avec peu de coefficients non nuls. On favorise ainsi les modèles simples.
- Exemple 2 : les **réseaux convolutifs (CNN)**. Leur architecture elle-même est une connaissance injectée : on suppose que l'information utile est **locale** (les pixels voisins sont liés) et que la même forme peut apparaître **n'importe où** dans l'image.

**Pourquoi c'est important pour l'explicabilité ?** Parce que cela nous dit que *comprendre un modèle* passe aussi par *comprendre ce qu'on lui a injecté* (données + biais).

### 1.5 Quelle connaissance peut-on extraire d'un modèle ? (diapos 22-26)

L'apprentissage produit une **hypothèse**, c'est-à-dire une **fonction de décision**. Exemples (diapo 23) : recommander un film, reconnaître des caractères ou des panneaux, évaluer une position au go, prédire un risque de criminalité, accorder un prêt, décider d'embaucher quelqu'un…

Deux situations (diapo 24) :

- la fonction de décision est utilisée **seule** ;
- ou elle est **intégrée dans un système d'inférence plus large** (par exemple un raisonnement juridique) : alors on doit pouvoir **questionner d'où vient la décision**.

**Interprétabilité de la fonction de décision (diapo 25)** : tous les modèles ne se valent pas.

```
Facile à interpréter   ←――――――――――――――――――――――→   Difficile à interpréter
Arbres de décision > fonctions linéaires > forêts aléatoires > SVM, réseaux de neurones
                                                      (nécessitent une analyse difficile)
```

**Interprétabilité du processus d'apprentissage (diapo 26)** :

- **Analyse de sensibilité** : « si je change cette valeur d'entrée, que se passe-t-il en sortie ? »
- **Explanation-Based Learning** : apprentissage guidé par des explications.

### 1.6 Quand a-t-on besoin d'interprétabilité ? (diapos 27-41)

**Quand elle n'est PAS nécessaire (diapos 27-28) :**

- quand le **risque est faible** (recommandation d'un film) ;
- quand il existe de **bonnes garanties de performance** (reconnaissance de caractères).

**Quand elle EST nécessaire (diapos 29-41), quatre grandes raisons :**

**① Décisions à haut risque** : opération chirurgicale, arrêt d'une centrale nucléaire, véhicule autonome. On ne délègue pas ces décisions à une boîte noire sans pouvoir les justifier.

**② Satisfaire la curiosité (c'est le but de la science)** : expliquer des résultats surprenants, des cas où aucune explication simple n'existe, ou quand la fonction de décision doit être intégrée dans une théorie du domaine.

**③ Déboguer** : comprendre pourquoi une décision est fausse.

- *Raisonnement contrefactuel* : « Qu'est-ce qui aurait dû être différent pour que la décision change ? »
- Exemple du vélo : si un vélo est reconnu parce qu'il a deux roues, que se passe-t-il quand une roue est cachée derrière des sacoches ?
- Exemple de biais : pourquoi le système semble-t-il favoriser un genre ?

**④ Exiger des systèmes prédictifs de meilleure qualité** : un système interprétable peut être **manipulé**. Si l'on sait qu'un prêt est accordé à partir de plus de 2 cartes de crédit, il suffit d'ouvrir 3 cartes. Pour éviter la manipulation, le système doit s'appuyer sur des **facteurs causaux** et non sur de simples corrélations.

### 1.7 L'effet « Clever Hans » (diapos 34-39)

**L'histoire.** Hans le Malin était un cheval allemand du début du XXᵉ siècle, réputé capable de calculer, lire et même résoudre des problèmes d'harmonie musicale en tapant du sabot. Le psychologue Oskar Pfungst a découvert que Hans n'avait **aucune** capacité de calcul : il **lisait des signaux corporels** émis inconsciemment par son maître ou par les personnes qui l'interrogeaient.

**Le parallèle en apprentissage supervisé.** Un modèle peut donner la bonne réponse **pour de mauvaises raisons**, en exploitant des **corrélations accidentelles (*spurious*) mais très prédictives** dans la base d'apprentissage.

Exemple du cours (diapos 38-39) : sur les 9963 images de chevaux de la base Pascal VOC, le réseau reconnaît « un cheval »… mais cela marche *sauf quand* la corrélation accidentelle disparaît.

**Complément :** dans l'étude de référence sur ce cas (Lapuschkin et al.), beaucoup d'images de chevaux contenaient une petite mention de copyright dans un coin de l'image. Le réseau avait appris à détecter cette mention, pas le cheval. Il suffisait de coller la mention sur une voiture pour qu'elle soit « reconnue comme un cheval ».

**Leçon :** un bon score de performance ne prouve **pas** que le modèle a compris la tâche. Seule une explication permet de détecter ce genre de tricherie involontaire.

### 1.8 Ce qu'on attend d'un algorithme « transparent » (diapo 42)

1. On peut y **injecter l'expertise humaine**.
2. Les **modèles appris sont interprétables**.

---

## 2. Les familles de méthodes

Le cours propose une **taxonomie** (diapos 44-45) qu'il faut bien avoir en tête, car elle structure tout le reste.

```Go
                      Interprétabilité en ML
                     /                       \
             BY-DESIGN                       POST-HOC
        (par conception / ante hoc)     (après coup)
                                        /            \
                           Model-agnostic            Model-specific
                       (indépendante du modèle)    (dépendante du modèle)
                         /           \
                     Locale        Globale
```

### Méthodes par conception (*by-design* / *ante hoc*)

Le modèle est **interprétable par nature** : le fait même de l'utiliser donne l'importance relative des descripteurs.
*Exemples : régression linéaire, arbre de décision.*

**Analogie :** c'est une recette de cuisine écrite par le cuisinier. On lit directement la recette.

### Méthodes post-hoc

On **découple** l'apprentissage du modèle et son explication : on entraîne d'abord un modèle (éventuellement très complexe), puis on essaie de l'expliquer **après coup**.

**Analogie :** on goûte un plat dont on ne connaît pas la recette, et on essaie de deviner les ingrédients.

Elles se divisent en deux :

|          | **Indépendantes du modèle (model-agnostic)**                                | **Dépendantes du modèle (model-specific)**                      |
| -------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------- |
| Principe | Traitent le modèle comme une boîte noire : on ne regarde que ses entrées/sorties | Exploitent la structure interne d'un type de modèle précis            |
| Avantage | Marchent avec n'importe quel modèle                                                | Plus riches, mais limités à une famille de modèles                   |
| Exemples | LIME, valeurs de Shapley                                                            | Cartes de saillance, identification des concepts (réseaux de neurones) |

Et les méthodes *model-agnostic* se distinguent encore par leur **portée** :

- **Locale** : explique la prédiction pour **un exemple** donné (« pourquoi *ce* patient est classé à risque ? »).
- **Globale** : explique la **stratégie de décision générale** du modèle.

---

## 3. Interprétabilité par conception (*by design*)

### 3.1 La régression linéaire (diapos 47-54)

**Le modèle.** Il prédit la sortie par une **somme pondérée** des d descripteurs :

```
ŷ = b₀ + b₁·x₁ + b₂·x₂ + … + b_d·x_d
```

Les coefficients bⱼ sont trouvés en **minimisant l'écart quadratique** entre prédictions et valeurs réelles sur les m exemples d'apprentissage (méthode des moindres carrés).

**Hypothèses nécessaires (diapo 48)**, sans quoi le modèle ou ses intervalles de confiance ne sont pas fiables :

| Hypothèse                                              | Signification simple                                                                                                                                    |
| ------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Fonction cible (quasi) linéaire                        | La relation réelle ressemble à une droite / un plan                                                                                                   |
| La sortie suit une loi normale sachant les descripteurs | Sinon les estimateurs de confiance sont invalides                                                                                                       |
| **Homoscédasticité**                            | La dispersion autour de la prédiction est la même partout. Ex. : quelle que soit la taille de la maison, l'incertitude sur le prix est du même ordre |
| Descripteurs non corrélés                             | Chaque variable apporte une information distincte                                                                                                       |

**Comment interpréter les coefficients ? (diapo 49)**
Le coefficient bⱼ dit **de combien varie la sortie quand on augmente xⱼ d'une unité** (toutes choses égales par ailleurs).
*Exemple : de combien le prix d'une maison augmente-t-il avec une pièce de plus ?*

**Lire le tableau de résultats d'une régression (diapos 49-50) :**

| Indicateur                     | Ce qu'il mesure                                                                            | Comment le lire                                                                |
| ------------------------------ | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------ |
| **t-statistique**        | Rapport entre le coefficient et son erreur standard                                        | Plus elle est grande (en valeur absolue), plus le coefficient est « solide » |
| **P > \|t\|** (p-valeur) | Probabilité d'observer un tel coefficient si la variable n'avait en réalité aucun effet | **< 0,05** : on considère que le descripteur a un impact significatif   |
| **[0,025 ; 0,975]**      | Intervalle de confiance à 95 % du coefficient                                             | Si l'intervalle contient 0, l'effet n'est pas clairement établi               |

**La matrice de corrélation (diapo 51)** : elle indique à quel point les descripteurs sont liés **entre eux**.

- +1 : corrélation positive maximale ;
- −1 : corrélation négative maximale ;
- 0 : pas de relation linéaire.
  Si la corrélation est forte, les descripteurs sont **redondants**, ce qui fragilise l'interprétation des coefficients.

**Le graphique des effets individuels (diapo 52)** : il montre la **contribution de chaque descripteur à la prédiction pour un exemple donné** (par exemple un patient). C'est déjà une forme d'explication **locale**.

**Avantages / inconvénients (diapo 54)**

| ✅ Avantages                                      | ⚠️ Inconvénients                                                  |
| ------------------------------------------------- | -------------------------------------------------------------------- |
| Simples à mettre en œuvre                       | Supposent des descripteurs décorrélés                             |
| Permettent d'obtenir des intervalles de confiance | Ne rendent pas bien compte des dépendances **non linéaires** |

**Le piège des variables corrélées.** Si deux descripteurs sont corrélés et que l'un a un très fort coefficient positif, l'autre peut recevoir un coefficient **négatif**, alors même qu'il est positivement corrélé à la cible. Le coefficient ne se lit donc plus comme « effet de cette variable » : il dépend des autres variables du modèle.

> **Exemple intuitif (Complément) :** pour prédire le prix d'une maison, on utilise la « surface » et le « nombre de pièces », très corrélés. Le modèle peut donner un très gros coefficient à la surface et un petit coefficient négatif aux pièces pour « corriger ». Conclure que « plus de pièces fait baisser le prix » serait une erreur d'interprétation.

### 3.2 Les arbres de décision (diapos 55-57)

**Le principe.** L'arbre pose une suite de questions simples (« surface > 80 m² ? », « étage > 3 ? »…) et chaque chemin de la racine à une feuille donne une prédiction. Il se lit comme un **jeu de 20 questions**.

**Avantages :**

- capturent naturellement les **interactions** entre descripteurs (l'effet d'une variable peut dépendre d'une autre) ;
- montrent **comment se forment différents groupes** dans les données ;
- la structure en arbre se prête bien à l'interprétation.

**Inconvénients :**

- peu adaptés aux **dépendances linéaires** (il faut beaucoup de découpes pour approcher une droite ; d'où l'idée des **arbres obliques**) ;
- **instables** : un petit changement dans la base d'apprentissage peut changer beaucoup l'arbre ;
- peuvent devenir **opaques** si les descripteurs sont mal choisis. Exemple du cours : l'explication des fins de parties aux échecs, où des descripteurs bruts (positions des pièces) ne permettent pas de formuler une règle lisible.

Le titre de la diapo 56 résume bien la situation : « **Lisibles… jusqu'à un certain point** ». Un arbre à 3 niveaux se lit facilement ; un arbre à 300 nœuds, non.

### 3.3 Bilan sur les modèles par conception (diapo 58)

- ✅ Plus facilement interprétables.
- ✅ On interprète **le modèle entier**, pas seulement une prédiction donnée.
- ⚠️ Souvent enclins au **sous-apprentissage** : trop simples pour capturer des relations complexes entre descripteurs et cible.

**C'est le dilemme central de l'IA explicable : interprétabilité ↔ performance.** Les modèles simples sont lisibles mais moins puissants ; les modèles puissants sont opaques. Les méthodes post-hoc cherchent à résoudre ce dilemme (même si l'article de Rudin cité en ressource, diapo 124, défend l'idée inverse : pour les décisions à haut risque, mieux vaut utiliser des modèles interprétables plutôt que d'expliquer des boîtes noires).

---

## 4. Interprétation post-hoc

## 4.a Méthodes indépendantes du modèle (*model-agnostic*)

Objectif (diapo 60) : **expliquer une prédiction individuelle**, c'est-à-dire répondre à « pourquoi le modèle a-t-il prédit *ça* pour *cet* exemple ? ». Deux méthodes : **LIME** et les **valeurs de Shapley**.

### 4.a.1 LIME (*Local Interpretable Model-agnostic Explanations*) (diapos 61-64)

**L'idée centrale.**
Le modèle complet f est peut-être très complexe, mais **si l'on regarde de très près autour d'un point**, il ressemble souvent à quelque chose de simple (une droite, un plan).

**Analogie :** la Terre est ronde, mais à l'échelle de votre quartier, une carte plane suffit. LIME construit une « carte plane locale » de la fonction complexe autour de l'exemple à expliquer.

**Lecture de la figure (diapo 62) :**

- la fonction de décision f est complexe (frontière entre zone bleue et zone rose) ;
- la **croix rouge** est l'exemple à expliquer ;
- les points proches sont pondérés par leur **proximité** (plus gros = plus proche) ;
- la **droite en pointillés g** est le modèle simple appris par LIME, qui approxime f **localement**.

**L'algorithme en 6 étapes (diapo 63) :**

| # | Étape                                                                                                | Pour comprendre                                           |
| - | ----------------------------------------------------------------------------------------------------- | --------------------------------------------------------- |
| 1 | Choisir l'exemple à expliquer                                                                        | Ex. : un patient donné                                   |
| 2 | Générer des **exemples virtuels** en permutant/perturbant des valeurs d'attributs             | On crée des « variantes » du patient                   |
| 3 | Attribuer un **poids** à chaque exemple virtuel selon sa distance à l'exemple                 | Les variantes proches comptent plus                       |
| 4 | **Prédire** leur classe avec le modèle appris (la boîte noire)                               | On interroge la boîte noire                              |
| 5 | Entraîner un **modèle surrogé simple** (ex. fonction linéaire) sur ces exemples étiquetés | On apprend à imiter la boîte noire **localement** |
| 6 | **Interpréter** le modèle surrogé                                                            | On lit les coefficients                                   |

**Remarques importantes (diapo 64) :**

- Le **choix de la fonction de distance** est crucial : il définit ce que veut dire « proche », donc la taille du voisinage local.
- Ribeiro et al. utilisent un **noyau exponentiel** de largeur 0,75 × √(nombre de descripteurs).
- LIME est disponible en Python (`lime`, `PiML`) et en R (`lime`, `iml`).

**Point de vigilance (Complément) :** le résultat dépend de la façon dont on génère les exemples virtuels et de la largeur du voisinage ; deux exécutions peuvent donner des explications un peu différentes.

### 4.a.2 Les valeurs de Shapley (diapos 66-81)

#### L'inspiration : la théorie des jeux (diapo 66)

- Chaque **descripteur** = un **joueur**.
- La **prédiction** = le **gain** de la partie.
- Les joueurs forment une **coalition** pour obtenir le gain.
- But : **rétribuer équitablement** chaque joueur selon sa contribution.

La méthode s'applique aux modèles linéaires **et** non linéaires.

#### Illustration pas à pas : le challenge Kaggle (diapos 67-71)

Trois élèves participent à un challenge :

- **E1** apporte l'expertise IODAA ;
- **E2** apporte une machine très puissante ;
- **E3** est un geek.

Ensemble, ils gagnent le 1ᵉʳ prix (10 000 $). Comment partager équitablement ?

**Étape 1 : que gagnerait chaque sous-groupe ?**

| Équipe      | Gain     |
| ------------ | -------- |
| E1 seul      | 7 500 $  |
| E2 seul      | 5 000 $  |
| E3 seul      | 2 500 $  |
| E1 + E2      | 10 000 $ |
| E1 + E3      | 7 500 $  |
| E2 + E3      | 5 000 $  |
| E1 + E2 + E3 | 10 000 $ |

**Étape 2 : l'astuce de Shapley, considérer tous les ordres d'arrivée.**
On imagine que les élèves rejoignent l'équipe un par un, dans tous les ordres possibles (3! = 6 ordres). À chaque arrivée, on note **combien le nouvel arrivant ajoute** au gain de l'équipe déjà formée (sa **contribution marginale**).

| Ordre          | Contribution de E1 | Contribution de E2 | Contribution de E3 |
| -------------- | ------------------ | ------------------ | ------------------ |
| E1 → E2 → E3 | 7 500              | 2 500              | 0                  |
| E1 → E3 → E2 | 7 500              | 2 500              | 0                  |
| E2 → E1 → E3 | 5 000              | 5 000              | 0                  |
| E2 → E3 → E1 | 5 000              | 5 000              | 0                  |
| E3 → E1 → E2 | 5 000              | 2 500              | 2 500              |
| E3 → E2 → E1 | 5 000              | 2 500              | 2 500              |

> Note : sur la diapo 68, certaines lignes sont présentées sous la forme « valeur cumulée → ajout ». J'ai ici directement noté ce que **chaque personne ajoute**. La diapo 71 corrige d'ailleurs une coquille de la diapo 70 sur l'ordre E3 → E1 → E2 (la dernière contribution est 2 500, pas 0).

**Étape 3 : moyenne sur les 6 ordres.**

```
Val(E1) = (7500 + 7500 + 5000 + 5000 + 5000 + 5000) / 6 = 5 833 $
Val(E2) = (2500 + 2500 + 5000 + 5000 + 2500 + 2500) / 6 = 3 333 $
Val(E3) = (   0 +    0 +    0 +    0 + 2500 + 2500) / 6 =   833 $
```

Vérification : 5 833 + 3 333 + 833 ≈ **10 000 $**, soit exactement le gain total. C'est la propriété d'**efficacité** : tout le gain est réparti, rien n'est perdu ni inventé.

> **Pourquoi moyenner sur tous les ordres ?** Parce que la contribution d'un joueur dépend de **qui est déjà là**. Par exemple, E3 n'apporte rien quand E1 et E2 sont déjà présents, mais apporte beaucoup quand il arrive en premier. Pour être équitable, on regarde toutes les façons possibles de construire la coalition.

#### Passage de la théorie des jeux au Machine Learning

| Théorie des jeux                      | Machine Learning                                  |
| -------------------------------------- | ------------------------------------------------- |
| Joueur                                 | Descripteur (feature)                             |
| Coalition                              | Sous-ensemble de descripteurs « connus »        |
| Gain                                   | Prédiction du modèle                            |
| Gain de référence (personne ne joue) | **Prédiction moyenne** du modèle E[ĥ(X)] |

**Ce qu'on explique :** l'**écart entre la prédiction pour un exemple et la prédiction moyenne**. Les valeurs de Shapley répartissent cet écart entre les descripteurs.

#### Exemple 1 : le salaire (diapos 72-73)

Modèle appris : **h(x₁, x₂) = 200·x₁ + 1000·x₂**

- x₁ = âge ∈ {18…60}, distribution supposée uniforme ;
- x₂ = diplôme ∈ {0, 1}, probabilités égales.

Pour une personne de 20 ans avec diplôme : h = 200×20 + 1000×1 = **5 000 €**.

**Quelle est la valeur de Shapley de x₂ (diplôme) ?**

*Étape 1 : la prédiction moyenne.*
Âge moyen = 39, proportion moyenne de diplômés = 0,5
E[h] = 200×39 + 1000×0,5 = 7 800 + 500 = **8 300 €**

*Étape 2 : contribution marginale du diplôme dans chaque coalition possible.*

- Quand on **ne connaît pas l'âge** (coalition vide) :
  prédiction avec diplôme connu = 200×39 + 1000×1 = 8 800 ; sans = 8 300 → apport = **500**
- Quand on **connaît déjà l'âge** (coalition {âge}) :
  avec diplôme = 5 000 ; sans diplôme connu = 200×20 + 1000×0,5 = 4 500 → apport = **500**

*Étape 3 : moyenne des apports.* Ici, 500 dans les deux cas, donc **φ(diplôme) = 500 €**.

**Résultat à retenir (Complément) :** pour un **modèle linéaire**, la valeur de Shapley d'un descripteur vaut simplement
`φⱼ = bⱼ × (xⱼ − moyenne de xⱼ)`.
Ici, φ(âge) = 200 × (20 − 39) = −3 800 €. La somme −3 800 + 500 = −3 300 = 5 000 − 8 300 ✓.

*Interprétation :* cette personne est jeune, donc sa prédiction est bien plus basse que la moyenne (−3 800 €), et son diplôme la remonte un peu (+500 €).

#### Exemple 2 : les appartements (diapo 74)

Le modèle prédit le prix selon la surface, l'étage, la proximité d'un parc, et l'interdiction ou non des chats.

Exemple à expliquer : `area = 50, floor = 2, park = nearby, cat = banned`.
Prix prédit : **300 k€**, prix moyen : **310 k€**. Il faut expliquer l'écart de **−10 k€**.

| Descripteur     | Contribution          |
| --------------- | --------------------- |
| area = 50       | +30 k€               |
| floor = 2       | +10 k€               |
| park = nearby   | 0 €                  |
| cat = banned    | −50 k€              |
| **Total** | **−10 k€** ✓ |

Les contributions s'additionnent exactement à l'écart par rapport à la moyenne : c'est la grande force de Shapley.

#### La formule générale (diapo 75)

```
φⱼ = Σ        [ |S|! · (d − |S| − 1)! / d! ] · ( val(S ∪ {j}) − val(S) )
     S ⊆ {1…d}\{j}
```

avec

```
val_x(S) = ∫ ĥ(x₁,…,x_d) dP_{X_C}  −  E[ĥ(X)]
```

**Décodage de la formule :**

| Symbole                     | Signification                                                                                                                                                                              |
| --------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| j                           | Le descripteur dont on veut la contribution                                                                                                                                                |
| S                           | Un sous-ensemble (une « coalition ») de descripteurs **ne contenant pas** j                                                                                                        |
| d                           | Nombre total de descripteurs                                                                                                                                                               |
| `val(S ∪ {j}) − val(S)` | **Contribution marginale** de j : ce que gagne la prédiction quand on ajoute j à la coalition S                                                                                    |
| `\|S\|!(d−\|S\|−1)!/d!`     | **Poids** : fraction des ordres d'arrivée dans lesquels j arrive juste après S. C'est ce qui réalise la « moyenne sur tous les ordres »                                         |
| `val_x(S)`                | Prédiction attendue quand **seuls les descripteurs de S sont connus** (valeurs de x), les autres (X_C) étant « moyennés » selon leur distribution, moins la prédiction moyenne |
| X_C                         | Les attributs **non inclus** dans la coalition S                                                                                                                                     |

**En une phrase :** la valeur de Shapley de j = la **moyenne pondérée** de tout ce que j apporte à la prédiction, dans toutes les coalitions possibles.

#### Deux difficultés pratiques (diapos 78-80)

**1. Comment faire quand on ne connaît pas la distribution des attributs ?**
On l'**estime** en tirant au hasard, dans l'**ensemble d'apprentissage**, des valeurs pour les variables « inconnues » de la coalition (idéalement toutes les valeurs) (diapo 79). Pour « ne pas connaître » une variable, on la remplace par des valeurs observées dans les données.

**2. Comment gérer l'explosion du nombre de coalitions ?**
Avec d descripteurs, il y a **2^d coalitions** : 20 descripteurs = plus d'un million, 50 descripteurs = hors de portée.
**Complément :** on utilise en pratique une approximation par **échantillonnage de Monte-Carlo** (on tire au hasard des ordres d'arrivée au lieu de les énumérer tous). Le livre de Molnar cité en ressources détaille cette approche (diapo 80).

#### Remarques finales (diapo 81)

- **LIME** donne l'**importance relative** de chaque attribut (approximation locale).
- **Shapley** donne la contribution « **exacte** » de chaque valeur mesurée de chaque attribut (avec la propriété que tout s'additionne).

> ⚠️ **Mise en garde essentielle (G. Zuin) :** *SHAP n'est pas une mesure de l'importance d'une variable dans le monde réel, mais de son importance **pour le modèle**.*
> Si le modèle a appris une corrélation absurde (effet Clever Hans), SHAP attribuera de l'importance à cette corrélation absurde. SHAP explique **le modèle**, pas **la réalité**.

### Comparaison LIME / Shapley (Complément)

|                        | LIME                                           | Shapley (SHAP)                                           |
| ---------------------- | ---------------------------------------------- | -------------------------------------------------------- |
| Idée                  | Modèle simple local qui imite la boîte noire | Partage équitable de la prédiction entre descripteurs  |
| Base théorique        | Approximation locale                           | Théorie des jeux (propriétés d'équité)              |
| Garantie d'additivité | Non                                            | Oui (somme = prédiction − moyenne)                     |
| Coût de calcul        | Faible                                         | Élevé (2^d coalitions, d'où approximations)           |
| Portée                | Locale                                         | Locale (et globale si on agrège sur plusieurs exemples) |

---

## 4.b Méthodes dépendantes du modèle (*model-specific*)

### 4.b.1 Réseaux de neurones en vision : les méthodes à base de gradient (diapos 83-99)

**La question posée :** pour une image classée « chat », **quels pixels** ont poussé le réseau à dire « chat » ? Le résultat se présente sous forme d'une **carte de saillance** (*saliency map*) ou d'une *heat map* : une image où les zones importantes sont « allumées ».

**L'idée commune à toutes ces méthodes : utiliser le gradient.**
Le gradient mesure **à quel point la sortie varie quand on modifie légèrement une entrée**. Un gradient élevé pour un pixel veut dire : « si je change un peu ce pixel, la sortie change beaucoup, donc ce pixel est important. »

#### ① Vanilla gradient (diapo 83)

On calcule le gradient du **logit** de la classe prédite y_c par rapport à chaque pixel de l'image d'entrée :

```
Carte de saillance = ∂ y_c / ∂ pixel
```

(le logit est le score brut avant le softmax ; y_c est celui de la classe prédite, de plus fort logit).

#### ② Input × Gradients (diapo 84)

**Le problème du gradient seul.** Considérons un modèle linéaire simple :

```
y = 100·x₁ + 5·x₂ + 1000      →   ∂y/∂x₁ = 100,   ∂y/∂x₂ = 5
```

Le gradient suggère que x₂ est peu important. Mais prenons un client de **30 ans** (x₁) et de **50 000 €** de salaire (x₂). Les échelles des variables sont très différentes ! En multipliant le gradient par la valeur d'entrée :

```
x₁ · ∂y/∂x₁ = 30 × 100      =   3 000
x₂ · ∂y/∂x₂ = 50 000 × 5    = 250 000
```

Le salaire apparaît alors **beaucoup plus important**, ce qui correspond mieux à son poids réel dans la prédiction.

> **À retenir :** le gradient seul mesure la *sensibilité* ; multiplier par l'entrée mesure la *contribution* réelle.

**Problèmes de ces méthodes sur les images (diapo 85) :**

1. Dans une image, les valeurs de pixels n'ont **pas de relation linéaire directe** avec la sortie.
2. La carte met en évidence la **structure de l'image** (contours, formes), ce qui est agréable à regarder mais **pas nécessairement ce que le système utilise réellement** pour prédire.

#### ③ Guided Backpropagation (GBP) (diapos 86-87, 92, 94)

**Idée :** modifier la rétropropagation standard en ne laissant passer que les **gradients positifs**.

- Dans la rétropropagation standard, les gradients négatifs servent à diminuer l'activation des éléments **non liés** à la classe cible.
- Pour une *visualisation*, on veut seulement voir ce qui est **associé** à la classe cible. En supprimant les gradients négatifs, on **réduit le bruit** et on obtient des visualisations plus nettes.
- La GBP peut aussi servir à regarder ce qui compte dans les **couches intermédiaires** (diapo 87 : les premières couches captent des formes basiques, les couches plus profondes des descripteurs plus abstraits).

**Limites (diapos 92 et 94) :**

- La GBP **ne distingue pas les classes**. Exemple : la fourrure est importante pour reconnaître un chat ; mais si plusieurs animaux à fourrure sont sur l'image, la GBP éclaire **toutes** les zones de fourrure, quel que soit l'animal. Elle montre ce qui contribue à la classe cible, pas ce qui lui est **spécifique**.
- Une carte de caractéristiques d'une couche antérieure peut contenir des éléments génériques (contours, textures). Un détecteur de contours pourrait contribuer positivement à une couche plus profonde **même si ce contour vient d'une région non pertinente** de l'image.

#### ④ Grad-CAM (*Gradient-weighted Class Activation Mapping*) (diapos 88, 95)

**Idée :** au lieu de propager le gradient jusqu'aux pixels d'entrée, on **s'arrête à la dernière couche convolutive**.

**Pourquoi ce choix ?** Cette couche, proche de la sortie, contient à la fois des **informations sémantiques** (« il y a un chat ») et **spatiales** (« à cet endroit »). Les gradients y sont les plus forts aux endroits des objets qui ont contribué positivement au logit cible.

| ✅ Points forts                                                                            | ⚠️ Limites                                                                     |
| ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------- |
| **Discrimine les classes** : montre les régions importantes pour une classe donnée | **Résolution faible** (les couches profondes sont petites) → carte floue |
| Rapide                                                                                     | Moins précise que d'autres méthodes                                            |
| Applicable à **tout** réseau convolutif                                            |                                                                                  |
| Sortie : une *heat map* intuitive                                                        |                                                                                  |

Cela explique son immense popularité « malgré ses défauts ».

**Exemples de la diapos 89-91 :**

- *Succès (diapo 90)* : même quand une image est modifiée par un **bruit adverse** de sorte que le réseau est trompé (il voit un « avion » avec une confiance > 0,9999), Grad-CAM localise encore correctement les catégories **originales** (boxer, chat tigré). Cela montre une certaine **robustesse**.
- *Échecs (diapo 91)* : des cas où l'image est classée « lévrier à 35 % » ou **incorrectement comme anguille à 70 %**, et où la carte de saillance n'est pas très convaincante.

#### ⑤ Guided Grad-CAM (diapo 93)

Les deux méthodes sont **complémentaires** :

- **Guided Backpropagation** : met en évidence des **détails fins**, mais ne discrimine pas les classes ;
- **Grad-CAM** : **discrimine les classes**, mais est grossière.

On les **combine** (multiplication des deux cartes) : on obtient une carte à la fois précise et spécifique à la classe.

#### ⑥ SmoothGrad : « ajouter du bruit pour enlever du bruit » (diapos 96-97)

Les cartes de gradient sont souvent **très bruitées** (le gradient fluctue brusquement d'un pixel à l'autre).

**Idée :** on crée plusieurs **copies bruitées** de l'image, on calcule la carte de saillance de chacune, puis on **fait la moyenne**. Le bruit s'annule en moyenne, ce qui laisse une carte plus lisse et plus lisible.

**Analogie :** prendre plusieurs photos légèrement tremblées et les superposer donne, paradoxalement, une image plus stable que d'en choisir une seule.

#### ⑦ Integrated Gradients (diapo 98)

*(la diapo ne contient que des illustrations)*
**Complément :** au lieu de calculer le gradient **en un seul point** (l'image), on part d'une image de **référence neutre** (par exemple une image noire) et on **somme les gradients** le long du chemin qui va de cette référence à l'image réelle. Cela évite le problème des gradients « saturés » (un pixel très important peut avoir un gradient proche de 0 si le réseau est déjà certain de sa réponse).

#### Les limites générales des méthodes à gradient (diapo 99)

1. **Difficile de savoir si l'explication est correcte** (réellement causale dans la prédiction).
2. **Biais de confirmation :** on peut croire qu'une explication est bonne simplement parce qu'elle satisfait nos attentes (« c'est bien la tête du chat qui est éclairée »).
3. **Instabilité :** de petites variations de l'image peuvent conduire à des explications très différentes. Exemple (Kindermans et al., 2019) : si on **ajoute une constante à tous les pixels**, le neurone de biais compense parfaitement, les poids restent les mêmes et **le modèle est identique**… pourtant les « explications » changent !

> Si deux modèles **strictement équivalents** produisent des explications différentes, alors l'explication ne reflète pas toujours le fonctionnement du modèle.

### 4.b.2 Les modèles à goulot d'étranglement conceptuel (*Concept Bottleneck Models*, CBM) (diapos 100-106)

**Le problème des cartes de saillance :** elles montrent **où** le réseau regarde, mais pas **quoi** il voit. Savoir qu'il regarde « la zone du genou » ne dit pas s'il a vu une « excroissance osseuse ».

**L'idée des CBM :** forcer le modèle à passer par des **concepts intermédiaires compréhensibles par l'humain** avant de prendre sa décision, au lieu d'aller directement des données brutes à l'étiquette finale.

```
       Modèle classique :       image brute ───────────────────────► classe

       Concept Bottleneck :     image brute ──► CONCEPTS ──► classe
                                (pixels)      (compréhensibles)
```

**Fonctionnement en deux étapes (diapo 102) :**

1. **Encodeur de concepts g(x)** : transforme les données brutes (pixels, texte) en un ensemble de **concepts définis par l'humain** c.
   - Radiographie du genou : « excroissances osseuses », « sclérose ».
   - Classification d'oiseaux : « couleur des ailes », « longueur du bec ».
2. **Prédicteur f(c)** : utilise **uniquement** les valeurs de concepts prédites pour produire la décision finale y.

On apprend donc un modèle de la forme **f(g(x))** avec :

- `g : ℝᵈ → ℝᵏ` (k concepts) ;
- `f : ℝᵏ → ℝⁿ` (n classes) ;
- **deux fonctions de coût dédiées** : une pour bien prédire les concepts (ĝ), une pour bien prédire la classe à partir des concepts (f̂).

**Données nécessaires (diapo 104) :** il faut un ensemble d'apprentissage de triplets **(xᵢ, yᵢ, cᵢ)** : l'entrée, l'étiquette **et** les concepts annotés pour chaque exemple. C'est le principal coût des CBM (il faut des annotations de concepts, souvent faites par des experts).

**Le grand avantage : l'intervention (diapo 105).**
Comme le modèle passe par des concepts, un humain peut **corriger un concept** (« non, il n'y a pas d'excroissance osseuse ici ») et voir **comment la prédiction change**. On peut ainsi corriger une classification en intervenant sur un seul concept (ou un groupe de concepts).

**Analogie :** c'est la différence entre un médecin qui dit « le résultat est positif » (boîte noire) et un médecin qui dit « j'ai observé ces trois signes, donc je conclus à cela » : on peut discuter chaque signe.

### 4.b.3 Comment évaluer un système explicatif ? (diapo 106)

Cinq critères proposés pour juger un système à concepts :

1. **Performance de prédiction** : elle ne doit pas être détériorée par la contrainte d'interprétabilité.
2. **Performance de reconnaissance des concepts intermédiaires** : les concepts sont-ils bien détectés ?
3. **Stabilité** des poids entre concepts et classes, quand on répète les exécutions ou qu'on varie les conditions initiales.
4. **Cohérence :** les concepts importants dans le **codage** sont-ils aussi importants pour le **calcul de la sortie** ?
5. **Sensibilité** de la sortie aux interventions sur les poids des concepts utilisés pour coder un exemple.

---

## 5. Quel avenir pour l'IA explicable ?

### 5.1 Ce qui manque au Machine Learning actuel (diapos 108-109)

> **Le focus exclusif sur la performance prédictive conduit à une formulation incomplète du problème d'apprentissage.**

On veut aussi :

- l'**interprétabilité des résultats** ;
- l'**interprétabilité du processus** d'apprentissage ;
- **mieux comprendre le monde**, en intégrant la fonction apprise dans une théorie existante.

**Conclusion forte du cours :** il faut d'une certaine manière **changer le critère inductif** utilisé en ML. Aujourd'hui on minimise une erreur de prédiction ; demain, il faudrait optimiser aussi l'intelligibilité, par exemple en l'intégrant dans la fonction de coût.

### 5.2 Les explications humaines sont-elles bonnes ? (diapos 110-116)

Avant d'exiger des explications des machines, regardons comment les humains s'expliquent.

**L'expérience de « l'effet spectateur » (*bystander effect*).**
Des participants, accueillis individuellement pour une « étude de marché sur les jeux », remplissent un questionnaire. Dans la pièce voisine, l'assistante grimpe sur une chaise, puis on entend un grand fracas et un cri de douleur (« Oh mon Dieu, mon pied… je ne peux plus bouger… »).

Deux conditions :

- **Condition 1 :** le participant est **seul** dans la pièce.
- **Condition 2 :** un autre « participant » (en réalité un complice) est présent et **réagit à peine**, haussant les épaules et continuant à remplir son questionnaire.

**Résultats :**

| Condition               | % des participants qui interviennent |
| ----------------------- | ------------------------------------ |
| Seul                    | **70 %**                       |
| Avec le complice passif | **7 %**                        |

**Les explications données après coup :**

- Seul : « Je n'étais pas sûr de ce qui s'était passé ; je me suis dit qu'il fallait au moins aller voir. »
- Avec le complice : les participants pensaient que ce n'était pas grave. **Aucun n'a jamais mentionné la présence de l'autre personne**, et ils affirmaient que celle-ci n'avait eu **aucune influence** sur leur décision.

**La leçon (diapo 116) :**

- Les raisons invoquées viennent **après coup**, une fois la décision prise.
- Elles sont **manifestement fausses** (la vraie cause, la présence du complice, est ignorée).
- **Les raisons sont des justifications** (des rationalisations), pas des comptes rendus fidèles du processus de décision.

**Pourquoi cela compte pour l'IA ?** Une explication post-hoc fournie par un modèle (ou par un humain) n'est pas nécessairement **la vraie cause** de la décision. C'est exactement le risque de LIME, Grad-CAM ou des *chains of thought* : produire une histoire plausible qui ne reflète pas le calcul réel.

### 5.3 Pourquoi cela devient crucial pour les machines (diapos 117-121)

**Les IA sont de plus en plus des sociétés d'agents** qui communiquent entre eux et prennent des décisions en fonction de ces interactions. **Chaque agent devrait pouvoir expliquer les « raisons » de ses sorties aux autres agents.**

**Cas du Tesla (7 mai 2016) (diapos 119-120).** Une voiture en mode autopilote est entrée en collision avec une semi-remorque traversant la route.

- Le **radar** avait détecté la semi-remorque, mais de nombreux panneaux routiers ont une « signature » radar similaire à celle d'une voiture.
- La **caméra** était incertaine : le ciel était d'un blanc laiteux et la remorque était blanche.

**Que se passerait-il si chaque capteur pouvait justifier ses doutes ?**

- Radar : « Je ne suis pas sûr, mais je ne crois pas que ce soit une semi-remorque, car cela ressemble à un panneau routier, comme j'en ai beaucoup détecté sur cette route. »
- Caméra : « Je ne suis pas sûre, mais je ne crois pas que ce soit une semi-remorque, car la couleur est proche de celle du ciel, laiteux aujourd'hui. »

Un module de fusion qui reçoit ces **justifications** (et pas seulement des scores) pourrait comprendre que les **deux erreurs ont des causes indépendantes et faibles**, et décider de façon plus prudente. Et si capteurs et modules avaient un **long historique d'échanges** et s'influençaient mutuellement…

**Interactions entre modules d'apprentissage : la publicité adaptative (diapo 121).**
Deux sous-systèmes : l'un place les liens publicitaires, l'autre choisit les publicités. Chacun apprend à partir des **données de clics**, qui dépendent à leur tour de **l'autre système** et de facteurs non contrôlés (prix, requêtes…). Ils s'influencent mutuellement, ce qui rend l'analyse causale très délicate (Bottou et al., 2013).

### 5.4 Questions ouvertes (diapos 122-123)

Le cours se termine volontairement par des questions **sans réponse définitive** :

| Question                                                                    | Pourquoi elle est intéressante                                                                                                                         |
| --------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Interprétabilité vs explicabilité ?**                            | Interpréter = comprendre le fonctionnement d'un modèle transparent ; expliquer = produire une justification (souvent après coup) d'un modèle opaque |
| **Comment évaluer une explication ?**                                | Une explication plausible n'est pas forcément fidèle                                                                                                  |
| **Explications et causalité ?**                                      | Corrélation ≠ causalité (cf. Clever Hans)                                                                                                            |
| **Une explication est-elle symbolique par nature ?**                  | Faut-il des concepts/mots pour expliquer ?                                                                                                              |
| **Explication « one-shot » ou par interactions ?**                  | Une bonne explication est peut-être un **dialogue**                                                                                              |
| **Expliquer permet-il de tromper le système ?**                      | Si je sais comment la machine raisonne, je peux la manipuler                                                                                            |
| **Pour qui ?**                                                        | Experts IA, experts du domaine, utilisateurs : trois publics, trois types d'explications                                                                |
| **Les « chains of thought » sont-elles des explications fiables ?** | Le raisonnement écrit par un LLM correspond-il à son calcul réel ? (cf. l'effet spectateur !)                                                        |
| **Place de la causalité : le graal ?**                               | Une explication causale serait robuste, non manipulable et généralisable                                                                              |

### Ressources (diapo 124)

- adataodyssey.com
- Christoph Molnar, *Interpretable Machine Learning* (3ᵉ éd., 2025)
- Cynthia Rudin, *Stop explaining black box machine learning models for high stakes decisions and use interpretable models instead*, Nature Machine Intelligence, 2019 (arXiv:1811.10154)

---

## 6. Fiche de révision

### Les idées-clés en 10 points

1. **Expliquer n'est pas nouveau** : les systèmes experts expliquaient via leurs règles. Le deep learning a rendu le problème aigu (boîtes noires).
2. **Le ML ne crée pas d'information** : tout vient des **données** + des **connaissances a priori** (choix des descripteurs, biais de représentation, architecture).
3. **L'interprétabilité est nécessaire** pour : décisions à haut risque, curiosité scientifique, débogage, et robustesse face à la manipulation.
4. **Clever Hans** : un bon score peut cacher une mauvaise raison (corrélation accidentelle).
5. **Taxonomie** : *by-design* vs *post-hoc* ; post-hoc = *model-agnostic* (local/global) ou *model-specific*.
6. **Modèles par conception** (régression linéaire, arbres) : lisibles mais enclins au sous-apprentissage.
7. **LIME** : approxime localement la boîte noire par un modèle simple, entraîné sur des exemples perturbés et pondérés par leur proximité.
8. **Shapley** : répartit équitablement l'écart (prédiction − moyenne) entre descripteurs, en moyennant les contributions marginales sur toutes les coalitions. ⚠️ Explique le *modèle*, pas le *monde*.
9. **Vision** : gradient, input×gradient, GBP, **Grad-CAM**, Guided Grad-CAM, SmoothGrad, Integrated Gradients. Aucune n'est garantie causale.
10. **CBM** : on force le passage par des **concepts humains**, ce qui permet l'**intervention**.

### Tableau récapitulatif des méthodes

| Méthode              | Famille                          | Portée                                   | Principe en une ligne                          | Limite principale                             |
| --------------------- | -------------------------------- | ----------------------------------------- | ---------------------------------------------- | --------------------------------------------- |
| Régression linéaire | By-design                        | Globale (+ locale via effets individuels) | Coefficients = effet d'une unité de variation | Hypothèses fortes, linéarité               |
| Arbre de décision    | By-design                        | Globale                                   | Suite de questions simples                     | Instable, opaque si trop grand                |
| LIME                  | Post-hoc, agnostique             | Locale                                    | Modèle simple local sur exemples perturbés   | Dépend du voisinage et de la distance        |
| Shapley / SHAP        | Post-hoc, agnostique             | Locale                                    | Partage équitable via coalitions              | Coût 2^d ; explique le modèle, pas le monde |
| Vanilla gradient      | Post-hoc, spécifique (réseaux) | Locale                                    | Sensibilité de la sortie aux pixels           | Bruité                                       |
| Input × Gradient     | idem                             | Locale                                    | Gradient × valeur d'entrée                   | Structure visuelle ≠ usage réel             |
| Guided Backprop       | idem                             | Locale                                    | Seuls les gradients positifs passent           | Ne discrimine pas les classes                 |
| Grad-CAM              | idem                             | Locale                                    | Gradient jusqu'à la dernière couche conv.    | Résolution faible                            |
| Guided Grad-CAM       | idem                             | Locale                                    | GBP × Grad-CAM                                | Hérite de limites des deux                   |
| SmoothGrad            | idem                             | Locale                                    | Moyenne sur copies bruitées                   | Coût de calcul                               |
| Concept Bottleneck    | Par conception (architecture)    | Globale + locale                          | Passer par des concepts humains                | Nécessite des annotations de concepts        |

---

## 7. Questions pour tester sa compréhension

1. Pourquoi dit-on que le ML « ne crée pas d'information » ? Donnez trois endroits où des connaissances a priori entrent dans un système d'apprentissage.
2. Citez quatre situations où l'interprétabilité est nécessaire, et deux où elle ne l'est pas.
3. Expliquez l'effet Clever Hans avec vos propres mots et donnez un exemple en apprentissage supervisé.
4. Quelle est la différence entre une méthode *by-design* et *post-hoc* ? Entre *model-agnostic* et *model-specific* ? Entre locale et globale ?
5. Une régression linéaire donne un coefficient négatif à un descripteur pourtant positivement corrélé à la cible. Quelle explication possible ?
6. Décrivez les 6 étapes de LIME. Pourquoi la fonction de distance est-elle cruciale ?
7. Dans l'exemple Kaggle, pourquoi doit-on considérer **tous les ordres** d'arrivée des élèves ?
8. Modèle h(x₁, x₂) = 200·x₁ + 1000·x₂, avec âge moyen 39 et 50 % de diplômés. Calculez les valeurs de Shapley des deux descripteurs pour une personne de 30 ans avec diplôme. *(Réponse : prédiction = 7 000 ; moyenne = 8 300 ; φ(âge) = 200×(30−39) = −1 800 ; φ(diplôme) = 1000×(1−0,5) = +500 ; somme = −1 300 = 7 000 − 8 300 ✓)*
9. Que signifie la phrase « SHAP mesure l'importance pour le modèle, pas pour le monde réel » ?
10. Pourquoi multiplier le gradient par l'entrée (Input × Gradient) peut-il changer la conclusion sur l'importance d'une variable ?
11. Pourquoi Grad-CAM s'arrête-t-il à la dernière couche convolutive, et quel en est le coût ?
12. Que montre l'expérience de Kindermans et al. (constante ajoutée aux pixels) sur la fiabilité des explications ?
13. Comment fonctionne un Concept Bottleneck Model ? Quel avantage apporte la possibilité d'« intervenir » ?
14. Que nous apprend l'expérience de l'effet spectateur sur la valeur des explications a posteriori, y compris celles d'une IA ?
15. Pourquoi l'explicabilité devient-elle encore plus importante dans les systèmes multi-agents (exemple Tesla) ?

---

## 8. Glossaire

| Terme                                          | Définition simple                                                                          |
| ---------------------------------------------- | ------------------------------------------------------------------------------------------- |
| **Boîte noire**                         | Modèle dont le fonctionnement interne n'est pas lisible par un humain                      |
| **Interprétabilité**                   | Capacité à comprendre *comment* le modèle fonctionne (souvent par construction)        |
| **Explicabilité**                       | Capacité à fournir une explication d'une décision, souvent après coup                   |
| **By-design / ante hoc**                 | Modèle interprétable par nature                                                           |
| **Post-hoc**                             | Explication produite après l'entraînement, séparément du modèle                        |
| **Model-agnostic**                       | Méthode utilisable avec n'importe quel modèle (on n'utilise que ses entrées/sorties)     |
| **Model-specific**                       | Méthode propre à une famille de modèles (ex. réseaux de neurones)                       |
| **Explication locale / globale**         | Pour un exemple précis / pour tout le comportement du modèle                              |
| **Descripteur (feature)**                | Variable d'entrée du modèle                                                               |
| **Corrélation accidentelle (spurious)** | Lien statistique présent dans les données mais sans rapport causal réel                  |
| **Clever Hans**                          | Un modèle qui réussit pour de mauvaises raisons                                           |
| **Modèle surrogé**                     | Modèle simple qui imite un modèle complexe (ex. dans LIME)                                |
| **Coalition**                            | Sous-ensemble de descripteurs (théorie des jeux / Shapley)                                 |
| **Contribution marginale**               | Ce qu'ajoute un descripteur quand on l'ajoute à une coalition                              |
| **Homoscédasticité**                   | Variance constante des erreurs quelle que soit la valeur des descripteurs                   |
| **Logit**                                | Score brut de sortie d'un réseau avant le softmax                                          |
| **Carte de saillance (saliency map)**    | Image indiquant l'importance de chaque pixel pour la décision                              |
| **Gradient**                             | Mesure de la sensibilité de la sortie à une petite variation de l'entrée                 |
| **Rétropropagation**                    | Algorithme qui calcule les gradients couche par couche dans un réseau                      |
| **Concept Bottleneck Model (CBM)**       | Modèle qui passe par des concepts humains intermédiaires avant de prédire                |
| **Contrefactuel**                        | « Que se passerait-il si telle valeur était différente ? »                              |
| **Biais de représentation**             | Contrainte sur l'espace d'hypothèses (ex. préférer les modèles linéaires parcimonieux) |
