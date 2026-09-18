# <p style="text-align: center;">Introduction à l'Apprentissage Automatique</p>

*Cours d'Antoine Cornuéjols (AgroParisTech – INRAe MIA Paris-Saclay, EKINOCS research group)*

## Plan du cours

1. Une science de l'apprentissage ?
2. Les grands types d'apprentissage
3. Le problème de l'apprentissage supervisé
4. Apprendre dans un espace d'hypothèses structuré *(à venir)*
5. Conclusion *(à venir)*

---

## Une science de l'apprentissage ?

### Pourquoi apprendre ?

Alors qu'on pourrait simplement coder les règles à la main.

**Funfact :** pour prédire la météo de demain, la méthode la plus simple est de dire qu'il fera le même temps qu'aujourd'hui.

**MAIS** reconnaître un chien : on peut le simplifier facilement (une boîte avec des pattes), mais ce n'est pas si simple en pratique, car il y a énormément d'espèces différentes.

**Motivation (slide) — des concepts difficiles à coder à la main**, donc il vaut mieux utiliser l'apprentissage :

- les mouvements permis pour un robot
- décider de recruter (ou pas) une personne
- les prédispositions à certains types de cancer

→ on préfère alors **apprendre à partir d'exemples** plutôt que coder des règles.

#### 1. Reconnaissance, prédiction

**La reconnaissance :**
exemple : image (lésions de peau, chihuahua vs. cookie, segmentation automatique en temps réel — voitures autonomes, imagerie satellite —, reconnaissance d'objets) ou de la parole (chatbots, assistants, reconnaissance du locuteur).

**La prédiction :**
exemple : prédiction par séries temporelles (pluie, bourse, ventes, insectes ravageurs).

#### 2. IA générative

*(apparition du terme : 2017, en sciences)*

- **AlphaFold** : de la séquence protéique à la configuration 3D de la protéine.
- **ITHACA** : archéologie — reconstitue des textes entiers à partir de fragments. Taux de bonne prédiction = 62 % contre 25 % pour les spécialistes humains. Fournit aussi une probabilité sur la région d'origine et la date du texte (entre 800 av. J.-C. et 800 apr. J.-C.).
- **Génération** : d'images (visages), de textes, de code…
- **AlphaGo** : tu connais.

**Comment cela marche ? → Adaptation / imitation :**

- apprentissage comportemental (ex. les « insectes » de Brooks pour apprendre à marcher, apprendre à agir sur une planète inconnue)
- apprentissage à jouer (s'adapter à l'adversaire, ne pas répéter ses erreurs passées, jouer en équipe — d'où l'exemple des équipes de robots-foot)

**Funfact :** on a gagné la coupe du monde de robot-foot car l'équipe française était la première équipe à avoir appris à mentir (feinter l'adversaire).

#### 3. IA agentique

**Définition :** systèmes (partiellement) autonomes, une société d'agents spécialisés, capables d'« agir » dans leur environnement.

**Funfact :** une IA agentique peut aller supprimer tout ton disque dur si elle en a besoin pour accomplir sa mission.

**Exemple :** aider à la découverte scientifique (ex. Co-Scientist : génération d'hypothèses, protocoles expérimentaux et validation, rédaction d'articles).

**Autres exemples cités :** préparer un voyage et effectuer les réservations ; créer et animer des communautés sur une plateforme sociale ; automatiser des opérations informatiques (création de code + documentation, débogage, maintenance).

---

Nous avons besoin de plus en plus de ressources de calcul.

**Problème universitaire :** nous n'avons pas les capacités que les grandes entreprises ont, et en plus il y a du *gate keeping*, car elles ont besoin de garder leurs secrets pour rester dans la course. (Un peu frustrant pour la recherche académique.)

**Funfact :** pour résoudre l'équation de Navier-Stokes, un futur système type « OpenAI next generation » mobiliserait plus de 10 000 agents, plus de 2 300 000 messages échangés, environ 130×10⁹ tokens, et plusieurs millions de dollars en électricité — pour illustrer la croissance (exponentielle, en échelle logarithmique) des ressources de calcul employées.

*Rappel : un petaflop = 10¹⁵ opérations flottantes par seconde.*

---

### Y a plein de données !

**Tout d'abord les capteurs :**

- Agriculture
- Météo

**Mais aussi en code informatique :**

- GitHub, GitLab, Stack Overflow, Hugging Face
- Internet en général
- LLM et autres

**Contre-exemples** *(domaines qui manquent de données) :*

- Alimentation (besoin d'un historique précis du comportement des gens)
- Usage de produits (pollution par solvants)
- Éducation et ce qu'il y a dans les cerveaux des gens

**Funfact :** l'enquête Nutrinet (~277 000 internautes suivis théoriquement sur des années) souffre en réalité d'un manque de données représentatives — panel à 80 % de femmes, milieux socio-professionnels élevés surreprésentés, abandons fréquents après quelques jours. Autre contre-exemple : l'éducation, où l'on a peu de données sur ce qui se passe réellement en classe ou devant un écran.

---

### Que sait-on faire et que ne peut-on pas faire ?

*Attention : on ne sait pas si c'est toujours le cas.*

**Le point central :** la question de l'explicabilité des réseaux de neurones profonds — pourquoi le réseau se trompe-t-il ?

**Adversarial learning :** il est possible de modifier très légèrement des images pour tromper l'IA (du bruit sur les pixels).

**Annotation d'images :** on peut modifier de façon flagrante pour les humains une image, sans que l'IA ne perçoive la différence (exemple : remplacer des images de jeunes par des vieux).

**Exemples :**

- cas en médecine : on tourne juste une image et le mélanome passe de bénin à cancéreux très facilement
- cas en image : voiture dans une piscine — la machine ne le perçoit pas car il n'y a jamais eu d'image d'une voiture dans une piscine, donc elle reste indifférente, alors que les humains réagissent immédiatement : « euh, non »

---

**Loi fondamentale de l'apprentissage :** il faut avoir des exemples d'erreurs pour apprendre réellement un langage ou autre.

**Citation (Tom Mitchell, 2006)**, l'un des pionniers du domaine :
*« How can we build computer systems that automatically improve with experience, and what are the fundamental laws that govern all learning processes? »*
— une bonne définition de ce qu'est le Machine Learning.

**De nouvelles possibilités ouvertes par l'apprentissage :**

- **Aider à comprendre** : des phénomènes complexes, des systèmes naturels.
- **Aider à décider** : pour des acteurs/secteurs multiples (agriculteurs, consommateurs, aménagement du territoire, santé…) et pour des choix qui peuvent être répétés, à grande latence (consommation, conduite d'une exploitation, politique environnementale).

---

### Brève histoire de l'IA

**Funfact :** l'effet Hans — un cheval qui a « appris à compter » en analysant en réalité le comportement de son maître, pour savoir quand il fallait arrêter de compter.

**Date du premier ordinateur : 1840.**
Il s'agissait de la machine analytique de Charles Babbage (projet des années 1830-1840, avec Ada Lovelace).

**Funfact :** elle ne fonctionnait pas, car la machine n'était pas suffisamment précise — en gros, on n'arrivait pas à savoir si un trou était bien percé ou non.

**Avant 1930 :** tout est force et énergie.
**Après 1930 :** tout devient information et code — c'est la **« cybernétique »** (théorie de la communication et du codage, Claude Shannon, 1948 ; code génétique, Watson, Crick et Rosalind Franklin).

→ **Idée clé :** les êtres vivants deviennent eux aussi des systèmes de traitement de signal.

**1. Pionniers de l'IA (1936-1956) :**

- **Turing** : les fonctions calculables sont définies par une machine.
- **Von Neumann** : architecture des ordinateurs, réflexions sur les automates.
- **McCulloch & Pitts (1943) : premier modèle du neurone !!** — une unité qui reçoit des infos 0/1 et qui renvoie des 0/1. Si on est en dessous d'un certain seuil de 1, elle renvoie un 0, sinon un 1.
  **Funfact :** 86 milliards de neurones dans notre cerveau. Il n'y a rien qui différencie fondamentalement un cerveau d'un ordinateur — une IA doit être capable de faire autant que nous.
- **Bcp de conférences** : les conférences Macy, qui donnent naissance à la première cybernétique.

**« L'espoir » de cette période :** l'intelligence met en jeu des processus généraux de raisonnement.

**2. IA comme méthodes générales de raisonnement (1956-1968) :**

**IA symbolique :** manipuler des règles et symboles explicites (lisibles par un humain) pour raisonner.

- le système CHECKER, qui bat tout le monde sauf une personne aux jeux de dames
- démonstrateurs de théorèmes (principe de résolution, à l'origine de Prolog)

→ **Idée clef :** en IA symbolique, il y a plein d'étapes discrètes différentes.

**Deuxième type d'IA : le connexionnisme** — faire apprendre les poids d'un réseau de neurones (boîte noire) pour résoudre le problème.

- le Perceptron, premier modèle « entraînable » de neurone, 1957-1962

**Différence :** explicite/interprétable (symbolique) vs. appris/opaque (connexionnisme) — mais le connexionnisme gère mieux le bruit et la complexité du monde réel.

**1968 : que sait-on faire ?**

- analogie : résoudre des schémas
- Eliza : premier balbutiement de « LLM », très primitif, mais les gens se trompaient (ils pensaient que la machine les comprenait)
- reconnaissance des caractères, utilisée dans les cartes de code

**1968 : les échecs de la période**

- la traduction automatique (ex. anglais → russe → anglais, ne redonnait pas le texte de départ)
- les experts ne sont pas des experts partout
- le Perceptron s'avère limité (il ne peut pas apprendre certaines fonctions simples, comme le XOR)

**3. « Knowledge is power » (1968-1980) :**

La machine a besoin de beaucoup de connaissances : mais comment les coder ? Et comment les utiliser ?

- **Représentation des connaissances** : représentations structurées (réseaux sémantiques, scripts, schémas et frames — précurseurs des langages objets), extensions de la logique (logiques non monotones, logiques temporelles, logique floue), de nombreux systèmes « intelligents » (ARCH, AM, MAGGIE, BORIS, PLANNER…), le projet CYC.
- **Utilisation des connaissances** : les systèmes experts.

→ **Problème de fond :** comment acquérir toutes les connaissances nécessaires ? Comment généraliser des expériences souvent limitées à des « problèmes jouets » ? → l'ingénierie des connaissances est un processus très lourd, peu systématisé, et difficile à maintenir.

**Funfact :** BORIS → une IA experte pour résumer les histoires d'adultères.

→ **« L'espoir » de cette période :** l'intelligence met en jeu beaucoup de connaissances, que l'on obtiendra par des processus généraux d'apprentissage.

**4. Méthodes générales d'apprentissage (1980 - aujourd'hui) :**

**Méthode symbolique :** nouvelles techniques (algorithme d'élimination, arbres de décision, méthode de l'Étoile) et nouveaux principes (espace des versions).
Nous avons de super modèles, mais faits dans des univers parfaits ; or dans la vraie vie (industrielle), il n'y a pas le même formalisme partout, ce qui pose problème pour raisonner.

**Connexionnisme :** renouveau du connexionnisme et du mouvement subsymbolique — Hopfield (1982), le Perceptron Multi-Couches (1985), IA distribuée (algorithmes génétiques, vie artificielle).
On ne sait pas ce qu'il se passe dedans, mais ces méthodes arrivent à résoudre des données bruitées.

→ Les réseaux de neurones s'imposent.

**5. Apprentissage statistique et applications (1995 - …) :**

- nouvelles méthodes sans raisonnement explicite (réseaux de neurones, algorithmes génétiques, réseaux bayésiens, apprentissage par renforcement)
- théorie de l'apprentissage statistique (hypothèse de données i.i.d.)

→ le tout guidé par les besoins industriels — on ne parle alors plus de « connaissances ».

**Depuis ~2022 :** retour des interactions avec les experts/utilisateurs et entre algorithmes adaptatifs, possible retour de la notion de connaissance (causalité), vers une rationalité limitée et des systèmes complexes hétérogènes à longue vie.

---

**L'histoire de l'IA en 4 points :**

- **1956-1969** : méthodes générales de raisonnement (démonstrateurs de théorèmes, logiques ; limites).
- **1970-1985** : les systèmes experts (représentation des connaissances : langages objets, réseaux sémantiques ; systèmes experts).
- **1985-2011** : méthodes générales d'apprentissage (méthodes symboliques, Perceptrons multi-couches, SVM, Boosting…).
- **2012 - …** : les réseaux de neurones profonds (réseaux convolutionnels, GANs, modèles génératifs et LLM).

---

## Les grands types d'apprentissage

### Apprentissage descriptif (non supervisé)

**Définition (slide) :** à propos d'un échantillon d'apprentissage S = {(xᵢ)}, on cherche à identifier des régularités qui rendent compte de S :

- sous forme de clusters (ex. mélange de Gaussiennes → clustering)
- sous forme de motifs fréquents (fouille de données)

**Définition simple :** à partir de données brutes, sans étiquettes ni indications, trouver des régularités qui les résument (groupes, clusters, motifs fréquents) — pour explorer et comprendre les données, sans savoir à l'avance ce qu'on cherche.

**Image :** on te met dans les caves du Louvre sans aucune indication, et on te demande de résumer ce qu'il y avait.

*Attention :* il y a besoin d'un expert pour confirmer les clusters définis. Généralement le clustering est utilisé dans des démarches exploratoires, et il faut faire attention car l'expert peut mal interpréter (trouver des explications même si les groupes sont dus au hasard).

→ Il y a donc une **grande dépendance au biais a priori**.

---

### Apprentissage prédictif (supervisé)

**Formalisation :** échantillon d'apprentissage S = {(x₁,y₁), (x₂,y₂), …, (xᵢ,yᵢ), …, (xₘ,yₘ)}, à partir duquel on apprend une fonction *f* (approchée par une hypothèse *h*). Pour un nouvel exemple x, on prédit y via h : x --h--> y. C'est l'apprentissage d'une fonction d'un espace d'entrée X vers un espace de sortie Y.

**Définition simple :** à partir d'exemples où entrée et sortie sont déjà liées (x, y), faire apprendre à la machine la fonction qui les relie — pour ensuite prédire y à partir de nouveaux x.

**Exemples de tâches :** spam ou pas spam ; article de politique ou de sport ; pathologie dont souffre un patient ; objet présent dans une image ; chat vs. chien.

**Différence discrimination / régression :**

- **Discrimination (classification)** : la sortie y est une classe/catégorie. Ex. prédire qu'un client va changer d'opérateur s'il cumule plus de 300 €/mois d'appels internationaux ET plus de 3 réclamations passées.
- **Régression** : la sortie y est une valeur continue. Ex. le nombre d'accidents déclarés par un conducteur est inversement proportionnel à l'ancienneté de son permis, avec des coefficients propres à chaque genre.

---

### Apprentissage prescriptif, pour « intervenir »

En quoi diffère-t-il du prédictif ? La sortie peut varier.

**Définition simple :** chercher des liens de **causalité** (pas juste de corrélation) pour savoir quelle action entreprendre — répondre à « que se passerait-il si j'intervenais ainsi ? »

**Exemples :**

- quelles recommandations faire à un consommateur pour qu'il baisse sa consommation d'aliments carnés ?
- quel serait l'impact si on doublait le prix de … ?
- quel rendement aurais-je eu l'année dernière si j'avais planté telle culture au lieu de telle autre ?

*Attention :* ne pas tomber dans le piège de la simple corrélation — on observe que les gens qui mangent des glaces sont souvent en maillot de bain ; si je veux vendre plus de glaces, il serait absurde d'en déduire qu'il faut demander aux gens de mettre un maillot de bain (confusion corrélation/causalité).

---

### Apprentissage par renforcement

**Formalisation :** les données d'apprentissage sont une séquence de perceptions, actions et récompenses (sₜ, aₜ, rₜ) pour t = 1, …, ∞, où la récompense rₜ peut dépendre d'actions passées, parfois bien antérieures à t. Le problème est d'inférer une fonction « situation perçue → action » de façon à maximiser un gain à long terme. C'est proche de l'apprentissage de réflexes.

Un agent dispose d'un canal de perception, d'un canal de récompense, et d'un canal d'action.

**Définition simple :** un agent apprend, par essais-erreurs, à agir dans un environnement (perception → action → récompense) de façon à maximiser un gain à long terme.

**Exemples :** tennis de table, hélicoptère, jeu de Go…

*Attention :* le système ne sait plus quoi faire si on change ne serait-ce qu'un tout petit peu les paramètres ; exemple : le tireur de pénalty qui ne sait plus tirer si le gardien fait n'importe quoi.

---

### Organisation des données

**Exemple de tableau de données** — chaque ligne est un exemple/instance, chaque colonne (sauf la dernière) est un descripteur/attribut (*feature*), la dernière colonne est l'étiquette/label :

| Identifiant | Genre | Age | Niveau études | Marié ? | Nb enfants | Revenu | Profession | À prospecter ? |
|---|---|---|---|---|---|---|---|---|
| I_21 | M | 43 | Bac+5 | Oui | 3 | 55 000 | Architecte | OUI |
| I_34 | M | 25 | Bac+2 | Non | 0 | 21 000 | Infirmier | NON |
| I_38 | F | 34 | Bac+8 | Oui | 2 | 35 000 | Chercheuse | OUI |
| I_39 | F | 67 | Bac | Oui | 5 | 20 000 | Retraitée | NON |

**Types de données rencontrées :** vectorielles, séquences, structurées, temporelles, spatiales.

---

## Problème de l'apprentissage supervisé

### Problème de l'induction

**Induction :** le fait de généraliser à partir d'exemples particuliers pour en tirer une règle générale (ex. « quelques émeraudes vertes » → « toutes les émeraudes sont vertes ») — c'est le principe même de l'apprentissage, mais qui nécessite toujours un biais pour être possible (sans biais, aucune généralisation n'est justifiée).

**Exemple chiffré donné en cours :**

Avec 4 attributs binaires (nombre 1/2 ; taille petit/grand ; forme cercle/carré ; couleur rouge/vert), il existe 2^(2⁴) = 2¹⁶ = 65 536 fonctions possibles de X vers Y.
Après 6 exemples d'apprentissage, il en reste encore 2¹⁰ = 1024.
Avec seulement 2 attributs, il y a 2^(2²) = 16 fonctions possibles, et il n'en reste plus que 2¹ = 2 après 3 exemples différents.

→ Sans biais supplémentaire, on ne peut pas choisir entre les hypothèses restantes.

*(Nous, humains, arrivons à généraliser car nous sommes tous biaisés — c'est le fait que les biais non efficaces sont morts au fil de l'évolution — donc nous sommes capables de trouver des hypothèses et de retrouver les solutions.)*

Tant qu'on n'a pas toutes les solutions à tous les cas, on ne peut pas être sûr. La question est donc : comment faire de la prédiction ?

**Citation (Leslie Valiant, « Probably Approximately Correct », 2013) :**
*« la généralisation/induction est un phénomène aussi routinier et reproductible que la chute des objets sous la gravité — il est donc raisonnable d'en attendre une explication scientifique quantitative. »*

Il est donc essentiel d'avoir un biais pour pouvoir faire de l'induction ; le biais doit être aligné avec la façon dont le monde réel est structuré.

**L'induction pose une double question :**

1. comment trouver de bonnes règles/hypothèses ? *(problème de l'invention)*
2. peut-on garantir quelque chose sur ces généralisations ? *(problème de la justification)*

---

### Point sur les biais

**Un biais**, c'est tout ce qui limite l'espace des hypothèses considérées, et qui favorise certaines hypothèses par rapport à d'autres.

**Types de biais :**

- **de représentation** *(déclaratif)* : limiter les possibilités (l'espace des hypothèses considérées).
- **de recherche** *(procédural)* : faire varier l'ordre d'exploration des données pour favoriser une hypothèse plutôt qu'une autre.

---

### Sous-apprentissage, bon apprentissage, sur-apprentissage

Comment garantir un niveau de performance ? Trois cas de figure typiques quand on ajuste une hypothèse h aux données :

- **Sous-apprentissage (underfitting)** : l'hypothèse est trop simple, elle ne capture pas assez la structure des données.
- **Bon apprentissage** : l'hypothèse capture bien la vraie structure sous-jacente, sans coller aux détails/au bruit.
- **Sur-apprentissage (overfitting)** : l'hypothèse colle trop aux exemples d'apprentissage (y compris au bruit), et généralise mal à de nouveaux exemples.

---

### No free-lunch theorem

*Toutes les méthodes d'induction se valent et sont égales au hasard.*

**Explication :** en moyenne sur l'ensemble de tous les problèmes possibles, tous les algorithmes d'apprentissage ont une performance équivalente — y compris à du hasard. Autrement dit, il **n'existe pas d'algorithme universellement meilleur que les autres sur tous les problèmes**. Il faut donc choisir le bon algorithme (le bon biais) en fonction de la classe de problèmes étudiée.

Cela mène à des **garanties « de lampadaire »** : une image pour dire qu'on ne peut donner que des garanties **conditionnelles**, pas absolues.

L'idée vient de la blague de l'ivrogne qui cherche ses clés sous le lampadaire — pas parce qu'il les a perdues là, mais parce que c'est le seul endroit où il y a de la lumière.

En apprentissage, c'est pareil :

- **Si** le monde réel satisfait les hypothèses qu'on a posées sur lui (le fameux « biais »),
- **Alors** l'algorithme produira une bonne hypothèse, proche de la vérité.
- **Mais si ce n'est pas le cas** (ex. le monde n'est pas « parcimonieux » comme on l'a supposé), l'algorithme peut produire de très mauvaises hypothèses — et on n'a aucune garantie dans ce cas.

Autrement dit : on ne peut garantir le succès que « là où la lumière porte » (là où nos hypothèses sont valides), pas partout. C'est directement lié au no free-lunch theorem : il n'y a pas de garantie universelle, seulement des garanties sous conditions.

---

### Les trois ingrédients de l'apprentissage artificiel

Ce sont 3 choses que tu vas voir forcément dans les articles.  
1. **Le choix de l'espace des hypothèses H** : en général H ≠ F (F étant l'espace de toutes les fonctions possibles).
2. **Le critère inductif** : comment évaluer chaque hypothèse en fonction de l'échantillon S.
3. **La méthode d'exploration de H** : comment trouver une bonne (voire optimale) hypothèse dans cet espace.

Quelle hypothèse choisir parmi toutes celles de H ? Il faut un moyen de calculer la performance d'une hypothèse, c'est-à-dire le coût d'une erreur de prédiction : c'est ce qu'on appelle une **fonction de perte** l(h(x), y), qui mesure l'écart entre la prédiction h(x) et la vraie valeur y.

À partir de cette fonction de perte, on peut définir deux notions de coût :

- Le **risque réel** (ou risque théorique) R(h) : c'est le coût moyen qu'on obtiendrait en espérance, si on connaissait la vraie distribution de probabilité p(x,y) des données :
  R(h) = ∫∫ₓᵧ l(h(x), y)·p(x,y) dx dy
  Problème : en pratique, on ne connaît jamais p(x,y) ! C'est justement ce qu'on cherche à apprendre.
- Le **risque empirique** R̂(h) : c'est l'erreur moyenne mesurée sur l'échantillon d'apprentissage S dont on dispose réellement (m exemples) :
  R̂(h) = (1/m) ∑ᵢ₌₁ᵐ l(h(xᵢ), yᵢ)

L'idée centrale de l'apprentissage statistique est d'utiliser le risque empirique R̂(h) (calculable, car on a les données) comme une **estimation** du risque réel R(h) (non calculable directement). Toute la question est de savoir à quel point cette estimation est fiable, et sous quelles conditions on peut garantir qu'une hypothèse qui minimise bien le risque empirique aura aussi un bon risque réel (c'est-à-dire qu'elle généralisera bien). C'est précisément l'objet de l'analyse dite **PAC Learning** (Probably Approximately Correct), formalisée notamment par Leslie Valiant, qui donne des garanties statistiques sur l'écart entre risque empirique et risque réel en fonction du nombre d'exemples et de la complexité de l'espace d'hypothèses H.

---

## Apprendre dans un espace d'hypothèses structuré

### L'apprentissage supervisé vu comme une recherche dans un espace

Une autre façon de voir l'apprentissage supervisé est de le considérer comme un jeu entre deux espaces :
- l'**espace des exemples** X, dans lequel se trouve le concept cible (la vraie frontière qu'on cherche à retrouver, inconnue) ;
- l'**espace des hypothèses** H, dans lequel l'algorithme explore les hypothèses candidates pour essayer d'approcher ce concept cible.

Apprendre revient donc à **chercher**, dans l'espace H, l'hypothèse qui approxime le mieux le concept cible inconnu. La question devient alors : comment contrôler intelligemment cette exploration de H, plutôt que de tester toutes les hypothèses une par une ?

### H est structuré : la relation de généralité

L'observation clé de cette partie est que l'espace des hypothèses H n'est pas une simple liste désordonnée : il est **structuré** par une relation de généralité, elle-même induite par la relation d'inclusion dans l'espace des exemples X. Concrètement, une hypothèse h₁ est dite **plus générale** qu'une hypothèse h₂ si l'ensemble des exemples couverts par h₁ inclut l'ensemble des exemples couverts par h₂ (h₁ "couvre" plus de cas que h₂). Cette relation définit un **ordre partiel** sur H : on peut se représenter H comme un treillis (un "diamant"), avec les hypothèses les plus générales en haut (celle qui couvre tout, notée ⊤) et les plus spécifiques en bas (celle qui ne couvre rien, notée ⊥).

Pour qualifier une hypothèse par rapport aux exemples d'apprentissage, on utilise ce vocabulaire :
- une hypothèse est **complète** si elle couvre bien tous les exemples positifs ;
- une hypothèse est **correcte** si elle ne couvre aucun exemple négatif ;
- une hypothèse **complète ET correcte** est dite **consistante** (ou cohérente) avec les données : c'est exactement ce qu'on cherche.

### Les opérateurs pour explorer H

Puisque H est structuré par cette relation de généralité, on peut le parcourir à l'aide de trois types d'opérateurs :
- la **généralisation** : transforme une description en une description plus générale (ex : supprimer une condition d'une conjonction "ferrari & rouge → chère" devient "ferrari → chère" ; ou ajouter une alternative ; ou étendre l'intervalle de valeurs autorisées pour un descripteur ; ou remonter dans une hiérarchie de valeurs, par exemple "chlore" et "brome" généralisés en "halogène") ;
- la **spécialisation** : l'opération inverse, qui restreint une description (produit une description plus contrainte, donc une conséquence logique de la description initiale) ;
- la **reformulation** : transforme une description en une nouvelle description logiquement équivalente (sans la rendre ni plus générale ni plus spécifique).

### L'espace des versions (Version Space)

Étant donné un échantillon d'exemples positifs et négatifs, l'ensemble de toutes les hypothèses de H qui sont consistantes avec ces exemples (complètes ET correctes) forme ce qu'on appelle **l'espace des versions** (Version Space). C'est le sous-ensemble de H qui reste "compatible" avec tout ce qu'on a observé jusque-là.

Le problème, c'est que cet espace des versions peut contenir énormément d'hypothèses (voire une infinité) — on ne va évidemment pas toutes les énumérer une par une ! L'observation fondamentale qui résout ce problème est la suivante : **grâce à la structure d'ordre partiel de H, l'espace des versions peut être entièrement caractérisé par seulement deux frontières (bornes)**, sans avoir à lister toutes les hypothèses intermédiaires :

- le **G-set** (borne supérieure) = l'ensemble des hypothèses **les plus générales** qui sont consistantes avec les exemples d'apprentissage ;
- le **S-set** (borne inférieure) = l'ensemble des hypothèses **les plus spécifiques** qui sont consistantes avec les exemples d'apprentissage.

**Pourquoi ça suffit ?** Parce que toute hypothèse "entre" S et G (au sens de la relation de généralité — plus générale qu'au moins un élément de S, et plus spécifique qu'au moins un élément de G) est automatiquement garantie d'être, elle aussi, consistante avec les exemples déjà vus. On n'a donc besoin de mémoriser que ces deux frontières S et G pour représenter tout l'espace des versions de façon compacte : c'est une représentation extrêmement économique en mémoire, comparée à l'énumération complète des hypothèses.

### Comment construit-on S et G ? L'algorithme d'élimination des candidats (Candidate Elimination)

L'idée de l'algorithme est de maintenir à jour S et G au fur et à mesure que de nouveaux exemples arrivent, en les faisant "converger" l'un vers l'autre.

**Initialisation :** S et G sont initialisés respectivement par l'ensemble des hypothèses les plus spécifiques (pour S) et les plus générales (pour G) qui sont consistantes avec le tout premier exemple positif fourni. Concrètement, S part de l'hypothèse la plus restrictive possible (l'exemple positif lui-même, tel quel), et G part de l'hypothèse la plus large possible (celle qui accepte tout, notée avec des "?" partout).

**Pour chaque nouvel exemple (positif ou négatif), on met à jour S et G :**

*Mise à jour de S (la frontière spécifique) :*
- si l'exemple est **négatif** : on élimine de S les hypothèses qui couvrent (à tort) cet exemple négatif ;
- si l'exemple est **positif** : on généralise **le moins possible** (minimalement) les hypothèses de S qui ne couvrent pas encore cet exemple, pour qu'elles le couvrent désormais ; puis on élimine de S les hypothèses qui couvriraient un exemple négatif déjà vu, et/ou celles qui seraient devenues plus générales qu'une autre hypothèse de S.

*Mise à jour de G (la frontière générale) :*
- si l'exemple est **positif** : on élimine de G les hypothèses qui ne couvrent pas cet exemple ;
- si l'exemple est **négatif** : on spécialise **le moins possible** (minimalement) les hypothèses de G qui couvrent (à tort) cet exemple négatif, pour qu'elles ne le couvrent plus ; puis on élimine de G les hypothèses qui ne seraient plus plus-générales qu'au moins une hypothèse de S, et/ou celles qui seraient devenues plus spécifiques qu'une autre hypothèse de G.

On répète ce processus jusqu'à convergence, ou jusqu'à ce que S = G = ∅ (ce qui signifierait qu'aucune hypothèse cohérente n'existe dans H pour expliquer les données — par exemple si les données sont bruitées, ou si le concept cible n'est pas exprimable dans H).

### Exemple travaillé : le jeu de golf (« Enjoy Sport »)

Voici l'exemple classique donné en exercice pour illustrer l'algorithme (à ne pas confondre avec le système LEX présenté juste après dans le cours, voir plus bas). On veut apprendre dans quelles conditions météo on a envie de jouer au golf, à partir d'exemples décrits par 6 attributs :

ciel × AirTemp × Humidité × Vent × Eau × Prévision

avec :
- ciel → soleil, nuage, pluie
- AirTemp → chaud, froid
- humidité → normal, élevé
- vent → fort, faible
- eau → chaude, fraîche
- prévision → égale, change

Petite remarque sur la taille de l'espace des hypothèses : si on autorisait n'importe quelle disjonction de valeurs, il y aurait 3×2⁵ = 96 valeurs possibles au total, donc 2⁹⁶ fonctions possibles — bien trop pour être exploré. C'est pourquoi on introduit un **biais de langage** (un biais de représentation) : on interdit les disjonctions, et on limite les hypothèses à de simples conjonctions d'attributs (chaque attribut prenant soit une valeur précise, soit "?" = n'importe laquelle, soit ∅ = aucune). Avec ce biais, il ne reste plus que 4×3⁵ = 972 hypothèses possibles (en comptant l'hypothèse vide ∅ et le joker "?" pour chaque attribut) — un espace bien plus raisonnable à explorer.

**Déroulé de l'algorithme :**

*Exemple 1 (+) : Soleil × chaud × normal × Fort × chaude × égale*  
S1 = {(Soleil × chaud × normal × Fort × chaude × égale)} 
— on initialise S avec l'exemple positif lui-même (hypothèse la plus spécifique possible).  
G1 = {(? × ? × ? × ? × ? × ?)}  
 — G reste l'hypothèse la plus générale, puisqu'elle couvre déjà cet exemple positif.

*Exemple 2 (+) : Soleil × chaud × élevé × Fort × chaude × égale*  
S2 = {(Soleil × chaud × ? × Fort × chaude × égale)}   
— seul l'attribut Humidité diffère entre S1 et ce nouvel exemple positif (normal vs élevé) ; on généralise minimalement uniquement cet attribut en "?".  
G2 = {(? × ? × ? × ? × ? × ?)}   
— inchangé, un exemple positif ne remet pas en cause G tant qu'il est couvert.

*Exemple 3 (−) : Pluie × froid × élevé × Fort × chaude × change*  
S3 = S2 = {(Soleil × chaud × ? × Fort × chaude × égale)}  
— cet exemple négatif n'est de toute façon pas couvert par S2 (Soleil ≠ Pluie), donc S ne bouge pas.  
G3 = {(Soleil × ? × ? × ? × ? × ?), (? × chaud × ? × ? × ? × ?), (? × ? × ? × ? × ? × égale)}  
— G2 couvrait cet exemple négatif (il couvre tout), il faut donc le spécialiser minimalement. On cherche toutes les façons de fixer UN SEUL attribut pour exclure cet exemple négatif, tout en restant cohérent avec S2 (c'est-à-dire en restant plus général qu'au moins une hypothèse de S) : fixer le ciel à Soleil, OU l'AirTemp à chaud, OU la Prévision à égale — ce sont les 3 seules spécialisations à un seul attribut qui excluent l'exemple 3 tout en restant compatibles avec S2.

*Exemple 4 (+) : Soleil × chaud × élevé × Fort × Fraîche × change*  
S4 = {(Soleil × chaud × ? × Fort × ? × ?)} 
— deux attributs diffèrent entre S3 et ce nouvel exemple (Eau : chaude vs fraîche ; Prévision : égale vs change), on les généralise tous les deux minimalement en "?".   
G4 = {(Soleil × ? × ? × ? × ? × ?), (? × chaud × ? × ? × ? × ?)}  
— parmi les 3 hypothèses de G3, celle qui fixait la Prévision à "égale" ne couvre plus ce nouvel exemple positif (dont la prévision est "change") : on l'élimine. Les deux autres couvrent bien l'exemple 4, elles sont conservées.  


**Bilan à ce stade (avec ces 4 exemples, le jeu de données classique de cet exercice) :**  
- S4 = {(Soleil × chaud × ? × Fort × ? × ?)}  
- G4 = {(Soleil × ? × ? × ? × ? × ?), (? × chaud × ? × ? × ? × ?)}  

On voit que S4 ≠ G4 : l'algorithme n'a **pas encore convergé**, il reste plusieurs hypothèses possibles dans l'espace des versions (toutes celles qui sont "entre" S4 et G4, plus générales que S4 et plus spécifiques qu'au moins une hypothèse de G4). Pour trancher davantage, il faudrait fournir de nouveaux exemples au système. Si un jour S = G (avec une seule hypothèse commune), l'algorithme a convergé vers une unique hypothèse consistante avec toutes les données observées.

### Propriétés et questions ouvertes sur l'algorithme

Quelques questions importantes à garder en tête sur cet algorithme :
- il est **incrémental** (on peut lui donner les exemples un par un, sans tout reprendre à zéro) ;
- quelle est sa **complexité** (en nombre d'hypothèses à maintenir dans S et G) ?
- comment utiliser le résultat **si la convergence n'a pas lieu** (plusieurs hypothèses restent possibles, comme dans l'exemple du golf ci-dessus) ?
- que signifie le cas où **S = G = ∅** ? (aucune hypothèse de H n'est cohérente avec les données — le concept cible n'est peut-être pas représentable dans l'espace d'hypothèses choisi, ou les données sont bruitées)
- l'algorithme pourrait-il être utilisé en mode **apprentissage actif** (c'est-à-dire en choisissant intelligemment quel exemple demander ensuite, plutôt que de les recevoir passivement) ?
- que faire si les **données sont bruitées** (un exemple mal étiqueté peut faire s'effondrer tout l'espace des versions à S = G = ∅) ?

### Illustration : le système LEX

À ne pas confondre avec l'exemple du golf ci-dessus : LEX est un vrai système historique qui a appliqué ce type d'approche pour apprendre des **heuristiques de résolution de problèmes** (en l'occurrence, des problèmes d'intégration symbolique en mathématiques). Le principe repose sur un cycle en quatre étapes :

**Génération de problèmes** → **Résolution de problèmes** (LEX tente de résoudre l'exercice généré, avec ses heuristiques déjà partiellement apprises) → **Critique** (analyse de la trace détaillée de la tentative de résolution) → **Généralisation** (chaque tentative de résolution devient un exemple d'apprentissage, qui vient affiner les heuristiques du système) → et le cycle repart vers la génération d'un nouveau problème.

C'est une bonne illustration du fait que l'algorithme d'élimination des candidats ne s'applique pas qu'à des exemples "statiques" fournis d'avance (comme le golf) : il peut aussi tourner en boucle, avec un système qui génère lui-même ses propres exemples d'apprentissage au fil de son expérience.

---

## Conclusion

Pour conclure ce premier cours d'introduction :

- Il existe plusieurs grands types de tâches d'apprentissage : supervisé, non-supervisé, par renforcement — mais aussi, plus généralement, l'apprentissage semi-supervisé, le ranking, l'apprentissage en ligne (*on-line learning*), etc.
- L'apprentissage supervisé peut se résumer à une **recherche dans un espace d'hypothèses** structuré, ce qui soulève plusieurs questions à se poser systématiquement face à un nouveau problème :
  - Comment coder les données d'entrée ? Quelle représentation choisir ?
  - Comment choisir l'espace des hypothèses H ?
  - Comment évaluer les hypothèses (quel critère inductif, quelle fonction de perte) ?
  - Comment explorer H efficacement (quelle méthode de recherche) ?

Ces quatre questions constituent en réalité le fil conducteur de tout le reste du cours : chaque nouvelle méthode d'apprentissage qu'on verra (arbres de décision, SVM, réseaux de neurones...) n'est jamais qu'une réponse particulière à ces mêmes quatre questions.