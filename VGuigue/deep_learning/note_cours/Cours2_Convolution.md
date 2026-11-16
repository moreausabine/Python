# Cours2_Convolution — Réseaux convolutionnels pour l'image

> Notes explicatives du cours de Vincent Guigue (AgroParisTech / INRAE / Paris-Saclay).
> Plan du cours : **Introduction → Convolution → Vision → Introspection → GNN**.
> À la fin : une section **« Boîte à outils PyTorch »** avec les commandes de base.

---

## Sommaire

1. [Introduction : les tâches en Computer Vision](#1-introduction--les-tâches-en-computer-vision)
2. [Pourquoi les MLP ne suffisent pas](#2-pourquoi-les-mlp-ne-suffisent-pas)
3. [Avant le deep learning : Bag of Visual Words](#3-avant-le-deep-learning--bag-of-visual-words)
4. [L'idée des réseaux convolutifs](#4-lidée-des-réseaux-convolutifs)
5. [La convolution 1D (pour bien comprendre)](#5-la-convolution-1d-pour-bien-comprendre)
6. [La convolution 2D](#6-la-convolution-2d)
7. [Le pooling](#7-le-pooling)
8. [La couche convolutionnelle « usuelle » et LeNet](#8-la-couche-convolutionnelle-usuelle-et-lenet)
9. [Récapitulatif des paramètres et calcul des dimensions](#9-récapitulatif-des-paramètres-et-calcul-des-dimensions)
10. [Fine-tuning et transfer learning](#10-fine-tuning-et-transfer-learning)
11. [Les grandes architectures](#11-les-grandes-architectures)
12. [Types de problèmes, modèles et métriques](#12-types-de-problèmes-modèles-et-métriques)
13. [Vision Transformer (ViT)](#13-vision-transformer-vit)
14. [CNN et représentations temps-fréquence](#14-cnn-et-représentations-temps-fréquence)
15. [Introspection : comprendre ce que voit un CNN](#15-introspection--comprendre-ce-que-voit-un-cnn)
16. [Des convolutions aux graphes (GNN)](#16-des-convolutions-aux-graphes-gnn)
17. [Boîte à outils PyTorch (les bases des bases)](#17-boîte-à-outils-pytorch-les-bases-des-bases)
18. [Fiche de révision express](#18-fiche-de-révision-express)

---

## 1. Introduction : les tâches en Computer Vision

Le **Computer Vision** consiste à faire « comprendre » des images à une machine. Selon ce qu'on veut en tirer, on parle de tâches différentes, qui se distinguent par **la finesse de la réponse attendue** :

| Tâche | Question posée | Sortie |
|---|---|---|
| **Classification** (object recognition) | « Qu'y a-t-il dans l'image ? » | Une étiquette (ex. *Chat*) |
| **Classification + localisation** | « Quoi, et où ? » (un seul objet) | Étiquette + boîte englobante |
| **Détection d'objets** | « Quels objets, et où ? » (plusieurs) | Liste de (classe, boîte) |
| **Segmentation sémantique** | « À quelle classe appartient *chaque pixel* ? » | Une image de labels (chat / balle / fond) |
| **Segmentation d'instances** | Comme ci-dessus, mais en distinguant *chaque individu* | Un masque par objet (chat 1, chat 2…) |
| **Image captioning** | « Décris l'image » | Une phrase |
| **Visual Question Answering** | « Combien de balles ? » | Une réponse en texte |
| **Génération d'images** | « Dessine un chat qui joue avec une balle » | Une image |

**Repère utile** : en segmentation sémantique, on ne parle pas d'« objets », seulement de *pixels* (deux chats collés = une seule zone « chat »). En segmentation d'instances, on sépare chaque chat.

Le domaine est très large : suivi d'objets, estimation de pose, navigation autonome, reconstruction 3D, compréhension de foules, scènes urbaines, relevés aériens, etc.

---

## 2. Pourquoi les MLP ne suffisent pas

Un **MLP (perceptron multicouche / fully-connected)** relie *chaque* entrée à *chaque* neurone. Appliqué à une image, cela pose plusieurs problèmes :

1. **Une image n'est qu'une grille de pixels** → sémantique très pauvre : un pixel isolé ne veut rien dire, c'est le *voisinage* qui crée du sens (un bord, une texture, un œil…).
2. **Dans un MLP, une entrée = un rôle propre** : le pixel (12, 45) a ses propres poids, distincts du pixel (13, 45). Le réseau est donc **dépendant de la position du pixel**.
3. Or, **dans une image, l'absolu n'a pas d'importance, le relatif est bien plus important** : un chat reste un chat qu'il soit à gauche ou à droite de l'image.
4. **Explosion du nombre de paramètres** : une image 224×224×3 = 150 528 entrées. Une seule couche dense de 1000 neurones = ~150 millions de poids. Ingérable et sujet au surapprentissage.

### Les invariants

Un objet peut apparaître sous de **multiples formes** : angle de vue, pose, éclairage, occlusion (partiellement caché), échelle, environnement. On veut un modèle **invariant** (ou robuste) à ces variations.

> 💡 **Intuition** : on veut un modèle qui dit « il y a un œil *quelque part* » plutôt que « il y a un œil exactement au pixel (120, 85) ».

---

## 3. Avant le deep learning : Bag of Visual Words

Avant les CNN, la pipeline classique avait **trois étapes** (inspirée du Bag of Words en texte) :

1. **Extraire un dictionnaire de features** : détecter des points clés (*keypoints*) dans l'image, ou quadriller en grille, et décrire chaque petit patch par un vecteur (descripteur, ex. SIFT).
2. **Agréger les features** : on fait un *clustering* (ex. k-means) de tous les descripteurs → chaque cluster est un **« mot visuel »** (un « vocabulaire »). Chaque image devient un **histogramme** du nombre d'occurrences de chaque mot visuel.
3. **Classifier** : un classifieur classique (SVM, etc.) est entraîné sur la **matrice des données** (lignes = images, colonnes = mots visuels).

**Limite** : les features sont **fixées à la main** (non apprises), et on perd l'information spatiale. Le deep learning va apprendre *les features et le classifieur ensemble*.

---

## 4. L'idée des réseaux convolutifs

Architecture typique (schéma NVidia) :

```
Image → [Convolution + ReLU → Pooling] → [Convolution + ReLU → Pooling] → ... → Flatten → Fully connected → Softmax
        \______________ Feature learning ______________/                         \_____ Classification _____/
```

- **Partie « feature learning »** : empilement de blocs *convolution + ReLU + pooling* qui extraient automatiquement des caractéristiques de plus en plus abstraites.
- **Partie « classification »** : on aplatit (*flatten*), puis couches denses, puis *softmax* pour obtenir des probabilités de classes.

### Les 3 atouts de la convolution

| Atout | Explication |
|---|---|
| **Peu de paramètres** | Un même petit filtre (ex. 3×3 = 9 poids) est réutilisé partout sur l'image : on **partage les poids**. |
| **Apprentissage des motifs à extraire** | Les valeurs du filtre ne sont pas choisies à la main : elles sont **apprises par rétropropagation**. |
| **Agrégation progressive des échelles** | Les premières couches voient de petits détails (contours), les suivantes combinent ces détails en formes, puis en objets. |

---

## 5. La convolution 1D (pour bien comprendre)

On commence par un signal 1D (ex. une courbe, un spectre) car c'est plus simple à visualiser.

### Le principe : une fenêtre glissante

Un **filtre** (petit vecteur de poids) **glisse** le long du signal. À chaque position, on calcule le **produit scalaire** entre le filtre et la portion du signal qu'il recouvre → ça donne **une valeur de sortie**. On obtient ainsi un nouveau signal (une *feature map*).

```
Signal :   [ 2  1  0  3  1  2  ... ]
Filtre :   [ w1 w2 w3 ]
             └────┘  ← position 1 : sortie[0] = 2·w1 + 1·w2 + 0·w3
                └────┘ ← position 2 : sortie[1] = 1·w1 + 0·w2 + 3·w3
```

**Ce que « détecte » un filtre** : il est grand quand le signal ressemble à son motif. Un filtre appris peut ainsi détecter un pic, une montée, une oscillation…

### Plusieurs filtres, puis pooling, puis couches suivantes

Le cours montre la construction progressive du schéma :

1. **Un filtre** glisse → une feature map.
2. **Plusieurs filtres** (couleurs différentes) → plusieurs feature maps : chaque filtre détecte un motif différent.
3. **Pooling (max / average)** : on résume les valeurs par fenêtres (cf. §7) → on réduit la taille et on gagne en *invariance* à la localisation précise.
4. **Pooling global** : on résume *tout* le signal en une valeur par filtre → on obtient un vecteur de taille fixe (= nombre de filtres), même si le signal est de longueur variable.
5. **Fully connected** : ce vecteur alimente un classifieur.
6. **On empile** : de nouveaux filtres glissent sur les sorties de la première couche (ils combinent des motifs simples en motifs complexes).

La **rétropropagation (error back-propagation)** met à jour *les coefficients des filtres* pour que les motifs détectés soient utiles à la tâche finale.

### Les paramètres clés : largeur / stride / padding

- **Largeur (kernel size)** : taille de la fenêtre.
- **Stride (pas)** : de combien on décale le filtre à chaque étape (stride 1 = on avance d'un cran ; stride 2 = on saute un cran sur deux → sortie 2× plus courte).
- **Padding** : on ajoute des zéros aux bords pour pouvoir centrer le filtre sur les extrémités et contrôler la taille de sortie.

### Invariance vs dépendance à la localisation

- La convolution est **équivariante à la translation** : si le motif se décale dans l'entrée, la réponse se décale pareillement dans la sortie.
- Le **pooling** apporte une **(quasi-)invariance** : la position exacte compte moins.
- Mais en combinant avec des couches denses finales, le réseau garde une certaine **dépendance à la localisation** (utile ou non selon la tâche).

### Champ réceptif et convolutions dilatées

Le **champ réceptif (receptive field)** d'un neurone = la zone de l'entrée qui influence sa valeur.

- Empiler des convolutions 3 → champ réceptif qui grandit (3, puis 5, 7…).
- **Convolution dilatée** (Yu, 2016) : le filtre « saute » des cases (ex. dilatation 2 : il regarde les positions 0, 2, 4). Le schéma du cours : dilatations 1, 2, 4 avec un filtre de taille 3 → champs réceptifs de **3, 7, 15**. On voit très large **sans ajouter de paramètres** ni réduire la résolution. Très utile en segmentation.

---

## 6. La convolution 2D

Même idée, mais le filtre est une petite **matrice** (ex. 3×3) qui glisse dans les deux directions de l'image.

### Exemple détaillé du cours

Image 5×5 :

```
1 1 1 0 0
0 1 1 1 0
0 0 1 1 1
0 0 1 1 0
0 1 1 0 0
```

Filtre 3×3 :

```
1 0 1
0 1 0
1 0 1
```

**Position en haut à gauche** : on superpose le filtre sur les 3×3 premières cases et on fait la somme des produits *terme à terme* :

```
Zone :        Filtre :       Produits :
1 1 1         1 0 1          1 0 1
0 1 1    ×    0 1 0    =     0 1 0     → somme = 1+1+1+1 = 4
0 0 1         1 0 1          0 0 1
```

On décale ensuite d'un pixel vers la droite, etc. On obtient la **carte convoluée (convolved feature)** 3×3 :

```
4 3 4
2 4 3
2 3 4
```

Taille de sortie : (5 − 3 + 0)/1 + 1 = **3**. ✔

> 📝 **Note technique** : ce que les librairies appellent « convolution » est en fait une *corrélation croisée* (le filtre n'est pas retourné). Comme les filtres sont appris, cela ne change rien en pratique.

### Images en couleur : les canaux

Une image RGB a **3 canaux**. Un filtre 2D « complet » a donc la forme `(in_channels × K × K)` : il couvre **tous les canaux d'entrée** à la fois, et la somme des produits donne **une seule valeur**.
Avec `out_channels` filtres, on obtient `out_channels` cartes de sortie (un « volume » en profondeur).

```
Entrée : (3, H, W)  ──  16 filtres de (3×5×5)  ──►  Sortie : (16, H', W')
```

---

## 7. Le pooling

**Rôle** : **réduire la dimensionnalité** de la sortie (sous-échantillonnage / *subsampling*), réduire le calcul, et ajouter de l'invariance aux petites translations.

| Type | Opération sur une fenêtre |
|---|---|
| **Max Pooling** | on garde le **maximum** (« le motif est-il présent quelque part dans la fenêtre ? ») |
| **Average Pooling** | on prend la **moyenne** |
| **Sum Pooling** | on fait la **somme** |

**Exemple du cours** : max pooling avec fenêtre 2×2 et stride 2 :

```
1 1 | 2 4              6 | 8
5 6 | 7 8     ──►     ───┼───
----+----              3 | 4
3 2 | 1 0
1 2 | 3 4
```

Chaque bloc 2×2 est remplacé par son max : 6, 8, 3, 4. L'image est divisée par 2 en hauteur et en largeur.

> 💡 **Le pooling n'a aucun paramètre à apprendre.**

---

## 8. La couche convolutionnelle « usuelle » et LeNet

### Le bloc de base

```
Input → Convolution (transformation affine) → Non-linéarité (ReLU) → Pooling → couche suivante
```

- **Convolution** : extrait des motifs (partie linéaire).
- **ReLU** : `max(0, x)` → apporte la non-linéarité (sans elle, empiler des convolutions reviendrait à une seule convolution).
- **Pooling** : réduit la taille.

**Exemple de dimensions (cours)** : image 11×11 → conv. 3×3 → **9×9** → max pooling (fenêtre 3×3, stride 3) → **3×3**.

### Exemple historique : LeNet-5 (reconnaissance de chiffres)

```
INPUT 32×32
 → Conv   → C1 : 6 cartes 28×28
 → Subsampling → S2 : 6 cartes 14×14
 → Conv   → C3 : 16 cartes 10×10
 → Subsampling → S4 : 16 cartes 5×5
 → Full connection → C5 : 120
 → Full connection → F6 : 84
 → Sortie : 10 classes (chiffres 0-9)
```

On observe le schéma classique : **la taille spatiale diminue, le nombre de cartes (canaux) augmente**.

### VGG (exemple de réseau profond)

Entrée 224×224×3, puis des blocs de convolutions 3×3 avec nombre de canaux croissant :
`64 → 128 → 256 → 512 → 512`, séparés par des max-pooling (résolution : 224 → 112 → 56 → 28 → 14 → 7), puis 2 couches denses de 4096 et un softmax.

---

## 9. Récapitulatif des paramètres et calcul des dimensions

### Les hyperparamètres d'une convolution (`nn.Conv2d`)

| Paramètre | Signification |
|---|---|
| `in_channels` | Nombre de canaux de l'entrée (1 = niveaux de gris, 3 = RGB, ou `out_channels` de la couche précédente) |
| `out_channels` | Nombre de **filtres** = nombre de cartes de sortie |
| `kernel_size` | Taille du filtre (3, 5, 7…) |
| `stride` | Pas de glissement |
| `padding` | Nombre de pixels (zéros) ajoutés sur chaque bord |
| `dilation` | Espacement entre les cases du filtre |

### Formule de la taille de sortie (par canal)

$$
\text{taille sortie} = \left\lfloor \frac{W - K + 2P}{S} \right\rfloor + 1
$$

- `W` = taille de l'entrée, `K` = taille du noyau, `P` = padding, `S` = stride.
- Avec une **dilatation `D`**, on remplace `K` par `K_eff = D·(K − 1) + 1`.

**Astuces à retenir :**
- `K=3, S=1, P=1` → la taille est **conservée** (« same » padding).
- `S=2` → la taille est **divisée par ~2**.
- Pas de padding (`P=0`) → la taille **diminue** de `K − 1`.

### Exemple du cours, pas à pas

```python
self.conv1 = nn.Conv2d(in_channels=1, out_channels=16,
                       kernel_size=5, stride=1, padding=0, dilation=1)
self.pool1 = nn.MaxPool2d(kernel_size=3, stride=1)
```

Pour une image 28×28 :
- Conv : (28 − 5 + 0)/1 + 1 = **24** → sortie `(16, 24, 24)`.
- Pooling : (24 − 3)/1 + 1 = **22** → sortie `(16, 22, 22)`.

### Nombre de paramètres d'une convolution

$$
\text{params} = \text{out\_channels} \times (\text{in\_channels} \times K \times K + 1)
$$

(le « +1 » est le biais de chaque filtre). Ici : 16 × (1×5×5 + 1) = **416 paramètres** — à comparer aux millions d'une couche dense !

---

## 10. Fine-tuning et transfer learning

### Problème de l'entraînement « from scratch »

Entraîner un réseau de zéro :
- donne de bons descripteurs…
- mais **demande énormément de données**,
- et la performance n'est **souvent pas optimale**.

### Solution : réutiliser un réseau pré-entraîné

Un réseau entraîné sur une grande base (ex. **ImageNet**, 1000 classes) a déjà appris des features utiles :
- **couches basses** : contours, couleurs, textures (génériques, réutilisables partout) ;
- **couches intermédiaires** : formes, parties d'objets ;
- **couches hautes** : concepts spécifiques à la tâche d'origine.

*(Visualisation de Zeiler & Fergus 2013 : Low-level → Mid-level → High-level features → classifieur entraînable.)*

### Les stratégies

| Stratégie | Description | Quand ? |
|---|---|---|
| **Extracteur de features** | On **gèle** tout le réseau, on **remplace** les couches denses finales par de nouvelles (adaptées à nos classes) et on n'entraîne **que celles-ci** | Peu de données, tâche proche |
| **Fine-tuning partiel** | On gèle les premières couches, on ré-entraîne les dernières | Données moyennes |
| **Fine-tuning complet** | On dégèle tout et on entraîne avec un *petit* learning rate | Beaucoup de données, tâche éloignée |

**Questions du cours** : *Combien de couches garder / ré-entraîner ? A-t-on besoin de tout fine-tuner ?*

### L'oubli catastrophique (catastrophic forgetting)

Si on ré-entraîne trop fort sur la tâche B, les paramètres s'éloignent de la zone de faible erreur de la tâche A (cf. schéma : θ*_A → θ*_B) : le réseau **« oublie »** ce qu'il savait faire. D'où l'intérêt de geler des couches et/ou d'utiliser un learning rate faible.

---

## 11. Les grandes architectures

### AlexNet (2012) — le déclencheur
- Convolutions 11×11, 5×5, 3×3.
- **Max-pooling, activations ReLU**.
- **Dropout** (désactive aléatoirement des neurones → limite le surapprentissage) et **data augmentation** (flips, rotations, recadrages… → plus de variété).
- A remporté ImageNet 2012 et lancé la révolution du deep learning en vision.

### VGG — la profondeur par des petits filtres
Principe : **empiler beaucoup de convolutions 3×3**. Deux convolutions 3×3 couvrent un champ réceptif 5×5 avec moins de paramètres, et ajoutent plus de non-linéarités. VGG-16 / VGG-19 (16 / 19 couches). Efficace mais **très gourmand** en paramètres et en calcul.

### ResNet — les connexions résiduelles
Problème : en rendant un réseau très profond, il devient **plus difficile à entraîner** (gradients qui disparaissent, dégradation).
Solution : le **skip connection** (flèches courbes sur le schéma) :

$$ y = F(x) + x $$

Le bloc n'apprend que la **correction** `F(x)` à ajouter à l'entrée. Le gradient peut « passer par le raccourci », ce qui permet d'entraîner des réseaux de 50, 101, 152 couches.

### Comparaison
Le graphique (An Analysis of Deep Neural Network Models for Practical Applications, 2017) compare *précision top-1* et *coût en opérations* : les ResNet / Inception offrent le meilleur compromis ; VGG est précis mais très coûteux ; AlexNet est léger mais moins précis. **Il n'y a pas de meilleur modèle absolu : tout dépend du compromis précision / calcul.**

### Reconstruction d'image et transfert de style (Gatys et al. 2016)
- On utilise un **CNN pré-entraîné** (discriminatif) comme outil de mesure.
- Idée : générer une image qui a **le contenu** d'une photo `p` et **le style** d'un tableau `a`.
- **Contenu** : comparé via les activations des couches profondes.
- **Style** : comparé via les **matrices de Gram** (corrélations entre cartes de features).
- On optimise **les pixels de l'image** `x` par descente de gradient :
  `L_total = α·L_contenu + β·L_style`.
- C'est un **problème inverse** sur le CNN (on ne modifie pas le réseau, on modifie l'image).

---

## 12. Types de problèmes, modèles et métriques

Le cours résume les trois grandes familles selon **ce que le réseau doit sortir**.

### a) Classification

- Entrée : image `I = RGB^(d×d)` → sortie : une classe `C`.
- Modèles : VGG, ResNet, etc.
- Dernière couche : `Linear(…, nb_classes)` + softmax.
- **Métriques** : accuracy, précision, rappel.
- *Question : comment passer au multi-classe (multi-label) ?* → on utilise une **sigmoïde** par classe (+ `BCEWithLogitsLoss`) au lieu d'un softmax, car plusieurs classes peuvent être vraies simultanément.
- ⇒ **Pensez transfer learning !**

### b) Détection d'objets

- Sortie : classe **+ boîte englobante** `B = (pos_x, pos_y, width, height)`.
- C'est donc **un problème de régression** (pour la boîte) combiné à de la classification.
- Modèles : **YOLO, R-CNN, Fast-R-CNN**.
- **Métrique : IoU (Intersection over Union)** :

$$ \text{IoU} = \frac{\text{aire(intersection)}}{\text{aire(union)}} $$

  IoU = 1 → boîtes identiques ; IoU = 0 → aucun recouvrement. On considère souvent une détection correcte si IoU > 0,5.

### c) Segmentation (pixel-level)

- Entrée : image → sortie : **une image** `I'` (un label par pixel).
- Modèle : **U-Net** (et SegNet) : architecture **encodeur-décodeur**.
  - **Encodeur** : convolutions + pooling → on comprend « quoi » (la résolution baisse).
  - **Décodeur** : convolutions transposées / upsampling → on remonte à la résolution d'origine.
  - **Skip connections** entre encodeur et décodeur (flèches grises) : réinjectent les détails fins pour localiser précisément les contours.
- **Spécificités par rapport à un auto-encodeur** : un auto-encodeur reconstruit *la même image* à partir d'un goulot d'étranglement ; le U-Net reconstruit *une autre image* (la carte de labels), et ses *skip connections* évitent de perdre l'information spatiale.
- **Métrique** : IoU calculée sur les masques de pixels.

---

## 13. Vision Transformer (ViT)

Idée : appliquer l'architecture **Transformer** (née en NLP) aux images.

1. On **découpe** l'image en **patchs** (ex. 16×16 pixels).
2. Chaque patch est **aplati** puis projeté linéairement (*Linear Projection of Flattened Patches*) → un vecteur (« token »).
3. On ajoute un **encodage de position** (sinon le Transformer ignore l'ordre) et un token spécial **[class]** appris.
4. Le **Transformer Encoder** (self-attention) traite la séquence.
5. Le token [class] passe dans une **MLP head** → classe.

**Question du cours : différences / complémentarités avec les CNN ?**

| | CNN | ViT |
|---|---|---|
| Biais inductif | Fort (localité, partage de poids, invariance à la translation) | Faible (apprend tout depuis les données) |
| Champ de vision | Local, grandit avec la profondeur | **Global dès la 1ʳᵉ couche** (self-attention entre tous les patchs) |
| Données nécessaires | Moins | **Beaucoup** (ou pré-entraînement) |
| Coût | Linéaire en taille d'image | Quadratique en nombre de patchs |

---

## 14. CNN et représentations temps-fréquence

Les CNN ne servent pas qu'aux photos : tout signal qu'on peut représenter comme une **image** est candidat. Exemple : **le son**, transformé en **spectrogramme** (axe x = temps, axe y = fréquence, couleur = intensité).

**Exemple : séparation de sources** (isoler la voix de chaque locuteur dans un mélange) :
- **Avant** : problème non supervisé, ICA (analyse en composantes indépendantes), SVD…
- **Maintenant** : **problème supervisé de classification dans le domaine temps-fréquence** : pour chaque case du spectrogramme, on prédit à quelle source elle appartient (un peu comme une segmentation d'image). Les CNN ont fait de gros progrès sur ce problème ces dernières années.

---

## 15. Introspection : comprendre ce que voit un CNN

Un réseau profond est une « boîte noire ». L'**introspection** cherche à comprendre : *quelle région de l'image déclenche la classification ? Quels filtres sont importants ?*

### Ce que les filtres apprennent
Visualisation (exemple Faces / Cars) : les **filtres deviennent de plus en plus abstraits** :
- couche 1 : bords, orientations, couleurs ;
- couche 2 : parties (yeux, nez, roues…) → ce sont des **« mots visuels élémentaires »** ;
- couche 3 : objets entiers (visages, voitures).

Les couches **conservent l'information topologique** (la structure spatiale).

### Méthodes d'explication

| Méthode | Principe | Intuition |
|---|---|---|
| **Occlusion** | On **masque** une zone de l'image (carré gris) qu'on déplace partout, et on mesure **comment la probabilité de la classe chute** | « Si je cache cette zone et que le modèle se trompe, c'est qu'elle était importante. » (Ex. cacher le museau du Poméranien fait chuter le score.) Simple, mais coûteux (un passage par position). |
| **Carte de saillance (saliency map)** | On calcule le **gradient de la sortie par rapport à l'image d'entrée** | Les pixels où le gradient est fort sont ceux auxquels la classification est **la plus sensible**. Un seul passage backward. |
| **Class Activation Map (CAM)** | Avec un **Global Average Pooling (GAP)** avant la couche finale : 1 filtre = 1 feature, pondérée par le poids `w_k` de la classe. On fait une **combinaison linéaire des cartes de features** : `CAM_c = Σ_k w_k^c · f_k` | « Quelles régions ont fait “voter” les filtres pour cette classe ? » On obtient une carte de chaleur des **régions d'intérêt**. |
| **LIME** (Local Interpretable Model-agnostic Explanations) | 1) découper l'image en **super-pixels** ; 2) **échantillonner** des perturbations (certains super-pixels masqués/allumés) ; 3) **pondérer** chaque échantillon par sa proximité avec l'image d'origine ; 4) apprendre un **modèle linéaire local** ; 5) interpréter les **coefficients** | Même si le modèle est globalement complexe et non linéaire, **localement** il est approximable par un modèle simple et interprétable. « Agnostic » = marche avec n'importe quel modèle. |

---

## 16. Des convolutions aux graphes (GNN)

**Idée** : la convolution fonctionne sur une **grille régulière** (chaque pixel a toujours 8 voisins). Mais beaucoup de données ne sont pas des grilles : molécules, réseaux sociaux, routes… ce sont des **graphes** (nœuds + arêtes, avec un nombre de voisins variable).

**Généralisation** : au lieu de « glisser un filtre sur une grille », on fait **passer des messages** entre voisins :

$$ n_i^{(2)} = \sum_{n_j \in \mathcal{N}(n_i)} \psi(n_i, n_j, e_{ji}) $$

- `n_i` : nœud courant ; `n_j` : ses voisins `𝒩(n_i)` ; `e_ji` : l'arête qui relie `j` à `i`.
- `ψ` : un **réseau de neurones** (ex. PMC / MLP) qui combine les informations. Le nombre de filtres = taille du vecteur de sortie.
- **Message passing** : chaque nœud agrège les messages de ses voisins.
- **Plus de couches ⇒ ↗ champ réceptif** : à la couche k, un nœud « voit » ses voisins à distance k (même logique que pour les CNN).
- On peut brancher des **losses** sur : les **nœuds** (classer chaque nœud), les **arcs** (prédire un lien), ou les **graphes entiers** (classer un graphe → il faut un **agrégateur**, ex. somme/moyenne/max de tous les nœuds).

> 💡 Une image est un cas particulier de graphe : chaque pixel est un nœud relié à ses voisins sur la grille.

---

## 17. Boîte à outils PyTorch (les bases des bases)

### 17.1 Imports de base

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from torchvision import datasets, transforms, models
```

### 17.2 Convention des dimensions (à retenir absolument)

| Type | Format PyTorch |
|---|---|
| Image 2D (batch) | `(N, C, H, W)` = (batch, canaux, hauteur, largeur) |
| Signal 1D (batch) | `(N, C, L)` |

```python
x = torch.randn(8, 3, 224, 224)   # batch de 8 images RGB 224×224
print(x.shape)                     # torch.Size([8, 3, 224, 224])
x = x.unsqueeze(0)                 # ajoute une dim batch (image seule → (1, C, H, W))
x = x.squeeze(0)                   # enlève la dim de taille 1
x = x.permute(0, 3, 1, 2)          # (N,H,W,C) -> (N,C,H,W)  (ex. depuis numpy/PIL)
```

### 17.3 Les couches essentielles

```python
# --- Convolutions ---
nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1, dilation=1)
nn.Conv1d(in_channels=1, out_channels=8, kernel_size=5)        # signaux 1D

# --- Pooling ---
nn.MaxPool2d(kernel_size=2, stride=2)       # réduit H et W par 2
nn.AvgPool2d(kernel_size=2)                  # moyenne
nn.AdaptiveAvgPool2d((1, 1))                 # Global Average Pooling → (N, C, 1, 1)
nn.AdaptiveAvgPool2d((7, 7))                 # force une taille de sortie fixe

# --- Activations ---
nn.ReLU()
nn.Sigmoid()
nn.Softmax(dim=1)          # (inutile avec CrossEntropyLoss, cf. plus bas)

# --- Régularisation / normalisation ---
nn.Dropout(p=0.5)
nn.BatchNorm2d(num_features=16)        # = nombre de canaux

# --- Mise en forme / couches denses ---
nn.Flatten()                           # (N, C, H, W) -> (N, C*H*W)
nn.Linear(in_features=1024, out_features=10)

# --- Convolution transposée (décodeur, upsampling) ---
nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=2, stride=2)
nn.Upsample(scale_factor=2)
```

### 17.4 Un CNN complet (modèle de départ)

```python
class SimpleCNN(nn.Module):
    def __init__(self, n_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),   # (1,28,28) -> (16,28,28)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # -> (16,14,14)
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> (32,14,14)
            nn.ReLU(),
            nn.MaxPool2d(2),                               # -> (32,7,7)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                  # -> 32*7*7 = 1568
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, n_classes),                     # logits (pas de softmax !)
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)

model = SimpleCNN()
```

### 17.5 Vérifier les dimensions (le réflexe n°1 pour déboguer)

```python
x = torch.randn(1, 1, 28, 28)
out = x
for layer in model.features:
    out = layer(out)
    print(layer.__class__.__name__, out.shape)

print(model)                                           # architecture
sum(p.numel() for p in model.parameters())             # nombre total de paramètres
sum(p.numel() for p in model.parameters() if p.requires_grad)   # paramètres entraînables
```

### 17.6 Données : transformations, data augmentation, DataLoader

```python
transform_train = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),          # data augmentation
    transforms.RandomRotation(10),              # data augmentation
    transforms.ToTensor(),                      # PIL -> tensor (C,H,W) valeurs [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],     # stats ImageNet
                         std=[0.229, 0.224, 0.225]),
])
transform_test = transforms.Compose([           # PAS d'augmentation au test
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_set = datasets.MNIST(root="data", train=True, download=True,
                           transform=transforms.ToTensor())
train_loader = DataLoader(train_set, batch_size=64, shuffle=True)

# Dossier d'images organisé en sous-dossiers = classes :
# data/train/chat/*.jpg, data/train/chien/*.jpg ...
ds = datasets.ImageFolder("data/train", transform=transform_train)
```

### 17.7 Fonction de perte et optimiseur

```python
criterion = nn.CrossEntropyLoss()        # classification multi-classe (prend les LOGITS, applique softmax en interne)
# nn.BCEWithLogitsLoss()                 # multi-label / binaire
# nn.MSELoss()                           # régression (ex. boîtes englobantes)

optimizer = optim.Adam(model.parameters(), lr=1e-3)
# optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### 17.8 Boucle d'entraînement (à connaître par cœur)

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

for epoch in range(10):
    model.train()                                   # active dropout, batchnorm en mode train
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()                       # 1. remet les gradients à zéro
        outputs = model(images)                     # 2. forward
        loss = criterion(outputs, labels)           # 3. calcul de la perte
        loss.backward()                             # 4. rétropropagation
        optimizer.step()                            # 5. mise à jour des poids

        total_loss += loss.item()
    print(f"Epoch {epoch+1} - loss: {total_loss/len(train_loader):.4f}")
```

### 17.9 Évaluation

```python
model.eval()                       # désactive dropout, fige batchnorm
correct, total = 0, 0
with torch.no_grad():              # pas de calcul de gradient (plus rapide, moins de mémoire)
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        preds = model(images).argmax(dim=1)          # classe prédite
        correct += (preds == labels).sum().item()
        total += labels.size(0)
print("Accuracy:", correct / total)
```

### 17.10 Sauvegarder / charger

```python
torch.save(model.state_dict(), "model.pth")                   # sauvegarde les poids
model.load_state_dict(torch.load("model.pth"))                # recharge (même architecture requise)
model.eval()
```

### 17.11 Transfer learning / fine-tuning

```python
# 1. Charger un réseau pré-entraîné sur ImageNet
model = models.resnet18(weights="DEFAULT")       # ou models.vgg16, models.alexnet...

# 2. GELER toutes les couches (extracteur de features)
for param in model.parameters():
    param.requires_grad = False

# 3. REMPLACER la dernière couche pour notre nombre de classes
n_classes = 5
model.fc = nn.Linear(model.fc.in_features, n_classes)    # (pour VGG : model.classifier[6])

# 4. N'optimiser que les paramètres entraînables
optimizer = optim.Adam(model.fc.parameters(), lr=1e-3)

# --- Fine-tuning partiel : dégeler le dernier bloc ---
for param in model.layer4.parameters():
    param.requires_grad = True
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)  # lr plus faible !
```

### 17.12 Introspection rapide

```python
# --- Carte de saillance (gradient de la sortie par rapport à l'image) ---
model.eval()
img = img.unsqueeze(0).requires_grad_(True)       # (1,C,H,W), on veut le gradient sur l'image
score = model(img)[0, classe_cible]               # score de la classe
score.backward()
saliency = img.grad.abs().max(dim=1)[0]           # (1,H,W) : carte de saillance

# --- Visualiser les filtres de la 1re couche ---
weights = model.conv1.weight.data                  # (out_channels, in_channels, K, K)

# --- Récupérer les cartes de features d'une couche (hook) ---
activations = {}
def hook(module, inp, out):
    activations["layer"] = out.detach()
model.layer1.register_forward_hook(hook)
_ = model(img)
print(activations["layer"].shape)
```

### 17.13 Visualisation avec matplotlib

```python
import matplotlib.pyplot as plt
plt.imshow(img.squeeze().permute(1, 2, 0))   # (C,H,W) -> (H,W,C)  ; pour du gris : plt.imshow(img.squeeze(), cmap="gray")
plt.show()
```

---

## 18. Fiche de révision express

### Les 10 idées à retenir

1. **MLP + images = mauvais** : trop de paramètres, dépendance à la position, pas d'invariance.
2. **Convolution = petit filtre qui glisse** + **poids partagés** → peu de paramètres, détection de motifs n'importe où.
3. Les filtres sont **appris par rétropropagation**.
4. **Pooling** = réduire la taille (max / moyenne) + invariance locale, **0 paramètre**.
5. Bloc standard : **Conv → ReLU → Pooling**, répété ; à la fin : **Flatten → Linear → Softmax**.
6. Plus on est profond : **résolution ↓, nombre de canaux ↑, abstraction ↑**, **champ réceptif ↗**.
7. **Fine-tuning** : réutiliser un réseau pré-entraîné (ImageNet) ; geler les premières couches ; attention à l'**oubli catastrophique**.
8. **ResNet** = connexions résiduelles `y = F(x) + x` pour entraîner des réseaux très profonds.
9. Selon la tâche : **classification** (accuracy), **détection** (boîtes, IoU, YOLO/R-CNN), **segmentation** (image→image, U-Net, IoU).
10. **Introspection** : occlusion, saillance (gradient), CAM (GAP + poids), LIME (modèle linéaire local).

### Formules à connaître

| Quoi | Formule |
|---|---|
| Taille de sortie (conv ou pooling) | `⌊(W − K + 2P) / S⌋ + 1` |
| Noyau effectif avec dilatation | `K_eff = D·(K − 1) + 1` |
| Paramètres d'une conv | `out_ch × (in_ch × K × K + 1)` |
| Bloc résiduel | `y = F(x) + x` |
| IoU | `aire(A ∩ B) / aire(A ∪ B)` |
| CAM | `CAM_c = Σ_k w_k^c · f_k` |
| Message passing | `n_i^(2) = Σ_{j∈𝒩(i)} ψ(n_i, n_j, e_ji)` |

### Pièges classiques

- ❌ Oublier `model.eval()` + `torch.no_grad()` à l'inférence.
- ❌ Oublier `optimizer.zero_grad()` (les gradients s'accumulent).
- ❌ Mettre un `Softmax` avant `CrossEntropyLoss` (elle l'applique déjà).
- ❌ Se tromper dans la dimension d'entrée du premier `nn.Linear` après `Flatten` → **calculer / afficher les shapes**.
- ❌ Mauvais ordre des dimensions : PyTorch attend `(N, C, H, W)`.
- ❌ Fine-tuner avec un learning rate trop élevé → oubli catastrophique.
- ❌ Ne pas normaliser les images avec les stats d'ImageNet quand on utilise un réseau pré-entraîné.
- ❌ Faire de la data augmentation sur le jeu de test.
