# Fiche : U-Net, que se passe-t-il dans la représentation ?

## 1. But du notebook

Un CNN classique *résume* l'image : il dit **quoi** (la classe) mais perd **où**. Pour la segmentation (une classe par pixel), il faut les deux. Le U-Net combine :

- un **encodeur** qui descend en résolution jusqu'à un **goulot** (bottleneck) ;
- un **décodeur** symétrique qui remonte en résolution ;
- des **connexions de saut** (skip connections) qui copient les cartes de l'encodeur vers le décodeur, niveau par niveau.

**Hypothèse testée expérimentalement** : les skips transportent le « où » (contours, position précise), le goulot transporte le « quoi » (contexte, identité de l'objet).

Le notebook est conçu pour tourner sur un portable : MNIST collé sur un canevas 64×64, réseaux d'environ 100k paramètres, quelques minutes d'entraînement. Des poids pré-entraînés sont fournis dans `models/`.

## 2. Notions principales

- **Encodeur / goulot / décodeur** et **skip connections** (concaténation sur l'axe des canaux).
- **`ConvTranspose2d`** : convolution « inverse » qui augmente la résolution (avec `kernel_size=2, stride=2`, elle double la taille).
- **`nn.Upsample`** : alternative sans paramètre appris pour augmenter la résolution.
- **Conv 1×1** : couche linéaire appliquée pixel par pixel (la « tête » de classification).
- **Segmentation sémantique** : 11 classes par pixel (0 = fond, `d+1` = chiffre `d`).
- **Métriques séparant « où » et « quoi »** :
  - `iou_fg` (« où ») : IoU entre pixels prédits « chiffre » et vrais pixels « chiffre », classes ignorées ;
  - `acc_cls` (« quoi ») : parmi les pixels bien détectés comme chiffres, proportion avec la bonne classe.
  - L'accuracy par pixel est trompeuse (le fond fait environ 90 % des pixels).
- **Champ réceptif (receptive field)** théorique et **effectif**.
- **Hooks et points d'accroche** (`nn.Identity` + forward hooks) pour lire ou remplacer le contenu des skips et du goulot.
- **Ablation** : remplacer un tenseur par zéro ou par sa moyenne et mesurer l'effet.
- **Images chimères** : forme (où) prise sur B, représentation (quoi) prise sur A.
- **Sondes linéaires (linear probes)** : classifieur linéaire sur une représentation gelée, pour mesurer l'information qui y est lisible. Il faut une baseline (réseau non entraîné).
- **Géométrie des représentations** : projection ACP (PCA) des vecteurs par position.
- **Dépendance du rôle du goulot à la tâche** : segmentation, identité, inpainting.

## 3. Structure du notebook

### Setup

Imports, choix du device (`cuda` / `mps` / `cpu`), `SummaryWriter` TensorBoard (avec classe de remplacement si absent).

### A. Construire un U-Net

- **A.1** Mini-exos : prédire les tailles de sortie de `Conv2d`, `MaxPool2d`, `ConvTranspose2d`, `Upsample`, `torch.cat`, conv 1×1. Questions sur le nombre de paramètres et sur la taille 64×64 (3 poolings successifs).
- **A.2** Réseau : `DoubleConv` fournie, classe `UNet` à compléter (décodeur et `forward`). Option `skips=False` : skips remplacés par des zéros (auto-encodeur).

### B. Tâche « quoi » + « où » : segmenter des chiffres MNIST

1. **B.1 Données** : 1 à 3 chiffres MNIST sur un canevas noir 64×64 (`make_canvas`, `random_positions`, `SegMNIST`). 10 000 images en train, 1 000 en test. Masque obtenu en seuillant le chiffre.
2. **B.2 Métriques** : `seg_counts` (à compléter), `edited` (gestionnaire de contexte pour poser les hooks), `evaluate_seg`.
3. **B.3 Trois architectures** (toutes avec `base=16`) :

   | modèle | description |
   |---|---|
   | `unet` | profondeur 3, avec skips |
   | `noskip` | même réseau, skips coupés (auto-encodeur) |
   | `shallow` | U-Net de profondeur 1, même largeur |

   Calcul à la main du champ réceptif théorique (`theoretical_rf`), puis `train` (AdamW + OneCycleLR) ou chargement des poids.
4. **B.4 Comparaison** : tableau `iou_fg` / `acc_cls` et visualisation (`show_batch`).

### C. Que contient la représentation ? (sur le `unet` entraîné)

- **C.0** `capture` : récupérer le contenu de tous les points d'accroche et visualiser les canaux.
- **C.1 Ablations** : goulot / chaque skip / tous les skips remplacés par la moyenne, et comparaison avec le remplacement par zéro.
- **C.2 Chimères** : `make_pairs` (mêmes positions, chiffres différents) et `chimera(model, xA, xB, from_A)`. On mesure la proportion de pixels avec la classe de B, la classe de A, ou autre.
- **C.3 Sondes linéaires** : max-pooling global d'un point d'accroche, puis classifieur linéaire qui prédit quels chiffres sont présents (F1 moyen). Comparaison de `skip0`, `skip1`, `skip2`, du goulot du `unet`, du goulot du `noskip` et d'un U-Net aléatoire.
- **C.4 Champ réceptif effectif** : gradient de la sortie au pixel central par rapport à l'entrée, comparé au champ théorique (`shallow` vs `unet`).
- **C.5 Géométrie** : vecteurs par position du goulot / `skip2` / `skip0`, projetés par ACP et colorés par classe majoritaire.

### D. Le rôle du goulot dépend de la tâche

Même architecture (`base=8`, sortie à 1 canal, perte MSE), deux tâches : **identité** et **inpainting** (carré 20×20 remplacé par du gris 0.5). Métriques `mse_trou` / `mse_hors`. Ablations « goulot → moyenne » et « skip0 → moyenne » sur les deux modèles.

### Pour aller plus loin

- La largeur est aussi un goulot (`base=8` : trop peu de canaux à pleine résolution pour 11 classes).
- Skips à gain appris (scalaire `α` par skip avec pénalité L1).
- Skip dropout (couper les skips aléatoirement à l'entraînement).
- Lien avec les modèles de diffusion (U-Net comme débruiteur).

## 4. Organisation du U-Net

Avec `base=8` et `depth=3`, les canaux sont `[8, 16, 32, 64]` (`base * 2**i`).

```
entrée 1x64x64
   │ enc0 (DoubleConv)
8x64x64 ───────────── skip0 ──────────────► concat ─► dec0 ─► 8x64x64 ─► conv 1x1 ─► 11x64x64
   │ pool                                      ▲ up
   │ enc1                                      │
16x32x32 ──────────── skip1 ──────► concat ─► dec1 ─► 16x32x32
   │ pool                              ▲ up
   │ enc2                              │
32x16x16 ───────────── skip2 ─► concat ─► dec2 ─► 32x16x16
   │ pool                          ▲ up
   │ goulot (DoubleConv)           │
64x8x8 ────────────────────────────┘
```

Éléments de l'architecture :

- **`DoubleConv`** : `Conv2d(3×3, padding=1)` → `BatchNorm2d` → `ReLU`, deux fois (`bias=False` devant la BatchNorm).
- **Encodeur** : `ModuleList` de `DoubleConv` + `MaxPool2d(2)` entre les niveaux.
- **Décodeur** : pour chaque niveau, `ConvTranspose2d(c_{i+1} → c_i, 2, stride=2)` qui double la résolution et divise les canaux par 2, puis `DoubleConv(2·c_i → c_i)` appliquée à la **concaténation** de la sortie de l'up et du skip.
- **Tête** : `Conv2d(c_0, n_classes, 1)`.
- **Points d'accroche** : `skip_tap[i]` et `bottleneck_tap` sont des `nn.Identity()` qui ne changent pas le calcul mais permettent de poser des hooks. `skip_tap[i]` ne doit agir que sur la branche skip, pas sur celle qui descend au niveau suivant.

Principe du `forward` (le `TODO` de A.2) :

1. Encodeur : à chaque niveau, calculer la sortie, la stocker (via le tap) dans `skips`, puis appliquer le pooling.
2. Passer dans le goulot (via `bottleneck_tap`).
3. Décodeur : up, récupérer le skip du niveau correspondant (en ordre inverse, remplacé par des zéros si `skips=False`), concaténer, `DoubleConv`.
4. Appliquer la tête 1×1.

**Champ réceptif théorique** (formule : `r ← r + (k-1)·j`, puis `j ← j·s`) :

| profondeur | pixel du goulot | pixel de sortie |
|---|---|---|
| 1 | 14×14 | 18×18 (moins qu'un chiffre de 28×28) |
| 3 | 68 px (tout le canevas) | plus large |

## 5. Commandes clés par notion

```python
# --- Tailles (mini-exos A.1) ---
nn.Conv2d(8, 16, kernel_size=3, padding=1)           # taille spatiale conservée
nn.MaxPool2d(2)                                      # divise H, W par 2
nn.ConvTranspose2d(8, 4, kernel_size=2, stride=2)    # double H, W
nn.Upsample(scale_factor=2, mode="nearest")          # double H, W, sans paramètre
torch.cat([a, b], dim=1)                             # concaténation sur les canaux (skip)
nn.Conv2d(8, 11, kernel_size=1)                      # tête pixel par pixel

# --- Briques du UNet ---
nn.ModuleList([...])                                 # listes de modules enregistrés
nn.Identity()                                        # point d'accroche pour hook
sum(p.numel() for p in model.parameters())           # nombre de paramètres

# --- Hooks avec remplacement (un hook qui RENVOIE une valeur remplace la sortie) ---
module.register_forward_hook(lambda m, inp, out: nouvelle_valeur)
with edited(model, {"bottleneck": lambda o: torch.zeros_like(o)}):
    out = model(x)                                   # forward avec goulot remis à zéro

# --- Capture des activations ---
acts = capture(model, x)                             # dict nom -> tenseur

# --- Ablation par la moyenne ---
means = {k: v.mean(0, keepdim=True) for k, v in capture(unet, xm).items()}
replace_by(means["skip0"])                           # t.expand_as(o)

# --- Chimère ---
edited(model, {k: (lambda o, t=acts_A[k]: t) for k in from_A})

# --- Entraînement ---
loss_fn = nn.CrossEntropyLoss()                      # segmentation (logits B×11×H×W, cible B×H×W)
loss_fn = nn.MSELoss()                               # identité / inpainting
optimizer = torch.optim.AdamW(model.parameters(), lr=max_lr)
scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=len(train_loader), epochs=epochs)

# --- Sauvegarde / chargement ---
torch.save({'model_state': model.state_dict()}, fichier)
model.load_state_dict(torch.load(fichier, map_location="cpu")['model_state'])

# --- Sonde linéaire ---
feats = acts[tap].amax(dim=(2, 3))                   # max-pooling global -> (B, C)
lin = nn.Linear(C, 10)
F.binary_cross_entropy_with_logits(lin(feats), present)

# --- Champ réceptif effectif ---
x.requires_grad_(True)
out[:, :, i, j].max(1).values.sum().backward()
x.grad.abs().mean(0)[0]

# --- Géométrie ---
F.avg_pool2d(onehot, k)                              # proportion de chaque classe par zone
torch.pca_lowrank(v, q=2, center=False)              # ACP en 2D
```

## 6. Points d'attention

- **Hypothèses fournies dans les commentaires du notebook** (à vérifier par l'expérience, le notebook n'a pas de sorties enregistrées) :
  - `shallow` (champ réceptif trop petit) devrait bien délimiter les pixels (`iou_fg`) mais se tromper de classe (`acc_cls`) ;
  - `noskip` devrait donner la bonne classe, mais avec des contours moins précis (reconstruits depuis une grille 8×8), donc un `iou_fg` plus faible.
- **Zéro vs moyenne** dans les ablations : le zéro est brutal (le décodeur reçoit des valeurs jamais vues, les statistiques de BatchNorm s'effondrent) et mesure autant la « surprise » du réseau que la perte d'information. La moyenne enlève l'information propre à l'image en gardant des valeurs plausibles.
- **Chimères** : tester avec des positions différentes entre A et B permet de savoir si ces représentations sont un vecteur global ou une carte spatiale.
- **Sondes** : la baseline aléatoire est indispensable, et la comparaison est biaisée par la dimension (nombre de canaux) qui diffère selon le point d'accroche.
- **Champ effectif vs théorique** : le champ réceptif effectif est plus petit que le théorique (le gradient décroît avec la distance). Cela éclaire l'échec de `shallow`.
- **Largeur** : `base=8` peut classer des chiffres entiers en « fond » (la probabilité du fond l'emporte à l'argmax quand le réseau hésite).
- **Parties à compléter** (`TODO`) : mini-exos A.1, `forward` de A.2, `seg_counts` (B.2), et les questions d'analyse de B.4, C.1, C.2, C.3, C.4, C.5 et D.
