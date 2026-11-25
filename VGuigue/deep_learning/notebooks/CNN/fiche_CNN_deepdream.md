# Fiche : CNN pré-entraîné, deepdream, reconstruction et transfert de style

## 1. But du notebook

Prendre un CNN pré-entraîné (GoogLeNet sur ImageNet) et le détourner de sa fonction de classification. On ne réentraîne pas les poids (weights). On fige le réseau et on optimise l'image d'entrée par descente de gradient, à partir d'une loss définie sur une couche intermédiaire.

Le notebook contient quatre parties :

- **A** : vérifier que le réseau fonctionne (classification).
- **B** : deepdream.
- **C** : reconstruction d'image à partir d'un embedding.
- **D** : transfert de style (style transfer), à compléter par l'étudiant (les `TODO`).

## 2. Notions principales

- **Transfer learning / modèle pré-entraîné** : on récupère GoogLeNet avec ses poids ImageNet et on le passe en mode inférence (`eval()`).
- **Preprocessing ImageNet** : `Resize`, `ToTensor`, `Normalize(mean, std)`, et `unsqueeze(0)` pour créer la dimension batch. Le `deprocess` inverse l'opération pour afficher l'image.
- **Logits vs probabilités** : la sortie brute (1000 scores) devient des probabilités avec un `softmax`. On lit ensuite le top-5.
- **Hooks (forward hooks)** : fonctions « crochet » qui capturent l'activation (feature map) d'une couche intermédiaire pendant le forward.
- **Optimisation sur l'entrée** : les poids sont figés, et c'est l'image (`requires_grad_(True)`) qui devient le paramètre optimisé.
- **Gradient ascent vs descent** : deepdream maximise la norme des activations (loss avec un signe moins). La reconstruction et le style transfer minimisent une distance.
- **Hiérarchie des features** : les couches basses encodent des textures et des contours, les couches profondes des motifs abstraits. Cela justifie `inception3a` pour le contenu et `inception5a` pour le style.
- **Matrice de Gram** : corrélations entre canaux (channels) d'une feature map, utilisée pour représenter le style (texture sans information spatiale).
- **Loss composite** : `α · loss_content + β · loss_style`, avec des poids à ajuster.
- **Optimisation** : Adam / AdamW et scheduler `OneCycleLR` pour converger plus vite.
- **Initialisation** : bruit blanc gaussien (`randn_like`) pour les parties C et D.

## 3. Structure du notebook

### Setup

Import de GoogLeNet, `model.eval()`, affichage de l'architecture (indispensable pour savoir où poser les hooks).

### A. Classification (sanity check)

1. Téléchargement de l'image exemple (le chien).
2. Preprocessing et forward sous `torch.no_grad()`.
3. Softmax puis top-5. Résultat : Samoyed à environ 94 %.
4. Téléchargement des labels ImageNet.

### B. Deepdream

1. Chargement de l'image (`load_image`) et copie de l'original.
2. Choix de la couche (`inception4b`) et pose du hook.
3. Boucle de 30 itérations : forward, `loss = -activations['feat'].norm()`, backward, `optimizer.step()`.
4. `clamp` des pixels, retrait du hook, `deprocess` et affichage avant/après (avec un zoom).

La loss devient de plus en plus négative, ce qui est normal : on maximise la norme.

### C. Reconstruction d'image

1. Choix du device (`cuda` / `mps` / `cpu`).
2. Hook sur `inception3a`, forward de l'image et sauvegarde de la cible (`target`). Son shape est `[1, 256, 62, 79]`.
3. Création d'une image de bruit avec gradient.
4. 1000 itérations : `loss = (target - activations['feat']).norm()`, avec AdamW + OneCycleLR.
5. Affichage de l'image reconstruite.

### D. Transfert de style

1. Chargement de l'image de contenu (chien) et de l'image de style (`vg.jpg`, Van Gogh).
2. Deux hooks : contenu sur `inception3a`, style sur `inception5a`.
3. Cibles fixes : `target_content` (feature map) et `target_gram` (Gram de l'image de style).
4. Optimisation du bruit sur 500 itérations avec `α=1` et `β=1e4`.
5. Affichage.

Le dernier bloc du notebook est une cellule technique de l'enseignant (génération du sujet à partir de la correction). Elle ne fait pas partie du contenu.

## 4. Organisation du CNN (GoogLeNet / Inception v1)

Le réseau est une succession d'étages, que l'on peut lister avec `model.named_modules()` :

| Étage | Modules | Rôle |
|---|---|---|
| **Stem** | `conv1` (7×7, 64, stride 2) → `maxpool1` → `conv2` (1×1) → `conv3` (3×3, 192) → `maxpool2` | extraction de features basiques, réduction de résolution |
| **Inception 3** | `inception3a` (192→256 canaux), `inception3b` (→480), `maxpool3` | features bas/moyen niveau |
| **Inception 4** | `inception4a` à `4e` (→832), `maxpool4` | features de niveau moyen/élevé |
| **Inception 5** | `inception5a`, `inception5b` (→1024) | features abstraites |
| **Tête** | `avgpool` (global, → 1×1) → `dropout(0.2)` → `fc` (1024→1000) | classification ImageNet |

Deux détails :

- `aux1` et `aux2` valent `None`. Ce sont les classifieurs auxiliaires, désactivés ici.
- Chaque `BasicConv2d` correspond à `Conv2d` + `BatchNorm2d` (suivis d'une ReLU dans le forward).

**Un module Inception** applique quatre branches en parallèle sur la même entrée, puis concatène leurs sorties sur la dimension des canaux :

- `branch1` : conv 1×1.
- `branch2` : conv 1×1 (réduction) → conv 3×3.
- `branch3` : conv 1×1 (réduction) → conv 3×3.
- `branch4` : maxpool 3×3 → conv 1×1.

Les convolutions 1×1 réduisent le nombre de canaux avant les convolutions plus coûteuses. C'est ce qui rend l'architecture efficace. Les noms comme `inception3a` sont ceux qu'on utilise dans les hooks.

## 5. Commandes clés par notion

```python
# --- Modèle pré-entraîné ---
model = googlenet(weights=GoogLeNet_Weights.IMAGENET1K_V1)
model.eval()
dict(model.named_modules())["inception4b"]      # récupérer une couche par son nom

# --- Preprocessing ---
transforms.Compose([Resize, CenterCrop, ToTensor, Normalize(mean, std)])
tensor.unsqueeze(0)                              # ajouter la dimension batch

# --- Inférence ---
with torch.no_grad():
    output = model(input_batch)
probs = torch.nn.functional.softmax(output[0], dim=0)
torch.topk(probs, 5)

# --- Hook (capture d'une couche intermédiaire) ---
def hook_fn(module, input, output):
    activations['feat'] = output
hook = target_layer.register_forward_hook(hook_fn)
...
hook.remove()                                    # à ne pas oublier

# --- Optimiser l'image (et non le modèle) ---
img.requires_grad_(True)
optimizer = torch.optim.Adam([img], lr=0.1)

# --- Boucle d'optimisation ---
optimizer.zero_grad()
model(img)                                       # remplit activations via le hook
loss = -activations['feat'].norm()               # deepdream (ascent, donc signe -)
loss = (target - activations['feat']).norm()     # reconstruction (descent)
loss.backward()
optimizer.step()
img.data = torch.clamp(img.data, -2, 2)          # stabilité

# --- Scheduler ---
OneCycleLR(optimizer, max_lr=..., steps_per_epoch=1, epochs=nepoch)
scheduler.step()

# --- Cibles fixes ---
with torch.no_grad():
    target = activations['feat'].clone()

# --- Bruit initial ---
noise = torch.randn_like(img)

# --- Style : matrice de Gram ---
f = x.view(C, H*W)
gram = (f @ f.T) / (C*H*W)
loss = alpha*loss_content + beta*loss_style

# --- Device ---
device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
```

## 6. Points d'attention

- **Le signe de la loss** : minimiser `-norm` revient à maximiser la norme. Sans le moins, deepdream ne fonctionne pas.
- **`.clone()`** sur les cibles : sinon elles seraient écrasées au prochain forward, puisque le hook réécrit `activations`.
- **`torch.no_grad()`** pour calculer les cibles, mais pas pendant l'optimisation de l'image.
- **`gram`** utilise `x.view(C, H*W)`, ce qui ne marche que pour un batch de taille 1.
- **Reconstruction (C)** : la loss atteint un minimum vers l'itération 120 (environ 278), puis remonte. Le `max_lr` de OneCycleLR est probablement trop élevé à ce stade du cycle, donc ce paramètre est à ajuster.
- **Style transfer (D)** : le rapport `β/α` pilote le compromis contenu/style. Il faut le tester.
