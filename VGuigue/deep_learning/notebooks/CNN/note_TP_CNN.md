<h1 align="center">Fiche : Fonctions utiles sur les CNN</h1>

---

---

## I. Convolutions et pooling avec PyTorch

### 1. `nn.Conv2d`

```python
nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0,
          dilation=1, bias=True, padding_mode='zeros')
```

| Paramètre       | Rôle                                                                                                                      | Dans le notebook                                                      |
| ---------------- | -------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------- |
| `in_channels`  | Nombre de canaux de l'entrée (1 = niveaux de gris, 3 = RGB)                                                               | `1`                                                                 |
| `out_channels` | Nombre de filtres appris, donc nombre de canaux en sortie                                                                  | `1`                                                                 |
| `kernel_size`  | Taille du filtre (`2` signifie 2×2)                                                                                     | `2` ou `3`                                                        |
| `stride`       | Pas de déplacement de la fenêtre. Un pas de 2 divise à peu près la taille par 2                                        | `stride=2` : sortie 2×2 pour une entrée 5×5                      |
| `padding`      | Bordure ajoutée tout autour de l'entrée, qui augmente la taille de sortie                                                | `padding=1` : sortie 6×6 pour une entrée 5×5                     |
| `padding_mode` | Valeur de remplissage (`'zeros'`, `'reflect'`, `'replicate'`, `'circular'`)                                        | `'zeros'`                                                           |
| `dilation`     | Espacement entre les cases du filtre (filtre « troué »), ce qui élargit le champ de vision sans ajouter de paramètres | `dilation=2` : sortie 3×3                                          |
| `bias`         | Ajoute un biais par filtre                                                                                                 | `False` uniquement pour le test ; en pratique on le laisse toujours |

#### Attributs utiles

- `conv.weight` : tenseur de forme `(out_channels, in_channels, kH, kW)`. Ici `(1, 1, 2, 2)`.
- Pour imposer les poids à la main : `conv.weight = nn.Parameter(poids)`. Utile pour tester, mais en pratique on garde l'initialisation aléatoire.

### 3. Pooling

```python
nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1)
nn.AvgPool2d(kernel_size, stride=None, padding=0)
```

- `MaxPool2d` garde le maximum de chaque fenêtre, `AvgPool2d` en fait la moyenne.
- Pas de paramètres à apprendre.
- Le pooling agit canal par canal, d'où l'absence de `in_channels` et `out_channels`.
- Par défaut, `stride = kernel_size`. Avec `kernel_size=2`, la hauteur et la largeur sont donc divisées par 2 (arrondi à l'inférieur).
- Il sert à réduire la taille et à introduire des invariances locales.

Exemple du notebook, avec une entrée 5×5 :

- max : `[[6, 8], [16, 18]]`
- moyenne : `[[3, 5], [13, 15]]`

### 6. Imports à connaître (pour la suite du TP)

- `nn` : couches (`Conv2d`, `Linear`, pooling, etc.).
- `F` (`torch.nn.functional`) : versions fonctionnelles (`relu`, `softmax`, etc.).
- `optim` : optimiseurs (SGD, Adam, etc.).
- `TensorDataset`, `DataLoader`, `random_split` : construire les jeux de données, les découper en train/val/test et itérer par mini-batchs.
- `tqdm` : barre de progression.

---

## II. Construire et entraîner un CNN avec PyTorch

*Complète la fiche 1 (Conv2d, pooling, imports). Source : notebook `2_1-CNN.ipynb`.*

### 1. Con#struire un modèle : `nn.Module`

```python
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()          # obligatoire en premier
        # définir les couches ici
    def forward(self, x):
        # dire comment x traverse les couches
        return x
```

- `__init__` : on **déclare** les couches (elles contiennent les paramètres à apprendre).
- `forward(x)` : on **décrit le calcul**. On l'appelle avec `model(x)` (jamais `model.forward(x)`).
- Le backward est automatique, il ne faut pas l'écrire.

### 2. Couches et conteneurs

| Fonction                                 | Rôle                                                                                                                           | Paramètres importants                      |
| ---------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------- |
| `nn.Sequential(l1, l2, ...)`           | Enchaîne des couches dans l'ordre et se comporte comme une seule couche                                                        | la liste des couches                        |
| `nn.Linear(in_features, out_features)` | Couche fully-connected : `y = Wx + b`                                                                                         | dimension d'entrée, dimension de sortie    |
| `nn.ReLU()`                            | Activation `max(0, x)`, ne change pas la taille                                                                               | aucun                                       |
| `nn.MaxPool2d(kernel_size, stride)`    | Pooling max (voir fiche 1)                                                                                                      | `stride` vaut `kernel_size` par défaut |
| `nn.AvgPool2d(kernel_size, stride)`    | Pooling moyenne                                                                                                                 | idem                                        |
| `torch.nn.init.xavier_uniform_(w)`     | Initialise les poids `w` (ex. `conv.weight`) pour stabiliser les gradients. Le `_` final signifie « modifie sur place » | le tenseur de poids                         |

**Structure classique d'un CNN**

- `self.features` : `Conv → ReLU → Pool → Conv → ReLU → Pool` (extraction des caractéristiques).
- `self.classifier` : `Linear → ReLU → Linear` (décision finale).
- Entre les deux, on aplatit : `x.view(x.size(0), -1)` donne `(B, C, H, W) → (B, C·H·W)`.

**Dimension d'entrée du premier `Linear`** = `C × H × W` en sortie des convolutions (calculée avec la formule de la fiche 1).

**Dernière couche** : autant de neurones que de classes (10 pour MNIST), sans activation.

### 3. Tenseurs : fonctions de manipulation

| Fonction                             | Rôle                                                              |
| ------------------------------------ | ------------------------------------------------------------------ |
| `x.view(a, b)` / `x.view(a, -1)` | Reformate le tenseur (`-1` = dimension déduite automatiquement) |
| `x.size()` ou `x.shape`          | Forme du tenseur                                                   |
| `x.size(0)`                        | Taille de la 1re dimension (la taille du batch)                    |
| `x.unsqueeze(0)`                   | Ajoute une dimension de taille 1 en position 0                     |
| `x.squeeze()`                      | Supprime les dimensions de taille 1                                |
| `x.to(device)`                     | Envoie le tenseur ou le modèle sur le CPU ou le GPU               |
| `torch.argmax(t, dim)`             | Indice du maximum le long d'une dimension (classe prédite)        |
| `t.mean()`, `t.sum()`            | Moyenne et somme                                                   |
| `t.item()`                         | Convertit un tenseur à 1 valeur en nombre Python                  |

`device` s'écrit `torch.device("cuda" if torch.cuda.is_available() else "cpu")`. Le modèle **et** les données doivent être sur le même device.

### 4. Inspecter un modèle

| Fonction                                   | Rôle                                                                                                                                                                                            |
| ------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `print(model)`                           | Affiche l'architecture (rapide)                                                                                                                                                                  |
| `model.named_parameters()`               | Itère sur `(nom, paramètre)` : taille avec `param.size()`, `param.requires_grad` indique si le paramètre est appris. Utile pour **geler** des couches (`requires_grad = False`) |
| `model.parameters()`                     | Itère sur les paramètres seuls (donné à l'optimiseur)                                                                                                                                        |
| `summary(model, input_size=(1, 28, 28))` | Résumé couche par couche avec tailles de sortie et nombre de paramètres (`from torchsummary import summary`, à installer avec pip). `input_size` est sans la dimension de batch          |

### 5. Entraînement : les briques

| Fonction                                    | Rôle                                                                                                                                                                                                                                   |
| ------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `nn.CrossEntropyLoss()`                   | Perte pour la classification multiclasse. Prend les **scores bruts** (logits) de forme `(B, nb_classes)` et les **indices de classes** `(B,)`. Applique le softmax en interne, donc pas de softmax en sortie du réseau |
| `optim.Adam(model.parameters(), lr=1e-3)` | Optimiseur qui met à jour les poids. `lr` = taux d'apprentissage                                                                                                                                                                     |
| `optimizer.zero_grad()`                   | Remet les gradients à 0 (sinon ils s'accumulent d'un batch à l'autre)                                                                                                                                                                 |
| `loss.backward()`                         | Rétropropagation : calcule les gradients                                                                                                                                                                                               |
| `optimizer.step()`                        | Met à jour les poids avec les gradients                                                                                                                                                                                                |
| `model.train()`                           | Mode entraînement (active dropout, batchnorm en apprentissage)                                                                                                                                                                         |
| `model.eval()`                            | Mode évaluation (désactive ces comportements)                                                                                                                                                                                         |
| `with torch.no_grad():`                   | Désactive le calcul des gradients, ce qui économise de la mémoire et du temps (validation, test)                                                                                                                                     |

**Ordre à retenir dans la boucle sur un batch**

```
zero_grad → forward → loss → backward → step
```

### 6. Boucle d'entraînement type

```python
def train(model, epochs, train_loader, test_loader):
    writer = SummaryWriter(f"{TB_PATH}/{model.name}")
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    loss = nn.CrossEntropyLoss()
    for epoch in tqdm(range(epochs)):
        model.train()
        for x, y in train_loader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            yhat = model(x)
            l = loss(yhat, y)
            l.backward()
            optimizer.step()
        # ... stockage des métriques dans writer
        model.eval()
        with torch.no_grad():
            for x, y in test_loader:
                ...   # forward + métriques uniquement
```

**Pondérer par la taille du batch** : on cumule `loss * len(x)` et `acc * len(x)`, puis on divise par le nombre total d'exemples, car le dernier batch peut être plus petit.

### 7. Métrique : accuracy

```python
def accuracy(yhat, y):
    return (torch.argmax(yhat, 1).view(y.size(0), -1) == y.view(-1, 1)).float().mean()
```

- `argmax(yhat, 1)` : classe prédite pour chaque exemple.
- Comparaison avec `y`, puis `.float().mean()` : proportion de bonnes prédictions.

### 8. Suivi avec TensorBoard

| Fonction                                   | Rôle                                                           |
| ------------------------------------------ | --------------------------------------------------------------- |
| `SummaryWriter(dossier)`                 | Crée un écrivain de logs (un dossier par modèle)             |
| `writer.add_scalar(nom, valeur, étape)` | Enregistre une valeur (ex. `'loss/train'`, `epoch`)         |
| `tensorboard --logdir <chemin>`          | Commande console pour visualiser les courbes dans le navigateur |

Les noms de dossier ne doivent pas contenir de `:` sous Windows (utiliser `time.strftime("%Y%m%d-%H%M%S")`).

---

---

## III. Visualiser et interpréter un CNN

### 1. Nouv#eautés dans l'entraînement

| Fonction                                                   | Rôle                                                                                                                  | Paramètres importants                                                                                                                 |
| ---------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| `torch.optim.AdamW(params, lr)`                          | Variante d'Adam avec une régularisation (weight decay) mieux gérée                                                  | `lr` : taux d'apprentissage                                                                                                          |
| `OneCycleLR(optimizer, max_lr, steps_per_epoch, epochs)` | Fait varier le `lr` pendant l'entraînement : il monte jusqu'à `max_lr` puis redescend. Accélère la convergence | `max_lr` : pic du lr ; `steps_per_epoch` : nombre de batchs par epoch (`len(train_loader)`) ; `epochs` : nombre total d'epochs |
| `scheduler.step()`                                       | Met à jour le `lr`. Avec OneCycleLR, on l'appelle **après chaque batch**, juste après `optimizer.step()`  | aucun                                                                                                                                  |

Import : `from torch.optim.lr_scheduler import OneCycleLR`.

### 2. Récupérer les sorties intermédiaires d'un réseau

| Fonction                             | Rôle                                                                               |
| ------------------------------------ | ----------------------------------------------------------------------------------- |
| `model.features`                   | Sous-réseau convolutionnel (le `nn.Sequential`)                                  |
| `model.features._modules.values()` | Liste des couches du `Sequential`, dans l'ordre, pour les parcourir une à une    |
| `m.forward(x)`                     | Applique **une seule couche** `m` à `x` (exécution pas à pas)          |
| `isinstance(m, nn.Conv2d)`         | Teste le type d'une couche (aussi `nn.MaxPool2d`, `nn.AvgPool2d`)               |
| `m.weight`                         | Poids d'une couche de convolution, de forme `(out_channels, in_channels, kH, kW)` |
| `w[j, 0]`                          | Filtre `j` appliqué au canal d'entrée 0 (matrice `kH × kW`)                  |

**Principe** : on fait passer l'image couche par couche et on stocke la sortie de celles qui nous intéressent.

### 4. Convertir un tenseur pour l'afficher

| Fonction                        | Rôle                                                                                                        |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| `.detach()`                   | Détache du graphe de calcul (obligatoire si le tenseur a un gradient)                                       |
| `.cpu()` ou `.to('cpu')`    | Ramène sur le CPU (matplotlib ne lit pas le GPU)                                                            |
| `np.array(t)` / `t.numpy()` | Convertit en tableau NumPy                                                                                   |
| `.squeeze()`                  | Supprime les dimensions de taille 1                                                                          |
| `.permute(1, 2, 0)`           | Réordonne les dimensions : `(C, H, W) → (H, W, C)`, format attendu par `imshow` pour une image couleur |

Chaîne habituelle : `t.detach().cpu().squeeze()`.

### 5. Saliency map : quels pixels ont influencé la décision ?

| Fonction                      | Rôle                                                                                                             |
| ----------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| `img.requires_grad = True`  | Demande le calcul du gradient **par rapport à l'image**                                                    |
| `model.zero_grad()`         | Remet à 0 les gradients des paramètres du modèle                                                               |
| `nn.Softmax(dim=1)(scores)` | Transforme les scores en probabilités (somme = 1). `dim=1` : le long des classes                               |
| `outputs[0, label]`         | Probabilité de la classe d'intérêt pour la 1re image du batch                                                  |
| `output.backward()`         | Calcule le gradient de cette probabilité par rapport à toutes les entrées                                      |
| `img.grad`                  | Gradient obtenu pour chaque pixel                                                                                 |
| `.abs()`                    | Valeur absolue (on s'intéresse à l'ampleur de l'influence)                                                      |
| `torch.max(t, dim=0)[0]`    | Maximum le long d'une dimension (ici sur les canaux couleur). `[0]` récupère les valeurs, `[1]` les indices |

**Idée** : plus le gradient d'un pixel est grand, plus une petite modification de ce pixel change la prédiction. Le softmax doit être remis à la main car il est inclus dans `CrossEntropyLoss` et absent du réseau.

### 6. Class Activation Map (CAM) : où le réseau regarde-t-il ?

**Modification du réseau** : le classifieur devient `pooling moyen global → une seule couche linéaire`.

| Fonction                         | Rôle                                                                          | Paramètres                                |
| -------------------------------- | ------------------------------------------------------------------------------ | ------------------------------------------ |
| `nn.Unflatten(dim, (C, H, W))` | Inverse de l'aplatissement : redonne une forme 3D                              | `dim=1` ; la forme cible                 |
| `nn.AvgPool2d(k, 1)`           | Avec `k` = taille de la carte (ici 16) : une seule valeur moyenne par filtre | taille de la fenêtre = taille de la carte |
| `nn.Flatten(start_dim)`        | Aplatit à partir de la dimension `start_dim` (ici 1 pour garder le batch)   | `1`                                      |
| `nn.Linear(16, 10)`            | Une entrée par filtre, une sortie par classe                                  |                                            |

```python
model.classifier = nn.Sequential(
    nn.Unflatten(1, (16, 16, 16)),
    nn.AvgPool2d(16, 1),
    nn.Flatten(1),
    nn.Linear(16, 10)
)
```

**Geler puis ré-entraîner (fine-tuning)**

```python
for p in model.parameters():
    p.requires_grad = False      # geler tout le réseau
model.classifier = nn.Sequential(...)   # les nouvelles couches sont créées après, donc apprenables
train(model, ...)
```

Les paramètres gelés ne sont pas modifiés par l'optimiseur. Une couche créée **après** le gel est entraînable par défaut.

**Calcul de la CAM**

| Étape             | Code                                    | Rôle                                                                     |
| ------------------ | --------------------------------------- | ------------------------------------------------------------------------- |
| Features           | `x = model.features(img)`             | Cartes de la dernière couche conv, forme `(1, 16, 16, 16)`             |
| Poids de la classe | `model.classifier[-1].weight[classe]` | Importance de chacun des 16 filtres pour cette classe                     |
| Pondération       | `x * w.view(1, -1, 1, 1)`             | Multiplie chaque carte par son poids (le `view` permet le broadcasting) |
| Somme              | `.sum(1)`                             | Somme sur les filtres : carte `(1, 16, 16)`                             |
| Affichage          | `imshow(cmap="bwr")`                  | Rouge = zones qui poussent vers la classe, bleu = qui s'y opposent        |

Autres fonctions utiles : `torch.nn.functional.softmax(out, -1)` (softmax fonctionnel) et `torch.max(out, dim=-1)[1].item()` (classe prédite).
