# Module 1 : Introduction au Deep Learning — Architectures en PyTorch

> **Notes explicatives** du cours de Vincent Guigue (Agro–IODAA, Semestre 1, inspiré de N. Baskiotis & B. Piwowarski).
> Objectif de ces notes : comprendre **le principe derrière chaque concept**, pas seulement retenir les formules.

---

## Sommaire

1. [Introduction : du neurone biologique au perceptron](#1-introduction--du-neurone-biologique-au-perceptron)
2. [Pourquoi un seul neurone ne suffit pas : limites et non-linéarité](#2-pourquoi-un-seul-neurone-ne-suffit-pas)
3. [Le vocabulaire de l'inférence (forward)](#3-le-vocabulaire-de-linférence-forward)
4. [Apprendre : fonction de coût et descente de gradient](#4-apprendre--coût-et-descente-de-gradient)
5. [La rétro-propagation (chain rule)](#5-la-rétro-propagation-chain-rule)
6. [Architecture modulaire](#6-architecture-modulaire)
7. [Premier réseau de neurones en PyTorch](#7-premier-réseau-de-neurones-en-pytorch)
8. [Autres modules importants : classification multi-classes](#8-classification-multi-classes--softmax-et-cross-entropie)
9. [Lutter contre le sur-apprentissage](#9-lutter-contre-le-sur-apprentissage)
10. [Améliorer le gradient](#10-améliorer-le-gradient)
11. [Optimisation](#11-optimisation--bien-descendre-la-colline)
12. [Conclusion : DataLoader, checkpointing, TensorBoard](#12-conclusion--les-outils-indispensables)
13. [🧾 Cheatsheet finale](#13--cheatsheet-finale)

---

## 1. Introduction : du neurone biologique au perceptron

### 1.1 L'inspiration biologique (« plus ou moins lointaine »)

Un neurone biologique :
- reçoit des signaux via ses **dendrites**,
- les **additionne** dans le **corps cellulaire**,
- et, si le total est assez fort, envoie un signal le long de l'**axone** vers d'autres neurones, via les **synapses**.

Le neurone artificiel reprend cette logique **de façon très simplifiée** :

| Neurone biologique | Neurone artificiel |
|---|---|
| Signaux reçus | Entrées $x_i$ (les *features*) |
| Force d'une synapse | Poids $w_i$ |
| Fusion des messages | **Addition** pondérée |
| Décision de « s'activer » | **Fonction d'activation** (au début : le signe) |

> 💡 **À retenir** : le cours précise que le nom « réseau de neurones » est *évocateur et vendeur*. Il ne faut pas croire qu'on simule un cerveau : un neurone artificiel est juste **un opérateur mathématique simple** (somme pondérée + non-linéarité).

### 1.2 Le perceptron : le tout premier modèle

**Idée** : on veut classer des points en deux catégories ($y = -1$ ou $y = +1$) en traçant une **droite** (un hyperplan en dimension $d$) qui les sépare.

**Le modèle** (sur des données $(\mathbf{x}, y) \in \mathbb{R}^d \times \{-1, 1\}$) :

1. **Score linéaire** : $f_{\mathbf{w}}(\mathbf{x}) = w_0 + \sum_{i=1}^{d} x_i w_i = w_0 + \langle \mathbf{x}, \mathbf{w} \rangle$
   - $w_0$ est le **biais** : il décale la droite (sans lui, elle passerait forcément par l'origine).
2. **Décision** : $g(x) = \text{sign}(x)$ → on prédit $+1$ si le score est positif, $-1$ sinon.
3. **Sortie** : $g(f(\mathbf{x})) = \text{sign}(\langle \mathbf{x}, \mathbf{w}\rangle)$

**Intuition géométrique** : $\mathbf{w}$ est un vecteur perpendiculaire à la droite de séparation. Un point est d'un côté ou de l'autre selon le signe du produit scalaire avec $\mathbf{w}$.

**Comment apprend-on ?** (algorithme du perceptron)

```
Tant qu'il n'y a pas convergence :
    pour chaque exemple (xⁱ, yⁱ) :
        si yⁱ × <w, xⁱ> < 0 :      # l'exemple est MAL classé
            w ← w + ε · yⁱ · xⁱ     # on "tire" w vers l'exemple
```

- **Pourquoi `yⁱ × <w, xⁱ> < 0` signifie « mal classé » ?** Si $y = +1$ et que le score est négatif (ou l'inverse), le produit est négatif : signe prédit ≠ signe réel.
- **Pourquoi la mise à jour fonctionne ?** On ajoute $\varepsilon y x$ à $\mathbf{w}$ : si $y=+1$, on rapproche $\mathbf{w}$ de $\mathbf{x}$ (le score de $x$ augmente) ; si $y=-1$, on l'éloigne.
- **Lien avec la descente de gradient** : cela revient à minimiser le coût $\max(0, -y f_{\mathbf{w}}(\mathbf{x}))$ — il vaut 0 si l'exemple est bien classé, et il est positif (proportionnel à l'erreur) sinon.

> ⚠️ *Remarque* : sur la slide, il est écrit `arg max` devant l'espérance de ce coût, mais comme c'est une **erreur** qu'on veut réduire, il s'agit bien d'un **`arg min`**.

---

## 2. Pourquoi un seul neurone ne suffit pas

### 2.1 Les limites du perceptron

Le perceptron ne trace qu'**une seule droite**. Il ne peut donc séparer que des données **linéairement séparables**. Le cours montre trois jeux de données où il échoue :

1. Un nuage de points rouges entouré/mélangé à des points bleus (non séparable par un plan).
2. Un **damier** (cases rouges/vertes alternées) — aucune droite ne peut séparer ça.
3. Trois paquets alignés : violet – **jaune** – violet. La classe jaune est « au milieu » : une seule droite ne peut pas isoler le milieu des deux côtés (il faudrait **deux** droites).

### 2.2 Première idée : empiler/combiner des neurones

Prenons 2 neurones en première couche ($f_1$, $f_2$) puis un neurone de sortie ($f_3$) :

$$f_1(\mathbf{x}) = w^{(1)}_{11}x_1 + w^{(1)}_{21}x_2 + w^{(1)}_{01}$$
$$f_2(\mathbf{x}) = w^{(1)}_{12}x_1 + w^{(1)}_{22}x_2 + w^{(1)}_{02}$$
$$f_3(\mathbf{x}) = w^{(2)}_{11}f_1(\mathbf{x}) + w^{(2)}_{21}f_2(\mathbf{x}) + w^{(2)}_{01}$$

**Question du cours : combiner des neurones est-il suffisant ?** → **NON.**

### 2.3 Preuve : le piège de la « composition linéaire »

Si on développe $f_3$ :

$$f_3(\mathbf{x}) = x_1\underbrace{(w^{(2)}_{11}w^{(1)}_{11} + w^{(2)}_{21}w^{(1)}_{12})}_{\text{une constante}} + x_2\underbrace{(w^{(2)}_{11}w^{(1)}_{21} + w^{(2)}_{21}w^{(1)}_{22})}_{\text{une constante}} + \underbrace{w^{(2)}_{01} + w^{(2)}_{11}w^{(1)}_{01} + w^{(2)}_{21}w^{(1)}_{02}}_{\text{une constante}}$$

On retrouve $f_3(\mathbf{x}) = A x_1 + B x_2 + C$ : **c'est exactement un perceptron** ! Peu importe le nombre de couches, une composition de fonctions linéaires reste linéaire.

> 💡 **L'analogie** : multiplier par 2 puis par 3 revient à multiplier par 6. Empiler des transformations linéaires n'ajoute aucune puissance expressive.

### 2.4 La solution : la non-linéarité

Il faut **intercaler une fonction non linéaire $g$** après chaque neurone (la **fonction d'activation**).

**Quelle fonction choisir ?**

- ❌ **La fonction signe** : sa dérivée est nulle partout (sauf en 0 où elle n'existe pas). Or on apprend par **gradient** → le gradient serait nul ou indéfini → impossible d'apprendre. C'est la « dérivée problématique » du cours.
- ✅ **La sigmoïde** : $g(x) = \dfrac{1}{1+e^{-x}}$ — une version **lisse** et dérivable de la marche d'escalier, qui écrase tout dans $]0, 1[$.
- ✅ **tanh**, **ReLU** (vu plus loin), etc.

### 2.5 L'exemple qui fait comprendre la puissance des 2 couches

Pour le jeu de données « violet – jaune – violet » (sur l'axe $x_1$) :

- $f_1 = \sigma(3x_1 - 3)$ : une marche d'escalier lisse qui passe de 0 à 1 autour de $x_1 = +1$.
- $f_2 = \sigma(3x_1 + 3)$ : même marche mais autour de $x_1 = -1$.
- $f_3 = f_2 - f_1 - 0{,}5$ : la **différence** de deux marches décalées donne une **bosse** (≈ 1 entre −1 et +1, ≈ 0 ailleurs), puis on recentre en retirant 0,5.

Résultat : $f_3 > 0$ au centre (classe jaune) et $f_3 < 0$ sur les côtés (classe violette). Avec **deux « droites » combinées** par la non-linéarité, on isole la zone du milieu, ce qui était impossible avec un seul neurone.

> 💡 **Intuition clé** : chaque neurone de la couche cachée découpe l'espace avec « sa » droite ; la couche suivante **combine** ces découpes pour fabriquer des formes complexes. Plus on a de neurones/couches, plus les formes réalisables sont riches.

---

## 3. Le vocabulaire de l'inférence (forward)

Pour un réseau avec activation sigmoïde $g$ :

```
 x ──► [ a₁ = w₀₁ + w₁₁x₁ + w₂₁x₂ ] ──► z₁ = g(a₁) ─┐
                                                      ├─► a₃ = w₀₁ + w₁₁z₁ + w₂₁z₂ ──► ŷ = g(a₃)
 x ──► [ a₂ = w₀₂ + w₁₂x₁ + w₂₂x₂ ] ──► z₂ = g(a₂) ─┘
```

| Terme | Signification |
|---|---|
| **Inférence** | Calculer la sortie du réseau pour une entrée : on l'appelle la **passe forward** (« en avant »). |
| $g$ | **Fonction d'activation** (c'est elle qui apporte la non-linéarité du réseau). |
| $a_i$ | **Activation** du neurone $i$ = la somme pondérée *avant* la non-linéarité (« pré-activation »). |
| $z_i$ | **Sortie** du neurone $i$ = $g(a_i)$ (la transformée non linéaire de l'activation). |
| $\hat{y}$ | **Prédiction** finale du réseau. |
| $y$ | **Vérité terrain** (*ground truth*), la vraie valeur. |

> ⚠️ Piège de vocabulaire : « activation » désigne parfois $a_i$ (comme ici) et parfois $z_i$ (dans d'autres cours). Ici : **$a$ = avant $g$, $z$ = après $g$**.

---

## 4. Apprendre : coût et descente de gradient

### 4.1 L'objectif : apprendre les poids

Un réseau n'est rien d'autre qu'une fonction $\hat{y} = F(\mathbf{x}; \mathbf{W})$ avec plein de paramètres $\mathbf{W}$. **Apprendre = trouver les poids qui font que $\hat{y}$ est le plus proche possible de $y$** sur les données.

Pour mesurer « proche », on choisit une **fonction de coût** (*loss*). Exemple : les **moindres carrés** :

$$L(\hat{y}, y) = (\hat{y} - y)^2$$

(Le cours pose la question « pourquoi est-ce un bon choix ? » — réponses classiques : c'est dérivable, convexe en $\hat y$, pénalise fortement les grosses erreurs, et correspond au maximum de vraisemblance sous un bruit gaussien.)

**Le vrai problème** : il y a beaucoup de poids. Quand l'erreur est grande, **quel poids est responsable ?** Comment répartir la « faute » entre eux ? → **la rétro-propagation de l'erreur** (section 5).

### 4.2 La descente de gradient

**Intuition : la randonnée dans le brouillard.** Vous êtes sur une montagne (la surface de coût) et vous voulez atteindre la vallée (coût minimal). Vous ne voyez rien, mais vous sentez la **pente sous vos pieds** : vous faites un pas dans la direction qui descend le plus.

- La pente = le **gradient** $\dfrac{\partial L}{\partial w_{ij}}$ : il dit « si j'augmente légèrement $w_{ij}$, de combien le coût augmente-t-il ? ».
- On avance dans le sens **opposé** au gradient :

$$w_{ij} \leftarrow w_{ij} - \underbrace{\varepsilon}_{\text{learning rate}} \; \frac{\partial L(\hat{y}, y)}{\partial w_{ij}}$$

- $\varepsilon$ (**learning rate**, taux d'apprentissage) = la **taille du pas**. Trop petit : on met une éternité. Trop grand : on dépasse la vallée et on oscille ou diverge.

**Le cycle complet d'un apprentissage** :

1. **Forward** : calcul de $\hat{y}$ (et stockage des valeurs intermédiaires).
2. **Backward** : calcul de tous les gradients $\partial L / \partial w_{ij}$.
3. **Optimisation** : mise à jour des poids par descente de gradient.

### 4.3 Espace de description vs espace des paramètres

Deux « vues » du même problème (slide 27) :

- **Espace de description** (gauche) : les axes sont les *variables* ($X_1$, $Y$). On y voit les **données** et **la droite** de régression $\mathbf{w}$.
- **Espace des paramètres** (droite) : les axes sont les *poids* $\mathbf{w}$ et le *coût* $C$. Chaque point représente **un modèle possible**, et sa hauteur est son erreur.

Quand on apprend, on ne déplace pas les données : on **déplace le point $\mathbf{w}$** sur la surface de coût vers son minimum $\mathbf{w}^\star$.

---

## 5. La rétro-propagation (chain rule)

### 5.1 Le principe : la règle de dérivation en chaîne

Le réseau est une **composition de fonctions** : $L(\hat y, y)$ dépend de $g(a_3)$, qui dépend de $a_3$, qui dépend de $z_1, z_2$, etc. Pour dériver une composition, on utilise la **chain rule** :

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial \text{(sortie)}} \times \frac{\partial \text{(sortie)}}{\partial \text{(étape précédente)}} \times \dots \times \frac{\partial \text{(...)}}{\partial w}$$

> 💡 **Analogie de la chaîne de production** : une usine fabrique un produit défectueux. Pour savoir quel atelier corriger, on remonte la chaîne à l'envers : l'erreur finale est répartie entre les ateliers selon **leur contribution** à cette erreur. C'est exactement la rétro-propagation.

### 5.2 Gradient de la dernière couche

Poids $w^{(2)}_{i1}$ (qui arrivent sur le neurone de sortie $a_3$) :

$$L = \big(g(a_3) - y\big)^2, \qquad a_3 = w^{(2)}_{01} + w^{(2)}_{11}z_1 + w^{(2)}_{21}z_2$$

On décompose en deux facteurs :

$$\frac{\partial L}{\partial w^{(2)}_{i1}} = \underbrace{\frac{\partial L}{\partial a_3}}_{\text{erreur au neurone}} \times \underbrace{\frac{\partial a_3}{\partial w^{(2)}_{i1}}}_{= z_i}$$

avec
$$\frac{\partial L}{\partial a_3} = \underbrace{2(g(a_3) - y)}_{\text{dérivée du carré}} \times \underbrace{g'(a_3)}_{\text{dérivée de l'activation}}$$

D'où :

$$\boxed{\frac{\partial L}{\partial w^{(2)}_{i1}} = 2\,g'(a_3)\,(\hat{y} - y)\,z_i}$$

**Lecture du résultat (très parlante)** :

- $(\hat{y}-y)$ : plus l'erreur de prédiction est grande, plus on corrige.
- $g'(a_3)$ : si le neurone est **saturé** (dérivée ≈ 0), la correction est quasi nulle → on retrouvera ce problème avec le *gradient vanishing*.
- $z_i$ : plus l'entrée $z_i$ était « active », plus ce poids a contribué à l'erreur, donc plus on le corrige. (Pour le biais, $z_0 = 1$.)

**Exemple numérique** (celui de la slide : $\hat y = 0{,}5$, $y=-1$) : si $g$ est la sigmoïde, $g'(a_3) = \hat y(1-\hat y) = 0{,}25$. Donc $\partial L/\partial a_3 = 2 \times 0{,}25 \times (0{,}5 - (-1)) = 0{,}75$. Chaque poids $w^{(2)}_{i1}$ reçoit un gradient de $0{,}75 \times z_i$. *(Note : une sigmoïde ne peut pas sortir −1 ; c'est un exemple purement illustratif.)*

### 5.3 Gradient de la première couche

Pour un poids $w^{(1)}_{i1}$ plus loin dans le réseau, on **prolonge la chaîne** :

$$\frac{\partial L}{\partial w^{(1)}_{i1}} = \frac{\partial L}{\partial a_1}\cdot\frac{\partial a_1}{\partial w^{(1)}_{i1}}, \qquad \frac{\partial L}{\partial a_1} = \underbrace{\frac{\partial L}{\partial a_3}}_{\text{erreur à propager}} \cdot \underbrace{g'(a_1)\,w^{(2)}_{11}}_{\text{poids de la connexion}}$$

Donc :
$$\frac{\partial L}{\partial w^{(1)}_{i1}} = \frac{\partial L}{\partial a_3} \; g'(a_1)\, w^{(2)}_{11} \; x_i$$

> 📌 **L'idée centrale** : l'erreur $\partial L/\partial a_3$ déjà calculée à la sortie est **réutilisée** et **renvoyée en arrière** à travers la connexion $w^{(2)}_{11}$ (puis modulée par $g'(a_1)$). On ne recalcule pas tout depuis zéro : c'est ce qui rend la méthode efficace.
>
> *(Sur la slide 11, le poids de la connexion est noté `w13` : il s'agit du poids $w^{(2)}_{11}$ qui relie le neurone 1 au neurone de sortie.)*

### 5.4 Cas général dans les couches intermédiaires

On note $\delta_j = \dfrac{\partial L}{\partial a_j}$ (l'**erreur au neurone $j$**). Alors :

$$\frac{\partial L}{\partial w_{ij}} = z_i \cdot \delta_j \qquad\text{et}\qquad \delta_j = \sum_k \big(g'(a_j)\, w_{jk}\big)\,\delta_k$$

(le $g'$ s'applique au neurone $j$ ; il peut être sorti de la somme.)

**Deux règles à retenir :**

1. **Gradient d'un poids** = (**valeur qui entre** dans le poids) × (**erreur qui sort** du neurone suivant) : $z_i \times \delta_j$.
2. **Quand l'erreur arrive de plusieurs sources → on additionne.** Un neurone $j$ qui alimente plusieurs neurones $k$ est responsable d'une partie de *chacune* de leurs erreurs : $\delta_j = \sum_k \dots$
3. L'erreur de la couche $j$ s'exprime en fonction de l'erreur de la couche $k$ suivante → on calcule **de la fin vers le début** (d'où « *back*-propagation »).

---

## 6. Architecture modulaire

### 6.1 L'idée : un réseau = un assemblage de modules (briques LEGO)

Au lieu de tout recalculer à la main pour chaque nouvelle architecture, on voit le réseau comme une **chaîne de modules** $M_1, M_2, \dots, M_K$, plus un bloc final : le **coût**.

```
 x ─► [M¹ (W¹)] ─► z¹ ─► [M² (W²)] ─► z² ─► ... ─► [Mᴷ (Wᴷ)] ─► ŷ ─► [ L(ŷ, y) ] ◄─ y
```

**Un module $M^k$ :**
- a des **entrées** : la sortie de la couche précédente $z^{k-1}$ ;
- a **éventuellement des paramètres** $W^{(k)}$ (qu'on peut voir comme des entrées supplémentaires) ;
- produit **une sortie** $z^k = M^k(z^{k-1}, W^k)$.

> 💡 **Pourquoi c'est génial ?** Chaque module n'a besoin de savoir faire que **deux choses** : (1) calculer sa sortie (forward) et (2) calculer sa dérivée (backward). Ensuite, on peut brancher les modules comme on veut : la chain rule s'occupe du reste. C'est la philosophie de PyTorch.

### 6.2 Les modules usuels

**① La couche linéaire (`Linear`)** — transformation paramétrée de $\mathbb{R}^d$ vers $\mathbb{R}^{d'}$ :

$$z^k = M^k(z^{k-1}, W^k) = {W^k}^{\top} z^{k-1}, \qquad W^k \in \mathbb{R}^{d \times d'}$$

Chaque sortie est une somme pondérée de **toutes** les entrées : $z^{k+1}_i = w_{1i}z^k_1 + w_{2i}z^k_2 + \dots + w_{di}z^k_d$ (d'où le nom *fully-connected*). C'est ce module qui contient les **paramètres à apprendre**.

**② Le module d'activation** (ex. `tanh`) — de $\mathbb{R}^d$ vers $\mathbb{R}^d$, appliqué **élément par élément** :

$$M^k(z^{k-1}, \emptyset) = \big(\tanh(z^{k-1}_1), \dots, \tanh(z^{k-1}_d)\big)$$

Il n'a **aucun paramètre** ($\emptyset$) : il ne fait qu'apporter la non-linéarité.

**③ Le coût** — bloc final à **deux entrées** : la sortie du réseau $\hat y$ et la supervision $y$. Il sort un nombre (l'erreur).

### 6.3 Apprentissage du réseau

Pour chaque module, on veut $\nabla_{W^k} L(\hat y, y)$ :

- **Module sans paramètre** (activation) : rien à apprendre → gradient par rapport aux paramètres **nul**. (Mais il doit quand même **laisser passer** l'erreur vers l'arrière !)
- **Autres modules** : **rétro-propagation**.

### 6.4 Zoom sur le module $k$ : formules de la rétro-propagation

**Deux faits** :

- $w^k_{ij}$ **n'impacte que** $z^k_j$ (la sortie $j$ de son module) ;
- $z^k_j$ **impacte tous** les $z^{k+1}_n$ de la couche suivante.

Donc, par la chain rule :

$$\frac{\partial L}{\partial w^k_{ij}} = \frac{\partial L}{\partial z^k_j}\cdot\frac{\partial M^k(z^{k-1}, W^k)}{\partial w^k_{ij}}, \qquad \frac{\partial L}{\partial z^k_j} = \sum_n \frac{\partial L}{\partial z^{k+1}_n}\cdot\frac{\partial M^{k+1}(z^k, W^{k+1})}{\partial z^k_j}$$

On introduit l'**erreur du module** : $\delta^k_j = \dfrac{\partial L}{\partial z^k_j}$. Le calcul devient une **récurrence** :

$$\delta^k_j = \sum_n \delta^{k+1}_n \cdot \frac{\partial M^{k+1}}{\partial z^k_j} \qquad\qquad \frac{\partial L}{\partial w^k_{ij}} = \delta^k_j \cdot \frac{\partial M^k}{\partial w^k_{ij}}$$

**Initialisation (dernière couche)** : $\delta^{end}_j = \dfrac{\partial L(z^{end}, y)}{\partial z^{end}_j}$ = le gradient du coût par rapport à la prédiction. Pour la MSE : $\delta^{end}_n = \frac{2}{N}(z^{end}_n - y_n)$.

### 6.5 Forward / backward en pratique

**Forward** — on calcule de gauche à droite :
$z^1 = M_1(\mathbf{x}, W^1)$, puis $z^k = M_k(z^{k-1}, W^k)$, … jusqu'à $\hat{\mathbf{y}}$.

> 🔑 **On STOCKE tous les $z^k$ au passage.** Pourquoi ? Parce que le backward en a besoin : le gradient d'un module linéaire contient $z^k_j$ (voir ci-dessous). C'est pourquoi l'entraînement consomme plus de mémoire que l'inférence simple.

**Backward** — on calcule de droite à gauche, en commençant par $\delta^{end}$, puis en propageant :

Pour un **module linéaire** :
- Gradient des poids : $\dfrac{\partial L}{\partial w^{k+1}_{jn}} = \delta^{end}_n \cdot z^k_j$ → (erreur de sortie) × (valeur d'entrée)
- Erreur à renvoyer en arrière : $\delta^k_j = \sum_n \delta^{end}_n \, w^{k+1}_{jn}$ → l'erreur est **redistribuée par les poids**

> 📌 **Moralité** : chaque module reçoit une erreur de sa droite, calcule (a) le gradient de ses propres poids et (b) l'erreur qu'il renvoie à sa gauche. Le réseau tout entier s'entraîne en faisant circuler cette erreur.

---

## 7. Premier réseau de neurones en PyTorch

### 7.1 Réseau Fully-Connected (FC / MLP)

Un réseau **fully-connected** est une **succession de couches linéaires et de fonctions d'activation**. Sur le schéma : Input (4) → Dense #1 (6) → Dense #2 (4) → Softmax (3 classes).

| ✅ Avantages | ⚠️ Limites |
|---|---|
| Idéal pour de **simples tâches de classification** (multi-classes aussi) | **Peu adapté aux entrées complexes** (texte, images) |
| On le **retrouve dans quasiment toutes les autres architectures** (comme brique) | **Très sujet au sur-apprentissage** quand on augmente le nombre de couches |

### 7.2 Utiliser des modules existants

```python
M = torch.nn.Linear(args)   # Création du module
z = M(x)                    # Inférence : appel direct (pas besoin d'écrire M.forward(x))
```

**Réseau à une seule couche linéaire** (= décision linéaire simple, une régression linéaire) :

```python
Xdim = housing_x.size(1)                       # nombre de features en entrée

net = torch.nn.Linear(Xdim, 1)                 # couche linéaire Xdim -> 1
mseloss = torch.nn.MSELoss()                   # fonction de coût
optim = torch.optim.SGD(params=net.parameters(), lr=EPS)   # optimiseur
#                      ^ on donne à l'optimiseur la liste des poids à ajuster

yhat = net(housing_x)                          # forward
loss = mseloss(net(housing_x).view(-1,1), housing_y.view(-1,1))
loss.backward()                                # backward : calcule tous les gradients
optim.step()                                   # mise à jour des poids
```

**Les rôles à bien distinguer :**

| Objet | Rôle |
|---|---|
| `net` | le modèle (contient les poids) |
| `mseloss` | mesure l'erreur |
| `optim` | sait **comment mettre à jour** les poids à partir des gradients |
| `.view(-1, 1)` | remet les tenseurs sous la forme `(N, 1)` pour que prédiction et cible aient la **même forme** (sinon, broadcasting silencieux et résultats faux !) |

### 7.3 Un réseau de neurones est un module

En PyTorch, un réseau complet est lui-même un **module** (`torch.nn.Module`). Il suffit de définir :

1. **les couches** (dans `__init__`),
2. **l'inférence** (`forward` : comment les entrées traversent les couches).

> 🎉 **Et c'est tout : le `backward` est automatique** (grâce à l'*autograd* de PyTorch, qui applique la chain rule à notre place).

```python
class DeuxCouches(torch.nn.Module):          # héritage de nn.Module
    def __init__(self):                      # définition des attributs (couches)
        super(DeuxCouches, self).__init__()  # obligatoire : initialise la classe mère
        self.un   = torch.nn.Linear(Xdim, 5)   # entrée -> 5 neurones cachés
        self.act  = torch.nn.Tanh()            # non-linéarité
        self.deux = torch.nn.Linear(5, 1)      # 5 neurones -> 1 sortie

    def forward(self, x):                    # inférence
        return self.deux(self.act(self.un(x)))   # un -> tanh -> deux

netDeuxCouches = DeuxCouches()               # instanciation
netDeuxCouches(housing_x)                    # usage en inférence
```

**Pourquoi `super().__init__()` ?** Il active toute la mécanique de `nn.Module` (enregistrement automatique des sous-couches, donc de leurs poids → `net.parameters()` les retrouve).

**Alternative plus courte** (utilisée à la slide 30) : `torch.nn.Sequential([...])` pour une simple chaîne de modules, sans écrire de classe.

### 7.4 La boucle d'apprentissage standard

La standardisation des modules **standardise** le processus d'apprentissage : c'est toujours la même recette.

```python
# Données
housing_x, housing_y = ...
# Définition de :
net   = ...   # réseau de neurones
loss  = ...   # coût
optim = ...   # optimiseur

for i in range(EPOCHS):                                   # boucle d'apprentissage
    loss = mseloss(net(housing_x).view(-1,1), housing_y.view(-1,1))   # 1. forward + coût
    print(f"iteration : {i}, loss : {loss}")
    optim.zero_grad()    # 2. remise à zéro des gradients
    loss.backward()      # 3. calcul des gradients
    optim.step()         # 4. mise à jour des paramètres
```

> ⚠️ **Pourquoi `zero_grad()` ?** PyTorch **accumule** les gradients à chaque `backward()` (il les additionne dans `param.grad`). Si on ne les remet pas à zéro, le gradient de l'itération précédente s'ajoute au nouveau : on obtiendrait de mauvaises mises à jour. **Ordre à retenir : `zero_grad → backward → step`.**

---

## 8. Classification multi-classes : Softmax et cross-entropie

### 8.1 Supervision multi-classes : le one-hot

Pour prédire **$K$ classes**, le réseau a **$K$ sorties** (une par classe). La vérité est codée en **vecteur one-hot** :

$$\mathbf{y} = (0, 0, \dots, 1, \dots, 0) \quad\text{avec } y_k = 1 \text{ pour la bonne classe, } y_i = 0 \text{ sinon}$$

> Pourquoi pas simplement un numéro de classe (1, 2, 3…) ? Parce que cela introduirait un **ordre artificiel** (la classe 3 serait « plus grande » que la 1, et « entre » 2 et 4). Le one-hot traite toutes les classes de façon symétrique.

### 8.2 Pourquoi la MSE est un mauvais choix ici

Avec une sigmoïde en sortie (valeurs dans $[0,1]$) et la MSE : exemple — sortie $(0{,}4;\ 0{,}8;\ 0{,}2;\dots)$, cible $(0; 1; 0;\dots)$, coût par composante $(0{,}16; 0{,}04; 0{,}04; …)$.

Problèmes :
- Ce qui compte pour la classification, c'est le **`argmax`** (quelle sortie est la plus grande), pas la **similarité exacte** avec le vecteur cible.
- On voudrait **maximiser la sortie de la bonne classe** *et* **minimiser celles des autres**, de façon **couplée** (si l'une monte, les autres doivent baisser). Les sigmoïdes indépendantes ne le garantissent pas.

### 8.3 SoftMax : transformer des scores en probabilités

$$\text{SoftMax}(\mathbf{z})_i = \frac{e^{z_i}}{\sum_{j=1}^{K} e^{z_j}}, \qquad \sum_{i=1}^{K}\text{SoftMax}(\mathbf{z})_i = 1$$

**Ce que ça fait** :
1. $e^{z_i}$ : rend tout **positif** et **amplifie les écarts** (le plus grand score écrase les autres → « renforcement du max »).
2. On divise par la somme : les sorties deviennent une **distribution de probabilité** (somme = 1).

*Exemple* : scores $(2, 1, 0)$ → $e^2, e^1, e^0 = (7{,}39;\ 2{,}72;\ 1)$, somme $= 11{,}11$ → probabilités $(0{,}67;\ 0{,}24;\ 0{,}09)$.

### 8.4 Coût cross-entropique (= maximum de vraisemblance multinomial)

$$CE(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{i=1}^{K} y_i \log(\hat{y}_i)$$

Avec un one-hot de la classe $k$, tous les termes sauf un s'annulent :

$$CE(\mathbf{y}, \hat{\mathbf{y}}) = -\log(\hat{y}_k)$$

**Intuition** : on regarde **la probabilité que le modèle a donnée à la bonne classe** et on prend $-\log$ :
- proba = 1 → coût = 0 (parfait) ;
- proba → 0 → coût → +∞ (le modèle est **très** pénalisé s'il était confiant dans la mauvaise réponse).

**Combinaison SoftMax + Cross-entropie** (simplification remarquable) :

$$CE(\mathbf{y}, \text{SoftMax}(\mathbf{z})) = -z_k + \log\Big(\sum_{j=1}^{K} e^{z_j}\Big)$$

$$\boxed{\frac{\partial CE}{\partial z_i} = \text{Softmax}(\mathbf{z})_i - \mathbb{1}_{i=k}}$$

→ Le gradient est simplement **(probabilité prédite) − (probabilité vraie)**. Très intuitif et numériquement stable : c'est pourquoi on associe quasi toujours les deux.

> 🛠️ **En pratique avec PyTorch** *(ajout utile, hors slides)* : `torch.nn.CrossEntropyLoss()` **intègre déjà le SoftMax** (via un `log_softmax` stable). On lui donne donc les **scores bruts (logits)** — **sans** SoftMax avant — et la cible sous forme d'**indices de classe** (entiers), pas de one-hot.

### 8.5 Cross-entropie binaire (BCE) : le cas multi-label

Quand **plusieurs classes peuvent être vraies en même temps** (ex. une image contenant à la fois « chat » et « canapé »), on traite chaque sortie comme **une Bernoulli indépendante** :

$$BCE(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{i=1}^{K} \Big[ y_i \log(\hat{y}_i) + (1 - y_i)\log(1-\hat{y}_i) \Big]$$

| Situation | Sortie | Coût |
|---|---|---|
| **Multi-classes** (1 seule bonne classe) | SoftMax | Cross-entropie |
| **Multi-label** (plusieurs vraies possibles) | Sigmoïde sur chaque sortie | BCE |

---

## 9. Lutter contre le sur-apprentissage

**Le problème (overfitting)** : un réseau très flexible finit par **apprendre par cœur** le jeu d'entraînement (y compris son bruit) et généralise mal à de nouvelles données. Toutes les techniques ci-dessous visent à **régulariser** le réseau.

| Technique | Principe | Intuition |
|---|---|---|
| **Régularisation des couches (L1, L2)** | Ajouter au coût un terme de pénalisation en $\lVert W \rVert^p$ sur les poids | On punit les **gros poids** → modèle plus « simple » et plus lisse. L2 réduit tous les poids ; L1 en annule certains (sparsité). |
| **Dropout** | À chaque itération, on **retire au hasard** quelques neurones | Le réseau ne peut plus compter sur un neurone précis → il doit répartir l'information → plus **robuste**. (Équivaut à entraîner un « ensemble » de sous-réseaux.) |
| **Data augmentation** | Perturber les données d'entrée (rotation, retournement, couleurs, contraste, cisaillement…) | On **multiplie les exemples** en gardant le même label → le modèle apprend les **invariances** (un papillon retourné reste un papillon). |
| **Gradient clipping** | **Borner la norme** du gradient rétro-propagé | Évite les **mises à jour explosives** (instabilité) quand un gradient devient énorme. |

**Dropout** (slide 25) : à gauche le réseau standard, à droite le même réseau après dropout, avec des neurones « barrés » (désactivés pour cette itération).
- ⚠️ En PyTorch : `net.train()` **active** le dropout ; `net.eval()` le **désactive** pour l'évaluation/l'inférence.

**Data augmentation** (slides 25) : exemple du papillon (version originale, dé-texturée, en niveaux de gris, contours renforcés, retournée/pivotée) et des chiffres « 15 » (ShearX, Invert, Solarize, AutoContrast, Equalize…). Ces transformations sont **paramétrables et disponibles dans `torchvision.transforms`** et peuvent être **appliquées automatiquement** à chaque batch.

---

## 10. Améliorer le gradient

### 10.1 Le problème du *gradient vanishing* (gradient qui disparaît)

Le gradient **tend à disparaître** :
- **dans les couches éloignées de la supervision** (début du réseau),
- **dans les sigmoïdes saturées**.

**Pourquoi ?** Rappelons que le gradient d'une couche est un **produit** de termes (chain rule) et contient $g'(a)$. Or pour la sigmoïde :

$$\sigma'(x) = \sigma(x)\big(1-\sigma(x)\big) \le 0{,}25$$

et $\sigma'(x) \approx 0$ dès que $|x|$ est grand (zones plates de la courbe = saturation). Multiplier **plein de nombres < 0,25** à travers de nombreuses couches → un produit qui tend vers 0. Les premières couches **n'apprennent presque plus**.

### 10.2 Solution n°1 : la fonction d'activation ReLU

$$\text{ReLU}(x) = \max(0, x)$$

- Pour $x > 0$ : dérivée **= 1** → le gradient passe **sans être écrasé** (« garde un gradient fort lorsque le neurone est activé »).
- Pour $x < 0$ : sortie et dérivée = 0 (le neurone est « éteint »).
- Bonus : très **simple et rapide** à calculer.

> 🛠️ *Ajout hors slide* : un neurone qui reste toujours dans la zone $x<0$ ne reçoit plus aucun gradient (« dying ReLU ») ; des variantes (LeakyReLU…) existent.

### 10.3 Solution n°2 : normaliser (BatchNorm / LayerNorm)

**Idée** : on **recentre et remet à l'échelle** les valeurs qui circulent dans le réseau, pour obtenir une meilleure **topologie** de la surface de coût.

> 💡 **Pourquoi centrer aide ?** Si les entrées ont des échelles très différentes, la surface de coût ressemble à une **vallée longue et étroite** (ellipses allongées) : le gradient « zigzague » d'un bord à l'autre au lieu d'avancer droit vers le minimum. Avec des données centrées/normalisées, les courbes de niveau deviennent des **cercles** et le chemin vers le minimum est direct (slide 26).

| Normalisation | Statistiques calculées sur… | Remarque |
|---|---|---|
| **BatchNorm** | Chaque **sortie** (feature), **sur tout le mini-batch** | Estimation sur chaque mini-batch ; garde une **moyenne mobile** ($\mu_{mov}$, $\sigma_{mov}$) utilisée à l'inférence |
| **LayerNorm** | Chaque **exemple**, **séparément**, sur **toutes ses dimensions** | Indépendant de la taille du batch |

**Étapes de BatchNorm** (schéma slide 26) :
1. Pour chaque feature $i$ du mini-batch : moyenne $\mu_i = \frac{1}{M}\sum A_i$ et écart-type $\sigma_i$.
2. **Normaliser** : $\hat{A}_i = \dfrac{A_i - \mu_i}{\sigma_i}$.
3. **Remettre à l'échelle et décaler** avec deux paramètres **appris** : $\tilde{BN}_i = \gamma \odot \hat{A}_i + \beta$ (pour que le réseau puisse annuler la normalisation si elle n'est pas utile).
4. Maintenir une moyenne mobile pour l'inférence.

Position typique : `Linear → Activation → BatchNorm → Linear …` (comme sur le schéma du cours).

---

## 11. Optimisation : bien descendre la colline

### 11.1 L'algorithme de descente de gradient

Pour un coût $C = \sum_{i=1}^n (\mathbf{x}_i w - y_i)^2$ :

1. **Initialiser** $\mathbf{w}_0$ ;
2. **En boucle** (en recalculant le gradient à chaque fois) :
$$\mathbf{w}^{t+1} = \mathbf{w}^t - \varepsilon \nabla_{\mathbf{w}} C \qquad (\varepsilon : \text{learning rate})$$

### 11.2 Paysages de coût : convexe vs réaliste

- **Régression linéaire + MSE** : la surface est un **bol convexe** (un seul minimum, on y arrive toujours).
- **Réseau de neurones** : surface **non convexe**, avec :
  - des **minima locaux** (petites vallées qui ne sont pas les plus profondes),
  - un **minimum global**,
  - des **points selles** (plats dans une direction, creux dans une autre — très fréquents en grande dimension).

> En pratique, les minima locaux sont rarement catastrophiques pour les gros réseaux ; les points selles et les plateaux ralentissent surtout l'apprentissage.

### 11.3 Batch, stochastic ou mini-batch ?

| Méthode | Gradient calculé sur… | Trajectoire (slide 28) | Caractéristiques |
|---|---|---|---|
| **Batch gradient descent** | **Tout** le dataset | Droite, régulière (bleue) | Précis mais **lent** et gourmand en mémoire |
| **Stochastic (SGD)** | **1 seul** exemple | Très bruitée (violette) | Rapide par pas, mais très **erratique** |
| **Mini-batch** | Un **petit lot** (ex. 32, 64, 128) | Compromis (verte) | ✅ **Le standard** : bon équilibre précision / vitesse / parallélisme GPU |

> 📌 **« Le nombre d'itérations n'est pas le coût ! »** : le vrai coût de calcul, c'est (nombre d'itérations) × (coût d'une itération). Une itération en SGD est peu chère, une en batch est chère : comparer uniquement le nombre d'itérations est trompeur. Un **epoch** = un passage complet sur toutes les données.

### 11.4 Variantes de la descente de gradient

Il existe de nombreuses variantes très efficaces (*momentum*, *RMSprop*, **Adam**, **AdamW**…) qui **adaptent le pas** et/ou **lissent la direction** (utile quand on zigzague dans une vallée étroite). Références du cours :
- **Sebastian Ruder** : https://ruder.io/optimizing-gradient-descent/
- **Francis Bach** : https://francisbach.com/

*Règle pratique* : **Adam (`lr=1e-3`)** est un excellent choix par défaut.

### 11.5 Fonction d'apprentissage standard (version « complète »)

```python
net = torch.nn.Sequential([...])
net.name = "mon_premier_reseau"
net = net.to(device)                                   # CPU ou GPU
MyLoss = torch.nn.MSELoss()
optim = torch.optim.Adam(params=net.parameters(), lr=1e-3)

summary = SummaryWriter(f"/tmp/logs/model-{time.asctime()}")   # logs TensorBoard
for epoch in tqdm(range(EPOCHS)):                      # tqdm : barre de progression
    net.train()                                        # mode entraînement (dropout actif...)
    cumloss = 0
    for xbatch, ybatch in train_loader:                # un mini-batch à la fois
        xbatch, ybatch = xbatch.to(device), ybatch.to(device)
        outputs = net(xbatch)                          # forward
        loss = MyLoss(outputs.view(-1), ybatch)        # coût
        optim.zero_grad()                              # reset des gradients
        loss.backward()                                # backward
        optim.step()                                   # mise à jour
        cumloss += loss.item()                         # .item() : tenseur -> nombre Python
    summary.add_scalar("loss/train loss", cumloss/len(train_loader), epoch)
```

**Les nouveautés par rapport à la boucle simple :**
- `.to(device)` : envoie modèle **et** données sur le même appareil (GPU si dispo) — sinon erreur.
- `train_loader` : fournit les mini-batchs (voir DataLoader, section 12).
- `net.train()` : mode entraînement (à l'évaluation → `net.eval()`).
- `loss.item()` : récupère un simple nombre (évite de garder le graphe de calcul en mémoire).
- `SummaryWriter` : écrit les courbes pour **TensorBoard**.

### 11.6 Faire varier le learning rate (scheduler)

Garder le même $\varepsilon$ toute la durée n'est pas optimal : grand au début (avancer vite), petit à la fin (se poser précisément). Un **scheduler** ajuste le learning rate au fil des itérations. Exemple avec **OneCycleLR** (monte puis redescend, pour **accélérer** la convergence) :

```python
from torch.optim.lr_scheduler import OneCycleLR

nepoch = 1000
max_lr = 5e-1                                          # version optimisée OneCycle
optimizer = torch.optim.AdamW([noise], lr=max_lr)
scheduler = OneCycleLR(optimizer, max_lr=max_lr, steps_per_epoch=1, epochs=nepoch)

# [...] dans la boucle :
loss = MyLoss(outputs.view(-1), ybatch)
loss.backward()
optimizer.step()
scheduler.step()      # <-- à appeler après optimizer.step()
```

---

## 12. Conclusion : les outils indispensables

### 12.1 Multiplication des modules et des hyperparamètres

- Une architecture = **beaucoup d'hyperparamètres** (nombre de couches, de neurones, learning rate, taille de batch, dropout…).
- Besoin de **normalisation / bonnes pratiques** pour :
  - **mieux comparer** les architectures,
  - avoir des **a priori** sur les bons paramètres (éviter de tout tester).
- **Connaître le lexique des modules** : pour **comprendre les enjeux** et **savoir lire les articles** scientifiques.

### 12.2 `Dataset` & `DataLoader` : le lien entre *n'importe quelles données* et *un modèle*

**Trois étapes** (slide 33) :

1. **Créer une classe `Dataset` personnalisée** : elle définit **comment les données sont chargées** et possède une méthode `__getitem__` (renvoie l'exemple n° *i*) — et `__len__` (nombre d'exemples).
2. **L'instancier** pour obtenir les jeux **train / validation / test**.
3. **Les envoyer dans un `DataLoader`** (classe de PyTorch) qui définit :
   - le **shuffling** (mélange à chaque epoch),
   - la **taille de batch**,
   - le **nombre de processus** de chargement en parallèle,
   - et bien plus encore.

```python
from torch.utils.data import Dataset, DataLoader

class MonDataset(Dataset):
    def __init__(self, X, y):
        self.X, self.y = X, y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, i):                  # renvoie (entrée, cible) pour l'indice i
        return self.X[i], self.y[i]

train_loader = DataLoader(MonDataset(Xtr, ytr), batch_size=64, shuffle=True, num_workers=2)
val_loader   = DataLoader(MonDataset(Xva, yva), batch_size=64, shuffle=False)
```

> 💡 Les trois ensembles ont des rôles distincts : **train** (on apprend dessus), **validation** (on règle les hyperparamètres / on détecte l'overfitting), **test** (évaluation finale, qu'on ne touche qu'une fois). `shuffle=True` pour le train seulement.

### 12.3 Check-pointing

**Sauvegarder régulièrement l'état du modèle** (poids, parfois optimiseur et epoch) pendant l'entraînement :
- on **reprend** un entraînement interrompu,
- on **garde le meilleur modèle** (selon la validation),
- on peut **réutiliser** le modèle plus tard sans ré-entraîner.

```python
torch.save(net.state_dict(), "checkpoint.pt")        # sauvegarde des poids
net.load_state_dict(torch.load("checkpoint.pt"))     # rechargement
```

### 12.4 TensorBoard

Outil de **visualisation du suivi d'entraînement**, à lancer **dans une autre fenêtre (ou sur un autre serveur)**. Il affiche les courbes de `loss` (par *run*, avec lissage), ce qui permet de **comparer plusieurs modèles** (ex. `DeuxCouches` vs `Sequentiel`) et de repérer l'overfitting (loss d'entraînement qui baisse mais pas celle de validation).

```bash
tensorboard --logdir /tmp/logs
```

---

## 13. 🧾 Cheatsheet finale

### 13.1 Les formules essentielles

| Concept | Formule |
|---|---|
| Perceptron | $\hat y = \text{sign}(w_0 + \langle \mathbf{w}, \mathbf{x}\rangle)$ |
| Mise à jour perceptron | si $y\langle \mathbf{w},\mathbf{x}\rangle<0$ : $\mathbf{w} \leftarrow \mathbf{w} + \varepsilon y \mathbf{x}$ |
| Pré-activation / sortie | $a = \sum w x + b$ ; $z = g(a)$ |
| Sigmoïde | $\sigma(x) = \frac{1}{1+e^{-x}}$ ; $\sigma' = \sigma(1-\sigma) \le 0{,}25$ |
| ReLU | $\max(0, x)$ ; dérivée $= 1$ si $x>0$, sinon $0$ |
| MSE | $L = (\hat y - y)^2$ (ou moyenne sur $N$ exemples) |
| Descente de gradient | $w \leftarrow w - \varepsilon\,\partial L/\partial w$ |
| Gradient d'un poids | $\dfrac{\partial L}{\partial w_{ij}} = z_i\,\delta_j$ |
| Erreur rétro-propagée | $\delta_j = g'(a_j)\sum_k w_{jk}\,\delta_k$ |
| Module linéaire (backward) | grad poids $= \delta^{k+1}_n z^k_j$ ; erreur renvoyée $=\sum_n \delta_n w_{jn}$ |
| SoftMax | $\dfrac{e^{z_i}}{\sum_j e^{z_j}}$ |
| Cross-entropie (one-hot) | $-\log(\hat y_k)$ |
| Gradient SoftMax + CE | $\text{Softmax}(\mathbf z)_i - \mathbb{1}_{i=k}$ |
| BCE (multi-label) | $-\sum_i [y_i\log\hat y_i + (1-y_i)\log(1-\hat y_i)]$ |
| BatchNorm | $\hat A = \frac{A-\mu}{\sigma}$ ; $out = \gamma\hat A + \beta$ |

### 13.2 Quelle activation / quel coût choisir ?

| Tâche | Dernière couche | Coût PyTorch | Cible |
|---|---|---|---|
| **Régression** | `Linear(…, 1)` (pas d'activation) | `nn.MSELoss()` | valeur réelle (même forme que la sortie) |
| **Classification binaire** | 1 sortie (logit) | `nn.BCEWithLogitsLoss()` *(ou Sigmoid + `BCELoss`)* | 0 / 1 (float) |
| **Multi-classes** (1 seule vraie) | `Linear(…, K)` (**logits**) | `nn.CrossEntropyLoss()` *(SoftMax intégré)* | **indice** de la classe (int) |
| **Multi-label** | `Linear(…, K)` | `nn.BCEWithLogitsLoss()` | vecteur de 0/1 (float) |

| Activation | Quand ? | Attention |
|---|---|---|
| **ReLU** | **Défaut** pour les couches cachées | neurones « morts » possibles |
| **tanh** | Sortie centrée en $[-1,1]$ | sature → gradient vanishing |
| **sigmoïde** | Sortie binaire / multi-label | sature (gradient ≤ 0,25) |
| **SoftMax** | Sortie multi-classes | ne pas l'ajouter avant `CrossEntropyLoss` |

### 13.3 Anatomie d'un modèle PyTorch

```python
import torch
class MonReseau(torch.nn.Module):
    def __init__(self, d_in, d_hidden, d_out):
        super().__init__()                           # TOUJOURS en premier
        self.fc1 = torch.nn.Linear(d_in, d_hidden)
        self.act = torch.nn.ReLU()
        self.drop = torch.nn.Dropout(0.2)
        self.fc2 = torch.nn.Linear(d_hidden, d_out)
    def forward(self, x):                            # pas de backward à écrire !
        return self.fc2(self.drop(self.act(self.fc1(x))))

# Équivalent compact
net = torch.nn.Sequential(
    torch.nn.Linear(d_in, d_hidden), torch.nn.ReLU(),
    torch.nn.Dropout(0.2), torch.nn.Linear(d_hidden, d_out))
```

### 13.4 Template d'entraînement complet

```python
device = "cuda" if torch.cuda.is_available() else "cpu"
net = MonReseau(...).to(device)
criterion = torch.nn.CrossEntropyLoss()
optim = torch.optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(EPOCHS):
    net.train()                                   # --- entraînement ---
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        loss = criterion(net(xb), yb)             # 1) forward + coût
        optim.zero_grad()                         # 2) reset gradients
        loss.backward()                           # 3) backward
        # torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)   # (optionnel) gradient clipping
        optim.step()                              # 4) mise à jour

    net.eval()                                    # --- validation ---
    with torch.no_grad():                         # pas de gradients : plus rapide, moins de mémoire
        val_loss = sum(criterion(net(x.to(device)), y.to(device)).item()
                       for x, y in val_loader) / len(val_loader)
```

### 13.5 Les 4 lignes magiques (ordre à ne jamais inverser)

```python
loss = criterion(net(x), y)   # 1. forward + coût
optim.zero_grad()             # 2. on vide les anciens gradients
loss.backward()               # 3. on calcule les nouveaux
optim.step()                  # 4. on met à jour les poids
```

### 13.6 Boîte à outils anti-sur-apprentissage

| Outil | Code PyTorch |
|---|---|
| L2 (weight decay) | `torch.optim.Adam(..., weight_decay=1e-4)` |
| Dropout | `torch.nn.Dropout(p=0.2)` + `net.train()` / `net.eval()` |
| Data augmentation | `torchvision.transforms` (`RandomHorizontalFlip`, `RandomRotation`, `ColorJitter`…) |
| Gradient clipping | `torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm)` |
| BatchNorm / LayerNorm | `torch.nn.BatchNorm1d(n)` / `torch.nn.LayerNorm(n)` |
| Scheduler | `OneCycleLR`, `StepLR`, `CosineAnnealingLR` (+ `scheduler.step()`) |
| Early stopping / checkpoint | `torch.save(net.state_dict(), path)` sur le meilleur score de validation |

### 13.7 Diagnostic rapide

| Symptôme | Cause probable | Piste |
|---|---|---|
| Loss **stagne** dès le début | learning rate trop petit, gradient vanishing, activation saturée | ↑ `lr`, ReLU, normaliser les entrées, BatchNorm |
| Loss **explose / NaN** | learning rate trop grand, gradients explosifs | ↓ `lr`, gradient clipping |
| Loss train ↓ mais **val ↑** | **sur-apprentissage** | dropout, L2, data augmentation, moins de couches, early stopping |
| Loss train **et** val élevées | **sous-apprentissage** | modèle plus gros, plus d'epochs, `lr` mieux réglé |
| Erreur de dimension / résultat bizarre | formes `(N,)` vs `(N,1)` | `.view(-1,1)` / `.view(-1)` |
| Erreur « device » | modèle et données sur des appareils différents | `.to(device)` pour les deux |
| Gradients qui s'accumulent | oubli de `zero_grad()` | `optim.zero_grad()` **avant** `backward()` |

### 13.8 Glossaire express

| Terme | Définition en une ligne |
|---|---|
| **Feature** | Variable d'entrée décrivant un exemple |
| **Biais** ($w_0$, $b$) | Poids constant qui décale la fonction |
| **Forward / backward** | Passe de calcul de la prédiction / des gradients |
| **Chain rule** | Dérivée d'une composition = produit des dérivées |
| **Rétro-propagation** | Application de la chain rule en remontant le réseau |
| **$\delta$** | Erreur (gradient du coût) à un neurone / module |
| **Epoch** | Un passage complet sur tout le jeu d'entraînement |
| **Mini-batch** | Petit lot d'exemples utilisé pour un pas de gradient |
| **Learning rate** ($\varepsilon$) | Taille du pas de la descente de gradient |
| **Logits** | Scores bruts avant SoftMax/sigmoïde |
| **One-hot** | Vecteur de 0 avec un seul 1 (la bonne classe) |
| **Overfitting** | Mémorisation du train, mauvaise généralisation |
| **Gradient vanishing** | Gradient qui s'éteint dans les couches profondes |
| **Hyperparamètre** | Réglage fixé *avant* l'apprentissage (lr, nb de couches, batch size…) |
| **Checkpoint** | Sauvegarde de l'état du modèle |

---

*Fin des notes — Module 1 : Introduction au Deep Learning, architectures en PyTorch.*
