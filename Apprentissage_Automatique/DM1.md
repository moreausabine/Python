
<p style="text-align: center;">DM 1</p>

# <p style="text-align: center;">DA IODAA : Cours d'Apprentissage Artificiel</p> 


## Exercice 1 : Les échantillons de Mars

### 1- Spécifiez le S-set et le G-set après la prise en compte de chacune des séquences

Les séquences d'exemple sont : 
| Exemple | Forme | Couleur | Composition | Résultat (Classe) |
| :--- | :--- | :--- | :--- | :---: |
| **E1** | Ellipsoïde | Orange | Amine | **+** |
| **E2** | Sphère | Rouge | Organo-phosphoré | **+** |
| **E3** | Pentaèdre | Vert | Sulfure | **-** |
| **E4** | Sphère | Rose | Amine | **-** |

- **E1** :  
S1 = {(Ellipsoïde & orange & amine)}  
G1 = {(? & ? & ?)}  
- **E2** :  
S2 = {(Conique & clare & organique),(Conique & foncée & organique)}  
G2 = {(? & ? & ?)}  
- **E3** :   
S3 = S2 = {(Conique & clare & organique),(Conique & foncée & organique)}
G3 = {(Conique & ? & ?),{(? & chaude & ?)},{(? & ? & organique)}}  
- **E4** :  
S4 = {(Conique & foncé & organique)}  
G4 = {(? & foncé & ?)} 

Le S-set est donc : {(Conique & foncé & organique)}  
Le G-set est donc : {(? & foncé & ?)}  

<br>

### 2- Comment seront alors classés les exemples suivants et pourquoi ?
En supposant qu'il est préférable d'avoir des faux négatifs que des faux positifs, les nouvelles séquences pourraient être classées ainsi : 
| Exemple | Forme | Couleur | Composition | Résultat (Classe) |
| :--- | :--- | :--- | :--- | :---: |
| **E5** | Sphère | marron | cétone | **+** |
| **E6** | Sphère | bleu | oxyde | **-** |
| **E7** | Pentaèdre | violet | cétone | **-** |

La séquence E5 aurait pour classe **+** car la "sphère" est une forme conique le "marron" est une couleur foncée, les "cétones" sont des organiques. Elle correspond donc parfaitement au S-set défini à la question précédente.

La séquence E6 aurait probablemet pour classe **-** car même si la forme est conique, sa couleur est froide et sa composition est minérale comme dans le cas de la séquence E3 qui était de classe -.

La séquence E7 aurait peut-être pour classe **-** car même si cette fois-ci la couleur et la composition sont respectivement foncée et organique, la forme est un pentaèdre comme dans l'exemple 4 qui était de classe -.


<br>

---

## Exercice 2 : Réflexions sur la fragmentation du G-set

### 1- Pourquoi le G-set peut facilement être de taille exponentielle en le nombre d’exemples d’apprentissage ?

G peut devenir très grand parce qu'un exemple négatif peut créer plusieurs spécialisations possibles, et chacune de ces spécialisations peut ensuite donner naissance à plusieurs autres spécialisations qui dans certains cas peuvent être chacune incomparable (chacune n'est pas plus spécifique que les autres). Cependant ces cas correspondent aux pires des cas lorsqu'il y a beaucoup d'attributs avec des hierarchies complexes. 

Ceci est une intuition cependant je n'ai pas réussi à trouver des exemples permettanty d'illustrer ce phénomène.

<br>

### 2- Quel remède pourrait permettre de limiter de phénomène ? Réfléchissez en particulier à l’utilisation de near-miss (contre-exemples critiques). Quel effet ont-ils sur le G-set ?

Le near-miss correspond au fait de donner des exemples négatifs  proches des exemles positifs.  
Ceci permet de limiter ou diminuer les hypothèses possibles dans le G-set car cela permet d'éliminer les généralisations trop larges. En effet, le but du G-set et S-set sont de délimiter l'espace des versions possible. Exclure des hypothèses proches des hypothèses positives dont déjà inclues dans le S-set permettent donc rapprocher le G-set du S-set et donc d'aider à la convergence.