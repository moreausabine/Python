
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
S2 = {(Conique & chaude & organique)}  
G2 = {(? & ? & ?)}  
- **E3** :   
S3 = S2 = {(Conique & chaude & organique)}  
G3 = {(Conique & ? & ?),{(? & chaude & ?)},{(? & ? & organique)}}  
- **E4** :  
S4 = {(Conique & chaude & organique)}  
G4 = {(Conique & ? & ?),(? & chaude & ?)} 

Le S-set est donc : {(Conique & chaude & organique)}  
Le G-set est donc : {(Conique & ? & ?),(? & chaude & ?)}  

### 2- Comment seront alors classés les exemples suivants et pourquoi ?
En supposant que les classes ne soient pas réparties aléatoirement, que j'ai les même biais que la personne ayant fait le DM et qu'il est préférable d'avoir des faux négatifs que des faux positifs, les nouvelles séquences pourraient être classées ainsi : 
| Exemple | Forme | Couleur | Composition | Résultat (Classe) |
| :--- | :--- | :--- | :--- | :---: |
| **E5** | Sphère | marron | cétone | **+** |
| **E6** | Sphère | bleu | oxyde | **-** |
| **E7** | Pentaèdre | violet | cétone | **-** |

La séquence E5 aurait pour classe **+** car la "sphère" est une forme conique le "marron" est une couleur foncée, les "cétones" sont des organiques. Elle correspond donc parfaitement au S-set défini à la question précédente.

La séquence E6 aurait probablemet pour classe **-** car même si la forme est conique, sa couleur est froide et sa composition est minérale comme dans le cas de la séquence E3 qui était de classe -.

La séquence E7 aurait peut-être pour classe **-** car même si cette fois-ci la couleur et la composition sont respectivement foncée et organique, la forme est un pentaèdre comme dans l'exemple 4 qui était de classe -.

---

## Exercice 2 : Réflexions sur la fragmentation du G-set

### 1- Pourquoi le G-set peut facilement être de taille exponentielle en le nombre d’exemples d’apprentissage ?

La question demande d'expliquer pourquoi, au fur et à mesure que l'on traite des exemples, le nombre d'hypothèses conservées dans le $G$-set peut augmenter de manière très rapide (exponentielle) au lieu de rester petit.