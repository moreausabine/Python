# <p style="text-align: center;">Introduction</p>

## Une science de l'apprentissage ?

### Pk apprendre ?   
Alors qu'on pourrait simplement coder (funfact c'est plus simple pour prédire la météo de demain c'est plus simple de dire qu'il fera le même temps que aujourd'hui)  

*MAIS* Reconnaitre un chien : on peut simplifier un chien facilement (boite avec des pattes) mais pas si simple car pas mal d'espèce

Y a des trucs qu'il est difficile de coder à la main donc il vaut mieux utiliser l'apprentissage

1. <ins>Reconnaissance, prediction :</ins>   
   exemple : lésions de peau / chihaha et cookie / temps réelsde segmentation automatique (cf voiture ou imagerie satellite) / reconnaissances d'objets  
   exemple : prédiction par séries tempoorelles (pluie, bourse, vente, inscetes ravageurs)


2. <ins>IA générative</ins>   
   (2017 en science)  
   AlphaFold : de séquence protéine à configuration protéine  
   ITHACA : archéologie pour reconstituer à partir de fragments des textes entiers (62% des choses reconstruites)  
   Génération d'images (visage)  
   AlphaGo : tu connais   
   
   Adaptation --> imiter un comportement : enfer sa mère  
   On a gagné la coupe du monde de robot-foot car l'équipe française était la première équipe à avoir appris à mentir  


3. <ins>IA Agentique</ins>  
   Groupe de discussion d'IA qui peut agir sur leur environnement (contrairement à un LLM qui est coincé)  
   *Fun fact* : il peut aller supprimer tout ton disque dur si il en a besoin pour sa mission  

   Exemple : une IA qui aide en discutant avec vous pour faire des hypothèses de recherche et des articles


Nous avons besoin de plus en plus de ressources de calcul. problème universitaire : nous avons pas les capacités que les grandes entreprises ont et en plus il y a du gate keeping car ils ont besoin de garder leur secret pour être encore dans la course. (C'est un peu chiant pour nous du coup)

---

### Y a pleins de données !

Tt d'abord les capteurs :
 - Agriculture  
 - Météo  

Mais aussi en code informatique :  
 - Github  
 - Internet quoi  
 - LLM et autres  

Contre exemple :   
 - Alimentation (nous avons besoin d'historique précis du comportement des gens)  
 - Usage de produits (pollutions par solvants)  
 - Education et ce qu'il y a dans les cerveaux des gens
  
---

### Que sait on faire et que ne pouvons nous pas faire ?

*Attention on ne sait pas si c'est tjs le cas*

Adversarial Learning :  
Il est possible de modifier très légérement des images pour tromper l'IA (du bruits sur les pixels)  

Annotation d'images :  
On peut modifier de façon flagrante pour les humains une image mais les IA ne voient pas la diff (exemple remplacer des images de jeunes par des vieux)

*Exemple :* 
- cas en médecine : on tourne juste une image et les mélanome passe de bénin à cancéreux très facilement
- Cas en image : voiture dans une piscine, la machine ne le perçoit pas car il n'y a jamais eu d'image d'une voiture dans une piscine donc elle est en mode osef alors que les humains sont en mode : euh nope 


**Loi fondamentale de l'apprentissage** : il faut avoir des exemples d'erreurs pour apprendre réellement un langage ou autre.


---

### Brève histoire de l'IA

*Fun Fact :* effet Hans : un cheval qui a "appris à compter" en analysant le comportement de son maître pour savoir quand il faut arreter de compter

**Date premier ordinateur** : 1840  
*Fun Fact :* Elle ne marchait pas car ce n'était pas suffisemment precis comme machine en gros on arrivait pas à savoir si c'était bien un trou ou pas

**Avant 1930** : Tt est force et énergie  
**Après 1930** : Tt devient information et code (Théorie de la communaocation et du coddage claude shannon 1948 et code génétique Watson Crick et Rosalind Franklin)


1. <ins>Pionnier de IA de 1936 à 1956 :</ins>
   - Turing
   - Von Neumann (architecture des ordi)
   - **McCulloth & Pitts (1943) : Premier modèle du neurone !!** : Chose qui reçoit des infos 0/1 et qui renvoit des 0/1. Si en dessous d'un certains seuil de 1 alors elle renvoie un 0 et sinon un 1  
   *Funfact :* 86 miliards de neurone dans notre cerveau  
   Il n'y a rien qui différencie un cerveau d'un ordinateur. Une IA doit être capable de faire autant que nous 
   - Hebb (1949): 
   - Bcp de conférence
    
  

2. <ins>IA comme méthdes généarles de raisonnement (1956-1968)</ins>
   Manipulation de l'info : IA symbolique  
   - Checker qui bat tt le monde sauf 1 personne aux jeux de dame  
   Premier connexionnisme
   - Perceprton premier modyle de neurone 1957-1962  
   
   IA symbolique : il y a pleins de différentes étapes discrètes

   1968 : que sait on faire :  
   - Analogie : résoudre des schémas
   - Eliza : premier début de LLM très primitif mais les gens se trompaient (pensaient que la machine les comprenaient)
   - Reconnaissance des caractères pour utiliser dans les cartes de code

    1968 : les échecs  
    - La traduction : anglais-russe-anglais
    - Les experts ne sont pas experts partout
  

3. <ins>Knowledge is power (1968-1980)</ins>  
   La machine a beosin de bcp de connaissances : mais comment les coder ? et comement les utiliser ?

   *Fun Fact :* BORIS --> une super IA qui saut bien résumer les histoires d'adultères

4. <ins>Méthodes générales d'apprentissage (1980-now)</ins>  
   *Méthode symbolique* : que l'on sait lire
   Nous avons de super modèles mais fait dans des univers parfaits or dans la vraie vie (industriel) il n'y a pas le même formalisme partout donc souci pour raisonner.

   *Connexionnsime* : On ne sait pas ce qu'il se passe dedans mais ils arrivent à résoudre les données bruitées

   --> Les réseaux de neurones s'imposent

---

## Grand type d'apprentissage

# Apprentissage descriptif (non supervisé)
Résumé des données : on te mets dans les caves du Louvre avec 0 indication et on te demande ce qu'il y avait  
Exemple : clustering  (mélange de Gaussienne) ou motifs fréquents

Attention : il y a besoin d'un expert pour confirmer les clusters défini  
Généralement le clustering c'ets dans des trucs exploratoires et il faut savoir faire attention car l'experts peut mal interpréter (trouver des explications même si les groupes sont du hasard)  

Il y a donc une grande dépendance au biais à priori

---

# Apprentissage prédictif (supervisé)
Un échantillon d'apprentissage où on relie entrée et sortie et on veut que la machine retrouve la fonction pour relier les 2

Dans ce cas là on n'a pas besoin de trouver une hypothèse, nous avons déjà l'idée on veut juste répindre à des problèmes

Attention différence entre régression et .... 

---

# Apprentissage prescriptif pour "intervenir"
En quoi différent du prédictif : la sortie peut varier

recherche de causalité

---

# Apprentissage par renforcement

agent qui a un canal de perception, un canal de récompense, un canal d'action

Exemple : tennis de table, hélicoptère, go...




Le système ne sait plus quoi faire si on change un tt petit peu les paramètres ; exemple le tireur qui ne sait plus tirer si le gardien fait n'importe quoi

Organisation des données

---

## Problème de l'apprentissage supervisé

### Problème de l'induction

Tant qu'on a pas toutes les solutions à tous les cas nous ne sommes pa sûres.
La question c'est donc : comment faire de la prédiction ?  

Ici on peut trouver car nous sommes tous biaisés car nous sommes humains (c'est issu du fait que les biais non efficaces sont morts) donc nous sommes capables de trouver des hypothèses et récupérer les solutions.

Il est donc essentiel d'avoir un biais pour pouvoir faire de l'induction, le biais doit être aligné avec ce que le grand architecte de l'univers a décidé. 

C'est quoi un biais : tt ce qui limite l'espace des hypothèses considérées et ce qui va favoriser certaines hypothèses par rapport à d'autres

type de biais :
- représentation : limiter les possibilités
- recherche : faire varier l'ordre des données pour favoriser une hypothèse plutôt qu'une autre

# No free-lunch thorem

Toutes les méthodes d'induction se valent et sont égales au hasard

Explication : *à remplir*


