=====================================================================
                      Description des fichiers
=====================================================================

image_plante : dossier des images utilisées pour l'herbier
plantes_degradees.json : données des différentes plantes
TP Mon herbier : énoncé

Herbier_code_Console.py : code avec affichage graphique dans la console (donc sans image)
Herbier_code_FenetreGraph.py : code avec affichage graphique dans une fenetre graphique (donc avec image) <-- beaucoup plus joli



=====================================================================
                    Logique de la construction
=====================================================================

Décrit que pour le fichier Herbier_code_console_commente.py
Il n'y aucune différence de logique entre les 2 seul l'affichage graphique est différent et s'il y a des input/print dans la classe Quiz (globalement)



1. Démarrage
Tout commence par Interagir().menu() (bloc if __name__ == "__main__"). 
Le constructeur de Interagir crée un unique objet Herbier, qui sera partagé par toutes les actions. menu() tourne ensuite en boucle jusqu'à ce que l'utilisateur tape 0.


2. Architecture générale
Le code est séparé en trois couches :
- Interagir s'occupe uniquement des input et des print.
- Herbier contient toute la logique (nettoyage, ajout, recherche, sauvegarde).
- Quiz gère le jeu de questions, en s'appuyant sur l'herbier.

Plante est une simple classe de données avec son affichage (__str__). 
Les fonctions du haut du fichier (enlever_accents, valider_plante, detecter_doublons, etc.) sont des outils appelés par ces classes.


3. Ajouter une plante (options 1 et 2 du menu)
Toutes les voies d'ajout convergent vers Herbier.integrer(), qui enchaîne trois étapes :
- nettoyer_plante() : transforme le dictionnaire brut en dictionnaire propre. Elle appelle un nettoyeur par champ (nettoyer_nom, nettoyer_famille, nettoyer_cycle, nettoyer_besoin), qui s'appuient tous sur nettoyage_generique() (minuscules, accents retirés, espaces normalisés). Elle appelle aussi valider_plante() et, s'il y a des erreurs, afficher_rapport().
- integrer() rappelle valider_plante() : s'il reste des erreurs, la plante est rejetée.
Sinon, un objet Plante est créé et passé à ajouter_plante(). Celle-ci vérifie les doublons avec detecter_doublons(), ajoute la plante, trie le classeur avec tri_par_nom, puis met à jour les listes de référence (mise_jour_besoins, mise_jour_famille, mise_jour_nom_plante).

La différence entre les deux options :
- Option 2 (menu) : ajouter_plante_herbier() collecte les saisies (champ par champ ou en une ligne avec //), puis appelle integrer() une seule fois.
- Option 1 (menu) : ajouter_pleins_plantes() appelle charger_json(), qui lit le fichier et appelle integrer() pour chaque plante de la liste.


4. Rechercher (option 3)
parcourir_herbier() propose deux modes :

Recherche par nom : appelle rechercher(), qui cherche le texte dans tous les champs (sauf la photo), sans tenir compte des accents ni de la casse.
Filtre par critères : demander() récupère et nettoie chaque critère, qui peut rester vide (via les nettoyeurs de Herbier). Puis filtrer() ne garde que les plantes respectant tous les critères renseignés.

Dans les deux cas, le résultat est affiché par afficher_liste_plante(). L'option 4 utilise le même affichage via afficher_classeur().


5. Supprimer et sauvegarder (options 5 et 6)
- Suppression : le nom saisi est nettoyé avec nettoyer_nom(), puis supprimer_plante() reconstruit le classeur sans la plante visée.
- Sauvegarde : sauvegarder_herbier() appelle sauvegarder_json(), qui convertit chaque Plante en dictionnaire avec vars() puis écrit le JSON.


6. Le quiz (option 7)
lancer_quiz() demande le nombre de questions et le mode (révision ou non), puis crée un objet Quiz et appelle quiz.quiz().

quiz() est le chef d'orchestre :
- Mode classique : quiz_classique() tire au sort à chaque tour entre deux types de question :
question_nom_verna(espece) : « Quelle est la famille de X ? » (elle utilise famille_de() pour la bonne réponse).
question_famille(famille) : « Quelle plante est de la famille Y ? » (elle utilise filtrer() pour trouver les bonnes réponses).

- Mode révision : l'utilisateur choisit une famille, puis quiz_revision() ne pose que des question_famille() sur cette famille.
Chaque question affiche 4 choix mélangés, appelle demander_reponse() pour valider la saisie, et retourne 1 ou 0. La note s'accumule dans self.note.
À la fin, quiz() calcule le pourcentage, affiche le score et un message d'encouragement. Si aucune question n'a pu être posée (herbier vide), il l'indique et s'arrête.

Les éléments déjà interrogés sont retirés de espece_questionable et famille_questionable, pour éviter les répétitions. C'est aussi ce qui limite le nombre de questions possibles.