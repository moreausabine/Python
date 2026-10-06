import json
import random as rd
import unicodedata
from pathlib import Path

# Dossier où se trouve ce script : sert de base pour les chemins des fichiers JSON
dossier_courant = Path(__file__).resolve().parent

# Valeurs acceptées pour le cycle de vie, et correspondance féminin -> forme canonique
CYCLES_AUTORISES = ['vivace', 'annuel', 'bisannuel']
SYNONYMES_CYCLE = {'annuelle': 'annuel', 'bisannuelle': 'bisannuel'}

# =====================================================================
#  Fonctions utiles
# =====================================================================

def tri_par_nom(plante):
    # Clé de tri pour classer l'herbier par ordre alphabétique (utilisée dans ajouter_plante)
    return plante.nom or ""


def enlever_accents(texte):
    """ Enleve les accents"""
    # NFD sépare les lettres de leurs accents, puis on retire les accents (catégorie 'Mn')
    nfd_form = unicodedata.normalize('NFD', texte)
    return "".join([c for c in nfd_form if unicodedata.category(c) != 'Mn'])


def valider_plante(plante):
    """Retourne la liste des erreurs. Liste vide = plante valide.
    Accepte un dictionnaire ou un objet Plante."""
    # vars(objet) transforme un objet en dictionnaire de ses attributs
    d = plante if isinstance(plante, dict) else vars(plante)
    erreurs = []

    # Champs obligatoires
    if not d.get("nom"):
        erreurs.append("nom vernaculaire manquant")
    if not d.get("nom_scientifique"):
        erreurs.append("nom scientifique manquant")
    if not d.get("famille"):
        erreurs.append("famille manquante")

    # Le cycle doit exister et faire partie des valeurs autorisées
    cycle = d.get("cycle")
    if not cycle:
        erreurs.append("cycle manquant")
    elif cycle not in CYCLES_AUTORISES:
        erreurs.append(f"cycle non valide : '{cycle}' (attendu : {', '.join(CYCLES_AUTORISES)})")

    if not isinstance(d.get("besoins"), list):
        erreurs.append("les besoins doivent être une liste")

    # la photo peut être absente : aucun test
    return erreurs


def afficher_rapport(plante, erreurs):
    # Affiche le nom de la plante puis la liste de ses erreurs de validation
    d = plante if isinstance(plante, dict) else vars(plante)
    print(d.get("nom") or d.get("nom_scientifique") or "(plante sans nom)")
    for e in erreurs:
        print(f"  Erreur : {e}")


def detecter_doublons(plantes):
    """Retourne la liste des couples (plante_d_origine, doublon) selon le nom scientifique.
    Complexité O(n) en moyenne grâce au dictionnaire."""
    vues = {}            # nom scientifique -> première plante rencontrée
    doublons = []
    for p in plantes:
        cle = p.nom_scientifique
        if not cle:
            continue
        if cle in vues:
            doublons.append((vues[cle], p))   # déjà vu : c'est un doublon
        else:
            vues[cle] = p
    return doublons


# =====================================================================
#  CLASSES
# =====================================================================

class Plante:
    """ Classe plante : pour avoir le print et  les attributs des plantes"""
    def __init__(self, nom, nom_scientifique, famille, cycle, besoins, photo=None):
        self.nom = nom
        self.nom_scientifique = nom_scientifique
        self.famille = famille
        self.cycle = cycle
        # Si les besoins arrivent en texte "soleil/ombre", on les transforme en liste
        if isinstance(besoins, str):
            besoins = [b.strip() for b in besoins.split("/") if b.strip()]
        self.besoins = besoins
        self.photo = photo

    def __str__(self):
        """ le print """
        return (f"{self.nom} ({self.nom_scientifique}) - {self.famille}\n"
                f"Cycle : {self.cycle}\n"
                f"Besoins : {', '.join(self.besoins)}")


class Herbier:
    """Toute la logique : données, nettoyage, ajout, recherche, sauvegarde."""

    def __init__(self):
        self.classeur = []   # liste des objets Plante
        # Listes de référence (utilisées pour valider les saisies et pour le quiz)
        self.besoins = ['soleil', 'mi-ombre', 'ombre', 'arrosage_faible', 'arrosage_moyen',
                        'arrosage_fort', 'eau_douce', 'eau_de_mer']
        self.cycle = list(CYCLES_AUTORISES)
        self.famille = []       # se remplit au fur et à mesure des ajouts
        self.nom_plante = []    # idem

    # ---------- gérer les données de l'herbier ----------
    def mise_jour_besoins(self, besoins):
        # Ajoute à la liste de référence les nouveaux besoins rencontrés
        if isinstance(besoins, str):
            besoins = besoins.split("/")

        for besoin in besoins:
            besoin = besoin.strip().lower()
            if besoin and besoin not in self.besoins:
                self.besoins.append(besoin)

    def mise_jour_famille(self, famille):
        # Mémorise la famille si elle est nouvelle
        if famille and famille not in self.famille:
            self.famille.append(famille)

    def mise_jour_nom_plante(self, nom_plante):
        # Mémorise le nom si il est nouveau
        if nom_plante and nom_plante not in self.nom_plante:
            self.nom_plante.append(nom_plante)

    def ajouter_plante(self, plante):
        # Dernière étape de l'ajout : reçoit un objet Plante déjà nettoyé et validé
        if not plante.nom:
            print("Plante sans nom : non ajoutée") 
            return False
        # On teste si l'ajout créerait un doublon (même nom scientifique)
        if detecter_doublons(self.classeur + [plante]):
            print('Plante déjà connue donc non apprise')
            print(plante)
            return False
        self.classeur.append(plante)
        self.classeur.sort(key=tri_par_nom)          # classeur toujours trié par nom
        self.mise_jour_besoins(plante.besoins)       # met à jour les listes de référence
        self.mise_jour_famille(plante.famille)
        self.mise_jour_nom_plante(plante.nom)
        return True

    def supprimer_plante(self, nom):
        # On reconstruit le classeur sans la plante visée ; True si quelque chose a été retiré
        avant = len(self.classeur)
        self.classeur = [p for p in self.classeur if p.nom != nom]
        return len(self.classeur) < avant

    # ---------- nettoyage ----------
    def nettoyage_generique(self, valeur):
        # Nettoyage de base : minuscules, sans accents, '_' -> espace, espaces multiples réduits
        if valeur is None:
            return None
        valeur = enlever_accents(str(valeur).strip().lower()).replace('_', ' ')
        valeur = ' '.join(valeur.split())      # espaces multiples -> un seul
        if valeur in ('', 'null'):
            return None
        return valeur

    def nettoyer_nom(self, valeur):
        """Nom vernaculaire ou scientifique : 'Solanum lycopersicum'."""
        nom = self.nettoyage_generique(valeur)
        return nom.capitalize() if nom else None

    def nettoyer_famille(self, valeur):
        famille = self.nettoyage_generique(valeur)
        # Corrige l'ancien suffixe "-ees" (sans accent) en "-eae" (ex : Rosacees -> Rosaceae)
        if famille and famille.endswith("ees"):
            famille = famille[:-3] + "eae"
        return famille.capitalize() if famille else None

    def nettoyer_cycle(self, valeur):
        cycle = self.nettoyage_generique(valeur)
        return SYNONYMES_CYCLE.get(cycle, cycle)     # 'annuelle' -> 'annuel'

    def nettoyer_besoin(self, valeur):
        besoin = self.nettoyage_generique(valeur)
        return besoin.replace(' ', '_') if besoin else None   # 'arrosage regulier' -> 'arrosage_regulier'

    def nettoyer_plante(self, donnees):
        """Prend un dictionnaire brut, retourne un dictionnaire propre avec toujours les 6 clés."""
        # Besoins : texte "a/b" ou liste -> liste nettoyée, sans vides ni doublons
        besoins = donnees.get("besoins")
        if isinstance(besoins, str):       
            besoins = besoins.split("/")
        besoins = [self.nettoyer_besoin(b) for b in (besoins or [])]
        besoins = list(dict.fromkeys(b for b in besoins if b))   # dédoublonne en gardant l'ordre

        photo = donnees.get("photo")
        photo = photo.strip() if isinstance(photo, str) else None

        # Chaque champ passe par son nettoyeur spécifique
        propre = {
            "nom": self.nettoyer_nom(donnees.get("nom")),
            "nom_scientifique": self.nettoyer_nom(donnees.get("nom_scientifique")),
            "famille": self.nettoyer_famille(donnees.get("famille")),
            "cycle": self.nettoyer_cycle(donnees.get("cycle")),
            "besoins": besoins,
            "photo": photo or None,
        }

        # Si le dictionnaire propre a des erreurs, on les affiche (l'appelant décidera quoi faire)
        erreurs = valider_plante(propre)
        if erreurs:
            afficher_rapport(propre, erreurs)
        return propre

    def integrer(self, donnees):
        """Nettoie, valide puis ajoute une plante. Retourne True si elle est ajoutée."""
        # Point d'entrée unique pour ajouter une plante (saisie manuelle ou JSON)
        propre = self.nettoyer_plante(donnees)
        if valider_plante(propre):          # liste non vide = invalide -> on abandonne
            return False
        return self.ajouter_plante(Plante(**propre))   # dict -> objet Plante

    # ---------- chargement / sauvegarde ----------
    def charger_json(self, path_json):
        """Charge un JSON, retourne le nombre de plantes ajoutées (ou None si erreur)."""
        try:
            with open(path_json, "r", encoding="utf-8") as fichier:
                donnees = json.load(fichier)
        except FileNotFoundError:
            print(f"Fichier introuvable : {path_json}")
            return None
        except json.JSONDecodeError:
            print("Le fichier n'est pas un JSON valide")
            return None

        if not isinstance(donnees, list):
            print("Le JSON doit contenir une liste de plantes")
            return None

        # Chaque entrée passe par integrer() ; on compte celles qui ont été ajoutées
        return sum(1 for p in donnees if self.integrer(p))

    def sauvegarder_json(self, nom_fichier):
        """Écrit l'herbier dans un fichier JSON. Retourne True si la sauvegarde a réussi."""
        donnees = [vars(p) for p in self.classeur]   # chaque Plante -> dictionnaire
        try:
            with open(nom_fichier, "w", encoding="utf-8") as f:
                json.dump(donnees, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"Impossible de sauvegarder : {e}")
            return False
        return True

    # ---------- parcourir l'herbier ----------
    def rechercher(self, chaine_car):
        # Recherche texte libre : la chaîne doit apparaître dans un champ quelconque (sauf photo)
        Liste_plante = []
        chaine_car = enlever_accents(chaine_car.strip().lower()).replace('_', ' ')
        for plante in self.classeur:
            for cle, valeur in vars(plante).items():
                if cle == "photo" or valeur is None:
                    continue
                texte = enlever_accents(str(valeur).lower()).replace('_', ' ')
                if chaine_car in texte:
                    Liste_plante.append(plante)
                    break     # un champ suffit, inutile de tester les autres
        return Liste_plante

    def filtrer(self, cycle=None, besoin=None, famille=None):
        # Filtre combinable : une plante est gardée si elle respecte TOUS les critères donnés
        cible = 0    # nombre de critères actifs
        if cycle is not None:
            cible += 1
        if besoin is not None:
            cible += 1
        if famille is not None:
            cible += 1

        Liste_plante = []
        for plante in self.classeur:
            trouve = 0   # nombre de critères respectés par cette plante
            if cycle is not None and plante.cycle == cycle:
                trouve += 1
            if besoin is not None and besoin in plante.besoins:
                trouve += 1
            if famille is not None and plante.famille == famille:
                trouve += 1

            if trouve == cible:
                Liste_plante.append(plante)

        return Liste_plante

    def famille_de(self, nom):
        """Retourne la famille d'une plante (nom vernaculaire ou scientifique), ou None si inconnue."""
        nom = self.nettoyer_nom(nom)
        if not nom:
            return None
        for plante in self.classeur:
            if nom in (plante.nom, plante.nom_scientifique):
                return plante.famille
        return None


class Quiz:
    def __init__(self, herbier, nb_question = 5, revision = False, note = 0):
        self.nb_question_max = int(nb_question) 
        self.herbier = herbier
        self.revision = revision      # True = quiz centré sur une seule famille
        self.note = note              # nombre de bonnes réponses
        # Copies des listes de l'herbier : on y retire les éléments déjà interrogés (pas de répétition)
        self.espece_questionable = list(self.herbier.nom_plante)
        self.famille_questionable = list(self.herbier.famille)
        self.question_posee = 0
 
    def demander_reponse(self, nb_choix):
        """Redemande tant que la saisie n'est pas un numéro entre 1 et nb_choix."""
        rep = input('Numéro de réponse : ')
        while not rep.strip().isdigit() or int(rep) < 1 or int(rep) > nb_choix: 
            print('Ecrivez un numéro de réponse juste')
            rep = input('Numéro de réponse : ')
        return int(rep)
 
    def question_nom_verna(self, espece):
        # Question type 1 : "Quelle est la famille de <espèce> ?"  (retourne 1 si juste, 0 sinon)
        print(f"Quelle est la famille de {espece} ?")
        reponse_juste = self.herbier.famille_de(espece)
        if reponse_juste in self.famille_questionable: 
            self.famille_questionable.remove(reponse_juste)
        # 1 bonne réponse + jusqu'à 3 mauvaises tirées au hasard, puis mélange
        Reponse = [reponse_juste]
        mauvaises = [f for f in self.herbier.famille if f != reponse_juste]
        Reponse += rd.sample(mauvaises, min(3, len(mauvaises)))
 
        rd.shuffle(Reponse)
 
        for k in range(len(Reponse)):
            print(f"{k+1} - {Reponse[k]}")
 
        rep = self.demander_reponse(len(Reponse))
 
        if Reponse[rep-1] == reponse_juste :
            print("Félicitation ! Vous avez trouvé la bonne réponse :)")
            return 1
        else :
            print(f"Dommage... La bonne réponse était {reponse_juste}")
            return 0
 
 
    def question_famille(self, miff):
        # Question type 2 : "Quelle plante est de la famille <miff> ?"  (retourne 1 ou 0)
        especes_correctes = [p.nom for p in self.herbier.filtrer(famille = miff)]
        print(f"Quelle plante est de la famille des {miff} ?")

        # On privilégie une espèce pas encore interrogée ; sinon on reprend n'importe laquelle
        candidats = [e for e in especes_correctes if e in self.espece_questionable] or especes_correctes
        reponse_juste = rd.choice(candidats)
        if reponse_juste in self.espece_questionable:
            self.espece_questionable.remove(reponse_juste)
 
        # Mauvaises réponses = plantes d'autres familles
        Reponse = [reponse_juste]
        mauvaises = [e for e in self.herbier.nom_plante if e not in especes_correctes] 
        Reponse += rd.sample(mauvaises, min(3, len(mauvaises)))
 
        rd.shuffle(Reponse)
 
        for k in range(len(Reponse)):
            print(f"{k+1} - {Reponse[k]}")
 
        rep = self.demander_reponse(len(Reponse))
 
        if Reponse[rep-1] == reponse_juste :
            print("Félicitation ! Vous avez trouvé la bonne réponse :)")
            return 1
        else :
            print(f"Dommage... La bonne réponse était {reponse_juste} :(")
            return 0
 
 
    def quiz_revision(self, miff):
        # Mode révision : uniquement des questions de type 2 sur la famille choisie
        self.espece_questionable = [p.nom for p in self.herbier.filtrer(famille = miff)]
        while self.question_posee < self.nb_question_max and len(self.espece_questionable) > 0:
            print('-------')
            print(f'Question {self.question_posee+1} sur la famille des {miff}')
            self.note = self.note + self.question_famille(miff)
            self.question_posee = self.question_posee + 1
 
 
    def quiz_classique(self):
        # Mode classique : tirage au sort entre question type 1 (espèce) et type 2 (famille)
        while self.question_posee < self.nb_question_max and len(self.espece_questionable) > 0 and len(self.famille_questionable) > 0:
            print('-------')
            print(f'Question {self.question_posee+1}')
            a = rd.randint(0,1)
            if a == 0:
                espece = rd.choice(self.espece_questionable)
                self.espece_questionable.remove(espece)
                point = self.question_nom_verna(espece)
            else : 
                miff = rd.choice(self.famille_questionable)
                self.famille_questionable.remove(miff)
                point = self.question_famille(miff)
            self.note = self.note + point
            self.question_posee = self.question_posee + 1
 
 
 
    def quiz(self):
        # Chef d'orchestre du quiz : choisit le mode, puis affiche le résultat final
        if self.revision : 
            print("Vous venez de lancer un quiz de révision, quelle famille de plante voulez vous résever ?")
            i = 0
            for fam in self.herbier.famille : 
                i = i + 1     
                print(f'{i} - {fam}')
            a = input('Numéro de la famille de plante (une seule) : ')
            while not a.strip().isdigit() or int(a)<1 or int(a)>len(self.herbier.famille):
                print('Veuillez sélectionner un numéro de famille correcte')
                a = input('Numéro de la famille de plante (une seule) : ')
 
            self.quiz_revision(self.herbier.famille[int(a)-1])
                
        else :
            self.quiz_classique()
 
        # Aucune question posée (herbier vide) : pas de score à afficher
        if self.question_posee == 0:
            print("Aucune question possible (herbier vide ?)")
            return
 
        pourcentage = round(self.note/self.question_posee*100, 1)
 
        print("=============================")
        print("          RESULTAT           ")
        print("=============================")
        print(f'Score : {self.note}/{self.question_posee}')
        print(f'Pourcentage : {pourcentage} %')
        print("=============================")
 
        # Message selon le score
        if pourcentage == 100 :
            print('Excellent !')
        elif pourcentage >= 80:
            print('Très bien !')
        elif pourcentage >= 60:
            print('Bien mais révise encore un peu plus !')
        else:
            print('Revoyez vos famille de plantes !')
        print("=============================")


# =====================================================================
#  INTERFACE GRAPHIQUE (Console)
# =====================================================================

class Interagir:
    """Interface utilisateur : saisies (input) et affichages (print) uniquement.
    Tout le reste est délégué à self.herbier et self.quiz"""

    def __init__(self):
        self.herbier = Herbier()    # un seul herbier, partagé par toutes les actions du menu

    # ---------- affichage ----------
    def afficher_classeur(self):
        self.afficher_liste_plante(self.herbier.classeur)

    def afficher_liste_plante(self, Liste_plante):
        # Affichage commun à "tout afficher" et aux résultats de recherche / filtre
        print("------------------------")
        if not Liste_plante:
            print('Aucune correspondance')
        else:
            for plante in Liste_plante:
                print(plante)         # appelle Plante.__str__
                print("---")
        print("------------------------")

    # ---------- saisie ----------
    def demander(self, question, valides, nettoyeur=None):
        """Retourne None si vide, sinon une valeur valide."""
        # Redemande tant que la saisie nettoyée n'est ni vide ni dans la liste "valides"
        nettoyeur = nettoyeur or self.herbier.nettoyage_generique
        while True:
            rep = nettoyeur(input(question))
            if rep is None or rep in valides:
                return rep
            print("Valeur non valide, réessayez (ou laissez vide).")

    # ---------- chargement / sauvegarde ----------
    def ajouter_pleins_plantes(self):
        # Option 1 du menu : charge un fichier JSON complet
        chemin = input("Chemin du fichier JSON (vide = plantes_degradees.json) : ").strip()
        chemin = chemin or 'plantes_degradees.json'
        n = self.herbier.charger_json(dossier_courant / chemin)
        if n is not None:
            print(f"{n} plante(s) ajoutée(s)")

    def sauvegarder_herbier(self):
        # Option 6 du menu
        chemin = input("Nom du fichier (vide = mon_herbier.json) : ").strip() or "mon_herbier.json"
        if self.herbier.sauvegarder_json(dossier_courant / chemin):
            print("Herbier sauvegardé")

    # ---------- ajout ----------
    def ajouter_plante_herbier(self):
        # Option 2 du menu : deux façons de saisir, qui aboutissent au même dictionnaire "donnees"
        choix = input('Ajouter une plante paramètre par paramètre (1) ou en une liste (2) ? : ').strip()

        if choix == '1':
            # Saisie guidée, champ par champ
            nom = input('Nom vernaculaire ? : ')
            nom_scientifique = input('Nom scientifique ? : ')
            famille = input('Famille ? : ')

            cycle = None
            while cycle is None:   # cycle  obligatoire
                cycle = self.demander('Cycle ? (annuel/bisannuel/vivace) : ',
                                      self.herbier.cycle, self.herbier.nettoyer_cycle)
                if cycle is None:
                    print("Le cycle est obligatoire")

            besoins = []
            while True:
                besoins.append(input('Besoin ? (ex : soleil, arrosage faible) : '))
                if input('Ajouter un autre besoin ? (oui/non) ').strip().lower() != 'oui':
                    break

            photo = input('Nom du fichier photo (vide = aucune) : ')
            donnees = {"nom": nom, "nom_scientifique": nom_scientifique, "famille": famille,
                       "cycle": cycle, "besoins": besoins, "photo": photo}

        elif choix == '2':
            # Saisie rapide en une ligne, champs séparés par //
            saisie = input('nom_verna//nom_scien//famille//cycle//besoin1/besoin2//photo : ')
            attributs = [e.strip() for e in saisie.split("//")]
            if len(attributs) < 5:
                print("Format invalide : il faut au moins 5 champs séparés par //")
                return
            donnees = {"nom": attributs[0], "nom_scientifique": attributs[1],
                       "famille": attributs[2], "cycle": attributs[3],
                       "besoins": attributs[4],
                       "photo": attributs[5] if len(attributs) > 5 else None}
        else:
            print("Choix non valide")
            return

        # Nettoyage + validation + ajout sont gérés par Herbier.integrer
        if self.herbier.integrer(donnees):
            print("Plante ajoutée")

    # ---------- parcours ----------
    def parcourir_herbier(self):
        # Option 3 du menu : recherche libre par nom, ou filtre par critères
        print("Voulez-vous rechercher les plantes par :")
        print("(1) nom")
        print("(2) famille, cycle, besoin")
        choix = input().strip()
        if choix == '1':
            nom = input('Nom recherché : ')
            resultats = self.herbier.rechercher(nom)
            self.afficher_liste_plante(resultats)
        elif choix == '2':
            print("Laissez vide un paramètre que vous ne voulez pas utiliser.")
            cycle = self.demander('Cycle recherché ? (vide = ignorer) : ',
                                  self.herbier.cycle, self.herbier.nettoyer_cycle)
            besoin = self.demander('Besoin recherché ? (vide = ignorer) : ',
                                   self.herbier.besoins, self.herbier.nettoyer_besoin)
            famille = self.demander('Famille recherchée ? (vide = ignorer) : ',
                                    self.herbier.famille, self.herbier.nettoyer_famille)
            resultats = self.herbier.filtrer(cycle, besoin, famille)
            self.afficher_liste_plante(resultats)
        else:
            print("Choix non valide")

    # ---------- quiz ----------        
    def lancer_quiz(self):
        # Option 7 du menu : configure le quiz (nombre de questions, mode) puis le lance
        print('Vous allez lancer un quiz quelques questions de configuration :')
        print('1 - Combien de questions voulez vous faire ?')
        print("(Attention nombre maximal de question limité par le nombre d'espèce dans l'herbier)")
        nb_question = input('(chiffre positif obligatoire !) ')
        while not nb_question.strip().isdigit() or int(nb_question) <= 0:
            nb_question = input('(chiffre positif obligatoire !) ')
        nb_question = int(nb_question)
 
        mauvaise_reponse = False
        while not mauvaise_reponse :     # boucle jusqu'à une réponse oui/non valide
            print("2 - Serait ce un quiz de révision d'une famille particulière ?")
            rev = input('(oui/non)').strip().lower()
            if rev == 'oui' :
                quiz = Quiz(self.herbier, nb_question = nb_question, revision=True)
                quiz.quiz()
                mauvaise_reponse = True
            elif rev == 'non' :
                quiz = Quiz(self.herbier, nb_question = nb_question)
                quiz.quiz()
                mauvaise_reponse = True




    # ---------- menu ----------
    def menu(self):
        # Boucle principale : affiche le menu et appelle la méthode correspondant au choix
        while True:
            print("\n===== HERBIER =====")
            print("1. Charger des plantes depuis un fichier JSON")
            print("2. Ajouter une plante")
            print("3. Rechercher / filtrer les plantes")
            print("4. Afficher tout l'herbier")
            print("5. Supprimer une plante")
            print("6. Sauvegarder l'herbier")
            print("7. Lancer un quiz")
            print("0. Quitter")
            choix = input("Votre choix : ").strip()

            if choix == '1':
                self.ajouter_pleins_plantes()
            elif choix == '2':
                self.ajouter_plante_herbier()
            elif choix == '3':
                self.parcourir_herbier()
            elif choix == '4':
                self.afficher_classeur()
            elif choix == '5':
                nom = self.herbier.nettoyer_nom(input("Nom de la plante à supprimer : "))
                if self.herbier.supprimer_plante(nom):
                    print("Plante supprimée")
                else:
                    print("Aucune plante de ce nom")
            elif choix == '6':
                self.sauvegarder_herbier()
            elif choix == '7':
                self.lancer_quiz()
            elif choix == '0':
                print("À bientôt !")
                break
            else:
                print("Choix non valide")


# =====================================================================
#  Tests rapides
# =====================================================================

# Point d'entrée : le menu ne se lance que si on exécute ce fichier directement
if __name__ == "__main__":
    Interagir().menu()