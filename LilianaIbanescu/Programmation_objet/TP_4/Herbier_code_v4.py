import contextlib
import io
import json
import shutil
import sys
import tkinter as tk
import unicodedata
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

# Pillow sert à afficher les photos : pip install pillow
try:
    from PIL import Image, ImageTk
    PILLOW_OK = True
except ImportError:
    PILLOW_OK = False

# Récupère le dossier contenant actuellement ce fichier
dossier_courant = Path(__file__).resolve().parent

DOSSIER_PHOTOS = dossier_courant / "image_plante"

CYCLES_AUTORISES = ['vivace', 'annuel', 'bisannuel']
SYNONYMES_CYCLE = {'annuelle': 'annuel', 'bisannuelle': 'bisannuel'}

# =====================================================================
#  Fonctions utiles
# =====================================================================

def tri_par_nom(plante):
    return plante.nom or ""


def enlever_accents(texte):
    nfd_form = unicodedata.normalize('NFD', texte)
    return "".join([c for c in nfd_form if unicodedata.category(c) != 'Mn'])


def valider_plante(plante):
    """Retourne la liste des erreurs. Liste vide = plante valide.
    Accepte un dictionnaire ou un objet Plante."""
    d = plante if isinstance(plante, dict) else vars(plante)
    erreurs = []

    if not d.get("nom"):
        erreurs.append("nom vernaculaire manquant")
    if not d.get("nom_scientifique"):
        erreurs.append("nom scientifique manquant")
    if not d.get("famille"):
        erreurs.append("famille manquante")

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
            doublons.append((vues[cle], p))
        else:
            vues[cle] = p
    return doublons


# =====================================================================
#  CLASSES
# =====================================================================

class Plante:
    def __init__(self, nom, nom_scientifique, famille, cycle, besoins, photo=None):
        self.nom = nom
        self.nom_scientifique = nom_scientifique
        self.famille = famille
        self.cycle = cycle
        if isinstance(besoins, str):
            besoins = [b.strip() for b in besoins.split("/") if b.strip()]
        self.besoins = besoins
        self.photo = photo

    def __str__(self):
        return (f"{self.nom} ({self.nom_scientifique}) - {self.famille}\n"
                f"Cycle : {self.cycle}\n"
                f"Besoins : {', '.join(self.besoins)}")


class Herbier:
    """Toute la logique : données, nettoyage, ajout, recherche, sauvegarde."""

    def __init__(self):
        self.classeur = []
        self.besoins = ['soleil', 'mi-ombre', 'ombre', 'arrosage_faible', 'arrosage_moyen',
                        'arrosage_fort', 'eau_douce', 'eau_de_mer']
        self.cycle = list(CYCLES_AUTORISES)
        self.famille = []

    # ---------- gérer les données de l'herbier ----------
    def mise_jour_besoins(self, besoins):
        if isinstance(besoins, str):
            besoins = besoins.split("/")

        for besoin in besoins:
            besoin = besoin.strip().lower()
            if besoin and besoin not in self.besoins:
                self.besoins.append(besoin)

    def mise_jour_famille(self, famille):
        # les familles sont déjà normalisées : pas de .lower()
        if famille and famille not in self.famille:
            self.famille.append(famille)

    def ajouter_plante(self, plante):
        if not plante.nom:
            print("Plante sans nom : non ajoutée")
            return False
        if detecter_doublons(self.classeur + [plante]):
            print('Plante déjà connue donc non apprise')
            print(plante)
            return False
        self.classeur.append(plante)
        self.classeur.sort(key=tri_par_nom)
        self.mise_jour_besoins(plante.besoins)
        self.mise_jour_famille(plante.famille)
        return True

    def supprimer_plante(self, nom):
        avant = len(self.classeur)
        self.classeur = [p for p in self.classeur if p.nom != nom]
        return len(self.classeur) < avant

    # ---------- nettoyage ----------
    def nettoyage_generique(self, valeur):
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
        besoins = donnees.get("besoins")
        if isinstance(besoins, str):                      # "soleil/arrosage faible"
            besoins = besoins.split("/")
        besoins = [self.nettoyer_besoin(b) for b in (besoins or [])]
        besoins = list(dict.fromkeys(b for b in besoins if b))   # enlève les vides et les doublons

        photo = donnees.get("photo")
        photo = photo.strip() if isinstance(photo, str) else None

        propre = {
            "nom": self.nettoyer_nom(donnees.get("nom")),
            "nom_scientifique": self.nettoyer_nom(donnees.get("nom_scientifique")),
            "famille": self.nettoyer_famille(donnees.get("famille")),
            "cycle": self.nettoyer_cycle(donnees.get("cycle")),
            "besoins": besoins,
            "photo": photo or None,
        }

        erreurs = valider_plante(propre)      # champs obligatoires + données manquantes
        if erreurs:
            afficher_rapport(propre, erreurs)
        return propre

    def integrer(self, donnees):
        """Nettoie, valide puis ajoute une plante. Retourne True si elle est ajoutée."""
        propre = self.nettoyer_plante(donnees)
        if valider_plante(propre):            # erreurs déjà affichées par nettoyer_plante
            return False
        return self.ajouter_plante(Plante(**propre))

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

        return sum(1 for p in donnees if self.integrer(p))

    def sauvegarder_json(self, nom_fichier):
        """Écrit l'herbier dans un fichier JSON. Retourne True si la sauvegarde a réussi."""
        donnees = [vars(p) for p in self.classeur]
        try:
            with open(nom_fichier, "w", encoding="utf-8") as f:
                json.dump(donnees, f, ensure_ascii=False, indent=2)
        except OSError as e:
            print(f"Impossible de sauvegarder : {e}")
            return False
        return True

    # ---------- parcourir l'herbier ----------
    def rechercher(self, chaine_car):
        Liste_plante = []
        chaine_car = enlever_accents(chaine_car.strip().lower()).replace('_', ' ')
        for plante in self.classeur:
            for cle, valeur in vars(plante).items():
                if cle == "photo" or valeur is None:
                    continue
                texte = enlever_accents(str(valeur).lower()).replace('_', ' ')
                if chaine_car in texte:
                    Liste_plante.append(plante)
                    break
        return Liste_plante

    def filtrer(self, cycle, besoin, famille):
        cible = 0
        if cycle is not None:
            cible += 1
        if besoin is not None:
            cible += 1
        if famille is not None:
            cible += 1

        Liste_plante = []
        for plante in self.classeur:
            trouve = 0
            if cycle is not None and plante.cycle == cycle:
                trouve += 1
            if besoin is not None and besoin in plante.besoins:
                trouve += 1
            if famille is not None and plante.famille == famille:
                trouve += 1

            if trouve == cible:
                Liste_plante.append(plante)

        return Liste_plante


class Interagir:
    """Interface utilisateur : saisies (input) et affichages (print) uniquement.
    Tout le reste est délégué à self.herbier."""

    def __init__(self):
        self.herbier = Herbier()

    # ---------- affichage ----------
    def afficher_classeur(self):
        self.afficher_liste_plante(self.herbier.classeur)

    def afficher_liste_plante(self, Liste_plante):
        print("------------------------")
        if not Liste_plante:
            print('Aucune correspondance')
        else:
            for plante in Liste_plante:
                print(plante)
                print("---")
        print("------------------------")

    # ---------- saisie ----------
    def demander(self, question, valides, nettoyeur=None):
        """Retourne None si vide, sinon une valeur valide."""
        nettoyeur = nettoyeur or self.herbier.nettoyage_generique
        while True:
            rep = nettoyeur(input(question))
            if rep is None or rep in valides:
                return rep
            print("Valeur non valide, réessayez (ou laissez vide).")

    # ---------- chargement / sauvegarde ----------
    def ajouter_pleins_plantes(self):
        chemin = input("Chemin du fichier JSON (vide = plantes_degradees.json) : ").strip()
        chemin = chemin or 'plantes_degradees.json'
        n = self.herbier.charger_json(dossier_courant / chemin)
        if n is not None:
            print(f"{n} plante(s) ajoutée(s)")

    def sauvegarder_herbier(self):
        chemin = input("Nom du fichier (vide = mon_herbier.json) : ").strip() or "mon_herbier.json"
        if self.herbier.sauvegarder_json(dossier_courant / chemin):
            print("Herbier sauvegardé")

    # ---------- ajout ----------
    def ajouter_plante_herbier(self):
        choix = input('Ajouter une plante paramètre par paramètre (1) ou en une liste (2) ? : ').strip()

        if choix == '1':
            nom = input('Nom vernaculaire ? : ')
            nom_scientifique = input('Nom scientifique ? : ')
            famille = input('Famille ? : ')

            cycle = None
            while cycle is None:   # le cycle est obligatoire
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
            saisie = input('nom_verna//nom_scien//famille//cycle//besoin1/besoin2//photo : ')
            attributs = [e.strip() for e in saisie.split("//")]   # on garde les champs vides
            if len(attributs) < 5:
                print("Format invalide : il faut au moins 5 champs séparés par //")
                return
            donnees = {"nom": attributs[0], "nom_scientifique": attributs[1],
                       "famille": attributs[2], "cycle": attributs[3],
                       "besoins": attributs[4],        # "besoin1/besoin2" : nettoyer_plante sait le découper
                       "photo": attributs[5] if len(attributs) > 5 else None}
        else:
            print("Choix non valide")
            return

        if self.herbier.integrer(donnees):
            print("Plante ajoutée")

    # ---------- parcours ----------
    def parcourir_herbier(self):
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

    # ---------- menu ----------
    def menu(self):
        while True:
            print("\n===== HERBIER =====")
            print("1. Charger des plantes depuis un fichier JSON")
            print("2. Ajouter une plante")
            print("3. Rechercher / filtrer les plantes")
            print("4. Afficher tout l'herbier")
            print("5. Supprimer une plante")
            print("6. Sauvegarder l'herbier")
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
            elif choix == '0':
                print("À bientôt !")
                break
            else:
                print("Choix non valide")


# =====================================================================
#  INTERFACE GRAPHIQUE (tkinter + Pillow)
# =====================================================================

class AppHerbier(tk.Tk):
    """Fenêtre graphique. Comme Interagir, elle ne contient que de l'interface :
    toute la logique est déléguée à self.herbier."""

    TOUS = "(tous)"
    FOND = "#f4f1ea"
    VERT = "#2e5339"
    VERT_CLAIR = "#4a7c59"
    ROUGE = "#a8403a"
    TAILLE_PHOTO = (360, 300)

    def __init__(self, herbier=None):
        super().__init__()
        self.title("Mon herbier")
        self.geometry("1020x640")
        self.minsize(940, 580)
        self.configure(bg=self.FOND)

        self.herbier = herbier or Herbier()
        self.plantes_affichees = []     # plantes actuellement dans la liste
        self.photo_tk = None            # référence à garder, sinon l'image disparaît
        self.ecrans = {}
        self.boutons_menu = {}
        self.combos = {}
        self.mappings = {}              # texte affiché -> valeur réelle, pour chaque filtre
        self.vars_besoins = {}

        ttk.Style(self).theme_use("clam")
        self.creer_menu()
        self.zone = tk.Frame(self, bg=self.FOND)
        self.zone.pack(side="left", fill="both", expand=True)
        self.creer_ecran_accueil()
        self.creer_ecran_parcourir()
        self.creer_ecran_ajouter()
        self.afficher_ecran("accueil")

    # ------------------------------------------------------------------
    #  Outils
    # ------------------------------------------------------------------
    def capturer(self, fonction, *args):
        """Exécute une méthode de Herbier et récupère ce qu'elle a 'print' (rapports d'erreurs...)."""
        tampon = io.StringIO()
        with contextlib.redirect_stdout(tampon):
            resultat = fonction(*args)
        return resultat, tampon.getvalue().strip()

    def bouton(self, parent, texte, commande, couleur=None, taille=12):
        couleur = couleur or self.VERT
        return tk.Button(parent, text=texte, command=commande, font=("Arial", taille),
                         bg=couleur, fg="white", activebackground=self.VERT_CLAIR,
                         activeforeground="white", relief="flat", padx=14, pady=7,
                         cursor="hand2")

    def nouvel_ecran(self, nom):
        frame = tk.Frame(self.zone, bg=self.FOND)
        self.ecrans[nom] = frame
        return frame

    def fenetre_resume(self, titre, resume, details):
        """Message simple, ou fenêtre avec les détails s'il y en a."""
        if not details:
            messagebox.showinfo(titre, resume)
            return
        fen = tk.Toplevel(self)
        fen.title(titre)
        fen.geometry("540x400")
        fen.transient(self)
        tk.Label(fen, text=resume, font=("Arial", 12, "bold")).pack(pady=(10, 4))
        tk.Label(fen, text="Détails :").pack(anchor="w", padx=12)
        cadre = tk.Frame(fen)
        cadre.pack(fill="both", expand=True, padx=10)
        scroll = ttk.Scrollbar(cadre)
        scroll.pack(side="right", fill="y")
        texte = tk.Text(cadre, wrap="word", yscrollcommand=scroll.set, font=("Courier", 10))
        texte.pack(side="left", fill="both", expand=True)
        scroll.config(command=texte.yview)
        texte.insert("1.0", details)
        texte.config(state="disabled")
        ttk.Button(fen, text="Fermer", command=fen.destroy).pack(pady=8)

    # ------------------------------------------------------------------
    #  Menu latéral et navigation entre écrans
    # ------------------------------------------------------------------
    def creer_menu(self):
        barre = tk.Frame(self, bg=self.VERT, width=210)
        barre.pack(side="left", fill="y")
        barre.pack_propagate(False)
        tk.Label(barre, text="HERBIER", font=("Arial", 18, "bold"),
                 bg=self.VERT, fg="white").pack(pady=(25, 25))

        ecrans = [("accueil", "Accueil"),
                  ("parcourir", "Parcourir l'herbier"),
                  ("ajouter", "Ajouter une plante")]
        for cle, texte in ecrans:
            b = self.bouton_menu(barre, texte, lambda c=cle: self.afficher_ecran(c))
            self.boutons_menu[cle] = b

        tk.Frame(barre, bg="#6f9a7b", height=1).pack(fill="x", padx=15, pady=12)
        self.bouton_menu(barre, "Charger un JSON", self.action_charger)
        self.bouton_menu(barre, "Sauvegarder", self.action_sauvegarder)
        self.bouton_menu(barre, "Quitter", self.destroy, pack_bas=True)

    def bouton_menu(self, parent, texte, commande, pack_bas=False):
        b = tk.Button(parent, text=texte, command=commande, font=("Arial", 12), anchor="w",
                      bg=self.VERT, fg="white", activebackground=self.VERT_CLAIR,
                      activeforeground="white", relief="flat", padx=18, pady=9, cursor="hand2")
        if pack_bas:
            b.pack(side="bottom", fill="x", pady=(0, 15))
        else:
            b.pack(fill="x")
        return b

    def afficher_ecran(self, nom):
        for ecran in self.ecrans.values():
            ecran.pack_forget()
        self.ecrans[nom].pack(fill="both", expand=True)
        for cle, b in self.boutons_menu.items():
            b.config(bg=self.VERT_CLAIR if cle == nom else self.VERT)

        if nom == "accueil":
            self.maj_stats()
        elif nom == "parcourir":
            self.maj_filtres()
            self.rafraichir()
        elif nom == "ajouter":
            self.maj_cases_besoins()

    def maj_apres_modif(self):
        """À appeler quand le contenu de l'herbier a changé."""
        self.maj_filtres()
        self.rafraichir()
        self.maj_stats()

    # ------------------------------------------------------------------
    #  Écran d'accueil
    # ------------------------------------------------------------------
    def creer_ecran_accueil(self):
        f = self.nouvel_ecran("accueil")
        tk.Label(f, text="Mon herbier", font=("Arial", 30, "bold"),
                 bg=self.FOND, fg=self.VERT).pack(pady=(80, 5))
        tk.Label(f, text="Parcourez vos plantes, ajoutez-en de nouvelles ou chargez un fichier.",
                 font=("Arial", 12), bg=self.FOND, fg="#555").pack()
        self.label_stats = tk.Label(f, text="", font=("Arial", 14), bg=self.FOND, fg="#333")
        self.label_stats.pack(pady=30)

        cadre = tk.Frame(f, bg=self.FOND)
        cadre.pack()
        self.bouton(cadre, "Parcourir l'herbier",
                    lambda: self.afficher_ecran("parcourir")).pack(side="left", padx=8)
        self.bouton(cadre, "Ajouter une plante",
                    lambda: self.afficher_ecran("ajouter")).pack(side="left", padx=8)
        self.bouton(cadre, "Charger un JSON", self.action_charger).pack(side="left", padx=8)

    def maj_stats(self):
        n = len(self.herbier.classeur)
        if n == 0:
            self.label_stats.config(text="L'herbier est vide : chargez un fichier JSON ou ajoutez une plante.")
        else:
            nf = len(self.herbier.famille)
            self.label_stats.config(text=f"{n} plante(s) dans {nf} famille(s)")

    # ------------------------------------------------------------------
    #  Écran « Parcourir »
    # ------------------------------------------------------------------
    def creer_ecran_parcourir(self):
        f = self.nouvel_ecran("parcourir")

        # ----- colonne de gauche : recherche, filtres, liste -----
        gauche = tk.Frame(f, bg=self.FOND)
        gauche.pack(side="left", fill="y", padx=(15, 8), pady=15)

        tk.Label(gauche, text="Recherche", bg=self.FOND).pack(anchor="w")
        self.var_recherche = tk.StringVar()
        self.var_recherche.trace_add("write", lambda *a: self.rafraichir())
        ttk.Entry(gauche, textvariable=self.var_recherche).pack(fill="x")

        for cle, titre in (("cycle", "Cycle"), ("besoin", "Besoin"), ("famille", "Famille")):
            tk.Label(gauche, text=titre, bg=self.FOND).pack(anchor="w", pady=(8, 0))
            combo = ttk.Combobox(gauche, state="readonly", values=[self.TOUS])
            combo.set(self.TOUS)
            combo.pack(fill="x")
            combo.bind("<<ComboboxSelected>>", lambda e: self.rafraichir())
            self.combos[cle] = combo
            self.mappings[cle] = {self.TOUS: None}

        ttk.Button(gauche, text="Réinitialiser les filtres",
                   command=self.reinitialiser_filtres).pack(fill="x", pady=8)

        self.label_compte = tk.Label(gauche, text="", bg=self.FOND, fg="#555")
        self.label_compte.pack(anchor="w")

        cadre_liste = tk.Frame(gauche)
        cadre_liste.pack(fill="both", expand=True, pady=(4, 0))
        scroll = ttk.Scrollbar(cadre_liste)
        scroll.pack(side="right", fill="y")
        self.liste = tk.Listbox(cadre_liste, width=30, font=("Arial", 11), activestyle="none",
                                exportselection=False, yscrollcommand=scroll.set)
        self.liste.pack(side="left", fill="both", expand=True)
        scroll.config(command=self.liste.yview)
        self.liste.bind("<<ListboxSelect>>", self.afficher_fiche)

        # ----- colonne de droite : fiche de la plante -----
        droite = tk.Frame(f, bg=self.FOND)
        droite.pack(side="left", fill="both", expand=True, padx=(8, 15), pady=15)

        cadre_photo = tk.Frame(droite, width=380, height=320, bg="white",
                               highlightbackground="#cfc9bb", highlightthickness=1)
        cadre_photo.pack()
        cadre_photo.pack_propagate(False)
        self.label_photo = tk.Label(cadre_photo, text="", bg="white", fg="#888", font=("Arial", 11))
        self.label_photo.pack(expand=True)

        self.label_nom = tk.Label(droite, text="", font=("Arial", 20, "bold"),
                                  bg=self.FOND, fg=self.VERT)
        self.label_nom.pack(pady=(12, 0))
        self.label_sci = tk.Label(droite, text="", font=("Arial", 12, "italic"),
                                  bg=self.FOND, fg="#555")
        self.label_sci.pack()
        self.label_infos = tk.Label(droite, text="", font=("Arial", 12), bg=self.FOND,
                                    justify="left", anchor="w", wraplength=380)
        self.label_infos.pack(pady=10)

        barre = tk.Frame(droite, bg=self.FOND)
        barre.pack(pady=5)
        self.bouton(barre, "< Précédent", lambda: self.aller_a(-1)).pack(side="left", padx=5)
        self.bouton(barre, "Suivant >", lambda: self.aller_a(1)).pack(side="left", padx=5)
        self.bouton(barre, "Supprimer", self.supprimer_selection,
                    couleur=self.ROUGE).pack(side="left", padx=(25, 5))

    def maj_filtres(self):
        """Remplit les listes déroulantes avec les valeurs actuelles de l'herbier."""
        sources = {"cycle": self.herbier.cycle,
                   "besoin": self.herbier.besoins,
                   "famille": self.herbier.famille}
        for cle, valeurs in sources.items():
            mapping = {self.TOUS: None}
            for v in sorted(valeurs):
                mapping[v.replace("_", " ")] = v      # on affiche sans '_', on filtre avec
            self.mappings[cle] = mapping
            combo = self.combos[cle]
            actuel = combo.get()
            combo["values"] = list(mapping)
            combo.set(actuel if actuel in mapping else self.TOUS)

    def valeur_filtre(self, cle):
        return self.mappings[cle].get(self.combos[cle].get())

    def reinitialiser_filtres(self):
        self.var_recherche.set("")
        for combo in self.combos.values():
            combo.set(self.TOUS)
        self.rafraichir()

    def rafraichir(self):
        """Recalcule la liste selon la recherche et les filtres (via Herbier)."""
        resultats = self.herbier.filtrer(self.valeur_filtre("cycle"),
                                         self.valeur_filtre("besoin"),
                                         self.valeur_filtre("famille"))
        texte = self.var_recherche.get().strip()
        if texte:
            trouvees = self.herbier.rechercher(texte)
            resultats = [p for p in resultats if p in trouvees]

        self.plantes_affichees = resultats
        self.liste.delete(0, tk.END)
        for p in resultats:
            self.liste.insert(tk.END, p.nom)
        total = len(self.herbier.classeur)
        self.label_compte.config(text=f"{len(resultats)} plante(s) sur {total}")

        if resultats:
            self.liste.selection_set(0)
            self.afficher_fiche()
        else:
            self.vider_fiche()

    def vider_fiche(self):
        self.label_nom.config(text="")
        self.label_sci.config(text="")
        vide = not self.herbier.classeur
        self.label_infos.config(text="L'herbier est vide.\nUtilisez « Charger un JSON »." if vide
                                else "Aucune plante ne correspond.")
        self.afficher_photo(None)

    def afficher_fiche(self, event=None):
        selection = self.liste.curselection()
        if not selection:
            return
        plante = self.plantes_affichees[selection[0]]
        besoins = ", ".join(b.replace("_", " ") for b in plante.besoins) or "aucun"
        self.label_nom.config(text=plante.nom)
        self.label_sci.config(text=plante.nom_scientifique)
        self.label_infos.config(text=(f"Famille : {plante.famille}\n"
                                      f"Cycle : {plante.cycle}\n"
                                      f"Besoins : {besoins}"))
        self.afficher_photo(plante.photo)

    def afficher_photo(self, nom_fichier):
        """Cherche l'image dans le dossier photos/ à côté du script."""
        self.photo_tk = None
        texte = "Pas de photo"
        if nom_fichier and not PILLOW_OK:
            texte = "Pillow n'est pas installé\n(pip install pillow)"
        elif nom_fichier:
            try:
                img = Image.open(DOSSIER_PHOTOS / nom_fichier)
                img.thumbnail(self.TAILLE_PHOTO)
                self.photo_tk = ImageTk.PhotoImage(img)
            except OSError:
                texte = f"Photo introuvable\n({nom_fichier})"

        if self.photo_tk:
            self.label_photo.config(image=self.photo_tk, text="")
        else:
            self.label_photo.config(image="", text=texte)

    def aller_a(self, decalage):
        """Passe à la plante précédente (-1) ou suivante (+1), en boucle."""
        if not self.plantes_affichees:
            return
        selection = self.liste.curselection()
        courant = selection[0] if selection else 0
        nouveau = (courant + decalage) % len(self.plantes_affichees)
        self.liste.selection_clear(0, tk.END)
        self.liste.selection_set(nouveau)
        self.liste.see(nouveau)
        self.afficher_fiche()

    def supprimer_selection(self):
        selection = self.liste.curselection()
        if not selection:
            messagebox.showinfo("Supprimer", "Sélectionnez d'abord une plante dans la liste.")
            return
        plante = self.plantes_affichees[selection[0]]
        if messagebox.askyesno("Supprimer", f"Supprimer « {plante.nom} » de l'herbier ?"):
            self.herbier.supprimer_plante(plante.nom)
            self.maj_apres_modif()

    # ------------------------------------------------------------------
    #  Écran « Ajouter »
    # ------------------------------------------------------------------
    def creer_ecran_ajouter(self):
        f = self.nouvel_ecran("ajouter")
        tk.Label(f, text="Ajouter une plante", font=("Arial", 20, "bold"),
                 bg=self.FOND, fg=self.VERT).pack(pady=(25, 10))

        form = tk.Frame(f, bg=self.FOND)
        form.pack()

        self.var_nom = tk.StringVar()
        self.var_sci = tk.StringVar()
        self.var_famille = tk.StringVar()
        self.var_cycle = tk.StringVar()
        self.var_autres = tk.StringVar()
        self.var_photo = tk.StringVar()

        champs = [("Nom vernaculaire *", self.var_nom),
                  ("Nom scientifique *", self.var_sci),
                  ("Famille *", self.var_famille)]
        for ligne, (titre, var) in enumerate(champs):
            tk.Label(form, text=titre, bg=self.FOND).grid(row=ligne, column=0, sticky="e", padx=8, pady=5)
            ttk.Entry(form, textvariable=var, width=38).grid(row=ligne, column=1, columnspan=2,
                                                             sticky="w", pady=5)

        tk.Label(form, text="Cycle *", bg=self.FOND).grid(row=3, column=0, sticky="e", padx=8, pady=5)
        ttk.Combobox(form, textvariable=self.var_cycle, values=CYCLES_AUTORISES,
                     state="readonly", width=20).grid(row=3, column=1, sticky="w", pady=5)

        tk.Label(form, text="Besoins", bg=self.FOND).grid(row=4, column=0, sticky="ne", padx=8, pady=5)
        self.cadre_besoins = tk.Frame(form, bg=self.FOND)
        self.cadre_besoins.grid(row=4, column=1, columnspan=2, sticky="w", pady=5)

        tk.Label(form, text="Autres besoins", bg=self.FOND).grid(row=5, column=0, sticky="e", padx=8, pady=5)
        ttk.Entry(form, textvariable=self.var_autres, width=38).grid(row=5, column=1, columnspan=2,
                                                                     sticky="w", pady=5)
        tk.Label(form, text="(séparés par des virgules, ex : support, sol drainant)",
                 bg=self.FOND, fg="#777", font=("Arial", 9)).grid(row=6, column=1, columnspan=2, sticky="w")

        tk.Label(form, text="Photo", bg=self.FOND).grid(row=7, column=0, sticky="e", padx=8, pady=5)
        ttk.Entry(form, textvariable=self.var_photo, width=30).grid(row=7, column=1, sticky="w", pady=5)
        ttk.Button(form, text="Choisir une image...", command=self.choisir_photo).grid(
            row=7, column=2, sticky="w", padx=6)

        barre = tk.Frame(f, bg=self.FOND)
        barre.pack(pady=20)
        self.bouton(barre, "Ajouter à l'herbier", self.valider_formulaire).pack(side="left", padx=6)
        self.bouton(barre, "Effacer", self.vider_formulaire, couleur="#7d7d7d").pack(side="left", padx=6)

    def maj_cases_besoins(self):
        """Reconstruit les cases à cocher d'après les besoins connus de l'herbier."""
        anciens = {b: v.get() for b, v in self.vars_besoins.items()}
        for widget in self.cadre_besoins.winfo_children():
            widget.destroy()
        self.vars_besoins = {}
        for i, besoin in enumerate(sorted(self.herbier.besoins)):
            var = tk.BooleanVar(value=anciens.get(besoin, False))
            self.vars_besoins[besoin] = var
            tk.Checkbutton(self.cadre_besoins, text=besoin.replace("_", " "), variable=var,
                           bg=self.FOND, activebackground=self.FOND, anchor="w").grid(
                row=i // 3, column=i % 3, sticky="w", padx=4)

    def choisir_photo(self):
        dossier = DOSSIER_PHOTOS
        chemin = filedialog.askopenfilename(
            title="Choisir une photo",
            initialdir=dossier if dossier.is_dir() else dossier_courant,
            filetypes=[("Images", "*.jpg *.jpeg *.png *.gif *.webp *.bmp"), ("Tous", "*.*")])
        if chemin:
            self.var_photo.set(chemin)

    def copier_photo(self, chemin):
        """Copie l'image choisie dans photos/ et retourne le nom du fichier (ou None)."""
        if not chemin:
            return None
        source = Path(chemin)
        if source.is_file():
            dossier = DOSSIER_PHOTOS
            cible = dossier / source.name
            try:
                dossier.mkdir(exist_ok=True)
                if source.resolve() != cible.resolve():
                    shutil.copy2(source, cible)
            except OSError:
                pass
        return source.name

    def vider_formulaire(self):
        for var in (self.var_nom, self.var_sci, self.var_famille, self.var_cycle,
                    self.var_autres, self.var_photo):
            var.set("")
        for var in self.vars_besoins.values():
            var.set(False)

    def valider_formulaire(self):
        besoins = [b for b, var in self.vars_besoins.items() if var.get()]
        besoins += self.var_autres.get().replace("/", ",").split(",")
        donnees = {"nom": self.var_nom.get(),
                   "nom_scientifique": self.var_sci.get(),
                   "famille": self.var_famille.get(),
                   "cycle": self.var_cycle.get(),
                   "besoins": besoins,
                   "photo": self.copier_photo(self.var_photo.get().strip())}

        ajoutee, log = self.capturer(self.herbier.integrer, donnees)
        if ajoutee:
            messagebox.showinfo("Herbier", "Plante ajoutée !")
            self.vider_formulaire()
            self.maj_apres_modif()
            self.maj_cases_besoins()
        else:
            messagebox.showwarning("Plante non ajoutée", log or "La plante n'a pas pu être ajoutée.")

    # ------------------------------------------------------------------
    #  Charger / sauvegarder
    # ------------------------------------------------------------------
    def action_charger(self):
        chemin = filedialog.askopenfilename(title="Charger un fichier JSON",
                                            initialdir=dossier_courant,
                                            filetypes=[("JSON", "*.json"), ("Tous", "*.*")])
        if not chemin:
            return
        n, log = self.capturer(self.herbier.charger_json, chemin)
        if n is None:
            messagebox.showerror("Erreur", log or "Impossible de lire ce fichier.")
            return
        self.maj_apres_modif()
        self.fenetre_resume("Chargement", f"{n} plante(s) ajoutée(s)", log)

    def action_sauvegarder(self):
        chemin = filedialog.asksaveasfilename(title="Sauvegarder l'herbier",
                                              initialdir=dossier_courant,
                                              initialfile="mon_herbier.json",
                                              defaultextension=".json",
                                              filetypes=[("JSON", "*.json")])
        if not chemin:
            return
        ok, log = self.capturer(self.herbier.sauvegarder_json, chemin)
        if ok:
            messagebox.showinfo("Sauvegarde", "Herbier sauvegardé.")
        else:
            messagebox.showerror("Erreur", log or "La sauvegarde a échoué.")


# =====================================================================
#  Tests rapides (décommente pour essayer)
# =====================================================================

# h = Herbier()
# print(h.nettoyer_plante({"nom": " tomate ", "nom_scientifique": "SOLANUM  lycopersicum",
#                          "famille": "Solanacées", "cycle": "Annuelle",
#                          "besoins": ["Soleil", "arrosage régulier"], "photo": "tomate.jpg"}))
# h.nettoyer_plante({"nom": "", "nom_scientifique": "Ocimum basilicum",
#                    "famille": "Lamiaceae", "cycle": "annuel", "besoins": ["soleil"]})
#
# a = Plante("Tomate", "Solanum lycopersicum", "Solanaceae", "annuel", ["soleil"])
# b = Plante("Tomate cerise", "Solanum lycopersicum", "Solanaceae", "annuel", ["soleil"])
# for original, doublon in detecter_doublons([a, b]):
#     print(f"{doublon.nom} est un doublon de {original.nom}")
#
# # Test de la sérialisation
# h.charger_json("plantes_degradees.json")
# h.sauvegarder_json("mon_herbier.json")
# h2 = Herbier()
# h2.charger_json("mon_herbier.json")
# print([vars(p) for p in h.classeur] == [vars(p) for p in h2.classeur])   # True


if __name__ == "__main__":
    if "--terminal" in sys.argv:
        Interagir().menu()          # version console : python Herbier_code_v5.py --terminal
    else:
        AppHerbier().mainloop()     # version graphique (par défaut)
