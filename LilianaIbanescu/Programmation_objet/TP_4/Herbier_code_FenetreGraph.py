import contextlib
import io
import json
import random as rd
import shutil
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
        self.nom_plante = []

    # ---------- gérer les données de l'herbier ----------
    def mise_jour_besoins(self, besoins):
        if isinstance(besoins, str):
            besoins = besoins.split("/")

        for besoin in besoins:
            besoin = besoin.strip().lower()
            if besoin and besoin not in self.besoins:
                self.besoins.append(besoin)

    def mise_jour_famille(self, famille):
        if famille and famille not in self.famille:
            self.famille.append(famille)

    def mise_jour_nom_plante(self, nom_plante):
        if nom_plante and nom_plante not in self.nom_plante:
            self.nom_plante.append(nom_plante)

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
        self.mise_jour_nom_plante(plante.nom)
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
        if isinstance(besoins, str):
            besoins = besoins.split("/")
        besoins = [self.nettoyer_besoin(b) for b in (besoins or [])]
        besoins = list(dict.fromkeys(b for b in besoins if b)) 

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

        erreurs = valider_plante(propre)
        if erreurs:
            afficher_rapport(propre, erreurs)
        return propre

    def integrer(self, donnees):
        """Nettoie, valide puis ajoute une plante. Retourne True si elle est ajoutée."""
        propre = self.nettoyer_plante(donnees)
        if valider_plante(propre):
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

    def famille_de(self, nom):
        """Retourne la famille d'une plante (nom vernaculaire ou scientifique), ou None si inconnue."""
        nom = self.nettoyer_nom(nom)
        if not nom:
            return None
        for plante in self.classeur:
            if nom in (plante.nom, plante.nom_scientifique):
                return plante.famille
        return None



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
    BONNE = "#3d8b4f"
    GRIS = "#9a9a9a"
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
        self.quiz_en_cours = None       # objet Quiz de la partie en cours
        self.question_courante = None   # (intitulé, choix, bonne réponse)
        self.reponse_donnee = False
        self.boutons_choix = []

        ttk.Style(self).theme_use("clam")
        self.creer_menu()
        self.zone = tk.Frame(self, bg=self.FOND)
        self.zone.pack(side="left", fill="both", expand=True)
        self.creer_ecran_accueil()
        self.creer_ecran_parcourir()
        self.creer_ecran_ajouter()
        self.creer_ecran_supprimer()
        self.creer_ecran_quiz()
        self.afficher_ecran("accueil")


    #  ------------------------------ Outils ------------------------------
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

    # ------------------------------ Menu latéral et navigation entre écrans ------------------------------
    def creer_menu(self):
        barre = tk.Frame(self, bg=self.VERT, width=210)
        barre.pack(side="left", fill="y")
        barre.pack_propagate(False)
        tk.Label(barre, text="HERBIER", font=("Arial", 18, "bold"),
                 bg=self.VERT, fg="white").pack(pady=(25, 25))

        ecrans = [("accueil", "Accueil"),
                  ("parcourir", "Parcourir l'herbier"),
                  ("ajouter", "Ajouter une plante"),
                  ("supprimer", "Supprimer une plante"),
                  ("quiz", "Quiz")]
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
        elif nom == "supprimer":
            self.maj_liste_suppression()
        elif nom == "quiz":
            self.maj_quiz_config()

    def resynchroniser_listes(self):
        """Herbier.supprimer_plante() ne met pas à jour les listes 'famille' et 'nom_plante'.
        On enlève ici ce qui n'existe plus, sinon les filtres et le quiz proposeraient
        des plantes ou des familles supprimées."""
        familles = {p.famille for p in self.herbier.classeur}
        noms = {p.nom for p in self.herbier.classeur}
        self.herbier.famille = [f for f in self.herbier.famille if f in familles]
        self.herbier.nom_plante = [n for n in self.herbier.nom_plante if n in noms]

    def maj_apres_modif(self):
        """À appeler quand le contenu de l'herbier a changé."""
        self.resynchroniser_listes()
        self.maj_filtres()
        self.rafraichir()
        self.maj_stats()

  
    #  -------------------------------------- Écran d'accueil --------------------------------------

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
        self.bouton(cadre, "Lancer un quiz",
                    lambda: self.afficher_ecran("quiz")).pack(side="left", padx=8)
        self.bouton(cadre, "Charger un JSON", self.action_charger).pack(side="left", padx=8)

    def maj_stats(self):
        n = len(self.herbier.classeur)
        if n == 0:
            self.label_stats.config(text="L'herbier est vide : chargez un fichier JSON ou ajoutez une plante.")
        else:
            nf = len(self.herbier.famille)
            self.label_stats.config(text=f"{n} plante(s) dans {nf} famille(s)")


    #  -------------------------------------- Écran « Parcourir » --------------------------------------
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


    #  --------------------- Écran « Ajouter » ---------------------
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

    #  ------------------------ Écran « Supprimer » ------------------------
    def creer_ecran_supprimer(self):
        f = self.nouvel_ecran("supprimer")
        self.plantes_suppr = []
        tk.Label(f, text="Supprimer une plante", font=("Arial", 20, "bold"),
                 bg=self.FOND, fg=self.VERT).pack(pady=(25, 5))
        tk.Label(f, text="Sélectionnez une ou plusieurs plantes (Ctrl ou Maj + clic), "
                         "puis cliquez sur Supprimer.",
                 bg=self.FOND, fg="#555").pack()

        cadre = tk.Frame(f, bg=self.FOND)
        cadre.pack(pady=12)
        tk.Label(cadre, text="Recherche", bg=self.FOND).pack(anchor="w")
        self.var_recherche_suppr = tk.StringVar()
        self.var_recherche_suppr.trace_add("write", lambda *a: self.maj_liste_suppression())
        ttk.Entry(cadre, textvariable=self.var_recherche_suppr, width=45).pack(fill="x", pady=(0, 8))

        cadre_liste = tk.Frame(cadre)
        cadre_liste.pack()
        scroll = ttk.Scrollbar(cadre_liste)
        scroll.pack(side="right", fill="y")
        self.liste_suppr = tk.Listbox(cadre_liste, width=50, height=14, font=("Arial", 11),
                                      selectmode="extended", activestyle="none",
                                      exportselection=False, yscrollcommand=scroll.set)
        self.liste_suppr.pack(side="left")
        scroll.config(command=self.liste_suppr.yview)

        self.bouton(f, "Supprimer la sélection", self.supprimer_plantes_selectionnees,
                    couleur=self.ROUGE).pack(pady=10)

    def maj_liste_suppression(self):
        texte = self.var_recherche_suppr.get().strip()
        self.plantes_suppr = self.herbier.rechercher(texte) if texte else list(self.herbier.classeur)
        self.liste_suppr.delete(0, tk.END)
        for p in self.plantes_suppr:
            self.liste_suppr.insert(tk.END, f"{p.nom}  ({p.nom_scientifique})")

    def supprimer_plantes_selectionnees(self):
        indices = self.liste_suppr.curselection()
        if not indices:
            messagebox.showinfo("Supprimer", "Sélectionnez d'abord au moins une plante.")
            return
        plantes = [self.plantes_suppr[i] for i in indices]
        noms = ", ".join(p.nom for p in plantes)
        if not messagebox.askyesno("Supprimer", f"Supprimer de l'herbier : {noms} ?"):
            return
        for p in plantes:
            self.herbier.supprimer_plante(p.nom)
        self.maj_apres_modif()
        self.maj_liste_suppression()



    # ----------------------------------- Écran « Quiz »  (3 étapes : configuration -> questions -> résultat) -----------------------------------
    def creer_ecran_quiz(self):
        f = self.nouvel_ecran("quiz")
        self.etapes_quiz = {nom: tk.Frame(f, bg=self.FOND) for nom in ("config", "question", "resultat")}
        self.creer_quiz_config(self.etapes_quiz["config"])
        self.creer_quiz_question(self.etapes_quiz["question"])
        self.creer_quiz_resultat(self.etapes_quiz["resultat"])

    def afficher_etape_quiz(self, nom):
        for etape in self.etapes_quiz.values():
            etape.pack_forget()
        self.etapes_quiz[nom].pack(fill="both", expand=True)

    # ----- étape 1 : configuration -----
    def creer_quiz_config(self, c):
        tk.Label(c, text="Quiz", font=("Arial", 20, "bold"),
                 bg=self.FOND, fg=self.VERT).pack(pady=(25, 5))
        tk.Label(c, text="Testez vos connaissances sur les familles de plantes de votre herbier.",
                 bg=self.FOND, fg="#555").pack()

        form = tk.Frame(c, bg=self.FOND)
        form.pack(pady=25)

        tk.Label(form, text="Nombre de questions", bg=self.FOND).grid(row=0, column=0, sticky="e",
                                                                       padx=8, pady=8)
        self.var_nb_questions = tk.StringVar(value="5")
        self.spin_nb_questions = ttk.Spinbox(form, from_=1, to=50, width=6,
                                             textvariable=self.var_nb_questions)
        self.spin_nb_questions.grid(row=0, column=1, sticky="w")

        self.var_mode_quiz = tk.StringVar(value="classique")
        for ligne, (texte, valeur) in enumerate([("Quiz classique (toutes les familles)", "classique"),
                                                 ("Révision d'une famille", "revision")], start=1):
            tk.Radiobutton(form, text=texte, value=valeur, variable=self.var_mode_quiz,
                           command=self.maj_mode_quiz, bg=self.FOND, activebackground=self.FOND,
                           anchor="w").grid(row=ligne, column=0, columnspan=2, sticky="w", padx=8, pady=2)

        self.combo_quiz_famille = ttk.Combobox(form, state="disabled", width=24)
        self.combo_quiz_famille.grid(row=3, column=0, columnspan=2, sticky="w", padx=34, pady=(0, 8))
        self.combo_quiz_famille.bind("<<ComboboxSelected>>", lambda e: self.maj_max_questions())

        self.label_quiz_info = tk.Label(c, text="", bg=self.FOND, fg=self.ROUGE, wraplength=520)
        self.label_quiz_info.pack()
        self.btn_lancer_quiz = self.bouton(c, "Lancer le quiz", self.lancer_quiz)
        self.btn_lancer_quiz.pack(pady=15)

    def max_questions(self):
        """Nombre maximal de questions pour le mode choisi (en révision : une par plante de la famille)."""
        if self.var_mode_quiz.get() == "revision":
            famille = self.combo_quiz_famille.get()
            maxi = len(self.herbier.filtrer(None, None, famille)) if famille else 1
        else:
            maxi = len(self.herbier.classeur)
        return max(1, maxi)

    def maj_max_questions(self):
        self.spin_nb_questions.config(to=self.max_questions())

    def maj_mode_quiz(self):
        revision = self.var_mode_quiz.get() == "revision"
        self.combo_quiz_famille.config(state="readonly" if revision else "disabled")
        self.maj_max_questions()

    def maj_quiz_config(self):
        """Remet l'écran de configuration à jour (familles disponibles, herbier assez rempli ?)."""
        familles = sorted(self.herbier.famille)
        self.combo_quiz_famille["values"] = familles
        if self.combo_quiz_famille.get() not in familles:
            self.combo_quiz_famille.set(familles[0] if familles else "")
        self.maj_mode_quiz()

        if len(self.herbier.classeur) < 4 or len(familles) < 2:
            self.label_quiz_info.config(text="Il faut au moins 4 plantes réparties dans au moins "
                                             "2 familles pour lancer un quiz.\n"
                                             "Chargez un fichier JSON ou ajoutez des plantes.")
            self.btn_lancer_quiz.config(state="disabled")
        else:
            self.label_quiz_info.config(text="")
            self.btn_lancer_quiz.config(state="normal")
        self.afficher_etape_quiz("config")

    def lancer_quiz(self):
        try:
            nb = int(self.var_nb_questions.get())
        except ValueError:
            nb = 0
        if nb < 1:
            messagebox.showwarning("Quiz", "Entrez un nombre de questions positif.")
            return
        nb = min(nb, self.max_questions())

        revision = self.var_mode_quiz.get() == "revision"
        famille = self.combo_quiz_famille.get() if revision else None
        if revision and not famille:
            messagebox.showwarning("Quiz", "Choisissez la famille à réviser.")
            return

        self.quiz_en_cours = Quiz(self.herbier, nb_question=nb, revision=revision)
        self.quiz_en_cours.demarrer(famille)
        self.poser_question_quiz()

    # ----- étape 2 : les questions -----
    def creer_quiz_question(self, q):
        self.bouton(q, "Abandonner", self.maj_quiz_config, couleur="#7d7d7d").pack(side="bottom", pady=15)

        self.label_quiz_titre = tk.Label(q, text="", font=("Arial", 12), bg=self.FOND, fg="#555")
        self.label_quiz_titre.pack(pady=(30, 2))
        self.label_quiz_score = tk.Label(q, text="", font=("Arial", 11), bg=self.FOND, fg="#555")
        self.label_quiz_score.pack()
        self.label_quiz_intitule = tk.Label(q, text="", font=("Arial", 18, "bold"), bg=self.FOND,
                                            fg=self.VERT, wraplength=600, justify="center")
        self.label_quiz_intitule.pack(pady=(20, 18))

        self.cadre_choix = tk.Frame(q, bg=self.FOND)
        self.cadre_choix.pack()
        self.label_quiz_verdict = tk.Label(q, text="", font=("Arial", 13, "bold"), bg=self.FOND,
                                           wraplength=600)
        self.label_quiz_verdict.pack(pady=14)
        self.btn_quiz_suivant = self.bouton(q, "Question suivante >", self.poser_question_quiz)

    def poser_question_quiz(self):
        quiz = self.quiz_en_cours
        question = quiz.question_suivante()
        if question is None:                       # plus de question : on affiche le résultat
            self.afficher_resultat_quiz()
            return

        intitule, choix, _ = question
        self.question_courante = question
        self.reponse_donnee = False

        titre = f"Question {quiz.question_posee + 1} / {quiz.nb_question_max}"
        if quiz.revision:
            titre += f"  -  famille des {quiz.famille_revisee}"
        self.label_quiz_titre.config(text=titre)
        self.label_quiz_score.config(text=f"Score actuel : {quiz.note}/{quiz.question_posee}")
        self.label_quiz_intitule.config(text=intitule)
        self.label_quiz_verdict.config(text="")
        self.btn_quiz_suivant.pack_forget()

        for widget in self.cadre_choix.winfo_children():
            widget.destroy()
        self.boutons_choix = []
        for i, texte in enumerate(choix):
            b = self.bouton(self.cadre_choix, texte, lambda i=i: self.repondre_quiz(i), taille=13)
            b.config(width=34)
            b.pack(pady=5)
            self.boutons_choix.append(b)
        self.afficher_etape_quiz("question")

    def repondre_quiz(self, indice):
        if self.reponse_donnee:                    # une seule réponse par question
            return
        self.reponse_donnee = True

        quiz = self.quiz_en_cours
        _, choix, reponse_juste = self.question_courante
        est_juste, message = quiz.repondre(choix[indice], reponse_juste)

        for k, b in enumerate(self.boutons_choix):
            if choix[k] == reponse_juste:
                b.config(bg=self.BONNE)
            elif k == indice:
                b.config(bg=self.ROUGE)
            else:
                b.config(bg=self.GRIS)
        self.label_quiz_verdict.config(text=message, fg=self.BONNE if est_juste else self.ROUGE)
        self.label_quiz_score.config(text=f"Score actuel : {quiz.note}/{quiz.question_posee}")

        fini = quiz.question_posee >= quiz.nb_question_max
        self.btn_quiz_suivant.config(text="Voir le résultat >" if fini else "Question suivante >")
        self.btn_quiz_suivant.pack(pady=5)

    # ----- étape 3 : le résultat -----
    def creer_quiz_resultat(self, r):
        tk.Label(r, text="Résultat", font=("Arial", 20, "bold"),
                 bg=self.FOND, fg=self.VERT).pack(pady=(60, 15))
        self.label_res_score = tk.Label(r, text="", font=("Arial", 26, "bold"), bg=self.FOND)
        self.label_res_score.pack()
        self.label_res_pct = tk.Label(r, text="", font=("Arial", 16), bg=self.FOND, fg="#555")
        self.label_res_pct.pack(pady=4)
        self.label_res_msg = tk.Label(r, text="", font=("Arial", 16, "italic"),
                                      bg=self.FOND, fg=self.VERT)
        self.label_res_msg.pack(pady=10)
        self.label_res_note = tk.Label(r, text="", bg=self.FOND, fg="#777", wraplength=520)
        self.label_res_note.pack()

        barre = tk.Frame(r, bg=self.FOND)
        barre.pack(pady=25)
        self.bouton(barre, "Rejouer", self.maj_quiz_config).pack(side="left", padx=6)
        self.bouton(barre, "Accueil", lambda: self.afficher_ecran("accueil"),
                    couleur="#7d7d7d").pack(side="left", padx=6)

    def afficher_resultat_quiz(self):
        quiz = self.quiz_en_cours
        bilan = quiz.bilan()
        if bilan is None:
            messagebox.showinfo("Quiz", "Aucune question n'a pu être posée.")
            self.maj_quiz_config()
            return
        pourcentage, appreciation = bilan
        self.label_res_score.config(text=f"Score : {quiz.note}/{quiz.question_posee}")
        self.label_res_pct.config(text=f"{pourcentage} %")
        self.label_res_msg.config(text=appreciation)
        if quiz.question_posee < quiz.nb_question_max:
            self.label_res_note.config(text=f"Le quiz s'est arrêté après {quiz.question_posee} "
                                            f"question(s) : l'herbier n'en permettait pas plus.")
        else:
            self.label_res_note.config(text="")
        self.afficher_etape_quiz("resultat")


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





if __name__ == "__main__":
    AppHerbier().mainloop()
