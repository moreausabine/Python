# Simulation écosystème proie prédateur
import math
import random
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


# =====================================================================
#  PARAMÈTRES : tout ce qu'il faut modifier pour équilibrer se trouve ici
# =====================================================================
class Config:

    class GRILLE:
        LARGEUR = 60
        HAUTEUR = 60

    class DEPART:
        NB_LAPINS = 60
        NB_LOUPS = 8
        NB_CAROTTES = 100

    class SIMU:
        SEED = None              # ex: 42 pour une simulation reproductible
        MAX_TOURS = None         # None = pas de limite
        PAUSE = 0.05              # pause entre deux affichages (secondes)
        AFFICHER_TOUS_LES = 1    # 1 = chaque tour, 5 = un tour sur 5...

    class LAPIN:
        ENERGIE_INITIALE = 100
        ENERGIE_MAX = 100
        PERTE_ENERGIE_PAR_TOUR = 5
        VITESSE = 1
        VISION = 15
        SEUIL_FAIM = 20          # en dessous : cherche à manger au lieu de fuir
        GAIN_CAROTTE = 15         # énergie gagnée en mangeant une carotte
        # Reproduction
        AGE_REPRO_MIN = 5
        ENERGIE_REPRO_MIN = 80
        COUT_REPRO = 20
        ENERGIE_NAISSANCE = 50
        PROBA_REPRO = 0.5        # chance de se reproduire quand c'est possible
        AGE_MAX = 60              # mort de vieillesse

    class LOUP:
        ENERGIE_INITIALE = 100
        ENERGIE_MAX = 100
        PERTE_ENERGIE_PAR_TOUR = 3
        VITESSE = 2
        VISION = 10
        GAIN_PROIE = 25           # énergie gagnée en mangeant un lapin
        # Reproduction
        AGE_REPRO_MIN = 5
        ENERGIE_REPRO_MIN = 70
        COUT_REPRO = 30
        ENERGIE_NAISSANCE = 50
        PROBA_REPRO = 0.08
        AGE_MAX = 120

    class CAROTTE:
        ENERGIE_INITIALE = 5
        ENERGIE_MAX = 5
        REGENERATION = 1         # énergie regagnée à chaque tour
        AGE_MAX = 30             # durée de vie maximale
        DENSITE_MAX = 0.15        # part maximale de cases occupées (0 à 1)
        # Dispersion
        PROBA_DISPERSION = 0.1   # chance de se reproduire à chaque tour
        RAYON = 3                # distance max de dispersion
        AGE_REPRO_MIN = 3        # se reproduit si age > AGE_REPRO_MIN
        ENERGIE_REPRO_MIN = 4
        COUT_REPRO = 3

# =====================================================================
#  ÊTRES VIVANTS (base commune aux animaux et aux carottes)
# =====================================================================
class Etre_vivant(ABC):
    CFG = None  # défini dans les sous-classes (Config.LAPIN, Config.LOUP, Config.CAROTTE)

    def __init__(self, x, y, energie=None):
        self.x = x
        self.y = y
        self.age = 0
        self.energie = self.CFG.ENERGIE_INITIALE if energie is None else energie

    def vieillir(self):
        self.age += 1

    def mort(self):
        return self.energie <= 0

    def perte_energie(self, qte):
        self.energie = max(self.energie - qte, 0)

    def gain_energie(self, qte):
        self.energie = min(self.energie + qte, self.CFG.ENERGIE_MAX)

    def peut_se_reproduire(self):
        return (not self.mort()
                and self.age >= self.CFG.AGE_REPRO_MIN
                and self.energie >= self.CFG.ENERGIE_REPRO_MIN)

    @abstractmethod
    def reproduire(self, environnement):
        pass


# =====================================================================
#  ANIMAUX
# =====================================================================
class Animal(Etre_vivant):

    def __init__(self, x, y, energie=None):
        super().__init__(x, y, energie)
        self.vitesse = self.CFG.VITESSE
        self.vision = self.CFG.VISION

    def mort(self):
        return super().mort() or self.age > self.CFG.AGE_MAX

    def vieillir(self):
        super().vieillir()
        self.perte_energie(self.CFG.PERTE_ENERGIE_PAR_TOUR)

    def deplacer(self, dx, dy, environnement):
        """Déplacement exact de (dx, dy) cases, avec rebouclage (monde torique)."""
        self.x = (self.x + dx) % environnement.largeur
        self.y = (self.y + dy) % environnement.hauteur

    def bouger(self, dx, dy, environnement):
        """Déplacement d'une case par unité de vitesse dans la direction (dx, dy)."""
        self.deplacer(dx * self.vitesse, dy * self.vitesse, environnement)

    def ecart(self, cible, environnement):
        """Écart signé (dx, dy) le plus court vers la cible, en tenant compte du rebouclage."""
        dx = (cible.x - self.x) % environnement.largeur
        if dx > environnement.largeur / 2:
            dx -= environnement.largeur

        dy = (cible.y - self.y) % environnement.hauteur
        if dy > environnement.hauteur / 2:
            dy -= environnement.hauteur

        return dx, dy

    def distance_avec(self, autre, environnement):
        dx, dy = self.ecart(autre, environnement)
        return math.sqrt(dx**2 + dy**2)

    def aller_vers(self, cible, environnement):
        """Avance vers la cible par le chemin le plus court, sans la dépasser."""
        dx, dy = self.ecart(cible, environnement)
        dx = max(-self.vitesse, min(self.vitesse, dx))
        dy = max(-self.vitesse, min(self.vitesse, dy))
        self.deplacer(dx, dy, environnement)

    def errer(self, environnement):
        """Déplacement aléatoire, sans mémoire de la direction précédente."""
        self.bouger(random.randint(-1, 1), random.randint(-1, 1), environnement)


# ------------------ Lapin

class Lapin(Animal):
    CFG = Config.LAPIN

    def fuir(self, environnement):
        if len(environnement.predateur) != 0:
            predateur_plus_proche = min(
                environnement.predateur,
                key=lambda p: self.distance_avec(p, environnement)
            )

            if self.distance_avec(predateur_plus_proche, environnement) <= self.vision:
                dx_t, dy_t = self.ecart(predateur_plus_proche, environnement)

                # signe opposé à la direction du prédateur
                dx = -1 if dx_t > 0 else (1 if dx_t < 0 else 0)
                dy = -1 if dy_t > 0 else (1 if dy_t < 0 else 0)

                self.bouger(dx, dy, environnement)
                return

        self.chercher_manger(environnement)

    def manger_sur_place(self, environnement):
        for carotte in environnement.ressource:
            self.manger(carotte)

    def chercher_manger(self, environnement):
        # on ignore les carottes déjà mangées ce tour-ci
        carottes = [c for c in environnement.ressource if not c.mort()]

        if carottes:
            carotte_plus_proche = min(
                carottes,
                key=lambda c: self.distance_avec(c, environnement)
            )

            if self.distance_avec(carotte_plus_proche, environnement) <= self.vision:
                self.aller_vers(carotte_plus_proche, environnement)
                return

        # rien en vue : déplacement aléatoire
        self.errer(environnement)

    def agir(self, environnement):
        self.manger_sur_place(environnement)   # mange ce qu'il y a sous lui, puis bouge

        if self.energie <= self.CFG.SEUIL_FAIM:
            self.chercher_manger(environnement)
        else:
            self.fuir(environnement)

    def reproduire(self, environnement):
        if self.peut_se_reproduire() and random.random() < self.CFG.PROBA_REPRO:
            self.perte_energie(self.CFG.COUT_REPRO)
            environnement.ajouter(Lapin(self.x, self.y, self.CFG.ENERGIE_NAISSANCE))

    def manger(self, carotte):
        if self.mort() or carotte.mort():   # un lapin mort ne mange pas
            return
        if carotte.x == self.x and carotte.y == self.y:
            carotte.perte_energie(carotte.energie)
            self.gain_energie(self.CFG.GAIN_CAROTTE)


# ------------------ Loup

class Loup(Animal):
    CFG = Config.LOUP

    def chasser(self, animal):
        if animal.mort():          # ne pas manger deux fois la même proie
            return
        if animal.x == self.x and animal.y == self.y:
            animal.perte_energie(animal.energie)
            self.gain_energie(self.CFG.GAIN_PROIE)

    def detection_proie(self, environnement):
        proies = [p for p in environnement.proie if not p.mort()]

        if proies:
            proie_plus_proche = min(
                proies,
                key=lambda p: self.distance_avec(p, environnement)
            )

            if self.distance_avec(proie_plus_proche, environnement) <= self.vision:
                self.aller_vers(proie_plus_proche, environnement)
                return

        # aucune proie en vue : errance avec inertie
        self.errer(environnement)

    def reproduire(self, environnement):
        if self.peut_se_reproduire() and random.random() < self.CFG.PROBA_REPRO:
            self.perte_energie(self.CFG.COUT_REPRO)
            environnement.ajouter(Loup(self.x, self.y, self.CFG.ENERGIE_NAISSANCE))


# =====================================================================
#  RESSOURCES
# =====================================================================
class Carotte(Etre_vivant):
    CFG = Config.CAROTTE

    def vieillir(self):
        super().vieillir()
        self.gain_energie(self.CFG.REGENERATION)

    def mort(self):
        # morte si mangée OU trop vieille
        return super().mort() or self.age > self.CFG.AGE_MAX

    def reproduire(self, environnement):
        """Dispersion : une nouvelle carotte apparaît à proximité."""
        if self.peut_se_reproduire() and random.random() < self.CFG.PROBA_DISPERSION:
            r = self.CFG.RAYON
            # le modulo garde la nouvelle carotte dans la grille (monde torique)
            x = (self.x + random.randint(-r, r)) % environnement.largeur
            y = (self.y + random.randint(-r, r)) % environnement.hauteur

            if environnement.case_libre(x, y):
                environnement.ajouter(Carotte(x, y))
                self.perte_energie(self.CFG.COUT_REPRO)


# =====================================================================
#  ENVIRONNEMENT
# =====================================================================
class Environnement:
    def __init__(self, largeur, hauteur):
        self.largeur = largeur
        self.hauteur = hauteur
        self.max_carottes = largeur * hauteur * Config.CAROTTE.DENSITE_MAX
        self.proie = []
        self.predateur = []
        self.ressource = []
        self.cases_carottes = set()   # cases déjà occupées par une carotte

    def case_libre(self, x, y):
        return (x, y) not in self.cases_carottes and len(self.ressource) < self.max_carottes

    def ajouter(self, chose):
        if isinstance(chose, Lapin):
            self.proie.append(chose)
        elif isinstance(chose, Loup):
            self.predateur.append(chose)
        else:
            # une seule carotte par case
            if (chose.x, chose.y) in self.cases_carottes:
                return
            self.ressource.append(chose)
            self.cases_carottes.add((chose.x, chose.y))

    def supprimer_morts(self):
        for lapin in self.proie[:]:
            if lapin.mort():
                self.proie.remove(lapin)

        for loup in self.predateur[:]:
            if loup.mort():
                self.predateur.remove(loup)

        for carotte in self.ressource[:]:
            if carotte.mort():
                self.ressource.remove(carotte)
                self.cases_carottes.discard((carotte.x, carotte.y))  # libère la case

    def simuler_tour(self):
        # Les lapins bougent d'abord
        for lapin in self.proie[:]:
            lapin.agir(self)

        # Puis chaque loup bouge et tente de capturer immédiatement
        for loup in self.predateur[:]:
            loup.detection_proie(self)
            for lapin in self.proie:
                loup.chasser(lapin)

        # Les lapins mangent
        for lapin in self.proie[:]:
            for carotte in self.ressource:
                lapin.manger(carotte)

        # Reproduction
        for lapin in self.proie[:]:
            lapin.reproduire(self)

        for loup in self.predateur[:]:
            loup.reproduire(self)

        for carotte in self.ressource[:]:
            carotte.reproduire(self)

        # Vieillir
        for lapin in self.proie[:]:
            lapin.vieillir()

        for loup in self.predateur[:]:
            loup.vieillir()

        for carotte in self.ressource[:]:
            carotte.vieillir()

        # Supprimer les morts
        self.supprimer_morts()


# =====================================================================
#  SIMULATION
# =====================================================================
class Simulation:
    def __init__(self):
        self.largeur = Config.GRILLE.LARGEUR
        self.hauteur = Config.GRILLE.HAUTEUR
        self.envi = Environnement(self.largeur, self.hauteur)
        self.generation = 0

        self.figure = None
        self.ax_grille = None
        self.ax_courbes = None
        self.ax_carottes = None

        # Historique pour les courbes
        self.historique = {"tours": [], "lapins": [], "loups": [], "carottes": []}

    def enregistrer(self):
        self.historique["tours"].append(self.generation)
        self.historique["lapins"].append(len(self.envi.proie))
        self.historique["loups"].append(len(self.envi.predateur))
        self.historique["carottes"].append(len(self.envi.ressource))

    def afficher(self):
        if self.figure is None:
            plt.ion()
            self.figure, (self.ax_grille, self.ax_courbes) = plt.subplots(
                1, 2, figsize=(13, 6)
            )
            self.ax_carottes = self.ax_courbes.twinx()  # axe secondaire pour les carottes

        self.ax_grille.clear()
        self.ax_courbes.clear()
        self.ax_carottes.clear()

        # ----- Grille -----
        grille = np.ones((self.hauteur, self.largeur, 3))

        for carotte in self.envi.ressource:
            grille[carotte.y % self.hauteur, carotte.x % self.largeur] = [1, 0.64, 0]

        for lapin in self.envi.proie:
            grille[lapin.y % self.hauteur, lapin.x % self.largeur] = [0, 0.8, 0]

        for loup in self.envi.predateur:
            grille[loup.y % self.hauteur, loup.x % self.largeur] = [1, 0, 0]

        self.ax_grille.imshow(grille, origin="lower", interpolation="nearest")
        self.ax_grille.set_xticks([])
        self.ax_grille.set_yticks([])
        self.ax_grille.set_title(f"Tour {self.generation}")

        # ----- Courbes -----
        h = self.historique
        l1, = self.ax_courbes.plot(h["tours"], h["lapins"], color="green", label="Lapins")
        l2, = self.ax_courbes.plot(h["tours"], h["loups"], color="red", label="Loups")
        l3, = self.ax_carottes.plot(
            h["tours"], h["carottes"], color="orange", linestyle="--", label="Carottes"
        )

        self.ax_courbes.set_xlabel("Tour")
        self.ax_courbes.set_ylabel("Animaux")
        self.ax_carottes.set_ylabel("Carottes")
        self.ax_courbes.set_title("Évolution des populations")
        self.ax_courbes.legend(handles=[l1, l2, l3], loc="upper left")

        self.figure.tight_layout()
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        plt.pause(Config.SIMU.PAUSE)

    def statistique(self):
        print("Tour : ", self.generation)
        print("Loup : ", len(self.envi.predateur))
        print("Lapin : ", len(self.envi.proie))
        print("Carotte : ", len(self.envi.ressource))
        print("Somme animaux : ", len(self.envi.predateur) + len(self.envi.proie))

    def placer_population_initiale(self):
        d = Config.DEPART
        for _ in range(d.NB_LAPINS):
            self.envi.ajouter(Lapin(random.randint(0, self.largeur - 1),
                                    random.randint(0, self.hauteur - 1)))

        for _ in range(d.NB_LOUPS):
            self.envi.ajouter(Loup(random.randint(0, self.largeur - 1),
                                   random.randint(0, self.hauteur - 1)))

        for _ in range(d.NB_CAROTTES):
            self.envi.ajouter(Carotte(random.randint(0, self.largeur - 1),
                                      random.randint(0, self.hauteur - 1)))

    def en_cours(self):
        if len(self.envi.proie) == 0 or len(self.envi.predateur) == 0:
            return False
        max_tours = Config.SIMU.MAX_TOURS
        return max_tours is None or self.generation < max_tours

    def start(self):
        if Config.SIMU.SEED is not None:
            random.seed(Config.SIMU.SEED)

        self.placer_population_initiale()
        self.enregistrer()  # état initial (tour 0)

        while self.en_cours():
            self.generation += 1
            self.envi.simuler_tour()
            self.enregistrer()
            self.statistique()
            if self.generation % Config.SIMU.AFFICHER_TOUS_LES == 0:
                self.afficher()

        # Dernier affichage pour voir l'état final
        self.afficher()
        print("Fin de la simulation au tour", self.generation)
        plt.ioff()
        plt.show()  # garde la fenêtre ouverte à la fin


if __name__ == "__main__":
    Simulation().start()