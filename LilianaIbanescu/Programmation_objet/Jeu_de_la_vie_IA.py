import random
import tkinter as tk


class Grille:
    def __init__(self, largeur, hauteur):
        self.largeur = largeur
        self.hauteur = hauteur

        self.cellules = [
            [False for _ in range(largeur)]
            for _ in range(hauteur)
        ]

    def initialiser_aleatoirement(self, probabilite=0.3):
        """Remplit la grille aléatoirement."""
        for y in range(self.hauteur):
            for x in range(self.largeur):
                self.cellules[y][x] = random.random( ) < probabilite

    def compter_voisins(self, x, y):
        """Compte les cellules vivantes autour d'une cellule."""
        nombre = 0

        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):

                # Ne pas compter la cellule elle-même
                if dx == 0 and dy == 0:
                    continue

                nx = x + dx
                ny = y + dy

                # Vérifier que la cellule est dans la grille
                if 0 <= nx < self.largeur and 0 <= ny < self.hauteur:
                    if self.cellules[ny][nx]:
                        nombre += 1

        return nombre

    def prochaine_generation(self):
        """Crée la génération suivante."""
        nouvelle_grille = Grille(self.largeur, self.hauteur)

        for y in range(self.hauteur):
            for x in range(self.largeur):

                voisins = self.compter_voisins(x, y)
                vivante = self.cellules[y][x]

                if vivante:
                    # Une cellule vivante survit avec 2 ou 3 voisins
                    if voisins == 2 or voisins == 3:
                        nouvelle_grille.cellules[y][x] = True
                else:
                    # Une cellule morte naît avec exactement 3 voisins
                    if voisins == 3:
                        nouvelle_grille.cellules[y][x] = True

        return nouvelle_grille


class JeuDeLaVie:
    def __init__(self, largeur=40, hauteur=25, taille_cellule=20):
        self.largeur = largeur
        self.hauteur = hauteur
        self.taille_cellule = taille_cellule

        self.grille = Grille(largeur, hauteur)

        self.generation = 0
        self.en_cours = False

        # Création de la fenêtre
        self.fenetre = tk.Tk()
        self.fenetre.title("Jeu de la Vie")

        # Canvas = zone dans laquelle on dessine la grille
        self.canvas = tk.Canvas(
            self.fenetre,
            width=largeur * taille_cellule,
            height=hauteur * taille_cellule,
            bg="white"
        )
        self.canvas.pack()

        # Boutons
        cadre_boutons = tk.Frame(self.fenetre)
        cadre_boutons.pack(pady=10)

        self.bouton_demarrer = tk.Button(
            cadre_boutons,
            text="Démarrer",
            command=self.demarrer
        )
        self.bouton_demarrer.pack(side=tk.LEFT, padx=5)

        self.bouton_pause = tk.Button(
            cadre_boutons,
            text="Pause",
            command=self.pause
        )
        self.bouton_pause.pack(side=tk.LEFT, padx=5)

        self.bouton_etape = tk.Button(
            cadre_boutons,
            text="Étape",
            command=self.etape
        )
        self.bouton_etape.pack(side=tk.LEFT, padx=5)

        self.bouton_aleatoire = tk.Button(
            cadre_boutons,
            text="Aléatoire",
            command=self.aleatoire
        )
        self.bouton_aleatoire.pack(side=tk.LEFT, padx=5)

        self.bouton_effacer = tk.Button(
            cadre_boutons,
            text="Effacer",
            command=self.effacer
        )
        self.bouton_effacer.pack(side=tk.LEFT, padx=5)

        # Texte indiquant la génération
        self.label_generation = tk.Label(
            self.fenetre,
            text="Génération : 0"
        )
        self.label_generation.pack()

        # Permet de cliquer sur les cellules
        self.canvas.bind("<Button-1>", self.cliquer)

        # Dessiner la grille initiale
        self.afficher()

    def afficher(self):
        """Affiche la grille dans le Canvas."""
        self.canvas.delete("all")

        for y in range(self.hauteur):
            for x in range(self.largeur):

                x1 = x * self.taille_cellule
                y1 = y * self.taille_cellule

                x2 = x1 + self.taille_cellule
                y2 = y1 + self.taille_cellule

                # Couleur de la cellule
                if self.grille.cellules[y][x]:
                    couleur = "#222222"
                else:
                    couleur = "white"

                self.canvas.create_rectangle(
                    x1,
                    y1,
                    x2,
                    y2,
                    fill=couleur,
                    outline="#cccccc"
                )

        self.label_generation.config(
            text=f"Génération : {self.generation}"
        )

    def cliquer(self, evenement):
        """Active ou désactive une cellule avec la souris."""

        x = evenement.x // self.taille_cellule
        y = evenement.y // self.taille_cellule

        if 0 <= x < self.largeur and 0 <= y < self.hauteur:

            self.grille.cellules[y][x] = not self.grille.cellules[y][x]

            self.afficher()

    def etape(self):
        """Passe à la génération suivante."""

        self.grille = self.grille.prochaine_generation()
        self.generation += 1

        self.afficher()

    def demarrer(self):
        """Démarre la simulation."""

        if not self.en_cours:
            self.en_cours = True
            self.animer()

    def pause(self):
        """Met la simulation en pause."""

        self.en_cours = False

    def animer(self):
        """Anime automatiquement le jeu."""

        if self.en_cours:
            self.etape()

            # Recommencer dans 100 ms
            self.fenetre.after(100, self.animer)

    def aleatoire(self):
        """Génère une nouvelle grille aléatoire."""

        self.pause()

        self.grille.initialiser_aleatoirement()

        self.generation = 0

        self.afficher()

    def effacer(self):
        """Efface toutes les cellules."""

        self.pause()

        self.grille = Grille(
            self.largeur,
            self.hauteur
        )

        self.generation = 0

        self.afficher()

    def lancer(self):
        """Lance la fenêtre."""

        self.fenetre.mainloop()


# --------------------------------------------------
# Programme principal
# --------------------------------------------------

if __name__ == "__main__":

    jeu = JeuDeLaVie(
        largeur=40,
        hauteur=25,
        taille_cellule=20
    )

    jeu.lancer()