import matplotlib.pyplot as plt
import numpy as np


# 1 définir la classe terrain
class Terrain():
    def __init__(self,dim):
        self.dim = dim
        grille = []
        for x in range(self.dim):
            grille.append([])
            for j in range(self.dim):
                grille[x].append(None)

        self.grille = grille


    def occupation(self):
        for x in range(self.dim):
            for y in range(self.dim):
                self.grille[x][y] = Cell(x,y,self)





# 2 définir la classe cellule

class Cell:
    def __init__(self, x, y, terrain, etat=False, voisine = 0,):
        self.x = x
        self.y = y
        self.etat = etat
        self.voisine = voisine
        self.terrain = terrain

    def nb_voisines_vivantes(self):
        pos_x = self.x
        pos_y = self.y
        L = [-1, 0, 1]
        voisine = 0
        for dx in L:
            for dy in L:
                if (pos_x + dx >= 0 and pos_x + dx < self.terrain.dim) and (pos_y + dy >= 0 and pos_y + dy < self.terrain.dim) :
                    if (dx != 0 or dy != 0) and self.terrain.grille[pos_x+dx][pos_y+dy].etat :
                            voisine += 1

        self.voisine = voisine

    def changement_etat(self):
        if self.voisine <=1 and self.etat :
            self.etat = False
        elif self.voisine == 3 and not self.etat :
            self.etat = True
        elif self.voisine > 3 and self.etat :
            self.etat = False


def affichage(terrain):
    grille_affichage = []

    for x in range(terrain.dim):
        ligne = []
        for y in range(terrain.dim):
            if terrain.grille[x][y].etat:
                ligne.append(0)
            else:
                ligne.append(1)
        grille_affichage.append(ligne)

    plt.cla()
    plt.imshow(grille_affichage, cmap="gray_r", interpolation="nearest")
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.pause(0.5)



# 3 faire l'initialisation
field = Terrain(100)
field.occupation()

#ajouter des cellules True
nb_cells = 50
X = np.random.choice(field.dim, nb_cells, replace=False)
Y = np.random.choice(field.dim, nb_cells, replace=False)
for x in X:
    for y in Y :
        field.grille[x][y].etat = True


# Les itératives
tps = 0
while tps != 100 :
    for x in range(field.dim):
        for y in range(field.dim):
            field.grille[x][y].nb_voisines_vivantes()

    for x in range(field.dim):
        for y in range(field.dim):
            field.grille[x][y].changement_etat()


    # l'affichage

    affichage(field)

    tps += 1

plt.show()