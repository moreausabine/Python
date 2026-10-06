import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import randint

# -----------------------------------------------------------------------------
#                               Données d'entrées
# -----------------------------------------------------------------------------

temp_max = 300

nb_carni = 10
nb_herbi = 20
nb_plante = 50

taille_monde = 10

# -----------------------------------------------------------------------------
#                               Classes
# -----------------------------------------------------------------------------

class Animal:
    def __init__(self,x=0,y=0, energie, esp_vie, age=0):
        self.x = x
        self.y = y
        self.age = age
        self.esp_vie = esp_vie
        self.energie = energie

    def bouge(self, fatigue):
        self.x = self.x + randint(-1,1)
        self.y = self.y + randint(-1,1)
        self.energie = self.energie - fatigue

    def repro(self, fatigue):
        # condition sur l'énergie
            # ajout d'un bonhomme à la liste correspondante
        self.energie = self.energie - fatigue

    def manger(self, bouffe):
        self.energie += bouffe


class Plante:
    def __init__(self,x=0,y=0, energie, esp_vie, age=0):
        self.x = x
        self.y = y
        self.age = age
        self.esp_vie = esp_vie
        self.energie = energie

    def repro(self)

Fourmillière = []
for k in range(nb_fourmi) :
    Fourmillière.append(Fourmis())

Deplacement = []
for i in range(temp):
    Position_inter = []
    for fourmi in Fourmillière :
        Position_inter.append([fourmi.x,fourmi.y])
        fourmi.bouge()
    Deplacement.append(Position_inter)