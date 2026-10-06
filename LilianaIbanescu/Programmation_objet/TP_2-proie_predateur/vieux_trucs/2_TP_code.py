# Simulation écosystème proie prédateur
import math
import random
from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import numpy as np


class Animal(ABC):
    def __init__(self,x,y,energie=100,age=0,vitesse=1):
        self.x = x
        self.y = y
        self.energie = energie
        self.age = age
        self.vitesse = vitesse
    
    def vieillir(self):
        self.age += 1
        self.energie += -5

    def mort(self):
        return self.energie <= 0

    def bouger(self,dx,dy,environnement):
        new_x = self.x + dx*self.vitesse
        if new_x <= environnement.largeur-1 and new_x >=0 :
            self.x = new_x
        elif new_x > environnement.largeur-1 :
            self.x = self.x + dx*self.vitesse - environnement.largeur
        elif new_x < 0 : 
            self.x += self.x + dx*self.vitesse + environnement.largeur

        new_y = self.y + dy*self.vitesse
        if new_y >=0 and new_y <= environnement.hauteur-1  :
            self.y = new_y
        elif new_y > environnement.hauteur-1 :
            self.y = self.y + dy*self.vitesse - environnement.hauteur
        elif new_y < 0 : 
            self.y += self.y + dy*self.vitesse + environnement.hauteur

    def perte_energie(self, qte):
        self.energie = max(self.energie - qte, 0)

    def gain_energie(self, qte):
        self.energie = min(self.energie + qte, 100)

    def distance_avec(self, autre_animal):
        return math.sqrt((self.x - autre_animal.x)**2 + (self.y - autre_animal.y)**2)

    def peut_se_reproduire(self):
        return self.age >=5 and self.energie >=60

    @abstractmethod
    def reproduire(self, environnement):
        pass

class Lapin(Animal):
    def __init__(self, x,y,energie=100,age=0,vitesse=1, vision = 15):
        super().__init__(x,y,energie,age,vitesse) 
        self.vision = vision

    def fuir(self, environnement):
        if len(environnement.predateur) != 0:
            #chercher distance predateurs
            predateur_plus_proche = environnement.predateur[0]
            distance_min = self.distance_avec(predateur_plus_proche)

            for predateur in environnement.predateur[1:]:
                dist = self.distance_avec(predateur)
                if dist < distance_min:
                    distance_min = dist
                    predateur_plus_proche = predateur

            if distance_min <= self.vision :
                # déplacer vers proie
                dx_t = predateur_plus_proche.x - self.x
                dy_t = predateur_plus_proche.y - self.y

                if dx_t >0:
                    dx = -1
                elif dx_t == 0 :
                    dx =0
                else :
                    dx = 1

                if dy_t >0:
                    dy = -1
                elif dy_t == 0 :
                    dy =0
                else :
                    dy = 1

                self.bouger(dx,dy, environnement)

            else :
                self.chercher_manger(environnement)

        else :
            self.chercher_manger(environnement)


    def chercher_manger(self, environnement):
        if len(environnement.ressource) != 0:
            #chercher distance predateurs
            carotte_plus_proche = environnement.ressource[0]
            distance_min = self.distance_avec(carotte_plus_proche)

            for carotte in environnement.ressource[1:]:
                dist = self.distance_avec(carotte)
                if dist < distance_min:
                    distance_min = dist
                    carotte_plus_proche = carotte

            if distance_min <= self.vision :
                # déplacer vers proie
                dx_t = carotte_plus_proche.x - self.x
                dy_t = carotte_plus_proche.y - self.y

                if dx_t >0:
                    dx = 1
                elif dx_t == 0 :
                    dx =0
                else :
                    dx = -1

                if dy_t >0:
                    dy = 1
                elif dy_t == 0 :
                    dy =0
                else :
                    dy = -1

                self.bouger(dx,dy, environnement)

            else :
                dx = random.randint(-1,1)
                dy = random.randint(-1,1)
                self.bouger(dx,dy, environnement)

        else :
            dx = random.randint(-1,1)
            dy = random.randint(-1,1)
            self.bouger(dx,dy, environnement)

    def agir(self,environnement):
        if self.energie <= 30:
            self.chercher_manger(environnement)
        else :
            self.fuir(environnement)

    def reproduire(self, environement):
        if self.peut_se_reproduire():
            self.perte_energie(30)
            environement.ajouter(Lapin(self.x, self.y))

    def manger(self, carotte):
        if carotte.x == self.x and carotte.y == self.y :
            carotte.perte_energie(carotte.energie)
            self.gain_energie(5)



class Loup(Animal):
    def __init__(self, x,y,energie=100,age=0,vitesse=2, vision = 10):
        super().__init__(x,y,energie,age,vitesse) 
        self.vision = vision

    def chasser(self, animal):
        if animal.x == self.x and animal.y == self.y :
            animal.perte_energie(animal.energie)
            self.gain_energie(5)

    def detection_proie(self,environnement):
        if len(environnement.proie) != 0:
            #chercher distance proies
            proie_plus_proche = environnement.proie[0]
            distance_min = self.distance_avec(proie_plus_proche)

            for proie in environnement.proie[1:]:
                dist = self.distance_avec(proie)
                if dist < distance_min:
                    distance_min = dist
                    proie_plus_proche = proie

            if distance_min <= self.vision :
                # déplacer vers proie
                dx_t = proie_plus_proche.x - self.x
                dy_t = proie_plus_proche.y - self.y

                if dx_t >0:
                    dx = 1
                elif dx_t == 0 :
                    dx =0
                else :
                    dx = -1

                if dy_t >0:
                    dy = 1
                elif dy_t == 0 :
                    dy =0
                else :
                    dy = -1

                self.bouger(dx,dy,environnement)
        
            else :
                dx = random.randint(-1,1)
                dy = random.randint(-1,1)
                self.bouger(dx,dy,environnement)

        else :
            dx = random.randint(-1,1)
            dy = random.randint(-1,1)
            self.bouger(dx,dy,environnement)

    def reproduire(self, environement):
        if self.peut_se_reproduire():
            self.perte_energie(30)
            environement.ajouter(Loup(self.x, self.y))

class Carotte:
    def __init__(self, x,y, age=0, energie = 5):
        self.x = x
        self.y = y
        self.age = age
        self.energie = energie

    def vieillir(self):
        self.age += 1
        self.gain_energie(1)

    def peut_repro(self):
        return self.age > 2

    def perte_energie(self, qte):
        self.energie = max(self.energie - qte, 0)

    def gain_energie(self, qte):
        self.energie = min(self.energie + qte, 5)

    def mort(self):
        return self.energie <= 0

    def dispersion(self, environnement):
        if self.peut_repro():
            environnement.ajouter(Carotte(self.x+random.randint(-10,10), self.y+random.randint(-10,10)))
            self.perte_energie(4) 

class Environnement:
    def __init__(self, largeur, hauteur):
        self.largeur = largeur
        self.hauteur = hauteur
        self.proie = []
        self.predateur = []
        self.ressource = []

    def ajouter(self,chose):
        if isinstance(chose, Lapin) :
            self.proie.append(chose)
        elif isinstance(chose, Loup) :
            self.predateur.append(chose)
        else :
            self.ressource.append(chose)

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

    def simuler_tour(self):
        # Tt le monde se déplace
        for loup in self.predateur :
            loup.detection_proie(self)

        for lapin in self.proie:
            lapin.agir(self)

        # Gestion énergie
        for loup in self.predateur :
            for lapin in self.proie :
                loup.chasser(lapin)

        for lapin in self.proie :
            for carotte in self.ressource :
                lapin.manger(carotte)

        # Reproduction
        for lapin in self.proie :
            lapin.reproduire(self)

        for loup in self.predateur :
            loup.reproduire(self)

        for carotte in self.ressource :
            carotte.dispersion(self)

        # Vieillir
        for lapin in self.proie :
            lapin.vieillir()

        for loup in self.predateur :
            loup.vieillir()

        for carotte in self.ressource :
            carotte.vieillir()

        #Supprimer les morts
        self.supprimer_morts()


class Simulation:
    def __init__(self,nb_lapin, nb_loup, nb_carotte, largeur=50, hauteur=50):
        self.largeur = largeur
        self.hauteur = hauteur
        self.envi = Environnement(self.largeur, self.hauteur)
        self.nb_lapin = nb_lapin
        self.nb_loup = nb_loup
        self.nb_carotte = nb_carotte
        self.generation = 0
        self.figure = None
        self.axes = None

    def afficher(self):
        if self.figure is None:
            plt.ion()
            self.figure, self.axes = plt.subplots()

        self.axes.clear()

        # Création d'une grille vide (fond blanc en RVB)
        grille = np.ones((self.hauteur, self.largeur, 3))

        # Coloration des proies (hauteur = Y, largeur = X)
        for carotte in self.envi.ressource:
            y = int(carotte.y) % self.hauteur
            x = int(carotte.x) % self.largeur
            grille[y, x] = [1, 0.64, 0]
        
        for lapin in self.envi.proie:
            y = int(lapin.y) % self.hauteur
            x = int(lapin.x) % self.largeur
            grille[y, x] = [0, 0.8, 0]

        # Coloration des prédateurs
        for loup in self.envi.predateur:
            y = int(loup.y) % self.hauteur
            x = int(loup.x) % self.largeur
            grille[y, x] = [1, 0, 0]

        # Affichage de la matrice avec origine en bas à gauche
        self.axes.imshow(grille, origin="lower", interpolation="nearest")

        # Suppression des axes, graduations et chiffres sur les bords
        self.axes.set_xticks([])
        self.axes.set_yticks([])

        self.axes.set_title(f"Tour {self.generation}")
        self.figure.canvas.draw()
        self.figure.canvas.flush_events()
        plt.pause(0.5)

    def statistique(self):
        print("Tour : ", self.generation)
        print("Loup : ", len(self.envi.predateur))
        print("Lapin : ", len(self.envi.proie))
        print("Somme : ", len(self.envi.predateur)+len(self.envi.proie))

    def start(self):
        for k in range(self.nb_lapin):
            self.envi.ajouter(Lapin(random.randint(0,self.largeur-1),random.randint(0,self.hauteur-1)))

        for k in range(self.nb_loup):
            self.envi.ajouter(Loup(random.randint(0,self.largeur-1),random.randint(0,self.hauteur-1)))

        for k in range(self.nb_carotte):
            self.envi.ajouter(Carotte(random.randint(0, self.largeur - 1), random.randint(0, self.hauteur - 1)))

        while len(self.envi.proie) != 0 and len(self.envi.predateur) != 0 :
            self.envi.simuler_tour()
            self.afficher()
            self.statistique()
            self.generation += 1

if __name__ == "__main__":
    jeu = Simulation(20,5,5)
    jeu.start()