# ------------------------------------ Yatsy --------------------------------------

# Combien de classes ?
# - Dés : (attributs : valeur // méthode : changer valeur)
# - Combinaison : (valeur des différentes combinaison // méthode : compter point)
# - Main : (attributs : score // méthode :)
# - Joueur : (attributs : Main // méthode : interraction avec jeu, lancés dés ; conserver dès ; valider)
# - Ordi : (attributs : Main // méthode : calcul de savoir ce qu'il veut faire, lancés dés ; conserver dès ; valider)
# - Jeu : (attributs : // méthode : tour)
# - Menu
# - Affichage

# Fun : prison à dés

# ---------------------------------------------------------------------------------

import random as rd


# Dés :

class Dice():
    def __init__(self, valeur = 0):
        self.valeur = valeur

    def lancer(self):
        self.valeur = rd.randint(1,6)


# Main :

class Main():
    def __init__(self, score = 0):
        self.score = score
        self.dices = []
        for k in range(5):
            self.dices.append(Dice())

    def lancer_des(self, id_dice):
        for k in id_dice:
            self.



