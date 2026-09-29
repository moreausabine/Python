import json
from pathlib import Path

# Récupère le dossier contenant actuellement Herbier_code.py
dossier_courant = Path(__file__).resolve().parent

# =====================================================================
#  Fonctions utiles
# =====================================================================

def tri_par_nom(plante):
    return plante.nom

def recherche_chaine_car(text, motif):
    debut = text.find(motif)
    if debut != -1:
        fin = debut + len(motif)
        return fin
    return False


# =====================================================================
#  CLASSE
# =====================================================================

class Plante:
    def __init__(self, nom, nom_scientifique, famille, cycle, besoins, photo=None):
        self.nom = nom
        self.nom_scientifique = nom_scientifique
        self.famille = famille
        self.cycle = cycle
        self.besoins = besoins
        self.photos = photo

    def __str__(self):
        rep = f"{self.nom} ({self.nom_scientifique}) - {self.famille}\nCycle : {self.cycle}\nBesoins : {self.besoins}"
        return rep


class Herbier:
    def __init__(self):
        self.classeur = []
        self.longueur = len(self.classeur)
        self.besoins = ['soleil', 'mi-ombre', 'ombre', 'arrosage faible', 'arrosage moyen', 'arrosage fort', 'eau douce', 'eau de mer']
        self.cycle = ['vivace', 'annuel', 'bisannuel']
        self.famille = []

    def mise_jour_besoins(self,Besoins):
        Besoins = Besoins.lower()
        besoins = [element for element in Besoins.split("/") if element]
        for besoin in besoins:
            if besoin.lower() not in self.besoins :
                self.besoins.append(besoin)

    def mise_jour_famille(self,famille):
        if famille.lower() not in self.famille :
            self.famille.append(famille)

    def ajouter_plante(self,plante):
        self.classeur.append(plante)
        self.classeur.sort(key=tri_par_nom)
        #Mettre à jour les listes de l'herbier : 
        self.mise_jour_besoins(plante.besoins)
        self.mise_jour_famille(plante.famille)

    def supprimer_plante(self, nom):
        for plante in self.classeur :
            if plante.nom == nom:
                self.classeur.remove(plante)
        #Mettre à jour les listes


    def afficher_classeur(self):
        print('------------------------')
        for plante in self.classeur:
            print(plante)
            print('---')
        print('------------------------')

    def afficher_liste_plante(self, Liste_plante):
        print("------------------------")
        if Liste_plante == []:
            print('Aucune correspondance')
        else :
            for plante in Liste_plante:
                print(plante)
                print("---")
        print("------------------------")


    def rechercher(self, chaine_car):
        Liste_plante = []
        for plante in self.classeur:
            trouve = False
            for valeur in vars(plante).values():
                if valeur is not None and chaine_car.lower() in str(valeur).lower():
                    trouve = True
                    break

            if trouve:
                Liste_plante.append(plante)

        self.afficher_liste_plante(Liste_plante)



    def filtrer(self, cycle, besoin, famille):
        cible = 0
        if cycle != None :
            cible += 1
        if besoin != None :
            cible += 1
        if famille != None :
            cible += 1

        Liste_plante = []
        for plante in self.classeur:
            trouve = 0
            if cycle != None and plante.cycle == cycle :
                trouve += 1
            if besoin != None and besoin in plante.besoins :
                trouve += 1
            if famille != None and plante.famille == famille :
                trouve += 1
            
            if trouve == cible:
                Liste_plante.append(plante)

        self.afficher_liste_plante(Liste_plante)

        



class Interagir:
    def __init__(self):
        self.herbier = Herbier()

    def nettoyage_generique(self, valeur):
        if valeur == None:
            return False
        # tout en minuscule
        valeur = valeur.lower()

        # supprimer les espaces de débuts et fins
        if valeur[0] == ' ' :
            valeur = valeur[1:]
        if valeur[-1] == ' ' :
            valeur = valeur[:-1]


        return valeur

    def nettoyage_donnees(self, donnees):
        for plante in donnees :
            # tt changer à part le lien de la photo
            for cle in list(plante.keys())[:-1]:
                if cle == "besoins":
                    for besoin in plante[cle]:
                        besoin = self.nettoyage_generique(besoin)
                        if not besoin :
                            # Ajouter quoi faire si vide
                            a = 1

                if cle == "famille":
                    valeur = plante[cle]
                    valeur = self.nettoyage_generique(valeur)
                    #Ajouter spécificité de fin de nom de famille



                
                


    def ajouter_pleins_plantes(self):
        chemin = input("Quel est le chemin du dossier que vous voulez intégrer à l'herbier ? ")
        chemin = 'plantes_degradees.json'
        path_json = dossier_courant / chemin
        with open(path_json, "r", encoding="utf-8") as fichier:
            donnees = json.load(fichier)

        donnees = self.nettoyage_donnees(donnees)

        # ajouter plante

    def ajouter_plante_herbier(self):
        choix = input('Voulez vous ajouter une plante paramètre par paramètres (1) ou en une liste (2) ? : ')
        if choix == '1':
            nom = input('Nom vernaculaire ? :')
            nom_scientifique = input('Nom scientifique ? :')
            famille = input('Famille ? :')
            bon = False
            while not bon :
                cycle = input('Cycle ? (annuel/bisannuel/vivace) : ')
                if cycle.lower() not in self.herbier.cycle:
                    print('Cycle non valide')
                    cycle = input('Cycle ? (annuel/bisannuel/vivace) : ')
                else : 
                    bon = True

            Besoin = ''
            bon = False
            while not bon :
                besoin = input('Besoin ? (annuel/bisannuel/vivace) : ')
                Besoin = Besoin + besoin + '/'
                choix2 = input('Voulez vous ajouter un besoin ? (oui/non)')
                if choix2.lower == 'non':
                    bon = True

        elif choix == '2':
            liste_parametres = input('Liste paramètres nom_verna//nom_scien//famille//cycle//besoin1/besoin2//chemin_photo : ')
            # diviser la liste paramètres en les données dont j'ai besoin pour ajouter la plante
            attributs = [element for element in liste_parametres.split("//") if element]
            nom = attributs[0]
            nom_scientifique = attributs[1]
            famille = attributs[2]
            cycle = attributs[3]
            besoin = attributs[4]

        self.herbier.ajouter_plante(Plante(nom, nom_scientifique, famille, cycle, besoin))

    def parcourir_herbier(self):
        print("Voulez rechercher les plantes par :")
        print("(1) nom")
        print("(2) famille, cycle, besoin ?")
        choix = input()
        if choix == '1':
            nom = input('Nom recherché : ')
            self.herbier.rechercher(nom)

        elif choix == '2':
            print("Rentrez successivement les informations demandées si vous n'avez pas envie de caractériser un paramètres laissez le vide")
            bon = False
            while not bon :
                cycle = input('Cycle recherché ? (annuel/bisannuel/vivace) : ')
                if cycle.lower() not in self.herbier.cycle:
                    print('Cycle non valide')
                    cycle = input('Cycle ? (annuel/bisannuel/vivace) : ')
                else : 
                    bon = True

            bon = False
            while not bon :
                besoin = input('Besoin recherché ? ')
                if besoin.lower() not in self.herbier.besoins:
                    print("Aucune plante n'a ce besoin")
                    choix2 = input('Voulez vous rentrer un autre besoin ? (oui/non)')
                    if choix2.lower == 'non':
                        bon = True
                else : 
                    bon = True

            bon = False
            while not bon :
                famille = input('Famille recherchée ? ')
                if famille.lower() not in self.herbier.famille:
                    print("Aucune plante de l'herbier n'est de cette famille")
                    choix2 = input('Voulez vous rentrer un autre famille ? (oui/non)')
                    if choix2.lower == 'non':
                        bon = True
                else : 
                    bon = True


            self.herbier.filtrer(cycle, besoin, famille)
        
        # phrase type : plantes [cycle] ayant besoin de [besoin] de la famille des [famille]



            





# =====================================================================
#  Exemple
# =====================================================================

# Test 1
"""tomate = Plante('Tomate', 'Solanum lycopersicum', 'Solanaceae', 'annuel', 'soleil/arrosage régulier')
lierre = Plante('Lierre grimpant', 'Hedera helix L.', 'Araliaceae', 'vivace', 'soleil/ support')
nenuphar = Plante('Nénuphar blanc ou Nymphéa', 'Nymphaea alba', 'Nymphaeaceae', 'vivace', 'eau douce')"""
#print(tomate)
#print(lierre)
#print(nenuphar)

# Test 2
#print('Test 2')
"""herbier = Herbier()
herbier.ajouter_plante(tomate)
herbier.ajouter_plante(lierre)
herbier.ajouter_plante(nenuphar)"""
#herbier.afficher_classeur()
#print()

# Test 3 
#print('Test 3')
#herbier.supprimer_plante('Tomate')
#herbier.afficher_classeur()
#herbier.ajouter_plante(tomate)
#print()

# Test 4
#print('Test 4')
#herbier.rechercher('Tomate')
#herbier.rechercher('eae')
#herbier.rechercher('zifnujezi')
#print()

# Test 5
#herbier.filtrer('vivace', None, None )
#herbier.filtrer('vivace', 'soleil', None )

test = Interagir()
test.ajouter_pleins_plantes()