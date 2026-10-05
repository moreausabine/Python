import json
import unicodedata
from pathlib import Path

# Récupère le dossier contenant actuellement Herbier_code.py
dossier_courant = Path(__file__).resolve().parent

# =====================================================================
#  Fonctions utiles
# =====================================================================

def tri_par_nom(plante):
    return plante.nom or ""

def enlever_accents(texte):
    nfd_form = unicodedata.normalize('NFD', texte)
    return "".join([c for c in nfd_form if unicodedata.category(c) != 'Mn'])


# =====================================================================
#  CLASSE
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
        rep = f"{self.nom} ({self.nom_scientifique}) - {self.famille}\nCycle : {self.cycle}\nBesoins : {', '.join(self.besoins)}"
        return rep


class Herbier:
    def __init__(self):
        self.classeur = []
        self.besoins = ['soleil', 'mi-ombre', 'ombre', 'arrosage faible', 'arrosage moyen', 'arrosage fort', 'eau douce', 'eau de mer']
        self.cycle = ['vivace', 'annuel', 'bisannuel']
        self.famille = []

    def mise_jour_besoins(self,besoins):
        if isinstance(besoins, str):
            besoins = besoins.split("/")

        for besoin in besoins:
            besoin = besoin.strip().lower()
            if besoin and besoin not in self.besoins:
                self.besoins.append(besoin)
        

    def mise_jour_famille(self,famille):
        if famille and famille.lower() not in self.famille :
            self.famille.append(famille)

    def ajouter_plante(self, plante):
        if not plante.nom:
            print("Plante sans nom : non ajoutée")
            return False
        for p in self.classeur:
            if p.nom_scientifique == plante.nom_scientifique and p.famille == plante.famille:
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
        chaine_car = enlever_accents(chaine_car.strip().lower())
        for plante in self.classeur:
            for cle, valeur in vars(plante).items():
                if cle == "photo" or valeur is None:
                    continue
                if chaine_car in enlever_accents(str(valeur).lower()):
                    Liste_plante.append(plante)
                    break
        return Liste_plante



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

        return Liste_plante

        



class Interagir:
    def __init__(self):
        self.herbier = Herbier()

    def demander(self, question, valides):
        """Retourne None si vide, sinon une valeur valide."""
        while True:
            rep = self.nettoyage_generique(input(question))
            if rep is None or rep in valides:
                return rep
            print("Valeur non valide, réessayez (ou laissez vide).")

    def nettoyer_famille(self, valeur):
        famille = self.nettoyage_generique(valeur)
        if famille and famille.endswith("ees"):
            famille = famille[:-3] + "eae"
        return famille

    def nettoyage_generique(self, valeur):
        if valeur is None:
            return None
        valeur = valeur.strip().lower()
        if valeur in ('', 'null'):
            return None
        return enlever_accents(valeur).replace('_', ' ')

    def nettoyage_donnees(self, donnees):
        for plante in donnees:
            for cle, valeur in plante.items():
                if cle == "photo":
                    continue
                if cle == "besoins":
                    nettoyes = [self.nettoyage_generique(b) for b in valeur]
                    plante[cle] = [b for b in nettoyes if b]
                elif cle == "famille":
                    plante[cle] = self.nettoyage_generique(valeur)
                    plante[cle] = self.nettoyer_famille(valeur)
                else:
                    plante[cle] = self.nettoyage_generique(valeur)
        return donnees


    def ajouter_pleins_plantes(self):
        chemin = input("Chemin du fichier JSON (vide = plantes_degradees.json) : ").strip()
        chemin = chemin or 'plantes_degradees.json'
        path_json = dossier_courant / chemin
        try:
            with open(path_json, "r", encoding="utf-8") as fichier:
                donnees = json.load(fichier)
        except FileNotFoundError:
            print(f"Fichier introuvable : {path_json}")
            return
        except json.JSONDecodeError:
            print("Le fichier n'est pas un JSON valide")
            return

        donnees = self.nettoyage_donnees(donnees)
        print(donnees)

        ajoutees = 0
        for p in donnees:
            if p["cycle"] not in self.herbier.cycle:
                print(f"Cycle non valide pour {p['nom']} : ignorée")
                continue
            plante = Plante(p["nom"], p["nom_scientifique"], p["famille"],
                            p["cycle"], p["besoins"], p["photo"])
            if self.herbier.ajouter_plante(plante):
                ajoutees += 1
        print(f"{ajoutees} plante(s) ajoutée(s)")


    def ajouter_plante_herbier(self):
        choix = input('Ajouter une plante paramètre par paramètre (1) ou en une liste (2) ? : ').strip()

        if choix == '1':
            nom = self.nettoyage_generique(input('Nom vernaculaire ? : '))
            nom_scientifique = self.nettoyage_generique(input('Nom scientifique ? : '))
            famille = self.nettoyage_generique(input('Famille ? : '))
            famille = self.nettoyer_famille(input('Famille ? : '))

            cycle = None
            while cycle is None:   # le cycle est obligatoire
                cycle = self.demander('Cycle ? (annuel/bisannuel/vivace) : ', self.herbier.cycle)
                print("Le cycle est obligatoire")
                
            besoins = []
            while True:
                b = self.nettoyage_generique(input('Besoin ? (ex : soleil, arrosage faible) : '))
                if b:
                    besoins.append(b)
                if input('Ajouter un autre besoin ? (oui/non) ').strip().lower() != 'oui':
                    break

            photo = input('Nom du fichier photo (vide = aucune) : ').strip() or None

        elif choix == '2':
            saisie = input('nom_verna//nom_scien//famille//cycle//besoin1/besoin2//photo : ')
            attributs = [e.strip() for e in saisie.split("//")]   # on garde les champs vides
            if len(attributs) < 5:
                print("Format invalide : il faut au moins 5 champs séparés par //")
                return

            nom, nom_scientifique, famille, cycle = [self.nettoyage_generique(a) for a in attributs[:4]]
            famille = self.nettoyer_famille(famille)
            if not nom or not nom_scientifique or not famille:
                print("Nom, nom scientifique et famille sont obligatoires")
                return
            if cycle not in self.herbier.cycle:
                print("Cycle non valide")
                return

            besoins = [self.nettoyage_generique(b) for b in attributs[4].split("/")]
            besoins = [b for b in besoins if b]
            photo = (attributs[5] or None) if len(attributs) > 5 else None

        else:
            print("Choix non valide")
            return

        if self.herbier.ajouter_plante(Plante(nom, nom_scientifique, famille, cycle, besoins, photo)):
            print("Plante ajoutée")

    def parcourir_herbier(self):
        print("Voulez-vous rechercher les plantes par :")
        print("(1) nom")
        print("(2) famille, cycle, besoin")
        choix = input().strip()
        if choix == '1':
            nom = input('Nom recherché : ')
            resultats = self.herbier.rechercher(nom)
            self.herbier.afficher_liste_plante(resultats)
        elif choix == '2':
            print("Laissez vide un paramètre que vous ne voulez pas utiliser.")
            cycle   = self.demander('Cycle recherché ? (vide = ignorer) : ', self.herbier.cycle)
            besoin  = self.demander('Besoin recherché ? (vide = ignorer) : ', self.herbier.besoins)
            famille = self.demander('Famille recherchée ? (vide = ignorer) : ', self.herbier.famille)
            resultats = self.herbier.filtrer(cycle, besoin, famille)
            self.herbier.afficher_liste_plante(resultats)
        else:
            print("Choix non valide")

    def menu(self):
        while True:
            print("\n===== HERBIER =====")
            print("1. Charger des plantes depuis un fichier JSON")
            print("2. Ajouter une plante")
            print("3. Rechercher / filtrer les plantes")
            print("4. Afficher tout l'herbier")
            print("5. Supprimer une plante")
            print("0. Quitter")
            choix = input("Votre choix : ").strip()

            if choix == '1':
                self.ajouter_pleins_plantes()
            elif choix == '2':
                self.ajouter_plante_herbier()
            elif choix == '3':
                self.parcourir_herbier()
            elif choix == '4':
                self.herbier.afficher_classeur()
            elif choix == '5':
                nom = self.nettoyage_generique(input("Nom de la plante à supprimer : "))
                if self.herbier.supprimer_plante(nom):
                    print("Plante supprimée")
                else:
                    print("Aucune plante de ce nom")
            elif choix == '0':
                print("À bientôt !")
                break
            else:
                print("Choix non valide")


            





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


if __name__ == "__main__":
    Interagir().menu()