"""
TP INCA/Ciqual - Squelette étudiant
Ne modifiez pas les noms des fonctions demandées.
"""

from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent

# Chemin absolu vers le dossier "donnees"
DOSSIER = BASE_DIR / "donnees"


individus = pd.read_csv(f"{DOSSIER}/individus.csv")
groupes = pd.read_csv(f"{DOSSIER}/groupes.csv")
aliments = pd.read_csv(f"{DOSSIER}/aliments.csv")
constituants = pd.read_csv(f"{DOSSIER}/constituants.csv")
composition = pd.read_csv(f"{DOSSIER}/composition.csv")
consommations = pd.read_csv(f"{DOSSIER}/consommations.csv")


# Partie A
Partie_A = False

if Partie_A :
    print("Individus :" )
    print(f'Nombre colonne {individus.shape[1]}')
    print(f'Nombre lignes {individus.shape[0]}')
    print(f'Noms des colonnes {individus.columns}')

    print("Groupes :" )
    print(f'Nombre colonne {groupes.shape[1]}')
    print(f'Nombre lignes {groupes.shape[0]}')
    print(f'Noms des colonnes {groupes.columns}')

    print("Aliments :" )
    print(f'Nombre colonne {aliments.shape[1]}')
    print(f'Nombre lignes {aliments.shape[0]}')
    print(f'Noms des colonnes {aliments.columns}')

    print("Constituants :" )
    print(f'Nombre colonne {constituants.shape[1]}')
    print(f'Nombre lignes {constituants.shape[0]}')
    print(f'Noms des colonnes {constituants.columns}')

    print("Composition :" )
    print(f'Nombre colonne {composition.shape[1]}')
    print(f'Nombre lignes {composition.shape[0]}')
    print(f'Noms des colonnes {composition.columns}')

    print("Consommation :" )
    print(f'Nombre colonne {consommations.shape[1]}')
    print(f'Nombre lignes {consommations.shape[0]}')
    print(f'Noms des colonnes {consommations   .columns}')



# Partie B recherche
def rechercher_aliment(texte):
    """Retourne les aliments dont le libellé contient texte."""
    resultat = aliments[aliments["libelle"].str.contains(texte, case=False, na=False)]
    return resultat


def composition_aliment(code_aliment):
    """Retourne la composition d'un aliment, avec nom et unité du constituant."""
    resultat = composition[composition["code_aliment"] == code_aliment].merge(constituants, on="code_const", how = "left")
    return resultat


# Parte C : Calcul nutritionnel
def apport_aliment(code_aliment, quantite_g):
    """Calcule les apports pour une quantité donnée en grammes."""
    compo = composition_aliment(code_aliment)
    apport = compo[['nom','teneur']]
    apport['teneur'] = apport['teneur'] * quantite_g / 100
    return apport


def apports_individu(id_individu):
    """Calcule les apports nutritionnels totaux d'un individu."""
    conso = consommations[consommations["id_individu"] == id_individu]

    tous_les_apports = []

    for id_ligne in range(len(conso)):
        apport = apport_aliment(conso.iloc[id_ligne]["code_aliment"], conso.iloc[id_ligne]["quantite_g"])
        if apport is not None and not apport.empty:
            apport_t = apport.T
            apport_t.columns = apport_t.iloc[0]
            apport_t = apport_t.iloc[[1]]
            tous_les_apports.append(apport_t)


    if tous_les_apports:
        liste_apports = pd.concat(tous_les_apports, ignore_index=True)
    else:
        liste_apports = pd.DataFrame()

    conso = conso.reset_index(drop=True)
    liste_apports = liste_apports.reset_index(drop=True)
    resultat = pd.concat([conso, liste_apports], axis=1)

    totaux_globaux = resultat[['Energie', 'Proteines', 'Lipides', 'Glucides', 'Fibres']].sum()
    
    return totaux_globaux




def apports_par_repas(id_individu):
    """Calcule les apports nutritionnels totaux d'un individu."""
    conso = consommations[consommations["id_individu"] == id_individu]

    tous_les_apports = []

    for id_ligne in range(len(conso)):
        apport = apport_aliment(conso.iloc[id_ligne]["code_aliment"], conso.iloc[id_ligne]["quantite_g"])
        if apport is not None and not apport.empty:
            apport_t = apport.T
            apport_t.columns = apport_t.iloc[0]
            apport_t = apport_t.iloc[[1]]
            tous_les_apports.append(apport_t)


    if tous_les_apports:
        liste_apports = pd.concat(tous_les_apports, ignore_index=True)
    else:
        liste_apports = pd.DataFrame()

    conso = conso.reset_index(drop=True)
    liste_apports = liste_apports.reset_index(drop=True)
    resultat = pd.concat([conso, liste_apports], axis=1)

    totaux_par_repas = resultat.groupby(['date', 'repas'])[['Energie', 'Proteines', 'Lipides', 'Glucides', 'Fibres']].sum()
    
    return totaux_par_repas


if __name__ == "__main__":
    #print(rechercher_aliment("pain"))
    #print(composition_aliment(1001))
    #print(apport_aliment(1001, 60))
    print(apports_individu(1))
    print(apports_par_repas(1))
