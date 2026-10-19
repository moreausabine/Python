"""
TP INCA/Ciqual - Squelette étudiant
Ne modifiez pas les noms des fonctions demandées.
"""

from pathlib import Path

import numpy as np
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
consommations_err = pd.read_csv(f"{DOSSIER}/consommations_erreur.csv")


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


# Partie D : Les jointures
def apports_individu_merge(id_individu):
    #colonnes qui nous interessent : 
    # - individus : id_individu, 
    # - consommations : id_individu, code_aliment,
    # - composition : code_aliment, code_const, teneur,
    # - constituants : code_const, nom, unite
    
    resultat = individus[individus["id_individu"] == id_individu].merge(consommations, on="id_individu", how = "left")
    resultat = resultat.drop(["sexe", "age", "poids_kg", "taille_cm", "id_conso"], axis=1)
    resultat = resultat.merge(composition, on="code_aliment", how = "left")
    resultat = resultat.merge(constituants, on="code_const", how = "left")
    resultat["apport"] = resultat["quantite_g"]*resultat["teneur"]/100
    resultat = resultat.groupby(["nom"]).sum()
    resultat = resultat.drop(["id_individu", "code_aliment", "date", "unite", "repas", "quantite_g", "code_const", "teneur"], axis=1)
    return resultat

# Partie E : Analayse par repas
def apports_individu_par_repas_merge(id_individu):
    #colonnes qui nous interessent : 
    # - individus : id_individu, 
    # - consommations : id_individu, code_aliment,
    # - composition : code_aliment, code_const, teneur,
    # - constituants : code_const, nom, unite
    
    resultat = individus[individus["id_individu"] == id_individu].merge(consommations, on="id_individu", how = "left")
    resultat = resultat.drop(["sexe", "age", "poids_kg", "taille_cm", "id_conso"], axis=1)
    resultat = resultat.merge(composition, on="code_aliment", how = "left")
    resultat = resultat.merge(constituants, on="code_const", how = "left")
    resultat["apport"] = resultat["quantite_g"]*resultat["teneur"]/100
    resultat = resultat.groupby(["date","repas", "nom"]).sum()
    resultat = resultat.drop(["id_individu", "code_aliment", "unite", "quantite_g", "code_const", "teneur"], axis=1)
    return resultat

# Partie F : Questions d'analyse

# Combien aliment diff ?
def nb_aliment_diff(id_individu):
    #colonnes qui nous interessent : 
    # - individus : id_individu, 
    # - consommations : id_individu, code_aliment,

    resultat = individus[individus["id_individu"] == id_individu].merge(consommations, on="id_individu", how = "left")
    resultat = resultat.drop(["sexe", "age", "poids_kg", "taille_cm", "id_conso"], axis=1)
    return resultat.shape[0]

# individu mange plus de bouffe en poids
def plus_manger():
    resultat = individus.merge(consommations, on="id_individu", how = "left")
    resultat = resultat.drop(["sexe", "age", "poids_kg", "taille_cm", "id_conso", "repas", "date", "code_aliment"], axis=1)
    resultat = resultat.groupby(["id_individu"]).sum()
    return resultat['quantite_g'].idxmax()


# quel individu à l'apport énergétique le plus élevé
def comp_apport_energetique():
    apport = []
    for id_individu in individus["id_individu"]:
        apport_par_ind = apports_individu_merge(id_individu)
        apport.append([id_individu, apport_par_ind.loc["Energie", "apport"]])

    energies = np.array([item[1] for item in apport], dtype=float)
    max_index = np.argmax(energies)

    return apport[max_index][0]


# Partie H : Qualité des données
def controler_donnes(consommations):
    conso_bis = consommations.copy()
    
    print("Valeurs manquantes")
    print(conso_bis.isna().sum())
    conso_bis = conso_bis.dropna()

    print("Valeurs aberrantes (Quantités négatives)")
    quantite_negative = conso_bis[conso_bis['quantite'] <= 0]
    print(f"Lignes avec quantité négative : {len(quantite_negative)}")
    print(quantite_negative)
    conso_bis = conso_bis[conso_bis['quantite'] > 0]

    print("Lignes identiques (Doublons)")
    doublons = conso_bis[conso_bis.duplicated(keep=False)]
    print(f"Nombre de lignes en double : {len(doublons)}")
    print(doublons)
    conso_bis = conso_bis.drop_duplicates()

    print("Code aliment non référencé")
    if 'code_aliment' in conso_bis.columns and 'code_aliment' in aliments.columns:
        code_pas_bon = conso_bis[~conso_bis['code_aliment'].isin(aliments['code_aliment'])]
        print(f"Lignes avec un code aliment invalide : {len(code_pas_bon)}")
        print(code_pas_bon)
        conso_bis = conso_bis[conso_bis['code_aliment'].isin(aliments['code_aliment'])]

    print(f"Nombre de lignes valides restantes : {len(conso_bis)}")
    
    return conso_bis


# Partie H - Défi Algorithmique
def aliment_plus_proteine(id_individu):
    #colonnes qui nous interessent : 
    # - individus : id_individu, 
    # - consommations : id_individu, code_aliment,
    # - composition : code_aliment, code_const, teneur,
    # - constituants : code_const, nom, unite
    
    resultat = individus[individus["id_individu"] == id_individu].merge(consommations, on="id_individu", how = "left")
    resultat = resultat.drop(["sexe", "age", "poids_kg", "taille_cm", "id_conso"], axis=1)
    resultat = resultat.merge(composition, on="code_aliment", how = "left")
    resultat = resultat.merge(constituants, on="code_const", how = "left")
    resultat["apport"] = resultat["quantite_g"]*resultat["teneur"]/100

    code_aliment_prot = resultat.loc[resultat["nom"] == "Proteines", ["code_aliment", "apport"]]

    return code_aliment_prot.loc[code_aliment_prot['apport'].idxmax(),"code_aliment"]

if __name__ == "__main__":
    #print(rechercher_aliment("pain"))
    #print(composition_aliment(1001))
    #print(apport_aliment(1001, 60))
    #print(apports_individu(1))
    #print(apports_par_repas(1))
    #print(apports_individu_merge(1))
    #print(apports_individu_par_repas_merge(1))
    #print(nb_aliment_diff(1))
    print(plus_manger())
    #print(comp_apport_energetique())
    #print(controler_donnes(consommations_err))
    #print(aliment_plus_proteine(1))