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


# =============================================================================
# Partie A : Découverte des données
# =============================================================================
Partie_A = False

if Partie_A:
    tables = {
        "Individus": individus,
        "Groupes": groupes,
        "Aliments": aliments,
        "Constituants": constituants,
        "Composition": composition,
        "Consommations": consommations,
    }
    for nom_table, df in tables.items():
        print(f"{nom_table} :")
        print(f"Nombre colonnes : {df.shape[1]}")
        print(f"Nombre lignes   : {df.shape[0]}")
        print(f"Noms des colonnes : {list(df.columns)}")
        print(df.head())
        df.info()
        print()

# --- Réponses A2 -------------------------------------------------------------
# 1. La table qui contient les individus est `individus` (clé : id_individu).
# 2. La table qui décrit les aliments est `aliments` (clé : code_aliment) ;
#    elle est rattachée à `groupes` via code_groupe.
# 3. Les valeurs nutritionnelles sont dans `composition` (colonne `teneur`,
#    pour 100 g d'aliment). Les noms et unités des constituants sont dans
#    `constituants`.
# 4. `composition` possède deux codes (code_aliment, code_const) car c'est une
#    table d'association (relation N-N) : un aliment contient plusieurs
#    constituants, et un constituant est présent dans plusieurs aliments.
#    La clé primaire est donc composée du couple (code_aliment, code_const) :
#    une teneur n'a de sens que pour un aliment ET un constituant donnés.


# =============================================================================
# Partie B : Recherche
# =============================================================================
def rechercher_aliment(texte):
    """Retourne les aliments dont le libellé contient texte (sans casse)."""
    resultat = aliments[aliments["libelle"].str.contains(texte, case=False, na=False)]
    return resultat


def composition_aliment(code_aliment):
    """Retourne la composition d'un aliment, avec nom et unité du constituant."""
    resultat = composition[composition["code_aliment"] == code_aliment].merge(
        constituants, on="code_const", how="left"
    )
    return resultat


# =============================================================================
# Partie C : Calcul nutritionnel
# =============================================================================
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


# =============================================================================
# Partie D : Les jointures
# =============================================================================
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



# --- Réponses Partie D -------------------------------------------------------
# 1. La jointure avec `aliments` se fait sur la colonne `code_aliment`
#    (clé étrangère dans consommations, clé primaire dans aliments).
# 2. La jointure avec `composition` se fait aussi sur `code_aliment`
#    (puis `composition` ⨝ `constituants` se fait sur `code_const`).
# 3. `composition` augmente fortement le nombre de lignes car la relation est
#    1-N : chaque consommation d'un aliment est dupliquée autant de fois que
#    cet aliment a de constituants (ex : 1 ligne de pain -> 1 ligne par
#    constituant : énergie, protéines, lipides, glucides, fibres...).
#    Nb lignes ≈ nb consommations × nb constituants par aliment.
# 4. Une ligne du DataFrame final = "la consommation d'un aliment par un
#    individu, à une date et un repas, pour un constituant donné", avec la
#    teneur pour 100 g et l'apport correspondant (teneur × quantité / 100).


# =============================================================================
# Partie E : Analyse par repas
# =============================================================================
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


def apports_par_repas_tableau(id_individu):
    """Apports par type de repas (toutes dates confondues), un constituant par colonne.
 
    Format proche de l'exemple de l'énoncé :
        repas   Energie   Proteines   Lipides ...
    """
    resultat = consommations[consommations["id_individu"] == id_individu]
    resultat = resultat.merge(composition, on="code_aliment", how="left")
    resultat = resultat.merge(constituants, on="code_const", how="left")
    resultat["apport"] = resultat["quantite_g"] * resultat["teneur"] / 100
    tableau = resultat.pivot_table(index="repas", columns="nom", values="apport", aggfunc="sum")
    return tableau


# =============================================================================
# Partie F : Questions d'analyse
# =============================================================================

# Q1 : Combien d'aliments différents sont consommés par l'individu 1 ?
def nb_aliment_diff(id_individu):
    resultat = individus[individus["id_individu"] == id_individu].merge(consommations, on="id_individu", how = "left")
    resultat = resultat.drop(["sexe", "age", "poids_kg", "taille_cm", "id_conso"], axis=1)
    return resultat.shape[0]


# Q6 : Quel individu consomme la plus grande quantité totale d'aliments ?
def plus_manger():
    quantites = consommations.groupby("id_individu")["quantite_g"].sum()
    return quantites.idxmax()


# Q7 : Quel individu possède l'apport énergétique total le plus élevé ?
def comp_apport_energetique():
    apport = []
    for id_individu in individus["id_individu"]:
        apport_par_ind = apports_individu_merge(id_individu)
        if "Energie" in apport_par_ind.index:
            apport.append([id_individu, apport_par_ind.loc["Energie", "apport"]])
        else:
            apport.append([id_individu, 0.0])

    energies = np.array([item[1] for item in apport], dtype=float)
    max_index = np.argmax(energies)

    return apport[max_index][0]


def repas_plus_energetique(id_individu):
    """Q4 : type de repas qui apporte le plus d'énergie (toutes dates cumulées)."""
    tableau = apports_par_repas_tableau(id_individu)
    return tableau["Energie"].idxmax()



# =============================================================================
# Partie G : Qualité des données
# =============================================================================
def controler_donnees(df=None):
    """Détecte et écarte les anomalies d'un DataFrame de consommations.

    Catégories contrôlées :
      1. valeurs manquantes ;
      2. quantités nulles ou négatives ;
      3. lignes identiques (doublons) ;
      4. codes aliments absents de la table `aliments` ;
      + contrôles sur `composition` : code constituant absent de `constituants`,
        teneur manquante.

    Retourne le DataFrame nettoyé (les anomalies sont affichées).
    """
    conso_bis = (consommations if df is None else df).copy()

    print("--- Valeurs manquantes ---")
    print(conso_bis.isna().sum())
    conso_bis = conso_bis.dropna()

    print("--- Quantités nulles ou négatives ---")
    quantite_invalide = conso_bis[conso_bis["quantite_g"] <= 0]
    print(f"Lignes avec quantité <= 0 : {len(quantite_invalide)}")
    print(quantite_invalide)
    conso_bis = conso_bis[conso_bis["quantite_g"] > 0]

    print("--- Lignes identiques (doublons) ---")
    doublons = conso_bis[conso_bis.duplicated(keep=False)]
    print(f"Nombre de lignes concernées par un doublon : {len(doublons)}")
    print(doublons)
    conso_bis = conso_bis.drop_duplicates()

    print("--- Code aliment non référencé ---")
    code_pas_bon = conso_bis[~conso_bis["code_aliment"].isin(aliments["code_aliment"])]
    print(f"Lignes avec un code aliment invalide : {len(code_pas_bon)}")
    print(code_pas_bon)
    conso_bis = conso_bis[conso_bis["code_aliment"].isin(aliments["code_aliment"])]

    print("--- Table composition ---")
    const_absent = composition[~composition["code_const"].isin(constituants["code_const"])]
    print(f"Lignes avec un code constituant absent : {len(const_absent)}")
    teneur_manquante = composition[composition["teneur"].isna()]
    print(f"Lignes avec une teneur manquante : {len(teneur_manquante)}")

    print(f"Nombre de lignes valides restantes : {len(conso_bis)}")
    return conso_bis


# =============================================================================
# Partie H : Défi algorithmique
# =============================================================================
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
    # Partie B
    print('Resultat de la fonction rechercher_aliment("pain") : \n', rechercher_aliment("pain"))
    print('Resultat de la fonction rechercher_aliment("POMME") : \n', rechercher_aliment("POMME"))
    print('Resultat de la fonction composition_aliment(1001) : \n', composition_aliment(1001))
    print()

    # Partie C
    print('Resultat de la fonction apport_aliment(1001, 60) : \n', apport_aliment(1001, 60))
    print('Resultat de la fonction apports_individu(1) : \n', apports_individu(1))
    print('Resultat de la fonction apports_par_repas(1) : \n', apports_par_repas(1))
    print()
    
    # Partie D
    print('Resultat de la fonction apports_individu_merge(1) : \n', apports_individu_merge(1))
    print()
    
    # Partie E
    print('Resultat de la fonction apports_individu_par_repas_merge(1) : \n', apports_individu_par_repas_merge(1))
    print('Resultat de la fonction apports_par_repas_tableau(1) : \n', apports_par_repas_tableau(1))
    print()
    
    # Partie F
    print('Q1 - Resultat de la fonction nb_aliment_diff(1) : ', nb_aliment_diff(1))
    print('Q2 - Apport total en energie de l\'individu 1 : ', apports_individu_merge(1).loc["Energie", "apport"])
    print('Q3 - Apport total en proteines de l\'individu 1 : ', apports_individu_merge(1).loc["Proteines", "apport"])
    print('Q4 - Resultat de la fonction repas_plus_energetique(1) : ', repas_plus_energetique(1))
    print('Q5 - Resultat de la fonction aliment_plus_proteine(1) : ', aliment_plus_proteine(1))
    print('Q6 - Resultat de la fonction plus_manger() : ', plus_manger())
    print('Q7 - Resultat de la fonction comp_apport_energetique() : ', comp_apport_energetique())
    print()
    
    # Partie G
    print('Resultat de la fonction controler_donnees(consommations_err) : \n', controler_donnees(consommations_err))
    print()
    
    # Partie H
    print('Resultat de la fonction aliment_plus_proteine(1) : ', aliment_plus_proteine(1))
    print()
    
