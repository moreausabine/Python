"""
Tests simples pour la simulation écosystème proie-prédateur.

Pas de classes de test, pas d'assert : chaque vérification est un if/else.
Si la condition est vraie, on affiche 'YES'. Sinon, le script s'arrête
immédiatement avec sys.exit() et un message expliquant ce qui a échoué.
"""

import importlib.util
import os
import sys

import matplotlib

matplotlib.use("Agg")  # pas d'affichage pendant les tests

# --- Chargement du module à tester ---
# Adapte ce nom si ton fichier source s'appelle différemment.
NOM_FICHIER_SOURCE = "2_TP_code_v2.py"

CHEMIN_SOURCE = os.path.join(os.path.dirname(os.path.abspath(__file__)), NOM_FICHIER_SOURCE)
spec = importlib.util.spec_from_file_location("ecosysteme", CHEMIN_SOURCE)
eco = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eco)

Lapin = eco.Lapin
Loup = eco.Loup
Carotte = eco.Carotte
Environnement = eco.Environnement
Config = eco.Config

# ---- TEST 1
print("Test 1 — Création d'un animal")
lapin = Lapin(5, 5)

if lapin.age == 0:
    print("YES")
else:
    sys.exit(f"ÉCHEC : l'âge devrait être 0 à la création, mais vaut {lapin.age}")

if lapin.energie == Config.LAPIN.ENERGIE_INITIALE:  # 100
    print("YES")
else:
    sys.exit(f"ÉCHEC : l'énergie devrait être {Config.LAPIN.ENERGIE_INITIALE} à la création, mais vaut {lapin.energie}")

# ---- TEST 2
print("Test 2 — Vieillissement")
lapin = Lapin(5, 5)
age_avant, energie_avant = lapin.age, lapin.energie

lapin.vieillir()

if lapin.age > age_avant:
    print("YES")
else:
    sys.exit(f"ÉCHEC : l'âge n'a pas augmenté après vieillir() (avant={age_avant}, après={lapin.age})")

if lapin.energie < energie_avant:
    print("YES")
else:
    sys.exit(f"ÉCHEC : l'énergie n'a pas diminué après vieillir() (avant={energie_avant}, après={lapin.energie})")


# ---- TEST 3
print("Test 3 — Mort")
lapin = Lapin(5, 5, energie=0)

if lapin.mort() is True:
    print("YES")
else:
    sys.exit("ÉCHEC : un animal avec 0 d'énergie devrait être considéré comme mort")


# ---- TEST 4
print("Test 4 — Déplacement")
env = Environnement(50, 50)
lapin = Lapin(5, 5)
x_avant, y_avant = lapin.x, lapin.y

lapin.bouger(1, 0, env)  # une case vers la droite

if (lapin.x, lapin.y) != (x_avant, y_avant):
    print("YES")
else:
    sys.exit(f"ÉCHEC : les coordonnées n'ont pas changé après bouger() (position restée ({x_avant}, {y_avant}))")

if lapin.x == (x_avant + lapin.vitesse) % env.largeur and lapin.y == y_avant:
    print("YES")
else:
    sys.exit(f"ÉCHEC : déplacement incorrect, attendu x={(x_avant + lapin.vitesse) % env.largeur}, y={y_avant}, obtenu x={lapin.x}, y={lapin.y}")


# ---- TEST 5
print("Test 5 — Chasse")
loup = Loup(10, 10, energie=50)   # énergie < max pour observer le gain
lapin = Lapin(10, 10)

loup.chasser(lapin)

if lapin.mort() is True:
    print("YES")
else:
    sys.exit("ÉCHEC : le lapin devrait être mort après avoir été chassé par le loup")

if loup.energie > 50:
    print("YES")
else:
    sys.exit(f"ÉCHEC : l'énergie du loup devrait avoir augmenté après la chasse, elle vaut {loup.energie}")


# ---- TEST 6
print("Test 6 — Environnement")
# Remarque : le code ne définit pas d'attribut unique `environnement.animaux` ;
# les lapins et les loups sont rangés séparément dans `proie` et `predateur`.
env = Environnement(50, 50)

for _ in range(3):
    env.ajouter(Lapin(0, 0))
for _ in range(2):
    env.ajouter(Loup(0, 0))

if len(env.proie) == 3:
    print("YES")
else:
    sys.exit(f"ÉCHEC : 3 lapins ajoutés, mais len(env.proie) vaut {len(env.proie)}")

if len(env.predateur) == 2:
    print("YES")
else:
    sys.exit(f"ÉCHEC : 2 loups ajoutés, mais len(env.predateur) vaut {len(env.predateur)}")

if len(env.proie) + len(env.predateur) == 5:
    print("YES")
else:
    sys.exit(f"ÉCHEC : le total attendu est 5, obtenu {len(env.proie) + len(env.predateur)}")


# ---- TEST 7
print("Test 7 — Suppression")
env = Environnement(50, 50)
lapin_mort = Lapin(0, 0, energie=0)
env.ajouter(lapin_mort)

if lapin_mort in env.proie:
    print("YES")
else:
    sys.exit("ÉCHEC : le lapin mort devrait être présent dans env.proie avant suppression")

env.supprimer_morts()

if lapin_mort not in env.proie:
    print("YES")
else:
    sys.exit("ÉCHEC : le lapin mort est toujours dans env.proie après supprimer_morts()")


print("\nTous les tests sont passés.")