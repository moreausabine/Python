question 1 : quelles métriques à utiliser dans le cas de 5 classes équilibrées et dans le cas de classes très désiquilibrées ?
équilibrées : accuracy
très désiquilibrées : 

question 2 : comment faire pour avoir perfa et perfv listes des accuracy scores des différentes groupes de variables pour Xv et Yv train et Xa et Ya test (c'est du forward on initialise à 0 et on construit)

mod = SVC_linear()
met = accuracy_score
perfa = []
perfv = []
index = [2,5,1, ...]

for i in range(len(index)):
    mod.fit(Xa[:, index[:i+1]], Ya)
    pa = met(mod.predict(Xa[:, index[:i+1]]), Ya)
    pv = met(mod.predict(Xv[:, index[:i+1]]), Yv)
    perfa.append(pa)
    perfv.append(pv)


question 3 : tracer les courbes de perf 


question 4 validation croisée entre 2 modèles mod 1 et pmod2 avec une validation croisée de 10 folds combien d'initialisation ?
20

question 5 : pourquoi on fait une validation croisée ? et à quoi ça sert dans la comparaison de modèles ?

ça sert à avoir une p-value pour savoir si c'est significativement meilleur