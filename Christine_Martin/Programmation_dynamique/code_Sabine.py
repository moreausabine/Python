import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
#                                      Données 
# -----------------------------------------------------------------------------

# Séquences
X = 'ACCGACTTAGACAGGT'
Y = 'TTACCGACGTATACAGCGTA'

# valeur de score
match = 2
mismatch = -2
indel = -4

"""# Séquences test 1
X = 'ACGGCTAT'
Y = 'ACTGTAT'

# valeur de score
match = 2
mismatch = -1
indel = -2"""

# -----------------------------------------------------------------------------
#                                 Fonctions 
# -----------------------------------------------------------------------------

# Partie 0 : trouver toutes les lettres de X et Y pour créer le tableau match 
def lettres_unique(X,Y):  
    """ récupérer les lettres uniques pour créer tableau des scores """
    Lettres_uniques = list(dict.fromkeys(X+Y))
    return Lettres_uniques

def Tab_score(Lettres_uniques, match, mismatch, indel):
    """ Créer le tableau score """
    n = len(Lettres_uniques)
    Score = np.full(shape=(n, n), fill_value=mismatch)
    Score[0, :] = indel
    Score[:, 0] = indel
    np.fill_diagonal(Score, match)
    Score[0, 0] = 0
    
    df_tab_match = pd.DataFrame(data=Score, index=Lettres_uniques, columns=Lettres_uniques)
    return df_tab_match


# Partie 1 : faire le tableau
def Tab_seq(X,Y):
    """ créer le tableau séquence """
    df_tab_seq = pd.DataFrame(data= np.zeros((len(Y), len(X)), dtype=int), index=list(Y), columns=list(X))
    return df_tab_seq


# Partie 2 : Remplir le tableau
def Remp_tab(df_score, df_seq) :
    """ Remplir le tableau de séquence 2 étapes : initialisation et les formules """
    # initialisation
    for k in range(len(df_seq.columns)-1):
        df_seq.iloc[0, k + 1] = df_seq.iloc[0, k] + indel
        
    for k in range(len(df_seq.index)-1):
        df_seq.iloc[k+1,0] = df_seq.iloc[k, 0] + indel
    
    max_matrix = np.empty((len(df_seq.index), len(df_seq.columns)), dtype=object)

    for i in range(max_matrix.shape[0]):
        for j in range(max_matrix.shape[1]):
            if j == 0 and i ==0 :
                max_matrix[i, j] = None
            elif i == 0:
                max_matrix[i, j] = (i, j - 1)
            elif j == 0:
                max_matrix[i, j] = (i - 1, j)
            else:
                max_matrix[i, j] = None
    
    # remplissage et Max
    for i in range(1,len(df_seq.index)):
        for j in range(1,len(df_seq.columns)):
            Sub = df_seq.iloc[i-1,j-1] + df_score.loc[df_seq.index[i],df_seq.columns[j]]
            Del = df_seq.iloc[i-1,j] + df_score.loc[df_seq.index[i],'-']
            Ins = df_seq.iloc[i,j-1] + df_score.loc['-',df_seq.columns[j]]
            val_max = max(Sub, Del, Ins)
            df_seq.iloc[i, j] = val_max
            max_positions = []

            if val_max == Sub:
                max_positions.append((i - 1, j - 1))
            if val_max == Del:
                max_positions.append((i - 1, j))
            if val_max == Ins:
                max_positions.append((i, j - 1))
            max_matrix[i, j] = max_positions
           

    df_max = pd.DataFrame(data=max_matrix, index=df_seq.index, columns=df_seq.columns)
    print(df_max)
    return df_seq, df_max


def find_way(df_max):
    """Trouver un seul chemin en prenant toujours le premier prédécesseur."""

    i = len(df_max.index) - 1
    j = len(df_max.columns) - 1

    X_new = ''
    Y_new = ''

    while i != 0 or j != 0:
        precedent = df_max.iloc[i, j]

        if isinstance(precedent, list):
            precedent = precedent[0]
        pi, pj = precedent

        if pi == i - 1 and pj == j - 1:
            # Substitution ou identité
            X_new = df_max.columns[j] + X_new
            Y_new = df_max.index[i] + Y_new

        elif pi == i - 1 and pj == j:
            # Insertion
            X_new = '-' + X_new
            Y_new = df_max.index[i] + Y_new

        elif pi == i and pj == j - 1:
            # Suppression
            X_new = df_max.columns[j] + X_new
            Y_new = '-' + Y_new

        i, j = pi, pj

    return X_new, Y_new

def parcours(df_max, i, j, X_new, Y_new, solutions):
    if i == 0 and j == 0:
        solutions.append((X_new, Y_new))
        return

    precedents = df_max.iloc[i, j]

    if not isinstance(precedents, list):
        precedents = [precedents]

    for pi, pj in precedents:
        X_temp = X_new
        Y_temp = Y_new
        if pi == i - 1 and pj == j - 1: # Substitution ou identité
            X_temp = df_max.columns[j] + X_temp
            Y_temp = df_max.index[i] + Y_temp
        elif pi == i - 1 and pj == j: # Insertion
            X_temp = '-' + X_temp
            Y_temp = df_max.index[i] + Y_temp
        elif pi == i and pj == j - 1: # Deletion
            X_temp = df_max.columns[j] + X_temp
            Y_temp = '-' + Y_temp

        # Continuer le parcours
        parcours(df_max, pi, pj, X_temp, Y_temp, solutions)


def find_way_ttes_sol(df_max):
    """Trouver toutes les solutions optimales."""

    solutions = []
    i = len(df_max.index) - 1
    j = len(df_max.columns) - 1
    parcours(df_max, i, j, '', '', solutions)

    return solutions



# -----------------------------------------------------------------------------
#                                      Main 
# -----------------------------------------------------------------------------

def main(X,Y,match, mismatch, indel):
    X = '-'+X
    Y = '-'+Y
    a = input('Utiliser les valeurs de match, mismatch et indel par défaut (1) ou utiliser des nouvelles (2) : ')
    if a == '2':
        match = input('match : ')
        mismatch = input('mismatch : ')
        indel = input('indel : ')
    df_score = Tab_score(lettres_unique(X,Y), match, mismatch, indel)
    print("Le tableau score est fait")
    #print(df_score)

    df_seq = Tab_seq(X,Y)
    print("Le tableau des sequences est fait")
    df_seq,df_max = Remp_tab(df_score, df_seq)
    print("Le tableau des sequences est rempli")
    print(df_seq)
    print('---')
    print("Voici la valeur d'alignement optimal", df_seq.iloc[-1,-1])

    b = input('Avoir une séquence optimale (1) ou toutes les séquences optimales : (2)')
    if b == '2':
        solutions = find_way_ttes_sol(df_max)

        print("Voici les alignements optimaux :")

        for n, (X_new, Y_new) in enumerate(solutions, 1):
            print(f"\nAlignement {n} :")
            print("X :", X_new)
            print("Y :", Y_new)

        print('---')

    else :        
        X_new, Y_new = find_way(df_max)
        print("Voici l'alignement correspondant :")
        print('X : ',X_new)
        print('Y : ',Y_new)
        print('---')
        


if __name__ == "__main__":
    main(X,Y,match, mismatch, indel)