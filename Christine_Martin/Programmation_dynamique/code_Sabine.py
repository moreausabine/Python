import pandas as pd
import numpy as np

# -----------------------------------------------------------------------------
#                                      Données 
# -----------------------------------------------------------------------------

# Séquences
#X = 'ACCGACTTAGACAGGT'
#Y = 'TTACCGACGTATACAGCGTA'

# valeur de score
#match = 2
#mismatch = -2
#indel = -4

# Séquences test 1
X = 'ACGGCTAT'
Y = 'ACTGTAT'

# valeur de score
match = 2
mismatch = -1
indel = -2

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
    
    # remplissage et Max
    for i in range(1,len(df_seq.index)):
        for j in range(1,len(df_seq.columns)):
            Sub = df_seq.iloc[i-1,j-1] + df_score.loc[df_seq.index[i],df_seq.columns[j]]
            Del = df_seq.iloc[i-1,j] + df_score.loc[df_seq.index[i],'-']
            Ins = df_seq.iloc[i,j-1] + df_score.loc['-',df_seq.columns[j]]
            val_max = max(Sub, Del, Ins)
            df_seq.iloc[i, j] = val_max
            
            if val_max == Sub:
                max_matrix[i, j] = (i - 1, j - 1)
            elif val_max == Del:
                max_matrix[i, j] = (i - 1, j)
            else:
                max_matrix[i, j] = (i, j - 1)           

    df_max = pd.DataFrame(data=max_matrix, index=df_seq.index, columns=df_seq.columns)
    
    return df_seq, df_max

# Partie 2 bis ou 3 : Retrouver le.s chemin.s
def trad(Etape, df_max):
    

def find_way(df_max):
    Etape =[df_max.iloc[-1,-1]]

    while Etape[-1] != (0,0):
        Etape.append(df_max.iloc[Etape[-1][0],Etape[-1][1]])

    return trad(Etape)

# -----------------------------------------------------------------------------
#                                      Main 
# -----------------------------------------------------------------------------

X = '-'+X
Y = '-'+Y
df_score = Tab_score(lettres_unique(X,Y), match, mismatch, indel)
print("tableau score fait")
#print(df_score)
print('---')

df_seq = Tab_seq(X,Y)
print("tableau seq fait")
#print(df_seq)
print('---')

df_seq,df_max = Remp_tab(df_score, df_seq)
print("tableau sequence rempli")
print(df_seq)
print('---')


Chemin = find_way(df_max)
print("Voici le chemiin")
print(Chemin)
print('---')