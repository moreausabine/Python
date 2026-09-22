import random as rd

import matplotlib.pyplot as plt

# -----------------------------------------------------------------
#                          Algorithme de tri
# -----------------------------------------------------------------

rd.seed(10)

tri = 'Rapide_2'

# Liste L :
N = 5
L = [ k for k in range(N)]
rd.shuffle(L)

L1 = L.copy()
L2 = L.copy()
print(L1)

# Par selction
def selection(L):
    L_etape = []
    for k in range(len(L)-1):
        min = L[k]
        index = 0
        for i in range(len(L[k:])) :
            if min >= L[k+i] :
                min = L1[k+i]
                index = i+k

        min = L.pop(index)
        L.append(L[k])
        L[k] = min
        L_etape.append(L)
    return L, L_etape


if tri == 'Selection' :
    L1, L_etape = selection(L1)
    print(L_etape)


# par quick
def quick(L):
    if len(L) <= 1:
        return L, 
    else : 
        pivot = L[-1]
        index = 0
        for k in range(len(L)):
            if L[index] > pivot:
                tmp = L[index]
                L.remove(tmp)
                L.append(tmp)
            else : 
                index += 1
        return quick(L[:index-1]) + [L[index-1]] + quick(L[index:])


if tri == 'Rapide' :
    L_tri = quick(L2)
    print(L_tri)



