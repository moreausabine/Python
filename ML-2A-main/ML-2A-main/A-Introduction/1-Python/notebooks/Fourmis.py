import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from random import randint

temp = 300
nb_fourmi = 100

class Fourmis:
    def __init__(self,x=0,y=0):
        self.x = x
        self.y = y

    def bouge(self):
        self.x = self.x + randint(-1,1)
        self.y = self.y + randint(-1,1)

Fourmillière = []
for k in range(nb_fourmi) :
    Fourmillière.append(Fourmis())

Deplacement = []
for i in range(temp):
    Position_inter = []
    for fourmi in Fourmillière :
        Position_inter.append([fourmi.x,fourmi.y])
        fourmi.bouge()
    Deplacement.append(Position_inter)

#print(Deplacement)

"""plt.figure(figsize=(8, 6))

for numero_fourmi in range(nb_fourmi):
    x = [tour[numero_fourmi][0] for tour in Deplacement]
    y = [tour[numero_fourmi][1] for tour in Deplacement]

    plt.plot(
        x,
        y,
        marker="o",
        label=f"Fourmi {numero_fourmi + 1}",
        alpha=0.4
    )

plt.title("Déplacement des fourmis à chaque tour")
plt.xlabel("Position X")
plt.ylabel("Position Y")
plt.axhline(0, color="black", linewidth=0.5)
plt.axvline(0, color="black", linewidth=0.5)
plt.grid(True)
plt.legend()
plt.show()"""



# Récupération des limites du graphique
positions = [
    position
    for tour in Deplacement
    for position in tour
]

x_values = [position[0] for position in positions]
y_values = [position[1] for position in positions]

marge = 2

fig, ax = plt.subplots(figsize=(8, 6))

ax.set_xlim(min(x_values) - marge, max(x_values) + marge)
ax.set_ylim(min(y_values) - marge, max(y_values) + marge)
ax.set_aspect("equal")
ax.grid(True)

fourmis = ax.scatter([], [], s=25, alpha=0.7)

def afficher_tour(numero_tour):
    positions_tour = Deplacement[numero_tour]

    fourmis.set_offsets(positions_tour)
    ax.set_title(
        f"Position des fourmis - Tour {numero_tour + 1}/{len(Deplacement)}"
    )

    return fourmis,

animation = FuncAnimation(
    fig,
    afficher_tour,
    frames=len(Deplacement),
    interval=50,
    blit=True,
    repeat=True
) 

plt.show()