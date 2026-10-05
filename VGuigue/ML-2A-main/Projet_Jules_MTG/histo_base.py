import json
from pathlib import Path

dossier_courant = Path(__file__).resolve().parent
chemin = 'default-cards.jsonl'
Dossier_json = dossier_courant / chemin

data = []

# Lecture ligne par ligne
with open(Dossier_json, "r", encoding="utf-8") as f:
    for line in f:
        data.append(json.loads(line))

print(f"{len(data)} objets chargés.")

print(type(data))
print(type(data[0]))

for 