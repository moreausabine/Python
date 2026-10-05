# Le projet

- on veut se concentrer que sur les cartes papier du jeu magic
- que sur les cartes pas les tokens
- prix en euro 
- à une date précise 01/10/2026 
- anglais ou seul langage possible

- foil ou pas (dédoubler): discuter avec le prof

est ce qu'on garde toutes les cartes : 
créature, artefact, instant, sorcery, enchantment, aura, planeswalker

on supprime
- planes
- les unsets
- cartes qui paraissent après le 1er octobre  


## Truc à prédire :

régressif : 
- le prix sur le marché la revente : 

classif :
- la couleur de mana
  Kmeans - 32 identités de couleur
  
  on fait comment : double face ?

  les colonnes prises en compte : le gameplay et c'est tt


## le plan d'action pour chaque prédiction

0- Preprocessing : regarder des corrélations entre les différentes colonnes pour éliminer des colonnes

1- faire des modèles simples en limitant les colonnes : limiter au max les OneHotEncoder

2- Faire réseaux de neurones pour pouvoir manier plus de données


## Les colonnes nécessaires

### Choses à devoir prendre en compte (simple) 
trouvable dans les bases de données :
- mana cost
- les effets (gainlife etc...)
- le nom de la carte (pour savoir ce que c'est) (genre moyen que les cartes Jace ou Urza coute plus cher le jean eude qui a une carte et aucun lore)
- la date de publication
- l'illustration (foil, full art, version)
- où a t elle été publiée : secret lair, truc exclusif...
- l'artiste
- si créature : force et défense
- le set genre spiderman osef 
- rareté

### Choses à devoir prendre en compte (peu simple) 
à créer :
- nombre de fois où il ya eu des reprints **avec la même illustration**
- A quel point le set s'est vendu
- pour créature : ratio entre force et défense et mana cost
- entrer proba de trouver (lier set et rareté)

### Choses à devoir prendre en compte (vraiment pas simple)
je sais pas si c'est possible mais :
- regarder les effets des autres cartes : en gros prendre en compte les synergies pour pouvoir prévoir si des vielles cartes vont booster
- nombre de fois dans des decks 


--> prendre en compte plus précisément les effets de la carte autrement que certains mots clefs

## Les colonnes

### attention au dédoublemenrt d'info :
les dates et set : redites ?

### les colones précises

{"object":"card",
**"id":"0000419b-0bba-4488-8f7a-6194544ce91e",**
**"oracle_id":"b34bb2dc-c1af-4d77-b0b3-a0fb342a5fc6",** 
"multiverse_ids":[668564],
"mtgo_id":129825,
"arena_id":91829,
"tcgplayer_id":558404,
"cardmarket_id":777725,
**"name":"Forest",**
**"lang":"en",**
**"released_at":"2024-08-02",**
"uri":"https://api.scryfall.com/cards/0000419b-0bba-4488-8f7a-6194544ce91e",
"scryfall_uri":"https://scryfall.com/card/blb/280/forest?utm_source=api",
**"layout":"normal",**
"highres_image":true,
"image_status":"highres_scan","image_updated_at":"2026-07-13T02:46:16Z",
"image_uris":{"small":"https://cards.scryfall.io/small/front/0/0/0000419b-0bba-4488-8f7a-6194544ce91e.jpg?1783910776",
"normal":"https://cards.scryfall.io/normal/front/0/0/0000419b-0bba-4488-8f7a-6194544ce91e.jpg?1783910776",
"large":"https://cards.scryfall.io/large/front/0/0/0000419b-0bba-4488-8f7a-6194544ce91e.jpg?1783910776",
"png":"https://cards.scryfall.io/png/front/0/0/0000419b-0bba-4488-8f7a-6194544ce91e.png?1783910776",
"art_crop":"https://cards.scryfall.io/art_crop/front/0/0/0000419b-0bba-4488-8f7a-6194544ce91e.jpg?1783910776",
"border_crop":"https://cards.scryfall.io/border_crop/front/0/0/0000419b-0bba-4488-8f7a-6194544ce91e.jpg?1783910776",
"thumb":"https://cards.scryfall.io/thumb/front/0/0/0000419b-0bba-4488-8f7a-6194544ce91e.webp?1783910776",
"grid":"https://cards.scryfall.io/grid/front/0/0/0000419b-0bba-4488-8f7a-6194544ce91e.webp?1783910776",
"display":"https://cards.scryfall.io/display/front/0/0/0000419b-0bba-4488-8f7a-6194544ce91e.webp?1783910776",
"art":"https://cards.scryfall.io/art/front/0/0/0000419b-0bba-4488-8f7a-6194544ce91e.webp?1783910776",
"crop":"https://cards.scryfall.io/crop/front/0/0/0000419b-0bba-4488-8f7a-6194544ce91e.webp?1783910776"},
**"mana_cost":"",**
**"cmc":0.0,**
**"type_line":"Basic Land — Forest",** # faire un one Hot Encoder
**"oracle_text":"({T}: Add {G}.)"**,
**"colors":[],**
**"color_identity":["G"],**
**"keywords":[],**
**"produced_mana":["G"],**
**"all_parts":[
    {"object":"related_card","id":"bddc66f7-4e94-4857-ba7d-6b0083d0bfa0",
    "component":"combo_piece",
    "name":"Forest",
    "type_line":"Basic Land — Forest",
    "uri":"https://api.scryfall.com/cards/bddc66f7-4e94-4857-ba7d-6b0083d0bfa0"},{"object":"related_card","id":"aed8ae06-ae54-4a16-8830-6547164ce6ae","component":"combo_piece","name":"Gilt-Leaf Alchemist","type_line":"Creature — Elf Druid","uri":"https://api.scryfall.com/cards/aed8ae06-ae54-4a16-8830-6547164ce6ae"}],** # nombre de carte mécanique liée

**"legalities":{"standard":"legal","future":"legal","historic":"legal","timeless":"legal","gladiator":"legal","pioneer":"legal","modern":"legal","legacy":"legal","pauper":"legal","vintage":"legal","penny":"legal","commander":"legal","oathbreaker":"legal","standardbrawl":"legal","brawl":"legal","competitivebrawl":"legal","alchemy":"legal","paupercommander":"legal","duel":"legal","oldschool":"not_legal","premodern":"legal","predh":"legal","tlr":"legal"},**
**"games":["paper","mtgo","arena"],** #pour filtrer
**"reserved":false,**
**"game_changer":false,**
**"foil":true,**
**"nonfoil":true,"**
**finishes":["nonfoil","foil"],***
"oversized":false,
**"promo":false,**
**"reprint":true,**
**"variation":false,**
*Ajouté par rapport à Fort*
**"card_faces": l**
**"defense": l** # pour les battles 
**"edhrec_rank": l**
**"loyalty": l**
**"power" : l**
**"toughness" : l**

"set_id":"a2f58272-bba6-439d-871e-7a46686ac018",
**"set":"blb",**
**"set_name":"Bloomburrow",**
**"set_type":"expansion",**
"set_uri":"https://api.scryfall.com/sets/a2f58272-bba6-439d-871e-7a46686ac018",
"set_search_uri":"https://api.scryfall.com/cards/search?order=set&q=e%3Ablb&unique=prints",
"scryfall_set_uri":"https://scryfall.com/sets/blb?utm_source=api",
"rulings_uri":"https://api.scryfall.com/cards/0000419b-0bba-4488-8f7a-6194544ce91e/rulings",
"prints_search_uri":"https://api.scryfall.com/cards/search?order=released&q=oracleid%3Ab34bb2dc-c1af-4d77-b0b3-a0fb342a5fc6&unique=prints",
**"collector_number":"280",**
**"digital":false,** # pour filtrer
**"rarity":"common",**
"card_back_id":"0aeebaf5-8c7d-4636-9e82-8c27447861f7",
**"artist":"David Robert Hovey",**
"artist_ids":["22ab27e3-6476-48f1-a9f7-9a9e86339030"],
"illustration_id":"fb2b1ca2-7440-48c2-81c8-84da0a45a626",
**"border_color":"black",**
"frame":"2015",
**"full_art":true,** # vérifier si elle existe en non full art
**"textless":false,**
**"booster":true,**
**"story_spotlight":false,**
**"prices":{"usd":"0.37","usd_foil":"0.54","usd_etched":null,*"eur":"0.27","eur_foil":"0.63"*,"tix":"0.03"},**
"related_uris":{"gatherer":"https://gatherer.wizards.com/Pages/Card/Details.aspx?multiverseid=668564&printed=false",
"tcgplayer_infinite_articles":"https://partner.tcgplayer.com/c/4931599/1830156/21018?subId1=api&trafcat=tcgplayer.com%2Fsearch%2Farticles&u=https%3A%2F%2Fwww.tcgplayer.com%2Fsearch%2Farticles%3FproductLineName%3Dmagic%26q%3DForest",
"tcgplayer_infinite_decks":"https://partner.tcgplayer.com/c/4931599/1830156/21018?subId1=api&trafcat=tcgplayer.com%2Fsearch%2Fdecks&u=https%3A%2F%2Fwww.tcgplayer.com%2Fsearch%2Fdecks%3FproductLineName%3Dmagic%26q%3DForest",
"edhrec":"https://edhrec.com/route/?cc=Forest"},
"purchase_uris":{"tcgplayer":"https://partner.tcgplayer.com/c/4931599/1830156/21018?subId1=api&u=https%3A%2F%2Fwww.tcgplayer.com%2Fproduct%2F558404%3Fpage%3D1",
"cardmarket":"https://www.cardmarket.com/en/Magic/Products?idProduct=777725&referrer=scryfall&utm_campaign=card_prices&utm_medium=text&utm_source=scryfall",
"cardhoarder":"https://www.cardhoarder.com/cards/129825?affiliate_id=scryfall&ref=card-profile&utm_campaign=affiliate&utm_medium=card&utm_source=scryfall"}}
