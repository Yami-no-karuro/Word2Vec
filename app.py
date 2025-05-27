from lib.utils import load_model
from lib.math import cosine_distance

from lib.embedder import get_word_embedding
from lib.embedder import get_sentence_embedding

model: dict = load_model("models/model.pkl")

word_pairs = [
    ("cane", "cucciolo"),       # <- Similar
    ("mamma", "madre"),         # <- Similar
    ("scuola", "aula"),         # <- Similar
    ("fuoco", "ghiaccio"),      # <- Different
    ("giovane", "anziano"),     # <- Different
    ("felice", "triste")        # <- Different
]

print("Word Similarity")
for w1, w2 in word_pairs:
    e1 = get_word_embedding(w1, model)
    e2 = get_word_embedding(w2, model)

    d = cosine_distance(e1, e2)
    print(f'"{w1}" ↔ "{w2}": distance = {d:.18f}')

sentence_pairs = [
    ("Il bambino gioca nel parco", "Un ragazzo si diverte nel giardino"),           # <- Similar
    ("La mamma spinge il passeggino", "Una donna cammina con un bambino"),          # <- Similar
    ("Il sole splende alto nel cielo", "Sta per piovere nel pomeriggio"),           # <- Slightly different
    ("Sto leggendo un libro interessante", "La macchina è parcheggiata lontano"),   # <- Different
    ("I negozianti espongono la merce", "Le vetrine sono piene di prodotti")        # <- Related
]

print("Sentence Similarity")
for s1, s2 in sentence_pairs:
    e1 = get_sentence_embedding(s1, model)
    e2 = get_sentence_embedding(s2, model)

    d = cosine_distance(e1, e2)
    print(f'"{s1}"\n↔\n"{s2}"\n→ distance = {d:.18f}\n')

