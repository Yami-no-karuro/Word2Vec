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

for w1, w2 in word_pairs:
    e1 = get_word_embedding(w1, model)
    e2 = get_word_embedding(w2, model)

    d = cosine_distance(e1, e2)
    print(f'"{w1}" ↔ "{w2}": distance = {d:.18f}')

