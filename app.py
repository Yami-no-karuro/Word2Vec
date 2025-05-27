from lib.utils import load_model
from lib.math import cosine_distance

from lib.embedder import get_word_embedding
from lib.embedder import get_sentence_embedding

model: dict = load_model("models/model.pkl")

embedding_a: list[float] = get_word_embedding("partecipanti", model)
embedding_b: list[float] = get_word_embedding("presenti", model)
print(embedding_a)

distance: float = cosine_distance(embedding_a, embedding_b)
print(f"The distance from \"partecipanti\" and \"presenti\" is: \"{distance:.18f}\"")

