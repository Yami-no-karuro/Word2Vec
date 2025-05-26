from lib.utils import load_model

from lib.embedder import get_word_embedding
from lib.embedder import get_sentence_embedding

print("Loading model...")
model: dict = load_model("models/model.pkl")

word_embedding: list[float] = get_word_embedding("Romanzo", model)
print(f"Word embedding: \"{word_embedding}\"")

sentence_embedding: list[float] = get_sentence_embedding("Lo stregone comincia a sospettare della natura dell'anello.", model)
print(f"Sentence embedding: \"{sentence_embedding}\"")

