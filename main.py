from lib.utils import load_model

def get_word_embedding(word: str, model: dict) -> list[float]:
    word: str = word.lower()
    w1: list[list[float]] = model["w1"]
    w2i: dict[str, int] = model["w2i"]

    idx: int = w2i.get(word, w2i.get("<unknown>"))
    return w1[idx]

print("Loading model...")
model: dict = load_model("models/model.pkl")

# ====
# Example...
# ====

result: list[float] = get_word_embedding("supercalifragilisticexpialidocious", model)
print(result)

