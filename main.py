from lib.utils import load_model

def get_word_embedding(word: str, model: dict) -> list[float] | None:
    if word in model["w2i"]:
        return model["w1"][model["w2i"][word]]

    return None

print("Loading model...")
model: dict = load_model("models/model.pkl")

# ====
# Example...
# ====

result: list[float] = get_word_embedding("frodo", model)
print(result)

