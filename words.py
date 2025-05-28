from lib.utils import load_model
from lib.math import cosine_distance
from lib.embedder import get_word_embedding

model: dict = load_model("models/model.pkl")

print("=== Words semantic distance ===")
print("(Type \"exit\" or \"CTRL + C\" to stop the program)")

while True:
    w1 = input("Word A: ").strip()
    if w1.lower() == "exit":
        break

    w2 = input("Word B: ").strip()
    if w2.lower() == "exit":
        break

    e1 = get_word_embedding(w1, model)
    e2 = get_word_embedding(w2, model)
    d = cosine_distance(e1, e2)
    print(f'"{w1}" ↔ "{w2}": distance = {d:.18f}\n')

