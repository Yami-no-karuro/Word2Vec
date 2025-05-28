from lib.utils import load_model
from lib.math import cosine_distance
from lib.embedder import get_sentence_embedding 

model: dict = load_model("models/model.pkl")

print("=== Sentence Similarity ===")
print("(Type \"exit\" or \"CTRL + C\" to stop the program)")

while True:
    s1 = input("Sentence A: ").strip()
    if s1.lower() == "exit":
        break

    s2 = input("Sentence B: ").strip()
    if s2.lower() == "exit":
        break

    e1 = get_sentence_embedding(s1, model)
    e2 = get_sentence_embedding(s2, model)
    d = cosine_distance(e1, e2)
    print(f'"{s1}" ↔ "{s2}": distance = {d:.18f}\n')

