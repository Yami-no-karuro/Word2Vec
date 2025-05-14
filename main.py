from dataclasses import dataclass

from lib.math import softmax

from lib.tokenizer import Token
from lib.tokenizer import get_tokens
from lib.tokenizer import get_identifiers

from lib.lex import build_vocabulary
from lib.lex import build_w2i_dict
from lib.lex import build_i2w_dict
from lib.lex import get_contiguous_pairs

import math
import random
import re

def get_word_embedding(word: str) -> list[float] | None:
    if word in w2i:
        return w1[w2i[word]]

    return None

# ====
# __MAIN__
# ====

with open("source/corpus.txt", "r", encoding = "utf-8") as file:
    input: str = file.read()

# Dev example...
input = "Il romanzo ha inizio con la festa del 111º compleanno di Bilbo e del 33° di suo nipote Frodo."

# Builds the vocabulary through input tokenization.
# Only the "Identifier" tokens are actually usefull, so the token list is further filtered before usage.
tokens: list[Token] = get_tokens(input)
identifiers: list[str] = get_identifiers(tokens)
vocab: list[str] = build_vocabulary(identifiers)

# Builds the "Word2Index" and "Index2Word" dictionaries
# The "Word2Index" dictionary maps every identifier in the vocabulary to the specific index.
# The "Index2Word" dictionary is basically "Word2Index" reverse.
w2i: dict[str, int] = build_w2i_dict(vocab)
i2w: dict[int, str] = build_i2w_dict(w2i)

# Builds the cotiguous pairs set.
# For example... "My name is Carlo": [("My", "name") ("name", "is"), ("is", "Carlo")]
pairs: set[tuple[str, str]] = get_contiguous_pairs(identifiers)

x: list[int] = []
y: list[int] = []
for target, context in pairs:
    if target in w2i and context in w2i:
        t_idx: int = w2i[target]
        x.append(t_idx)

    if context in w2i:
        c_idx: int = w2i[context]
        y.append(c_idx)

embedding_dim: int = 1024
vocab_size: int = len(vocab)

w1: list[list[float]] = []
for word_index in range(vocab_size):
    mbd_vector: list[float] = []

    for dimension in range(embedding_dim):
        random_value: float = random.uniform(-0.01, 0.01)
        mbd_vector.append(random_value)

    w1.append(mbd_vector)

w2: list[list[float]] = []
for dimension in range(embedding_dim):
    out_vector: list[float] = []

    for word_index in range(vocab_size):
        random_value: float = random.uniform(-0.01, 0.01)
        out_vector.append(random_value)

    w2.append(out_vector)

# ====
# Learning
# ====

learning_rate: float = 0.05
epochs: int = 10

for epoch in range(epochs):
    total_loss: float = 0.0

    for i in range(len(x)):
        target_idx: int = x[i]
        context_idx: int = y[i]

        v: list[float] = w1[target_idx]
        z: list[float] = []

        for i_vocab in range(vocab_size):
            dot: float = 0.0
            for d in range(embedding_dim):
                dot += w2[d][i_vocab] * v[d]

            z.append(dot)

        y_pred: list[float] = softmax(z)

        error: list[float] = [p for p in y_pred]
        error[context_idx] -= 1.0

        for d in range(embedding_dim):
            for i_vocab in range(vocab_size):
                gradient: float = error[i_vocab] * v[d]
                w2[d][i_vocab] -= learning_rate * gradient

        for d in range(embedding_dim):
            grad: float = 0.0
            for i_vocab in range(vocab_size):
                grad += error[i_vocab] * w2[d][i_vocab]

            w1[target_idx][d] -= learning_rate * grad

        loss: float = -math.log(y_pred[context_idx] + 1e-10)
        total_loss += loss

    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")

result = get_word_embedding("frodo")
print(result)

