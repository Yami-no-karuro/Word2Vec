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

# 1. Vocabulary and dictionary building.
# This process allows to map words to numerical indexes.
# The "Word2Idex" dictionary specifically maps word to integers and the "Index2Words" is the exact opposite.
#
# Example...
# - input:         "Il romanzo ha inizio con la festa..."
# - tokens:        [Token(type: "Identifier", content: "Il") ...]
# - identifiers:   ["il", "romanzo", "ha", "inizio"...]
# - vocab:         ["il", "romanzo", "ha", "inizio"...] <- Only uniques values
# - w2i:           {("il": 0), ("romanzo": 1), ...}
# - i2w:           {(0: "in"), (1: "romanzo"), ...}

with open("source/sm.txt", "r", encoding = "utf-8") as file:
    input: str = file.read()

tokens: list[Token] = get_tokens(input)
identifiers: list[str] = get_identifiers(tokens)
vocab: list[str] = build_vocabulary(identifiers)

w2i: dict[str, int] = build_w2i_dict(vocab)
i2w: dict[int, str] = build_i2w_dict(w2i)

# 2. Contiguous pairs retrival.
# This process extracts contiguous words in order to identify recurring semantic patterns.
# The pairs are than converted in numerical indexes as "x -> target", "y -> context".
#
# Example...
# - pairs: [("il", "romanzo"), ("romanzo", "ha"), ...] <- Only uniques values
# - x:     [0, 2, 19, ...] 
# - x:     [4, 7, 51, ...] 

x: list[int] = []
y: list[int] = []
pairs: set[tuple[str, str]] = get_contiguous_pairs(identifiers)

for target, context in pairs:
    if target in w2i and context in w2i:
        t_idx: int = w2i[target]
        x.append(t_idx)

    if context in w2i:
        c_idx: int = w2i[context]
        y.append(c_idx)

# 3. Model parameters.
# The embedding size and the vocabulary size are configured.

embedding_dim: int = 1024
vocab_size: int = len(vocab)

# 4. Embedding matrix and projection matrix.
# The "Embedding Matrix" or w1 is a "vocab_size * embedding_dim" matrix where every row represents a specific word embedding.
# The "Projection Matrix" or w2 is a "vocab_size * embedding_dim" matrix where every column represents a weigth associated to a specific vocabulary word.

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

