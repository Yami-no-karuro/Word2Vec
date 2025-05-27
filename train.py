from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

from lib.math import softmax
from lib.utils import dump_model

from lib.tokenizer import Token
from lib.tokenizer import get_tokens
from lib.tokenizer import get_identifiers

from lib.lex import build_vocabulary
from lib.lex import build_w2i_dict
from lib.lex import build_i2w_dict
from lib.lex import get_contiguous_pairs

import threading
import math
import random
import pickle
import re

# ====
# Training Data
# ====

# 1. Vocabulary and dictionary.
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

with open("source/corpus.txt", "r", encoding = "utf-8") as file:
    input: str = file.read()

tokens: list[Token] = get_tokens(input)
identifiers: list[str] = get_identifiers(tokens)
identifiers.append("<unknown>")

vocab: list[str] = build_vocabulary(identifiers)
w2i: dict[str, int] = build_w2i_dict(vocab)
i2w: dict[int, str] = build_i2w_dict(w2i)

# 2. Contiguous pairs.
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

vocab_size: int = len(vocab)
embedding_dim: int = 768
learning_rate: float = 0.05
epochs: int = 50

# 4. Embedding matrix.
# The "Embedding Matrix" or "w1" is a [vocab_size][embedding_dim] matrix used to retrieve the embedding of a given input word (target).
# Each row w1[i] represents the embedding vector of the word with index i in the vocabulary.
#
# Think of it as:
#
#        embedding_dim →
#      ┌─────────────────────┐
#      │ "il"      → [ ... ] │
#      │ "romanzo" → [ ... ] │
#      │ "ha"      → [ ... ] │
#      │ "inizio"  → [ ... ] │
#  ↓   │   ...               │
# vocab_size

w1: list[list[float]] = []
for word_index in range(vocab_size):
    mbd_vector: list[float] = []

    for dimension in range(embedding_dim):
        random_value: float = random.uniform(-0.01, 0.01)
        mbd_vector.append(random_value)

    w1.append(mbd_vector)

# 5. Output Projection Matrix.
# The "Output Projection Matrix" is a [embedding_dim][vocab_size] matrix used to transform the embedding vector into a probability distribution over all words.
# Each column w2[:,i] represents the output vector associated with the word at index i.
# It’s used to project the input embedding into a distribution over the vocabulary via softmax.
#
# Think of it as:
#
#             vocab_size →
#           "il" "romanzo"  "ha" ...
#        ┌──────────────────────────┐
#   dim0 │ w2[0][0] w2[0][1] ...    │
#   dim1 │ w2[1][0] w2[1][1] ...    │
#   ...  │         ...              │
#        └──────────────────────────┘
#       ↑
#  embedding_dim

w2: list[list[float]] = []
for dimension in range(embedding_dim):
    out_vector: list[float] = []

    for word_index in range(vocab_size):
        random_value: float = random.uniform(-0.01, 0.01)
        out_vector.append(random_value)

    w2.append(out_vector)

# ====
# Model Training 
# ====

# For each (target, context) word pair:
# 1. Retrieves the embedding vector of the target word from w1.
# 2. Projects this vector to the output space using w2.
# 3. Computes the softmax distribution (predicted context probabilities).
# 4. Calculates the error between prediction and actual context word.
# 5. Updates both w2 (output projection) and w1 (embedding) using gradient descent.
# 6. Accumulates the cross-entropy loss for reporting.

lock = threading.Lock()

def train_pair(i: int) -> float:
    target_idx: int = x[i]
    context_idx: int = y[i]

    # 1. Gets the embedding vector of the target word from w1.
    v: list[float] = w1[target_idx]

    # 2. Projects v into vocabulary space by computing dot product with w2.
    z: list[float] = []
    for i_vocab in range(vocab_size):
        dot: float = 0.0
        for d in range(embedding_dim):
            dot += w2[d][i_vocab] * v[d]
        z.append(dot)

    # 3. Applies softmax to get predicted probability distribution.
    y_pred: list[float] = softmax(z)

    # 4. Computes the error vector.
    error: list[float] = [p for p in y_pred]
    error[context_idx] -= 1.0

    # 5. Updates w2 e w1 with lock to avoid race conditions
    with lock:
        for d in range(embedding_dim):
            for i_vocab in range(vocab_size):
                gradient: float = error[i_vocab] * v[d]
                w2[d][i_vocab] -= learning_rate * gradient

        for d in range(embedding_dim):
            grad: float = 0.0
            for i_vocab in range(vocab_size):
                grad += error[i_vocab] * w2[d][i_vocab]

            w1[target_idx][d] -= learning_rate * grad

    # 6. Computes loss for monitoring.
    loss: float = -math.log(y_pred[context_idx] + 1e-10)
    return loss

print("Starting training iterations...")
for epoch in range(epochs):
    total_loss: float = 0.0

    with ThreadPoolExecutor(max_workers = 8) as executor:
        losses = list(executor.map(train_pair, range(len(x))))

    total_loss = sum(losses)
    print(f"[{epoch + 1}/{epochs}] - loss: {total_loss:.4f}")

print("Traning completed...")
dump_model("models/model.pkl", {
    "w1": w1,
    "w2": w2,
    "w2i": w2i,
    "i2w": i2w,
    "vocab": vocab
})

