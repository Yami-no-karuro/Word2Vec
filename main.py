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
import pickle
import re

def get_word_embedding(word: str) -> list[float] | None:
    if word in w2i:
        return w1[w2i[word]]

    return None

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

with open("source/sm.txt", "r", encoding = "utf-8") as file:
    input: str = file.read()

tokens: list[Token] = get_tokens(input)
identifiers: list[str] = get_identifiers(tokens)
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

embedding_dim: int = 1024
vocab_size: int = len(vocab)

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
#
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
#
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

learning_rate: float = 0.05
epochs: int = 10

print("Starting training iterations...")
for epoch in range(epochs):
    total_loss: float = 0.0

    for i in range(len(x)):
        target_idx: int = x[i] # The center word index (target)
        context_idx: int = y[i] # The surrounding word index (context)

        # 1. Gets embedding vector of the target word from w1.
        v: list[float] = w1[target_idx]

        # 2. Projects v into vocabulary space by computing dot product with w2.
        #    z[i] will be the unnormalized logit for word i.
        z: list[float] = []
        for i_vocab in range(vocab_size):
            dot: float = 0.0
            for d in range(embedding_dim):
                dot += w2[d][i_vocab] * v[d]
            z.append(dot)

        # 3. Applies softmax to get predicted probability distribution.
        y_pred: list[float] = softmax(z)

        # 4. Computes the error vector.
        #    The true label is a one-hot vector where only context_idx is 1.
        error: list[float] = [p for p in y_pred]
        error[context_idx] -= 1.0

        # 5. Update w2 (output projection matrix).
        #    Using gradient: dL/dw2[d][i_vocab] = error[i_vocab] * v[d].
        for d in range(embedding_dim):
            for i_vocab in range(vocab_size):
                gradient: float = error[i_vocab] * v[d]
                w2[d][i_vocab] -= learning_rate * gradient

        # 6. Update w1 (embedding matrix).
        #    Using gradient: dL/dw1[target_idx][d] = sum(error[i_vocab] * w2[d][i_vocab]).
        for d in range(embedding_dim):
            grad: float = 0.0
            for i_vocab in range(vocab_size):
                grad += error[i_vocab] * w2[d][i_vocab]

            w1[target_idx][d] -= learning_rate * grad

        # 7. Compute loss for monitoring.
        #    Cross-entropy loss for the predicted probability of the actual context word.
        loss: float = -math.log(y_pred[context_idx] + 1e-10)  # <- Avoids log(0)
        total_loss += loss

    print(f"[{epoch + 1}/{epochs}] - loss: {total_loss:.4f}")
print("Traning completed...")

