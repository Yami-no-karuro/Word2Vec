# Word2Vec (Skip Gram)

## Word2Vec model, in Python

### Intro

**Word2Vec** is a technique in **Natural Language Processing** (NLP) for obtaining **Vector** representations of words.  
These vectors capture information about the meaning of the word based on the surrounding words.  
The **Word2Vec** algorithm estimates these representations by modeling text in a large corpus.  
Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence.  
(More on [Word2Vec](https://en.wikipedia.org/wiki/Word2vec), [NPL](https://en.wikipedia.org/wiki/Natural_language_processing) and [Vectors](https://en.wikipedia.org/wiki/Vector_space) on [Wikipedia](https://en.wikipedia.org/))

### 1. Vocabulary, Word2Index and Index2Word

This phase of the training process allows to map words to numerical indexes.  
The text goes through a tokenization process, then the identifiers are extracted and organized.

The first piece of information that we need is the **vocabulary**.  
The **vocabulary** contains every word in the corpus.

```python
def build_vocabulary(identifiers: list[str]) -> list[str]:
    vocab: set[str] = set()
    for identifier in identifiers:
        vocab.add(identifier)

    vocab: list[str] = sorted(vocab)
    return vocab
```

After the vocabulary we need to create the **Word2Index** dictionary and his reverse, to map words to numerical indexes.

```python
# Word2Index
def build_w2i_dict(vocabulary: list[str]) -> dict[str, int]:
    w2i: dict[str, int] = dict()
    for index in range(len(vocabulary)):
        word: str = vocabulary[index]
        w2i[word] = index

    return w2i

# Index2Words 
def build_i2w_dict(w2i: dict[str, int]) -> dict[int, str]:
    i2w: dict[int, str] = dict()
    for word, index in w2i.items():
        i2w[index] = word

    return i2w
```

### 2. Contiguous Pairs

This process extracts **contiguous words** in order to identify **recurring semantic patterns**.  
The pairs are than converted in numerical indexes as **x -> target**, **y -> context**.

```python
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
```

### 3. Model Parameters

At this point, model parameters such as the **embedding size**, should be defined.  
In this example we'll be using 1024 as **embedding size**.

### 4. Embedding Matrix

The **embedding matrix** or **w1** is a `[vocab_size][embedding_dim]` matrix used to retrieve the embedding of a given input word (target).  
Each row `w1[i]` represents the embedding vector of the word with index i in the vocabulary.

```python
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
```

