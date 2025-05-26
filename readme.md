# Word2Vec (Skip Gram)

## Word2Vec model, in Python

### Intro

**Word2Vec** is a technique in **Natural Language Processing** (NLP) for obtaining **Vector** representations of words.  
These vectors capture information about the meaning of the word based on the surrounding words.  
The **Word2Vec** algorithm estimates these representations by modeling text in a large corpus.  
Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence.  
(More on [Word2Vec](https://en.wikipedia.org/wiki/Word2vec), [NPL](https://en.wikipedia.org/wiki/Natural_language_processing) and [Vectors](https://en.wikipedia.org/wiki/Vector_space) on [Wikipedia](https://en.wikipedia.org/))

### Vocabulary, Word2Index and Index2Word

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

