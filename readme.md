# Word2Vec (Skip Gram)

## Word2Vec model, in Python

### Intro

**Word2Vec** is a technique in **Natural Language Processing** (NLP) for obtaining **Vector** representations of words.  
These vectors capture information about the meaning of the word based on the surrounding words.  
The **Word2Vec** algorithm estimates these representations by modeling text in a large corpus.  
Once trained, such a model can detect synonymous words or suggest additional words for a partial sentence.  
(More on [Word2Vec](https://en.wikipedia.org/wiki/Word2vec), [NPL](https://en.wikipedia.org/wiki/Natural_language_processing) and [Vectors](https://en.wikipedia.org/wiki/Vector_space) on [Wikipedia](https://en.wikipedia.org/))

### What is this example about?

This project is a **from-scratch** implementation of the **Skip-Gram** variant of the **Word2Vec** model using pure Python.  
The objective is to learn word **embeddings—dense vector representations** by scanning a corpus of text and predicting the surrounding context words given a target word.  
It is designed for **educational purposes** and focuses on clarity and simplicity, rather than computational efficiency or performance.  
The training follows these key steps:

- Tokenization of the input text.
- Extraction of identifiers (words).
- Building a vocabulary of unique tokens.
- Encoding pairs of contiguous words as input/output training samples.
- Initializing the embedding and projection matrices.
- Training the model using a simple form of gradient descent.
- Saving the resulting model to disk.

### Vocabulary

Each unique word is assigned an integer ID:
- `w2i`: maps word → index
- `i2w`: maps index → word

### Training Data

The model looks at pairs of **contiguous words** (e.g., `("il", "romanzo")`) to learn meaningful relationships.  
These pairs are encoded as integer indices:

- `x`: list of target word indices
- `y`: list of context word indices

### Model Structure

- `w1`: Embedding matrix of shape `[vocab_size][embedding_dim]`
- `w2`: Output projection matrix of shape `[embedding_dim][vocab_size]`

The model computes:

- `v` = embedding of target word → `w1[target_idx]`
- `z` = dot product `w2^T · v` → unnormalized logits
- `y_pred` = softmax(z) → predicted probabilities

Weights are updated with gradient descent and loss is tracked over time.

### Example

Given a sentence like:

    "Il romanzo ha inizio con la festa..."

The model will extract pairs like:

    ("il", "romanzo"), ("romanzo", "ha"), ("ha", "inizio"), ...

These will be used to train the model to associate the embedding of `"il"` with the context `"romanzo"`, and so on.
