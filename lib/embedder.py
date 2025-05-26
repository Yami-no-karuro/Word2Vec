def mean_embedding(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []

    num_vectors: int = len(vectors)
    vector_len: int = len(vectors[0])
    summed: list[float] = [0.0] * vector_len

    for vec in vectors:
        for i in range(vector_len):
            summed[i] += vec[i]

    mean: list[float] = []
    for value in summed:
        mean.append(value / num_vectors)

    return mean

def get_word_embedding(word: str, model: dict) -> list[float]:
    word: str = word.lower()
    w1: list[list[float]] = model["w1"]
    w2i: dict[str, int] = model["w2i"]

    idx: int = w2i.get(word, w2i.get("<unknown>"))
    return w1[idx]

def get_sentence_embedding(sentence: str, model: dict) -> list[float]:
    words: list[str] = sentence.lower().split()
    embeddings: list[list[float]] = []

    for word in words:
        embedding: list[float] = get_word_embedding(word, model)
        embeddings.append(embedding)

    return mean_embedding(embeddings)

