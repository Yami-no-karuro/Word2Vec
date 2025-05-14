def build_vocabulary(identifiers: list[str]) -> list[str]:
    vocab: set[str] = set()
    for identifier in identifiers:
        vocab.add(identifier)

    vocab: list[str] = sorted(vocab)
    return vocab

def build_w2i_dict(vocabulary: list[str]) -> dict[str, int]:
    w2i: dict[str, int] = dict()
    for index in range(len(vocabulary)):
        word: str = vocabulary[index]
        w2i[word] = index

    return w2i

def build_i2w_dict(w2i: dict[str, int]) -> dict[int, str]:
    i2w: dict[int, str] = dict()
    for word, index in w2i.items():
        i2w[index] = word

    return i2w

def get_contiguous_pairs(identifiers: list[str]) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    for idx, _ in enumerate(identifiers):
        if idx > 0 and idx < len(identifiers) - 1:
            pairs.add((identifiers[idx - 1], identifiers[idx]))
            pairs.add((identifiers[idx], identifiers[idx + 1]))

    return pairs

