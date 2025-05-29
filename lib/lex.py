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

def build_frequency_map(identifiers: list[str]) -> dict[str, int]:
    frequency: dict[str, int] = {}
    for identifier in identifiers:
        if identifier not in frequency:
            frequency[identifier] = 1
        else:
            frequency[identifier] += 1

    return frequency

def get_contiguous_pairs(identifiers: list[str], window_size: int = 2) -> set[tuple[str, str]]:
    pairs: set[tuple[str, str]] = set()
    length: int = len(identifiers)

    for idx in range(length):
        target: str = identifiers[idx]
        for offset in range(1, window_size + 1):
            left_idx: int = idx - offset
            right_idx: int = idx + offset

            if left_idx >= 0:
                pairs.add((target, identifiers[left_idx]))
            if right_idx < length:
                pairs.add((target, identifiers[right_idx]))

    return pairs

