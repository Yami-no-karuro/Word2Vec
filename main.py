from tokenizer import Token, get_tokens

def get_identifiers(tokens: list[Token]) -> list[str]:
    identifiers: list[str] = []
    for token in tokens:
        if token.type == "Identifier":
            word: str = token.content.lower()
            identifiers.append(word)

    return identifiers

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

# ====
# __MAIN__
# ====

input: str = """Il romanzo ha inizio con la festa del 111º compleanno di Bilbo e del 33° di suo nipote Frodo. 
    Alla fine della festa, Bilbo comunica a tutti i presenti che intende lasciare la Contea per sempre e, dopo essersi infilato l'anello, sparisce, ma viene raggiunto a Casa Baggins da Gandalf, che riesce a convincerlo a lasciare l'anello a Frodo. 
    Lo stregone comincia a sospettare della natura dell'anello, perciò consiglia a Frodo di non adoperarlo mai e si allontana da Casa Baggins alla ricerca della verità."""

tokens: list[Token] = get_tokens(input)
identifiers: list[str] = get_identifiers(tokens)
vocab: list[str] = build_vocabulary(identifiers)

w2i: dict[str, int] = build_w2i_dict(vocab)
i2w: dict[int, str] = build_i2w_dict(w2i)

pairs: set[tuple[str, str]] = get_contiguous_pairs(identifiers)

X: list[int] = []
Y: list[int] = []

for target, context in pairs:
    if target in w2i and context in w2i:
        t_idx: int = w2i[target]
        X.append(t_idx)

    if context in w2i:
        c_idx: int = w2i[context]
        Y.append(c_idx)

print("X (Target):", X)
print("Y (Context):", Y)

