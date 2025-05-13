from dataclasses import dataclass

import re

@dataclass
class Token:
    type: str
    content: str

def get_tokens(input: str) -> list[Token]:
    words: list[str] = re.findall(
        r"\w+|[^\w\s]",
        input,
        re.UNICODE
    )

    tokens: list[Token] = []
    for index, word in enumerate(words):
        t_type: str = "Unknown"
        if word.isalpha():
            t_type = "Identifier"
        elif word.isdigit():
            t_type = "Number"
        elif re.match(r"\W", word):
            t_type = "Punctuation"

        tokens.append(Token(
            type = t_type, 
            content = word
        ))

    return tokens

@dataclass
class Sentence:
    length: int
    content: list[Token]

def get_sentences(input: str) -> list[Sentence]:
    tokens: list[Token] = get_tokens(input)
    sentences: list[Sentences] = []

    s_len: int = 0
    sentence: list[Token] = []
    for _, token in enumerate(tokens):
        sentence.append(token)
        s_len = s_len + 1

        if token.type == "Punctuation" and (
            token.content == "." or 
            token.content == "!" or 
            token.content == "?"
        ):
            sentences.append(Sentence(
                length = s_len,
                content = sentence 
            ))

            sentence = []
            s_len = 0
            continue
   
    return sentences

