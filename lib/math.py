import math

def softmax(z: list[float]) -> list[float]:
    max_z: float = max(z)
    exp_z: list[float] = []
    for value in z:
        exp_value: float = math.exp(value - max_z)
        exp_z.append(exp_value)

    sum_exp: float = 0.0
    for value in exp_z:
        sum_exp += value

    result: list[float] = []
    for value in exp_z:
        softmax_value: float = value / sum_exp
        result.append(softmax_value)

    return result

def dot_product(vec1: list[float], vec2: list[float]) -> float:
    result: float = 0.0
    i: int = 0

    while i < len(vec1):
        result = result + (vec1[i] * vec2[i])
        i = i + 1

    return result

def vector_norm(vec: list[float]) -> float:
    sum_of_squares: float = 0.0
    i: int = 0

    while i < len(vec):
        sum_of_squares = sum_of_squares + (vec[i] * vec[i])
        i = i + 1

    x: float = sum_of_squares
    tolerance: float = 1e-10
    guess: float = x
    
    while True:
        new_guess: float = 0.5 * (guess + x / guess)
        if abs(new_guess - guess) < tolerance:
            break
        guess = new_guess

    return guess

def cosine_distance(vec1: list[float], vec2: list[float]) -> float:
    dp: float = dot_product(vec1, vec2)
    norm1: float = vector_norm(vec1)
    norm2: float = vector_norm(vec2)

    if norm1 == 0.0 or norm2 == 0.0:
        return 1.0

    cosine_similarity: float = dp / (norm1 * norm2)
    distance: float = 1.0 - cosine_similarity
    return distance

