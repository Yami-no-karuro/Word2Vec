import math

def softmax(z: list[float]) -> list[float]:
    max_z: float = max(z)
    exp_z: list[float] = []
    for value in z:
        exp_value = math.exp(value - max_z)
        exp_z.append(exp_value)

    sum_exp: float = 0.0
    for value in exp_z:
        sum_exp += value

    result: list[float] = []
    for value in exp_z:
        softmax_value = value / sum_exp
        result.append(softmax_value)

    return result

