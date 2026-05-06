import math

def produto_escalar(a, b):
    """Calcula o produto escalar de dois vetores."""
    return sum(x * y for x, y in zip(a, b))

def subtrair_vetores(a, b):
    """Calcula a subtração de dois vetores (a - b)."""
    return [x - y for x, y in zip(a, b)]

def multiplicar_escalar(s, v):
    """Multiplica um vetor v por um escalar s."""
    return [s * x for x in v]

def distancia_euclidiana(a, b):
    """Calcula a distância euclidiana entre dois vetores."""
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))

def calcular_media(vetores):
    """Calcula o vetor médio (protótipo) de uma lista de vetores."""
    if not vetores:
        return []
    n = len(vetores)
    dim = len(vetores[0])
    media = [0.0] * dim
    for v in vetores:
        for i in range(dim):
            media[i] += v[i]
    return [x / n for x in media]

def discriminante(x, mj):
    """
    Calcula dj(x) = x^t * mj - 0.5 * mj^t * mj
    Esta é a função discriminante para o Classificador de Distância Mínima.
    """
    termo1 = produto_escalar(x, mj)
    termo2 = 0.5 * produto_escalar(mj, mj)
    return termo1 - termo2

def coeficientes_superficie_decisao(mi, mj):
    """
    Calcula os coeficientes para a superfície de decisão dij(x) = 0
    dij(x) = (mi - mj)^t * x - 0.5 * (mi^t * mi - mj^t * mj) = 0
    Retorna: (w, b) onde w é o vetor de pesos e b é a constante de viés (bias).
    Equação: w * x + b = 0
    """
    w = subtrair_vetores(mi, mj)
    b = -0.5 * (produto_escalar(mi, mi) - produto_escalar(mj, mj))
    return w, b
