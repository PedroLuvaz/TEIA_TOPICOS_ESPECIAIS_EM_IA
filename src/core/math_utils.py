"""
Algebra linear em Python puro para o Classificador de Distancia Minima.

Todas as operacoes sao implementadas com listas nativas e lacos for —
sem numpy, scipy ou qualquer biblioteca de algebra.

Formulas centrais:
  Prototipo:      m_j = (1/N_j) * sum(x  para x em omega_j)
  Discriminante:  d_j(x) = x^T * m_j  -  (1/2) * m_j^T * m_j
  Fronteira:      w = m_i - m_j,   b = -(1/2)*(||m_i||^2 - ||m_j||^2)
"""

import math


def produto_escalar(a, b):
    """
    a^T * b = sum( a_i * b_i )
    Operacao base para o discriminante e para o calculo do bias da fronteira.
    """
    return sum(x * y for x, y in zip(a, b))


def subtrair_vetores(a, b):
    """
    a - b  (componente a componente)
    Usado para calcular w = m_i - m_j (vetor normal a fronteira de decisao).
    """
    return [x - y for x, y in zip(a, b)]


def multiplicar_escalar(s, v):
    """s * v  (escalar por vetor)"""
    return [s * x for x in v]


def distancia_euclidiana(a, b):
    """
    ||a - b|| = sqrt( sum( (a_i - b_i)^2 ) )
    Regra de decisao: argmin_j ||x - m_j||  (menor distancia = classe predita).
    Matematicamente equivalente a argmax_j d_j(x).
    """
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def calcular_media(vetores):
    """
    m = (1/N) * sum(v  para v em vetores)
    Calcula o prototipo (centroide) de uma classe a partir de suas amostras de treino.
    """
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
    Funcao Discriminante Linear:
        d_j(x) = x^T * m_j  -  (1/2) * m_j^T * m_j

    Derivacao: minimizar ||x - m_j||^2 expande para x^Tx - 2*x^T*m_j + m_j^T*m_j.
    Como x^Tx e constante (nao depende de j), minimizar a distancia equivale a
    maximizar  x^T*m_j - (1/2)*m_j^T*m_j  =  d_j(x).

    Regra de decisao: argmax_j d_j(x)  <==>  argmin_j ||x - m_j||
    """
    return produto_escalar(x, mj) - 0.5 * produto_escalar(mj, mj)


def coeficientes_superficie_decisao(mi, mj):
    """
    Fronteira entre classes i e j: d_i(x) = d_j(x)  =>  w^T * x + b = 0

    Derivacao:
        d_i(x) - d_j(x) = 0
        (m_i - m_j)^T * x  -  (1/2)*(||m_i||^2 - ||m_j||^2)  =  0

    Coeficientes:
        w = m_i - m_j
        b = -(1/2) * (||m_i||^2 - ||m_j||^2)

    Para plotar a reta em 2D: x2 = (-w1*x1 - b) / w2

    Retorna: (w, b)
    """
    w = subtrair_vetores(mi, mj)
    b = -0.5 * (produto_escalar(mi, mi) - produto_escalar(mj, mj))
    return w, b
