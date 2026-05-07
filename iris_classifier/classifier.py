"""
Logica de treinamento e classificacao do Classificador de Distancia Minima.

TREINAMENTO — calcula os prototipos (vetores medios) de cada classe:
    m_j = (1/N_j) * sum(x  para x em omega_j)

CLASSIFICACAO MULTICLASSE — Funcao Discriminante (argmax):
    d_j(x) = x^T * m_j  -  (1/2) * m_j^T * m_j
    classe* = argmax_j  d_j(x)   <==>   argmin_j  ||x - m_j||

CLASSIFICACAO BINARIA — Distancia Euclidiana (argmin):
    classe* = argmin_{i,j}  ||x - m_{i,j}||
"""

from math_utils import calcular_media, discriminante, distancia_euclidiana


def treinar(dados_treino, indices_atributos):
    """
    Fase de treinamento: calcula m_j = (1/N_j) * sum(x) para cada classe j.
    Retorna: dict {classe: vetor_prototipo}
    """
    prototipos = {}
    classes = sorted(list(set(d['classe'] for d in dados_treino)))

    for classe in classes:
        amostras = [d['atributos'] for d in dados_treino if d['classe'] == classe]
        atributos_sel = [[s[i] for i in indices_atributos] for s in amostras]
        prototipos[classe] = calcular_media(atributos_sel)

    return prototipos


def predizer_todas_classes(x, prototipos, indices_atributos):
    """
    Classificacao multiclasse pela Funcao Discriminante:
        d_j(x) = x^T * m_j  -  (1/2) * m_j^T * m_j
        classe* = argmax_j  d_j(x)

    Matematicamente equivalente a argmin_j ||x - m_j||, sem calcular raiz quadrada.
    Retorna: (dict {classe: score}, classe_vencedora)
    """
    x_sel = [x[i] for i in indices_atributos]
    scores = {classe: discriminante(x_sel, mj) for classe, mj in prototipos.items()}
    vencedor = max(scores, key=scores.get)
    return scores, vencedor


def predizer_binario(x, pi, pj, classe_i, classe_j, indices_atributos):
    """
    Classificacao binaria pela distancia euclidiana:
        classe* = argmin(||x - pi||, ||x - pj||)

    Usado no Experimento iii para calcular acuracia por par de classes.
    """
    x_sel = [x[i] for i in indices_atributos]
    return classe_i if distancia_euclidiana(x_sel, pi) < distancia_euclidiana(x_sel, pj) else classe_j
