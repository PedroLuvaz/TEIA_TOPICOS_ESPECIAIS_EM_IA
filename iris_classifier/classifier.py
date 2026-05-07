from math_utils import calcular_media, discriminante, distancia_euclidiana

def treinar(dados_treino, indices_atributos):
    """
    Calcula o protótipo (vetor médio) de cada classe:
        m_j = (1/N_j) * sum(x)  para toda amostra x da classe j
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
    Classifica x usando a Função Discriminante:
        d_j(x) = x^T * m_j  -  0.5 * m_j^T * m_j
    Regra de decisão: argmax_j d_j(x)  (maior score = classe predita)

    Equivalente a argmin ||x - m_j|| (menor distância euclidiana),
    mas sem calcular a raiz quadrada.

    Retorna: dict {classe: score_discriminante}, classe_vencedora
    """
    x_sel = [x[i] for i in indices_atributos]

    scores = {classe: discriminante(x_sel, mj) for classe, mj in prototipos.items()}
    vencedor = max(scores, key=scores.get)

    return scores, vencedor


def predizer_binario(x, pi, pj, classe_i, classe_j, indices_atributos):
    """
    Classifica x entre duas classes via distância euclidiana direta:
        escolhe a classe cujo protótipo está mais próximo de x.
    Usado no Experimento iii (superfícies de decisão binárias).
    """
    x_sel = [x[i] for i in indices_atributos]
    return classe_i if distancia_euclidiana(x_sel, pi) < distancia_euclidiana(x_sel, pj) else classe_j
