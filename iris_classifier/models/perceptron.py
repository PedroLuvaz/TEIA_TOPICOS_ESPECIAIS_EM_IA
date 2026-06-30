"""
Perceptron de Rosenblatt — Python puro.

Regra de aprendizado (classificacao binaria):
    Ativacao limiar: y = +1  se  w^T · x_aug >= 0,  senao  y = -1
    Atualizacao (apenas quando erra):
        w  <-  w + p · (d - y) · x_aug       [x_aug = [1, x1, x2, ...]]

Convergencia garantida SOMENTE para dados linearmente separaveis
(Teorema da Convergencia do Perceptron, Rosenblatt 1957).
"""


def _sgn(net):
    """Ativacao limiar: retorna +1 se net >= 0, senao -1."""
    return 1 if net >= 0 else -1


def treinar_perceptron(dados_treino, classe_pos, classe_neg,
                       indices_atributos, taxa_aprendizado=0.03,
                       max_epocas=100):
    """
    Treina o Perceptron binario para um par de classes.

    Parametros
    ----------
    dados_treino      : lista de {'atributos': [...], 'classe': str}
    classe_pos        : classe mapeada para d = +1
    classe_neg        : classe mapeada para d = -1
    indices_atributos : lista de indices das features a usar (ex: [2, 3])
    taxa_aprendizado  : p — taxa de aprendizado (padrao 0.03)
    max_epocas        : limite de iteracoes (padrao 100)

    Retorna
    -------
    w                : list — pesos [w0(bias), w1, ..., wn]
    historico_erros  : list — numero de erros por epoca
    epocas_treinadas : int  — epocas ate convergir (ou max_epocas se nao convergiu)
    """
    # Montar conjunto binario com bias na posicao 0
    amostras = []
    for d in dados_treino:
        if d['classe'] == classe_pos:
            x_sel = [d['atributos'][i] for i in indices_atributos]
            amostras.append(([1.0] + x_sel, +1))
        elif d['classe'] == classe_neg:
            x_sel = [d['atributos'][i] for i in indices_atributos]
            amostras.append(([1.0] + x_sel, -1))

    n_pesos = 1 + len(indices_atributos)   # bias + n features
    w = [0.0] * n_pesos
    historico_erros = []

    for _ in range(max_epocas):
        n_erros = 0
        for x_aug, d_val in amostras:
            # Ativacao: net = w^T * x_aug
            net = sum(wi * xi for wi, xi in zip(w, x_aug))
            y = _sgn(net)
            if y != d_val:
                n_erros += 1
                # Atualizacao: w <- w + p*(d-y)*x_aug
                delta = taxa_aprendizado * (d_val - y)
                w = [wi + delta * xi for wi, xi in zip(w, x_aug)]

        historico_erros.append(n_erros)
        if n_erros == 0:
            # Convergiu: todos os pontos classificados corretamente
            break

    return w, historico_erros, len(historico_erros)


def predizer_perceptron(x_atributos, w):
    """
    Classifica uma amostra com os pesos treinados.

    Parametros
    ----------
    x_atributos : features ja selecionadas pelos indices (sem o bias)
    w           : vetor de pesos [w0, w1, ..., wn]

    Retorna
    -------
    +1 ou -1
    """
    x_aug = [1.0] + list(x_atributos)
    net = sum(wi * xi for wi, xi in zip(w, x_aug))
    return _sgn(net)


def acuracia_binaria_perceptron(dados, w, classe_pos, classe_neg,
                                indices_atributos):
    """
    Calcula a acuracia binaria do Perceptron num conjunto de dados.

    Retorna: float no intervalo [0.0, 1.0]
    """
    corretos = total = 0
    for d in dados:
        if d['classe'] not in (classe_pos, classe_neg):
            continue
        x_sel = [d['atributos'][i] for i in indices_atributos]
        y = predizer_perceptron(x_sel, w)
        pred = classe_pos if y == 1 else classe_neg
        if pred == d['classe']:
            corretos += 1
        total += 1
    return corretos / total if total > 0 else 0.0
