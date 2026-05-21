"""
Regra Delta (Widrow-Hoff / Adaline) — Python puro.

Diferenca fundamental em relacao ao Perceptron:
    A atualizacao usa a saida LINEAR  net = w^T · x  (nao o limiar sgn).
    Isso minimiza o Erro Quadratico Medio (MSE) mesmo quando os dados
    NAO sao linearmente separaveis.

Regra de atualizacao (modo online/estocastico, por amostra):
    net   = w^T · x_aug
    erro  = d - net            (saida linear, nao limiarizada)
    w    <- w + p · erro · x_aug

Criterio de parada: numero maximo de epocas.
Para dados nao separaveis (ex.: XOR), o MSE converge ao minimo
possivel, mas nunca a zero — prova da limitacao dos classificadores
lineares.
"""


def _treinar_delta(amostras, n_pesos, max_epocas, taxa_aprendizado):
    """
    Nucleo do algoritmo: aplica a Regra Delta por max_epocas.

    Retorna (w, historico_mse) onde historico_mse[k] = MSE medio na epoca k+1.
    O MSE e calculado ANTES das atualizacoes da epoca (valor de referencia).
    """
    w = [0.0] * n_pesos
    historico_mse = []

    for _ in range(max_epocas):
        soma_eq = 0.0
        for x_aug, d_val in amostras:
            net = sum(wi * xi for wi, xi in zip(w, x_aug))
            erro = d_val - net
            soma_eq += erro ** 2
            # Atualizacao online: w <- w + p * erro * x_aug
            w = [wi + taxa_aprendizado * erro * xi
                 for wi, xi in zip(w, x_aug)]

        # MSE medio da epoca (acumula erros apos cada atualizacao)
        historico_mse.append(soma_eq / len(amostras))

    return w, historico_mse


def treinar_delta_iris(dados_treino, classe_pos, classe_neg,
                       indices_atributos, taxa_aprendizado=0.02,
                       max_epocas=100):
    """
    Treina a Regra Delta para um par de classes do Iris.
    Mapeamento: classe_pos -> d = +1,  classe_neg -> d = -1.

    Parametros
    ----------
    dados_treino      : lista de {'atributos': [...], 'classe': str}
    classe_pos        : classe mapeada para d = +1
    classe_neg        : classe mapeada para d = -1
    indices_atributos : lista de indices das features (ex: [2, 3])
    taxa_aprendizado  : p (padrao 0.02)
    max_epocas        : numero de epocas de treinamento

    Retorna
    -------
    w               : list — pesos treinados [w0, w1, ..., wn]
    historico_mse   : list — MSE por epoca
    epocas_treinadas: int  — sempre max_epocas (sem parada antecipada)
    """
    amostras = []
    for d in dados_treino:
        if d['classe'] == classe_pos:
            x_sel = [d['atributos'][i] for i in indices_atributos]
            amostras.append(([1.0] + x_sel, +1.0))
        elif d['classe'] == classe_neg:
            x_sel = [d['atributos'][i] for i in indices_atributos]
            amostras.append(([1.0] + x_sel, -1.0))

    n_pesos = 1 + len(indices_atributos)
    w, historico_mse = _treinar_delta(
        amostras, n_pesos, max_epocas, taxa_aprendizado)
    return w, historico_mse, max_epocas


def treinar_delta_xor(max_epocas=200, taxa_aprendizado=0.02):
    """
    Treina a Regra Delta no problema XOR (nao linearmente separavel).

    Padroes de entrada (com bias):
        x_aug = [1, x1, x2]
        (0, 0) -> d = 0    (0, 1) -> d = 1
        (1, 0) -> d = 1    (1, 1) -> d = 0

    Nota: como o XOR nao e linearmente separavel, o MSE nunca atinge
    zero — demonstrando o limite teorico dos classificadores lineares.

    Retorna
    -------
    w             : list — pesos treinados [w0, w1, w2]
    historico_mse : list — MSE por epoca
    """
    amostras = [
        ([1.0, 0.0, 0.0], 0.0),   # (0, 0) -> 0
        ([1.0, 0.0, 1.0], 1.0),   # (0, 1) -> 1
        ([1.0, 1.0, 0.0], 1.0),   # (1, 0) -> 1
        ([1.0, 1.0, 1.0], 0.0),   # (1, 1) -> 0
    ]
    w, historico_mse = _treinar_delta(amostras, 3, max_epocas,
                                      taxa_aprendizado)
    return w, historico_mse


def predizer_delta(x_atributos, w, classe_pos, classe_neg):
    """
    Classifica uma amostra Iris com os pesos da Regra Delta.
    Usa sgn(net) para decisao binaria: net >= 0 -> classe_pos.

    Retorna: classe_pos ou classe_neg
    """
    x_aug = [1.0] + list(x_atributos)
    net = sum(wi * xi for wi, xi in zip(w, x_aug))
    return classe_pos if net >= 0 else classe_neg


def predizer_delta_xor(x1, x2, w):
    """
    Classifica um ponto XOR usando os pesos treinados.
    Limiar em 0.5 (ponto medio entre 0 e 1).

    Retorna: 0 ou 1
    """
    net = w[0] + w[1] * x1 + w[2] * x2
    return 1 if net >= 0.5 else 0


def acuracia_binaria_delta(dados, w, classe_pos, classe_neg,
                           indices_atributos):
    """
    Calcula a acuracia binaria da Regra Delta num conjunto de dados.

    Retorna: float no intervalo [0.0, 1.0]
    """
    corretos = total = 0
    for d in dados:
        if d['classe'] not in (classe_pos, classe_neg):
            continue
        x_sel = [d['atributos'][i] for i in indices_atributos]
        pred = predizer_delta(x_sel, w, classe_pos, classe_neg)
        if pred == d['classe']:
            corretos += 1
        total += 1
    return corretos / total if total > 0 else 0.0


# ===========================================================================
# One-vs-All (Um Contra Todos) — multiclasse
# ===========================================================================
def treinar_delta_ova(dados_treino, indices_atributos,
                      taxa_aprendizado=0.02, max_epocas=200):
    """
    Treina a Regra Delta no esquema Um-Contra-Todos (One-vs-All).

    Para cada classe c, treina um classificador binario:
        d = +1   se   amostra pertence a c
        d = -1   caso contrario

    A predicao multiclasse usa argmax sobre os 3 nets:
        classe* = argmax_c  (w_c^T · x_aug)

    Retorna
    -------
    pesos    : dict {classe: w}            — 1 vetor de pesos por classe
    historico: dict {classe: [mse/epoca]}  — 1 historico por classe
    n_epocas : int                         — max_epocas (sem parada antecipada)
    """
    classes = sorted(set(d['classe'] for d in dados_treino))
    n_pesos = 1 + len(indices_atributos)

    pesos = {}
    historico = {}

    for c in classes:
        amostras = []
        for d in dados_treino:
            x_sel = [d['atributos'][i] for i in indices_atributos]
            d_val = +1.0 if d['classe'] == c else -1.0
            amostras.append(([1.0] + x_sel, d_val))

        w, hist = _treinar_delta(amostras, n_pesos, max_epocas,
                                 taxa_aprendizado)
        pesos[c] = w
        historico[c] = hist

    return pesos, historico, max_epocas


def predizer_delta_ova(x_atributos, pesos):
    """
    Predicao multiclasse via argmax dos nets dos classificadores OvA.

    Retorna: (classe_vencedora, dict {classe: net})
    """
    x_aug = [1.0] + list(x_atributos)
    nets = {}
    for c, w in pesos.items():
        nets[c] = sum(wi * xi for wi, xi in zip(w, x_aug))
    vencedor = max(nets, key=nets.get)
    return vencedor, nets


def acuracia_delta_ova(dados, pesos, indices_atributos):
    """
    Acuracia multiclasse do classificador OvA.
    Retorna: float em [0.0, 1.0]
    """
    corretos = total = 0
    for d in dados:
        x_sel = [d['atributos'][i] for i in indices_atributos]
        pred, _ = predizer_delta_ova(x_sel, pesos)
        if pred == d['classe']:
            corretos += 1
        total += 1
    return corretos / total if total > 0 else 0.0
