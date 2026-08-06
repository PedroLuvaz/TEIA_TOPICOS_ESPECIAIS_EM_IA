"""
Testes de significancia para comparar classificadores — Python puro.

Contexto
--------
O projeto ja trazia o teste Z sobre o Kappa. Ele tem uma limitacao
metodologica importante: assume que os dois Kappas sao **independentes**,
somando as variancias. Mas quando dois classificadores sao avaliados no
MESMO conjunto de teste, os resultados sao **pareados** — eles acertam e
erram as mesmas amostras dificeis. Tratar como independente joga poder
estatistico fora e torna o teste conservador demais.

Este modulo acrescenta os testes apropriados:

* **McNemar** — o teste padrao para comparar dois classificadores no mesmo
  conjunto. Olha so os casos em que os dois discordam.
* **Bootstrap pareado** — intervalo de confianca para a diferenca de
  QUALQUER metrica (MCC, F1, precisao, revocacao...), reamostrando as
  mesmas amostras para os dois classificadores.
* **Teste de permutacao** — alternativa nao parametrica: sob a hipotese
  nula, trocar as predicoes de A e B numa amostra nao muda nada.

Tambem implementa o **MCC multiclasse** (Gorodkin, 2004), que o projeto
so tinha na versao binaria (um-contra-resto).

Sem numpy/scipy — apenas `math` e `random`.
"""

import math
import random


# ===========================================================================
# MCC multiclasse
# ===========================================================================
def mcc_multiclasse(matriz, classes):
    """
    Coeficiente de Matthews generalizado para K classes (Gorodkin, 2004).

        MCC = (c·s - Σ_k p_k t_k) /
              sqrt((s² - Σ_k p_k²) · (s² - Σ_k t_k²))

    onde, para a matriz no formato deste projeto (matriz[predito][real]):
        t_k = total de amostras cuja classe real e k   (soma da coluna k)
        p_k = total de amostras preditas como k        (soma da linha k)
        c   = total de acertos (traco da matriz)
        s   = total de amostras

    Vale -1 (discordancia total) a +1 (perfeito); 0 e o nivel do acaso.
    Diferente do acerto global, continua informativo com classes
    desbalanceadas.
    """
    t = {k: sum(matriz[p].get(k, 0) for p in classes) for k in classes}
    p = {k: sum(matriz[k].get(r, 0) for r in classes) for k in classes}
    c = sum(matriz[k].get(k, 0) for k in classes)
    s = sum(t.values())

    if s == 0:
        return 0.0

    numerador = c * s - sum(p[k] * t[k] for k in classes)
    termo_p = s * s - sum(p[k] ** 2 for k in classes)
    termo_t = s * s - sum(t[k] ** 2 for k in classes)
    denominador = math.sqrt(termo_p * termo_t)

    if denominador < 1e-12:
        return 0.0
    return numerador / denominador


# ===========================================================================
# Metricas calculadas direto das listas de predicoes
# ===========================================================================
def _matriz_de(predicoes, gabarito, classes):
    m = {p: {r: 0 for r in classes} for p in classes}
    for pred, real in zip(predicoes, gabarito):
        if pred in m and real in m[pred]:
            m[pred][real] += 1
    return m


def _binario(matriz, classe, classes):
    """(VP, FP, FN, VN) da classe contra todas as outras."""
    vp = matriz[classe][classe]
    fp = sum(matriz[classe][r] for r in classes if r != classe)
    fn = sum(matriz[p][classe] for p in classes if p != classe)
    vn = sum(matriz[p][r] for p in classes for r in classes
             if p != classe and r != classe)
    return vp, fp, fn, vn


def acerto_global_de(predicoes, gabarito, classes=None):
    if not gabarito:
        return 0.0
    return sum(1 for p, r in zip(predicoes, gabarito) if p == r) / len(gabarito)


def mcc_de(predicoes, gabarito, classes):
    return mcc_multiclasse(_matriz_de(predicoes, gabarito, classes), classes)


def kappa_de(predicoes, gabarito, classes):
    """Kappa de Cohen calculado direto das listas."""
    m = _matriz_de(predicoes, gabarito, classes)
    n = len(gabarito)
    if n == 0:
        return 0.0
    ag = sum(m[k][k] for k in classes) / n
    ac = sum(
        (sum(m[k].values()) * sum(m[p][k] for p in classes)) / (n * n)
        for k in classes
    )
    return (ag - ac) / (1 - ac) if abs(1 - ac) > 1e-12 else 0.0


def _media_por_classe(predicoes, gabarito, classes, fn):
    """Macro-media de uma metrica binaria sobre as classes."""
    m = _matriz_de(predicoes, gabarito, classes)
    valores = []
    for c in classes:
        vp, fp, fn_, vn = _binario(m, c, classes)
        valores.append(fn(vp, fp, fn_, vn))
    return sum(valores) / len(valores) if valores else 0.0


def precisao_de(predicoes, gabarito, classes):
    return _media_por_classe(
        predicoes, gabarito, classes,
        lambda vp, fp, fn, vn: vp / (vp + fp) if vp + fp else 0.0)


def revocacao_de(predicoes, gabarito, classes):
    return _media_por_classe(
        predicoes, gabarito, classes,
        lambda vp, fp, fn, vn: vp / (vp + fn) if vp + fn else 0.0)


def especificidade_de(predicoes, gabarito, classes):
    return _media_por_classe(
        predicoes, gabarito, classes,
        lambda vp, fp, fn, vn: vn / (vn + fp) if vn + fp else 0.0)


def f1_de(predicoes, gabarito, classes):
    def _f1(vp, fp, fn, vn):
        p = vp / (vp + fp) if vp + fp else 0.0
        r = vp / (vp + fn) if vp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0
    return _media_por_classe(predicoes, gabarito, classes, _f1)


#: Metricas disponiveis para os testes. Todas recebem (predicoes, gabarito, classes).
METRICAS = {
    'mcc': ('Coeficiente de Matthews (MCC)', mcc_de),
    'kappa': ('Kappa de Cohen', kappa_de),
    'acerto_global': ('Acerto Global', acerto_global_de),
    'f1': ('F1 (macro)', f1_de),
    'precisao': ('Precisão (macro)', precisao_de),
    'revocacao': ('Revocação (macro)', revocacao_de),
    'especificidade': ('Especificidade (macro)', especificidade_de),
}


# ===========================================================================
# Teste de McNemar — o teste pareado correto
# ===========================================================================
def _binomial_cdf(k, n, p=0.5):
    """P(X <= k) para X ~ Binomial(n, p), somando os termos exatos."""
    total = 0.0
    for i in range(k + 1):
        total += math.comb(n, i) * (p ** i) * ((1 - p) ** (n - i))
    return min(1.0, total)


def _qui2_p_valor_1gl(x):
    """
    p-valor da qui-quadrado com 1 grau de liberdade.

    Com 1 gl vale a identidade  P(X² > x) = 2·(1 - Φ(√x)),  entao basta a
    funcao de distribuicao normal — que ja existe no projeto via erf.
    """
    if x <= 0:
        return 1.0
    z = math.sqrt(x)
    phi = 0.5 * (1 + math.erf(z / math.sqrt(2)))
    return 2 * (1 - phi)


def mcnemar(predicoes_a, predicoes_b, gabarito, exato=None):
    """
    Teste de McNemar para dois classificadores no MESMO conjunto de teste.

    Monta a tabela 2x2 dos acertos pareados:

                       B acertou   B errou
        A acertou          a          b
        A errou            c          d

    Só `b` e `c` (os discordantes) carregam informacao: se os dois
    classificadores fossem equivalentes, esperar-se-ia b ≈ c.

        H0: b e c vem da mesma distribuicao (os classificadores empatam)

    Para b + c >= 25 usa a aproximacao qui-quadrado com correcao de
    continuidade de Edwards; abaixo disso usa o teste binomial exato, que
    e o correto para amostras pequenas — o caso do Iris, com 45 amostras
    de teste.

    Retorna dict com a tabela, a estatistica, o p-valor e o veredito.
    """
    a = b = c = d = 0
    for pa, pb, real in zip(predicoes_a, predicoes_b, gabarito):
        ca, cb = pa == real, pb == real
        if ca and cb:
            a += 1
        elif ca and not cb:
            b += 1
        elif not ca and cb:
            c += 1
        else:
            d += 1

    discordantes = b + c
    if exato is None:
        exato = discordantes < 25

    if discordantes == 0:
        return {
            'a': a, 'b': b, 'c': c, 'd': d,
            'discordantes': 0,
            'metodo': 'nenhum',
            'estatistica': None,
            'p_valor': 1.0,
            'significativo': False,
            'observacao': ('Os dois classificadores acertaram e erraram '
                           'exatamente as mesmas amostras — não há nada a testar.'),
        }

    if exato:
        # Binomial bicaudal: P(X <= min(b,c)) * 2, com X ~ Bin(b+c, 0.5)
        k = min(b, c)
        p_valor = min(1.0, 2 * _binomial_cdf(k, discordantes, 0.5))
        estatistica = k
        metodo = 'binomial exato'
        observacao = (f'Poucos discordantes ({discordantes} < 25): usado o '
                      'teste binomial exato, mais confiável que a '
                      'aproximação qui-quadrado nesse regime.')
    else:
        estatistica = (abs(b - c) - 1) ** 2 / discordantes
        p_valor = _qui2_p_valor_1gl(estatistica)
        metodo = 'qui-quadrado com correção de continuidade'
        observacao = (f'{discordantes} discordantes: a aproximação '
                      'qui-quadrado é adequada.')

    return {
        'a': a, 'b': b, 'c': c, 'd': d,
        'discordantes': discordantes,
        'metodo': metodo,
        'estatistica': estatistica,
        'p_valor': p_valor,
        'significativo': p_valor < 0.05,
        'observacao': observacao,
    }


# ===========================================================================
# Bootstrap pareado
# ===========================================================================
def _percentil(valores_ordenados, q):
    """Percentil por interpolacao linear (q em [0, 1])."""
    if not valores_ordenados:
        return 0.0
    if len(valores_ordenados) == 1:
        return valores_ordenados[0]
    pos = q * (len(valores_ordenados) - 1)
    baixo = int(math.floor(pos))
    alto = min(baixo + 1, len(valores_ordenados) - 1)
    peso = pos - baixo
    return valores_ordenados[baixo] * (1 - peso) + valores_ordenados[alto] * peso


def bootstrap_diferenca(predicoes_a, predicoes_b, gabarito, classes,
                        fn_metrica, n_reamostragens=2000, semente=42,
                        alfa=0.05):
    """
    Intervalo de confianca da DIFERENCA (A - B) de uma metrica, por
    bootstrap **pareado**.

    A cada reamostragem, sorteia indices com reposicao e calcula a metrica
    dos dois classificadores **nas mesmas amostras** — preservando o
    pareamento. Se o intervalo nao contem zero, a diferenca e significativa
    ao nivel alfa.

    Funciona para qualquer metrica, inclusive as que nao tem formula
    fechada de variancia — como o MCC.
    """
    rng = random.Random(semente)
    n = len(gabarito)
    if n == 0:
        raise ValueError('Conjunto de teste vazio.')

    obs_a = fn_metrica(predicoes_a, gabarito, classes)
    obs_b = fn_metrica(predicoes_b, gabarito, classes)
    diferenca_observada = obs_a - obs_b

    diferencas = []
    for _ in range(n_reamostragens):
        idx = [rng.randrange(n) for _ in range(n)]
        g = [gabarito[i] for i in idx]
        pa = [predicoes_a[i] for i in idx]
        pb = [predicoes_b[i] for i in idx]
        diferencas.append(fn_metrica(pa, g, classes) - fn_metrica(pb, g, classes))

    diferencas.sort()
    baixo = _percentil(diferencas, alfa / 2)
    alto = _percentil(diferencas, 1 - alfa / 2)

    media = sum(diferencas) / len(diferencas)
    variancia = (sum((x - media) ** 2 for x in diferencas)
                 / (len(diferencas) - 1)) if len(diferencas) > 1 else 0.0

    return {
        'metrica_a': obs_a,
        'metrica_b': obs_b,
        'diferenca': diferenca_observada,
        'ic_baixo': baixo,
        'ic_alto': alto,
        'erro_padrao': variancia ** 0.5,
        'contem_zero': baixo <= 0 <= alto,
        'significativo': not (baixo <= 0 <= alto),
        'n_reamostragens': n_reamostragens,
        'confianca': 1 - alfa,
        'distribuicao': _histograma(diferencas, 40),
    }


def _histograma(valores, n_faixas):
    """Histograma leve, para o frontend desenhar a distribuicao."""
    if not valores:
        return {'faixas': [], 'contagens': []}
    minimo, maximo = valores[0], valores[-1]
    if abs(maximo - minimo) < 1e-12:
        return {'faixas': [minimo], 'contagens': [len(valores)]}
    largura = (maximo - minimo) / n_faixas
    contagens = [0] * n_faixas
    for v in valores:
        i = min(n_faixas - 1, int((v - minimo) / largura))
        contagens[i] += 1
    faixas = [minimo + (i + 0.5) * largura for i in range(n_faixas)]
    return {'faixas': faixas, 'contagens': contagens}


# ===========================================================================
# Teste de permutacao
# ===========================================================================
def teste_permutacao(predicoes_a, predicoes_b, gabarito, classes,
                     fn_metrica, n_permutacoes=2000, semente=42):
    """
    Teste de permutacao pareado.

    Sob H0 (os classificadores sao equivalentes), qual das duas predicoes
    veio de A e qual veio de B e irrelevante. Entao, para cada amostra,
    trocamos as predicoes com probabilidade 1/2 e recalculamos a diferenca.

    O p-valor e a proporcao de permutacoes cuja diferenca (em modulo)
    ficou pelo menos tao extrema quanto a observada.
    """
    rng = random.Random(semente)
    n = len(gabarito)

    obs = (fn_metrica(predicoes_a, gabarito, classes)
           - fn_metrica(predicoes_b, gabarito, classes))
    alvo = abs(obs)

    extremos = 0
    for _ in range(n_permutacoes):
        pa, pb = [], []
        for i in range(n):
            if rng.random() < 0.5:
                pa.append(predicoes_a[i])
                pb.append(predicoes_b[i])
            else:
                pa.append(predicoes_b[i])
                pb.append(predicoes_a[i])
        dif = fn_metrica(pa, gabarito, classes) - fn_metrica(pb, gabarito, classes)
        if abs(dif) >= alvo - 1e-12:
            extremos += 1

    # Correcao de Davison & Hinkley: evita p-valor exatamente zero
    p_valor = (extremos + 1) / (n_permutacoes + 1)

    return {
        'diferenca_observada': obs,
        'p_valor': p_valor,
        'significativo': p_valor < 0.05,
        'n_permutacoes': n_permutacoes,
        'extremos': extremos,
    }


# ===========================================================================
# Comparacao completa
# ===========================================================================
def comparar(predicoes_a, predicoes_b, gabarito, classes, metrica='mcc',
             n_reamostragens=2000, n_permutacoes=2000, semente=42):
    """
    Roda os tres testes de uma vez para um par de classificadores.

    Devolve o resultado de McNemar (que independe da metrica escolhida,
    pois compara acertos), mais o bootstrap e a permutacao para a metrica
    pedida.
    """
    if metrica not in METRICAS:
        raise ValueError(
            f"metrica deve ser uma de {sorted(METRICAS)}, recebida '{metrica}'")

    nome_metrica, fn = METRICAS[metrica]

    return {
        'metrica': metrica,
        'nome_metrica': nome_metrica,
        'n_amostras': len(gabarito),
        'mcnemar': mcnemar(predicoes_a, predicoes_b, gabarito),
        'bootstrap': bootstrap_diferenca(
            predicoes_a, predicoes_b, gabarito, classes, fn,
            n_reamostragens, semente),
        'permutacao': teste_permutacao(
            predicoes_a, predicoes_b, gabarito, classes, fn,
            n_permutacoes, semente),
    }
