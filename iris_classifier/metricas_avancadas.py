"""
Metricas Avancadas de Qualidade para Classificadores
=====================================================
Implementacoes em Python puro das metricas apresentadas na Aula PR_51:

  1. Acerto Global (Ag)
  2. Acuracia do Produtor (sensibilidade / recall por classe)
  3. Acuracia do Usuario (precisao por classe)
  4. Coeficiente Kappa
  5. Coeficiente Tau
  6. Coeficiente de Matthews (MCC) — problema binario
  7. Fb Score (b=1 e b=2) — problema binario

Todas as formulas sao fieis ao material do Prof. Robson Pequeno de Sousa.
"""

import math


# ===========================================================================
# Bloco 1 — Metricas multiclasse (extraidas da matriz de confusao)
# ===========================================================================

def acerto_global(matriz, classes):
    """
    Ag = (1/m) * sum(a_ii)
    Soma da diagonal principal dividida pelo total de amostras.
    """
    total = sum(matriz[r][p] for r in classes for p in classes)
    if total == 0:
        return 0.0
    acertos = sum(matriz[c][c] for c in classes)
    return acertos / total


def acuracia_produtor(matriz, classe):
    """
    A_pi = a_ii / a_{+i}   (coluna i — quantos do mundo real foram corretamente capturados)
    Equivale a Sensibilidade / Recall para classificacao multiclasse.
    a_{+i} = soma da coluna i (total de amostras reais da classe i).
    """
    total_real = sum(matriz[r][classe] for r in matriz)   # coluna
    if total_real == 0:
        return 0.0
    return matriz[classe][classe] / total_real


def acuracia_usuario(matriz, classe):
    """
    A_ui = a_ii / a_{i+}   (linha i — das que predisse como i, quantas eram i)
    Equivale a Precisao / VPP para classificacao multiclasse.
    a_{i+} = soma da linha i (total de predicoes da classe i).
    """
    total_predito = sum(matriz[classe][p] for p in matriz[classe])   # linha
    if total_predito == 0:
        return 0.0
    return matriz[classe][classe] / total_predito


# ===========================================================================
# Bloco 2 — Coeficiente Kappa
# ===========================================================================

def _acerto_casual(matriz, classes):
    """
    A_a = sum_i(a_{i+} * a_{+i}) / m^2
    Probabilidade de concordancia esperada por acaso.
    """
    m = sum(matriz[r][p] for r in classes for p in classes)
    if m == 0:
        return 0.0
    soma = 0.0
    for c in classes:
        linha_i  = sum(matriz[c][p] for p in classes)           # a_{i+}
        coluna_i = sum(matriz[r][c] for r in classes)           # a_{+i}
        soma += linha_i * coluna_i
    return soma / (m * m)


def kappa(matriz, classes):
    """
    K = (Ag - Aa) / (1 - Aa)
    K in ]-inf, 1] : 1 = perfeito, 0 = aleatorio, negativo = pior que aleatorio.
    Interpretacao (Landis & Koch 1977):
      <0.20  fraco | 0.21-0.40 razoavel | 0.41-0.60 moderado
      0.61-0.80 substancial | 0.81-1.00 quase perfeito
    """
    ag = acerto_global(matriz, classes)
    aa = _acerto_casual(matriz, classes)
    if abs(1 - aa) < 1e-12:
        return 1.0
    return (ag - aa) / (1 - aa)


def variancia_kappa(matriz, classes):
    """
    Variancia do Kappa (Congalton & Green, 2009).
    Necessaria para o teste de significancia entre dois classificadores.
    """
    m = sum(matriz[r][p] for r in classes for p in classes)
    if m == 0:
        return 0.0

    ag = acerto_global(matriz, classes)
    aa = _acerto_casual(matriz, classes)

    # phi_1 = Ag (acerto global)
    phi1 = ag
    # phi_2 = Aa (acerto casual)
    phi2 = aa

    # phi_3 = (1/m^2) * sum_i a_ii*(a_{i+} + a_{+i})
    phi3 = 0.0
    for c in classes:
        aii = matriz[c][c]
        linha  = sum(matriz[c][p] for p in classes)
        coluna = sum(matriz[r][c] for r in classes)
        phi3 += aii * (linha + coluna)
    phi3 /= (m * m)

    # phi_4 = (1/m^3) * sum_i sum_j a_ij*(a_{j+} + a_{+i})
    phi4 = 0.0
    for ri in classes:
        for rj in classes:
            aij = matriz[ri][rj]
            linha_j  = sum(matriz[rj][p] for p in classes)
            coluna_i = sum(matriz[r][ri] for r in classes)
            phi4 += aij * (linha_j + coluna_i)
    phi4 /= (m * m * m)

    den1 = (1 - phi2) ** 2
    den2 = (1 - phi2) ** 3
    den3 = (1 - phi2) ** 4

    if abs(den1) < 1e-12 or abs(den2) < 1e-12 or abs(den3) < 1e-12:
        return 0.0

    t1 = phi1 * (1 - phi1) / den1
    t2 = 2 * (1 - phi1) * (2 * phi1 * phi2 - phi3) / den2
    t3 = ((1 - phi1) ** 2) * (phi4 - 4 * phi2 ** 2) / den3

    return (t1 + t2 + t3) / m


# ===========================================================================
# Bloco 3 — Coeficiente Tau
# ===========================================================================

def tau(matriz, classes):
    """
    tau = (Ag - 1/C) / (1 - 1/C)    tau in [-1, 1]
    Alternativa ao Kappa que assume distribuicao uniforme entre classes.
    1/C e a probabilidade de acerto por acaso com C classes equiprovaveis.
    """
    c = len(classes)
    ag = acerto_global(matriz, classes)
    if abs(1 - 1.0 / c) < 1e-12:
        return 1.0
    return (ag - 1.0 / c) / (1.0 - 1.0 / c)


def variancia_tau(matriz, classes):
    """
    sigma_tau^2 = (1/m) * Ag*(1-Ag) / (1 - 1/C)^2
    """
    m = sum(matriz[r][p] for r in classes for p in classes)
    if m == 0:
        return 0.0
    c = len(classes)
    ag = acerto_global(matriz, classes)
    denom = (1.0 - 1.0 / c) ** 2
    if abs(denom) < 1e-12:
        return 0.0
    return (ag * (1 - ag)) / (denom * m)


# ===========================================================================
# Bloco 4 — Teste de significancia entre dois classificadores (Z)
# ===========================================================================

def z_kappa(k1, var1, k2, var2):
    """
    Z_k = (k1 - k2) / sqrt(var1 + var2)
    Teste Z para diferenca entre dois Kappas.
    """
    den = math.sqrt(var1 + var2)
    if den < 1e-12:
        return 0.0
    return (k1 - k2) / den


def z_tau(t1, var1, t2, var2):
    """
    Z_tau = (tau1 - tau2) / sqrt(var1 + var2)
    Teste Z para diferenca entre dois Taus.
    """
    den = math.sqrt(var1 + var2)
    if den < 1e-12:
        return 0.0
    return (t1 - t2) / den


def p_valor_z(z):
    """
    p-valor bilateral: P(|Z| > z) = 2*(1 - Phi(|z|))
    Approximacao da funcao erfc para calcular sem scipy.
    """
    return 2.0 * (1.0 - _phi(abs(z)))


def _phi(z):
    """Funcao de distribuicao acumulada N(0,1) — aproximacao de Horner."""
    # Abramowitz & Stegun 26.2.17
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    poly = t * (0.319381530
                + t * (-0.356563782
                       + t * (1.781477937
                              + t * (-1.821255978
                                     + t * 1.330274429))))
    phi_neg_z = (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * z * z) * poly
    if z >= 0:
        return 1.0 - phi_neg_z
    return phi_neg_z


# ===========================================================================
# Bloco 5 — Metricas binarias (VP, FP, FN, VN extraidas de problema dois classes)
# ===========================================================================

def _extrair_binario(matriz, classe_pos, classes):
    """
    Extrai VP, FP, FN, VN da matriz multiclasse em visao One-vs-Rest
    para a classe_pos.
    """
    vp = matriz[classe_pos][classe_pos]
    fp = sum(matriz[r][classe_pos] for r in classes if r != classe_pos)
    fn = sum(matriz[classe_pos][p] for p in classes if p != classe_pos)
    vn = sum(matriz[r][p] for r in classes for p in classes
             if r != classe_pos and p != classe_pos)
    return vp, fp, fn, vn


def sensibilidade(vp, fn):
    """VP / (VP + FN) — recall / acuracia do produtor binario."""
    if vp + fn == 0:
        return 0.0
    return vp / (vp + fn)


def especificidade(vn, fp):
    """VN / (VN + FP)."""
    if vn + fp == 0:
        return 0.0
    return vn / (vn + fp)


def precisao_binaria(vp, fp):
    """VP / (VP + FP) — acuracia do usuario."""
    if vp + fp == 0:
        return 0.0
    return vp / (vp + fp)


def mcc(vp, vn, fp, fn):
    """
    Coeficiente de Matthews:
        MCC = (VP*VN - FP*FN) / sqrt((VP+FP)(VP+FN)(VN+FP)(VN+FN))
    MCC in [-1, 1]: -1 discordancia total, 0 aleatorio, 1 perfeito.
    Robusto para classes desbalanceadas.
    """
    num = vp * vn - fp * fn
    den = math.sqrt((vp + fp) * (vp + fn) * (vn + fp) * (vn + fn))
    if den < 1e-12:
        return 0.0
    return num / den


def fb_score(vp, fp, fn, b=1):
    """
    Fb = (1 + b^2) * (Pr * Re) / (b^2 * Pr + Re)
    b=1  → F1 (equilibrio precisao/recall)
    b=2  → F2 (maior peso a revocacao — preferido em rastreamento)
    b=0.5 → F0.5 (maior peso a precisao)
    """
    pr = precisao_binaria(vp, fp)
    re = sensibilidade(vp, fn)
    denom = (b * b * pr) + re
    if denom < 1e-12:
        return 0.0
    return (1 + b * b) * pr * re / denom


# ===========================================================================
# Bloco 6 — Relatorio consolidado para um par de classes (problema binario)
# ===========================================================================

def relatorio_binario(matriz, classe_pos, classes):
    """
    Calcula todas as metricas binarias para classe_pos vs resto.
    Retorna dict com todas as metricas.
    """
    vp, fp, fn, vn = _extrair_binario(matriz, classe_pos, classes)
    s   = sensibilidade(vp, fn)
    e   = especificidade(vn, fp)
    pr  = precisao_binaria(vp, fp)
    f1  = fb_score(vp, fp, fn, b=1)
    f2  = fb_score(vp, fp, fn, b=2)
    m_  = mcc(vp, vn, fp, fn)
    return {
        'vp': vp, 'fp': fp, 'fn': fn, 'vn': vn,
        'sensibilidade': s,
        'especificidade': e,
        'precisao': pr,
        'f1': f1,
        'f2': f2,
        'mcc': m_,
    }


# ===========================================================================
# Bloco 7 — Relatorio multiclasse completo
# ===========================================================================

def relatorio_completo(predicoes, gabarito, classes, nome_modelo=''):
    """
    Dado listas de predicoes e gabarito, retorna dict com todas as metricas.
    """
    # Matriz de confusao
    matriz = {real: {pred: 0 for pred in classes} for real in classes}
    for pred, real in zip(predicoes, gabarito):
        if real in matriz and pred in matriz[real]:
            matriz[real][pred] += 1

    ag = acerto_global(matriz, classes)
    k  = kappa(matriz, classes)
    t  = tau(matriz, classes)
    vk = variancia_kappa(matriz, classes)
    vt = variancia_tau(matriz, classes)

    por_classe = {}
    for c in classes:
        vp, fp, fn, vn = _extrair_binario(matriz, c, classes)
        por_classe[c] = {
            'acuracia_produtor': acuracia_produtor(matriz, c),
            'acuracia_usuario':  acuracia_usuario(matriz, c),
            'sensibilidade':     sensibilidade(vp, fn),
            'especificidade':    especificidade(vn, fp),
            'precisao':          precisao_binaria(vp, fp),
            'f1':                fb_score(vp, fp, fn, b=1),
            'f2':                fb_score(vp, fp, fn, b=2),
            'mcc':               mcc(vp, vn, fp, fn),
            'vp': vp, 'fp': fp, 'fn': fn, 'vn': vn,
        }

    return {
        'nome': nome_modelo,
        'matriz': matriz,
        'acerto_global': ag,
        'kappa': k,
        'variancia_kappa': vk,
        'tau': t,
        'variancia_tau': vt,
        'por_classe': por_classe,
    }