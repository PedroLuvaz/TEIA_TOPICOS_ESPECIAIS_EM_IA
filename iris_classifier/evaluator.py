"""
Metricas de avaliacao do classificador — todas implementadas em Python puro.

Para cada classe j (visao One-vs-Rest):
  VP = Verdadeiro Positivo  (predito j  e  real j)
  FP = Falso Positivo       (predito j  e  real != j)
  FN = Falso Negativo       (predito != j  e  real j)

  Precisao(j)  = VP / (VP + FP)        — dos que eu disse que sao j, quantos sao mesmo j?
  Revocacao(j) = VP / (VP + FN)        — dos que eram j, quantos eu acertei?
  F1(j)        = 2 * P * R / (P + R)   — media harmonica de precisao e revocacao
"""


def acuracia(predicoes, gabarito):
    """
    Acuracia = predicoes corretas / total
    Metrica global do classificador.
    """
    if not predicoes:
        return 0.0
    corretos = sum(1 for p, gt in zip(predicoes, gabarito) if p == gt)
    return corretos / len(predicoes)


def matriz_confusao(predicoes, gabarito, classes):
    """
    Constroi a matriz de confusao C onde:
        C[real][predito] = quantidade de amostras com classe real = i e predicao = j

    Retorna: dict {classe_real: {classe_predita: contagem}}

    Diagonal principal = acertos. Fora da diagonal = erros (e mostra ONDE o
    classificador se confunde, nao apenas QUANTO).
    """
    matriz = {real: {pred: 0 for pred in classes} for real in classes}
    for pred, real in zip(predicoes, gabarito):
        if real in matriz and pred in matriz[real]:
            matriz[real][pred] += 1
    return matriz


def precisao_por_classe(matriz, classe):
    """
    Precisao(j) = VP / (VP + FP)
    VP = matriz[j][j]              (predito j e real j)
    FP = sum(matriz[i][j] para i != j)   (predito j e real i != j)
    """
    vp = matriz[classe][classe]
    fp = sum(matriz[real][classe] for real in matriz if real != classe)
    if vp + fp == 0:
        return 0.0
    return vp / (vp + fp)


def revocacao_por_classe(matriz, classe):
    """
    Revocacao(j) = VP / (VP + FN)
    VP = matriz[j][j]
    FN = sum(matriz[j][k] para k != j)   (real j mas predito k != j)
    """
    vp = matriz[classe][classe]
    fn = sum(matriz[classe][pred] for pred in matriz[classe] if pred != classe)
    if vp + fn == 0:
        return 0.0
    return vp / (vp + fn)


def f1_por_classe(matriz, classe):
    """
    F1(j) = 2 * P * R / (P + R)  — media harmonica.
    Penaliza mais quando P ou R e baixo (uma das duas isolada nao basta).
    """
    p = precisao_por_classe(matriz, classe)
    r = revocacao_por_classe(matriz, classe)
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)


def imprimir_matriz_confusao(matriz, classes):
    """
    Imprime a matriz de confusao formatada no terminal.
    Linhas = classe real | Colunas = classe predita.
    """
    larg_col = max(12, max(len(c) for c in classes) + 2)
    larg_lin = max(16, max(len(c) for c in classes) + 2)
    cabecalho = "Real \\ Predito".ljust(larg_lin) + "".join(c.ljust(larg_col) for c in classes) + "Total"
    print(cabecalho)
    print("-" * len(cabecalho))
    for real in classes:
        total_linha = sum(matriz[real].values())
        linha = real.ljust(larg_lin) + "".join(str(matriz[real][pred]).ljust(larg_col) for pred in classes)
        linha += str(total_linha)
        print(linha)


def imprimir_metricas_por_classe(matriz, classes):
    """
    Imprime tabela com Precisao, Revocacao e F1 para cada classe.
    """
    larg = max(12, max(len(c) for c in classes) + 2)
    print(f"{'Classe'.ljust(larg)}{'Precisao'.ljust(12)}{'Revocacao'.ljust(12)}{'F1'.ljust(10)}")
    print("-" * (larg + 12 + 12 + 10))
    for c in classes:
        p = precisao_por_classe(matriz, c)
        r = revocacao_por_classe(matriz, c)
        f1 = f1_por_classe(matriz, c)
        print(f"{c.ljust(larg)}{f'{p:.4f}'.ljust(12)}{f'{r:.4f}'.ljust(12)}{f'{f1:.4f}'.ljust(10)}")
