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
        C[predito][real] = quantidade de amostras com classe real = j e predicao = i

    Retorna: dict {classe_predita: {classe_real: contagem}}

    Diagonal principal = acertos. Fora da diagonal = erros.
    """
    matriz = {pred: {real: 0 for real in classes} for pred in classes}
    for pred, real in zip(predicoes, gabarito):
        if pred in matriz and real in matriz[pred]:
            matriz[pred][real] += 1
    return matriz


def precisao_por_classe(matriz, classe):
    """
    Precisao(j) = VP / (VP + FP)
    VP = matriz[j][j]                    (predito j e real j)
    FP = sum(matriz[j][real] para real != j)  (predito j mas real != j)
    """
    vp = matriz[classe][classe]
    fp = sum(matriz[classe][real] for real in matriz[classe] if real != classe)
    if vp + fp == 0:
        return 0.0
    return vp / (vp + fp)


def revocacao_por_classe(matriz, classe):
    """
    Revocacao(j) = VP / (VP + FN)
    VP = matriz[j][j]
    FN = sum(matriz[pred][classe] para pred != j)  (real j mas predito != j)
    """
    vp = matriz[classe][classe]
    fn = sum(matriz[pred][classe] for pred in matriz if pred != classe)
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
    Linhas = classe predita | Colunas = classe real.
    Apresenta o total das linhas, colunas e total geral.
    """
    larg_col = max(12, max(len(c) for c in classes) + 2)
    larg_lin = max(16, max(len(c) for c in classes) + 2)
    cabecalho = "Predito \\ Real".ljust(larg_lin) + "".join(c.ljust(larg_col) for c in classes) + "Total"
    print(cabecalho)
    print("-" * len(cabecalho))
    
    totais_colunas = {real: 0 for real in classes}
    total_geral = 0
    
    for pred in classes:
        total_linha = sum(matriz[pred][real] for real in classes)
        linha = pred.ljust(larg_lin) + "".join(str(matriz[pred][real]).ljust(larg_col) for real in classes)
        linha += str(total_linha)
        print(linha)
        
        for real in classes:
            totais_colunas[real] += matriz[pred][real]
        total_geral += total_linha
        
    print("-" * len(cabecalho))
    linha_total = "Total".ljust(larg_lin) + "".join(str(totais_colunas[real]).ljust(larg_col) for real in classes)
    linha_total += str(total_geral)
    print(linha_total)


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
