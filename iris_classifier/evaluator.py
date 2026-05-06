def acuracia(predicoes, gabarito):
    """Calcula a proporção de predições corretas."""
    if not predicoes:
        return 0.0
    corretos = sum(1 for p, gt in zip(predicoes, gabarito) if p == gt)
    return corretos / len(predicoes)

def matriz_confusao(predicoes, gabarito, classes):
    """
    Calcula uma matriz de confusão manualmente.
    Linhas = Gabarito (Real), Colunas = Predito
    Retorna: dict representando a matriz
    """
    matriz = {classe_real: {classe_pred: 0 for classe_pred in classes} for classe_real in classes}

    for p, gt in zip(predicoes, gabarito):
        if gt in matriz and p in matriz[gt]:
            matriz[gt][p] += 1

    return matriz

def imprimir_matriz_confusao(mc, classes):
    """Imprime a matriz de confusão de forma legível."""
    print("\nMatriz de Confusão:")
    cabecalho = "Real \\ Pred  | " + " | ".join(f"{c:10}" for c in classes)
    print(cabecalho)
    print("-" * len(cabecalho))
    for classe_real in classes:
        linha = f"{classe_real:10} | " + " | ".join(f"{mc[classe_real][classe_pred]:10}" for classe_pred in classes)
        print(linha)

def precisao_por_classe(mc, classes):
    """
    Calcula a precisão para cada classe: TP / (TP + FP).
    Precisão responde: das amostras que o modelo disse ser da classe X, quantas realmente são?
    Retorna: dict {classe: precisao}
    """
    resultado = {}
    for classe in classes:
        tp = mc[classe][classe]
        fp = sum(mc[outra][classe] for outra in classes if outra != classe)
        denominador = tp + fp
        resultado[classe] = tp / denominador if denominador > 0 else 0.0
    return resultado

def revocacao_por_classe(mc, classes):
    """
    Calcula a revocação (recall) para cada classe: TP / (TP + FN).
    Revocação responde: das amostras reais da classe X, quantas o modelo identificou corretamente?
    Retorna: dict {classe: revocacao}
    """
    resultado = {}
    for classe in classes:
        tp = mc[classe][classe]
        fn = sum(mc[classe][outra] for outra in classes if outra != classe)
        denominador = tp + fn
        resultado[classe] = tp / denominador if denominador > 0 else 0.0
    return resultado

def f1_por_classe(mc, classes):
    """
    Calcula o F1-Score para cada classe: 2 * (P * R) / (P + R).
    F1 é a média harmônica entre precisão e revocação — balanceia os dois.
    Retorna: dict {classe: f1}
    """
    precisoes = precisao_por_classe(mc, classes)
    revocacoes = revocacao_por_classe(mc, classes)
    resultado = {}
    for classe in classes:
        p = precisoes[classe]
        r = revocacoes[classe]
        denominador = p + r
        resultado[classe] = 2 * p * r / denominador if denominador > 0 else 0.0
    return resultado

def imprimir_metricas_por_classe(mc, classes):
    """Imprime tabela completa: Classe | Precisão | Revocação | F1-Score."""
    precisoes = precisao_por_classe(mc, classes)
    revocacoes = revocacao_por_classe(mc, classes)
    f1s = f1_por_classe(mc, classes)

    print("\nMétricas por Classe:")
    cabecalho = f"{'Classe':12} | {'Precisão':10} | {'Revocação':10} | {'F1-Score':10}"
    print(cabecalho)
    print("-" * len(cabecalho))
    for classe in classes:
        print(f"{classe:12} | {precisoes[classe]:10.4f} | {revocacoes[classe]:10.4f} | {f1s[classe]:10.4f}")

    media_p = sum(precisoes.values()) / len(classes)
    media_r = sum(revocacoes.values()) / len(classes)
    media_f1 = sum(f1s.values()) / len(classes)
    print("-" * len(cabecalho))
    print(f"{'Média (macro)':12} | {media_p:10.4f} | {media_r:10.4f} | {media_f1:10.4f}")
