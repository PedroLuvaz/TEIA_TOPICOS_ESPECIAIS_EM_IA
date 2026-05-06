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
