def acuracia(predicoes, gabarito):
    """
    Acuracia = numero de predicoes corretas / total de amostras
    Metrica principal para avaliar o classificador nos 3 experimentos.
    """
    if not predicoes:
        return 0.0
    corretos = sum(1 for p, gt in zip(predicoes, gabarito) if p == gt)
    return corretos / len(predicoes)
