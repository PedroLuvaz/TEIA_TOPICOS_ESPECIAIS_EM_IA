from math_utils import calcular_media, discriminante

def treinar(dados_treino, indices_atributos):
    """
    Treina o Classificador de Distância Mínima calculando o vetor médio (protótipo)
    para cada classe usando os índices de atributos especificados.
    Retorna: dict {classe: vetor_prototipo}
    """
    prototipos = {}
    classes = sorted(list(set(d['classe'] for d in dados_treino)))
    
    for classe in classes:
        amostras_da_classe = [d['atributos'] for d in dados_treino if d['classe'] == classe]
        # Extrair apenas os atributos selecionados
        atributos_selecionados = [[s[i] for i in indices_atributos] for s in amostras_da_classe]
        prototipos[classe] = calcular_media(atributos_selecionados)
        
    return prototipos

def predizer_todas_classes(x, prototipos, indices_atributos):
    """
    Prediz a classe para uma amostra x calculando dj(x) para todos os protótipos.
    Retorna: dict {classe: valor_dj}, classe_vencedora
    """
    # Extrair apenas os atributos selecionados de x
    x_selecionado = [x[i] for i in indices_atributos]
    
    pontuacoes = {}
    for classe, mj in prototipos.items():
        pontuacoes[classe] = discriminante(x_selecionado, mj)
        
    vencedor = max(pontuacoes, key=pontuacoes.get)
    return pontuacoes, vencedor

def predizer_binario(x, pi, pj, classe_i, classe_j, indices_atributos):
    """
    Prediz entre duas classes (classe_i, classe_j) com protótipos pi, pj.
    """
    x_selecionado = [x[i] for i in indices_atributos]
    di = discriminante(x_selecionado, pi)
    dj = discriminante(x_selecionado, pj)
    
    return classe_i if di > dj else classe_j
