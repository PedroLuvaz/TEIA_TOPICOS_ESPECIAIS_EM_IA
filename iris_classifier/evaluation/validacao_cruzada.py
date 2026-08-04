"""
Validacao cruzada k-fold estratificada — Python puro.

Motivacao
---------
Avaliar um classificador num unico split de 70/30 deixa apenas 45 amostras
de teste. Com tao poucas amostras, um erro a mais ou a menos muda a acuracia
em 2,2 pontos, e o resultado varia muito conforme a semente sorteada: no
Iris (petalas), o mesmo classificador vai de 86,7% a 100,0% so trocando a
semente.

A validacao cruzada resolve isso: o conjunto e dividido em k partes, e cada
uma serve de teste uma vez enquanto as outras k-1 treinam. Assim **toda**
amostra e testada exatamente uma vez, e o desvio-padrao entre as dobras
mostra o quanto o resultado e estavel.

Repetindo o processo com sementes diferentes (`repeticoes`), obtem-se uma
estimativa ainda mais confiavel — util para comparar classificadores cujas
acuracias diferem por pouco.

Sem numpy/scipy/sklearn — apenas listas nativas e lacos for.
"""

import random


def dobras_estratificadas(dados, k=5, semente=42):
    """
    Divide `dados` em k dobras preservando a proporcao de cada classe.

    Retorna: lista de k listas de amostras.
    """
    if k < 2:
        raise ValueError('k deve ser no minimo 2.')

    rng = random.Random(semente)

    # Agrupa por classe e embaralha dentro de cada grupo
    por_classe = {}
    for d in dados:
        por_classe.setdefault(d['classe'], []).append(d)

    dobras = [[] for _ in range(k)]
    for classe in sorted(por_classe):
        amostras = por_classe[classe][:]
        rng.shuffle(amostras)
        # Distribui em rodizio — mantem as dobras equilibradas mesmo quando
        # o numero de amostras da classe nao e multiplo de k
        for i, amostra in enumerate(amostras):
            dobras[i % k].append(amostra)

    return dobras


def validar_cruzado(dados, treinar_fn, predizer_fn, classes,
                    k=5, repeticoes=1, semente=42):
    """
    Roda a validacao cruzada k-fold (opcionalmente repetida).

    Parametros
    ----------
    dados       : lista de {'atributos': [...], 'classe': str}
    treinar_fn  : funcao(dados_treino) -> modelo
    predizer_fn : funcao(modelo, amostra) -> classe predita
    classes     : lista de nomes de classe
    k           : numero de dobras
    repeticoes  : quantas vezes repetir o k-fold (com sementes diferentes)
    semente     : semente base

    Retorna
    -------
    dict com:
        acuracias      : lista com a acuracia de cada dobra (todas as repeticoes)
        media          : media das acuracias
        desvio         : desvio-padrao amostral
        minimo, maximo : extremos observados
        matriz         : matriz de confusao acumulada (predito x real)
        n_avaliacoes   : k * repeticoes
    """
    acuracias = []
    matriz = {p: {r: 0 for r in classes} for p in classes}

    for rep in range(repeticoes):
        dobras = dobras_estratificadas(dados, k, semente + rep)

        for i in range(k):
            teste = dobras[i]
            treino = [d for j, dobra in enumerate(dobras) if j != i for d in dobra]

            modelo = treinar_fn(treino)

            corretos = 0
            for amostra in teste:
                pred = predizer_fn(modelo, amostra)
                real = amostra['classe']
                if pred in matriz and real in matriz[pred]:
                    matriz[pred][real] += 1
                if pred == real:
                    corretos += 1

            acuracias.append(corretos / len(teste) if teste else 0.0)

    n = len(acuracias)
    media = sum(acuracias) / n if n else 0.0
    if n > 1:
        variancia = sum((a - media) ** 2 for a in acuracias) / (n - 1)
    else:
        variancia = 0.0

    return {
        'acuracias': acuracias,
        'media': media,
        'desvio': variancia ** 0.5,
        'minimo': min(acuracias) if acuracias else 0.0,
        'maximo': max(acuracias) if acuracias else 0.0,
        'matriz': matriz,
        'n_avaliacoes': n,
        'k': k,
        'repeticoes': repeticoes,
    }


def intervalo_confianca(media, desvio, n, z=1.96):
    """
    Intervalo de confianca de 95% para a media das acuracias.

    Usa o erro-padrao da media (desvio / raiz(n)) — a aproximacao normal e
    razoavel para o numero de dobras usado aqui.
    """
    if n <= 1:
        return (media, media)
    erro_padrao = desvio / (n ** 0.5)
    return (max(0.0, media - z * erro_padrao),
            min(1.0, media + z * erro_padrao))
