"""
Catalogo unico dos modelos de classificacao do aplicativo.

Por que existe
--------------
Ate aqui cada modelo vivia dentro do seu laboratorio: para trocar de
classificador o usuario trocava de tela. O requisito da entrega e outro — "o
projeto deve disponibilizar ao usuario opcoes de definicao do modelo a ser
utilizado no processo de classificacao, bem como a parametrizacao do modelo".

Este modulo transforma os modelos em ITENS DE UM CATALOGO: cada um declara
seu nome, sua descricao, o esquema dos proprios hiperparametros e as funcoes
de treino e predicao. Com isso:

  · a tela "Classificar" monta os controles sozinha, a partir do esquema;
  · os testes de significancia comparam qualquer par do catalogo;
  · acrescentar um modelo novo e acrescentar uma entrada aqui.

Nenhuma matematica mora neste arquivo — tudo e delegado aos modulos em
Python puro de `iris_classifier/models/`.

Esquema de um parametro
-----------------------
    {'id', 'rotulo', 'tipo', 'padrao', 'min', 'max', 'passo', 'opcoes',
     'ajuda'}

`tipo` e um de: 'inteiro', 'numero', 'opcoes'. O valor 0 em
`profundidade_max` significa "sem limite" (a interface mostra isso no rotulo)
porque um controle deslizante nao tem como emitir `None`.
"""
from functools import lru_cache

from models.bayes_classifier import (predizer_todas_classes_bayes,
                                     treinar_bayes)
from models.classifier import predizer_todas_classes, treinar as treinar_dm
from models.delta_rule import predizer_delta_ova, treinar_delta_ova
from models.mlp_multiclasse import treinar_mlp_multiclasse
from models.perceptron import (predizer_perceptron_ova,
                               treinar_perceptron_ova)
from models.random_forest import treinar_floresta

from .core import indices_de, obter_split


# ---------------------------------------------------------------------------
# Adaptadores: cada modelo exposto com a mesma interface
# (treinar -> objeto, predizer(objeto, atributos) -> classe)
# ---------------------------------------------------------------------------
def _dm_treinar(treino, idx, **_):
    return {'prototipos': treinar_dm(treino, idx), 'idx': idx}


def _dm_predizer(m, atributos):
    return predizer_todas_classes(atributos, m['prototipos'], m['idx'])[1]


def _dm_scores(m, atributos):
    return predizer_todas_classes(atributos, m['prototipos'], m['idx'])[0]


def _bayes_treinar(naive):
    def _treinar(treino, idx, **_):
        return {'modelo': treinar_bayes(treino, idx, naive=naive), 'idx': idx}
    return _treinar


def _bayes_predizer(m, atributos):
    return predizer_todas_classes_bayes(atributos, m['modelo'], m['idx'])[1]


def _bayes_scores(m, atributos):
    return predizer_todas_classes_bayes(atributos, m['modelo'], m['idx'])[0]


def _delta_treinar(treino, idx, taxa=0.02, max_epocas=200, **_):
    pesos, historico, _ = treinar_delta_ova(treino, idx, taxa, max_epocas)
    return {'pesos': pesos, 'idx': idx, 'historico': historico}


def _delta_predizer(m, atributos):
    return predizer_delta_ova([atributos[i] for i in m['idx']], m['pesos'])[0]


def _delta_scores(m, atributos):
    return predizer_delta_ova([atributos[i] for i in m['idx']], m['pesos'])[1]


def _perceptron_treinar(treino, idx, taxa=0.03, max_epocas=100, **_):
    pesos, historico, epocas = treinar_perceptron_ova(treino, idx, taxa,
                                                      max_epocas)
    return {'pesos': pesos, 'idx': idx, 'historico': historico,
            'epocas': epocas}


def _perceptron_predizer(m, atributos):
    return predizer_perceptron_ova([atributos[i] for i in m['idx']],
                                   m['pesos'])[0]


def _perceptron_scores(m, atributos):
    return predizer_perceptron_ova([atributos[i] for i in m['idx']],
                                   m['pesos'])[1]


def _floresta_treinar(treino, idx, n_arvores=50, criterio='gini',
                      profundidade_max=0, max_atributos='sqrt',
                      min_amostras_folha=1, semente=42, **_):
    return treinar_floresta(
        treino, idx,
        n_arvores=n_arvores, criterio=criterio,
        # 0 no controle deslizante = "sem limite" para a arvore.
        profundidade_max=profundidade_max or None,
        max_atributos=max_atributos, min_amostras_folha=min_amostras_folha,
        semente=semente)


def _floresta_predizer(m, atributos):
    return m.predizer(atributos)


def _floresta_scores(m, atributos):
    return m.probabilidades(atributos)


def _mlp_treinar(treino, idx, n_ocultos=8, taxa=0.3, epocas=300, semente=42,
                 **_):
    return treinar_mlp_multiclasse(treino, idx, n_ocultos=n_ocultos,
                                   taxa_aprendizado=taxa, epocas=epocas,
                                   semente=semente)


def _mlp_predizer(m, atributos):
    return m.predizer(atributos)


def _mlp_scores(m, atributos):
    return m.saidas(atributos)


# ---------------------------------------------------------------------------
# Catalogo
# ---------------------------------------------------------------------------
MODELOS = {
    'distancia_minima': {
        'nome': 'Distância Mínima',
        'grupo': 'Lineares',
        'descricao': ('Um protótipo (vetor médio) por classe; decide pelo '
                      'maior discriminante linear, o que equivale à menor '
                      'distância euclidiana ao protótipo.'),
        'rotulo_score': 'dⱼ(x)',
        'parametros': [],
        'treinar': _dm_treinar,
        'predizer': _dm_predizer,
        'scores': _dm_scores,
    },
    'perceptron_ova': {
        'nome': 'Perceptron OvA',
        'grupo': 'Lineares',
        'descricao': ('Perceptron de Rosenblatt, um por classe (Um-Contra-'
                      'Todos). Só converge se as classes forem linearmente '
                      'separáveis — caso contrário oscila até o limite de '
                      'épocas.'),
        'rotulo_score': 'net',
        'parametros': [
            {'id': 'taxa', 'rotulo': 'Taxa de aprendizado', 'tipo': 'numero',
             'padrao': 0.03, 'min': 0.001, 'max': 1.0, 'passo': 0.001,
             'ajuda': 'Tamanho da correção aplicada a cada erro.'},
            {'id': 'max_epocas', 'rotulo': 'Máximo de épocas',
             'tipo': 'inteiro', 'padrao': 100, 'min': 1, 'max': 2000,
             'passo': 10,
             'ajuda': 'Limite de passagens pelo treino se não convergir.'},
        ],
        'treinar': _perceptron_treinar,
        'predizer': _perceptron_predizer,
        'scores': _perceptron_scores,
    },
    'delta_ova': {
        'nome': 'Regra Delta OvA',
        'grupo': 'Lineares',
        'descricao': ('Widrow-Hoff (Adaline): minimiza o erro quadrático por '
                      'gradiente descendente. Ao contrário do Perceptron, '
                      'produz uma solução mesmo sem separabilidade linear.'),
        'rotulo_score': 'net',
        'parametros': [
            {'id': 'taxa', 'rotulo': 'Taxa de aprendizado', 'tipo': 'numero',
             'padrao': 0.02, 'min': 0.001, 'max': 1.0, 'passo': 0.001,
             'ajuda': 'Passo do gradiente descendente.'},
            {'id': 'max_epocas', 'rotulo': 'Épocas', 'tipo': 'inteiro',
             'padrao': 200, 'min': 1, 'max': 2000, 'passo': 10,
             'ajuda': 'A Regra Delta não tem parada antecipada: roda todas.'},
        ],
        'treinar': _delta_treinar,
        'predizer': _delta_predizer,
        'scores': _delta_scores,
    },
    'bayes': {
        'nome': 'Bayes Ótimo (QDA)',
        'grupo': 'Probabilísticos',
        'descricao': ('Assume normal multivariada por classe, cada uma com '
                      'sua matriz de covariância — a fronteira resultante é '
                      'quadrática.'),
        'rotulo_score': 'ln p(x|ω)·P(ω)',
        'parametros': [],
        'treinar': _bayes_treinar(False),
        'predizer': _bayes_predizer,
        'scores': _bayes_scores,
    },
    'naive': {
        'nome': 'Naive Bayes',
        'grupo': 'Probabilísticos',
        'descricao': ('Mesma ideia do Bayes ótimo, mas supondo atributos '
                      'independentes: a covariância vira diagonal.'),
        'rotulo_score': 'ln p(x|ω)·P(ω)',
        'parametros': [],
        'treinar': _bayes_treinar(True),
        'predizer': _bayes_predizer,
        'scores': _bayes_scores,
    },
    'floresta': {
        'nome': 'Floresta Aleatória',
        'grupo': 'Seminário',
        'descricao': ('Modelo apresentado no seminário: um comitê de árvores '
                      'CART treinadas em amostras bootstrap, com sorteio de '
                      'atributos em cada nó e decisão por voto majoritário.'),
        'rotulo_score': 'proporção de votos',
        'parametros': [
            {'id': 'n_arvores', 'rotulo': 'Número de árvores',
             'tipo': 'inteiro', 'padrao': 50, 'min': 1, 'max': 300,
             'passo': 1,
             'ajuda': 'Mais árvores reduzem a variância e estabilizam o voto.'},
            {'id': 'criterio', 'rotulo': 'Critério de divisão',
             'tipo': 'opcoes', 'padrao': 'gini',
             'opcoes': [{'valor': 'gini', 'rotulo': 'Índice Gini'},
                        {'valor': 'entropia', 'rotulo': 'Entropia (ganho)'}],
             'ajuda': 'Medida de impureza usada para escolher cada divisão.'},
            {'id': 'profundidade_max', 'rotulo': 'Profundidade máxima',
             'tipo': 'inteiro', 'padrao': 0, 'min': 0, 'max': 20, 'passo': 1,
             'ajuda': '0 = sem limite (árvores crescem até a folha pura).'},
            {'id': 'max_atributos', 'rotulo': 'Atributos sorteados por nó',
             'tipo': 'opcoes', 'padrao': 'sqrt',
             'opcoes': [{'valor': 'sqrt', 'rotulo': '√p (padrão)'},
                        {'valor': 'log2', 'rotulo': 'log₂ p'},
                        {'valor': 'todos', 'rotulo': 'Todos (vira bagging)'}],
             'ajuda': 'É o sorteio que descorrelaciona as árvores.'},
            {'id': 'min_amostras_folha', 'rotulo': 'Mínimo de amostras/folha',
             'tipo': 'inteiro', 'padrao': 1, 'min': 1, 'max': 20, 'passo': 1,
             'ajuda': 'Valores maiores podam a árvore e reduzem sobreajuste.'},
            {'id': 'semente', 'rotulo': 'Semente aleatória', 'tipo': 'inteiro',
             'padrao': 42, 'min': 0, 'max': 9999, 'passo': 1,
             'ajuda': 'Fixa bootstrap e sorteio de atributos — reprodutível.'},
        ],
        'treinar': _floresta_treinar,
        'predizer': _floresta_predizer,
        'scores': _floresta_scores,
    },
    'mlp': {
        'nome': 'Rede Feedforward (MLP)',
        'grupo': 'Redes Neurais',
        'descricao': ('Perceptron multicamadas com backpropagation, escrito '
                      'do zero no Lab 5. Entradas normalizadas em [0,1] e '
                      'saída 1-de-C (um neurônio por classe).'),
        'rotulo_score': 'ativação de saída',
        'parametros': [
            {'id': 'n_ocultos', 'rotulo': 'Neurônios na camada oculta',
             'tipo': 'inteiro', 'padrao': 8, 'min': 1, 'max': 40, 'passo': 1,
             'ajuda': 'Capacidade da rede: poucos não aprendem, muitos '
                      'decoram.'},
            {'id': 'taxa', 'rotulo': 'Taxa de aprendizado', 'tipo': 'numero',
             'padrao': 0.3, 'min': 0.01, 'max': 2.0, 'passo': 0.01,
             'ajuda': 'Passo do gradiente na retropropagação.'},
            {'id': 'epocas', 'rotulo': 'Épocas', 'tipo': 'inteiro',
             'padrao': 300, 'min': 10, 'max': 2000, 'passo': 10,
             'ajuda': 'Passagens completas pelo conjunto de treino.'},
            {'id': 'semente', 'rotulo': 'Semente aleatória', 'tipo': 'inteiro',
             'padrao': 42, 'min': 0, 'max': 9999, 'passo': 1,
             'ajuda': 'Fixa os pesos iniciais — resultado reprodutível.'},
        ],
        'treinar': _mlp_treinar,
        'predizer': _mlp_predizer,
        'scores': _mlp_scores,
    },
}

# Ordem de exibicao na interface e nas comparacoes.
ORDEM = ['distancia_minima', 'perceptron_ova', 'delta_ova', 'bayes', 'naive',
         'mlp', 'floresta']


class ModeloInvalido(ValueError):
    """Id de modelo inexistente ou parametro fora do esquema."""


def info(modelo: str):
    cfg = MODELOS.get(modelo)
    if cfg is None:
        raise ModeloInvalido(
            f"Modelo inválido: '{modelo}'. Use um de: {', '.join(ORDEM)}.")
    return cfg


def catalogo():
    """Catalogo serializavel — o que a interface usa para montar os controles."""
    return [
        {'id': mid, 'nome': MODELOS[mid]['nome'],
         'grupo': MODELOS[mid]['grupo'],
         'descricao': MODELOS[mid]['descricao'],
         'rotulo_score': MODELOS[mid]['rotulo_score'],
         'parametros': MODELOS[mid]['parametros']}
        for mid in ORDEM
    ]


def normalizar_parametros(modelo: str, valores: dict | None):
    """
    Valida os parametros recebidos contra o esquema do modelo.

    Chaves desconhecidas sao ignoradas; ausentes recebem o padrao; numeros
    fora da faixa sao trazidos para dentro dela (em vez de derrubar a
    requisicao — a interface ja limita os controles).
    """
    valores = valores or {}
    saida = {}
    for p in info(modelo)['parametros']:
        bruto = valores.get(p['id'], p['padrao'])
        if p['tipo'] == 'opcoes':
            validos = [o['valor'] for o in p['opcoes']]
            saida[p['id']] = bruto if bruto in validos else p['padrao']
            continue
        try:
            numero = float(bruto)
        except (TypeError, ValueError):
            numero = float(p['padrao'])
        numero = max(p['min'], min(p['max'], numero))
        saida[p['id']] = int(round(numero)) if p['tipo'] == 'inteiro' else numero
    return saida


def treinar_modelo(modelo: str, treino, idx, parametros: dict | None = None):
    """Treina um modelo do catalogo. Retorna (objeto, parametros_normalizados)."""
    cfg = info(modelo)
    params = normalizar_parametros(modelo, parametros)
    return cfg['treinar'](treino, idx, **params), params


def predizer(modelo: str, objeto, atributos):
    return info(modelo)['predizer'](objeto, atributos)


def scores(modelo: str, objeto, atributos):
    """Pontuacao por classe (discriminante, net, log-verossimilhanca, voto…)."""
    return info(modelo)['scores'](objeto, atributos)


# ---------------------------------------------------------------------------
# Predicoes de todos os modelos — base das comparacoes
# ---------------------------------------------------------------------------
@lru_cache(maxsize=32)
def predicoes_de_todos(dataset: str, atributos: str, proporcao: float = 0.7):
    """
    Treina TODOS os modelos do catalogo (com os parametros padrao) no mesmo
    split e devolve {id: (nome, [predicoes do teste])}.

    O pareamento sobre o MESMO conjunto de teste e o que autoriza McNemar e o
    bootstrap pareado: os modelos acertam e erram as mesmas amostras difíceis.

    O resultado e cacheado por (dataset, atributos, proporcao) — sem isso, a
    matriz de significancia retreinaria os sete modelos para cada par.
    """
    _, treino, teste = obter_split(dataset, proporcao)
    idx = indices_de(atributos, dataset)

    preds = {}
    for mid in ORDEM:
        objeto, _ = treinar_modelo(mid, treino, idx)
        preds[mid] = (MODELOS[mid]['nome'],
                      [predizer(mid, objeto, d['atributos']) for d in teste])
    return preds


def esquecer_predicoes():
    """Descarta o cache de predicoes (usado quando os dados mudam)."""
    predicoes_de_todos.cache_clear()
