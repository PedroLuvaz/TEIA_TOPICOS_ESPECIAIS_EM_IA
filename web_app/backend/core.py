"""
Nucleo compartilhado do backend web.

Centraliza o carregamento dos dados, as configuracoes de atributos/datasets
e um cache simples em memoria — de modo que os routers apenas orquestrem
chamadas aos modelos em Python puro de `iris_classifier/`.

Nenhuma matematica e reimplementada aqui: tudo e delegado aos modulos ja
existentes do projeto (regra do CLAUDE.md — Python puro, sem numpy).

Multiplos datasets
------------------
Cada dataset declara suas proprias classes, features e combinacoes de
atributos no registro `DATASETS`. Os routers nunca assumem "as 3 classes do
Iris": pedem `classes_de(dataset)`, `config_atributos_de(dataset)` e assim
por diante. E o que permite o dataset categorico do seminario (fim de semana,
4 classes, 3 atributos) rodar nas mesmas telas do Iris.
"""
import os
import random
import sys
from functools import lru_cache

# --- PATH: permitir importar os pacotes internos de iris_classifier/ -------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IRIS_DIR = os.path.join(BASE_DIR, 'iris_classifier')
for _p in (IRIS_DIR, BASE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.data_loader import (carregar_dados_iris,  # noqa: E402
                              carregar_fim_de_semana, split_estratificado)

# ---------------------------------------------------------------------------
# Configuracao de atributos do Iris (espelha a da GUI Tkinter)
# ---------------------------------------------------------------------------
_ATRIBUTOS_IRIS = {
    'petalas': {
        'indices': [2, 3],
        'nome': 'Petalas',
        'eixo_x': 'Comprimento da Petala (cm)',
        'eixo_y': 'Largura da Petala (cm)',
    },
    'sepalas': {
        'indices': [0, 1],
        'nome': 'Sepalas',
        'eixo_x': 'Comprimento da Sepala (cm)',
        'eixo_y': 'Largura da Sepala (cm)',
    },
    'todas': {
        'indices': [0, 1, 2, 3],
        'nome': 'Todas as 4 features',
        'eixo_x': 'Comprimento da Petala (cm)',
        'eixo_y': 'Largura da Petala (cm)',
    },
}

_FEATURES_IRIS = [
    'Comprimento da Sepala', 'Largura da Sepala',
    'Comprimento da Petala', 'Largura da Petala',
]

_CLASSES_IRIS = ['setosa', 'versicolor', 'virginica']

# ---------------------------------------------------------------------------
# Dataset do seminario: fim de semana (categorico, codificado em ordinais)
# ---------------------------------------------------------------------------
_EIXO_CLIMA = 'Clima  (0=Sol · 1=Vento · 2=Chuva)'
_EIXO_PAIS = 'Pais visitam?  (0=Nao · 1=Sim)'
_EIXO_DINHEIRO = 'Dinheiro  (0=Pobre · 1=Rico)'

_ATRIBUTOS_FDS = {
    'clima_pais': {
        'indices': [0, 1],
        'nome': 'Clima x Pais',
        'eixo_x': _EIXO_CLIMA,
        'eixo_y': _EIXO_PAIS,
    },
    'clima_dinheiro': {
        'indices': [0, 2],
        'nome': 'Clima x Dinheiro',
        'eixo_x': _EIXO_CLIMA,
        'eixo_y': _EIXO_DINHEIRO,
    },
    'pais_dinheiro': {
        'indices': [1, 2],
        'nome': 'Pais x Dinheiro',
        'eixo_x': _EIXO_PAIS,
        'eixo_y': _EIXO_DINHEIRO,
    },
    'todos': {
        'indices': [0, 1, 2],
        'nome': 'Todos os 3 atributos',
        'eixo_x': _EIXO_CLIMA,
        'eixo_y': _EIXO_PAIS,
    },
}

_FEATURES_FDS = ['Clima', 'Pais visitam?', 'Dinheiro']
_CLASSES_FDS = ['Cinema', 'Compras', 'Ficar em casa', 'Tenis']

# Rotulos dos valores categoricos, por indice de atributo — usados nos
# eixos e nas memorias de calculo para nao mostrar so 0/1/2.
_VALORES_FDS = {
    0: ['Sol', 'Vento', 'Chuva'],
    1: ['Nao', 'Sim'],
    2: ['Pobre', 'Rico'],
}


def _pares(classes):
    """Todos os pares (i, j) com i < j — para os classificadores binarios."""
    return [(classes[i], classes[j])
            for i in range(len(classes))
            for j in range(i + 1, len(classes))]


# ---------------------------------------------------------------------------
# Registro de datasets
# ---------------------------------------------------------------------------
DATASETS = {
    'v1': {
        'id': 'v1',
        'nome': 'Iris Original',
        'descricao': 'Base classica de Fisher (150 amostras)',
        'tipo': 'continuo',
        'caminho': os.path.join(BASE_DIR, 'data', 'Iris data.xls'),
        'classes': _CLASSES_IRIS,
        'features': _FEATURES_IRIS,
        'atributos': _ATRIBUTOS_IRIS,
        'atributos_padrao': 'petalas',
        'valores': None,
    },
    'v2': {
        'id': 'v2',
        'nome': 'Iris Separavel',
        'descricao': 'Variante linearmente separavel (v2)',
        'tipo': 'continuo',
        'caminho': os.path.join(BASE_DIR, 'data', 'iris_data_02.xlsx'),
        'classes': _CLASSES_IRIS,
        'features': _FEATURES_IRIS,
        'atributos': _ATRIBUTOS_IRIS,
        'atributos_padrao': 'petalas',
        'valores': None,
    },
    'fds': {
        'id': 'fds',
        'nome': 'Fim de Semana (seminario)',
        'descricao': ('Dataset categorico do seminario de Florestas '
                      'Aleatorias, com 1000 instancias e 8% de ruido'),
        'tipo': 'categorico',
        'caminho': os.path.join(BASE_DIR, 'data', 'fim_de_semana_1000.csv'),
        'classes': _CLASSES_FDS,
        'features': _FEATURES_FDS,
        'atributos': _ATRIBUTOS_FDS,
        'atributos_padrao': 'clima_pais',
        'valores': _VALORES_FDS,
    },
}

DATASET_PADRAO = 'v1'

# Compatibilidade: alguns pontos do projeto (Lab 5, exercicios fixos do Iris)
# so fazem sentido no Iris e continuam usando estas constantes.
CLASSES = _CLASSES_IRIS
NOMES_FEATURES = _FEATURES_IRIS
CONFIG_ATRIBUTOS = _ATRIBUTOS_IRIS
PARES_CLASSES = _pares(_CLASSES_IRIS)
CAMINHOS_DADOS = {k: v['caminho'] for k, v in DATASETS.items()}


# ---------------------------------------------------------------------------
# Acesso ao registro
# ---------------------------------------------------------------------------
def info(dataset: str = DATASET_PADRAO):
    """Metadados do dataset. Levanta ValueError se o id nao existir."""
    cfg = DATASETS.get(dataset)
    if cfg is None:
        raise ValueError(
            f"Dataset invalido: '{dataset}'. Use um de: "
            f"{', '.join(sorted(DATASETS))}.")
    return cfg


def classes_de(dataset: str = DATASET_PADRAO):
    return info(dataset)['classes']


def features_de(dataset: str = DATASET_PADRAO):
    return info(dataset)['features']


def config_atributos_de(dataset: str = DATASET_PADRAO):
    return info(dataset)['atributos']


def pares_de(dataset: str = DATASET_PADRAO):
    return _pares(classes_de(dataset))


def eh_categorico(dataset: str = DATASET_PADRAO):
    return info(dataset)['tipo'] == 'categorico'


def rotulo_valor(dataset: str, atributo: int, valor):
    """
    Traduz o codigo numerico de um atributo categorico no rotulo original.

    Ex.: rotulo_valor('fds', 0, 1.0) -> 'Vento'. Para datasets continuos (ou
    valores fora da tabela) devolve o proprio numero formatado.
    """
    tabela = info(dataset)['valores']
    if tabela and atributo in tabela:
        try:
            indice = int(round(float(valor)))
        except (TypeError, ValueError):
            return str(valor)
        if 0 <= indice < len(tabela[atributo]):
            return tabela[atributo][indice]
    return f'{valor:.2f}' if isinstance(valor, float) else str(valor)


# ---------------------------------------------------------------------------
# Carregamento e split (cacheados — o arquivo so e lido uma vez por dataset)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=8)
def carregar(dataset: str):
    """Carrega e devolve a lista completa de amostras do dataset informado."""
    cfg = info(dataset)
    caminho = cfg['caminho']
    if not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Dataset '{dataset}' nao encontrado. Esperado em: {caminho}")

    if cfg['tipo'] == 'categorico':
        # numerico=True: os codigos ordinais das colunas `_cod`, para que os
        # classificadores numericos do projeto consumam o mesmo arquivo.
        return carregar_fim_de_semana(caminho, numerico=True)
    return carregar_dados_iris(caminho)


@lru_cache(maxsize=64)
def _split_cache(dataset: str, proporcao: float, semente: int):
    dados = carregar(dataset)
    treino, teste = split_estratificado(
        dados, proporcao_treino=proporcao, semente=semente)
    return treino, teste


def obter_split(dataset: str = DATASET_PADRAO, proporcao: float = 0.7,
                semente: int = 42):
    """
    Retorna (dados_completos, treino, teste) com o mesmo split estratificado
    usado na GUI desktop — garantindo resultados identicos entre as interfaces.
    """
    treino, teste = _split_cache(dataset, proporcao, semente)
    return carregar(dataset), treino, teste


def indices_de(atributos: str, dataset: str = DATASET_PADRAO):
    """Traduz a chave de atributos nos indices das features do dataset."""
    cfg = config_atributos_de(dataset).get(atributos)
    if cfg is None:
        raise ValueError(
            f"Conjunto de atributos invalido para '{dataset}': '{atributos}'. "
            f"Use um de: {', '.join(sorted(config_atributos_de(dataset)))}.")
    return cfg['indices']


def indices_plot(atributos: str, dataset: str = DATASET_PADRAO):
    """
    Indices usados para plotar em 2D.

    Quando o conjunto tem mais de 2 atributos, projeta nos dois primeiros do
    par padrao do dataset (petalas no Iris, Clima x Pais no fim de semana).
    """
    idx = indices_de(atributos, dataset)
    if len(idx) <= 2:
        return idx
    padrao = info(dataset)['atributos_padrao']
    return config_atributos_de(dataset)[padrao]['indices'][:2]


def config_de(atributos: str, dataset: str = DATASET_PADRAO):
    """Configuracao completa (nome, indices, rotulos dos eixos)."""
    cfg = config_atributos_de(dataset).get(atributos)
    if cfg is None:
        raise ValueError(
            f"Conjunto de atributos invalido para '{dataset}': '{atributos}'.")
    return cfg


# ---------------------------------------------------------------------------
# Serializacao para o frontend
# ---------------------------------------------------------------------------
def serializar_amostras(dados, indices, dados_treino=None, jitter=0.0):
    """
    Converte amostras para JSON, marcando quais pertencem ao conjunto de treino.
    Usa identidade de objeto (id) — o split devolve as mesmas instancias.

    `jitter` desloca aleatoriamente os pontos no grafico. Serve so para os
    datasets categoricos: com 3 atributos discretos, 1000 amostras cairiam
    sobre 12 posicoes exatas e o grafico viraria 12 pontos. O deslocamento e
    deterministico (semente fixa) para nao "tremer" a cada renderizacao, e
    afeta apenas x/y — `atributos` continua com os valores originais.
    """
    ids_treino = set(id(d) for d in (dados_treino or []))
    rng = random.Random(42) if jitter else None

    saida = []
    for d in dados:
        x = d['atributos'][indices[0]]
        y = d['atributos'][indices[1]]
        if rng:
            x += rng.uniform(-jitter, jitter)
            y += rng.uniform(-jitter, jitter)
        saida.append({
            'x': x,
            'y': y,
            'classe': d['classe'],
            'treino': id(d) in ids_treino,
            'atributos': d['atributos'],
        })
    return saida


def jitter_de(dataset: str = DATASET_PADRAO):
    """Deslocamento recomendado no scatter — zero para datasets continuos."""
    return 0.22 if eh_categorico(dataset) else 0.0


def limites_com_margem(dados, indices, margem=0.5):
    """Bounding box dos dados com uma margem, para desenhar fronteiras."""
    xs = [d['atributos'][indices[0]] for d in dados]
    ys = [d['atributos'][indices[1]] for d in dados]
    return {
        'x_min': min(xs) - margem, 'x_max': max(xs) + margem,
        'y_min': min(ys) - margem, 'y_max': max(ys) + margem,
    }


def malha(limites, resolucao=90):
    """Gera os eixos de uma grade regular dentro dos limites informados."""
    passo_x = (limites['x_max'] - limites['x_min']) / (resolucao - 1)
    passo_y = (limites['y_max'] - limites['y_min']) / (resolucao - 1)
    eixo_x = [limites['x_min'] + k * passo_x for k in range(resolucao)]
    eixo_y = [limites['y_min'] + k * passo_y for k in range(resolucao)]
    return eixo_x, eixo_y
