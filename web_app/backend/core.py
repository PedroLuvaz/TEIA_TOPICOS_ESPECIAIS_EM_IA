"""
Nucleo compartilhado do backend web.

Centraliza o carregamento dos dados, as configuracoes de atributos/datasets
e um cache simples em memoria — de modo que os routers apenas orquestrem
chamadas aos modelos em Python puro de `iris_classifier/`.

Nenhuma matematica e reimplementada aqui: tudo e delegado aos modulos ja
existentes do projeto (regra do CLAUDE.md — Python puro, sem numpy).
"""
import os
import sys
from functools import lru_cache

# --- PATH: permitir importar os pacotes internos de iris_classifier/ -------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IRIS_DIR = os.path.join(BASE_DIR, 'iris_classifier')
for _p in (IRIS_DIR, BASE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.data_loader import carregar_dados_iris, split_estratificado  # noqa: E402

# ---------------------------------------------------------------------------
# Constantes de dominio (espelham as usadas na GUI Tkinter)
# ---------------------------------------------------------------------------
CLASSES = ['setosa', 'versicolor', 'virginica']

CAMINHOS_DADOS = {
    'v1': os.path.join(BASE_DIR, 'data', 'Iris data.xls'),
    'v2': os.path.join(BASE_DIR, 'data', 'iris_data_02.xlsx'),
}

DATASETS = [
    {'id': 'v1', 'nome': 'Iris Original', 'descricao': 'Base classica de Fisher (150 amostras)'},
    {'id': 'v2', 'nome': 'Iris Separavel', 'descricao': 'Variante linearmente separavel (v2)'},
]

CONFIG_ATRIBUTOS = {
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

NOMES_FEATURES = [
    'Comprimento da Sepala', 'Largura da Sepala',
    'Comprimento da Petala', 'Largura da Petala',
]

PARES_CLASSES = [
    ('setosa', 'versicolor'),
    ('setosa', 'virginica'),
    ('versicolor', 'virginica'),
]


# ---------------------------------------------------------------------------
# Carregamento e split (cacheados — o .xls so e lido uma vez por dataset)
# ---------------------------------------------------------------------------
@lru_cache(maxsize=4)
def carregar(dataset: str):
    """Carrega e devolve a lista completa de amostras do dataset informado."""
    caminho = CAMINHOS_DADOS.get(dataset)
    if not caminho or not os.path.exists(caminho):
        raise FileNotFoundError(
            f"Dataset '{dataset}' nao encontrado. Esperado em: {caminho}")
    return carregar_dados_iris(caminho)


@lru_cache(maxsize=32)
def _split_cache(dataset: str, proporcao: float, semente: int):
    dados = carregar(dataset)
    treino, teste = split_estratificado(
        dados, proporcao_treino=proporcao, semente=semente)
    return treino, teste


def obter_split(dataset: str = 'v1', proporcao: float = 0.7, semente: int = 42):
    """
    Retorna (dados_completos, treino, teste) com o mesmo split estratificado
    usado na GUI desktop — garantindo resultados identicos entre as interfaces.
    """
    treino, teste = _split_cache(dataset, proporcao, semente)
    return carregar(dataset), treino, teste


def indices_de(atributos: str):
    """Traduz a chave de atributos ('petalas'/'sepalas'/'todas') nos indices."""
    cfg = CONFIG_ATRIBUTOS.get(atributos)
    if cfg is None:
        raise ValueError(f"Conjunto de atributos invalido: '{atributos}'")
    return cfg['indices']


def indices_plot(atributos: str):
    """
    Indices usados para plotar em 2D. Para 'todas' (4D) projetamos nas
    petalas, que sao as features mais discriminantes — mesma convencao da GUI.
    """
    return [2, 3] if atributos == 'todas' else indices_de(atributos)


def serializar_amostras(dados, indices, dados_treino=None):
    """
    Converte amostras para JSON, marcando quais pertencem ao conjunto de treino.
    Usa identidade de objeto (id) — o split devolve as mesmas instancias.
    """
    ids_treino = set(id(d) for d in (dados_treino or []))
    return [
        {
            'x': d['atributos'][indices[0]],
            'y': d['atributos'][indices[1]],
            'classe': d['classe'],
            'treino': id(d) in ids_treino,
            'atributos': d['atributos'],
        }
        for d in dados
    ]


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
