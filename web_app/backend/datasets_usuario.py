"""
Registro das bases de dados enviadas pelo usuario (arquivos .txt).

Atende ao requisito "o aplicativo devera ser alimentado pela base de dados do
usuario, no formato txt": o arquivo enviado pela interface e gravado em
`data/enviados/`, junto de um `.json` com a configuracao de leitura escolhida
(delimitador, cabecalho, coluna de classe, colunas ignoradas). A partir dai
ele vira um dataset como qualquer outro — aparece no mesmo seletor do Iris e
do dataset do seminario, e roda em todas as telas.

Por que gravar em disco
-----------------------
Manter so em memoria faria a base sumir a cada `--reload` do uvicorn. Gravando
o texto original mais a configuracao, a base volta sozinha quando o servidor
reinicia, e o usuario nao precisa reenviar o arquivo no meio da defesa.

Formato de um registro (`data/enviados/usr_ab12cd34.json`)
---------------------------------------------------------
    {
      "id": "usr_ab12cd34",
      "nome": "Vinho (UCI)",
      "arquivo_original": "wine.txt",
      "criado_em": "2026-08-20T21:15:03",
      "leitura": {"delimitador": "virgula", "cabecalho": "nao",
                  "coluna_classe": 0, "colunas_ignoradas": []}
    }

Este modulo nao importa `core` — a dependencia e de mao unica (`core` conhece
os datasets do usuario, nao o contrario), o que evita import circular.
"""
import hashlib
import json
import os
import sys
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IRIS_DIR = os.path.join(BASE_DIR, 'iris_classifier')
for _p in (IRIS_DIR, BASE_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from data.leitor_texto import ErroLeitura, carregar  # noqa: E402

PASTA = os.path.join(BASE_DIR, 'data', 'enviados')

# Limite de bases guardadas ao mesmo tempo — a pasta e um espaco de trabalho,
# nao um repositorio.
MAX_BASES = 20

# Quantos pares de atributos gerar no menu "Atributos". Com muitas features o
# menu viraria uma lista interminavel de combinacoes.
MAX_FEATURES_EM_PARES = 4

_cache = None   # {id: cfg} — montado na primeira consulta


# ---------------------------------------------------------------------------
# Combinacoes de atributos oferecidas na interface
# ---------------------------------------------------------------------------
def _combinacoes(features):
    """
    Monta o menu "Atributos" de uma base do usuario.

    Todas as telas 2D do projeto (dispersao, superficies de decisao, regioes)
    trabalham com um par de features; as telas que aceitam mais dimensoes usam
    a entrada 'todos'. Geramos os pares entre as primeiras
    `MAX_FEATURES_EM_PARES` colunas e a combinacao completa.
    """
    n = len(features)
    combos = {}
    limite = min(n, MAX_FEATURES_EM_PARES)
    for i in range(limite):
        for j in range(i + 1, limite):
            combos[f'par_{i}_{j}'] = {
                'indices': [i, j],
                'nome': f'{features[i]} × {features[j]}',
                'eixo_x': features[i],
                'eixo_y': features[j],
            }
    if n > 2:
        combos['todos'] = {
            'indices': list(range(n)),
            'nome': f'Todos os {n} atributos',
            'eixo_x': features[0],
            'eixo_y': features[1],
        }
    return combos


def _padrao(combos):
    return 'par_0_1' if 'par_0_1' in combos else next(iter(combos))


# ---------------------------------------------------------------------------
# Persistencia
# ---------------------------------------------------------------------------
def _garantir_pasta():
    os.makedirs(PASTA, exist_ok=True)


def _caminho_txt(id_dataset):
    return os.path.join(PASTA, f'{id_dataset}.txt')


def _caminho_json(id_dataset):
    return os.path.join(PASTA, f'{id_dataset}.json')


def _novo_id(nome, conteudo):
    digest = hashlib.sha1(
        (nome + conteudo[:4096] + datetime.now().isoformat()).encode('utf-8')
    ).hexdigest()[:8]
    return f'usr_{digest}'


def _cfg_de(meta, base):
    """
    Traduz (metadados gravados + resultado da leitura) na mesma estrutura dos
    datasets nativos do `core.DATASETS`.
    """
    features = base['features']
    combos = _combinacoes(features)
    return {
        'id': meta['id'],
        'nome': meta['nome'],
        'descricao': (f"Base do usuário · {base['n_amostras']} amostras · "
                      f"{len(features)} atributos · {len(base['classes'])} classes"),
        'tipo': base['tipo'],
        'origem': 'usuario',
        'caminho': _caminho_txt(meta['id']),
        'classes': base['classes'],
        'features': features,
        'atributos': combos,
        'atributos_padrao': _padrao(combos),
        # `valores` chega do leitor com chaves int (posicao no vetor de
        # atributos) — o mesmo formato do dataset categorico do seminario.
        'valores': base['valores'],
        'leitura': meta['leitura'],
        'arquivo_original': meta.get('arquivo_original', ''),
        'criado_em': meta.get('criado_em', ''),
        'avisos': base['avisos'],
        'n_amostras': base['n_amostras'],
        'coluna_classe': base['coluna_classe'],
        'nome_coluna_classe': base['nome_coluna_classe'],
    }


def _ler_registro(id_dataset):
    """Le o par .json/.txt do disco e devolve a configuracao do dataset."""
    with open(_caminho_json(id_dataset), encoding='utf-8') as f:
        meta = json.load(f)
    with open(_caminho_txt(id_dataset), encoding='utf-8') as f:
        texto = f.read()
    base = carregar(texto, **meta['leitura'])
    return _cfg_de(meta, base)


def registrados():
    """
    Todas as bases enviadas, no formato do registro `core.DATASETS`.

    Bases cujo arquivo tenha sido apagado ou corrompido sao simplesmente
    ignoradas — a interface nunca quebra por causa de um upload antigo.
    """
    global _cache
    if _cache is not None:
        return _cache

    _cache = {}
    if not os.path.isdir(PASTA):
        return _cache

    for arquivo in sorted(os.listdir(PASTA)):
        if not arquivo.endswith('.json'):
            continue
        id_dataset = arquivo[:-5]
        try:
            _cache[id_dataset] = _ler_registro(id_dataset)
        except (OSError, ValueError, KeyError):
            continue
    return _cache


def importar(nome, conteudo, delimitador='auto', cabecalho='auto',
             coluna_classe=None, colunas_ignoradas=()):
    """
    Valida, grava e registra uma base enviada. Devolve a configuracao criada.

    A validacao e a propria leitura: se `leitor_texto.carregar` levantar
    `ErroLeitura`, nada e gravado e a mensagem sobe para a interface.
    """
    nome = (nome or '').strip() or 'Base do usuário'
    base = carregar(conteudo, delimitador=delimitador, cabecalho=cabecalho,
                    coluna_classe=coluna_classe,
                    colunas_ignoradas=colunas_ignoradas)

    atuais = registrados()
    if len(atuais) >= MAX_BASES:
        raise ErroLeitura(
            f'Limite de {MAX_BASES} bases enviadas atingido. Remova alguma '
            'antes de importar outra.')

    id_dataset = _novo_id(nome, conteudo)
    meta = {
        'id': id_dataset,
        'nome': nome,
        'arquivo_original': '',
        'criado_em': datetime.now().isoformat(timespec='seconds'),
        # Grava a configuracao JA RESOLVIDA (e nao 'auto'), para que releituras
        # futuras produzam exatamente a mesma base, mesmo que a heuristica de
        # deteccao mude.
        'leitura': {
            'delimitador': base['delimitador'],
            'cabecalho': 'sim' if base['cabecalho'] else 'nao',
            'coluna_classe': base['coluna_classe'],
            'colunas_ignoradas': base['colunas_ignoradas'],
        },
    }

    _garantir_pasta()
    with open(_caminho_txt(id_dataset), 'w', encoding='utf-8') as f:
        f.write(conteudo)
    with open(_caminho_json(id_dataset), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    cfg = _cfg_de(meta, base)
    atuais[id_dataset] = cfg
    return cfg


def renomear(id_dataset, nome):
    """Troca o nome exibido de uma base enviada."""
    cfg = registrados().get(id_dataset)
    if cfg is None:
        raise KeyError(id_dataset)
    nome = (nome or '').strip()
    if not nome:
        raise ErroLeitura('O nome da base não pode ficar vazio.')
    with open(_caminho_json(id_dataset), encoding='utf-8') as f:
        meta = json.load(f)
    meta['nome'] = nome
    with open(_caminho_json(id_dataset), 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    cfg['nome'] = nome
    return cfg


def remover(id_dataset):
    """Apaga a base enviada (arquivo e metadados). Idempotente."""
    atuais = registrados()
    atuais.pop(id_dataset, None)
    for caminho in (_caminho_txt(id_dataset), _caminho_json(id_dataset)):
        try:
            os.remove(caminho)
        except OSError:
            pass


def carregar_dados(cfg):
    """Le as amostras de uma base enviada, aplicando a configuracao gravada."""
    with open(cfg['caminho'], encoding='utf-8') as f:
        texto = f.read()
    return carregar(texto, **cfg['leitura'])['dados']


def esquecer_cache():
    """Descarta o cache em memoria — o proximo acesso rele o disco."""
    global _cache
    _cache = None
