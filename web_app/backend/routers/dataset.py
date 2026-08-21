"""Rotas de metadados, exploracao e IMPORTACAO dos datasets."""
import os

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from .. import datasets_usuario, modelos
from ..core import (DATASET_PADRAO, carregar, classes_de,
                    config_atributos_de, config_de, features_de, indices_plot,
                    invalidar_cache, jitter_de, obter_split, pares_de,
                    serializar_amostras, todos)

# Importado depois de `..core`/`datasets_usuario`: sao eles que colocam
# `iris_classifier/` no sys.path.
from data.leitor_texto import (DELIMITADORES, MAX_LINHAS,  # noqa: E402
                               ROTULOS_DELIMITADOR, ErroLeitura, analisar)

router = APIRouter(prefix='/api/dataset', tags=['dataset'])

# O conteudo do .txt viaja como string dentro do JSON (o projeto nao depende
# de `python-multipart`). O teto de caracteres evita que um arquivo enorme
# derrube o servidor antes mesmo de o leitor recusa-lo pelo numero de linhas.
MAX_CARACTERES = 8_000_000


def _atributos_serializados(dataset):
    return [
        {'id': k, 'nome': v['nome'], 'indices': v['indices'],
         'eixo_x': v['eixo_x'], 'eixo_y': v['eixo_y']}
        for k, v in config_atributos_de(dataset).items()
    ]


@router.get('/metadata')
def metadata():
    """
    Opcoes disponiveis na interface.

    Cada dataset traz suas proprias classes, features e combinacoes de
    atributos — o frontend nao assume as 3 classes do Iris em lugar nenhum.
    As chaves de topo (`classes`, `features`, ...) descrevem o dataset padrao
    e existem para o codigo que ainda nao seleciona dataset (Lab 5).
    """
    datasets = []
    for cfg in todos().values():
        existe = os.path.exists(cfg['caminho'])
        # O numero de amostras muda o custo da validacao cruzada, entao a
        # interface precisa dele para escolher um numero de repeticoes sensato.
        try:
            n_amostras = len(carregar(cfg['id'])) if existe else 0
        except Exception:
            n_amostras = 0
        datasets.append({
            'id': cfg['id'],
            'nome': cfg['nome'],
            'descricao': cfg['descricao'],
            'tipo': cfg['tipo'],
            'disponivel': existe,
            'n_amostras': n_amostras,
            'classes': cfg['classes'],
            'features': cfg['features'],
            'atributos': _atributos_serializados(cfg['id']),
            'atributos_padrao': cfg['atributos_padrao'],
            'pares': [{'pos': a, 'neg': b} for a, b in pares_de(cfg['id'])],
            'valores': cfg['valores'],
            # Campos exclusivos das bases enviadas pelo usuario.
            'origem': cfg.get('origem', 'projeto'),
            'arquivo_original': cfg.get('arquivo_original', ''),
            'criado_em': cfg.get('criado_em', ''),
            'avisos': cfg.get('avisos', []),
            'coluna_classe': cfg.get('nome_coluna_classe', ''),
        })

    return {
        'datasets': datasets,
        'dataset_padrao': DATASET_PADRAO,
        'atributos': _atributos_serializados(DATASET_PADRAO),
        'classes': classes_de(DATASET_PADRAO),
        'features': features_de(DATASET_PADRAO),
        'pares': [{'pos': a, 'neg': b} for a, b in pares_de(DATASET_PADRAO)],
    }


@router.get('/amostras')
def amostras(dataset: str = 'v1', atributos: str = 'petalas',
             proporcao: float = Query(0.7, ge=0.1, le=0.9),
             semente: int = 42):
    """Amostras projetadas em 2D, ja marcadas como treino ou teste."""
    try:
        dados, treino, teste = obter_split(dataset, proporcao, semente)
        idx = indices_plot(atributos, dataset)
        cfg = config_de(atributos, dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        'amostras': serializar_amostras(dados, idx, treino,
                                        jitter=jitter_de(dataset)),
        'total': len(dados),
        'n_treino': len(treino),
        'n_teste': len(teste),
        'eixo_x': cfg['eixo_x'],
        'eixo_y': cfg['eixo_y'],
    }


@router.get('/estatisticas')
def estatisticas(dataset: str = 'v1'):
    """Media, desvio, minimo e maximo de cada feature, por classe."""
    try:
        dados, _, _ = obter_split(dataset)
        nomes_features = features_de(dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    resumo = {}
    for classe in classes_de(dataset):
        amostras_c = [d['atributos'] for d in dados if d['classe'] == classe]
        n = len(amostras_c)
        por_feature = []
        for j, nome in enumerate(nomes_features):
            valores = [a[j] for a in amostras_c]
            media = sum(valores) / n if n else 0.0
            variancia = sum((v - media) ** 2 for v in valores) / (n - 1) if n > 1 else 0.0
            por_feature.append({
                'feature': nome,
                'media': media,
                'desvio': variancia ** 0.5,
                'minimo': min(valores) if valores else 0.0,
                'maximo': max(valores) if valores else 0.0,
            })
        resumo[classe] = {'n': n, 'features': por_feature}
    return {'por_classe': resumo, 'features': nomes_features}


# ===========================================================================
# Importacao da base do usuario (.txt)
# ===========================================================================
class LeituraRequest(BaseModel):
    """Conteudo do arquivo mais as opcoes de leitura escolhidas na interface."""
    conteudo: str = Field(..., description='Texto integral do arquivo enviado')
    delimitador: str = 'auto'
    cabecalho: str = 'auto'                 # 'auto' | 'sim' | 'nao'
    coluna_classe: int | None = None
    colunas_ignoradas: list[int] = []


class ImportarRequest(LeituraRequest):
    nome: str = 'Base do usuário'
    arquivo_original: str = ''


class RenomearRequest(BaseModel):
    nome: str


def _validar_tamanho(conteudo: str):
    if len(conteudo) > MAX_CARACTERES:
        raise HTTPException(
            status_code=413,
            detail=f'Arquivo grande demais ({len(conteudo) // 1024} KB). O '
                   f'limite é {MAX_CARACTERES // 1024} KB / {MAX_LINHAS} linhas.')


@router.get('/opcoes-leitura')
def opcoes_leitura():
    """Delimitadores aceitos e limites da importacao — alimenta o formulario."""
    return {
        'delimitadores': [{'id': k, 'nome': ROTULOS_DELIMITADOR[k]}
                          for k in DELIMITADORES],
        'max_linhas': MAX_LINHAS,
        'max_caracteres': MAX_CARACTERES,
    }


@router.post('/analisar')
def analisar_arquivo(req: LeituraRequest):
    """
    Pre-visualiza o arquivo SEM importa-lo.

    Devolve o delimitador detectado, se ha cabecalho, o perfil de cada coluna
    (nome, tipo, quantos valores distintos) e um palpite da coluna de classe.
    E o que permite ao usuario conferir a leitura antes de confirmar.
    """
    _validar_tamanho(req.conteudo)
    try:
        return analisar(req.conteudo, req.delimitador, req.cabecalho)
    except ErroLeitura as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post('/importar')
def importar_arquivo(req: ImportarRequest):
    """
    Importa a base do usuario e a registra como um dataset do aplicativo.

    A partir daqui ela aparece no seletor "Base de dados" de todas as telas —
    classificacao, metricas, testes de significancia e o modelo do seminario.
    """
    _validar_tamanho(req.conteudo)
    try:
        cfg = datasets_usuario.importar(
            req.nome, req.conteudo,
            delimitador=req.delimitador, cabecalho=req.cabecalho,
            coluna_classe=req.coluna_classe,
            colunas_ignoradas=req.colunas_ignoradas)
    except ErroLeitura as e:
        raise HTTPException(status_code=400, detail=str(e))
    except OSError as e:
        raise HTTPException(status_code=500,
                            detail=f'Falha ao gravar o arquivo: {e}')

    invalidar_cache()
    modelos.esquecer_predicoes()
    return {
        'id': cfg['id'],
        'nome': cfg['nome'],
        'descricao': cfg['descricao'],
        'tipo': cfg['tipo'],
        'classes': cfg['classes'],
        'features': cfg['features'],
        'n_amostras': cfg['n_amostras'],
        'coluna_classe': cfg['nome_coluna_classe'],
        'atributos_padrao': cfg['atributos_padrao'],
        'avisos': cfg['avisos'],
        'leitura': cfg['leitura'],
    }


@router.get('/enviados')
def listar_enviados():
    """Bases .txt ja importadas, com a configuracao de leitura de cada uma."""
    return {'datasets': [
        {'id': c['id'], 'nome': c['nome'], 'descricao': c['descricao'],
         'tipo': c['tipo'], 'n_amostras': c['n_amostras'],
         'classes': c['classes'], 'features': c['features'],
         'coluna_classe': c['nome_coluna_classe'],
         'arquivo_original': c['arquivo_original'],
         'criado_em': c['criado_em'], 'leitura': c['leitura'],
         'avisos': c['avisos']}
        for c in datasets_usuario.registrados().values()
    ]}


@router.patch('/enviados/{id_dataset}')
def renomear_enviado(id_dataset: str, req: RenomearRequest):
    """Renomeia uma base importada."""
    try:
        cfg = datasets_usuario.renomear(id_dataset, req.nome)
    except KeyError:
        raise HTTPException(status_code=404, detail='Base não encontrada.')
    except ErroLeitura as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {'id': cfg['id'], 'nome': cfg['nome']}


@router.delete('/enviados/{id_dataset}')
def remover_enviado(id_dataset: str):
    """Remove a base importada e os caches associados a ela."""
    if id_dataset not in datasets_usuario.registrados():
        raise HTTPException(status_code=404, detail='Base não encontrada.')
    datasets_usuario.remover(id_dataset)
    invalidar_cache()
    modelos.esquecer_predicoes()
    return {'removido': id_dataset}
