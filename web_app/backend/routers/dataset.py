"""Rotas de metadados e exploracao dos datasets."""
import os

from fastapi import APIRouter, HTTPException, Query

from ..core import (DATASETS, DATASET_PADRAO, carregar, classes_de,
                    config_atributos_de, config_de, features_de, indices_plot,
                    jitter_de, obter_split, pares_de, serializar_amostras)

router = APIRouter(prefix='/api/dataset', tags=['dataset'])


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
    for cfg in DATASETS.values():
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
