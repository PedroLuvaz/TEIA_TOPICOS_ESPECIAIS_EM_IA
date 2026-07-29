"""Rotas de metadados e exploracao do dataset Iris."""
import os

from fastapi import APIRouter, HTTPException, Query

from ..core import (CAMINHOS_DADOS, CLASSES, CONFIG_ATRIBUTOS, DATASETS,
                    NOMES_FEATURES, PARES_CLASSES, indices_plot, obter_split,
                    serializar_amostras)

router = APIRouter(prefix='/api/dataset', tags=['dataset'])


@router.get('/metadata')
def metadata():
    """Opcoes disponiveis na interface (datasets, atributos, classes, pares)."""
    return {
        'datasets': [
            {**d, 'disponivel': os.path.exists(CAMINHOS_DADOS[d['id']])}
            for d in DATASETS
        ],
        'atributos': [
            {'id': k, 'nome': v['nome'], 'indices': v['indices'],
             'eixo_x': v['eixo_x'], 'eixo_y': v['eixo_y']}
            for k, v in CONFIG_ATRIBUTOS.items()
        ],
        'classes': CLASSES,
        'features': NOMES_FEATURES,
        'pares': [{'pos': a, 'neg': b} for a, b in PARES_CLASSES],
    }


@router.get('/amostras')
def amostras(dataset: str = 'v1', atributos: str = 'petalas',
             proporcao: float = Query(0.7, ge=0.1, le=0.9),
             semente: int = 42):
    """Amostras projetadas em 2D, ja marcadas como treino ou teste."""
    try:
        dados, treino, teste = obter_split(dataset, proporcao, semente)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_plot(atributos)
    cfg = CONFIG_ATRIBUTOS[atributos]
    return {
        'amostras': serializar_amostras(dados, idx, treino),
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
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    resumo = {}
    for classe in CLASSES:
        amostras_c = [d['atributos'] for d in dados if d['classe'] == classe]
        n = len(amostras_c)
        por_feature = []
        for j, nome in enumerate(NOMES_FEATURES):
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
    return {'por_classe': resumo, 'features': NOMES_FEATURES}
