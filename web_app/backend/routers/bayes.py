"""
Lab 4 — Classificador Otimo de Bayes (QDA) e Naive Bayes.

Alem do treino e das regioes de decisao quadraticas, expoe o teste de
normalidade multivariada (Henze-Zirkler / Mardia) calculado em Python puro.
"""
import math

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.math_utils import distancia_mahalanobis_quad
from evaluation.metricas_avancadas import (p_valor_z, relatorio_completo,
                                           z_kappa)
from evaluation.mvn_tester import calcular_mvn_python
from models.bayes_classifier import (predizer_todas_classes_bayes,
                                     treinar_bayes)

from ..core import (CLASSES, CONFIG_ATRIBUTOS, indices_de, indices_plot,
                    limites_com_margem, malha, obter_split,
                    serializar_amostras)

router = APIRouter(prefix='/api/bayes', tags=['bayes'])


class PredicaoRequest(BaseModel):
    dataset: str = 'v1'
    atributos: str = 'petalas'
    naive: bool = False
    valores: list[float]


def _score(x_sel, params):
    """d_j(x) = -0.5·ln|Sigma_j| - 0.5·(x-m)^T Sigma^-1 (x-m)"""
    d2 = distancia_mahalanobis_quad(x_sel, params['media'], params['inv_cov'])
    return -0.5 * math.log(params['det']) - 0.5 * d2


@router.get('/treinar')
def treinar(dataset: str = 'v1', atributos: str = 'petalas',
            proporcao: float = Query(0.7, ge=0.1, le=0.9)):
    """Treina Bayes Otimo e Naive Bayes lado a lado, com teste Z entre eles."""
    try:
        dados, treino, teste = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(atributos)
    gabarito = [d['classe'] for d in teste]

    resultados = {}
    for chave, naive in (('bayes', False), ('naive', True)):
        modelo = treinar_bayes(treino, idx, naive=naive)
        preds = [predizer_todas_classes_bayes(d['atributos'], modelo, idx)[1]
                 for d in teste]
        nome = 'Bayes Otimo (QDA)' if not naive else 'Naive Bayes'
        resultados[chave] = {
            'relatorio': relatorio_completo(preds, gabarito, CLASSES, nome),
            'parametros': {
                c: {
                    'media': modelo[c]['media'],
                    'cov': modelo[c]['cov'],
                    'det': modelo[c]['det'],
                    'inv_cov': modelo[c]['inv_cov'],
                }
                for c in CLASSES
            },
        }

    ra, rb = resultados['bayes']['relatorio'], resultados['naive']['relatorio']
    z = z_kappa(ra['kappa'], ra['variancia_kappa'], rb['kappa'], rb['variancia_kappa'])
    p = p_valor_z(z)

    idx_plot = indices_plot(atributos)
    cfg = CONFIG_ATRIBUTOS[atributos]
    return {
        **resultados,
        'teste_z': {'z': z, 'p': p, 'significativo': p < 0.05},
        'amostras': serializar_amostras(dados, idx_plot, treino),
        'n_treino': len(treino),
        'n_teste': len(teste),
        'eixo_x': cfg['eixo_x'],
        'eixo_y': cfg['eixo_y'],
        'dimensoes': len(idx),
    }


@router.get('/regioes')
def regioes(dataset: str = 'v1', atributos: str = 'petalas',
            classificador: str = Query('bayes', pattern='^(bayes|naive)$'),
            proporcao: float = Query(0.7, ge=0.1, le=0.9),
            resolucao: int = Query(90, ge=20, le=200)):
    """
    Regioes de decisao quadraticas. Devolve tambem a superficie de diferenca
    de scores por par de classes, permitindo tracar a fronteira exata (nivel 0)
    com marching squares no frontend — sem "escadinhas".
    """
    try:
        dados, treino, _ = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(atributos)
    idx_plot = indices_plot(atributos)
    modelo = treinar_bayes(treino, idx, naive=(classificador == 'naive'))

    lim = limites_com_margem(dados, idx_plot)
    eixo_x, eixo_y = malha(lim, resolucao)

    fixos = {}
    for j in idx:
        if j not in idx_plot:
            valores = [d['atributos'][j] for d in dados]
            fixos[j] = sum(valores) / len(valores)

    grade = []
    superficies = {f'{a}|{b}': [] for a, b in
                   [('setosa', 'versicolor'), ('setosa', 'virginica'),
                    ('versicolor', 'virginica')]}

    for y in eixo_y:
        linha_classe = []
        linhas_par = {k: [] for k in superficies}
        for x in eixo_x:
            ponto = []
            for j in idx:
                if j == idx_plot[0]:
                    ponto.append(x)
                elif j == idx_plot[1]:
                    ponto.append(y)
                else:
                    ponto.append(fixos[j])

            scores = {c: _score(ponto, modelo[c]) for c in CLASSES}
            linha_classe.append(CLASSES.index(max(scores, key=scores.get)))
            for chave in superficies:
                a, b = chave.split('|')
                linhas_par[chave].append(scores[a] - scores[b])

        grade.append(linha_classe)
        for chave in superficies:
            superficies[chave].append(linhas_par[chave])

    return {'grade': grade, 'superficies': superficies,
            'eixo_x': eixo_x, 'eixo_y': eixo_y,
            'limites': lim, 'classes': CLASSES}


@router.get('/normalidade')
def normalidade(dataset: str = 'v1', atributos: str = 'petalas'):
    """
    Teste de aderencia a normalidade multivariada por classe
    (Henze-Zirkler e Mardia), implementado em Python puro.
    """
    try:
        dados, _, _ = obter_split(dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(atributos)
    try:
        resultado = calcular_mvn_python(dados, idx)
    except Exception as e:  # pragma: no cover — protege a UI de erro numerico
        raise HTTPException(status_code=500,
                            detail=f'Falha ao calcular MVN: {e}')

    return {'resultado': resultado, 'atributos': atributos,
            'indices': idx, 'n_features': len(idx)}


@router.post('/predizer')
def predizer(req: PredicaoRequest):
    """Classifica um vetor arbitrario com Bayes ou Naive Bayes."""
    try:
        _, treino, _ = obter_split(req.dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(req.atributos)
    if len(req.valores) != len(idx):
        raise HTTPException(
            status_code=400,
            detail=f'Esperados {len(idx)} valores, recebidos {len(req.valores)}.')

    modelo = treinar_bayes(treino, idx, naive=req.naive)
    scores = {c: _score(req.valores, modelo[c]) for c in CLASSES}
    mahalanobis = {
        c: distancia_mahalanobis_quad(req.valores, modelo[c]['media'],
                                      modelo[c]['inv_cov'])
        for c in CLASSES
    }
    return {
        'classe': max(scores, key=scores.get),
        'scores': scores,
        'mahalanobis': mahalanobis,
        'valores': req.valores,
    }
