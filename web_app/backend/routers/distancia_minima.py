"""
Lab 1 — Classificador de Distancia Minima.

Delega toda a matematica para `models/classifier.py` e `core/math_utils.py`
(Python puro, sem numpy), expondo protótipos, fronteiras lineares, regioes
de decisao e predicao de amostras arbitrarias.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.math_utils import (coeficientes_superficie_decisao,
                             distancia_euclidiana, discriminante)
from evaluation.metricas_avancadas import relatorio_completo
from models.classifier import predizer_todas_classes, treinar

from ..core import (CLASSES, CONFIG_ATRIBUTOS, PARES_CLASSES, indices_de,
                    indices_plot, limites_com_margem, malha, obter_split,
                    serializar_amostras)

router = APIRouter(prefix='/api/distancia-minima', tags=['distancia-minima'])


class PredicaoRequest(BaseModel):
    dataset: str = 'v1'
    atributos: str = 'petalas'
    proporcao: float = 0.7
    valores: list[float]


def _contexto(dataset, atributos, proporcao):
    try:
        dados, treino, teste = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    idx = indices_de(atributos)
    prototipos = treinar(treino, idx)
    return dados, treino, teste, idx, prototipos


@router.get('/treinar')
def treinar_modelo(dataset: str = 'v1', atributos: str = 'petalas',
                   proporcao: float = Query(0.7, ge=0.1, le=0.9)):
    """
    Treina o classificador e devolve protótipos, metricas completas,
    equacoes das fronteiras e as amostras para plotagem.
    """
    dados, treino, teste, idx, prototipos = _contexto(dataset, atributos, proporcao)

    preds = [predizer_todas_classes(d['atributos'], prototipos, idx)[1] for d in teste]
    gabarito = [d['classe'] for d in teste]
    relatorio = relatorio_completo(preds, gabarito, CLASSES, 'Distancia Minima')

    # Equacoes das fronteiras para cada par de classes
    fronteiras = []
    for ci, cj in PARES_CLASSES:
        w, b = coeficientes_superficie_decisao(prototipos[ci], prototipos[cj])
        fronteiras.append({
            'classe_i': ci, 'classe_j': cj,
            'w': w, 'b': b,
            'equacao': _formatar_equacao(w, b, atributos),
        })

    idx_plot = indices_plot(atributos)
    cfg = CONFIG_ATRIBUTOS[atributos]
    return {
        'prototipos': {c: prototipos[c] for c in CLASSES},
        'prototipos_plot': {
            c: {'x': prototipos[c][idx.index(idx_plot[0])] if idx_plot[0] in idx else prototipos[c][0],
                'y': prototipos[c][idx.index(idx_plot[1])] if idx_plot[1] in idx else prototipos[c][1]}
            for c in CLASSES
        },
        'relatorio': relatorio,
        'fronteiras': fronteiras,
        'amostras': serializar_amostras(dados, idx_plot, treino),
        'n_treino': len(treino),
        'n_teste': len(teste),
        'eixo_x': cfg['eixo_x'],
        'eixo_y': cfg['eixo_y'],
        'dimensoes': len(idx),
    }


@router.get('/regioes')
def regioes(dataset: str = 'v1', atributos: str = 'petalas',
            proporcao: float = Query(0.7, ge=0.1, le=0.9),
            resolucao: int = Query(90, ge=20, le=200)):
    """
    Grade de regioes de decisao: para cada ponto da malha, o indice da classe
    vencedora. O frontend renderiza isso como um heatmap suave.
    """
    dados, treino, _, idx, prototipos = _contexto(dataset, atributos, proporcao)
    idx_plot = indices_plot(atributos)
    lim = limites_com_margem(dados, idx_plot)
    eixo_x, eixo_y = malha(lim, resolucao)

    # Para o modo 4D, as features nao plotadas ficam fixas na media global
    fixos = _valores_fixos(dados, idx, idx_plot)

    grade = []
    for y in eixo_y:
        linha = []
        for x in eixo_x:
            ponto = _montar_ponto(x, y, idx, idx_plot, fixos)
            scores = {c: discriminante(ponto, prototipos[c]) for c in CLASSES}
            linha.append(CLASSES.index(max(scores, key=scores.get)))
        grade.append(linha)

    return {'grade': grade, 'eixo_x': eixo_x, 'eixo_y': eixo_y,
            'limites': lim, 'classes': CLASSES}


@router.post('/predizer')
def predizer(req: PredicaoRequest):
    """Classifica um vetor arbitrario, devolvendo scores e distancias."""
    _, treino, _, idx, prototipos = _contexto(req.dataset, req.atributos, req.proporcao)

    if len(req.valores) != len(idx):
        raise HTTPException(
            status_code=400,
            detail=f'Esperados {len(idx)} valores para "{req.atributos}", '
                   f'recebidos {len(req.valores)}.')

    scores = {c: discriminante(req.valores, prototipos[c]) for c in CLASSES}
    distancias = {c: distancia_euclidiana(req.valores, prototipos[c]) for c in CLASSES}
    vencedor = max(scores, key=scores.get)

    return {
        'classe': vencedor,
        'scores': scores,
        'distancias': distancias,
        'prototipos': {c: prototipos[c] for c in CLASSES},
        'valores': req.valores,
    }


# ---------------------------------------------------------------------------
def _valores_fixos(dados, idx, idx_plot):
    """Media global das features que nao estao no plano de plotagem."""
    fixos = {}
    for j in idx:
        if j not in idx_plot:
            valores = [d['atributos'][j] for d in dados]
            fixos[j] = sum(valores) / len(valores)
    return fixos


def _montar_ponto(x, y, idx, idx_plot, fixos):
    """Monta o vetor de features na ordem de `idx`, variando so o plano 2D."""
    ponto = []
    for j in idx:
        if j == idx_plot[0]:
            ponto.append(x)
        elif j == idx_plot[1]:
            ponto.append(y)
        else:
            ponto.append(fixos[j])
    return ponto


def _formatar_equacao(w, b, atributos):
    """Monta a string da equacao da fronteira: w1*x1 + w2*x2 + b = 0."""
    termos = [f'{wi:+.4f}·x{i + 1}' for i, wi in enumerate(w)]
    return f'{" ".join(termos)} {b:+.4f} = 0'
