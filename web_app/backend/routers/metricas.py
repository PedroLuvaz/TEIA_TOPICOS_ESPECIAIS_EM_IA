"""
Lab 3 — Metricas Avancadas de Qualidade.

Duas frentes:
  · avaliar uma matriz de confusao arbitraria (editavel na interface)
  · comparar todos os classificadores do projeto no mesmo split, com
    teste Z de significancia de Kappa entre cada par
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from evaluation.metricas_avancadas import (acerto_global, kappa, p_valor_z,
                                           relatorio_completo, tau,
                                           variancia_kappa, variancia_tau,
                                           z_kappa, z_tau)
from models.bayes_classifier import (predizer_todas_classes_bayes,
                                     treinar_bayes)
from models.classifier import predizer_todas_classes, treinar
from models.delta_rule import predizer_delta_ova, treinar_delta_ova

from ..core import CLASSES, indices_de, obter_split

router = APIRouter(prefix='/api/metricas', tags=['metricas'])


class MatrizRequest(BaseModel):
    """Matriz de confusao: matriz[predito][real] = contagem."""
    matriz: dict[str, dict[str, int]]
    classes: list[str] | None = None
    nome: str = 'Matriz personalizada'


class ComparacaoMatrizesRequest(BaseModel):
    matriz_a: dict[str, dict[str, int]]
    matriz_b: dict[str, dict[str, int]]
    classes: list[str] | None = None
    nome_a: str = 'Classificador A'
    nome_b: str = 'Classificador B'


def _metricas_da_matriz(matriz, classes, nome):
    """Reconstroi o relatorio completo a partir de uma matriz ja pronta."""
    predicoes, gabarito = [], []
    for pred in classes:
        for real in classes:
            n = int(matriz.get(pred, {}).get(real, 0))
            predicoes.extend([pred] * n)
            gabarito.extend([real] * n)
    if not predicoes:
        raise HTTPException(status_code=400,
                            detail='A matriz de confusao esta vazia.')
    return relatorio_completo(predicoes, gabarito, classes, nome)


@router.post('/avaliar')
def avaliar(req: MatrizRequest):
    """Calcula todas as metricas de uma matriz de confusao informada."""
    classes = req.classes or CLASSES
    relatorio = _metricas_da_matriz(req.matriz, classes, req.nome)
    total = sum(sum(linha.values()) for linha in relatorio['matriz'].values())
    return {'relatorio': relatorio, 'total_amostras': total, 'classes': classes}


@router.post('/comparar-matrizes')
def comparar_matrizes(req: ComparacaoMatrizesRequest):
    """Teste Z de Kappa e de Tau entre duas matrizes de confusao."""
    classes = req.classes or CLASSES
    ra = _metricas_da_matriz(req.matriz_a, classes, req.nome_a)
    rb = _metricas_da_matriz(req.matriz_b, classes, req.nome_b)

    zk = z_kappa(ra['kappa'], ra['variancia_kappa'], rb['kappa'], rb['variancia_kappa'])
    zt = z_tau(ra['tau'], ra['variancia_tau'], rb['tau'], rb['variancia_tau'])

    return {
        'a': ra, 'b': rb,
        'kappa': {'z': zk, 'p': p_valor_z(zk), 'significativo': p_valor_z(zk) < 0.05},
        'tau': {'z': zt, 'p': p_valor_z(zt), 'significativo': p_valor_z(zt) < 0.05},
    }


@router.get('/comparar-modelos')
def comparar_modelos(dataset: str = 'v1', atributos: str = 'petalas',
                     proporcao: float = Query(0.7, ge=0.1, le=0.9)):
    """
    Avalia todos os classificadores multiclasse do projeto no mesmo split
    e aplica o teste Z de Kappa em cada par — a comparacao central do Lab 3.
    """
    try:
        _, treino, teste = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(atributos)
    gabarito = [d['classe'] for d in teste]
    relatorios = {}

    prototipos = treinar(treino, idx)
    relatorios['distancia_minima'] = relatorio_completo(
        [predizer_todas_classes(d['atributos'], prototipos, idx)[1] for d in teste],
        gabarito, CLASSES, 'Distancia Minima')

    pesos_ova, _, _ = treinar_delta_ova(treino, idx)
    relatorios['delta_ova'] = relatorio_completo(
        [predizer_delta_ova([d['atributos'][i] for i in idx], pesos_ova)[0] for d in teste],
        gabarito, CLASSES, 'Regra Delta OvA')

    for chave, naive, nome in (('bayes', False, 'Bayes Otimo (QDA)'),
                               ('naive', True, 'Naive Bayes')):
        modelo = treinar_bayes(treino, idx, naive=naive)
        relatorios[chave] = relatorio_completo(
            [predizer_todas_classes_bayes(d['atributos'], modelo, idx)[1] for d in teste],
            gabarito, CLASSES, nome)

    chaves = list(relatorios.keys())
    comparacoes = []
    for i in range(len(chaves)):
        for j in range(i + 1, len(chaves)):
            ra, rb = relatorios[chaves[i]], relatorios[chaves[j]]
            z = z_kappa(ra['kappa'], ra['variancia_kappa'],
                        rb['kappa'], rb['variancia_kappa'])
            p = p_valor_z(z)
            comparacoes.append({
                'a': chaves[i], 'b': chaves[j],
                'nome_a': ra['nome'], 'nome_b': rb['nome'],
                'z': z, 'p': p, 'significativo': p < 0.05,
            })

    return {'relatorios': relatorios, 'comparacoes': comparacoes,
            'n_teste': len(teste), 'classes': CLASSES}


@router.get('/simular')
def simular(acerto: float = Query(0.9, ge=0.0, le=1.0),
            n_por_classe: int = Query(15, ge=1, le=500)):
    """
    Gera uma matriz de confusao sintetica com o acerto global desejado,
    distribuindo os erros uniformemente — util para explorar como Kappa e
    Tau reagem a diferentes niveis de acerto.
    """
    n_classes = len(CLASSES)
    acertos_por_classe = round(n_por_classe * acerto)
    erros = n_por_classe - acertos_por_classe

    matriz = {p: {r: 0 for r in CLASSES} for p in CLASSES}
    for i, real in enumerate(CLASSES):
        matriz[real][real] = acertos_por_classe
        if erros > 0:
            outras = [c for c in CLASSES if c != real]
            base, resto = divmod(erros, len(outras))
            for k, pred in enumerate(outras):
                matriz[pred][real] = base + (1 if k < resto else 0)

    relatorio = _metricas_da_matriz(matriz, CLASSES, f'Simulacao ({acerto:.0%})')
    return {'relatorio': relatorio, 'acerto_alvo': acerto,
            'n_por_classe': n_por_classe}


@router.get('/curva-kappa')
def curva_kappa(n_por_classe: int = Query(15, ge=1, le=500),
                passos: int = Query(21, ge=3, le=101)):
    """
    Curva de Acerto Global x Kappa x Tau, varrendo o acerto de 0% a 100%.
    Mostra visualmente por que Kappa e mais rigoroso que o acerto bruto.
    """
    pontos = []
    for k in range(passos):
        alvo = k / (passos - 1)
        acertos = round(n_por_classe * alvo)
        erros = n_por_classe - acertos
        matriz = {p: {r: 0 for r in CLASSES} for p in CLASSES}
        for real in CLASSES:
            matriz[real][real] = acertos
            if erros > 0:
                outras = [c for c in CLASSES if c != real]
                base, resto = divmod(erros, len(outras))
                for i, pred in enumerate(outras):
                    matriz[pred][real] = base + (1 if i < resto else 0)
        pontos.append({
            'acerto_alvo': alvo,
            'acerto_global': acerto_global(matriz, CLASSES),
            'kappa': kappa(matriz, CLASSES),
            'tau': tau(matriz, CLASSES),
            'var_kappa': variancia_kappa(matriz, CLASSES),
            'var_tau': variancia_tau(matriz, CLASSES),
        })
    return {'pontos': pontos}
