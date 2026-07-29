"""
Lab 2 — Perceptron de Rosenblatt e Regra Delta (Widrow-Hoff / Adaline).

Cobre os quatro experimentos da aba desktop equivalente:
  · perceptron binario (par de classes)
  · regra delta binaria (par de classes)
  · regra delta One-vs-All (multiclasse)
  · XOR com regra delta — limite dos classificadores lineares
"""
from fastapi import APIRouter, HTTPException, Query

from evaluation.metricas_avancadas import relatorio_completo
from models.delta_rule import (acuracia_binaria_delta, predizer_delta,
                               predizer_delta_ova, treinar_delta_iris,
                               treinar_delta_ova, treinar_delta_xor)
from models.perceptron import (acuracia_binaria_perceptron,
                               predizer_perceptron, treinar_perceptron)

from ..core import (CLASSES, CONFIG_ATRIBUTOS, indices_de, indices_plot,
                    limites_com_margem, obter_split, serializar_amostras)

router = APIRouter(prefix='/api/perceptron-delta', tags=['perceptron-delta'])


@router.get('/binario')
def binario(algoritmo: str = Query('perceptron', pattern='^(perceptron|delta)$'),
            dataset: str = 'v1', atributos: str = 'petalas',
            classe_pos: str = 'setosa', classe_neg: str = 'versicolor',
            taxa: float = Query(0.03, gt=0, le=1),
            max_epocas: int = Query(100, ge=1, le=2000),
            proporcao: float = Query(0.7, ge=0.1, le=0.9)):
    """Treina Perceptron ou Regra Delta para um par de classes."""
    if classe_pos == classe_neg:
        raise HTTPException(status_code=400,
                            detail='As duas classes devem ser diferentes.')
    try:
        dados, treino, teste = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(atributos)

    if algoritmo == 'perceptron':
        w, historico, epocas = treinar_perceptron(
            treino, classe_pos, classe_neg, idx, taxa, max_epocas)
        rotulo_historico = 'erros'
        convergiu = historico[-1] == 0 if historico else False
        acc_treino = acuracia_binaria_perceptron(treino, w, classe_pos, classe_neg, idx)
        acc_teste = acuracia_binaria_perceptron(teste, w, classe_pos, classe_neg, idx)
        preditor = lambda x: predizer_perceptron(x, w)  # noqa: E731
        mapear = lambda y: classe_pos if y == 1 else classe_neg  # noqa: E731
    else:
        w, historico, epocas = treinar_delta_iris(
            treino, classe_pos, classe_neg, idx, taxa, max_epocas)
        rotulo_historico = 'mse'
        convergiu = bool(historico) and historico[-1] < 0.01
        acc_treino = acuracia_binaria_delta(treino, w, classe_pos, classe_neg, idx)
        acc_teste = acuracia_binaria_delta(teste, w, classe_pos, classe_neg, idx)
        preditor = lambda x: predizer_delta(x, w, classe_pos, classe_neg)  # noqa: E731
        mapear = lambda y: y  # noqa: E731

    # Matriz de confusao binaria no conjunto de teste
    teste_par = [d for d in teste if d['classe'] in (classe_pos, classe_neg)]
    preds, gabarito = [], []
    for d in teste_par:
        x_sel = [d['atributos'][i] for i in idx]
        preds.append(mapear(preditor(x_sel)))
        gabarito.append(d['classe'])
    relatorio = relatorio_completo(preds, gabarito, [classe_pos, classe_neg],
                                  'Perceptron' if algoritmo == 'perceptron' else 'Regra Delta')

    idx_plot = indices_plot(atributos)
    cfg = CONFIG_ATRIBUTOS[atributos]
    dados_par = [d for d in dados if d['classe'] in (classe_pos, classe_neg)]

    return {
        'algoritmo': algoritmo,
        'pesos': w,
        'historico': historico,
        'rotulo_historico': rotulo_historico,
        'epocas': epocas,
        'convergiu': convergiu,
        'acuracia_treino': acc_treino,
        'acuracia_teste': acc_teste,
        'relatorio': relatorio,
        'amostras': serializar_amostras(dados_par, idx_plot, treino),
        'limites': limites_com_margem(dados_par, idx_plot),
        'classe_pos': classe_pos,
        'classe_neg': classe_neg,
        'eixo_x': cfg['eixo_x'],
        'eixo_y': cfg['eixo_y'],
        'bidimensional': len(idx) == 2,
    }


@router.get('/ova')
def ova(dataset: str = 'v1', atributos: str = 'petalas',
        taxa: float = Query(0.02, gt=0, le=1),
        max_epocas: int = Query(200, ge=1, le=2000),
        proporcao: float = Query(0.7, ge=0.1, le=0.9)):
    """Regra Delta multiclasse no esquema Um-Contra-Todos."""
    try:
        dados, treino, teste = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(atributos)
    pesos, historico, epocas = treinar_delta_ova(treino, idx, taxa, max_epocas)

    preds, gabarito = [], []
    for d in teste:
        x_sel = [d['atributos'][i] for i in idx]
        preds.append(predizer_delta_ova(x_sel, pesos)[0])
        gabarito.append(d['classe'])
    relatorio = relatorio_completo(preds, gabarito, CLASSES, 'Regra Delta OvA')

    idx_plot = indices_plot(atributos)
    cfg = CONFIG_ATRIBUTOS[atributos]
    return {
        'pesos': pesos,
        'historico': historico,
        'epocas': epocas,
        'relatorio': relatorio,
        'amostras': serializar_amostras(dados, idx_plot, treino),
        'limites': limites_com_margem(dados, idx_plot),
        'eixo_x': cfg['eixo_x'],
        'eixo_y': cfg['eixo_y'],
        'bidimensional': len(idx) == 2,
    }


@router.get('/xor')
def xor(taxa: float = Query(0.02, gt=0, le=1),
        max_epocas: int = Query(300, ge=1, le=5000)):
    """
    XOR com Regra Delta — demonstra o limite teorico dos classificadores
    lineares: o MSE estaciona proximo de 0,25 e nunca zera.
    """
    w, historico = treinar_delta_xor(max_epocas=max_epocas, taxa_aprendizado=taxa)

    padroes = []
    for x1, x2, alvo in [(0.0, 0.0, 0), (0.0, 1.0, 1), (1.0, 0.0, 1), (1.0, 1.0, 0)]:
        net = w[0] + w[1] * x1 + w[2] * x2
        padroes.append({
            'x1': x1, 'x2': x2, 'alvo': alvo,
            'net': net,
            'previsto': 1 if net >= 0.5 else 0,
            'correto': (1 if net >= 0.5 else 0) == alvo,
        })

    return {
        'pesos': w,
        'historico': historico,
        'mse_final': historico[-1] if historico else None,
        'mse_teorico': 0.25,
        'padroes': padroes,
        'acertos': sum(1 for p in padroes if p['correto']),
    }
