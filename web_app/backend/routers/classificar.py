"""
Classificacao — escolha e parametrizacao do modelo.

Esta e a tela que atende diretamente ao enunciado da entrega: "opcoes de
definicao do modelo a ser utilizado no processo de classificacao, bem como a
parametrizacao do modelo". Um unico endpoint treina QUALQUER modelo do
catalogo (`backend/modelos.py`) sobre QUALQUER base (inclusive a .txt
importada pelo usuario), com os hiperparametros que a interface enviar.

Rotas
-----
GET  /api/classificar/modelos    catalogo + esquema de parametros
POST /api/classificar/treinar    treina, avalia e devolve todas as metricas
POST /api/classificar/regioes    regioes de decisao no plano 2D
POST /api/classificar/predizer   classifica uma amostra digitada
"""
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from evaluation.metricas_avancadas import relatorio_completo

from .. import modelos as M
from ..core import (classes_de, config_de, features_de, indices_de,
                    indices_plot, jitter_de, limites_com_margem, malha,
                    obter_split, serializar_amostras)

router = APIRouter(prefix='/api/classificar', tags=['classificar'])


class ClassificarRequest(BaseModel):
    dataset: str = 'v1'
    atributos: str = 'petalas'
    proporcao: float = Field(0.7, ge=0.1, le=0.9)
    modelo: str = 'distancia_minima'
    # Livre por construcao: cada modelo tem o seu conjunto. A validacao contra
    # o esquema acontece em `modelos.normalizar_parametros`.
    parametros: dict = {}


class RegioesRequest(ClassificarRequest):
    resolucao: int = Field(80, ge=20, le=160)


class PredicaoRequest(ClassificarRequest):
    valores: list[float]


def _contexto(req: ClassificarRequest):
    """Split + indices + modelo treinado, com os erros ja traduzidos em HTTP."""
    try:
        dados, treino, teste = obter_split(req.dataset, req.proporcao)
        idx = indices_de(req.atributos, req.dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        objeto, params = M.treinar_modelo(req.modelo, treino, idx,
                                          req.parametros)
    except M.ModeloInvalido as e:
        raise HTTPException(status_code=400, detail=str(e))
    except (ValueError, ZeroDivisionError) as e:
        # Ex.: covariancia singular no Bayes quando uma classe tem poucas
        # amostras — erro do DADO, nao do servidor.
        raise HTTPException(
            status_code=400,
            detail=f'Não foi possível treinar o modelo com esta configuração: {e}')
    return dados, treino, teste, idx, objeto, params


def _medias(dados, n_features):
    """Media global de cada feature — usada para fixar as dimensoes nao plotadas."""
    return [sum(d['atributos'][j] for d in dados) / len(dados)
            for j in range(n_features)]


def _extras(modelo, objeto):
    """
    Informacoes especificas de cada modelo, quando existirem.

    E o que faz a tela generica nao ficar pobre: a floresta mostra OOB e
    importancias, a rede mostra a curva de erro, os lineares mostram em
    quantas epocas convergiram.
    """
    if modelo == 'floresta':
        return {
            'oob': {'acuracia': objeto.acuracia_oob, 'erro': objeto.erro_oob},
            'importancias': [
                {'indice': a, 'importancia': objeto.importancias[a]}
                for a in sorted(objeto.importancias,
                                key=lambda x: -objeto.importancias[x])],
            'arvores': len(objeto.arvores),
        }
    if modelo == 'mlp':
        hist = objeto.historico_erro
        # No maximo 120 pontos: o suficiente para desenhar a curva.
        passo = max(1, len(hist) // 120)
        return {
            'curva_erro': [{'epoca': i + 1, 'erro': hist[i]}
                           for i in range(0, len(hist), passo)],
            'erro_final': hist[-1] if hist else None,
        }
    if modelo == 'perceptron_ova':
        return {
            'epocas_por_classe': objeto['epocas'],
            'convergiu': {c: (h[-1] == 0 if h else False)
                          for c, h in objeto['historico'].items()},
            'curvas': {c: h for c, h in objeto['historico'].items()},
        }
    if modelo == 'delta_ova':
        return {'curvas': {c: h for c, h in objeto['historico'].items()}}
    return {}


@router.get('/modelos')
def listar_modelos():
    """Catalogo de modelos com o esquema de hiperparametros de cada um."""
    return {'modelos': M.catalogo(), 'ordem': M.ORDEM}


@router.post('/treinar')
def treinar(req: ClassificarRequest):
    """
    Treina o modelo escolhido com os parametros escolhidos e devolve o
    relatorio completo de qualidade sobre o conjunto de teste.
    """
    dados, treino, teste, idx, objeto, params = _contexto(req)
    CLASSES = classes_de(req.dataset)
    cfg_attr = config_de(req.atributos, req.dataset)
    idx_plot = indices_plot(req.atributos, req.dataset)

    inicio = time.perf_counter()
    preds = [M.predizer(req.modelo, objeto, d['atributos']) for d in teste]
    preds_treino = [M.predizer(req.modelo, objeto, d['atributos'])
                    for d in treino]
    ms_predicao = (time.perf_counter() - inicio) * 1000

    gabarito = [d['classe'] for d in teste]
    nome = M.info(req.modelo)['nome']
    relatorio = relatorio_completo(preds, gabarito, CLASSES, nome)
    relatorio_treino = relatorio_completo(
        preds_treino, [d['classe'] for d in treino], CLASSES,
        f'{nome} (treino)')

    return {
        'relatorio': relatorio,
        # O acerto no treino ao lado do acerto no teste denuncia sobreajuste —
        # e o que se espera ver numa floresta sem limite de profundidade.
        'acerto_treino': relatorio_treino['acerto_global'],
        'extras': _extras(req.modelo, objeto),
        'modelo': {'id': req.modelo, 'nome': nome,
                   'descricao': M.info(req.modelo)['descricao'],
                   'rotulo_score': M.info(req.modelo)['rotulo_score']},
        'parametros': params,
        'amostras': serializar_amostras(dados, idx_plot, treino,
                                        jitter=jitter_de(req.dataset)),
        'classes': CLASSES,
        'features': [features_de(req.dataset)[i] for i in idx],
        # Indices no vetor completo de atributos: o formulario de predicao
        # precisa deles para saber o que cada campo representa, e o grafico
        # para saber quais duas dimensoes ele esta desenhando.
        'indices': idx,
        'indices_plot': idx_plot,
        'medias': [sum(d['atributos'][i] for d in dados) / len(dados)
                   for i in idx],
        'n_treino': len(treino),
        'n_teste': len(teste),
        'dimensoes': len(idx),
        'eixo_x': cfg_attr['eixo_x'],
        'eixo_y': cfg_attr['eixo_y'],
        'ms_predicao': ms_predicao,
    }


@router.post('/regioes')
def regioes(req: RegioesRequest):
    """
    Regioes de decisao do modelo escolhido.

    Percorre uma malha regular do plano 2D e pergunta ao modelo qual classe
    ele atribui a cada ponto. Quando o conjunto de atributos tem mais de duas
    dimensoes, as demais ficam fixas na media global — e a mesma convencao
    das telas dos laboratorios.
    """
    dados, _, _, idx, objeto, _ = _contexto(req)
    CLASSES = classes_de(req.dataset)
    idx_plot = indices_plot(req.atributos, req.dataset)
    n_features = len(features_de(req.dataset))

    lim = limites_com_margem(dados, idx_plot)
    eixo_x, eixo_y = malha(lim, req.resolucao)
    base = _medias(dados, n_features)

    grade = []
    for y in eixo_y:
        linha = []
        for x in eixo_x:
            ponto = list(base)
            ponto[idx_plot[0]] = x
            ponto[idx_plot[1]] = y
            classe = M.predizer(req.modelo, objeto, ponto)
            linha.append(CLASSES.index(classe) if classe in CLASSES else -1)
        grade.append(linha)

    return {'grade': grade, 'eixo_x': eixo_x, 'eixo_y': eixo_y,
            'limites': lim, 'classes': CLASSES}


@router.post('/predizer')
def predizer(req: PredicaoRequest):
    """Classifica uma amostra digitada pelo usuario, com as pontuacoes."""
    dados, _, _, idx, objeto, params = _contexto(req)
    CLASSES = classes_de(req.dataset)

    if len(req.valores) != len(idx):
        raise HTTPException(
            status_code=400,
            detail=f'Esperados {len(idx)} valores para "{req.atributos}", '
                   f'recebidos {len(req.valores)}.')

    # O vetor completo tem uma posicao por feature do dataset; as que nao
    # fazem parte do conjunto de atributos escolhido ficam na media global.
    ponto = _medias(dados, len(features_de(req.dataset)))
    for k, j in enumerate(idx):
        ponto[j] = req.valores[k]

    return {
        'classe': M.predizer(req.modelo, objeto, ponto),
        'scores': M.scores(req.modelo, objeto, ponto),
        'rotulo_score': M.info(req.modelo)['rotulo_score'],
        'valores': req.valores,
        'parametros': params,
    }
