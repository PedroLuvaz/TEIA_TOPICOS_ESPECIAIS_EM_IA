"""
Rotas do Classificador de Distancia Minima — endpoints para prototipos,
classificacao, fronteiras de decisao e metricas.

Endpoints:
    GET  /api/prototypes       — Prototipos para os indices solicitados
    POST /api/classify         — Classifica uma amostra
    GET  /api/boundaries       — Coeficientes das fronteiras de decisao (3 pares)
    GET  /api/distancia/metrics — Metricas completas do classificador de distancia
"""

from flask import Blueprint, request, jsonify, current_app

import sys
import os

# Importacoes dos modulos do classificador
from classifier import treinar as treinar_prototipos, predizer_todas_classes, predizer_binario
from math_utils import coeficientes_superficie_decisao
from evaluator import acuracia, matriz_confusao, precisao_por_classe, revocacao_por_classe, f1_por_classe
from data_loader import filtrar_por_classes

bp_classificador = Blueprint('classificador', __name__)


def _parse_indices(param_string, padrao=None):
    """Converte string '2,3' em lista de inteiros [2, 3]."""
    if not param_string:
        return padrao or [2, 3]
    try:
        return [int(i.strip()) for i in param_string.split(',')]
    except ValueError:
        return padrao or [2, 3]


@bp_classificador.route('/api/prototypes', methods=['GET'])
def prototipos():
    """
    Retorna os prototipos (vetores medios) para os indices de atributos especificados.

    Query params:
        features  — indices separados por virgula (padrao: '2,3')
        dataset   — 'v1' (padrao) ou 'v2'
    """
    try:
        estado = current_app.config['ESTADO']
        indices = _parse_indices(
            request.args.get('features'),
            estado['INDICES_PETALA']
        )
        versao = request.args.get('dataset', 'v1')

        if versao == 'v2' and estado['treino_v2']:
            treino = estado['treino_v2']
        else:
            treino = estado['treino_v1']

        prototipos = treinar_prototipos(treino, indices)

        return jsonify({
            'indices_atributos': indices,
            'prototipos': {c: list(v) for c, v in prototipos.items()},
        })

    except Exception as e:
        return jsonify({'erro': f'Erro ao calcular prototipos: {str(e)}'}), 500


@bp_classificador.route('/api/classify', methods=['POST'])
def classificar():
    """
    Classifica uma amostra usando o classificador de distancia minima.

    Body JSON:
        x                  — lista de atributos (ex: [4.5, 1.5])
        indices_atributos  — lista de indices (ex: [2, 3])
        dataset            — 'v1' (padrao) ou 'v2'
    """
    try:
        estado = current_app.config['ESTADO']
        corpo = request.get_json()

        if not corpo or 'x' not in corpo:
            return jsonify({'erro': 'Campo "x" obrigatorio no body JSON'}), 400

        x = corpo['x']
        indices = corpo.get('indices_atributos', estado['INDICES_PETALA'])
        versao = corpo.get('dataset', 'v1')

        if versao == 'v2' and estado['treino_v2']:
            treino = estado['treino_v2']
        else:
            treino = estado['treino_v1']

        prototipos = treinar_prototipos(treino, indices)

        # Criar vetor completo se x ja estiver selecionado pelos indices
        if len(x) == len(indices):
            # x ja contem apenas os atributos selecionados — montar vetor completo
            x_completo = [0.0] * (max(indices) + 1)
            for idx, val in zip(indices, x):
                x_completo[idx] = val
        else:
            x_completo = x

        scores, vencedor = predizer_todas_classes(x_completo, prototipos, indices)

        return jsonify({
            'x': x,
            'indices_atributos': indices,
            'scores': scores,
            'classe_predita': vencedor,
        })

    except Exception as e:
        return jsonify({'erro': f'Erro ao classificar: {str(e)}'}), 500


@bp_classificador.route('/api/boundaries', methods=['GET'])
def fronteiras():
    """
    Retorna os coeficientes (w, b) das superficies de decisao para todos os
    pares de classes.

    Query params:
        features  — indices separados por virgula (padrao: '2,3')
        dataset   — 'v1' (padrao) ou 'v2'
    """
    try:
        estado = current_app.config['ESTADO']
        indices = _parse_indices(
            request.args.get('features'),
            estado['INDICES_PETALA']
        )
        versao = request.args.get('dataset', 'v1')

        if versao == 'v2' and estado['treino_v2']:
            treino = estado['treino_v2']
        else:
            treino = estado['treino_v1']

        prototipos = treinar_prototipos(treino, indices)
        pares = estado['PARES_BINARIOS']

        resultado = []
        for classe_i, classe_j in pares:
            pi = prototipos[classe_i]
            pj = prototipos[classe_j]
            w, b = coeficientes_superficie_decisao(pi, pj)
            resultado.append({
                'par': [classe_i, classe_j],
                'w': list(w),
                'b': b,
                'prototipo_i': list(pi),
                'prototipo_j': list(pj),
            })

        return jsonify({
            'indices_atributos': indices,
            'fronteiras': resultado,
        })

    except Exception as e:
        return jsonify({'erro': f'Erro ao calcular fronteiras: {str(e)}'}), 500


@bp_classificador.route('/api/distancia/metrics', methods=['GET'])
def metricas_distancia():
    """
    Retorna metricas completas do classificador de distancia minima
    (acuracia, matriz de confusao, precisao/revocacao/F1 por classe).

    Query params:
        features  — indices separados por virgula (padrao: '2,3')
        dataset   — 'v1' (padrao) ou 'v2'
    """
    try:
        estado = current_app.config['ESTADO']
        indices = _parse_indices(
            request.args.get('features'),
            estado['INDICES_PETALA']
        )
        versao = request.args.get('dataset', 'v1')

        if versao == 'v2' and estado['treino_v2']:
            treino = estado['treino_v2']
            teste = estado['teste_v2']
        else:
            treino = estado['treino_v1']
            teste = estado['teste_v1']

        classes = estado['CLASSES']
        prototipos = treinar_prototipos(treino, indices)

        # Predicoes no conjunto de teste
        predicoes = []
        gabarito = []
        for amostra in teste:
            _, vencedor = predizer_todas_classes(amostra['atributos'], prototipos, indices)
            predicoes.append(vencedor)
            gabarito.append(amostra['classe'])

        acc = acuracia(predicoes, gabarito)
        cm = matriz_confusao(predicoes, gabarito, classes)

        metricas_classe = {}
        for c in classes:
            metricas_classe[c] = {
                'precisao': round(precisao_por_classe(cm, c), 4),
                'revocacao': round(revocacao_por_classe(cm, c), 4),
                'f1': round(f1_por_classe(cm, c), 4),
            }

        # Metricas por par de classes (fronteiras)
        pares_metricas = []
        for classe_i, classe_j in estado['PARES_BINARIOS']:
            dados_par = filtrar_por_classes(teste, [classe_i, classe_j])
            pi = prototipos[classe_i]
            pj = prototipos[classe_j]

            pred_par = []
            gab_par = []
            for amostra in dados_par:
                pred = predizer_binario(
                    amostra['atributos'], pi, pj,
                    classe_i, classe_j, indices
                )
                pred_par.append(pred)
                gab_par.append(amostra['classe'])

            acc_par = acuracia(pred_par, gab_par)
            pares_metricas.append({
                'par': [classe_i, classe_j],
                'acuracia': round(acc_par, 4),
                'total_amostras': len(dados_par),
            })

        return jsonify({
            'indices_atributos': indices,
            'acuracia_global': round(acc, 4),
            'matriz_confusao': cm,
            'metricas_por_classe': metricas_classe,
            'metricas_por_par': pares_metricas,
            'prototipos': {c: list(v) for c, v in prototipos.items()},
            'total_treino': len(treino),
            'total_teste': len(teste),
        })

    except Exception as e:
        return jsonify({'erro': f'Erro ao calcular metricas: {str(e)}'}), 500
