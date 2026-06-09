"""
Rotas de dados — endpoints para consultar informacoes do dataset Iris.

Endpoints:
    GET /api/data          — Resumo do dataset (total, treino, teste, classes)
    GET /api/data/samples  — Amostras reais (com filtros via query params)
"""

from flask import Blueprint, request, jsonify, current_app

bp_dados = Blueprint('dados', __name__)


@bp_dados.route('/api/data', methods=['GET'])
def info_dataset():
    """
    Retorna informacoes gerais sobre o dataset carregado.

    Query params opcionais:
        dataset  — 'v1' (padrao) ou 'v2'
    """
    try:
        estado = current_app.config['ESTADO']
        versao = request.args.get('dataset', 'v1')

        if versao == 'v2':
            if estado['dados_v2'] is None:
                return jsonify({'erro': 'Dataset v2 nao disponivel'}), 404
            dados = estado['dados_v2']
            treino = estado['treino_v2']
            teste = estado['teste_v2']
        else:
            dados = estado['dados_v1']
            treino = estado['treino_v1']
            teste = estado['teste_v1']

        classes = sorted(list(set(d['classe'] for d in dados)))
        contagem_classes = {}
        for c in classes:
            contagem_classes[c] = sum(1 for d in dados if d['classe'] == c)

        return jsonify({
            'dataset': versao,
            'total': len(dados),
            'treino': len(treino),
            'teste': len(teste),
            'classes': classes,
            'contagem_por_classe': contagem_classes,
        })

    except Exception as e:
        return jsonify({'erro': f'Erro ao consultar dados: {str(e)}'}), 500


@bp_dados.route('/api/data/samples', methods=['GET'])
def amostras():
    """
    Retorna as amostras do dataset.

    Query params opcionais:
        dataset   — 'v1' (padrao) ou 'v2'
        split     — 'todos' (padrao), 'treino' ou 'teste'
        classe    — filtra por classe (ex: 'setosa')
        limite    — numero maximo de amostras retornadas
        offset    — deslocamento para paginacao
    """
    try:
        estado = current_app.config['ESTADO']
        versao = request.args.get('dataset', 'v1')
        split = request.args.get('split', 'todos')
        classe_filtro = request.args.get('classe', None)
        limite = request.args.get('limite', None, type=int)
        offset = request.args.get('offset', 0, type=int)

        if versao == 'v2':
            if estado['dados_v2'] is None:
                return jsonify({'erro': 'Dataset v2 nao disponivel'}), 404
            if split == 'treino':
                resultado = estado['treino_v2']
            elif split == 'teste':
                resultado = estado['teste_v2']
            else:
                resultado = estado['dados_v2']
        else:
            if split == 'treino':
                resultado = estado['treino_v1']
            elif split == 'teste':
                resultado = estado['teste_v1']
            else:
                resultado = estado['dados_v1']

        # Filtro por classe
        if classe_filtro:
            resultado = [d for d in resultado if d['classe'] == classe_filtro]

        total_filtrado = len(resultado)

        # Paginacao
        resultado = resultado[offset:]
        if limite:
            resultado = resultado[:limite]

        return jsonify({
            'dataset': versao,
            'split': split,
            'total_filtrado': total_filtrado,
            'retornados': len(resultado),
            'offset': offset,
            'amostras': resultado,
        })

    except Exception as e:
        return jsonify({'erro': f'Erro ao consultar amostras: {str(e)}'}), 500
