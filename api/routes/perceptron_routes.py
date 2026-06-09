"""
Rotas do Perceptron de Rosenblatt — endpoints para treinamento e predicao.

Endpoints:
    POST /api/perceptron/train   — Treina o Perceptron para um par de classes
    POST /api/perceptron/predict — Prediz uma amostra com pesos treinados
"""

from flask import Blueprint, request, jsonify, current_app

from data_loader import carregar_dados_iris, split_estratificado, filtrar_por_classes
from perceptron import treinar_perceptron, predizer_perceptron, acuracia_binaria_perceptron

bp_perceptron = Blueprint('perceptron', __name__)


@bp_perceptron.route('/api/perceptron/train', methods=['POST'])
def treinar():
    """
    Treina o Perceptron binario para um par de classes.

    Body JSON:
        classe_pos         — classe positiva (d=+1), ex: 'setosa'
        classe_neg         — classe negativa (d=-1), ex: 'versicolor'
        indices_atributos  — lista de indices (ex: [2, 3])
        taxa_aprendizado   — (opcional) padrao 0.03
        max_epocas         — (opcional) padrao 100
        proporcao_treino   — (opcional) padrao 0.7
        semente            — (opcional) padrao 42
        dataset            — (opcional) 'v1' ou 'v2'
    """
    try:
        estado = current_app.config['ESTADO']
        corpo = request.get_json()

        if not corpo:
            return jsonify({'erro': 'Body JSON obrigatorio'}), 400

        classe_pos = corpo.get('classe_pos')
        classe_neg = corpo.get('classe_neg')

        if not classe_pos or not classe_neg:
            return jsonify({
                'erro': 'Campos "classe_pos" e "classe_neg" sao obrigatorios'
            }), 400

        indices = corpo.get('indices_atributos', estado['INDICES_PETALA'])
        taxa = corpo.get('taxa_aprendizado', 0.03)
        max_epocas = corpo.get('max_epocas', 100)
        prop_treino = corpo.get('proporcao_treino', 0.7)
        semente = corpo.get('semente', 42)
        versao = corpo.get('dataset', 'v1')

        # Determinar dataset
        if versao == 'v2' and estado['dados_v2']:
            dados = estado['dados_v2']
        else:
            dados = estado['dados_v1']

        # Novo split se parametros customizados
        if prop_treino != 0.7 or semente != 42:
            treino, teste = split_estratificado(dados, prop_treino, semente)
        else:
            if versao == 'v2' and estado['treino_v2']:
                treino = estado['treino_v2']
                teste = estado['teste_v2']
            else:
                treino = estado['treino_v1']
                teste = estado['teste_v1']

        # Treinar
        w, historico_erros, epocas = treinar_perceptron(
            treino, classe_pos, classe_neg, indices,
            taxa_aprendizado=taxa, max_epocas=max_epocas
        )

        # Calcular acuracias
        acc_treino = acuracia_binaria_perceptron(
            treino, w, classe_pos, classe_neg, indices
        )
        acc_teste = acuracia_binaria_perceptron(
            teste, w, classe_pos, classe_neg, indices
        )

        # Convergencia
        convergiu = len(historico_erros) > 0 and historico_erros[-1] == 0

        # Salvar pesos no estado para uso posterior
        chave = f'{classe_pos}_vs_{classe_neg}'
        estado['perceptron_pesos'][chave] = {
            'w': w,
            'classe_pos': classe_pos,
            'classe_neg': classe_neg,
            'indices': indices,
        }
        estado['historicos'][f'perceptron_{chave}'] = historico_erros

        return jsonify({
            'modelo': 'perceptron',
            'classe_pos': classe_pos,
            'classe_neg': classe_neg,
            'indices_atributos': indices,
            'taxa_aprendizado': taxa,
            'max_epocas': max_epocas,
            'epocas_treinadas': epocas,
            'convergiu': convergiu,
            'pesos': w,
            'historico_erros': historico_erros,
            'acuracia_treino': round(acc_treino, 4),
            'acuracia_teste': round(acc_teste, 4),
            'total_treino': len(filtrar_por_classes(treino, [classe_pos, classe_neg])),
            'total_teste': len(filtrar_por_classes(teste, [classe_pos, classe_neg])),
        })

    except Exception as e:
        return jsonify({'erro': f'Erro ao treinar Perceptron: {str(e)}'}), 500


@bp_perceptron.route('/api/perceptron/predict', methods=['POST'])
def predizer():
    """
    Prediz uma amostra usando pesos do Perceptron ja treinados.

    Body JSON:
        x                  — atributos selecionados (ex: [4.5, 1.5])
        classe_pos         — classe positiva usada no treinamento
        classe_neg         — classe negativa usada no treinamento
        pesos              — (opcional) vetor de pesos [w0, w1, ..., wn]
                             Se nao fornecido, usa os ultimos pesos treinados.
    """
    try:
        estado = current_app.config['ESTADO']
        corpo = request.get_json()

        if not corpo or 'x' not in corpo:
            return jsonify({'erro': 'Campo "x" obrigatorio'}), 400

        x = corpo['x']
        classe_pos = corpo.get('classe_pos')
        classe_neg = corpo.get('classe_neg')

        # Pesos fornecidos ou salvos
        pesos = corpo.get('pesos')
        if pesos is None:
            if not classe_pos or not classe_neg:
                return jsonify({
                    'erro': 'Forneça "pesos" ou "classe_pos"/"classe_neg" para usar pesos salvos'
                }), 400
            chave = f'{classe_pos}_vs_{classe_neg}'
            salvo = estado['perceptron_pesos'].get(chave)
            if not salvo:
                return jsonify({
                    'erro': f'Nenhum Perceptron treinado para {chave}. Treine primeiro via POST /api/perceptron/train'
                }), 404
            pesos = salvo['w']
            classe_pos = salvo['classe_pos']
            classe_neg = salvo['classe_neg']

        resultado = predizer_perceptron(x, pesos)
        classe_predita = classe_pos if resultado == 1 else classe_neg

        return jsonify({
            'x': x,
            'saida_numerica': resultado,
            'classe_predita': classe_predita,
            'classe_pos': classe_pos,
            'classe_neg': classe_neg,
        })

    except Exception as e:
        return jsonify({'erro': f'Erro ao predizer com Perceptron: {str(e)}'}), 500
