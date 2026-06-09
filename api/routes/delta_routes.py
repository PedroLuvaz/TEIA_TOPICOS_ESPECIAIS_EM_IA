"""
Rotas da Regra Delta (Widrow-Hoff / Adaline) — endpoints para treinamento
binario, multiclasse (OvA), XOR e predicao.

Endpoints:
    POST /api/delta/train       — Treina Regra Delta binaria (ou XOR)
    POST /api/delta-ova/train   — Treina Regra Delta multiclasse (One-vs-All)
    POST /api/delta/predict     — Prediz com pesos da Regra Delta binaria
    POST /api/delta-ova/predict — Prediz com pesos OvA (multiclasse)
"""

from flask import Blueprint, request, jsonify, current_app

from data_loader import split_estratificado, filtrar_por_classes
from delta_rule import (
    treinar_delta_iris, predizer_delta, acuracia_binaria_delta,
    treinar_delta_ova, predizer_delta_ova, acuracia_delta_ova,
    treinar_delta_xor, predizer_delta_xor,
)

bp_delta = Blueprint('delta', __name__)


@bp_delta.route('/api/delta/train', methods=['POST'])
def treinar_delta():
    """
    Treina a Regra Delta para um par de classes ou para o problema XOR.

    Body JSON (Iris binario):
        classe_pos         — classe positiva (d=+1)
        classe_neg         — classe negativa (d=-1)
        indices_atributos  — lista de indices (ex: [2, 3])
        taxa_aprendizado   — (opcional) padrao 0.02
        max_epocas         — (opcional) padrao 100
        proporcao_treino   — (opcional) padrao 0.7
        semente            — (opcional) padrao 42
        dataset            — (opcional) 'v1' ou 'v2'

    Body JSON (XOR):
        modo               — 'xor'
        taxa_aprendizado   — (opcional) padrao 0.02
        max_epocas         — (opcional) padrao 200
    """
    try:
        estado = current_app.config['ESTADO']
        corpo = request.get_json()

        if not corpo:
            return jsonify({'erro': 'Body JSON obrigatorio'}), 400

        # ---------------------------------------------------------------
        # Modo XOR
        # ---------------------------------------------------------------
        if corpo.get('modo') == 'xor':
            taxa = corpo.get('taxa_aprendizado', 0.02)
            max_epocas = corpo.get('max_epocas', 200)

            w, historico_mse = treinar_delta_xor(
                max_epocas=max_epocas, taxa_aprendizado=taxa
            )

            # Testar predicoes nos 4 padroes
            padroes = [(0, 0), (0, 1), (1, 0), (1, 1)]
            esperado = [0, 1, 1, 0]
            predicoes = [predizer_delta_xor(x1, x2, w) for x1, x2 in padroes]
            acertos = sum(1 for p, e in zip(predicoes, esperado) if p == e)

            estado['historicos']['delta_xor'] = historico_mse

            return jsonify({
                'modelo': 'delta_xor',
                'pesos': w,
                'historico_mse': historico_mse,
                'max_epocas': max_epocas,
                'taxa_aprendizado': taxa,
                'predicoes': [
                    {'entrada': list(p), 'esperado': e, 'predito': pr}
                    for p, e, pr in zip(padroes, esperado, predicoes)
                ],
                'acertos': acertos,
                'total': 4,
                'mse_final': round(historico_mse[-1], 6) if historico_mse else None,
                'mensagem': 'XOR nao e linearmente separavel — MSE nunca atinge zero',
            })

        # ---------------------------------------------------------------
        # Modo Iris binario
        # ---------------------------------------------------------------
        classe_pos = corpo.get('classe_pos')
        classe_neg = corpo.get('classe_neg')

        if not classe_pos or not classe_neg:
            return jsonify({
                'erro': 'Campos "classe_pos" e "classe_neg" sao obrigatorios (ou use modo="xor")'
            }), 400

        indices = corpo.get('indices_atributos', estado['INDICES_PETALA'])
        taxa = corpo.get('taxa_aprendizado', 0.02)
        max_epocas = corpo.get('max_epocas', 100)
        prop_treino = corpo.get('proporcao_treino', 0.7)
        semente = corpo.get('semente', 42)
        versao = corpo.get('dataset', 'v1')

        if versao == 'v2' and estado['dados_v2']:
            dados = estado['dados_v2']
        else:
            dados = estado['dados_v1']

        if prop_treino != 0.7 or semente != 42:
            treino, teste = split_estratificado(dados, prop_treino, semente)
        else:
            if versao == 'v2' and estado['treino_v2']:
                treino = estado['treino_v2']
                teste = estado['teste_v2']
            else:
                treino = estado['treino_v1']
                teste = estado['teste_v1']

        w, historico_mse, epocas = treinar_delta_iris(
            treino, classe_pos, classe_neg, indices,
            taxa_aprendizado=taxa, max_epocas=max_epocas
        )

        acc_treino = acuracia_binaria_delta(treino, w, classe_pos, classe_neg, indices)
        acc_teste = acuracia_binaria_delta(teste, w, classe_pos, classe_neg, indices)

        # Salvar pesos
        chave = f'{classe_pos}_vs_{classe_neg}'
        estado['delta_pesos'][chave] = {
            'w': w,
            'classe_pos': classe_pos,
            'classe_neg': classe_neg,
            'indices': indices,
        }
        estado['historicos'][f'delta_{chave}'] = historico_mse

        return jsonify({
            'modelo': 'delta_iris',
            'classe_pos': classe_pos,
            'classe_neg': classe_neg,
            'indices_atributos': indices,
            'taxa_aprendizado': taxa,
            'max_epocas': max_epocas,
            'epocas_treinadas': epocas,
            'pesos': w,
            'historico_mse': historico_mse,
            'mse_final': round(historico_mse[-1], 6) if historico_mse else None,
            'acuracia_treino': round(acc_treino, 4),
            'acuracia_teste': round(acc_teste, 4),
            'total_treino': len(filtrar_por_classes(treino, [classe_pos, classe_neg])),
            'total_teste': len(filtrar_por_classes(teste, [classe_pos, classe_neg])),
        })

    except Exception as e:
        return jsonify({'erro': f'Erro ao treinar Regra Delta: {str(e)}'}), 500


@bp_delta.route('/api/delta-ova/train', methods=['POST'])
def treinar_ova():
    """
    Treina a Regra Delta no esquema One-vs-All (multiclasse).

    Body JSON:
        indices_atributos  — lista de indices (ex: [2, 3])
        taxa_aprendizado   — (opcional) padrao 0.02
        max_epocas         — (opcional) padrao 200
        proporcao_treino   — (opcional) padrao 0.7
        semente            — (opcional) padrao 42
        dataset            — (opcional) 'v1' ou 'v2'
    """
    try:
        estado = current_app.config['ESTADO']
        corpo = request.get_json() or {}

        indices = corpo.get('indices_atributos', estado['INDICES_PETALA'])
        taxa = corpo.get('taxa_aprendizado', 0.02)
        max_epocas = corpo.get('max_epocas', 200)
        prop_treino = corpo.get('proporcao_treino', 0.7)
        semente = corpo.get('semente', 42)
        versao = corpo.get('dataset', 'v1')

        if versao == 'v2' and estado['dados_v2']:
            dados = estado['dados_v2']
        else:
            dados = estado['dados_v1']

        if prop_treino != 0.7 or semente != 42:
            treino, teste = split_estratificado(dados, prop_treino, semente)
        else:
            if versao == 'v2' and estado['treino_v2']:
                treino = estado['treino_v2']
                teste = estado['teste_v2']
            else:
                treino = estado['treino_v1']
                teste = estado['teste_v1']

        pesos, historico, n_epocas = treinar_delta_ova(
            treino, indices,
            taxa_aprendizado=taxa, max_epocas=max_epocas
        )

        acc_treino = acuracia_delta_ova(treino, pesos, indices)
        acc_teste = acuracia_delta_ova(teste, pesos, indices)

        # Salvar pesos
        estado['delta_ova_pesos'] = {
            'pesos': {c: list(w) for c, w in pesos.items()},
            'indices': indices,
        }
        for c, hist in historico.items():
            estado['historicos'][f'delta_ova_{c}'] = hist

        return jsonify({
            'modelo': 'delta_ova',
            'indices_atributos': indices,
            'taxa_aprendizado': taxa,
            'max_epocas': max_epocas,
            'epocas_treinadas': n_epocas,
            'pesos': {c: list(w) for c, w in pesos.items()},
            'historico_mse': {c: list(h) for c, h in historico.items()},
            'acuracia_treino': round(acc_treino, 4),
            'acuracia_teste': round(acc_teste, 4),
            'total_treino': len(treino),
            'total_teste': len(teste),
        })

    except Exception as e:
        return jsonify({'erro': f'Erro ao treinar Delta OvA: {str(e)}'}), 500


@bp_delta.route('/api/delta/predict', methods=['POST'])
def predizer_delta_rota():
    """
    Prediz uma amostra usando pesos da Regra Delta binaria.

    Body JSON:
        x                  — atributos selecionados (ex: [4.5, 1.5])
        classe_pos         — classe positiva usada no treinamento
        classe_neg         — classe negativa usada no treinamento
        pesos              — (opcional) vetor de pesos [w0, w1, ..., wn]
    """
    try:
        estado = current_app.config['ESTADO']
        corpo = request.get_json()

        if not corpo or 'x' not in corpo:
            return jsonify({'erro': 'Campo "x" obrigatorio'}), 400

        x = corpo['x']
        classe_pos = corpo.get('classe_pos')
        classe_neg = corpo.get('classe_neg')

        pesos_w = corpo.get('pesos')
        if pesos_w is None:
            if not classe_pos or not classe_neg:
                return jsonify({
                    'erro': 'Forneca "pesos" ou "classe_pos"/"classe_neg" para usar pesos salvos'
                }), 400
            chave = f'{classe_pos}_vs_{classe_neg}'
            salvo = estado['delta_pesos'].get(chave)
            if not salvo:
                return jsonify({
                    'erro': f'Nenhuma Regra Delta treinada para {chave}. Treine primeiro.'
                }), 404
            pesos_w = salvo['w']
            classe_pos = salvo['classe_pos']
            classe_neg = salvo['classe_neg']

        classe_predita = predizer_delta(x, pesos_w, classe_pos, classe_neg)

        return jsonify({
            'x': x,
            'classe_predita': classe_predita,
            'classe_pos': classe_pos,
            'classe_neg': classe_neg,
        })

    except Exception as e:
        return jsonify({'erro': f'Erro ao predizer com Delta: {str(e)}'}), 500


@bp_delta.route('/api/delta-ova/predict', methods=['POST'])
def predizer_ova_rota():
    """
    Prediz uma amostra usando pesos da Regra Delta OvA (multiclasse).

    Body JSON:
        x     — atributos selecionados (ex: [4.5, 1.5])
        pesos — (opcional) dict {classe: [w0, w1, ..., wn]}
    """
    try:
        estado = current_app.config['ESTADO']
        corpo = request.get_json()

        if not corpo or 'x' not in corpo:
            return jsonify({'erro': 'Campo "x" obrigatorio'}), 400

        x = corpo['x']

        pesos_dict = corpo.get('pesos')
        if pesos_dict is None:
            salvo = estado.get('delta_ova_pesos')
            if not salvo:
                return jsonify({
                    'erro': 'Nenhuma Regra Delta OvA treinada. Treine primeiro via POST /api/delta-ova/train'
                }), 404
            pesos_dict = salvo['pesos']

        classe_predita, nets = predizer_delta_ova(x, pesos_dict)

        return jsonify({
            'x': x,
            'classe_predita': classe_predita,
            'nets': nets,
        })

    except Exception as e:
        return jsonify({'erro': f'Erro ao predizer com Delta OvA: {str(e)}'}), 500
