"""
Rotas de métricas — endpoints para calcular métricas avançadas e testes de hipóteses (teste Z).

Endpoints:
    POST /api/metricas/train-all - Treina todos os 6 classificadores e retorna os relatórios completos.
    POST /api/metricas/z-test    - Executa o teste Z de Kappa e Tau para comparar 2 classificadores.
"""

from flask import Blueprint, request, jsonify, current_app
import math
import sys
import os

# Adiciona o diretório do classificador ao path para importação
DIRETORIO_API = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRETORIO_PROJETO = os.path.dirname(DIRETORIO_API)
DIRETORIO_CLASSIFICADOR = os.path.join(DIRETORIO_PROJETO, 'iris_classifier')
if DIRETORIO_CLASSIFICADOR not in sys.path:
    sys.path.insert(0, DIRETORIO_CLASSIFICADOR)

from data_loader import split_estratificado, filtrar_por_classes
from classifier import treinar, predizer_todas_classes, predizer_binario
from math_utils import distancia_euclidiana
from perceptron import treinar_perceptron, predizer_perceptron
from delta_rule import treinar_delta_iris, treinar_delta_ova, predizer_delta_ova
from metricas_avancadas import (
    relatorio_completo, z_kappa, z_tau, p_valor_z
)

bp_metricas = Blueprint('metricas', __name__)

@bp_metricas.route('/api/metricas/train-all', methods=['POST'])
def treinar_todos_classificadores():
    """
    Treina todos os 6 classificadores simultaneamente nos dados selecionados
    e retorna o relatório completo de métricas de cada um.
    
    JSON Body:
        dataset           - 'v1' ou 'v2' (padrão 'v1')
        atributos         - 'petalas', 'sepalas' ou 'todas' (padrão 'petalas')
        proporcao_treino  - float de 0.1 a 0.9 (padrão 0.7)
        semente           - int (padrão 42)
        comparacao        - 'todas', 'setosa_versicolor', 'versicolor_virginica', 'setosa_virginica' (padrão 'todas')
    """
    try:
        estado = current_app.config['ESTADO']
        data = request.get_json() or {}
        
        versao = data.get('dataset', 'v1')
        atributos = data.get('atributos', 'petalas')
        prop_treino = float(data.get('proporcao_treino', 0.7))
        semente = int(data.get('semente', 42))
        comparacao = data.get('comparacao', 'todas')
        
        # Seleciona o dataset base
        if versao == 'v2':
            if estado['dados_v2'] is None:
                return jsonify({'erro': 'Dataset v2 não disponível'}), 404
            dados_base = estado['dados_v2']
        else:
            dados_base = estado['dados_v1']
            
        # Define os índices dos atributos
        if atributos == 'petalas':
            indices = [2, 3]
        elif atributos == 'sepalas':
            indices = [0, 1]
        else:
            indices = [0, 1, 2, 3]
            
        # Split estratificado personalizado se semente ou proporção mudar
        treino, teste = split_estratificado(dados_base, proporcao_treino=prop_treino, semente=semente)
        
        # Filtra classes se for comparação binária
        classes_sel = ['setosa', 'versicolor', 'virginica']
        if comparacao == 'setosa_versicolor':
            classes_sel = ['setosa', 'versicolor']
        elif comparacao == 'versicolor_virginica':
            classes_sel = ['versicolor', 'virginica']
        elif comparacao == 'setosa_virginica':
            classes_sel = ['setosa', 'virginica']
            
        if comparacao != 'todas':
            treino = filtrar_por_classes(treino, classes_sel)
            teste = filtrar_por_classes(teste, classes_sel)
            
        resultados = {}
        preds_por_modelo = {}
        
        # Função auxiliar para registrar predições no dicionário final
        def registrar(nome, preds, gab):
            report = relatorio_completo(preds, gab, classes_sel, nome)
            resultados[nome] = report
            preds_por_modelo[nome] = (preds, gab)
            
        # 1. Distância Mínima
        preds, gab = _pred_dist_minima(treino, teste, classes_sel, indices)
        registrar('Dist. Minima', preds, gab)
        
        # 2. Distância Máxima
        preds, gab = _pred_dist_maxima(treino, teste, classes_sel, indices)
        registrar('Dist. Maxima', preds, gab)
        
        # 3. Superfície de Decisão 2a2 (ou binária)
        nome_sup = 'Superficie 2a2' if len(classes_sel) == 3 else 'Superficie Binaria'
        preds, gab = _pred_ova_superficie(treino, teste, classes_sel, indices)
        registrar(nome_sup, preds, gab)
        
        # 4. Perceptron 2a2 (ou binário)
        nome_perc = 'Perceptron 2a2' if len(classes_sel) == 3 else 'Perceptron Binario'
        preds, gab = _pred_perceptron_ova(treino, teste, classes_sel, indices)
        registrar(nome_perc, preds, gab)
        
        # 5. Delta Binária OvA (ou binário)
        nome_delta_bin = 'Delta Bin. OvA' if len(classes_sel) == 3 else 'Delta Binario'
        preds, gab = _pred_delta_bin_ova(treino, teste, classes_sel, indices)
        registrar(nome_delta_bin, preds, gab)
        
        # 6. Delta OvA (ou binário por nets)
        nome_delta_ova = 'Delta OvA' if len(classes_sel) == 3 else 'Delta Binario (Nets)'
        preds, gab = _pred_delta_ova(treino, teste, classes_sel, indices)
        registrar(nome_delta_ova, preds, gab)
        
        # Salva o resultado no estado da aplicação para uso posterior (ex: gráfico ou Z-test)
        estado['resultados_metricas'] = resultados
        estado['classes_metricas'] = classes_sel
        
        return jsonify({
            'resultados': resultados,
            'classes': classes_sel,
            'indices': indices
        })
        
    except Exception as e:
        return jsonify({'erro': f'Erro ao treinar modelos: {str(e)}'}), 500

@bp_metricas.route('/api/metricas/z-test', methods=['POST'])
def teste_z_comparacao():
    """
    Executa o teste Z de hipóteses para Kappa e Tau entre dois modelos.
    
    JSON Body:
        modelo_a - Nome do classificador A
        modelo_b - Nome do classificador B
    """
    try:
        estado = current_app.config['ESTADO']
        data = request.get_json() or {}
        
        modelo_a = data.get('modelo_a')
        modelo_b = data.get('modelo_b')
        
        resultados = estado.get('resultados_metricas')
        if not resultados:
            return jsonify({'erro': 'Modelos ainda não foram treinados. Chame /api/metricas/train-all primeiro.'}), 400
            
        m_a = resultados.get(modelo_a)
        m_b = resultados.get(modelo_b)
        
        if not m_a or not m_b:
            return jsonify({'erro': f'Modelos especificados não encontrados ({modelo_a} ou {modelo_b})'}), 404
            
        k1, vk1 = m_a['kappa'], m_a['variancia_kappa']
        k2, vk2 = m_b['kappa'], m_b['variancia_kappa']
        t1, vt1 = m_a['tau'], m_a['variancia_tau']
        t2, vt2 = m_b['tau'], m_b['variancia_tau']
        
        zk = z_kappa(k1, vk1, k2, vk2)
        zt = z_tau(t1, vt1, t2, vt2)
        pzk = p_valor_z(zk)
        pzt = p_valor_z(zt)
        
        return jsonify({
            'modelo_a': modelo_a,
            'modelo_b': modelo_b,
            'kappa': {
                'val_a': k1,
                'val_b': k2,
                'var_a': vk1,
                'var_b': vk2,
                'z_calculado': zk,
                'p_valor': pzk,
                'significativo': pzk < 0.05
            },
            'tau': {
                'val_a': t1,
                'val_b': t2,
                'var_a': vt1,
                'var_b': vt2,
                'z_calculado': zt,
                'p_valor': pzt,
                'significativo': pzt < 0.05
            }
        })
        
    except Exception as e:
        return jsonify({'erro': f'Erro no teste Z: {str(e)}'}), 500

# ---------------------------------------------------------------------------
# Funções Auxiliares de Predição (Idênticas à tab_metricas_avancadas.py)
# ---------------------------------------------------------------------------
def _pred_dist_minima(treino, teste, classes, indices):
    proto = treinar(treino, indices)
    preds, gab = [], []
    for a in teste:
        _, pred = predizer_todas_classes(a['atributos'], proto, indices)
        preds.append(pred)
        gab.append(a['classe'])
    return preds, gab

def _pred_dist_maxima(treino, teste, classes, indices):
    proto = treinar(treino, indices)
    preds, gab = [], []
    for a in teste:
        x = [a['atributos'][i] for i in indices]
        dists = {c: distancia_euclidiana(x, proto[c]) for c in classes}
        preds.append(max(dists, key=dists.get))
        gab.append(a['classe'])
    return preds, gab

def _pred_ova_superficie(treino, teste, classes, indices):
    proto = treinar(treino, indices)
    preds, gab = [], []
    
    pares_locais = [('setosa', 'versicolor'), ('versicolor', 'virginica'), ('setosa', 'virginica')]
    if len(classes) == 2:
        pares_locais = [(classes[0], classes[1])]

    for a in teste:
        votos = {c: 0 for c in classes}
        for ci, cj in pares_locais:
            if ci in classes and cj in classes:
                venc = predizer_binario(a['atributos'], proto[ci], proto[cj], ci, cj, indices)
                votos[venc] += 1
        preds.append(max(votos, key=votos.get))
        gab.append(a['classe'])
    return preds, gab

def _pred_perceptron_ova(treino, teste, classes, indices):
    if len(classes) == 2:
        cp, cn = classes[0], classes[1]
        w, _, _ = treinar_perceptron(treino, cp, cn, indices, 0.03, 200)
        preds, gab = [], []
        for a in teste:
            x = [a['atributos'][i] for i in indices]
            y = predizer_perceptron(x, w)
            preds.append(cp if y == 1 else cn)
            gab.append(a['classe'])
        return preds, gab
    else:
        # Perceptron Hierárquico 2a2
        treino_s1 = []
        for d in treino:
            lbl = 'setosa' if d['classe'] == 'setosa' else 'not_setosa'
            treino_s1.append({'atributos': d['atributos'], 'classe': lbl})
        w1, _, _ = treinar_perceptron(treino_s1, 'setosa', 'not_setosa', indices, 0.03, 200)
        
        treino_s2 = filtrar_por_classes(treino, ['versicolor', 'virginica'])
        w2, _, _ = treinar_perceptron(treino_s2, 'versicolor', 'virginica', indices, 0.03, 200)
        
        preds, gab = [], []
        for a in teste:
            x = [a['atributos'][i] for i in indices]
            y1 = predizer_perceptron(x, w1)
            if y1 == 1:
                preds.append('setosa')
            else:
                y2 = predizer_perceptron(x, w2)
                preds.append('versicolor' if y2 == 1 else 'virginica')
            gab.append(a['classe'])
        return preds, gab

def _pred_delta_bin_ova(treino, teste, classes, indices):
    if len(classes) == 2:
        cp, cn = classes[0], classes[1]
        w, _, _ = treinar_delta_iris(treino, cp, cn, indices, 0.02, 300)
        preds, gab = [], []
        for a in teste:
            x = [1.0] + [a['atributos'][i] for i in indices]
            net = sum(wi * xi for wi, xi in zip(w, x))
            preds.append(cp if net >= 0 else cn)
            gab.append(a['classe'])
        return preds, gab
    else:
        pesos = {}
        pares_locais = [('setosa', 'versicolor'), ('versicolor', 'virginica'), ('setosa', 'virginica')]
        for cp, cn in pares_locais:
            treino_par = filtrar_por_classes(treino, [cp, cn])
            w, _, _ = treinar_delta_iris(treino_par, cp, cn, indices, 0.02, 300)
            pesos[(cp, cn)] = (w, cp, cn)
        preds, gab = [], []
        for a in teste:
            votos = {c: 0 for c in classes}
            for (w, cp, cn) in pesos.values():
                x = [1.0] + [a['atributos'][i] for i in indices]
                net = sum(wi * xi for wi, xi in zip(w, x))
                votos[cp if net >= 0 else cn] += 1
            preds.append(max(votos, key=votos.get))
            gab.append(a['classe'])
        return preds, gab

def _pred_delta_ova(treino, teste, classes, indices):
    if len(classes) == 2:
        return _pred_delta_bin_ova(treino, teste, classes, indices)
    
    pesos, _, _ = treinar_delta_ova(treino, indices, 0.02, 300)
    preds, gab = [], []
    for a in teste:
        x = [a['atributos'][i] for i in indices]
        pred, _ = predizer_delta_ova(x, pesos)
        preds.append(pred)
        gab.append(a['classe'])
    return preds, gab
