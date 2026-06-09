"""
Rotas de gráficos — endpoints para gerar e retornar plots do matplotlib como imagens base64.

Endpoints:
    GET /api/plots/scatter     - Dispersão das 3 classes
    GET /api/plots/boundary    - Superfície de decisão binária (par de classes)
    GET /api/plots/confusion   - Heatmap da matriz de confusão para o classificador de distância mínima
    GET /api/plots/convergence - Gráfico de convergência (Erros/MSE por época) do Perceptron ou Delta
"""

from flask import Blueprint, request, jsonify, current_app
import io
import base64
import sys
import os

# Configura o matplotlib para o modo não interativo antes de qualquer importação
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Adiciona o diretório do classificador ao path para importação
DIRETORIO_API = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIRETORIO_PROJETO = os.path.dirname(DIRETORIO_API)
DIRETORIO_CLASSIFICADOR = os.path.join(DIRETORIO_PROJETO, 'iris_classifier')
if DIRETORIO_CLASSIFICADOR not in sys.path:
    sys.path.insert(0, DIRETORIO_CLASSIFICADOR)

from data_loader import split_estratificado, filtrar_por_classes
from classifier import treinar
from visualizer import (
    plotar_dispersao_todas_classes,
    plotar_superficie_decisao,
    plotar_matriz_confusao
)

bp_graficos = Blueprint('graficos', __name__)

def fig_para_base64():
    """Salva a figura atual em um buffer de memória e a converte para string base64."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=110)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close('all')  # Libera memória
    return f"data:image/png;base64,{img_base64}"

@bp_graficos.route('/api/plots/scatter', methods=['GET'])
def plot_scatter():
    """
    Retorna o gráfico de dispersão das classes do dataset.
    
    Query Params:
        dataset   - 'v1' ou 'v2'
        atributos - 'petalas', 'sepalas' ou 'todas'
    """
    try:
        estado = current_app.config['ESTADO']
        versao = request.args.get('dataset', 'v1')
        atributos = request.args.get('atributos', 'petalas')
        
        if versao == 'v2':
            if estado['dados_v2'] is None:
                return jsonify({'erro': 'Dataset v2 não disponível'}), 404
            dados = estado['dados_v2']
            treino = estado['treino_v2']
            teste = estado['teste_v2']
        else:
            dados = estado['dados_v1']
            treino = estado['treino_v1']
            teste = estado['teste_v1']
            
        if atributos == 'petalas':
            indices = [2, 3]
            nomes = ['Comprimento da Pétala', 'Largura da Pétala']
        elif atributos == 'sepalas':
            indices = [0, 1]
            nomes = ['Comprimento da Sépalas', 'Largura da Sépalas']
        else:
            indices = [0, 1, 2, 3]
            nomes = ['C. Sépala', 'L. Sépala', 'C. Pétala', 'L. Pétala']
            
        prototipos = treinar(treino, indices)
        
        # Plota usando o visualizer existente
        buf = io.BytesIO()
        plotar_dispersao_todas_classes(
            dados,
            indices,
            prototipos=prototipos,
            dados_treino=treino,
            dados_teste=teste,
            nomes_atributos=nomes,
            caminho_salvar=buf
        )
        
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close('all')
        
        return jsonify({'image': f"data:image/png;base64,{img_base64}"})
        
    except Exception as e:
        return jsonify({'erro': f'Erro ao gerar gráfico de dispersão: {str(e)}'}), 500

@bp_graficos.route('/api/plots/boundary', methods=['GET'])
def plot_boundary():
    """
    Retorna a superfície de decisão binária entre duas classes.
    
    Query Params:
        dataset   - 'v1' ou 'v2'
        atributos - 'petalas', 'sepalas'
        classe1   - Primeira classe (ex: 'setosa')
        classe2   - Segunda classe (ex: 'versicolor')
    """
    try:
        estado = current_app.config['ESTADO']
        versao = request.args.get('dataset', 'v1')
        atributos = request.args.get('atributos', 'petalas')
        classe1 = request.args.get('classe1')
        classe2 = request.args.get('classe2')
        
        if not classe1 or not classe2:
            return jsonify({'erro': 'Parâmetros classe1 e classe2 são obrigatórios'}), 400
            
        if versao == 'v2':
            if estado['dados_v2'] is None:
                return jsonify({'erro': 'Dataset v2 não disponível'}), 404
            treino = estado['treino_v2']
            teste = estado['teste_v2']
        else:
            treino = estado['treino_v1']
            teste = estado['teste_v1']
            
        if atributos == 'petalas':
            indices = [2, 3]
            nomes = ['Comprimento da Pétala', 'Largura da Pétala']
        else:
            indices = [0, 1]
            nomes = ['Comprimento da Sépalas', 'Largura da Sépalas']
            
        # Filtra dados para o par binário
        treino_bin = filtrar_por_classes(treino, [classe1, classe2])
        teste_bin = filtrar_por_classes(teste, [classe1, classe2])
        
        # Calcula protótipos locais
        prototipos = treinar(treino_bin, indices)
        pi = prototipos[classe1]
        pj = prototipos[classe2]
        
        dados_c1 = [d for d in treino_bin + teste_bin if d['classe'] == classe1]
        dados_c2 = [d for d in treino_bin + teste_bin if d['classe'] == classe2]
        
        buf = io.BytesIO()
        plotar_superficie_decisao(
            pi, pj,
            dados_c1, dados_c2,
            classe1, classe2,
            indices,
            dados_treino=treino_bin,
            dados_teste=teste_bin,
            nomes_atributos=nomes,
            titulo=f"Fronteira: {classe1.capitalize()} vs {classe2.capitalize()}",
            caminho_salvar=buf
        )
        
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close('all')
        
        return jsonify({'image': f"data:image/png;base64,{img_base64}"})
        
    except Exception as e:
        return jsonify({'erro': f'Erro ao gerar superfície de decisão: {str(e)}'}), 500

@bp_graficos.route('/api/plots/confusion', methods=['GET'])
def plot_confusion():
    """
    Retorna o heatmap da matriz de confusão.
    
    Query Params:
        modelo - Nome do modelo (ex: 'Dist. Minima', 'Perceptron 2a2', etc.)
    """
    try:
        estado = current_app.config['ESTADO']
        modelo_nome = request.args.get('modelo', 'Dist. Minima')
        
        resultados = estado.get('resultados_metricas')
        if not resultados or modelo_nome not in resultados:
            return jsonify({'erro': f'Treine todos os modelos primeiro ou especifique um modelo válido. ({modelo_nome} não encontrado)'}), 400
            
        report = resultados[modelo_nome]
        matriz = report['matriz']  # O formato original é dict de dicts
        classes = estado.get('classes_metricas', ['setosa', 'versicolor', 'virginica'])
        
        # Converte dict de dict para lista de listas
        matriz_lista = []
        for p in classes:
            linha = []
            for r in classes:
                linha.append(matriz.get(p, {}).get(r, 0))
            matriz_lista.append(linha)
            
        buf = io.BytesIO()
        plotar_matriz_confusao(
            matriz_lista,
            classes,
            titulo=f"Matriz de Confusao — {modelo_nome}",
            caminho_salvar=buf
        )
        
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close('all')
        
        return jsonify({'image': f"data:image/png;base64,{img_base64}"})
        
    except Exception as e:
        return jsonify({'erro': f'Erro ao gerar matriz de confusão: {str(e)}'}), 500

@bp_graficos.route('/api/plots/convergence', methods=['POST'])
def plot_convergence():
    """
    Retorna o gráfico de convergência de erros/MSE do Perceptron ou Delta.
    
    JSON Body:
        historico       - lista de floats/ints
        metrica_nome    - 'Erros de Classificacao' ou 'Erro Quadratico Medio (MSE)'
        modelo_titulo   - 'Perceptron' ou 'Regra Delta'
    """
    try:
        data = request.get_json() or {}
        historico = data.get('historico', [])
        metrica_nome = data.get('metrica_nome', 'Erros')
        modelo_titulo = data.get('modelo_titulo', 'Modelo')
        
        if not historico:
            return jsonify({'erro': 'Histórico vazio ou inválido'}), 400
            
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.set_facecolor('#13131d')
        fig.patch.set_facecolor('#06060b')
        
        # Plotagem estilizada
        ax.plot(range(1, len(historico) + 1), historico, color='#4a9eff', linewidth=2, label=metrica_nome)
        
        # Configuração dos eixos
        ax.set_title(f"Convergência do Treinamento — {modelo_titulo}", color='#e8e8f0', fontsize=12, pad=15)
        ax.set_xlabel("Épocas", color='#8888a0', fontsize=10)
        ax.set_ylabel(metrica_nome, color='#8888a0', fontsize=10)
        ax.tick_params(colors='#8888a0', which='both')
        
        # Grid estilizada
        ax.grid(True, color='rgba(255, 255, 255, 0.05)', linestyle='--')
        
        # Legenda
        leg = ax.legend(facecolor='#0d0d14', edgecolor='rgba(255, 255, 255, 0.1)')
        for text in leg.get_texts():
            text.set_color('#e8e8f0')
            
        # Bordas coloridas
        for spine in ax.spines.values():
            spine.set_color('rgba(255, 255, 255, 0.1)')
            
        # Base64
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=110)
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode('utf-8')
        plt.close('all')
        
        return jsonify({'image': f"data:image/png;base64,{img_base64}"})
        
    except Exception as e:
        return jsonify({'erro': f'Erro ao gerar gráfico de convergência: {str(e)}'}), 500
