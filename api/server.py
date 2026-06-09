"""
Servidor Flask — API backend do Classificador Iris.

Responsabilidades:
    1. Adicionar iris_classifier/ ao sys.path para importar os modulos puros.
    2. Carregar os datasets (v1 e v2 quando disponivel) ao iniciar.
    3. Pre-calcular prototipos para os indices padrao [2, 3] (petalas).
    4. Registrar todos os Blueprints de rotas (dados, classificador, perceptron,
       regra delta, metricas avancadas e graficos).
    5. Servir o diretorio frontend/ como arquivos estaticos.
    6. Executar na porta 5000 em modo debug.

Todas as respostas sao JSON com Content-Type adequado.
"""

import os
import sys
import mimetypes

# Corrige problema de tipo MIME no Windows para arquivos estáticos JS/CSS
mimetypes.add_type('application/javascript', '.js')
mimetypes.add_type('text/css', '.css')

# ---------------------------------------------------------------------------
# Configuracao de caminhos
# ---------------------------------------------------------------------------
DIRETORIO_API = os.path.dirname(os.path.abspath(__file__))
DIRETORIO_PROJETO = os.path.dirname(DIRETORIO_API)
DIRETORIO_CLASSIFICADOR = os.path.join(DIRETORIO_PROJETO, 'iris_classifier')
DIRETORIO_DADOS = os.path.join(DIRETORIO_PROJETO, 'data')
DIRETORIO_FRONTEND = os.path.join(DIRETORIO_PROJETO, 'frontend')

# Garante que o pacote iris_classifier pode ser importado
sys.path.insert(0, DIRETORIO_CLASSIFICADOR)
sys.path.insert(0, DIRETORIO_PROJETO)

# ---------------------------------------------------------------------------
# Importacoes Flask
# ---------------------------------------------------------------------------
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

# ---------------------------------------------------------------------------
# Importacoes dos modulos do classificador
# ---------------------------------------------------------------------------
from data_loader import carregar_dados_iris, split_estratificado
from classifier import treinar as treinar_prototipos

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
CLASSES = ['setosa', 'versicolor', 'virginica']
PARES_BINARIOS = [
    ('setosa', 'versicolor'),
    ('versicolor', 'virginica'),
    ('setosa', 'virginica'),
]
INDICES_PETALA = [2, 3]
INDICES_SEPALA = [0, 1]
INDICES_TODOS = [0, 1, 2, 3]

# ---------------------------------------------------------------------------
# Carregamento dos dados na inicializacao
# ---------------------------------------------------------------------------
CAMINHO_V1 = os.path.join(DIRETORIO_DADOS, 'Iris data.xls')
CAMINHO_V2 = os.path.join(DIRETORIO_DADOS, 'iris_data_02.xlsx')

print('[servidor] Carregando dataset v1...')
dados_v1 = carregar_dados_iris(CAMINHO_V1)
print(f'[servidor] Dataset v1 carregado: {len(dados_v1)} amostras')

dados_v2 = None
if os.path.exists(CAMINHO_V2):
    print('[servidor] Carregando dataset v2...')
    dados_v2 = carregar_dados_iris(CAMINHO_V2)
    print(f'[servidor] Dataset v2 carregado: {len(dados_v2)} amostras')
else:
    print('[servidor] Dataset v2 nao encontrado — sera ignorado.')

# Split padrao (70/30, semente 42)
treino_v1, teste_v1 = split_estratificado(dados_v1, proporcao_treino=0.7, semente=42)
print(f'[servidor] Split v1 — Treino: {len(treino_v1)} | Teste: {len(teste_v1)}')

treino_v2, teste_v2 = (None, None)
if dados_v2:
    treino_v2, teste_v2 = split_estratificado(dados_v2, proporcao_treino=0.7, semente=42)
    print(f'[servidor] Split v2 — Treino: {len(treino_v2)} | Teste: {len(teste_v2)}')

# Prototipos padrao (petalas)
prototipos_padrao = treinar_prototipos(treino_v1, INDICES_PETALA)
print(f'[servidor] Prototipos padrao calculados (indices {INDICES_PETALA}):')
for classe, vetor in prototipos_padrao.items():
    print(f'  {classe}: {[round(v, 4) for v in vetor]}')

# ---------------------------------------------------------------------------
# Estado global compartilhado entre rotas
# ---------------------------------------------------------------------------
estado = {
    'dados_v1': dados_v1,
    'dados_v2': dados_v2,
    'treino_v1': treino_v1,
    'teste_v1': teste_v1,
    'treino_v2': treino_v2,
    'teste_v2': teste_v2,
    'prototipos_padrao': prototipos_padrao,
    'CLASSES': CLASSES,
    'PARES_BINARIOS': PARES_BINARIOS,
    'INDICES_PETALA': INDICES_PETALA,
    'INDICES_SEPALA': INDICES_SEPALA,
    'INDICES_TODOS': INDICES_TODOS,
    # Pesos treinados (preenchidos sob demanda pelas rotas)
    'perceptron_pesos': {},
    'delta_pesos': {},
    'delta_ova_pesos': {},
    'historicos': {},
}

# ---------------------------------------------------------------------------
# Criacao da aplicacao Flask
# ---------------------------------------------------------------------------
app = Flask(
    __name__,
    static_folder=DIRETORIO_FRONTEND,
    static_url_path='',
)
CORS(app)

# Disponibilizar estado para os Blueprints
app.config['ESTADO'] = estado

# ---------------------------------------------------------------------------
# Registro dos Blueprints
# ---------------------------------------------------------------------------
from routes.data_routes import bp_dados
from routes.classifier_routes import bp_classificador
from routes.perceptron_routes import bp_perceptron
from routes.delta_routes import bp_delta
from routes.metrics_routes import bp_metricas
from routes.plots_routes import bp_graficos

app.register_blueprint(bp_dados)
app.register_blueprint(bp_classificador)
app.register_blueprint(bp_perceptron)
app.register_blueprint(bp_delta)
app.register_blueprint(bp_metricas)
app.register_blueprint(bp_graficos)


# ---------------------------------------------------------------------------
# Rota raiz — serve o frontend
# ---------------------------------------------------------------------------
@app.route('/')
def raiz():
    """Serve o index.html do frontend."""
    return send_from_directory(DIRETORIO_FRONTEND, 'index.html')


@app.route('/<path:caminho>')
def arquivos_estaticos(caminho):
    """Serve qualquer arquivo estatico do frontend."""
    return send_from_directory(DIRETORIO_FRONTEND, caminho)


# ---------------------------------------------------------------------------
# Tratamento global de erros
# ---------------------------------------------------------------------------
@app.errorhandler(404)
def nao_encontrado(erro):
    return jsonify({'erro': 'Recurso nao encontrado', 'codigo': 404}), 404


@app.errorhandler(500)
def erro_interno(erro):
    return jsonify({'erro': 'Erro interno do servidor', 'codigo': 500}), 500


# ---------------------------------------------------------------------------
# Ponto de entrada
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print(f'\n[servidor] Frontend servido de: {DIRETORIO_FRONTEND}')
    print('[servidor] Iniciando na porta 5000...\n')
    app.run(host='0.0.0.0', port=5000, debug=True)
