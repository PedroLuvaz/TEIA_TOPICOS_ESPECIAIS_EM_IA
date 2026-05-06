import os
import sys

# Adiciona o diretório pai ao path para permitir imports dos módulos locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import carregar_dados_iris, split_estratificado, filtrar_por_classes
from classifier import treinar, predizer_todas_classes, predizer_binario
from evaluator import acuracia, matriz_confusao, imprimir_matriz_confusao
from visualizer import plotar_superficie_decisao, plotar_dispersao_todas_classes

# Configurações de Caminhos
RAIZ_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_DADOS = os.path.join(RAIZ_PROJETO, "data", "Iris data.xls")
PASTA_OUTPUTS = os.path.join(RAIZ_PROJETO, "outputs")

# Configurações do Experimento
INDICES_ATRIBUTOS = [2, 3]  # Comprimento da Pétala, Largura da Pétala
CLASSES = ['setosa', 'versicolor', 'virginica']

def executar_experimentos():
    # Garantir que a pasta de outputs existe
    if not os.path.exists(PASTA_OUTPUTS):
        os.makedirs(PASTA_OUTPUTS)

    # 1. Carregar dados
    if not os.path.exists(CAMINHO_DADOS):
        print(f"Erro: Arquivo {CAMINHO_DADOS} não encontrado.")
        return
        
    dados = carregar_dados_iris(CAMINHO_DADOS)
    print(f"Total de amostras carregadas: {len(dados)}")
    
    # 2. Split Estratificado 70/30
    dados_treino, dados_teste = split_estratificado(dados, proporcao_treino=0.7, semente=42)
    print(f"Treino: {len(dados_treino)} amostras | Teste: {len(dados_teste)} amostras")
    
    # --- EXPERIMENTO i & ii: Classificador de Distância Mínima (3 classes) ---
    print("\n" + "="*70)
    print("EXPERIMENTO i & ii: Classificador de Distância Mínima (3 classes)")
    print("="*70)
    
    # Treino
    prototipos = treinar(dados_treino, INDICES_ATRIBUTOS)
    print("\nProtótipos (Vetores Médios):")
    for classe, proto in prototipos.items():
        print(f"  {classe:10}: {proto}")
        
    # Teste
    predicoes = []
    gabarito = []
    
    print("\nResultados de Classificação (Amostras de Teste):")
    print(f"{'Classe Real':12} | {'d_setosa':10} | {'d_versicolor':10} | {'d_virginica':10} | {'Predição':12}")
    print("-" * 80)
    
    for amostra in dados_teste:
        scores, vencedor = predizer_todas_classes(amostra['atributos'], prototipos, INDICES_ATRIBUTOS)
        predicoes.append(vencedor)
        gabarito.append(amostra['classe'])
        
        print(f"{amostra['classe']:12} | {scores['setosa']:10.4f} | {scores['versicolor']:10.4f} | {scores['virginica']:10.4f} | {vencedor:12}")
        
    # Avaliação
    acc = acuracia(predicoes, gabarito)
    print(f"\nAcurácia Geral: {acc:.2%}")
    
    mc = matriz_confusao(predicoes, gabarito, CLASSES)
    imprimir_matriz_confusao(mc, CLASSES)
    
    # Visualização Geral
    caminho_scatter = os.path.join(PASTA_OUTPUTS, "iris_dispersao_geral.png")
    plotar_dispersao_todas_classes(dados, INDICES_ATRIBUTOS, prototipos, caminho_salvar=caminho_scatter)
    
    # --- EXPERIMENTO iii: Superfícies de Decisão (Pares de Classes) ---
    print("\n" + "="*70)
    print("EXPERIMENTO iii: Superfícies de Decisão (Pares de Classes)")
    print("="*70)
    
    pares = [
        ('virginica', 'setosa'),
        ('setosa', 'versicolor'),
        ('versicolor', 'virginica')
    ]
    
    for classe_i, classe_j in pares:
        print(f"\nAnalisando Par: {classe_i} vs {classe_j}")
        
        # Filtrar dados para o par específico
        treino_par = filtrar_por_classes(dados_treino, [classe_i, classe_j])
        teste_par = filtrar_por_classes(dados_teste, [classe_i, classe_j])
        
        # Recalcular protótipos para o par (usando apenas amostras das duas classes envolvidas)
        prototipos_par = treinar(treino_par, INDICES_ATRIBUTOS)
        pi = prototipos_par[classe_i]
        pj = prototipos_par[classe_j]
        
        # Classificar teste binário
        predicoes_par = []
        gabarito_par = []
        for amostra in teste_par:
            pred = predizer_binario(amostra['atributos'], pi, pj, classe_i, classe_j, INDICES_ATRIBUTOS)
            predicoes_par.append(pred)
            gabarito_par.append(amostra['classe'])
            
        acc_par = acuracia(predicoes_par, gabarito_par)
        print(f"Acurácia do Par: {acc_par:.2%}")
        
        # Plotar superfície
        dados_c1 = [d for d in dados if d['classe'] == classe_i]
        dados_c2 = [d for d in dados if d['classe'] == classe_j]
        
        nome_arquivo = f"superficie_{classe_i}_{classe_j}.png"
        caminho_plot = os.path.join(PASTA_OUTPUTS, nome_arquivo)
        
        plotar_superficie_decisao(pi, pj, dados_c1, dados_c2, classe_i, classe_j, INDICES_ATRIBUTOS, 
                                  titulo=f"Superfície: {classe_i} vs {classe_j}",
                                  caminho_salvar=caminho_plot)

if __name__ == "__main__":
    executar_experimentos()
