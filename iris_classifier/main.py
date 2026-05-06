import os
import sys

# Adiciona o diretório pai ao path para permitir imports dos módulos locais
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import carregar_dados_iris, split_estratificado, filtrar_por_classes
from classifier import treinar, predizer_todas_classes, predizer_binario
from evaluator import (acuracia, matriz_confusao, imprimir_matriz_confusao,
                       imprimir_metricas_por_classe)
from visualizer import (plotar_superficie_decisao, plotar_dispersao_todas_classes,
                        plotar_matriz_confusao)
from math_utils import coeficientes_superficie_decisao

# Configurações de Caminhos
RAIZ_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_DADOS = os.path.join(RAIZ_PROJETO, "data", "Iris data.xls")
PASTA_OUTPUTS = os.path.join(RAIZ_PROJETO, "outputs")

# Configurações do Experimento
INDICES_ATRIBUTOS = [2, 3]  # Comprimento da Pétala, Largura da Pétala
CLASSES = ['setosa', 'versicolor', 'virginica']

NOMES_ATRIBUTOS = {
    0: 'Comp. Sépala',
    1: 'Larg. Sépala',
    2: 'Comp. Pétala',
    3: 'Larg. Pétala',
}

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
    print("\nProtótipos (Vetores Médios) — atributos utilizados: Pétala")
    for classe, proto in prototipos.items():
        print(f"  {classe:10}: {[round(v, 4) for v in proto]}")

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
    imprimir_metricas_por_classe(mc, CLASSES)

    # Visualizações
    caminho_scatter = os.path.join(PASTA_OUTPUTS, "iris_dispersao_geral.png")
    plotar_dispersao_todas_classes(dados, INDICES_ATRIBUTOS, prototipos, caminho_salvar=caminho_scatter)

    caminho_confusao = os.path.join(PASTA_OUTPUTS, "matriz_confusao.png")
    plotar_matriz_confusao(mc, CLASSES, caminho_salvar=caminho_confusao)

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

        treino_par = filtrar_por_classes(dados_treino, [classe_i, classe_j])
        teste_par = filtrar_por_classes(dados_teste, [classe_i, classe_j])

        prototipos_par = treinar(treino_par, INDICES_ATRIBUTOS)
        pi = prototipos_par[classe_i]
        pj = prototipos_par[classe_j]

        # Imprimir equação numérica da fronteira de decisão
        w, b = coeficientes_superficie_decisao(pi, pj)
        print(f"  Equação da Fronteira (dij(x) = 0):")
        print(f"    w = [{w[0]:.4f}, {w[1]:.4f}]   b = {b:.4f}")
        print(f"    {w[0]:.4f}*x1 + {w[1]:.4f}*x2 + {b:.4f} = 0")

        predicoes_par = []
        gabarito_par = []
        for amostra in teste_par:
            pred = predizer_binario(amostra['atributos'], pi, pj, classe_i, classe_j, INDICES_ATRIBUTOS)
            predicoes_par.append(pred)
            gabarito_par.append(amostra['classe'])

        acc_par = acuracia(predicoes_par, gabarito_par)
        print(f"  Acurácia do Par: {acc_par:.2%}")

        dados_c1 = [d for d in dados if d['classe'] == classe_i]
        dados_c2 = [d for d in dados if d['classe'] == classe_j]

        nome_arquivo = f"superficie_{classe_i}_{classe_j}.png"
        caminho_plot = os.path.join(PASTA_OUTPUTS, nome_arquivo)

        plotar_superficie_decisao(pi, pj, dados_c1, dados_c2, classe_i, classe_j, INDICES_ATRIBUTOS,
                                  titulo=f"Superfície: {classe_i} vs {classe_j}",
                                  caminho_salvar=caminho_plot)

    # --- EXPERIMENTO COMPARATIVO: Sépalas vs Pétalas ---
    print("\n" + "="*70)
    print("EXPERIMENTO COMPARATIVO: Sépalas [0,1] vs Pétalas [2,3]")
    print("="*70)
    print("Objetivo: mostrar que a escolha dos atributos impacta diretamente")
    print("a separabilidade linear e, consequentemente, a acurácia do modelo.")
    print()

    for indices, rotulo in [([2, 3], "Pétalas"), ([0, 1], "Sépalas")]:
        nomes = f"{NOMES_ATRIBUTOS[indices[0]]} + {NOMES_ATRIBUTOS[indices[1]]}"
        prototipos_exp = treinar(dados_treino, indices)
        preds_exp = []
        gab_exp = []
        for amostra in dados_teste:
            _, vencedor = predizer_todas_classes(amostra['atributos'], prototipos_exp, indices)
            preds_exp.append(vencedor)
            gab_exp.append(amostra['classe'])

        acc_exp = acuracia(preds_exp, gab_exp)
        mc_exp = matriz_confusao(preds_exp, gab_exp, CLASSES)
        erros = sum(1 for p, g in zip(preds_exp, gab_exp) if p != g)
        print(f"  [{rotulo}] {nomes:35} | Acuracia: {acc_exp:.2%}  ({erros} erros em {len(dados_teste)} amostras)")

    print()
    print("  Conclusão: Pétalas são linearmente separáveis para as 3 classes.")
    print("  Sépalas apresentam sobreposição entre Versicolor e Virginica,")
    print("  tornando impossível separação perfeita com um classificador linear.")

if __name__ == "__main__":
    executar_experimentos()
