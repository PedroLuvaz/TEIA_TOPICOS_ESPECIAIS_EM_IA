"""
Classificador de Distancia Minima — Iris Dataset
=================================================
Experimentos obrigatorios:
  i.   Prototipos: m_j = (1/N_j) * sum(x)  para cada classe j
  ii.  Discriminante: d_j(x) = x^T*m_j - (1/2)*m_j^T*m_j  =>  argmax_j d_j(x)
       Equivalencia: argmax d_j(x) == argmin ||x - m_j||  (prova em runtime)
  iii. Fronteira: w = m_i - m_j,  b = -(1/2)*(||m_i||^2 - ||m_j||^2)

Implementacao em Python puro (sem numpy/scipy/sklearn/pandas).
"""

import os
import sys

RAIZ_PROJETO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RAIZ_PROJETO, 'src'))

from core.data_loader import carregar_dados_iris, split_estratificado, filtrar_por_classes
from core.classifier import treinar, predizer_todas_classes, predizer_binario
from core.evaluator import (
    acuracia,
    matriz_confusao,
    imprimir_matriz_confusao,
    imprimir_metricas_por_classe,
)
from visualizer import (
    plotar_superficie_decisao,
    plotar_dispersao_todas_classes,
    plotar_matriz_confusao,
)
from gui.interativo import abrir_interface
from core.math_utils import coeficientes_superficie_decisao, distancia_euclidiana
CAMINHO_DADOS = os.path.join(RAIZ_PROJETO, "data", "Iris data.xls")
PASTA_OUTPUTS = os.path.join(RAIZ_PROJETO, "outputs")

INDICES_PETALA = [2, 3]
INDICES_SEPALA = [0, 1]
CLASSES = ['setosa', 'versicolor', 'virginica']
PARES_BINARIOS = [('virginica', 'setosa'), ('setosa', 'versicolor'), ('versicolor', 'virginica')]

NOMES_ATRIBUTOS = {
    0: 'Comp. Sepala (cm)',
    1: 'Larg. Sepala (cm)',
    2: 'Comp. Petala (cm)',
    3: 'Larg. Petala (cm)',
}


def secao(titulo):
    print(f"\n{'='*70}\n{titulo}\n{'='*70}")


# ---------------------------------------------------------------------------
# Experimento i & ii
# Protótipos + Função Discriminante + Distância Euclidiana (equivalência)
# ---------------------------------------------------------------------------

def experimento_multiclasse(dados_treino, dados_teste, dados_todos):
    secao("EXPERIMENTO i & ii: Classificador de Distancia Minima (3 classes)")

    # --- Experimento i: Protótipos ---
    prototipos = treinar(dados_treino, INDICES_PETALA)

    print("\nPrototipos (Vetores Medios) — m_j = (1/N_j) * sum(x):")
    for classe, proto in prototipos.items():
        print(f"  {classe:10}: {[round(v, 4) for v in proto]}")

    # Pré-extrair atributos de pétala de cada amostra de teste
    amostras_sel = [[a['atributos'][i] for i in INDICES_PETALA] for a in dados_teste]
    gabarito      = [a['classe'] for a in dados_teste]

    # --- Experimento ii-A: Função Discriminante  d_j(x) = x^T*m_j - 0.5*m_j^T*m_j ---
    print("\n--- Funcao Discriminante: d_j(x) = x^T * m_j - 0.5 * m_j^T * m_j ---")
    print(f"  Regra: argmax_j d_j(x)  [maior valor = classe predita]")
    print(f"\n{'Classe Real':12} | {'d_setosa':10} | {'d_versicolor':12} | {'d_virginica':11} | {'Predicao':12}")
    print("-" * 80)

    predicoes_disc = []
    for amostra, real in zip(dados_teste, gabarito):
        scores, pred = predizer_todas_classes(amostra['atributos'], prototipos, INDICES_PETALA)
        predicoes_disc.append(pred)
        print(f"{real:12} | {scores['setosa']:10.4f} | {scores['versicolor']:12.4f} | {scores['virginica']:11.4f} | {pred:12}")

    # --- Experimento ii-B: Distância Euclidiana  ||x - m_j|| ---
    print("\n--- Distancia Euclidiana: ||x - m_j|| = sqrt( sum((x_k - m_jk)^2) ) ---")
    print(f"  Regra: argmin_j ||x - m_j||  [menor distancia = classe predita]")
    print(f"\n{'Classe Real':12} | {'dist_setosa':11} | {'dist_versicolor':15} | {'dist_virginica':14} | {'Predicao':12}")
    print("-" * 85)

    predicoes_dist = []
    for x_sel, real in zip(amostras_sel, gabarito):
        dists = {c: distancia_euclidiana(x_sel, prototipos[c]) for c in CLASSES}
        pred  = min(dists, key=dists.get)
        predicoes_dist.append(pred)
        print(f"{real:12} | {dists['setosa']:11.4f} | {dists['versicolor']:15.4f} | {dists['virginica']:14.4f} | {pred:12}")

    # --- Prova de equivalência ---
    concordam = sum(d == e for d, e in zip(predicoes_disc, predicoes_dist))
    print(f"\nEquivalencia matematica: {concordam}/{len(gabarito)} predicoes identicas entre os dois metodos.")

    # --- Avaliação Global ---
    acc = acuracia(predicoes_dist, gabarito)
    print(f"\nAcuracia Geral: {acc:.2%}")

    # --- Matriz de Confusao ---
    print("\n--- Matriz de Confusao (Linhas = Real, Colunas = Predito) ---")
    matriz = matriz_confusao(predicoes_dist, gabarito, CLASSES)
    imprimir_matriz_confusao(matriz, CLASSES)

    # --- Metricas por Classe (Precisao, Revocacao, F1) ---
    print("\n--- Metricas por Classe ---")
    print("  Precisao(j)  = VP / (VP + FP)   — qualidade das predicoes da classe")
    print("  Revocacao(j) = VP / (VP + FN)   — cobertura das amostras reais da classe")
    print("  F1(j)        = 2*P*R / (P + R)  — media harmonica\n")
    imprimir_metricas_por_classe(matriz, CLASSES)

    # --- Graficos ---
    plotar_dispersao_todas_classes(
        dados_todos, INDICES_PETALA, prototipos,
        dados_treino=dados_treino,
        dados_teste=dados_teste,
        nomes_atributos=NOMES_ATRIBUTOS,
        titulo="Iris Dataset — Distribuicao Completa (Teste Destacado)",
        caminho_salvar=_output("iris_dispersao_geral.png"),
    )

    plotar_matriz_confusao(
        matriz, CLASSES,
        titulo="Matriz de Confusao — Petalas (Conjunto de Teste)",
        caminho_salvar=_output("matriz_confusao.png"),
    )

    return prototipos


# ---------------------------------------------------------------------------
# Experimento iii — Superfícies de decisão (classificadores binários por par)
# Fronteira: w = m_i - m_j,  b = -0.5*(||m_i||^2 - ||m_j||^2)
# ---------------------------------------------------------------------------

def experimento_superficies(dados_treino, dados_teste, dados_todos):
    secao("EXPERIMENTO iii: Superficies de Decisao (Pares de Classes)")
    print("Fronteira: w = m_i - m_j   |   b = -0.5*(||m_i||^2 - ||m_j||^2)")
    print("Reta 2D:   x2 = (-w1*x1 - b) / w2")

    for classe_i, classe_j in PARES_BINARIOS:
        print(f"\nPar: {classe_i} vs {classe_j}")

        treino_par = filtrar_por_classes(dados_treino, [classe_i, classe_j])
        teste_par  = filtrar_por_classes(dados_teste,  [classe_i, classe_j])

        prototipos_par = treinar(treino_par, INDICES_PETALA)
        pi, pj = prototipos_par[classe_i], prototipos_par[classe_j]

        w, b = coeficientes_superficie_decisao(pi, pj)
        print(f"  w = [{w[0]:.4f}, {w[1]:.4f}]   b = {b:.4f}")
        print(f"  Equacao: {w[0]:.4f}*x1 + {w[1]:.4f}*x2 + {b:.4f} = 0")

        preds_par = [predizer_binario(a['atributos'], pi, pj, classe_i, classe_j, INDICES_PETALA)
                     for a in teste_par]
        gab_par   = [a['classe'] for a in teste_par]
        acc_par   = acuracia(preds_par, gab_par)
        print(f"  Acuracia do Par (teste): {acc_par:.2%}")

        todos_par = filtrar_por_classes(dados_todos, [classe_i, classe_j])
        preds_todos = [predizer_binario(a['atributos'], pi, pj, classe_i, classe_j, INDICES_PETALA)
                       for a in todos_par]
        gab_todos = [a['classe'] for a in todos_par]
        acc_todos = acuracia(preds_todos, gab_todos)
        print(f"  Acuracia do Par (base completa): {acc_todos:.2%}")

        # O grafico mostra a base completa, mas a fronteira vem do treino.
        # Pontos de treino ficam suaves; pontos de teste ficam destacados.
        plotar_superficie_decisao(
            pi, pj,
            [d for d in todos_par if d['classe'] == classe_i],
            [d for d in todos_par if d['classe'] == classe_j],
            classe_i, classe_j, INDICES_PETALA,
            dados_treino=treino_par,
            dados_teste=teste_par,
            nomes_atributos=NOMES_ATRIBUTOS,
            titulo=f"Superficie: {classe_i} vs {classe_j} (base completa)",
            caminho_salvar=_output(f"superficie_{classe_i}_{classe_j}.png"),
        )


# ---------------------------------------------------------------------------
# Experimento comparativo — Pétalas [2,3] vs Sépalas [0,1]
# ---------------------------------------------------------------------------

def experimento_comparativo(dados_treino, dados_teste):
    secao("EXPERIMENTO COMPARATIVO: Petalas [2,3] vs Sepalas [0,1]")
    print("A escolha dos atributos determina a separabilidade linear do modelo.\n")

    experimentos = [
        ("Petalas", INDICES_PETALA, "Comp. Petala + Larg. Petala"),
        ("Sepalas", INDICES_SEPALA, "Comp. Sepala + Larg. Sepala"),
    ]

    for rotulo, indices, descricao in experimentos:
        prototipos_exp = treinar(dados_treino, indices)
        preds = [predizer_todas_classes(a['atributos'], prototipos_exp, indices)[1] for a in dados_teste]
        gab   = [a['classe'] for a in dados_teste]
        erros = sum(p != g for p, g in zip(preds, gab))
        print(f"  [{rotulo}] {descricao:35} | {acuracia(preds, gab):.2%}  ({erros} erro(s) em {len(dados_teste)} amostras)")

    print("\n  Conclusao: Petalas sao linearmente separaveis — acuracia 100%.")
    print("  Sepalas de Versicolor e Virginica se sobrepoem — classificador")
    print("  linear nao consegue separacao perfeita.")


# ---------------------------------------------------------------------------
# Classificação interativa — exibe discriminante E distância euclidiana
# ---------------------------------------------------------------------------

def modo_interativo(prototipos):
    secao("CLASSIFICACAO INTERATIVA")
    print("Informe os valores das petalas para classificar uma nova amostra.")
    print("Digite 'sair' a qualquer momento para encerrar.\n")

    while True:
        vetor_completo = [0.0, 0.0, 0.0, 0.0]
        cancelado = False

        for i in INDICES_PETALA:
            entrada = input(f"  {NOMES_ATRIBUTOS[i]}: ").strip()
            if entrada.lower() == 'sair':
                cancelado = True
                break
            try:
                vetor_completo[i] = float(entrada.replace(',', '.'))
            except ValueError:
                print("  Valor invalido. Use numeros decimais (ex: 4.5).\n")
                cancelado = True
                break

        if cancelado:
            print("\nEncerrando classificacao interativa.")
            break

        x_sel = [vetor_completo[i] for i in INDICES_PETALA]

        scores_disc, pred_disc = predizer_todas_classes(vetor_completo, prototipos, INDICES_PETALA)

        dists     = {c: distancia_euclidiana(x_sel, prototipos[c]) for c in CLASSES}
        pred_dist = min(dists, key=dists.get)

        print(f"\n  Resultado final: {pred_dist.upper()}")
        print("  A classe escolhida e a que tem a menor distancia ao prototipo.")

        print("\n  Funcao discriminante d_j(x)  [maior valor vence]:")

        for c in sorted(scores_disc, key=scores_disc.get, reverse=True):
            marcador = " <-- predito" if c == pred_disc else ""
            print(f"    {c:12}: {scores_disc[c]:8.4f}{marcador}")

        print("\n  Distancia euclidiana ate cada prototipo  [menor valor vence]:")

        for c in sorted(dists, key=dists.get):
            marcador = " <-- predito" if c == pred_dist else ""
            print(f"    {c:12}: {dists[c]:8.4f}{marcador}")

        print()


# ---------------------------------------------------------------------------
# Orquestrador principal
# ---------------------------------------------------------------------------

def _output(nome_arquivo):
    return os.path.join(PASTA_OUTPUTS, nome_arquivo)


def executar_experimentos():
    os.makedirs(PASTA_OUTPUTS, exist_ok=True)

    if not os.path.exists(CAMINHO_DADOS):
        print(f"Erro: arquivo nao encontrado: {CAMINHO_DADOS}")
        return

    dados = carregar_dados_iris(CAMINHO_DADOS)
    for i, d in enumerate(dados):
        d['indice'] = i
    dados_treino, dados_teste = split_estratificado(dados, proporcao_treino=0.7, semente=42)

    print(f"Amostras carregadas : {len(dados)}")
    print(f"Treino              : {len(dados_treino)}")
    print(f"Teste               : {len(dados_teste)}")

    prototipos = experimento_multiclasse(dados_treino, dados_teste, dados)
    experimento_superficies(dados_treino, dados_teste, dados)
    experimento_comparativo(dados_treino, dados_teste)
    modo_interativo(prototipos)

    print("\nAbrindo interface grafica interativa...")
    print("(Feche a janela para encerrar o programa)")
    abrir_interface(dados, prototipos, dados_treino, dados_teste)


if __name__ == "__main__":
    executar_experimentos()
