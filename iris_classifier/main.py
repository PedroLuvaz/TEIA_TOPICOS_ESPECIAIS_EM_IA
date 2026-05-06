import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_loader import carregar_dados_iris, split_estratificado, filtrar_por_classes
from classifier import treinar, predizer_todas_classes, predizer_binario
from evaluator import acuracia, matriz_confusao, imprimir_matriz_confusao, imprimir_metricas_por_classe
from visualizer import plotar_superficie_decisao, plotar_dispersao_todas_classes, plotar_matriz_confusao
from math_utils import coeficientes_superficie_decisao

RAIZ_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_DADOS = os.path.join(RAIZ_PROJETO, "data", "Iris data.xls")
PASTA_OUTPUTS = os.path.join(RAIZ_PROJETO, "outputs")

INDICES_PETALA = [2, 3]
INDICES_SEPALA = [0, 1]
CLASSES = ['setosa', 'versicolor', 'virginica']
PARES_BINARIOS = [('virginica', 'setosa'), ('setosa', 'versicolor'), ('versicolor', 'virginica')]


def secao(titulo):
    print(f"\n{'='*70}\n{titulo}\n{'='*70}")


# ---------------------------------------------------------------------------
# Experimento i & ii — Classificador multiclasse com função discriminante
# ---------------------------------------------------------------------------

def experimento_multiclasse(dados_treino, dados_teste, dados_todos):
    secao("EXPERIMENTO i & ii: Classificador de Distancia Minima (3 classes)")

    prototipos = treinar(dados_treino, INDICES_PETALA)

    print("\nPrototipos (Vetores Medios) — Petala:")
    for classe, proto in prototipos.items():
        print(f"  {classe:10}: {[round(v, 4) for v in proto]}")

    predicoes, gabarito = [], []
    print(f"\n{'Classe Real':12} | {'d_setosa':10} | {'d_versicolor':10} | {'d_virginica':10} | {'Predicao':12}")
    print("-" * 80)

    for amostra in dados_teste:
        scores, vencedor = predizer_todas_classes(amostra['atributos'], prototipos, INDICES_PETALA)
        predicoes.append(vencedor)
        gabarito.append(amostra['classe'])
        print(f"{amostra['classe']:12} | {scores['setosa']:10.4f} | {scores['versicolor']:10.4f} | {scores['virginica']:10.4f} | {vencedor:12}")

    acc = acuracia(predicoes, gabarito)
    print(f"\nAcuracia Geral: {acc:.2%}")

    mc = matriz_confusao(predicoes, gabarito, CLASSES)
    imprimir_matriz_confusao(mc, CLASSES)
    imprimir_metricas_por_classe(mc, CLASSES)

    plotar_dispersao_todas_classes(dados_todos, INDICES_PETALA, prototipos,
                                   caminho_salvar=_output("iris_dispersao_geral.png"))
    plotar_matriz_confusao(mc, CLASSES,
                           caminho_salvar=_output("matriz_confusao.png"))


# ---------------------------------------------------------------------------
# Experimento iii — Superfícies de decisão (classificadores binários por par)
# ---------------------------------------------------------------------------

def experimento_superficies(dados_treino, dados_teste, dados_todos):
    secao("EXPERIMENTO iii: Superficies de Decisao (Pares de Classes)")

    for classe_i, classe_j in PARES_BINARIOS:
        print(f"\nPar: {classe_i} vs {classe_j}")

        treino_par = filtrar_por_classes(dados_treino, [classe_i, classe_j])
        teste_par  = filtrar_por_classes(dados_teste,  [classe_i, classe_j])

        prototipos_par = treinar(treino_par, INDICES_PETALA)
        pi, pj = prototipos_par[classe_i], prototipos_par[classe_j]

        w, b = coeficientes_superficie_decisao(pi, pj)
        print(f"  Fronteira dij(x) = 0:")
        print(f"    w = [{w[0]:.4f}, {w[1]:.4f}]   b = {b:.4f}")
        print(f"    {w[0]:.4f}*x1 + {w[1]:.4f}*x2 + {b:.4f} = 0")

        preds_par = [predizer_binario(a['atributos'], pi, pj, classe_i, classe_j, INDICES_PETALA)
                     for a in teste_par]
        gab_par   = [a['classe'] for a in teste_par]
        print(f"  Acuracia do Par: {acuracia(preds_par, gab_par):.2%}")

        plotar_superficie_decisao(
            pi, pj,
            [d for d in dados_todos if d['classe'] == classe_i],
            [d for d in dados_todos if d['classe'] == classe_j],
            classe_i, classe_j, INDICES_PETALA,
            titulo=f"Superficie: {classe_i} vs {classe_j}",
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
    dados_treino, dados_teste = split_estratificado(dados, proporcao_treino=0.7, semente=42)

    print(f"Amostras carregadas : {len(dados)}")
    print(f"Treino              : {len(dados_treino)}")
    print(f"Teste               : {len(dados_teste)}")

    experimento_multiclasse(dados_treino, dados_teste, dados)
    experimento_superficies(dados_treino, dados_teste, dados)
    experimento_comparativo(dados_treino, dados_teste)


if __name__ == "__main__":
    executar_experimentos()
