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

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data.data_loader import carregar_dados_iris, split_estratificado, filtrar_por_classes
from models.classifier import treinar, predizer_todas_classes, predizer_binario
from models.bayes_classifier import treinar_bayes, predizer_todas_classes_bayes, predizer_binario_bayes
from models.mlp_sklearn import treinar_mlp_iris, prever_mlp_iris
from evaluation.mvn_tester import executar_analise_mvn
from evaluation.metricas_avancadas import relatorio_completo, z_kappa, p_valor_z
from evaluation.evaluator import (
    acuracia,
    matriz_confusao,
    imprimir_matriz_confusao,
    imprimir_metricas_por_classe,
)
from visualization.visualizer import (
    plotar_superficie_decisao,
    plotar_dispersao_todas_classes,
    plotar_matriz_confusao,
    plotar_superficie_decisao_bayes,
)
from core.math_utils import coeficientes_superficie_decisao, distancia_euclidiana

RAIZ_PROJETO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMINHO_DADOS = os.path.join(RAIZ_PROJETO, "data", "Iris data.xls")
PASTA_OUTPUTS = os.path.join(RAIZ_PROJETO, "outputs")

INDICES_PETALA = [2, 3]
INDICES_SEPALA = [0, 1]
INDICES_TODAS  = [0, 1, 2, 3]
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
    print("\n--- Matriz de Confusao (Linhas = Predito, Colunas = Real) ---")
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
# Experimento comparativo — Todas as 4 caracteristicas [0,1,2,3]
# ---------------------------------------------------------------------------

def experimento_todas_caracteristicas(dados_treino, dados_teste):
    secao("EXPERIMENTO COMPARATIVO: Todas as 4 Caracteristicas [0,1,2,3]")
    print("Classificador de distancia minima usando sepalas + petalas (espaco 4D).")
    print("Mesmo split 70/30 estratificado (semente 42). Sem grafico de fronteira:")
    print("a superficie de decisao em 4D nao e plotavel em um plano.\n")

    prototipos_4d = treinar(dados_treino, INDICES_TODAS)

    print("Prototipos (Vetores Medios) nas 4 dimensoes:")
    for classe, proto in prototipos_4d.items():
        print(f"  {classe:10}: {[round(v, 4) for v in proto]}")

    preds = [predizer_todas_classes(a['atributos'], prototipos_4d, INDICES_TODAS)[1]
             for a in dados_teste]
    gab   = [a['classe'] for a in dados_teste]
    erros = sum(p != g for p, g in zip(preds, gab))

    acc = acuracia(preds, gab)
    print(f"\n  [Todas] Sepalas + Petalas (4 atributos)    | {acc:.2%}  ({erros} erro(s) em {len(dados_teste)} amostras)")

    print("\n--- Matriz de Confusao (Linhas = Predito, Colunas = Real) ---")
    matriz_4d = matriz_confusao(preds, gab, CLASSES)
    imprimir_matriz_confusao(matriz_4d, CLASSES)

    print("\n  Conclusao: usar as 4 caracteristicas mantem desempenho alto, mas a")
    print("  sobreposicao das sepalas (versicolor x virginica) pode introduzir")
    print("  pequeno ruido em relacao ao uso exclusivo das petalas.")


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


# ---------------------------------------------------------------------------
# Experimentos com Bayes Ótimo e Naive Bayes (Nova Feature)
# ---------------------------------------------------------------------------

def experimento_bayes(dados_treino, dados_teste, dados_todos):
    secao("EXPERIMENTOS DE BAYES OTIMO E NAIVE BAYES")
    
    # 1. Verificação de Normalidade Multivariada com R
    print("1. VERIFICACAO DE NORMALIDADE MULTIVARIADA (R - Pacote MVN)\n")
    relatorio_r, dados_mvn, r_ok = executar_analise_mvn(CAMINHO_DADOS, PASTA_OUTPUTS)
    print(relatorio_r)
    
    # 2. Treinamento dos Classificadores Bayes Ótimo (QDA) e Naive Bayes (4 features)
    print("2. CLASSIFICACAO MULTICLASSE (3 Classes, 4 atributos)")
    
    model_bayes = treinar_bayes(dados_treino, INDICES_TODAS, naive=False)
    model_naive = treinar_bayes(dados_treino, INDICES_TODAS, naive=True)
    
    # Avaliar Bayes Ótimo
    preds_bayes = []
    gab = [d['classe'] for d in dados_teste]
    for d in dados_teste:
        _, pred = predizer_todas_classes_bayes(d['atributos'], model_bayes, INDICES_TODAS)
        preds_bayes.append(pred)
        
    # Avaliar Naive Bayes
    preds_naive = []
    for d in dados_teste:
        _, pred = predizer_todas_classes_bayes(d['atributos'], model_naive, INDICES_TODAS)
        preds_naive.append(pred)
        
    # Calcular métricas completas usando metricas_avancadas
    rel_bayes = relatorio_completo(preds_bayes, gab, CLASSES, "Bayes Otimo")
    rel_naive = relatorio_completo(preds_naive, gab, CLASSES, "Naive Bayes")
    
    # Exibir relatório de Bayes Ótimo
    print("\n>>> CLASSIFICADOR BAYES OTIMO (QDA) <<<")
    print(f"  Acuracia Global: {rel_bayes['acerto_global']:.2%}")
    print(f"  Indice Kappa:    {rel_bayes['kappa']:.4f} (Var: {rel_bayes['variancia_kappa']:.6f})")
    print("\nMatriz de Confusao:")
    imprimir_matriz_confusao(rel_bayes['matriz'], CLASSES)
    print("\nMetricas por Classe:")
    for c in CLASSES:
        mc = rel_bayes['por_classe'][c]
        print(f"  Classe {c:10}: Produtor (Sens.): {mc['acuracia_produtor']:.4f} | Usuario (Prec.): {mc['acuracia_usuario']:.4f} | F1: {mc['f1']:.4f}")
        
    # Exibir relatório de Naive Bayes
    print("\n>>> CLASSIFICADOR NAIVE BAYES <<<")
    print(f"  Acuracia Global: {rel_naive['acerto_global']:.2%}")
    print(f"  Indice Kappa:    {rel_naive['kappa']:.4f} (Var: {rel_naive['variancia_kappa']:.6f})")
    print("\nMatriz de Confusao:")
    imprimir_matriz_confusao(rel_naive['matriz'], CLASSES)
    print("\nMetricas por Classe:")
    for c in CLASSES:
        mc = rel_naive['por_classe'][c]
        print(f"  Classe {c:10}: Produtor (Sens.): {mc['acuracia_produtor']:.4f} | Usuario (Prec.): {mc['acuracia_usuario']:.4f} | F1: {mc['f1']:.4f}")
        
    # Teste de significância de Kappa entre os dois
    z_stat = z_kappa(rel_bayes['kappa'], rel_bayes['variancia_kappa'], rel_naive['kappa'], rel_naive['variancia_kappa'])
    p_val = p_valor_z(z_stat)
    
    print("\n>>> COMPARACAO E SIGNIFICANCIA DE KAPPA (Item e) <<<")
    print(f"  Estatistica Z: {z_stat:.4f}")
    print(f"  p-valor:       {p_val:.6f}")
    if p_val < 0.05:
        print("  Resultado: Existe diferenca estatisticamente significativa entre as acuracias dos classificadores (ao nivel de 5%).")
        if rel_bayes['kappa'] > rel_naive['kappa']:
            print("  O Classificador de Bayes Otimo e estatisticamente superior.")
        else:
            print("  O Classificador Naive Bayes e estatisticamente superior.")
    else:
        print("  Resultado: Nao existe diferenca estatisticamente significativa entre as acuracias dos classificadores (ao nivel de 5%).")
        if rel_bayes['acerto_global'] > rel_naive['acerto_global']:
            print(f"  Numericamente, o Bayes Otimo teve maior acuracia ({rel_bayes['acerto_global']:.2%} vs {rel_naive['acerto_global']:.2%}), mas a diferenca nao e significativa.")
        elif rel_bayes['acerto_global'] < rel_naive['acerto_global']:
            print(f"  Numericamente, o Naive Bayes teve maior acuracia ({rel_naive['acerto_global']:.2%} vs {rel_naive['acerto_global']:.2%}), mas a diferenca nao e significativa.")
        else:
            print("  Os dois classificadores apresentaram acuracia identica no conjunto de teste.")
            
    # 3. Superfícies de Decisão (Pares de classes, usando Pétalas [2,3])
    print("\n3. SUPERFICIES DE DECISAO E CLASSIFICACAO BINARIA (Pares de Classes, Petalas [2,3])")
    
    for classe_i, classe_j in PARES_BINARIOS:
        print(f"\n------------------------------------------")
        print(f"Par: {classe_i} vs {classe_j}")
        print(f"------------------------------------------")
        
        # Filtrar dados para o par
        treino_par = filtrar_por_classes(dados_treino, [classe_i, classe_j])
        teste_par  = filtrar_por_classes(dados_teste,  [classe_i, classe_j])
        todos_par  = filtrar_por_classes(dados_todos,  [classe_i, classe_j])
        
        # Treinar modelos locais 2D nas pétalas
        model_bayes_2d = treinar_bayes(treino_par, INDICES_PETALA, naive=False)
        model_naive_2d = treinar_bayes(treino_par, INDICES_PETALA, naive=True)
        
        # Predições
        preds_bayes_par = [predizer_binario_bayes(a['atributos'], model_bayes_2d, classe_i, classe_j, INDICES_PETALA) for a in teste_par]
        preds_naive_par = [predizer_binario_bayes(a['atributos'], model_naive_2d, classe_i, classe_j, INDICES_PETALA) for a in teste_par]
        gab_par = [a['classe'] for a in teste_par]
        
        # Relatórios de Métricas
        classes_par = [classe_i, classe_j]
        rel_b = relatorio_completo(preds_bayes_par, gab_par, classes_par, "Bayes Otimo 2D")
        rel_n = relatorio_completo(preds_naive_par, gab_par, classes_par, "Naive Bayes 2D")
        
        print("\n--> Bayes Otimo (QDA):")
        print(f"  Acuracia: {rel_b['acerto_global']:.2%}")
        print(f"  Kappa:    {rel_b['kappa']:.4f}")
        imprimir_matriz_confusao(rel_b['matriz'], classes_par)
        
        print("\n--> Naive Bayes:")
        print(f"  Acuracia: {rel_n['acerto_global']:.2%}")
        print(f"  Kappa:    {rel_n['kappa']:.4f}")
        imprimir_matriz_confusao(rel_n['matriz'], classes_par)
        
        # Plotar superfícies de decisão
        plotar_superficie_decisao_bayes(
            model_bayes_2d,
            [d for d in todos_par if d['classe'] == classe_i],
            [d for d in todos_par if d['classe'] == classe_j],
            classe_i, classe_j, INDICES_PETALA,
            dados_treino=treino_par,
            dados_teste=teste_par,
            nomes_atributos=NOMES_ATRIBUTOS,
            titulo=f"Bayes Otimo: {classe_i} vs {classe_j}",
            caminho_salvar=_output(f"bayes_otimo_superficie_{classe_i}_{classe_j}.png"),
        )
        
        plotar_superficie_decisao_bayes(
            model_naive_2d,
            [d for d in todos_par if d['classe'] == classe_i],
            [d for d in todos_par if d['classe'] == classe_j],
            classe_i, classe_j, INDICES_PETALA,
            dados_treino=treino_par,
            dados_teste=teste_par,
            nomes_atributos=NOMES_ATRIBUTOS,
            titulo=f"Naive Bayes: {classe_i} vs {classe_j}",
            caminho_salvar=_output(f"naive_bayes_superficie_{classe_i}_{classe_j}.png"),
        )


# ---------------------------------------------------------------------------
# Lab 5 — Item (ii): Rede Feedforward (MLP) vs Bayes Otimo vs Naive Bayes
# Unica parte do projeto que usa biblioteca de ML (scikit-learn), conforme
# explicitamente permitido pelo enunciado apenas para este experimento.
# ---------------------------------------------------------------------------

def experimento_mlp_iris(dados_treino, dados_teste):
    secao("LAB 5 - ITEM (ii): FEEDFORWARD (MLP) vs BAYES OTIMO vs NAIVE BAYES")
    print("Classificacao das 3 especies do Iris (4 atributos), split 70/30 estratificado.")
    print("MLP treinada com scikit-learn (unico experimento do projeto com lib de ML).\n")

    gab = [d['classe'] for d in dados_teste]

    # --- Treinar os 3 modelos ---
    modelo_mlp = treinar_mlp_iris(dados_treino, INDICES_TODAS, semente=42)
    model_bayes = treinar_bayes(dados_treino, INDICES_TODAS, naive=False)
    model_naive = treinar_bayes(dados_treino, INDICES_TODAS, naive=True)

    # --- Predicoes no conjunto de teste ---
    preds_mlp = prever_mlp_iris(modelo_mlp, dados_teste, INDICES_TODAS)
    preds_bayes = [predizer_todas_classes_bayes(d['atributos'], model_bayes, INDICES_TODAS)[1]
                   for d in dados_teste]
    preds_naive = [predizer_todas_classes_bayes(d['atributos'], model_naive, INDICES_TODAS)[1]
                   for d in dados_teste]

    # --- Relatorio completo (acerto global, kappa, tau, precisao/recall/F1/F2/MCC por classe) ---
    rel_mlp = relatorio_completo(preds_mlp, gab, CLASSES, "Feedforward (MLP)")
    rel_bayes = relatorio_completo(preds_bayes, gab, CLASSES, "Bayes Otimo")
    rel_naive = relatorio_completo(preds_naive, gab, CLASSES, "Naive Bayes")

    for rel in (rel_mlp, rel_bayes, rel_naive):
        print(f"\n>>> {rel['nome'].upper()} <<<")
        print(f"  Acerto Global: {rel['acerto_global']:.2%}")
        print(f"  Kappa:         {rel['kappa']:.4f} (Var: {rel['variancia_kappa']:.6f})")
        print(f"  Tau:           {rel['tau']:.4f}")
        print("\n  Matriz de Confusao:")
        imprimir_matriz_confusao(rel['matriz'], CLASSES)
        print("\n  Metricas por Classe:")
        for c in CLASSES:
            mc = rel['por_classe'][c]
            print(f"    {c:10}: Precisao={mc['precisao']:.4f}  Recall={mc['sensibilidade']:.4f}  "
                  f"F1={mc['f1']:.4f}  F2={mc['f2']:.4f}  MCC={mc['mcc']:.4f}")

    # --- Testes Z de significancia de Kappa (todos os pares) ---
    print("\n>>> TESTES Z DE SIGNIFICANCIA DE KAPPA (todos os pares) <<<")
    pares = [
        ("Feedforward (MLP)", rel_mlp, "Bayes Otimo", rel_bayes),
        ("Feedforward (MLP)", rel_mlp, "Naive Bayes", rel_naive),
        ("Bayes Otimo", rel_bayes, "Naive Bayes", rel_naive),
    ]
    for nome_a, rel_a, nome_b, rel_b in pares:
        z = z_kappa(rel_a['kappa'], rel_a['variancia_kappa'], rel_b['kappa'], rel_b['variancia_kappa'])
        p = p_valor_z(z)
        veredito = "diferenca estatisticamente significativa" if p < 0.05 else "sem diferenca significativa"
        print(f"  {nome_a} x {nome_b}: Z={z:.4f}  p={p:.6f}  ({veredito} a 5%)")


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

    prototipos = experimento_multiclasse(dados_treino, dados_teste, dados)
    experimento_superficies(dados_treino, dados_teste, dados)
    experimento_comparativo(dados_treino, dados_teste)
    experimento_todas_caracteristicas(dados_treino, dados_teste)
    
    # Executar os experimentos de Bayes & Naive Bayes (Nova Feature)
    experimento_bayes(dados_treino, dados_teste, dados)

    # Lab 5 — Feedforward (MLP) vs Bayes Otimo vs Naive Bayes (Nova Feature)
    experimento_mlp_iris(dados_treino, dados_teste)

    modo_interativo(prototipos)


if __name__ == "__main__":
    executar_experimentos()
