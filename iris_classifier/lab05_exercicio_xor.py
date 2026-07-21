"""
Lab 5 — Exercicio B (slides 42-43): Problema XOR com MLP — 1 Epoca
====================================================================

Enunciado do slide: "Resolva o problema XOR utilizando uma MLP de acordo
com a arquitetura da rede fig 12.28(b). Exercite com uma epoca apenas.
Implemente a arquitetura de rede acima."

A Figura 12.28(b) mostra a arquitetura MINIMA que resolve o XOR: 2 entradas
-> 2 neuronios ocultos (com bias) -> 1 neuronio de saida (com bias), com
pesos genericos rotulados w1..w9 no slide (sem valores numericos dados —
a figura ilustra apenas a TOPOLOGIA da solucao). Os pesos iniciais abaixo
foram escolhidos para esta demonstracao (valores pequenos e simetricos,
tipicos de uma inicializacao aleatoria pequena):

    Pesos entrada->oculta:
        h1: w = [+0.50, +0.50]   bias = -0.20
        h2: w = [-0.50, -0.50]   bias = +0.30
    Pesos oculta->saida:
        saida: w = [+0.60, -0.60]   bias = -0.10

    Taxa de aprendizagem: eta = 0.5

Tabela-verdade do XOR (4 padroes de treino):
    (0,0) -> 0     (0,1) -> 1     (1,0) -> 1     (1,1) -> 0

"1 epoca" = os 4 padroes sao apresentados UMA VEZ, em sequencia, com
atualizacao dos pesos apos CADA padrao (modo online/estocastico — mesma
convencao usada em perceptron.py e delta_rule.py deste projeto). Como o
XOR nao e linearmente separavel, uma unica epoca NAO e suficiente para a
rede convergir — o objetivo do exercicio e observar a direcao do ajuste
dos pesos e a leve reducao do erro, nao a convergencia completa (que
exigiria varias centenas de epocas, como classicamente demonstrado na
literatura de backpropagation).
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mlp_backprop import RedeFeedforward

PADROES = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
]
TAXA_APRENDIZADO = 0.5

PESOS_OCULTA = [
    [0.50, 0.50],     # h1
    [-0.50, -0.50],   # h2
]
BIAS_OCULTA = [-0.20, 0.30]

PESOS_SAIDA = [
    [0.60, -0.60],    # saida
]
BIAS_SAIDA = [-0.10]


def main():
    rede = RedeFeedforward(
        n_entradas=2, n_ocultos=2, n_saidas=1,
        pesos_oculta=[row[:] for row in PESOS_OCULTA], bias_oculta=BIAS_OCULTA[:],
        pesos_saida=[row[:] for row in PESOS_SAIDA], bias_saida=BIAS_SAIDA[:],
    )

    print("=" * 74)
    print("LAB 5 - EXERCICIO B: PROBLEMA XOR COM MLP (1 EPOCA)")
    print("=" * 74)
    print(f"\nArquitetura: 2 entradas -> 2 ocultos -> 1 saida  (Fig. 12.28b)")
    print(f"Taxa de aprendizagem: eta={TAXA_APRENDIZADO}")
    print(f"Pesos iniciais: oculta={PESOS_OCULTA}  bias_oculta={BIAS_OCULTA}")
    print(f"                saida={PESOS_SAIDA}   bias_saida={BIAS_SAIDA}\n")

    print("Previsao ANTES do treino (pesos iniciais):")
    for x, t in PADROES:
        out = rede.prever(x)[0]
        print(f"  x={x}  alvo={t[0]:.0f}  out={out:.4f}")

    print("\n--- PROCESSANDO 1 EPOCA (4 padroes, modo online) ---")
    erro_epoca = 0.0
    for i, (x, t) in enumerate(PADROES):
        r = rede.passo_treinamento(x, t, TAXA_APRENDIZADO)
        erro_epoca += r['erro_total']
        print(f"\nPadrao {i + 1}: x={x}  alvo={t[0]:.0f}")
        print(f"  out_h1={r['saida_oculta'][0]:.4f}  out_h2={r['saida_oculta'][1]:.4f}  "
              f"out={r['saida_rede'][0]:.4f}  erro={r['erro_total']:.5f}")
        print(f"  delta_saida={r['delta_saida'][0]:.6f}  "
              f"delta_h1={r['delta_oculta'][0]:.6f}  delta_h2={r['delta_oculta'][1]:.6f}")
        print(f"  pesos_oculta -> {[[round(v, 4) for v in row] for row in r['w_oculta_depois']]}  "
              f"bias_oculta -> {[round(v, 4) for v in r['b_oculta_depois']]}")
        print(f"  pesos_saida  -> {[round(v, 4) for v in r['w_saida_depois'][0]]}  "
              f"bias_saida -> {r['b_saida_depois'][0]:.4f}")

    print(f"\nErro medio da epoca (media dos 4 erros, calculados antes de cada "
          f"atualizacao): {erro_epoca / 4:.5f}")

    print("\n--- PREVISAO APOS 1 EPOCA (pesos atualizados) ---")
    for x, t in PADROES:
        out = rede.prever(x)[0]
        pred = 1 if out >= 0.5 else 0
        acertou = "OK" if pred == int(t[0]) else "ainda errado"
        print(f"  x={x}  alvo={t[0]:.0f}  out={out:.4f}  classe={pred}  ({acertou})")

    print("\nConclusao: apos apenas 1 epoca as saidas permanecem proximas de 0.5 "
          "(regiao de maxima incerteza da sigmoide) — o XOR nao e linearmente "
          "separavel e a rede precisa de muitas epocas de gradiente descendente "
          "para separar corretamente os 4 padroes.")


if __name__ == "__main__":
    main()
