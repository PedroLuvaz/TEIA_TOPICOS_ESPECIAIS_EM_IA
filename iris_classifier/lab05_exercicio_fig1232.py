"""
Lab 5 — Exercicio A (slide 34): Treinamento da Rede da Figura 12.32 — 1 Iteracao
==================================================================================

Enunciado do slide: "Treine a rede abaixo, em que a saida desejada e 1 para C1
e 0 para C2, so uma interacao [iteracao]."

A rede e a mesma "pequena rede totalmente conectada" apresentada nos slides
"Exemplo" (Figuras 12.32/12.33), reaproveitada aqui com um alvo explicito:

    Entradas:          x = [3, 0, 1]
    Pesos entrada->oculta (W2):
        b1: w = [0.1, 0.2, 0.6]
        b2: w = [0.4, 0.3, 0.1]
    Bias oculta (b2 no slide):  [0.4, 0.2]
    Pesos oculta->saida (W3):
        c1: w = [0.2, 0.1]
        c2: w = [0.1, 0.4]
    Bias saida (b3 no slide):   [0.6, 0.3]

    Saida desejada:    C1 = 1   C2 = 0
    Taxa de aprendizagem: eta = 0.5  (nao especificada no slide para este
                          exercicio; adotado o mesmo valor do exemplo
                          didatico completo "i1/i2" da mesma aula, usado
                          como referencia de ordem de grandeza razoavel)

Valores de alimentacao adiante (ja verificados nos slides "Exemplo"):
    out_b1 = 0.7858   out_b2 = 0.8176
    out_c1 = 0.6982   out_c2 = 0.6694
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mlp_backprop import RedeFeedforward

ENTRADAS = [3.0, 0.0, 1.0]
ALVO = [1.0, 0.0]  # C1 = 1, C2 = 0
TAXA_APRENDIZADO = 0.5

PESOS_OCULTA = [
    [0.1, 0.2, 0.6],   # b1
    [0.4, 0.3, 0.1],   # b2
]
BIAS_OCULTA = [0.4, 0.2]

PESOS_SAIDA = [
    [0.2, 0.1],   # c1
    [0.1, 0.4],   # c2
]
BIAS_SAIDA = [0.6, 0.3]


def main():
    rede = RedeFeedforward(
        n_entradas=3, n_ocultos=2, n_saidas=2,
        pesos_oculta=PESOS_OCULTA, bias_oculta=BIAS_OCULTA,
        pesos_saida=PESOS_SAIDA, bias_saida=BIAS_SAIDA,
    )

    print("=" * 74)
    print("LAB 5 - EXERCICIO A: REDE DA FIGURA 12.32 (1 ITERACAO)")
    print("=" * 74)
    print(f"\nEntradas: x = {ENTRADAS}")
    print(f"Saida desejada: C1={ALVO[0]:.0f}  C2={ALVO[1]:.0f}")
    print(f"Taxa de aprendizagem: eta={TAXA_APRENDIZADO}\n")

    r = rede.passo_treinamento(ENTRADAS, ALVO, TAXA_APRENDIZADO)

    print("--- PASSO 1: ALIMENTACAO ADIANTE (FORWARD) ---")
    print(f"  out_b1 = {r['saida_oculta'][0]:.4f}  (slide: 0.7858)")
    print(f"  out_b2 = {r['saida_oculta'][1]:.4f}  (slide: 0.8176)")
    print(f"  out_c1 = {r['saida_rede'][0]:.4f}  (slide: 0.6982)")
    print(f"  out_c2 = {r['saida_rede'][1]:.4f}  (slide: 0.6694)")
    print(f"  Erro total E = {r['erro_total']:.5f}")

    print("\n--- PASSO 2: RETROPROPAGACAO (DELTAS) ---")
    print(f"  delta_c1 = {r['delta_saida'][0]:.6f}")
    print(f"  delta_c2 = {r['delta_saida'][1]:.6f}")
    print(f"  delta_b1 = {r['delta_oculta'][0]:.6f}")
    print(f"  delta_b2 = {r['delta_oculta'][1]:.6f}")

    print(f"\n--- PASSO 3: PESOS ATUALIZADOS (eta = {TAXA_APRENDIZADO}) ---")
    print("  Camada de saida:")
    print(f"    c1: {PESOS_SAIDA[0]} -> {[round(v, 5) for v in r['w_saida_depois'][0]]}")
    print(f"    c2: {PESOS_SAIDA[1]} -> {[round(v, 5) for v in r['w_saida_depois'][1]]}")
    print(f"    bias: {BIAS_SAIDA} -> {[round(v, 5) for v in r['b_saida_depois']]}")
    print("  Camada oculta:")
    print(f"    b1: {PESOS_OCULTA[0]} -> {[round(v, 5) for v in r['w_oculta_depois'][0]]}")
    print(f"    b2: {PESOS_OCULTA[1]} -> {[round(v, 5) for v in r['w_oculta_depois'][1]]}")
    print(f"    bias: {BIAS_OCULTA} -> {[round(v, 5) for v in r['b_oculta_depois']]}")

    print("\n--- PASSO 4: NOVA PREVISAO (apos 1 atualizacao) ---")
    nova_saida = rede.prever(ENTRADAS)
    novo_erro = rede.erro_total(nova_saida, ALVO)
    print(f"  out_c1 = {nova_saida[0]:.4f}  (era {r['saida_rede'][0]:.4f}, alvo=1)")
    print(f"  out_c2 = {nova_saida[1]:.4f}  (era {r['saida_rede'][1]:.4f}, alvo=0)")
    print(f"  Novo erro total E = {novo_erro:.5f}  (era {r['erro_total']:.5f} antes da atualizacao)")


if __name__ == "__main__":
    main()
