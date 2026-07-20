"""
Lab 5 — Item (i): Rede Feedforward para Reconhecimento de Galinha vs Homem
============================================================================

Reproduz o exemplo da Aula PR_711 (slides 27-36): uma rede totalmente
conectada 2-2-2 (2 entradas, 2 neuronios ocultos, 2 neuronios de saida),
com os pesos iniciais dados no slide, para classificar entre "homem" (c1)
e "galinha" (c2).

Arquitetura (nomenclatura do slide):
    Entradas:         a1 = 0.15   a2 = 0.35
    Pesos ocultos:    w1 (a1->b1) = 0.10   w2 (a1->b2) = 0.20
                      w3 (a2->b1) = 0.12   w4 (a2->b2) = 0.17
    Bias ocultos:     bw1 (->b1)  = 0.80   bw2 (->b2)  = 0.25
    Pesos saida:      w5 (b1->c1) = 0.05   w6 (b1->c2) = 0.40
                      w7 (b2->c1) = 0.33   w8 (b2->c2) = 0.07
    Bias saida:       bw3 (->c1)  = 0.15   bw4 (->c2)  = 0.70

    Saida desejada:   c1 (homem)   = 0
                      c2 (galinha) = 1

    Taxa de aprendizagem: eta = 0.05  (definida no enunciado do laboratorio)

Valores esperados do slide (conferidos e batidos nesta implementacao):
    out_b1 = 0.7020   out_b2 = 0.5841
    out_c1 = 0.5934   out_c2 = 0.7353
    Erro total E = 0.21108
"""
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.mlp_backprop import RedeFeedforward

ENTRADAS = [0.15, 0.35]
ALVO = [0.0, 1.0]  # c1 = homem (0), c2 = galinha (1)
TAXA_APRENDIZADO = 0.05

# w_oculta[i][j] = peso da entrada j para o neuronio oculto i
PESOS_OCULTA = [
    [0.10, 0.12],   # b1: w1 (de a1), w3 (de a2)
    [0.20, 0.17],   # b2: w2 (de a1), w4 (de a2)
]
BIAS_OCULTA = [0.80, 0.25]   # bw1 (->b1), bw2 (->b2)

# w_saida[i][j] = peso do neuronio oculto j para o neuronio de saida i
PESOS_SAIDA = [
    [0.05, 0.33],   # c1: w5 (de b1), w7 (de b2)
    [0.40, 0.07],   # c2: w6 (de b1), w8 (de b2)
]
BIAS_SAIDA = [0.15, 0.70]    # bw3 (->c1), bw4 (->c2)


def main():
    rede = RedeFeedforward(
        n_entradas=2, n_ocultos=2, n_saidas=2,
        pesos_oculta=PESOS_OCULTA, bias_oculta=BIAS_OCULTA,
        pesos_saida=PESOS_SAIDA, bias_saida=BIAS_SAIDA,
    )

    print("=" * 74)
    print("LAB 5 - ITEM (i): REDE FEEDFORWARD - GALINHA vs HOMEM")
    print("=" * 74)
    print(f"\nEntradas: a1={ENTRADAS[0]}, a2={ENTRADAS[1]}")
    print(f"Saida desejada: homem(c1)={ALVO[0]}, galinha(c2)={ALVO[1]}")
    print(f"Taxa de aprendizagem: eta={TAXA_APRENDIZADO}\n")

    resultado = rede.passo_treinamento(ENTRADAS, ALVO, TAXA_APRENDIZADO)

    print("--- PASSO 1: ALIMENTACAO ADIANTE (FORWARD) ---")
    print(f"  out_b1 = {resultado['saida_oculta'][0]:.4f}  (slide: 0.7020)")
    print(f"  out_b2 = {resultado['saida_oculta'][1]:.4f}  (slide: 0.5841)")
    print(f"  out_c1 (homem)   = {resultado['saida_rede'][0]:.4f}  (slide: 0.5934)")
    print(f"  out_c2 (galinha) = {resultado['saida_rede'][1]:.4f}  (slide: 0.7353)")
    print(f"  Erro total E = {resultado['erro_total']:.5f}  (slide: 0.21108)")

    print("\n--- PASSO 2: RETROPROPAGACAO (DELTAS) ---")
    print(f"  delta_c1 = {resultado['delta_saida'][0]:.6f}")
    print(f"  delta_c2 = {resultado['delta_saida'][1]:.6f}")
    print(f"  delta_b1 = {resultado['delta_oculta'][0]:.6f}")
    print(f"  delta_b2 = {resultado['delta_oculta'][1]:.6f}")

    print(f"\n--- PASSO 3: PESOS ATUALIZADOS (eta = {TAXA_APRENDIZADO}) ---")
    print("  Camada de saida:")
    print(f"    w5 (b1->c1): {PESOS_SAIDA[0][0]:.5f} -> {resultado['w_saida_depois'][0][0]:.5f}")
    print(f"    w6 (b1->c2): {PESOS_SAIDA[1][0]:.5f} -> {resultado['w_saida_depois'][1][0]:.5f}")
    print(f"    w7 (b2->c1): {PESOS_SAIDA[0][1]:.5f} -> {resultado['w_saida_depois'][0][1]:.5f}")
    print(f"    w8 (b2->c2): {PESOS_SAIDA[1][1]:.5f} -> {resultado['w_saida_depois'][1][1]:.5f}")
    print(f"    bw3 (->c1):  {BIAS_SAIDA[0]:.5f} -> {resultado['b_saida_depois'][0]:.5f}")
    print(f"    bw4 (->c2):  {BIAS_SAIDA[1]:.5f} -> {resultado['b_saida_depois'][1]:.5f}")
    print("  Camada oculta:")
    print(f"    w1 (a1->b1): {PESOS_OCULTA[0][0]:.5f} -> {resultado['w_oculta_depois'][0][0]:.5f}")
    print(f"    w2 (a1->b2): {PESOS_OCULTA[1][0]:.5f} -> {resultado['w_oculta_depois'][1][0]:.5f}")
    print(f"    w3 (a2->b1): {PESOS_OCULTA[0][1]:.5f} -> {resultado['w_oculta_depois'][0][1]:.5f}")
    print(f"    w4 (a2->b2): {PESOS_OCULTA[1][1]:.5f} -> {resultado['w_oculta_depois'][1][1]:.5f}")
    print(f"    bw1 (->b1):  {BIAS_OCULTA[0]:.5f} -> {resultado['b_oculta_depois'][0]:.5f}")
    print(f"    bw2 (->b2):  {BIAS_OCULTA[1]:.5f} -> {resultado['b_oculta_depois'][1]:.5f}")

    print("\n--- PASSO 4: NOVA PREVISAO (apos 1 atualizacao) ---")
    nova_saida = rede.prever(ENTRADAS)
    novo_erro = rede.erro_total(nova_saida, ALVO)
    print(f"  out_c1 (homem)   = {nova_saida[0]:.4f}")
    print(f"  out_c2 (galinha) = {nova_saida[1]:.4f}")
    print(f"  Novo erro total E = {novo_erro:.5f}  (era {resultado['erro_total']:.5f} antes da atualizacao)")


if __name__ == "__main__":
    main()
