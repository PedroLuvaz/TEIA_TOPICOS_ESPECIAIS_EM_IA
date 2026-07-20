"""
Perceptron Multicamadas (MLP) com Backpropagation — Python puro.

Rede feedforward totalmente conectada de 3 camadas (entrada -> oculta -> saida),
com ativacao sigmoide em todos os neuronios (oculta e saida), seguindo o
material da Aula PR_711 (Prof. Robson Pequeno de Sousa).

Notacao (mesma do slide):
    a_j(l-1)       : ativacao de entrada vinda da camada l-1
    z_i(l) = sum_j w_ij(l) * a_j(l-1) + b_i(l)   : entrada liquida (net) do neuronio i na camada l
    a_i(l) = h(z_i(l)) = sigmoide(z_i(l))        : saida (ativacao) do neuronio i na camada l

Algoritmo de treinamento — Retropropagacao do Erro (Backpropagation):
    Erro quadratico:         E = (1/2) * sum_i (t_i - z_i)^2   (t = alvo, z = saida da rede)
    delta da camada saida:   delta_o = (saida - alvo) * saida * (1 - saida)
    delta da camada oculta:  delta_h = (sum_o delta_o * w_ho) * saida_h * (1 - saida_h)
    gradiente:               dE/dw = delta * entrada_do_peso
    atualizacao (gradiente descendente): w_novo = w - taxa_aprendizado * dE/dw

Sem numpy/scipy/sklearn — apenas listas nativas e lacos for.
"""

import math
import random


def sigmoide(x):
    """h(x) = 1 / (1 + e^-x)"""
    return 1.0 / (1.0 + math.exp(-x))


class RedeFeedforward:
    """
    MLP totalmente conectado de 3 camadas: entrada -> oculta -> saida.
    Ativacao sigmoide em ambas as camadas (oculta e saida).

    Os pesos e bias podem ser fornecidos explicitamente (para reproduzir um
    exemplo didatico com valores iniciais dados no slide) ou inicializados
    aleatoriamente (para treinar do zero).
    """

    def __init__(self, n_entradas, n_ocultos, n_saidas,
                 pesos_oculta=None, bias_oculta=None,
                 pesos_saida=None, bias_saida=None, semente=None):
        """
        pesos_oculta[i][j] = peso do neuronio de entrada j para o neuronio oculto i
        pesos_saida[i][j]  = peso do neuronio oculto j para o neuronio de saida i
        bias_oculta[i]     = bias do neuronio oculto i
        bias_saida[i]      = bias do neuronio de saida i
        """
        if semente is not None:
            random.seed(semente)

        self.n_entradas = n_entradas
        self.n_ocultos = n_ocultos
        self.n_saidas = n_saidas

        self.w_oculta = pesos_oculta if pesos_oculta is not None else [
            [random.uniform(-0.5, 0.5) for _ in range(n_entradas)] for _ in range(n_ocultos)
        ]
        self.b_oculta = bias_oculta if bias_oculta is not None else [
            random.uniform(-0.5, 0.5) for _ in range(n_ocultos)
        ]
        self.w_saida = pesos_saida if pesos_saida is not None else [
            [random.uniform(-0.5, 0.5) for _ in range(n_ocultos)] for _ in range(n_saidas)
        ]
        self.b_saida = bias_saida if bias_saida is not None else [
            random.uniform(-0.5, 0.5) for _ in range(n_saidas)
        ]

    def forward(self, entradas):
        """
        Alimentacao adiante (feedforward).
        Retorna (saida_oculta, saida_rede) — ambas listas de ativacoes sigmoide.
        """
        saida_oculta = []
        for i in range(self.n_ocultos):
            net = self.b_oculta[i] + sum(
                self.w_oculta[i][j] * entradas[j] for j in range(self.n_entradas)
            )
            saida_oculta.append(sigmoide(net))

        saida_rede = []
        for i in range(self.n_saidas):
            net = self.b_saida[i] + sum(
                self.w_saida[i][j] * saida_oculta[j] for j in range(self.n_ocultos)
            )
            saida_rede.append(sigmoide(net))

        return saida_oculta, saida_rede

    def erro_total(self, saida_rede, alvo):
        """E = (1/2) * sum_i (alvo_i - saida_i)^2"""
        return 0.5 * sum((alvo[i] - saida_rede[i]) ** 2 for i in range(self.n_saidas))

    def passo_treinamento(self, entradas, alvo, taxa_aprendizado):
        """
        Executa UM passo completo de treinamento para 1 amostra:
          1. Alimentacao adiante (forward)
          2. Retropropagacao do erro (calculo dos deltas)
          3. Atualizacao dos pesos e bias (gradiente descendente)

        Retorna um dict com os valores intermediarios (para fins didaticos /
        memoria de calculo): saida_oculta, saida_rede, erro_total,
        delta_saida, delta_oculta e os pesos/bias apos a atualizacao.
        """
        saida_oculta, saida_rede = self.forward(entradas)
        erro = self.erro_total(saida_rede, alvo)

        # --- Retropropagacao: camada de saida ---
        # delta_o = (saida - alvo) * saida * (1 - saida)
        delta_saida = [
            (saida_rede[i] - alvo[i]) * saida_rede[i] * (1 - saida_rede[i])
            for i in range(self.n_saidas)
        ]

        # --- Retropropagacao: camada oculta ---
        # delta_h = (sum_o delta_o * w_ho) * saida_h * (1 - saida_h)
        delta_oculta = []
        for j in range(self.n_ocultos):
            soma = sum(delta_saida[i] * self.w_saida[i][j] for i in range(self.n_saidas))
            delta_oculta.append(soma * saida_oculta[j] * (1 - saida_oculta[j]))

        # --- Gradientes e atualizacao: pesos/bias da camada de saida ---
        w_saida_novo = [row[:] for row in self.w_saida]
        b_saida_novo = self.b_saida[:]
        for i in range(self.n_saidas):
            for j in range(self.n_ocultos):
                grad = delta_saida[i] * saida_oculta[j]
                w_saida_novo[i][j] = self.w_saida[i][j] - taxa_aprendizado * grad
            b_saida_novo[i] = self.b_saida[i] - taxa_aprendizado * delta_saida[i]

        # --- Gradientes e atualizacao: pesos/bias da camada oculta ---
        w_oculta_novo = [row[:] for row in self.w_oculta]
        b_oculta_novo = self.b_oculta[:]
        for i in range(self.n_ocultos):
            for j in range(self.n_entradas):
                grad = delta_oculta[i] * entradas[j]
                w_oculta_novo[i][j] = self.w_oculta[i][j] - taxa_aprendizado * grad
            b_oculta_novo[i] = self.b_oculta[i] - taxa_aprendizado * delta_oculta[i]

        resultado = {
            'saida_oculta': saida_oculta,
            'saida_rede': saida_rede,
            'erro_total': erro,
            'delta_saida': delta_saida,
            'delta_oculta': delta_oculta,
        }

        self.w_oculta, self.b_oculta = w_oculta_novo, b_oculta_novo
        self.w_saida, self.b_saida = w_saida_novo, b_saida_novo

        resultado['w_oculta_depois'] = self.w_oculta
        resultado['b_oculta_depois'] = self.b_oculta
        resultado['w_saida_depois'] = self.w_saida
        resultado['b_saida_depois'] = self.b_saida
        return resultado

    def treinar(self, X, Y, taxa_aprendizado=0.1, epocas=1000):
        """
        Treina a rede em multiplas amostras por multiplas epocas (modo
        online/estocastico: atualiza os pesos apos cada amostra).

        X: lista de vetores de entrada
        Y: lista de vetores de saida desejada (mesma ordem de X)

        Retorna: historico_erro (lista com o erro medio de cada epoca).
        """
        historico_erro = []
        for _ in range(epocas):
            erro_epoca = 0.0
            for entradas, alvo in zip(X, Y):
                resultado = self.passo_treinamento(entradas, alvo, taxa_aprendizado)
                erro_epoca += resultado['erro_total']
            historico_erro.append(erro_epoca / len(X))
        return historico_erro

    def prever(self, entradas):
        """Retorna a saida da rede (lista de ativacoes) para uma amostra."""
        _, saida_rede = self.forward(entradas)
        return saida_rede
