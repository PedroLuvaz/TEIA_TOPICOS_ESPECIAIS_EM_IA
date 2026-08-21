"""
Rede feedforward multiclasse treinada por backpropagation — Python puro.

O Lab 5 ja tinha a `RedeFeedforward` (mlp_backprop.py), mas sempre aplicada a
exercicios especificos: o XOR, a rede da figura 12.32, o "galinha vs homem".
Aqui ela vira um classificador de uso geral, para qualquer base com qualquer
numero de classes — e o que permite a rede aparecer na tela de classificacao
ao lado dos demais modelos e entrar nos testes de significancia.

Duas decisoes de engenharia importam:

1. **Normalizacao min-max.** A sigmoide satura rapido: com entradas na escala
   de centimetros (ou pior, de milhares), os gradientes somem e a rede nao sai
   do lugar. Cada atributo e reescalado para [0, 1] usando o MINIMO E O MAXIMO
   DO TREINO — nunca do teste, para nao vazar informacao.
2. **Codificacao 1-de-C (one-hot).** A camada de saida tem um neuronio por
   classe; o alvo e 1 no neuronio da classe correta e 0 nos demais. A predicao
   e o argmax das ativacoes de saida.

Sem numpy, sem scikit-learn: so listas e lacos, como o resto do projeto.
"""
from models.mlp_backprop import RedeFeedforward


class RedeMulticlasse:
    """Envolve a `RedeFeedforward` com escala, one-hot e argmax."""

    def __init__(self, classes, indices_atributos, minimos, amplitudes, rede,
                 historico_erro):
        self.classes = classes
        self.indices = indices_atributos
        self.minimos = minimos
        self.amplitudes = amplitudes
        self.rede = rede
        self.historico_erro = historico_erro

    # -- pre-processamento ---------------------------------------------------
    def _entrada(self, atributos):
        """Seleciona as features usadas e reescala cada uma para [0, 1]."""
        return [(atributos[i] - self.minimos[k]) / self.amplitudes[k]
                for k, i in enumerate(self.indices)]

    # -- inferencia ----------------------------------------------------------
    def saidas(self, atributos):
        """Ativacao de cada neuronio de saida, ja associada a sua classe."""
        ativacoes = self.rede.prever(self._entrada(atributos))
        return {c: ativacoes[k] for k, c in enumerate(self.classes)}

    def predizer(self, atributos):
        """Classe do neuronio de saida mais ativo (argmax)."""
        saidas = self.saidas(atributos)
        return max(saidas, key=saidas.get)


def treinar_mlp_multiclasse(dados_treino, indices_atributos, n_ocultos=8,
                            taxa_aprendizado=0.3, epocas=300, semente=42):
    """
    Treina a rede em qualquer base do projeto.

    Parametros
    ----------
    dados_treino      : [{'atributos': [...], 'classe': str}, ...]
    indices_atributos : indices das features a usar
    n_ocultos         : neuronios da camada oculta
    taxa_aprendizado  : passo do gradiente descendente
    epocas            : passagens completas pelo conjunto de treino
    semente           : fixa a inicializacao aleatoria dos pesos

    Retorna uma `RedeMulticlasse` pronta para predizer.
    """
    classes = sorted(set(d['classe'] for d in dados_treino))

    # Escala calculada SO com o treino.
    minimos, amplitudes = [], []
    for i in indices_atributos:
        valores = [d['atributos'][i] for d in dados_treino]
        menor, maior = min(valores), max(valores)
        minimos.append(menor)
        # Atributo constante: amplitude 1 evita divisao por zero (a entrada
        # vira sempre 0, e a rede simplesmente ignora essa dimensao).
        amplitudes.append((maior - menor) or 1.0)

    X, Y = [], []
    for d in dados_treino:
        X.append([(d['atributos'][i] - minimos[k]) / amplitudes[k]
                  for k, i in enumerate(indices_atributos)])
        Y.append([1.0 if c == d['classe'] else 0.0 for c in classes])

    rede = RedeFeedforward(len(indices_atributos), n_ocultos, len(classes),
                           semente=semente)
    historico = rede.treinar(X, Y, taxa_aprendizado=taxa_aprendizado,
                             epocas=epocas)

    return RedeMulticlasse(classes, list(indices_atributos), minimos,
                           amplitudes, rede, historico)
