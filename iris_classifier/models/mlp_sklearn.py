"""
Rede Feedforward (MLP) via scikit-learn — Item (ii) do Lab 5.

O enunciado permite explicitamente o uso de bibliotecas de Machine Learning
apenas para este experimento (classificacao multiclasse do Iris, comparada
com Bayes Otimo e Naive Bayes). O item (i) do mesmo laboratorio (galinha vs
homem) e todo o restante do projeto permanecem em Python puro — ver
`iris_classifier/models/mlp_backprop.py`.
"""
from sklearn.neural_network import MLPClassifier


def treinar_mlp_iris(dados_treino, indices_atributos, semente=42,
                      camadas_ocultas=(8,), max_iter=3000):
    """
    Treina um MLPClassifier (rede feedforward totalmente conectada, treinada
    por retropropagacao do erro) para classificar as 3 especies do Iris.

    Retorna o modelo treinado (sklearn.neural_network.MLPClassifier).
    """
    X = [[d['atributos'][i] for i in indices_atributos] for d in dados_treino]
    y = [d['classe'] for d in dados_treino]

    modelo = MLPClassifier(
        hidden_layer_sizes=camadas_ocultas,
        activation='logistic',
        solver='adam',
        max_iter=max_iter,
        random_state=semente,
    )
    modelo.fit(X, y)
    return modelo


def prever_mlp_iris(modelo, dados, indices_atributos):
    """Retorna a lista de classes preditas pelo modelo para os dados dados."""
    X = [[d['atributos'][i] for i in indices_atributos] for d in dados]
    return list(modelo.predict(X))
