# Lab 5 — Relatório de Experimentos: Feedforward (MLP) e Backpropagation

**Disciplina:** Tópicos Especiais em Inteligência Artificial (TEIA)
**UEPB 2026**
**Grupo:** Erick Nathan · Laura Barbosa · Pedro Lucas
**Referência:** Aula PR_711 (Prof. Robson Pequeno de Sousa)

---

## 1. Introdução

Este relatório documenta os dois experimentos exigidos no Lab 5:

- **Item (i):** implementação, em Python puro, de uma rede feedforward totalmente conectada (2 entradas → 2 neurônios ocultos → 2 neurônios de saída) treinada por retropropagação do erro, para o exemplo didático "reconhecimento de galinha e homem".
- **Item (ii):** classificação das 3 espécies do Iris usando uma rede feedforward (via `scikit-learn`, uso explicitamente permitido pelo enunciado apenas para este item), comparada com o Classificador Ótimo de Bayes (QDA) e o Naive Bayes, avaliando todas as métricas de qualidade.

---

## 2. Item (i) — Rede "Galinha vs Homem"

### 2.1 Arquitetura e Parâmetros

| Parâmetro | Valor |
|---|---|
| Entradas | $a_1 = 0{,}15$ &nbsp;&nbsp; $a_2 = 0{,}35$ |
| Pesos entrada → oculta | $w_1=0{,}10$ &nbsp; $w_2=0{,}20$ &nbsp; $w_3=0{,}12$ &nbsp; $w_4=0{,}17$ |
| Bias da camada oculta | $bw_1=0{,}80$ (→ $b_1$) &nbsp;&nbsp; $bw_2=0{,}25$ (→ $b_2$) |
| Pesos oculta → saída | $w_5=0{,}05$ &nbsp; $w_6=0{,}40$ &nbsp; $w_7=0{,}33$ &nbsp; $w_8=0{,}07$ |
| Bias da camada de saída | $bw_3=0{,}15$ (→ $c_1$) &nbsp;&nbsp; $bw_4=0{,}70$ (→ $c_2$) |
| Saída desejada | homem ($c_1$) = 0 &nbsp;&nbsp; galinha ($c_2$) = 1 |
| Taxa de aprendizagem | $\eta = 0{,}05$ |
| Ativação | Sigmoide, em ambas as camadas |

Implementação: `iris_classifier/models/mlp_backprop.py` (classe `RedeFeedforward`) e script demonstrativo `iris_classifier/lab05_galinha_homem.py`, ambos em **Python puro**, sem `numpy`/`scipy`/`scikit-learn`.

### 2.2 Passo 1 — Alimentação Adiante (Forward)

| Grandeza | Valor calculado | Valor do slide |
|---|:---:|:---:|
| $\text{out}_{b_1}$ | $0{,}7020$ | $0{,}7020$ |
| $\text{out}_{b_2}$ | $0{,}5841$ | $0{,}5841$ |
| $\text{out}_{c_1}$ (homem) | $0{,}5934$ | $0{,}5934$ |
| $\text{out}_{c_2}$ (galinha) | $0{,}7353$ | $0{,}7353$ |
| Erro total $E$ | $0{,}21107$ | $0{,}21108$ |

> [!TIP]
> **Verificação:** os quatro valores de ativação batem exatamente com os apresentados no slide da Aula PR_711; a diferença de $0{,}00001$ no erro total é apenas arredondamento de casas decimais. Isso confirma que a implementação em Python puro reproduz fielmente a matemática do material.

### 2.3 Passo 2 — Retropropagação (Deltas)

$$\delta_{c_1} = (\text{out}_{c_1} - t_1)\cdot \text{out}_{c_1}(1-\text{out}_{c_1}) = 0{,}143167$$
$$\delta_{c_2} = (\text{out}_{c_2} - t_2)\cdot \text{out}_{c_2}(1-\text{out}_{c_2}) = -0{,}051519$$
$$\delta_{b_1} = \left(\delta_{c_1} w_5 + \delta_{c_2} w_6\right)\text{out}_{b_1}(1-\text{out}_{b_1}) = -0{,}002813$$
$$\delta_{b_2} = \left(\delta_{c_1} w_7 + \delta_{c_2} w_8\right)\text{out}_{b_2}(1-\text{out}_{b_2}) = 0{,}010601$$

### 2.4 Passo 3 — Pesos Atualizados ($\eta = 0{,}05$)

| Peso | Antes | Depois |
|---|:---:|:---:|
| $w_5$ ($b_1{\to}c_1$) | $0{,}05000$ | $0{,}04497$ |
| $w_6$ ($b_1{\to}c_2$) | $0{,}40000$ | $0{,}40181$ |
| $w_7$ ($b_2{\to}c_1$) | $0{,}33000$ | $0{,}32582$ |
| $w_8$ ($b_2{\to}c_2$) | $0{,}07000$ | $0{,}07150$ |
| $bw_3$ ($\to c_1$) | $0{,}15000$ | $0{,}14284$ |
| $bw_4$ ($\to c_2$) | $0{,}70000$ | $0{,}70258$ |
| $w_1$ ($a_1{\to}b_1$) | $0{,}10000$ | $0{,}10002$ |
| $w_2$ ($a_1{\to}b_2$) | $0{,}20000$ | $0{,}19992$ |
| $w_3$ ($a_2{\to}b_1$) | $0{,}12000$ | $0{,}12005$ |
| $w_4$ ($a_2{\to}b_2$) | $0{,}17000$ | $0{,}16981$ |
| $bw_1$ ($\to b_1$) | $0{,}80000$ | $0{,}80014$ |
| $bw_2$ ($\to b_2$) | $0{,}25000$ | $0{,}24947$ |

### 2.5 Passo 4 — Nova Previsão (após 1 atualização)

| Grandeza | Antes da atualização | Depois de 1 passo |
|---|:---:|:---:|
| $\text{out}_{c_1}$ (homem) | $0{,}5934$ | $0{,}5902$ |
| $\text{out}_{c_2}$ (galinha) | $0{,}7353$ | $0{,}7362$ |
| Erro total $E$ | $0{,}21107$ | $0{,}20894$ |

> [!IMPORTANT]
> **Conclusão:** após apenas **1 passo** de retropropagação, o erro total caiu de $0{,}21107$ para $0{,}20894$ (redução de $\approx 1{,}0\%$) — a saída $c_1$ (homem) se afastou levemente de 0 (piorou) enquanto $c_2$ (galinha) se aproximou de 1 (melhorou), e o efeito líquido foi uma redução do erro agregado, confirmando que o passo de gradiente descendente está na direção correta. Repetindo esse processo por várias épocas, o erro tende a cair continuamente até a rede classificar corretamente a amostra de treino.

---

## 3. Item (ii) — Feedforward (MLP) vs Bayes Ótimo vs Naive Bayes no Iris

### 3.1 Configuração

- **Dataset:** Iris (150 amostras, 4 atributos, 3 classes), split estratificado 70% treino / 30% teste, `seed=42` (mesma partição usada em todos os laboratórios anteriores).
- **Rede Feedforward:** `sklearn.neural_network.MLPClassifier`, 1 camada oculta com 8 neurônios, ativação logística (sigmoide), solver `adam`.
- **Bayes Ótimo (QDA) e Naive Bayes:** mesma implementação em Python puro dos laboratórios anteriores (`iris_classifier/models/bayes_classifier.py`).

### 3.2 Resultados Globais

| Modelo | Acerto Global | Kappa | Tau |
|---|:---:|:---:|:---:|
| **Feedforward (MLP)** | $100{,}00\%$ | $1{,}0000$ | $1{,}0000$ |
| **Bayes Ótimo (QDA)** | $97{,}78\%$ | $0{,}9667$ | $0{,}9667$ |
| **Naive Bayes** | $97{,}78\%$ | $0{,}9667$ | $0{,}9667$ |

#### Matriz de Confusão — Feedforward (MLP)
```text
Predito \ Real  setosa      versicolor  virginica   Total
---------------------------------------------------------
setosa          15          0           0           15
versicolor      0           15          0           15
virginica       0           0           15          15
---------------------------------------------------------
Total           15          15          15          45
```

#### Matriz de Confusão — Bayes Ótimo & Naive Bayes (idênticas)
```text
Predito \ Real  setosa      versicolor  virginica   Total
---------------------------------------------------------
setosa          15          0           0           15
versicolor      0           14          0           14
virginica       0           1           15          16
---------------------------------------------------------
Total           15          15          15          45
```

### 3.3 Métricas por Classe (One-vs-Rest)

| Classe | Modelo | Precisão | Recall | F1 | F2 | MCC |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Setosa | MLP / Bayes / Naive | $1{,}0000$ | $1{,}0000$ | $1{,}0000$ | $1{,}0000$ | $1{,}0000$ |
| Versicolor | Feedforward (MLP) | $1{,}0000$ | $1{,}0000$ | $1{,}0000$ | $1{,}0000$ | $1{,}0000$ |
| Versicolor | Bayes / Naive | $1{,}0000$ | $0{,}9333$ | $0{,}9655$ | $0{,}9459$ | $0{,}9504$ |
| Virginica | Feedforward (MLP) | $1{,}0000$ | $1{,}0000$ | $1{,}0000$ | $1{,}0000$ | $1{,}0000$ |
| Virginica | Bayes / Naive | $0{,}9375$ | $1{,}0000$ | $0{,}9677$ | $0{,}9868$ | $0{,}9520$ |

### 3.4 Testes Z de Significância de Kappa (todos os pares)

$$Z = \frac{K_1 - K_2}{\sqrt{\text{Var}(K_1) + \text{Var}(K_2)}}$$

| Par | Z | p-valor | Veredito (5%) |
|---|:---:|:---:|---|
| Feedforward (MLP) × Bayes Ótimo | $1{,}0234$ | $0{,}306124$ | sem diferença significativa |
| Feedforward (MLP) × Naive Bayes | $1{,}0234$ | $0{,}306124$ | sem diferença significativa |
| Bayes Ótimo × Naive Bayes | $0{,}0000$ | $1{,}000000$ | sem diferença significativa |

> [!TIP]
> **Interpretação:** embora a rede feedforward tenha acertado $100\%$ das amostras de teste contra $97{,}78\%$ dos classificadores bayesianos (1 amostra de *versicolor* confundida com *virginica*), o teste Z mostra que essa diferença **não é estatisticamente significativa** ao nível de 5% — o conjunto de teste (45 amostras) é pequeno demais para diferenciar com confiança um único erro a mais ou a menos. Isso é consistente com o que já se observava nos laboratórios anteriores: nas pétalas, o Iris é quase perfeitamente separável, e os três paradigmas (redes neurais, distâncias probabilísticas Gaussianas) convergem para desempenho equivalente.

---

## 4. Conclusão

1. **Item (i):** a implementação em Python puro do `RedeFeedforward` reproduziu exatamente os valores de ativação apresentados no slide da Aula PR_711 (diferença apenas de arredondamento no erro total), validando a correção da alimentação adiante e do algoritmo de retropropagação implementados do zero.
2. **Item (ii):** a rede feedforward (via `scikit-learn`) atingiu acurácia perfeita no conjunto de teste do Iris, superando numericamente o Bayes Ótimo e o Naive Bayes — mas o teste Z de Kappa confirma que a diferença não é estatisticamente significativa dado o tamanho do conjunto de teste. Os três classificadores são estatisticamente equivalentes para este problema, cada um chegando lá por um caminho matemático distinto: distâncias probabilísticas Gaussianas (Bayes/Naive) versus otimização iterativa por gradiente descendente com camada oculta não linear (MLP).
