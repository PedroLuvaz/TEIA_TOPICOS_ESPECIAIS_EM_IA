# Lab 5 — Relatório de Experimentos: Feedforward (MLP) e Backpropagation

**Disciplina:** Tópicos Especiais em Inteligência Artificial (TEIA)
**UEPB 2026**
**Grupo:** Erick Nathan · Laura Barbosa · Pedro Lucas
**Referência:** Aula PR_711 (Prof. Robson Pequeno de Sousa)

---

## 1. Introdução

Este relatório documenta os experimentos do Lab 5, organizados em duas abas da GUI:

- **Lab 5.0** (`tab_xor.py`): o exemplo didático genérico do slide 37 (rede 2-2-2, $i_1/i_2$) e o exercício do XOR (slide 36), resolvido com a arquitetura mínima da Fig. 12.28(b) em 1 época, com fronteira de decisão 2D e curva de convergência interativas.
- **Lab 5.1** (`tab_feedforward.py`): os dois itens exigidos no enunciado formal — **item (i)**, implementação em Python puro de uma rede feedforward totalmente conectada (2 entradas → 2 ocultos → 2 saídas) para o exemplo "reconhecimento de galinha e homem"; e **item (ii)**, classificação das 3 espécies do Iris com uma rede feedforward (via `scikit-learn`, uso explicitamente permitido pelo enunciado apenas para este item), comparada com o Classificador Ótimo de Bayes (QDA) e o Naive Bayes — além do Exercício A (slide 34).

---

## 2. Lab 5.0 — XOR com MLP (slides 36-37)

### 2.1 Exemplo Didático (slide 37) — Rede 2-2-2 Genérica

**Arquitetura e parâmetros**

| Parâmetro | Valor |
|---|---|
| Entradas | $i_1 = 0{,}05$ &nbsp;&nbsp; $i_2 = 0{,}10$ |
| Pesos entrada → oculta | $h_1: [0{,}15;\ 0{,}20]$ &nbsp;&nbsp; $h_2: [0{,}25;\ 0{,}30]$ |
| Bias da camada oculta | $b_1 = 0{,}35$ (**único, compartilhado por $h_1$ e $h_2$**) |
| Pesos oculta → saída | $o_1: [0{,}40;\ 0{,}45]$ &nbsp;&nbsp; $o_2: [0{,}50;\ 0{,}55]$ |
| Bias da camada de saída | $b_2 = 0{,}60$ (**único, compartilhado por $o_1$ e $o_2$**) |
| Saída desejada | $o_1 = 0{,}01$ &nbsp;&nbsp; $o_2 = 0{,}99$ |
| Taxa de aprendizagem | $\eta = 0{,}5$ |

Diferente de todos os outros exemplos deste laboratório, aqui **o bias é um único valor por camada**, compartilhado por todos os neurônios dela — o gradiente do bias soma os deltas de **todos** os neurônios da camada ($\partial E/\partial b = \sum \delta$), em vez de cada neurônio ter seu próprio bias independente. Implementado em `JanelaMemoriaCalculoMLP` via o parâmetro `bias_compartilhado=True`.

**1ª iteração — Forward, erro e deltas**

| Grandeza | Valor calculado | Valor do slide |
|---|:---:|:---:|
| $\text{out}_{h_1}$ | $0{,}593270$ | $0{,}593270$ |
| $\text{out}_{h_2}$ | $0{,}596884$ | $0{,}596884$ |
| $\text{out}_{o_1}$ | $0{,}751365$ | $0{,}751365$ |
| $\text{out}_{o_2}$ | $0{,}772928$ | $0{,}772928$ |
| Erro total $E$ | $0{,}298371$ | $0{,}298371$ |
| $\delta_{o_1}$ | $0{,}138499$ | $0{,}138499$ |
| $\delta_{o_2}$ | $-0{,}038098$ | $-0{,}038098$ |
| $\delta_{h_1}$ | $0{,}008771$ | $0{,}008771$ |
| $\delta_{h_2}$ | $0{,}009954$ | $0{,}009954$ |

**Pesos e bias atualizados**

| Parâmetro | Antes | Depois | Slide |
|---|:---:|:---:|:---:|
| $w_5,\,w_6,\,w_7,\,w_8$ | $0{,}40;\ 0{,}45;\ 0{,}50;\ 0{,}55$ | $0{,}358916;\ 0{,}408666;\ 0{,}511301;\ 0{,}561370$ | idem |
| $w_1,\,w_2,\,w_3,\,w_4$ | $0{,}15;\ 0{,}20;\ 0{,}25;\ 0{,}30$ | $0{,}149781;\ 0{,}199561;\ 0{,}249751;\ 0{,}299502$ | idem |
| $b_2$ (saída, compartilhado) | $0{,}60$ | $0{,}549800$ | $0{,}549800$ |
| $b_1$ (oculta, compartilhado) | $0{,}35$ | $0{,}340637$ | $0{,}340637$ |

**2ª iteração completa (mesma entrada, pesos já atualizados)**

| Grandeza | Valor calculado | Slide |
|---|:---:|:---:|
| $\text{out}_{o_1}$ | $0{,}732024$ | $0{,}732024$ |
| $\text{out}_{o_2}$ | $0{,}765985$ | $0{,}765985$ |
| Novo erro total $E$ | $0{,}285751$ | $0{,}285751$ |

> [!TIP]
> **Verificação:** todos os valores acima — forward, deltas, pesos/bias atualizados e a 2ª iteração completa — batem **exatamente** com os slides 38-42 da Aula PR_711, incluindo a convenção de bias compartilhado. Isso confirma que `RedeFeedforward` reproduz corretamente as duas convenções de bias usadas ao longo do laboratório (independente por neurônio, no item i e no Exercício A; compartilhado por camada, apenas neste exemplo).

### 2.2 Exercício XOR (slide 36) — Arquitetura Fig. 12.28(b), 1 Época

**Arquitetura e parâmetros**

| Parâmetro | Valor |
|---|---|
| Arquitetura | 2 entradas → 2 ocultos → 1 saída (Fig. 12.28b) |
| Pesos entrada → oculta | $h_1: [0{,}50;\ 0{,}50]$ &nbsp;&nbsp; $h_2: [-0{,}50;\ -0{,}50]$ |
| Bias da camada oculta | $h_1=-0{,}20$ &nbsp;&nbsp; $h_2=0{,}30$ |
| Pesos oculta → saída | $[0{,}60;\ -0{,}60]$ |
| Bias da camada de saída | $-0{,}10$ |
| Taxa de aprendizagem | $\eta = 0{,}5$ |

*A Fig. 12.28(b) do slide mostra apenas a topologia (rótulos genéricos $w_1..w_9$, sem valores numéricos) — os pesos acima foram escolhidos pelo grupo para a demonstração, por isso não há "valor do slide" para comparar, diferente do exemplo didático e do Exercício A.*

Implementação: `iris_classifier/lab05_exercicio_xor.py` (script) e `iris_classifier/gui/tab_xor.py` (aba interativa).

**Previsões antes da época (pesos iniciais)**

| Padrão | Alvo | Saída |
|---|:---:|:---:|
| $(0,0)$ | $0$ | $0{,}4565$ |
| $(0,1)$ | $1$ | $0{,}4936$ |
| $(1,0)$ | $1$ | $0{,}4936$ |
| $(1,1)$ | $0$ | $0{,}5287$ |

**1 época = 4 padrões processados em sequência (modo online — atualização de pesos após cada padrão)**

Erro médio da época (calculado antes de cada atualização, mesma convenção de `perceptron.py`/`delta_rule.py`): $0{,}13156$.

**Previsões depois da época (pesos atualizados)**

| Padrão | Alvo | Saída | Classificação |
|---|:---:|:---:|---|
| $(0,0)$ | $0$ | $0{,}4579$ | ainda ambíguo |
| $(0,1)$ | $1$ | $0{,}4950$ | ainda ambíguo |
| $(1,0)$ | $1$ | $0{,}4950$ | ainda ambíguo |
| $(1,1)$ | $0$ | $0{,}5300$ | ainda ambíguo |

> [!IMPORTANT]
> **Conclusão:** após apenas 1 época, todas as saídas permanecem muito próximas de $0{,}5$ (a região de máxima incerteza da sigmoide) — o algoritmo mal começou a mover os pesos. Isso é o resultado **esperado e correto**: o XOR não é linearmente separável, e mesmo com a arquitetura correta (camada oculta), o gradiente descendente precisa de muitas épocas para convergir. O exercício pedia explicitamente apenas 1 época — o objetivo pedagógico é observar a direção do ajuste dos pesos, não a convergência.

**Além do exercício — treino interativo:** a aba Lab 5.0 permite continuar o treino além da 1 época pedida. Com estes pesos iniciais, o erro médio fica quase estacionado até por volta da época 500 (ainda $\approx 0{,}130$) e só converge de fato entre as épocas 2000-5000 (erro $<0{,}002$, todos os 4 padrões classificados corretamente). Isso demonstra, na prática, que a MLP com camada oculta **eventualmente resolve** o XOR — algo que a Regra Delta linear (Aba 2 do projeto) nunca consegue fazer, mesmo com épocas ilimitadas (seu MSE estaciona em $0{,}25$).

---

## 3. Lab 5.1  ·  Item (i) — Rede "Galinha vs Homem"

### 3.1 Arquitetura e Parâmetros

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

### 3.2 Passo 1 — Alimentação Adiante (Forward)

| Grandeza | Valor calculado | Valor do slide |
|---|:---:|:---:|
| $\text{out}_{b_1}$ | $0{,}7020$ | $0{,}7020$ |
| $\text{out}_{b_2}$ | $0{,}5841$ | $0{,}5841$ |
| $\text{out}_{c_1}$ (homem) | $0{,}5934$ | $0{,}5934$ |
| $\text{out}_{c_2}$ (galinha) | $0{,}7353$ | $0{,}7353$ |
| Erro total $E$ | $0{,}21107$ | $0{,}21108$ |

> [!TIP]
> **Verificação:** os quatro valores de ativação batem exatamente com os apresentados no slide da Aula PR_711; a diferença de $0{,}00001$ no erro total é apenas arredondamento de casas decimais. Isso confirma que a implementação em Python puro reproduz fielmente a matemática do material.

### 3.3 Passo 2 — Retropropagação (Deltas)

$$\delta_{c_1} = (\text{out}_{c_1} - t_1)\cdot \text{out}_{c_1}(1-\text{out}_{c_1}) = 0{,}143167$$
$$\delta_{c_2} = (\text{out}_{c_2} - t_2)\cdot \text{out}_{c_2}(1-\text{out}_{c_2}) = -0{,}051519$$
$$\delta_{b_1} = \left(\delta_{c_1} w_5 + \delta_{c_2} w_6\right)\text{out}_{b_1}(1-\text{out}_{b_1}) = -0{,}002813$$
$$\delta_{b_2} = \left(\delta_{c_1} w_7 + \delta_{c_2} w_8\right)\text{out}_{b_2}(1-\text{out}_{b_2}) = 0{,}010601$$

### 3.4 Passo 3 — Pesos Atualizados ($\eta = 0{,}05$)

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

### 3.5 Passo 4 — Nova Previsão (após 1 atualização)

| Grandeza | Antes da atualização | Depois de 1 passo |
|---|:---:|:---:|
| $\text{out}_{c_1}$ (homem) | $0{,}5934$ | $0{,}5902$ |
| $\text{out}_{c_2}$ (galinha) | $0{,}7353$ | $0{,}7362$ |
| Erro total $E$ | $0{,}21107$ | $0{,}20894$ |

> [!IMPORTANT]
> **Conclusão:** após apenas **1 passo** de retropropagação, o erro total caiu de $0{,}21107$ para $0{,}20894$ (redução de $\approx 1{,}0\%$) — a saída $c_1$ (homem) se afastou levemente de 0 (piorou) enquanto $c_2$ (galinha) se aproximou de 1 (melhorou), e o efeito líquido foi uma redução do erro agregado, confirmando que o passo de gradiente descendente está na direção correta. Repetindo esse processo por várias épocas, o erro tende a cair continuamente até a rede classificar corretamente a amostra de treino.

---

## 4. Lab 5.1  ·  Item (ii) — Feedforward (MLP) vs Bayes Ótimo vs Naive Bayes no Iris

### 4.1 Configuração

- **Dataset:** Iris (150 amostras, 4 atributos, 3 classes), split estratificado 70% treino / 30% teste, `seed=42` (mesma partição usada em todos os laboratórios anteriores).
- **Rede Feedforward:** `sklearn.neural_network.MLPClassifier`, 1 camada oculta com 8 neurônios, ativação logística (sigmoide), solver `adam`.
- **Bayes Ótimo (QDA) e Naive Bayes:** mesma implementação em Python puro dos laboratórios anteriores (`iris_classifier/models/bayes_classifier.py`).

### 4.2 Resultados Globais

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

### 4.3 Métricas por Classe (One-vs-Rest)

| Classe | Modelo | Precisão | Recall | F1 | F2 | MCC |
|---|---|:---:|:---:|:---:|:---:|:---:|
| Setosa | MLP / Bayes / Naive | $1{,}0000$ | $1{,}0000$ | $1{,}0000$ | $1{,}0000$ | $1{,}0000$ |
| Versicolor | Feedforward (MLP) | $1{,}0000$ | $1{,}0000$ | $1{,}0000$ | $1{,}0000$ | $1{,}0000$ |
| Versicolor | Bayes / Naive | $1{,}0000$ | $0{,}9333$ | $0{,}9655$ | $0{,}9459$ | $0{,}9504$ |
| Virginica | Feedforward (MLP) | $1{,}0000$ | $1{,}0000$ | $1{,}0000$ | $1{,}0000$ | $1{,}0000$ |
| Virginica | Bayes / Naive | $0{,}9375$ | $1{,}0000$ | $0{,}9677$ | $0{,}9868$ | $0{,}9520$ |

### 4.4 Testes Z de Significância de Kappa (todos os pares)

$$Z = \frac{K_1 - K_2}{\sqrt{\text{Var}(K_1) + \text{Var}(K_2)}}$$

| Par | Z | p-valor | Veredito (5%) |
|---|:---:|:---:|---|
| Feedforward (MLP) × Bayes Ótimo | $1{,}0234$ | $0{,}306124$ | sem diferença significativa |
| Feedforward (MLP) × Naive Bayes | $1{,}0234$ | $0{,}306124$ | sem diferença significativa |
| Bayes Ótimo × Naive Bayes | $0{,}0000$ | $1{,}000000$ | sem diferença significativa |

> [!TIP]
> **Interpretação:** embora a rede feedforward tenha acertado $100\%$ das amostras de teste contra $97{,}78\%$ dos classificadores bayesianos (1 amostra de *versicolor* confundida com *virginica*), o teste Z mostra que essa diferença **não é estatisticamente significativa** ao nível de 5% — o conjunto de teste (45 amostras) é pequeno demais para diferenciar com confiança um único erro a mais ou a menos. Isso é consistente com o que já se observava nos laboratórios anteriores: nas pétalas, o Iris é quase perfeitamente separável, e os três paradigmas (redes neurais, distâncias probabilísticas Gaussianas) convergem para desempenho equivalente.

---

## 5. Lab 5.1  ·  Exercício Extra (slide 34)

### 5.1 Exercício A (slide 34) — Rede da Figura 12.32, 1 Iteração

**Arquitetura e parâmetros**

| Parâmetro | Valor |
|---|---|
| Entradas | $x = [3{,}0;\ 0{,}0;\ 1{,}0]$ |
| Pesos entrada → oculta | $b_1: [0{,}1;\ 0{,}2;\ 0{,}6]$ &nbsp;&nbsp; $b_2: [0{,}4;\ 0{,}3;\ 0{,}1]$ |
| Bias da camada oculta | $b_1=0{,}4$ &nbsp;&nbsp; $b_2=0{,}2$ |
| Pesos oculta → saída | $c_1: [0{,}2;\ 0{,}1]$ &nbsp;&nbsp; $c_2: [0{,}1;\ 0{,}4]$ |
| Bias da camada de saída | $c_1=0{,}6$ &nbsp;&nbsp; $c_2=0{,}3$ |
| Saída desejada | $C_1 = 1$ &nbsp;&nbsp; $C_2 = 0$ |
| Taxa de aprendizagem | $\eta = 0{,}5$ *(não especificada no slide para este exercício — valor escolhido pelo grupo, mesma ordem de grandeza do exemplo didático completo da aula)* |

Implementação: `iris_classifier/lab05_exercicio_fig1232.py`, reaproveitando `RedeFeedforward` (Python puro).

**Passo 1 — Forward**

| Grandeza | Valor |
|---|:---:|
| $\text{out}_{b_1}$ | $0{,}7858$ |
| $\text{out}_{b_2}$ | $0{,}8176$ |
| $\text{out}_{c_1}$ | $0{,}6982$ |
| $\text{out}_{c_2}$ | $0{,}6694$ |
| Erro total $E$ | $0{,}26960$ |

**Passo 2 — Deltas**

| Delta | Valor |
|---|:---:|
| $\delta_{c_1}$ | $-0{,}063582$ |
| $\delta_{c_2}$ | $0{,}148140$ |
| $\delta_{b_1}$ | $0{,}000353$ |
| $\delta_{b_2}$ | $0{,}007890$ |

**Passo 3/4 — Nova previsão (após a única iteração, $\eta=0{,}5$)**

| Grandeza | Antes | Depois |
|---|:---:|:---:|
| $\text{out}_{c_1}$ | $0{,}6982$ | $0{,}7131$ |
| $\text{out}_{c_2}$ | $0{,}6694$ | $0{,}6304$ |
| Erro total $E$ | $0{,}26960$ | $0{,}23986$ |

> [!TIP]
> **Conclusão:** após a única iteração pedida pelo enunciado, o erro caiu de $0{,}26960$ para $0{,}23986$ ($\approx 11\%$ de redução) — $c_1$ se aproximou do alvo 1 e $c_2$ se aproximou do alvo 0, confirmando que o gradiente descendente moveu os pesos na direção correta já no primeiro passo.

---

## 6. Conclusão

1. **Lab 5.0 — Exemplo Didático (slide 37):** a implementação em Python puro do `RedeFeedforward` reproduziu exatamente os valores de todos os slides (38-42) — forward, deltas, atualização de pesos com bias compartilhado por camada e a 2ª iteração completa — validando a correção do algoritmo de retropropagação implementado do zero em ambas as convenções de bias usadas no laboratório.
2. **Lab 5.0 — Exercício XOR (slide 36):** o experimento com apenas 1 época demonstra, na prática, por que a camada oculta não linear é indispensável (o problema não é linearmente separável) e por que uma única época é insuficiente para convergência. O treino interativo além do exercício confirma que a MLP eventualmente resolve o XOR (convergência entre as épocas 2000-5000) — reforçando o mesmo limite teórico já observado com a Regra Delta linear na Aba 2, que nunca resolve o XOR (MSE estacionado em 0,25).
3. **Lab 5.1 — Item (i):** a implementação em Python puro do `RedeFeedforward` reproduziu exatamente os valores de ativação apresentados no slide da Aula PR_711 (diferença apenas de arredondamento no erro total).
4. **Lab 5.1 — Item (ii):** a rede feedforward (via `scikit-learn`) atingiu acurácia perfeita no conjunto de teste do Iris, superando numericamente o Bayes Ótimo e o Naive Bayes — mas o teste Z de Kappa confirma que a diferença não é estatisticamente significativa dado o tamanho do conjunto de teste. Os três classificadores são estatisticamente equivalentes para este problema, cada um chegando lá por um caminho matemático distinto: distâncias probabilísticas Gaussianas (Bayes/Naive) versus otimização iterativa por gradiente descendente com camada oculta não linear (MLP).
5. **Lab 5.1 — Exercício A:** a mesma implementação de backprop foi reutilizada para uma rede maior (3→2→2), confirmando que o motor generaliza para diferentes tamanhos de arquitetura sem alterações de código.
