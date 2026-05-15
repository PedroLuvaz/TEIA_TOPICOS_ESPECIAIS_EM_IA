# Teoria Completa — Classificadores Lineares

> Material de estudo detalhado para a prova. Cobre toda a teoria implementada no projeto:
> Classificador de Distância Mínima (Aba 1) · Perceptron de Rosenblatt (Aba 2) · Regra Delta / Adaline (Aba 2) · Problema XOR (Aba 2).

---

## 1. Reconhecimento de Padrões — Visão Geral

**Reconhecimento de Padrões** é a área que estuda como máquinas aprendem a identificar categorias a partir de dados. O pipeline básico é:

```
Dados Brutos → Extração de Features → Treinamento → Classificação → Avaliação
```

- **Feature (Atributo):** Uma característica mensurável de uma amostra. No Iris: comprimento da pétala, largura da sépala, etc.
- **Classe (Rótulo):** A categoria da amostra. No Iris: Setosa, Versicolor, Virginica.
- **Modelo:** A função matemática que mapeia features → classes.
- **Treinamento:** Ajuste dos parâmetros do modelo usando amostras rotuladas.

---

## 2. A Base de Dados Iris

Proposta por **Ronald Fisher em 1936** em seu artigo _"The use of multiple measurements in taxonomic problems"_. É um dos datasets mais famosos em ML.

| Característica | Valor |
|---|---|
| Total de amostras | 150 |
| Amostras por classe | 50 (balanceado) |
| Número de features | 4 |
| Número de classes | 3 |

**As 4 features (atributos):**
- Índice 0: Comprimento da Sépala (cm)
- Índice 1: Largura da Sépala (cm)
- Índice 2: **Comprimento da Pétala (cm)** ← mais discriminante
- Índice 3: **Largura da Pétala (cm)** ← mais discriminante

**As 3 classes:**
- `setosa` — pétalas muito pequenas, completamente separável
- `versicolor` — pétalas médias
- `virginica` — pétalas maiores

**Fato importante:** Usando apenas as pétalas (índices [2,3]), as 3 classes são **linearmente separáveis**. Usando sépalas (índices [0,1]), versicolor e virginica se sobrepõem.

---

## 3. Preparação dos Dados: Split Estratificado

### Por que não um split simples?

O dataset Iris está ordenado: as 50 primeiras amostras são Setosa, as 50 seguintes Versicolor, as 50 últimas Virginica. Se pegarmos os primeiros 70% diretamente, o conjunto de teste poderia não conter Setosa nem parte de Versicolor.

### Split Estratificado

Garante que a **proporção de classes seja mantida** em treino e teste.

**Algoritmo implementado (`data_loader.py`):**
1. Separar amostras por classe
2. Embaralhar cada grupo com `random.seed(42)` (reprodutibilidade)
3. Pegar os primeiros 70% de cada grupo para treino
4. Os 30% restantes de cada grupo para teste

**Resultado com 150 amostras (50 por classe):**
- Treino: 35 por classe × 3 = **105 amostras**
- Teste: 15 por classe × 3 = **45 amostras**

### Por que `random.seed(42)`?

Garante que qualquer pessoa que execute o código obtenha exatamente os mesmos resultados. É uma boa prática de reproducibilidade científica.

---

## 4. Classificador de Distância Mínima

### Intuição

> "Classifique uma nova amostra como pertencente à classe cujo **protótipo** está mais próximo dela."

É o classificador mais simples baseado em protótipos. Assume que cada classe pode ser resumida por um único ponto representativo no espaço de features.

### Passo 1: Calcular os Protótipos (Treinamento)

O **protótipo** (ou centróide) de uma classe é o vetor médio de todas as suas amostras de treinamento.

$$m_j = \frac{1}{N_j} \sum_{x \in \omega_j} x$$

Onde:
- $m_j$ = protótipo da classe $j$
- $N_j$ = número de amostras de treino da classe $j$
- $\omega_j$ = conjunto de amostras da classe $j$
- $x$ = vetor de features de uma amostra

**Exemplo numérico (pétalas, 2D):**
- Setosa: média de [1.4, 0.3], [1.4, 0.2], ... → $m_{setosa} \approx [1.48, 0.24]$
- Versicolor: média de [4.7, 1.4], ... → $m_{versicolor} \approx [4.26, 1.33]$
- Virginica: média de [6.0, 2.5], ... → $m_{virginica} \approx [5.55, 2.03]$

**No código:** `calcular_media()` em `math_utils.py`

### Passo 2: Classificar (Predição)

Para classificar uma nova amostra $x$, calculamos a distância euclidiana até cada protótipo e escolhemos o mais próximo:

$$\hat{y} = \arg\min_j \|x - m_j\|$$

**No entanto**, o projeto usa a **Função Discriminante Linear** — matematicamente equivalente, mas mais eficiente computacionalmente.

---

## 5. Função Discriminante Linear

### Derivação Completa

Queremos minimizar $\|x - m_j\|^2$. Expandindo:

$$\|x - m_j\|^2 = (x - m_j)^T(x - m_j)$$
$$= x^T x - 2x^T m_j + m_j^T m_j$$

Como $x^T x$ é constante para todos os $j$ (não depende da classe), **minimizar** $\|x - m_j\|^2$ equivale a **maximizar**:

$$d_j(x) = x^T m_j - \frac{1}{2} m_j^T m_j$$

> Isso é a **Função Discriminante do Classificador de Distância Mínima**.

### Por que "linear"?

Porque $d_j(x)$ é uma função **linear** em $x$: é um produto escalar (combinação linear) mais uma constante.

### Regra de Decisão (Regra do Máximo)

$$\hat{y} = \arg\max_j \; d_j(x)$$

Calculamos $d_{setosa}(x)$, $d_{versicolor}(x)$, $d_{virginica}(x)$ e escolhemos a classe com maior valor.

**No código:** `discriminante(x, mj)` em `math_utils.py`, `predizer_todas_classes()` em `classifier.py`

### Verificação intuitiva

Se $x$ está mais próximo de $m_i$ do que de $m_j$, então:
$$\|x - m_i\|^2 < \|x - m_j\|^2$$
$$\Rightarrow -2x^T m_i + m_i^T m_i < -2x^T m_j + m_j^T m_j$$
$$\Rightarrow x^T m_i - \frac{1}{2}m_i^T m_i > x^T m_j - \frac{1}{2}m_j^T m_j$$
$$\Rightarrow d_i(x) > d_j(x) \checkmark$$

---

## 6. Superfície de Decisão

### O que é?

A superfície de decisão entre duas classes $i$ e $j$ é o conjunto de pontos onde o classificador está "em dúvida" — onde as duas funções discriminantes são iguais:

$$d_{ij}(x) = d_i(x) - d_j(x) = 0$$

### Derivação dos Coeficientes

Substituindo a definição de $d_j(x)$:

$$d_i(x) - d_j(x) = 0$$
$$\left(x^T m_i - \frac{1}{2}m_i^T m_i\right) - \left(x^T m_j - \frac{1}{2}m_j^T m_j\right) = 0$$
$$(m_i - m_j)^T x - \frac{1}{2}(m_i^T m_i - m_j^T m_j) = 0$$

Portanto:

$$\underbrace{w^T x + b}_{=0} \quad \text{onde:}$$

$$\boxed{w = m_i - m_j}$$

$$\boxed{b = -\frac{1}{2}\left(\|m_i\|^2 - \|m_j\|^2\right)}$$

### Interpretação Geométrica

- $w = m_i - m_j$ é o vetor que aponta de $m_j$ para $m_i$
- A superfície de decisão é um **hiperplano perpendicular** a esse vetor
- Em 2D (pétalas), é uma **reta**
- A reta passa pelo **ponto médio** entre os dois protótipos
- É equidistante de $m_i$ e $m_j$

### Plotar a Reta em 2D

Para plotar, isolamos $x_2$:
$$w_1 x_1 + w_2 x_2 + b = 0$$
$$x_2 = \frac{-w_1 x_1 - b}{w_2}$$

**No código:** `coeficientes_superficie_decisao()` em `math_utils.py`, `plotar_superficie_decisao()` em `visualizer.py`

---

## 7. Separabilidade Linear

### Definição

Um conjunto de dados é **linearmente separável** se existe um hiperplano que separa perfeitamente as classes sem nenhum erro.

- Em 2D: uma reta separa as nuvens de pontos
- Em 3D: um plano separa
- Em $n$D: um hiperplano de dimensão $n-1$ separa

### Iris com Pétalas vs Sépalas

**Pétalas (índices [2,3]) → Linearmente separável:**
- Setosa: pétalas muito pequenas (comp. < 2cm), completamente à parte
- Versicolor vs Virginica: há uma fronteira clara, embora próxima

**Sépalas (índices [0,1]) → NÃO linearmente separável:**
- Versicolor e Virginica se sobrepõem no espaço das sépalas
- Nenhuma reta consegue separar perfeitamente

### Implicação Prática

O Classificador de Distância Mínima é um **classificador linear**. Ele só consegue 100% de acurácia quando os dados são linearmente separáveis. Para dados não separáveis, seria necessário usar classificadores não-lineares (redes neurais, SVM com kernel, etc.).

---

## 8. Métricas de Avaliação

### Acurácia

$$\text{Acurácia} = \frac{\text{Número de predições corretas}}{\text{Total de amostras}}$$

**Limitação:** Com classes desbalanceadas, a acurácia pode ser enganosa. Com Iris (balanceado), é confiável.

### Matriz de Confusão

Tabela que mostra a contagem de predições corretas e incorretas por classe.

```
            Predito
            Setosa   Versicolor  Virginica
Real Setosa    15          0          0
     Versicolor  0         15          0
     Virginica   0          0         15
```

- Diagonal principal: acertos (TP de cada classe)
- Fora da diagonal: erros (confusões entre classes)

### Precisão (Precision)

Para cada classe $j$:

$$\text{Precisão}_j = \frac{TP_j}{TP_j + FP_j}$$

**Pergunta respondida:** "Das vezes que o modelo disse 'é classe $j$', quantas vezes estava certo?"

**Quando importa:** Quando o custo de um falso positivo é alto (ex: diagnóstico de doença rara — não quero alarmar pacientes saudáveis).

### Revocação (Recall / Sensibilidade)

$$\text{Revocação}_j = \frac{TP_j}{TP_j + FN_j}$$

**Pergunta respondida:** "Das amostras que realmente eram classe $j$, quantas o modelo encontrou?"

**Quando importa:** Quando o custo de um falso negativo é alto (ex: detecção de câncer — não quero perder casos reais).

### F1-Score

$$F1_j = \frac{2 \cdot \text{Precisão}_j \cdot \text{Revocação}_j}{\text{Precisão}_j + \text{Revocação}_j}$$

É a **média harmônica** entre Precisão e Revocação. Penaliza modelos que são ótimos em uma métrica mas ruins na outra. Valor entre 0 e 1; 1 é perfeito.

### Macro-Média

A média simples de uma métrica sobre todas as classes. Trata todas as classes igualmente independente do tamanho.

$$\text{F1-macro} = \frac{1}{K}\sum_{j=1}^{K} F1_j$$

---

## 9. Resumo do Pipeline Completo

```
1. CARREGAR          iris_data.xls → 150 amostras {atributos: [4 floats], classe: str}
        ↓
2. SPLIT             Split Estratificado 70/30 → treino(105) + teste(45)
        ↓
3. TREINAR           Para cada classe: m_j = média dos vetores de pétala
        ↓            → protótipos: {setosa: [...], versicolor: [...], virginica: [...]}
4. PREDIZER          Para cada amostra de teste:
                     Calcular d_j(x) = xᵀmⱼ − ½mⱼᵀmⱼ para cada j
                     Retornar classe com maior d_j(x)
        ↓
5. AVALIAR           Acurácia, Matriz de Confusão, Precisão, Revocação, F1
        ↓
6. VISUALIZAR        Dispersão geral, heatmap de confusão, superfícies de decisão
```

---

## 10. Conexão com a Teoria de Bayes (contexto avançado)

O Classificador de Distância Mínima pode ser derivado como caso especial do **Classificador de Bayes** assumindo:
- Classes equiprováveis: $P(\omega_j) = 1/K$
- Distribuições gaussianas com mesma matriz de covariância: $\Sigma_j = \sigma^2 I$

Nesse caso, maximizar a função discriminante de Bayes reduz a minimizar a distância euclidiana ao protótipo — que é exatamente o que fazemos.

Isso justifica formalmente por que o algoritmo funciona para dados com distribuição aproximadamente gaussiana e classes bem separadas.

---

## 11. Perceptron de Rosenblatt

### Contexto histórico

O **Perceptron** foi proposto por Frank Rosenblatt em 1957 como modelo computacional do neurônio biológico. É o classificador linear mais simples com aprendizado adaptativo — diferentemente do Classificador de Distância Mínima, os pesos são ajustados iterativamente com base nos erros cometidos.

### Arquitetura

```
x₁ ──w₁──┐
x₂ ──w₂──┤  net = w₀·1 + w₁·x₁ + ... + wₙ·xₙ   →   y = sgn(net)
  ⋮       ├─→[ net ]──→[ sgn ]──→ y ∈ {+1, −1}
xₙ ──wₙ──┘
 1 ──w₀──┘  (bias — sempre 1, permite deslocar a fronteira)
```

O vetor aumentado (com bias) é: $x_\text{aug} = [1,\; x_1,\; x_2,\; \ldots,\; x_n]^T$

O vetor de pesos: $w = [w_0,\; w_1,\; w_2,\; \ldots,\; w_n]^T$

### Função de Ativação: Degrau Bipolar (sgn)

$$y = \text{sgn}(\text{net}) = \begin{cases} +1 & \text{se } w^T x_\text{aug} \geq 0 \\ -1 & \text{se } w^T x_\text{aug} < 0 \end{cases}$$

### Regra de Aprendizado

O Perceptron **só atualiza os pesos quando erra**. Dado um par de treinamento $(x, d)$ com $d \in \{+1, -1\}$:

$$w^{(t+1)} = w^{(t)} + p \cdot (d - y) \cdot x_\text{aug}$$

Onde:
- $p$ = taxa de aprendizado (ex: $p = 0{,}03$)
- $d$ = saída desejada (+1 ou −1)
- $y$ = saída atual = $\text{sgn}(w^T x_\text{aug})$
- $(d - y) \in \{0,\; +2,\; -2\}$ — zero quando correto

**Caso prático:** Se $d = +1$ e $y = -1$ (erro), então $(d-y) = +2$ e os pesos aumentam na direção de $x_\text{aug}$, empurrando a fronteira para o lado correto.

### Algoritmo Completo

```
Inicializar: w = [0, 0, ..., 0]

Para cada época até max_epocas:
    n_erros = 0
    Para cada amostra (x_aug, d) no treino:
        net = w^T · x_aug
        y = sgn(net)
        Se y ≠ d:
            n_erros += 1
            w = w + p · (d - y) · x_aug
    historico_erros.append(n_erros)
    Se n_erros == 0: parar (convergiu)
```

**No código:** `treinar_perceptron()` em `perceptron.py`

### Teorema da Convergência do Perceptron

> **Se os dados forem linearmente separáveis, o Perceptron converge em um número finito de iterações.**

Prova (Rosenblatt, 1957): Seja $\gamma$ a margem de separação (distância mínima de qualquer ponto à fronteira ótima) e $R = \max \|x_\text{aug}\|$. O número de atualizações é limitado por:

$$t_{\max} \leq \left(\frac{R}{\gamma}\right)^2$$

**Corolário:** Se os dados **não** são linearmente separáveis, o algoritmo **nunca para** (oscila indefinidamente). Por isso limitamos o treinamento com `max_epocas`.

### Resultados no Iris (pétalas [2,3])

| Par | Épocas | Convergiu | Acurácia (teste) |
|---|---|---|---|
| Setosa × Versicolor | 6 | ✓ | 100% |
| Setosa × Virginica | 5 | ✓ | 100% |
| Versicolor × Virginica | 100 | ✗ | ~50–80% |

O par Versicolor × Virginica não converge porque há amostras sobrepostas — mesmo com pétalas, a separação não é perfeita (os 5 erros da Aba 1 correspondem justamente a esses pontos fronteiriços).

---

## 12. Regra Delta (Widrow-Hoff / Adaline)

### Motivação

O Perceptron tem dois problemas:
1. Só converge para dados linearmente separáveis
2. A atualização usa $y = \text{sgn}(\text{net})$ — uma função não-diferenciável — o que impede o uso de gradiente descente

A **Regra Delta** (proposta por Widrow e Hoff em 1960, modelo ADALINE — *ADAptive LInear NEuron*) resolve ambos:
- Usa a saída **linear** $\text{net}$ na atualização (diferenciável)
- Minimiza o Erro Quadrático Médio (MSE), que tem solução analítica
- Converge para o **mínimo global de MSE** mesmo com dados sobrepostos

### Arquitetura

Idêntica ao Perceptron, mas o aprendizado usa a saída antes da limiarização:

```
x_aug ──→ [ net = w^T x_aug ] ──→ TREINAMENTO: usa net diretamente
                                  CLASSIFICAÇÃO: y = sgn(net)
```

### Função de Custo: MSE

$$E(w) = \frac{1}{N} \sum_{k=1}^{N} (d_k - \text{net}_k)^2 = \frac{1}{N} \sum_{k=1}^{N} (d_k - w^T x_k)^2$$

Onde $d_k \in \{+1, -1\}$ é o alvo e $\text{net}_k = w^T x_k$ é a saída linear.

### Derivação da Regra de Atualização

Calculamos o gradiente de $E$ em relação a $w_i$:

$$\frac{\partial E}{\partial w_i} = \frac{-2}{N} \sum_{k=1}^{N} (d_k - \text{net}_k) \cdot x_{k,i}$$

Gradiente descente: $w_i \leftarrow w_i - \eta \frac{\partial E}{\partial w_i}$

Na forma **online (estocástica)** — atualiza após cada amostra, descartando $N$ e absorvendo 2 em $p$:

$$\boxed{w \leftarrow w + p \cdot (d - \text{net}) \cdot x_\text{aug}}$$

### Diferença Fundamental em Relação ao Perceptron

| Característica | Perceptron | Regra Delta |
|---|---|---|
| Saída usada na atualização | $y = \text{sgn}(\text{net})$ (limiar) | $\text{net} = w^T x$ (linear) |
| Atualiza quando | Erra | **Sempre** (mesmo quando acerta) |
| O que minimiza | Erros de classificação | MSE (Erro Quadrático Médio) |
| Garante convergência | Só para dados separáveis | Sempre (ao mínimo de MSE) |
| Classificação final | $\text{sgn}(\text{net})$ | $\text{sgn}(\text{net})$ |

A regra de atualização parece semelhante, mas o erro $\delta$ é diferente:
- Perceptron: $\delta = d - \text{sgn}(\text{net}) \in \{-2, 0, +2\}$
- Regra Delta: $\delta = d - \text{net} \in \mathbb{R}$ (contínuo)

### Convergência e Superfície de Erro

A função de custo $E(w)$ é uma **parabolóide convexa** em $w$ — tem um único mínimo global. O gradiente descente converge garantidamente a esse mínimo.

Para dados **linearmente separáveis**: o MSE converge a zero (a fronteira ótima é encontrada).

Para dados **não separáveis**: o MSE converge ao menor valor possível — mas nunca a zero. A fronteira encontrada é o **melhor compromisso linear** entre os grupos sobrepostos.

### Resultados no Iris (pétalas [2,3])

| Par | MSE inicial | MSE final (200 ep.) | Acurácia (teste) |
|---|---|---|---|
| Setosa × Versicolor | 0.3332 | 0.0727 | 100% |
| Setosa × Virginica | 0.3734 | 0.0503 | 100% |
| Versicolor × Virginica | 0.1298 | ~0.14 | ~50–80% |

O par Ver×Vir com p=0.02 mostra comportamento esperado para dados sobrepostos: a Regra Delta encontra o compromisso linear mas não elimina todos os erros.

---

## 13. O Problema XOR — Limite dos Classificadores Lineares

### A Função XOR

A função booleana XOR (Ou-Exclusivo) é definida por:

| $x_1$ | $x_2$ | $d = x_1 \oplus x_2$ |
|---|---|---|
| 0 | 0 | **0** |
| 0 | 1 | **1** |
| 1 | 0 | **1** |
| 1 | 1 | **0** |

Os padrões com saída 0: $\{(0,0),\, (1,1)\}$ — estão nos cantos da diagonal principal.
Os padrões com saída 1: $\{(0,1),\, (1,0)\}$ — estão nos cantos da outra diagonal.

### Por que Nenhuma Reta Separa o XOR?

Para classificar perfeitamente, precisaríamos que:
- $w_0 + w_1 \cdot 0 + w_2 \cdot 0 < 0$ (para $(0,0)$, saída 0)
- $w_0 + w_1 \cdot 0 + w_2 \cdot 1 \geq 0$ (para $(0,1)$, saída 1)
- $w_0 + w_1 \cdot 1 + w_2 \cdot 0 \geq 0$ (para $(1,0)$, saída 1)
- $w_0 + w_1 \cdot 1 + w_2 \cdot 1 < 0$ (para $(1,1)$, saída 0)

Somando as inequações das saídas 1:
$$2w_0 + w_1 + w_2 \geq 0$$

Somando as inequações das saídas 0:
$$2w_0 + w_1 + w_2 < 0$$

Contradição: $2w_0 + w_1 + w_2$ não pode ser simultaneamente $\geq 0$ e $< 0$.

**Conclusão:** O XOR é matematicamente impossível de resolver com um único classificador linear.

### MSE Mínimo Teórico para o XOR

Dada a impossibilidade de separação perfeita, qual é o menor MSE que um classificador linear pode atingir?

Por simetria, os 4 padrões têm a mesma influência no centróide. O melhor valor constante que minimiza o MSE é a média dos alvos:

$$\bar{d} = \frac{0 + 1 + 1 + 0}{4} = 0{,}5$$

O MSE mínimo é então:

$$E_{\min} = \frac{1}{4}\sum_{k=1}^{4}(d_k - 0{,}5)^2 = \frac{4 \cdot 0{,}25}{4} = \boxed{0{,}25}$$

### O Que Acontece na Prática

Quando treinamos a Regra Delta no XOR por muitas épocas com $p$ pequeno:
- Os pesos convergem para $w \approx [0{,}5, 0, 0]$ (bias ≈ 0.5, pesos ≈ 0)
- A saída linear é $\approx 0{,}5$ para todas as entradas
- O MSE converge para $\approx 0{,}25$

Isso confirma que a Regra Delta encontrou o mínimo global da superfície quadrática — mas esse mínimo não representa uma separação útil.

### Solução: Redes Neurais Multicamada

O XOR foi o problema que motivou o desenvolvimento das **Redes Neurais com múltiplas camadas** (MLPs). Com uma camada oculta de 2 neurônios, o XOR pode ser resolvido perfeitamente usando ativações não-lineares (ex: sigmoide, ReLU).

> Esta extensão corresponde a Aba 3 (futura implementação com sklearn/MLP).

---

## 14. Comparação dos Três Classificadores

| | Distância Mínima | Perceptron | Regra Delta |
|---|---|---|---|
| **Treinamento** | Calcular média | Iterativo (por erros) | Iterativo (sempre) |
| **Saída usada no ajuste** | N/A (sem iteração) | $\text{sgn}(\text{net})$ | $\text{net}$ (linear) |
| **Fronteira de decisão** | Hiperplano equidistante | Hiperplano geral | Hiperplano (min. MSE) |
| **Garante convergência** | Sempre (1 passo) | Só se separável | Sempre (ao mín. MSE) |
| **O que minimiza** | Distância ao protótipo | Erros de class. | MSE |
| **Dados não separáveis** | Classifica (com erros) | Não converge | Converge ao mín. MSE |
| **Implementação** | `classifier.py` | `perceptron.py` | `delta_rule.py` |

### Equivalência Assintótica

Para dados linearmente separáveis, todos os três classificadores encontram **fronteiras equivalentes** (a menos de escala dos pesos). A diferença está no processo de aprendizado, não no resultado final.

Para dados não separáveis, apenas a Regra Delta tem comportamento bem definido (mínimo de MSE). O Perceptron oscila; o Classificador de Distância Mínima simplesmente classifica com os protótipos calculados.

---

## 15. Interface Gráfica: Aba 2 — Perceptron & Delta

A Aba 2 da GUI implementa os três experimentos acima de forma interativa:

| Modo | O que demonstra |
|---|---|
| **Perceptron + Setosa×Versicolor** | Convergência rápida (6 épocas), fronteira clara |
| **Perceptron + Versicolor×Virginica** | Não-convergência, sobreposição de classes |
| **Regra Delta + qualquer par** | Curva de MSE decrescente, fronteira minimizando MSE |
| **XOR (Regra Delta)** | MSE converge a 0.25, fronteira inútil, limite linear |

**Como interpretar os gráficos:**
- **Subplot esquerdo:** Scatter dos dados + fronteira de decisão (reta âmbar tracejada)
- **Subplot direito:** Curva de convergência — erros/época (Perceptron) ou MSE/época (Delta)

---

## 16. Pipeline Completo do Projeto (Ambas as Abas)

```
DADOS (150 amostras, 4 features, 3 classes)
       │
       ▼
SPLIT ESTRATIFICADO 70/30  (seed=42)
  Treino: 105  │  Teste: 45
       │
  ┌────┴──────────────────────────────┐
  │ ABA 1 — Distância Mínima          │ ABA 2 — Perceptron / Delta
  │                                    │
  │ Treino: m_j = média de cada classe │ Treino: w iterativo
  │ Predição: argmax d_j(x)            │ Perceptron: w ← w + p(d−y)x  [se errar]
  │ Fronteira: w=m_i−m_j, b=...        │ Delta:      w ← w + p(d−net)x [sempre]
  │ Acurácia pétalas: 100%             │ XOR: MSE → 0.25 (não converge)
  │ Acurácia sépalas: ~82%             │
  └────────────────────────────────────┘
       │
       ▼
AVALIAÇÃO: Acurácia · Matriz de Confusão · Precisão · Revocação · F1
       │
       ▼
VISUALIZAÇÃO: Dispersão · Fronteiras · Heatmap · Convergência
```
