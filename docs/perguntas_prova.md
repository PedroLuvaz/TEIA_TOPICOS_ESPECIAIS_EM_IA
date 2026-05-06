# Perguntas e Respostas para a Prova

> Questões prováveis com respostas completas. Estude ativamente: cubra as respostas e tente responder antes de ler.

---

## Q1. O que é um Classificador de Distância Mínima?

**Resposta:**

É um classificador baseado em protótipos que atribui uma amostra desconhecida à classe cujo protótipo (vetor médio das amostras de treinamento) é mais próximo, segundo a distância euclidiana.

Formalmente: dado um conjunto de protótipos $\{m_1, m_2, \ldots, m_K\}$, a classificação de $x$ é:

$$\hat{y} = \arg\min_j \|x - m_j\|$$

É um classificador **linear** — sua fronteira de decisão entre dois protótipos é sempre um hiperplano (reta em 2D) perpendicular ao segmento que une os dois protótipos.

---

## Q2. Como se calcula o protótipo de uma classe?

**Resposta:**

O protótipo $m_j$ é o vetor médio (centróide) de todas as amostras de treinamento pertencentes à classe $j$:

$$m_j = \frac{1}{N_j} \sum_{x \in \omega_j} x$$

No código (`math_utils.py`, função `calcular_media`), isso é implementado somando os valores de cada atributo de todas as amostras da classe e dividindo pelo número de amostras — tudo em Python puro com laços `for`.

Geometricamente, o protótipo é o "centro de gravidade" da nuvem de pontos da classe no espaço de features.

---

## Q3. Derive a função discriminante a partir da distância euclidiana.

**Resposta:**

Queremos minimizar $\|x - m_j\|^2$ (equivale a minimizar $\|x - m_j\|$, mais eficiente sem raiz quadrada). Expandindo:

$$\|x - m_j\|^2 = (x - m_j)^T(x - m_j) = x^Tx - 2x^Tm_j + m_j^Tm_j$$

O termo $x^Tx$ é **constante** para todos os $j$ (não depende da classe). Portanto, minimizar $\|x - m_j\|^2$ é equivalente a minimizar apenas:

$$-2x^Tm_j + m_j^Tm_j$$

Que por sua vez é equivalente a **maximizar**:

$$d_j(x) = x^Tm_j - \frac{1}{2}m_j^Tm_j$$

Esta é a **Função Discriminante Linear** do Classificador de Distância Mínima. A regra de decisão é: $\hat{y} = \arg\max_j d_j(x)$.

---

## Q4. O que é e como se calcula a superfície de decisão entre duas classes?

**Resposta:**

A superfície de decisão entre as classes $i$ e $j$ é o conjunto de pontos onde as duas funções discriminantes são iguais: $d_i(x) = d_j(x)$, ou seja, $d_{ij}(x) = d_i(x) - d_j(x) = 0$.

Substituindo:

$$\left(x^Tm_i - \tfrac{1}{2}m_i^Tm_i\right) - \left(x^Tm_j - \tfrac{1}{2}m_j^Tm_j\right) = 0$$

$$(m_i - m_j)^T x - \frac{1}{2}(m_i^Tm_i - m_j^Tm_j) = 0$$

Portanto: $w^Tx + b = 0$, onde:

$$w = m_i - m_j \qquad b = -\frac{1}{2}(\|m_i\|^2 - \|m_j\|^2)$$

**Interpretação geométrica:** É um hiperplano perpendicular ao vetor $w = m_i - m_j$, posicionado no ponto médio entre os dois protótipos. Em 2D (pétalas), é uma reta.

Para plotar em 2D: $x_2 = \dfrac{-w_1 x_1 - b}{w_2}$

---

## Q5. Por que a acurácia é 100% com pétalas mas não com sépalas?

**Resposta:**

Porque as **pétalas** tornam o dataset Iris **linearmente separável**, enquanto as **sépalas** não.

Com pétalas (comprimento + largura):
- Setosa tem pétalas muito pequenas (< 2cm) — completamente isolada das outras
- Versicolor e Virginica têm regiões distintas — uma reta consegue separá-las

Com sépalas (comprimento + largura):
- As distribuições de Versicolor e Virginica se **sobrepõem** no espaço 2D das sépalas
- Nenhuma reta consegue separar perfeitamente os dois grupos

O Classificador de Distância Mínima é um **classificador linear** — só funciona perfeitamente quando os dados são linearmente separáveis. Para dados não separáveis, modelos não-lineares (SVM com kernel RBF, redes neurais) seriam necessários.

---

## Q6. O que é split estratificado e por que é necessário?

**Resposta:**

Split estratificado é uma técnica de divisão de dados que **garante que a proporção de classes seja mantida** tanto no conjunto de treino quanto no de teste.

**Por que é necessário no Iris:**
O dataset está ordenado por classe (50 Setosa, depois 50 Versicolor, depois 50 Virginica). Um split simples dos primeiros 70% deixaria quase só Setosa e Versicolor no treino e quase só Virginica no teste — o modelo nunca aprenderia a reconhecer Virginica.

**Como implementamos:**
1. Separar amostras por classe
2. Embaralhar cada grupo separadamente (com `random.seed(42)`)
3. Pegar 70% de cada grupo para treino e 30% para teste

**Resultado:** 35 amostras de cada classe no treino (105 total) e 15 de cada no teste (45 total).

---

## Q7. O que representa cada elemento da matriz de confusão?

**Resposta:**

A matriz de confusão é uma tabela $K \times K$ onde a linha representa a classe **real** e a coluna representa a classe **predita**.

- **Diagonal principal** (posição $[i][i]$): amostras da classe $i$ corretamente classificadas como $i$ → **Verdadeiros Positivos (TP)**
- **Fora da diagonal**, posição $[i][j]$ com $j \neq i$: amostras reais da classe $i$ classificadas erroneamente como $j$
  - Para a classe $i$: são **Falsos Negativos (FN)** — o modelo não reconheceu $i$
  - Para a classe $j$: são **Falsos Positivos (FP)** — o modelo disse "$j$" errado

Com 100% de acurácia (pétalas), todos os elementos fora da diagonal são 0.

---

## Q8. Qual a diferença entre precisão e revocação? Quando cada uma importa?

**Resposta:**

$$\text{Precisão}_j = \frac{TP_j}{TP_j + FP_j} \qquad \text{"Das que eu disse que eram } j\text{, quantas eram?"}$$

$$\text{Revocação}_j = \frac{TP_j}{TP_j + FN_j} \qquad \text{"Das que eram } j\text{, quantas eu encontrei?"}$$

**Precisão importa mais quando:** o custo de um Falso Positivo é alto.
- Exemplo: filtro de spam (não quero marcar e-mails legítimos como spam)
- Exemplo: recomendação de produtos (não quero recomendar itens irrelevantes)

**Revocação importa mais quando:** o custo de um Falso Negativo é alto.
- Exemplo: diagnóstico de câncer (não quero deixar passar nenhum caso real)
- Exemplo: detecção de fraude (não quero deixar transações fraudulentas passarem)

**F1-Score** equilibra as duas. Útil quando não há preferência clara por uma delas.

---

## Q9. O que é produto escalar e como é usado no classificador?

**Resposta:**

O produto escalar de dois vetores $a$ e $b$ de dimensão $n$ é:

$$a^T b = \sum_{i=1}^{n} a_i \cdot b_i$$

No código (`math_utils.py`, função `produto_escalar`), é implementado como:
```python
sum(x * y for x, y in zip(a, b))
```

**Como é usado:**
1. Na função discriminante: $d_j(x) = \underbrace{x^T m_j}_{\text{produto escalar}} - \frac{1}{2}\underbrace{m_j^T m_j}_{\text{norma ao quadrado}}$
2. No cálculo de $b$ da fronteira: $b = -\frac{1}{2}(m_i^T m_i - m_j^T m_j)$

**Interpretação geométrica:** O produto escalar $x^T m_j$ mede o quanto $x$ "aponta na direção" de $m_j$. Quanto maior, mais alinhados estão.

---

## Q10. Por que o classificador de distância mínima é chamado de "linear"?

**Resposta:**

Porque sua fronteira de decisão é sempre um **hiperplano linear** no espaço de features.

A função discriminante $d_j(x) = x^T m_j - \frac{1}{2}m_j^T m_j$ é linear em $x$: é uma combinação linear dos componentes de $x$ (pesos = componentes de $m_j$) mais uma constante.

A fronteira $d_i(x) - d_j(x) = 0$ resulta em:
$$w^T x + b = 0$$

Que é a equação geral de um hiperplano. Em 2D, é uma reta; em 3D, um plano; em $n$D, um hiperplano de dimensão $n-1$.

**Limitação:** Classificadores lineares não conseguem separar classes cujas fronteiras naturais são curvas (não-lineares). Nesses casos, usa-se SVM com kernel, redes neurais, etc.

---

## Q11. Como o código garante reprodutibilidade?

**Resposta:**

Usando `random.seed(42)` antes de embaralhar as amostras no split estratificado (`data_loader.py`). Ao fixar a semente do gerador de números aleatórios, o embaralhamento é sempre o mesmo em qualquer execução, garantindo que:
- Os conjuntos de treino e teste sejam sempre idênticos
- Os resultados (acurácia, protótipos, etc.) sejam reproduzíveis
- Qualquer pessoa execute e obtenha os mesmos números

O valor 42 é convencional na comunidade de ML (referência ao livro "O Guia do Mochileiro das Galáxias").

---

## Q12. Qual é a complexidade computacional do treinamento e da predição?

**Resposta:**

**Treinamento:**
- Para cada classe, calcula a média de suas $N_j$ amostras de dimensão $d$
- Complexidade: $O(N \cdot d)$ onde $N$ = total de amostras, $d$ = dimensão

**Predição de uma amostra:**
- Calcula $d_j(x)$ para cada uma das $K$ classes: $O(K \cdot d)$
- Para 3 classes e 2 atributos: apenas 6 multiplicações + 3 somas

O Classificador de Distância Mínima é extremamente eficiente computacionalmente — o treinamento é apenas calcular médias, e a predição é apenas calcular produtos escalares.
