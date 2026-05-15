# Perguntas e Respostas para a Prova

> Questões prováveis com respostas completas. Estude ativamente: cubra as respostas e tente responder antes de ler.
> Cobre: Distância Mínima (Q1–Q12) · Perceptron (Q13–Q17) · Regra Delta (Q18–Q21) · XOR e Comparação (Q22–Q25).

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

---

## Q13. O que é o Perceptron e como ele aprende?

**Resposta:**

O **Perceptron de Rosenblatt (1957)** é o classificador linear mais simples com aprendizado iterativo. Diferente do Classificador de Distância Mínima (que calcula médias em um único passo), o Perceptron ajusta os pesos gradualmente com base nos erros.

**Estrutura:**
- Entradas: vetor aumentado $x_\text{aug} = [1, x_1, x_2, \ldots, x_n]^T$ (1 é o bias)
- Pesos: $w = [w_0, w_1, w_2, \ldots, w_n]^T$ — inicializados em 0
- Ativação: $\text{net} = w^T x_\text{aug}$
- Saída: $y = \text{sgn}(\text{net}) = +1$ se $\text{net} \geq 0$, senão $-1$

**Regra de aprendizado** (atualiza somente quando erra):

$$w \leftarrow w + p \cdot (d - y) \cdot x_\text{aug}$$

- Se $d = +1$ e $y = -1$ (erro): $(d-y) = +2$, pesos crescem na direção de $x_\text{aug}$
- Se $d = -1$ e $y = +1$ (erro): $(d-y) = -2$, pesos diminuem
- Se $d = y$ (acerto): $(d-y) = 0$, sem atualização

O aprendizado é **supervisionado** e **online** (atualiza amostra a amostra).

---

## Q14. Qual é o Teorema da Convergência do Perceptron?

**Resposta:**

O Teorema da Convergência (Rosenblatt, 1957) garante:

> **Se os dados de treinamento são linearmente separáveis, o Perceptron converge em um número finito de iterações, independentemente da taxa de aprendizado $p > 0$.**

O número máximo de atualizações de pesos é:

$$t_{\max} \leq \left(\frac{R}{\gamma}\right)^2$$

Onde:
- $R = \max_k \|x_{\text{aug},k}\|$ — norma máxima das amostras aumentadas
- $\gamma$ — margem de separação (distância do ponto mais próximo ao hiperplano ótimo)

**Implicações práticas:**
1. Se os dados são separáveis: o algoritmo sempre termina com zero erros
2. Se os dados **não** são separáveis: o algoritmo **nunca converge** — oscila indefinidamente entre estados de pesos, por isso limitamos o número de épocas (`max_epocas`)
3. A taxa de aprendizado $p$ não afeta *se* o algoritmo converge, mas *quantas* iterações leva

**No projeto:** Setosa × Versicolor com pétalas converge em 6 épocas. Versicolor × Virginica não converge em 100 épocas — há sobreposição.

---

## Q15. Por que o Perceptron falha em Versicolor × Virginica com pétalas?

**Resposta:**

Porque Versicolor e Virginica **não são perfeitamente linearmente separáveis** com as pétalas, mesmo que a acurácia do Classificador de Distância Mínima pareça 100% no conjunto de teste de 30 amostras.

O que acontece no dado completo (150 amostras):
- Existem **5 amostras** que cruzam a fronteira entre Versicolor e Virginica
- Essas amostras são de fato biologicamente "fronteiriças" — Iris de tamanho intermediário
- No conjunto de teste de 45 amostras, essas 5 amostras podem ou não aparecer

O Perceptron, diferente do Classificador de Distância Mínima, **itera sobre todos os dados** e não pode separar perfeitamente se há qualquer sobreposição. Por isso oscila e nunca atinge zero erros.

**Conclusão pedagógica:** A acurácia 100% do Classificador de Distância Mínima foi "sorte" do split aleatório — as amostras sobrepostas caíram no treino ou não apareceram no teste. O Perceptron é mais honesto: detecta a sobreposição e não converge.

---

## Q16. O que é a Regra Delta e como ela difere do Perceptron?

**Resposta:**

A **Regra Delta** (Widrow e Hoff, 1960 — modelo ADALINE: ADAptive LInear NEuron) é uma variante que usa a saída **linear** $\text{net}$ na atualização, em vez da saída limiarizada $y = \text{sgn}(\text{net})$:

$$w \leftarrow w + p \cdot (d - \text{net}) \cdot x_\text{aug}$$

**Diferença fundamental:**

| | Perceptron | Regra Delta |
|---|---|---|
| Erro | $d - \text{sgn}(\text{net}) \in \{-2, 0, +2\}$ | $d - \text{net} \in \mathbb{R}$ (contínuo) |
| Atualiza quando | Somente se errar | **Sempre** |
| Minimiza | Erros de classificação | MSE = $\frac{1}{N}\sum(d-\text{net})^2$ |
| Convergência | Só se separável | **Sempre** (ao mín. MSE) |

A Regra Delta é essencialmente **gradiente descendente** na superfície de erro MSE, que é uma parabolóide convexa — tem um único mínimo global, garantindo convergência.

---

## Q17. Derive a regra de atualização da Regra Delta a partir do gradiente.

**Resposta:**

A função de custo é o MSE:
$$E(w) = \frac{1}{N} \sum_{k=1}^{N} (d_k - \text{net}_k)^2 = \frac{1}{N} \sum_{k=1}^{N} \left(d_k - \sum_i w_i x_{k,i}\right)^2$$

Gradiente em relação ao peso $w_i$:
$$\frac{\partial E}{\partial w_i} = \frac{-2}{N} \sum_{k=1}^{N} (d_k - \text{net}_k) \cdot x_{k,i}$$

Regra de gradiente descendente (lote): $w_i \leftarrow w_i - \alpha \frac{\partial E}{\partial w_i}$

Na versão **online** (estocástica, por amostra), processamos um $(x_k, d_k)$ de cada vez, com $\alpha = p/2$:

$$w_i \leftarrow w_i + p \cdot (d_k - \text{net}_k) \cdot x_{k,i}$$

Ou na forma vetorial:

$$w \leftarrow w + p \cdot (d - \text{net}) \cdot x_\text{aug}$$

Esta é a **Regra Delta** — essencialmente gradiente descendente estocástico (SGD) na superfície MSE.

---

## Q18. Por que a Regra Delta converge mesmo para dados não separáveis?

**Resposta:**

Porque a função de custo MSE

$$E(w) = \frac{1}{N} \sum_{k=1}^{N} (d_k - w^T x_k)^2$$

é uma **função quadrática convexa** em $w$ — sua superfície de erro no espaço de pesos é uma parabolóide com um único mínimo global.

Isso significa que, independentemente do ponto de partida e do dado ser separável ou não, o gradiente descendente sempre converge ao mínimo global de $E(w)$.

**Para dados separáveis:** $E_{\min} = 0$ (fronteira perfeita existe).

**Para dados sobrepostos:** $E_{\min} > 0$ (o mínimo representa o melhor compromisso linear possível). Os pesos encontrados correspondem ao hiperplano que minimiza a soma dos erros quadráticos ao redor da fronteira de sobreposição.

**Contraste com o Perceptron:** O Perceptron minimiza erros de classificação (função não-convexa, não-diferenciável) — não há garantia de convergência para dados não separáveis.

---

## Q19. O que é o problema XOR e por que ele é importante?

**Resposta:**

O problema XOR é a função booleana $d = x_1 \oplus x_2$ com a tabela verdade:

| $(x_1, x_2)$ | $d$ | Grupo |
|---|---|---|
| $(0, 0)$ | 0 | Diagonal principal |
| $(1, 1)$ | 0 | Diagonal principal |
| $(0, 1)$ | 1 | Outra diagonal |
| $(1, 0)$ | 1 | Outra diagonal |

Os dois grupos (diagonal principal vs. outra diagonal) são distribuídos simetricamente no quadrado unitário — **nenhuma reta pode separá-los**.

**Por que é importante historicamente:** Em 1969, Minsky e Papert demonstraram que o Perceptron simples (1 camada) não consegue aprender o XOR. Isso causou o "Inverno da IA" — décadas de desinvestimento em redes neurais. A solução veio com o **algoritmo Backpropagation** para redes multicamada (1986), que pode resolver XOR com 1 camada oculta de 2 neurônios.

**No projeto:** O XOR serve como demonstração da limitação fundamental dos classificadores lineares — independentemente do algoritmo de treinamento, nenhum hiperplano resolve o problema.

---

## Q20. Calcule o MSE mínimo teórico do XOR com um classificador linear.

**Resposta:**

Por simetria, os 4 padrões têm a mesma influência na superfície de MSE. O melhor classificador linear faz a saída constante igual à média dos alvos:

$$\bar{d} = \frac{d_1 + d_2 + d_3 + d_4}{4} = \frac{0 + 1 + 1 + 0}{4} = 0{,}5$$

O MSE mínimo é:

$$E_{\min} = \frac{1}{4} \sum_{k=1}^{4} (d_k - 0{,}5)^2 = \frac{(0-0{,}5)^2 + (1-0{,}5)^2 + (1-0{,}5)^2 + (0-0{,}5)^2}{4}$$

$$= \frac{4 \times 0{,}25}{4} = \boxed{0{,}25}$$

**Interpretação:** Qualquer classificador linear no XOR erra sistematicamente com MSE ≥ 0,25. Na prática, a Regra Delta converge para os pesos $w \approx [0{,}5, 0, 0]$ (bias ≈ 0,5, outros ≈ 0), produzindo saída constante ≈ 0,5 — e MSE ≈ 0,25 confirmado experimentalmente.

---

## Q21. Prove que o XOR não é linearmente separável.

**Resposta:**

Suponha por absurdo que exista $w = [w_0, w_1, w_2]$ tal que o Perceptron classifica corretamente todos os 4 padrões:

- $(0,0) \to d=0$ (saída $-1$): $w_0 < 0$ ... (I)
- $(0,1) \to d=1$ (saída $+1$): $w_0 + w_2 \geq 0$ ... (II)
- $(1,0) \to d=1$ (saída $+1$): $w_0 + w_1 \geq 0$ ... (III)
- $(1,1) \to d=0$ (saída $-1$): $w_0 + w_1 + w_2 < 0$ ... (IV)

Somando (II) e (III):

$$2w_0 + w_1 + w_2 \geq 0 \quad \cdots (V)$$

Somando (I) e (IV):

$$2w_0 + w_1 + w_2 < 0 \quad \cdots (VI)$$

(V) e (VI) são **contraditórias**. Logo, não existe tal $w$ — o XOR não é linearmente separável. $\square$

---

## Q22. Compare os três classificadores implementados no projeto.

**Resposta:**

| | Dist. Mínima | Perceptron | Regra Delta |
|---|---|---|---|
| Arquivo | `classifier.py` | `perceptron.py` | `delta_rule.py` |
| Tipo de treinamento | Analítico (1 passo) | Iterativo (por erros) | Iterativo (sempre) |
| O que ajusta | Calcula protótipos | $w$ via sgn | $w$ via net linear |
| Função minimizada | $\|x - m_j\|$ | Erros de class. | MSE |
| Convergência garantida | Sempre | Só se separável | Sempre |
| Dados sobrepostos | Classifica c/ erros | Oscila | Mín. MSE |
| Fronteira de decisão | Perpendicular a $m_i - m_j$ | Qualquer hiperplano | Hiperplano de mín. MSE |

**No Iris (pétalas):**
- Distância Mínima: 100% no teste (5 erros na base completa)
- Perceptron Set×Ver: converge 6 épocas, 100%
- Perceptron Ver×Vir: não converge em 100 épocas
- Regra Delta Set×Ver: MSE 0,33→0,07, 100%

**Observação:** Para dados perfeitamente separáveis, todos os três encontram fronteiras similares. A diferença aparece nos dados sobrepostos.

---

## Q23. O que significa "taxa de aprendizado" e como ela afeta o treinamento?

**Resposta:**

A **taxa de aprendizado** $p$ (ou $\alpha$, $\eta$) controla o tamanho do passo de atualização dos pesos a cada iteração.

**Efeito no Perceptron:**
- A convergência é garantida para qualquer $p > 0$ (desde que os dados sejam separáveis)
- $p$ grande: pesos oscilam mais antes de convergir
- $p$ pequena: converge mais suavemente, mas pode precisar de mais épocas

**Efeito na Regra Delta:**
- $p$ muito grande: gradiente descendente pode **ultrapassar** o mínimo e divergir (instabilidade)
- $p$ muito pequena: convergência muito lenta
- Existe um $p_{\max}$ acima do qual o treinamento diverge (depende da escala dos dados)
- Regra prática: $p < \frac{2}{\lambda_{\max}}$ onde $\lambda_{\max}$ é o maior autovalor da matriz de covariância

**No projeto:**
- Perceptron: $p = 0{,}03$ (padrão)
- Regra Delta: $p = 0{,}02$ (menor, pois atualiza em todos os passos)

---

## Q24. Qual é a diferença entre o MSE antes e depois do limiar no Perceptron vs. Regra Delta?

**Resposta:**

Esta é a diferença central entre os dois algoritmos:

**Perceptron:** O erro usado na atualização é:

$$\delta_\text{perc} = d - \underbrace{\text{sgn}(\text{net})}_{\text{saída limiarizada}} \in \{-2, 0, +2\}$$

A função sgn é **não-diferenciável** em $\text{net}=0$ e tem gradiente zero em todo o resto. Isso significa que o Perceptron não pode ser interpretado como gradiente descendente em nenhuma função de custo suave.

**Regra Delta:** O erro usado na atualização é:

$$\delta_\text{delta} = d - \underbrace{\text{net}}_{\text{saída linear}} \in \mathbb{R}$$

Como $\text{net} = w^T x$ é linear em $w$, o gradiente de $E = (d - \text{net})^2$ em relação a $w$ é bem definido em todos os pontos, permitindo gradiente descendente clássico.

**Consequência prática:** A Regra Delta tem uma superfície de erro suave (parabolóide convexa) com garantia de convergência. O Perceptron navega por uma superfície não-suave sem garantia equivalente para dados sobrepostos.

---

## Q25. Como resolver o XOR com uma rede neural? (Resposta conceitual)

**Resposta:**

O XOR requer uma **rede neural multicamada** (MLP — Multi-Layer Perceptron) com pelo menos uma **camada oculta**.

**Arquitetura mínima para XOR:**

```
Entrada: x₁, x₂
   ↓
Camada Oculta: 2 neurônios com ativação não-linear (ex: sigmoide)
   ↓
Saída: 1 neurônio com limiar
```

**Intuição:** A camada oculta aprende a **transformar o espaço de features** — cria novas representações $h_1$, $h_2$ a partir de $x_1$, $x_2$ onde o XOR **se torna linearmente separável**. A camada de saída então aplica um classificador linear nas novas representações.

**Por exemplo**, os neurônios ocultos podem aprender:
- $h_1 \approx x_1 \text{ OR } x_2$ (ativo se pelo menos um é 1)
- $h_2 \approx x_1 \text{ AND } x_2$ (ativo só se ambos são 1)

Então: $d = h_1 \text{ AND NOT } h_2$, que é linearmente separável no espaço $(h_1, h_2)$.

**No projeto:** Esta extensão corresponde à Aba 3 futura (Redes Neurais com sklearn/MLP).
