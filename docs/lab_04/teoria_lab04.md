# Lab 4 — Teoria Completa: Classificadores Probabilísticos (Bayes Ótimo & Naive Bayes)

**Referência:** Aulas de Reconhecimento de Padrões — Prof. Robson Pequeno de Sousa
**Implementação:** `iris_classifier/bayes_classifier.py` e `iris_classifier/mvn_tester.py` (Python puro, sem numpy/scipy/sklearn para treino/predição)
**Interface:** Aba 4 — *Bayes & Normalidade* da GUI (`iris_classifier/gui/tab_bayes.py`)

---

## 1. Classificadores Probabilísticos: Abordagem Generativa

Diferente dos classificadores baseados em distâncias rígidas (Distância Mínima) ou fronteiras ajustadas iterativamente com base em erros (Perceptron, Regra Delta), os **classificadores probabilísticos** modelam os dados de forma estatística.

A meta é estimar a probabilidade a posteriori de uma amostra $x \in \mathbb{R}^d$ pertencer a uma classe $\omega_j$:

$$P(\omega_j | x)$$

Pela **Regra de Bayes**, expressamos essa probabilidade em função da verossimilhança (densidade condicional) $p(x | \omega_j)$ e da probabilidade a priori da classe $P(\omega_j)$:

$$P(\omega_j | x) = \frac{p(x | \omega_j) P(\omega_j)}{p(x)}$$

Onde $p(x) = \sum_{k=1}^C p(x | \omega_k) P(\omega_k)$ é o fator de normalização (evidência). Para fins de classificação (Maximização da Probabilidade a Posteriori — regra MAP), o termo $p(x)$ é constante entre as classes e pode ser ignorado.

Se as classes forem equiprováveis (priores iguais, $P(\omega_j) = 1/C$), a regra de decisão se simplifica para:

$$\hat{y} = \arg\max_{j} p(x | \omega_j)$$

---

## 2. A Premissa de Normalidade Multivariada (Gaussiana)

Para modelar analiticamente a densidade condicional $p(x | \omega_j)$, assumimos que cada classe segue uma distribuição **Normal Multivariada (Gaussiana)** de dimensão $d$:

$$p(x | \omega_j) = \frac{1}{(2\pi)^{d/2} |\Sigma_j|^{1/2}} \exp\left( -\frac{1}{2} (x - m_j)^T \Sigma_j^{-1} (x - m_j) \right)$$

Onde:
- $m_j \in \mathbb{R}^d$ é o vetor de médias (centróide) da classe $\omega_j$.
- $\Sigma_j \in \mathbb{R}^{d \times d}$ é a matriz de covariância da classe $\omega_j$.
- $|\Sigma_j|$ é o determinante de $\Sigma_j$.
- $\Sigma_j^{-1}$ é a matriz inversa de $\Sigma_j$.

### Verificação de Aderência à Normalidade (R - Pacote MVN)
Antes de adotar o modelo Gaussiano, é crucial verificar se os dados de fato seguem uma distribuição normal multivariada. No laboratório, realizamos os seguintes testes para cada classe separadamente usando o ambiente R:
1. **Teste de Henze-Zirkler (HZ):** Mede a distância entre a função geradora de momentos empírica e a teórica de uma normal. Se o $p$-valor for $> 0.05$, aceita-se a hipótese nula $H_0$ de que os dados provêm de uma distribuição normal multivariada.
2. **Teste de Mardia:** Avalia a assimetria (skewness) e a curtose (kurtosis) multivariadas. Complementa o teste de HZ para fornecer maior robustez à análise.

---

## 3. Estimação de Parâmetros e Regularização (Treinamento)

### 3.1. Vetor de Médias Amostral
$$m_j = \frac{1}{N_j} \sum_{x \in \omega_j} x$$

### 3.2. Matriz de Covariância Amostral
$$\Sigma_j = \frac{1}{N_j - 1} \sum_{x \in \omega_j} (x - m_j)(x - m_j)^T$$

### 3.3. Regularização de Ridge (Tikhonov)
Em dados reais, ou quando o número de amostras de treino de uma classe é pequeno em relação à dimensão das variáveis, a matriz de covariância $\Sigma_j$ pode ser singular (não-inversível), resultando em determinante nulo $|\Sigma_j| = 0$. 

Para garantir a estabilidade numérica e evitar divisões por zero, aplicamos uma regularização do tipo **Ridge** adicionando um pequeno ruído $\epsilon = 10^{-9}$ à diagonal principal de $\Sigma_j$:

$$\Sigma'_j = \Sigma_j + \epsilon \cdot I$$

Onde $I$ é a matriz identidade de dimensão $d \times d$.

---

## 4. Classificador Bayes Ótimo (QDA)

O **Classificador de Bayes Ótimo** (também chamado de Análise Discriminante Quadrática — *Quadratic Discriminant Analysis - QDA*) assume que cada classe possui sua própria matriz de covariância $\Sigma_j$.

### Função Discriminante
Queremos maximizar a densidade probabilística condicional (sob priores iguais). Aplicando o logaritmo natural ($\ln$) na verossimilhança gaussiana (uma transformação estritamente crescente que preserva o ponto de máximo), obtemos a função discriminante quadrática:

$$d_j(x) = \ln p(x | \omega_j) = -\frac{d}{2}\ln(2\pi) - \frac{1}{2} \ln |\Sigma'_j| - \frac{1}{2} (x - m_j)^T (\Sigma'_j)^{-1} (x - m_j)$$

Ignorando o termo constante $-\frac{d}{2}\ln(2\pi)$, definimos a função discriminante de Bayes Ótimo (QDA) como:

$$\boxed{d_j^{\text{Bayes}}(x) = -\frac{1}{2} \ln |\Sigma'_j| - \frac{1}{2} d_M^2(x, m_j)}$$

Onde $d_M^2(x, m_j) = (x - m_j)^T (\Sigma'_j)^{-1} (x - m_j)$ é a **distância de Mahalanobis quadrada** entre a amostra $x$ e a média $m_j$.

### Geometria das Superfícies de Decisão
Como as matrizes de covariância $\Sigma_j$ variam entre as classes, a fronteira de decisão (superfície onde $d_i(x) = d_j(x)$) é uma equação de segunda ordem em $x$. No plano 2D, as superfícies de decisão resultantes são **curvas não-lineares** (parábolas, hipérboles ou elipses).

---

## 5. Classificador Naive Bayes

O classificador **Naive Bayes** simplifica o modelo gaussiano assumindo a hipótese (ingênua) de que todas as características (features) são **condicionalmente independentes** dada a classe. 

Matematicamente, isso significa que a covariância entre quaisquer dois atributos distintos é zero. Logo, a matriz de covariância estimativa $\Sigma_j$ é forçada a ser uma **matriz diagonal**:

$$\Sigma_j^{\text{Naive}} = \text{diag}(\sigma_{j1}^2, \sigma_{j2}^2, \dots, \sigma_{jd}^2)$$

Onde $\sigma_{ji}^2$ é a variância amostral do atributo $i$ na classe $j$.

### Função Discriminante Simplificada
Com a matriz de covariância diagonal, o determinante e a inversa tornam-se extremamente simples de calcular de forma direta:
- Determinante: $|\Sigma'_j| = \prod_{i=1}^d \sigma_{ji}^2$ (ou $\prod_{i=1}^d (\sigma_{ji}^2 + \epsilon)$)
- Inversa: $(\Sigma'_j)^{-1} = \text{diag}\left( \frac{1}{\sigma_{j1}^2 + \epsilon}, \dots, \frac{1}{\sigma_{jd}^2 + \epsilon} \right)$

A distância de Mahalanobis simplifica-se para a soma ponderada das diferenças quadráticas por atributo (distância Euclidiana normalizada):

$$d_M^2(x, m_j) = \sum_{i=1}^d \frac{(x_i - m_{ji})^2}{\sigma_{ji}^2}$$

Substituindo na expressão discriminante, a função de Naive Bayes é dada por:

$$\boxed{d_j^{\text{Naive}}(x) = -\frac{1}{2} \sum_{i=1}^d \ln(\sigma_{ji}^2) - \frac{1}{2} \sum_{i=1}^d \frac{(x_i - m_{ji})^2}{\sigma_{ji}^2}}$$

### Geometria das Superfícies de Decisão
A fronteira de decisão do Naive Bayes também é quadrática caso as variâncias sejam diferentes entre as classes. No entanto, por assumir independência dos eixos coordenados, as densidades de probabilidade formam elipsoides **alinhados com os eixos cartesianos** (sem rotação).

---

## 6. Comparação Matemática dos Classificadores

| Aspecto | Distância Mínima | Naive Bayes | Bayes Ótimo (QDA) |
|---|---|---|---|
| **Modelo da Covariância** | Assume $\Sigma_j = \sigma^2 I$ (igual p/ todos) | Diagonal $\Sigma_j = \text{diag}(\sigma_{ji}^2)$ | Livre $\Sigma_j$ (completa para cada classe) |
| **Inversa da Matriz** | Não necessária | Trivial ($1/\sigma_{ji}^2$) | Exige inversão matricial ($d \times d$) |
| **Fronteira de Decisão** | Linear (Hiperplano reto) | Quadrática (elipsoides alinhados) | Quadrática geral (curva livre) |
| **Complexidade de Treino** | $O(N \cdot d)$ | $O(N \cdot d)$ | $O(N \cdot d^2 + C \cdot d^3)$ |
| **Relação geométrica** | Passa no ponto médio dos protótipos | Curvas orientadas nos eixos | Curvas deformadas pela dispersão dos dados |

---

## 7. Álgebra Linear em Python Puro (Sem Numpy)

Para manter o rigor acadêmico do laboratório de implementar tudo a partir do zero, as operações matriciais cruciais para o Bayes Ótimo foram desenvolvidas em `math_utils.py` usando algoritmos tradicionais:

1. **Determinante de Matriz (`det_matriz`):** Implementado por meio de expansão por cofatores de Laplace (para matrizes pequenas $2 \times 2$) e decomposição/eliminação para matrizes maiores.
2. **Inversão Matricial (`inv_matriz`):** Desenvolvida através do algoritmo de **Eliminação de Gauss-Jordan com Pivotamento Parcial**, garantindo estabilidade numérica ao dividir pelos coeficientes pivôs máximos de cada coluna.
3. **Produto e Transposição Vetorial/Matricial:** Codificados puramente por laços aninhados (`for`) em Python.

---

## 8. Teste de Significância de Diferença de Kappas (Z-Test)

Após obter a acurácia e a matriz de confusão no conjunto de teste (30% das amostras), calculamos o coeficiente Kappa $K$ e sua variância teórica $\text{Var}(K)$ para ambos os classificadores.

Para avaliar se a diferença de desempenho é estatisticamente significativa no conjunto de teste, realizamos o **Teste Z de Kappas**:

$$Z = \frac{K_{\text{Bayes}} - K_{\text{Naive}}}{\sqrt{\text{Var}(K_{\text{Bayes}}) + \text{Var}(K_{\text{Naive}})}}$$

- **Hipótese Nula ($H_0$):** $K_{\text{Bayes}} = K_{\text{Naive}}$ (Desempenho dos classificadores é idêntico além do acaso).
- **Hipótese Alternativa ($H_1$):** $K_{\text{Bayes}} \neq K_{\text{Naive}}$ (Existe diferença real de acerto).
- **Regra de Decisão:** Ao nível de significância de $\alpha = 5\%$, rejeita-se $H_0$ se $|Z| > 1.96$.
- **Cálculo do p-valor:** Estimado de forma bicaudal integrando numericamente a densidade de probabilidade da normal padrão por meio da aproximação racional de Abramowitz & Stegun.

---

*Disciplina: Tópicos Especiais em Inteligência Artificial · UEPB · 2026*  
*Grupo: Erick Nathan · Laura Barbosa · Pedro Lucas*
