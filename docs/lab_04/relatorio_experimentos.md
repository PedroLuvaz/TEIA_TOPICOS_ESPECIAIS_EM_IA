# Lab 4 — Relatório de Experimentos: Bayes Ótimo e Naive Bayes

**Disciplina:** Tópicos Especiais em Inteligência Artificial (TEIA)  
**UEPB 2026**  
**Grupo:** Erick Nathan · Laura Barbosa · Pedro Lucas  

---

## 1. Introdução e Objetivo

Este relatório documenta a implementação prática e análise comparativa entre dois classificadores probabilísticos generativos aplicados ao conjunto de dados **Iris**:
1. **Classificador de Bayes Ótimo** (também conhecido como Análise Discriminante Quadrática - *QDA*).
2. **Classificador Naive Bayes** (que assume independência condicional dos atributos).

Ambos os modelos foram implementados em **Python puro** (sem uso de `numpy`, `scipy` ou `scikit-learn` para a lógica de álgebra linear e machine learning), realizando-se toda a manipulação matricial (determinantes, inversões via eliminação de Gauss-Jordan com pivotamento parcial, matrizes de covariância e distância de Mahalanobis) com laços `for` e estruturas de dados nativas.

Além disso, integrou-se o ambiente **R** utilizando o pacote **MVN** para realizar a verificação estatística de **Aderência à Distribuição Normal Multivariada** para cada uma das três espécies (*setosa*, *versicolor*, e *virginica*), utilizando os testes de **Henze-Zirkler** e **Mardia**.

---

## 2. Fundamentação Matemática e Modelagem

### 2.1 Densidade de Probabilidade e Classificador MAP
Assumindo que os dados de cada classe $C_j$ seguem uma distribuição Normal Multivariada com vetor de médias $m_j$ e matriz de covariância $\Sigma_j$:

$$P(x|C_j) = \frac{1}{(2\pi)^{d/2} |\Sigma_j|^{1/2}} \exp\left( -\frac{1}{2} (x - m_j)^T \Sigma_j^{-1} (x - m_j) \right)$$

Pela regra do **Máximo a Posteriori (MAP)**, a classe predita $\hat{y}$ é dada por:

$$\hat{y} = \arg\max_{j} P(C_j|x) = \arg\max_{j} \frac{P(x|C_j) P(C_j)}{P(x)}$$

Como a probabilidade marginal $P(x)$ é constante para todas as classes, ela pode ser suprimida. Além disso, por especificação do problema, assume-se que as probabilidades a priori são iguais ($P(C_1) = P(C_2) = P(C_3) = 1/3$), permitindo também suprimir o termo $P(C_j)$. Aplicando o logaritmo natural (função estritamente crescente) e descartando termos constantes independentes de $j$ como $-\frac{d}{2}\ln(2\pi)$, obtemos a **Função Discriminante Log-Verossimilhança**:

$$d_j(x) = -\frac{1}{2} \ln |\Sigma_j| - \frac{1}{2} (x - m_j)^T \Sigma_j^{-1} (x - m_j)$$

Onde a segunda parcela corresponde a metade do quadrado da **Distância de Mahalanobis** da amostra $x$ ao centróide $m_j$.

### 2.2 Diferenciação entre Bayes Ótimo e Naive Bayes
*   **Bayes Ótimo (QDA):** Estima a matriz de covariância completa $\Sigma_j$ para cada classe, capturando as correlações entre os atributos (comprimento e largura de sépalas e pétalas). Isso resulta em fronteiras de decisão quadráticas (não-lineares).
*   **Naive Bayes:** Assume independência condicional completa entre os atributos. Matematicamente, isso equivale a forçar todas as covariâncias (termos fora da diagonal de $\Sigma_j$) a $0$. A matriz $\Sigma_j$ torna-se uma matriz diagonal onde $\Sigma_{j,ii} = \sigma_{ji}^2$ (variância do atributo $i$ na classe $j$). A distância de Mahalanobis simplifica-se para a soma ponderada das distâncias euclidianas normalizadas pelas variâncias:

$$d_j(x) = -\frac{1}{2} \sum_{i=1}^d \ln(\sigma_{ji}^2) - \frac{1}{2} \sum_{i=1}^d \frac{(x_i - m_{ji})^2}{\sigma_{ji}^2}$$

---

## 3. Verificação de Normalidade Multivariada no R

A análise de aderência à distribuição Normal Multivariada foi realizada em R usando o pacote `MVN` sobre as 4 variáveis numéricas de cada classe. 

> [!IMPORTANT]
> **Hipóteses dos Testes:**
> *   $H_0$: Os dados vêm de uma distribuição Normal Multivariada.
> *   $H_1$: Os dados desviam significativamente da normalidade.
> *   *Critério:* Se o $p\text{-valor} > 0.05$, aceita-se $H_0$ (dados normais).

### 3.1 Resultados Obtidos
Abaixo constam os valores das estatísticas de teste e $p$-valores gerados pelo pacote `MVN`:

| Classe | Teste Henze-Zirkler (HZ) | Mardia Skewness (Assimetria) | Mardia Kurtosis (Curtose) | Aderência MVN ($H_0$) |
| :--- | :--- | :--- | :--- | :--- |
| **Setosa** | Stat = 0.9481, **p = 0.0496** | Stat = 22.4678, p = 0.3159 | Stat = 0.5842, p = 0.5591 | **NÃO** (p < 0.05 no HZ) |
| **Versicolor** | Stat = 0.4072, **p = 0.3802** | Stat = 17.1829, p = 0.6409 | Stat = 0.4902, p = 0.6241 | **SIM** |
| **Virginica** | Stat = 0.6482, **p = 0.0882** | Stat = 26.0418, p = 0.1639 | Stat = 0.1118, p = 0.9109 | **SIM** |

### 3.2 Análise Crítica
1.  **Setosa:** Rejeita-se a normalidade multivariada no nível de $5\%$ pelo teste Henze-Zirkler (pois $p = 0.0496 < 0.05$), embora passe confortavelmente no teste de assimetria e curtose de Mardia. Isso ocorre devido à alta sensibilidade do HZ a pequenas flutuações e correlações nas dimensões das pétalas e sépalas dessa espécie.
2.  **Versicolor & Virginica:** Apresentam forte aderência à normalidade multivariada, passando com folga em todos os testes estatísticos ($p$-valores muito superiores a $0.05$).
3.  **Implicação Prática:** A suposição de normalidade multivariada da densidade condicional $P(x|C_j)$ exigida pelo classificador quadrático de Bayes é **válida** para a maior parte das classes e serve como excelente aproximação paramétrica mesmo para a classe Setosa, que desvia por margem mínima da normalidade.

---

## 4. Avaliação Comparativa dos Modelos (Split 70/30)

O dataset foi dividido mantendo **70% para treino** (105 amostras, sendo 35 de cada classe) e **30% para teste** (45 amostras, sendo 15 de cada classe), com semente aleatória `random.seed(42)` para garantir reprodutibilidade total.

### 4.1 Resultados de Classificação Multiclasse (4 Variáveis)

Tanto o classificador **Bayes Ótimo (QDA)** quanto o **Naive Bayes** apresentaram desempenho idêntico no conjunto de teste:

*   **Acurácia Global (Acerto Global):** $97.78\%$ ($44$ acertos em $45$ amostras)
*   **Índice Kappa:** $0.9667$ (Variância: $0.001061$)
    *   *Classificação de Kappa (Landis & Koch):* **Quase Perfeito** ($0.81 - 1.00$)
*   **Total de Erros:** Apenas $1$ amostra errada (uma amostra real da classe *versicolor* classificada como *virginica*).

#### Matriz de Confusão Compartilhada (Bayes Ótimo & Naive Bayes)
```text
Predito \ Real  setosa      versicolor  virginica   Total
---------------------------------------------------------
setosa          15          0           0           15
versicolor      0           14          0           14
virginica       0           1           15          16
---------------------------------------------------------
Total           15          15          15          45
```

#### Métricas de Qualidade por Classe (One-vs-Rest)
| Classe | Acurácia Produtor (Recall) | Acurácia Usuário (Precisão) | F1-Score |
| :--- | :---: | :---: | :---: |
| **Setosa** | $1.0000$ | $1.0000$ | $1.0000$ |
| **Versicolor** | $0.9333$ | $1.0000$ | $0.9655$ |
| **Virginica** | $1.0000$ | $0.9375$ | $0.9677$ |

---

## 5. Teste de Significância de Kappa (Z-test) - Item (e)

Para responder formalmente qual classificador tem desempenho estatisticamente superior, realizamos o **Teste Z de Significância da diferença entre dois Kappas** (Congalton & Green, 2009):

$$Z = \frac{K_{\text{Bayes}} - K_{\text{Naive}}}{\sqrt{\text{Var}(K_{\text{Bayes}}) + \text{Var}(K_{\text{Naive}})}}$$

Substituindo os valores obtidos na classificação multiclasse:
*   $K_{\text{Bayes}} = 0.9667$ e $\text{Var}(K_{\text{Bayes}}) = 0.001061$
*   $K_{\text{Naive}} = 0.9667$ e $\text{Var}(K_{\text{Naive}}) = 0.001061$

$$Z = \frac{0.9667 - 0.9667}{\sqrt{0.001061 + 0.001061}} = 0.0000$$

O $p$-valor bilateral correspondente a $Z = 0.0000$ é:

$$p\text{-valor} = 1.0000$$

> [!TIP]
> **Veredito Estatístico:**
> Como $p\text{-valor} = 1.0000 > 0.05$, **não rejeitamos** a hipótese nula de igualdade. Não existe diferença estatisticamente significativa entre as acurácias dos classificadores de Bayes Ótimo e Naive Bayes na base Iris. Ambos exibem desempenho idêntico no conjunto de teste ($97.78\%$).

---

## 6. Superfícies de Decisão Par a Par (Item c)

As superfícies de decisão foram mapeadas avaliando a fronteira em que $d_i(x) = d_j(x)$ para as 2 variáveis mais discriminantes (Comprimento e Largura da Pétala). Foram plotados os contornos quadráticos gerados pelas equações no plano e salvos nos respectivos arquivos no diretório `outputs/`:

### i) Setosa vs Virginica
*   **Resultado de Teste:** $100\%$ de acurácia (ambos os modelos).
*   **Comentário:** As duas classes são amplamente separáveis, a Setosa está num cluster muito isolado.
*   **Gráficos salvos em:**
    *   [Bayes Ótimo](../../outputs/bayes_otimo_superficie_virginica_setosa.png)
    *   [Naive Bayes](../../outputs/naive_bayes_superficie_virginica_setosa.png)

### ii) Versicolor vs Virginica
*   **Resultado de Teste:** $100\%$ de acurácia (ambos os modelos) quando o subset do par é avaliado de forma binária e local (embora na multiclasse ocorra 1 erro de sobreposição).
*   **Comentário:** A fronteira de decisão quadrática do Bayes Ótimo adapta-se muito melhor à curvatura entre as duas distribuições, enquanto a do Naive Bayes (fronteira elipsoidal ortogonal) é forçada a ser paralela aos eixos devido à nulidade da covariância.
*   **Gráficos salvos em:**
    *   [Bayes Ótimo](../../outputs/bayes_otimo_superficie_versicolor_virginica.png)
    *   [Naive Bayes](../../outputs/naive_bayes_superficie_versicolor_virginica.png)

### iii) Setosa vs Versicolor
*   **Resultado de Teste:** $100\%$ de acurácia (ambos os modelos).
*   **Comentário:** Separação perfeita, a Setosa é linearmente e quadraticamente separável com folga de Versicolor.
*   **Gráficos salvos em:**
    *   [Bayes Ótimo](../../outputs/bayes_otimo_superficie_setosa_versicolor.png)
    *   [Naive Bayes](../../outputs/naive_bayes_superficie_setosa_versicolor.png)
