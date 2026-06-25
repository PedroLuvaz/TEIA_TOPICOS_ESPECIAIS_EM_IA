# Formulário — Classificadores Lineares

> Referência rápida para consulta durante a prova.
> Cobre: Distância Mínima · Perceptron · Regra Delta · XOR.

---

## Notação

| Símbolo | Significado |
|---|---|
| $x$ | Vetor de features de uma amostra |
| $x_\text{aug}$ | Vetor aumentado com bias: $[1, x_1, x_2, \ldots, x_n]^T$ |
| $m_j$ | Protótipo (vetor médio) da classe $j$ |
| $N_j$ | Número de amostras de treino da classe $j$ |
| $\omega_j$ | Conjunto de amostras da classe $j$ |
| $K$ | Número de classes |
| $w$ | Vetor de pesos (Perceptron/Delta: inclui bias $w_0$) |
| $w_0$ | Bias (limiar) — peso associado à entrada constante 1 |
| $b$ | Bias da fronteira de decisão (Distância Mínima) |
| $p$ | Taxa de aprendizado |
| $d$ | Saída desejada (alvo): $+1$ ou $-1$ (binário) |
| $\text{net}$ | Ativação linear: $\text{net} = w^T x_\text{aug}$ |
| $y$ | Saída do classificador: $\text{sgn}(\text{net})$ |
| $E$ | Erro Quadrático Médio (MSE) |
| $TP_j$ | True Positives da classe $j$ |
| $FP_j$ | False Positives da classe $j$ |
| $FN_j$ | False Negatives da classe $j$ |

---

## Treinamento: Protótipo

$$\boxed{m_j = \frac{1}{N_j} \sum_{x \in \omega_j} x}$$

Vetor médio de todas as amostras de treino da classe $j$.

---

## Distância Euclidiana

$$\boxed{\|a - b\| = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}}$$

Usada diretamente na classificação binária (Exp. iii) e no modo interativo.
A Função Discriminante (abaixo) é matematicamente equivalente e evita a raiz quadrada.

---

## Função Discriminante (Regra de Decisão)

$$\boxed{d_j(x) = x^T m_j - \frac{1}{2} m_j^T m_j}$$

**Regra:** Classificar $x$ na classe $j^*$ tal que:

$$\boxed{j^* = \arg\max_j \; d_j(x)}$$

**Equivalência:** Maximizar $d_j(x)$ = Minimizar $\|x - m_j\|^2$

**Derivação:**
$$\|x - m_j\|^2 = x^Tx - \underbrace{2x^Tm_j + m_j^Tm_j}_{\text{parte que depende de } j}$$

Como $x^Tx$ é constante → minimizar $-2x^Tm_j + m_j^Tm_j$ → maximizar $x^Tm_j - \tfrac{1}{2}m_j^Tm_j$

---

## Coeficientes da Fronteira de Decisão

Para a fronteira entre classes $i$ e $j$ (onde $d_i(x) = d_j(x)$):

$$\boxed{w = m_i - m_j}$$

$$\boxed{b = -\frac{1}{2}\left(\|m_i\|^2 - \|m_j\|^2\right) = -\frac{1}{2}(m_i^T m_i - m_j^T m_j)}$$

**Equação da fronteira:** $w^T x + b = 0$

---

## Reta da Fronteira em 2D (para plotar)

A partir de $w_1 x_1 + w_2 x_2 + b = 0$:

$$\boxed{x_2 = \frac{-w_1 x_1 - b}{w_2}}$$

---

## Produto Escalar

$$\boxed{a^T b = \sum_{i=1}^{n} a_i \cdot b_i}$$

Implementado como `produto_escalar(a, b)` com `zip` em Python puro.

---

## Métricas de Avaliação

### Acurácia

$$\boxed{\text{Acurácia} = \frac{\sum_j TP_j}{\text{Total de amostras}}}$$

### Precisão por Classe

$$\boxed{\text{Precisão}_j = \frac{TP_j}{TP_j + FP_j}}$$

> "Das que eu disse que eram $j$, quantas eram?"

### Revocação por Classe

$$\boxed{\text{Revocação}_j = \frac{TP_j}{TP_j + FN_j}}$$

> "Das que eram $j$, quantas eu encontrei?"

### F1-Score por Classe

$$\boxed{F1_j = \frac{2 \cdot \text{Precisão}_j \cdot \text{Revocação}_j}{\text{Precisão}_j + \text{Revocação}_j}}$$

### Macro-Média

$$\boxed{\overline{F1} = \frac{1}{K}\sum_{j=1}^{K} F1_j}$$

---

## Estrutura da Matriz de Confusão

```
              Predito →
         Setosa  Versicolor  Virginica
Real  Setosa    [ TP_s   FP_s→v   FP_s→vi ]
   ↓  Versicolor[ FN_v←s  TP_v   FP_v→vi ]
      Virginica [ FN_vi←s FN_vi←v  TP_vi  ]
```

- **Diagonal:** acertos (TP de cada classe)
- **Linha $i$, coluna $j \neq i$:** amostras reais de $i$ classificadas como $j$ (FN para $i$, contribui para FP de $j$)

---

## Separabilidade Linear

| Atributos | Classes separáveis? | Acurácia esperada |
|---|---|---|
| Pétalas [2,3] | Sim (perfeitamente) | 100% |
| Sépalas [0,1] | Parcialmente (Setosa sim; Versicolor/Virginica não) | ~80% |

---

## Parâmetros do Experimento

| Parâmetro | Valor |
|---|---|
| Total de amostras | 150 (50 por classe) |
| Proporção de treino | 70% (105 amostras) |
| Proporção de teste | 30% (45 amostras) |
| Semente aleatória | `random.seed(42)` |
| Atributos padrão | índices [2, 3] — pétalas |
| Classes | setosa, versicolor, virginica |

---

## Equivalências NumPy (versão futura)

| Python puro | NumPy |
|---|---|
| `produto_escalar(a, b)` | `np.dot(a, b)` |
| `distancia_euclidiana(a, b)` | `np.linalg.norm(a - b)` |
| `calcular_media(vetores)` | `np.mean(X, axis=0)` |
| `discriminante(x, mj)` | `np.dot(x, mj) - 0.5*np.dot(mj, mj)` |
| `treinar(...)` | `NearestCentroid().fit(X, y)` |

---

## Perceptron de Rosenblatt

### Vetor Aumentado (com Bias)

$$\boxed{x_\text{aug} = \begin{bmatrix} 1 \\ x_1 \\ x_2 \\ \vdots \\ x_n \end{bmatrix}, \quad w = \begin{bmatrix} w_0 \\ w_1 \\ w_2 \\ \vdots \\ w_n \end{bmatrix}}$$

### Ativação e Saída

$$\boxed{\text{net} = w^T x_\text{aug} = w_0 + w_1 x_1 + w_2 x_2 + \cdots + w_n x_n}$$

$$\boxed{y = \text{sgn}(\text{net}) = \begin{cases} +1 & \text{se net} \geq 0 \\ -1 & \text{se net} < 0 \end{cases}}$$

### Regra de Atualização (somente quando erra)

$$\boxed{w^{(t+1)} = w^{(t)} + p \cdot (d - y) \cdot x_\text{aug}}$$

- $d \in \{+1, -1\}$ — saída desejada
- $y = \text{sgn}(\text{net})$ — saída atual
- $(d - y) \in \{-2,\; 0,\; +2\}$; zero quando acerta (sem atualização)

### Limite do Número de Atualizações (Teorema da Convergência)

$$\boxed{t_{\max} \leq \left(\frac{R}{\gamma}\right)^2}$$

- $R = \max_k \|x_{\text{aug},k}\|$ — norma máxima das amostras
- $\gamma$ = margem de separação (distância do ponto mais próximo ao hiperplano ótimo)

Válido **somente para dados linearmente separáveis**.

---

## Regra Delta (Widrow-Hoff / Adaline)

### Função de Custo: MSE

$$\boxed{E(w) = \frac{1}{N} \sum_{k=1}^{N} (d_k - \text{net}_k)^2}$$

Onde $\text{net}_k = w^T x_{\text{aug},k}$ é a saída **linear** (sem limiarização).

### Regra de Atualização (modo online — atualiza a cada amostra)

$$\boxed{w \leftarrow w + p \cdot (d - \text{net}) \cdot x_\text{aug}}$$

- $d \in \{+1, -1\}$ — saída desejada
- $\text{net} = w^T x_\text{aug}$ — saída **linear** (antes de qualquer limiar)
- Atualiza para **todas as amostras**, não só as erradas

### Derivação (Gradiente Descendente)

$$\frac{\partial E}{\partial w_i} = \frac{-2}{N} \sum_k (d_k - \text{net}_k) \cdot x_{k,i}$$

$$w_i \leftarrow w_i - \eta \frac{\partial E}{\partial w_i} \quad \Longrightarrow \quad w_i \leftarrow w_i + \underbrace{p}_{\eta/N} (d - \text{net}) \, x_i$$

### Classificação Final

$$\boxed{\hat{y} = \text{sgn}(\text{net}) = \text{sgn}(w^T x_\text{aug})}$$

A Regra Delta **treina** com a saída linear mas **classifica** com o sinal da saída.

---

## Comparação: Perceptron vs. Regra Delta

| | Perceptron | Regra Delta |
|---|---|---|
| Erro utilizado | $\delta = d - \text{sgn}(\text{net})$ | $\delta = d - \text{net}$ |
| Quando atualiza | Somente se errar | Sempre |
| Converge para | Fronteira qualquer (se separável) | Mínimo de MSE |
| Dados sobrepostos | Oscila / não para | Converge ao mínimo |
| Função minimizada | Erros de classificação | $E = \frac{1}{N}\sum(d-\text{net})^2$ |

---

## Problema XOR

### Tabela Verdade

| $x_1$ | $x_2$ | $d$ |
|---|---|---|
| 0 | 0 | **0** |
| 0 | 1 | **1** |
| 1 | 0 | **1** |
| 1 | 1 | **0** |

### MSE Mínimo Teórico (Linear)

$$\boxed{E_{\min}^{\text{XOR}} = \frac{1}{4}\sum_{k=1}^{4}(d_k - \bar{d})^2 = 0{,}25}$$

Onde $\bar{d} = \frac{0+1+1+0}{4} = 0{,}5$ é a média dos alvos.

**Interpretação:** Nenhum classificador linear pode ter MSE menor que 0,25 no XOR. Se o MSE converge a ≈ 0,25, o modelo encontrou o melhor compromisso possível — mas não resolve o problema.

### Prova de Inseparabilidade

Supondo que existisse $w$ tal que:
- $(0,0) \to d=0$: $w_0 < 0$
- $(0,1) \to d=1$: $w_0 + w_2 \geq 0$
- $(1,0) \to d=1$: $w_0 + w_1 \geq 0$
- $(1,1) \to d=0$: $w_0 + w_1 + w_2 < 0$

Somando as duas desigualdades $\geq 0$: $2w_0 + w_1 + w_2 \geq 0$

Somando as duas $< 0$: $w_0 + (w_0 + w_1 + w_2) < 0 \Rightarrow 2w_0 + w_1 + w_2 < 0$

**Contradição** → sistema impossível → XOR não é linearmente separável. $\square$

---

## Parâmetros do Projeto

| Parâmetro | Valor |
|---|---|
| Total de amostras | 150 (50 por classe) |
| Proporção de treino | 70% (105 amostras) |
| Proporção de teste | 30% (45 amostras) |
| Semente aleatória | `random.seed(42)` |
| Atributos padrão | índices [2, 3] — pétalas |
| Classes | setosa, versicolor, virginica |
| Taxa de aprendizado Perceptron | $p = 0{,}03$ |
| Taxa de aprendizado Delta | $p = 0{,}02$ |
| Máx. épocas Perceptron | 100 |
| Máx. épocas Delta (Iris) | 200 |
| Máx. épocas Delta (XOR) | 300 |

---

## Bayes Ótimo e Naive Bayes

### Vetor de Médias (Centróide)

$$\boxed{m_j = \frac{1}{N_j} \sum_{x \in \omega_j} x}$$

### Matriz de Covariância Estimada

$$\boxed{\Sigma_j = \frac{1}{N_j - 1} \sum_{x \in \omega_j} (x - m_j)(x - m_j)^T}$$

### Covariância Regularizada (Ridge)

$$\boxed{\Sigma'_j = \Sigma_j + \epsilon \cdot I}$$

*(Evita matrizes singulares não-inversíveis, adicionando $\epsilon = 10^{-9}$ à diagonal principal)*

### Determinante de $\Sigma'_j$

$$\boxed{|\Sigma'_j| = \det(\Sigma'_j)}$$

### Distância de Mahalanobis Quadrada

$$\boxed{d_M^2(x, m_j) = (x - m_j)^T (\Sigma'_j)^{-1} (x - m_j)}$$

### Função Discriminante de Bayes Ótimo (QDA)

$$\boxed{d_j(x) = -\frac{1}{2} \ln |\Sigma'_j| - \frac{1}{2} (x - m_j)^T (\Sigma'_j)^{-1} (x - m_j)}$$

### Função Discriminante de Naive Bayes

$$\boxed{d_j(x) = -\frac{1}{2} \sum_{i=1}^d \ln(\sigma_{ji}^2) - \frac{1}{2} \sum_{i=1}^d \frac{(x_i - m_{ji})^2}{\sigma_{ji}^2}}$$

### Regra de Decisão MAP (Priors Iguais)

$$\boxed{\hat{y} = \arg\max_j \; d_j(x)}$$

### Teste de Significância de Kappa (Z-test)

$$\boxed{Z = \frac{K_1 - K_2}{\sqrt{\text{Var}(K_1) + \text{Var}(K_2)}}}$$

- $H_0$: $K_1 = K_2$ (desempenho idêntico).
- Rejeita-se $H_0$ se $|Z| > 1{,}96$ (nível de significância de $5\%$, p-valor $< 0{,}05$).

