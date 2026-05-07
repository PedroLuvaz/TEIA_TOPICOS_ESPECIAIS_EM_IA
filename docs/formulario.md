# Formulário — Classificador de Distância Mínima

> Referência rápida para consulta durante a prova.

---

## Notação

| Símbolo | Significado |
|---|---|
| $x$ | Vetor de features de uma amostra (ex: `[comp_petala, larg_petala]`) |
| $m_j$ | Protótipo (vetor médio) da classe $j$ |
| $N_j$ | Número de amostras de treino da classe $j$ |
| $\omega_j$ | Conjunto de amostras da classe $j$ |
| $K$ | Número de classes |
| $w$ | Vetor de pesos da fronteira de decisão |
| $b$ | Bias (constante) da fronteira de decisão |
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
