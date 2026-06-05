# Lab 3 — Item 3: Exercícios da Aula PR51

**Referência:** Slide 15 da Aula PR51 — Prof. Robson Pequeno de Sousa  
**Tarefa:** Calcular Kappa, Tau, variâncias e teste de significância para dois classificadores (A e B) com 4 classes.

---

## Matrizes de Confusão

### Classificação A

|         | w1 (pred) | w2 (pred) | w3 (pred) | w4 (pred) | **Total Real** |
|---------|:---------:|:---------:|:---------:|:---------:|:--------------:|
| **w1**  | 140       | 20        | 0         | 0         | 160            |
| **w2**  | 10        | 130       | 0         | 0         | 140            |
| **w3**  | 5         | 0         | 150       | 10        | 165            |
| **w4**  | 15        | 10        | 0         | 120       | 145            |
| **Total Pred** | 170 | 160      | 150       | 130       | **610**        |

### Classificação B

|         | w1 (pred) | w2 (pred) | w3 (pred) | w4 (pred) | **Total Real** |
|---------|:---------:|:---------:|:---------:|:---------:|:--------------:|
| **w1**  | 140       | 30        | 2         | 0         | 172            |
| **w2**  | 10        | 110       | 5         | 0         | 125            |
| **w3**  | 0         | 10        | 140       | 0         | 150            |
| **w4**  | 20        | 10        | 3         | 140       | 173            |
| **Total Pred** | 170 | 160      | 150       | 140       | **620**        |

---

## Cálculo do Acerto Global

$$A_g = \frac{1}{m} \sum_{i=1}^{c} a_{ii}$$

**Classificação A:** $A_g = \frac{140+130+150+120}{610} = \frac{540}{610} = 0.885246$ **(88.52%)**

**Classificação B:** $A_g = \frac{140+110+140+140}{620} = \frac{530}{620} = 0.854839$ **(85.48%)**

---

## Acurácia do Produtor e do Usuário

$$A_{pi} = \frac{a_{ii}}{a_{+i}} \quad \text{(Produtor — sensibilidade)}
\qquad
A_{ui} = \frac{a_{ii}}{a_{i+}} \quad \text{(Usuário — precisão)}$$

### Classificação A

| Classe | VP  | Total Real (linha) | Total Pred (coluna) | Ac. Produtor | Ac. Usuário |
|--------|:---:|:-----------:|:-----------:|:------------:|:-----------:|
| w1     | 140 | 160         | 170         | 140/170 = **82.35%** | 140/160 = **87.50%** |
| w2     | 130 | 140         | 160         | 130/160 = **81.25%** | 130/140 = **92.86%** |
| w3     | 150 | 165         | 150         | 150/150 = **100.00%** | 150/165 = **90.91%** |
| w4     | 120 | 145         | 130         | 120/130 = **92.31%** | 120/145 = **82.76%** |

### Classificação B

| Classe | VP  | Total Real (linha) | Total Pred (coluna) | Ac. Produtor | Ac. Usuário |
|--------|:---:|:-----------:|:-----------:|:------------:|:-----------:|
| w1     | 140 | 172         | 170         | 140/170 = **82.35%** | 140/172 = **81.40%** |
| w2     | 110 | 125         | 160         | 110/160 = **68.75%** | 110/125 = **88.00%** |
| w3     | 140 | 150         | 150         | 140/150 = **93.33%** | 140/150 = **93.33%** |
| w4     | 140 | 173         | 140         | 140/140 = **100.00%** | 140/173 = **80.92%** |

---

## Coeficiente Kappa

$$K = \frac{A_g - A_a}{1 - A_a}
\qquad
A_a = \frac{\sum_{i=1}^{c} a_{i+} \cdot a_{+i}}{m^2}$$

### Classificação A

$$A_a = \frac{(160 \times 170) + (140 \times 160) + (165 \times 150) + (145 \times 130)}{610^2}$$
$$= \frac{27200 + 22400 + 24750 + 18850}{372100} = \frac{93200}{372100} = 0.250470$$

$$K_A = \frac{0.885246 - 0.250470}{1 - 0.250470} = \frac{0.634776}{0.749530} = \mathbf{0.846899}$$

> **Interpretação: Substancial** (0.61 < K ≤ 0.80 → Substancial; K > 0.80 → Quase Perfeito)  
> K = 0.8469 está acima de 0.80 → **Quase Perfeito**

### Classificação B

$$A_a = \frac{(172 \times 170) + (125 \times 160) + (150 \times 150) + (173 \times 140)}{620^2}$$
$$= \frac{29240 + 20000 + 22500 + 24220}{384400} = \frac{95960}{384400} = 0.249636$$

$$K_B = \frac{0.854839 - 0.249636}{1 - 0.249636} = \frac{0.605203}{0.750364} = \mathbf{0.806546}$$

> **Interpretação: Quase Perfeito** (K > 0.80)

---

## Coeficiente Tau

$$\tau = \frac{A_g - \frac{1}{C}}{1 - \frac{1}{C}} \quad \text{com } C = 4 \text{ classes} \Rightarrow \frac{1}{C} = 0.25$$

**Classificação A:**
$$\tau_A = \frac{0.885246 - 0.25}{1 - 0.25} = \frac{0.635246}{0.75} = \mathbf{0.846995}$$

**Classificação B:**
$$\tau_B = \frac{0.854839 - 0.25}{1 - 0.25} = \frac{0.604839}{0.75} = \mathbf{0.806452}$$

---

## Variância dos Coeficientes

### Variância do Kappa (Congalton & Green, 2009)

$$\sigma^2_k = \frac{1}{m}\left(\frac{\phi_1(1-\phi_1)}{(1-\phi_2)^2} + \frac{2(1-\phi_1)(2\phi_1\phi_2-\phi_3)}{(1-\phi_2)^3} + \frac{(1-\phi_1)^2(\phi_4-4\phi_2^2)}{(1-\phi_2)^4}\right)$$

| Coeficiente | Classificação A | Classificação B |
|-------------|:--------------:|:--------------:|
| $\sigma^2_k$ | **0.00027827**  | **0.00032598**  |

### Variância do Tau

$$\sigma^2_\tau = \frac{1}{m} \cdot \frac{A_g(1-A_g)}{\left(1-\frac{1}{C}\right)^2}$$

**Classificação A:**
$$\sigma^2_{\tau_A} = \frac{1}{610} \cdot \frac{0.885246 \times 0.114754}{0.75^2} = \frac{0.101567}{0.5625 \times 610} = \mathbf{0.00029606}$$

**Classificação B:**
$$\sigma^2_{\tau_B} = \frac{1}{620} \cdot \frac{0.854839 \times 0.145161}{0.75^2} = \mathbf{0.00035581}$$

---

## Teste de Significância entre A e B

$$Z_k = \frac{k_1 - k_2}{\sqrt{\sigma^2_{k_1} + \sigma^2_{k_2}}} \to N(0,1)
\qquad
Z_\tau = \frac{\tau_1 - \tau_2}{\sqrt{\sigma^2_{\tau_1} + \sigma^2_{\tau_2}}} \to N(0,1)$$

> **H₀:** não há diferença entre os coeficientes  
> **H₁:** há diferença  
> Região crítica bilateral: **|Z| > 1.96** para α = 0.05

### Teste Z — Kappa

$$Z_k = \frac{0.846899 - 0.806546}{\sqrt{0.00027827 + 0.00032598}} = \frac{0.040353}{\sqrt{0.00060425}} = \frac{0.040353}{0.024582} = \mathbf{1.6416}$$

$$p\text{-valor} = P(|Z| > 1.6416) = \mathbf{0.1007}$$

### Teste Z — Tau

$$Z_\tau = \frac{0.846995 - 0.806452}{\sqrt{0.00029606 + 0.00035581}} = \frac{0.040543}{\sqrt{0.00065187}} = \frac{0.040543}{0.025531} = \mathbf{1.5879}$$

$$p\text{-valor} = P(|Z| > 1.5879) = \mathbf{0.1123}$$

---

## Resultados Consolidados

| Métrica           | Classificação A | Classificação B |
|-------------------|:--------------:|:--------------:|
| Total de amostras | 610             | 620             |
| Acerto Global     | 88.52%          | 85.48%          |
| Kappa (K)         | 0.846899        | 0.806546        |
| Interpretação K   | Quase Perfeito  | Quase Perfeito  |
| Tau (τ)           | 0.846995        | 0.806452        |
| Var(Kappa)        | 0.00027827      | 0.00032598      |
| Var(Tau)          | 0.00029606      | 0.00035581      |
| Z Kappa (A vs B)  | **1.6416**      | —               |
| p-valor Kappa     | **0.1007**      | —               |
| Z Tau   (A vs B)  | **1.5879**      | —               |
| p-valor Tau       | **0.1123**      | —               |

---

## Conclusão

Como $|Z_k| = 1.6416 < 1.96$ e $p = 0.1007 > 0.05$, **não se rejeita H₀** pelo
coeficiente Kappa. Da mesma forma, $|Z_\tau| = 1.5879 < 1.96$ e $p = 0.1123 > 0.05$,
**não se rejeita H₀** pelo coeficiente Tau.

Ambos os testes concordam: **não há evidência estatística suficiente para afirmar
que os classificadores A e B são diferentes** ao nível de significância de 5%.
Apesar do Classificador A apresentar Kappa ligeiramente superior (0.8469 vs 0.8065),
essa diferença pode ser atribuída à variação amostral — não é estatisticamente
significativa.

---

*Disciplina: Tópicos Especiais em Inteligência Artificial · UEPB · 2026*  
*Grupo: Erick Nathan · Laura Barbosa · Pedro Lucas*