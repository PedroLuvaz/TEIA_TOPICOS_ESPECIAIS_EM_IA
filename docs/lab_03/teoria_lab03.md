# Lab 3 — Teoria Completa: Métricas Avançadas de Qualidade de Classificadores

**Referência:** Aula PR51 — Prof. Robson Pequeno de Sousa
**Implementação:** `iris_classifier/metricas_avancadas.py` (Python puro, sem numpy/scipy/sklearn)
**Interface:** Aba 3 — *Métricas Avançadas* da GUI (`iris_classifier/gui/tab_metricas_avancadas.py`)

---

## 1. Por que a acurácia simples não basta?

A acurácia (acerto global) responde apenas *"quantos acertei no total"*. Ela esconde
dois problemas:

1. **Acerto por acaso** — um classificador aleatório em um problema de 2 classes
   balanceadas já acerta ~50%. Dizer "acertei 60%" sem descontar o acaso superestima
   a qualidade real.
2. **Classes desbalanceadas** — se 95% das amostras são da classe A, um classificador
   que responde sempre "A" tem 95% de acurácia e zero utilidade.

As métricas do Lab 3 (Kappa, Tau, MCC, Fb) existem exatamente para **descontar o
acaso** e **enxergar o desempenho por classe**.

---

## 2. Matriz de Confusão

Toda métrica deste laboratório é extraída da matriz de confusão `a_ij`, onde cada
célula conta quantas amostras da classe real *i* foram preditas como classe *j*
(convenção do slide: **linha = real, coluna = predito**; na GUI a matriz é exibida
com linha = predito, coluna = real — as fórmulas se ajustam à convenção usada).

Notação:

- `m` — total de amostras
- `a_ii` — diagonal principal (acertos da classe i)
- `a_i+` — soma da linha i
- `a_+i` — soma da coluna i
- `C` — número de classes

---

## 3. Acerto Global (Ag)

```
Ag = (1/m) · Σ a_ii
```

Soma da diagonal dividida pelo total. É a acurácia tradicional.
**Limitação:** não desconta o acaso nem distingue classes.

---

## 4. Acurácia do Produtor e do Usuário

Métricas **por classe**, vindas do sensoriamento remoto (Congalton & Green):

| Métrica | Fórmula | Pergunta que responde | Equivalente |
|---|---|---|---|
| **Acurácia do Produtor** | `A_pi = a_ii / a_+i` | "Das amostras que *realmente são* da classe i, quantas o classificador capturou?" | Sensibilidade / Recall |
| **Acurácia do Usuário** | `A_ui = a_ii / a_i+` | "Das amostras que o classificador *disse ser* da classe i, quantas eram mesmo?" | Precisão / VPP |

- Produtor baixo → o classificador **deixa escapar** amostras da classe (muitos FN).
- Usuário baixo → o classificador **inventa** amostras da classe (muitos FP).

---

## 5. Coeficiente Kappa (K)

```
K = (Ag − Aa) / (1 − Aa)        com        Aa = Σ (a_i+ · a_+i) / m²
```

`Aa` é o **acerto casual**: a probabilidade de concordância esperada por puro acaso,
estimada a partir das distribuições marginais (totais de linha × totais de coluna).
O Kappa mede, portanto, **quanto o classificador acerta além do acaso**, normalizado
pelo máximo possível de melhora.

- `K = 1` → concordância perfeita
- `K = 0` → desempenho igual ao acaso
- `K < 0` → pior que o acaso

### Interpretação (Landis & Koch, 1977)

| Faixa de K | Interpretação |
|:---:|:---:|
| ≤ 0.20 | Fraco |
| 0.21 – 0.40 | Razoável |
| 0.41 – 0.60 | Moderado |
| 0.61 – 0.80 | Substancial |
| > 0.80 | Quase Perfeito |

### Variância do Kappa (Congalton & Green, 2009)

Necessária para o teste de significância:

```
σ²_K = (1/m) · [ φ₁(1−φ₁)/(1−φ₂)²
               + 2(1−φ₁)(2φ₁φ₂−φ₃)/(1−φ₂)³
               + (1−φ₁)²(φ₄−4φ₂²)/(1−φ₂)⁴ ]
```

onde `φ₁ = Ag`, `φ₂ = Aa`, `φ₃ = (1/m²)·Σ a_ii(a_i+ + a_+i)` e
`φ₄ = (1/m³)·Σ_i Σ_j a_ij(a_j+ + a_+i)`.

---

## 6. Coeficiente Tau (τ)

```
τ = (Ag − 1/C) / (1 − 1/C)
```

Mesma ideia do Kappa, mas o acerto casual é fixado em `1/C` (classes
**equiprováveis**), em vez de estimado pelas marginais.

```
σ²_τ = (1/m) · Ag(1−Ag) / (1 − 1/C)²
```

**Kappa vs Tau:** o Tau assume distribuição uniforme entre classes — hipótese mais
restritiva. O Kappa usa a distribuição real das predições, por isso é considerado
**mais confiável** quando os dois divergem.

---

## 7. Teste de Significância entre dois classificadores (Z)

Comparar dois classificadores apenas pelo valor de K ou τ não basta: a diferença
pode ser fruto da variação amostral. O teste Z verifica se ela é **estatisticamente
significativa**:

```
Z_K = (K₁ − K₂) / √(σ²_K1 + σ²_K2)          Z_τ = (τ₁ − τ₂) / √(σ²_τ1 + σ²_τ2)
```

- **H₀:** não há diferença entre os coeficientes
- **H₁:** há diferença
- **Região crítica bilateral:** `|Z| > 1.96` para α = 5% (Z ~ N(0,1))
- **p-valor:** `p = P(|Z| > z) = 2·(1 − Φ(|z|))` — calculado em Python puro pela
  aproximação de Abramowitz & Stegun (26.2.17) da CDF normal.

Se `|Z| > 1.96` (equivalente a `p < 0.05`), rejeita-se H₀: os classificadores são
genuinamente diferentes.

---

## 8. Métricas binárias (problema de 2 classes)

Para um par de classes (ou visão One-vs-Rest), extraem-se VP, FP, FN, VN:

| Métrica | Fórmula |
|---|---|
| Sensibilidade (Recall) | `VP / (VP + FN)` |
| Especificidade | `VN / (VN + FP)` |
| Precisão (VPP) | `VP / (VP + FP)` |
| **Fb Score** | `Fb = (1+b²)·(Pr·Re) / (b²·Pr + Re)` |
| **MCC (Matthews)** | `(VP·VN − FP·FN) / √((VP+FP)(VP+FN)(VN+FP)(VN+FN))` |

- **F1 (b=1):** equilíbrio entre precisão e recall.
- **F2 (b=2):** dá mais peso à revocação — preferido quando perder um positivo
  custa caro (ex.: rastreamento, diagnóstico).
- **MCC ∈ [−1, 1]:** robusto a desbalanceamento; 0 = aleatório, 1 = perfeito.

---

## 9. Resultados do Lab 3 no Iris

Resumo dos experimentos (detalhes em `item_02.md` e `item_03.md`):

### Item 2 — Comparação dos classificadores (pétalas, split 70/30, seed 42)

- **OvA Superfície (Distância Mínima):** Ag = 100%, K = 1.0 — as pétalas são
  linearmente separáveis e os protótipos analíticos aproveitam isso plenamente.
- **Perceptron OvA:** Ag = 77.78%, K = 0.6667 — Versicolor × Virginica não são
  linearmente separáveis, o treinamento iterativo esbarra no limite de épocas.
- **Perceptron vs Delta OvA:** Z_K = 2.4071 (p = 0.0161) → **rejeita H₀** pelo
  Kappa; Z_τ = 1.1859 (p = 0.2357) → não rejeita pelo Tau. Prevalece o Kappa:
  o Perceptron é genuinamente superior neste cenário.

### Item 3 — Exercício do Slide 15 (matrizes A e B, 4 classes)

- K_A = 0.8469 e K_B = 0.8065 — ambos "Quase Perfeito".
- Z_K = 1.6416 (p = 0.1007) e Z_τ = 1.5879 (p = 0.1123) → **não se rejeita H₀**:
  a vantagem de A é atribuível à variação amostral.
- Reproduzível na GUI: Aba 3 → sub-aba **Exercícios PR51** (matrizes editáveis,
  cálculo ao vivo em Python puro).

---

*Disciplina: Tópicos Especiais em Inteligência Artificial · UEPB · 2026*
*Grupo: Erick Nathan · Laura Barbosa · Pedro Lucas*
