# Lab 3 — Item 2: Análise Comparativa dos Classificadores

**Dataset:** Iris · 150 amostras · Atributos de pétala [2,3] · Split 70/30 · Seed 42  
**Classificadores avaliados:** Distância Mínima, Distância Máxima, OvA Superfície, Perceptron OvA, Delta Bin. OvA, Delta OvA

---

## 2.1 Superfície de Decisão (OvA) vs Perceptron OvA

| Classificador    | Acerto Global | Kappa  | Tau    | Interpretação     |
|------------------|:-------------:|:------:|:------:|:-----------------:|
| OvA Superfície   | 100.00%       | 1.0000 | 1.0000 | Quase Perfeito    |
| Perceptron OvA   |  77.78%       | 0.6667 | 0.6667 | Substancial       |

### Existe diferença significativa?

Sim, e é expressiva. O classificador de superfície de decisão atingiu **K = 1.0**
(concordância perfeita), enquanto o Perceptron ficou em **K = 0.6667** (nível Substancial),
uma diferença de 0.3333 pontos no coeficiente Kappa.

### O que significa essa diferença?

O classificador de superfície de decisão utiliza os protótipos (vetores médios)
calculados analiticamente a partir dos dados de treino, o que para o dataset Iris
com os atributos de pétala é suficiente para uma separação perfeita das três classes.

O Perceptron, por sua vez, depende de convergência iterativa. Como as classes
*Versicolor* e *Virginica* **não são linearmente separáveis**, o algoritmo encerra
o treinamento ao atingir o limite de épocas sem encontrar uma fronteira ideal,
resultando em erros de classificação entre essas duas classes e redução no acerto global.

---

## 2.2 Perceptron OvA vs Regra Delta OvA

| Métrica          | Perceptron OvA | Delta OvA |
|------------------|:--------------:|:---------:|
| Acerto Global    | **77.78%**     | 66.67%    |
| Kappa (K)        | **0.666667**   | 0.500000  |
| Tau (τ)          | **0.666667**   | 0.500000  |
| Var(Kappa)       | 0.004609       | 0.000185  |
| Var(Tau)         | 0.008642       | 0.011111  |

### Teste de Significância

| Teste    | Z calculado | p-valor | Conclusão (α = 5%)       |
|----------|:-----------:|:-------:|:------------------------:|
| Z Kappa  | 2.4071      | 0.0161  | **Rejeita H₀**           |
| Z Tau    | 1.1859      | 0.2357  | Não rejeita H₀           |

> **H₀:** não há diferença entre os coeficientes  
> **H₁:** há diferença  
> Região crítica bilateral: |Z| > 1.96 para α = 0.05

### Qual classificador tem maior acurácia?

O **Perceptron OvA** apresentou maior acurácia (77.78% vs 66.67%) e maior Kappa
(0.6667 vs 0.5000).

### O teste de Kappa confirma essa diferença?

**Sim.** Com Z = 2.4071 e p = 0.0161 < 0.05, **rejeita-se H₀** pelo coeficiente
Kappa. A diferença de desempenho entre os dois classificadores é estatisticamente
significativa — o Perceptron OvA é genuinamente superior ao Delta OvA para este
conjunto de dados e atributos.

### O teste de Tau confirma?

**Não ao mesmo nível.** Com p = 0.2357 > 0.05, o Tau **não rejeita H₀**.
Isso ocorre porque o Tau assume distribuição uniforme entre as classes (1/C = 1/3),
hipótese mais restritiva. O Kappa, ao considerar a distribuição real das predições
no cálculo das variâncias, é mais sensível à diferença real entre os classificadores
neste cenário.

---

## Conclusão Geral

O **Perceptron OvA** obteve o maior acerto global (77.78%) e o maior Kappa (0.6667)
entre os classificadores baseados em aprendizado iterativo.

A diferença em relação ao Delta OvA é **estatisticamente significativa pelo teste
de Kappa** (p = 0.0161 < 0.05), mas **não pelo teste de Tau** (p = 0.2357 > 0.05).
Em situações de divergência entre os dois testes, o **Kappa é considerado mais
confiável** por não pressupor distribuição uniforme entre as classes.

O melhor classificador geral do experimento foi a **OvA Superfície** (Distância
Mínima), com acerto de 100% e K = 1.0, por utilizar fronteiras analíticas que
aproveitam plenamente a separabilidade linear dos atributos de pétala.

---

*Disciplina: Tópicos Especiais em Inteligência Artificial · UEPB · 2026*  
*Grupo: Erick Nathan · Laura Barbosa · Pedro Lucas*