# Guia de Explicação para o Professor

Este documento serve como um guia para apresentar e explicar o projeto. Ele detalha a lógica implementada, as escolhas arquiteturais e a matemática por trás do classificador.

---

## 1. Introdução e Restrições

**Ponto Chave de Apresentação:** Destaque que o projeto foi construído "do zero" (from scratch).
- **Sem bibliotecas de Machine Learning:** Não usamos Scikit-Learn, NumPy ou SciPy. Toda a álgebra linear (produto escalar, distâncias, médias) foi implementada com laços e listas nativas do Python no arquivo `math_utils.py`.
- **Por que isso é importante?** Mostra domínio sobre os conceitos matemáticos ensinados em sala, provando que o algoritmo não é uma "caixa preta".

---

## 2. Preparação dos Dados (Split Estratificado)

**Onde encontrar no código:** `data_loader.py`

**Explicação:**
Se pegássemos os dados e dividíssemos simplesmente os primeiros 70% para treino e os 30% finais para teste, correríamos o risco de deixar classes inteiras de fora do treinamento, já que o dataset original é ordenado por classe (50 amostras seguidas de cada).

Para resolver isso, implementamos o **Split Estratificado**:
1. Agrupamos os dados por classe.
2. Embaralhamos (shuffle) as amostras dentro de cada classe (usando uma semente fixa `random.seed(42)` para reprodutibilidade).
3. Pegamos 70% (35 amostras) **de cada classe** para treino.
4. Pegamos os 30% restantes (15 amostras) **de cada classe** para teste.

Isso garante que o modelo aprenda de forma justa sobre todas as espécies de íris.

---

## 3. A Matemática: Classificador de Distância Mínima

**Onde encontrar no código:** `math_utils.py` e `classifier.py`

O Classificador de Distância Mínima assume que cada classe pode ser representada por um único ponto no espaço: o seu **protótipo**.

### Passo A: Cálculo dos Protótipos (Vetores Médios)
Durante o treinamento (Experimento i), calculamos o vetor médio $m_j$ para cada classe $j$.

Fórmula:
$$ m_j = \frac{1}{N_j} \sum_{x \in \omega_j} x $$

No código (`calcular_media`), simplesmente somamos os valores de todas as características (Comprimento e Largura da Pétala) das amostras de treino de uma classe e dividimos pela quantidade de amostras.

### Passo B: Classificação por Distância Euclidiana

Para classificar uma nova amostra de teste $x$ (Experimento ii), calculamos a **distância euclidiana** de $x$ até cada protótipo e escolhemos o mais próximo:

$$\|x - m_j\| = \sqrt{\sum_{k} (x_k - m_{j,k})^2}$$

**Como decidimos?**
Calculamos $\|x - m_{setosa}\|$, $\|x - m_{versicolor}\|$ e $\|x - m_{virginica}\|$. A classe escolhida é aquela com a **menor distância** (Regra do Mínimo). Isso é visível na tabela impressa no terminal — a coluna `dist_*` com menor valor define a predição.

> **Nota técnica:** Existe uma forma equivalente chamada **Função Discriminante**, $d_j(x) = x^T m_j - \frac{1}{2}\|m_j\|^2$, que produz a mesma classificação sem calcular a raiz quadrada. No código, usamos a distância euclidiana diretamente (`distancia_euclidiana` em `math_utils.py`) para manter clareza conceitual. A função discriminante permanece em `math_utils.py` para o cálculo das superfícies de decisão.

---

## 4. Superfícies de Decisão (Pares de Classes)

**Onde encontrar no código:** Experimento iii no `main.py` e em `visualizer.py`

Para desenhar as retas que separam as classes nos gráficos, treinamos classificadores binários (duas classes por vez).

A fronteira de decisão entre a classe $i$ e a classe $j$ ocorre exatamente onde as funções discriminantes se igualam:
$$ d_{ij}(x) = d_i(x) - d_j(x) = 0 $$

Substituindo a fórmula da função discriminante, chegamos aos coeficientes da reta ($w \cdot x + b = 0$):

1. **Vetor de Pesos ($w$):**
   $$ w = m_i - m_j $$
   *(A diferença entre os dois protótipos)*

2. **Viés / Constante ($b$):**
   $$ b = -\frac{1}{2} (m_i^T \cdot m_i - m_j^T \cdot m_j) $$

No código (`coeficientes_superficie_decisao`), esses dois valores são calculados.
No `visualizer.py`, isolamos o $x_2$ na equação da reta para poder plotá-la no plano cartesiano 2D:
$$ x_2 = \frac{-w_1 \cdot x_1 - b}{w_2} $$

---

## 5. Resultados e Conclusão

- **Atributos Utilizados:** O código está configurado para usar os índices `[2, 3]`, que correspondem ao **Comprimento da Pétala** e **Largura da Pétala**.
- **Acurácia:** O modelo atinge **100% de acurácia**.
- **Por quê?** Como visível nos gráficos gerados na pasta `outputs/`, o conjunto de dados Iris, quando analisado apenas pelas pétalas, é **linearmente separável** (especialmente a Setosa em relação às outras). O Classificador de Distância Mínima é um classificador linear perfeito para este cenário, posicionando a fronteira de decisão de forma equidistante entre os protótipos.

---

## 6. Experimento Comparativo: Sépalas vs Pétalas

**Onde encontrar no código:** última seção do `main.py` — "EXPERIMENTO COMPARATIVO"

Este experimento demonstra que a **escolha dos atributos (features)** impacta diretamente a capacidade de separação linear do classificador.

| Atributos | Índices | Acurácia Esperada |
|---|---|---|
| Comprimento + Largura da **Pétala** | [2, 3] | **100%** |
| Comprimento + Largura da **Sépala** | [0, 1] | **~82%** |

**Por que a diferença?**
- As pétalas de Setosa são muito menores que as das outras duas classes — separação perfeita.
- As sépalas de Versicolor e Virginica se **sobrepõem** consideravelmente no espaço de features: nenhuma reta consegue separá-las perfeitamente.
- Um classificador de distância mínima é um **classificador linear** — sua fronteira é sempre uma reta (em 2D) ou hiperplano. Ele só funciona com 100% de acurácia quando os dados são **linearmente separáveis**.

**Como apresentar ao professor:** Este experimento prova que o aluno entende a relação entre separabilidade linear, escolha de features e a limitação do classificador implementado — não apenas que ele "rodou e deu 100%".

---

## 7. Estrutura Modular e Responsabilidade de Cada Arquivo

| Arquivo | Responsabilidade | Matemática central |
|---|---|---|
| `math_utils.py` | Toda a álgebra linear em Python puro | `produto_escalar`, `distancia_euclidiana`, `discriminante`, `coeficientes_superficie_decisao` |
| `data_loader.py` | Leitura do XLS + split estratificado | Agrupamento por classe, shuffle com `seed=42` |
| `classifier.py` | Treinamento e predição | `treinar` → protótipos; `predizer_todas_classes` → argmax $d_j(x)$; `predizer_binario` → argmin distância |
| `evaluator.py` | Métrica de avaliação | Acurácia |
| `visualizer.py` | Gráficos matplotlib | Dispersão, superfícies de decisão, heatmap de confusão |
| `main.py` | Orquestrador | Executa experimentos i, ii, iii + comparativo + interativo |

**Ponto importante:** `math_utils.py` não conhece nada de Iris — é uma biblioteca de álgebra genérica. Isso é uma boa prática de separação de responsabilidades.

---

## 8. Futura Migração para Bibliotecas (NumPy / Scikit-learn)

O projeto foi intencionalmente construído sem bibliotecas de ML para demonstrar domínio matemático. Em uma versão futura, cada função pura tem um equivalente direto:

| Função Pura (`math_utils.py`) | Equivalente NumPy/Sklearn |
|---|---|
| `produto_escalar(a, b)` | `np.dot(a, b)` |
| `distancia_euclidiana(a, b)` | `np.linalg.norm(np.array(a) - np.array(b))` |
| `calcular_media(vetores)` | `np.mean(X, axis=0)` |
| `discriminante(x, mj)` | `np.dot(x, mj) - 0.5 * np.dot(mj, mj)` |
| `treinar(dados, indices)` | `sklearn.neighbors.NearestCentroid().fit(X, y)` |
| `predizer_todas_classes(...)` | `NearestCentroid().predict(x)` |

**Como apresentar ao professor:** "A estrutura modular foi pensada para facilitar a migração. O `math_utils.py` pode ser trocado por NumPy sem mudar nada no restante do código — basta substituir as funções uma a uma."

A mesma separação de responsabilidades (`data_loader → classifier → evaluator → visualizer`) continuará válida mesmo com bibliotecas.
