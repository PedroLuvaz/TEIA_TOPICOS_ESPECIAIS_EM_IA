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

### Passo B: Função Discriminante
Para classificar uma nova amostra de teste $x$ (Experimento ii), não calculamos a distância euclidiana pura, mas sim a **Função Discriminante Linear**, que é matematicamente equivalente e computacionalmente mais eficiente, conforme ensinado em aula.

Fórmula exata implementada (`discriminante`):
$$ d_j(x) = x^T \cdot m_j - \frac{1}{2} m_j^T \cdot m_j $$

**Como decidimos?**
Calculamos $d_{setosa}(x)$, $d_{versicolor}(x)$ e $d_{virginica}(x)$. A classe escolhida é aquela que apresentar o **maior valor numérico** (Regra do Máximo). Isso é visível na tabela impressa no terminal.

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
