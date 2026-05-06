# Teoria Completa — Classificador de Distância Mínima

> Material de estudo detalhado para a prova. Cobre toda a teoria implementada no projeto.

---

## 1. Reconhecimento de Padrões — Visão Geral

**Reconhecimento de Padrões** é a área que estuda como máquinas aprendem a identificar categorias a partir de dados. O pipeline básico é:

```
Dados Brutos → Extração de Features → Treinamento → Classificação → Avaliação
```

- **Feature (Atributo):** Uma característica mensurável de uma amostra. No Iris: comprimento da pétala, largura da sépala, etc.
- **Classe (Rótulo):** A categoria da amostra. No Iris: Setosa, Versicolor, Virginica.
- **Modelo:** A função matemática que mapeia features → classes.
- **Treinamento:** Ajuste dos parâmetros do modelo usando amostras rotuladas.

---

## 2. A Base de Dados Iris

Proposta por **Ronald Fisher em 1936** em seu artigo _"The use of multiple measurements in taxonomic problems"_. É um dos datasets mais famosos em ML.

| Característica | Valor |
|---|---|
| Total de amostras | 150 |
| Amostras por classe | 50 (balanceado) |
| Número de features | 4 |
| Número de classes | 3 |

**As 4 features (atributos):**
- Índice 0: Comprimento da Sépala (cm)
- Índice 1: Largura da Sépala (cm)
- Índice 2: **Comprimento da Pétala (cm)** ← mais discriminante
- Índice 3: **Largura da Pétala (cm)** ← mais discriminante

**As 3 classes:**
- `setosa` — pétalas muito pequenas, completamente separável
- `versicolor` — pétalas médias
- `virginica` — pétalas maiores

**Fato importante:** Usando apenas as pétalas (índices [2,3]), as 3 classes são **linearmente separáveis**. Usando sépalas (índices [0,1]), versicolor e virginica se sobrepõem.

---

## 3. Preparação dos Dados: Split Estratificado

### Por que não um split simples?

O dataset Iris está ordenado: as 50 primeiras amostras são Setosa, as 50 seguintes Versicolor, as 50 últimas Virginica. Se pegarmos os primeiros 70% diretamente, o conjunto de teste poderia não conter Setosa nem parte de Versicolor.

### Split Estratificado

Garante que a **proporção de classes seja mantida** em treino e teste.

**Algoritmo implementado (`data_loader.py`):**
1. Separar amostras por classe
2. Embaralhar cada grupo com `random.seed(42)` (reprodutibilidade)
3. Pegar os primeiros 70% de cada grupo para treino
4. Os 30% restantes de cada grupo para teste

**Resultado com 150 amostras (50 por classe):**
- Treino: 35 por classe × 3 = **105 amostras**
- Teste: 15 por classe × 3 = **45 amostras**

### Por que `random.seed(42)`?

Garante que qualquer pessoa que execute o código obtenha exatamente os mesmos resultados. É uma boa prática de reproducibilidade científica.

---

## 4. Classificador de Distância Mínima

### Intuição

> "Classifique uma nova amostra como pertencente à classe cujo **protótipo** está mais próximo dela."

É o classificador mais simples baseado em protótipos. Assume que cada classe pode ser resumida por um único ponto representativo no espaço de features.

### Passo 1: Calcular os Protótipos (Treinamento)

O **protótipo** (ou centróide) de uma classe é o vetor médio de todas as suas amostras de treinamento.

$$m_j = \frac{1}{N_j} \sum_{x \in \omega_j} x$$

Onde:
- $m_j$ = protótipo da classe $j$
- $N_j$ = número de amostras de treino da classe $j$
- $\omega_j$ = conjunto de amostras da classe $j$
- $x$ = vetor de features de uma amostra

**Exemplo numérico (pétalas, 2D):**
- Setosa: média de [1.4, 0.3], [1.4, 0.2], ... → $m_{setosa} \approx [1.48, 0.24]$
- Versicolor: média de [4.7, 1.4], ... → $m_{versicolor} \approx [4.26, 1.33]$
- Virginica: média de [6.0, 2.5], ... → $m_{virginica} \approx [5.55, 2.03]$

**No código:** `calcular_media()` em `math_utils.py`

### Passo 2: Classificar (Predição)

Para classificar uma nova amostra $x$, calculamos a distância euclidiana até cada protótipo e escolhemos o mais próximo:

$$\hat{y} = \arg\min_j \|x - m_j\|$$

**No entanto**, o projeto usa a **Função Discriminante Linear** — matematicamente equivalente, mas mais eficiente computacionalmente.

---

## 5. Função Discriminante Linear

### Derivação Completa

Queremos minimizar $\|x - m_j\|^2$. Expandindo:

$$\|x - m_j\|^2 = (x - m_j)^T(x - m_j)$$
$$= x^T x - 2x^T m_j + m_j^T m_j$$

Como $x^T x$ é constante para todos os $j$ (não depende da classe), **minimizar** $\|x - m_j\|^2$ equivale a **maximizar**:

$$d_j(x) = x^T m_j - \frac{1}{2} m_j^T m_j$$

> Isso é a **Função Discriminante do Classificador de Distância Mínima**.

### Por que "linear"?

Porque $d_j(x)$ é uma função **linear** em $x$: é um produto escalar (combinação linear) mais uma constante.

### Regra de Decisão (Regra do Máximo)

$$\hat{y} = \arg\max_j \; d_j(x)$$

Calculamos $d_{setosa}(x)$, $d_{versicolor}(x)$, $d_{virginica}(x)$ e escolhemos a classe com maior valor.

**No código:** `discriminante(x, mj)` em `math_utils.py`, `predizer_todas_classes()` em `classifier.py`

### Verificação intuitiva

Se $x$ está mais próximo de $m_i$ do que de $m_j$, então:
$$\|x - m_i\|^2 < \|x - m_j\|^2$$
$$\Rightarrow -2x^T m_i + m_i^T m_i < -2x^T m_j + m_j^T m_j$$
$$\Rightarrow x^T m_i - \frac{1}{2}m_i^T m_i > x^T m_j - \frac{1}{2}m_j^T m_j$$
$$\Rightarrow d_i(x) > d_j(x) \checkmark$$

---

## 6. Superfície de Decisão

### O que é?

A superfície de decisão entre duas classes $i$ e $j$ é o conjunto de pontos onde o classificador está "em dúvida" — onde as duas funções discriminantes são iguais:

$$d_{ij}(x) = d_i(x) - d_j(x) = 0$$

### Derivação dos Coeficientes

Substituindo a definição de $d_j(x)$:

$$d_i(x) - d_j(x) = 0$$
$$\left(x^T m_i - \frac{1}{2}m_i^T m_i\right) - \left(x^T m_j - \frac{1}{2}m_j^T m_j\right) = 0$$
$$(m_i - m_j)^T x - \frac{1}{2}(m_i^T m_i - m_j^T m_j) = 0$$

Portanto:

$$\underbrace{w^T x + b}_{=0} \quad \text{onde:}$$

$$\boxed{w = m_i - m_j}$$

$$\boxed{b = -\frac{1}{2}\left(\|m_i\|^2 - \|m_j\|^2\right)}$$

### Interpretação Geométrica

- $w = m_i - m_j$ é o vetor que aponta de $m_j$ para $m_i$
- A superfície de decisão é um **hiperplano perpendicular** a esse vetor
- Em 2D (pétalas), é uma **reta**
- A reta passa pelo **ponto médio** entre os dois protótipos
- É equidistante de $m_i$ e $m_j$

### Plotar a Reta em 2D

Para plotar, isolamos $x_2$:
$$w_1 x_1 + w_2 x_2 + b = 0$$
$$x_2 = \frac{-w_1 x_1 - b}{w_2}$$

**No código:** `coeficientes_superficie_decisao()` em `math_utils.py`, `plotar_superficie_decisao()` em `visualizer.py`

---

## 7. Separabilidade Linear

### Definição

Um conjunto de dados é **linearmente separável** se existe um hiperplano que separa perfeitamente as classes sem nenhum erro.

- Em 2D: uma reta separa as nuvens de pontos
- Em 3D: um plano separa
- Em $n$D: um hiperplano de dimensão $n-1$ separa

### Iris com Pétalas vs Sépalas

**Pétalas (índices [2,3]) → Linearmente separável:**
- Setosa: pétalas muito pequenas (comp. < 2cm), completamente à parte
- Versicolor vs Virginica: há uma fronteira clara, embora próxima

**Sépalas (índices [0,1]) → NÃO linearmente separável:**
- Versicolor e Virginica se sobrepõem no espaço das sépalas
- Nenhuma reta consegue separar perfeitamente

### Implicação Prática

O Classificador de Distância Mínima é um **classificador linear**. Ele só consegue 100% de acurácia quando os dados são linearmente separáveis. Para dados não separáveis, seria necessário usar classificadores não-lineares (redes neurais, SVM com kernel, etc.).

---

## 8. Métricas de Avaliação

### Acurácia

$$\text{Acurácia} = \frac{\text{Número de predições corretas}}{\text{Total de amostras}}$$

**Limitação:** Com classes desbalanceadas, a acurácia pode ser enganosa. Com Iris (balanceado), é confiável.

### Matriz de Confusão

Tabela que mostra a contagem de predições corretas e incorretas por classe.

```
            Predito
            Setosa   Versicolor  Virginica
Real Setosa    15          0          0
     Versicolor  0         15          0
     Virginica   0          0         15
```

- Diagonal principal: acertos (TP de cada classe)
- Fora da diagonal: erros (confusões entre classes)

### Precisão (Precision)

Para cada classe $j$:

$$\text{Precisão}_j = \frac{TP_j}{TP_j + FP_j}$$

**Pergunta respondida:** "Das vezes que o modelo disse 'é classe $j$', quantas vezes estava certo?"

**Quando importa:** Quando o custo de um falso positivo é alto (ex: diagnóstico de doença rara — não quero alarmar pacientes saudáveis).

### Revocação (Recall / Sensibilidade)

$$\text{Revocação}_j = \frac{TP_j}{TP_j + FN_j}$$

**Pergunta respondida:** "Das amostras que realmente eram classe $j$, quantas o modelo encontrou?"

**Quando importa:** Quando o custo de um falso negativo é alto (ex: detecção de câncer — não quero perder casos reais).

### F1-Score

$$F1_j = \frac{2 \cdot \text{Precisão}_j \cdot \text{Revocação}_j}{\text{Precisão}_j + \text{Revocação}_j}$$

É a **média harmônica** entre Precisão e Revocação. Penaliza modelos que são ótimos em uma métrica mas ruins na outra. Valor entre 0 e 1; 1 é perfeito.

### Macro-Média

A média simples de uma métrica sobre todas as classes. Trata todas as classes igualmente independente do tamanho.

$$\text{F1-macro} = \frac{1}{K}\sum_{j=1}^{K} F1_j$$

---

## 9. Resumo do Pipeline Completo

```
1. CARREGAR          iris_data.xls → 150 amostras {atributos: [4 floats], classe: str}
        ↓
2. SPLIT             Split Estratificado 70/30 → treino(105) + teste(45)
        ↓
3. TREINAR           Para cada classe: m_j = média dos vetores de pétala
        ↓            → protótipos: {setosa: [...], versicolor: [...], virginica: [...]}
4. PREDIZER          Para cada amostra de teste:
                     Calcular d_j(x) = xᵀmⱼ − ½mⱼᵀmⱼ para cada j
                     Retornar classe com maior d_j(x)
        ↓
5. AVALIAR           Acurácia, Matriz de Confusão, Precisão, Revocação, F1
        ↓
6. VISUALIZAR        Dispersão geral, heatmap de confusão, superfícies de decisão
```

---

## 10. Conexão com a Teoria de Bayes (contexto avançado)

O Classificador de Distância Mínima pode ser derivado como caso especial do **Classificador de Bayes** assumindo:
- Classes equiprováveis: $P(\omega_j) = 1/K$
- Distribuições gaussianas com mesma matriz de covariância: $\Sigma_j = \sigma^2 I$

Nesse caso, maximizar a função discriminante de Bayes reduz a minimizar a distância euclidiana ao protótipo — que é exatamente o que fazemos.

Isso justifica formalmente por que o algoritmo funciona para dados com distribuição aproximadamente gaussiana e classes bem separadas.
