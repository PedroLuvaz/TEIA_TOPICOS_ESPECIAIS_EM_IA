# Lab 1 — Teoria Completa: Classificador de Distância Mínima

**Disciplina:** Tópicos Especiais em Inteligência Artificial (TEIA)
**UEPB 2026**
**Grupo:** Erick Nathan · Laura Barbosa · Pedro Lucas
**Referência:** Aula PR3 (Prof. Robson Pequeno de Sousa)

---

## 1. Enunciado

> Utilize a base de dados Data Iris e implemente os modelos estudados na aula PR3:
> **(i)** classificador de distância mínima para as três classes;
> **(ii)** função de decisão para as três classes utilizando o classificador de
> máximo;
> **(iii)** superfície de decisão para duas classes.
>
> Detalhamento: usar **70%** dos dados de cada classe para treinamento e os
> **30%** restantes para teste. No item (iii), planejar o classificador com o
> esquema: Virgínica × Setosa; Setosa × Versicolor; Versicolor × Virgínica.

Os três itens são o mesmo classificador visto de três ângulos: o item (i) é a
regra de decisão, o item (ii) é a forma algébrica equivalente que evita a raiz
quadrada, e o item (iii) é a fronteira que essa regra desenha no espaço de
atributos.

---

## 2. A base e a preparação dos dados

A base Iris tem **150 amostras**, **4 atributos** numéricos e **3 classes**
balanceadas (50 amostras cada):

| Índice | Atributo |
|:---:|---|
| 0 | Comprimento da sépala (cm) |
| 1 | Largura da sépala (cm) |
| 2 | Comprimento da pétala (cm) |
| 3 | Largura da pétala (cm) |

### 2.1 Split estratificado 70/30

O enunciado pede 70% **de cada classe**, e não 70% do total. A diferença
importa: um sorteio simples sobre as 150 amostras poderia, por azar, deixar uma
classe sub-representada no treino e enviesar o protótipo dela.

O procedimento é, para cada classe separadamente: embaralhar as 50 amostras e
separar as 35 primeiras para treino e as 15 restantes para teste. Resultado:
**105 amostras de treino e 45 de teste**, com 15 de cada classe no teste.

A semente aleatória é fixada em **42**, de modo que a execução é reprodutível —
os números do relatório são os mesmos que a banca verá rodando o programa.

---

## 3. Item (i) — Classificador de Distância Mínima

### 3.1 Ideia

Cada classe é resumida a **um único ponto**: o vetor médio das suas amostras de
treino, chamado **protótipo**. Uma amostra desconhecida é atribuída à classe
cujo protótipo está mais próximo.

$$m_j = \frac{1}{N_j}\sum_{x \in \omega_j} x$$

onde $N_j$ é o número de amostras de treino da classe $j$ e $\omega_j$ o
conjunto dessas amostras. O protótipo tem a mesma dimensão dos dados: quatro
componentes quando se usam os quatro atributos, duas quando se usa apenas o par
de pétalas.

### 3.2 Regra de decisão

$$\text{classe}(x) = \arg\min_j \; \|x - m_j\|,
\qquad \|a - b\| = \sqrt{\sum_{i=1}^{n}(a_i - b_i)^2}$$

É o classificador mais simples possível dentro da família dos lineares: uma
média por classe, uma distância por classe, e o menor valor vence.

### 3.3 Treinamento

Não há iteração, taxa de aprendizado nem critério de parada. O "treinamento" é
o cálculo direto das médias — o que torna este classificador **determinístico**:
rodando duas vezes com o mesmo split, os protótipos são idênticos.

---

## 4. Item (ii) — Função de decisão e classificador de máximo

### 4.1 Da distância mínima para o máximo discriminante

Comparar distâncias exige calcular uma raiz quadrada por classe. Isso pode ser
evitado. Expandindo o quadrado da distância:

$$\|x - m_j\|^2 = (x - m_j)^T(x - m_j) = x^Tx - 2\,x^Tm_j + m_j^Tm_j$$

O termo $x^Tx$ **não depende de $j$**: é o mesmo para todas as classes e, numa
comparação, pode ser descartado. Sobra minimizar $-2x^Tm_j + m_j^Tm_j$, o que
equivale a **maximizar**:

$$\boxed{\;d_j(x) = x^{T}m_j - \tfrac{1}{2}\,m_j^{T}m_j\;}$$

### 4.2 Regra do máximo

$$\text{classe}(x) = \arg\max_j \; d_j(x)$$

Esta é a **função de decisão** pedida no item (ii), e $d_j$ é chamada de
**função discriminante linear** — linear porque é uma função afim de $x$: um
produto escalar mais uma constante.

A equivalência é exata, não aproximada: o classificador de máximo e o de
distância mínima produzem **sempre** a mesma resposta. A vantagem é
computacional (nenhuma raiz quadrada) e conceitual — a forma $w^Tx + b$ conecta
este classificador ao Perceptron e à Regra Delta do Lab 2.

---

## 5. Item (iii) — Superfície de decisão entre duas classes

### 5.1 Dedução

A fronteira entre as classes $i$ e $j$ é o lugar geométrico dos pontos em que o
classificador fica indeciso, isto é, onde os dois discriminantes se igualam:

$$d_i(x) = d_j(x)$$
$$x^Tm_i - \tfrac{1}{2}m_i^Tm_i = x^Tm_j - \tfrac{1}{2}m_j^Tm_j$$
$$x^T(m_i - m_j) - \tfrac{1}{2}\left(\|m_i\|^2 - \|m_j\|^2\right) = 0$$

Que é a equação de um **hiperplano** $w^Tx + b = 0$, com:

$$\boxed{\;w = m_i - m_j\;}
\qquad
\boxed{\;b = -\tfrac{1}{2}\left(\|m_i\|^2 - \|m_j\|^2\right)\;}$$

### 5.2 A reta no plano 2D

Com dois atributos, $w_1x_1 + w_2x_2 + b = 0$, e para desenhar basta isolar
$x_2$:

$$x_2 = \frac{-w_1x_1 - b}{w_2}$$

### 5.3 Interpretação geométrica

O vetor $w = m_i - m_j$ é a direção que liga os dois protótipos, e a fronteira é
perpendicular a ela. Mais que isso: a fronteira passa exatamente pelo **ponto
médio** do segmento que une $m_i$ e $m_j$ — ou seja, é a **mediatriz** desse
segmento.

Uma consequência prática: a fronteira depende **apenas das médias**. A dispersão
das classes, a forma da nuvem de pontos e a correlação entre atributos não são
levadas em conta. Duas classes com médias próximas mas dispersões muito
diferentes recebem uma fronteira ruim — limitação que o Lab 4 corrige com o
classificador de Bayes, ao incorporar a matriz de covariância.

### 5.4 Os três pares do enunciado

| Par | Expectativa |
|---|---|
| Virgínica × Setosa | Classes muito distantes; separação trivial |
| Setosa × Versicolor | Setosa é isolada das outras duas; separação limpa |
| Versicolor × Virgínica | Classes vizinhas, com sobreposição parcial |

---

## 6. Métricas usadas na avaliação

Sobre as 45 amostras de teste:

$$\text{Acurácia} = \frac{\text{acertos}}{\text{total}}$$

E a **matriz de confusão**, com a convenção adotada em todo o projeto —
**linha = classe predita, coluna = classe real**. A diagonal concentra os
acertos; cada célula fora dela mostra exatamente qual confusão o classificador
cometeu.

As métricas mais refinadas (Kappa, Tau, MCC e os testes de significância) são o
assunto do Lab 3 e estão em [`../lab_03/teoria_lab03.md`](../lab_03/teoria_lab03.md).

---

## 7. Implementação

Tudo em Python puro — sem `numpy`, `pandas` ou bibliotecas de aprendizado de
máquina. A álgebra linear foi escrita com laços, listas e `zip`.

| Arquivo | Papel |
|---|---|
| `iris_classifier/core/math_utils.py` | `produto_escalar`, `distancia_euclidiana`, `calcular_media`, `discriminante`, `coeficientes_superficie_decisao` |
| `iris_classifier/models/classifier.py` | `treinar` (protótipos) e `predizer_todas_classes` (regra do máximo) |
| `iris_classifier/data/data_loader.py` | Leitura do `.xls` e `split_estratificado` |
| `iris_classifier/evaluation/evaluator.py` | Acurácia e matriz de confusão |

**No aplicativo:** aba *Distância Mínima* — protótipos, discriminantes,
fronteiras e regiões de decisão, com a memória de cálculo mostrando a
substituição numérica de cada etapa. O modelo também aparece no catálogo da aba
*Classificar*.

Os resultados obtidos estão em
[`relatorio_experimentos.md`](relatorio_experimentos.md).
