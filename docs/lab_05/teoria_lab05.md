# Lab 5 — Teoria Completa: Perceptron Multicamadas (MLP) e Backpropagation

**Referência:** Aula PR_711 — Redes Neurais Artificiais (Prof. Robson Pequeno de Sousa)
**Implementação (item i):** `iris_classifier/models/mlp_backprop.py` (Python puro, sem numpy/scipy/sklearn)
**Implementação (item ii):** `iris_classifier/models/mlp_sklearn.py` (scikit-learn, uso explicitamente permitido pelo enunciado)
**Interface:** Aba 5 — *Feedforward (MLP)* da GUI (`iris_classifier/gui/tab_feedforward.py`)

---

## 1. Por que Redes Multicamadas?

Os classificadores lineares estudados nos laboratórios anteriores (Distância Mínima, Perceptron, Regra Delta) só conseguem separar classes que sejam **linearmente separáveis** — a fronteira de decisão é sempre uma reta (2D) ou hiperplano (nD). O problema **XOR** é o contraexemplo clássico: nenhuma reta separa as classes `{(0,0),(1,1)}` de `{(0,1),(1,0)}`.

A solução é empilhar camadas de neurônios: uma **camada oculta** entre a entrada e a saída permite que a rede aprenda combinações não lineares dos atributos originais, tornando o problema linearmente separável *no espaço da camada oculta*.

---

## 2. Modelo do Neurônio Artificial

Cada neurônio $i$ da camada $l$ calcula uma **entrada líquida** (net) a partir das saídas da camada anterior, e aplica uma **função de ativação** $h$:

$$z_i(l) = \sum_{j=1}^{n_{l-1}} w_{ij}(l)\, a_j(l-1) + b_i(l)$$

$$a_i(l) = h\big(z_i(l)\big)$$

Onde:
- $a_j(l-1)$ é a saída (ativação) do neurônio $j$ na camada anterior — para a camada de entrada, $a_j(1) = x_j$ (as próprias componentes do vetor de atributos).
- $w_{ij}(l)$ é o peso da conexão do neurônio $j$ (camada $l-1$) para o neurônio $i$ (camada $l$).
- $b_i(l)$ é o **bias** do neurônio $i$ — equivalente a uma entrada extra fixa em 1, com peso próprio.

### Função de Ativação Sigmoide

$$h(z) = \sigma(z) = \frac{1}{1 + e^{-z}}$$

Usada em todas as camadas (oculta e saída) nesta implementação. Sua derivada tem uma forma particularmente simples em função da própria saída, o que será essencial no backprop:

$$\sigma'(z) = \sigma(z)\big(1 - \sigma(z)\big)$$

---

## 3. Alimentação Adiante (Feedforward)

Para uma rede de 3 camadas (entrada → oculta → saída), o cálculo em um único passo à frente é:

1. **Camada de entrada:** $a(1) = x$
2. **Camada oculta:** $z(2) = W(2)\, a(1) + b(2)$ e $a(2) = \sigma\big(z(2)\big)$
3. **Camada de saída:** $z(3) = W(3)\, a(2) + b(3)$ e $a(3) = \sigma\big(z(3)\big)$

A rede atribui o padrão de entrada à classe $k$ cujo neurônio de saída tem a maior ativação: $a_k(3) > a_j(3)$ para todo $j \neq k$.

---

## 4. Algoritmo de Treinamento: Backpropagation (Retropropagação do Erro)

### 4.1 Função de Erro

O erro quadrático total entre as saídas desejadas $t_i$ e as saídas reais $z_i$ da rede é:

$$E = \frac{1}{2} \sum_{i=1}^{n} (t_i - z_i)^2$$

O objetivo do treinamento é ajustar todos os pesos e bias da rede para **minimizar** $E$, usando **gradiente descendente**:

$$w_{\text{novo}} = w - \eta \cdot \frac{\partial E}{\partial w}$$

onde $\eta$ é a **taxa de aprendizagem**.

### 4.2 Gradiente da Camada de Saída

Pela regra da cadeia, para um peso $w$ que liga um neurônio oculto $h$ a um neurônio de saída $o$:

$$\frac{\partial E}{\partial w} = \underbrace{\frac{\partial E}{\partial z_o}}_{\delta_o} \cdot \frac{\partial z_o}{\partial w} = \delta_o \cdot \text{out}_h$$

Onde o **termo de erro (delta)** da camada de saída é:

$$\delta_o = (z_o - t_o)\, z_o (1 - z_o)$$

Este delta combina três fatores: (1) o erro bruto $(z_o - t_o)$, (2) a derivada da sigmoide $z_o(1-z_o)$ — que "trava" o ajuste quando o neurônio já está saturado (próximo de 0 ou 1) — e (3) implicitamente, o próprio gradiente descendente na direção que reduz $E$.

### 4.3 Gradiente da Camada Oculta

Para um peso $w$ que liga um neurônio de entrada a um neurônio oculto $h$, o erro precisa ser **retropropagado** de volta através de todos os neurônios de saída conectados a $h$:

$$\delta_h = \left(\sum_{o} \delta_o \cdot w_{ho}\right) \text{out}_h (1 - \text{out}_h)$$

$$\frac{\partial E}{\partial w} = \delta_h \cdot \text{out}_{\text{entrada}}$$

**Ponto de atenção (destacado no slide):** se a rede tiver múltiplos neurônios de saída, cada um contribui com seu próprio $\delta_o$ para o cálculo de $\delta_h$ — a soma percorre **todas** as conexões de saída do neurônio oculto, não apenas uma.

### 4.4 Atualização dos Bias

Os bias seguem a mesma regra de atualização, tratando a "entrada" como constante igual a 1:

$$b_{\text{novo}} = b - \eta \cdot \delta$$

onde $\delta$ é o delta do próprio neurônio (de saída ou oculto).

### 4.5 Resumo do Algoritmo (1 amostra)

1. **Forward:** calcular $a(2)$ (saída oculta) e $a(3)$ (saída da rede).
2. **Erro:** calcular $E$ comparando $a(3)$ com o alvo $t$.
3. **Backward (saída):** $\delta_o = (a_o(3) - t_o)\, a_o(3)(1 - a_o(3))$ para cada neurônio de saída.
4. **Backward (oculta):** $\delta_h = \left(\sum_o \delta_o w_{ho}\right) a_h(2)(1 - a_h(2))$ para cada neurônio oculto.
5. **Atualização:** $w \leftarrow w - \eta \cdot \delta \cdot \text{entrada}$ e $b \leftarrow b - \eta \cdot \delta$, para todos os pesos e bias da rede.

Repetindo esses 5 passos para todas as amostras de treino, por várias épocas, a rede converge para um mínimo (local) da função de erro.

---

## 5. Item (i): Rede "Galinha vs Homem" — Implementação em Python Puro

O item (i) do laboratório pede uma rede totalmente conectada com:
- **2 entradas** ($a_1, a_2$),
- **2 neurônios na camada oculta** ($b_1, b_2$),
- **2 neurônios na camada de saída** ($c_1$ = homem, $c_2$ = galinha),
- ativação sigmoide em ambas as camadas,
- pesos iniciais dados no slide (ver `iris_classifier/lab05_galinha_homem.py`),
- taxa de aprendizagem $\eta = 0{,}05$,
- saída desejada: homem ($c_1$) = 0, galinha ($c_2$) = 1.

Toda a lógica (`RedeFeedforward` em `iris_classifier/models/mlp_backprop.py`) é implementada com listas nativas e laços `for` — **sem** `numpy`, `scipy` ou `scikit-learn`, seguindo a mesma restrição das demais implementações do projeto (Distância Mínima, Perceptron, Regra Delta, Bayes).

Os resultados numéricos completos (forward pass, deltas, pesos atualizados) estão em `relatorio_experimentos.md`.

---

## 6. Item (ii): Classificação Feedforward do Iris (uso permitido de biblioteca de ML)

O item (ii) pede a classificação das 3 espécies do Iris usando uma rede feedforward, comparando o resultado com o Classificador Ótimo de Bayes (QDA) e o Naive Bayes — ambos já implementados em Python puro nos laboratórios anteriores (`iris_classifier/models/bayes_classifier.py`).

Diferente do item (i), o enunciado **permite explicitamente** o uso de bibliotecas de Machine Learning para este experimento específico. Por isso, a rede feedforward do item (ii) é treinada com `sklearn.neural_network.MLPClassifier` (`iris_classifier/models/mlp_sklearn.py`) — o único ponto de todo o projeto que usa uma biblioteca de ML.

A avaliação reaproveita a mesma infraestrutura de métricas já usada nos laboratórios anteriores (`iris_classifier/evaluation/metricas_avancadas.py`):

- **Acerto Global (Ag)** e **Coeficiente Kappa** (com variância, para o teste de significância).
- **Coeficiente Tau** (alternativa ao Kappa assumindo classes equiprováveis).
- **Métricas por classe (One-vs-Rest):** Acurácia do Produtor (recall), Acurácia do Usuário (precisão), F1, F2 (maior peso à revocação) e Coeficiente de Matthews (MCC).
- **Teste Z de significância de Kappa**, comparando cada par de classificadores (MLP × Bayes, MLP × Naive, Bayes × Naive).

---

## 7. Bônus Interativo: Reconhecimento de Imagem em Pixels

O slide da Aula PR_711 ilustra o problema "galinha vs homem" original como o reconhecimento de uma imagem de **8×8 pixels** (64 valores de cinza, um por neurônio de entrada `a1...a64`) — o exemplo numérico com apenas 2 entradas (item i) é uma simplificação didática desse problema maior.

Para tornar essa ideia tangível, a Aba 5 inclui um **canvas de 8×8 pixels pintável à mão livre**, ao lado de uma **segunda rede própria** (independente da rede do item i), com arquitetura **64 entradas → 10 neurônios ocultos → 1 saída**, treinada do zero (`RedeFeedforward`, Python puro) usando apenas dois padrões de referência desenhados à mão (inspirados nas silhuetas "Man" e "Chicken" do slide) como dados de treino:

- padrão "Homem" → saída alvo 0
- padrão "Galinha" → saída alvo 1

A cada pixel pintado ou apagado, a alimentação adiante é recalculada instantaneamente e a interface mostra:
1. A saída da rede (0 a 1) com uma barra de progresso colorida.
2. O rótulo textual ("mais parecido com HOMEM/GALINHA" ou "ambíguo").
3. A ativação de cada um dos 10 neurônios da camada oculta, como um mosaico de tons de cinza — uma visualização direta de "o que a camada oculta está enxergando" naquele desenho.

Como a rede é treinada com apenas 2 exemplos, ela não generaliza no sentido estatístico tradicional — o objetivo é **pedagógico**: deixar visível, em tempo real, como pequenas mudanças na entrada (pixels) se propagam pela rede e alteram a saída, reforçando de forma interativa a mesma matemática de alimentação adiante estudada no item (i).

---

## 8. Estrutura Modular

| Arquivo | Responsabilidade | Restrição |
|---|---|---|
| `iris_classifier/models/mlp_backprop.py` | Rede feedforward + backprop do zero (item i) | Python puro — sem libs de ML |
| `iris_classifier/lab05_galinha_homem.py` | Script demonstrativo do item (i), reproduz os valores do slide | Python puro |
| `iris_classifier/models/mlp_sklearn.py` | Wrapper fino do `MLPClassifier` para o Iris (item ii) | scikit-learn permitido |
| `iris_classifier/main.py` (`experimento_mlp_iris`) | Orquestra item (ii): treina os 3 modelos, calcula métricas, testes Z | reaproveita `evaluation/` |
| `iris_classifier/gui/tab_feedforward.py` | Aba 5 da GUI — memória de cálculo do item (i) + comparativo do item (ii) | — |

---

*Disciplina: Tópicos Especiais em Inteligência Artificial · UEPB · 2026*
*Grupo: Erick Nathan · Laura Barbosa · Pedro Lucas*
