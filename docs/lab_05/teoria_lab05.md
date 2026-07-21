# Lab 5 — Teoria Completa: Perceptron Multicamadas (MLP) e Backpropagation

**Referência:** Aula PR_711 — Redes Neurais Artificiais (Prof. Robson Pequeno de Sousa)
**Implementação (item i):** `iris_classifier/models/mlp_backprop.py` (Python puro, sem numpy/scipy/sklearn)
**Implementação (item ii):** `iris_classifier/models/mlp_sklearn.py` (scikit-learn, uso explicitamente permitido pelo enunciado)
**Interface:** duas abas da GUI —
**Lab 5.0** *XOR (MLP)* (`iris_classifier/gui/tab_xor.py`, slides 36-37) e
**Lab 5.1** *Feedforward (MLP)* (`iris_classifier/gui/tab_feedforward.py`, itens i/ii + exercício extra do slide 34)

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

## 8. Lab 5.0 — XOR com MLP (slides 36-37)

O laboratório abre com uma aba dedicada, **Lab 5.0** (`iris_classifier/gui/tab_xor.py`), que resolve o exercício do XOR (slide 36) usando o exemplo didático genérico do slide 37 como demonstração do algoritmo. Ela antecede a aba "Lab 5.1" (itens i/ii do enunciado formal) porque tem arquitetura, pesos e visualização próprios.

### 8.1 Exemplo Didático (slide 37) — Rede 2-2-2 Genérica

O slide 37 ("Exemplo didático: treinando uma rede de 3 camadas") **não resolve o XOR** — é um exemplo completo e genérico de como a conta do backpropagation é feita passo a passo, com números diferentes do restante do laboratório: $i_1=0{,}05$, $i_2=0{,}10$, alvo $o_1=0{,}01$ e $o_2=0{,}99$, pesos $w_1..w_8$ e $\eta=0{,}5$. Os slides 38-43 mostram a solução completa (1ª e 2ª iteração, e uma curva de convergência até a época 1000), o que permite conferir a implementação numericamente, exatamente como já era feito para o item (i).

**Particularidade importante deste exemplo:** ao contrário de todos os outros exemplos do laboratório (item i, Exercício A), aqui o bias $b_1$ e $b_2$ é **um único valor por camada, compartilhado por todos os neurônios dela** — não um bias independente por neurônio. Isso é visível na fórmula de atualização do bias do slide: $\partial E/\partial b = \sum \delta$ (soma os deltas de **todos** os neurônios da camada, não apenas um). A classe `JanelaMemoriaCalculoMLP` foi estendida com o parâmetro `bias_compartilhado=True` especificamente para reproduzir essa convenção; sem ele, cada neurônio atualizaria seu próprio bias de forma independente (a convenção padrão, usada no item i e no Exercício A) e o resultado numérico divergiria do slide a partir da 2ª iteração.

Verificação: com `bias_compartilhado=True`, a implementação reproduz **exatamente** todos os valores dos slides 38-42 — alimentação adiante ($\text{out}_{h_1}=0{,}593270$, $\text{out}_{h_2}=0{,}596884$, $\text{out}_{o_1}=0{,}751365$, $\text{out}_{o_2}=0{,}772928$, $E=0{,}298371$), deltas, pesos/bias atualizados ($b_1'=0{,}340637$, $b_2'=0{,}549800$) e a 2ª iteração completa ($E=0{,}285751$).

### 8.2 Exercício XOR (slide 36) — Arquitetura Fig. 12.28(b), 1 Época

Enunciado: *"Resolva o problema XOR utilizando uma MLP de acordo com a arquitetura da rede fig 12.28(b). Exercicte com uma época apenas. Implemente a arquitetura de rede acima."*

A Figura 12.28(b) mostra apenas a **topologia mínima** que resolve o XOR (2 entradas → 2 ocultos → 1 saída, pesos rotulados genericamente $w_1...w_9$, sem valores numéricos no slide) — por isso os pesos iniciais usados na demonstração (`iris_classifier/lab05_exercicio_xor.py`) foram escolhidos pelo grupo, com $\eta=0{,}5$.

"1 época" = os 4 padrões da tabela-verdade do XOR são apresentados **uma vez cada**, em sequência, com atualização de pesos após cada padrão (modo online/estocástico — mesma convenção de `perceptron.py` e `delta_rule.py` deste projeto). Como o XOR não é linearmente separável, uma única época não é suficiente para a rede convergir: as saídas permanecem próximas de 0,5 (região de máxima incerteza da sigmoide) mesmo após processar os 4 padrões.

**Painel interativo:** a Aba Lab 5.0 mostra, ao vivo, uma fronteira de decisão 2D (mapa de calor da saída da rede sobre o plano $x_1 \times x_2$, com os 4 pontos do XOR sobrepostos) e uma curva de convergência (erro médio por época). Os botões "Rodar exatamente 1 época" reproduzem o exercício tal como pedido no slide; os botões "+500/+2000 épocas" continuam o treino além do mínimo exigido — com estes pesos iniciais o erro fica quase estacionado até por volta da época 500 e só converge de fato entre as épocas 2000-5000, o que **confirma na prática** por que uma camada oculta não linear é indispensável para o XOR (o mesmo limite já demonstrado com a Regra Delta linear na Aba 2, onde o MSE estaciona em 0,25 sem nunca zerar) e, ao mesmo tempo, mostra que — diferente da Regra Delta — a MLP eventualmente resolve o problema.

---

## 9. Lab 5.1 — Exercício Extra (slide 34)

Além dos itens (i) e (ii) do enunciado formal, a Aba "Lab 5.1" também resolve um exercício adicional do slide, reaproveitando o mesmo motor `RedeFeedforward` (Python puro).

### 9.1 Exercício A (slide 34) — Rede da Figura 12.32, 1 Iteração

Enunciado: *"Treine a rede abaixo, em que a saída desejada é 1 para C1 e 0 para C2, só uma interação [iteração]."* — reaproveita a "pequena rede totalmente conectada" do exemplo numérico completo da aula (Figuras 12.32/12.33), agora com um alvo explícito ($C_1=1$, $C_2=0$) e pedindo apenas **1 iteração** (1 passo de forward + backward + atualização de pesos).

Arquitetura: **3 entradas → 2 ocultos → 2 saídas**, pesos dados no exemplo do slide. O slide não especifica uma taxa de aprendizagem para este exercício específico — foi adotado $\eta = 0{,}5$ (mesma ordem de grandeza do exemplo didático do slide 37), documentado explicitamente no código (`iris_classifier/lab05_exercicio_fig1232.py`).

Resultado: o erro total cai de $E=0{,}26960$ para $E=0{,}23986$ após a única iteração — confirmando que o passo de gradiente descendente caminha na direção correta, mesmo sem convergir em apenas 1 iteração.

---

## 10. Estrutura Modular

| Arquivo | Responsabilidade | Restrição |
|---|---|---|
| `iris_classifier/models/mlp_backprop.py` | Rede feedforward + backprop do zero (item i) | Python puro — sem libs de ML |
| `iris_classifier/lab05_galinha_homem.py` | Script demonstrativo do item (i), reproduz os valores do slide | Python puro |
| `iris_classifier/lab05_exercicio_fig1232.py` | Script do Exercício A (slide 34), rede Fig. 12.32, 1 iteração | Python puro |
| `iris_classifier/lab05_exercicio_xor.py` | Script do exercício XOR (slide 36), MLP, 1 época | Python puro |
| `iris_classifier/models/mlp_sklearn.py` | Wrapper fino do `MLPClassifier` para o Iris (item ii) | scikit-learn permitido |
| `iris_classifier/main.py` (`experimento_mlp_iris`) | Orquestra item (ii): treina os 3 modelos, calcula métricas, testes Z | reaproveita `evaluation/` |
| `iris_classifier/gui/tab_xor.py` | Aba Lab 5.0 — exemplo didático (slide 37) + XOR interativo (slide 36) | — |
| `iris_classifier/gui/tab_feedforward.py` | Aba Lab 5.1 — item (i), item (ii) e Exercício A | — |

---

*Disciplina: Tópicos Especiais em Inteligência Artificial · UEPB · 2026*
*Grupo: Erick Nathan · Laura Barbosa · Pedro Lucas*
