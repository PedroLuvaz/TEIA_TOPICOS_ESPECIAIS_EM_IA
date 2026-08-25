# Lab 2 — Teoria Completa: Perceptron, Regra Delta e o Problema XOR

**Disciplina:** Tópicos Especiais em Inteligência Artificial (TEIA)
**UEPB 2026**
**Grupo:** Erick Nathan · Laura Barbosa · Pedro Lucas
**Referência:** Aula PR4 (Prof. Robson Pequeno de Sousa)

---

## 1. Enunciados

O laboratório reúne três atividades da aula PR4.

**(A) Perceptron.** Estimar a superfície de decisão para classificar as classes
da Iris com estratégias binárias: plotar o diagrama de dispersão de $X_1$ e
$X_2$ das três classes num mesmo gráfico; inicializar os pesos aleatoriamente ou
com $w(1) = (0,0,0,0,0)$; usar $p = 0{,}03$; classificar Setosa × Versicolor,
Versicolor × Virgínica e Setosa × Virgínica; elaborar o fluxo de classificação
binária com os vetores de pesos ótimos; e observar que **versicolor e virgínica
não são linearmente separáveis** — treinar no máximo 100 épocas e examinar o
vetor de pesos resultante.

**(B) Regra Delta.** Usar a estratégia **um contra todos** para classificar as
três classes; pesos iniciais aleatórios ou $w(1) = (0,0,0,0,0)$; parâmetro de
aprendizagem parametrizável pelo usuário ou $p = 0{,}02$; saída desejada
$d = +1$ para a classe 1 e $d = -1$ para a classe 2; número de épocas
parametrizável, sendo o **critério de parada o número de épocas**; e plotar o
gráfico de convergência **época × MSE**, com
$\text{MSE} = \text{erro quadrático total} / n$.

**(C) XOR.** Implementar o problema do XOR com a Regra Delta, critério de parada
por número de épocas, plotando o gráfico de convergência. Saída desejada:
$d = 0$ para $0 \oplus 0$ e $1 \oplus 1$; $d = 1$ para $1 \oplus 0$ e
$0 \oplus 1$.

A linha que costura as três: **aprender a fronteira a partir dos erros**, em vez
de calculá-la diretamente das médias como no Lab 1 — e descobrir onde essa ideia
esbarra num limite intransponível.

---

## 2. O neurônio linear

### 2.1 Vetor aumentado e ativação

O bias entra como um peso associado a uma entrada constante igual a 1, o que
permite tratar tudo como um único produto escalar:

$$x_{\text{aug}} = [1, x_1, x_2, \ldots, x_n]^T,
\qquad w = [w_0, w_1, \ldots, w_n]^T$$

$$\text{net} = w^T x_{\text{aug}} = w_0 + \sum_{i=1}^{n} w_i x_i$$

Daí o $w(1) = (0,0,0,0,0)$ do enunciado ter **cinco** componentes: um bias mais
os quatro atributos da Iris.

### 2.2 Saída

$$y = \text{sgn}(\text{net}) = \begin{cases} +1, & \text{net} \ge 0 \\
-1, & \text{net} < 0 \end{cases}$$

A fronteira de decisão é o conjunto onde $\text{net} = 0$, ou seja
$w^Tx_{\text{aug}} = 0$ — um hiperplano, exatamente como no Lab 1. A diferença
está em **como** se chega a ele.

---

## 3. Perceptron de Rosenblatt

### 3.1 Regra de aprendizado

O perceptron só age quando erra:

$$\boxed{\;w \leftarrow w + p\,(d - y)\,x_{\text{aug}}\;}$$

com $d \in \{+1, -1\}$ o alvo, $y$ a saída atual e $p$ a taxa de aprendizado.

Quando $y = d$ o fator $(d-y)$ zera e nada muda. Quando erra, $(d - y) = \pm 2$
e o vetor de pesos é empurrado na direção da amostra mal classificada. É
aprendizado por **correção de erro**, amostra a amostra.

### 3.2 Por que a correção funciona

Se uma amostra de alvo $+1$ foi classificada como $-1$, então $\text{net} < 0$.
A atualização soma $2p\,x_{\text{aug}}$ a $w$, e o novo produto escalar passa a
ser:

$$w_{\text{novo}}^T x_{\text{aug}} = w^T x_{\text{aug}} + 2p\,\|x_{\text{aug}}\|^2$$

Como $\|x_{\text{aug}}\|^2 > 0$, o `net` daquela amostra **aumenta** — anda na
direção de acertá-la. O mesmo argumento vale, com sinal trocado, no outro caso.

### 3.3 Teorema da convergência e sua consequência prática

Se as duas classes forem **linearmente separáveis**, o perceptron converge para
uma solução em um número finito de passos, qualquer que seja $p > 0$ e o $w$
inicial. Se **não** forem, o algoritmo nunca para: sempre há alguma amostra mal
classificada empurrando os pesos, e eles oscilam indefinidamente.

Daí a instrução do enunciado de limitar a 100 épocas para o par
versicolor × virgínica. O limite não é um detalhe de implementação: é a única
forma de encerrar o treinamento num problema não separável. E o vetor de pesos
resultante é simplesmente **o estado em que o algoritmo estava quando o relógio
parou** — não uma solução ótima em nenhum sentido.

### 3.4 Critério de parada

O algoritmo encerra quando ocorre o que vier primeiro:

1. uma época inteira sem nenhum erro (convergiu); ou
2. o limite de épocas.

---

## 4. Regra Delta (Widrow-Hoff / Adaline)

### 4.1 A mudança de perspectiva

O perceptron mede o erro **depois** da função sinal: ou acertou, ou errou. A
Regra Delta mede o erro **antes**, sobre a saída linear:

$$e = d - \text{net}$$

Isso muda tudo. Mesmo quando a amostra está do lado certo da fronteira, se o
`net` estiver longe do alvo ainda há erro a reduzir — o treinamento continua
refinando a solução em vez de parar no primeiro arranjo que funciona.

### 4.2 Função de custo e gradiente

Minimiza-se o erro quadrático:

$$E = \frac{1}{2}\sum_k (d_k - \text{net}_k)^2$$

$$\frac{\partial E}{\partial w} = -\sum_k (d_k - \text{net}_k)\,x_{\text{aug},k}$$

Descendo o gradiente, e atualizando amostra a amostra (modo estocástico):

$$\boxed{\;w \leftarrow w + p\,(d - \text{net})\,x_{\text{aug}}\;}$$

A fórmula é quase idêntica à do perceptron — a diferença é que $\text{net}$
substitui $y = \text{sgn}(\text{net})$, e essa única troca transforma um
algoritmo de correção de erro num método de mínimos quadrados.

### 4.3 O MSE por época

Conforme o enunciado, o erro de cada época é a soma dos erros quadráticos de
todas as iterações daquela época, dividida pelo número de amostras de treino:

$$\boxed{\;\text{MSE}_{\text{época}} = \frac{1}{n}\sum_{k=1}^{n} (d_k - \text{net}_k)^2\;}$$

O gráfico **época × MSE** é a curva de convergência. Sua leitura:

- queda rápida e estabilização em valor baixo → problema separável, solução boa;
- estabilização em **patamar alto** → o modelo linear atingiu o melhor que
  consegue, e o resíduo é a sobreposição entre as classes;
- oscilação → taxa de aprendizado alta demais.

### 4.4 Critério de parada

Número de épocas, parametrizável pelo usuário. Diferentemente do perceptron, a
Regra Delta **não tem parada antecipada**: como o erro é contínuo, ele quase
nunca chega exatamente a zero.

---

## 5. Perceptron × Regra Delta

| | Perceptron | Regra Delta |
|---|---|---|
| Erro medido | após o sinal: $d - y$ | antes do sinal: $d - \text{net}$ |
| Atualiza quando | só ao errar | sempre |
| Se as classes são separáveis | converge em passos finitos | aproxima a solução de mínimos quadrados |
| Se não são | oscila para sempre | estabiliza num compromisso |
| Parada | erro zero **ou** limite de épocas | número de épocas |
| Saída do treino | um hiperplano que separa | o hiperplano de menor erro quadrático |

---

## 6. Estratégias de classificação

### 6.1 Binária, par a par

Cada problema envolve duas classes, com alvos $d = +1$ e $d = -1$. Os três pares
do enunciado — Setosa × Versicolor, Versicolor × Virgínica e Setosa × Virgínica
— produzem três vetores de pesos independentes e três fronteiras.

### 6.2 Um contra todos (One-vs-All)

Para classificar as **três** classes de uma vez, treina-se um classificador por
classe:

$$d = +1 \text{ se a amostra pertence à classe } c, \qquad
d = -1 \text{ caso contrário}$$

E a decisão multiclasse é o maior `net` entre os três:

$$\text{classe}(x) = \arg\max_c \; w_c^T x_{\text{aug}}$$

Note o paralelo com o Lab 1: também ali a decisão multiclasse era um
$\arg\max$ de funções lineares. A diferença é a origem dos coeficientes —
médias, num caso; aprendizado por erro, no outro.

---

## 7. O problema XOR

### 7.1 A tabela

| $x_1$ | $x_2$ | $d$ |
|:---:|:---:|:---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

### 7.2 Por que nenhuma reta resolve

Os pontos de saída 1 — $(0,1)$ e $(1,0)$ — ficam em cantos **opostos** do
quadrado, e o mesmo vale para os de saída 0. Qualquer reta que deixe $(0,1)$ e
$(1,0)$ de um lado necessariamente deixa também $(0,0)$ ou $(1,1)$ junto. É a
demonstração clássica de que o XOR não é linearmente separável.

Formalmente, seria preciso satisfazer ao mesmo tempo:

$$w_0 < 0{,}5 \quad (0,0)\to 0, \qquad w_0 + w_2 \ge 0{,}5 \quad (0,1)\to 1$$
$$w_0 + w_1 \ge 0{,}5 \quad (1,0)\to 1, \qquad w_0 + w_1 + w_2 < 0{,}5 \quad (1,1)\to 0$$

Somando a segunda e a terceira: $2w_0 + w_1 + w_2 \ge 1$. Da primeira e da
quarta: $2w_0 + w_1 + w_2 < 1$. Contradição — **o sistema não tem solução**.

### 7.3 O que se espera ver

Com alvos 0 e 1 e limiar em 0,5, a melhor resposta linear possível é prever
sempre algo próximo de 0,5 para os quatro padrões, o que dá erro quadrático
médio de $(0{,}5)^2 = 0{,}25$. O MSE **estaciona nesse patamar** e não desce
mais, por mais épocas que se rode. Não é falha do algoritmo: é a medida exata do
limite de um neurônio único.

A saída desse impasse é acrescentar uma camada oculta — o assunto do Lab 5.

---

## 8. Implementação

Python puro, sem bibliotecas de aprendizado de máquina.

| Arquivo | Papel |
|---|---|
| `iris_classifier/models/perceptron.py` | `treinar_perceptron` (binário), `predizer_perceptron`, `acuracia_binaria_perceptron` e `treinar_perceptron_ova` (multiclasse) |
| `iris_classifier/models/delta_rule.py` | `_treinar_delta` (núcleo), `treinar_delta_iris` (binário), `treinar_delta_ova` (um contra todos), `treinar_delta_xor` e as funções de predição |
| `iris_classifier/core/math_utils.py` | Produto escalar e operações de vetor |

**No aplicativo:** aba *Perceptron & Delta*, com quatro sub-abas — Perceptron,
Regra Delta, Delta OvA e XOR —, cada uma com o diagrama de dispersão, a
fronteira estimada e a curva de convergência. O Perceptron OvA e a Regra Delta
OvA também aparecem no catálogo da aba *Classificar*, com taxa e épocas
ajustáveis.

Os resultados obtidos estão em
[`relatorio_experimentos.md`](relatorio_experimentos.md).
