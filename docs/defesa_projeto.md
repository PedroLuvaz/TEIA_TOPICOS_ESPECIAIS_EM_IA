# Guia de Defesa do Projeto — Reconhecimento de Padrões

> **Documento único de apoio à defesa.** Explica o que o aplicativo faz, toda a
> teoria por trás de cada modelo, as métricas, os testes de significância e a
> arquitetura do software. Os documentos por laboratório continuam existindo com
> mais detalhe: este aqui costura tudo numa linha só, na ordem em que a defesa
> tende a acontecer.
>
> Universidade Estadual da Paraíba · Tópicos Especiais em Inteligência
> Artificial · Erick Nathan, Laura Barbosa e Pedro Lucas.

---

## Sumário

1. [O que a entrega pedia e onde cada item está](#1-o-que-a-entrega-pedia-e-onde-cada-item-está)
2. [Como rodar e roteiro de demonstração](#2-como-rodar-e-roteiro-de-demonstração)
3. [Fundamentos: dados, treino e teste](#3-fundamentos-dados-treino-e-teste)
4. [Os sete modelos, um a um](#4-os-sete-modelos-um-a-um)
5. [Métricas de qualidade](#5-métricas-de-qualidade)
6. [Comparação de modelos e testes de significância](#6-comparação-de-modelos-e-testes-de-significância)
7. [A base do usuário em .txt](#7-a-base-do-usuário-em-txt)
8. [Arquitetura do software](#8-arquitetura-do-software)
9. [Resultados de referência](#9-resultados-de-referência)
10. [Roteiro de fala e perguntas prováveis](#10-roteiro-de-fala-e-perguntas-prováveis)
11. [Mapa da documentação](#11-mapa-da-documentação)

---

## 1. O que a entrega pedia e onde cada item está

As instruções do professor, e a resposta do projeto a cada uma:

| Pedido | Onde está no aplicativo | Detalhe |
|---|---|---|
| *"Opções de definição do modelo a ser utilizado no processo de classificação, bem como a parametrização do modelo"* | Aba **Classificar**: seletor com sete modelos e os hiperparâmetros de cada um | [`classificar_modelos.md`](classificar_modelos.md) |
| *"O aplicativo deverá ser alimentado pela base de dados do usuário, no formato txt"* | Botão **Importar .txt**, no painel de configuração presente em todas as telas | [`importar_dados_txt.md`](importar_dados_txt.md) |
| *"Disponibilizar as métricas de qualidade"* | Acerto global, Kappa, Tau, acurácia do produtor e do usuário, especificidade, F1, F2, MCC, matriz de confusão, validação cruzada com IC 95% | [§5](#5-métricas-de-qualidade) |
| *"Comparação de modelos utilizando os testes de significância vistos em sala"* | Teste Z de Kappa e de Tau, McNemar, bootstrap pareado e teste de permutação, para qualquer par dos sete modelos | [§6](#6-comparação-de-modelos-e-testes-de-significância) |
| *"As equipes devem disponibilizar no aplicativo o modelo apresentado na defesa"* | **Florestas Aleatórias**: aba própria (árvores navegáveis, OOB, importâncias) e entrada no catálogo de classificação | [`seminario_florestas_aleatorias.md`](seminario_florestas_aleatorias.md) |

Uma frase que resume a resposta ao professor: **o aplicativo deixou de ser um
demonstrador do Iris e virou uma ferramenta de classificação** — o usuário traz
os dados dele, escolhe o modelo, ajusta os parâmetros, mede a qualidade e testa
se a diferença entre dois modelos é estatisticamente real.

---

## 2. Como rodar e roteiro de demonstração

### 2.1 Subir o aplicativo

O caminho mais curto, para apresentar: **duplo clique em `Iniciar Projeto.bat`**
(Windows) ou `./iniciar.sh` (macOS/Linux). O script instala o que falta, compila
o frontend na primeira vez, sobe o servidor e abre o navegador.

Para desenvolver, dois terminais:

```bash
python -m uvicorn web_app.backend.main:app --reload --port 8000
```

```bash
npm --prefix web_app/frontend run dev
```

Interface em `http://localhost:5173`; documentação automática da API em
`http://localhost:8000/docs`.

### 2.2 Roteiro de 10 minutos

| Tempo | O que mostrar | Frase-guia |
|---|---|---|
| 0:00 | Aba **Classificar** com Iris + pétalas + Distância Mínima | "Esta é a tela de uso: base, modelo, parâmetros." |
| 1:00 | Trocar para **Floresta Aleatória** e mexer em *nº de árvores* e *profundidade* | "Os controles vêm do esquema que o próprio modelo publica." |
| 2:30 | Apontar *acerto no treino* vs *acerto no teste* com profundidade sem limite | "Aqui o sobreajuste fica visível: 94% no treino, 73% no teste." |
| 3:30 | Clicar no gráfico para classificar um ponto | "A predição mostra a pontuação que o modelo dá a cada classe." |
| 4:30 | **Importar .txt** — a tela já abre com `data/exemplos/iris.txt` lido; basta apontar a prévia (ou clicar no botão do outro exemplo) | "A base é do usuário; o delimitador e a classe são detectados." |
| 6:00 | Rodar um modelo sobre a base importada | "Tudo o que existe no app passa a valer para essa base." |
| 7:00 | Aba **Métricas Avançadas → Significância** | "Comparar dois números não basta: aqui está o teste pareado." |
| 8:30 | Matriz de todos os pares + validação cruzada | "E aqui os sete modelos comparados de uma vez." |
| 9:30 | Aba **Florestas Aleatórias** (seminário) | "O modelo que apresentamos, com as árvores navegáveis." |

Dica de demonstração: **use sépalas, não pétalas**. Nas pétalas quase todo
modelo marca 100% e nenhuma diferença é significativa — o que é um resultado
honesto, mas não mostra o aplicativo trabalhando. Nas sépalas as classes se
sobrepõem e os testes têm o que dizer.

---

## 3. Fundamentos: dados, treino e teste

### 3.1 O problema

Reconhecimento de padrões supervisionado: dado um vetor de atributos
$x = [x_1, \dots, x_d]$, atribuir a ele uma classe $\omega_j$ entre $C$
possíveis, a partir de exemplos rotulados. Cada modelo deste projeto é uma
maneira diferente de traçar as **fronteiras de decisão** que dividem o espaço de
atributos em regiões.

### 3.2 As bases

| Base | Amostras | Atributos | Classes | Papel |
|---|---|---|---|---|
| Iris Original (`v1`) | 150 | 4 numéricos | 3 | Base clássica de Fisher, usada nos laboratórios |
| Iris Separável (`v2`) | 150 | 4 numéricos | 3 | Variante linearmente separável |
| Fim de Semana (`fds`) | 1000 | 3 categóricos | 4 | Base do seminário, com 8% de ruído injetado |
| `usr_*` | livre | livre | livre | Qualquer `.txt` importado pelo usuário |

### 3.3 Split estratificado

O conjunto é dividido em **70% treino / 30% teste**, *por classe*, com
`random.seed(42)`. Estratificar importa: um sorteio simples poderia deixar uma
classe inteira fora do treino. No Iris isso dá 105 amostras de treino e 45 de
teste (15 por classe).

A semente fixa é o que torna os números reproduzíveis — o professor vê na
defesa os mesmos valores do relatório.

### 3.4 Por que 45 amostras de teste enganam

Com 45 amostras, um erro a mais ou a menos move a acurácia em 2,2 pontos. Por
isso o projeto oferece **validação cruzada k-fold estratificada com
repetições**: toda amostra é testada, e o desvio entre as dobras diz o quanto o
número é confiável. É a diferença entre "97,8%" e "97,6% ± 2,5%".

---

## 4. Os sete modelos, um a um

Todos escritos do zero, em Python puro — sem `numpy`, `pandas`, `scipy` ou
`scikit-learn`. A única exceção do projeto inteiro é o item (ii) do Lab 5, onde
o enunciado permitia explicitamente a biblioteca, e que fica isolado em
`models/mlp_sklearn.py`.

### 4.1 Classificador de Distância Mínima

**Ideia.** Cada classe é resumida por um ponto: o vetor médio (protótipo). Uma
amostra pertence à classe do protótipo mais próximo.

**Treinamento** — um protótipo por classe:

$$m_j = \frac{1}{N_j}\sum_{x \in \omega_j} x$$

**Decisão** — em vez de calcular distâncias com raiz quadrada, usa-se a função
discriminante linear equivalente:

$$d_j(x) = x^{T}m_j - \tfrac{1}{2}m_j^{T}m_j \qquad
\text{classe} = \arg\max_j d_j(x)$$

A equivalência vem de expandir $\|x - m_j\|^2 = x^Tx - 2x^Tm_j + m_j^Tm_j$: o
termo $x^Tx$ é o mesmo para todas as classes, então minimizar a distância é
maximizar $x^Tm_j - \frac12 m_j^Tm_j$.

**Fronteira.** Igualando $d_i(x) = d_j(x)$ obtém-se um hiperplano:

$$w = m_i - m_j, \qquad b = -\tfrac{1}{2}\left(\|m_i\|^2 - \|m_j\|^2\right),
\qquad w^Tx + b = 0$$

Em 2D, a reta para plotar é $x_2 = (-w_1x_1 - b)/w_2$. Geometricamente, é a
**mediatriz** do segmento que liga os dois protótipos.

**Hiperparâmetros:** nenhum. O modelo é totalmente determinado pelos dados.

**Quando falha.** Quando as classes têm dispersões muito diferentes ou formas
alongadas: o protótipo ignora a covariância. É exatamente o que o Bayes corrige.

*Código:* `models/classifier.py`, `core/math_utils.py`.

### 4.2 Perceptron (Rosenblatt) e Perceptron OvA

**Ideia.** Aprender os pesos de um hiperplano corrigindo-os a cada erro.

**Modelo.** Com o vetor aumentado $x_{\text{aug}} = [1, x_1, \dots, x_d]$:

$$\text{net} = w^Tx_{\text{aug}}, \qquad y = \text{sgn}(\text{net})$$

**Regra de aprendizado** — só atualiza quando erra:

$$w \leftarrow w + p\,(d - y)\,x_{\text{aug}}$$

com $d \in \{+1, -1\}$ o alvo e $p$ a taxa de aprendizado.

**Teorema da convergência.** Se as duas classes forem linearmente separáveis, o
Perceptron converge em um número finito de passos. Se **não** forem, ele oscila
para sempre — daí o limite de épocas.

**Multiclasse (OvA).** Treina-se um Perceptron por classe, rerrotulando as
amostras em $+1$ (a classe) e $-1$ (todo o resto); a decisão é
$\arg\max_c \text{net}_c$. O algoritmo binário é idêntico ao do Lab 2: muda só a
rotulagem.

**Hiperparâmetros:** taxa de aprendizado, máximo de épocas.

**Leitura didática.** No Iris com pétalas, "setosa contra o resto" converge em
poucas épocas; "versicolor contra o resto" e "virginica contra o resto" nunca
convergem, porque essas classes não são separáveis do resto por uma reta. Por
isso o Perceptron OvA fica em 66,7% — e a interface mostra um selo por classe
dizendo quem convergiu.

*Código:* `models/perceptron.py`.

### 4.3 Regra Delta (Widrow-Hoff / Adaline)

**Ideia.** Em vez de corrigir só quando erra, minimizar continuamente o erro
quadrático da **saída linear** (antes do limiar):

$$E = \frac{1}{2}\sum (d - \text{net})^2, \qquad
w \leftarrow w + p\,(d - \text{net})\,x_{\text{aug}}$$

**Diferença essencial para o Perceptron.** O Perceptron usa o erro *depois* do
sinal e para quando acerta tudo; a Regra Delta usa o erro *antes* do sinal e
continua ajustando, produzindo uma solução de mínimos quadrados mesmo quando as
classes se sobrepõem. Em compensação, ela não tem parada antecipada: roda todas
as épocas.

**Multiclasse:** mesmo esquema Um-Contra-Todos, com $\arg\max$ dos *nets*.

**Hiperparâmetros:** taxa de aprendizado, épocas.

**O XOR.** Com o XOR, o MSE da Regra Delta estaciona em 0,25 e nunca zera —
demonstração empírica de que um único neurônio não resolve um problema não
linearmente separável. É o que motiva a camada oculta do Lab 5.

*Código:* `models/delta_rule.py`.

### 4.4 Bayes Ótimo (QDA)

**Ideia.** Modelar a distribuição de cada classe e escolher a mais provável —
abordagem *generativa*, ao contrário das anteriores, que só traçam fronteiras.

**Regra de Bayes.** Com prioris iguais, maximizar $P(\omega_j|x)$ equivale a
maximizar $p(x|\omega_j)$. Assumindo normal multivariada:

$$p(x|\omega_j) = \frac{1}{(2\pi)^{d/2}|\Sigma_j|^{1/2}}
\exp\left(-\tfrac{1}{2}(x-m_j)^T\Sigma_j^{-1}(x-m_j)\right)$$

**Treinamento** — média e covariância amostrais por classe:

$$m_j = \frac{1}{N_j}\sum x, \qquad
\Sigma_j = \frac{1}{N_j-1}\sum (x-m_j)(x-m_j)^T$$

**Discriminante** (log da verossimilhança, sem a constante):

$$d_j(x) = -\tfrac{1}{2}\ln|\Sigma_j| - \tfrac{1}{2}\,d_M^2(x, m_j),
\qquad d_M^2 = (x-m_j)^T\Sigma_j^{-1}(x-m_j)$$

$d_M$ é a **distância de Mahalanobis**: a distância euclidiana "corrigida" pela
forma da nuvem de pontos da classe.

**Regularização de Ridge.** Com poucas amostras, $\Sigma_j$ pode ser singular.
Soma-se $\epsilon I$ com $\epsilon = 10^{-9}$ à diagonal, garantindo inversa e
determinante bem definidos.

**Geometria.** Como cada classe tem sua covariância, a fronteira é **quadrática**
— parábolas, elipses ou hipérboles no plano.

**Álgebra em Python puro.** Determinante e inversa saem de eliminação de
**Gauss-Jordan** implementada à mão (`core/math_utils.py`).

**Premissa a verificar.** O modelo assume normalidade multivariada. O projeto
testa isso com **Henze-Zirkler** e **Mardia** (pacote MVN do R, com fallback de
resultados pré-calculados quando o R não está instalado) — ver aba
*Bayes & Normalidade*.

*Código:* `models/bayes_classifier.py`, `evaluation/mvn_tester.py`.

### 4.5 Naive Bayes

Mesma formulação, com uma hipótese a mais: os atributos são **condicionalmente
independentes** dada a classe. A covariância vira diagonal:

$$d_j(x) = -\tfrac{1}{2}\sum_{i=1}^{d}\ln \sigma_{ji}^2
          -\tfrac{1}{2}\sum_{i=1}^{d}\frac{(x_i - m_{ji})^2}{\sigma_{ji}^2}$$

Determinante e inversa ficam triviais (produto e recíprocos das variâncias), e
as elipses de densidade ficam **alinhadas aos eixos**, sem rotação.

Na prática o Naive Bayes costuma empatar com o QDA — e às vezes superá-lo,
porque estima muito menos parâmetros ($d$ variâncias em vez de $d(d+1)/2$
covariâncias) e portanto sofre menos com amostras pequenas.

### 4.6 Rede Feedforward (MLP) com Backpropagation

**Ideia.** Empilhar neurônios em camadas para construir fronteiras não lineares.

**Neurônio.** $\text{net} = \sum w_ix_i + b$, seguido da sigmoide

$$\sigma(z) = \frac{1}{1+e^{-z}}, \qquad \sigma'(z) = \sigma(z)\,(1-\sigma(z))$$

A derivada em função da própria saída é o que torna o backpropagation barato.

**Feedforward.** Camada oculta $h = \sigma(W^{(1)}x + b^{(1)})$; camada de saída
$o = \sigma(W^{(2)}h + b^{(2)})$.

**Erro.** $E = \frac{1}{2}\sum_k (t_k - o_k)^2$.

**Retropropagação** — gradiente da saída para a entrada:

$$\delta_k^{\text{saída}} = (t_k - o_k)\,o_k(1-o_k)$$
$$\delta_j^{\text{oculta}} = h_j(1-h_j)\sum_k \delta_k^{\text{saída}} w_{kj}$$
$$w \leftarrow w + p\,\delta\,(\text{entrada do peso})$$

**Uso multiclasse neste projeto.** Duas providências transformam a rede dos
exercícios num classificador geral (`models/mlp_multiclasse.py`):

1. **Normalização min-max** das entradas para $[0,1]$, calculada **só com o
   treino** (usar o teste vazaria informação). Sem isso a sigmoide satura e o
   gradiente desaparece.
2. **Codificação 1-de-C** na saída: um neurônio por classe, alvo 1 na classe
   correta e 0 nas demais; decisão por $\arg\max$.

**Hiperparâmetros:** neurônios ocultos, taxa de aprendizado, épocas, semente.

**O que observar na defesa.** A curva de erro por época: descendo e estabilizando
significa que aprendeu; oscilando, taxa alta demais; parada alta, capacidade ou
épocas insuficientes.

*Código:* `models/mlp_backprop.py` (a rede), `models/mlp_multiclasse.py` (o
classificador).

### 4.7 Florestas Aleatórias — o modelo do seminário

**Árvore de decisão (CART).** Divide recursivamente o espaço por perguntas do
tipo $x_i \le t$, escolhendo a divisão que mais reduz a impureza:

$$\text{Gini}(S) = 1 - \sum_c p_c^2, \qquad
H(S) = -\sum_c p_c\log_2 p_c$$

$$\text{Ganho} = I(S) - \frac{|S_{\text{esq}}|}{|S|}I(S_{\text{esq}})
                        - \frac{|S_{\text{dir}}|}{|S|}I(S_{\text{dir}})$$

As fronteiras são **degraus paralelos aos eixos** — uma diferença visual
imediata em relação às retas da distância mínima e às curvas do Bayes.

**O problema da árvore única.** Cresce até folhas puras, decora o treino e muda
completamente se poucas amostras mudarem: variância alta.

**Duas fontes de aleatoriedade.**

1. **Bagging** — cada árvore treina numa amostra bootstrap (sorteio com
   reposição, do mesmo tamanho do treino). Em média, $1 - (1-1/n)^n \approx
   63{,}2\%$ das amostras entram; as restantes são as **out-of-bag**.
2. **Subespaço aleatório** — em cada nó, só um subconjunto sorteado de atributos
   (√p por padrão) pode ser usado na divisão. Sem isso, todas as árvores
   escolheriam o mesmo atributo forte no topo e ficariam correlacionadas.

**Decisão.** Voto majoritário das árvores; a proporção de votos serve de medida
de confiança.

**Erro out-of-bag.** Cada árvore é testada nas amostras que ficaram fora do seu
bootstrap. A média disso é uma estimativa de generalização **sem separar um
conjunto de validação** — validação de graça, embutida no bagging.

**Importância dos atributos.** Soma da redução de impureza atribuída a cada
atributo em todas as árvores, ponderada pelo número de amostras do nó.

**Hiperparâmetros:** nº de árvores, critério (Gini/entropia), profundidade
máxima, atributos sorteados por nó, mínimo de amostras por folha, semente.

**Versão categórica (ID3).** Para o dataset do seminário existe também uma
floresta com divisões *multi-way* por valor de atributo, no estilo ID3 dos
slides (`models/floresta_categorica.py`).

*Código:* `models/random_forest.py`. Teoria completa em
[`seminario_florestas_aleatorias.md`](seminario_florestas_aleatorias.md) e a
base em [`seminario_dataset_fim_de_semana.md`](seminario_dataset_fim_de_semana.md).

---

## 5. Métricas de qualidade

### 5.1 Matriz de confusão

Toda métrica nasce dela. Convenção do projeto: **linha = classe predita**,
**coluna = classe real**. A diagonal concentra os acertos.

### 5.2 Acerto global e o problema dele

$$A_g = \frac{\sum_i x_{ii}}{N}$$

Simples e enganoso: numa base com 95% de uma classe, o classificador que
responde sempre essa classe marca 95%. Daí as métricas seguintes.

### 5.3 Acerto casual, Kappa e Tau

$$A_c = \frac{1}{N^2}\sum_i x_{i+}\,x_{+i}, \qquad
\kappa = \frac{A_g - A_c}{1 - A_c}$$

O acerto casual estima o quanto se acertaria por sorte, dados os totais
marginais. O Kappa desconta isso: $\kappa = 1$ é perfeito, $0$ é o acaso,
negativo é pior que o acaso. Escala de Landis & Koch: > 0,80 quase perfeito;
0,61–0,80 substancial; 0,41–0,60 moderado; ≤ 0,40 fraco.

O **Tau** usa a mesma ideia, mas fixa o acaso em $1/C$ (classes equiprováveis):

$$\tau = \frac{A_g - 1/C}{1 - 1/C}$$

Quando os dois divergem, o Kappa é o mais confiável, porque usa a distribuição
real das predições em vez de supor uniformidade.

**Variâncias** (necessárias para o teste Z) — Kappa pela fórmula de Congalton &
Green (2009), com os quatro termos $\varphi_1..\varphi_4$; Tau por
$\sigma^2_\tau = \frac{1}{N}\cdot\frac{A_g(1-A_g)}{(1-1/C)^2}$.

### 5.4 Métricas por classe (One-vs-Rest)

Cada classe vira uma matriz 2×2 (VP, FP, FN, VN):

| Métrica | Fórmula | Pergunta que responde |
|---|---|---|
| Acurácia do produtor (revocação) | $VP/(VP+FN)$ | "Das que eram desta classe, quantas achei?" |
| Acurácia do usuário (precisão) | $VP/(VP+FP)$ | "Das que eu disse serem desta classe, quantas eram?" |
| Especificidade | $VN/(VN+FP)$ | "Das que não eram, quantas rejeitei?" |
| F1 | $2PR/(P+R)$ | Equilíbrio entre as duas |
| F-beta | $(1+\beta^2)PR/(\beta^2P + R)$ | F2 dá mais peso à revocação |

### 5.5 Coeficiente de Matthews (MCC)

É a correlação de Pearson entre predição e gabarito. Vale $+1$ na predição
perfeita, $0$ no acaso, $-1$ na inversão total:

$$\text{MCC} = \frac{VP\cdot VN - FP\cdot FN}
{\sqrt{(VP+FP)(VP+FN)(VN+FP)(VN+FN)}}$$

A versão **multiclasse** (Gorodkin, 2004) usa a matriz de confusão inteira, sem
passar por One-vs-Rest. O MCC é a métrica mais robusta a desbalanceamento do
conjunto — por isso é o padrão nos testes de significância do aplicativo.

### 5.6 Validação cruzada

k-fold estratificado com repetições. Reporta média, desvio, mínimo, máximo e o
intervalo de confiança:

$$\bar{A} \pm z\,\frac{s}{\sqrt{n}}$$

Além disso acumula a matriz de confusão de todas as dobras e recalcula Kappa e
Tau sobre ela — o que dá um Kappa muito mais estável que o do split único.

*Código:* `evaluation/metricas_avancadas.py`, `evaluation/validacao_cruzada.py`.
Teoria detalhada em [`lab_03/teoria_lab03.md`](lab_03/teoria_lab03.md).

---

## 6. Comparação de modelos e testes de significância

### 6.1 O ponto central

Dizer que A tirou 84,4% e B tirou 82,2% **não** significa que A é melhor: a
diferença pode ser variação amostral. Testar significância é responder "essa
diferença sobreviveria a outro conjunto de teste?".

### 6.2 Teste Z de Kappa (e de Tau)

$$Z = \frac{\kappa_A - \kappa_B}{\sqrt{\sigma^2(\kappa_A) + \sigma^2(\kappa_B)}}$$

- $H_0$: não há diferença entre os coeficientes.
- Bilateral, $\alpha = 5\%$: rejeita-se $H_0$ se $|Z| > 1{,}96$.
- p-valor $= 2(1 - \Phi(|z|))$, com $\Phi$ pela aproximação de
  Abramowitz & Stegun em Python puro.

**Limitação, e é importante saber dizê-la.** Somar as variâncias supõe que as
duas estimativas são independentes. Como os dois classificadores são avaliados
**no mesmo conjunto de teste**, eles acertam as mesmas amostras fáceis e erram
as mesmas difíceis: $\text{Cov}(\kappa_A, \kappa_B) > 0$. Ignorar esse termo
infla o denominador e torna o teste conservador demais. É por isso que o projeto
implementa também os três testes **pareados** abaixo.

### 6.3 McNemar

Monta a tabela 2×2 dos acertos pareados:

|  | B acertou | B errou |
|---|---|---|
| **A acertou** | a | b |
| **A errou** | c | d |

Só os discordantes $b$ e $c$ carregam informação: se os modelos fossem
equivalentes, esperaríamos $b \approx c$.

- $b + c \ge 25$: qui-quadrado com correção de continuidade de Edwards,
  $\chi^2 = (|b-c|-1)^2/(b+c)$, 1 grau de liberdade.
- $b + c < 25$: **teste binomial exato**, $2\,P(X \le \min(b,c))$ com
  $X \sim \text{Bin}(b+c, 0{,}5)$ — o caso do Iris, com 45 amostras de teste.
- $b + c = 0$: os modelos são idênticos amostra a amostra; não há o que testar.

### 6.4 Bootstrap pareado

Reamostra com reposição **os índices do conjunto de teste** (os mesmos índices
para os dois modelos, preservando o pareamento), recalcula a métrica escolhida
em cada reamostragem e acumula a distribuição da diferença
$\Delta = M_A - M_B$. O intervalo de confiança de 95% são os percentis 2,5 e
97,5. Se o IC **não contém zero**, a diferença é significativa.

Vantagem: funciona para qualquer métrica (MCC, Kappa, F1, acerto global…), sem
fórmula fechada de variância.

### 6.5 Teste de permutação

Sob $H_0$ os dois modelos são intercambiáveis. Então, para cada amostra,
sorteia-se se as predições de A e B trocam de lugar; recalcula-se a diferença;
repete-se milhares de vezes. O p-valor é a proporção de permutações cuja
diferença é tão extrema quanto a observada. É não paramétrico: não supõe
distribuição nenhuma.

### 6.6 Como ler o veredito

O aplicativo mostra os três resultados lado a lado, e eles costumam concordar.
Quando divergem, a leitura é:

- **McNemar significativo, bootstrap não** — os modelos erram amostras
  diferentes, mas o efeito sobre a métrica agregada é pequeno.
- **Bootstrap significativo, McNemar não** — a diferença aparece na métrica
  (por exemplo, concentrada numa classe), mas o total de acertos é parecido.
- **Nenhum significativo** — não há evidência de que os modelos difiram; diga
  exatamente isso, e não "os modelos são iguais". Não rejeitar $H_0$ não prova
  $H_0$.

*Código:* `evaluation/testes_significancia.py`. Detalhamento em
[`lab_03/testes_significancia.md`](lab_03/testes_significancia.md).

---

## 7. A base do usuário em .txt

Pipeline completo, em Python puro (`data/leitor_texto.py`):

```text
arquivo .txt  →  detecta delimitador  →  detecta cabeçalho  →  perfila colunas
      →  usuário confere e ajusta  →  codifica categorias  →  vira dataset
```

- **Delimitador**: vírgula, ponto e vírgula, tabulação, barra vertical ou
  espaços. Vence o candidato que divide as 50 primeiras linhas de forma mais
  consistente.
- **Cabeçalho**: detectado por dois sinais — primeira linha textual com corpo
  numérico; ou, em bases totalmente categóricas, rótulos que não reaparecem na
  própria coluna.
- **Coluna de classe**: sugerida por nome típico (`classe`, `label`, `target`,
  `decisao`…) ou pela última coluna categórica curta; sempre ajustável.
- **Atributos categóricos**: viram códigos $0..k-1$, com a tabela de rótulos
  preservada para a interface exibir `Sol` em vez de `0`.
- **Colunas ignoradas**: ids e codificações redundantes podem ser desmarcados.
- **Validações**: mínimo de 2 atributos, mínimo de 2 classes, tetos de linhas,
  colunas e classes; linhas com valores ausentes são descartadas com aviso.

A base importada é gravada em `data/enviados/` (texto + configuração de
leitura), sobrevive a reinícios do servidor e aparece no mesmo seletor do Iris.
Guia completo: [`importar_dados_txt.md`](importar_dados_txt.md).

---

## 8. Arquitetura do software

### 8.1 Camadas

```text
React + Vite  ──HTTP/JSON──►  FastAPI  ──chamadas diretas──►  Python puro
(web_app/frontend)            (web_app/backend)               (iris_classifier/)
   interface                   orquestração                    matemática
```

A regra que organiza tudo: **o backend não reimplementa matemática**. Os routers
carregam dados, chamam os módulos de `iris_classifier/` e serializam a resposta.
Isso é o que garante que a interface web, a GUI desktop e o CLI produzam
exatamente os mesmos números.

### 8.2 Registro de bases

`web_app/backend/core.py` mantém um registro em que cada base declara suas
classes, features e combinações de atributos. Nenhuma tela assume "as 3 classes
do Iris" — elas perguntam à base. É isso que permite o dataset categórico do
seminário (4 classes, 3 atributos) e as bases `.txt` do usuário rodarem nas
mesmas telas, sem código condicional espalhado.

### 8.3 Catálogo de modelos

`web_app/backend/modelos.py` é a peça nova mais importante. Cada modelo declara:

```python
'floresta': {
    'nome': 'Floresta Aleatória',
    'grupo': 'Seminário',
    'descricao': '…',
    'parametros': [ {…esquema de cada hiperparâmetro…} ],
    'treinar':  _floresta_treinar,
    'predizer': _floresta_predizer,
    'scores':   _floresta_scores,
}
```

Consequências práticas:

- a tela **Classificar** monta os controles a partir do esquema — não há
  formulário escrito à mão para cada modelo;
- os **testes de significância** e a **validação cruzada** varrem o mesmo
  catálogo, então um modelo novo entra nas comparações automaticamente;
- a validação de parâmetros é central: valores fora da faixa são trazidos para
  dentro dela, chaves desconhecidas ignoradas, ausentes recebem o padrão.

### 8.4 Caches

Três, todos por chave e invalidados quando as bases mudam:

1. leitura do arquivo (`carregar`);
2. split estratificado (`obter_split`);
3. predições de todos os modelos por (base, atributos, proporção) — sem ele, a
   matriz de significância de 21 pares retreinaria os sete modelos a cada par.

### 8.5 Memórias de cálculo

Cada laboratório tem um botão que abre a **memória de cálculo**: fórmula em
LaTeX, referência ao arquivo e à linha onde a função está implementada (via
módulo `inspect`) e a substituição numérica passo a passo. É o recurso a usar
quando a pergunta for "de onde saiu esse número?".

---

## 9. Resultados de referência

Split 70/30, semente 42. Percentual de acerto no conjunto de teste.

### 9.1 Iris, split único

| Modelo | Pétalas | Sépalas | 4 features |
|---|---|---|---|
| Distância Mínima | 100,0 | 82,2 | 97,8 |
| Perceptron OvA | 66,7 | 64,4 | 73,3 |
| Regra Delta OvA | 66,7 | 33,3 | 44,4 |
| Bayes Ótimo (QDA) | 100,0 | 80,0 | 97,8 |
| Naive Bayes | 100,0 | 84,4 | 97,8 |
| Rede Feedforward | 97,8 | 68,9 | 97,8 |
| Floresta Aleatória | 100,0 | 73,3 | 100,0 |

**Como explicar os 100%:** as pétalas separam as três espécies quase
perfeitamente — é uma propriedade da base, não um erro de implementação. Nas
sépalas as classes se sobrepõem e as diferenças aparecem.

**Como explicar os 33,3% da Regra Delta nas sépalas:** com classes sobrepostas e
escalas parecidas, os três discriminantes OvA ficam quase idênticos e o
$\arg\max$ colapsa numa classe só. É o limite do modelo linear, não um bug — e é
exatamente a motivação histórica das redes multicamadas.

### 9.2 Iris, validação cruzada (5 dobras × 5 repetições, pétalas)

| Modelo | Acurácia média |
|---|---|
| Bayes Ótimo (QDA) | 97,6% ± 2,5% |
| Distância Mínima | 96,3% ± 3,4% |
| Naive Bayes | 96,0% ± 3,6% |
| Floresta Aleatória | 95,5% ± 3,6% |
| Regra Delta OvA | 66,7% ± 0,0% |
| Perceptron OvA | 64,7% ± 3,5% |

O contraste com a tabela anterior é o argumento inteiro da validação cruzada:
o que parecia 100% vira 96–98% quando todas as amostras são testadas.

### 9.3 Efeito dos hiperparâmetros (demonstrações prontas)

**Sobreajuste da floresta (Iris, sépalas):**

| Profundidade máxima | Treino | Teste |
|---|---|---|
| 1 | 65,7% | 60,0% |
| 2 | 79,0% | 80,0% |
| 3 | 81,0% | 77,8% |
| sem limite | 94,3% | 73,3% |

Sem limite, a floresta decora o treino e perde 21 pontos no teste. Com
profundidade 2, treino e teste andam juntos.

**Número de árvores (Iris, sépalas):** 1 árvore → 64,4%; 10 → 71,1%; 50 → 73,3%;
200 → 73,3%. O ganho satura: acrescentar árvores reduz variância até um ponto, e
depois não muda mais.

**Atributos sorteados por nó (base do seminário, 3 atributos):** √p → 82,8%;
log₂ p → 82,8%; **todos → 94,4%**. Com só três atributos, sortear um por nó
cega a árvore; o parâmetro que normalmente ajuda, aqui atrapalha. É um ótimo
exemplo de que hiperparâmetro bom depende da base.

### 9.4 Base do seminário (fim de semana, 1000 amostras, 3 atributos)

| Modelo | Acerto |
|---|---|
| Bayes / Naive / Rede | 94,4% |
| Floresta Aleatória (√p) | 82,8% |
| Distância Mínima | 70,9% |
| Perceptron / Delta OvA | 11,9% |

Dois pontos para comentar: o teto de ~94% é o **ruído de 8%** injetado na
geração da base (erro irredutível); e o desastre dos modelos lineares acontece
porque os atributos são categóricos codificados em inteiros — a ordem
`Sol < Vento < Chuva` não significa nada, e um hiperplano sobre esses códigos
não tem sentido geométrico.

---

## 10. Roteiro de fala e perguntas prováveis

### 10.1 Divisão sugerida

| Parte | Conteúdo |
|---|---|
| Abertura | O que o app faz e como ele atende aos quatro pedidos da entrega |
| Demonstração | O roteiro de 10 minutos da [§2.2](#22-roteiro-de-10-minutos) |
| Teoria | O modelo do seminário (florestas) e um modelo clássico para contraste |
| Avaliação | Métricas e por que os testes de significância são necessários |
| Fechamento | Arquitetura: catálogo de modelos e a regra do Python puro |

### 10.2 Perguntas prováveis

**"Por que quase tudo dá 100%?"**
Porque nas pétalas o Iris é quase linearmente separável. Mostre as sépalas
(80% ± ) e a validação cruzada (96–98%): o 100% é do split, não do modelo.

**"Como o usuário troca de modelo?"**
Na aba Classificar, no seletor de modelo; os controles de parâmetro mudam junto,
porque são gerados a partir do esquema que cada modelo publica.

**"E se eu trouxer minha própria base?"**
Importar .txt, conferir a leitura detectada e importar. A base vale em todas as
telas, inclusive nos testes de significância.

**"Qual a diferença entre o teste Z e o McNemar?"**
O Z compara duas estatísticas agregadas supondo independência; como os dois
modelos são avaliados no mesmo conjunto, existe covariância positiva e o Z fica
conservador. O McNemar é pareado: olha amostra a amostra, só os discordantes.

**"Por que o teste binomial exato e não o qui-quadrado?"**
Porque com 45 amostras os discordantes costumam ser menos de 25, regime em que a
aproximação qui-quadrado é ruim. O app escolhe sozinho e informa o método usado.

**"Por que o Perceptron não converge?"**
Só existe garantia de convergência para classes linearmente separáveis. Na
decomposição Um-Contra-Todos, "versicolor contra o resto" não é separável — a
interface mostra o selo de convergência por classe.

**"Qual a vantagem da floresta sobre uma árvore?"**
Reduz variância. Uma árvore decora o treino; o comitê média os erros
independentes. E o OOB dá uma estimativa de generalização sem gastar dados com
validação.

**"O que é o erro out-of-bag?"**
Cada árvore vê ~63% das amostras; as ~37% restantes servem para testá-la. A
média desses testes é o OOB.

**"Vocês usaram numpy?"**
Não. Toda a matemática é Python puro — inclusive inversão de matrizes por
Gauss-Jordan, backpropagation e a leitura do `.txt`. A única exceção é o item
(ii) do Lab 5, onde o enunciado permitia scikit-learn, e que está isolado num
módulo separado.

**"Como vocês sabem que a implementação está correta?"**
Três frentes: as memórias de cálculo reproduzem passo a passo os exemplos dos
slides; os exercícios da aula (XOR, figura 12.32, galinha vs homem) batem com os
valores esperados; e os classificadores concordam entre si nas bases onde
deveriam concordar.

**"Por que o MCC e não a acurácia?"**
Porque a acurácia mente com classes desbalanceadas. O MCC usa as quatro células
da matriz 2×2 (ou a matriz inteira, na versão multiclasse) e só fica alto quando
o modelo vai bem em todas as classes.

---

## 11. Mapa da documentação

| Documento | Conteúdo |
|---|---|
| [`classificar_modelos.md`](classificar_modelos.md) | Tela de escolha do modelo e o que cada hiperparâmetro faz |
| [`importar_dados_txt.md`](importar_dados_txt.md) | Formato do `.txt`, heurísticas de leitura, limites, problemas comuns |
| [`interface_web.md`](interface_web.md) | Arquitetura da interface web, todas as rotas da API |
| [`teoria_completa.md`](teoria_completa.md) | Teoria dos classificadores lineares (Labs 1 e 2) |
| [`formulario.md`](formulario.md) | Folha de fórmulas para consulta rápida |
| [`lab_03/teoria_lab03.md`](lab_03/teoria_lab03.md) | Métricas avançadas: Kappa, Tau, variâncias, teste Z |
| [`lab_03/testes_significancia.md`](lab_03/testes_significancia.md) | McNemar, bootstrap pareado, permutação, MCC multiclasse |
| [`lab_04/teoria_lab04.md`](lab_04/teoria_lab04.md) | Bayes Ótimo, Naive Bayes, normalidade multivariada |
| [`lab_05/teoria_lab05.md`](lab_05/teoria_lab05.md) | MLP, backpropagation, XOR |
| [`seminario_florestas_aleatorias.md`](seminario_florestas_aleatorias.md) | Teoria completa do seminário |
| [`seminario_dataset_fim_de_semana.md`](seminario_dataset_fim_de_semana.md) | A base categórica do seminário |
| [`README.md`](README.md) | Índice de toda a documentação, organizado por laboratório |
