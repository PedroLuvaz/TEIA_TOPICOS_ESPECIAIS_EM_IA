# Guia de Explicação para o Professor

Este documento serve como um guia para apresentar e explicar o projeto. Ele detalha a lógica implementada, as escolhas arquiteturais e a matemática por trás de cada classificador.

**Conteúdo coberto:**
- §1–§10: Distância Mínima (Aba 1) — split, protótipos, discriminante, fronteiras, GUI
- §11–§14: Perceptron & Regra Delta & XOR (Aba 2) — teoria, resultados, demonstração
- §15–§16: Bayes Ótimo & Naive Bayes (Aba 4) — normalidade multivariada, teste Z
- §18–§20: Feedforward (MLP) & Backpropagation (Aba 5) — Lab 5

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

### Núcleo (Python puro — sem bibliotecas de ML)

| Arquivo | Responsabilidade | Matemática central |
|---|---|---|
| `math_utils.py` | Toda a álgebra linear em Python puro | `produto_escalar`, `distancia_euclidiana`, `discriminante`, `coeficientes_superficie_decisao` |
| `data_loader.py` | Leitura do XLS + split estratificado | Agrupamento por classe, shuffle com `seed=42` |
| `classifier.py` | Treinamento e predição | `treinar` → protótipos; `predizer_todas_classes` → argmax $d_j(x)$; `predizer_binario` → argmin distância |
| `evaluator.py` | Métrica de avaliação | Acurácia |
| `visualizer.py` | Gráficos matplotlib (saída em `outputs/`) | Dispersão, superfícies de decisão, heatmap de confusão |
| `main.py` | Orquestrador CLI | Executa experimentos i, ii, iii + comparativo + interativo |

### Camada de apresentação (GUI Tkinter)

| Arquivo | Responsabilidade |
|---|---|
| `run_gui.py` | Ponto de entrada da interface (`python iris_classifier/run_gui.py`) |
| `gui/app.py` | Janela principal (cabeçalho + notebook de 4 abas + rodapé) |
| `gui/theme.py` | Paleta editorial escura, tipografia (Cambria/Segoe/Consolas), estilos `ttk` |
| `gui/widgets.py` | Componentes reutilizáveis (`Card`, `MetricBlock`) |
| `gui/tab_distancia_minima.py` | Aba ativa — controles, gráfico embarcado, métricas dinâmicas, análise textual |
| `gui/janela_calculos.py` | **Janela de Memória de Cálculo** (fórmulas LaTeX renderizadas via mathtext + substituição numérica passo a passo) |

**Ponto importante:** `math_utils.py` não conhece nada de Iris — é uma biblioteca de álgebra genérica. A camada GUI **não duplica** matemática: ela importa e chama os módulos puros. Isso é separação de responsabilidades.

---

## 8. Demonstração Interativa: Interface Gráfica

Além do modo CLI (`python iris_classifier/main.py`), o projeto inclui uma interface gráfica completa em **Tkinter + matplotlib**, executada com:

```bash
python iris_classifier/run_gui.py
```

### O que a interface oferece

A janela principal usa um **notebook de abas** preparado para receber implementações futuras do projeto. A aba ativa, **Distância Mínima**, mostra:

| Painel | Conteúdo |
|---|---|
| **Atributos do Modelo** | Toggle entre Pétalas `[2,3]` e Sépalas `[0,1]` — ao trocar, o modelo é re-treinado e todos os painéis atualizam |
| **Visualização** | Toggle entre dispersão geral e fronteira de cada par (3 pares de classes) |
| **Classificar Amostra** | Entrada manual de valores `(x₁, x₂)` para classificar uma nova amostra |
| **Predição** | Resultado destacado com a classe vencedora, valor de `d_max` e scores das outras classes |
| **Memória de Cálculo** | Botão que abre janela secundária com fórmulas matemáticas e substituição numérica |
| **Acurácia teste** | Cor dinâmica: verde (≥95%), âmbar (≥80%), vermelho (<80%) |
| **Erros base completa** | Mostra erros do modelo nas 150 amostras (treino + teste) — revela o overlap real |
| **Análise** | Texto explicativo gerado dinamicamente sobre separabilidade vs sobreposição |

### Pontos didáticos para apresentação

1. **Mostre a alternância Pétalas ↔ Sépalas:**
   - Pétalas: 100% de acurácia no teste, 5 erros na base completa (todos versicolor↔virginica)
   - Sépalas: 82.22% no teste, 27 erros na base completa
   - O texto da Análise muda automaticamente explicando o porquê.

2. **Mostre as fronteiras dos pares:**
   - `Setosa × Versicolor` e `Setosa × Virginica` — fronteira clara, regiões bem separadas
   - `Versicolor × Virginica` — pontos atravessam a linha (são os erros do modelo)

3. **Classifique uma amostra de borda:**
   - Com pétalas: `(5.0, 1.7)` é caso ambíguo — ver os 3 scores discriminantes lado a lado.

---

## 9. Janela de Memória de Cálculo

**Onde encontrar no código:** `iris_classifier/gui/janela_calculos.py`

Este é o painel mais importante para defender o domínio matemático ao professor: ele mostra as **fórmulas em LaTeX renderizadas** (via `matplotlib.mathtext`) e logo abaixo a **substituição numérica** com os valores reais do modelo treinado.

### Estrutura das 4 seções

**Seção 1 — Protótipos (Vetores Médios)**
- Fórmula: $m_j = \frac{1}{N_j} \sum_{x \in \omega_j} x$
- Substituição para cada classe:
  ```
  N_set = 35   (amostras de treino)
  m_set = (1/35) · [Σ Comp.Pétala, Σ Larg.Pétala]
        = [1.4800, 0.2486]
  ```

**Seção 2 — Função Discriminante**
- Fórmulas:
  - $d_j(x) = x^T m_j - \frac{1}{2} m_j^T m_j$
  - $j^* = \arg\max_j\, d_j(x)$
- Substituição completa com $x = [4.5, 1.5]$:
  ```
  classe versicolor:
    m_ver = [4.2371, 1.3229]
    x · m_ver = 4.50·4.2371 + 1.50·1.3229 = 21.0514
    m_ver · m_ver = 4.2371² + 1.3229² = 19.7033
    d_ver(x) = 21.0514 - ½·19.7033 = +11.1998
  ```
- Resultado destacado: `argmax → VERSICOLOR (d = +11.1998)`

**Seção 3 — Equivalência Argmax ≡ Argmin**
- Fórmulas:
  - $\|x - m_j\| = \sqrt{\sum_k (x_k - m_{jk})^2}$
  - $\arg\max_j\, d_j(x) \equiv \arg\min_j\, \|x - m_j\|$
  - Expansão: $\|x - m_j\|^2 = x^T x - 2\,x^T m_j + m_j^T m_j$
- Validação numérica lado a lado: o maior `d_j(x)` corresponde à menor `||x - m_j||`.

**Seção 4 — Fronteiras de Decisão (3 pares)**
- Fórmulas:
  - $w = m_i - m_j$
  - $b = -\frac{1}{2}(\|m_i\|^2 - \|m_j\|^2)$
  - $x_2 = \frac{-w_1 x_1 - b}{w_2}$ (forma plotável)
- Para cada par:
  ```
  PAR setosa × versicolor:
    m_set = [1.4800, 0.2486]    m_ver = [4.2371, 1.3229]
    w = [-2.7571, -1.0743]
    ||m_set||² = 2.2522    ||m_ver||² = 19.7033
    b = +8.7256
    Equação: -2.7571 x1 -1.0743 x2 +8.7256 = 0
    Reta:    x2 = -2.5665·x1 + 8.1222
  ```

### Por que isso é poderoso para a apresentação

1. **Você está provando que entende a matemática**, não só rodando código. As mesmas fórmulas do `formulario.md` aparecem na tela com substituição numérica em tempo real.
2. **Transparência total:** o professor pode pedir "calcula d_versicolor para [4.5, 1.5]" e a janela já mostra cada passo.
3. **Conecta as 4 representações** (protótipo, discriminante, distância, fronteira) numa visualização única.
4. **Atualiza dinamicamente:** trocou pétalas para sépalas, todos os números recalculam.

---

## 10. Futura Migração para Bibliotecas (NumPy / Scikit-learn)

O projeto foi intencionalmente construído sem bibliotecas de ML para demonstrar domínio matemático. Em uma versão futura, cada função pura tem um equivalente direto:

| Função Pura | Equivalente NumPy/Sklearn |
|---|---|
| `produto_escalar(a, b)` | `np.dot(a, b)` |
| `distancia_euclidiana(a, b)` | `np.linalg.norm(np.array(a) - np.array(b))` |
| `calcular_media(vetores)` | `np.mean(X, axis=0)` |
| `discriminante(x, mj)` | `np.dot(x, mj) - 0.5 * np.dot(mj, mj)` |
| `treinar_perceptron(...)` | `sklearn.linear_model.Perceptron().fit(X, y)` |
| `treinar_delta_iris(...)` | `sklearn.linear_model.SGDClassifier(loss='squared_error').fit(X,y)` |
| `treinar_delta_xor(...)` | Não existe equivalente direto para demonstração didática |

**Como apresentar ao professor:** "A estrutura modular foi pensada para facilitar a migração. Os módulos `math_utils.py`, `perceptron.py` e `delta_rule.py` podem ser trocados por NumPy/sklearn sem mudar nada no restante do código — basta substituir as funções uma a uma."

---

## 11. Aba 2 — Perceptron & Regra Delta: O que Implementamos

### Ponto Chave de Apresentação

A Aba 2 implementa **três experimentos do PR4**, todos em Python puro:

1. **Perceptron de Rosenblatt** — classificador binário com aprendizado por correção de erros
2. **Regra Delta (Adaline / Widrow-Hoff)** — gradiente descendente na superfície MSE
3. **XOR com Regra Delta** — demonstração do limite fundamental dos classificadores lineares

**Por que isso é importante?** Enquanto a Aba 1 calculava protótipos em um único passo analítico, a Aba 2 mostra o aprendizado iterativo — o professor pode ver o processo de convergência em tempo real nos gráficos.

### Arquivos implementados

| Arquivo | Responsabilidade |
|---|---|
| `iris_classifier/perceptron.py` | Perceptron puro: sgn, regra de atualização, convergência |
| `iris_classifier/delta_rule.py` | Regra Delta pura: gradiente MSE, XOR |
| `iris_classifier/gui/tab_perceptron_delta.py` | Aba 2: controles, 2 subplots, métricas, análise |

---

## 12. Demonstração da Aba 2: Roteiro de Apresentação

Execute: `python iris_classifier/run_gui.py` e clique na aba **Perceptron & Delta**.

### Experimento 1 — Perceptron, dados separáveis

1. **Algoritmo:** Perceptron | **Par:** Setosa × Versicolor | **Atributos:** Pétalas
2. **Hiperparâmetros:** p = 0,03 | Max. Épocas = 100
3. Clicar em **Treinar**

**O que mostrar:**
- O gráfico direito (Convergência): o número de erros **cai para zero em 6 épocas** — convergência garantida pelo Teorema de Rosenblatt para dados separáveis
- O gráfico esquerdo (Dispersão): a linha âmbar separa perfeitamente o azul (Setosa) do verde (Versicolor)
- Métricas: **Acurácia 100%** · Épocas **6 / 100** · Convergência: **Convergiu**

**Frase para o professor:** *"O Perceptron convergiu em apenas 6 épocas porque Setosa e Versicolor são linearmente separáveis com as pétalas. O Teorema da Convergência de Rosenblatt garante que sempre haverá convergência em tempo finito quando os dados são separáveis."*

---

### Experimento 2 — Perceptron, dados sobrepostos

1. **Par:** Versicolor × Virginica | restante igual
2. Clicar em **Treinar**

**O que mostrar:**
- Gráfico de convergência: os erros **nunca chegam a zero** — o algoritmo oscila até atingir `max_epocas`
- Status abaixo do botão: *"Limite 100 épocas (não convergiu)"* em vermelho
- Métricas: Acurácia **~50–80%** · Convergência: **Erros: X**

**Frase para o professor:** *"Versicolor e Virginica têm amostras biologicamente fronteiriças que se cruzam no espaço das pétalas. Isso é exatamente o que vimos na Aba 1 (os 5 erros). O Perceptron detecta a impossibilidade e não para — corolário direto do Teorema da Convergência."*

---

### Experimento 3 — Regra Delta, comparação com o Perceptron

1. **Algoritmo:** Regra Delta | **Par:** Setosa × Versicolor | **Pétalas** | p = 0,02 | 200 épocas
2. Clicar em **Treinar**

**O que mostrar:**
- Gráfico de convergência: curva MSE **decrescendo suavemente** (parabolóide convexa)
- Comparação com Perceptron: convergência mais lenta (200 vs 6 épocas) mas **garantida pela teoria**

3. Trocar para **Versicolor × Virginica** e treinar novamente

**O que mostrar:**
- O MSE ainda converge (embora a um valor positivo)
- Diferença do Perceptron: a Regra Delta **não oscila**, encontra o melhor compromisso linear
- **Frase:** *"A Regra Delta usa a saída linear (net) para calcular o erro, não o limiar (sgn). Isso cria uma superfície de erro quadrática convexa — sempre tem mínimo global, independente da separabilidade dos dados."*

---

### Experimento 4 — XOR: o limite dos classificadores lineares

1. **Algoritmo:** XOR (Delta) | p = 0,02 | Max. Épocas = 300
2. Clicar em **Treinar**

**O que mostrar:**
- Gráfico esquerdo: os 4 pontos XOR — {(0,0),(1,1)} em azul vs {(0,1),(1,0)} em coral
- A linha âmbar (fronteira linear) **não separa os dois grupos** — é impossível
- Gráfico direito: MSE converge para **~0,25** (o mínimo teórico provado matematicamente)
- Texto de análise: explica a prova de inseparabilidade e a solução via MLP

**Frase para o professor:** *"O XOR é o problema que motivou a pesquisa em redes multicamada. A Regra Delta encontra o melhor classificador linear possível — mas esse melhor tem MSE = 0,25, confirmando matematicamente a impossibilidade. A solução exige uma camada oculta com ativação não-linear."*

---

## 13. Matemática Central da Aba 2 (resumo para apresentação)

### Perceptron — por que o erro é (d − y)?

Queremos que a saída $y = \text{sgn}(w^T x)$ concorde com o alvo $d$. Quando erram:

- Se $d = +1$ e $y = -1$: o vetor de pesos precisa crescer em direção a $x$ → soma $+2 \cdot p \cdot x$
- Se $d = -1$ e $y = +1$: o vetor de pesos precisa diminuir → soma $-2 \cdot p \cdot x$

Isso é exatamente $(d - y) \cdot x = (\pm 2) \cdot x$.

**No código (`perceptron.py`):**
```python
if y != d_val:
    delta = taxa_aprendizado * (d_val - y)   # ±2p
    w = [wi + delta * xi for wi, xi in zip(w, x_aug)]
```

### Regra Delta — por que usar net e não sgn?

A função sgn não é diferenciável — seu gradiente é zero em quase todo lugar, impossibilitando gradiente descendente. Usando net diretamente:

$$E(w) = (d - w^T x)^2 \quad \Rightarrow \quad \frac{\partial E}{\partial w} = -2(d - w^T x) \cdot x$$

$$w \leftarrow w - \frac{p}{2} \cdot \frac{\partial E}{\partial w} = w + p(d - \text{net}) \cdot x$$

**No código (`delta_rule.py`):**
```python
net = sum(wi * xi for wi, xi in zip(w, x_aug))
erro = d_val - net            # erro linear, não limiarizado
w = [wi + taxa_aprendizado * erro * xi for wi, xi in zip(w, x_aug)]
```

---

## 14. Pontos Didáticos para Defender para o Professor

1. **Progressão pedagógica:** Distância Mínima (sem iteração) → Perceptron (iteração por erro) → Delta Rule (gradiente) → XOR (limite linear) — a sequência mostra evolução histórica e conceitual do campo

2. **Python puro em tudo:** `perceptron.py` e `delta_rule.py` usam apenas `sum()`, `zip()` e operações com listas — zero numpy. O professor pode auditar linha a linha

3. **Convergência como experimento:** O aluno pode ver ao vivo a curva de convergência mudar ao trocar o par de classes — não só "saber" que Versicolor×Virginica não converge, mas **ver** o gráfico oscilar

4. **XOR como ponte:** A demonstração experimental do MSE = 0,25 valida matematicamente a prova teórica de inseparabilidade — conecta teoria à prática

5. **Mesmo dataset, resultados diferentes:** Usar o Iris em ambas as abas mostra como o mesmo problema pode ser abordado com algoritmos distintos, facilitando comparação direta

---

## 15. Classificadores Probabilísticos (Aba 4 — Bayes & Naive Bayes)

**Ponto Chave de Apresentação:** Explique que saímos dos classificadores lineares baseados puramente em distâncias rígidas ou hiperplanos de separação e entramos no domínio dos **classificadores baseados em densidades de probabilidade**.

### Passo A: Aderência à Normalidade Multivariada
*   **A premissa:** Para usar a densidade condicional normal, precisamos verificar se os dados de cada classe $C_j$ de fato seguem uma distribuição normal multivariada.
*   **O Teste:** Usamos o ambiente R (via pacote `MVN`) para rodar os testes de **Henze-Zirkler** (HZ) e **Mardia** (assimetria e curtose).
*   **O Resultado no Iris:** 
    *   *Setosa* rejeita levemente a normalidade multivariada estrita ($p = 0.0496$), embora passe em assimetria e curtose.
    *   *Versicolor* e *Virginica* têm forte aderência ($p$-valores $> 0.05$).
*   **O Fallback:** Se o R não estiver instalado no computador atual, o sistema detecta isso dinamicamente e exibe os resultados reais precalculados gerados pelo R, mantendo a experiência fluida sem quebrar a execução.

### Passo B: Bayes Ótimo (QDA) vs Naive Bayes
*   **Bayes Ótimo:** Estima a matriz de covariância completa $\Sigma_j$ para cada classe. A fronteira de decisão resultante é **quadrática** (curva não-linear) porque cada classe tem sua própria covariância.
*   **Naive Bayes:** Assume independência condicional completa entre as variáveis, o que zera todos os termos fora da diagonal principal de $\Sigma_j$. A fronteira resultante torna-se elipsoidal e ortogonal aos eixos.
*   **Implementação em Python Puro:** Mostre ao professor que a inversão de matrizes e determinantes foi feita à mão usando **eliminação de Gauss-Jordan com pivotamento parcial** e **expansão de cofatores de Laplace** em `math_utils.py` (sem NumPy).

---

## 16. Teste de Significância de Kappa (Item e)

**Onde encontrar no código:** `metricas_avancadas.py` e `main.py`

**Como funciona:**
Para verificar se um classificador é estatisticamente melhor que outro, não basta comparar a acurácia bruta, pois a diferença pode ser fruto do acaso. Usamos o **Teste Z de diferença de Kappas** (Congalton & Green, 2009):

$$Z = \frac{K_1 - K_2}{\sqrt{\text{Var}(K_1) + \text{Var}(K_2)}}$$

No conjunto de teste (split 70/30):
*   Ambos os modelos obtiveram acurácia de **$97.78\%$** e Kappa $K = 0.9667$ no teste.
*   A estatística $Z = 0.0000$, com $p\text{-valor} = 1.0000$.
*   **Veredito:** Não há diferença estatisticamente significativa entre as duas acurácias ao nível de 5%. Ambos os modelos são equivalentes para esta partição.

---

## 17. Defesa do Projeto ao Professor (Novos Pontos)

1.  **Matemática Linear e Inversão Matricial em Python Puro:** Mostre a implementação de `inv_matriz` usando pivotamento parcial. Isso prova que o grupo compreende álgebra linear computacional avançada, fundamental para processamento numérico estável.
2.  **Integração Multilinguagem (Python + R):** O projeto conecta-se ao R escrevendo scripts intermediários e executando-os em subprocessos, consumindo arquivos CSV de intercâmbio. Essa integração dinâmica é valorizada na prática científica.
3.  **Fronteiras de Decisão Não-Lineares no Plano (Contour Plots):** Exiba os gráficos da pasta `outputs/` mostrando as fronteiras parabólicas e hiperbólicas geradas pelos contornos das funções de decisão log-probabilísticas de Bayes. Isso contrasta perfeitamente com as fronteiras retas das abas anteriores.
4.  **Memória de Cálculo LaTeX Dinâmica:** A janela secundária de memória de cálculo para a aba Bayes apresenta no formato LaTeX os vetores de médias calculados e as matrizes de covariância estimadas da classe, permitindo ao professor auditar as contas.

---

## 18. Redes Neurais Multicamadas (Aba 5 — Feedforward/Backpropagation)

**Ponto Chave de Apresentação:** Depois dos classificadores lineares e probabilísticos, o Lab 5 introduz **redes neurais multicamadas (MLP)**, treinadas pelo algoritmo de **retropropagação do erro (backpropagation)** — a base de todo o aprendizado profundo moderno.

### Item (i) — Rede "Galinha vs Homem" (Python puro)
*   **O que é:** uma rede 2 entradas → 2 neurônios ocultos → 2 neurônios de saída, com ativação sigmoide em ambas as camadas, usando os pesos iniciais exatos do slide da Aula PR_711.
*   **Onde encontrar no código:** `iris_classifier/models/mlp_backprop.py` (classe `RedeFeedforward`) e `iris_classifier/lab05_galinha_homem.py`.
*   **Prova de correção:** a alimentação adiante (forward) da implementação reproduz **exatamente** os valores do slide ($\text{out}_{b_1}=0{,}7020$, $\text{out}_{b_2}=0{,}5841$, $\text{out}_{c_1}=0{,}5934$, $\text{out}_{c_2}=0{,}7353$, $E=0{,}21108$) — mostre isso ao professor como evidência de que a matemática do backprop foi implementada corretamente do zero.
*   **Como apresentar:** clique em "Rodar 1 passo de treinamento" na Aba 5 e mostre que o erro total cai de $0{,}21107$ para $0{,}20894$ após uma única atualização de pesos — a rede está de fato aprendendo na direção correta do gradiente.

### Item (ii) — Feedforward vs Bayes Ótimo vs Naive Bayes (Iris)
*   **Única exceção à regra "sem bibliotecas de ML":** o próprio enunciado permite `scikit-learn` apenas para este experimento. Isso está isolado em `iris_classifier/models/mlp_sklearn.py` — o restante do projeto (incluindo o item i deste mesmo laboratório) continua 100% Python puro.
*   **Resultado:** a rede feedforward atingiu $100\%$ de acurácia no conjunto de teste, contra $97{,}78\%$ de Bayes Ótimo e Naive Bayes (ambos erram a mesma única amostra de *versicolor*).
*   **Ponto sutil e importante:** apesar da diferença numérica, o **teste Z de Kappa** mostra que essa diferença **não é estatisticamente significativa** ($Z=1{,}0234$, $p=0{,}306$) — 45 amostras de teste são poucas para diferenciar com confiança um único erro a mais ou a menos. Isso é uma ótima oportunidade de discutir com o professor a diferença entre "melhor na prática" e "melhor com significância estatística".

---

## 19. Estrutura Modular do Lab 5

| Arquivo | Responsabilidade | Restrição |
|---|---|---|
| `iris_classifier/models/mlp_backprop.py` | Rede feedforward + backprop do zero (item i) | Python puro |
| `iris_classifier/lab05_galinha_homem.py` | Script demonstrativo do item (i) | Python puro |
| `iris_classifier/models/mlp_sklearn.py` | Wrapper do `MLPClassifier` para o Iris (item ii) | scikit-learn permitido |
| `iris_classifier/main.py` (`experimento_mlp_iris`) | Orquestra item (ii): treina os 3 modelos, métricas, testes Z | reaproveita `evaluation/` |
| `iris_classifier/gui/tab_feedforward.py` | Aba 5 da GUI | — |

---

## 20. Defesa do Lab 5 ao Professor

1.  **Progressão pedagógica completa:** Distância Mínima → Perceptron/Delta → Bayes → **Feedforward/Backprop** — do classificador linear mais simples até a rede neural multicamada, cobrindo toda a evolução histórica do reconhecimento de padrões estudada na disciplina.
2.  **Regra da cadeia em ação:** o backprop implementado em `mlp_backprop.py` é uma aplicação direta e auditável da regra da cadeia do cálculo diferencial — cada `delta` no código corresponde exatamente a um termo da derivação do slide.
3.  **Uso disciplinado de bibliotecas:** o projeto usa `scikit-learn` em **um único lugar** (`mlp_sklearn.py`), exatamente onde o enunciado permite, e em nenhum outro — mostra domínio de quando uma biblioteca é apropriada e quando a implementação do zero é exigida.
4.  **Significância estatística acima da acurácia bruta:** o teste Z de Kappa entre os 3 modelos reforça uma lição central da disciplina — comparar acurácias brutas sem teste de hipótese pode levar a conclusões equivocadas sobre qual modelo é "melhor".

