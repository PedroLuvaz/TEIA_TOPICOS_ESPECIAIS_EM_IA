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
