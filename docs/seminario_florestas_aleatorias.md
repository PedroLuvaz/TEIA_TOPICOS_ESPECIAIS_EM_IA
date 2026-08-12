# Seminário — Florestas Aleatórias (Random Forests)

**Disciplina:** Tópicos Especiais em Inteligência Artificial (TEIA) · UEPB 2026
**Grupo:** Erick Nathan · Laura Barbosa · Pedro Lucas
**Referência principal:** Breiman, L. (2001). *Random Forests*. Machine Learning, 45(1), 5–32.
**Implementação:** `iris_classifier/models/random_forest.py` — Python puro, sem `scikit-learn`
**Interface:** aba *Florestas Aleatórias* (`web_app/frontend/src/pages/Floresta.tsx`)

---

## 1. O problema que as florestas resolvem

Uma árvore de decisão sozinha tem uma qualidade e um defeito:

- **Qualidade:** é interpretável. Dá para ler a árvore e explicar exatamente
  por que uma amostra foi classificada daquele jeito.
- **Defeito:** tem **variância alta**. Pequenas mudanças no conjunto de treino
  produzem árvores bem diferentes — trocar poucas amostras muda o atributo
  escolhido na raiz, e daí toda a estrutura abaixo.

A ideia de Breiman é combinar muitas árvores **propositalmente diferentes
entre si** e decidir por voto da maioria. A média de muitos preditores com
erros pouco correlacionados tem variância menor que a de cada um isolado, sem
aumentar o viés.

> [!IMPORTANT]
> O ganho depende de as árvores serem **descorrelacionadas**. Se todas
> chegassem à mesma estrutura, votar não acrescentaria nada. Por isso a
> Floresta Aleatória usa duas fontes independentes de aleatoriedade.

---

## 2. Árvore de decisão — a peça base

### 2.1 Impureza

Uma divisão é boa quando separa as classes: os nós filhos ficam mais "puros"
que o pai. A impureza mede o quanto as classes estão misturadas num nó.

**Índice de Gini** (padrão):

$$G = 1 - \sum_{k} p_k^2$$

**Entropia:**

$$H = -\sum_{k} p_k \log_2 p_k$$

Ambas valem 0 num nó puro. Com 3 classes equilibradas, o máximo do Gini é
$1 - 1/3 = 0{,}6667$ e o da entropia é $\log_2 3 = 1{,}585$.

Na prática as duas produzem árvores muito parecidas; o Gini é mais barato de
calcular (não tem logaritmo).

### 2.2 Ganho de uma divisão

Para uma divisão que parte o nó em esquerda e direita:

$$\text{ganho} = I(\text{pai}) - \left[\frac{n_{esq}}{n} I(\text{esq}) + \frac{n_{dir}}{n} I(\text{dir})\right]$$

O algoritmo (CART) testa, para cada atributo, os **pontos médios entre valores
consecutivos distintos** e escolhe a divisão de maior ganho. Testar só os
pontos médios basta: qualquer limiar entre dois valores consecutivos produz
exatamente a mesma partição.

### 2.3 Geometria das fronteiras

Cada divisão é do tipo `atributo ≤ limiar` — sempre **paralela a um eixo**.
Por isso as fronteiras de uma árvore são **escadas de retângulos**, nunca
retas oblíquas ou curvas.

Este é o contraste mais visível com os laboratórios anteriores: Distância
Mínima e Perceptron produzem retas em qualquer ângulo; Bayes produz cônicas;
a floresta produz degraus alinhados aos eixos.

---

## 3. As duas fontes de aleatoriedade

### 3.1 Bagging (bootstrap aggregating)

Cada árvore treina numa amostra sorteada **com reposição** do conjunto
original, do mesmo tamanho. Algumas amostras entram repetidas, outras não
entram.

A probabilidade de uma amostra específica ficar de fora é:

$$P(\text{fora}) = \left(1 - \frac{1}{n}\right)^{n} \xrightarrow[n \to \infty]{} e^{-1} \approx 0{,}368$$

Ou seja, cada árvore vê em média **63,2%** das amostras distintas, e **36,8%**
ficam de fora — as amostras *out-of-bag* (OOB) daquela árvore.

> [!TIP]
> Na aba do seminário, o painel "a floresta em números" mostra a proporção
> medida. Com o Iris ela fica em torno de 63,3% — batendo com a teoria.

### 3.2 Subespaço aleatório de atributos

Em cada nó, a busca pela melhor divisão considera apenas $m$ atributos
sorteados dos $p$ disponíveis, tipicamente:

$$m = \lfloor\sqrt{p}\rfloor$$

Sem isso, um atributo muito discriminante apareceria na raiz de quase todas as
árvores e elas ficariam parecidas demais. É esta a diferença entre uma
Floresta Aleatória e um *bagging* puro de árvores.

> [!NOTE]
> Na interface dá para escolher "Todos" em *atributos por nó* — isso desliga o
> subespaço aleatório e transforma a floresta em bagging puro. Comparar as duas
> configurações é um bom momento da apresentação.

---

## 4. Erro out-of-bag

Cada amostra é classificada apenas pelas árvores que **não a viram** no
treino, e o voto dessas árvores é comparado com a classe real.

Isso dá uma estimativa de generalização **sem separar um conjunto de
validação** — ela sai de graça do próprio processo de bagging.

> [!IMPORTANT]
> No Iris, o OOB costuma ficar **abaixo** da acurácia no conjunto de teste
> (ex.: 90,5% de OOB contra 100% no teste). Isso não é um defeito: o OOB usa
> todas as 105 amostras de treino, cada uma avaliada por ~37% das árvores,
> enquanto o teste usa só 45 amostras avaliadas pela floresta inteira. O OOB é
> a estimativa mais honesta das duas.

---

## 5. Importância dos atributos

Soma, sobre todos os nós de todas as árvores, do ganho de impureza que cada
atributo proporcionou — ponderado pelo número de amostras que passaram pelo
nó, e normalizada para somar 1:

$$\text{imp}(j) = \frac{\sum_{\text{nós que dividem por } j} n_{\text{nó}} \cdot \text{ganho}}{\sum_{j'}\sum_{\text{nós de } j'} n_{\text{nó}} \cdot \text{ganho}}$$

É a *mean decrease in impurity*. Com as 4 features do Iris, o resultado
medido foi:

| Atributo | Importância |
|---|---:|
| Comprimento da Pétala | 43,1% |
| Largura da Pétala | 42,8% |
| Comprimento da Sépala | 10,6% |
| Largura da Sépala | 3,5% |

As pétalas concentram **85,9%** da importância — resultado clássico do Iris, e
uma explicação quantitativa de por que os classificadores dos laboratórios
anteriores vão tão bem usando só as pétalas.

---

## 6. Resultados medidos

Validação cruzada 5-fold com 3 repetições, floresta de 50 árvores, critério
Gini:

| Atributos | Floresta | Árvore única | Bayes (QDA) | Distância Mínima |
|---|---:|---:|---:|---:|
| Pétalas | ~95% | ~94% | ~97,6% | ~96,3% |
| Sépalas | ~76% | ~70% | ~76,4% | ~80,9% |

Leitura honesta destes números:

1. **No Iris a floresta não vence por larga margem.** A base é pequena (150
   amostras), com poucos atributos e classes quase separáveis nas pétalas. O
   ensemble brilha em problemas com muitos atributos ruidosos — não é o caso
   aqui.
2. **A vantagem sobre a árvore única aparece nas sépalas**, onde as classes se
   sobrepõem: é exatamente onde a variância de uma árvore isolada machuca, e
   onde a média do ensemble ajuda.
3. **O Bayes Ótimo compete bem** porque as classes do Iris são
   aproximadamente gaussianas — a suposição do QDA é razoável nesta base.

> [!TIP]
> Na apresentação, vale trocar os atributos para **Sépalas** ao vivo: é onde a
> diferença entre floresta e árvore única fica visível, e onde o teste Z passa
> a acusar significância.

---

## 7. Como explorar na interface

A aba *Florestas Aleatórias* tem três sub-abas:

**A floresta** — métricas (teste, OOB, árvore única, Kappa), regiões de
decisão em escada, importância dos atributos e a floresta em números
(bootstrap, OOB, profundidade média). Clicando no gráfico, aparece a votação
árvore a árvore naquele ponto.

**Árvores individuais** — o diagrama de qualquer árvore da floresta, com a
condição de cada nó, o ganho, o número de amostras e a distribuição das
classes. Percorrendo o índice dá para ver como as árvores diferem entre si —
a descorrelação que o ensemble depende.

**Comparativo** — validação cruzada da floresta contra a árvore única, o Bayes
e a Distância Mínima, com média, desvio e intervalo de confiança, mais o teste
Z entre cada par.

O botão **Ver teoria e cálculos** abre a memória de cálculo com sete seções:
impureza, ganho, bagging, subespaço aleatório, erro OOB, importância dos
atributos e a votação de uma amostra — todas com a substituição numérica.

---

## 8. Parâmetros e o que cada um faz

| Parâmetro | Efeito |
|---|---|
| **Número de árvores** | Mais árvores reduzem a variância do ensemble. O ganho satura — a partir de ~100 a curva achata. Nunca causa *overfitting*. |
| **Critério** | Gini ou entropia. Produzem árvores parecidas; Gini é mais barato. |
| **Atributos por nó** | `√p` é o padrão. "Todos" desliga o subespaço aleatório (vira bagging puro). |
| **Profundidade máxima** | Limita o crescimento. Sem limite, cada árvore cresce até as folhas puras — o que é comum em florestas, já que o ensemble controla o *overfitting*. |

> [!NOTE]
> Aumentar o número de árvores **não** causa sobreajuste — é uma propriedade
> demonstrada por Breiman. O erro de generalização converge para um limite
> conforme o número de árvores cresce. O que causa sobreajuste é o contrário:
> árvores individuais profundas demais **e** correlacionadas demais.

---

## 9. Onde as Florestas se encaixam no curso

| Classificador | Fronteira | Parâmetros estimados |
|---|---|---|
| Distância Mínima | reta / hiperplano | vetores médios |
| Perceptron, Regra Delta | reta / hiperplano | pesos por gradiente |
| Bayes Ótimo (QDA) | cônica | médias e covariâncias |
| MLP (Lab 5) | curva suave arbitrária | pesos por backpropagation |
| **Floresta Aleatória** | **escada alinhada aos eixos** | **divisões por ganho de impureza** |

A floresta é o único método não paramétrico do conjunto: não assume forma
funcional nenhuma para a fronteira nem distribuição para os dados. Ela apenas
particiona o espaço recursivamente até separar as classes.

---

## 10. Estrutura do código

| Arquivo | Responsabilidade |
|---|---|
| `iris_classifier/models/random_forest.py` | Árvore CART, bagging, subespaço aleatório, OOB, importâncias — Python puro |
| `iris_classifier/evaluation/validacao_cruzada.py` | k-fold estratificado, usado no comparativo |
| `web_app/backend/routers/floresta.py` | API: treino, árvores, regiões, predição, validação e memória de cálculo |
| `web_app/frontend/src/pages/Floresta.tsx` | Página do seminário |
| `web_app/frontend/src/components/ArvoreDecisao.tsx` | Diagrama SVG da árvore |

Funções centrais de `random_forest.py`:

- `gini` / `entropia` — impureza de um nó
- `melhor_divisao` — busca exaustiva pelo maior ganho entre os atributos sorteados
- `construir_arvore` — recursão com critérios de parada e acúmulo das importâncias
- `FlorestaAleatoria.treinar` — bagging + subespaço + cálculo do OOB
- `caminho_decisao` — sequência de decisões até a folha, usada na memória de cálculo

---

*Tópicos Especiais em Inteligência Artificial · UEPB 2026*

---

## Dataset em escala: as 10 instâncias viram 1000

O exemplo do fim de semana usado em toda a apresentação tem 10 padrões — o
suficiente para as contas à mão, mas não para avaliar um classificador (o erro
OOB do slide 25 deu 44,4% justamente por isso).

[`seminario_dataset_fim_de_semana.md`](seminario_dataset_fim_de_semana.md)
descreve a versão com **1000 instâncias** do mesmo problema: como o conceito foi
extraído das 10 linhas originais, como o ruído de rótulo cria um teto teórico de
acerto, e os resultados da floresta ID3 multi-way (erro OOB de 7,59% contra 7,00%
de ruído injetado).

Esse documento também registra **duas divergências de contagem encontradas nos
slides 20 e 21** (Árvores 2 e 3), com o efeito de cada uma sobre as conclusões.

```bash
python seminario_fim_de_semana.py
```
