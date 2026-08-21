# Tela Classificar — escolha e parametrização do modelo

> Requisito atendido: *"O projeto deve disponibilizar ao usuário opções de
> definição do modelo a ser utilizado no processo de classificação, bem como a
> parametrização do modelo."*

A aba **Classificar** é a porta de entrada do aplicativo. Nela o usuário
escolhe a base (inclusive um `.txt` próprio), escolhe o **modelo** e ajusta os
**hiperparâmetros**, roda a classificação e vê todas as métricas de qualidade —
sem trocar de tela para trocar de classificador.

As abas de laboratório continuam existindo: elas mostram a *dedução* de cada
método (memórias de cálculo, protótipos, épocas, árvores). A tela Classificar é
o uso aplicado de todos eles.

---

## 1. Catálogo de modelos

Sete classificadores, todos escritos do zero em Python puro:

| Modelo | Grupo | Hiperparâmetros |
|---|---|---|
| Distância Mínima | Lineares | — |
| Perceptron OvA | Lineares | taxa de aprendizado, máximo de épocas |
| Regra Delta OvA | Lineares | taxa de aprendizado, épocas |
| Bayes Ótimo (QDA) | Probabilísticos | — |
| Naive Bayes | Probabilísticos | — |
| Rede Feedforward (MLP) | Redes Neurais | neurônios ocultos, taxa, épocas, semente |
| Floresta Aleatória | Seminário | nº de árvores, critério, profundidade, atributos por nó, mínimo por folha, semente |

Distância Mínima, Bayes e Naive Bayes **não têm** hiperparâmetros: são
determinados inteiramente pelos dados de treino (médias e covariâncias). O que
se varia neles é a base, o conjunto de atributos e a proporção treino/teste.

### Como o catálogo é definido

Cada modelo é uma entrada em `web_app/backend/modelos.py`, declarando nome,
descrição, funções de treino e predição, e o **esquema** dos próprios
parâmetros:

```python
{'id': 'n_arvores', 'rotulo': 'Número de árvores', 'tipo': 'inteiro',
 'padrao': 50, 'min': 1, 'max': 300, 'passo': 1,
 'ajuda': 'Mais árvores reduzem a variância e estabilizam o voto.'}
```

A interface **não** tem controles escritos à mão: ela lê o esquema em
`/api/classificar/modelos` e monta um slider para `inteiro`/`numero` e um
select para `opcoes`. Acrescentar um modelo ao catálogo faz a tela crescer
sozinha — e o modelo novo aparece automaticamente nos testes de significância e
na validação cruzada.

Valores fora da faixa são trazidos para dentro dela em vez de derrubar a
requisição; chaves desconhecidas são ignoradas; ausentes recebem o padrão.

---

## 2. O que cada parâmetro faz

**Perceptron OvA / Regra Delta OvA**
- *Taxa de aprendizado* — tamanho da correção a cada erro. Alta demais oscila;
  baixa demais não sai do lugar dentro do limite de épocas.
- *Épocas* — o Perceptron para assim que zera os erros (se as classes forem
  linearmente separáveis); a Regra Delta roda todas, sem parada antecipada.

**Rede Feedforward (MLP)**
- *Neurônios ocultos* — capacidade da rede. Poucos não aprendem, muitos decoram.
- *Taxa de aprendizado* — passo do gradiente na retropropagação.
- *Épocas* — passagens completas pelo treino; acompanhe a curva de erro.
- *Semente* — fixa os pesos iniciais, tornando o resultado reprodutível.

**Floresta Aleatória**
- *Número de árvores* — mais árvores reduzem a variância do comitê.
- *Critério* — Gini ou entropia, a medida de impureza de cada divisão.
- *Profundidade máxima* — `0` significa sem limite (árvores crescem até a folha
  pura). Limitar poda o sobreajuste.
- *Atributos sorteados por nó* — √p, log₂ p ou todos. É o sorteio que
  descorrelaciona as árvores; com "todos", a floresta vira bagging puro.
- *Mínimo de amostras por folha* — poda simples contra sobreajuste.
- *Semente* — fixa bootstrap e sorteios.

---

## 3. O que a tela mostra

**Quatro indicadores no topo**
- *Acerto no teste* — sobre amostras nunca vistas.
- *Acerto no treino* — exibido ao lado de propósito: a diferença entre os dois
  é a leitura direta de **sobreajuste**. Uma floresta sem limite de
  profundidade costuma marcar quase 100% no treino.
- *Kappa* — concordância corrigida pelo acerto casual.
- *Tempo de predição* — em milissegundos, sobre treino + teste.

**Regiões de decisão** — o modelo é aplicado a uma malha do plano e cada ponto
é pintado com a classe vencedora. Clicar no gráfico classifica aquele ponto.
Quando o conjunto de atributos tem mais de duas dimensões, as demais ficam
fixas na média global (a classificação usa todas).

**Métricas completas** — acerto global, Kappa, Tau e Var(κ); matriz de
confusão; e por classe: acurácia do produtor (revocação), do usuário
(precisão), especificidade, F1, F2 e MCC.

**Classificar uma amostra** — um campo por atributo, preenchido com a média de
cada um. O resultado traz a classe e a pontuação atribuída a cada classe, com o
nome certo para cada modelo: `dⱼ(x)` na distância mínima, `net` nos lineares,
`ln p(x|ω)·P(ω)` no Bayes, proporção de votos na floresta, ativação de saída na
rede.

**Painéis específicos do modelo**
- Floresta: erro out-of-bag e importância dos atributos.
- MLP: curva de erro por época.
- Perceptron/Delta: curva de convergência por classe e, no Perceptron, um selo
  dizendo se cada classificador binário convergiu.

---

## 4. Comparação entre modelos

A comparação continua na aba **Métricas Avançadas**, agora sobre o mesmo
catálogo de sete modelos:

- **Validação cruzada** — k-fold estratificado com repetições, média ± desvio e
  IC 95%. Por padrão roda todos menos a rede (que treinaria k × repetições
  vezes); a API aceita `modelos=bayes,floresta,mlp` para incluí-la.
- **Split único** — todos no mesmo split, com teste Z de Kappa entre cada par.
- **Significância** — McNemar, bootstrap pareado e teste de permutação para
  qualquer par, em qualquer métrica. Ver
  [`lab_03/testes_significancia.md`](lab_03/testes_significancia.md).

Os três testes exigem **predições pareadas**: todos os modelos são avaliados no
mesmo conjunto de teste, e o resultado fica em cache por
(base, atributos, proporção) — sem isso, a matriz de 21 pares retreinaria os
sete modelos a cada par.

---

## 5. Rotas da API

| Método | Rota | Descrição |
|---|---|---|
| GET | `/api/classificar/modelos` | Catálogo + esquema dos parâmetros |
| POST | `/api/classificar/treinar` | Treina, avalia e devolve todas as métricas |
| POST | `/api/classificar/regioes` | Regiões de decisão no plano 2D |
| POST | `/api/classificar/predizer` | Classifica uma amostra informada |

Corpo típico:

```json
{
  "dataset": "v1",
  "atributos": "petalas",
  "proporcao": 0.7,
  "modelo": "floresta",
  "parametros": {"n_arvores": 100, "criterio": "entropia"}
}
```

---

## 6. Dois modelos criados para esta tela

Ambos em Python puro, seguindo a regra do projeto.

**Perceptron Um-Contra-Todos** (`models/perceptron.py`) — treina um Perceptron
de Rosenblatt por classe, rerrotulando as amostras em `+1` (a classe) e `−1`
(todo o resto), e decide pelo argmax dos *nets*. O algoritmo binário é o mesmo
do Lab 2: muda apenas a rotulagem.

**Rede feedforward multiclasse** (`models/mlp_multiclasse.py`) — envolve a
`RedeFeedforward` do Lab 5 com duas providências: normalização min-max das
entradas para [0, 1] (a sigmoide satura com valores grandes, e os gradientes
somem) calculada **só com o treino**, e codificação 1-de-C na saída, com um
neurônio por classe e decisão por argmax.
