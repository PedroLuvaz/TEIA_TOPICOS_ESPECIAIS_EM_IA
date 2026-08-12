# Dataset do seminário em escala — "fim de semana" com 1000 instâncias

**Motivação:** o seminário de Florestas Aleatórias usa, do início ao fim, o mesmo
conjunto de **10 padrões** do material da disciplina (slide 9). Dez instâncias
bastam para fazer as contas à mão, mas não para avaliar um classificador — no
próprio seminário o erro OOB deu 44,4% justamente por isso (B = 3 árvores, 10
instâncias, 4 classes). Este documento descreve a versão com **1000 instâncias**
do mesmo problema.

**Arquivos:**

| Arquivo | Papel |
|---|---|
| [`iris_classifier/data/gerar_fim_de_semana.py`](../iris_classifier/data/gerar_fim_de_semana.py) | Gerador do CSV |
| [`data/fim_de_semana_1000.csv`](../data/fim_de_semana_1000.csv) | As 1000 instâncias |
| [`iris_classifier/data/data_loader.py`](../iris_classifier/data/data_loader.py) | `carregar_fim_de_semana()` |
| [`iris_classifier/models/floresta_categorica.py`](../iris_classifier/models/floresta_categorica.py) | Floresta ID3 multi-way |
| [`iris_classifier/seminario_fim_de_semana.py`](../iris_classifier/seminario_fim_de_semana.py) | Experimento completo (blocos A, B e C) |

---

## 1. O problema original (slide 9)

Decidir o programa do fim de semana a partir de 3 atributos categóricos:

| # | Clima | Pais visitam? | Dinheiro | Decisão |
|---|-------|---------------|----------|---------|
| 1 | Sol | Sim | Rico | Cinema |
| 2 | Sol | Não | Rico | Tênis |
| 3 | Vento | Sim | Rico | Cinema |
| 4 | Chuva | Sim | Pobre | Cinema |
| 5 | Chuva | Não | Rico | Ficar em casa |
| 6 | Chuva | Sim | Pobre | Cinema |
| 7 | Vento | Não | Pobre | Cinema |
| 8 | Vento | Não | Rico | Compras |
| 9 | Vento | Sim | Rico | Cinema |
| 10 | Sol | Não | Rico | Tênis |

`Clima ∈ {Sol, Vento, Chuva}`, `Pais ∈ {Sim, Não}`, `Dinheiro ∈ {Rico, Pobre}`,
4 classes de decisão.

---

## 2. Como as 1000 instâncias são geradas

### 2.1 O conceito

As 10 linhas acima definem, **sem ambiguidade**, uma função completa sobre as
`3 × 2 × 2 = 12` combinações possíveis:

```
Pais = Sim                     ->  Cinema
Pais = Não  e  Dinheiro = Pobre ->  Cinema
Pais = Não  e  Dinheiro = Rico:
    Clima = Sol                ->  Tênis
    Clima = Vento              ->  Compras
    Clima = Chuva              ->  Ficar em casa
```

`validar_conceito()` verifica, a cada execução, que a regra reproduz as 10 linhas
originais e cobre as 12 combinações. Se alguma divergir, o gerador levanta
`AssertionError` — o dataset nunca sai silenciosamente errado.

### 2.2 Os atributos

Sorteados de forma independente, com as frequências marginais observadas nas 10
instâncias originais:

| Atributo | Distribuição |
|---|---|
| Clima | Sol 30% · Vento 40% · Chuva 30% |
| Pais | Sim 50% · Não 50% |
| Dinheiro | Rico 70% · Pobre 30% |

*Ressalva honesta:* as 10 linhas originais não são sorteios independentes dessas
marginais (as linhas 4 e 6 são idênticas, por exemplo). Usar as marginais
empíricas é a escolha natural e defensável, mas é uma **decisão de modelagem**,
não algo que os slides prescrevem.

### 2.3 O ruído de rótulo — e por que ele é necessário

Uma fração `taxa_ruido` (padrão **8%**) das instâncias tem o rótulo trocado por
outra classe, sorteada uniformemente entre as 3 restantes.

Sem ruído o problema seria uma função determinística dos atributos, e qualquer
árvore razoável acertaria **100%** — exatamente a crítica que o professor fez às
métricas do projeto. Com ruído existe um **teto teórico de acerto**: o
classificador ótimo acerta `1 − taxa_ruido`. Assim o erro OOB passa a medir algo
real e os testes de significância têm o que comparar.

A coluna `ruido` do CSV marca quais linhas foram alteradas, para que o
experimento continue auditável.

No arquivo versionado (semente 42): **70 rótulos trocados = 7,00%**, logo o teto
teórico de acerto é **93,00%**.

### 2.4 Formato do CSV

```
id,clima,pais,dinheiro,decisao,clima_cod,pais_cod,dinheiro_cod,ruido
1,Vento,Sim,Rico,Cinema,1,1,1,0
2,Chuva,Nao,Pobre,Cinema,2,0,0,0
3,Vento,Sim,Rico,Cinema,1,1,1,0
```

| Coluna | Conteúdo |
|---|---|
| `id` | 1..N |
| `clima`, `pais`, `dinheiro` | valores categóricos em texto |
| `decisao` | a classe |
| `clima_cod` | Sol = 0, Vento = 1, Chuva = 2 |
| `pais_cod` | Não = 0, Sim = 1 |
| `dinheiro_cod` | Pobre = 0, Rico = 1 |
| `ruido` | 1 se o rótulo foi trocado |

Distribuição obtida (semente 42, N = 1000):

| | |
|---|---|
| Clima | Vento 41,1% · Chuva 30,0% · Sol 28,9% |
| Pais | Não 50,8% · Sim 49,2% |
| Dinheiro | Rico 69,9% · Pobre 30,1% |
| Decisão | Cinema 60,0% · Compras 15,7% · Ficar em casa 12,4% · Tênis 11,9% |

As classes ficam **desbalanceadas** (60% / 15,7% / 12,4% / 11,9%), o que é
proposital: é nesse regime que Kappa e MCC ganham sentido frente ao acerto global.

### 2.5 Regenerar

```bash
python -m data.gerar_fim_de_semana
```

```bash
python -m data.gerar_fim_de_semana --n 5000 --ruido 0.15 --semente 7
```

---

## 3. A floresta categórica (ID3 multi-way)

`models/random_forest.py` implementa a **CART binária** (`atributo <= limiar`),
que é o certo para atributos contínuos como os do Iris. O seminário, porém, faz
as contas sobre atributos categóricos usando **ID3**: entropia, ganho de
informação e divisão **multi-way** — um ramo por valor do atributo.

Codificar "Sol/Vento/Chuva" como 0/1/2 e deixar a CART cortar por limiar
funcionaria, mas inventaria uma ordem que não existe (`Sol < Vento < Chuva`) e
não reproduziria as contas apresentadas. Por isso
`models/floresta_categorica.py` é um módulo separado, que implementa o algoritmo
do slide 17 ao pé da letra — sem tocar no CART que já roda no Iris.

O custo dessa escolha aparece no Bloco C (seção 6): a floresta CART sobre os
códigos ordinais fica **11,6 pontos abaixo** da floresta categórica.

---

## 4. Bloco A — validação contra os slides

O script confere cada número da apresentação. **A árvore única (slide 10) e a
Árvore 1 (slide 19) conferem exatamente:**

| Grandeza | Calculado | Slide |
|---|---|---|
| H(S) da raiz | 1,5710 | 1,5710 |
| Gain(S, Clima) | 0,6955 | 0,70 |
| Gain(S, Pais) | 0,6100 | 0,61 |
| Gain(S, Dinheiro) | 0,2813 | 0,2816 |
| Árvore 1 — H(S) | 1,3610 | 1,3610 |
| Árvore 1 — Gain(Pais) | 1,0000 | 1,0000 |
| Árvore 1 — Gain(Clima) | 0,7245 | 0,7245 |
| OOB das 3 árvores | {6,7,8} {3,5,6,8} {2,6} | idem |
| Votação do padrão novo (slide 23) | 3 × Cinema | 3 × Cinema |

### 4.1 Duas divergências reais nos slides

> **As Árvores 2 e 3 da apresentação têm erro de contagem de classes.** As
> distribuições afirmadas nos slides não correspondem às amostras bootstrap
> listadas no slide 18.

**Árvore 2** — bootstrap `{1,1,1,2,4,4,7,9,9,10}`:
as instâncias **2 e 10 são Tênis**, logo a distribuição é `Cinema = 8, Tênis = 2`.
O slide 20 diz `Cinema = 9, Tênis = 1`.

| | Calculado | Slide |
|---|---|---|
| H(S) | 0,7219 | 0,4690 |
| Gain(Pais) | 0,4464 | 0,2690 |
| Gain(Dinheiro) | 0,1177 | 0,1080 |

A **conclusão do slide não muda** — Pais continua vencendo e sendo a raiz.

**Árvore 3** — bootstrap `{1,3,4,4,5,7,7,8,9,10}`:
a instância **8 é Compras**, logo a distribuição é
`Cinema = 7, Tênis = 1, Ficar em casa = 1, Compras = 1`.
O slide 21 diz `Cinema = 6, Tênis = 3, Ficar em casa = 1` — omite Compras e conta
3 Tênis onde só existe 1.

| | Calculado | Slide |
|---|---|---|
| H(S) | 1,3568 | 1,2955 |
| Gain(Pais) | 0,3958 | 0,4200 |
| Gain(Dinheiro) | 0,2813 | 0,6100 |

Aqui a **conclusão muda**: com a contagem correta, **Pais** vence Dinheiro
(0,3958 > 0,2813) e vira a raiz da Árvore 3 — enquanto o slide 21 conclui
Dinheiro. Isso afeta em cascata:

- o **slide 22** ("nenhuma das três árvores usa Clima na raiz") continua
  verdadeiro, mas o argumento fica mais fraco: as três passam a usar **Pais**,
  não duas Pais e uma Dinheiro;
- a tabela do **slide 27** (importância) muda, porque Dinheiro deixa de ter um
  split de raiz.

O erro OOB do slide 25 e a votação do slide 23 **não** dependem dessas contagens
e continuam válidos.

Rodar só essa verificação:

```bash
python seminario_fim_de_semana.py --bloco a
```

---

## 5. Bloco B — a floresta em 1000 instâncias

`FlorestaCategorica`, critério entropia, split estratificado 70/30 (698 treino,
302 teste), `mtry = round(√3) = 2`, semente 42.

```bash
python seminario_fim_de_semana.py --bloco b --n-arvores 300
```

### 5.1 Desempenho

| Métrica | Valor |
|---|---|
| Acerto Global | **94,37%** |
| Kappa | 0,9024 |
| Tau | 0,9249 |
| MCC (multiclasse) | 0,9035 |
| **Erro OOB** | **7,59%** |

**A leitura mais importante:** o erro OOB (7,59%) reproduz quase exatamente a
taxa de ruído injetada (7,00%). É a demonstração numérica de que o OOB estima o
erro de generalização "de graça", como afirma o slide 24 — e de que a floresta
aprendeu o conceito inteiro, deixando só o erro irredutível.

O acerto de 94,37% fica ligeiramente **acima** do teto de 93% porque o teto vale
para o conjunto completo; o split de teste específico contém uma proporção de
rótulos ruidosos um pouco menor, e parte do ruído coincide com a classe predita.

### 5.2 Matriz de confusão (linha = predito, coluna = real)

| | Cinema | Compras | Ficar em casa | Tênis |
|---|---|---|---|---|
| **Cinema** | 177 | 6 | 1 | 4 |
| **Compras** | 2 | 42 | 2 | 1 |
| **Ficar em casa** | 1 | 0 | 35 | 0 |
| **Tênis** | 0 | 0 | 0 | 31 |

Os 17 erros são quase todos rótulos ruidosos — não há confusão sistemática entre
classes.

### 5.3 Importância dos atributos (MDI)

| Atributo | Importância normalizada |
|---|---|
| Clima | 34,0% |
| Pais visitam? | 33,6% |
| Dinheiro | 32,4% |

**Diferente do exemplo de 10 instâncias**, onde Dinheiro era claramente o mais
fraco (19,5% no slide 27). Com 1000 instâncias os três empatam, e isso está
certo: o conceito verdadeiro precisa dos três atributos em sequência — Pais
separa Cinema, Dinheiro separa o ramo "Não", e Clima decide entre Tênis, Compras
e Ficar em casa. Nenhum é dispensável.

É um bom argumento para a apresentação: **com 10 instâncias a importância era
ruído amostral**, não estrutura do problema.

### 5.4 Efeito do mtry (slide 22)

Atributo usado na raiz, em 300 árvores:

| Atributo | Árvores | % |
|---|---|---|
| Pais visitam? | 163 | 54,3% |
| Clima | 137 | 45,7% |
| Dinheiro | 0 | 0% |

Com `mtry = 2` de `p = 3`, Pais entra no sorteio em ~67% dos casos e vence sempre
que entra; nos ~33% restantes o par é {Clima, Dinheiro} e Clima vence. É
exatamente a decorrelação que o slide 22 descreve, agora com números estáveis.

---

## 6. Bloco C — o mesmo CSV nos outros módulos

Carregando com `numerico=True` (colunas `_cod`), o mesmo arquivo alimenta os
demais classificadores do projeto, no mesmo split:

```bash
python seminario_fim_de_semana.py --bloco c
```

| Classificador | Acerto | Kappa | MCC |
|---|---|---|---|
| Naive Bayes | 94,37% | 0,9024 | 0,9035 |
| Bayes Ótimo (QDA) | 94,37% | 0,9024 | 0,9035 |
| Floresta CART (Lab) | 82,78% | 0,6995 | 0,7196 |
| Distância Mínima | 70,86% | 0,5861 | 0,6290 |
| Regra Delta OvA | 11,92% | 0,0000 | 0,0000 |

Três leituras que valem para a apresentação:

1. **Bayes atinge o teto.** Com 3 atributos discretos e 12 combinações, o
   classificador de Bayes essencialmente memoriza a tabela de decisão — e empata
   com a floresta categórica em 94,37%. Naive e QDA dão idêntico porque, aqui, os
   atributos realmente são independentes por construção (seção 2.2).

2. **A codificação ordinal custa caro.** A floresta CART do Lab, sobre os mesmos
   dados codificados 0/1/2, cai para 82,78% — **11,6 pontos** abaixo da floresta
   categórica. Um corte por limiar não consegue isolar "Vento" sozinho, porque a
   codificação o colocou entre Sol e Chuva. É a justificativa empírica para o
   módulo ID3 separado da seção 3.

3. **A Regra Delta OvA colapsa.** Kappa = 0 e acerto de 11,92%, que é exatamente
   a proporção da classe Tênis — ela prediz uma única classe para tudo. Não é um
   bug: são 4 classes num espaço ordinal de 3 dimensões em que os separadores
   lineares um-contra-todos ficam todos degenerados. É o mesmo limite do
   *versicolor* no Iris e do XOR do Lab 5, reaparecendo num terceiro contexto.

### 6.1 Ressalva sobre a codificação ordinal

As colunas `_cod` usam **codificação ordinal**, que impõe uma ordem inexistente a
`Clima` (Sol < Vento < Chuva). Para `Pais` e `Dinheiro`, que são binários, não há
perda. Para `Clima`, há — e o item 2 acima mede exatamente quanto.

A alternativa seria *one-hot*, mas ela quebraria o Bayes Ótimo: as colunas
somariam 1 e a matriz de covariância ficaria singular. A escolha aqui é o
compromisso mais simples que mantém o arquivo utilizável por todos os módulos
sem alterar nenhum deles.

---

## 7. Na interface web

O dataset aparece no seletor **Base de dados** de todas as abas, ao lado das duas
variantes do Iris. Trocar para *Fim de Semana (seminário)* muda tudo o que
depende do dataset:

| O que muda | Antes (fixo no Iris) | Agora |
|---|---|---|
| Classes | 3 | vêm do dataset (4 aqui) |
| Conjuntos de atributos | pétalas / sépalas / todas | Clima×Pais, Clima×Dinheiro, Pais×Dinheiro, todos |
| Pares para o Perceptron | 3 | 6 |
| Cores das classes | 3 fixas | paleta atribuída na ordem do dataset |
| Rótulos dos eixos | cm | `Clima (0=Sol · 1=Vento · 2=Chuva)` |

### 7.1 Como isso foi feito

- **`web_app/backend/core.py`** ganhou um registro `DATASETS` em que cada base
  declara classes, features, combinações de atributos e o tipo (contínuo ou
  categórico). `classes_de(dataset)`, `features_de(dataset)`,
  `config_atributos_de(dataset)` e `pares_de(dataset)` substituíram as
  constantes globais.
- **Os 7 routers** passaram a resolver as classes por dataset. `indices_de` e
  `indices_plot` receberam o parâmetro `dataset`.
- **O frontend** deixou de ter listas fixas de classes: `classesDoRelatorio()`
  lê as classes da própria matriz de confusão devolvida pela API, e o hook
  `usarDataset()` entrega classes, features, atributos e pares do dataset
  selecionado.
- **Trocar de dataset é atômico**: `usarConfig` ajusta o conjunto de atributos
  na mesma atualização de estado, porque as chaves são diferentes por dataset
  ('petalas' × 'clima_pais') e corrigir depois dispararia uma requisição
  inválida.

### 7.2 Jitter no gráfico

Com 3 atributos discretos, as 1000 amostras cairiam sobre 12 posições exatas e o
gráfico viraria 12 pontos. Para os datasets categóricos o backend aplica um
deslocamento aleatório de ±0,22 nas coordenadas de plotagem — determinístico
(semente fixa), para não tremer a cada renderização, e aplicado **apenas** a
`x`/`y`; o vetor `atributos` de cada amostra continua com os valores originais.

### 7.3 Um bug pré-existente que apareceu no caminho

A variância do Kappa (fórmula de Congalton & Green) é um estimador
**assintótico** e pode devolver um valor negativo para classificadores
degenerados — o caso da Regra Delta OvA, que aqui prediz uma classe só. O
`z_kappa` fazia `sqrt(var1 + var2)` direto e quebrava com `math domain error`.

O bug **já existia no Iris**: `comparar-modelos` com sépalas devolvia 500. A
correção foi limitar a variância a zero na origem (`variancia_kappa` e
`variancia_tau`), que é o mesmo tratamento já dado ao classificador perfeito.

Confirmado que a correção **não altera** o exercício do slide 15: o Z continua
1,6416, exatamente como no material da disciplina.

---

*Disciplina: Tópicos Especiais em Inteligência Artificial · UEPB · 2026*
*Grupo: Erick Nathan · Laura Barbosa · Pedro Lucas*
