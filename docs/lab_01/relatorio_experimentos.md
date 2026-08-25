# Lab 1 — Relatório de Experimentos: Classificador de Distância Mínima

**Disciplina:** Tópicos Especiais em Inteligência Artificial (TEIA)
**UEPB 2026**
**Grupo:** Erick Nathan · Laura Barbosa · Pedro Lucas
**Referência:** Aula PR3 (Prof. Robson Pequeno de Sousa)

---

## 1. Configuração do experimento

| Parâmetro | Valor |
|---|---|
| Base | Iris (`data/Iris data.xls`) — 150 amostras, 4 atributos, 3 classes |
| Split | 70% treino / 30% teste, **estratificado por classe** |
| Amostras | 105 de treino · 45 de teste (15 por classe) |
| Semente | 42 (resultados reprodutíveis) |
| Conjuntos de atributos avaliados | 4 atributos · pétalas [2,3] · sépalas [0,1] |

Todos os números abaixo foram produzidos pelo próprio código do projeto e podem
ser reproduzidos com `python iris_classifier/main.py` ou pela aba
*Distância Mínima* do aplicativo.

---

## 2. Item (i) — Classificador de distância mínima para as três classes

### 2.1 Protótipos calculados (105 amostras de treino)

Com os quatro atributos, $m_j = \frac{1}{35}\sum x$:

| Classe | Compr. sépala | Larg. sépala | Compr. pétala | Larg. pétala |
|---|:---:|:---:|:---:|:---:|
| Setosa | 5,0029 | 3,4143 | 1,4800 | 0,2486 |
| Versicolor | 5,9371 | 2,7743 | 4,2371 | 1,3229 |
| Virgínica | 6,5000 | 2,9114 | 5,5200 | 1,9886 |

Já se lê algo nos números: as **pétalas** crescem de forma clara e monótona
entre as três classes (1,48 → 4,24 → 5,52 cm de comprimento), enquanto as
**sépalas** quase não distinguem versicolor de virgínica (5,94 → 6,50 cm, com
larguras praticamente iguais). Essa observação antecipa todos os resultados
seguintes.

### 2.2 Desempenho no conjunto de teste

| Conjunto de atributos | Acertos (45) | Acurácia | Erros na base completa (150) |
|---|:---:|:---:|:---:|
| Pétalas [2,3] | 45 | **100,00%** | 5 |
| Todos os 4 atributos | 44 | **97,78%** | 10 |
| Sépalas [0,1] | 37 | **82,22%** | 27 |

### 2.3 Matrizes de confusão (linha = predito, coluna = real)

**Pétalas — 100,00%**

| Predito \ Real | Setosa | Versicolor | Virgínica |
|---|:---:|:---:|:---:|
| Setosa | **15** | 0 | 0 |
| Versicolor | 0 | **15** | 0 |
| Virgínica | 0 | 0 | **15** |

**Quatro atributos — 97,78%**

| Predito \ Real | Setosa | Versicolor | Virgínica |
|---|:---:|:---:|:---:|
| Setosa | **15** | 0 | 0 |
| Versicolor | 0 | **14** | 0 |
| Virgínica | 0 | 1 | **15** |

O único erro é uma versicolor classificada como virgínica — a confusão esperada
entre as duas classes vizinhas. Nenhuma setosa é confundida em nenhum cenário.

**Sépalas — 82,22%**

| Predito \ Real | Setosa | Versicolor | Virgínica |
|---|:---:|:---:|:---:|
| Setosa | **15** | 2 | 0 |
| Versicolor | 0 | **9** | 2 |
| Virgínica | 0 | 4 | **13** |

Aqui os 8 erros se concentram entre versicolor e virgínica (6 dos 8), mais 2
versicolores atraídas pelo protótipo da setosa.

---

## 3. Item (ii) — Função de decisão pelo classificador de máximo

A decisão é $\arg\max_j d_j(x)$, com
$d_j(x) = x^Tm_j - \frac{1}{2}m_j^Tm_j$.

Exemplo com a primeira amostra do conjunto de teste, no plano das pétalas —
$x = [1{,}40 \;;\; 0{,}20]$, classe real **setosa**:

| Classe | $d_j(x)$ | Decisão |
|---|---:|:---:|
| Setosa | **+0,9956** | ← máximo |
| Versicolor | −3,6551 | |
| Virgínica | −9,0867 | |

O maior valor é o da setosa, e a classificação está correta. A margem é
larga — quase 4,7 unidades de discriminante até a segunda colocada —, o que
indica um ponto bem no interior da região da setosa, longe de qualquer
fronteira.

**Verificação da equivalência.** Classificando as 45 amostras de teste pelas
duas formulações — menor distância euclidiana e maior discriminante — as
respostas coincidem em 45 de 45 casos, como a dedução da seção 4 da teoria
prevê. Não se trata de uma aproximação numérica: as duas expressões diferem por
$x^Tx$, termo idêntico para todas as classes.

---

## 4. Item (iii) — Superfícies de decisão para os três pares

Coeficientes obtidos no plano das pétalas, com $w = m_i - m_j$ e
$b = -\frac{1}{2}(\|m_i\|^2 - \|m_j\|^2)$:

### 4.1 Virgínica × Setosa

$$+4{,}0400\,x_1 + 1{,}7400\,x_2 - 16{,}0863 = 0$$

Reta para plotagem: $x_2 = (-4{,}0400\,x_1 + 16{,}0863)\,/\,1{,}7400$
**Acurácia binária no teste: 30/30 = 100,00%**

### 4.2 Setosa × Versicolor

$$-2{,}7571\,x_1 - 1{,}0743\,x_2 + 8{,}7256 = 0$$

Reta para plotagem: $x_2 = (2{,}7571\,x_1 - 8{,}7256)\,/\,(-1{,}0743)$
**Acurácia binária no teste: 30/30 = 100,00%**

### 4.3 Versicolor × Virgínica

$$-1{,}2829\,x_1 - 0{,}6657\,x_2 + 7{,}3607 = 0$$

Reta para plotagem: $x_2 = (1{,}2829\,x_1 - 7{,}3607)\,/\,(-0{,}6657)$
**Acurácia binária no teste: 30/30 = 100,00%**

### 4.4 Leitura dos coeficientes

O par virgínica × setosa tem os maiores coeficientes em módulo ($w_1 = 4{,}04$):
os protótipos estão longe um do outro, e o discriminante cresce depressa ao se
afastar da fronteira. Já o par versicolor × virgínica tem os menores
($w_1 = -1{,}28$) — protótipos vizinhos, fronteira "frouxa", pequenas variações
de $x$ mudam pouco o discriminante. É a tradução algébrica da proximidade entre
essas duas classes.

Vale registrar a diferença entre os itens: **par a par**, no plano das pétalas,
os três problemas binários são resolvidos com 100% de acerto. Os 5 erros que
aparecem na base completa (seção 2.2) surgem apenas na decisão **multiclasse**,
quando as três fronteiras competem simultaneamente.

---

## 5. Análise dos resultados

**Por que as pétalas acertam 100%?** É uma propriedade da base, não mérito do
classificador. No plano das pétalas as três espécies formam agrupamentos quase
disjuntos; qualquer classificador linear razoável acerta quase tudo. Convém
dizer isso explicitamente na defesa, para que o 100% não seja lido como
sobreajuste ou erro de implementação.

**Por que 4 atributos acertam menos que 2?** Parece contraintuitivo, mas é
consistente: as sépalas acrescentam ruído. Como o classificador de distância
mínima trata todos os atributos com o mesmo peso e ignora a covariância, incluir
duas variáveis pouco informativas dilui a informação das pétalas. Um
classificador que estima a covariância — o Bayes do Lab 4 — não sofre tanto com
isso.

**Por que as sépalas ficam em 82%?** Porque versicolor e virgínica têm
protótipos muito próximos nesse plano (5,94 × 6,50 no comprimento; 2,77 × 2,91
na largura). A mediatriz entre eles corta o meio de uma região onde as duas
nuvens se misturam, e 6 das 8 falhas caem exatamente aí.

**Limitação estrutural.** Todos os erros observados vêm da mesma fonte: o
protótipo descarta a forma da nuvem de pontos. Duas classes com a mesma média e
dispersões diferentes seriam indistinguíveis para este modelo. Essa é
precisamente a lacuna que os laboratórios seguintes atacam — o Lab 2 aprende a
fronteira a partir dos erros, e o Lab 4 modela a distribuição completa de cada
classe.

---

## 6. Onde reproduzir

| Como | O quê |
|---|---|
| Aba **Distância Mínima** do aplicativo | Protótipos, discriminantes, as três fronteiras e as regiões de decisão; clicar no gráfico classifica um ponto arbitrário |
| Botão **Ver teoria e cálculos** | Memória de cálculo com a substituição numérica passo a passo |
| Aba **Classificar** | O mesmo modelo no catálogo, com métricas completas e matriz de confusão |
| `python iris_classifier/main.py` | Todos os experimentos no terminal, com os gráficos salvos em `outputs/` |

Teoria completa em [`teoria_lab01.md`](teoria_lab01.md).
