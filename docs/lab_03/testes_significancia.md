# Testes de Significância além do Kappa

**Motivação:** observação do professor — *"verificar a significância do teste;
especificamente, após determinar o coeficiente de Matthews e os demais, fazer o
comparativo e deixar pronto para o usuário testar."*

**Implementação:** `iris_classifier/evaluation/testes_significancia.py`
(Python puro, sem numpy/scipy/sklearn)
**Interface web:** Lab 3 → aba **Métricas** → sub-aba **Significância**
**Endpoints:** `/api/metricas/classificadores`, `/api/metricas/significancia`,
`/api/metricas/significancia/matriz`, `/api/metricas/significancia/memoria`

---

## 1. O problema com o teste Z de Kappa

O teste Z visto no Lab 3 compara dois Kappas assim:

```
        κ_A − κ_B
Z = ─────────────────────
    √( σ²(κ_A) + σ²(κ_B) )
```

Somar as variâncias no denominador é a variância da diferença **apenas quando as
duas estimativas são independentes**:

```
Var(κ_A − κ_B) = Var(κ_A) + Var(κ_B) − 2·Cov(κ_A, κ_B)
```

Nos nossos experimentos os dois classificadores são avaliados **no mesmo conjunto
de teste**. Eles acertam as mesmas amostras fáceis e erram as mesmas amostras
difíceis, então `Cov(κ_A, κ_B) > 0`. Ao ignorar esse termo, o denominador fica
maior do que deveria, o `Z` fica menor e o teste **subestima a significância** —
ele é conservador demais.

A alternativa correta é usar testes **pareados**, que trabalham amostra a amostra
em vez de trabalhar com duas estatísticas agregadas. São os três desta aba.

---

## 2. Coeficiente de Matthews (MCC)

O MCC é o coeficiente de correlação de Pearson entre o vetor de predições e o
vetor do gabarito. Vale `+1` na predição perfeita, `0` no acaso e `−1` na
inversão total.

### 2.1 Versão binária (uma classe contra o resto)

```
              VP·VN − FP·FN
MCC = ────────────────────────────────────
      √( (VP+FP)(VP+FN)(VN+FP)(VN+FN) )
```

Já existia no Lab 3 por classe (`metricas_avancadas.mcc`). A versão *macro*
usada aqui é a média dos MCC um-contra-resto das três classes.

### 2.2 Versão multiclasse (Gorodkin, 2004)

Generalização direta para `K` classes, calculada de uma vez sobre a matriz de
confusão inteira:

```
                 c·s − Σ_k p_k·t_k
MCC_K = ───────────────────────────────────────
        √( (s² − Σ_k p_k²) · (s² − Σ_k t_k²) )
```

onde, na convenção `matriz[predito][real]` usada no projeto:

- `c` = acertos totais (diagonal)
- `s` = total de amostras
- `p_k` = total predito como `k` (soma da linha `k`)
- `t_k` = total real de `k` (soma da coluna `k`)

Implementado em `mcc_multiclasse()`. Quando o denominador é ~0 (um classificador
degenerado que responde sempre a mesma classe) o retorno é `0.0`, que é a
convenção padrão.

**Por que o MCC e não a acurácia:** o MCC usa as quatro células da matriz 2×2,
enquanto acurácia e F1 ignoram os verdadeiros negativos. Num problema
desbalanceado a acurácia premia o classificador que responde sempre a classe
majoritária; o MCC não.

---

## 3. Teste 1 — McNemar

Monta a tabela 2×2 dos **acertos pareados**:

|              | B acertou | B errou |
|--------------|-----------|---------|
| **A acertou** | a         | b       |
| **A errou**   | c         | d       |

As células `a` (ambos acertaram) e `d` (ambos erraram) não dizem nada sobre qual
classificador é melhor — são descartadas. Sob H₀ (os dois empatam), cada
discordância é um cara-ou-coroa justo, então `b` e `c` deveriam ser parecidos.

**Aproximação qui-quadrado** (usada quando `b + c ≥ 25`), com a correção de
continuidade de Edwards:

```
χ² = (|b − c| − 1)² / (b + c)     com 1 grau de liberdade
```

**Binomial exato** (usado quando `b + c < 25` — o caso do Iris, com 45 amostras
de teste):

```
p = 2 · P(X ≤ min(b, c)),    X ~ Bin(b + c, 0,5)
```

O p-valor do qui-quadrado com 1 grau de liberdade sai da identidade
`P(χ²₁ > x) = 2·(1 − Φ(√x))`, reaproveitando a função de distribuição normal já
implementada no Lab 3.

**Validação:** com `b = 10, c = 0` o resultado é `p = 0,001953`, exatamente
`2·(0,5)¹⁰`. Com `b = 3, c = 0`, `p = 0,25 = 2·(0,5)³`. Com dois classificadores
idênticos, `p = 1,0` e o teste avisa que não há nada a testar.

---

## 4. Teste 2 — Bootstrap pareado

Reamostra o conjunto de teste **com reposição** `B` vezes (padrão 2000). Em cada
reamostragem os índices sorteados carregam junto o par
`(predição de A, predição de B)` — é isso que preserva o pareamento. Recalcula a
métrica escolhida para os dois e guarda a diferença `Δ* = M(A*) − M(B*)`.

O intervalo de confiança de 95% são os percentis 2,5% e 97,5% da distribuição
das `Δ*`:

```
IC₉₅% = [ Δ*(2,5%), Δ*(97,5%) ]
```

**Se o IC não contém zero, a diferença é significativa a 5%.**

Vantagem sobre o McNemar: além do sim/não, entrega o **tamanho do efeito com
incerteza**. Vantagem sobre o teste Z: funciona para *qualquer* métrica —
MCC, F1, precisão, especificidade — sem precisar de uma fórmula analítica de
variância para cada uma.

O endpoint devolve também um histograma de 40 faixas dessa distribuição, que a
interface plota com as linhas do IC, do zero e da diferença observada.

---

## 5. Teste 3 — Permutação

Sob H₀ os dois classificadores são intercambiáveis: trocar aleatoriamente as
predições de A e B numa mesma amostra não deveria mudar nada. Fazendo essa troca
`B` vezes constrói-se a distribuição de `Δ` sob a hipótese nula.

```
      1 + #{ |Δ*| ≥ |Δ_obs| }
p = ──────────────────────────
             1 + B
```

O `+1` no numerador e no denominador evita `p = 0` e mantém o teste válido
(correção de Davison & Hinkley). É por isso que o menor p-valor possível com
2000 permutações é `1/2001 ≈ 0,0005`.

---

## 6. Métricas disponíveis para o teste

O registro `METRICAS` liga o identificador da API à função que calcula a métrica
a partir de `(predições, gabarito, classes)`:

| id                | Métrica                        |
|-------------------|--------------------------------|
| `mcc`             | Coeficiente de Matthews (macro)|
| `kappa`           | Kappa de Cohen                 |
| `acerto_global`   | Acerto Global                  |
| `f1`              | F1 (macro)                     |
| `precisao`        | Precisão (macro)               |
| `revocacao`       | Revocação (macro)              |
| `especificidade`  | Especificidade (macro)         |

O McNemar independe da métrica escolhida (ele só olha acertos); o bootstrap e a
permutação usam a métrica selecionada.

---

## 7. Classificadores comparáveis

Todos treinados no mesmo split e avaliados no mesmo conjunto de teste — o
pareamento é o que torna os testes válidos:

| id                 | Classificador          | Origem              |
|--------------------|------------------------|---------------------|
| `distancia_minima` | Distância Mínima       | Lab 1               |
| `delta_ova`        | Regra Delta OvA        | Lab 2               |
| `bayes`            | Bayes Ótimo (QDA)      | Lab 4               |
| `naive`            | Naive Bayes            | Lab 4               |
| `floresta`         | Floresta Aleatória     | Seminário           |

São 10 pares possíveis, todos testados de uma vez na sub-aba **Todos os pares**.

---

## 8. Resultados no Iris

### 8.1 Pétalas (split 70/30, semente 42) — o cenário "tudo 100%"

| Classificador      | MCC     |
|--------------------|---------|
| Distância Mínima   | 1,0000  |
| Bayes Ótimo (QDA)  | 1,0000  |
| Naive Bayes        | 1,0000  |
| Floresta Aleatória | 1,0000  |
| Regra Delta OvA    | 0,6124  |

Quatro classificadores empatam em 1,0000 — e os três testes concordam: `0/3`
selos para todos os pares entre eles (`p = 1,0`, IC `[0, 0]`). **Empate perfeito
é empate estatístico.** Só a Regra Delta OvA se separa, com `3/3` selos em todos
os seus pares (`p < 0,0001` no McNemar).

O caso `Bayes × Delta OvA` em detalhe:

```
tabela de McNemar:  a=30  b=15  c=0  d=0
método:             binomial exato (15 < 25)
p McNemar:          0,000061  = 2·(0,5)¹⁵
Δ MCC:              +0,3876
IC 95% bootstrap:   [+0,278, +0,482]   (não contém zero)
p permutação:       0,0010
teste Z de Kappa:   Z = 36,74   p ≈ 0
```

### 8.2 Sépalas — o cenário interessante

Trocando os atributos, os classificadores se aproximam e os testes passam a
discordar entre si — que é justamente onde a discussão fica rica:

| Par                                | Δ MCC   | IC 95%             | p McNemar | p Permut. | Selos |
|------------------------------------|---------|--------------------|-----------|-----------|-------|
| Delta OvA × Naive Bayes            | −0,7718 | [−0,900, −0,601]   | < 0,0001  | 0,0017    | 3/3   |
| Distância Mínima × Delta OvA       | +0,7399 | [+0,566, +0,894]   | < 0,0001  | 0,0017    | 3/3   |
| Delta OvA × Bayes                  | −0,7047 | [−0,866, −0,521]   | 0,0001    | 0,0017    | 3/3   |
| Delta OvA × Floresta               | −0,6013 | [−0,769, −0,400]   | 0,0013    | 0,0033    | 3/3   |
| **Naive Bayes × Floresta**         | +0,1705 | [+0,036, +0,317]   | 0,0625    | 0,0532    | **1/3** |
| Distância Mínima × Floresta        | +0,1386 | [−0,001, +0,301]   | 0,2188    | 0,1015    | 0/3   |
| Bayes × Floresta                   | +0,1034 | [−0,034, +0,253]   | 0,3750    | 0,1664    | 0/3   |
| Bayes × Naive Bayes                | −0,0671 | [−0,180, +0,000]   | 0,5000    | 0,4958    | 0/3   |
| Distância Mínima × Bayes           | +0,0352 | [−0,089, +0,174]   | 1,0000    | 0,7537    | 0/3   |
| Distância Mínima × Naive Bayes     | −0,0319 | [−0,125, +0,065]   | 1,0000    | 0,4742    | 0/3   |

O par **Naive Bayes × Floresta** é o exemplo didático da aba: o bootstrap acusa
diferença (IC não contém zero), mas o McNemar (`p = 0,0625`) e a permutação
(`p = 0,0532`) ficam logo acima de 5%. É um caso de **limiar**, em que a
recomendação é reportar o intervalo de confiança em vez de um sim/não.

---

## 9. Como usar na interface

1. Abrir a aba **Métricas** → sub-aba **Significância**.
2. Em **Testar um par**: escolher classificador A, classificador B e a métrica.
   O padrão já vem em MCC, como pedido.
3. Ajustar reamostragens e permutações se quiser intervalos mais estáveis.
4. Ler o **veredito** (quantos dos 3 testes rejeitaram H₀), a tabela de McNemar,
   o histograma do bootstrap e o contraste com o teste Z clássico.
5. Botão **Ver cálculos** — memória de cálculo completa, com as fórmulas em
   LaTeX e a substituição numérica de cada teste.
6. Em **Todos os pares**: os 10 pares de uma vez, ordenados pelo p-valor do
   McNemar, com os selos M / B / P.

---

## 10. Como conciliar os três testes

- **Concordam (3/3 ou 0/3):** conclusão sólida, é o caso mais comum.
- **McNemar diverge do bootstrap:** o McNemar é o mais conservador, porque olha
  só acertos e ignora *qual* classe foi predita; se a métrica escolhida é
  sensível à distribuição dos erros (MCC, F1), o bootstrap enxerga uma diferença
  que o McNemar não vê.
- **Diferença no limiar (p entre 0,03 e 0,07):** reportar o IC 95% do bootstrap,
  não o sim/não. Nesse regime o resultado depende muito do split.

---

*Disciplina: Tópicos Especiais em Inteligência Artificial · UEPB · 2026*
*Grupo: Erick Nathan · Laura Barbosa · Pedro Lucas*
