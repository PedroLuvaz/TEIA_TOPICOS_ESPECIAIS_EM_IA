# Lab 2 — Relatório de Experimentos: Perceptron, Regra Delta e XOR

**Disciplina:** Tópicos Especiais em Inteligência Artificial (TEIA)
**UEPB 2026**
**Grupo:** Erick Nathan · Laura Barbosa · Pedro Lucas
**Referência:** Aula PR4 (Prof. Robson Pequeno de Sousa)

---

## 1. Configuração do experimento

| Parâmetro | Valor |
|---|---|
| Base | Iris — 150 amostras, 4 atributos, 3 classes |
| Split | 70% treino / 30% teste, estratificado e aleatorizado por classe |
| Amostras | 105 de treino · 45 de teste |
| Semente | 42 (resultados reprodutíveis) |
| Pesos iniciais | $w(1) = (0,0,0,0,0)$, conforme o enunciado |
| Taxa — Perceptron | $p = 0{,}03$ |
| Taxa — Regra Delta | $p = 0{,}02$ (ajustável na interface) |
| Épocas — Perceptron | máximo de 100 |
| Épocas — Regra Delta | 200 (ajustável na interface) |

---

## 2. Item (a) — Diagrama de dispersão das três classes

O gráfico de dispersão com as três classes num mesmo plano está na aba
*Perceptron & Delta* do aplicativo, e pode ser desenhado para qualquer par de
atributos pelo seletor **Atributos**:

- **$X_1$ × $X_2$ (sépalas):** as nuvens de versicolor e virgínica se
  interpenetram largamente; só a setosa se destaca. É a visualização que
  antecipa a dificuldade do par versicolor × virgínica.
- **Pétalas:** os três agrupamentos aparecem quase disjuntos, com uma faixa
  estreita de contato entre versicolor e virgínica.

Cada ponto é desenhado preenchido quando pertence ao treino e vazado quando
pertence ao teste, de modo que a separação 70/30 fica visível no próprio
gráfico.

---

## 3. Itens (b), (c) e (e) — Perceptron nas três estratégias binárias

Pesos iniciais zerados, $p = 0{,}03$, limite de 100 épocas.

### 3.1 Com os quatro atributos — $w$ de cinco componentes

| Par | Convergiu? | Épocas | Acurácia treino | Acurácia teste |
|---|---|:---:|:---:|:---:|
| Setosa × Versicolor | **Sim** | 6 | 100,00% | 100,00% |
| Setosa × Virgínica | **Sim** | 5 | 100,00% | 100,00% |
| Versicolor × Virgínica | **Não** | 100 (limite) | 97,14% | 96,67% |

Vetores de pesos ao fim do treinamento, na ordem
$[w_0, w_{\text{comp.sép}}, w_{\text{larg.sép}}, w_{\text{comp.pét}}, w_{\text{larg.pét}}]$:

```
Setosa × Versicolor      w = [+0.0600, +0.1140, +0.2940, -0.4740, -0.2100]
Setosa × Virgínica       w = [+0.1200, +0.1260, +0.4320, -0.7140, -0.2640]
Versicolor × Virgínica   w = [+1.5600, +1.6440, +2.5920, -3.2880, -1.9860]
```

### 3.2 Com o par de pétalas — $w$ de três componentes

| Par | Convergiu? | Épocas | Acurácia treino | Acurácia teste |
|---|---|:---:|:---:|:---:|
| Setosa × Versicolor | **Sim** | 6 | 100,00% | 100,00% |
| Setosa × Virgínica | **Sim** | 5 | 100,00% | 100,00% |
| Versicolor × Virgínica | **Não** | 100 (limite) | 51,43% | 50,00% |

```
Setosa × Versicolor      w = [+0.3000, -0.0780, -0.1020]
Setosa × Virgínica       w = [+0.4200, -0.1200, -0.1440]
Versicolor × Virgínica   w = [+2.7600, -0.7080, -0.5040]
```

### 3.3 Item (e) — o que o vetor de pesos mostrou

O enunciado pede exatamente esta observação, e o resultado é o mais instrutivo
do laboratório.

**Os dois pares separáveis convergem depressa:** 5 e 6 épocas, com erro zero.
Uma vez que nenhuma amostra é mal classificada, o fator $(d-y)$ zera para todas
e o vetor de pesos **congela** — não muda mais, por mais épocas que se rode.

**O par versicolor × virgínica não converge.** Ao fim das 100 épocas ainda
restavam 3 amostras mal classificadas (com 4 atributos) e 2 (com pétalas).
Repare na **escala** dos pesos: no par não separável eles são uma ordem de
grandeza maiores que nos separáveis — $+2{,}59$ e $-3{,}29$ contra $+0{,}29$ e
$-0{,}47$. É o efeito acumulado de cem épocas de correções que nunca se anulam:
o vetor foi empurrado de um lado para o outro sem nunca encontrar repouso.

**O vetor final é um instantâneo arbitrário.** Ele não é ótimo em nenhum
sentido — é apenas onde o algoritmo estava quando o limite de épocas o
interrompeu. A prova está no contraste entre as duas tabelas: com os quatro
atributos o instantâneo calhou de ser bom (96,67% no teste), enquanto com as
pétalas calhou de ser ruim (50,00%, equivalente a jogar uma moeda). Mesmo
algoritmo, mesmos dados, mesma taxa — o que mudou foi apenas em que ponto da
oscilação o relógio parou.

Essa é a diferença prática entre o Perceptron e a Regra Delta: parado no meio de
um problema não separável, o Perceptron não oferece garantia nenhuma sobre o
resultado.

---

## 4. Item (d) — Fluxo de classificação binária

Com os vetores de pesos estimados, a classificação de uma amostra nova segue
sempre os mesmos quatro passos:

```text
   amostra x = [x1, x2, x3, x4]
            │
            ▼
   x_aug = [1, x1, x2, x3, x4]        (acrescenta a entrada do bias)
            │
            ▼
   net = w0·1 + w1·x1 + ... + w4·x4   (produto escalar com o w treinado)
            │
            ▼
   y = sgn(net)                        (função sinal)
            │
     ┌──────┴──────┐
     ▼             ▼
  net ≥ 0       net < 0
  classe +1     classe −1
```

Para o esquema completo das três classes, o mesmo fluxo é executado com os três
vetores de pesos, e a decisão final é o maior `net` — a estratégia Um-Contra-Todos
da seção seguinte. O aplicativo mostra esse fluxo em funcionamento: clicando em
qualquer ponto do gráfico, ele exibe o `net` calculado e a classe resultante.

---

## 5. Regra Delta — estratégia Um-Contra-Todos

Pesos zerados, $p = 0{,}02$, 200 épocas, $d = +1$ para a classe em foco e
$d = -1$ para as demais. Atributos: pétalas.

### 5.1 Pesos e convergência por classificador

| Classificador | $w$ final | MSE inicial | MSE final |
|---|---|:---:|:---:|
| Setosa vs resto | [+1,2466, −0,2581, −0,5589] | 0,2326 | **0,0919** |
| Versicolor vs resto | [−0,6407, −0,1067, +0,1388] | 0,2933 | **0,1842** |
| Virgínica vs resto | [−1,6059, +0,3648, +0,4202] | 0,1615 | **0,0930** |

### 5.2 Desempenho multiclasse

| Conjunto | Acurácia |
|---|:---:|
| Treino (105) | 66,67% |
| Teste (45) | **66,67%** |

### 5.3 Leitura das curvas de convergência

As três curvas caem e estabilizam, mas em **patamares bem diferentes**, e é aí
que está a informação:

- **Setosa vs resto** (MSE 0,092) e **virgínica vs resto** (0,093) descem a
  valores baixos: cada uma dessas classes ocupa uma extremidade do espaço das
  pétalas e é razoavelmente separável do restante por uma reta.
- **Versicolor vs resto** estaciona no dobro disso (0,184). A razão é
  geométrica: a versicolor fica **espremida entre** as outras duas. Não existe
  reta que a isole do resto — as amostras de setosa ficam de um lado e as de
  virgínica do outro, e ambas deveriam receber $d = -1$.

Como o classificador da versicolor nunca produz o maior `net` em lugar nenhum, o
$\arg\max$ praticamente nunca escolhe essa classe. As 15 versicolores do teste
são perdidas, e o acerto se fixa em 30/45 = 66,67% — exatamente as setosas e as
virgínicas.

O número não é um defeito da implementação: é a fronteira do que uma combinação
de discriminantes lineares consegue fazer com uma classe geometricamente
intermediária. O mesmo 66,67% aparece no Perceptron OvA da aba *Classificar*,
por razão idêntica.

### 5.4 Parametrização exigida pelo enunciado

Tanto a taxa de aprendizado quanto o número de épocas são ajustáveis na
interface — sliders na aba *Perceptron & Delta* e no catálogo da aba
*Classificar* —, e o critério de parada é o número de épocas, como pedido. Os
valores padrão são os do enunciado ($p = 0{,}02$).

---

## 6. Regra Delta binária, par a par

Mesmos parâmetros, com $d = +1$ para a primeira classe e $d = -1$ para a
segunda:

| Par | $w$ final | MSE inicial → final | Acurácia teste |
|---|---|:---:|:---:|
| Setosa × Versicolor | [+1,6745, −0,5083, −0,4093] | 0,3332 → **0,0727** | 100,00% |
| Setosa × Virgínica | [+1,4679, −0,2949, −0,5741] | 0,3734 → **0,0503** | 100,00% |
| Versicolor × Virgínica | [+1,5723, −0,3537, −0,4369] | 0,1298 → **0,1401** | 50,00% |

O contraste é nítido. Nos pares separáveis o MSE cai a menos de um quarto do
valor inicial. No par versicolor × virgínica ele **sobe** ligeiramente
(0,1298 → 0,1401) e ali fica: o gradiente empurra os pesos para reduzir o erro
das amostras de uma classe e, ao fazê-lo, aumenta o das outras. A curva plana
num patamar alto é a assinatura visual de um problema que o modelo linear não
resolve.

---

## 7. XOR com a Regra Delta

Padrões e alvos conforme o enunciado — $d = 0$ para $(0,0)$ e $(1,1)$; $d = 1$
para $(0,1)$ e $(1,0)$ — com limiar de decisão em 0,5.

| Épocas | $w$ final | MSE inicial → final |
|:---:|---|:---:|
| 200 | [+0,4929, −0,0061, +0,0040] | 0,4917 → **0,2604** |
| 1000 | [+0,5102, −0,0204, −0,0102] | 0,4917 → **0,2603** |

### 7.1 Saídas da rede treinada (200 épocas)

| Entrada | `net` | Saída | Esperado | |
|:---:|---:|:---:|:---:|:---:|
| (0, 0) | +0,4929 | 0 | 0 | ✔ |
| (0, 1) | +0,4969 | 0 | 1 | ✘ |
| (1, 0) | +0,4867 | 0 | 1 | ✘ |
| (1, 1) | +0,4908 | 0 | 0 | ✔ |

### 7.2 Análise

Os quatro `net` são praticamente iguais, todos rondando **0,49**. Os pesos das
entradas encolheram para perto de zero ($-0{,}006$ e $+0{,}004$) e só o bias
sobreviveu, em 0,49. Ou seja: incapaz de separar os padrões, o algoritmo
convergiu para a melhor resposta constante possível — **responder sempre a média
dos alvos**, que é 0,5.

O MSE reflete isso com precisão: prever 0,5 quando os alvos são 0 e 1 dá erro
quadrático $(0{,}5)^2 = 0{,}25$ por padrão. O valor medido, **0,2604**, está a
menos de meio ponto percentual desse piso teórico. Quintuplicar as épocas — de
200 para 1000 — melhora o MSE na quarta casa decimal (0,2604 → 0,2603) e não
altera nada: **não é uma questão de treinar mais**, e sim de o modelo não
possuir capacidade de representação para a tarefa.

O acerto fica em 2 de 4 padrões — e mesmo esses dois são acidentais, resultado
de todos os `net` caírem logo abaixo do limiar de 0,5.

Este resultado é o argumento histórico que motivou as redes multicamadas: com
uma camada oculta de dois neurônios, o mesmo problema é resolvido com erro
próximo de zero, como mostra o [Lab 5](../lab_05/relatorio_experimentos.md).

---

## 8. Síntese

| Experimento | Resultado | O que ele ensina |
|---|:---:|---|
| Perceptron — Setosa × Versicolor | 100% em 6 épocas | Convergência garantida quando há separabilidade |
| Perceptron — Setosa × Virgínica | 100% em 5 épocas | Idem |
| Perceptron — Versicolor × Virgínica | 96,67% (4 atr.) / 50,00% (pétalas) | Sem separabilidade, o resultado depende de onde o limite de épocas interrompeu |
| Delta binária — pares separáveis | 100%, MSE < 0,08 | Solução de mínimos quadrados, estável |
| Delta binária — par sobreposto | 50%, MSE estagnado | O erro residual mede a sobreposição |
| Delta Um-Contra-Todos | 66,67% | A classe do meio não é isolável por retas |
| XOR | 2/4, MSE 0,2604 ≈ 0,25 | Limite teórico do neurônio único |

Os "fracassos" desta tabela são tão importantes quanto os acertos: eles
delimitam com precisão onde termina o alcance dos classificadores lineares e
justificam os laboratórios seguintes — Bayes (Lab 4), que modela a distribuição
de cada classe, e as redes multicamadas (Lab 5), que quebram a restrição da
fronteira linear.

---

## 9. Onde reproduzir

| Como | O quê |
|---|---|
| Aba **Perceptron & Delta** | Sub-abas Perceptron, Regra Delta, Delta OvA e XOR, com dispersão, fronteira e curva de convergência |
| Aba **Classificar** | Perceptron OvA e Regra Delta OvA no catálogo, com taxa e épocas ajustáveis |
| Botão **Ver teoria e cálculos** | Memória de cálculo com as atualizações de peso passo a passo |
| `python iris_classifier/main.py` | Todos os experimentos no terminal |

Teoria completa em [`teoria_lab02.md`](teoria_lab02.md).
