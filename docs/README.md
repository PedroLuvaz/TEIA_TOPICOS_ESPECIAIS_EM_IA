# Índice da documentação

Todo o material escrito do projeto, organizado **por laboratório**. Cada seção
diz onde está a teoria, onde está o relatório de resultados, qual módulo do
código implementa aquilo e em que tela do aplicativo o assunto aparece.

**Projeto:** Reconhecimento de Padrões — Tópicos Especiais em IA (UEPB)
**Equipe:** Erick Nathan · Laura Barbosa · Pedro Lucas

---

## Comece por aqui

| Documento | Para quê |
|---|---|
| [`defesa_projeto.md`](defesa_projeto.md) | **O guia único da defesa.** Costura tudo: requisitos da entrega, teoria dos sete modelos, métricas, testes de significância, arquitetura, resultados medidos e banco de perguntas |
| [`../GUIA_DO_PROFESSOR.pdf`](../GUIA_DO_PROFESSOR.pdf) | O mesmo guia em PDF, pronto para imprimir, com o mapa da documentação |
| [`../TUTORIAL_RODAR_PROJETO.md`](../TUTORIAL_RODAR_PROJETO.md) | Como instalar e abrir o projeto em dois cliques, sem terminal |
| [`../README.md`](../README.md) | Visão geral do repositório e estrutura de pastas |
| [`interface_web.md`](interface_web.md) | Arquitetura da interface web e todas as rotas da API |

Se for ler um documento só, leia o `defesa_projeto.md`. Os documentos por
laboratório abaixo entram no detalhe de cada assunto.

---

## Lab 1 · Classificador de Distância Mínima

Protótipos (vetores médios), função discriminante linear e as fronteiras de
decisão entre cada par de classes.

| Documento | O que tem dentro |
|---|---|
| [`lab_01/teoria_lab01.md`](lab_01/teoria_lab01.md) | Enunciado da aula PR3, split estratificado, protótipos, a equivalência entre distância mínima e classificador de máximo, e a dedução da superfície de decisão |
| [`lab_01/relatorio_experimentos.md`](lab_01/relatorio_experimentos.md) | Protótipos calculados, acurácias, matrizes de confusão e as equações das três fronteiras, item por item do enunciado |
| [`teoria_completa.md`](teoria_completa.md) | §1 a §10: material de estudo mais extenso, com o pipeline completo |
| [`formulario.md`](formulario.md) | Folha de fórmulas: protótipo, distância euclidiana, discriminante, coeficientes da fronteira |

**Código:** `iris_classifier/models/classifier.py` · `iris_classifier/core/math_utils.py`
**Tela:** *Distância Mínima* · **API:** `/api/distancia-minima/*`

---

## Lab 2 · Perceptron e Regra Delta

Aprendizado por correção de erro (Rosenblatt) e por gradiente descendente
(Widrow-Hoff), o esquema Um-Contra-Todos e o limite do XOR.

| Documento | O que tem dentro |
|---|---|
| [`lab_02/teoria_lab02.md`](lab_02/teoria_lab02.md) | Enunciados da aula PR4, neurônio linear, regra do Perceptron e o teorema da convergência, Regra Delta e o MSE por época, Um-Contra-Todos e a prova de que o XOR não tem solução linear |
| [`lab_02/relatorio_experimentos.md`](lab_02/relatorio_experimentos.md) | Vetores de pesos, épocas até convergir, curvas de MSE e o XOR estagnado em 0,26 — item por item do enunciado |
| [`teoria_completa.md`](teoria_completa.md) | §11 a §16: material de estudo mais extenso |
| [`formulario.md`](formulario.md) | Regras de atualização de pesos, MSE, comparação Perceptron × Delta |

**Código:** `iris_classifier/models/perceptron.py` · `iris_classifier/models/delta_rule.py`
**Tela:** *Perceptron & Delta* · **API:** `/api/perceptron-delta/*`

---

## Lab 3 · Métricas avançadas e significância

Onde mora a resposta ao pedido de *"métricas de qualidade e comparação de
modelos com testes de significância"*.

| Documento | O que tem dentro |
|---|---|
| [`lab_03/teoria_lab03.md`](lab_03/teoria_lab03.md) | Matriz de confusão, acerto global, acerto casual, Kappa e sua variância (Congalton & Green), Tau, métricas por classe e o teste Z |
| [`lab_03/testes_significancia.md`](lab_03/testes_significancia.md) | Por que o teste Z é conservador com modelos pareados, e os três testes que resolvem: McNemar, bootstrap pareado e permutação; MCC multiclasse |
| [`lab_03/item_02.md`](lab_03/item_02.md) | Item 2 do laboratório: comparação dos classificadores no Iris, com Kappa e Tau lado a lado |
| [`lab_03/item_03.md`](lab_03/item_03.md) | Item 3: exercício do slide 15 (matrizes A e B, 4 classes) resolvido passo a passo |

**Código:** `iris_classifier/evaluation/metricas_avancadas.py` · `testes_significancia.py` · `validacao_cruzada.py`
**Tela:** *Métricas Avançadas* (validação cruzada · split único · significância · matriz editável) · **API:** `/api/metricas/*`

---

## Lab 4 · Bayes Ótimo e Naive Bayes

Classificadores generativos: modelar a distribuição de cada classe em vez de
apenas traçar fronteiras.

| Documento | O que tem dentro |
|---|---|
| [`lab_04/teoria_lab04.md`](lab_04/teoria_lab04.md) | Normal multivariada, estimação de média e covariância, regularização de Ridge, discriminante do QDA, distância de Mahalanobis e a simplificação do Naive Bayes |
| [`lab_04/relatorio_experimentos.md`](lab_04/relatorio_experimentos.md) | Resultados no Iris, testes de normalidade (Henze-Zirkler e Mardia) e comparação entre os dois classificadores |

**Código:** `iris_classifier/models/bayes_classifier.py` · `iris_classifier/evaluation/mvn_tester.py`
**Tela:** *Bayes & Normalidade* · **API:** `/api/bayes/*`

---

## Lab 5 · Redes neurais (MLP e backpropagation)

| Documento | O que tem dentro |
|---|---|
| [`lab_05/teoria_lab05.md`](lab_05/teoria_lab05.md) | Neurônio artificial, sigmoide, feedforward, dedução do backpropagation, o XOR com camada oculta e os exercícios dos slides |
| [`lab_05/relatorio_experimentos.md`](lab_05/relatorio_experimentos.md) | Item (i) — rede em Python puro; item (ii) — rede com scikit-learn comparada a Bayes; resultados de cada exercício |

**Código:** `iris_classifier/models/mlp_backprop.py` (rede do zero) · `mlp_multiclasse.py` (classificador multiclasse) · `mlp_sklearn.py` (item ii)
**Telas:** *Lab 5.0 · XOR*, *Lab 5.1 · Feedforward* e *Construtor de Rede* · **API:** `/api/lab5/*`

---

## Seminário · Florestas Aleatórias

O modelo apresentado na defesa do seminário, disponível no aplicativo conforme
pedido pelo professor.

| Documento | O que tem dentro |
|---|---|
| [`seminario_florestas_aleatorias.md`](seminario_florestas_aleatorias.md) | Árvore CART, impureza de Gini e entropia, bagging, subespaço aleatório de atributos, voto majoritário, erro out-of-bag e importância dos atributos |
| [`seminario_dataset_fim_de_semana.md`](seminario_dataset_fim_de_semana.md) | A base categórica do seminário levada a 1000 instâncias, com 8% de ruído: como foi gerada e o que o ruído significa no teto de acerto |

**Código:** `iris_classifier/models/random_forest.py` · `floresta_categorica.py` (ID3 multi-way) · `iris_classifier/data/gerar_fim_de_semana.py` · `seminario/` (animações)
**Tela:** *Florestas Aleatórias* · **API:** `/api/floresta/*`

---

## Entrega final

Os quatro pedidos do professor e o que foi construído para atender a cada um.

| Documento | O que tem dentro |
|---|---|
| [`defesa_projeto.md`](defesa_projeto.md) | Guia único: requisitos × implementação, teoria dos sete modelos, métricas, testes, arquitetura, resultados e perguntas prováveis |
| [`classificar_modelos.md`](classificar_modelos.md) | Tela *Classificar*: catálogo dos sete modelos, o que cada hiperparâmetro faz e como a comparação entre modelos funciona |
| [`importar_dados_txt.md`](importar_dados_txt.md) | Importação da base do usuário em `.txt`: formato aceito, heurísticas de leitura, limites e problemas comuns |
| [`interface_web.md`](interface_web.md) | Arquitetura da interface web, o que cada página faz e a tabela completa de rotas da API |
| [`../TUTORIAL_RODAR_PROJETO.md`](../TUTORIAL_RODAR_PROJETO.md) | Instalação e execução para quem nunca abriu o projeto |

**Código:** `iris_classifier/data/leitor_texto.py` (leitor de `.txt`) · `web_app/backend/modelos.py` (catálogo de modelos) · `datasets_usuario.py` · `routers/classificar.py`
**Telas:** *Classificar* e o botão *Importar .txt*, presente em todas as páginas

---

## Mapa rápido: pergunta → documento

| Se a pergunta for… | Vá para |
|---|---|
| "Como eu rodo isso?" | [`../TUTORIAL_RODAR_PROJETO.md`](../TUTORIAL_RODAR_PROJETO.md) |
| "Onde está o requisito X da entrega?" | [`defesa_projeto.md` §1](defesa_projeto.md) |
| "Como funciona este modelo?" | Seção do laboratório correspondente, acima |
| "O que essa métrica significa?" | [`lab_03/teoria_lab03.md`](lab_03/teoria_lab03.md) |
| "Essa diferença é significativa?" | [`lab_03/testes_significancia.md`](lab_03/testes_significancia.md) |
| "Como troco o modelo e os parâmetros?" | [`classificar_modelos.md`](classificar_modelos.md) |
| "Como uso minha própria base?" | [`importar_dados_txt.md`](importar_dados_txt.md) |
| "Que rota da API faz isso?" | [`interface_web.md` §4](interface_web.md) |
| "Qual fórmula era mesmo?" | [`formulario.md`](formulario.md) |

---

## Nota de organização

Dois documentos foram removidos por terem sido absorvidos pelo
[`defesa_projeto.md`](defesa_projeto.md): o antigo `guia_professor.md` (roteiro
de apresentação que ainda descrevia as abas da interface desktop) e o
`perguntas_prova.md` (banco de questões das provas dos Labs 1 e 2). O conteúdo
equivalente está no guia da defesa — roteiro de demonstração na §2 e perguntas
prováveis na §10. O histórico do Git preserva as versões antigas.
