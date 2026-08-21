# Contexto e Regras do Projeto — CLAUDE.md

Este arquivo instrui o agente Claude ao atuar neste projeto. Sempre obedeça às diretrizes abaixo.

---

## 1. Objetivo do Projeto

- Projeto acadêmico: **Tópicos Especiais em IA** (TEIA)
- Começou como um **Classificador de Distância Mínima** para a base **Iris** e cresceu para uma aplicação com sete classificadores, todos escritos do zero.
- Três experimentos originais (Lab 1):
  1. Cálculo de protótipos (vetores médios) e classificação multiclasse (3 classes).
  2. Função discriminante linear e regra de decisão por máximo.
  3. Superfícies de decisão (fronteiras lineares) para todos os pares de classes.
- Requisitos da **entrega final** (definidos pelo professor):
  1. O usuário escolhe o **modelo** e **parametriza** o modelo — aba *Classificar*.
  2. O aplicativo é alimentado pela **base do usuário em .txt** — botão *Importar .txt*.
  3. **Métricas de qualidade** e **comparação de modelos com testes de significância**.
  4. O **modelo do seminário** (Florestas Aleatórias) disponível no aplicativo.

---

## 2. Restrições Técnicas Estritas

- **PROIBIDO:** `numpy`, `scipy`, `pandas` — qualquer biblioteca de ML ou álgebra avançada. Isso vale também para a leitura de dados: o leitor de `.txt` do usuário (`iris_classifier/data/leitor_texto.py`) é Python puro, sem `pandas` e sem o módulo `csv`.
- **OBRIGATÓRIO:** Toda a matemática (produto escalar, subtração de vetores, distâncias, médias, covariâncias, gradientes, impureza) DEVE ser feita em **Python puro** com laços `for`, listas nativas e `zip`, em `iris_classifier/core/math_utils.py` e nos módulos de `iris_classifier/models/`.
- **Única exceção:** `scikit-learn` no Lab 5, item (ii) — permitido explicitamente pelo enunciado daquele item, e isolado em `models/mlp_sklearn.py`.
- **Bibliotecas externas permitidas:** `xlrd`/`openpyxl` (planilhas), `matplotlib` (gráficos), `fastapi`/`uvicorn` (interface web).
- O backend web **não reimplementa matemática**: os routers apenas orquestram chamadas aos módulos de `iris_classifier/`.

---

## 3. Estrutura de Arquivos

```
TEIA_TOPICOS_ESPECIAIS_EM_IA/
├── iris_classifier/
│   ├── main.py              # Orquestrador CLI de todos os laboratorios
│   ├── run_gui.py           # Interface desktop (Tkinter)
│   ├── core/math_utils.py   # Algebra linear em Python puro
│   ├── data/
│   │   ├── data_loader.py   # Leitura das planilhas e split estratificado
│   │   ├── leitor_texto.py  # Leitor generico do .txt do usuario (Python puro)
│   │   └── gerar_fim_de_semana.py
│   ├── models/              # classifier, perceptron, delta_rule, bayes_classifier,
│   │                        # mlp_backprop, mlp_multiclasse, random_forest,
│   │                        # floresta_categorica, mlp_sklearn
│   ├── evaluation/          # evaluator, metricas_avancadas, testes_significancia,
│   │                        # validacao_cruzada, mvn_tester
│   ├── visualization/       # visualizer.py (matplotlib)
│   └── gui/                 # Abas da interface desktop
├── web_app/
│   ├── backend/             # FastAPI: main, core, modelos (catalogo),
│   │   └── routers/         # datasets_usuario, traco, lab5_config
│   └── frontend/            # React 18 + Vite + TypeScript + Tailwind v4
├── data/
│   ├── Iris data.xls        # Base original — NÃO ALTERAR
│   ├── fim_de_semana_1000.csv
│   ├── exemplos/            # .txt de exemplo para a importacao
│   └── enviados/            # Bases do usuario (nao versionado)
├── docs/                    # defesa_projeto, classificar_modelos,
│                            # importar_dados_txt, interface_web, teoria, labs
├── outputs/                 # Graficos gerados (criado automaticamente)
├── CLAUDE.md · GEMINI.md · README.md · requirements.txt
```

---

## 4. Padrões de Código

- **Idioma:** Todo código, comentários, docstrings, `print`s e documentação em **Português do Brasil**.
- **Split estratificado:** 70% treino / 30% teste *por classe*, com `random.seed(42)`.
- **Atributos padrão:** índices `[2, 3]` (Comprimento e Largura da Pétala) — no Iris.
- **Nomes das classes:** `'setosa'`, `'versicolor'`, `'virginica'` (minúsculas) — no Iris.
- **Nada no código pode assumir "as 3 classes do Iris".** Cada base declara suas classes, features e combinações de atributos (`web_app/backend/core.py`), e as telas perguntam à base. É o que permite o dataset categórico do seminário e as bases `.txt` do usuário rodarem nas mesmas telas.
- **Modelo novo** entra no catálogo `web_app/backend/modelos.py` declarando nome, descrição, esquema de parâmetros e as funções de treino/predição — a interface, os testes de significância e a validação cruzada passam a incluí-lo sozinhos.
- **Saída de gráficos:** sempre em `outputs/`. Nunca exibir com `plt.show()` — sempre `plt.savefig()`.
- **Não alterar** o arquivo `data/Iris data.xls`.

---

## 5. Como Executar

```bash
python iris_classifier/main.py        # CLI: todos os experimentos
python iris_classifier/run_gui.py     # Interface desktop (Tkinter)
```

Interface web (a que atende aos requisitos da entrega):

```bash
python -m uvicorn web_app.backend.main:app --reload --port 8000
npm --prefix web_app/frontend run dev
```

Saída esperada no terminal:
- Total de amostras carregadas: 150
- Treino: 105 | Teste: 45
- Protótipos das 3 classes
- Tabela de scores discriminantes por amostra
- Acurácia geral: 100.00% (pétalas são linearmente separáveis)
- Matriz de confusão
- Métricas por classe (Precisão, Revocação, F1)
- Equações numéricas das fronteiras de decisão (3 pares)
- Acurácia do experimento comparativo com sépalas (~80%)
- Gráficos salvos em `outputs/`

---

## 6. Matemática Central (referência rápida)

**Protótipo:**  `m_j = (1/N_j) · Σ x`  para toda amostra `x` da classe `j`

**Função Discriminante:**  `d_j(x) = xᵀ·m_j − ½·m_j ᵀ·m_j`

**Regra de Decisão:**  `classe = argmax_j d_j(x)`

**Coeficientes da Fronteira:**  `w = m_i − m_j`,  `b = −½·(‖m_i‖² − ‖m_j‖²)`

**Reta no plano 2D:**  `x₂ = (−w₁·x₁ − b) / w₂`
