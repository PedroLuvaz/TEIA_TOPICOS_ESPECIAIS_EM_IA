# Reconhecimento de Padrões - Iris Dataset (Laboratório de Inteligência Artificial)

Este projeto é uma aplicação científica completa para modelagem, visualização e classificação de padrões sobre o famoso conjunto de dados **Iris**. Ele implementa diversos classificadores clássicos a partir do zero (usando apenas **Python puro e Álgebra Linear**, sem bibliotecas de Machine Learning como Scikit-Learn ou NumPy). A única exceção é o Lab 5, item (ii), onde o próprio enunciado permite explicitamente o uso do `scikit-learn` para treinar a rede feedforward comparada com os classificadores de Bayes.

O projeto é equipado com uma interface gráfica rica e interativa desenvolvida em Tkinter e integrada ao ambiente R para testes de hipóteses avançados.

---

## 📂 Estrutura do Projeto

```text
.
├── data/
│   └── Iris data.xls               # Base de dados original
├── docs/
│   ├── formulario.md               # Formulário resumo de todas as equações do projeto
│   ├── guia_professor.md           # Roteiro didático e guia de defesa do projeto
│   ├── teoria_completa.md          # Manual teórico completo sobre os classificadores lineares
│   ├── lab_03/                     # Documentação específica de métricas avançadas (Lab 3)
│   │   ├── teoria_lab03.md
│   │   ├── item_02.md
│   │   └── item_03.md
│   ├── lab_04/                     # Documentação específica de Bayes & Normalidade (Lab 4)
│   │   ├── teoria_lab04.md
│   │   └── relatorio_experimentos.md
│   └── lab_05/                     # Documentação específica de Feedforward/Backprop (Lab 5)
│       ├── teoria_lab05.md
│       └── relatorio_experimentos.md
├── iris_classifier/
│   ├── classifier.py               # Lógica de treino/predição (Distância Mínima)
│   ├── perceptron.py               # Algoritmo de aprendizado do Perceptron de Rosenblatt
│   ├── delta_rule.py               # Algoritmo da Regra Delta (Widrow-Hoff / Adaline)
│   ├── bayes_classifier.py         # Classificador Bayes Ótimo (QDA) e Naive Bayes (Python puro)
│   ├── mvn_tester.py               # Integração com R (pacote MVN) para testes de normalidade
│   ├── metricas_avancadas.py       # Cálculo de Kappa, Tau, variâncias, Z-test, Fb, MCC
│   ├── models/mlp_backprop.py      # Rede feedforward + backprop do zero (Lab 5, item i — Python puro)
│   ├── models/mlp_sklearn.py       # Wrapper do MLPClassifier para o Iris (Lab 5, item ii — sklearn permitido)
│   ├── lab05_galinha_homem.py      # Script demonstrativo do Lab 5, item i (reproduz os valores do slide)
│   ├── lab05_exercicio_fig1232.py  # Exercicio extra do Lab 5.1 (slide 34): rede Fig. 12.32, 1 iteracao
│   ├── lab05_exercicio_xor.py      # Exercicio XOR do Lab 5.0 (slide 36): XOR com MLP, 1 epoca
│   ├── data_loader.py              # Leitura do Excel e separação estratificada dos dados
│   ├── evaluator.py                # Cálculo de acurácia básica e matriz de confusão
│   ├── main.py                     # Ponto de entrada CLI (orquestra todos os experimentos)
│   ├── math_utils.py               # Álgebra Linear do zero (inversão Gauss-Jordan, det, cov)
│   ├── visualizer.py               # Geração de gráficos e contours de decisão (Matplotlib)
│   ├── run_gui.py                  # Inicialização da Interface Gráfica (Tkinter)
│   └── gui/                        # Interface Gráfica do Usuário (GUI)
│       ├── app.py                  # Janela principal e controle de abas
│       ├── theme.py                # Design System (Slate Light, tipografia e estilos ttk)
│       ├── widgets.py              # Componentes visuais personalizados (Cards, KPI blocks)
│       ├── tab_distancia_minima.py # Painel do Classificador de Distância Mínima
│       ├── tab_perceptron_delta.py  # Painel de Perceptron e Regra Delta (convergência, XOR)
│       ├── tab_metricas_avancadas.py# Painel com simulações, matrizes editáveis e teste Z
│       ├── tab_bayes.py            # Painel dos Classificadores Probabilísticos de Bayes
│       ├── tab_xor.py              # Painel do Lab 5.0 — XOR (MLP) + exemplo didático (slides 36-37)
│       ├── tab_feedforward.py      # Painel do Lab 5.1 — Feedforward (MLP) e Backpropagation (Iris)
│       └── janela_calculos.py      # Memória de Cálculo LaTeX dinâmica por aba
├── web_app/                        # Interface Web (React + FastAPI)
│   ├── backend/                    # API FastAPI — expõe os modelos via JSON
│   │   ├── main.py                 # App, CORS e entrega do build em produção
│   │   ├── core.py                 # Carregamento de dados, split e cache
│   │   ├── lab5_config.py          # Configurações dos exercícios do Lab 5
│   │   └── routers/                # Um router por laboratório
│   └── frontend/                   # React 18 + Vite + TypeScript + Tailwind v4
│       └── src/
│           ├── lib/                # Cliente da API e tipos
│           ├── components/         # Design system e visualizações
│           └── pages/              # Uma página por laboratório
├── outputs/                        # Gráficos e resultados gerados automaticamente
└── requirements.txt                # Dependências de execução (xlrd, matplotlib, pillow, scikit-learn)
```

---

## ⚙️ Pré-requisitos e Instalação

As bibliotecas externas em Python são o `xlrd` (leitura do `.xls`), `matplotlib` (plotagem), `pillow` (renderização de imagens) e o `scikit-learn` (usado apenas no Lab 5, item ii, para a rede feedforward comparada com Bayes). O laboratório opcionalmente requer uma instalação funcional de **R** com o pacote **MVN** instalado para os testes de normalidade multivariada (caso não esteja instalado, o programa usa um fallback seguro de resultados pré-calculados).

Para instalar as dependências de Python:

```bash
pip install -r requirements.txt
```

---

## 🚀 Como Executar

### 1. Modo CLI (Terminal)
Para rodar todos os experimentos sequenciais de todos os laboratórios (incluindo cálculo de matrizes, acurácias, curvas de erro, testes de normalidade no R e teste Z de Kappa), gerando os gráficos de fronteira na pasta `outputs/`:

```bash
python iris_classifier/main.py
```

### 2. Modo GUI (Interface Gráfica)
Para abrir o laboratório interativo multimapas de visualização:

```bash
python iris_classifier/run_gui.py
```

### 3. Modo Web (React + FastAPI)
Interface web com os mesmos experimentos, reutilizando integralmente os modelos
em Python puro. Em desenvolvimento, use dois terminais:

```bash
python -m uvicorn web_app.backend.main:app --reload --port 8000
```

```bash
npm --prefix web_app/frontend run dev
```

Acesse **http://localhost:5173**. Para rodar em produção num único servidor,
gere o build e suba apenas o backend (que passa a servir a interface em
http://localhost:8000):

```bash
npm --prefix web_app/frontend run build
python -m uvicorn web_app.backend.main:app --port 8000
```

> Na primeira vez, instale as dependências do frontend com
> `npm --prefix web_app/frontend install`.
> O guia completo — arquitetura, rotas da API e solução de problemas — está em
> [`docs/interface_web.md`](docs/interface_web.md).

---

## 🖥️ Recursos e Abas da Interface Gráfica

A GUI do projeto foi projetada seguindo regras estéticas premium (**Slate Light Mode**, tipografia refinada e transições suaves), dividindo-se em:

### **Aba 1: Distância Mínima**
*   Classificação baseada em protótipos de classes.
*   Alterna entre pétalas `[2,3]` e sépalas `[0,1]` em tempo real.
*   Interface para classificar amostra manual e plot de retas de fronteiras de decisão.

### **Aba 2: Perceptron & Regra Delta**
*   Treinamento interativo iterativo de classificadores lineares.
*   Exibição em tempo real da curva de convergência (Erros/Época ou MSE/Época).
*   Demonstração prática da separabilidade linear vs sobreposição de classes e a impossibilidade teórica da resolução do **XOR** sob um limite linear (MSE estacionando em $0.25$).

### **Aba 3: Métricas Avançadas**
*   Cálculo automático de estatísticas robustas: Coeficientes **Kappa ($K$)** e **Tau ($\tau$)**, suas variâncias e teste de hipóteses Z.
*   Métricas binárias detalhadas: Sensibilidade, Especificidade, F1-Score, F2-Score e Coeficiente de Matthews (MCC).
*   Sub-aba interativa para auditar e testar o exercício clássico do slide 15 do Prof. Robson (comparando Matrizes de Confusão A e B em tempo real).

### **Aba 4: Bayes & Normalidade**
*   Classificação Bayesianas Probabilística por meio do **Bayes Ótimo (QDA)** e do **Naive Bayes**.
*   Conexão em tempo real ao R para testar a Aderência à Normalidade Multivariada das classes via testes de **Henze-Zirkler** e **Mardia**.
*   Superfícies de decisão não-lineares (parabólicas e hiperbólicas) geradas dinamicamente via contour lines de log-probabilidade.

### **Aba 5 · Lab 5.0: XOR (MLP)**
*   **Exemplo didático (slide 37):** memória de cálculo passo a passo de uma rede 2-2-2 genérica, com bias único compartilhado por camada — reproduz exatamente os valores dos slides 38-42, incluindo a 2ª iteração completa.
*   **Exercício XOR (slide 36):** resolve o XOR com a arquitetura mínima da Fig. 12.28(b) (2 entradas → 2 ocultos → 1 saída) em 1 época, com sua própria memória de cálculo em LaTeX.
*   **Painel interativo:** fronteira de decisão 2D ao vivo (mapa de calor da saída da rede) e curva de convergência, com botões para treinar além da 1 época exigida e observar o XOR sendo efetivamente resolvido (algo que a Regra Delta linear, na Aba 2, nunca consegue fazer).

### **Aba 6 · Lab 5.1: Feedforward (MLP)**
*   **Item (i):** memória de cálculo passo a passo da rede 2-2-2 "galinha vs homem" (Python puro) — alimentação adiante, deltas da retropropagação e pesos atualizados, reproduzindo os valores exatos do slide da Aula PR_711.
*   **Bônus interativo:** um canvas de 8x8 pixels pintável à mão livre (inspirado na Figura do slide), classificado em tempo real por uma segunda rede própria (64 entradas → 10 ocultos → 1 saída), com visualização ao vivo da saída e da ativação de cada neurônio da camada oculta.
*   **Item (ii):** comparação da rede feedforward (`scikit-learn`, único ponto do projeto com biblioteca de ML) com o Bayes Ótimo e o Naive Bayes na classificação do Iris — tabela com todas as métricas de qualidade (Acerto Global, Kappa, Tau, Precisão, Recall, F1, F2, MCC) e testes Z de significância entre os 3 modelos.
*   **Exercício extra (slide 34):** treina a rede maior do exemplo completo da aula (Fig. 12.32) por 1 iteração, com sua própria memória de cálculo em LaTeX.

---

## 📐 Janela de Memória de Cálculo (LaTeX)
Cada aba possui um botão dedicado para abrir a **Janela de Memória de Cálculo**. Ela renderiza, utilizando a sintaxe **LaTeX** e o motor mathtext do Matplotlib:
1. As fórmulas teóricas do laboratório.
2. A referência exata ao arquivo e linha de código (via módulo `inspect`) onde a função matemática está programada.
3. A substituição passo a passo dos parâmetros do modelo com os valores reais da base Iris e o cálculo final da predição.

---

## 👥 Grupo e Autores

*   **Erick Nathan**
*   **Laura Barbosa**
*   **Pedro Lucas**

*Universidade Estadual da Paraíba (UEPB) · Tópicos Especiais em Inteligência Artificial · 2026*
