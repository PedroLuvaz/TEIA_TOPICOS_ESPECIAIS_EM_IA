# Reconhecimento de Padrões - Iris Dataset (Laboratório de Inteligência Artificial)

Este projeto é uma aplicação científica completa para modelagem, visualização e classificação de padrões sobre o famoso conjunto de dados **Iris**. Ele implementa diversos classificadores clássicos a partir do zero (usando apenas **Python puro e Álgebra Linear**, sem bibliotecas de Machine Learning como Scikit-Learn ou NumPy).

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
│   └── lab_04/                     # Documentação específica de Bayes & Normalidade (Lab 4)
│       ├── teoria_lab04.md
│       └── relatorio_experimentos.md
├── iris_classifier/
│   ├── classifier.py               # Lógica de treino/predição (Distância Mínima)
│   ├── perceptron.py               # Algoritmo de aprendizado do Perceptron de Rosenblatt
│   ├── delta_rule.py               # Algoritmo da Regra Delta (Widrow-Hoff / Adaline)
│   ├── bayes_classifier.py         # Classificador Bayes Ótimo (QDA) e Naive Bayes (Python puro)
│   ├── mvn_tester.py               # Integração com R (pacote MVN) para testes de normalidade
│   ├── metricas_avancadas.py       # Cálculo de Kappa, Tau, variâncias, Z-test, Fb, MCC
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
│       └── janela_calculos.py      # Memória de Cálculo LaTeX dinâmica por aba
├── outputs/                        # Gráficos e resultados gerados automaticamente
└── requirements.txt                # Dependências básicas de execução (xlrd, matplotlib, pillow)
```

---

## ⚙️ Pré-requisitos e Instalação

As únicas bibliotecas externas em Python são o `xlrd` (leitura do `.xls`), `matplotlib` (plotagem) e `pillow` (renderização de imagens). O laboratório opcionalmente requer uma instalação funcional de **R** com o pacote **MVN** instalado para os testes de normalidade multivariada (caso não esteja instalado, o programa usa um fallback seguro de resultados pré-calculados).

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
