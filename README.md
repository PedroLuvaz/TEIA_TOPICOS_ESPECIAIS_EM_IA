# Classificador de Distância Mínima - Iris Dataset

Este projeto implementa um **Classificador de Distância Mínima** a partir do zero (usando apenas Python puro e Álgebra Linear, sem bibliotecas de Machine Learning como Scikit-Learn ou NumPy). O objetivo é classificar amostras do famoso conjunto de dados Iris.

## 📂 Estrutura do Projeto

```text
.
├── data/
│   └── Iris data.xls             # Base de dados original
├── docs/
│   └── guia_professor.md         # Explicação detalhada da matemática e do código
├── iris_classifier/
│   ├── classifier.py             # Lógica de treinamento e predição
│   ├── data_loader.py            # Leitura do Excel e separação estratificada dos dados
│   ├── evaluator.py              # Cálculo de acurácia e matriz de confusão
│   ├── main.py                   # Ponto de entrada CLI (orquestra os experimentos)
│   ├── math_utils.py             # Operações matemáticas puras (vetores, médias, etc)
│   ├── visualizer.py             # Geração de gráficos (Matplotlib)
│   ├── run_gui.py                # Ponto de entrada da interface gráfica (Tkinter)
│   └── gui/                      # Pacote da GUI — preparado para novas abas
│       ├── app.py                # Janela principal (cabeçalho + notebook + rodapé)
│       ├── theme.py              # Paleta editorial escura, tipografia, estilos ttk
│       ├── widgets.py            # Cartões e blocos de métrica reutilizáveis
│       └── tab_distancia_minima.py  # Aba do Classificador de Distância Mínima
├── outputs/                      # Pasta onde os gráficos são salvos automaticamente
├── README.md                     # Este arquivo
└── requirements.txt              # Dependências do projeto
```

## ⚙️ Pré-requisitos e Instalação

As únicas bibliotecas externas utilizadas são o `xlrd` (para ler o arquivo `.xls` antigo) e o `matplotlib` (exclusivamente para a geração dos gráficos).

Para evitar conflitos com outras bibliotecas na sua máquina, é recomendado o uso de um ambiente virtual (venv).

### 1. Criar e Ativar o Ambiente Virtual

**No Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**No Linux/macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 2. Instalar as Dependências

Com o ambiente virtual ativado, instale as bibliotecas necessárias:

```bash
pip install -r requirements.txt
```

## 🚀 Como Executar

### Modo CLI (terminal)

Para rodar todos os experimentos e gerar os gráficos no terminal:

```bash
python iris_classifier/main.py
```

### Modo GUI (interface gráfica — Tkinter)

Para abrir a interface visual com tema escuro, métricas em tempo real, gráficos interativos e classificação manual:

```bash
python iris_classifier/run_gui.py
```

A janela já está estruturada com **abas** prontas para receber implementações futuras (Aba 2, Aba 3, Aba 4). A aba **Distância Mínima** continua usando exclusivamente Python puro.

**Grupo:** Erick Nathan · Laura Barbosa · Pedro Lucas

## 🔄 Fluxo de Execução (`main.py`)

O código segue o seguinte pipeline:

1. **Carregamento:** Lê o arquivo `Iris data.xls` ignorando o cabeçalho.
2. **Separação (Split Estratificado):** Divide os dados em **70% para treino** e **30% para teste**, garantindo que a proporção de cada classe seja mantida (35 amostras de treino e 15 de teste para cada uma das 3 classes).
3. **Experimentos i e ii (Classificação Multiclasse):**
   - Calcula os protótipos (vetores médios) das 3 classes usando os dados de treino.
   - Aplica a Função Discriminante para todas as amostras de teste.
   - Retorna a predição (classe com maior valor de função discriminante), a acurácia geral e a Matriz de Confusão.
4. **Visualização Geral:** Plota e salva um gráfico de dispersão com todas as amostras.
5. **Experimento iii (Superfícies de Decisão):**
   - Isola as classes em pares (A vs B, B vs C, C vs A).
   - Recalcula os protótipos apenas para os dados daquele par.
   - Calcula os coeficientes da reta de decisão ($w^T \cdot x + b = 0$).
   - Avalia a acurácia binária e gera/salva o gráfico da fronteira de decisão.

Para uma explicação aprofundada das fórmulas matemáticas e de como apresentar este projeto, consulte o arquivo **[docs/guia_professor.md](docs/guia_professor.md)**.
