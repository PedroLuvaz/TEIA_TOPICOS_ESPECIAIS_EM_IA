# Contexto e Regras do Projeto — CLAUDE.md

Este arquivo instrui o agente Claude ao atuar neste projeto. Sempre obedeça às diretrizes abaixo.

---

## 1. Objetivo do Projeto

- Projeto acadêmico: **Tópicos Especiais em IA** (TEIA)
- Implementação de um **Classificador de Distância Mínima** (Minimum Distance Classifier) para a base de dados **Iris** (150 amostras, 4 features, 3 classes).
- Três experimentos obrigatórios:
  1. Cálculo de protótipos (vetores médios) e classificação multiclasse (3 classes).
  2. Função discriminante linear e regra de decisão por máximo.
  3. Superfícies de decisão (fronteiras lineares) para todos os pares de classes.

---

## 2. Restrições Técnicas Estritas

- **PROIBIDO:** `numpy`, `scipy`, `scikit-learn`, `pandas` — qualquer biblioteca de ML ou álgebra avançada.
- **OBRIGATÓRIO:** Toda a matemática (produto escalar, subtração de vetores, distâncias, médias) DEVE ser feita em **Python puro** com laços `for`, listas nativas e `zip`, no arquivo `iris_classifier/math_utils.py`.
- **Bibliotecas externas permitidas:** apenas `xlrd` (leitura do `.xls`) e `matplotlib` (gráficos).

---

## 3. Estrutura de Arquivos

```
TEIA_TOPICOS_ESPECIAIS_EM_IA/
├── iris_classifier/
│   ├── main.py          # Orquestrador — executa os 3 experimentos
│   ├── data_loader.py   # Leitura do XLS e split estratificado
│   ├── math_utils.py    # Álgebra linear em Python puro (produto escalar, distância, média, discriminante)
│   ├── classifier.py    # Lógica de classificação (treinar, predizer)
│   ├── evaluator.py     # Métricas: acurácia, matriz de confusão, precisão, revocação, F1
│   └── visualizer.py    # Gráficos: dispersão, superfícies de decisão, heatmap da confusão
├── data/
│   └── Iris data.xls    # Base original — NÃO ALTERAR
├── docs/
│   ├── guia_professor.md      # Guia de apresentação ao professor
│   ├── teoria_completa.md     # Teoria completa para estudo
│   ├── formulario.md          # Folha de fórmulas rápidas
│   └── perguntas_prova.md     # Perguntas e respostas para a prova
├── outputs/             # Gráficos gerados (criado automaticamente)
├── CLAUDE.md            # Este arquivo
├── GEMINI.md            # Regras para o agente Gemini
├── README.md            # Visão geral do projeto
└── requirements.txt     # xlrd==2.0.2, matplotlib==3.10.8
```

---

## 4. Padrões de Código

- **Idioma:** Todo código, comentários, docstrings, `print`s e documentação em **Português do Brasil**.
- **Split estratificado:** 70% treino / 30% teste *por classe*, com `random.seed(42)`.
- **Atributos padrão:** índices `[2, 3]` (Comprimento e Largura da Pétala).
- **Nomes das classes:** `'setosa'`, `'versicolor'`, `'virginica'` (minúsculas).
- **Saída de gráficos:** sempre em `outputs/`. Nunca exibir com `plt.show()` — sempre `plt.savefig()`.
- **Não alterar** o arquivo `data/Iris data.xls`.

---

## 5. Como Executar

```bash
python iris_classifier/main.py
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
