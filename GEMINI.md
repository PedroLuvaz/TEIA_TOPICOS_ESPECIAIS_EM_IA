# Contexto e Regras do Projeto

Este arquivo serve como instrução fundamental para o agente Gemini CLI ao atuar neste projeto. Sempre obedeça às diretrizes abaixo:

## 1. Objetivo do Projeto
- Este é um projeto acadêmico de Tópicos Especiais em IA.
- O objetivo é implementar um **Classificador de Distância Mínima** para a base de dados Iris.
- O código realiza três experimentos: 
  1) Cálculo de protótipos e classificação de 3 classes.
  2) Função discriminante e regra de decisão por máximo.
  3) Superfícies de decisão para pares de classes (classificadores binários).

## 2. Restrições Técnicas Estritas
- **NENHUMA biblioteca de Machine Learning ou Álgebra Avançada é permitida.** NÃO use `numpy`, `scipy`, `scikit-learn` ou `pandas`.
- Toda a matemática (álgebra linear, cálculo de médias, distâncias, produto escalar) DEVE ser feita em Python puro (laços `for`, listas nativas, `zip`), localizados em `iris_classifier/math_utils.py`.
- As únicas bibliotecas externas permitidas são `xlrd` (para leitura da base XLS) e `matplotlib` (para geração de gráficos no `visualizer.py`).

## 3. Padrões de Código e Estrutura
- **Idioma:** Todo o código, comentários, docstrings, saídas no terminal (prints) e documentação devem ser escritos obrigatoriamente em **Português do Brasil**.
- **Split Estratificado:** A divisão de treino e teste DEVE ser estratificada (garantindo 70% de treino e 30% de teste *por classe*), usando `random.seed(42)`.
- **Organização de Arquivos:** Mantenha a separação de responsabilidades (leitura de dados, matemática, classificador, métricas, visualização e o orquestrador `main.py`).
- Não altere o `Iris data.xls` original localizado na pasta `data/`.
- Salve sempre os gráficos gerados na pasta `outputs/`.
