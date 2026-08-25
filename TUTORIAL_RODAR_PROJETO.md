# Como rodar o projeto — passo a passo

**Projeto:** Reconhecimento de Padrões — Tópicos Especiais em IA (UEPB)
**Equipe:** Erick Nathan · Laura Barbosa · Pedro Lucas

Este tutorial é para quem está abrindo o projeto pela primeira vez. São
**dois cliques**: um para instalar, outro para abrir. Nada precisa ser digitado
no terminal.

---

## Resumo em uma linha

> Duplo clique em **`dependencias.bat`** e espere terminar.
> Depois, duplo clique em **`Iniciar Projeto.bat`**. O navegador abre sozinho.

---

## O que precisa estar instalado no computador

| Programa | Precisa? | Observação |
|---|---|---|
| **Python 3.10 ou mais novo** | **Sim** | Se não tiver, o `dependencias.bat` se oferece para instalar |
| Node.js | Não | A interface já vem compilada dentro do projeto |
| R | Não | Só para o teste extra de normalidade; sem ele o app usa valores já calculados |

Também é preciso **conexão com a internet** na primeira execução, porque o
instalador baixa as bibliotecas.

---

## Passo 1 — Instalar (só uma vez)

1. Abra a pasta do projeto.
2. Dê um **duplo clique** no arquivo **`dependencias.bat`**.
3. Vai abrir uma janela preta. Aperte qualquer tecla quando ele pedir.
4. Espere. Na primeira vez leva de 2 a 5 minutos: ele está baixando as
   bibliotecas.
5. No fim aparece a mensagem:

   ```text
   ===================================================================
     PRONTO! As dependencias estao instaladas.
   ===================================================================
   ```

6. Ele pergunta se quer abrir o projeto agora. Digite **S** para abrir na hora,
   ou **N** para abrir depois.

### Se aparecer "O Python nao foi encontrado"

O próprio instalador resolve, de duas formas:

- **Automática:** ele pergunta *"Instalar o Python agora?"*. Digite **S**,
  aceite as janelas que aparecerem, **feche a janela preta** e dê um duplo
  clique no `dependencias.bat` outra vez.
- **Manual:** baixe em <https://www.python.org/downloads/>, clique no botão
  amarelo *Download Python* e execute o arquivo baixado.
  ⚠️ **Marque a caixinha "Add python.exe to PATH"** antes de clicar em
  *Install Now* — sem ela o Windows não encontra o Python depois.
  Terminada a instalação, feche a janela preta e rode o `dependencias.bat` de novo.

### Se o Windows mostrar um aviso azul ("O Windows protegeu o computador")

É o SmartScreen avisando que o arquivo veio da internet. Clique em
**"Mais informações"** e depois em **"Executar assim mesmo"**. O arquivo é um
script de texto — dá para abrir no Bloco de Notas e ler tudo o que ele faz.

---

## Passo 2 — Abrir o projeto

1. Duplo clique em **`Iniciar Projeto.bat`**.
2. A janela preta mostra o progresso e, em poucos segundos, o **navegador abre
   sozinho** em `http://127.0.0.1:8000`.
3. Pronto: a aplicação está rodando.

> **Não feche a janela preta** enquanto estiver usando o projeto — é ela que
> mantém o programa no ar. Para encerrar, feche o navegador e aperte
> **Ctrl + C** na janela preta (ou simplesmente feche a janela).

---

## Passo 3 — O que fazer dentro da aplicação

A barra da esquerda tem as telas. Sugestão de percurso:

### 1. Classificar — a tela principal

É onde o modelo é escolhido e parametrizado.

- **Base de dados:** Iris Original, a variante separável, a base do seminário
  ou uma base sua importada.
- **Modelo:** sete opções — Distância Mínima, Perceptron OvA, Regra Delta OvA,
  Bayes Ótimo (QDA), Naive Bayes, Rede Feedforward (MLP) e Floresta Aleatória.
- **Parametrização:** os controles mudam conforme o modelo. Na floresta, por
  exemplo, dá para mexer no número de árvores, no critério de divisão e na
  profundidade máxima.
- Abaixo aparecem as métricas, a matriz de confusão e as regiões de decisão.
  **Clicando no gráfico**, o ponto clicado é classificado na hora.

> Dica: escolha **Sépalas** em *Atributos*. Nas pétalas quase todo modelo acerta
> 100% e as diferenças somem; nas sépalas as classes se sobrepõem e dá para ver
> cada modelo se comportando de um jeito.

### 2. Importar uma base sua (.txt)

1. No painel *Configuração do experimento*, clique em **Importar .txt**.
2. Clique em **Escolher arquivo** e selecione um `.txt`.
   Para testar, há dois prontos na pasta do projeto:
   - `data/exemplos/iris.txt`
   - `data/exemplos/fim_de_semana.txt`
3. A tela mostra como entendeu o arquivo: separador, cabeçalho, qual coluna é a
   classe e uma prévia das primeiras linhas. Tudo pode ser corrigido ali mesmo.
4. Clique em **Importar esta base**. A partir daí ela vale em todas as telas.

O formato aceito é simples: uma amostra por linha, colunas separadas por
vírgula, ponto e vírgula, tabulação ou espaços, e uma das colunas com o nome da
classe. Detalhes em [`docs/importar_dados_txt.md`](docs/importar_dados_txt.md).

### 3. Métricas Avançadas — comparação e significância

Quatro sub-abas: validação cruzada, split único, testes de significância
(McNemar, bootstrap pareado e permutação) e a matriz editável. É aqui que os
sete modelos são comparados entre si.

### 4. Florestas Aleatórias — o seminário

O modelo apresentado na defesa, com as árvores navegáveis uma a uma, o erro
out-of-bag e a importância de cada atributo.

---

## Se alguma coisa der errado

| O que aconteceu | O que fazer |
|---|---|
| A janela preta abre e fecha na hora | Rode o `dependencias.bat` primeiro; se já rodou, clique com o botão direito no arquivo → *Executar como administrador* |
| "Dependências Python ausentes" | Rode o `dependencias.bat` |
| "O Python nao foi encontrado" | Ver o [Passo 1](#se-aparecer-o-python-nao-foi-encontrado) |
| O navegador não abriu sozinho | Abra o navegador e digite `http://127.0.0.1:8000` |
| "Porta 8000 ocupada" | Não é problema: o programa procura a próxima porta livre e mostra o endereço certo na janela preta |
| A página abre em branco | Espere 5 segundos e atualize com **F5** |
| Antivírus bloqueou | Os arquivos `.bat` são texto puro; abra no Bloco de Notas para conferir e libere a pasta do projeto |
| Nada funciona | Apague a pasta `venv` e rode o `dependencias.bat` de novo |

---

## Para quem estiver no macOS ou Linux

Mesma coisa, pelo terminal, dentro da pasta do projeto:

```bash
./dependencias.sh
```

```bash
./iniciar.sh
```

Se aparecer erro de permissão, rode antes `chmod +x dependencias.sh iniciar.sh`.

---

## Outros jeitos de rodar (opcional)

**Interface do computador (Tkinter)** — não usa navegador:

```bash
venv\Scripts\python.exe iris_classifier\run_gui.py
```

**Todos os experimentos no terminal** — imprime protótipos, matrizes, métricas e
salva os gráficos em `outputs/`:

```bash
venv\Scripts\python.exe iris_classifier\main.py
```

**Modo desenvolvedor**, com recompilação automática do frontend (precisa do
Node.js instalado):

```bash
venv\Scripts\python.exe iniciar.py --dev
```

---

## Notas para a equipe

- A pasta `web_app/frontend/dist` (interface compilada) **é versionada de
  propósito**: é o que permite rodar o projeto só com o Python, sem instalar o
  Node.js. Depois de mexer no frontend, recompile e comite junto:

  ```bash
  npm --prefix web_app/frontend run build
  ```

- A pasta `venv` **não** é versionada — cada máquina cria a sua ao rodar o
  instalador.
- Documentação completa do projeto: [`docs/defesa_projeto.md`](docs/defesa_projeto.md).
