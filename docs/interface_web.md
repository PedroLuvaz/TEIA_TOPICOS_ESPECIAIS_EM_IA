# Interface Web — Guia Completo

Interface web dos laboratórios de Reconhecimento de Padrões, construída sobre
os **mesmos modelos em Python puro** da interface desktop (Tkinter). Nenhuma
matemática é reimplementada: o backend apenas expõe, via JSON, o que já existe
em `iris_classifier/`.

---

## 1. Arquitetura

```
web_app/
├── backend/                    # API FastAPI (só orquestra — sem matemática nova)
│   ├── main.py                 # App, CORS e entrega do build em produção
│   ├── core.py                 # Carregamento de dados, split e cache
│   ├── lab5_config.py          # Configurações dos exercícios do Lab 5
│   └── routers/
│       ├── dataset.py          # Metadados, amostras e estatísticas
│       ├── distancia_minima.py # Lab 1
│       ├── perceptron_delta.py # Lab 2
│       ├── metricas.py         # Lab 3
│       ├── bayes.py            # Lab 4
│       └── lab5.py             # Labs 5.0 e 5.1
└── frontend/                   # React 18 + Vite + TypeScript + Tailwind v4
    └── src/
        ├── lib/                # Cliente da API e tipos
        ├── components/         # Design system e visualizações
        └── pages/              # Uma página por laboratório
```

**Princípio central:** o backend importa `models/`, `evaluation/` e `core/` do
`iris_classifier/` exatamente como a GUI desktop faz. Se um número aparece
diferente entre as duas interfaces, é bug — não diferença de implementação.

---

## 2. Como rodar

### 2.0 Jeito rápido — um duplo clique (recomendado para apresentar)

Na raiz do projeto:

- **Windows:** duplo clique em **`Iniciar Projeto.bat`**
- **macOS / Linux:** `./iniciar.sh`

O script sobe o servidor, espera ele responder e abre o navegador já na
interface. Se o frontend ainda não estiver compilado, ele compila sozinho na
primeira execução (aí precisa do Node instalado); nas vezes seguintes só o
Python roda, e a abertura é quase instantânea.

Opções úteis:

```bash
python iniciar.py --dev          # hot reload do Vite, para desenvolver
python iniciar.py --rebuild      # recompila a interface antes de subir
python iniciar.py --porta 9000   # troca a porta
python iniciar.py --sem-browser  # não abre o navegador
```

Se a porta 8000 estiver ocupada, ele procura a próxima livre e avisa qual
usou. Para encerrar, `Ctrl+C` na janela — os dois processos são finalizados
juntos.

> Para a apresentação, deixe a interface já compilada
> (`npm --prefix web_app/frontend run build`). Assim o duplo clique não
> depende do Node e abre em segundos.

### 2.1 Pré-requisitos

- **Python 3.10+** com o venv do projeto já criado
- **Node.js 20+** (testado no 24) e npm

### 2.2 Instalação (só na primeira vez)

Dependências Python (a partir da raiz do projeto):

```bash
pip install -r requirements.txt
```

Dependências do frontend:

```bash
npm --prefix web_app/frontend install
```

### 2.3 Modo desenvolvimento (dois terminais)

**Terminal 1 — backend** (a partir da raiz do projeto):

```bash
python -m uvicorn web_app.backend.main:app --reload --port 8000
```

**Terminal 2 — frontend:**

```bash
npm --prefix web_app/frontend run dev
```

Acesse **http://localhost:5173**. O Vite encaminha `/api` para a porta 8000
automaticamente, então não é preciso configurar nada.

### 2.4 Modo produção (um único servidor)

Gere o build do frontend e suba apenas o backend — ele passa a servir a
interface na mesma porta:

```bash
npm --prefix web_app/frontend run build
python -m uvicorn web_app.backend.main:app --port 8000
```

Acesse **http://localhost:8000**.

### 2.5 Comandos úteis

| Comando | O que faz |
|---|---|
| `npm --prefix web_app/frontend run dev` | Servidor de desenvolvimento com hot reload |
| `npm --prefix web_app/frontend run build` | Build de produção em `web_app/frontend/dist/` |
| `npm --prefix web_app/frontend run preview` | Serve o build localmente para conferência |
| `npm --prefix web_app/frontend run typecheck` | Checagem de tipos sem gerar arquivos |

### 2.6 Documentação interativa da API

Com o backend no ar, o FastAPI publica a documentação automática em:

- **http://localhost:8000/docs** — Swagger UI (permite testar cada rota)
- **http://localhost:8000/redoc** — ReDoc

---

## 3. O que cada página faz

### Classificar (aplicação)
A tela principal da entrega: escolha do **modelo** entre os sete do catálogo e
**parametrização** de cada um por controles gerados a partir do esquema que o
backend publica. Traz métricas completas, matriz de confusão, regiões de
decisão, predição de amostra digitada e painéis próprios de cada modelo (OOB e
importâncias na floresta, curva de erro na rede, convergência por classe nos
lineares). Detalhes em [`classificar_modelos.md`](classificar_modelos.md).

### Distância Mínima (Lab 1)
Protótipos (vetores médios), função discriminante linear e as três fronteiras
de decisão. Regiões de decisão renderizadas como mapa de calor suavizado.
**Clique no gráfico** para classificar um ponto arbitrário e ver os scores
discriminantes e as distâncias euclidianas a cada protótipo.

### Perceptron & Delta (Lab 2)
Quatro experimentos em sub-abas:
- **Perceptron** — ativação por limiar, erros por época, convergência garantida
  apenas em dados linearmente separáveis
- **Regra Delta** — saída linear, curva de MSE
- **Delta OvA** — multiclasse por Um-Contra-Todos, uma curva por classificador
- **XOR** — demonstra o limite teórico: o MSE estaciona em 0,25 e nunca zera

### Bases de dados

O seletor **Base de dados**, presente em todas as abas, oferece três opções:

| id | Base | Classes | Atributos |
|---|---|---|---|
| `v1` | Iris Original | 3 | pétalas · sépalas · todas |
| `v2` | Iris Separável | 3 | pétalas · sépalas · todas |
| `fds` | Fim de Semana (seminário) | 4 | Clima×Pais · Clima×Dinheiro · Pais×Dinheiro · todos |
| `usr_*` | Bases importadas pelo usuário (.txt) | livre | pares gerados a partir das colunas |

O botão **Importar .txt**, no mesmo painel, carrega a base do usuário: o
delimitador, o cabeçalho e a coluna de classe são detectados e podem ser
ajustados, e a base importada passa a valer em todas as telas. Guia completo em
[`importar_dados_txt.md`](importar_dados_txt.md).

Nada na interface assume as 3 classes do Iris: classes, features, pares de
classes e cores vêm do dataset selecionado (`/api/dataset/metadata`). A base do
seminário é categórica — ver
[`seminario_dataset_fim_de_semana.md`](seminario_dataset_fim_de_semana.md).

### Métricas Avançadas (Lab 3)
- **Validação cruzada** — k-fold estratificado com repetições, média ± desvio e
  IC 95%; é a resposta ao "por que tudo dá quase 100%" do split único
- **Split único** — todos os classificadores no mesmo split, com teste Z de
  Kappa entre cada par
- **Significância** — testes pareados de significância para o MCC e as demais
  métricas (McNemar, bootstrap pareado e permutação). Ver
  [`lab_03/testes_significancia.md`](lab_03/testes_significancia.md)
  - *Testar um par* — escolha A, B e a métrica; veredito dos 3 testes, tabela de
    McNemar, histograma do bootstrap com o IC 95% e contraste com o teste Z
  - *Todos os pares* — os 10 pares de uma vez, com os selos M / B / P
- **Matriz editável** — edite as células e veja Kappa, Tau e todas as métricas
  reagirem em tempo real
- **Ag × Kappa × Tau** — curva mostrando por que o acerto global superestima a
  qualidade do classificador

### Bayes & Normalidade (Lab 4)
Bayes Ótimo (QDA) e Naive Bayes lado a lado, com as matrizes de covariância
estimadas, fronteiras **quadráticas** traçadas por marching squares (curvas
suaves, sem serrilhado) e os testes de normalidade multivariada de
Henze-Zirkler e Mardia por classe.

### Lab 5.0 · XOR (MLP)
- Memória de cálculo do **exemplo didático (slide 37)** — inclui a convenção de
  bias único compartilhado por camada, conferida contra os slides 38-42
- Memória de cálculo do **exercício XOR (slide 36)** — 1 época, 4 padrões
- **Treino interativo** com superfície de saída da rede e curva de convergência

### Lab 5.1 · Feedforward
- **Item (i)** — memória de cálculo da rede 2-2-2 "galinha vs homem"
- **Exercício extra (slide 34)** — rede da Fig. 12.32, 1 iteração
- **Linha do tempo** — treine além da iteração única que o enunciado pede e
  acompanhe a rede convergindo
- **Bônus interativo** — canvas 8×8 pintável, classificado ao vivo por uma rede
  64-10-1, com as ativações da camada oculta visíveis
- **Item (ii)** — comparativo MLP (scikit-learn) × Bayes Ótimo × Naive Bayes no
  Iris, com todas as métricas e testes Z

### Seminário · Florestas Aleatórias
Implementação própria em Python puro (árvore CART, bagging, subespaço
aleatório, OOB, importâncias). Três sub-abas: **A floresta** (métricas,
regiões em escada, importâncias, votação ao clicar), **Árvores individuais**
(diagrama navegável de cada árvore) e **Comparativo** (validação cruzada
contra árvore única, Bayes e Distância Mínima). Teoria completa em
[`seminario_florestas_aleatorias.md`](seminario_florestas_aleatorias.md).

### Construtor de Rede
Monte a MLP do zero: adicione ou remova neurônios de cada camada, edite cada
peso e bias na mão, defina os padrões de treino e veja o diagrama reagir em
tempo real. A coluna **saída agora** é calculada no próprio navegador enquanto
você edita, antes mesmo de treinar. Ao treinar, o resultado aparece na mesma
linha do tempo dos exercícios da aula — e é possível adotar os pesos de
qualquer época como novo ponto de partida.

---

## 3.1 Linha do tempo do treinamento

Presente no Lab 5.0, no Lab 5.1 e no Construtor. O backend treina todas as
épocas de uma vez e devolve o histórico completo de erro mais **snapshots dos
pesos** espaçados logaritmicamente — resolução densa nas primeiras épocas, onde
o aprendizado muda rápido, e esparsa no fim.

Arrastar o slider (ou usar o play, com velocidade 1×/2×/4×) percorre esses
snapshots. Como os pesos vêm junto, a superfície de decisão e o diagrama da
rede são **recalculados no navegador** a cada quadro: o `forward` da MLP foi
replicado em TypeScript (`src/lib/utils.ts`), então não há uma chamada de rede
por quadro. O treino em si continua exclusivamente no Python puro do backend.

---

## 4. Rotas da API

Todas as rotas ficam sob `/api`. Os parâmetros comuns são `dataset`
(`v1`/`v2`/`fds` ou o id de uma base importada), `atributos` (as chaves que o
dataset declara em `/api/dataset/metadata`) e `proporcao` (fração de treino).

| Método | Rota | Descrição |
|---|---|---|
| GET | `/api/health` | Disponibilidade e se o scikit-learn está instalado |
| GET | `/api/dataset/metadata` | Opções disponíveis na interface |
| GET | `/api/dataset/amostras` | Amostras projetadas em 2D, marcadas treino/teste |
| GET | `/api/dataset/estatisticas` | Média, desvio, mínimo e máximo por classe |
| GET | `/api/dataset/opcoes-leitura` | Delimitadores aceitos e limites da importação |
| POST | `/api/dataset/analisar` | Pré-visualiza um .txt sem importá-lo |
| POST | `/api/dataset/importar` | Importa a base do usuário |
| GET | `/api/dataset/enviados` | Lista as bases importadas |
| PATCH | `/api/dataset/enviados/{id}` | Renomeia uma base importada |
| DELETE | `/api/dataset/enviados/{id}` | Remove uma base importada |
| GET | `/api/classificar/modelos` | Catálogo de modelos + esquema dos parâmetros |
| POST | `/api/classificar/treinar` | Treina o modelo escolhido e avalia |
| POST | `/api/classificar/regioes` | Regiões de decisão de qualquer modelo |
| POST | `/api/classificar/predizer` | Classifica uma amostra informada |
| GET | `/api/distancia-minima/treinar` | Protótipos, métricas e fronteiras |
| GET | `/api/distancia-minima/regioes` | Grade de regiões de decisão |
| POST | `/api/distancia-minima/predizer` | Classifica um vetor arbitrário |
| GET | `/api/perceptron-delta/binario` | Perceptron ou Delta para um par de classes |
| GET | `/api/perceptron-delta/ova` | Delta multiclasse Um-Contra-Todos |
| GET | `/api/perceptron-delta/xor` | XOR com Regra Delta (limite linear) |
| GET | `/api/metricas/comparar-modelos` | Todos os classificadores + testes Z |
| POST | `/api/metricas/avaliar` | Métricas de uma matriz de confusão informada |
| POST | `/api/metricas/comparar-matrizes` | Teste Z entre duas matrizes |
| GET | `/api/metricas/curva-kappa` | Curva Ag × Kappa × Tau |
| GET | `/api/bayes/treinar` | Bayes e Naive lado a lado, com teste Z |
| GET | `/api/bayes/regioes` | Regiões + superfícies para as fronteiras exatas |
| GET | `/api/bayes/normalidade` | Henze-Zirkler e Mardia por classe |
| POST | `/api/bayes/predizer` | Classifica um vetor arbitrário |
| GET | `/api/metricas/validacao-cruzada` | k-fold dos classificadores (parâmetro `modelos`) |
| GET | `/api/metricas/classificadores` | Classificadores e métricas testáveis |
| GET | `/api/metricas/significancia` | McNemar + bootstrap + permutação de um par |
| GET | `/api/metricas/significancia/matriz` | Todos os pares de uma vez |
| GET | `/api/metricas/significancia/memoria` | Memória de cálculo dos 3 testes |
| GET | `/api/floresta/treinar` | Treina a floresta: métricas, OOB, importâncias |
| GET | `/api/floresta/arvore/{i}` | Estrutura completa de uma árvore |
| GET | `/api/floresta/regioes` | Regiões de decisão + confiança do voto |
| POST | `/api/floresta/predizer` | Votação árvore a árvore de um ponto |
| GET | `/api/floresta/validacao-cruzada` | Floresta × árvore única × demais |
| GET | `/api/floresta/memoria` | Memória de cálculo do seminário |
| GET | `/api/lab5/exercicios` | Lista os exercícios do Lab 5 |
| GET | `/api/lab5/memoria/{id}` | Traço completo da memória de cálculo |
| POST | `/api/lab5/trajetoria/{id}` | Histórico de erro + snapshots dos pesos |
| POST | `/api/lab5/rede/trajetoria` | Treina uma rede montada pelo usuário |
| POST | `/api/lab5/rede/memoria` | Memória de cálculo de uma rede customizada |
| GET | `/api/lab5/xor/inicial` | Estado da rede XOR na época 0 |
| POST | `/api/lab5/xor/treinar` | Treina N épocas a partir de pesos informados |
| GET | `/api/lab5/imagem/padroes` | Padrões 8×8 de referência |
| POST | `/api/lab5/imagem/prever` | Classifica um desenho 8×8 |
| GET | `/api/lab5/iris/comparar` | Item (ii): MLP × Bayes × Naive |

Os identificadores de exercício aceitos em `/api/lab5/memoria/{id}` são:
`didatico`, `xor`, `galinha-homem` e `fig-1232`.

---

## 5. Decisões de implementação

**Fronteiras suaves.** As regiões de decisão são calculadas numa grade no
backend e pintadas numa canvas na resolução nativa, depois ampliadas com
interpolação. Para o Bayes, o backend também envia a superfície de diferença de
scores por par de classes, e o frontend traça o nível zero com *marching
squares* — o que produz curvas realmente lisas em vez de escadinhas.

**Cache do split.** `core.py` memoiza a leitura do `.xls` e o split
estratificado, de modo que trocar de aba não recarrega o arquivo. A semente
continua sendo 42, igual à GUI desktop — os resultados são idênticos.

**Padrões de pixels compartilhados.** Os padrões 8×8 do Lab 5.1 foram movidos
para `iris_classifier/models/padroes_pixels.py`, um módulo sem dependência de
interface, consumido tanto pela GUI Tkinter quanto pela API.

**Roteamento por hash.** A navegação usa `#/lab-5-0` em vez de `react-router`.
Isso mantém as URLs compartilháveis, elimina uma dependência com CVEs abertos e
reduz o bundle.

**Carregamento sob demanda.** Cada laboratório é um chunk separado. O
carregamento inicial fica em torno de 60 kB comprimidos; recharts e KaTeX só
são baixados quando a página que os usa abre.

---

## 6. Solução de problemas

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| Tela mostra "Não foi possível carregar" | Backend fora do ar | Suba o uvicorn na porta 8000 |
| `Dataset 'v2' não encontrado` | O arquivo `data/iris_data_02.xlsx` não está no repositório | Use a base `Iris Original`, ou gere o v2 localmente |
| Item (ii) do Lab 5.1 devolve 503 | scikit-learn ausente | `pip install scikit-learn` |
| Porta 5173 ocupada | Outro Vite rodando | Encerre o processo anterior ou rode com `--port` |
| Mudanças no backend não aparecem | Uvicorn sem `--reload` | Reinicie com a flag `--reload` |

---

*Tópicos Especiais em Inteligência Artificial · UEPB 2026*
*Grupo: Erick Nathan · Laura Barbosa · Pedro Lucas*
