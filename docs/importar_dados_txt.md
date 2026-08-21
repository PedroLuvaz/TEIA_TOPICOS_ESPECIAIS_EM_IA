# Importar a base de dados do usuário (.txt)

> Requisito atendido: *"O aplicativo deverá ser alimentado pela base de dados do
> usuário, no formato txt."*

O aplicativo não está preso ao Iris. Qualquer arquivo de texto com uma amostra
por linha pode ser importado pela interface e passa a funcionar em **todas** as
telas: classificação, métricas, validação cruzada, testes de significância e o
modelo do seminário.

---

## 1. Onde fica

O botão **Importar .txt** está no canto do painel *Configuração do
experimento* — o mesmo painel que aparece em todas as páginas. Ou seja: a base
pode ser trocada de dentro de qualquer laboratório, sem sair da tela.

Fluxo em três passos:

1. **Escolher arquivo** — o conteúdo é lido no navegador e enviado para o
   backend local, que devolve uma pré-visualização.
2. **Conferir a leitura** — delimitador, cabeçalho, coluna de classe e quais
   colunas ignorar. A prévia mostra as primeiras linhas já separadas em colunas.
3. **Importar** — a base é gravada em `data/enviados/` e selecionada
   automaticamente.

---

## 2. Formato aceito

Uma amostra por linha; colunas separadas por um delimitador constante:

```text
5.1,3.5,1.4,0.2,setosa
4.9,3.0,1.4,0.2,setosa
7.0,3.2,4.7,1.4,versicolor
```

| Item | O que é aceito |
|---|---|
| Extensão | `.txt`, `.csv`, `.tsv`, `.data` (o conteúdo é que importa) |
| Delimitador | vírgula, ponto e vírgula, tabulação, barra vertical ou espaços |
| Cabeçalho | opcional — detectado automaticamente |
| Decimal | ponto (`3.5`) ou vírgula (`3,5`, quando o delimitador não é vírgula) |
| Atributos | numéricos **ou** categóricos (texto) — podem ser misturados |
| Classe | qualquer coluna, em qualquer posição |
| Comentários | linhas começando com `#`, `%` ou `//` são ignoradas |
| Ausentes | `?`, `NA`, `N/A`, `null`, `-`, `.` ou vazio → a linha é descartada |

**Dois exemplos prontos** acompanham o projeto:

- `data/exemplos/iris.txt` — 150 linhas, vírgula, sem cabeçalho, atributos
  numéricos;
- `data/exemplos/fim_de_semana.txt` — 300 linhas, ponto e vírgula, com
  cabeçalho, atributos **categóricos** (`Sol`, `Chuva`, `Rico`…).

---

## 3. O que o leitor faz sozinho

Tudo em Python puro, em `iris_classifier/data/leitor_texto.py` — sem `pandas`,
sem `numpy`, sem o módulo `csv`.

**Delimitador.** Cada candidato é testado nas primeiras 50 linhas. Vence o que
divide o arquivo em colunas de forma mais consistente (mais linhas concordando
com o mesmo número de campos); no empate, vence o que produz mais colunas.

**Cabeçalho.** Duas evidências, nesta ordem:

1. a primeira linha é toda textual e o corpo do arquivo tem números — o tipo da
   primeira linha destoa do resto;
2. em bases inteiramente categóricas (onde a evidência acima não serve), o
   sinal é que os textos da primeira linha **não reaparecem na própria
   coluna** — `clima` nunca é um valor da coluna clima, mas `Sol` é.

**Coluna de classe.** Procura, nesta ordem: um nome típico de rótulo
(`classe`, `class`, `label`, `target`, `decisao`, `species`, `y`…); a última
coluna, se for categórica com poucos valores distintos; a primeira coluna
categórica curta; a última coluna, por convenção.

**Atributos categóricos.** Viram códigos inteiros `0..k-1` na ordem alfabética
dos rótulos, e a tabela de rótulos é guardada: a interface continua mostrando
`Sol` em vez de `0`. É o mesmo tratamento que o dataset categórico do seminário
já recebia.

**Combinações de atributos.** O menu *Atributos* é gerado a partir das colunas:
todos os pares entre as quatro primeiras features, mais a entrada "todos". Os
gráficos 2D usam o par escolhido; os modelos usam o conjunto inteiro quando a
opção "todos" está ativa.

---

## 4. Limites e validações

| Limite | Valor | Motivo |
|---|---|---|
| Linhas | 20 000 | Tudo é treinado em Python puro a cada requisição |
| Colunas | 60 | Idem |
| Classes | 20 | Acima disso, quase certamente a coluna escolhida não é o rótulo |
| Tamanho do arquivo | ~8 MB | O conteúdo viaja como texto dentro do JSON |
| Bases guardadas | 20 | `data/enviados/` é área de trabalho, não repositório |

A importação é recusada, com mensagem explicando o motivo, quando:

- sobra menos de **dois atributos** (os gráficos e as superfícies de decisão são
  desenhados em 2D);
- a coluna de classe escolhida tem **um único valor**;
- a coluna de classe tem valores distintos demais (provavelmente é um `id`);
- uma coluna de texto tem valores distintos demais para ser um atributo
  categórico — marque-a como ignorada;
- nenhuma linha válida foi lida (delimitador ou cabeçalho errados).

Avisos que **não** impedem a importação, mas aparecem na interface: linhas
descartadas por tamanho diferente, por valores ausentes ou não convertíveis, e
classes com menos de 4 amostras (o split estratificado fica frágil).

---

## 5. Onde a base fica guardada

```text
data/enviados/
├── usr_ab12cd34.txt     # o arquivo original, como enviado
└── usr_ab12cd34.json    # nome exibido + configuração de leitura resolvida
```

Gravar em disco tem um motivo prático: a base sobrevive ao reinício do servidor
(e ao `--reload` do uvicorn). Ninguém precisa reenviar o arquivo no meio da
defesa. A pasta está no `.gitignore` — dados do usuário não entram no
repositório.

O `.json` guarda a configuração **já resolvida** (`virgula`, `nao`, coluna 4) e
não `auto`: assim, releituras futuras produzem exatamente a mesma base, mesmo
que a heurística de detecção mude.

---

## 6. Rotas da API

| Método | Rota | Descrição |
|---|---|---|
| GET | `/api/dataset/opcoes-leitura` | Delimitadores aceitos e limites |
| POST | `/api/dataset/analisar` | Pré-visualiza o arquivo sem importar |
| POST | `/api/dataset/importar` | Importa e registra a base |
| GET | `/api/dataset/enviados` | Lista as bases importadas |
| PATCH | `/api/dataset/enviados/{id}` | Renomeia |
| DELETE | `/api/dataset/enviados/{id}` | Remove a base e limpa os caches |

O conteúdo do arquivo trafega como string dentro do JSON — o projeto não
depende de `python-multipart`. Exemplo com `curl`:

```bash
curl -X POST http://localhost:8000/api/dataset/importar -H "Content-Type: application/json" -d "{\"nome\":\"Iris TXT\",\"conteudo\":\"5.1,3.5,1.4,0.2,setosa\n4.9,3.0,1.4,0.2,setosa\n...\"}"
```

---

## 7. Problemas comuns

**"Não foi possível identificar as colunas do arquivo."**
O arquivo tem uma coluna só, ou usa um separador fora da lista. Confira se não
é largura fixa (colunas alinhadas por espaços em quantidade variável funcionam;
alinhadas *sem* separador, não).

**A prévia mostra tudo numa coluna só.**
Escolha o delimitador manualmente no seletor — a detecção automática pode se
confundir quando o texto contém o próprio separador.

**A primeira linha virou amostra (ou o cabeçalho virou classe).**
Force com *Primeira linha → É cabeçalho / Já são dados*.

**Uma coluna de `id` está entrando como atributo.**
Desmarque-a na lista de colunas antes de importar. Ids são numéricos e
correlacionam com nada — só atrapalham o classificador.

**Acurácia estranhamente perfeita.**
Verifique se alguma coluna é uma codificação da própria classe (como as colunas
`*_cod` do dataset do seminário). Desmarque-a.
