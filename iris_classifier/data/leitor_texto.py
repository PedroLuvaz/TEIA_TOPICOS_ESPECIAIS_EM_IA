"""
Leitor generico de bases de dados em texto (.txt / .csv / .tsv / .data).

Este modulo e a porta de entrada para a base de dados DO USUARIO — o
requisito "o aplicativo devera ser alimentado pela base de dados do usuario,
no formato txt". Ele nao sabe nada sobre Iris nem sobre o dataset do
seminario: recebe texto puro, descobre o delimitador, decide se ha cabecalho,
separa os atributos da coluna de classe e devolve a MESMA estrutura usada em
todo o projeto:

    {'atributos': [float, ...], 'classe': 'nome_da_classe'}

Regras do projeto respeitadas
-----------------------------
- Python puro: nenhum `pandas`, `numpy` ou modulo `csv`. Apenas lacos, listas
  e `str.split`.
- Atributos categoricos (texto) sao codificados em inteiros 0..k-1, na ordem
  alfabetica dos rotulos, e a tabela de rotulos volta em `valores` para que a
  interface mostre 'Sol' em vez de 0 — o mesmo tratamento que o dataset
  categorico do seminario ja recebia.

Uso tipico
----------
    analise = analisar(texto)              # o que o arquivo parece ser
    base    = carregar(texto, coluna_classe=analise['coluna_classe_sugerida'])
"""

# Delimitadores oferecidos na interface. `None` = qualquer espaco em branco
# (formato classico dos arquivos .data do repositorio UCI alinhados a coluna).
DELIMITADORES = {
    'virgula': ',',
    'ponto_virgula': ';',
    'tabulacao': '\t',
    'pipe': '|',
    'espaco': None,
}

ROTULOS_DELIMITADOR = {
    'virgula': 'Vírgula  ,',
    'ponto_virgula': 'Ponto e vírgula  ;',
    'tabulacao': 'Tabulação',
    'pipe': 'Barra vertical  |',
    'espaco': 'Espaços em branco',
}

# Guarda-corpos: o app e didatico, roda tudo em Python puro e refaz o
# treinamento a cada requisicao. Bases gigantes travariam a interface.
MAX_LINHAS = 20000
MAX_COLUNAS = 60
MAX_CLASSES = 20

# Marcadores comuns de valor ausente.
AUSENTES = {'', '?', 'na', 'n/a', 'nan', 'null', '-', '.'}


class ErroLeitura(ValueError):
    """Erro de formato do arquivo enviado, com mensagem para o usuario."""


# ---------------------------------------------------------------------------
# Utilitarios de baixo nivel
# ---------------------------------------------------------------------------
def _linhas_uteis(texto):
    """Linhas nao vazias, sem comentarios (#, %, //) e sem BOM."""
    texto = texto.lstrip('﻿')
    uteis = []
    for bruta in texto.replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        linha = bruta.strip()
        if not linha:
            continue
        if linha.startswith('#') or linha.startswith('%') or linha.startswith('//'):
            continue
        uteis.append(linha)
    return uteis


def _dividir(linha, delimitador):
    """Quebra a linha nos campos, ja sem espacos e aspas nas bordas."""
    if delimitador is None:
        campos = linha.split()
    else:
        campos = linha.split(delimitador)
    return [c.strip().strip('"').strip("'").strip() for c in campos]


def _eh_numero(texto):
    """True se o campo puder virar float — aceita virgula decimal."""
    try:
        _para_numero(texto)
        return True
    except (ValueError, AttributeError):
        return False


def _para_numero(texto):
    """Converte para float aceitando virgula decimal ('3,5' -> 3.5)."""
    if texto.count(',') == 1 and texto.count('.') == 0:
        texto = texto.replace(',', '.')
    return float(texto)


def _ausente(texto):
    return texto.strip().lower() in AUSENTES


# ---------------------------------------------------------------------------
# Deteccao automatica
# ---------------------------------------------------------------------------
def detectar_delimitador(linhas):
    """
    Escolhe o delimitador que parte o arquivo em colunas de forma consistente.

    Criterio: entre os candidatos que produzem mais de uma coluna, vence o que
    mantem o MESMO numero de campos no maior numero de linhas; havendo empate,
    vence o que produz mais colunas. E o raciocinio do `Sniffer` do modulo
    `csv`, escrito a mao para nao depender de biblioteca.
    """
    amostra = linhas[:50]
    melhor, melhor_nota = None, (-1, -1)

    for chave, delim in DELIMITADORES.items():
        contagens = [len(_dividir(linha, delim)) for linha in amostra]
        contagens = [c for c in contagens if c > 1]
        if not contagens:
            continue
        # Moda das contagens: quantas linhas concordam com o formato dominante.
        dominante = max(set(contagens), key=contagens.count)
        concordam = contagens.count(dominante)
        nota = (concordam, dominante)
        if nota > melhor_nota:
            melhor, melhor_nota = chave, nota

    if melhor is None:
        raise ErroLeitura(
            'Não foi possível identificar as colunas do arquivo. Verifique se '
            'os valores estão separados por vírgula, ponto e vírgula, '
            'tabulação, barra vertical ou espaços.')
    return melhor


def detectar_cabecalho(matriz):
    """
    Decide se a primeira linha e cabecalho.

    Duas evidencias, nesta ordem:

    1. A primeira linha e toda textual e o corpo tem algum numero — o tipo da
       primeira linha destoa do tipo do arquivo (caso do Iris com cabecalho).
    2. Bases inteiramente categoricas (o dataset do seminario, por exemplo)
       nao acionam a evidencia acima: ali o sinal e que os textos da primeira
       linha NAO reaparecem na propria coluna. 'clima' nunca e um valor da
       coluna clima, mas 'Sol' e.
    """
    if len(matriz) < 2:
        return False
    primeira, corpo = matriz[0], matriz[1:]
    if any(_eh_numero(c) for c in primeira):
        return False
    if any(_eh_numero(c) for linha in corpo[:8] for c in linha):
        return True

    amostra = corpo[:200]
    fora = 0
    for i, valor in enumerate(primeira):
        coluna = set(linha[i] for linha in amostra if i < len(linha))
        if valor not in coluna:
            fora += 1
    # Maioria dos rotulos ausente da propria coluna = cabecalho.
    return fora >= max(1, (len(primeira) + 1) // 2)


def _resolver(texto, delimitador='auto', cabecalho='auto'):
    """
    Etapa comum de `analisar` e `carregar`.

    Devolve (chave_delimitador, tem_cabecalho, nomes, linhas_de_dados, n_colunas).
    """
    linhas = _linhas_uteis(texto)
    if not linhas:
        raise ErroLeitura('O arquivo está vazio ou só contém comentários.')
    if len(linhas) > MAX_LINHAS + 1:
        raise ErroLeitura(
            f'O arquivo tem mais de {MAX_LINHAS} linhas. Como todos os modelos '
            'são treinados em Python puro a cada requisição, envie uma amostra '
            'menor da base.')

    chave = delimitador if delimitador != 'auto' else detectar_delimitador(linhas)
    if chave not in DELIMITADORES:
        raise ErroLeitura(
            f"Delimitador inválido: '{chave}'. Use um de: "
            f"{', '.join(sorted(DELIMITADORES))}.")
    delim = DELIMITADORES[chave]

    matriz = [_dividir(linha, delim) for linha in linhas]
    tamanhos = [len(l) for l in matriz]
    n_colunas = max(set(tamanhos), key=tamanhos.count)
    if n_colunas < 2:
        raise ErroLeitura(
            'Cada linha precisa ter pelo menos duas colunas: um atributo e a '
            'classe. Confira o delimitador escolhido.')
    if n_colunas > MAX_COLUNAS:
        raise ErroLeitura(
            f'O arquivo tem {n_colunas} colunas — o limite da aplicação é '
            f'{MAX_COLUNAS}.')

    if cabecalho == 'auto':
        tem_cabecalho = detectar_cabecalho(matriz)
    elif cabecalho in ('sim', True):
        tem_cabecalho = True
    elif cabecalho in ('nao', 'não', False):
        tem_cabecalho = False
    else:
        raise ErroLeitura(
            f"Valor inválido para cabeçalho: '{cabecalho}'. Use 'auto', "
            "'sim' ou 'nao'.")

    if tem_cabecalho:
        nomes = [c if c else f'Coluna {i + 1}' for i, c in enumerate(matriz[0])]
        corpo = matriz[1:]
    else:
        nomes = [f'Coluna {i + 1}' for i in range(n_colunas)]
        corpo = matriz

    if not corpo:
        raise ErroLeitura('O arquivo só tem cabeçalho, sem linhas de dados.')

    # Completa os nomes caso o cabecalho seja mais curto que o corpo.
    while len(nomes) < n_colunas:
        nomes.append(f'Coluna {len(nomes) + 1}')

    return chave, tem_cabecalho, nomes[:n_colunas], corpo, n_colunas


def _perfil_coluna(corpo, indice, nome=''):
    """Tipo, valores distintos e exemplos de uma coluna."""
    valores = [linha[indice] for linha in corpo
               if indice < len(linha) and not _ausente(linha[indice])]
    numerica = bool(valores) and all(_eh_numero(v) for v in valores)
    distintos = sorted(set(valores))
    return {
        'indice': indice,
        'nome': nome,
        'tipo': 'numerico' if numerica else 'categorico',
        'n_distintos': len(distintos),
        'distintos': distintos,
        'exemplos': distintos[:6],
        'n_ausentes': len(corpo) - len(valores),
    }


# ---------------------------------------------------------------------------
# API publica
# ---------------------------------------------------------------------------
def analisar(texto, delimitador='auto', cabecalho='auto'):
    """
    Inspeciona o arquivo SEM importa-lo — alimenta a tela de pre-visualizacao.

    Devolve o delimitador detectado, se ha cabecalho, o perfil de cada coluna
    (tipo, quantos valores distintos, exemplos) e um palpite de qual coluna
    guarda a classe.
    """
    chave, tem_cabecalho, nomes, corpo, n_colunas = _resolver(
        texto, delimitador, cabecalho)

    colunas = []
    for i in range(n_colunas):
        perfil = _perfil_coluna(corpo, i, nomes[i])
        # A lista completa de valores distintos so interessa para colunas
        # categoricas curtas — evita devolver 1000 rotulos ao navegador.
        perfil['valores'] = (perfil['distintos']
                             if perfil['tipo'] == 'categorico'
                             and perfil['n_distintos'] <= 30 else None)
        del perfil['distintos']
        colunas.append(perfil)

    return {
        'delimitador': chave,
        'delimitador_rotulo': ROTULOS_DELIMITADOR[chave],
        'cabecalho': tem_cabecalho,
        'n_colunas': n_colunas,
        'n_linhas': len(corpo),
        'colunas': colunas,
        'coluna_classe_sugerida': _sugerir_coluna_classe(colunas, n_colunas),
        'previa': [linha[:n_colunas] for linha in corpo[:8]],
        'nomes': nomes,
    }


def _sugerir_coluna_classe(colunas, n_colunas):
    """
    Palpite da coluna de classe.

    Prioridade: (1) nome tipico de rotulo; (2) ultima coluna, se categorica e
    com poucos valores; (3) primeira coluna categorica curta; (4) ultima
    coluna, por convencao.
    """
    tipicos = ('classe', 'class', 'rotulo', 'rótulo', 'label', 'target',
               'categoria', 'decisao', 'decisão', 'especie', 'espécie',
               'species', 'y')
    for c in colunas:
        if c['nome'].strip().lower() in tipicos:
            return c['indice']

    ultima = colunas[n_colunas - 1]
    if ultima['tipo'] == 'categorico' and 2 <= ultima['n_distintos'] <= MAX_CLASSES:
        return ultima['indice']

    for c in colunas:
        if c['tipo'] == 'categorico' and 2 <= c['n_distintos'] <= MAX_CLASSES:
            return c['indice']

    return n_colunas - 1


def carregar(texto, delimitador='auto', cabecalho='auto', coluna_classe=None,
             colunas_ignoradas=()):
    """
    Converte o texto na base de dados usada pelo resto do projeto.

    Parametros
    ----------
    texto              : conteudo integral do arquivo
    delimitador        : chave de `DELIMITADORES` ou 'auto'
    cabecalho          : 'auto' | 'sim' | 'nao'
    coluna_classe      : indice da coluna de rotulos (negativo conta do fim);
                         None usa a sugestao automatica
    colunas_ignoradas  : indices de colunas a descartar (ex.: um 'id')

    Retorna um dicionario com:
      dados     — [{'atributos': [float, ...], 'classe': str}, ...]
      features  — nomes das colunas de atributo, na ordem do vetor
      classes   — rotulos distintos, ordenados
      valores   — {posicao_no_vetor: [rotulos]} das colunas categoricas
      tipo      — 'categorico' se TODOS os atributos forem categoricos
      avisos    — mensagens sobre linhas descartadas, classes escassas etc.
    """
    chave, tem_cabecalho, nomes, corpo, n_colunas = _resolver(
        texto, delimitador, cabecalho)

    if coluna_classe is None:
        perfis_todos = [_perfil_coluna(corpo, i, nomes[i])
                        for i in range(n_colunas)]
        coluna_classe = _sugerir_coluna_classe(perfis_todos, n_colunas)
    if coluna_classe < 0:
        coluna_classe += n_colunas
    if not 0 <= coluna_classe < n_colunas:
        raise ErroLeitura(
            f'Coluna de classe inválida: {coluna_classe}. O arquivo tem '
            f'{n_colunas} colunas (0 a {n_colunas - 1}).')

    ignoradas = {i if i >= 0 else i + n_colunas for i in colunas_ignoradas}
    ignoradas.discard(coluna_classe)
    indices_atributos = [i for i in range(n_colunas)
                         if i != coluna_classe and i not in ignoradas]
    if not indices_atributos:
        raise ErroLeitura('Nenhuma coluna de atributo sobrou após as exclusões.')
    if len(indices_atributos) < 2:
        raise ErroLeitura(
            'A aplicação precisa de pelo menos DOIS atributos: os gráficos de '
            'dispersão e as superfícies de decisão são desenhados em 2D.')

    # --- 1a passada: tipo de cada coluna de atributo ------------------------
    perfis = {i: _perfil_coluna(corpo, i, nomes[i]) for i in indices_atributos}
    categoricas = {i: perfis[i]['distintos'] for i in indices_atributos
                   if perfis[i]['tipo'] == 'categorico'}

    for i, rotulos in categoricas.items():
        if len(rotulos) > MAX_CLASSES * 3:
            raise ErroLeitura(
                f"A coluna '{nomes[i]}' tem {len(rotulos)} valores de texto "
                'distintos. Colunas assim (identificadores, nomes) devem ser '
                'marcadas como ignoradas na importação.')

    codigos = {i: {rotulo: float(k) for k, rotulo in enumerate(rotulos)}
               for i, rotulos in categoricas.items()}

    # --- 2a passada: montar as amostras -------------------------------------
    dados = []
    descartadas_tamanho = descartadas_ausente = descartadas_valor = 0

    for linha in corpo:
        if len(linha) < n_colunas:
            descartadas_tamanho += 1
            continue
        classe = linha[coluna_classe].strip()
        if _ausente(classe):
            descartadas_ausente += 1
            continue
        if any(_ausente(linha[i]) for i in indices_atributos):
            descartadas_ausente += 1
            continue
        try:
            atributos = [codigos[i][linha[i]] if i in codigos
                         else _para_numero(linha[i])
                         for i in indices_atributos]
        except (ValueError, KeyError):
            descartadas_valor += 1
            continue
        dados.append({'atributos': atributos, 'classe': classe})

    if not dados:
        raise ErroLeitura(
            'Nenhuma linha válida foi lida. Confira o delimitador, o cabeçalho '
            'e a coluna de classe escolhidos.')

    classes = sorted(set(d['classe'] for d in dados))
    if len(classes) < 2:
        raise ErroLeitura(
            f"A coluna de classe escolhida ('{nomes[coluna_classe]}') só tem um "
            'valor. Escolha a coluna que contém os rótulos.')
    if len(classes) > MAX_CLASSES:
        raise ErroLeitura(
            f"A coluna '{nomes[coluna_classe]}' tem {len(classes)} valores "
            f'distintos — mais que o limite de {MAX_CLASSES} classes. Ela '
            'provavelmente não é a coluna de rótulos.')

    # Cada classe precisa de amostras suficientes para o split estratificado
    # deixar ao menos uma no treino e uma no teste.
    escassas = [c for c in classes
                if sum(1 for d in dados if d['classe'] == c) < 4]

    avisos = []
    if descartadas_tamanho:
        avisos.append(f'{descartadas_tamanho} linha(s) com número de colunas '
                      'diferente do restante foram descartadas.')
    if descartadas_ausente:
        avisos.append(f'{descartadas_ausente} linha(s) com valores ausentes '
                      'foram descartadas.')
    if descartadas_valor:
        avisos.append(f'{descartadas_valor} linha(s) com valores não '
                      'convertíveis foram descartadas.')
    if escassas:
        avisos.append('Classes com menos de 4 amostras (o split estratificado '
                      'fica frágil): ' + ', '.join(escassas) + '.')

    # `valores` e indexado pela POSICAO no vetor de atributos, nao pela coluna
    # original — e assim que o resto do projeto consulta os rotulos.
    valores = {pos: categoricas[i]
               for pos, i in enumerate(indices_atributos) if i in categoricas}

    return {
        'dados': dados,
        'features': [nomes[i] for i in indices_atributos],
        'classes': classes,
        'valores': valores or None,
        'tipo': ('categorico' if len(valores) == len(indices_atributos)
                 else 'continuo'),
        'coluna_classe': coluna_classe,
        'nome_coluna_classe': nomes[coluna_classe],
        'colunas_ignoradas': sorted(ignoradas),
        'delimitador': chave,
        'cabecalho': tem_cabecalho,
        'avisos': avisos,
        'n_amostras': len(dados),
    }
