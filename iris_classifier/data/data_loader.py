import xlrd
import random

def _linhas_xlsx(caminho):
    """Gerador de linhas de um arquivo .xlsx via openpyxl."""
    import openpyxl
    wb = openpyxl.load_workbook(caminho, read_only=True, data_only=True)
    ws = wb.active
    for row in ws.iter_rows(values_only=True):
        yield list(row)
    wb.close()

def _linhas_xls(caminho):
    """Gerador de linhas de um arquivo .xls via xlrd."""
    wb = xlrd.open_workbook(caminho)
    ws = wb.sheet_by_index(0)
    for i in range(ws.nrows):
        yield ws.row_values(i)

def carregar_dados_iris(caminho_arquivo):
    """
    Le os dados da Iris do arquivo XLS ou XLSX.
    Col 0-3: atributos (features)
    Col 4: classe ('setosa', 'versicolor', 'virginica')
    """
    if caminho_arquivo.lower().endswith('.xlsx'):
        gerador = _linhas_xlsx(caminho_arquivo)
    else:
        gerador = _linhas_xls(caminho_arquivo)

    dados = []
    for linha in gerador:
        try:
            atributos = [float(x) for x in linha[:4]]
            classe = str(linha[4]).strip().lower()
            if not classe:
                continue
            dados.append({'atributos': atributos, 'classe': classe})
        except (ValueError, IndexError, TypeError):
            continue
    return dados

def split_estratificado(dados, proporcao_treino=0.7, semente=42):
    """
    Realiza o split estratificado: proporcao_treino de cada classe para treino.
    """
    random.seed(semente)
    conjunto_treino = []
    conjunto_teste = []
    
    # Identificar classes únicas
    classes = sorted(list(set(d['classe'] for d in dados)))
    
    for classe in classes:
        dados_da_classe = [d for d in dados if d['classe'] == classe]
        random.shuffle(dados_da_classe)
        
        n_treino = int(len(dados_da_classe) * proporcao_treino)
        conjunto_treino += dados_da_classe[:n_treino]
        conjunto_teste += dados_da_classe[n_treino:]
        
    return conjunto_treino, conjunto_teste

def filtrar_por_classes(dados, classes_para_manter):
    """
    Filtra os dados para incluir apenas amostras das classes especificadas.
    """
    return [d for d in dados if d['classe'] in classes_para_manter]


# ---------------------------------------------------------------------------
# Dataset do seminario de Florestas Aleatorias (fim de semana)
# ---------------------------------------------------------------------------
FDS_COLUNAS_TEXTO = ['clima', 'pais', 'dinheiro']
FDS_COLUNAS_CODIGO = ['clima_cod', 'pais_cod', 'dinheiro_cod']
FDS_NOMES_ATRIBUTOS = ['Clima', 'Pais visitam?', 'Dinheiro']
FDS_CLASSES = ['Cinema', 'Compras', 'Ficar em casa', 'Tenis']


def carregar_fim_de_semana(caminho_arquivo, numerico=False):
    """
    Le o CSV gerado por `data.gerar_fim_de_semana`.

    Devolve a mesma estrutura usada no resto do projeto:

        {'atributos': [...], 'classe': 'Cinema', 'ruido': 0, 'id': 1}

    `numerico=False` (padrao) devolve os atributos em texto ('Sol', 'Sim',
    'Rico') — o formato esperado por `models.floresta_categorica`, que faz as
    divisoes multi-way do ID3 como nos slides.

    `numerico=True` devolve os codigos ordinais das colunas `_cod`, para que
    os demais modulos do projeto (Distancia Minima, Bayes, Regra Delta,
    metricas) possam consumir o mesmo arquivo sem alteracao.

    A chave `ruido` marca as instancias cujo rotulo foi trocado na geracao —
    util para separar o erro irredutivel do erro do classificador.
    """
    with open(caminho_arquivo, 'r', encoding='utf-8') as f:
        linhas = [linha.rstrip('\n\r') for linha in f if linha.strip()]

    if not linhas:
        return []

    cabecalho = [c.strip() for c in linhas[0].split(',')]
    pos = {nome: i for i, nome in enumerate(cabecalho)}

    colunas = FDS_COLUNAS_CODIGO if numerico else FDS_COLUNAS_TEXTO
    faltando = [c for c in colunas + ['decisao'] if c not in pos]
    if faltando:
        raise ValueError(
            f'CSV sem as colunas esperadas: {", ".join(faltando)}. '
            f'Gere o arquivo com `python -m data.gerar_fim_de_semana`.')

    dados = []
    for linha in linhas[1:]:
        campos = linha.split(',')
        if len(campos) < len(cabecalho):
            continue
        atributos = [campos[pos[c]].strip() for c in colunas]
        if numerico:
            atributos = [float(v) for v in atributos]
        dados.append({
            'atributos': atributos,
            'classe': campos[pos['decisao']].strip(),
            'ruido': int(campos[pos['ruido']]) if 'ruido' in pos else 0,
            'id': int(campos[pos['id']]) if 'id' in pos else len(dados) + 1,
        })
    return dados
