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
