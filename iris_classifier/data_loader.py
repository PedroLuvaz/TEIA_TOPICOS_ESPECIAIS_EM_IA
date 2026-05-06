import xlrd
import random

def carregar_dados_iris(caminho_arquivo):
    """
    Lê os dados da Iris do arquivo XLS.
    Estrutura esperada:
    Col 0-3: atributos (features)
    Col 4: classe ('setosa', 'versicolor', 'virginica')
    """
    dados = []
    workbook = xlrd.open_workbook(caminho_arquivo)
    planilha = workbook.sheet_by_index(0)
    
    for i in range(planilha.nrows):
        linha = planilha.row_values(i)
        # Pular cabeçalho ou linhas malformadas
        try:
            atributos = [float(x) for x in linha[:4]]
            classe = str(linha[4]).strip().lower()
            if not classe:
                continue
            amostra = {
                'atributos': atributos,
                'classe': classe
            }
            dados.append(amostra)
        except (ValueError, IndexError):
            # Provavelmente cabeçalho ou linha vazia
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
