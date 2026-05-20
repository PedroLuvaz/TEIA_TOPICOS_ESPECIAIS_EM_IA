"""
Abre direto a interface interativa do Iris Dataset.
Uso: python run_gui.py
"""

import os
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RAIZ, 'src'))

from core.data_loader import carregar_dados_iris, split_estratificado
from core.classifier import treinar
from gui.interativo import abrir_interface

CAMINHO_DADOS = os.path.join(RAIZ, 'data', 'Iris data.xls')
INDICES_PETALA = [2, 3]

dados = carregar_dados_iris(CAMINHO_DADOS)
for i, d in enumerate(dados):
    d['indice'] = i

dados_treino, dados_teste = split_estratificado(dados, proporcao_treino=0.7, semente=42)
prototipos = treinar(dados_treino, INDICES_PETALA)

abrir_interface(dados, prototipos, dados_treino, dados_teste)
