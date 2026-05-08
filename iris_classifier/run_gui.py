"""Ponto de entrada da interface grafica.

Uso:
    python iris_classifier/run_gui.py

Coloca o diretorio iris_classifier/ no sys.path para que os modulos
de Python puro (data_loader, classifier, math_utils, evaluator) sejam
importaveis a partir do pacote gui/.
"""
import os
import sys

PASTA_IRIS = os.path.dirname(os.path.abspath(__file__))
PROJETO    = os.path.dirname(PASTA_IRIS)

for p in (PROJETO, PASTA_IRIS):
    if p not in sys.path:
        sys.path.insert(0, p)

from gui.app import iniciar


if __name__ == '__main__':
    iniciar()
