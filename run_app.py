"""Abre o aplicativo grafico completo (Distancia Minima + Perceptron & Delta).

Uso: python run_app.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

from gui.app import iniciar

if __name__ == '__main__':
    iniciar()
