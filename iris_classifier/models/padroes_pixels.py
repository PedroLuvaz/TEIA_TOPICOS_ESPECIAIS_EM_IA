"""
Padroes de referencia 8x8 do bonus interativo do Lab 5 (Aula PR_711).

O slide ilustra o problema "galinha vs homem" como o reconhecimento de uma
imagem de 8x8 pixels (64 valores de cinza, um por neuronio de entrada
a1...a64). Estes dois padroes foram desenhados a mao inspirados nas
silhuetas "Man" e "Chicken" do slide.

Valores: 0.0 (fundo claro) a 1.0 (pixel escuro / silhueta).

Modulo sem dependencias de interface — compartilhado entre a GUI Tkinter
(`gui/tab_feedforward.py`) e a API web (`web_app/backend`).
"""

HOMEM_PIXELS = [
    [0.00, 0.30, 0.40, 0.40, 0.30, 0.00, 0.00, 0.00],
    [0.00, 0.40, 0.45, 0.45, 0.40, 0.00, 0.00, 0.00],
    [0.00, 0.40, 0.90, 0.90, 0.40, 0.00, 0.00, 0.00],
    [0.35, 0.45, 0.50, 0.50, 0.45, 0.25, 0.15, 0.00],
    [0.35, 0.55, 0.60, 0.60, 0.55, 0.30, 0.20, 0.00],
    [0.30, 0.50, 0.50, 0.50, 0.45, 0.00, 0.00, 0.00],
    [0.00, 0.25, 0.00, 0.25, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.15, 0.00, 0.15, 0.00, 0.00, 0.00, 0.00],
]

GALINHA_PIXELS = [
    [0.00, 0.30, 0.40, 0.20, 0.00, 0.00, 0.00, 0.00],
    [0.00, 0.45, 0.90, 0.55, 0.15, 0.00, 0.00, 0.00],
    [0.10, 0.55, 0.65, 0.60, 0.35, 0.25, 0.10, 0.00],
    [0.00, 0.40, 0.60, 0.60, 0.55, 0.45, 0.10, 0.00],
    [0.00, 0.20, 0.55, 0.60, 0.55, 0.35, 0.00, 0.00],
    [0.00, 0.00, 0.50, 0.55, 0.20, 0.05, 0.00, 0.00],
    [0.00, 0.00, 0.20, 0.15, 0.05, 0.00, 0.00, 0.00],
    [0.00, 0.00, 0.05, 0.05, 0.00, 0.00, 0.00, 0.00],
]


def achatar(grade):
    """Converte uma grade 8x8 no vetor de 64 entradas da rede."""
    return [v for linha in grade for v in linha]
