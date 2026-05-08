"""Janela principal do aplicativo IRIS - TEIA.

Cabecalho identitario + tabbar com a aba ativa (Distancia Minima)
e abas placeholder ja preparadas para futuras implementacoes que
usarao bibliotecas de ML (scikit-learn, etc).
"""
import tkinter as tk
from tkinter import ttk

from . import theme as T
from .tab_distancia_minima import TabDistanciaMinima


GRUPO = 'Erick Nathan   ·   Laura Barbosa   ·   Pedro Lucas'
DISCIPLINA = 'Topicos Especiais em Inteligencia Artificial'

ABAS_FUTURAS = [
    ('Aba 2', 'em breve'),
    ('Aba 3', 'em breve'),
    ('Aba 4', 'em breve'),
]


# ===========================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('IRIS  ·  Classificador de Distancia Minima')
        self.geometry('1320x880')
        self.minsize(1180, 800)

        T.aplicar_tema(self)

        self._construir_cabecalho()
        self._construir_notebook()
        self._construir_rodape()

    # ------------------------------------------------------------------
    def _construir_cabecalho(self):
        # Faixa fina ambar no topo (assinatura visual)
        tk.Frame(self, bg=T.ACCENT, height=2).pack(fill='x', side='top')

        cab = tk.Frame(self, bg=T.BG, height=96)
        cab.pack(fill='x', side='top')
        cab.pack_propagate(False)

        # Esquerda — kicker + titulo
        esq = tk.Frame(cab, bg=T.BG)
        esq.pack(side='left', fill='y', padx=(28, 0), pady=18)
        tk.Label(esq, text='IRIS  ·  TEIA  ·  2026',
                 bg=T.BG, fg=T.ACCENT,
                 font=T.FONT_KICKER).pack(anchor='w')
        tk.Label(esq, text='Classificador de Distancia Minima',
                 bg=T.BG, fg=T.FG,
                 font=T.FONT_DISPLAY).pack(anchor='w', pady=(2, 0))
        tk.Label(esq,
                 text='Python puro  ·  sem numpy  ·  sem scikit-learn',
                 bg=T.BG, fg=T.FG_MUTED,
                 font=T.FONT_SUBTITLE).pack(anchor='w', pady=(2, 0))

        # Direita — credito do grupo
        dir_ = tk.Frame(cab, bg=T.BG)
        dir_.pack(side='right', fill='y', padx=(0, 28), pady=18)
        tk.Label(dir_, text='GRUPO',
                 bg=T.BG, fg=T.ACCENT,
                 font=T.FONT_KICKER).pack(anchor='e')
        tk.Label(dir_, text=GRUPO,
                 bg=T.BG, fg=T.FG,
                 font=('Cambria', 12, 'bold')).pack(anchor='e', pady=(2, 0))
        tk.Label(dir_, text=DISCIPLINA,
                 bg=T.BG, fg=T.FG_MUTED,
                 font=T.FONT_SUBTITLE).pack(anchor='e', pady=(2, 0))

        # Linha separadora
        tk.Frame(self, bg=T.BORDER, height=1).pack(fill='x', side='top')

    # ------------------------------------------------------------------
    def _construir_notebook(self):
        # Espaco de respiro entre o cabecalho e o notebook
        wrap = tk.Frame(self, bg=T.BG)
        wrap.pack(fill='both', expand=True, padx=20, pady=(8, 0))

        self.notebook = ttk.Notebook(wrap)
        self.notebook.pack(fill='both', expand=True)

        # Aba ativa
        aba1 = TabDistanciaMinima(self.notebook)
        self.notebook.add(aba1, text='   Distancia Minima   ')

        # Abas placeholder
        for nome, status in ABAS_FUTURAS:
            ph = self._construir_placeholder(nome, status)
            self.notebook.add(ph, text=f'   {nome}   ')

        # Desabilitar abas futuras
        for i in range(1, 1 + len(ABAS_FUTURAS)):
            self.notebook.tab(i, state='disabled')

    def _construir_placeholder(self, nome, status):
        f = tk.Frame(self.notebook, bg=T.BG)
        c = tk.Frame(f, bg=T.BG)
        c.place(relx=0.5, rely=0.5, anchor='center')

        tk.Label(c, text=status.upper(), bg=T.BG, fg=T.ACCENT,
                 font=T.FONT_KICKER).pack(anchor='w')
        tk.Label(c, text=nome, bg=T.BG, fg=T.FG,
                 font=T.FONT_DISPLAY).pack(anchor='w', pady=(4, 0))
        tk.Label(c,
                 text=('Esta aba sera implementada em fase futura do projeto.'),
                 bg=T.BG, fg=T.FG_MUTED, font=T.FONT_BODY,
                 wraplength=560, justify='left'
                ).pack(anchor='w', pady=(14, 0))
        return f

    # ------------------------------------------------------------------
    def _construir_rodape(self):
        tk.Frame(self, bg=T.BORDER, height=1).pack(fill='x', side='bottom')
        rod = tk.Frame(self, bg=T.BG_PANEL, height=30)
        rod.pack(fill='x', side='bottom')
        rod.pack_propagate(False)
        tk.Label(rod,
                 text='Iris  ·  150 amostras  ·  4 atributos  ·  3 classes  ·  '
                      'split 70/30  ·  seed 42',
                 bg=T.BG_PANEL, fg=T.FG_MUTED,
                 font=T.FONT_SUBTITLE).pack(side='left', padx=20, pady=5)
        tk.Label(rod,
                 text='UEPB  ·  Topicos Especiais em IA  ·  2026',
                 bg=T.BG_PANEL, fg=T.FG_DIM,
                 font=T.FONT_SUBTITLE).pack(side='right', padx=20, pady=5)


# ===========================================================================
def iniciar():
    App().mainloop()
