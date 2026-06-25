"""Janela principal do aplicativo IRIS - TEIA.

Cabecalho identitario + tabbar com as abas ativas:
  Aba 1 — Classificador de Distancia Minima
  Aba 2 — Perceptron & Regra Delta
  Aba 3 — Metricas Avancadas (Aula PR_51)  ← NOVA
  Aba 4 — placeholder
"""
import tkinter as tk
from tkinter import ttk

from . import theme as T
from .tab_distancia_minima import TabDistanciaMinima
from .tab_perceptron_delta import TabPerceptronDelta
from .tab_metricas_avancadas import TabMetricasAvancadas
from .tab_bayes import TabBayes

_SCROLL_UNIT = 3

GRUPO      = 'Erick Nathan   ·   Laura Barbosa   ·   Pedro Lucas'
DISCIPLINA = 'Topicos Especiais em Inteligencia Artificial'

# ABAS_FUTURAS = [
#     ('Aba 4', 'em breve'),
# ]


# ===========================================================================
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title('IRIS  ·  Classificadores Lineares  ·  TEIA')
        self.geometry('1320x900')
        self.minsize(1180, 620)

        T.aplicar_tema(self)

        self._construir_scroll_wrapper()
        self._construir_cabecalho()
        self._construir_notebook()
        self._construir_rodape()

    # ------------------------------------------------------------------
    def _construir_scroll_wrapper(self):
        outer = tk.Frame(self, bg=T.BG)
        outer.pack(fill='both', expand=True)

        self._scrollbar = ttk.Scrollbar(outer, orient='vertical')
        self._scrollbar.pack(side='right', fill='y')

        self._canvas = tk.Canvas(
            outer, bg=T.BG, highlightthickness=0,
            yscrollcommand=self._scrollbar.set
        )
        self._canvas.pack(side='left', fill='both', expand=True)
        self._scrollbar.config(command=self._canvas.yview)

        self._inner = tk.Frame(self._canvas, bg=T.BG)
        self._win_id = self._canvas.create_window(
            (0, 0), window=self._inner, anchor='nw'
        )

        self._inner.bind('<Configure>', self._ao_redimensionar_inner)
        self._canvas.bind('<Configure>', self._ao_redimensionar_canvas)
        self.bind_all('<MouseWheel>', self._ao_rolar_mouse)

    def _ao_redimensionar_inner(self, event):
        self._canvas.configure(scrollregion=self._canvas.bbox('all'))

    def _ao_redimensionar_canvas(self, event):
        self._canvas.itemconfig(self._win_id, width=event.width)

    def _ao_rolar_mouse(self, event):
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)) * _SCROLL_UNIT,
                                  'units')

    # ------------------------------------------------------------------
    def _chip(self, master, texto, fg=None, bg=None):
        """Pequena etiqueta arredondada-aparente com informacao do estudo."""
        chip = tk.Label(master, text=texto,
                        bg=bg or T.BG_PANEL, fg=fg or T.FG_MUTED,
                        font=(T.FONT_FAMILY, 8, 'bold'),
                        padx=10, pady=3,
                        highlightthickness=1,
                        highlightbackground=T.BORDER)
        chip.pack(side='left', padx=(0, 6))
        return chip

    def _construir_cabecalho(self):
        tk.Frame(self._inner, bg=T.ACCENT, height=3).pack(fill='x')

        cab = tk.Frame(self._inner, bg=T.BG_CARD)
        cab.pack(fill='x')

        topo = tk.Frame(cab, bg=T.BG_CARD)
        topo.pack(fill='x', padx=28, pady=(16, 0))

        esq = tk.Frame(topo, bg=T.BG_CARD)
        esq.pack(side='left', fill='y')
        tk.Label(esq, text='ESTUDO EXPERIMENTAL  ·  BASE IRIS  ·  UEPB 2026',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP,
                 font=T.FONT_KICKER).pack(anchor='w')
        tk.Label(esq, text='Classificadores Lineares — Iris',
                 bg=T.BG_CARD, fg=T.FG,
                 font=T.FONT_DISPLAY).pack(anchor='w', pady=(2, 0))

        dir_ = tk.Frame(topo, bg=T.BG_CARD)
        dir_.pack(side='right', fill='y')
        tk.Label(dir_, text=DISCIPLINA, bg=T.BG_CARD, fg=T.FG_MUTED,
                 font=T.FONT_SUBTITLE).pack(anchor='e')
        tk.Label(dir_, text=GRUPO, bg=T.BG_CARD, fg=T.FG,
                 font=(T.FONT_FAMILY_TITLE, 11, 'bold')).pack(anchor='e', pady=(2, 0))

        chips = tk.Frame(cab, bg=T.BG_CARD)
        chips.pack(fill='x', padx=28, pady=(10, 14))
        self._chip(chips, 'Python puro', fg=T.ACCENT_DEEP, bg=T.ACCENT_SOFT)
        self._chip(chips, '150 amostras')
        self._chip(chips, '4 atributos')
        self._chip(chips, '3 classes')
        self._chip(chips, 'split 70/30 estratificado')
        self._chip(chips, 'seed 42')
        self._chip(chips, 'sem numpy · sem scikit-learn')

        tk.Frame(self._inner, bg=T.BORDER, height=1).pack(fill='x')

    # ------------------------------------------------------------------
    def _construir_notebook(self):
        wrap = tk.Frame(self._inner, bg=T.BG)
        wrap.pack(fill='both', expand=True, padx=20, pady=(8, 0))

        self.notebook = ttk.Notebook(wrap)
        self.notebook.pack(fill='both', expand=True)

        # Aba 1
        aba1 = TabDistanciaMinima(self.notebook)
        self.notebook.add(aba1, text='   Distancia Minima   ')

        # Aba 2
        aba2 = TabPerceptronDelta(self.notebook)
        self.notebook.add(aba2, text='   Perceptron & Delta   ')

        # Aba 3 — NOVA: Metricas Avancadas
        aba3 = TabMetricasAvancadas(self.notebook)
        self.notebook.add(aba3, text='   Metricas Avancadas   ')

        # Aba 4 — NOVA: Bayes & Normalidade
        aba4 = TabBayes(self.notebook)
        self.notebook.add(aba4, text='   Bayes & Normalidade   ')

        # Abas placeholder
        # for nome, status in ABAS_FUTURAS:
        #     ph = self._construir_placeholder(nome, status)
        #     self.notebook.add(ph, text=f'   {nome}   ')

        # Desabilitar apenas as abas futuras
        # for i in range(3, 3 + len(ABAS_FUTURAS)):
        #     self.notebook.tab(i, state='disabled')

    # def _construir_placeholder(self, nome, status):
    #     f = tk.Frame(self.notebook, bg=T.BG)
    #     c = tk.Frame(f, bg=T.BG)
    #     c.place(relx=0.5, rely=0.5, anchor='center')
    #     tk.Label(c, text=status.upper(), bg=T.BG, fg=T.ACCENT,
    #              font=T.FONT_KICKER).pack(anchor='w')
    #     tk.Label(c, text=nome, bg=T.BG, fg=T.FG,
    #              font=T.FONT_DISPLAY).pack(anchor='w', pady=(4, 0))
    #     tk.Label(c,
    #              text='Esta aba sera implementada em fase futura do projeto.',
    #              bg=T.BG, fg=T.FG_MUTED, font=T.FONT_BODY,
    #              wraplength=560, justify='left').pack(anchor='w', pady=(14, 0))
    #     return f

    # ------------------------------------------------------------------
    def _construir_rodape(self):
        tk.Frame(self._inner, bg=T.BORDER, height=1).pack(fill='x')
        rod = tk.Frame(self._inner, bg=T.BG_CARD, height=30)
        rod.pack(fill='x')
        rod.pack_propagate(False)
        tk.Label(rod,
                 text='Fisher (1936)  ·  Iris flower data set  ·  '
                      'matematica em Python puro (math_utils.py)',
                 bg=T.BG_CARD, fg=T.FG_MUTED,
                 font=T.FONT_SUBTITLE).pack(side='left', padx=20, pady=5)
        tk.Label(rod,
                 text='UEPB  ·  Topicos Especiais em IA  ·  2026',
                 bg=T.BG_CARD, fg=T.FG_DIM,
                 font=T.FONT_SUBTITLE).pack(side='right', padx=20, pady=5)


# ===========================================================================
def iniciar():
    App().mainloop()