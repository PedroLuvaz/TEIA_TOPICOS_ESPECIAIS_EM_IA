"""
Aba 2 — Perceptron & Regra Delta.

Tres experimentos do PR4:
  1. Perceptron de Rosenblatt — classificacao binaria do Iris (3 pares)
  2. Regra Delta (Adaline / Widrow-Hoff) — Iris binario, convergencia MSE
  3. XOR com Regra Delta — limite dos classificadores lineares

Layout:
  Coluna esquerda (320 px): cartoes de controle
  Coluna direita (flex): figura matplotlib com 2 subplots
      ax_scatter  (5/8): dispersao + fronteira de decisao
      ax_conv     (3/8): curva de convergencia (erros ou MSE por epoca)
  Painel inferior: 3 MetricBlocks + cartao de analise textual

Toda a matematica vem de perceptron.py e delta_rule.py (Python puro).
"""
import os
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

from data.data_loader import carregar_dados_iris, split_estratificado
from models.perceptron import treinar_perceptron, acuracia_binaria_perceptron
from models.delta_rule import (treinar_delta_iris, treinar_delta_xor,
                               acuracia_binaria_delta,
                               treinar_delta_ova, predizer_delta_ova,
                               acuracia_delta_ova)

from . import theme as T
from .widgets import Card, MetricBlock, separador
from .janela_calculos import JanelaMemoriaCalculoPD


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PROJETO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
CAMINHO_DADOS_V1 = os.path.join(PROJETO_ROOT, 'data', 'Iris data.xls')
CAMINHO_DADOS_V2 = os.path.join(PROJETO_ROOT, 'data', 'iris_data_02.xlsx')
CAMINHOS_DADOS = {'v1': CAMINHO_DADOS_V1, 'v2': CAMINHO_DADOS_V2}

_DS_OPCOES = ['Iris Original  ·  v1', 'Iris Separavel  ·  v2']
_DS_CHAVE  = {'Iris Original  ·  v1': 'v1', 'Iris Separavel  ·  v2': 'v2'}
_DS_LABEL  = {'v1': 'Iris Original  ·  v1', 'v2': 'Iris Separavel  ·  v2'}

CLASSES = ['setosa', 'versicolor', 'virginica']

CONF_ATTR = {
    'petalas': {
        'rotulo_ui': 'Petalas  ·  [2,3]',
        'indices': [2, 3],
        'eixo_x': 'Comp. Petala (cm)',
        'eixo_y': 'Larg. Petala (cm)',
    },
    'sepalas': {
        'rotulo_ui': 'Sepalas  ·  [0,1]',
        'indices': [0, 1],
        'eixo_x': 'Comp. Sepala (cm)',
        'eixo_y': 'Larg. Sepala (cm)',
    },
    'todas': {
        'rotulo_ui': 'Todas (4 Features) · [0,1,2,3]',
        'indices': [0, 1, 2, 3],
        'eixo_x': 'Comp. Petala (cm)',
        'eixo_y': 'Larg. Petala (cm)',
    },
}

PAR_POR_MODO = {
    'sv': ('setosa',     'versicolor'),
    'vv': ('versicolor', 'virginica'),
    'sg': ('setosa',     'virginica'),
}
PAR_ROTULO = {
    'sv': 'Setosa  x  Versicolor',
    'vv': 'Versicolor  x  Virginica',
    'sg': 'Setosa  x  Virginica',
}

CORES_CLASSE = {
    'setosa':     T.DATA_BLUE,
    'versicolor': T.DATA_MINT,
    'virginica':  T.DATA_CORAL,
}
MARCADORES_CLASSE = {
    'setosa':     'o',
    'versicolor': '^',
    'virginica':  's',
}

ALGO_DEFAULTS = {
    'perceptron': {'taxa': '0.03', 'epocas': '100'},
    'delta':      {'taxa': '0.02', 'epocas': '200'},
    'delta_ova':  {'taxa': '0.02', 'epocas': '200'},
    'xor':        {'taxa': '0.02', 'epocas': '300'},
}

# Nomes das 4 features (na ordem do arquivo .xls)
FEATURES = [
    ('s_comp', 'Comp. Sepala',  0),
    ('s_larg', 'Larg. Sepala',  1),
    ('p_comp', 'Comp. Petala',  2),
    ('p_larg', 'Larg. Petala',  3),
]


# ---------------------------------------------------------------------------
# Classe da aba
# ---------------------------------------------------------------------------
class TabPerceptronDelta(tk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, bg=T.BG, **kw)

        # --- estado ---
        self.dados = []
        self.dados_treino = []
        self.dados_teste = []

        self.var_dataset = tk.StringVar(value='v1')
        self.var_algo  = tk.StringVar(value='perceptron')
        self.var_par   = tk.StringVar(value='sv')
        self.var_attr  = tk.StringVar(value='petalas')
        self.var_taxa  = tk.StringVar(value='0.03')
        self.var_epocas = tk.StringVar(value='100')
        self.var_prop_treino = tk.StringVar(value='0.70')
        self.var_semente = tk.StringVar(value='42')

        self.w = None
        self.pesos_ova = None         # dict {classe: w} para Delta OvA
        self.historico = []
        self.historico_ova = None     # dict {classe: [mse/epoca]} para OvA
        self.n_epocas_treinadas = 0
        self.acc_teste = None
        self.convergiu = False
        self._classe_pos = None
        self._classe_neg = None

        # 4 campos da classificacao manual (uma StringVar por feature)
        self.vars_test = {
            's_comp': tk.StringVar(value='5.0'),
            's_larg': tk.StringVar(value='3.4'),
            'p_comp': tk.StringVar(value='4.5'),
            'p_larg': tk.StringVar(value='1.5'),
        }

        self._construir_layout()
        self._carregar_dados()
        self._atualizar_labels_testar()
        self._desenhar_inicial()
        self._escrever_analise_inicial()

    # -----------------------------------------------------------------------
    # Layout principal
    # -----------------------------------------------------------------------
    def _construir_layout(self):
        self.columnconfigure(0, minsize=320)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._coluna_controles()
        self._coluna_visualizacao()

    # -----------------------------------------------------------------------
    # Coluna esquerda — controles
    # -----------------------------------------------------------------------
    def _coluna_controles(self):
        self._wrap_esq = tk.Frame(self, bg=T.BG)
        self._wrap_esq.grid(row=0, column=0, sticky='nsew',
                            padx=(T.PAD_PAGE, T.GAP), pady=12)
        self._wrap_esq.columnconfigure(0, weight=1)

        # --- Seletor de dataset (sempre visivel, fora do relayoutar) ---
        fds = tk.Frame(self._wrap_esq, bg=T.BG)
        fds.pack(fill='x', pady=(0, T.GAP_SM))
        tk.Label(fds, text='Dataset:', bg=T.BG, fg=T.FG_MUTED,
                 font=T.FONT_LABEL).pack(side='left', padx=(0, 8))
        self._combo_dataset = ttk.Combobox(
            fds, values=_DS_OPCOES, state='readonly',
            width=22, font=T.FONT_BODY)
        self._combo_dataset.set(_DS_LABEL[self.var_dataset.get()])
        self._combo_dataset.pack(side='left', fill='x', expand=True)
        self._combo_dataset.bind('<<ComboboxSelected>>',
                                 self._ao_selecionar_dataset)

        # --- Card 1: Algoritmo ---
        self.card_algo = Card(self._wrap_esq, titulo='algoritmo')
        for val, label in [
            ('perceptron', 'Perceptron'),
            ('delta',      'Regra Delta  (Adaline)'),
            ('delta_ova',  'Regra Delta  ·  OvA (comparativo)'),
            ('xor',        'XOR  —  Regra Delta'),
        ]:
            tk.Radiobutton(
                self.card_algo, text=label,
                value=val, variable=self.var_algo,
                bg=T.BG_CARD, fg=T.FG,
                selectcolor=T.BG_HOVER,
                activebackground=T.BG_CARD, activeforeground=T.ACCENT_DEEP,
                font=T.FONT_BODY, anchor='w',
                borderwidth=0, highlightthickness=0,
                command=self._ao_trocar_algo,
            ).pack(fill='x', padx=14, pady=2)
        tk.Frame(self.card_algo, bg=T.BG_CARD, height=4).pack()

        # --- Card 2: Par de classes e Atributos (Iris) lado a lado ---
        self.frame_config_row = tk.Frame(self._wrap_esq, bg=T.BG)
        
        self.card_par = Card(self.frame_config_row, titulo='par de classes')
        for chave, rotulo in PAR_ROTULO.items():
            tk.Radiobutton(
                self.card_par, text=rotulo,
                value=chave, variable=self.var_par,
                bg=T.BG_CARD, fg=T.FG,
                selectcolor=T.BG_HOVER,
                activebackground=T.BG_CARD, activeforeground=T.ACCENT_DEEP,
                font=T.FONT_BODY, anchor='w',
                borderwidth=0, highlightthickness=0,
                command=self._ao_mudar_config,
            ).pack(fill='x', padx=14, pady=1)
        tk.Frame(self.card_par, bg=T.BG_CARD, height=2).pack()

        self.card_attr = Card(self.frame_config_row, titulo='atributos')
        for chave, cfg in CONF_ATTR.items():
            tk.Radiobutton(
                self.card_attr, text=cfg['rotulo_ui'],
                value=chave, variable=self.var_attr,
                bg=T.BG_CARD, fg=T.FG,
                selectcolor=T.BG_HOVER,
                activebackground=T.BG_CARD, activeforeground=T.ACCENT_DEEP,
                font=T.FONT_BODY, anchor='w',
                borderwidth=0, highlightthickness=0,
                command=self._ao_mudar_config,
            ).pack(fill='x', padx=14, pady=1)
        tk.Frame(self.card_attr, bg=T.BG_CARD, height=2).pack()

        # --- Card 3: Divisao dos dados ---
        self.card_split = Card(self._wrap_esq, titulo='divisao dos dados')
        form_s = tk.Frame(self.card_split, bg=T.BG_CARD)
        form_s.pack(fill='x', padx=14, pady=(2, 6))
        form_s.columnconfigure(1, weight=1)
        
        tk.Label(form_s, text='Proporcao Treino',
                 bg=T.BG_CARD, fg=T.FG_MUTED,
                 font=T.FONT_LABEL, anchor='w'
                ).grid(row=0, column=0, sticky='w', pady=(0, 2))
        ttk.Entry(form_s, textvariable=self.var_prop_treino,
                  font=T.FONT_MONO, width=9
                 ).grid(row=0, column=1, sticky='ew', padx=(8, 0))
                 
        tk.Label(form_s, text='Semente (Seed)',
                 bg=T.BG_CARD, fg=T.FG_MUTED,
                 font=T.FONT_LABEL, anchor='w'
                ).grid(row=1, column=0, sticky='w', pady=(4, 2))
        ttk.Entry(form_s, textvariable=self.var_semente,
                  font=T.FONT_MONO, width=9
                 ).grid(row=1, column=1, sticky='ew', padx=(8, 0))

        # --- Card 4: Hiperparâmetros e Treinamento ---
        self.card_hiper = Card(self._wrap_esq, titulo='hiperparametros & treino')
        form = tk.Frame(self.card_hiper, bg=T.BG_CARD)
        form.pack(fill='x', padx=14, pady=(2, 0))
        form.columnconfigure(1, weight=1)

        tk.Label(form, text='Taxa aprendizado  p',
                 bg=T.BG_CARD, fg=T.FG_MUTED,
                 font=T.FONT_LABEL, anchor='w'
                ).grid(row=0, column=0, sticky='w', pady=(0, 2))
        ttk.Entry(form, textvariable=self.var_taxa,
                 font=T.FONT_MONO, width=9
                 ).grid(row=0, column=1, sticky='ew', padx=(8, 0))

        tk.Label(form, text='Max. Epocas',
                 bg=T.BG_CARD, fg=T.FG_MUTED,
                 font=T.FONT_LABEL, anchor='w'
                 ).grid(row=1, column=0, sticky='w', pady=(4, 2))
        ttk.Entry(form, textvariable=self.var_epocas,
                  font=T.FONT_MONO, width=9
                 ).grid(row=1, column=1, sticky='ew', padx=(8, 0))

        # --- Botao Treinar (dentro do card de hiperparâmetros) ---
        self._frame_btn = tk.Frame(self.card_hiper, bg=T.BG_CARD)
        self._frame_btn.pack(fill='x', padx=14, pady=(8, 6))
        ttk.Button(
            self._frame_btn, text='Treinar Modelo  >',
            style='Primary.TButton',
            command=self._treinar,
        ).pack(fill='x', ipady=2)
        self.lbl_status = tk.Label(
            self._frame_btn, text='Modelo nao treinado.',
            bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_MONO_SM,
            anchor='w', wraplength=280, justify='left',
        )
        self.lbl_status.pack(fill='x', pady=(4, 0))

        # --- Card: classificar amostra, predição e memória ---
        self.card_testar = Card(self._wrap_esq, titulo='classificacao & predicao')
        form_t = tk.Frame(self.card_testar, bg=T.BG_CARD)
        form_t.pack(fill='x', padx=14, pady=(2, 0))
        form_t.columnconfigure(1, weight=1)
        self.lbls_test = {}
        for row, (chave, nome_legivel, _idx) in enumerate(FEATURES):
            lbl = tk.Label(form_t, text=nome_legivel,
                           bg=T.BG_CARD, fg=T.FG_MUTED,
                           font=T.FONT_LABEL, anchor='w')
            lbl.grid(row=row, column=0, sticky='w', pady=(0, 1))
            self.lbls_test[chave] = lbl
            ttk.Entry(form_t, textvariable=self.vars_test[chave],
                      font=T.FONT_MONO, width=10
                     ).grid(row=row, column=1, sticky='ew',
                            padx=(8, 0), pady=(0, 1))

        ttk.Button(self.card_testar, text='Classificar Amostra  >',
                   style='Primary.TButton',
                   command=self._classificar_amostra_pd
                  ).pack(fill='x', padx=14, pady=(8, 6))

        # Divisor sutil
        self.sep_pred = separador(self.card_testar, padx=14, pady=4)

        # Predição (dentro do mesmo card)
        self.lbl_pred_pd = tk.Label(self.card_testar, text='—',
                                    bg=T.BG_CARD, fg=T.FG_DIM,
                                    font=T.FONT_VALUE_BIG, anchor='w')
        self.lbl_pred_pd.pack(fill='x', padx=14, pady=(2, 1))
        
        self.lbl_pred_sub_pd = tk.Label(
            self.card_testar, text='aguardando treinamento',
            bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_MONO_SM,
            anchor='w', justify='left', wraplength=280)
        self.lbl_pred_sub_pd.pack(fill='x', padx=14, pady=(0, 2))
        
        self.lbl_equacao = tk.Label(
            self.card_testar, text='—',
            bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_MONO_SM,
            anchor='w', justify='left', wraplength=280)
        self.lbl_equacao.pack(fill='x', padx=14, pady=(0, 6))

        # Botão de memória de cálculo (no rodapé do card de classificação)
        self.btn_memoria_pd = ttk.Button(
            self.card_testar, text='Abrir memoria de calculo  >',
            style='Primary.TButton',
            command=self._abrir_memoria_calculo_pd
        )

        # --- Card: amostras do dataset (tabela compacta de 3 linhas) ---
        self.card_dataset = Card(self._wrap_esq, titulo='amostras do dataset')
        frame_tree = tk.Frame(self.card_dataset, bg=T.BG_CARD)
        frame_tree.pack(fill='x', padx=14, pady=(0, 8))
        frame_tree.columnconfigure(0, weight=1)

        cols = ('idx', 'cls', 'sc', 'sl', 'pc', 'pl')
        self.tree_dataset = ttk.Treeview(
            frame_tree, columns=cols, show='headings', height=10,
            selectmode='browse')
        self.tree_dataset.heading('idx', text='#')
        self.tree_dataset.heading('cls', text='Classe')
        self.tree_dataset.heading('sc',  text='S.Cmp')
        self.tree_dataset.heading('sl',  text='S.Lrg')
        self.tree_dataset.heading('pc',  text='P.Cmp')
        self.tree_dataset.heading('pl',  text='P.Lrg')
        self.tree_dataset.column('idx', width=30,  anchor='e', stretch=False)
        self.tree_dataset.column('cls', width=70,  anchor='w')
        self.tree_dataset.column('sc',  width=42,  anchor='e', stretch=False)
        self.tree_dataset.column('sl',  width=42,  anchor='e', stretch=False)
        self.tree_dataset.column('pc',  width=42,  anchor='e', stretch=False)
        self.tree_dataset.column('pl',  width=42,  anchor='e', stretch=False)
        self.tree_dataset.grid(row=0, column=0, sticky='nsew')

        scroll_tree = ttk.Scrollbar(
            frame_tree, orient='vertical',
            command=self.tree_dataset.yview)
        self.tree_dataset.configure(yscrollcommand=scroll_tree.set)
        scroll_tree.grid(row=0, column=1, sticky='ns')

        self.tree_dataset.tag_configure(
            'setosa',     foreground=T.DATA_BLUE)
        self.tree_dataset.tag_configure(
            'versicolor', foreground=T.DATA_MINT)
        self.tree_dataset.tag_configure(
            'virginica',  foreground=T.DATA_CORAL)
        self.tree_dataset.bind(
            '<<TreeviewSelect>>', self._ao_selecionar_amostra_dataset)

        # Layout inicial
        self._relayoutar_controles()

    def _relayoutar_controles(self):
        """Empacota os cartoes na ordem certa conforme o algoritmo selecionado."""
        for w in (self.card_algo, self.frame_config_row, self.card_par, self.card_attr,
                  self.card_split, self.card_hiper, self.card_testar, self.card_dataset):
            w.pack_forget()
        self.btn_memoria_pd.pack_forget()

        algo = self.var_algo.get()
        GAP = (T.GAP_SM, 0)   # espacamento compacto entre cards
        self.card_algo.pack(fill='x')

        show_par = algo not in ('xor', 'delta_ova')
        show_attr = algo != 'xor'

        if show_par or show_attr:
            self.frame_config_row.pack(fill='x', pady=GAP)
            self.card_par.pack_forget()
            self.card_attr.pack_forget()
            
            if show_par and show_attr:
                self.card_par.pack(side='left', fill='both', expand=True, padx=(0, 4))
                self.card_attr.pack(side='right', fill='both', expand=True, padx=(4, 0))
            elif show_par:
                self.card_par.pack(fill='both', expand=True)
            elif show_attr:
                self.card_attr.pack(fill='both', expand=True)

        if algo != 'xor':
            self.card_split.pack(fill='x', pady=GAP)

        self.card_hiper.pack(fill='x', pady=GAP)

        if algo != 'xor':
            self.card_testar.pack(fill='x', pady=GAP)
            self.card_dataset.pack(fill='x', pady=GAP)

        if algo in ('perceptron', 'delta') and self.var_attr.get() != 'todas':
            self.btn_memoria_pd.pack(fill='x', padx=14, pady=(2, 10))

    # -----------------------------------------------------------------------
    # Coluna direita — figura + metricas
    # -----------------------------------------------------------------------
    def _coluna_visualizacao(self):
        wrap = tk.Frame(self, bg=T.BG)
        wrap.grid(row=0, column=1, sticky='nsew',
                  padx=(T.GAP, T.PAD_PAGE), pady=T.PAD_PAGE)
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=3)
        wrap.rowconfigure(1, weight=2)

        # --- Figura matplotlib (2 subplots) ---
        painel = tk.Frame(wrap, bg=T.BG_PANEL,
                          highlightthickness=1,
                          highlightbackground=T.BORDER,
                          highlightcolor=T.BORDER)
        painel.grid(row=0, column=0, sticky='nsew')
        painel.columnconfigure(0, weight=1)
        painel.rowconfigure(0, weight=1)
        painel.rowconfigure(1, weight=0)   # toolbar — altura fixa

        self.figura = Figure(figsize=(9.5, 4.2), dpi=100,
                             facecolor=T.BG_CARD)
        self.ax_sc = None
        self.ax_cv = None
        self._montar_subplots(com_convergencia=True)

        self.canvas = FigureCanvasTkAgg(self.figura, master=painel)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew',
                                         padx=8, pady=(8, 2))

        # Barra de navegacao: zoom, pan, home, salvar
        self.toolbar = NavigationToolbar2Tk(self.canvas, painel,
                                            pack_toolbar=False)
        self.toolbar.update()
        self._estilizar_toolbar(self.toolbar)
        self.toolbar.grid(row=1, column=0, sticky='ew', padx=6, pady=(0, 6))

        # --- Painel inferior: metricas + analise ---
        inferior = tk.Frame(wrap, bg=T.BG)
        inferior.grid(row=1, column=0, sticky='nsew', pady=(14, 0))
        inferior.columnconfigure(0, weight=1)
        inferior.columnconfigure(1, weight=2)
        inferior.rowconfigure(0, weight=1)

        # Coluna de metricas
        col_m = tk.Frame(inferior, bg=T.BG)
        col_m.grid(row=0, column=0, sticky='nsew', padx=(0, 14))
        col_m.columnconfigure(0, weight=1)

        self.metric_epocas = MetricBlock(col_m, 'epocas treinadas', '—')
        self.metric_epocas.grid(row=0, column=0, sticky='ew')
        self.metric_conv   = MetricBlock(col_m, 'convergencia', '—')
        self.metric_conv.grid(row=1, column=0, sticky='ew', pady=(T.GAP, 0))

        # Cartao de analise textual
        card = Card(inferior, titulo='analise')
        card.grid(row=0, column=1, sticky='nsew')
        self.txt_analise = tk.Text(
            card, height=9, wrap='word',
            bg=T.BG_CARD, fg=T.FG,
            font=T.FONT_BODY,
            relief='flat', borderwidth=0,
            highlightthickness=0,
            padx=14, pady=2,
            spacing1=2, spacing3=4,
        )
        self.txt_analise.pack(fill='both', expand=True, padx=14, pady=(2, 14))
        self.txt_analise.tag_configure('hl',   foreground=T.ACCENT_DEEP,
                                       font=T.FONT_TEXT_HL)
        self.txt_analise.tag_configure('ok',   foreground=T.SUCCESS,
                                       font=T.FONT_TEXT_HL)
        self.txt_analise.tag_configure('err',  foreground=T.DANGER,
                                       font=T.FONT_TEXT_HL)
        self.txt_analise.tag_configure('mono', foreground=T.FG,
                                       font=T.FONT_MONO)
        self.txt_analise.configure(state='disabled')

    # -----------------------------------------------------------------------
    # Dados
    # -----------------------------------------------------------------------
    def _carregar_dados(self):
        caminho = CAMINHOS_DADOS[self.var_dataset.get()]
        if not os.path.exists(caminho):
            self.lbl_status.configure(
                text=f'Dados nao encontrados: {os.path.basename(caminho)}',
                fg=T.DANGER)
            return
        self.dados = carregar_dados_iris(caminho)
        
        try:
            prop = float(self.var_prop_treino.get())
            if not (0.1 <= prop <= 0.9):
                prop = 0.7
        except ValueError:
            prop = 0.7

        try:
            sem_str = self.var_semente.get().strip()
            sem = int(sem_str) if sem_str else None
        except ValueError:
            sem = 42

        self.dados_treino, self.dados_teste = split_estratificado(
            self.dados, proporcao_treino=prop, semente=sem)
        self._popular_tree_dataset()

    def _popular_tree_dataset(self):
        """Recarrega o Treeview com as 150 amostras do dataset atual."""
        # Limpa entradas atuais
        for iid in self.tree_dataset.get_children():
            self.tree_dataset.delete(iid)
        # Preenche
        for i, d in enumerate(self.dados):
            atr = d['atributos']
            classe = d['classe']
            self.tree_dataset.insert(
                '', 'end', iid=str(i),
                values=(i + 1, classe,
                        f'{atr[0]:.2f}', f'{atr[1]:.2f}',
                        f'{atr[2]:.2f}', f'{atr[3]:.2f}'),
                tags=(classe,))

    def _ao_selecionar_amostra_dataset(self, event=None):
        """Quando o usuario clica numa linha do Treeview, preenche os 4 campos."""
        sel = self.tree_dataset.selection()
        if not sel:
            return
        try:
            idx = int(sel[0])
        except (ValueError, IndexError):
            return
        if idx < 0 or idx >= len(self.dados):
            return
        atr = self.dados[idx]['atributos']
        self.vars_test['s_comp'].set(f'{atr[0]:.2f}')
        self.vars_test['s_larg'].set(f'{atr[1]:.2f}')
        self.vars_test['p_comp'].set(f'{atr[2]:.2f}')
        self.vars_test['p_larg'].set(f'{atr[3]:.2f}')

    # -----------------------------------------------------------------------
    # Eventos dos controles
    # -----------------------------------------------------------------------
    def _ao_trocar_algo(self):
        algo = self.var_algo.get()
        # Atualizar defaults de hiperparametros
        d = ALGO_DEFAULTS[algo]
        self.var_taxa.set(d['taxa'])
        self.var_epocas.set(d['epocas'])
        # Reexibir cards adequados
        self._relayoutar_controles()
        # Reconstruir figura: Perceptron nao mostra curva de convergencia
        self._montar_subplots(com_convergencia=(algo != 'perceptron'))
        # Resetar estado e redesenhar
        self.w = None
        self.pesos_ova = None
        self.historico = []
        self.historico_ova = None
        self._resetar_metricas()
        if algo != 'xor':
            self.lbl_pred_pd.configure(text='—', fg=T.FG_DIM)
            self.lbl_pred_sub_pd.configure(text='aguardando treinamento')
            self.lbl_equacao.configure(
                text='—  (treine o modelo)', fg=T.FG_MUTED)
        self._desenhar_inicial()
        self._escrever_analise_inicial()

    def _ao_mudar_config(self):
        """Ao mudar par ou atributos: limpa modelo e redesenha scatter."""
        self.w = None
        self.pesos_ova = None
        self.historico = []
        self.historico_ova = None
        self._resetar_metricas()
        self._atualizar_labels_testar()
        self.lbl_pred_pd.configure(text='—', fg=T.FG_DIM)
        self.lbl_pred_sub_pd.configure(text='aguardando treinamento')
        self.lbl_equacao.configure(text='—  (treine o modelo)', fg=T.FG_MUTED)
        self._relayoutar_controles()
        self._desenhar_inicial()
        self._escrever_analise_inicial()

    def _ao_selecionar_dataset(self, event=None):
        self.var_dataset.set(_DS_CHAVE[self._combo_dataset.get()])
        self._ao_mudar_dataset()

    def _ao_mudar_dataset(self):
        """Ao trocar entre v1 e v2: recarrega dados e limpa estado."""
        self.w = None
        self.pesos_ova = None
        self.historico = []
        self.historico_ova = None
        self._resetar_metricas()
        self.lbl_status.configure(text='Modelo nao treinado.', fg=T.FG_MUTED)
        if self.var_algo.get() != 'xor':
            self.lbl_pred_pd.configure(text='—', fg=T.FG_DIM)
            self.lbl_pred_sub_pd.configure(text='aguardando treinamento')
            self.lbl_equacao.configure(
                text='—  (treine o modelo)', fg=T.FG_MUTED)
        self._carregar_dados()
        self._desenhar_inicial()
        self._escrever_analise_inicial()

    # -----------------------------------------------------------------------
    # Classificacao manual
    # -----------------------------------------------------------------------
    def _abrir_memoria_calculo_pd(self):
        if self.w is None or self._classe_pos is None:
            self.lbl_status.configure(
                text='Treine o modelo primeiro.', fg=T.DANGER)
            return
        cfg = CONF_ATTR[self.var_attr.get()]
        indices = cfg['indices']
        # Le os 4 campos e extrai os 2 atributos efetivamente usados
        amostra4 = self._ler_amostra_4features()
        if amostra4 is None:
            amostra = [4.5, 1.5]
        else:
            amostra = [amostra4[indices[0]], amostra4[indices[1]]]
        JanelaMemoriaCalculoPD(
            self,
            algo=self.var_algo.get(),
            w=self.w,
            classe_pos=self._classe_pos,
            classe_neg=self._classe_neg,
            eixos=(cfg['eixo_x'].replace(' (cm)', ''),
                   cfg['eixo_y'].replace(' (cm)', '')),
            taxa=self.var_taxa.get(),
            epocas_treinadas=self.n_epocas_treinadas,
            historico=self.historico,
            amostra=amostra,
        )

    def _atualizar_labels_testar(self):
        """Destaca em ambar os 2 atributos usados pelo modelo treinado."""
        cfg = CONF_ATTR[self.var_attr.get()]
        ativos = set(cfg['indices'])   # ex: {2, 3} para petalas
        for chave, _nome, idx in FEATURES:
            cor = T.ACCENT_DEEP if idx in ativos else T.FG_DIM
            self.lbls_test[chave].configure(fg=cor)

    def _ler_amostra_4features(self):
        """Le os 4 campos da classificacao manual. Retorna lista [s_c, s_l, p_c, p_l]
        ou None em caso de erro."""
        try:
            return [float(self.vars_test[k].get().replace(',', '.'))
                    for k, _n, _i in FEATURES]
        except ValueError:
            return None

    def _classificar_amostra_pd(self):
        algo = self.var_algo.get()
        # Algum modelo treinado?
        treinado = (self.w is not None) or (self.pesos_ova is not None)
        if not treinado:
            self.lbl_pred_pd.configure(text='—', fg=T.DANGER)
            self.lbl_pred_sub_pd.configure(text='treine o modelo primeiro')
            return

        amostra4 = self._ler_amostra_4features()
        if amostra4 is None:
            self.lbl_pred_pd.configure(text='—', fg=T.DANGER)
            self.lbl_pred_sub_pd.configure(
                text='valores invalidos (use numeros decimais)')
            return

        indices = CONF_ATTR[self.var_attr.get()]['indices']
        x_sel = [amostra4[i] for i in indices]

        if algo == 'delta_ova' and self.pesos_ova is not None:
            # OvA: argmax dos 3 nets
            pred, nets = predizer_delta_ova(x_sel, self.pesos_ova)
            cor = CORES_CLASSE[pred]
            self.lbl_pred_pd.configure(text=pred.upper(), fg=cor)
            linhas_nets = '\n'.join(
                f'  net_{c[:3]}  =  {nets[c]:+.4f}'
                + ('   ← vencedor' if c == pred else '')
                for c in sorted(nets))
            self.lbl_pred_sub_pd.configure(
                text=f'argmax dos 3 classificadores:\n{linhas_nets}')
        else:
            # Binario
            net = self.w[0] + sum(wi * xi for wi, xi in zip(self.w[1:], x_sel))
            pred = self._classe_pos if net > 0 else self._classe_neg
            cor = CORES_CLASSE[pred]
            self.lbl_pred_pd.configure(text=pred.upper(), fg=cor)
            self.lbl_pred_sub_pd.configure(
                text=f'net = {net:+.4f}\n'
                     f'{self._classe_pos} se net > 0  |  '
                     f'{self._classe_neg} se net <= 0')

    # -----------------------------------------------------------------------
    # Treinamento
    # -----------------------------------------------------------------------
    def _treinar(self):
        try:
            taxa   = float(self.var_taxa.get())
            epocas = int(self.var_epocas.get())
            if taxa <= 0 or epocas <= 0:
                raise ValueError
        except ValueError:
            self.lbl_status.configure(
                text='Erro: taxa > 0 e epocas > 0 (inteiro).',
                fg=T.DANGER)
            return

        self._carregar_dados()
        if not self.dados:
            return

        algo = self.var_algo.get()
        # Limpa OvA antes de cada treino para nao confundir estado
        self.pesos_ova = None
        self.historico_ova = None

        if algo == 'xor':
            self._treinar_xor(taxa, epocas)
        elif algo == 'perceptron':
            self._treinar_perceptron_iris(taxa, epocas)
        elif algo == 'delta_ova':
            self._treinar_delta_ova_iris(taxa, epocas)
        else:
            self._treinar_delta_iris(taxa, epocas)

        self._desenhar_treinado()
        self._atualizar_metricas()
        self._atualizar_analise()
        self._atualizar_equacao()

    def _treinar_perceptron_iris(self, taxa, epocas):
        par    = self.var_par.get()
        indices = CONF_ATTR[self.var_attr.get()]['indices']
        cp, cn = PAR_POR_MODO[par]

        w, hist, n_ep = treinar_perceptron(
            self.dados_treino, cp, cn, indices, taxa, epocas)

        self.w = w
        self.historico = hist
        self.n_epocas_treinadas = n_ep
        self.convergiu = hist[-1] == 0
        self.acc_teste = acuracia_binaria_perceptron(
            self.dados_teste, w, cp, cn, indices)
        self._classe_pos = cp
        self._classe_neg = cn
        self._indices    = indices

        status = (f'Convergiu em {n_ep} epocas.' if self.convergiu
                  else f'Limite {epocas} epocas (nao convergiu).')
        self.lbl_status.configure(
            text=status,
            fg=T.SUCCESS if self.convergiu else T.DANGER)

    def _treinar_delta_iris(self, taxa, epocas):
        par    = self.var_par.get()
        indices = CONF_ATTR[self.var_attr.get()]['indices']
        cp, cn = PAR_POR_MODO[par]

        w, hist, n_ep = treinar_delta_iris(
            self.dados_treino, cp, cn, indices, taxa, epocas)

        self.w = w
        self.historico = hist
        self.n_epocas_treinadas = n_ep
        self.convergiu = True
        self.acc_teste = acuracia_binaria_delta(
            self.dados_teste, w, cp, cn, indices)
        self._classe_pos = cp
        self._classe_neg = cn
        self._indices    = indices

        self.lbl_status.configure(
            text=f'Treinada em {n_ep} epocas. MSE final: {hist[-1]:.4f}.',
            fg=T.FG_MUTED)

    def _treinar_xor(self, taxa, epocas):
        w, hist = treinar_delta_xor(max_epocas=epocas,
                                    taxa_aprendizado=taxa)
        self.w = w
        self.historico = hist
        self.n_epocas_treinadas = epocas
        self.convergiu = False
        self.acc_teste = None

        self.lbl_status.configure(
            text=f'XOR — MSE final: {hist[-1]:.4f} (teorico: 0.2500).',
            fg=T.FG_MUTED)

    def _treinar_delta_ova_iris(self, taxa, epocas):
        """Treina 3 classificadores Delta (Um-Contra-Todos)."""
        indices = CONF_ATTR[self.var_attr.get()]['indices']
        pesos, hist, n_ep = treinar_delta_ova(
            self.dados_treino, indices, taxa, epocas)

        self.pesos_ova = pesos
        self.historico_ova = hist
        self.n_epocas_treinadas = n_ep
        self.convergiu = True
        self.acc_teste = acuracia_delta_ova(
            self.dados_teste, pesos, indices)
        self._indices = indices

        # Para uso na curva de convergencia / display
        mse_finais = [h[-1] for h in hist.values()]
        mse_medio = sum(mse_finais) / len(mse_finais)
        self.lbl_status.configure(
            text=f'Treinado em {n_ep} epocas. MSE medio final: {mse_medio:.4f}.',
            fg=T.FG_MUTED)

    # -----------------------------------------------------------------------
    # Equacao treinada (exibida no card de predicao)
    # -----------------------------------------------------------------------
    def _atualizar_equacao(self):
        algo = self.var_algo.get()
        if algo == 'xor':
            return
        if algo == 'delta_ova' and self.pesos_ova is not None:
            partes = []
            for classe in sorted(self.pesos_ova):
                w = self.pesos_ova[classe]
                if len(w) == 3:
                    partes.append(
                        f'{classe[:3]}:  '
                        f'{w[0]:+.3f}  {w[1]:+.3f}·x1  {w[2]:+.3f}·x2  =  0')
                elif len(w) == 5:
                    partes.append(
                        f'{classe[:3]}:  '
                        f'{w[0]:+.3f}  {w[1]:+.3f}·x1  {w[2]:+.3f}·x2  {w[3]:+.3f}·x3  {w[4]:+.3f}·x4  =  0')
            self.lbl_equacao.configure(
                text='\n'.join(partes), fg=T.FG)
        elif self.w is not None:
            if len(self.w) == 3:
                w0, w1, w2 = self.w[0], self.w[1], self.w[2]
                texto = (f'{w0:+.4f}  {w1:+.4f}·x1  {w2:+.4f}·x2  =  0\n\n'
                         f'(decisao: classe positiva se net > 0)')
            elif len(self.w) == 5:
                w0, w1, w2, w3, w4 = self.w[0], self.w[1], self.w[2], self.w[3], self.w[4]
                texto = (f'{w0:+.4f}  {w1:+.4f}·x1  {w2:+.4f}·x2  {w3:+.4f}·x3  {w4:+.4f}·x4  =  0\n\n'
                         f'(decisao: classe positiva se net > 0)')
            else:
                texto = 'Vetor de pesos de tamanho inesperado'
            self.lbl_equacao.configure(text=texto, fg=T.FG)
        else:
            self.lbl_equacao.configure(
                text='—  (treine o modelo)', fg=T.FG_MUTED)

    # -----------------------------------------------------------------------
    # Graficos
    # -----------------------------------------------------------------------
    def _montar_subplots(self, com_convergencia):
        """Reconstroi a figura com 1 ou 2 subplots conforme o algoritmo.

        - com_convergencia=False  →  apenas scatter (Perceptron)
        - com_convergencia=True   →  scatter + curva de convergencia (Delta/OvA/XOR)
        """
        self.figura.clear()
        if com_convergencia:
            gs = self.figura.add_gridspec(
                1, 2, width_ratios=[5, 3],
                left=0.07, right=0.97,
                bottom=0.15, top=0.90,
                wspace=0.35,
            )
            self.ax_sc = self.figura.add_subplot(gs[0])
            self.ax_cv = self.figura.add_subplot(gs[1])
        else:
            gs = self.figura.add_gridspec(
                1, 1,
                left=0.07, right=0.97,
                bottom=0.15, top=0.90,
            )
            self.ax_sc = self.figura.add_subplot(gs[0])
            self.ax_cv = None

    @staticmethod
    def _estilizar_toolbar(toolbar):
        """Aplica o tema escuro a barra de ferramentas do matplotlib."""
        toolbar.configure(background=T.BG_PANEL)
        for child in toolbar.winfo_children():
            try:
                child.configure(background=T.BG_PANEL,
                                activebackground=T.BG_HOVER,
                                relief='flat', borderwidth=0)
            except Exception:
                pass

    def _estilizar_ax(self, ax):
        """Aplica o tema escuro ao subplot."""
        ax.set_facecolor(T.BG_PANEL)
        ax.tick_params(colors=T.FG_MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(T.BORDER)
        ax.grid(color=T.BORDER, linewidth=0.5, alpha=0.6)
        ax.title.set_color(T.FG)
        ax.xaxis.label.set_color(T.FG_MUTED)
        ax.yaxis.label.set_color(T.FG_MUTED)

    def _desenhar_inicial(self):
        """Scatter inicial (sem fronteira) + subplot de convergencia vazio."""
        self._desenhar_scatter(com_fronteira=False)
        if self.ax_cv is not None:
            self._desenhar_conv_vazio()
        self.canvas.draw()

    def _desenhar_treinado(self):
        """Scatter + fronteira de decisao + curva de convergencia."""
        self._desenhar_scatter(com_fronteira=True)
        if self.ax_cv is not None:
            self._desenhar_curva_convergencia()
        self.canvas.draw()

    def _desenhar_scatter(self, com_fronteira=False):
        ax = self.ax_sc
        ax.cla()
        self._estilizar_ax(ax)

        algo = self.var_algo.get()

        if algo == 'xor':
            self._scatter_xor(ax, com_fronteira)
        elif algo == 'delta_ova':
            self._scatter_iris_ova(ax, com_fronteira)
        else:
            self._scatter_iris(ax, com_fronteira)

    def _scatter_iris(self, ax, com_fronteira):
        attr_key = self.var_attr.get()
        indices_plot = [2, 3] if attr_key == 'todas' else CONF_ATTR[attr_key]['indices']
        cfg     = CONF_ATTR[attr_key]
        par     = self.var_par.get()
        cp, cn  = PAR_POR_MODO[par]

        # Plota apenas as duas classes do par selecionado
        for classe in [cp, cn]:
            pts = [d for d in self.dados if d['classe'] == classe]
            if not pts:
                continue
            x1 = [d['atributos'][indices_plot[0]] for d in pts]
            x2 = [d['atributos'][indices_plot[1]] for d in pts]
            ax.scatter(x1, x2,
                       color=CORES_CLASSE[classe],
                       marker=MARCADORES_CLASSE[classe],
                       s=30, alpha=0.75, linewidths=0.4,
                       edgecolors=T.BG, label=classe.capitalize(),
                       zorder=3)

        ax.set_xlabel(cfg['eixo_x'], fontsize=8)
        ax.set_ylabel(cfg['eixo_y'], fontsize=8)

        titulo = f'{cp.capitalize()}  ×  {cn.capitalize()}'
        if attr_key == 'todas':
            titulo += ' (Proj. 2D - Treino 4D)'

        # Fronteira de decisao
        if com_fronteira and self.w is not None and len(self.w) == 3:
            self._plotar_fronteira_2d(ax, self.w, indices_plot)

        ax.set_title(titulo, fontsize=9, pad=6)
        ax.legend(fontsize=7, facecolor=T.BG_PANEL,
                  edgecolor=T.BORDER, labelcolor=T.FG_MUTED,
                  loc='upper left')

    def _scatter_iris_ova(self, ax, com_fronteira):
        """Scatter com as 3 classes + 3 fronteiras lineares (OvA)."""
        attr_key = self.var_attr.get()
        indices_plot = [2, 3] if attr_key == 'todas' else CONF_ATTR[attr_key]['indices']
        cfg     = CONF_ATTR[attr_key]

        # Pontos das 3 classes
        for classe in CLASSES:
            pts = [d for d in self.dados if d['classe'] == classe]
            if not pts:
                continue
            x1 = [d['atributos'][indices_plot[0]] for d in pts]
            x2 = [d['atributos'][indices_plot[1]] for d in pts]
            ax.scatter(x1, x2,
                       color=CORES_CLASSE[classe],
                       marker=MARCADORES_CLASSE[classe],
                       s=30, alpha=0.75, linewidths=0.4,
                       edgecolors=T.BG, label=classe.capitalize(),
                       zorder=3)

        ax.set_xlabel(cfg['eixo_x'], fontsize=8)
        ax.set_ylabel(cfg['eixo_y'], fontsize=8)
        titulo_ova = 'OvA  ·  3 classificadores Delta'
        if attr_key == 'todas':
            titulo_ova += ' (Proj. 2D - Treino 4D)'
        ax.set_title(titulo_ova, fontsize=9, pad=6)

        # 3 fronteiras de decisao — uma por classificador
        if com_fronteira and self.pesos_ova is not None:
            primeira_classe = next(iter(self.pesos_ova))
            if len(self.pesos_ova[primeira_classe]) == 3:
                todos_x1 = [d['atributos'][indices_plot[0]] for d in self.dados]
                x1_min = min(todos_x1) - 0.5
                x1_max = max(todos_x1) + 0.5
                n = 200
                step = (x1_max - x1_min) / (n - 1)
                x1_pts = [x1_min + k * step for k in range(n)]

                for classe in sorted(self.pesos_ova):
                    w = self.pesos_ova[classe]
                    w0, w1, w2 = w[0], w[1], w[2]
                    cor = CORES_CLASSE[classe]
                    if abs(w2) > 1e-8:
                        x2_pts = [(-w0 - w1 * x) / w2 for x in x1_pts]
                        ax.plot(x1_pts, x2_pts, color=cor, ls='--',
                                lw=1.4, alpha=0.85,
                                label=f'fronteira {classe[:3]}', zorder=5)
                    elif abs(w1) > 1e-8:
                        ax.axvline(-w0 / w1, color=cor, ls='--',
                                   lw=1.4, alpha=0.85,
                                   label=f'fronteira {classe[:3]}', zorder=5)

        ax.legend(fontsize=7, facecolor=T.BG_PANEL,
                  edgecolor=T.BORDER, labelcolor=T.FG_MUTED,
                  loc='best')

    def _scatter_xor(self, ax, com_fronteira):
        """Scatter dos 4 pontos XOR."""
        padroes = [
            (0.0, 0.0, 0),
            (0.0, 1.0, 1),
            (1.0, 0.0, 1),
            (1.0, 1.0, 0),
        ]
        cores_xor = {0: T.DATA_BLUE, 1: T.DATA_CORAL}
        marcadores_xor = {0: 'o', 1: '^'}
        labels = {0: 'Classe 0  {(0,0),(1,1)}', 1: 'Classe 1  {(0,1),(1,0)}'}
        plotados = set()

        for x1, x2, cl in padroes:
            lbl = labels[cl] if cl not in plotados else ''
            ax.scatter(x1, x2,
                       color=cores_xor[cl], marker=marcadores_xor[cl],
                       s=120, zorder=4, linewidths=0.6,
                       edgecolors=T.FG, label=lbl)
            ax.annotate(f'({int(x1)},{int(x2)})→{cl}',
                        (x1, x2), textcoords='offset points',
                        xytext=(8, 6), fontsize=7, color=T.FG_MUTED)
            plotados.add(cl)

        ax.set_xlim(-0.4, 1.4)
        ax.set_ylim(-0.4, 1.4)
        ax.set_xlabel('x₁', fontsize=9)
        ax.set_ylabel('x₂', fontsize=9)
        ax.set_title('XOR  —  Problema Nao Separavel', fontsize=9, pad=6)

        if com_fronteira and self.w is not None:
            w0, w1, w2 = self.w
            n = 100
            step = 1.8 / (n - 1)
            x1_pts = [-0.4 + k * step for k in range(n)]
            if abs(w2) > 1e-9:
                x2_pts = [(-w0 - w1 * x) / w2 for x in x1_pts]
                ax.plot(x1_pts, x2_pts, color=T.ACCENT, ls='--',
                        lw=1.5, label='Fronteira (linear)', zorder=5)
            elif abs(w1) > 1e-9:
                ax.axvline(-w0 / w1, color=T.ACCENT, ls='--',
                           lw=1.5, label='Fronteira', zorder=5)

        ax.legend(fontsize=7, facecolor=T.BG_PANEL,
                  edgecolor=T.BORDER, labelcolor=T.FG_MUTED)

    def _plotar_fronteira_2d(self, ax, w, indices):
        """Desenha a reta de decisao w0 + w1*x1 + w2*x2 = 0."""
        if len(w) != 3:
            return
        todos_x1 = [d['atributos'][indices[0]] for d in self.dados]
        x1_min = min(todos_x1) - 0.5
        x1_max = max(todos_x1) + 0.5
        n = 200
        step = (x1_max - x1_min) / (n - 1)
        x1_pts = [x1_min + k * step for k in range(n)]

        w0, w1, w2 = w[0], w[1], w[2]
        if abs(w2) > 1e-8:
            x2_pts = [(-w0 - w1 * x) / w2 for x in x1_pts]
            ax.plot(x1_pts, x2_pts, color=T.ACCENT, ls='--',
                    lw=1.5, label='Fronteira', zorder=5)
        elif abs(w1) > 1e-8:
            ax.axvline(-w0 / w1, color=T.ACCENT, ls='--',
                       lw=1.5, label='Fronteira', zorder=5)

    def _desenhar_conv_vazio(self):
        ax = self.ax_cv
        ax.cla()
        self._estilizar_ax(ax)
        ax.set_title('Convergencia', fontsize=9, pad=6)
        ax.text(0.5, 0.5, 'Treine o modelo\npara ver a curva',
                transform=ax.transAxes,
                ha='center', va='center',
                fontsize=8, color=T.FG_DIM)

    def _desenhar_curva_convergencia(self):
        ax = self.ax_cv
        ax.cla()
        self._estilizar_ax(ax)

        algo = self.var_algo.get()

        if algo == 'delta_ova':
            if not self.historico_ova:
                self._desenhar_conv_vazio()
                return
            # 3 curvas — uma por classificador
            for classe in sorted(self.historico_ova):
                hist = self.historico_ova[classe]
                epocas = list(range(1, len(hist) + 1))
                ax.plot(epocas, hist, color=CORES_CLASSE[classe],
                        lw=1.4, alpha=0.85,
                        label=classe[:3], zorder=3)
            ax.set_ylabel('MSE', fontsize=7)
            ax.set_title('Delta OvA — MSE por classificador',
                         fontsize=9, pad=6)
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=7, facecolor=T.BG_PANEL,
                      edgecolor=T.BORDER, labelcolor=T.FG_MUTED,
                      loc='upper right')
            ax.set_xlabel('Epoca', fontsize=7)
            ax.tick_params(axis='both', labelsize=7)
            return

        if not self.historico:
            self._desenhar_conv_vazio()
            return

        epocas = list(range(1, len(self.historico) + 1))

        if algo == 'perceptron':
            cor = T.SUCCESS if self.convergiu else T.DANGER
            ax.plot(epocas, self.historico, color=cor, lw=1.5, zorder=3)
            ax.set_ylabel('Erros / Epoca', fontsize=7)
            ax.set_title('Perceptron — Erros', fontsize=9, pad=6)
            ax.set_ylim(bottom=0)
        else:
            # Regra Delta binaria ou XOR
            cor = T.DATA_MINT if algo == 'delta' else T.DATA_CORAL
            ax.plot(epocas, self.historico, color=cor, lw=1.5, zorder=3)
            ax.set_ylabel('MSE', fontsize=7)
            titulo = ('Regra Delta — MSE' if algo == 'delta'
                      else 'XOR — MSE (nao converge)')
            ax.set_title(titulo, fontsize=9, pad=6)
            ax.set_ylim(bottom=0)

            ax.axhline(self.historico[-1], color=T.BORDER_HARD,
                       ls=':', lw=0.8, zorder=2)

        ax.set_xlabel('Epoca', fontsize=7)
        ax.tick_params(axis='both', labelsize=7)

    # -----------------------------------------------------------------------
    # Metricas e analise
    # -----------------------------------------------------------------------
    def _resetar_metricas(self):
        self.metric_epocas.set('—')
        self.metric_conv.set('—')

    def _atualizar_metricas(self):
        algo = self.var_algo.get()

        # Epocas
        max_ep = int(self.var_epocas.get())
        self.metric_epocas.set(f'{self.n_epocas_treinadas} / {max_ep}')

        # Convergencia
        if algo == 'perceptron':
            erros = self.historico[-1] if self.historico else '—'
            self.metric_conv.set(
                'Convergiu' if self.convergiu else f'Erros: {erros}',
                T.SUCCESS if self.convergiu else T.DANGER)
        elif algo == 'delta_ova':
            if self.historico_ova:
                mse_medio = sum(h[-1] for h in self.historico_ova.values()) \
                            / len(self.historico_ova)
                self.metric_conv.set(f'MSE medio: {mse_medio:.4f}', T.FG)
            else:
                self.metric_conv.set('—')
        else:
            mse = self.historico[-1] if self.historico else 0.0
            self.metric_conv.set(f'{mse:.4f}', T.FG)

    def _txt_set(self, texto_com_tags):
        """Escreve no widget de analise com suporte a marcacoes.

        Formato de entrada: lista de (texto, tag_ou_None).
        """
        self.txt_analise.configure(state='normal')
        self.txt_analise.delete('1.0', 'end')
        for trecho, tag in texto_com_tags:
            if tag:
                self.txt_analise.insert('end', trecho, tag)
            else:
                self.txt_analise.insert('end', trecho)
        self.txt_analise.configure(state='disabled')

    def _escrever_analise_inicial(self):
        algo = self.var_algo.get()
        if algo == 'xor':
            self._txt_set([
                ('Problema XOR\n\n', 'hl'),
                ('O XOR e o exemplo classico de funcao booleana NAO linearmente separavel. '
                 'Nenhuma reta (hiperplano) consegue dividir os padroes\n'
                 '  ', None),
                ('{(0,0), (1,1)} → 0', 'mono'),
                ('  de  ', None),
                ('{(0,1), (1,0)} → 1', 'mono'),
                ('\n\nTreine com a Regra Delta e observe que o MSE '
                 'converge a ', None),
                ('0.2500', 'err'),
                (' — o minimo teorico de um classificador linear.', None),
            ])
        elif algo == 'delta_ova':
            self._txt_set([
                ('Regra Delta — Um Contra Todos (OvA)\n\n', 'hl'),
                ('Treina ', None), ('3 classificadores binarios', 'hl'),
                (' (um por classe vs resto). A predicao multiclasse usa ', None),
                ('argmax', 'mono'),
                (' dos 3 nets: a classe cujo classificador retorna o maior valor vence.\n\n'
                 'Clique em ', None),
                ('Treinar  >', 'hl'),
                (' para ajustar os 3 vetores de pesos.', None),
            ])
        else:
            par = self.var_par.get()
            cp, cn = PAR_POR_MODO[par]
            nome_algo = 'Perceptron' if algo == 'perceptron' else 'Regra Delta'
            self._txt_set([
                (f'{nome_algo} — {cp.capitalize()} × {cn.capitalize()}\n\n', 'hl'),
                ('Clique em ', None),
                ('Treinar  >', 'hl'),
                (' para ajustar os pesos pelo algoritmo selecionado.\n\n'
                 'Apos o treinamento, a fronteira de decisao sera exibida '
                 'no grafico e a curva de convergencia aparecera a direita.', None),
            ])

    def _atualizar_analise(self):
        algo = self.var_algo.get()
        if algo == 'delta_ova':
            if not self.historico_ova:
                return
            self._analise_delta_ova()
            return
        if not self.historico:
            return
        if algo == 'perceptron':
            self._analise_perceptron()
        elif algo == 'delta':
            self._analise_delta_iris()
        else:
            self._analise_xor()

    def _analise_perceptron(self):
        par = self.var_par.get()
        cp, cn = PAR_POR_MODO[par]
        n_ep = self.n_epocas_treinadas
        
        if len(self.w) == 3:
            w0, w1, w2 = self.w[0], self.w[1], self.w[2]
            eq_concreta = f'{w0:+.4f}  {w1:+.4f}·x1  {w2:+.4f}·x2  =  0'
            form_text = 'w0 + w1·x1 + w2·x2 = 0\n'
        else:
            w0, w1, w2, w3, w4 = self.w[0], self.w[1], self.w[2], self.w[3], self.w[4]
            eq_concreta = f'{w0:+.4f}  {w1:+.4f}·x1  {w2:+.4f}·x2  {w3:+.4f}·x3  {w4:+.4f}·x4  =  0'
            form_text = 'w0 + w1·x1 + w2·x2 + w3·x3 + w4·x4 = 0\n'

        if self.convergiu:
            self._txt_set([
                ('Perceptron convergiu\n\n', 'ok'),
                (f'Convergiu em ', None),
                (f'{n_ep}', 'hl'), (' epocas, p = ', None),
                (self.var_taxa.get(), 'mono'),
                (f'.  Acuracia teste: ', None),
                (f'{self.acc_teste*100:.2f}%', 'ok'), ('.\n\n', None),
                ('Fronteira: ', None),
                (form_text, 'mono'),
                (f'  {eq_concreta}\n\n', 'mono'),
                (f'{cp.capitalize()} × {cn.capitalize()}', 'hl'),
                (' sao linearmente separaveis nestes atributos.', None),
            ])
        else:
            self._txt_set([
                ('Perceptron nao convergiu\n\n', 'err'),
                (f'Limite de ', None), (f'{n_ep}', 'hl'),
                (' epocas atingido (p = ', None),
                (self.var_taxa.get(), 'mono'), (').\n\n', None),
                ('Ultima fronteira: ', None),
                (f'{eq_concreta}\n\n', 'mono'),
                (f'{cp.capitalize()} × {cn.capitalize()}', 'hl'),
                (' tem sobreposicao — nenhuma reta separa perfeitamente. '
                 'Tente Petalas [2,3] ou outro par.', None),
            ])

    def _analise_delta_iris(self):
        par = self.var_par.get()
        cp, cn = PAR_POR_MODO[par]
        mse_ini = self.historico[0] if self.historico else 0
        mse_fin = self.historico[-1] if self.historico else 0
        reducao = (1 - mse_fin / mse_ini) * 100 if mse_ini > 1e-12 else 0

        if len(self.w) == 3:
            w0, w1, w2 = self.w[0], self.w[1], self.w[2]
            eq_concreta = f'{w0:+.4f}  {w1:+.4f}·x1  {w2:+.4f}·x2  =  0'
            form_text = 'w0 + w1·x1 + w2·x2 = 0\n'
        else:
            w0, w1, w2, w3, w4 = self.w[0], self.w[1], self.w[2], self.w[3], self.w[4]
            eq_concreta = f'{w0:+.4f}  {w1:+.4f}·x1  {w2:+.4f}·x2  {w3:+.4f}·x3  {w4:+.4f}·x4  =  0'
            form_text = 'w0 + w1·x1 + w2·x2 + w3·x3 + w4·x4 = 0\n'

        self._txt_set([
            ('Regra Delta — MSE convergiu\n\n', 'hl'),
            (f'{cp.capitalize()} × {cn.capitalize()}', 'hl'),
            (', p = ', None), (self.var_taxa.get(), 'mono'),
            (f', {self.n_epocas_treinadas} epocas.  ', None),
            ('Acc teste: ', None),
            (f'{self.acc_teste*100:.2f}%', 'ok'), ('.\n\n', None),
            ('MSE: ', None), (f'{mse_ini:.4f}', 'mono'),
            (' → ', None), (f'{mse_fin:.4f}', 'mono'),
            (f'  ({reducao:.1f}% reducao)\n\n', None),
            ('Fronteira: ', None),
            (form_text, 'mono'),
            (f'  {eq_concreta}', 'mono'),
        ])

    def _analise_delta_ova(self):
        cor_acc = (T.SUCCESS if self.acc_teste and self.acc_teste >= 0.9
                   else T.ACCENT)

        partes = [
            ('Regra Delta — OvA (3 classes)\n\n', 'hl'),
            ('3 classificadores', 'hl'),
            (f', p = ', None), (self.var_taxa.get(), 'mono'),
            (f', {self.n_epocas_treinadas} epocas.  Acc teste: ', None),
            (f'{self.acc_teste*100:.2f}%', 'ok' if cor_acc == T.SUCCESS else 'hl'),
            ('.\n\nFronteiras  (w_c^T·x_aug = 0):\n', None),
        ]
        for classe in sorted(self.pesos_ova):
            w = self.pesos_ova[classe]
            if len(w) == 3:
                partes.append(
                    (f'  {classe[:3]}:  '
                     f'{w[0]:+.3f}  {w[1]:+.3f}·x1  {w[2]:+.3f}·x2  = 0\n',
                     'mono'))
            else:
                partes.append(
                    (f'  {classe[:3]}:  '
                     f'{w[0]:+.3f}  {w[1]:+.3f}·x1  {w[2]:+.3f}·x2  {w[3]:+.3f}·x3  {w[4]:+.3f}·x4  = 0\n',
                     'mono'))
        partes.append(('\nPredicao: ', None))
        partes.append(('argmax_c net_c', 'mono'))
        partes.append((
            '.   Versicolor (no meio) tem dificuldade — limitacao do OvA linear.',
            None))

        self._txt_set(partes)

    def _analise_xor(self):
        mse_ini = self.historico[0] if self.historico else 0
        mse_fin = self.historico[-1] if self.historico else 0

        self._txt_set([
            ('XOR — Limite Linear\n\n', 'err'),
            ('MSE inicial: ', None), (f'{mse_ini:.4f}', 'mono'), ('\n', None),
            ('MSE final:   ', None), (f'{mse_fin:.4f}', 'mono'),
            ('  (teorico minimo: ', None), ('0.2500', 'err'), (')\n\n', None),
            ('O problema XOR ', None),
            ('nao e linearmente separavel', 'err'),
            ('. O MSE converge ao minimo possivel de um classificador '
             'linear — exatamente 0.25 — pois os dois grupos estao '
             'simetricamente espalhados em torno do centroide.\n\n'
             'Para resolver o XOR e necessaria uma ', None),
            ('rede neural multicamada', 'hl'),
            (' (MLP com pelo menos 1 camada oculta).', None),
        ])
