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
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from data_loader import carregar_dados_iris, split_estratificado
from perceptron import treinar_perceptron, acuracia_binaria_perceptron
from delta_rule import (treinar_delta_iris, treinar_delta_xor,
                        acuracia_binaria_delta)

from . import theme as T
from .widgets import Card, MetricBlock


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
PROJETO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
CAMINHO_DADOS = os.path.join(PROJETO_ROOT, 'data', 'Iris data.xls')

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
    'xor':        {'taxa': '0.02', 'epocas': '300'},
}


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

        self.var_algo  = tk.StringVar(value='perceptron')
        self.var_par   = tk.StringVar(value='sv')
        self.var_attr  = tk.StringVar(value='petalas')
        self.var_taxa  = tk.StringVar(value='0.03')
        self.var_epocas = tk.StringVar(value='100')

        self.w = None
        self.historico = []
        self.n_epocas_treinadas = 0
        self.acc_teste = None
        self.convergiu = False

        self._construir_layout()
        self._carregar_dados()
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
                            padx=(20, 10), pady=20)
        self._wrap_esq.columnconfigure(0, weight=1)

        # --- Card 1: Algoritmo ---
        self.card_algo = Card(self._wrap_esq, titulo='algoritmo')
        for val, label in [
            ('perceptron', 'Perceptron'),
            ('delta',      'Regra Delta  (Adaline)'),
            ('xor',        'XOR  —  Regra Delta'),
        ]:
            tk.Radiobutton(
                self.card_algo, text=label,
                value=val, variable=self.var_algo,
                bg=T.BG_CARD, fg=T.FG,
                selectcolor=T.BG_HOVER,
                activebackground=T.BG_CARD, activeforeground=T.ACCENT,
                font=T.FONT_BODY, anchor='w',
                borderwidth=0, highlightthickness=0,
                command=self._ao_trocar_algo,
            ).pack(fill='x', padx=14, pady=2)
        tk.Frame(self.card_algo, bg=T.BG_CARD, height=8).pack()

        # --- Card 2: Par de classes (Iris) ---
        self.card_par = Card(self._wrap_esq, titulo='par de classes')
        for chave, rotulo in PAR_ROTULO.items():
            tk.Radiobutton(
                self.card_par, text=rotulo,
                value=chave, variable=self.var_par,
                bg=T.BG_CARD, fg=T.FG,
                selectcolor=T.BG_HOVER,
                activebackground=T.BG_CARD, activeforeground=T.ACCENT,
                font=T.FONT_BODY, anchor='w',
                borderwidth=0, highlightthickness=0,
                command=self._ao_mudar_config,
            ).pack(fill='x', padx=14, pady=2)
        tk.Frame(self.card_par, bg=T.BG_CARD, height=8).pack()

        # --- Card 3: Atributos (Iris) ---
        self.card_attr = Card(self._wrap_esq, titulo='atributos')
        for chave, cfg in CONF_ATTR.items():
            tk.Radiobutton(
                self.card_attr, text=cfg['rotulo_ui'],
                value=chave, variable=self.var_attr,
                bg=T.BG_CARD, fg=T.FG,
                selectcolor=T.BG_HOVER,
                activebackground=T.BG_CARD, activeforeground=T.ACCENT,
                font=T.FONT_BODY, anchor='w',
                borderwidth=0, highlightthickness=0,
                command=self._ao_mudar_config,
            ).pack(fill='x', padx=14, pady=2)
        tk.Frame(self.card_attr, bg=T.BG_CARD, height=8).pack()

        # --- Card 4: Hiperparâmetros ---
        self.card_hiper = Card(self._wrap_esq, titulo='hiperparametros')
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
                ).grid(row=1, column=0, sticky='w', pady=(8, 2))
        ttk.Entry(form, textvariable=self.var_epocas,
                  font=T.FONT_MONO, width=9
                 ).grid(row=1, column=1, sticky='ew', padx=(8, 0))
        tk.Frame(self.card_hiper, bg=T.BG_CARD, height=12).pack()

        # --- Botao Treinar ---
        self._frame_btn = tk.Frame(self._wrap_esq, bg=T.BG)
        ttk.Button(
            self._frame_btn, text='Treinar  >',
            style='Primary.TButton',
            command=self._treinar,
        ).pack(fill='x', ipady=4)
        self.lbl_status = tk.Label(
            self._frame_btn, text='Modelo nao treinado.',
            bg=T.BG, fg=T.FG_MUTED, font=T.FONT_MONO_SM,
            anchor='w', wraplength=280, justify='left',
        )
        self.lbl_status.pack(fill='x', pady=(4, 0))

        # Layout inicial
        self._relayoutar_controles()

    def _relayoutar_controles(self):
        """Empacota os cartoes na ordem certa, ocultando os de Iris no modo XOR."""
        # Remove tudo primeiro
        for w in (self.card_algo, self.card_par, self.card_attr,
                  self.card_hiper, self._frame_btn):
            w.pack_forget()

        # Reempacota em ordem
        self.card_algo.pack(fill='x')

        if self.var_algo.get() != 'xor':
            self.card_par.pack(fill='x', pady=(14, 0))
            self.card_attr.pack(fill='x', pady=(14, 0))

        self.card_hiper.pack(fill='x', pady=(14, 0))
        self._frame_btn.pack(fill='x', pady=(14, 0))

    # -----------------------------------------------------------------------
    # Coluna direita — figura + metricas
    # -----------------------------------------------------------------------
    def _coluna_visualizacao(self):
        wrap = tk.Frame(self, bg=T.BG)
        wrap.grid(row=0, column=1, sticky='nsew', padx=(10, 20), pady=20)
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

        self.figura = Figure(figsize=(9.5, 4.2), dpi=100,
                             facecolor=T.BG_PANEL)
        gs = self.figura.add_gridspec(
            1, 2, width_ratios=[5, 3],
            left=0.07, right=0.97,
            bottom=0.15, top=0.90,
            wspace=0.35,
        )
        self.ax_sc = self.figura.add_subplot(gs[0])
        self.ax_cv = self.figura.add_subplot(gs[1])

        self.canvas = FigureCanvasTkAgg(self.figura, master=painel)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew',
                                         padx=8, pady=8)

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

        self.metric_acc    = MetricBlock(col_m, 'acuracia teste',   '—')
        self.metric_acc.grid(row=0, column=0, sticky='ew')
        self.metric_epocas = MetricBlock(col_m, 'epocas treinadas', '—')
        self.metric_epocas.grid(row=1, column=0, sticky='ew', pady=(10, 0))
        self.metric_conv   = MetricBlock(col_m, 'convergencia', '—')
        self.metric_conv.grid(row=2, column=0, sticky='ew', pady=(10, 0))

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
        self.txt_analise.tag_configure('hl',   foreground=T.ACCENT,
                                       font=('Segoe UI Semibold', 10))
        self.txt_analise.tag_configure('ok',   foreground=T.SUCCESS,
                                       font=('Segoe UI Semibold', 10))
        self.txt_analise.tag_configure('err',  foreground=T.DANGER,
                                       font=('Segoe UI Semibold', 10))
        self.txt_analise.tag_configure('mono', foreground=T.FG,
                                       font=T.FONT_MONO)
        self.txt_analise.configure(state='disabled')

    # -----------------------------------------------------------------------
    # Dados
    # -----------------------------------------------------------------------
    def _carregar_dados(self):
        if not os.path.exists(CAMINHO_DADOS):
            self.lbl_status.configure(
                text=f'Dados nao encontrados: {CAMINHO_DADOS}',
                fg=T.DANGER)
            return
        self.dados = carregar_dados_iris(CAMINHO_DADOS)
        self.dados_treino, self.dados_teste = split_estratificado(
            self.dados, proporcao_treino=0.7, semente=42)

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
        # Resetar estado e redesenhar
        self.w = None
        self.historico = []
        self._resetar_metricas()
        self._desenhar_inicial()
        self._escrever_analise_inicial()

    def _ao_mudar_config(self):
        """Ao mudar par ou atributos: limpa modelo e redesenha scatter."""
        self.w = None
        self.historico = []
        self._resetar_metricas()
        self._desenhar_inicial()
        self._escrever_analise_inicial()

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

        algo = self.var_algo.get()

        if algo == 'xor':
            self._treinar_xor(taxa, epocas)
        elif algo == 'perceptron':
            self._treinar_perceptron_iris(taxa, epocas)
        else:
            self._treinar_delta_iris(taxa, epocas)

        self._desenhar_treinado()
        self._atualizar_metricas()
        self._atualizar_analise()

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

    # -----------------------------------------------------------------------
    # Graficos
    # -----------------------------------------------------------------------
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
        self._desenhar_conv_vazio()
        self.canvas.draw()

    def _desenhar_treinado(self):
        """Scatter + fronteira de decisao + curva de convergencia."""
        self._desenhar_scatter(com_fronteira=True)
        self._desenhar_curva_convergencia()
        self.canvas.draw()

    def _desenhar_scatter(self, com_fronteira=False):
        ax = self.ax_sc
        ax.cla()
        self._estilizar_ax(ax)

        algo = self.var_algo.get()

        if algo == 'xor':
            self._scatter_xor(ax, com_fronteira)
        else:
            self._scatter_iris(ax, com_fronteira)

    def _scatter_iris(self, ax, com_fronteira):
        indices = CONF_ATTR[self.var_attr.get()]['indices']
        cfg     = CONF_ATTR[self.var_attr.get()]

        # Plota todas as 3 classes (todos os dados)
        for classe in CLASSES:
            pts = [d for d in self.dados if d['classe'] == classe]
            if not pts:
                continue
            x1 = [d['atributos'][indices[0]] for d in pts]
            x2 = [d['atributos'][indices[1]] for d in pts]
            ax.scatter(x1, x2,
                       color=CORES_CLASSE[classe],
                       marker=MARCADORES_CLASSE[classe],
                       s=30, alpha=0.75, linewidths=0.4,
                       edgecolors=T.BG, label=classe.capitalize(),
                       zorder=3)

        ax.set_xlabel(cfg['eixo_x'], fontsize=8)
        ax.set_ylabel(cfg['eixo_y'], fontsize=8)

        par = self.var_par.get()
        cp, cn = PAR_POR_MODO[par]
        titulo = f'{cp.capitalize()}  ×  {cn.capitalize()}'

        # Fronteira de decisao
        if com_fronteira and self.w is not None and len(self.w) == 3:
            self._plotar_fronteira_2d(ax, self.w, indices)

        ax.set_title(titulo, fontsize=9, pad=6)
        ax.legend(fontsize=7, facecolor=T.BG_PANEL,
                  edgecolor=T.BORDER, labelcolor=T.FG_MUTED,
                  loc='upper left')

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

        if not self.historico:
            self._desenhar_conv_vazio()
            return

        algo = self.var_algo.get()
        epocas = list(range(1, len(self.historico) + 1))

        if algo == 'perceptron':
            cor = T.SUCCESS if self.convergiu else T.DANGER
            ax.plot(epocas, self.historico, color=cor, lw=1.5, zorder=3)
            ax.set_ylabel('Erros / Epoca', fontsize=7)
            ax.set_title('Perceptron — Erros', fontsize=9, pad=6)
            ax.set_ylim(bottom=0)
        else:
            # Regra Delta ou XOR
            cor = T.DATA_MINT if algo == 'delta' else T.DATA_CORAL
            ax.plot(epocas, self.historico, color=cor, lw=1.5, zorder=3)
            ax.set_ylabel('MSE', fontsize=7)
            titulo = ('Regra Delta — MSE' if algo == 'delta'
                      else 'XOR — MSE (nao converge)')
            ax.set_title(titulo, fontsize=9, pad=6)
            ax.set_ylim(bottom=0)

            # Linha do valor final (referencia visual)
            ax.axhline(self.historico[-1], color=T.BORDER_HARD,
                       ls=':', lw=0.8, zorder=2)

        ax.set_xlabel('Epoca', fontsize=7)
        ax.tick_params(axis='both', labelsize=7)

    # -----------------------------------------------------------------------
    # Metricas e analise
    # -----------------------------------------------------------------------
    def _resetar_metricas(self):
        self.metric_acc.set('—')
        self.metric_acc.lbl_valor.configure(fg=T.FG)
        self.metric_epocas.set('—')
        self.metric_conv.set('—')

    def _atualizar_metricas(self):
        algo = self.var_algo.get()

        # Acuracia
        if self.acc_teste is not None:
            pct = self.acc_teste * 100
            cor = (T.SUCCESS if pct >= 95 else
                   T.ACCENT  if pct >= 80 else T.DANGER)
            self.metric_acc.set(f'{pct:.2f}%', cor)
        else:
            self.metric_acc.set('N/A (XOR)', T.FG_MUTED)

        # Epocas
        max_ep = int(self.var_epocas.get())
        self.metric_epocas.set(f'{self.n_epocas_treinadas} / {max_ep}')

        # Convergencia
        if algo == 'perceptron':
            erros = self.historico[-1] if self.historico else '—'
            self.metric_conv.set(
                'Convergiu' if self.convergiu else f'Erros: {erros}',
                T.SUCCESS if self.convergiu else T.DANGER)
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

        if self.convergiu:
            self._txt_set([
                ('Perceptron convergiu\n\n', 'ok'),
                (f'O algoritmo convergiu em ', None),
                (f'{n_ep}', 'hl'), (' epocas', None),
                (' com p = ', None), (self.var_taxa.get(), 'mono'), ('.\n\n', None),
                (f'{cp.capitalize()} × {cn.capitalize()}', 'hl'),
                (' sao linearmente separaveis com os atributos selecionados. '
                 'A fronteira  ', None),
                ('w₀ + w₁·x₁ + w₂·x₂ = 0', 'mono'),
                (f'\nsepara corretamente todas as amostras de treino.\n\n'
                 f'Acuracia no teste: ', None),
                (f'{self.acc_teste*100:.2f}%', 'ok'), ('.', None),
            ])
        else:
            self._txt_set([
                ('Perceptron nao convergiu\n\n', 'err'),
                (f'Atingido o limite de ', None),
                (f'{n_ep}', 'hl'), (' epocas', None),
                (f' sem convergencia (p = ', None),
                (self.var_taxa.get(), 'mono'), (').\n\n', None),
                (f'{cp.capitalize()} × {cn.capitalize()}', 'hl'),
                (' possuem sobreposicao neste espaco de atributos. '
                 'Nenhuma reta separa perfeitamente as duas classes.\n\n'
                 'Experimente: troque os atributos para Petalas [2,3] '
                 'ou selecione um par com maior separacao.', None),
            ])

    def _analise_delta_iris(self):
        par = self.var_par.get()
        cp, cn = PAR_POR_MODO[par]
        mse_ini = self.historico[0] if self.historico else 0
        mse_fin = self.historico[-1] if self.historico else 0
        reducao = (1 - mse_fin / mse_ini) * 100 if mse_ini > 1e-12 else 0

        self._txt_set([
            ('Regra Delta — convergencia MSE\n\n', 'hl'),
            (f'{cp.capitalize()} × {cn.capitalize()}', 'hl'),
            (', p = ', None), (self.var_taxa.get(), 'mono'),
            (f', {self.n_epocas_treinadas} epocas.\n\n', None),
            ('MSE inicial: ', None),  (f'{mse_ini:.4f}', 'mono'), ('\n', None),
            ('MSE final:   ', None),  (f'{mse_fin:.4f}', 'mono'),
            (f'  (reducao de {reducao:.1f}%)\n\n', None),
            ('A Regra Delta minimiza o MSE independentemente '
             'da separabilidade linear — ao contrario do Perceptron, '
             'nao exige dados linearmente separaveis para convergir.\n\n'
             'Acuracia no teste: ', None),
            (f'{self.acc_teste*100:.2f}%', 'ok'), ('.', None),
        ])

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
