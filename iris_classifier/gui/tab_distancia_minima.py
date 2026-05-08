"""
Aba 1 - Classificador de Distancia Minima.

Composicao:
- Coluna esquerda: controles (atributos, modo do grafico, classificacao
  manual, predicao destacada).
- Coluna direita: grafico matplotlib embarcado (alterna entre dispersao
  geral e fronteiras dos pares).
- Painel inferior direito: metricas em destaque e analise textual sobre
  separabilidade / sobreposicao das classes.

Toda matematica vem dos modulos de Python puro (math_utils, classifier).
A aba so orquestra entrada do usuario, plotagem e analise.
"""
import os
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from data_loader import (carregar_dados_iris, split_estratificado,
                         filtrar_por_classes)
from classifier import treinar, predizer_todas_classes
from evaluator import acuracia
from math_utils import coeficientes_superficie_decisao

from . import theme as T
from .widgets import Card, MetricBlock


# ---------------------------------------------------------------------------
# Constantes do experimento
# ---------------------------------------------------------------------------
PROJETO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
CAMINHO_DADOS = os.path.join(PROJETO_ROOT, 'data', 'Iris data.xls')

CLASSES = ['setosa', 'versicolor', 'virginica']

CONFIGURACOES_ATRIBUTOS = {
    'petalas': {
        'rotulo': 'Petalas',
        'rotulo_ui': 'Petalas  ·  [2,3]',
        'indices': [2, 3],
        'eixo_x': 'Comp. Petala',
        'eixo_y': 'Larg. Petala',
        'unidade': 'cm',
    },
    'sepalas': {
        'rotulo': 'Sepalas',
        'rotulo_ui': 'Sepalas  ·  [0,1]',
        'indices': [0, 1],
        'eixo_x': 'Comp. Sepala',
        'eixo_y': 'Larg. Sepala',
        'unidade': 'cm',
    },
}

CORES_CLASSE = {
    'setosa':     T.DATA_BLUE,
    'versicolor': T.DATA_MINT,
    'virginica':  T.DATA_CORAL,
}

MODOS_GRAFICO = [
    ('Dispersao geral',         'dispersao'),
    ('Setosa  x  Versicolor',   'sv'),
    ('Versicolor  x  Virginica','vv'),
    ('Setosa  x  Virginica',    'sg'),
]

PAR_POR_MODO = {
    'sv': ('setosa', 'versicolor'),
    'vv': ('versicolor', 'virginica'),
    'sg': ('setosa', 'virginica'),
}


# ===========================================================================
# Aba
# ===========================================================================
class TabDistanciaMinima(tk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, bg=T.BG, **kw)

        # Estado
        self.dados = []
        self.dados_treino = []
        self.dados_teste = []
        self.prototipos = {}
        self.atributos_ativos = tk.StringVar(value='petalas')
        self.modo_grafico = tk.StringVar(value='dispersao')
        self.var_x = tk.StringVar(value='4.5')
        self.var_y = tk.StringVar(value='1.5')

        self._construir_layout()
        self._carregar_dados()
        self._atualizar_modelo()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _construir_layout(self):
        # 2 colunas: controles fixos a esquerda, viz flexivel a direita
        self.columnconfigure(0, minsize=320)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._coluna_controles()
        self._coluna_visualizacao()

    # ---- Coluna esquerda ----
    def _coluna_controles(self):
        wrap = tk.Frame(self, bg=T.BG)
        wrap.grid(row=0, column=0, sticky='nsew', padx=(20, 10), pady=20)
        wrap.columnconfigure(0, weight=1)

        # 1. Atributos
        card = Card(wrap, titulo='atributos do modelo')
        card.grid(row=0, column=0, sticky='ew')
        for chave, cfg in CONFIGURACOES_ATRIBUTOS.items():
            tk.Radiobutton(card, text=cfg['rotulo_ui'],
                           value=chave, variable=self.atributos_ativos,
                           bg=T.BG_CARD, fg=T.FG,
                           selectcolor=T.BG_HOVER,
                           activebackground=T.BG_CARD,
                           activeforeground=T.ACCENT,
                           font=T.FONT_BODY, anchor='w',
                           borderwidth=0, highlightthickness=0,
                           command=self._ao_trocar_atributos
                          ).pack(fill='x', padx=14, pady=2)
        # respiro inferior
        tk.Frame(card, bg=T.BG_CARD, height=10).pack()

        # 2. Modo do grafico
        card = Card(wrap, titulo='visualizacao')
        card.grid(row=1, column=0, sticky='ew', pady=(14, 0))
        for rotulo, valor in MODOS_GRAFICO:
            tk.Radiobutton(card, text=rotulo,
                           value=valor, variable=self.modo_grafico,
                           bg=T.BG_CARD, fg=T.FG,
                           selectcolor=T.BG_HOVER,
                           activebackground=T.BG_CARD,
                           activeforeground=T.ACCENT,
                           font=T.FONT_BODY, anchor='w',
                           borderwidth=0, highlightthickness=0,
                           command=self._desenhar_grafico
                          ).pack(fill='x', padx=14, pady=2)
        tk.Frame(card, bg=T.BG_CARD, height=10).pack()

        # 3. Classificacao manual
        card = Card(wrap, titulo='classificar amostra')
        card.grid(row=2, column=0, sticky='ew', pady=(14, 0))

        form = tk.Frame(card, bg=T.BG_CARD)
        form.pack(fill='x', padx=14, pady=(2, 0))
        form.columnconfigure(1, weight=1)

        self.lbl_x = tk.Label(form, text='Comp. Petala',
                              bg=T.BG_CARD, fg=T.FG_MUTED,
                              font=T.FONT_LABEL, anchor='w')
        self.lbl_x.grid(row=0, column=0, sticky='w', pady=(0, 2))
        ttk.Entry(form, textvariable=self.var_x,
                  font=T.FONT_MONO, width=10
                 ).grid(row=0, column=1, sticky='ew', padx=(8, 0))

        self.lbl_y = tk.Label(form, text='Larg. Petala',
                              bg=T.BG_CARD, fg=T.FG_MUTED,
                              font=T.FONT_LABEL, anchor='w')
        self.lbl_y.grid(row=1, column=0, sticky='w', pady=(8, 2))
        ttk.Entry(form, textvariable=self.var_y,
                  font=T.FONT_MONO, width=10
                 ).grid(row=1, column=1, sticky='ew', padx=(8, 0))

        ttk.Button(card, text='Classificar  >',
                   style='Primary.TButton',
                   command=self._classificar_amostra
                  ).pack(fill='x', padx=14, pady=(14, 14))

        # 4. Predicao (resultado em destaque)
        card = Card(wrap, titulo='predicao')
        card.grid(row=3, column=0, sticky='ew', pady=(14, 0))
        self.lbl_pred = tk.Label(card, text='—',
                                 bg=T.BG_CARD, fg=T.FG_DIM,
                                 font=T.FONT_VALUE_HUGE, anchor='w')
        self.lbl_pred.pack(fill='x', padx=14, pady=(2, 2))
        self.lbl_pred_sub = tk.Label(card,
                                     text='aguardando entrada',
                                     bg=T.BG_CARD, fg=T.FG_MUTED,
                                     font=T.FONT_MONO_SM, anchor='w',
                                     justify='left', wraplength=280)
        self.lbl_pred_sub.pack(fill='x', padx=14, pady=(0, 14))

        # spacer no fundo
        wrap.rowconfigure(4, weight=1)

    # ---- Coluna direita ----
    def _coluna_visualizacao(self):
        wrap = tk.Frame(self, bg=T.BG)
        wrap.grid(row=0, column=1, sticky='nsew', padx=(10, 20), pady=20)
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=3)
        wrap.rowconfigure(1, weight=2)

        # ---- Grafico ----
        painel = tk.Frame(wrap, bg=T.BG_PANEL,
                          highlightthickness=1,
                          highlightbackground=T.BORDER,
                          highlightcolor=T.BORDER)
        painel.grid(row=0, column=0, sticky='nsew')
        painel.columnconfigure(0, weight=1)
        painel.rowconfigure(0, weight=1)

        self.figura = Figure(figsize=(7, 4.2), dpi=100, facecolor=T.BG_PANEL)
        self.ax = self.figura.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figura, master=painel)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew',
                                         padx=10, pady=10)

        # ---- Painel inferior ----
        inferior = tk.Frame(wrap, bg=T.BG)
        inferior.grid(row=1, column=0, sticky='nsew', pady=(14, 0))
        inferior.columnconfigure(0, weight=1)
        inferior.columnconfigure(1, weight=2)
        inferior.rowconfigure(0, weight=1)

        # Coluna metricas
        col_m = tk.Frame(inferior, bg=T.BG)
        col_m.grid(row=0, column=0, sticky='nsew', padx=(0, 14))
        col_m.columnconfigure(0, weight=1)

        self.metric_acc = MetricBlock(col_m, 'acuracia teste', '—')
        self.metric_acc.grid(row=0, column=0, sticky='ew')
        self.metric_split = MetricBlock(col_m, 'treino  ·  teste', '—')
        self.metric_split.grid(row=1, column=0, sticky='ew', pady=(10, 0))
        self.metric_erros = MetricBlock(col_m, 'erros base completa', '—')
        self.metric_erros.grid(row=2, column=0, sticky='ew', pady=(10, 0))

        # Analise textual
        card = Card(inferior, titulo='analise')
        card.grid(row=0, column=1, sticky='nsew')
        self.txt_analise = tk.Text(card, height=9, wrap='word',
                                   bg=T.BG_CARD, fg=T.FG,
                                   font=T.FONT_BODY,
                                   relief='flat', borderwidth=0,
                                   highlightthickness=0,
                                   padx=14, pady=2,
                                   spacing1=2, spacing3=4)
        self.txt_analise.pack(fill='both', expand=True,
                              padx=14, pady=(2, 14))
        self.txt_analise.tag_configure('hl', foreground=T.ACCENT,
                                        font=('Segoe UI Semibold', 10))
        self.txt_analise.tag_configure('mono', foreground=T.FG,
                                        font=T.FONT_MONO)
        self.txt_analise.configure(state='disabled')

    # ------------------------------------------------------------------
    # Dados e modelo
    # ------------------------------------------------------------------
    def _carregar_dados(self):
        if not os.path.exists(CAMINHO_DADOS):
            self._set_analise(f'Erro: arquivo nao encontrado em\n{CAMINHO_DADOS}')
            return
        self.dados = carregar_dados_iris(CAMINHO_DADOS)
        self.dados_treino, self.dados_teste = split_estratificado(
            self.dados, proporcao_treino=0.7, semente=42)

    def _atualizar_modelo(self):
        if not self.dados:
            return
        cfg = CONFIGURACOES_ATRIBUTOS[self.atributos_ativos.get()]
        indices = cfg['indices']
        self.prototipos = treinar(self.dados_treino, indices)

        # Avaliar no teste
        preds_t, gab_t = self._predizer_lista(self.dados_teste, indices)
        acc_teste = acuracia(preds_t, gab_t)

        # Erros na base completa (mostra overlap real)
        preds_all, gab_all = self._predizer_lista(self.dados, indices)
        erros_total = sum(1 for p, g in zip(preds_all, gab_all) if p != g)
        erros_vv = sum(1 for p, g in zip(preds_all, gab_all)
                       if {p, g} == {'versicolor', 'virginica'} and p != g)

        # Atualizar metricas
        cor_acc = T.SUCCESS if acc_teste >= 0.95 else (
                  T.ACCENT if acc_teste >= 0.80 else T.DANGER)
        self.metric_acc.set(f'{acc_teste * 100:.2f}%', cor=cor_acc)
        self.metric_split.set(
            f'{len(self.dados_treino)}  ·  {len(self.dados_teste)}')
        self.metric_erros.set(
            f'{erros_total}  /  {len(self.dados)}',
            cor=T.SUCCESS if erros_total == 0 else T.ACCENT)

        # Atualizar rotulos do form de classificacao
        self.lbl_x.configure(text=cfg['eixo_x'])
        self.lbl_y.configure(text=cfg['eixo_y'])

        self._atualizar_analise(acc_teste, erros_total, erros_vv)
        self._desenhar_grafico()

    def _predizer_lista(self, lista, indices):
        preds, gab = [], []
        for amostra in lista:
            _, p = predizer_todas_classes(amostra['atributos'],
                                          self.prototipos, indices)
            preds.append(p)
            gab.append(amostra['classe'])
        return preds, gab

    def _ao_trocar_atributos(self):
        # Reseta resultado da classificacao manual ao trocar atributos
        self.lbl_pred.configure(text='—', fg=T.FG_DIM)
        self.lbl_pred_sub.configure(text='aguardando entrada')
        self._atualizar_modelo()

    def _classificar_amostra(self):
        cfg = CONFIGURACOES_ATRIBUTOS[self.atributos_ativos.get()]
        try:
            v1 = float(self.var_x.get().replace(',', '.'))
            v2 = float(self.var_y.get().replace(',', '.'))
        except ValueError:
            self.lbl_pred.configure(text='—', fg=T.DANGER)
            self.lbl_pred_sub.configure(
                text='valores invalidos (use numeros decimais)')
            return

        x = [0.0] * 4
        x[cfg['indices'][0]] = v1
        x[cfg['indices'][1]] = v2
        scores, vencedor = predizer_todas_classes(x, self.prototipos,
                                                  cfg['indices'])

        cor = CORES_CLASSE[vencedor]
        self.lbl_pred.configure(text=vencedor.upper(), fg=cor)
        outros = '   '.join(f'{c[:3]}={scores[c]:+.2f}'
                            for c in CLASSES if c != vencedor)
        self.lbl_pred_sub.configure(
            text=f'd_max = {scores[vencedor]:+.3f}\n{outros}')

    # ------------------------------------------------------------------
    # Texto de analise
    # ------------------------------------------------------------------
    def _atualizar_analise(self, acc_teste, erros_total, erros_vv):
        atrib = self.atributos_ativos.get()
        if atrib == 'petalas':
            partes = [
                ('Setosa', 'hl'),
                ' fica isolada — petalas pequenas (~1.5 x 0.25 cm) ',
                'versus versicolor (~4.3 x 1.3) e virginica (~5.5 x 2.0).\n\n',
                'Versicolor e virginica ',
                ('compartilham uma faixa em torno de 4.8 a 5.2 cm de '
                 'comprimento de petala', 'hl'),
                ' — sao parentes botanicos proximos. Por isso o classificador '
                'comete ',
                (f'{erros_total}', 'hl'),
                ' erro(s) na base completa, ',
                (f'{erros_vv}', 'hl'),
                ' deles entre essas duas. No teste com seed 42 a separacao '
                'cai exatamente fora da zona de overlap, atingindo ',
                (f'{acc_teste * 100:.2f}%', 'hl'), '.',
            ]
        else:
            partes = [
                'Sepalas trazem ',
                ('forte sobreposicao', 'hl'),
                ' entre as tres classes — versicolor e virginica '
                'ocupam praticamente a mesma regiao do plano. ',
                'Um classificador linear ',
                ('nao consegue separar regioes que ja se cruzam '
                 'no espaco de atributos', 'hl'),
                f', entao a acuracia cai para ',
                (f'{acc_teste * 100:.2f}%', 'hl'),
                ' e ha ',
                (f'{erros_total}', 'hl'),
                ' erro(s) ao classificar a base completa.\n\n',
                'Compare: trocando para Petalas a acuracia volta para ~100%.',
            ]
        self._set_analise_rico(partes)

    def _set_analise(self, texto):
        self.txt_analise.configure(state='normal')
        self.txt_analise.delete('1.0', 'end')
        self.txt_analise.insert('1.0', texto)
        self.txt_analise.configure(state='disabled')

    def _set_analise_rico(self, partes):
        self.txt_analise.configure(state='normal')
        self.txt_analise.delete('1.0', 'end')
        for parte in partes:
            if isinstance(parte, tuple):
                texto, tag = parte
                self.txt_analise.insert('end', texto, tag)
            else:
                self.txt_analise.insert('end', parte)
        self.txt_analise.configure(state='disabled')

    # ------------------------------------------------------------------
    # Grafico
    # ------------------------------------------------------------------
    def _desenhar_grafico(self):
        if not self.dados:
            return
        cfg = CONFIGURACOES_ATRIBUTOS[self.atributos_ativos.get()]
        indices = cfg['indices']
        modo = self.modo_grafico.get()

        self.figura.clear()
        self.figura.set_facecolor(T.BG_PANEL)
        self.ax = self.figura.add_subplot(111, facecolor=T.BG_CARD)

        for spine in self.ax.spines.values():
            spine.set_color(T.BORDER_HARD)
            spine.set_linewidth(0.8)
        self.ax.tick_params(colors=T.FG_MUTED, labelsize=8,
                            length=4, width=0.8)
        self.ax.grid(True, color=T.BORDER, linestyle=':',
                     linewidth=0.6, alpha=0.7)

        if modo == 'dispersao':
            self._desenhar_dispersao(indices, cfg)
        else:
            self._desenhar_par(indices, cfg, PAR_POR_MODO[modo])

        self.ax.title.set_color(T.FG)
        self.ax.xaxis.label.set_color(T.FG_MUTED)
        self.ax.yaxis.label.set_color(T.FG_MUTED)
        self.figura.tight_layout(pad=1.2)
        self.canvas.draw()

    def _desenhar_dispersao(self, indices, cfg):
        ids_treino = set(id(d) for d in self.dados_treino)
        # Setosa por ultimo (pequena, ficaria encoberta)
        for classe in ['virginica', 'versicolor', 'setosa']:
            cor = CORES_CLASSE[classe]
            treino = [d for d in self.dados
                      if d['classe'] == classe and id(d) in ids_treino]
            teste = [d for d in self.dados
                     if d['classe'] == classe and id(d) not in ids_treino]
            self._scatter(treino, indices, cor,
                          alpha=0.55, edge=T.BG_CARD, size=30)
            self._scatter(teste, indices, cor,
                          alpha=1.0, edge='white', size=58,
                          linewidth=1.0, label=classe)

        for classe, p in self.prototipos.items():
            cor = CORES_CLASSE[classe]
            self.ax.scatter(p[0], p[1], color=cor, marker='X', s=260,
                            edgecolors='white', linewidths=1.6,
                            zorder=6)
            self.ax.annotate(f'm_{classe[:3]}', (p[0], p[1]),
                             xytext=(10, 8), textcoords='offset points',
                             color=T.FG, fontsize=9,
                             fontweight='bold', fontfamily='Cambria')

        self.ax.set_xlabel(f"{cfg['eixo_x']} ({cfg['unidade']})")
        self.ax.set_ylabel(f"{cfg['eixo_y']} ({cfg['unidade']})")
        self.ax.set_title('Distribuicao completa  ·  teste destacado',
                          color=T.FG, fontsize=11, pad=10, loc='left',
                          fontfamily='Cambria', fontweight='bold')
        self._legenda()

    def _desenhar_par(self, indices, cfg, par):
        classe_i, classe_j = par
        treino_par = filtrar_por_classes(self.dados_treino, list(par))
        prots = treinar(treino_par, indices)
        pi, pj = prots[classe_i], prots[classe_j]
        w, b = coeficientes_superficie_decisao(pi, pj)

        ids_treino = set(id(d) for d in self.dados_treino)
        cor_i = CORES_CLASSE[classe_i]
        cor_j = CORES_CLASSE[classe_j]

        for classe, cor in [(classe_i, cor_i), (classe_j, cor_j)]:
            treino = [d for d in self.dados
                      if d['classe'] == classe and id(d) in ids_treino]
            teste = [d for d in self.dados
                     if d['classe'] == classe and id(d) not in ids_treino]
            self._scatter(treino, indices, cor,
                          alpha=0.55, edge=T.BG_CARD, size=30)
            self._scatter(teste, indices, cor,
                          alpha=1.0, edge='white', size=58,
                          linewidth=1.0, label=classe)

        self.ax.scatter(pi[0], pi[1], color=cor_i, marker='X', s=260,
                        edgecolors='white', linewidths=1.6, zorder=6)
        self.ax.scatter(pj[0], pj[1], color=cor_j, marker='X', s=260,
                        edgecolors='white', linewidths=1.6, zorder=6)

        x1_pts = [d['atributos'][indices[0]] for d in self.dados
                  if d['classe'] in par]
        x2_pts = [d['atributos'][indices[1]] for d in self.dados
                  if d['classe'] in par]
        x1_min, x1_max = min(x1_pts) - 0.4, max(x1_pts) + 0.4
        x2_min, x2_max = min(x2_pts) - 0.4, max(x2_pts) + 0.4

        # Fronteira + sombreamento das regioes
        if abs(w[1]) > 1e-9:
            xs = [x1_min, x1_max]
            ys = [(-w[0] * x - b) / w[1] for x in xs]
            # Topo da figura pertence a quem? Se w[1]>0, regiao i fica acima.
            if w[1] > 0:
                self.ax.fill_between(xs, ys, [x2_max, x2_max],
                                     color=cor_i, alpha=0.06, zorder=1)
                self.ax.fill_between(xs, [x2_min, x2_min], ys,
                                     color=cor_j, alpha=0.06, zorder=1)
            else:
                self.ax.fill_between(xs, [x2_min, x2_min], ys,
                                     color=cor_i, alpha=0.06, zorder=1)
                self.ax.fill_between(xs, ys, [x2_max, x2_max],
                                     color=cor_j, alpha=0.06, zorder=1)
            self.ax.plot(xs, ys, color=T.ACCENT, linewidth=2.0,
                         linestyle='--',
                         label='fronteira  d_ij(x) = 0', zorder=5)
        else:
            xv = -b / w[0]
            self.ax.axvline(xv, color=T.ACCENT, linewidth=2.0,
                            linestyle='--',
                            label='fronteira  d_ij(x) = 0', zorder=5)

        self.ax.set_xlim(x1_min, x1_max)
        self.ax.set_ylim(x2_min, x2_max)
        self.ax.set_xlabel(f"{cfg['eixo_x']} ({cfg['unidade']})")
        self.ax.set_ylabel(f"{cfg['eixo_y']} ({cfg['unidade']})")
        self.ax.set_title(f'Fronteira  ·  {classe_i}  x  {classe_j}',
                          color=T.FG, fontsize=11, pad=10, loc='left',
                          fontfamily='Cambria', fontweight='bold')
        self._legenda()

    def _scatter(self, dados, indices, cor, alpha=0.85, edge='white',
                 size=44, linewidth=0.5, label=None):
        if not dados:
            return
        x = [d['atributos'][indices[0]] for d in dados]
        y = [d['atributos'][indices[1]] for d in dados]
        self.ax.scatter(x, y, color=cor, alpha=alpha,
                        edgecolors=edge, linewidths=linewidth,
                        s=size, label=label, zorder=4)

    def _legenda(self):
        leg = self.ax.legend(loc='best', frameon=True,
                             facecolor=T.BG_PANEL,
                             edgecolor=T.BORDER_HARD,
                             labelcolor=T.FG, fontsize=8,
                             framealpha=0.95)
        if leg:
            leg.get_frame().set_linewidth(0.8)
