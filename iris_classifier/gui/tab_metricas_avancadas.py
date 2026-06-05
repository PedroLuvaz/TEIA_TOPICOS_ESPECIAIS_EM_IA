"""
Aba 3 — Metricas Avancadas (Aula PR_51)
========================================
Classifica o Iris com 6 abordagens:
  1. Distancia Minima        — argmin ||x - m_j||
  2. Distancia Maxima        — argmax ||x - m_j||  (baseline inferior)
  3. Superficie de decisao OvA — voto por par
  4. Perceptron OvA          — 3 pares, voto por net
  5. Regra Delta binaria OvA — 3 pares, voto por net
  6. Regra Delta OvA         — argmax nets

Sub-abas:
  [Comparativo]        — tabela todos classificadores x metricas globais
  [Detalhe por Classe] — produtor, usuario, F1, F2, MCC por classe (OvR)
  [Pares de Classes]   — MCC e Fb (b=1, b=2) para cada par (set×ver, ver×vir, set×vir)
  [Matriz Confusao]    — heatmap colorido
  [Grafico]            — barras Ag / Kappa / Tau
  [Comparacao K & T]   — teste Z de Kappa e Tau: Perceptron vs Delta  (Item 2)
  [Exercicios PR51]    — exercicios do slide com matrizes A e B         (Item 3)

Toda matematica em Python puro — sem numpy/scipy/sklearn.
"""
import os
import sys
import tkinter as tk
from tkinter import ttk
import math

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

PROJETO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
IRIS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (PROJETO_ROOT, IRIS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from data_loader import carregar_dados_iris, split_estratificado, filtrar_por_classes
from classifier  import treinar, predizer_todas_classes, predizer_binario
from math_utils  import distancia_euclidiana
from perceptron  import treinar_perceptron
from delta_rule  import (treinar_delta_iris, treinar_delta_ova, predizer_delta_ova)
from metricas_avancadas import (
    relatorio_completo, relatorio_binario,
    kappa, tau, variancia_kappa, variancia_tau,
    z_kappa, z_tau, p_valor_z,
    acerto_global, acuracia_produtor, acuracia_usuario,
    fb_score, mcc as mcc_fn,
)

from . import theme as T
from .widgets import Card, MetricBlock
from .janela_calculos import JanelaMemoriaCalculoMetricas

# ---------------------------------------------------------------------------
CAMINHO_DADOS = os.path.join(PROJETO_ROOT, 'data', 'Iris data.xls')
CLASSES = ['setosa', 'versicolor', 'virginica']
IDX_PETALA = [2, 3]
IDX_SEPALA = [0, 1]
PARES = [('setosa', 'versicolor'), ('versicolor', 'virginica'), ('setosa', 'virginica')]
ROTULO_PAR = {
    ('setosa', 'versicolor'):  'Setosa × Versicolor',
    ('versicolor', 'virginica'): 'Versicolor × Virginica',
    ('setosa', 'virginica'):   'Setosa × Virginica',
}

CORES_CLASSE = {
    'setosa':     T.DATA_BLUE,
    'versicolor': T.DATA_MINT,
    'virginica':  T.DATA_CORAL,
}

KAPPA_INTERP = [
    (0.81, 'Quase Perfeito'),
    (0.61, 'Substancial'),
    (0.41, 'Moderado'),
    (0.21, 'Razoavel'),
    (0.00, 'Fraco'),
    (-999, 'Nenhum'),
]

def interpretar_kappa(k):
    for limiar, rotulo in KAPPA_INTERP:
        if k > limiar:
            return rotulo
    return 'Nenhum'


# ===========================================================================
class TabMetricasAvancadas(tk.Frame):

    def __init__(self, master, **kw):
        super().__init__(master, bg=T.BG, **kw)

        self.dados        = []
        self.dados_treino = []
        self.dados_teste  = []
        self.resultados   = {}   # nome_modelo -> relatorio_completo(...)
        self.preds_por_modelo = {}  # nome -> (preds, gab)
        self._treinado    = False

        self.var_atributos  = tk.StringVar(value='petalas')
        self.var_modelo_sel = tk.StringVar(value='')
        self.var_classe_sel = tk.StringVar(value='setosa')

        self._construir_layout()
        self._carregar_dados()

    # -----------------------------------------------------------------------
    # Layout
    # -----------------------------------------------------------------------
    def _construir_layout(self):
        self.columnconfigure(0, minsize=285)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._col_esq()
        self._col_dir()

    def _col_esq(self):
        wrap = tk.Frame(self, bg=T.BG)
        wrap.grid(row=0, column=0, sticky='nsew', padx=(16, 8), pady=12)
        wrap.columnconfigure(0, weight=1)

        # Atributos
        card = Card(wrap, titulo='atributos do modelo')
        card.grid(row=0, column=0, sticky='ew')
        for val, lbl in [('petalas', 'Petalas  ·  [2,3]'),
                          ('sepalas', 'Sepalas  ·  [0,1]')]:
            tk.Radiobutton(card, text=lbl, value=val,
                           variable=self.var_atributos,
                           bg=T.BG_CARD, fg=T.FG, selectcolor=T.BG_HOVER,
                           activebackground=T.BG_CARD, activeforeground=T.ACCENT,
                           font=T.FONT_BODY, anchor='w',
                           borderwidth=0, highlightthickness=0
                          ).pack(fill='x', padx=14, pady=2)
        tk.Frame(card, bg=T.BG_CARD, height=4).pack()

        ttk.Button(wrap, text='Treinar e Calcular Metricas  >',
                   style='Primary.TButton',
                   command=self._treinar_tudo
                  ).grid(row=1, column=0, sticky='ew', pady=(8, 0))

        self.lbl_status = tk.Label(wrap, text='Aguardando treinamento.',
                                   bg=T.BG, fg=T.FG_MUTED,
                                   font=T.FONT_MONO_SM, anchor='w',
                                   wraplength=255, justify='left')
        self.lbl_status.grid(row=2, column=0, sticky='ew', pady=(4, 0))

        # Seletor de modelo
        card2 = Card(wrap, titulo='selecionar modelo')
        card2.grid(row=3, column=0, sticky='ew', pady=(10, 0))
        self._frame_radios_modelo = tk.Frame(card2, bg=T.BG_CARD)
        self._frame_radios_modelo.pack(fill='x', padx=14, pady=(0, 6))

        # Seletor de classe (metricas OvR)
        card3 = Card(wrap, titulo='classe  (OvR)')
        card3.grid(row=4, column=0, sticky='ew', pady=(8, 0))
        for c in CLASSES:
            tk.Radiobutton(card3, text=c.capitalize(),
                           value=c, variable=self.var_classe_sel,
                           bg=T.BG_CARD, fg=CORES_CLASSE[c],
                           selectcolor=T.BG_HOVER,
                           activebackground=T.BG_CARD, activeforeground=T.ACCENT,
                           font=T.FONT_BODY, anchor='w',
                           borderwidth=0, highlightthickness=0,
                           command=self._atualizar_metricas_globais
                          ).pack(fill='x', padx=14, pady=1)
        tk.Frame(card3, bg=T.BG_CARD, height=4).pack()

        # Legenda Kappa
        card4 = Card(wrap, titulo='interpretacao kappa')
        card4.grid(row=5, column=0, sticky='ew', pady=(8, 0))
        for faixa, desc, cor in [
            ('> 0.80', 'Quase Perfeito', T.SUCCESS),
            ('> 0.60', 'Substancial',    T.DATA_MINT),
            ('> 0.40', 'Moderado',       T.ACCENT),
            ('> 0.20', 'Razoavel',       T.FG_MUTED),
            ('<= 0.20','Fraco/Nenhum',   T.DANGER),
        ]:
            f = tk.Frame(card4, bg=T.BG_CARD)
            f.pack(fill='x', padx=14, pady=1)
            tk.Label(f, text=faixa, bg=T.BG_CARD, fg=cor,
                     font=T.FONT_MONO_SM, width=8, anchor='w').pack(side='left')
            tk.Label(f, text=desc, bg=T.BG_CARD, fg=T.FG_MUTED,
                     font=T.FONT_LABEL, anchor='w').pack(side='left', padx=(4, 0))
        tk.Frame(card4, bg=T.BG_CARD, height=4).pack()

        # Memoria de calculo das metricas
        card5 = Card(wrap, titulo='memoria de calculo')
        card5.grid(row=6, column=0, sticky='ew', pady=(8, 0))
        ttk.Button(card5, text='Abrir memoria de calculo  >',
                   style='Primary.TButton',
                   command=self._abrir_memoria_metricas
                  ).pack(fill='x', padx=14, pady=(2, 10))

        wrap.rowconfigure(7, weight=1)

    def _col_dir(self):
        wrap = tk.Frame(self, bg=T.BG)
        wrap.grid(row=0, column=1, sticky='nsew', padx=(8, 16), pady=12)
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=0)
        wrap.rowconfigure(1, weight=1)
        wrap.rowconfigure(2, weight=0)

        # Faixa de metricas globais
        self._faixa_global = tk.Frame(wrap, bg=T.BG)
        self._faixa_global.grid(row=0, column=0, sticky='ew')
        self._faixa_global.columnconfigure(list(range(5)), weight=1)

        self.mb_ag     = MetricBlock(self._faixa_global, 'Acerto Global',  '—')
        self.mb_kappa  = MetricBlock(self._faixa_global, 'Kappa',          '—')
        self.mb_interp = MetricBlock(self._faixa_global, 'Interpretacao',  '—')
        self.mb_tau    = MetricBlock(self._faixa_global, 'Tau',            '—')
        self.mb_n      = MetricBlock(self._faixa_global, 'Amostras Teste', '—')
        for i, mb in enumerate([self.mb_ag, self.mb_kappa, self.mb_interp,
                                 self.mb_tau, self.mb_n]):
            mb.grid(row=0, column=i, sticky='ew', padx=(0 if i == 0 else 6, 0))

        # Notebook central com todas as abas
        nb_wrap = tk.Frame(wrap, bg=T.BG)
        nb_wrap.grid(row=1, column=0, sticky='nsew', pady=(10, 0))
        nb_wrap.columnconfigure(0, weight=1)
        nb_wrap.rowconfigure(0, weight=1)

        self.nb = ttk.Notebook(nb_wrap)
        self.nb.grid(row=0, column=0, sticky='nsew')

        self._aba_comp    = tk.Frame(self.nb, bg=T.BG_CARD)
        self._aba_detalhe = tk.Frame(self.nb, bg=T.BG_CARD)
        self._aba_pares   = tk.Frame(self.nb, bg=T.BG_CARD)
        self._aba_matriz  = tk.Frame(self.nb, bg=T.BG_CARD)
        self._aba_graf    = tk.Frame(self.nb, bg=T.BG_CARD)
        self._aba_comp2   = tk.Frame(self.nb, bg=T.BG_CARD)

        self.nb.add(self._aba_comp,    text='  Comparativo  ')
        self.nb.add(self._aba_detalhe, text='  Detalhe por Classe  ')
        self.nb.add(self._aba_pares,   text='  Pares de Classes  ')
        self.nb.add(self._aba_matriz,  text='  Matriz de Confusao  ')
        self.nb.add(self._aba_graf,    text='  Grafico  ')
        self.nb.add(self._aba_comp2,   text='  Comparacao K & T  ')

        for aba in [self._aba_comp, self._aba_detalhe, self._aba_pares,
                    self._aba_matriz, self._aba_graf, self._aba_comp2]:
            tk.Label(aba,
                     text='Clique em  "Treinar e Calcular Metricas"  para iniciar.',
                     bg=T.BG_CARD, fg=T.FG_DIM, font=T.FONT_BODY
                    ).place(relx=0.5, rely=0.5, anchor='center')


        # Painel inferior — metricas binarias OvR
        painel_bin = tk.Frame(wrap, bg=T.BG,
                              highlightthickness=1,
                              highlightbackground=T.BORDER)
        painel_bin.grid(row=2, column=0, sticky='ew', pady=(10, 0))
        painel_bin.columnconfigure(list(range(6)), weight=1)
        tk.Label(painel_bin,
                 text='METRICAS BINARIAS  OvR  —  classe selecionada',
                 bg=T.BG, fg=T.ACCENT, font=T.FONT_KICKER, anchor='w'
                ).grid(row=0, column=0, columnspan=6, sticky='w', padx=10, pady=(8, 4))

        self.mb_sens  = MetricBlock(painel_bin, 'Sensibilidade',  '—')
        self.mb_espec = MetricBlock(painel_bin, 'Especificidade', '—')
        self.mb_prec  = MetricBlock(painel_bin, 'Precisao (VPP)', '—')
        self.mb_f1    = MetricBlock(painel_bin, 'F1  (b=1)',      '—')
        self.mb_f2    = MetricBlock(painel_bin, 'F2  (b=2)',      '—')
        self.mb_mcc   = MetricBlock(painel_bin, 'MCC (Matthews)', '—')
        for i, mb in enumerate([self.mb_sens, self.mb_espec, self.mb_prec,
                                 self.mb_f1, self.mb_f2, self.mb_mcc]):
            mb.grid(row=1, column=i, sticky='ew',
                    padx=(0 if i == 0 else 6, 0), pady=(0, 8))

    # -----------------------------------------------------------------------
    # Dados
    # -----------------------------------------------------------------------
    def _carregar_dados(self):
        if not os.path.exists(CAMINHO_DADOS):
            self.lbl_status.configure(
                text=f'Dados nao encontrados:\n{CAMINHO_DADOS}', fg=T.DANGER)
            return
        self.dados = carregar_dados_iris(CAMINHO_DADOS)
        self.dados_treino, self.dados_teste = split_estratificado(
            self.dados, proporcao_treino=0.7, semente=42)
        self.lbl_status.configure(
            text=f'{len(self.dados)} amostras  '
                 f'({len(self.dados_treino)} treino / {len(self.dados_teste)} teste).',
            fg=T.FG_MUTED)

    # -----------------------------------------------------------------------
    # Treinar todos os classificadores
    # -----------------------------------------------------------------------
    def _treinar_tudo(self):
        if not self.dados:
            self._carregar_dados()
        if not self.dados:
            return

        indices = IDX_PETALA if self.var_atributos.get() == 'petalas' else IDX_SEPALA
        self.lbl_status.configure(text='Treinando...', fg=T.ACCENT)
        self.update()

        resultados = {}
        preds_por = {}

        def registrar(nome, preds, gab):
            resultados[nome] = relatorio_completo(preds, gab, CLASSES, nome)
            preds_por[nome]  = (preds, gab)

        p, g = self._pred_dist_minima(indices);        registrar('Dist. Minima', p, g)
        p, g = self._pred_dist_maxima(indices);        registrar('Dist. Maxima', p, g)
        p, g = self._pred_ova_superficie(indices);     registrar('OvA Superficie', p, g)
        p, g = self._pred_perceptron_ova(indices);     registrar('Perceptron OvA', p, g)
        p, g = self._pred_delta_bin_ova(indices);      registrar('Delta Bin. OvA', p, g)
        p, g = self._pred_delta_ova(indices);          registrar('Delta OvA', p, g)

        self.resultados       = resultados
        self.preds_por_modelo = preds_por
        self._treinado        = True
        self._indices_usados  = indices

        nomes = list(resultados.keys())
        self.var_modelo_sel.set(nomes[0])
        self._reconstruir_radios(nomes)
        self._atualizar_metricas_globais()
        self._preencher_comparativo()
        self._preencher_detalhe()
        self._preencher_pares()
        self._preencher_matriz()
        self._desenhar_grafico()
        self._preencher_comparacao_kt()

        self.lbl_status.configure(
            text=f'Concluido. {len(nomes)} classificadores avaliados.',
            fg=T.SUCCESS)

    # -----------------------------------------------------------------------
    # Classificadores
    # -----------------------------------------------------------------------
    def _pred_dist_minima(self, indices):
        proto = treinar(self.dados_treino, indices)
        preds, gab = [], []
        for a in self.dados_teste:
            _, pred = predizer_todas_classes(a['atributos'], proto, indices)
            preds.append(pred); gab.append(a['classe'])
        return preds, gab

    def _pred_dist_maxima(self, indices):
        proto = treinar(self.dados_treino, indices)
        preds, gab = [], []
        for a in self.dados_teste:
            x = [a['atributos'][i] for i in indices]
            dists = {c: distancia_euclidiana(x, proto[c]) for c in CLASSES}
            preds.append(max(dists, key=dists.get))
            gab.append(a['classe'])
        return preds, gab

    def _pred_ova_superficie(self, indices):
        proto = treinar(self.dados_treino, indices)
        preds, gab = [], []
        for a in self.dados_teste:
            votos = {c: 0 for c in CLASSES}
            for ci, cj in PARES:
                venc = predizer_binario(a['atributos'],
                                        proto[ci], proto[cj], ci, cj, indices)
                votos[venc] += 1
            preds.append(max(votos, key=votos.get))
            gab.append(a['classe'])
        return preds, gab

    def _pred_perceptron_ova(self, indices):
        pesos = {}
        for cp, cn in PARES:
            treino_par = filtrar_por_classes(self.dados_treino, [cp, cn])
            w, _, _ = treinar_perceptron(treino_par, cp, cn, indices, 0.03, 200)
            pesos[(cp, cn)] = (w, cp, cn)
        preds, gab = [], []
        for a in self.dados_teste:
            votos = {c: 0 for c in CLASSES}
            for (w, cp, cn) in pesos.values():
                x = [1.0] + [a['atributos'][i] for i in indices]
                net = sum(wi * xi for wi, xi in zip(w, x))
                votos[cp if net >= 0 else cn] += 1
            preds.append(max(votos, key=votos.get))
            gab.append(a['classe'])
        return preds, gab

    def _pred_delta_bin_ova(self, indices):
        pesos = {}
        for cp, cn in PARES:
            treino_par = filtrar_por_classes(self.dados_treino, [cp, cn])
            w, _, _ = treinar_delta_iris(treino_par, cp, cn, indices, 0.02, 300)
            pesos[(cp, cn)] = (w, cp, cn)
        preds, gab = [], []
        for a in self.dados_teste:
            votos = {c: 0 for c in CLASSES}
            for (w, cp, cn) in pesos.values():
                x = [1.0] + [a['atributos'][i] for i in indices]
                net = sum(wi * xi for wi, xi in zip(w, x))
                votos[cp if net >= 0 else cn] += 1
            preds.append(max(votos, key=votos.get))
            gab.append(a['classe'])
        return preds, gab

    def _pred_delta_ova(self, indices):
        pesos, _, _ = treinar_delta_ova(self.dados_treino, indices, 0.02, 300)
        preds, gab = [], []
        for a in self.dados_teste:
            x = [a['atributos'][i] for i in indices]
            pred, _ = predizer_delta_ova(x, pesos)
            preds.append(pred); gab.append(a['classe'])
        return preds, gab

    # -----------------------------------------------------------------------
    # Helpers UI
    # -----------------------------------------------------------------------
    def _reconstruir_radios(self, nomes):
        for w in self._frame_radios_modelo.winfo_children():
            w.destroy()
        for nome in nomes:
            tk.Radiobutton(
                self._frame_radios_modelo,
                text=nome, value=nome,
                variable=self.var_modelo_sel,
                bg=T.BG_CARD, fg=T.FG, selectcolor=T.BG_HOVER,
                activebackground=T.BG_CARD, activeforeground=T.ACCENT,
                font=T.FONT_BODY, anchor='w',
                borderwidth=0, highlightthickness=0,
                command=self._ao_trocar_modelo,
            ).pack(fill='x', pady=1)

    def _ao_trocar_modelo(self):
        self._atualizar_metricas_globais()
        self._preencher_detalhe()
        self._preencher_pares()
        self._preencher_matriz()

    def _abrir_memoria_metricas(self):
        """Abre janela de memoria de calculo das metricas do modelo selecionado."""
        if not self._treinado or not self.resultados:
            self.lbl_status.configure(
                text='Treine os modelos primeiro.', fg=T.DANGER)
            return
        nome = self.var_modelo_sel.get()
        if not nome or nome not in self.resultados:
            return

        # Se houver Perceptron OvA e Delta OvA, passa o par para o teste Z
        perc = self.resultados.get('Perceptron OvA')
        delt = self.resultados.get('Delta OvA')
        perc_vs_delta = (perc, delt) if (perc and delt) else None

        JanelaMemoriaCalculoMetricas(
            self,
            nome_modelo=nome,
            relatorio=self.resultados[nome],
            classes=CLASSES,
            classe_foco=self.var_classe_sel.get(),
            perc_vs_delta=perc_vs_delta,
        )

    def _atualizar_metricas_globais(self):
        nome = self.var_modelo_sel.get()
        if not nome or nome not in self.resultados:
            return
        rel = self.resultados[nome]
        ag = rel['acerto_global']
        k  = rel['kappa']
        t  = rel['tau']
        m  = sum(rel['matriz'][r][p] for r in CLASSES for p in CLASSES)

        self.mb_ag.set(f'{ag*100:.2f}%',
                       T.SUCCESS if ag >= 0.9 else T.ACCENT if ag >= 0.7 else T.DANGER)
        self.mb_kappa.set(f'{k:.4f}',
                          T.SUCCESS if k > 0.80 else T.ACCENT if k > 0.40 else T.DANGER)
        self.mb_interp.set(interpretar_kappa(k),
                           T.SUCCESS if k > 0.80 else T.ACCENT if k > 0.40 else T.DANGER)
        self.mb_tau.set(f'{t:.4f}',
                        T.SUCCESS if t > 0.80 else T.ACCENT if t > 0.40 else T.DANGER)
        self.mb_n.set(str(m))

        # Metricas OvR binarias
        classe = self.var_classe_sel.get()
        pc = rel['por_classe'].get(classe, {})
        if pc:
            def c_m(v): return T.SUCCESS if v >= 0.9 else T.ACCENT if v >= 0.7 else T.DANGER
            self.mb_sens.set( f'{pc["sensibilidade"]*100:.2f}%',  c_m(pc["sensibilidade"]))
            self.mb_espec.set(f'{pc["especificidade"]*100:.2f}%', c_m(pc["especificidade"]))
            self.mb_prec.set( f'{pc["precisao"]*100:.2f}%',       c_m(pc["precisao"]))
            self.mb_f1.set(   f'{pc["f1"]:.4f}',                  c_m(pc["f1"]))
            self.mb_f2.set(   f'{pc["f2"]:.4f}',                  c_m(pc["f2"]))
            mv = pc["mcc"]
            self.mb_mcc.set(  f'{mv:.4f}',
                              T.SUCCESS if mv > 0.8 else T.ACCENT if mv > 0.4 else T.DANGER)

    # -----------------------------------------------------------------------
    # Aba Comparativo
    # -----------------------------------------------------------------------
    def _preencher_comparativo(self):
        for w in self._aba_comp.winfo_children():
            w.destroy()

        frame = tk.Frame(self._aba_comp, bg=T.BG_CARD)
        frame.pack(fill='both', expand=True, padx=8, pady=6)

        colunas  = ['Modelo', 'Acerto Global', 'Kappa', 'Tau',
                    'set AP', 'ver AP', 'vir AP',
                    'set AU', 'ver AU', 'vir AU']
        larguras = [120, 100, 80, 80, 65, 65, 65, 65, 65, 65]

        canvas   = tk.Canvas(frame, bg=T.BG_CARD, highlightthickness=0)
        scroll_y = ttk.Scrollbar(frame, orient='vertical',   command=canvas.yview)
        scroll_x = ttk.Scrollbar(frame, orient='horizontal', command=canvas.xview)
        inner    = tk.Frame(canvas, bg=T.BG_CARD)
        inner.bind('<Configure>',
                   lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        canvas.pack(side='left', fill='both', expand=True)
        scroll_y.pack(side='right', fill='y')
        scroll_x.pack(side='bottom', fill='x')

        def cel(row, col, texto, cor=T.FG, bg=T.BG_PANEL, larg=80, bold=False):
            f = ('Consolas', 9, 'bold') if bold else T.FONT_MONO_SM
            tk.Label(inner, text=texto, bg=bg, fg=cor, font=f,
                     width=larg // 8, anchor='center',
                     highlightthickness=1, highlightbackground=T.BORDER
                    ).grid(row=row, column=col, sticky='nsew', padx=1, pady=1)

        for j, (nm, larg) in enumerate(zip(colunas, larguras)):
            cel(0, j, nm, cor=T.ACCENT, larg=larg, bold=True)

        for i, (nome, rel) in enumerate(self.resultados.items()):
            ag = rel['acerto_global']
            k  = rel['kappa']
            t  = rel['tau']
            bg = T.BG_CARD if i % 2 == 0 else T.BG_PANEL
            cor_ag = T.SUCCESS if ag >= 0.9 else T.ACCENT if ag >= 0.7 else T.DANGER
            cor_k  = T.SUCCESS if k > 0.80 else T.ACCENT if k > 0.40 else T.DANGER

            vals  = [nome, f'{ag*100:.2f}%', f'{k:.4f}', f'{t:.4f}']
            cores = [T.FG, cor_ag, cor_k, cor_k]
            for c in CLASSES:
                pc = rel['por_classe'][c]
                vals.append(f'{pc["acuracia_produtor"]*100:.1f}%')
                cores.append(CORES_CLASSE[c])
            for c in CLASSES:
                pc = rel['por_classe'][c]
                vals.append(f'{pc["acuracia_usuario"]*100:.1f}%')
                cores.append(CORES_CLASSE[c])

            for j, (v, cor, larg) in enumerate(zip(vals, cores, larguras)):
                cel(i + 1, j, v, cor=cor, bg=bg, larg=larg)

    # -----------------------------------------------------------------------
    # Aba Detalhe por Classe (OvR)
    # -----------------------------------------------------------------------
    def _preencher_detalhe(self):
        for w in self._aba_detalhe.winfo_children():
            w.destroy()
        nome = self.var_modelo_sel.get()
        if not nome or nome not in self.resultados:
            return
        rel = self.resultados[nome]

        outer = tk.Frame(self._aba_detalhe, bg=T.BG_CARD)
        outer.pack(fill='both', expand=True, padx=10, pady=8)

        tk.Label(outer,
                 text=f'Modelo: {nome}  |  Metricas por Classe — visao OvR (One-vs-Rest)',
                 bg=T.BG_CARD, fg=T.ACCENT, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(0, 6))

        cols  = ['Classe', 'Ac.Prod (Sens)', 'Ac.Usu (Prec)',
                 'F1 (b=1)', 'F2 (b=2)', 'MCC', 'VP', 'FP', 'FN', 'VN']
        largs = [90, 110, 110, 80, 80, 80, 40, 40, 40, 40]

        canvas = tk.Canvas(outer, bg=T.BG_CARD, highlightthickness=0, height=130)
        inner  = tk.Frame(canvas, bg=T.BG_CARD)
        inner.bind('<Configure>',
                   lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.pack(fill='x')

        def cel(row, col, texto, cor=T.FG, bold=False):
            bg = T.BG_PANEL if row == 0 else (T.BG_CARD if row % 2 else T.BG)
            f  = ('Consolas', 9, 'bold') if bold else T.FONT_MONO_SM
            tk.Label(inner, text=texto, bg=bg, fg=cor, font=f,
                     anchor='center', width=largs[col] // 8,
                     highlightthickness=1, highlightbackground=T.BORDER
                    ).grid(row=row, column=col, sticky='nsew', padx=1, pady=1)

        for j, nm in enumerate(cols):
            cel(0, j, nm, cor=T.ACCENT, bold=True)

        for i, c in enumerate(CLASSES):
            pc  = rel['por_classe'][c]
            cor = CORES_CLASSE[c]
            for j, (v, cr) in enumerate(zip(
                [c.capitalize(),
                 f'{pc["acuracia_produtor"]*100:.2f}%',
                 f'{pc["acuracia_usuario"]*100:.2f}%',
                 f'{pc["f1"]:.4f}', f'{pc["f2"]:.4f}', f'{pc["mcc"]:.4f}',
                 str(pc['vp']), str(pc['fp']), str(pc['fn']), str(pc['vn'])],
                [cor] + [T.FG] * 9
            )):
                cel(i + 1, j, v, cor=cr)

        # Bloco de metricas globais
        tk.Frame(outer, bg=T.BORDER, height=1).pack(fill='x', pady=(10, 6))
        fg = tk.Frame(outer, bg=T.BG_CARD)
        fg.pack(fill='x')
        ag = rel['acerto_global']
        k  = rel['kappa']
        t  = rel['tau']

        for col, (rot, val, cor) in enumerate([
            ('Acerto Global', f'{ag*100:.4f}%',
             T.SUCCESS if ag >= 0.9 else T.ACCENT),
            ('Kappa', f'{k:.6f}',
             T.SUCCESS if k > 0.80 else T.ACCENT if k > 0.40 else T.DANGER),
            ('Interpretacao', interpretar_kappa(k),
             T.SUCCESS if k > 0.80 else T.ACCENT if k > 0.40 else T.DANGER),
            ('Tau', f'{t:.6f}',
             T.SUCCESS if t > 0.80 else T.ACCENT if t > 0.40 else T.DANGER),
            ('Var(Kappa)', f'{rel["variancia_kappa"]:.6f}', T.FG_MUTED),
            ('Var(Tau)',   f'{rel["variancia_tau"]:.6f}',   T.FG_MUTED),
        ]):
            bloco = tk.Frame(fg, bg=T.BG_CARD,
                             highlightthickness=1, highlightbackground=T.BORDER)
            bloco.grid(row=0, column=col, sticky='ew',
                       padx=(0 if col == 0 else 4, 0))
            fg.columnconfigure(col, weight=1)
            tk.Label(bloco, text=rot.upper(), bg=T.BG_CARD, fg=T.ACCENT,
                     font=T.FONT_KICKER, anchor='w').pack(
                fill='x', padx=8, pady=(6, 0))
            tk.Label(bloco, text=val, bg=T.BG_CARD, fg=cor,
                     font=T.FONT_MONO, anchor='w').pack(
                fill='x', padx=8, pady=(2, 6))

    # -----------------------------------------------------------------------
    # Aba Pares de Classes — MCC e Fb por par (set×ver, ver×vir, set×vir)
    # -----------------------------------------------------------------------
    def _preencher_pares(self):
        for w in self._aba_pares.winfo_children():
            w.destroy()
        nome = self.var_modelo_sel.get()
        if not nome or nome not in self.resultados:
            return
        rel = self.resultados[nome]

        outer = tk.Frame(self._aba_pares, bg=T.BG_CARD)
        outer.pack(fill='both', expand=True, padx=10, pady=8)

        tk.Label(outer,
                 text=f'Modelo: {nome}  |  MCC e Fb Score — problema DUAS CLASSES (por par)',
                 bg=T.BG_CARD, fg=T.ACCENT, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(0, 4))

        tk.Label(outer,
                 text='Cada par e tratado como classificacao binaria pura:\n'
                      '  VP = acertos da classe i  |  VN = acertos da classe j  '
                      '|  FP/FN = trocas entre i e j',
                 bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_LABEL,
                 justify='left', anchor='w'
                ).pack(fill='x', pady=(0, 8))

        # Para cada par, extrai VP/FP/FN/VN da matriz APENAS com as duas classes
        for ci, cj in PARES:
            mat = rel['matriz']
            # Visao binaria pura: so as duas classes do par
            vp = mat[ci][ci]
            fn = mat[ci][cj]
            fp = mat[cj][ci]
            vn = mat[cj][cj]
            total_par = vp + fn + fp + vn

            f1  = fb_score(vp, fp, fn, b=1)
            f2  = fb_score(vp, fp, fn, b=2)
            mv  = mcc_fn(vp, vn, fp, fn)
            sens = vp / (vp + fn) if (vp + fn) > 0 else 0.0
            prec = vp / (vp + fp) if (vp + fp) > 0 else 0.0
            ac   = (vp + vn) / total_par if total_par > 0 else 0.0

            cor_i = CORES_CLASSE[ci]
            cor_j = CORES_CLASSE[cj]

            bloco = tk.Frame(outer, bg=T.BG_PANEL,
                             highlightthickness=1, highlightbackground=T.BORDER)
            bloco.pack(fill='x', pady=(0, 8))

            # Titulo do par
            tit = tk.Frame(bloco, bg=T.BG_PANEL)
            tit.pack(fill='x', padx=10, pady=(6, 4))
            tk.Label(tit, text=ROTULO_PAR[(ci, cj)],
                     bg=T.BG_PANEL, fg=T.FG,
                     font=('Cambria', 11, 'bold'), anchor='w').pack(side='left')
            tk.Label(tit, text=f'  ({total_par} amostras de teste)',
                     bg=T.BG_PANEL, fg=T.FG_MUTED,
                     font=T.FONT_LABEL, anchor='w').pack(side='left')

            # Mini matriz 2x2
            grid = tk.Frame(bloco, bg=T.BG_PANEL)
            grid.pack(side='left', padx=10, pady=(0, 8))

            def cel2(row, col, texto, bg=T.BG_CARD, fg=T.FG, bold=False):
                f = ('Consolas', 9, 'bold') if bold else T.FONT_MONO_SM
                tk.Label(grid, text=texto, bg=bg, fg=fg, font=f,
                         width=10, anchor='center',
                         highlightthickness=1, highlightbackground=T.BORDER
                        ).grid(row=row, column=col, padx=1, pady=1, sticky='nsew')

            cel2(0, 0, 'Real \\ Pred', bg=T.BG_PANEL, fg=T.FG_MUTED)
            cel2(0, 1, ci.capitalize(), bg=cor_i, fg='white', bold=True)
            cel2(0, 2, cj.capitalize(), bg=cor_j, fg='white', bold=True)
            cel2(1, 0, ci.capitalize(), bg=cor_i, fg='white', bold=True)
            cel2(2, 0, cj.capitalize(), bg=cor_j, fg='white', bold=True)

            # VP (diagonal = acerto da classe ci)
            bg_vp = '#1a7f37' if vp > 0 else T.BG_CARD
            cel2(1, 1, f'VP = {vp}', bg=bg_vp, fg='white' if vp > 0 else T.FG)
            # FN (ci predito como cj)
            bg_fn = '#cf222e' if fn > 0 else T.BG_CARD
            cel2(1, 2, f'FN = {fn}', bg=bg_fn, fg='white' if fn > 0 else T.FG)
            # FP (cj predito como ci)
            bg_fp = '#cf222e' if fp > 0 else T.BG_CARD
            cel2(2, 1, f'FP = {fp}', bg=bg_fp, fg='white' if fp > 0 else T.FG)
            # VN (diagonal = acerto da classe cj)
            bg_vn = '#1a7f37' if vn > 0 else T.BG_CARD
            cel2(2, 2, f'VN = {vn}', bg=bg_vn, fg='white' if vn > 0 else T.FG)

            # Metricas do par
            metr = tk.Frame(bloco, bg=T.BG_PANEL)
            metr.pack(side='left', fill='x', expand=True, padx=10, pady=(0, 8))

            def bloquinho(parent, rotulo, valor, cor):
                b = tk.Frame(parent, bg=T.BG_CARD,
                             highlightthickness=1, highlightbackground=T.BORDER)
                b.pack(side='left', padx=(0, 6), ipadx=6, ipady=4)
                tk.Label(b, text=rotulo.upper(), bg=T.BG_CARD, fg=T.ACCENT,
                         font=T.FONT_KICKER, anchor='w').pack(
                    fill='x', padx=6, pady=(4, 0))
                tk.Label(b, text=valor, bg=T.BG_CARD, fg=cor,
                         font=T.FONT_HEADLINE, anchor='w').pack(
                    fill='x', padx=6, pady=(0, 4))

            c_mv = T.SUCCESS if mv > 0.8 else T.ACCENT if mv > 0.4 else T.DANGER
            c_f  = T.SUCCESS if f1 > 0.9 else T.ACCENT if f1 > 0.7 else T.DANGER

            bloquinho(metr, 'Acuracia',   f'{ac*100:.2f}%',
                      T.SUCCESS if ac >= 0.9 else T.ACCENT)
            bloquinho(metr, 'Sens.',      f'{sens*100:.2f}%',
                      T.SUCCESS if sens >= 0.9 else T.ACCENT)
            bloquinho(metr, 'Precisao',   f'{prec*100:.2f}%',
                      T.SUCCESS if prec >= 0.9 else T.ACCENT)
            bloquinho(metr, 'F1  (b=1)',  f'{f1:.4f}', c_f)
            bloquinho(metr, 'F2  (b=2)',  f'{f2:.4f}', c_f)
            bloquinho(metr, 'MCC',        f'{mv:.4f}',  c_mv)

    # -----------------------------------------------------------------------
    # Aba Matriz de Confusao
    # -----------------------------------------------------------------------
    def _preencher_matriz(self):
        for w in self._aba_matriz.winfo_children():
            w.destroy()
        nome = self.var_modelo_sel.get()
        if not nome or nome not in self.resultados:
            return
        rel    = self.resultados[nome]
        matriz = rel['matriz']

        outer = tk.Frame(self._aba_matriz, bg=T.BG_CARD)
        outer.pack(fill='both', expand=True, padx=12, pady=8)

        tk.Label(outer,
                 text=f'Matriz de Confusao  —  {nome}  |  '
                      f'Linhas = Real  ·  Colunas = Predito',
                 bg=T.BG_CARD, fg=T.ACCENT, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(0, 8))

        grid  = tk.Frame(outer, bg=T.BG_CARD)
        grid.pack(anchor='w')
        vals  = [[matriz[r][p] for p in CLASSES] for r in CLASSES]
        v_max = max(max(l) for l in vals) or 1

        def bg_cel(v, diag):
            t = v / v_max
            if diag:
                r = int(255 * (1 - 0.6 * t)); g = int(255 * (1 - 0.4 * t)); b = 255
            else:
                r = 255; g = int(255 * (1 - 0.7 * t)); b = int(255 * (1 - 0.7 * t))
            return f'#{r:02x}{g:02x}{b:02x}'

        tk.Label(grid, text='Real \\ Pred', bg=T.BG_PANEL, fg=T.FG_MUTED,
                 font=T.FONT_KICKER, width=10, anchor='center',
                 highlightthickness=1, highlightbackground=T.BORDER
                ).grid(row=0, column=0, padx=2, pady=2)

        for j, c in enumerate(CLASSES):
            tk.Label(grid, text=c.capitalize(), bg=CORES_CLASSE[c],
                     fg='white', font=T.FONT_KICKER, width=10, anchor='center',
                     highlightthickness=1, highlightbackground=T.BORDER
                    ).grid(row=0, column=j + 1, padx=2, pady=2)

        for i, real in enumerate(CLASSES):
            tk.Label(grid, text=real.capitalize(), bg=CORES_CLASSE[real],
                     fg='white', font=T.FONT_KICKER, width=10, anchor='center',
                     highlightthickness=1, highlightbackground=T.BORDER
                    ).grid(row=i + 1, column=0, padx=2, pady=2)
            for j, pred in enumerate(CLASSES):
                v   = matriz[real][pred]
                bg  = bg_cel(v, i == j)
                cfg = 'white' if (v / v_max > 0.4) else T.FG
                tk.Label(grid, text=str(v), bg=bg, fg=cfg,
                         font=('Consolas', 11, 'bold'), width=10, anchor='center',
                         highlightthickness=1, highlightbackground=T.BORDER
                        ).grid(row=i + 1, column=j + 1, padx=2, pady=2)

        # Acuracia produtor e usuario
        info = tk.Frame(outer, bg=T.BG_CARD)
        info.pack(fill='x', pady=(12, 0))
        tk.Label(info, text='ACURACIA DO PRODUTOR', bg=T.BG_CARD, fg=T.ACCENT,
                 font=T.FONT_KICKER, anchor='w'
                ).grid(row=0, column=0, columnspan=3, sticky='w', pady=(0, 4))
        tk.Label(info, text='ACURACIA DO USUARIO', bg=T.BG_CARD, fg=T.ACCENT,
                 font=T.FONT_KICKER, anchor='w'
                ).grid(row=0, column=3, columnspan=3, sticky='w', pady=(0, 4),
                       padx=(20, 0))
        for col, c in enumerate(CLASSES):
            pc  = rel['por_classe'][c]
            cor = CORES_CLASSE[c]
            for offset, chave in [(0, 'acuracia_produtor'), (3, 'acuracia_usuario')]:
                b = tk.Frame(info, bg=T.BG_PANEL,
                             highlightthickness=1, highlightbackground=T.BORDER)
                b.grid(row=1, column=col + offset, sticky='ew',
                       padx=(20 if (col == 0 and offset == 3) else
                              (0 if col == 0 else 4), 0))
                info.columnconfigure(col + offset, weight=1)
                tk.Label(b, text=c.capitalize(), bg=T.BG_PANEL, fg=cor,
                         font=T.FONT_KICKER, anchor='w').pack(
                    fill='x', padx=6, pady=(4, 0))
                tk.Label(b, text=f'{pc[chave]*100:.2f}%',
                         bg=T.BG_PANEL, fg=cor,
                         font=T.FONT_HEADLINE, anchor='w').pack(
                    fill='x', padx=6, pady=(0, 4))

    # -----------------------------------------------------------------------
    # Aba Grafico
    # -----------------------------------------------------------------------
    def _desenhar_grafico(self):
        for w in self._aba_graf.winfo_children():
            w.destroy()
        if not self.resultados:
            return

        fig = Figure(figsize=(9, 3.8), dpi=95, facecolor=T.BG_CARD)
        fig.subplots_adjust(left=0.06, right=0.98, bottom=0.22, top=0.85, wspace=0.3)
        ax1 = fig.add_subplot(1, 3, 1)
        ax2 = fig.add_subplot(1, 3, 2)
        ax3 = fig.add_subplot(1, 3, 3)

        nomes  = list(self.resultados.keys())
        ag_lst = [self.resultados[n]['acerto_global'] * 100 for n in nomes]
        k_lst  = [self.resultados[n]['kappa']          for n in nomes]
        t_lst  = [self.resultados[n]['tau']             for n in nomes]
        xs     = range(len(nomes))

        def estilizar(ax, titulo, ymin=0, ymax=1.15):
            ax.set_facecolor(T.BG_PANEL)
            ax.tick_params(colors=T.FG_MUTED, labelsize=7)
            for s in ax.spines.values():
                s.set_color(T.BORDER)
            ax.grid(axis='y', color=T.BORDER, linewidth=0.5, alpha=0.6)
            ax.set_title(titulo, color=T.FG, fontsize=9, pad=6,
                         fontfamily='Cambria', fontweight='bold')
            ax.set_xticks(xs)
            ax.set_xticklabels([n.replace(' ', '\n') for n in nomes],
                               fontsize=6.5, color=T.FG_MUTED)
            ax.set_ylim(ymin, ymax)

        c_ag = [T.SUCCESS if v >= 90 else T.ACCENT if v >= 70 else T.DANGER for v in ag_lst]
        ax1.bar(xs, ag_lst, color=c_ag, edgecolor=T.BG_PANEL, linewidth=0.5, zorder=3)
        for x, v in zip(xs, ag_lst):
            ax1.text(x, v + 0.5, f'{v:.1f}%', ha='center', va='bottom',
                     fontsize=6.5, color=T.FG_MUTED)
        ax1.set_ylabel('Acerto (%)', fontsize=7, color=T.FG_MUTED)
        estilizar(ax1, 'Acerto Global', ymin=0, ymax=115)

        c_k = [T.SUCCESS if v > 0.80 else T.ACCENT if v > 0.40 else T.DANGER for v in k_lst]
        ax2.bar(xs, k_lst, color=c_k, edgecolor=T.BG_PANEL, linewidth=0.5, zorder=3)
        for x, v in zip(xs, k_lst):
            ax2.text(x, max(v, 0) + 0.01, f'{v:.3f}', ha='center', va='bottom',
                     fontsize=6.5, color=T.FG_MUTED)
        ax2.axhline(0.80, color=T.SUCCESS, ls='--', lw=0.8, alpha=0.6)
        ax2.axhline(0.40, color=T.ACCENT,  ls='--', lw=0.8, alpha=0.6)
        ax2.set_ylabel('Kappa', fontsize=7, color=T.FG_MUTED)
        estilizar(ax2, 'Coeficiente Kappa', ymin=-0.2)

        c_t = [T.SUCCESS if v > 0.80 else T.ACCENT if v > 0.40 else T.DANGER for v in t_lst]
        ax3.bar(xs, t_lst, color=c_t, edgecolor=T.BG_PANEL, linewidth=0.5, zorder=3)
        for x, v in zip(xs, t_lst):
            ax3.text(x, max(v, 0) + 0.01, f'{v:.3f}', ha='center', va='bottom',
                     fontsize=6.5, color=T.FG_MUTED)
        ax3.axhline(0.80, color=T.SUCCESS, ls='--', lw=0.8, alpha=0.6)
        ax3.set_ylabel('Tau', fontsize=7, color=T.FG_MUTED)
        estilizar(ax3, 'Coeficiente Tau', ymin=-0.3)

        canvas = FigureCanvasTkAgg(fig, master=self._aba_graf)
        canvas.get_tk_widget().pack(fill='both', expand=True, padx=6, pady=6)
        canvas.draw()

    # -----------------------------------------------------------------------
    # Aba Comparacao Kappa & Tau — Perceptron vs Delta  (Item 2 do Lab 3)
    # -----------------------------------------------------------------------
    def _preencher_comparacao_kt(self):
        for w in self._aba_comp2.winfo_children():
            w.destroy()

        outer = tk.Frame(self._aba_comp2, bg=T.BG_CARD)
        outer.pack(fill='both', expand=True, padx=14, pady=10)

        tk.Label(outer,
                 text='ITEM 2 — Teste de Significancia: Perceptron OvA  vs  Delta OvA',
                 bg=T.BG_CARD, fg=T.ACCENT, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(0, 4))
        tk.Label(outer,
                 text='Verifica se a diferenca entre os dois classificadores e '
                      'estatisticamente significativa ao nivel de 5%.\n'
                      'H0: nao ha diferenca entre os coeficientes  |  '
                      'H1: ha diferenca',
                 bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_LABEL,
                 justify='left', anchor='w', wraplength=820
                ).pack(fill='x', pady=(0, 10))

        # Verificar se ambos os modelos existem
        perc = self.resultados.get('Perceptron OvA')
        delt = self.resultados.get('Delta OvA')
        if not perc or not delt:
            tk.Label(outer, text='Treine os modelos primeiro.',
                     bg=T.BG_CARD, fg=T.DANGER, font=T.FONT_BODY).pack()
            return

        k1  = perc['kappa'];          k2  = delt['kappa']
        t1  = perc['tau'];            t2  = delt['tau']
        vk1 = perc['variancia_kappa']; vk2 = delt['variancia_kappa']
        vt1 = perc['variancia_tau'];   vt2 = delt['variancia_tau']
        ag1 = perc['acerto_global'];   ag2 = delt['acerto_global']

        zk  = z_kappa(k1, vk1, k2, vk2)
        zt  = z_tau(t1, vt1, t2, vt2)
        pzk = p_valor_z(zk)
        pzt = p_valor_z(zt)
        sig_k = pzk < 0.05
        sig_t = pzt < 0.05

        # Tabela comparativa
        tab = tk.Frame(outer, bg=T.BG_CARD)
        tab.pack(fill='x', pady=(0, 14))

        def th(col, texto):
            tk.Label(tab, text=texto, bg=T.BG_PANEL, fg=T.ACCENT,
                     font=('Consolas', 9, 'bold'), anchor='center',
                     width=18, highlightthickness=1,
                     highlightbackground=T.BORDER
                    ).grid(row=0, column=col, padx=1, pady=1, sticky='nsew')

        def td(row, col, texto, cor=T.FG):
            bg = T.BG_CARD if row % 2 else T.BG_PANEL
            tk.Label(tab, text=texto, bg=bg, fg=cor,
                     font=T.FONT_MONO_SM, anchor='center',
                     width=18, highlightthickness=1,
                     highlightbackground=T.BORDER
                    ).grid(row=row, column=col, padx=1, pady=1, sticky='nsew')

        for col, h in enumerate(['Metrica', 'Perceptron OvA', 'Delta OvA',
                                  'Z calculado', 'p-valor', 'Conclusao (5%)']):
            th(col, h)

        linhas = [
            ('Acerto Global',
             f'{ag1*100:.2f}%', f'{ag2*100:.2f}%',
             '—', '—',
             'Maior: ' + ('Perceptron' if ag1 > ag2 else 'Delta OvA')),
            ('Kappa',
             f'{k1:.6f}', f'{k2:.6f}',
             f'{zk:.4f}', f'{pzk:.4f}',
             'SIGNIFICATIVO' if sig_k else 'nao significativo'),
            ('Tau',
             f'{t1:.6f}', f'{t2:.6f}',
             f'{zt:.4f}', f'{pzt:.4f}',
             'SIGNIFICATIVO' if sig_t else 'nao significativo'),
            ('Var(Kappa)',
             f'{vk1:.6f}', f'{vk2:.6f}', '—', '—', '—'),
            ('Var(Tau)',
             f'{vt1:.6f}', f'{vt2:.6f}', '—', '—', '—'),
        ]

        for r, (m, v1, v2, z, p, conc) in enumerate(linhas):
            cor_conc = (T.DANGER if 'SIGNIFICATIVO' in conc
                        else T.SUCCESS if 'nao' in conc else T.FG_MUTED)
            td(r + 1, 0, m, T.FG_MUTED)
            td(r + 1, 1, v1)
            td(r + 1, 2, v2)
            td(r + 1, 3, z)
            td(r + 1, 4, p)
            td(r + 1, 5, conc, cor_conc)

        # Interpretacao
        tk.Frame(outer, bg=T.BORDER, height=1).pack(fill='x', pady=(4, 8))

        maior_acc = 'Perceptron OvA' if ag1 > ag2 else ('Delta OvA' if ag2 > ag1 else 'empate')
        maior_k   = 'Perceptron OvA' if k1 > k2 else ('Delta OvA' if k2 > k1 else 'empate')

        texto_interp = (
            f'Maior acuracia:  {maior_acc}  '
            f'(Perceptron {ag1*100:.2f}%  vs  Delta {ag2*100:.2f}%)\n'
            f'Maior Kappa:     {maior_k}  '
            f'(Perceptron K={k1:.4f}  vs  Delta K={k2:.4f})\n\n'
        )
        if sig_k or sig_t:
            texto_interp += (
                f'Conclusao: a diferenca entre os classificadores e '
                f'ESTATISTICAMENTE SIGNIFICATIVA (p < 0.05).\n'
                f'Isso indica que o desempenho superior de um deles '
                f'nao e resultado do acaso.\n'
                f'Rejeita-se H0: os dois classificadores diferem entre si.'
            )
        else:
            texto_interp += (
                f'Conclusao: a diferenca entre os classificadores NAO e '
                f'estatisticamente significativa (p >= 0.05).\n'
                f'Nao ha evidencia suficiente para rejeitar H0.\n'
                f'Os dois classificadores tem desempenho equivalente '
                f'para este conjunto de dados e atributos.'
            )

        cor_txt = T.DANGER if (sig_k or sig_t) else T.SUCCESS
        tk.Label(outer, text=texto_interp,
                 bg=T.BG_CARD, fg=cor_txt, font=T.FONT_MONO_SM,
                 justify='left', anchor='w', wraplength=820
                ).pack(fill='x')

    # -----------------------------------------------------------------------