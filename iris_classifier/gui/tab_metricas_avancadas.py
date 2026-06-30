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

from data.data_loader import carregar_dados_iris, split_estratificado, filtrar_por_classes
from models.classifier  import treinar, predizer_todas_classes, predizer_binario
from core.math_utils  import distancia_euclidiana
from models.perceptron  import treinar_perceptron, predizer_perceptron
from models.delta_rule  import (treinar_delta_iris, treinar_delta_ova, predizer_delta_ova)
from evaluation.metricas_avancadas import (
    relatorio_completo, relatorio_binario,
    kappa, tau, variancia_kappa, variancia_tau,
    z_kappa, z_tau, p_valor_z,
    matriz_binaria_ovr, kappa_por_classe, z_classes,
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
        self.var_prop_treino = tk.StringVar(value='0.70')
        self.var_semente     = tk.StringVar(value='42')
        self.var_comparacao  = tk.StringVar(value='todas')
        self.var_comp_modelo1 = tk.StringVar(value='')
        self.var_comp_modelo2 = tk.StringVar(value='')
        self.var_sig_classe_a = tk.StringVar(value='setosa')
        self.var_sig_classe_b = tk.StringVar(value='versicolor')

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
        wrap.grid(row=0, column=0, sticky='nsew', padx=(T.PAD_PAGE, T.GAP), pady=T.PAD_PAGE)
        wrap.columnconfigure(0, weight=1)

        # Atributos
        card = Card(wrap, titulo='atributos do modelo')
        card.grid(row=0, column=0, sticky='ew')
        for val, lbl in [('petalas', 'Petalas  ·  [2,3]'),
                          ('sepalas', 'Sepalas  ·  [0,1]'),
                          ('todas',   'Todas (4 Features) · [0,1,2,3]')]:
            tk.Radiobutton(card, text=lbl, value=val,
                           variable=self.var_atributos,
                           bg=T.BG_CARD, fg=T.FG, selectcolor=T.BG_HOVER,
                           activebackground=T.BG_CARD, activeforeground=T.ACCENT,
                           font=T.FONT_BODY, anchor='w',
                           borderwidth=0, highlightthickness=0
                          ).pack(fill='x', padx=14, pady=2)
        tk.Frame(card, bg=T.BG_CARD, height=4).pack()

        # Divisão dos dados
        card_s = Card(wrap, titulo='divisao dos dados')
        card_s.grid(row=1, column=0, sticky='ew', pady=(8, 0))
        form_s = tk.Frame(card_s, bg=T.BG_CARD)
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

        # Seleção de comparação (Classes)
        card_cc = Card(wrap, titulo='comparacao (classes)')
        card_cc.grid(row=2, column=0, sticky='ew', pady=(8, 0))
        for val, lbl in [
            ('todas', '3 Classes (Todas)'),
            ('setosa_versicolor', 'Setosa × Versicolor'),
            ('versicolor_virginica', 'Versicolor × Virginica'),
            ('setosa_virginica', 'Setosa × Virginica')
        ]:
            tk.Radiobutton(card_cc, text=lbl, value=val,
                           variable=self.var_comparacao,
                           bg=T.BG_CARD, fg=T.FG, selectcolor=T.BG_HOVER,
                           activebackground=T.BG_CARD, activeforeground=T.ACCENT,
                           font=T.FONT_BODY, anchor='w',
                           borderwidth=0, highlightthickness=0,
                           command=self._ao_mudar_comparacao
                          ).pack(fill='x', padx=14, pady=2)

        ttk.Button(wrap, text='Treinar e Calcular Metricas  >',
                   style='Primary.TButton',
                   command=self._treinar_tudo
                  ).grid(row=3, column=0, sticky='ew', pady=(10, 0))

        self.lbl_status = tk.Label(wrap, text='Aguardando treinamento.',
                                   bg=T.BG, fg=T.FG_MUTED,
                                   font=T.FONT_MONO_SM, anchor='w',
                                   wraplength=255, justify='left')
        self.lbl_status.grid(row=4, column=0, sticky='ew', pady=(4, 0))

        # Seletor de modelo
        card2 = Card(wrap, titulo='selecionar modelo')
        card2.grid(row=5, column=0, sticky='ew', pady=(10, 0))
        self._frame_radios_modelo = tk.Frame(card2, bg=T.BG_CARD)
        self._frame_radios_modelo.pack(fill='x', padx=14, pady=(0, 6))

        # Seletor de classe (metricas OvR)
        card3 = Card(wrap, titulo='classe  (OvR)')
        card3.grid(row=6, column=0, sticky='ew', pady=(8, 0))
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
        card4.grid(row=7, column=0, sticky='ew', pady=(8, 0))
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
        card5.grid(row=8, column=0, sticky='ew', pady=(8, 0))
        ttk.Button(card5, text='Abrir memoria de calculo  >',
                   style='Primary.TButton',
                   command=self._abrir_memoria_metricas
                  ).pack(fill='x', padx=14, pady=(2, 10))

        wrap.rowconfigure(9, weight=1)

    def _col_dir(self):
        wrap = tk.Frame(self, bg=T.BG)
        wrap.grid(row=0, column=1, sticky='nsew', padx=(T.GAP, T.PAD_PAGE), pady=T.PAD_PAGE)
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
        self._aba_pr51    = tk.Frame(self.nb, bg=T.BG_CARD)

        self.nb.add(self._aba_comp,    text='  Comparativo  ')
        self.nb.add(self._aba_detalhe, text='  Detalhe por Classe  ')
        self.nb.add(self._aba_pares,   text='  Pares de Classes  ')
        self.nb.add(self._aba_matriz,  text='  Matriz de Confusao  ')
        self.nb.add(self._aba_graf,    text='  Grafico  ')
        self.nb.add(self._aba_comp2,   text='  Comparacao K & T  ')
        self.nb.add(self._aba_pr51,    text='  Exercicios PR51  ')

        for aba in [self._aba_comp, self._aba_detalhe, self._aba_pares,
                    self._aba_matriz, self._aba_graf, self._aba_comp2]:
            tk.Label(aba,
                     text= 'Clique em  "Treinar e Calcular Metricas"  para iniciar.',
                     bg=T.BG_CARD, fg=T.FG_DIM, font=T.FONT_BODY
                    ).place(relx=0.5, rely=0.5, anchor='center')

        # Exercicio PR51 nao depende de treinamento — montar imediatamente
        self._construir_aba_pr51()


        # Painel inferior — metricas binarias OvR
        painel_bin = tk.Frame(wrap, bg=T.BG,
                              highlightthickness=1,
                              highlightbackground=T.BORDER)
        painel_bin.grid(row=2, column=0, sticky='ew', pady=(10, 0))
        painel_bin.columnconfigure(list(range(6)), weight=1)
        tk.Label(painel_bin,
                 text='METRICAS BINARIAS  OvR  —  classe selecionada',
                 bg=T.BG, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
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
        self.lbl_status.configure(
            text=f'{len(self.dados)} amostras  '
                 f'({len(self.dados_treino)} treino / {len(self.dados_teste)} teste).',
            fg=T.FG_MUTED)

    # -----------------------------------------------------------------------
    # Treinar todos os classificadores
    # -----------------------------------------------------------------------
    def _ao_mudar_comparacao(self):
        comp = self.var_comparacao.get()
        if comp == 'setosa_versicolor':
            self.var_classe_sel.set('setosa')
        elif comp == 'versicolor_virginica':
            self.var_classe_sel.set('versicolor')
        elif comp == 'setosa_virginica':
            self.var_classe_sel.set('setosa')
        else:
            self.var_classe_sel.set('setosa')

    def _obter_dados_filtrados(self):
        comp = self.var_comparacao.get()
        if comp == 'todas':
            return self.dados_treino, self.dados_teste, CLASSES
        elif comp == 'setosa_versicolor':
            classes_sel = ['setosa', 'versicolor']
        elif comp == 'versicolor_virginica':
            classes_sel = ['versicolor', 'virginica']
        else:
            classes_sel = ['setosa', 'virginica']
            
        treino = filtrar_por_classes(self.dados_treino, classes_sel)
        teste = filtrar_por_classes(self.dados_teste, classes_sel)
        return treino, teste, classes_sel

    # -----------------------------------------------------------------------
    # Treinar todos os classificadores
    # -----------------------------------------------------------------------
    def _treinar_tudo(self):
        self._carregar_dados()
        if not self.dados:
            return

        attr_sel = self.var_atributos.get()
        if attr_sel == 'petalas':
            indices = [2, 3]
        elif attr_sel == 'sepalas':
            indices = [0, 1]
        else:
            indices = [0, 1, 2, 3]

        self.lbl_status.configure(text='Treinando...', fg=T.ACCENT)
        self.update()

        treino_f, teste_f, classes_sel = self._obter_dados_filtrados()
        resultados = {}
        preds_por = {}

        def registrar(nome, preds, gab):
            resultados[nome] = relatorio_completo(preds, gab, classes_sel, nome)
            preds_por[nome]  = (preds, gab)

        p, g = self._pred_dist_minima(treino_f, teste_f, classes_sel, indices);        registrar('Dist. Minima', p, g)
        p, g = self._pred_dist_maxima(treino_f, teste_f, classes_sel, indices);        registrar('Dist. Maxima', p, g)
        p, g = self._pred_ova_superficie(treino_f, teste_f, classes_sel, indices);     registrar('Superficie 2a2' if len(classes_sel) == 3 else 'Superficie Binaria', p, g)
        p, g = self._pred_perceptron_ova(treino_f, teste_f, classes_sel, indices);     registrar('Perceptron 2a2' if len(classes_sel) == 3 else 'Perceptron Binario', p, g)
        p, g = self._pred_delta_bin_ova(treino_f, teste_f, classes_sel, indices);      registrar('Delta Bin. OvA' if len(classes_sel) == 3 else 'Delta Binario', p, g)
        p, g = self._pred_delta_ova(treino_f, teste_f, classes_sel, indices);          registrar('Delta OvA' if len(classes_sel) == 3 else 'Delta Binario (Nets)', p, g)

        self.resultados       = resultados
        self.preds_por_modelo = preds_por
        self._treinado        = True
        self._indices_usados  = indices
        self._classes_usadas  = classes_sel

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
    def _pred_dist_minima(self, treino, teste, classes, indices):
        proto = treinar(treino, indices)
        preds, gab = [], []
        for a in teste:
            _, pred = predizer_todas_classes(a['atributos'], proto, indices)
            preds.append(pred); gab.append(a['classe'])
        return preds, gab

    def _pred_dist_maxima(self, treino, teste, classes, indices):
        proto = treinar(treino, indices)
        preds, gab = [], []
        for a in teste:
            x = [a['atributos'][i] for i in indices]
            dists = {c: distancia_euclidiana(x, proto[c]) for c in classes}
            preds.append(max(dists, key=dists.get))
            gab.append(a['classe'])
        return preds, gab

    def _pred_ova_superficie(self, treino, teste, classes, indices):
        proto = treinar(treino, indices)
        preds, gab = [], []
        
        if len(classes) == 2:
            pares_locais = [(classes[0], classes[1])]
        else:
            pares_locais = PARES

        for a in teste:
            votos = {c: 0 for c in classes}
            for ci, cj in pares_locais:
                venc = predizer_binario(a['atributos'],
                                        proto[ci], proto[cj], ci, cj, indices)
                votos[venc] += 1
            preds.append(max(votos, key=votos.get))
            gab.append(a['classe'])
        return preds, gab

    def _pred_perceptron_ova(self, treino, teste, classes, indices):
        if len(classes) == 2:
            cp, cn = classes[0], classes[1]
            w, _, _ = treinar_perceptron(treino, cp, cn, indices, 0.03, 200)
            preds, gab = [], []
            for a in teste:
                x = [a['atributos'][i] for i in indices]
                y = predizer_perceptron(x, w)
                preds.append(cp if y == 1 else cn)
                gab.append(a['classe'])
            return preds, gab
        else:
            # Perceptron Hierárquico 2a2: Setosa (A) vs Versicolor/Virginica (B e C)
            treino_s1 = []
            for d in treino:
                lbl = 'setosa' if d['classe'] == 'setosa' else 'not_setosa'
                treino_s1.append({'atributos': d['atributos'], 'classe': lbl})
            w1, _, _ = treinar_perceptron(treino_s1, 'setosa', 'not_setosa', indices, 0.03, 200)
            
            # Segunda superfície: Versicolor vs Virginica
            treino_s2 = filtrar_por_classes(treino, ['versicolor', 'virginica'])
            w2, _, _ = treinar_perceptron(treino_s2, 'versicolor', 'virginica', indices, 0.03, 200)
            
            preds, gab = [], []
            for a in teste:
                x = [a['atributos'][i] for i in indices]
                y1 = predizer_perceptron(x, w1)
                if y1 == 1:
                    preds.append('setosa')
                else:
                    y2 = predizer_perceptron(x, w2)
                    preds.append('versicolor' if y2 == 1 else 'virginica')
                gab.append(a['classe'])
            return preds, gab

    def _pred_delta_bin_ova(self, treino, teste, classes, indices):
        if len(classes) == 2:
            cp, cn = classes[0], classes[1]
            w, _, _ = treinar_delta_iris(treino, cp, cn, indices, 0.02, 300)
            preds, gab = [], []
            for a in teste:
                x = [1.0] + [a['atributos'][i] for i in indices]
                net = sum(wi * xi for wi, xi in zip(w, x))
                preds.append(cp if net >= 0 else cn)
                gab.append(a['classe'])
            return preds, gab
        else:
            pesos = {}
            for cp, cn in PARES:
                treino_par = filtrar_por_classes(treino, [cp, cn])
                w, _, _ = treinar_delta_iris(treino_par, cp, cn, indices, 0.02, 300)
                pesos[(cp, cn)] = (w, cp, cn)
            preds, gab = [], []
            for a in teste:
                votos = {c: 0 for c in classes}
                for (w, cp, cn) in pesos.values():
                    x = [1.0] + [a['atributos'][i] for i in indices]
                    net = sum(wi * xi for wi, xi in zip(w, x))
                    votos[cp if net >= 0 else cn] += 1
                preds.append(max(votos, key=votos.get))
                gab.append(a['classe'])
            return preds, gab

    def _pred_delta_ova(self, treino, teste, classes, indices):
        pesos, _, _ = treinar_delta_ova(treino, indices, 0.02, 300)
        preds, gab = [], []
        for a in teste:
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
        self._preencher_comparacao_kt()

    def _abrir_memoria_metricas(self):
        """Abre janela de memoria de calculo das metricas do modelo selecionado."""
        if not self._treinado or not self.resultados:
            self.lbl_status.configure(
                text='Treine os modelos primeiro.', fg=T.DANGER)
            return
        nome = self.var_modelo_sel.get()
        if not nome or nome not in self.resultados:
            return

        # Se houver Perceptron e Delta (OvA ou binarios), passa o par para o teste Z
        perc = next((self.resultados[n] for n in self.resultados
                     if 'Perceptron' in n), None)
        delt = next((self.resultados[n] for n in self.resultados
                     if 'Delta' in n and 'Bin' not in n), None)
        perc_vs_delta = (perc, delt) if (perc and delt) else None

        JanelaMemoriaCalculoMetricas(
            self,
            nome_modelo=nome,
            relatorio=self.resultados[nome],
            classes=self._classes_usadas if hasattr(self, '_classes_usadas') else CLASSES,
            classe_foco=self.var_classe_sel.get(),
            perc_vs_delta=perc_vs_delta
        )

    def _atualizar_metricas_globais(self):
        nome = self.var_modelo_sel.get()
        if not nome or nome not in self.resultados:
            return
        rel = self.resultados[nome]
        ag = rel['acerto_global']
        k  = rel['kappa']
        t  = rel['tau']
        classes = self._classes_usadas if hasattr(self, '_classes_usadas') else CLASSES
        m  = sum(rel['matriz'][p][r] for p in classes for r in classes)

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
        if classe not in classes:
            classe = classes[0]
            self.var_classe_sel.set(classe)

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

        classes = self._classes_usadas if hasattr(self, '_classes_usadas') else CLASSES
        colunas  = ['Modelo', 'Acerto Global', 'Kappa', 'Tau']
        for c in classes:
            colunas.append(f'{c[:3]} AP')
        for c in classes:
            colunas.append(f'{c[:3]} AU')
            
        larguras = [120, 100, 80, 80] + [65]*len(classes)*2

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
            f = T.FONT_CELL_BOLD if bold else T.FONT_MONO_SM
            tk.Label(inner, text=texto, bg=bg, fg=cor, font=f,
                     width=larg // 8, anchor='center',
                     highlightthickness=1, highlightbackground=T.BORDER
                    ).grid(row=row, column=col, sticky='nsew', padx=1, pady=1)

        for j, (nm, larg) in enumerate(zip(colunas, larguras)):
            cel(0, j, nm, cor=T.ACCENT_DEEP, larg=larg, bold=True)

        for i, (nome, rel) in enumerate(self.resultados.items()):
            ag = rel['acerto_global']
            k  = rel['kappa']
            t  = rel['tau']
            bg = T.BG_CARD if i % 2 == 0 else T.BG_PANEL
            cor_ag = T.SUCCESS if ag >= 0.9 else T.ACCENT if ag >= 0.7 else T.DANGER
            cor_k  = T.SUCCESS if k > 0.80 else T.ACCENT if k > 0.40 else T.DANGER

            vals  = [nome, f'{ag*100:.2f}%', f'{k:.4f}', f'{t:.4f}']
            cores = [T.FG, cor_ag, cor_k, cor_k]
            for c in classes:
                pc = rel['por_classe'][c]
                vals.append(f'{pc["acuracia_produtor"]*100:.1f}%')
                cores.append(CORES_CLASSE[c])
            for c in classes:
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
        classes = self._classes_usadas if hasattr(self, '_classes_usadas') else CLASSES

        outer = tk.Frame(self._aba_detalhe, bg=T.BG_CARD)
        outer.pack(fill='both', expand=True, padx=10, pady=8)

        tk.Label(outer,
                 text=f'Modelo: {nome}  |  Metricas por Classe — visao OvR (One-vs-Rest)',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
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
            f  = T.FONT_CELL_BOLD if bold else T.FONT_MONO_SM
            tk.Label(inner, text=texto, bg=bg, fg=cor, font=f,
                     anchor='center', width=largs[col] // 8,
                     highlightthickness=1, highlightbackground=T.BORDER
                    ).grid(row=row, column=col, sticky='nsew', padx=1, pady=1)

        for j, nm in enumerate(cols):
            cel(0, j, nm, cor=T.ACCENT_DEEP, bold=True)

        for i, c in enumerate(classes):
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
            tk.Label(bloco, text=rot.upper(), bg=T.BG_CARD, fg=T.ACCENT_DEEP,
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
        classes = self._classes_usadas if hasattr(self, '_classes_usadas') else CLASSES

        outer = tk.Frame(self._aba_pares, bg=T.BG_CARD)
        outer.pack(fill='both', expand=True, padx=10, pady=8)

        tk.Label(outer,
                 text=f'Modelo: {nome}  |  MCC e Fb Score — problema DUAS CLASSES (por par)',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(0, 4))

        tk.Label(outer,
                 text='Cada par e tratado como classificacao binaria pura:\n'
                      '  VP = acertos da classe i  |  VN = acertos da classe j  '
                      '|  FP/FN = trocas entre i e j',
                 bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_LABEL,
                 justify='left', anchor='w'
                ).pack(fill='x', pady=(0, 8))

        if len(classes) == 2:
            pares_locais = [(classes[0], classes[1])]
        else:
            pares_locais = PARES

        for ci, cj in pares_locais:
            mat = rel['matriz']
            # Visao binaria pura: so as duas classes do par.
            # Como pred e linha e real e coluna:
            vp = mat[ci][ci]
            fn = mat[cj][ci] # pred=cj, real=ci
            fp = mat[ci][cj] # pred=ci, real=cj
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
            tk.Label(tit, text=ROTULO_PAR.get((ci, cj), f'{ci.capitalize()} x {cj.capitalize()}'),
                     bg=T.BG_PANEL, fg=T.FG,
                     font=(T.FONT_FAMILY_TITLE, 11, 'bold'), anchor='w').pack(side='left')
            tk.Label(tit, text=f'  ({total_par} amostras de teste)',
                     bg=T.BG_PANEL, fg=T.FG_MUTED,
                     font=T.FONT_LABEL, anchor='w').pack(side='left')

            # Mini matriz 2x2
            grid = tk.Frame(bloco, bg=T.BG_PANEL)
            grid.pack(side='left', padx=10, pady=(0, 8))

            def cel2(row, col, texto, bg=T.BG_CARD, fg=T.FG, bold=False):
                f = T.FONT_CELL_BOLD if bold else T.FONT_MONO_SM
                tk.Label(grid, text=texto, bg=bg, fg=fg, font=f,
                         width=10, anchor='center',
                         highlightthickness=1, highlightbackground=T.BORDER
                        ).grid(row=row, column=col, padx=1, pady=1, sticky='nsew')

            cel2(0, 0, 'Pred \\ Real', bg=T.BG_PANEL, fg=T.FG_MUTED)
            cel2(0, 1, ci.capitalize(), bg=cor_i, fg='white', bold=True)
            cel2(0, 2, cj.capitalize(), bg=cor_j, fg='white', bold=True)
            cel2(1, 0, ci.capitalize(), bg=cor_i, fg='white', bold=True)
            cel2(2, 0, cj.capitalize(), bg=cor_j, fg='white', bold=True)

            # VP (diagonal = acerto da classe ci)
            bg_vp = T.SUCCESS if vp > 0 else T.BG_CARD
            cel2(1, 1, f'VP = {vp}', bg=bg_vp, fg='white' if vp > 0 else T.FG)
            # FN (real ci predito como cj)
            bg_fn = T.DANGER if fn > 0 else T.BG_CARD
            cel2(2, 1, f'FN = {fn}', bg=bg_fn, fg='white' if fn > 0 else T.FG)
            # FP (real cj predito como ci)
            bg_fp = T.DANGER if fp > 0 else T.BG_CARD
            cel2(1, 2, f'FP = {fp}', bg=bg_fp, fg='white' if fp > 0 else T.FG)
            # VN (diagonal = acerto da classe cj)
            bg_vn = T.SUCCESS if vn > 0 else T.BG_CARD
            cel2(2, 2, f'VN = {vn}', bg=bg_vn, fg='white' if vn > 0 else T.FG)

            # Metricas do par
            metr = tk.Frame(bloco, bg=T.BG_PANEL)
            metr.pack(side='left', fill='x', expand=True, padx=10, pady=(0, 8))

            def bloquinho(parent, rotulo, valor, cor):
                b = tk.Frame(parent, bg=T.BG_CARD,
                             highlightthickness=1, highlightbackground=T.BORDER)
                b.pack(side='left', padx=(0, 6), ipadx=6, ipady=4)
                tk.Label(b, text=rotulo.upper(), bg=T.BG_CARD, fg=T.ACCENT_DEEP,
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
        classes = self._classes_usadas if hasattr(self, '_classes_usadas') else CLASSES
        n = len(classes)

        outer = tk.Frame(self._aba_matriz, bg=T.BG_CARD)
        outer.pack(fill='both', expand=True, padx=12, pady=8)

        tk.Label(outer,
                 text=f'Matriz de Confusao  —  {nome}  |  '
                      f'Linhas = Predito  ·  Colunas = Real',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(0, 8))

        grid  = tk.Frame(outer, bg=T.BG_CARD)
        grid.pack(anchor='w')
        vals  = [[matriz[pred][real] for real in classes] for pred in classes]
        v_max = max(max(l) for l in vals) or 1

        def bg_cel(v, diag):
            t = v / v_max
            if diag:
                r = int(255 * (1 - 0.6 * t)); g = int(255 * (1 - 0.4 * t)); b = 255
            else:
                r = 255; g = int(255 * (1 - 0.7 * t)); b = int(255 * (1 - 0.7 * t))
            return f'#{r:02x}{g:02x}{b:02x}'

        tk.Label(grid, text='Pred \\ Real', bg=T.BG_PANEL, fg=T.FG_MUTED,
                 font=T.FONT_KICKER, width=10, anchor='center',
                 highlightthickness=1, highlightbackground=T.BORDER
                ).grid(row=0, column=0, padx=2, pady=2)

        for j, c in enumerate(classes):
            tk.Label(grid, text=c.capitalize(), bg=CORES_CLASSE[c],
                     fg='white', font=T.FONT_KICKER, width=10, anchor='center',
                     highlightthickness=1, highlightbackground=T.BORDER
                    ).grid(row=0, column=j + 1, padx=2, pady=2)
                    
        # Coluna de total da linha
        tk.Label(grid, text='Total', bg=T.BG_PANEL, fg=T.ACCENT_DEEP,
                 font=T.FONT_KICKER, width=10, anchor='center',
                 highlightthickness=1, highlightbackground=T.BORDER
                ).grid(row=0, column=n + 1, padx=2, pady=2)

        totais_colunas = {real: 0 for real in classes}
        total_geral = 0

        for i, pred in enumerate(classes):
            tk.Label(grid, text=pred.capitalize(), bg=CORES_CLASSE[pred],
                     fg='white', font=T.FONT_KICKER, width=10, anchor='center',
                     highlightthickness=1, highlightbackground=T.BORDER
                    ).grid(row=i + 1, column=0, padx=2, pady=2)
            
            total_linha = sum(matriz[pred][real] for real in classes)
            
            for j, real in enumerate(classes):
                v   = matriz[pred][real]
                bg  = bg_cel(v, i == j)
                cfg = 'white' if (v / v_max > 0.4) else T.FG
                tk.Label(grid, text=str(v), bg=bg, fg=cfg,
                         font=T.FONT_CELL_LG, width=10, anchor='center',
                         highlightthickness=1, highlightbackground=T.BORDER
                        ).grid(row=i + 1, column=j + 1, padx=2, pady=2)
                totais_colunas[real] += v
                
            cel_tot_linha = tk.Label(grid, text=str(total_linha), bg=T.BG_PANEL, fg=T.FG,
                                     font=T.FONT_CELL_LG, width=10, anchor='center',
                                     highlightthickness=1, highlightbackground=T.BORDER)
            cel_tot_linha.grid(row=i + 1, column=n + 1, padx=2, pady=2)
            total_geral += total_linha

        # Linha inferior de Totais das colunas
        tk.Label(grid, text='Total', bg=T.BG_PANEL, fg=T.ACCENT_DEEP,
                 font=T.FONT_KICKER, width=10, anchor='center',
                 highlightthickness=1, highlightbackground=T.BORDER
                ).grid(row=n + 1, column=0, padx=2, pady=2)
                
        for j, real in enumerate(classes):
            v_col = totais_colunas[real]
            tk.Label(grid, text=str(v_col), bg=T.BG_PANEL, fg=T.FG,
                     font=T.FONT_CELL_LG, width=10, anchor='center',
                     highlightthickness=1, highlightbackground=T.BORDER
                    ).grid(row=n + 1, column=j + 1, padx=2, pady=2)
                    
        # Célula inferior direita com total geral
        tk.Label(grid, text=str(total_geral), bg=T.BG_PANEL, fg=T.ACCENT_DEEP,
                 font=T.FONT_CELL_LG, width=10, anchor='center',
                 highlightthickness=1, highlightbackground=T.BORDER
                ).grid(row=n + 1, column=n + 1, padx=2, pady=2)

        # Acuracia produtor e usuario
        info = tk.Frame(outer, bg=T.BG_CARD)
        info.pack(fill='x', pady=(12, 0))
        tk.Label(info, text='ACURACIA DO PRODUTOR (Sensibilidade / Colunas)', bg=T.BG_CARD, fg=T.ACCENT_DEEP,
                 font=T.FONT_KICKER, anchor='w'
                ).grid(row=0, column=0, columnspan=n, sticky='w', pady=(0, 4))
        tk.Label(info, text='ACURACIA DO USUARIO (Precisao / Linhas)', bg=T.BG_CARD, fg=T.ACCENT_DEEP,
                 font=T.FONT_KICKER, anchor='w'
                ).grid(row=0, column=n, columnspan=n, sticky='w', pady=(0, 4),
                       padx=(20, 0))
        for col, c in enumerate(classes):
            pc  = rel['por_classe'][c]
            cor = CORES_CLASSE[c]
            for offset, chave in [(0, 'acuracia_produtor'), (n, 'acuracia_usuario')]:
                b = tk.Frame(info, bg=T.BG_PANEL,
                             highlightthickness=1, highlightbackground=T.BORDER)
                b.grid(row=1, column=col + offset, sticky='ew',
                       padx=(20 if (col == 0 and offset == n) else
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
                         fontfamily=T.FONT_FAMILY_NAME, fontweight='bold')
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
    # Aba Comparacao Kappa & Tau — Selecao Dinamica de Modelos
    # -----------------------------------------------------------------------
    def _preencher_comparacao_kt(self):
        for w in self._aba_comp2.winfo_children():
            w.destroy()

        outer = tk.Frame(self._aba_comp2, bg=T.BG_CARD)
        outer.pack(fill='both', expand=True, padx=14, pady=10)

        nomes_modelos = list(self.resultados.keys())
        if not nomes_modelos:
            tk.Label(outer, text='Treine os modelos primeiro.',
                     bg=T.BG_CARD, fg=T.DANGER, font=T.FONT_BODY).pack()
            return

        m1 = self.var_comp_modelo1.get()
        m2 = self.var_comp_modelo2.get()

        if m1 not in nomes_modelos:
            perc_opt = [n for n in nomes_modelos if 'Perceptron' in n]
            self.var_comp_modelo1.set(perc_opt[0] if perc_opt else nomes_modelos[0])
        if m2 not in nomes_modelos:
            delta_opt = [n for n in nomes_modelos if 'Delta' in n and 'Bin.' not in n]
            self.var_comp_modelo2.set(delta_opt[0] if delta_opt else (nomes_modelos[1] if len(nomes_modelos) > 1 else nomes_modelos[0]))

        p_name = self.var_comp_modelo1.get()
        d_name = self.var_comp_modelo2.get()

        # Seletores de Modelos para Comparacao
        frame_seletores = tk.Frame(outer, bg=T.BG_CARD)
        frame_seletores.pack(fill='x', pady=(0, 10))
        
        tk.Label(frame_seletores, text='Classificador A:', bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_LABEL).pack(side='left', padx=(0, 6))
        cb1 = ttk.Combobox(frame_seletores, textvariable=self.var_comp_modelo1, values=nomes_modelos, state='readonly', font=T.FONT_BODY, width=22)
        cb1.pack(side='left', padx=(0, 20))
        cb1.bind('<<ComboboxSelected>>', lambda e: self._preencher_comparacao_kt())

        tk.Label(frame_seletores, text='Classificador B:', bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_LABEL).pack(side='left', padx=(0, 6))
        cb2 = ttk.Combobox(frame_seletores, textvariable=self.var_comp_modelo2, values=nomes_modelos, state='readonly', font=T.FONT_BODY, width=22)
        cb2.pack(side='left')
        cb2.bind('<<ComboboxSelected>>', lambda e: self._preencher_comparacao_kt())

        tk.Label(outer,
                 text=f'ITEM 2 — Teste de Significancia: {p_name}  vs  {d_name}',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(10, 4))
        tk.Label(outer,
                 text='Verifica se a diferenca entre os dois classificadores e '
                       'estatisticamente significativa ao nivel de 5%.\n'
                       'H0: nao ha diferenca entre os coeficientes  |  '
                       'H1: ha diferenca',
                 bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_LABEL,
                 justify='left', anchor='w', wraplength=820
                ).pack(fill='x', pady=(0, 10))

        perc = self.resultados.get(p_name)
        delt = self.resultados.get(d_name)

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
            tk.Label(tab, text=texto, bg=T.BG_PANEL, fg=T.ACCENT_DEEP,
                     font=T.FONT_CELL_BOLD, anchor='center',
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

        for col, h in enumerate(['Metrica', p_name, d_name,
                                  'Z calculado', 'p-valor', 'Conclusao (5%)']):
            th(col, h)

        linhas = [
            ('Acerto Global',
             f'{ag1*100:.2f}%', f'{ag2*100:.2f}%',
             '—', '—',
             'Maior: ' + (p_name if ag1 > ag2 else d_name)),
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

        maior_acc = p_name if ag1 > ag2 else (d_name if ag2 > ag1 else 'empate')
        maior_k   = p_name if k1 > k2 else (d_name if k2 > k1 else 'empate')

        calculos_passos = (
            f"MEMORIA DE CALCULO DO TESTE Z:\n"
            f"1. Teste de Significancia para Kappa:\n"
            f"   Formula: Z_k = (K_A - K_B) / sqrt(Var(K_A) + Var(K_B))\n"
            f"   Calculo: Z_k = ({k1:.6f} - {k2:.6f}) / sqrt({vk1:.8f} + {vk2:.8f})\n"
            f"            Z_k = {k1 - k2:+.6f} / {math.sqrt(vk1 + vk2):.8f}\n"
            f"            Z_k = {zk:+.4f}   ==>   p-valor = {pzk:.4f} "
            f"({'p < 0.05' if pzk < 0.05 else 'p >= 0.05'})\n\n"
            f"2. Teste de Significancia para Tau:\n"
            f"   Formula: Z_t = (Tau_A - Tau_B) / sqrt(Var(Tau_A) + Var(Tau_B))\n"
            f"   Cálculo: Z_t = ({t1:.6f} - {t2:.6f}) / sqrt({vt1:.8f} + {vt2:.8f})\n"
            f"            Z_t = {t1 - t2:+.6f} / {math.sqrt(vt1 + vt2):.8f}\n"
            f"            Z_t = {zt:+.4f}   ==>   p-valor = {pzt:.4f} "
            f"({'p < 0.05' if pzt < 0.05 else 'p >= 0.05'})\n\n"
            f"--------------------------------------------------------------------------------\n"
        )

        texto_interp = (
            calculos_passos +
            f'Maior acuracia:  {maior_acc}  '
            f'({p_name} {ag1*100:.2f}%  vs  {d_name} {ag2*100:.2f}%)\n'
            f'Maior Kappa:     {maior_k}  '
            f'({p_name} K={k1:.4f}  vs  {d_name} K={k2:.4f})\n\n'
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

        # Secao adicional: significancia entre classes (OvR)
        self._secao_significancia_classes(outer)

    # -----------------------------------------------------------------------
    # Significancia entre Classes (OvR) — teste Z de Kappa classe A vs B
    # -----------------------------------------------------------------------
    def _secao_significancia_classes(self, outer):
        classes = self._classes_usadas if hasattr(self, '_classes_usadas') else CLASSES

        tk.Frame(outer, bg=T.BORDER, height=1).pack(fill='x', pady=(12, 8))
        nome_modelo = self.var_modelo_sel.get()
        tk.Label(outer,
                 text=f'SIGNIFICANCIA ENTRE CLASSES (OvR)  —  modelo: {nome_modelo}',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(0, 4))
        tk.Label(outer,
                 text='Compara o Kappa OvR (classe vs resto) de duas classes do '
                      'MESMO classificador.\n'
                      'H0: os Kappas das duas classes nao diferem  |  '
                      'regiao critica |Z| > 1.96 (a = 5%).',
                 bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_LABEL,
                 justify='left', anchor='w', wraplength=820
                ).pack(fill='x', pady=(0, 8))

        if not nome_modelo or nome_modelo not in self.resultados:
            tk.Label(outer, text='Selecione um modelo treinado.',
                     bg=T.BG_CARD, fg=T.DANGER, font=T.FONT_BODY,
                     anchor='w').pack(fill='x')
            return

        # Seletores de classes A e B
        ca = self.var_sig_classe_a.get()
        cb = self.var_sig_classe_b.get()
        if ca not in classes:
            ca = classes[0]
            self.var_sig_classe_a.set(ca)
        if cb not in classes or cb == ca:
            cb = next((c for c in classes if c != ca), classes[-1])
            self.var_sig_classe_b.set(cb)

        sel = tk.Frame(outer, bg=T.BG_CARD)
        sel.pack(fill='x', pady=(0, 10))
        tk.Label(sel, text='Classe A:', bg=T.BG_CARD, fg=T.FG_MUTED,
                 font=T.FONT_LABEL).pack(side='left', padx=(0, 6))
        cba = ttk.Combobox(sel, textvariable=self.var_sig_classe_a,
                           values=classes, state='readonly',
                           font=T.FONT_BODY, width=14)
        cba.pack(side='left', padx=(0, 20))
        cba.bind('<<ComboboxSelected>>', lambda e: self._preencher_comparacao_kt())
        tk.Label(sel, text='Classe B:', bg=T.BG_CARD, fg=T.FG_MUTED,
                 font=T.FONT_LABEL).pack(side='left', padx=(0, 6))
        cbb = ttk.Combobox(sel, textvariable=self.var_sig_classe_b,
                           values=classes, state='readonly',
                           font=T.FONT_BODY, width=14)
        cbb.pack(side='left')
        cbb.bind('<<ComboboxSelected>>', lambda e: self._preencher_comparacao_kt())

        if ca == cb:
            tk.Label(outer, text='Escolha duas classes diferentes.',
                     bg=T.BG_CARD, fg=T.DANGER, font=T.FONT_BODY,
                     anchor='w').pack(fill='x')
            return

        matriz = self.resultados[nome_modelo]['matriz']
        zc, k_a, var_a, k_b, var_b = z_classes(matriz, ca, cb)
        sig = abs(zc) > 1.96   # inclui o caso Z infinito
        z_txt = ('+Infinito' if zc > 0 else '-Infinito') if math.isinf(zc) \
                else f'{zc:+.4f}'

        # Mini-matrizes 2x2 OvR lado a lado (estilo aba Pares de Classes)
        faixa_mat = tk.Frame(outer, bg=T.BG_CARD)
        faixa_mat.pack(fill='x', pady=(0, 10))

        for col, (classe, k_v, var_v) in enumerate(
                [(ca, k_a, var_a), (cb, k_b, var_b)]):
            m2 = matriz_binaria_ovr(matriz, classe)
            vp = m2[classe][classe]
            fp = m2[classe]['resto']
            fn = m2['resto'][classe]
            vn = m2['resto']['resto']
            cor_c = CORES_CLASSE.get(classe, T.ACCENT)

            bloco = tk.Frame(faixa_mat, bg=T.BG_PANEL,
                             highlightthickness=1, highlightbackground=T.BORDER)
            bloco.grid(row=0, column=col, sticky='n',
                       padx=(0 if col == 0 else 16, 0))

            tk.Label(bloco,
                     text=f'{classe.capitalize()}  vs  Resto',
                     bg=T.BG_PANEL, fg=cor_c,
                     font=(T.FONT_FAMILY_TITLE, 11, 'bold'), anchor='w'
                    ).pack(fill='x', padx=10, pady=(6, 4))

            grid = tk.Frame(bloco, bg=T.BG_PANEL)
            grid.pack(padx=10, pady=(0, 6))

            def cel2(row, c_, texto, bg=T.BG_CARD, fg=T.FG, bold=False,
                     _grid=grid):
                f = T.FONT_CELL_BOLD if bold else T.FONT_MONO_SM
                tk.Label(_grid, text=texto, bg=bg, fg=fg, font=f,
                         width=10, anchor='center',
                         highlightthickness=1, highlightbackground=T.BORDER
                        ).grid(row=row, column=c_, padx=1, pady=1,
                               sticky='nsew')

            cel2(0, 0, 'Pred \\ Real', bg=T.BG_PANEL, fg=T.FG_MUTED)
            cel2(0, 1, classe.capitalize(), bg=cor_c, fg='white', bold=True)
            cel2(0, 2, 'Resto', bg=T.BG_HOVER, fg=T.FG, bold=True)
            cel2(1, 0, classe.capitalize(), bg=cor_c, fg='white', bold=True)
            cel2(2, 0, 'Resto', bg=T.BG_HOVER, fg=T.FG, bold=True)

            cel2(1, 1, f'VP = {vp}',
                 bg=T.SUCCESS if vp > 0 else T.BG_CARD,
                 fg='white' if vp > 0 else T.FG)
            cel2(1, 2, f'FP = {fp}',
                 bg=T.DANGER if fp > 0 else T.BG_CARD,
                 fg='white' if fp > 0 else T.FG)
            cel2(2, 1, f'FN = {fn}',
                 bg=T.DANGER if fn > 0 else T.BG_CARD,
                 fg='white' if fn > 0 else T.FG)
            cel2(2, 2, f'VN = {vn}',
                 bg=T.SUCCESS if vn > 0 else T.BG_CARD,
                 fg='white' if vn > 0 else T.FG)

            tk.Label(bloco,
                     text=f'Kappa = {k_v:.6f}    Var = {var_v:.8f}',
                     bg=T.BG_PANEL, fg=T.FG, font=T.FONT_MONO_SM, anchor='w'
                    ).pack(fill='x', padx=10, pady=(0, 8))

        # Cards de resultado do teste Z
        res = tk.Frame(outer, bg=T.BG_CARD)
        res.pack(fill='x', pady=(0, 8))

        def card_res(col, rotulo, valor, cor):
            b = tk.Frame(res, bg=T.BG_PANEL,
                         highlightthickness=1, highlightbackground=T.BORDER)
            b.grid(row=0, column=col, sticky='ew',
                   padx=(0 if col == 0 else 6, 0))
            res.columnconfigure(col, weight=1)
            tk.Label(b, text=rotulo.upper(), bg=T.BG_PANEL, fg=T.ACCENT_DEEP,
                     font=T.FONT_KICKER, anchor='w').pack(
                fill='x', padx=8, pady=(6, 0))
            tk.Label(b, text=valor, bg=T.BG_PANEL, fg=cor,
                     font=T.FONT_HEADLINE, anchor='w').pack(
                fill='x', padx=8, pady=(2, 6))

        cor_z = T.DANGER if sig else T.SUCCESS
        card_res(0, f'Kappa {ca[:3]}.', f'{k_a:.4f}',
                 CORES_CLASSE.get(ca, T.FG))
        card_res(1, f'Kappa {cb[:3]}.', f'{k_b:.4f}',
                 CORES_CLASSE.get(cb, T.FG))
        card_res(2, 'Z calculado', z_txt, cor_z)
        card_res(3, 'Z critico (95%)', '1.96', T.FG_MUTED)
        card_res(4, 'Veredito',
                 'SIGNIFICATIVA' if sig else 'NAO SIGNIFICATIVA', cor_z)

        # Memoria de calculo + conclusao
        den_txt = ('0 (variancias nulas)'
                   if (var_a + var_b) < 1e-24
                   else f'{math.sqrt(var_a + var_b):.8f}')
        memoria = (
            f'Z = (K_{ca} - K_{cb}) / sqrt(Var_{ca} + Var_{cb})\n'
            f'  = ({k_a:.6f} - {k_b:.6f}) / sqrt({var_a:.8f} + {var_b:.8f})\n'
            f'  = {k_a - k_b:+.6f} / {den_txt}   =   {z_txt}'
        )
        tk.Label(outer, text=memoria, bg=T.BG_PANEL, fg=T.FG,
                 font=T.FONT_MONO_SM, justify='left', anchor='w',
                 highlightthickness=1, highlightbackground=T.BORDER
                ).pack(fill='x', ipadx=10, ipady=8, pady=(0, 6))

        if sig:
            conclusao = (
                f'Como |Z| > 1.96, ha diferenca SIGNIFICATIVA entre os Kappas '
                f'das classes {ca.capitalize()} e {cb.capitalize()} ao nivel '
                f'de 5% — rejeita-se H0: o classificador "{nome_modelo}" nao '
                f'trata as duas classes com a mesma qualidade.'
            )
        else:
            conclusao = (
                f'Como |Z| <= 1.96, a diferenca entre os Kappas das classes '
                f'{ca.capitalize()} e {cb.capitalize()} NAO e significativa '
                f'a 95% — nao se rejeita H0: o classificador "{nome_modelo}" '
                f'apresenta qualidade equivalente nas duas classes.'
            )
        tk.Label(outer, text=conclusao,
                 bg=T.BG_CARD, fg=T.DANGER if sig else T.SUCCESS,
                 font=T.FONT_BODY, justify='left', anchor='w', wraplength=820
                ).pack(fill='x')

    # -----------------------------------------------------------------------
    # Aba Exercicios PR51 — Item 3 (Slide 15 da Aula PR51)
    # Matrizes A e B editaveis + todos os calculos em Python puro
    # -----------------------------------------------------------------------
    _PR51_CLASSES = ['w1', 'w2', 'w3', 'w4']
    _PR51_MATRIZ_A = [[140, 20, 0, 0],
                      [10, 130, 0, 0],
                      [5, 0, 150, 10],
                      [15, 10, 0, 120]]
    _PR51_MATRIZ_B = [[140, 30, 2, 0],
                      [10, 110, 5, 0],
                      [0, 10, 140, 0],
                      [20, 10, 3, 140]]

    def _construir_aba_pr51(self):
        outer = tk.Frame(self._aba_pr51, bg=T.BG_CARD)
        outer.pack(fill='both', expand=True, padx=14, pady=10)

        tk.Label(outer,
                 text='ITEM 3 — Exercicios da Aula PR51 (Slide 15)  ·  '
                      'Prof. Robson Pequeno de Sousa',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(0, 2))
        tk.Label(outer,
                 text='Duas classificacoes (A e B) com 4 classes. As celulas sao '
                      'editaveis: altere os valores e clique em "Recalcular" para '
                      'refazer Kappa, Tau, variancias e o teste Z em Python puro.',
                 bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_LABEL,
                 justify='left', anchor='w', wraplength=860
                ).pack(fill='x', pady=(0, 10))

        # --- duas matrizes editaveis lado a lado ---
        faixa = tk.Frame(outer, bg=T.BG_CARD)
        faixa.pack(fill='x')

        self._pr51_entries = {}   # ('A'|'B', i, j) -> tk.Entry
        for rot, dados, col in [('A', self._PR51_MATRIZ_A, 0),
                                ('B', self._PR51_MATRIZ_B, 1)]:
            bloco = tk.Frame(faixa, bg=T.BG_PANEL,
                             highlightthickness=1, highlightbackground=T.BORDER)
            bloco.grid(row=0, column=col, sticky='n', padx=(0 if col == 0 else 16, 0))
            tk.Label(bloco, text=f'Classificacao {rot}',
                     bg=T.BG_PANEL, fg=T.FG,
                     font=(T.FONT_FAMILY_TITLE, 11, 'bold')
                    ).grid(row=0, column=0, columnspan=6, sticky='w',
                           padx=8, pady=(6, 4))

            tk.Label(bloco, text='Real \\ Pred', bg=T.BG_PANEL, fg=T.FG_MUTED,
                     font=T.FONT_KICKER, width=9, anchor='center'
                    ).grid(row=1, column=0, padx=2, pady=2)
            for j, c in enumerate(self._PR51_CLASSES):
                tk.Label(bloco, text=c, bg=T.BG_HOVER, fg=T.FG,
                         font=T.FONT_KICKER, width=7, anchor='center'
                        ).grid(row=1, column=j + 1, padx=2, pady=2)

            for i, c in enumerate(self._PR51_CLASSES):
                tk.Label(bloco, text=c, bg=T.BG_HOVER, fg=T.FG,
                         font=T.FONT_KICKER, width=9, anchor='center'
                        ).grid(row=i + 2, column=0, padx=2, pady=2)
                for j in range(4):
                    e = tk.Entry(bloco, width=7, justify='center',
                                 font=T.FONT_MONO_SM, relief='flat',
                                 bg=T.BG_CARD, fg=T.FG,
                                 insertbackground=T.ACCENT,
                                 highlightthickness=1,
                                 highlightbackground=T.BORDER_HARD,
                                 highlightcolor=T.ACCENT)
                    e.insert(0, str(dados[i][j]))
                    e.grid(row=i + 2, column=j + 1, padx=2, pady=2)
                    self._pr51_entries[(rot, i, j)] = e
            tk.Frame(bloco, bg=T.BG_PANEL, height=6).grid(row=6, column=0)

        botoes = tk.Frame(faixa, bg=T.BG_CARD)
        botoes.grid(row=0, column=2, sticky='n', padx=(16, 0))
        ttk.Button(botoes, text='Recalcular  >', style='Primary.TButton',
                   command=self._calcular_pr51).pack(fill='x')
        ttk.Button(botoes, text='Restaurar slide',
                   command=self._restaurar_pr51).pack(fill='x', pady=(6, 0))

        # --- container dos resultados (refeito a cada recalculo) ---
        self._pr51_resultados = tk.Frame(outer, bg=T.BG_CARD)
        self._pr51_resultados.pack(fill='both', expand=True, pady=(12, 0))
        self._calcular_pr51()

    def _restaurar_pr51(self):
        for rot, dados in [('A', self._PR51_MATRIZ_A), ('B', self._PR51_MATRIZ_B)]:
            for i in range(4):
                for j in range(4):
                    e = self._pr51_entries[(rot, i, j)]
                    e.delete(0, 'end')
                    e.insert(0, str(dados[i][j]))
        self._calcular_pr51()

    def _ler_matriz_pr51(self, rot):
        """Le as entries e devolve matriz dict {real: {pred: int}}."""
        W = self._PR51_CLASSES
        matriz = {ci: {cj: 0 for cj in W} for ci in W}
        for i, ci in enumerate(W):
            for j, cj in enumerate(W):
                txt = self._pr51_entries[(rot, i, j)].get().strip()
                try:
                    v = int(txt)
                    if v < 0:
                        raise ValueError
                except ValueError:
                    raise ValueError(
                        f'Matriz {rot}, linha {ci}, coluna {cj}: '
                        f'valor invalido "{txt}" (use inteiros >= 0).')
                matriz[ci][cj] = v
        return matriz

    def _calcular_pr51(self):
        for w in self._pr51_resultados.winfo_children():
            w.destroy()
        outer = self._pr51_resultados
        W = self._PR51_CLASSES

        try:
            mat_a = self._ler_matriz_pr51('A')
            mat_b = self._ler_matriz_pr51('B')
        except ValueError as err:
            tk.Label(outer, text=str(err), bg=T.BG_CARD, fg=T.DANGER,
                     font=T.FONT_BODY, anchor='w').pack(fill='x')
            return

        res = {}
        for rot, mat in [('A', mat_a), ('B', mat_b)]:
            m  = sum(mat[r][p] for r in W for p in W)
            ag = acerto_global(mat, W)
            k  = kappa(mat, W)
            t  = tau(mat, W)
            vk = variancia_kappa(mat, W)
            vt = variancia_tau(mat, W)
            res[rot] = {'m': m, 'ag': ag, 'k': k, 't': t, 'vk': vk, 'vt': vt,
                        'mat': mat}

        zk  = z_kappa(res['A']['k'], res['A']['vk'], res['B']['k'], res['B']['vk'])
        zt  = z_tau(res['A']['t'], res['A']['vt'], res['B']['t'], res['B']['vt'])
        pzk = p_valor_z(zk)
        pzt = p_valor_z(zt)
        sig_k = pzk < 0.05
        sig_t = pzt < 0.05

        # --- tabela consolidada A vs B ---
        tk.Label(outer, text='RESULTADOS CONSOLIDADOS',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(0, 4))

        tab = tk.Frame(outer, bg=T.BG_CARD)
        tab.pack(anchor='w', pady=(0, 12))

        def cel(row, col, texto, cor=T.FG, bold=False, larg=20):
            bg = T.BG_PANEL if row == 0 else (T.BG_CARD if row % 2 else T.BG_PANEL)
            f  = T.FONT_CELL_BOLD if bold else T.FONT_MONO_SM
            tk.Label(tab, text=texto, bg=bg, fg=cor, font=f,
                     width=larg, anchor='center',
                     highlightthickness=1, highlightbackground=T.BORDER
                    ).grid(row=row, column=col, padx=1, pady=1, sticky='nsew')

        for col, h in enumerate(['Metrica', 'Classificacao A', 'Classificacao B']):
            cel(0, col, h, cor=T.ACCENT_DEEP, bold=True)

        def cor_k(v):
            return T.SUCCESS if v > 0.80 else T.ACCENT if v > 0.40 else T.DANGER

        linhas = [
            ('Total de amostras (m)', f"{res['A']['m']}", f"{res['B']['m']}",
             T.FG, T.FG),
            ('Acerto Global (Ag)',
             f"{res['A']['ag']*100:.4f}%", f"{res['B']['ag']*100:.4f}%",
             T.SUCCESS if res['A']['ag'] >= 0.8 else T.ACCENT,
             T.SUCCESS if res['B']['ag'] >= 0.8 else T.ACCENT),
            ('Kappa (K)', f"{res['A']['k']:.6f}", f"{res['B']['k']:.6f}",
             cor_k(res['A']['k']), cor_k(res['B']['k'])),
            ('Interpretacao K',
             interpretar_kappa(res['A']['k']), interpretar_kappa(res['B']['k']),
             cor_k(res['A']['k']), cor_k(res['B']['k'])),
            ('Tau (t)', f"{res['A']['t']:.6f}", f"{res['B']['t']:.6f}",
             cor_k(res['A']['t']), cor_k(res['B']['t'])),
            ('Var(Kappa)', f"{res['A']['vk']:.8f}", f"{res['B']['vk']:.8f}",
             T.FG_MUTED, T.FG_MUTED),
            ('Var(Tau)', f"{res['A']['vt']:.8f}", f"{res['B']['vt']:.8f}",
             T.FG_MUTED, T.FG_MUTED),
        ]
        for r, (nome, va, vb, ca, cb) in enumerate(linhas):
            cel(r + 1, 0, nome, T.FG_MUTED)
            cel(r + 1, 1, va, ca)
            cel(r + 1, 2, vb, cb)

        # --- acuracia do produtor e do usuario por classe ---
        tk.Label(outer, text='ACURACIA DO PRODUTOR E DO USUARIO POR CLASSE',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(0, 4))

        tab2 = tk.Frame(outer, bg=T.BG_CARD)
        tab2.pack(anchor='w', pady=(0, 12))

        def cel2(row, col, texto, cor=T.FG, bold=False):
            bg = T.BG_PANEL if row == 0 else (T.BG_CARD if row % 2 else T.BG_PANEL)
            f  = T.FONT_CELL_BOLD if bold else T.FONT_MONO_SM
            tk.Label(tab2, text=texto, bg=bg, fg=cor, font=f,
                     width=14, anchor='center',
                     highlightthickness=1, highlightbackground=T.BORDER
                    ).grid(row=row, column=col, padx=1, pady=1, sticky='nsew')

        for col, h in enumerate(['Classe', 'A · Produtor', 'A · Usuario',
                                  'B · Produtor', 'B · Usuario']):
            cel2(0, col, h, cor=T.ACCENT_DEEP, bold=True)
        for r, c in enumerate(W):
            cel2(r + 1, 0, c, T.FG_MUTED)
            cel2(r + 1, 1, f"{acuracia_produtor(res['A']['mat'], c)*100:.2f}%")
            cel2(r + 1, 2, f"{acuracia_usuario(res['A']['mat'], c)*100:.2f}%")
            cel2(r + 1, 3, f"{acuracia_produtor(res['B']['mat'], c)*100:.2f}%")
            cel2(r + 1, 4, f"{acuracia_usuario(res['B']['mat'], c)*100:.2f}%")

        # --- teste de significancia ---
        tk.Label(outer, text='TESTE DE SIGNIFICANCIA  A vs B   '
                             '(H0: nao ha diferenca  ·  regiao critica |Z| > 1.96, a = 5%)',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(0, 4))

        memoria = (
            f"Z_k = (K_A - K_B) / sqrt(Var(K_A) + Var(K_B))\n"
            f"    = ({res['A']['k']:.6f} - {res['B']['k']:.6f}) / "
            f"sqrt({res['A']['vk']:.8f} + {res['B']['vk']:.8f})\n"
            f"    = {res['A']['k'] - res['B']['k']:+.6f} / "
            f"{math.sqrt(res['A']['vk'] + res['B']['vk']):.8f}"
            f"  =  {zk:+.4f}   ==>   p-valor = {pzk:.4f}  "
            f"({'rejeita H0' if sig_k else 'nao rejeita H0'})\n\n"
            f"Z_t = (t_A - t_B) / sqrt(Var(t_A) + Var(t_B))\n"
            f"    = ({res['A']['t']:.6f} - {res['B']['t']:.6f}) / "
            f"sqrt({res['A']['vt']:.8f} + {res['B']['vt']:.8f})\n"
            f"    = {res['A']['t'] - res['B']['t']:+.6f} / "
            f"{math.sqrt(res['A']['vt'] + res['B']['vt']):.8f}"
            f"  =  {zt:+.4f}   ==>   p-valor = {pzt:.4f}  "
            f"({'rejeita H0' if sig_t else 'nao rejeita H0'})"
        )
        tk.Label(outer, text=memoria, bg=T.BG_PANEL, fg=T.FG,
                 font=T.FONT_MONO_SM, justify='left', anchor='w',
                 highlightthickness=1, highlightbackground=T.BORDER
                ).pack(fill='x', ipadx=10, ipady=8, pady=(0, 8))

        if sig_k or sig_t:
            conclusao = (
                'Conclusao: ha diferenca ESTATISTICAMENTE SIGNIFICATIVA entre as '
                'classificacoes A e B ao nivel de 5% '
                f"(rejeita-se H0 pelo {'Kappa' if sig_k else ''}"
                f"{' e ' if sig_k and sig_t else ''}{'Tau' if sig_t else ''})."
            )
            cor_c = T.DANGER
        else:
            conclusao = (
                'Conclusao: como |Z| < 1.96 e p > 0.05 em ambos os testes, NAO se '
                'rejeita H0 — nao ha evidencia estatistica suficiente para afirmar '
                'que as classificacoes A e B sao diferentes. A vantagem de A no '
                'Kappa pode ser atribuida a variacao amostral.'
            )
            cor_c = T.SUCCESS
        tk.Label(outer, text=conclusao, bg=T.BG_CARD, fg=cor_c,
                 font=T.FONT_BODY, justify='left', anchor='w', wraplength=860
                ).pack(fill='x')

    # -----------------------------------------------------------------------