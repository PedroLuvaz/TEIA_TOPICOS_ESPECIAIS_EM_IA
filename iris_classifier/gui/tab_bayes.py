"""
Aba 4 — Bayes e Normalidade.

Aba desenvolvida para implementar:
  1. Classificador Bayes Otimo (QDA)
  2. Classificador Naive Bayes
  3. Verificacao de Normalidade Multivariada (HZ e Mardia) executando R ou Fallback.
  4. Superficies de decisao quadraticas/nao-lineares em 2D.
  5. Metricas completas e Teste de Significancia de Kappa (Z-test).
"""
import os
import tkinter as tk
from tkinter import ttk
import math

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure

from data.data_loader import carregar_dados_iris, split_estratificado, filtrar_por_classes
from models.bayes_classifier import treinar_bayes, predizer_todas_classes_bayes, predizer_binario_bayes
from evaluation.mvn_tester import executar_analise_mvn
from evaluation.metricas_avancadas import relatorio_completo, z_kappa, z_tau, p_valor_z
from core.math_utils import distancia_mahalanobis_quad

from . import theme as T
from .widgets import Card, MetricBlock, separador

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
    'todas': {
        'rotulo': 'Todas',
        'rotulo_ui': 'Todas (4 Features) · [0,1,2,3]',
        'indices': [0, 1, 2, 3],
        # Para visualizacao em 2D projetamos nas petalas
        'eixo_x': 'Comp. Petala',
        'eixo_y': 'Larg. Petala',
        'unidade': 'cm',
    },
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
# Classe da Aba
# ===========================================================================
class TabBayes(tk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, bg=T.BG, **kw)

        # Estado
        self.dados = []
        self.dados_treino = []
        self.dados_teste = []
        self.model_bayes = {}
        self.model_naive = {}
        self.rel_bayes = {}
        self.rel_naive = {}
        self.dados_mvn = {}
        self.r_disponivel = False
        
        self.var_dataset = tk.StringVar(value='v1')
        self.var_attr = tk.StringVar(value='petalas')
        self.var_classifier = tk.StringVar(value='bayes') # 'bayes' ou 'naive'
        self.var_modo_grafico = tk.StringVar(value='dispersao')
        
        # Classificacao manual
        self.var_sx = tk.StringVar(value='5.8')
        self.var_sy = tk.StringVar(value='3.0')
        self.var_px = tk.StringVar(value='4.5')
        self.var_py = tk.StringVar(value='1.5')

        self._construir_layout()
        self._carregar_dados()
        self._executar_analise_normalidade()
        self._atualizar_modelo()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _construir_layout(self):
        self.columnconfigure(0, minsize=340)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        self._coluna_controles()
        self._coluna_visualizacao()

    # ---- Coluna esquerda ----
    def _coluna_controles(self):
        wrap = tk.Frame(self, bg=T.BG)
        wrap.grid(row=0, column=0, sticky='nsew',
                  padx=(T.PAD_PAGE, T.GAP), pady=12)
        wrap.columnconfigure(0, weight=1)

        # 1. Configuração do Modelo
        card_config = Card(wrap, titulo='configuracao do modelo')
        card_config.grid(row=0, column=0, sticky='ew', pady=(0, T.GAP_SM))

        fds = tk.Frame(card_config, bg=T.BG_CARD)
        fds.pack(fill='x', padx=T.CARD_PADX, pady=(2, 4))
        tk.Label(fds, text='Dataset:', bg=T.BG_CARD, fg=T.FG_MUTED,
                 font=T.FONT_LABEL).pack(side='left', padx=(0, 8))
        self._combo_dataset = ttk.Combobox(
            fds, values=_DS_OPCOES, state='readonly',
            width=22, font=T.FONT_BODY)
        self._combo_dataset.set(_DS_OPCOES[0])
        self._combo_dataset.pack(side='left', fill='x', expand=True)
        self._combo_dataset.bind('<<ComboboxSelected>>', self._ao_mudar_dataset_cb)

        # Atributos RadioButtons
        tk.Label(card_config, text='Atributos ativos:', bg=T.BG_CARD, fg=T.FG_MUTED,
                 font=T.FONT_LABEL, anchor='w').pack(fill='x', padx=T.CARD_PADX, pady=(6, 2))
        for chave, cfg in CONFIGURACOES_ATRIBUTOS.items():
            tk.Radiobutton(card_config, text=cfg['rotulo_ui'],
                           variable=self.var_attr, value=chave,
                           bg=T.BG_CARD, fg=T.FG,
                           selectcolor=T.BG_HOVER,
                           activebackground=T.BG_CARD,
                           activeforeground=T.ACCENT_DEEP,
                           font=T.FONT_BODY, anchor='w',
                           borderwidth=0, highlightthickness=0,
                           command=self._ao_trocar_atributos
                          ).pack(fill='x', padx=T.CARD_PADX, pady=1)

        # Classificador RadioButtons
        tk.Label(card_config, text='Classificador:', bg=T.BG_CARD, fg=T.FG_MUTED,
                 font=T.FONT_LABEL, anchor='w').pack(fill='x', padx=T.CARD_PADX, pady=(8, 2))
        tk.Radiobutton(card_config, text='Bayes Otimo (QDA - Cov. Completa)',
                       variable=self.var_classifier, value='bayes',
                       bg=T.BG_CARD, fg=T.FG, selectcolor=T.BG_HOVER,
                       activebackground=T.BG_CARD, activeforeground=T.ACCENT_DEEP,
                       font=T.FONT_BODY, anchor='w', borderwidth=0, highlightthickness=0,
                       command=self._ao_trocar_classificador
                      ).pack(fill='x', padx=T.CARD_PADX, pady=1)
        tk.Radiobutton(card_config, text='Naive Bayes (Cov. Diagonal)',
                       variable=self.var_classifier, value='naive',
                       bg=T.BG_CARD, fg=T.FG, selectcolor=T.BG_HOVER,
                       activebackground=T.BG_CARD, activeforeground=T.ACCENT_DEEP,
                       font=T.FONT_BODY, anchor='w', borderwidth=0, highlightthickness=0,
                       command=self._ao_trocar_classificador
                      ).pack(fill='x', padx=T.CARD_PADX, pady=1)

        # 2. Modo do Gráfico
        card_grafico = Card(wrap, titulo='modo do grafico (2D)')
        card_grafico.grid(row=1, column=0, sticky='ew', pady=(0, T.GAP_SM))
        for rotulo, modo in MODOS_GRAFICO:
            tk.Radiobutton(card_grafico, text=rotulo,
                           variable=self.var_modo_grafico, value=modo,
                           bg=T.BG_CARD, fg=T.FG,
                           selectcolor=T.BG_HOVER,
                           activebackground=T.BG_CARD,
                           activeforeground=T.ACCENT_DEEP,
                           font=T.FONT_BODY, anchor='w',
                           borderwidth=0, highlightthickness=0,
                           command=self._desenhar_grafico
                          ).pack(fill='x', padx=T.CARD_PADX, pady=2)

        # 3. Classificação Manual
        card_classif = Card(wrap, titulo='classificacao manual')
        card_classif.grid(row=2, column=0, sticky='ew', pady=(0, T.GAP_SM))
        form = tk.Frame(card_classif, bg=T.BG_CARD)
        form.pack(fill='x', padx=T.CARD_PADX, pady=(2, 0))
        form.columnconfigure(1, weight=1)

        self.lbl_sx_title = tk.Label(form, text='Comp. Sepala', bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_LABEL, anchor='w')
        self.lbl_sx_title.grid(row=0, column=0, sticky='w', pady=(0, 2))
        self.ent_sx = ttk.Entry(form, textvariable=self.var_sx, font=T.FONT_MONO, width=10)
        self.ent_sx.grid(row=0, column=1, sticky='ew', padx=(8, 0))

        self.lbl_sy_title = tk.Label(form, text='Larg. Sepala', bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_LABEL, anchor='w')
        self.lbl_sy_title.grid(row=1, column=0, sticky='w', pady=(4, 2))
        self.ent_sy = ttk.Entry(form, textvariable=self.var_sy, font=T.FONT_MONO, width=10)
        self.ent_sy.grid(row=1, column=1, sticky='ew', padx=(8, 0))

        self.lbl_px_title = tk.Label(form, text='Comp. Petala', bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_LABEL, anchor='w')
        self.lbl_px_title.grid(row=2, column=0, sticky='w', pady=(4, 2))
        self.ent_px = ttk.Entry(form, textvariable=self.var_px, font=T.FONT_MONO, width=10)
        self.ent_px.grid(row=2, column=1, sticky='ew', padx=(8, 0))

        self.lbl_py_title = tk.Label(form, text='Larg. Petala', bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_LABEL, anchor='w')
        self.lbl_py_title.grid(row=3, column=0, sticky='w', pady=(4, 2))
        self.ent_py = ttk.Entry(form, textvariable=self.var_py, font=T.FONT_MONO, width=10)
        self.ent_py.grid(row=3, column=1, sticky='ew', padx=(8, 0))

        ttk.Button(card_classif, text='Classificar Amostra  >',
                    style='Primary.TButton',
                    command=self._classificar_amostra
                   ).pack(fill='x', padx=T.CARD_PADX, pady=(10, 6))
        
        separador(card_classif)
        
        self.lbl_pred = tk.Label(card_classif, text='—',
                                 bg=T.BG_CARD, fg=T.FG_DIM,
                                 font=T.FONT_VALUE_BIG, anchor='w')
        self.lbl_pred.pack(fill='x', padx=T.CARD_PADX, pady=(2, 1))
        
        self.lbl_pred_sub = tk.Label(card_classif, text='aguardando entrada',
                                     bg=T.BG_CARD, fg=T.FG_MUTED,
                                     font=T.FONT_MONO_SM, anchor='w',
                                     justify='left', wraplength=280)
        self.lbl_pred_sub.pack(fill='x', padx=T.CARD_PADX, pady=(0, 6))

        # Memoria de calculo (abre janela com formulas + substituicao)
        self.btn_memoria = ttk.Button(
            card_classif, text='Abrir memoria de calculo  >',
            style='Primary.TButton',
            command=self._abrir_memoria_calculo)
        self.btn_memoria.pack(fill='x', padx=T.CARD_PADX, pady=(2, 10))

        # 4. Normalidade Multivariada (R MVN)
        card_mvn = Card(wrap, titulo='normalidade multivariada (r)')
        card_mvn.grid(row=3, column=0, sticky='ew')
        
        self.lbl_r_status = tk.Label(card_mvn, text='Verificando R...', bg=T.BG_CARD, fg=T.FG_MUTED,
                                     font=T.FONT_KICKER, anchor='w')
        self.lbl_r_status.pack(fill='x', padx=T.CARD_PADX, pady=(2, 4))
        
        frame_txt_mvn = tk.Frame(card_mvn, bg=T.BG_PANEL,
                                 highlightthickness=1,
                                 highlightbackground=T.BORDER)
        frame_txt_mvn.pack(fill='x', padx=T.CARD_PADX, pady=(2, 6))

        self.txt_mvn = tk.Text(frame_txt_mvn, height=14, wrap='word',
                               bg=T.BG_PANEL, fg=T.FG_MUTED,
                               font=T.FONT_MONO_SM,
                               relief='flat', borderwidth=0,
                               highlightthickness=0,
                               padx=8, pady=6)
        sb_mvn = tk.Scrollbar(frame_txt_mvn, orient='vertical',
                              command=self.txt_mvn.yview)
        self.txt_mvn.configure(yscrollcommand=sb_mvn.set)
        self.txt_mvn.pack(side='left', fill='both', expand=True)
        sb_mvn.pack(side='right', fill='y')
        self.txt_mvn.configure(state='disabled')
        
        ttk.Button(card_mvn, text='Recalcular Normalidade no R',
                   style='Ghost.TButton',
                   command=self._executar_analise_normalidade
                  ).pack(fill='x', padx=T.CARD_PADX, pady=(2, 8))

        wrap.rowconfigure(4, weight=1)

    # ---- Coluna direita ----
    def _coluna_visualizacao(self):
        wrap = tk.Frame(self, bg=T.BG)
        wrap.grid(row=0, column=1, sticky='nsew',
                  padx=(T.GAP, T.PAD_PAGE), pady=T.PAD_PAGE)
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=2)   # grafico
        wrap.rowconfigure(1, weight=0)   # kpis
        wrap.rowconfigure(2, weight=3)   # notebook metricas

        # Plot Canvas
        painel = tk.Frame(wrap, bg=T.BG_CARD,
                          highlightthickness=1,
                          highlightbackground=T.BORDER,
                          highlightcolor=T.BORDER)
        painel.grid(row=0, column=0, sticky='nsew')
        painel.columnconfigure(0, weight=1)
        painel.rowconfigure(0, weight=1)
        painel.rowconfigure(1, weight=0)

        self.figura = Figure(figsize=(7, 4.2), dpi=100, facecolor=T.BG_CARD)
        self.ax = self.figura.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figura, master=painel)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew',
                                         padx=10, pady=(10, 2))

        self.toolbar = NavigationToolbar2Tk(self.canvas, painel, pack_toolbar=False)
        self.toolbar.update()
        self._estilizar_toolbar(self.toolbar)
        self.toolbar.grid(row=1, column=0, sticky='ew', padx=6, pady=(0, 6))

        # KPI blocks — linha horizontal (row=1)
        kpi_frame = tk.Frame(wrap, bg=T.BG)
        kpi_frame.grid(row=1, column=0, sticky='ew', pady=(T.GAP_SM, T.GAP_SM))
        kpi_frame.columnconfigure(0, weight=1)
        kpi_frame.columnconfigure(1, weight=1)
        kpi_frame.columnconfigure(2, weight=1)

        self.metric_acc   = MetricBlock(kpi_frame, 'acuracia teste', '—')
        self.metric_kappa = MetricBlock(kpi_frame, 'indice kappa', '—')
        self.metric_z     = MetricBlock(kpi_frame, 'Z (comparativo Kappa)', '—')
        self.metric_acc.grid(row=0, column=0, sticky='ew')
        self.metric_kappa.grid(row=0, column=1, sticky='ew', padx=(T.GAP_SM, 0))
        self.metric_z.grid(row=0, column=2, sticky='ew', padx=(T.GAP_SM, 0))

        # Notebook de metricas completas (row=2)
        self._nb_metricas = ttk.Notebook(wrap)
        self._nb_metricas.grid(row=2, column=0, sticky='nsew')

        self._aba_metricas_tab = tk.Frame(self._nb_metricas, bg=T.BG_CARD)
        self._aba_matriz_tab   = tk.Frame(self._nb_metricas, bg=T.BG_CARD)
        self._aba_kappa_tab    = tk.Frame(self._nb_metricas, bg=T.BG_CARD)

        self._nb_metricas.add(self._aba_metricas_tab, text='  Metricas Completas (d)  ')
        self._nb_metricas.add(self._aba_matriz_tab,   text='  Matriz de Confusao  ')
        self._nb_metricas.add(self._aba_kappa_tab,    text='  Comparacao Kappa (e)  ')

        for aba in [self._aba_metricas_tab, self._aba_matriz_tab, self._aba_kappa_tab]:
            tk.Label(aba, text='Aguardando dados...',
                     bg=T.BG_CARD, fg=T.FG_DIM, font=T.FONT_BODY
                    ).place(relx=0.5, rely=0.5, anchor='center')

    @staticmethod
    def _estilizar_toolbar(toolbar):
        toolbar.configure(background=T.BG_CARD)
        for child in toolbar.winfo_children():
            try:
                child.configure(background=T.BG_CARD,
                                activebackground=T.BG_HOVER,
                                relief='flat', borderwidth=0)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Carregamento e Orquestração
    # ------------------------------------------------------------------
    def _carregar_dados(self):
        caminho = CAMINHOS_DADOS[self.var_dataset.get()]
        if os.path.exists(caminho):
            self.dados = carregar_dados_iris(caminho)
            self.dados_treino, self.dados_teste = split_estratificado(
                self.dados, proporcao_treino=0.7, semente=42)
        else:
            self.dados = []
            self.dados_treino = []
            self.dados_teste = []

    def _executar_analise_normalidade(self):
        caminho_dados = CAMINHOS_DADOS[self.var_dataset.get()]
        pasta_out = os.path.join(PROJETO_ROOT, 'outputs')
        
        # Executar analise MVN
        relatorio, dados_mvn, r_ok = executar_analise_mvn(caminho_dados, pasta_out)
        self.dados_mvn = dados_mvn
        self.r_disponivel = r_ok
        
        # Atualizar UI
        status_text = "R-MVN STATUS: " + ("R DISPONIVEL (MVN OK)" if r_ok else "R NAO DISPONIVEL (FALLBACK REAL)")
        status_color = T.SUCCESS if r_ok else T.ACCENT_DEEP
        self.lbl_r_status.configure(text=status_text, fg=status_color)
        
        self.txt_mvn.configure(state='normal')
        self.txt_mvn.delete('1.0', 'end')
        self.txt_mvn.insert('1.0', relatorio)
        self.txt_mvn.configure(state='disabled')

    def _atualizar_modelo(self):
        if not self.dados_treino:
            return
            
        attr_key = self.var_attr.get()
        indices_sel = CONFIGURACOES_ATRIBUTOS[attr_key]['indices']
        
        # Treinar os dois modelos (Bayes Ótimo e Naive Bayes) na base atual
        self.model_bayes = treinar_bayes(self.dados_treino, indices_sel, naive=False)
        self.model_naive = treinar_bayes(self.dados_treino, indices_sel, naive=True)
        
        # Fazer predicoes no conjunto de teste para calcular as metricas completas
        gab = [d['classe'] for d in self.dados_teste]
        
        # Bayes Ótimo
        preds_bayes = []
        for d in self.dados_teste:
            _, pred = predizer_todas_classes_bayes(d['atributos'], self.model_bayes, indices_sel)
            preds_bayes.append(pred)
        self.rel_bayes = relatorio_completo(preds_bayes, gab, CLASSES, "Bayes Otimo")
        
        # Naive Bayes
        preds_naive = []
        for d in self.dados_teste:
            _, pred = predizer_todas_classes_bayes(d['atributos'], self.model_naive, indices_sel)
            preds_naive.append(pred)
        self.rel_naive = relatorio_completo(preds_naive, gab, CLASSES, "Naive Bayes")
        
        # Configurar campos manuais visiveis
        self._ajustar_campos_classificacao_manual()
        
        # Atualizar KPIs de acordo com o classificador selecionado
        self._atualizar_kpi_e_analise()
        
        # Desenhar grafico
        self._desenhar_grafico()

    def _ajustar_campos_classificacao_manual(self):
        attr_key = self.var_attr.get()
        if attr_key == 'todas':
            self.lbl_sx_title.grid()
            self.ent_sx.grid()
            self.lbl_sy_title.grid()
            self.ent_sy.grid()
        else:
            self.lbl_sx_title.grid_remove()
            self.ent_sx.grid_remove()
            self.lbl_sy_title.grid_remove()
            self.ent_sy.grid_remove()
            
        # Atualiza labels das entradas principais
        cfg = CONFIGURACOES_ATRIBUTOS[attr_key]
        if attr_key == 'todas':
            # Se todas, x e y no form representam as petalas
            self.lbl_px_title.configure(text='Comp. Petala')
            self.lbl_py_title.configure(text='Larg. Petala')
        else:
            self.lbl_px_title.configure(text=cfg['eixo_x'])
            self.lbl_py_title.configure(text=cfg['eixo_y'])

    def _ao_mudar_dataset_cb(self, event):
        selecionado = self._combo_dataset.get()
        chave = _DS_CHAVE[selecionado]
        self.var_dataset.set(chave)
        self._carregar_dados()
        self._executar_analise_normalidade()
        self._atualizar_modelo()

    def _ao_trocar_atributos(self):
        # Se for selecionado 'todas' e o grafico estiver em modo par, força para dispersao
        if self.var_attr.get() == 'todas' and self.var_modo_grafico.get() != 'dispersao':
            self.var_modo_grafico.set('dispersao')
        
        self.lbl_pred.configure(text='—', fg=T.FG_DIM)
        self.lbl_pred_sub.configure(text='aguardando entrada')
        self._atualizar_modelo()

    def _ao_trocar_classificador(self):
        self._atualizar_kpi_e_analise()
        self._desenhar_grafico()

    def _classificar_amostra(self):
        attr_key = self.var_attr.get()
        cfg = CONFIGURACOES_ATRIBUTOS[attr_key]
        
        try:
            px = float(self.var_px.get().replace(',', '.'))
            py = float(self.var_py.get().replace(',', '.'))
            if attr_key == 'todas':
                sx = float(self.var_sx.get().replace(',', '.'))
                sy = float(self.var_sy.get().replace(',', '.'))
        except ValueError:
            self.lbl_pred.configure(text='—', fg=T.DANGER)
            self.lbl_pred_sub.configure(text='Valores invalidos. Use numeros decimais.')
            return

        # Montar o vetor x de acordo com os atributos selecionados
        if attr_key == 'todas':
            x = [sx, sy, px, py]
            model = self.model_bayes if self.var_classifier.get() == 'bayes' else self.model_naive
            scores, vencedor = predizer_todas_classes_bayes(x, model, cfg['indices'])
        else:
            x = [0.0] * 4
            x[cfg['indices'][0]] = px
            x[cfg['indices'][1]] = py
            model = self.model_bayes if self.var_classifier.get() == 'bayes' else self.model_naive
            scores, vencedor = predizer_todas_classes_bayes(x, model, cfg['indices'])

        cor = CORES_CLASSE[vencedor]
        self.lbl_pred.configure(text=vencedor.upper(), fg=cor)
        
        outros_scores = '  '.join(f'{c[:3]}={scores[c]:+.2f}' for c in CLASSES if c != vencedor)
        self.lbl_pred_sub.configure(
            text=f'score = {scores[vencedor]:+.3f}\n{outros_scores}'
        )

    # ------------------------------------------------------------------
    # Métricas e Análise
    # ------------------------------------------------------------------
    def _atualizar_kpi_e_analise(self):
        cls_key = self.var_classifier.get()
        rel_atual = self.rel_bayes if cls_key == 'bayes' else self.rel_naive

        if not rel_atual:
            return

        acc = rel_atual['acerto_global']
        k   = rel_atual['kappa']

        # Calcular Z e p-valor entre os dois modelos
        z_stat = z_kappa(self.rel_bayes['kappa'], self.rel_bayes['variancia_kappa'],
                         self.rel_naive['kappa'],  self.rel_naive['variancia_kappa'])
        p_val  = p_valor_z(z_stat)

        # KPI blocks
        self.metric_acc.set(f"{acc:.2%}", T.SUCCESS if acc > 0.9 else T.ACCENT_DEEP)
        self.metric_kappa.set(f"{k:.4f}",
                              T.SUCCESS if k > 0.80 else T.ACCENT if k > 0.40 else T.DANGER)
        self.metric_z.set(f"Z={z_stat:.3f}  p={p_val:.3f}",
                          T.DANGER if p_val < 0.05 else T.SUCCESS)

        # Preencher abas do notebook
        self._preencher_metricas_completas()
        self._preencher_matriz_confusao()
        self._preencher_comparacao_kappa()

    # ------------------------------------------------------------------
    # Plotagem Matplotlib
    # ------------------------------------------------------------------
    def _desenhar_grafico(self):
        self.ax.clear()
        self.ax.set_facecolor(T.BG_CARD)
        
        # Configurar bordas do subplot
        for spine in self.ax.spines.values():
            spine.set_color(T.BORDER_HARD)
            spine.set_linewidth(0.8)
        self.ax.tick_params(colors=T.FG_MUTED, labelsize=8, length=4, width=0.8)
        self.ax.grid(True, color=T.BORDER, linestyle=':', linewidth=0.6, alpha=0.7)
        
        attr_key = self.var_attr.get()
        cfg = CONFIGURACOES_ATRIBUTOS[attr_key]
        indices = cfg['indices']
        
        modo = self.var_modo_grafico.get()
        cls_key = self.var_classifier.get()
        model_params = self.model_bayes if cls_key == 'bayes' else self.model_naive
        
        if attr_key == 'todas':
            # Se 4D, sempre exibe dispersao projetada
            self._desenhar_dispersao([2, 3], CONFIGURACOES_ATRIBUTOS['petalas'], model_params, proj_4d=True)
        elif modo == 'dispersao':
            self._desenhar_dispersao(indices, cfg, model_params)
        else:
            # Fronteira binaria
            classe_i, classe_j = PAR_POR_MODO[modo]
            self._desenhar_fronteira_binaria(indices, cfg, model_params, classe_i, classe_j)
            
        self.ax.set_xlabel(cfg['eixo_x'] + f" ({cfg['unidade']})")
        self.ax.set_ylabel(cfg['eixo_y'] + f" ({cfg['unidade']})")
        self.ax.title.set_color(T.FG)
        self.ax.xaxis.label.set_color(T.FG_MUTED)
        self.ax.yaxis.label.set_color(T.FG_MUTED)
        self.figura.tight_layout(pad=1.2)
        self.canvas.draw()

    def _desenhar_dispersao(self, indices, cfg, model_params, proj_4d=False):
        ids_treino = set(id(d) for d in self.dados_treino)
        
        # Plotar as amostras
        for classe in ['virginica', 'versicolor', 'setosa']:
            cor = CORES_CLASSE[classe]
            marcador = MARCADORES_CLASSE[classe]
            
            treino = [d for d in self.dados if d['classe'] == classe and id(d) in ids_treino]
            teste = [d for d in self.dados if d['classe'] == classe and id(d) not in ids_treino]
            
            x1_tr = [d['atributos'][indices[0]] for d in treino]
            x2_tr = [d['atributos'][indices[1]] for d in treino]
            self.ax.scatter(x1_tr, x2_tr, color=cor, marker=marcador, label=f"{classe} (treino)",
                            edgecolors='white', linewidths=0.6, s=50, alpha=0.8, zorder=3)
            
            x1_te = [d['atributos'][indices[0]] for d in teste]
            x2_te = [d['atributos'][indices[1]] for d in teste]
            self.ax.scatter(x1_te, x2_te, color=cor, marker=marcador, label=f"{classe} (teste)",
                            edgecolors='black', linewidths=1.0, s=70, alpha=0.9, zorder=4)

        # Plotar vetores médios
        for classe in CLASSES:
            cor = CORES_CLASSE[classe]
            if proj_4d:
                # Projeta média correspondente às pétalas (índices 2 e 3)
                # O model_params de 'todas' tem 4 dimensões
                media = [model_params[classe]['media'][2], model_params[classe]['media'][3]]
            else:
                media = model_params[classe]['media']
                
            self.ax.scatter(media[0], media[1], color=cor, marker='X', s=200,
                            edgecolors='black', linewidths=1.5, zorder=5, label=f"Media {classe}")
            self.ax.text(media[0] + 0.05, media[1] + 0.05, classe, fontweight='bold', fontsize=8)

        titulo = f"Dispersao Geral - {'Projecao Petalas (4D)' if proj_4d else cfg['rotulo']}"
        self.ax.set_title(titulo, fontsize=11, fontweight='bold')
        self.ax.legend(loc='best', fontsize=8, framealpha=0.9)

    def _desenhar_fronteira_binaria(self, indices, cfg, model_params, classe_i, classe_j):
        # 1. Filtrar dados das duas classes
        dados_par = filtrar_por_classes(self.dados, [classe_i, classe_j])
        treino_par = filtrar_por_classes(self.dados_treino, [classe_i, classe_j])
        
        # Coletar pontos
        dados_c1 = [d for d in dados_par if d['classe'] == classe_i]
        dados_c2 = [d for d in dados_par if d['classe'] == classe_j]
        
        x1_c1 = [d['atributos'][indices[0]] for d in dados_c1]
        x2_c1 = [d['atributos'][indices[1]] for d in dados_c1]
        x1_c2 = [d['atributos'][indices[0]] for d in dados_c2]
        x2_c2 = [d['atributos'][indices[1]] for d in dados_c2]

        todos_x1 = x1_c1 + x1_c2
        todos_x2 = x2_c1 + x2_c2
        margem = 0.4
        x1_min = min(todos_x1) - margem
        x1_max = max(todos_x1) + margem
        x2_min = min(todos_x2) - margem
        x2_max = max(todos_x2) + margem

        cor_i = CORES_CLASSE[classe_i]
        cor_j = CORES_CLASSE[classe_j]
        marcador_i = MARCADORES_CLASSE[classe_i]
        marcador_j = MARCADORES_CLASSE[classe_j]

        # 2. Avaliar discriminante local num grid 2D
        resolucao = 100
        grid_x1 = [x1_min + (x1_max - x1_min) * k / (resolucao - 1) for k in range(resolucao)]
        grid_x2 = [x2_min + (x2_max - x2_min) * k / (resolucao - 1) for k in range(resolucao)]
        
        X = [[x1 for x1 in grid_x1] for _ in range(resolucao)]
        Y = [[x2 for _ in range(resolucao)] for x2 in grid_x2]
        Z = [[0.0 for _ in range(resolucao)] for _ in range(resolucao)]
        
        # Treinar modelo local 2D para as duas classes para gerar a fronteira local correta
        local_model = treinar_bayes(treino_par, indices, naive=(self.var_classifier.get() == 'naive'))
        params_i = local_model[classe_i]
        params_j = local_model[classe_j]
        
        for r in range(resolucao):
            x2 = grid_x2[r]
            for c in range(resolucao):
                x1 = grid_x1[c]
                pt = [x1, x2]
                
                d_m_i = distancia_mahalanobis_quad(pt, params_i['media'], params_i['inv_cov'])
                score_i = -0.5 * math.log(params_i['det']) - 0.5 * d_m_i
                
                d_m_j = distancia_mahalanobis_quad(pt, params_j['media'], params_j['inv_cov'])
                score_j = -0.5 * math.log(params_j['det']) - 0.5 * d_m_j
                
                Z[r][c] = score_i - score_j

        # Colorir regiões
        self.ax.contourf(X, Y, Z, levels=[-999999, 0.0, 999999], colors=[cor_j, cor_i], alpha=0.13)
        # Linha da fronteira
        self.ax.contour(X, Y, Z, levels=[0.0], colors=['black'], linestyles='--', linewidths=1.8)

        # Plotar pontos
        ids_treino = set(id(d) for d in self.dados_treino)
        for classe, dados_classe, cor, marcador in [
            (classe_i, dados_c1, cor_i, marcador_i),
            (classe_j, dados_c2, cor_j, marcador_j),
        ]:
            dados_tr = [d for d in dados_classe if id(d) in ids_treino]
            x1_tr = [d['atributos'][indices[0]] for d in dados_tr]
            x2_tr = [d['atributos'][indices[1]] for d in dados_tr]
            self.ax.scatter(x1_tr, x2_tr, color=cor, marker=marcador, label=f'{classe} (treino)',
                            edgecolors='white', linewidths=0.6, s=50, alpha=0.8, zorder=3)

            dados_te = [d for d in dados_classe if id(d) not in ids_treino]
            x1_te = [d['atributos'][indices[0]] for d in dados_te]
            x2_te = [d['atributos'][indices[1]] for d in dados_te]
            self.ax.scatter(x1_te, x2_te, color=cor, marker=marcador, label=f'{classe} (teste)',
                            edgecolors='black', linewidths=1.0, s=70, alpha=0.9, zorder=4)

        # Médias (Centróides)
        self.ax.scatter(params_i['media'][0], params_i['media'][1], color=cor_i, marker='X', s=200,
                        edgecolors='black', linewidths=1.5, label=f'Media {classe_i}', zorder=5)
        self.ax.scatter(params_j['media'][0], params_j['media'][1], color=cor_j, marker='X', s=200,
                        edgecolors='black', linewidths=1.5, label=f'Media {classe_j}', zorder=5)

        self.ax.set_xlim(x1_min, x1_max)
        self.ax.set_ylim(x2_min, x2_max)
        
        # Nome do modelo no título
        nome_mod = "Bayes Otimo (QDA)" if self.var_classifier.get() == 'bayes' else "Naive Bayes"
        self.ax.set_title(f"{nome_mod}: {classe_i} vs {classe_j}", fontsize=11, fontweight='bold')
        
        # Dummy line para legenda
        from matplotlib.lines import Line2D
        handles, labels = self.ax.get_legend_handles_labels()
        handles.append(Line2D([0], [0], color='black', linestyle='--', linewidth=1.8))
        labels.append('Fronteira di(x)=dj(x)')
        
        self.ax.legend(handles, labels, loc='best', fontsize=8, framealpha=0.9)

    # ------------------------------------------------------------------
    # Item (d) — Metricas completas (tabela comparativa Bayes vs Naive)
    # ------------------------------------------------------------------
    def _preencher_metricas_completas(self):
        for w in self._aba_metricas_tab.winfo_children():
            w.destroy()

        if not self.rel_bayes or not self.rel_naive:
            tk.Label(self._aba_metricas_tab, text='Aguardando dados...',
                     bg=T.BG_CARD, fg=T.FG_DIM, font=T.FONT_BODY
                    ).place(relx=0.5, rely=0.5, anchor='center')
            return

        outer = tk.Frame(self._aba_metricas_tab, bg=T.BG_CARD)
        outer.pack(fill='both', expand=True, padx=8, pady=6)

        attr_key = self.var_attr.get()
        tk.Label(outer,
                 text=f'Item (d) — Metricas de Qualidade  |  '
                      f'{CONFIGURACOES_ATRIBUTOS[attr_key]["rotulo_ui"]}',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(0, 3))
        tk.Label(outer,
                 text='Verde = melhor entre os dois classificadores. '
                      'Ac.Prod = Acuracia do Produtor (Sensibilidade). '
                      'Ac.Usu = Acuracia do Usuario (Precisao).',
                 bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_LABEL,
                 justify='left', anchor='w', wraplength=680
                ).pack(fill='x', pady=(0, 6))

        frame = tk.Frame(outer, bg=T.BG_CARD)
        frame.pack(fill='both', expand=True)

        canvas = tk.Canvas(frame, bg=T.BG_CARD, highlightthickness=0)
        sb_y   = ttk.Scrollbar(frame, orient='vertical', command=canvas.yview)
        inner  = tk.Frame(canvas, bg=T.BG_CARD)
        inner.bind('<Configure>',
                   lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=sb_y.set)
        canvas.pack(side='left', fill='both', expand=True)
        sb_y.pack(side='right', fill='y')

        def cel(row, col, texto, cor=T.FG, bg_c=T.BG_PANEL, larg=18, bold=False):
            f = T.FONT_CELL_BOLD if bold else T.FONT_MONO_SM
            tk.Label(inner, text=texto, bg=bg_c, fg=cor, font=f,
                     width=larg, anchor='center',
                     highlightthickness=1, highlightbackground=T.BORDER
                    ).grid(row=row, column=col, sticky='nsew', padx=1, pady=1)

        def melhor_cor(v1_str, v2_str):
            try:
                f1 = float(str(v1_str).replace('%', '')) / (100.0 if '%' in str(v1_str) else 1.0)
                f2 = float(str(v2_str).replace('%', '')) / (100.0 if '%' in str(v2_str) else 1.0)
                if f1 > f2: return T.SUCCESS, T.FG
                if f2 > f1: return T.FG, T.SUCCESS
            except Exception:
                pass
            return T.FG, T.FG

        # Cabecalho
        for j, h in enumerate(['Metrica / Classe', 'Bayes Otimo (QDA)', 'Naive Bayes']):
            cel(0, j, h, cor=T.ACCENT_DEEP, bold=True)

        row = 1

        # Metricas globais
        for label, v1, v2 in [
            ('Acerto Global (Ag)',
             f'{self.rel_bayes["acerto_global"]:.2%}',
             f'{self.rel_naive["acerto_global"]:.2%}'),
            ('Kappa',
             f'{self.rel_bayes["kappa"]:.6f}',
             f'{self.rel_naive["kappa"]:.6f}'),
            ('Tau',
             f'{self.rel_bayes["tau"]:.6f}',
             f'{self.rel_naive["tau"]:.6f}'),
            ('Var(Kappa)',
             f'{self.rel_bayes["variancia_kappa"]:.8f}',
             f'{self.rel_naive["variancia_kappa"]:.8f}'),
        ]:
            bg_r = T.BG_CARD if row % 2 == 0 else T.BG_PANEL
            c1, c2 = melhor_cor(v1, v2)
            cel(row, 0, label, cor=T.FG_MUTED, bg_c=bg_r)
            cel(row, 1, v1, cor=c1, bg_c=bg_r)
            cel(row, 2, v2, cor=c2, bg_c=bg_r)
            row += 1

        # Metricas por classe
        for c in CLASSES:
            pc_b = self.rel_bayes['por_classe'][c]
            pc_n = self.rel_naive['por_classe'][c]
            cor_c = CORES_CLASSE[c]

            for label, v1, v2 in [
                (f'{c.capitalize()}  Ac. Produtor',
                 f'{pc_b["acuracia_produtor"]:.2%}',
                 f'{pc_n["acuracia_produtor"]:.2%}'),
                (f'{c.capitalize()}  Ac. Usuario',
                 f'{pc_b["acuracia_usuario"]:.2%}',
                 f'{pc_n["acuracia_usuario"]:.2%}'),
                (f'{c.capitalize()}  Sensibilidade',
                 f'{pc_b["sensibilidade"]:.2%}',
                 f'{pc_n["sensibilidade"]:.2%}'),
                (f'{c.capitalize()}  Especificidade',
                 f'{pc_b["especificidade"]:.2%}',
                 f'{pc_n["especificidade"]:.2%}'),
                (f'{c.capitalize()}  F1 (b=1)',
                 f'{pc_b["f1"]:.4f}',
                 f'{pc_n["f1"]:.4f}'),
                (f'{c.capitalize()}  F2 (b=2)',
                 f'{pc_b["f2"]:.4f}',
                 f'{pc_n["f2"]:.4f}'),
                (f'{c.capitalize()}  MCC',
                 f'{pc_b["mcc"]:.4f}',
                 f'{pc_n["mcc"]:.4f}'),
            ]:
                bg_r = T.BG_CARD if row % 2 == 0 else T.BG_PANEL
                c1, c2 = melhor_cor(v1, v2)
                cel(row, 0, label, cor=cor_c, bg_c=bg_r)
                cel(row, 1, v1, cor=c1, bg_c=bg_r)
                cel(row, 2, v2, cor=c2, bg_c=bg_r)
                row += 1

    # ------------------------------------------------------------------
    # Item (d) — Matriz de Confusao visual
    # ------------------------------------------------------------------
    def _preencher_matriz_confusao(self):
        for w in self._aba_matriz_tab.winfo_children():
            w.destroy()

        cls_key = self.var_classifier.get()
        rel     = self.rel_bayes if cls_key == 'bayes' else self.rel_naive
        nome    = 'Bayes Otimo (QDA)' if cls_key == 'bayes' else 'Naive Bayes'

        if not rel:
            tk.Label(self._aba_matriz_tab, text='Aguardando dados...',
                     bg=T.BG_CARD, fg=T.FG_DIM, font=T.FONT_BODY
                    ).place(relx=0.5, rely=0.5, anchor='center')
            return

        outer = tk.Frame(self._aba_matriz_tab, bg=T.BG_CARD)
        outer.pack(fill='both', expand=True, padx=12, pady=8)

        tk.Label(outer,
                 text=f'Matriz de Confusao — {nome}  |  Linhas = Predito  ·  Colunas = Real',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(0, 8))

        matriz  = rel['matriz']
        classes = CLASSES
        n       = len(classes)

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

        tk.Label(grid, text='Total', bg=T.BG_PANEL, fg=T.ACCENT_DEEP,
                 font=T.FONT_KICKER, width=10, anchor='center',
                 highlightthickness=1, highlightbackground=T.BORDER
                ).grid(row=0, column=n + 1, padx=2, pady=2)

        totais_colunas = {real: 0 for real in classes}
        total_geral    = 0

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

            tk.Label(grid, text=str(total_linha), bg=T.BG_PANEL, fg=T.FG,
                     font=T.FONT_CELL_LG, width=10, anchor='center',
                     highlightthickness=1, highlightbackground=T.BORDER
                    ).grid(row=i + 1, column=n + 1, padx=2, pady=2)
            total_geral += total_linha

        tk.Label(grid, text='Total', bg=T.BG_PANEL, fg=T.ACCENT_DEEP,
                 font=T.FONT_KICKER, width=10, anchor='center',
                 highlightthickness=1, highlightbackground=T.BORDER
                ).grid(row=n + 1, column=0, padx=2, pady=2)

        for j, real in enumerate(classes):
            tk.Label(grid, text=str(totais_colunas[real]),
                     bg=T.BG_PANEL, fg=T.FG,
                     font=T.FONT_CELL_LG, width=10, anchor='center',
                     highlightthickness=1, highlightbackground=T.BORDER
                    ).grid(row=n + 1, column=j + 1, padx=2, pady=2)

        tk.Label(grid, text=str(total_geral), bg=T.BG_PANEL, fg=T.ACCENT_DEEP,
                 font=T.FONT_CELL_LG, width=10, anchor='center',
                 highlightthickness=1, highlightbackground=T.BORDER
                ).grid(row=n + 1, column=n + 1, padx=2, pady=2)

        # Acuracia produtor e usuario
        info = tk.Frame(outer, bg=T.BG_CARD)
        info.pack(fill='x', pady=(12, 0))
        tk.Label(info, text='ACURACIA DO PRODUTOR (Sensibilidade — colunas)',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
                ).grid(row=0, column=0, columnspan=n, sticky='w', pady=(0, 4))
        tk.Label(info, text='ACURACIA DO USUARIO (Precisao — linhas)',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
                ).grid(row=0, column=n, columnspan=n, sticky='w', pady=(0, 4), padx=(20, 0))

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

    # ------------------------------------------------------------------
    # Item (e) — Teste Z de Kappa: Bayes Otimo vs Naive Bayes
    # ------------------------------------------------------------------
    def _preencher_comparacao_kappa(self):
        for w in self._aba_kappa_tab.winfo_children():
            w.destroy()

        if not self.rel_bayes or not self.rel_naive:
            tk.Label(self._aba_kappa_tab, text='Aguardando dados...',
                     bg=T.BG_CARD, fg=T.FG_DIM, font=T.FONT_BODY
                    ).place(relx=0.5, rely=0.5, anchor='center')
            return

        outer = tk.Frame(self._aba_kappa_tab, bg=T.BG_CARD)
        outer.pack(fill='both', expand=True, padx=14, pady=10)

        attr_key = self.var_attr.get()
        k1  = self.rel_bayes['kappa']
        k2  = self.rel_naive['kappa']
        t1  = self.rel_bayes['tau']
        t2  = self.rel_naive['tau']
        vk1 = self.rel_bayes['variancia_kappa']
        vk2 = self.rel_naive['variancia_kappa']
        vt1 = self.rel_bayes['variancia_tau']
        vt2 = self.rel_naive['variancia_tau']
        ag1 = self.rel_bayes['acerto_global']
        ag2 = self.rel_naive['acerto_global']

        zk  = z_kappa(k1, vk1, k2, vk2)
        zt  = z_tau(t1, vt1, t2, vt2)
        pzk = p_valor_z(zk)
        pzt = p_valor_z(zt)
        sig_k = pzk < 0.05
        sig_t = pzt < 0.05

        tk.Label(outer,
                 text='Item (e) — Qual classificador tem maior acuracia? '
                      'Teste de Significancia de Kappa (Z-test)',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(0, 3))
        tk.Label(outer,
                 text=f'Atributos: {CONFIGURACOES_ATRIBUTOS[attr_key]["rotulo_ui"]}   |   '
                      'H0: K_Bayes = K_Naive  |  H1: K_Bayes ≠ K_Naive  (alfa = 5%)',
                 bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_LABEL,
                 justify='left', anchor='w', wraplength=720
                ).pack(fill='x', pady=(0, 10))

        # Tabela comparativa
        tab = tk.Frame(outer, bg=T.BG_CARD)
        tab.pack(fill='x', pady=(0, 12))

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

        for col, h in enumerate(['Metrica', 'Bayes Otimo', 'Naive Bayes',
                                  'Z calculado', 'p-valor', 'Conclusao (5%)']):
            th(col, h)

        melhor_ag = ('Bayes Otimo' if ag1 > ag2 else
                     ('Naive Bayes' if ag2 > ag1 else 'Empate'))

        linhas = [
            ('Acerto Global',
             f'{ag1:.2%}', f'{ag2:.2%}',
             '—', '—',
             f'Maior: {melhor_ag}'),
            ('Kappa',
             f'{k1:.6f}', f'{k2:.6f}',
             f'{zk:.4f}', f'{pzk:.4f}',
             'SIGNIFICATIVO' if sig_k else 'nao significativo'),
            ('Tau',
             f'{t1:.6f}', f'{t2:.6f}',
             f'{zt:.4f}', f'{pzt:.4f}',
             'SIGNIFICATIVO' if sig_t else 'nao significativo'),
            ('Var(Kappa)',
             f'{vk1:.8f}', f'{vk2:.8f}', '—', '—', '—'),
            ('Var(Tau)',
             f'{vt1:.8f}', f'{vt2:.8f}', '—', '—', '—'),
        ]

        for r, (m, v1, v2, z, p, conc) in enumerate(linhas):
            cor_conc = (T.DANGER if 'SIGNIFICATIVO' in conc and 'nao' not in conc
                        else T.SUCCESS if 'nao' in conc
                        else T.FG_MUTED)
            # Destacar melhor valor (verde para maior)
            if m == 'Acerto Global':
                c1 = T.SUCCESS if ag1 > ag2 else T.FG
                c2 = T.SUCCESS if ag2 > ag1 else T.FG
            elif m == 'Kappa':
                c1 = T.SUCCESS if k1 > k2 else T.FG
                c2 = T.SUCCESS if k2 > k1 else T.FG
            elif m == 'Tau':
                c1 = T.SUCCESS if t1 > t2 else T.FG
                c2 = T.SUCCESS if t2 > t1 else T.FG
            else:
                c1 = c2 = T.FG

            td(r + 1, 0, m, T.FG_MUTED)
            td(r + 1, 1, v1, c1)
            td(r + 1, 2, v2, c2)
            td(r + 1, 3, z)
            td(r + 1, 4, p)
            td(r + 1, 5, conc, cor_conc)

        # Conclusao narrativa
        tk.Frame(outer, bg=T.BORDER, height=1).pack(fill='x', pady=(4, 8))

        if sig_k:
            melhor = 'Bayes Otimo (QDA)' if k1 > k2 else 'Naive Bayes'
            conclusao = (
                f'CONCLUSAO: Existe diferenca ESTATISTICAMENTE SIGNIFICATIVA '
                f'(Z = {zk:.4f},  p = {pzk:.6f}  <  0.05).\n'
                f'O classificador {melhor} possui Kappa significativamente superior '
                f'e deve ser preferido para este conjunto de atributos.'
            )
            cor_conc_lbl = T.DANGER
        else:
            melhor = 'Bayes Otimo (QDA)' if k1 > k2 else 'Naive Bayes'
            if abs(k1 - k2) < 1e-9:
                conclusao = (
                    f'CONCLUSAO: Os dois classificadores possuem desempenho IDENTICO '
                    f'(Z = {zk:.4f},  p = {pzk:.6f}  >=  0.05).\n'
                    f'Nao ha diferenca estatisticamente significativa ao nivel de 5%.'
                )
            else:
                conclusao = (
                    f'CONCLUSAO: NAO existe diferenca estatisticamente significativa '
                    f'(Z = {zk:.4f},  p = {pzk:.6f}  >=  0.05).\n'
                    f'O {melhor} possui maior acuracia nominal '
                    f'({ag1:.2%} vs {ag2:.2%}), porem a diferenca nao e '
                    f'significativa ao nivel de 5%.'
                )
            cor_conc_lbl = T.SUCCESS

        tk.Label(outer, text=conclusao,
                 bg=T.BG_CARD, fg=cor_conc_lbl, font=T.FONT_BODY,
                 justify='left', anchor='w', wraplength=720
                ).pack(fill='x', pady=(0, 10))

    def _abrir_memoria_calculo(self):
        """Abre a janela de memoria de calculo para Bayes & Naive Bayes."""
        from .janela_calculos import JanelaMemoriaCalculoBayes
        
        attr_key = self.var_attr.get()
        indices_sel = CONFIGURACOES_ATRIBUTOS[attr_key]['indices']
        
        # Obter os valores atuais do form
        try:
            px = float(self.var_px.get().replace(',', '.'))
            py = float(self.var_py.get().replace(',', '.'))
            if attr_key == 'todas':
                sx = float(self.var_sx.get().replace(',', '.'))
                sy = float(self.var_sy.get().replace(',', '.'))
                amostra = [sx, sy, px, py]
            else:
                amostra = [0.0] * 4
                amostra[indices_sel[0]] = px
                amostra[indices_sel[1]] = py
        except ValueError:
            # Fallback para amostra padrao
            amostra = [5.8, 3.0, 4.5, 1.5]
            
        JanelaMemoriaCalculoBayes(
            self,
            self.model_bayes,
            self.model_naive,
            CONFIGURACOES_ATRIBUTOS[attr_key]['rotulo_ui'],
            indices_sel,
            amostra=amostra,
            rel_bayes=self.rel_bayes,
            rel_naive=self.rel_naive
        )
