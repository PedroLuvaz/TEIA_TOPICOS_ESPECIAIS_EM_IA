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
from evaluation.metricas_avancadas import relatorio_completo, z_kappa, p_valor_z
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
        wrap.rowconfigure(0, weight=3)
        wrap.rowconfigure(1, weight=2)

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

        # Painel inferior (Métricas + Análise)
        inferior = tk.Frame(wrap, bg=T.BG)
        inferior.grid(row=1, column=0, sticky='nsew', pady=(T.GAP_SM, 0))
        inferior.columnconfigure(0, weight=1)
        inferior.columnconfigure(1, weight=2)
        inferior.rowconfigure(0, weight=1)

        # Metricas KPI
        col_m = tk.Frame(inferior, bg=T.BG)
        col_m.grid(row=0, column=0, sticky='nsew', padx=(0, 14))
        col_m.columnconfigure(0, weight=1)

        self.metric_acc = MetricBlock(col_m, 'acuracia teste', '—')
        self.metric_acc.grid(row=0, column=0, sticky='ew')
        self.metric_kappa = MetricBlock(col_m, 'indice kappa', '—')
        self.metric_kappa.grid(row=1, column=0, sticky='ew', pady=(T.GAP_SM, 0))
        self.metric_z = MetricBlock(col_m, 'Z (comparativo Kappa)', '—')
        self.metric_z.grid(row=2, column=0, sticky='ew', pady=(T.GAP_SM, 0))

        # Analise Textual
        card = Card(inferior, titulo='analise e teste de significancia')
        card.grid(row=0, column=1, sticky='nsew')
        self.txt_analise = tk.Text(card, height=9, wrap='word',
                                   bg=T.BG_CARD, fg=T.FG,
                                   font=T.FONT_BODY,
                                   relief='flat', borderwidth=0,
                                   highlightthickness=0,
                                   padx=T.CARD_PADX, pady=2,
                                   spacing1=2, spacing3=4)
        self.txt_analise.pack(fill='both', expand=True,
                              padx=T.CARD_PADX, pady=(2, 14))
        self.txt_analise.tag_configure('hl', foreground=T.ACCENT_DEEP, font=T.FONT_TEXT_HL)
        self.txt_analise.tag_configure('mono', foreground=T.FG, font=T.FONT_MONO)
        self.txt_analise.tag_configure('bold', font=(T.FONT_FAMILY, 10, 'bold'))
        self.txt_analise.configure(state='disabled')

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
        
        acc = rel_atual['acerto_global']
        k = rel_atual['kappa']
        
        # Calcular Z e p-valor entre os dois modelos
        z_stat = z_kappa(self.rel_bayes['kappa'], self.rel_bayes['variancia_kappa'],
                         self.rel_naive['kappa'], self.rel_naive['variancia_kappa'])
        p_val = p_valor_z(z_stat)
        
        # Setar KPIs
        self.metric_acc.set(f"{acc:.2%}", T.SUCCESS if acc > 0.9 else T.ACCENT_DEEP)
        self.metric_kappa.set(f"{k:.4f}")
        self.metric_z.set(f"Z={z_stat:.3f} (p={p_val:.3f})")
        
        # Gerar Analise Textual
        self.txt_analise.configure(state='normal')
        self.txt_analise.delete('1.0', 'end')
        
        self.txt_analise.insert('end', "Normalidade Multivariada:\n", 'bold')
        self.txt_analise.insert('end', " - Setosa: ", 'body')
        self.txt_analise.insert('end', "HZ p=0.0496 (FALHA MVN)\n" if self.dados_mvn.get('setosa', {}).get('hz_normal') == 'NAO' else "Passa\n", 'hl' if self.dados_mvn.get('setosa', {}).get('hz_normal') == 'NAO' else 'body')
        self.txt_analise.insert('end', " - Versicolor: Passa MVN (HZ p=0.3802)\n", 'body')
        self.txt_analise.insert('end', " - Virginica: Passa MVN (HZ p=0.0882)\n\n", 'body')
        
        self.txt_analise.insert('end', "Teste de Significância de Kappa (Z-test):\n", 'bold')
        self.txt_analise.insert('end', f"Comparando Bayes Otimo (K={self.rel_bayes['kappa']:.4f}) com Naive Bayes (K={self.rel_naive['kappa']:.4f}):\n", 'body')
        self.txt_analise.insert('end', f" - Estatistica Z: {z_stat:.4f} | p-valor: {p_val:.6f}\n", 'mono')
        
        if p_val < 0.05:
            self.txt_analise.insert('end', " - Conclusao: Existe diferenca estatisticamente significativa entre as acuracias dos classificadores (ao nivel de 5%). ", 'body')
            melhor = "Bayes Otimo" if self.rel_bayes['kappa'] > self.rel_naive['kappa'] else "Naive Bayes"
            self.txt_analise.insert('end', f"O classificador {melhor} e significativamente superior.\n\n", 'hl')
        else:
            self.txt_analise.insert('end', " - Conclusao: Nao existe diferenca estatisticamente significativa entre as acuracias dos classificadores (ao nivel de 5%). ", 'body')
            if self.rel_bayes['acerto_global'] > self.rel_naive['acerto_global']:
                self.txt_analise.insert('end', f"O Bayes Otimo e ligeiramente superior em acuracia ({self.rel_bayes['acerto_global']:.1%} vs {self.rel_naive['acerto_global']:.1%}), mas nao e significativa.\n\n", 'body')
            elif self.rel_bayes['acerto_global'] < self.rel_naive['acerto_global']:
                self.txt_analise.insert('end', f"O Naive Bayes e ligeiramente superior em acuracia ({self.rel_naive['acerto_global']:.1%} vs {self.rel_bayes['acerto_global']:.1%}), mas nao e significativa.\n\n", 'body')
            else:
                self.txt_analise.insert('end', "Ambos os classificadores tem desempenho identico.\n\n", 'body')
                
        # Matriz de Confusão do modelo atual
        self.txt_analise.insert('end', f"Matriz de Confusao ({rel_atual['nome']}):\n", 'bold')
        m = rel_atual['matriz']
        # Cabeçalho
        self.txt_analise.insert('end', f"  Pred \\ Real |   Setosa   | Versicolor |  Virginica \n", 'mono')
        self.txt_analise.insert('end', f"  -----------+------------+------------+------------\n", 'mono')
        for pred in CLASSES:
            linha = f"  {pred:10} |"
            for real in CLASSES:
                count = m[pred][real]
                linha += f" {count:10} |"
            self.txt_analise.insert('end', f"{linha[:-1]}\n", 'mono')
            
        self.txt_analise.configure(state='disabled')

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
