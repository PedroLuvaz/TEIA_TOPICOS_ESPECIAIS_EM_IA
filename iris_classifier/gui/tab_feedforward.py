"""
Aba 5 — Feedforward (MLP) — Aula PR_711 (Lab 5)
=================================================
Implementa os dois itens do laboratorio:

  (i)  Rede feedforward 2-2-2 (Python puro, backprop do zero) para o
       exemplo didatico "Galinha vs Homem", com os pesos iniciais do slide.
  (ii) Rede feedforward via scikit-learn (unico ponto do projeto com lib
       de ML, uso explicitamente permitido pelo enunciado) para classificar
       o Iris, comparada com Bayes Otimo e Naive Bayes usando todas as
       metricas de qualidade (Acerto Global, Kappa, Tau, Precisao, Recall,
       F1, F2, MCC) e teste Z de significancia entre os 3 modelos.
"""
import os
import sys
import tkinter as tk
from tkinter import ttk

PROJETO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
IRIS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (PROJETO_ROOT, IRIS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from data.data_loader import carregar_dados_iris, split_estratificado
from models.bayes_classifier import treinar_bayes, predizer_todas_classes_bayes
from models.mlp_sklearn import treinar_mlp_iris, prever_mlp_iris
from models.mlp_backprop import RedeFeedforward
from evaluation.metricas_avancadas import relatorio_completo, z_kappa, p_valor_z

from . import theme as T
from .widgets import Card, MetricBlock, separador
from .janela_calculos import JanelaMemoriaCalculoMLP

# ---------------------------------------------------------------------------
CAMINHO_DADOS = os.path.join(PROJETO_ROOT, 'data', 'Iris data.xls')
CLASSES = ['setosa', 'versicolor', 'virginica']
INDICES_TODAS = [0, 1, 2, 3]

CORES_CLASSE = {
    'setosa':     T.DATA_BLUE,
    'versicolor': T.DATA_MINT,
    'virginica':  T.DATA_CORAL,
}

MODELOS = ['Feedforward (MLP)', 'Bayes Otimo', 'Naive Bayes']

# --- Exemplo (i): Galinha vs Homem — pesos exatos do slide da Aula PR_711 ---
GH_ENTRADAS = [0.15, 0.35]
GH_ALVO = [0.0, 1.0]  # c1 = homem (0), c2 = galinha (1)
GH_TAXA_APRENDIZADO = 0.05
GH_PESOS_OCULTA = [[0.10, 0.12], [0.20, 0.17]]
GH_BIAS_OCULTA = [0.80, 0.25]
GH_PESOS_SAIDA = [[0.05, 0.33], [0.40, 0.07]]
GH_BIAS_SAIDA = [0.15, 0.70]

# --- Bonus interativo: reconhecimento de imagem 8x8 pixels ---
# Padroes de referencia desenhados a mao (inspirados no slide da Aula PR_711:
# "Man" e "Chicken" em uma grade de 8x8 pixels). Valores 0.0 (fundo claro) a
# 1.0 (pixel escuro / silhueta).
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
COR_HOMEM = '#2563EB'
COR_GALINHA = T.ACCENT


def _achatar(grade):
    return [v for linha in grade for v in linha]


# ===========================================================================
class TabFeedforward(tk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, bg=T.BG, **kw)

        self.dados = []
        self.dados_treino = []
        self.dados_teste = []
        self.resultados = {}  # nome_modelo -> relatorio_completo(...)
        self.var_matriz_sel = tk.StringVar(value=MODELOS[0])

        # Bonus interativo: rede propria (64 entradas -> 10 ocultos -> 1 saida),
        # treinada do zero nos 2 padroes de referencia (Python puro).
        self.pixels = [[0.0] * 8 for _ in range(8)]
        self.rede_imagem = RedeFeedforward(n_entradas=64, n_ocultos=10, n_saidas=1, semente=42)
        self.rede_imagem.treinar(
            [_achatar(HOMEM_PIXELS), _achatar(GALINHA_PIXELS)], [[0.0], [1.0]],
            taxa_aprendizado=0.5, epocas=4000)

        self._construir_layout()
        self._carregar_dados()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    def _construir_layout(self):
        self.columnconfigure(0, minsize=300)
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)
        self._coluna_controles()
        self._coluna_resultados()

    def _coluna_controles(self):
        wrap = tk.Frame(self, bg=T.BG)
        wrap.grid(row=0, column=0, sticky='nsew', padx=(T.PAD_PAGE, T.GAP), pady=T.PAD_PAGE)
        wrap.columnconfigure(0, weight=1)

        card_info = Card(wrap, titulo='sobre esta aba')
        card_info.grid(row=0, column=0, sticky='ew')
        tk.Label(card_info,
                 text='Item (i): rede 2-2-2 em Python puro (sem libs de ML), '
                      'reproduzindo o exemplo "galinha vs homem" do slide.\n\n'
                      'Item (ii): rede feedforward via scikit-learn (unico '
                      'experimento do projeto com lib de ML — uso permitido '
                      'pelo enunciado) classificando o Iris, comparada com '
                      'Bayes Otimo e Naive Bayes.',
                 bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_BODY,
                 justify='left', anchor='w', wraplength=260
                ).pack(fill='x', padx=T.CARD_PADX, pady=(0, 10))

        card_gh = Card(wrap, titulo='item (i) — galinha vs homem')
        card_gh.grid(row=1, column=0, sticky='ew', pady=(T.GAP_SM, 0))
        tk.Label(card_gh,
                 text=f'Entradas: a1={GH_ENTRADAS[0]}  a2={GH_ENTRADAS[1]}\n'
                      f'Alvo: homem(c1)={GH_ALVO[0]:.0f}  galinha(c2)={GH_ALVO[1]:.0f}\n'
                      f'Taxa de aprendizagem: eta={GH_TAXA_APRENDIZADO}',
                 bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_MONO_SM,
                 justify='left', anchor='w'
                ).pack(fill='x', padx=T.CARD_PADX, pady=(0, 6))
        ttk.Button(card_gh, text='Abrir memoria de calculo  >',
                   style='Primary.TButton',
                   command=self._abrir_memoria_galinha_homem
                  ).pack(fill='x', padx=T.CARD_PADX, pady=(0, 10))

        card_ii = Card(wrap, titulo='item (ii) — iris comparativo')
        card_ii.grid(row=2, column=0, sticky='ew', pady=(T.GAP_SM, 0))
        ttk.Button(card_ii, text='Treinar e Comparar Modelos  >',
                   style='Primary.TButton',
                   command=self._treinar_tudo
                  ).pack(fill='x', padx=T.CARD_PADX, pady=(4, 6))
        self.lbl_status = tk.Label(card_ii, text='Aguardando treinamento.',
                                   bg=T.BG_CARD, fg=T.FG_MUTED,
                                   font=T.FONT_MONO_SM, anchor='w',
                                   wraplength=260, justify='left')
        self.lbl_status.pack(fill='x', padx=T.CARD_PADX, pady=(0, 10))

        wrap.rowconfigure(3, weight=1)

    def _coluna_resultados(self):
        wrap = tk.Frame(self, bg=T.BG)
        wrap.grid(row=0, column=1, sticky='nsew', padx=(T.GAP, T.PAD_PAGE), pady=T.PAD_PAGE)
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=0)
        wrap.rowconfigure(1, weight=1)

        faixa = tk.Frame(wrap, bg=T.BG)
        faixa.grid(row=0, column=0, sticky='ew')
        faixa.columnconfigure(list(range(3)), weight=1)
        self.mb_acc = MetricBlock(faixa, 'Acerto Global (MLP)', '—')
        self.mb_kappa = MetricBlock(faixa, 'Kappa (MLP)', '—')
        self.mb_z = MetricBlock(faixa, 'Z (MLP x Bayes)', '—')
        for i, mb in enumerate([self.mb_acc, self.mb_kappa, self.mb_z]):
            mb.grid(row=0, column=i, sticky='ew', padx=(0 if i == 0 else T.GAP_SM, 0))

        nb_wrap = tk.Frame(wrap, bg=T.BG)
        nb_wrap.grid(row=1, column=0, sticky='nsew', pady=(10, 0))
        nb_wrap.columnconfigure(0, weight=1)
        nb_wrap.rowconfigure(0, weight=1)

        self.nb = ttk.Notebook(nb_wrap)
        self.nb.grid(row=0, column=0, sticky='nsew')

        self._aba_imagem = tk.Frame(self.nb, bg=T.BG_CARD)
        self._aba_comp = tk.Frame(self.nb, bg=T.BG_CARD)
        self._aba_matriz = tk.Frame(self.nb, bg=T.BG_CARD)

        self.nb.add(self._aba_imagem, text='  Reconhecimento de Imagem (i)  ')
        self.nb.add(self._aba_comp, text='  Comparativo (ii)  ')
        self.nb.add(self._aba_matriz, text='  Matriz de Confusao  ')

        for aba in (self._aba_comp, self._aba_matriz):
            tk.Label(aba, text='Clique em "Treinar e Comparar Modelos" para iniciar.',
                     bg=T.BG_CARD, fg=T.FG_DIM, font=T.FONT_BODY
                    ).place(relx=0.5, rely=0.5, anchor='center')

        self._construir_aba_imagem()

    # ------------------------------------------------------------------
    # Dados
    # ------------------------------------------------------------------
    def _carregar_dados(self):
        if not os.path.exists(CAMINHO_DADOS):
            self.lbl_status.configure(
                text=f'Dados nao encontrados:\n{CAMINHO_DADOS}', fg=T.DANGER)
            return
        self.dados = carregar_dados_iris(CAMINHO_DADOS)
        self.dados_treino, self.dados_teste = split_estratificado(
            self.dados, proporcao_treino=0.7, semente=42)

    # ------------------------------------------------------------------
    # Item (ii) — treinar e comparar
    # ------------------------------------------------------------------
    def _treinar_tudo(self):
        if not self.dados_treino:
            self._carregar_dados()
        if not self.dados_treino:
            return

        self.lbl_status.configure(text='Treinando...', fg=T.ACCENT)
        self.update()

        gab = [d['classe'] for d in self.dados_teste]

        modelo_mlp = treinar_mlp_iris(self.dados_treino, INDICES_TODAS, semente=42)
        model_bayes = treinar_bayes(self.dados_treino, INDICES_TODAS, naive=False)
        model_naive = treinar_bayes(self.dados_treino, INDICES_TODAS, naive=True)

        preds_mlp = prever_mlp_iris(modelo_mlp, self.dados_teste, INDICES_TODAS)
        preds_bayes = [predizer_todas_classes_bayes(d['atributos'], model_bayes, INDICES_TODAS)[1]
                       for d in self.dados_teste]
        preds_naive = [predizer_todas_classes_bayes(d['atributos'], model_naive, INDICES_TODAS)[1]
                       for d in self.dados_teste]

        self.resultados = {
            'Feedforward (MLP)': relatorio_completo(preds_mlp, gab, CLASSES, 'Feedforward (MLP)'),
            'Bayes Otimo': relatorio_completo(preds_bayes, gab, CLASSES, 'Bayes Otimo'),
            'Naive Bayes': relatorio_completo(preds_naive, gab, CLASSES, 'Naive Bayes'),
        }

        rel_mlp = self.resultados['Feedforward (MLP)']
        rel_bayes = self.resultados['Bayes Otimo']
        ag, k = rel_mlp['acerto_global'], rel_mlp['kappa']
        z = z_kappa(rel_mlp['kappa'], rel_mlp['variancia_kappa'],
                    rel_bayes['kappa'], rel_bayes['variancia_kappa'])
        p = p_valor_z(z)

        self.mb_acc.set(f'{ag:.2%}', T.SUCCESS if ag >= 0.9 else T.ACCENT)
        self.mb_kappa.set(f'{k:.4f}', T.SUCCESS if k > 0.80 else T.ACCENT if k > 0.40 else T.DANGER)
        self.mb_z.set(f'Z={z:.3f}  p={p:.3f}', T.DANGER if p < 0.05 else T.SUCCESS)

        self._preencher_comparativo()
        self._preencher_matriz()

        self.lbl_status.configure(
            text=f'Concluido. {len(self.dados_teste)} amostras de teste.', fg=T.SUCCESS)

    # ------------------------------------------------------------------
    # Aba Comparativo
    # ------------------------------------------------------------------
    def _preencher_comparativo(self):
        for w in self._aba_comp.winfo_children():
            w.destroy()

        outer = tk.Frame(self._aba_comp, bg=T.BG_CARD)
        outer.pack(fill='both', expand=True, padx=8, pady=6)

        tk.Label(outer,
                 text='Feedforward (MLP) x Bayes Otimo x Naive Bayes — todas as metricas de qualidade',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(0, 6))

        frame = tk.Frame(outer, bg=T.BG_CARD)
        frame.pack(fill='both', expand=True)
        canvas = tk.Canvas(frame, bg=T.BG_CARD, highlightthickness=0)
        sb_y = ttk.Scrollbar(frame, orient='vertical', command=canvas.yview)
        inner = tk.Frame(canvas, bg=T.BG_CARD)
        inner.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.create_window((0, 0), window=inner, anchor='nw')
        canvas.configure(yscrollcommand=sb_y.set)
        canvas.pack(side='left', fill='both', expand=True)
        sb_y.pack(side='right', fill='y')

        def cel(row, col, texto, cor=T.FG, bg_c=T.BG_PANEL, larg=20, bold=False):
            f = T.FONT_CELL_BOLD if bold else T.FONT_MONO_SM
            tk.Label(inner, text=texto, bg=bg_c, fg=cor, font=f,
                     width=larg, anchor='center',
                     highlightthickness=1, highlightbackground=T.BORDER
                    ).grid(row=row, column=col, sticky='nsew', padx=1, pady=1)

        def melhor_col(vals):
            """Retorna o indice do maior valor numerico entre uma lista de strings tipo '12.34%'."""
            try:
                nums = [float(v.replace('%', '')) for v in vals]
                return nums.index(max(nums))
            except ValueError:
                return -1

        for j, h in enumerate(['Metrica / Classe'] + MODELOS):
            cel(0, j, h, cor=T.ACCENT_DEEP, bold=True)

        row = 1
        for label, chave, fmt in [
            ('Acerto Global (Ag)', 'acerto_global', lambda v: f'{v:.2%}'),
            ('Kappa', 'kappa', lambda v: f'{v:.6f}'),
            ('Tau', 'tau', lambda v: f'{v:.6f}'),
            ('Var(Kappa)', 'variancia_kappa', lambda v: f'{v:.8f}'),
        ]:
            vals = [fmt(self.resultados[m][chave]) for m in MODELOS]
            idx_melhor = melhor_col(vals) if chave != 'variancia_kappa' else -1
            bg_r = T.BG_CARD if row % 2 == 0 else T.BG_PANEL
            cel(row, 0, label, cor=T.FG_MUTED, bg_c=bg_r)
            for j, v in enumerate(vals):
                cor = T.SUCCESS if j == idx_melhor else T.FG
                cel(row, j + 1, v, cor=cor, bg_c=bg_r)
            row += 1

        for c in CLASSES:
            cor_c = CORES_CLASSE[c]
            for label, chave, fmt in [
                (f'{c.capitalize()}  Precisao', 'precisao', lambda v: f'{v:.2%}'),
                (f'{c.capitalize()}  Recall', 'sensibilidade', lambda v: f'{v:.2%}'),
                (f'{c.capitalize()}  F1 (b=1)', 'f1', lambda v: f'{v:.4f}'),
                (f'{c.capitalize()}  F2 (b=2)', 'f2', lambda v: f'{v:.4f}'),
                (f'{c.capitalize()}  MCC', 'mcc', lambda v: f'{v:.4f}'),
            ]:
                vals = [fmt(self.resultados[m]['por_classe'][c][chave]) for m in MODELOS]
                idx_melhor = melhor_col(vals)
                bg_r = T.BG_CARD if row % 2 == 0 else T.BG_PANEL
                cel(row, 0, label, cor=cor_c, bg_c=bg_r)
                for j, v in enumerate(vals):
                    cor = T.SUCCESS if j == idx_melhor else T.FG
                    cel(row, j + 1, v, cor=cor, bg_c=bg_r)
                row += 1

        separador(outer)
        tk.Label(outer,
                 text='Testes Z de significancia de Kappa (todos os pares):',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(4, 2))
        pares = [
            ('Feedforward (MLP)', 'Bayes Otimo'),
            ('Feedforward (MLP)', 'Naive Bayes'),
            ('Bayes Otimo', 'Naive Bayes'),
        ]
        for nome_a, nome_b in pares:
            ra, rb = self.resultados[nome_a], self.resultados[nome_b]
            z = z_kappa(ra['kappa'], ra['variancia_kappa'], rb['kappa'], rb['variancia_kappa'])
            p = p_valor_z(z)
            veredito = 'diferenca significativa' if p < 0.05 else 'sem diferenca significativa'
            cor = T.DANGER if p < 0.05 else T.SUCCESS
            tk.Label(outer, text=f'  {nome_a} x {nome_b}: Z={z:.4f}  p={p:.6f}  ({veredito} a 5%)',
                     bg=T.BG_CARD, fg=cor, font=T.FONT_MONO_SM, anchor='w'
                    ).pack(fill='x')

    # ------------------------------------------------------------------
    # Aba Matriz de Confusao (com seletor de modelo)
    # ------------------------------------------------------------------
    def _preencher_matriz(self):
        for w in self._aba_matriz.winfo_children():
            w.destroy()

        outer = tk.Frame(self._aba_matriz, bg=T.BG_CARD)
        outer.pack(fill='both', expand=True, padx=12, pady=8)

        sel = tk.Frame(outer, bg=T.BG_CARD)
        sel.pack(fill='x', pady=(0, 8))
        tk.Label(sel, text='Modelo:', bg=T.BG_CARD, fg=T.FG_MUTED,
                 font=T.FONT_LABEL).pack(side='left', padx=(0, 8))
        for m in MODELOS:
            tk.Radiobutton(sel, text=m, value=m, variable=self.var_matriz_sel,
                           bg=T.BG_CARD, fg=T.FG, selectcolor=T.BG_HOVER,
                           activebackground=T.BG_CARD, activeforeground=T.ACCENT_DEEP,
                           font=T.FONT_BODY, borderwidth=0, highlightthickness=0,
                           command=self._redesenhar_matriz
                          ).pack(side='left', padx=(0, 10))

        self._matriz_grid_holder = tk.Frame(outer, bg=T.BG_CARD)
        self._matriz_grid_holder.pack(anchor='w')
        self._redesenhar_matriz()

    def _redesenhar_matriz(self):
        for w in self._matriz_grid_holder.winfo_children():
            w.destroy()
        nome = self.var_matriz_sel.get()
        if nome not in self.resultados:
            return
        rel = self.resultados[nome]
        matriz = rel['matriz']
        classes = CLASSES
        n = len(classes)
        grid = self._matriz_grid_holder
        vals = [[matriz[pred][real] for real in classes] for pred in classes]
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
        total_geral = 0
        for i, pred in enumerate(classes):
            tk.Label(grid, text=pred.capitalize(), bg=CORES_CLASSE[pred],
                     fg='white', font=T.FONT_KICKER, width=10, anchor='center',
                     highlightthickness=1, highlightbackground=T.BORDER
                    ).grid(row=i + 1, column=0, padx=2, pady=2)
            total_linha = sum(matriz[pred][real] for real in classes)
            for j, real in enumerate(classes):
                v = matriz[pred][real]
                bg = bg_cel(v, i == j)
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
            tk.Label(grid, text=str(totais_colunas[real]), bg=T.BG_PANEL, fg=T.FG,
                     font=T.FONT_CELL_LG, width=10, anchor='center',
                     highlightthickness=1, highlightbackground=T.BORDER
                    ).grid(row=n + 1, column=j + 1, padx=2, pady=2)
        tk.Label(grid, text=str(total_geral), bg=T.BG_PANEL, fg=T.ACCENT_DEEP,
                 font=T.FONT_CELL_LG, width=10, anchor='center',
                 highlightthickness=1, highlightbackground=T.BORDER
                ).grid(row=n + 1, column=n + 1, padx=2, pady=2)

    # ------------------------------------------------------------------
    # Aba Reconhecimento de Imagem — bonus interativo do item (i)
    # ------------------------------------------------------------------
    def _construir_aba_imagem(self):
        outer = tk.Frame(self._aba_imagem, bg=T.BG_CARD)
        outer.pack(fill='both', expand=True, padx=14, pady=10)

        tk.Label(outer,
                 text='DESENHE UM PADRAO DE 8x8 PIXELS E VEJA A REDE CLASSIFICAR EM TEMPO REAL',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(0, 2))
        tk.Label(outer,
                 text='Rede propria (64 entradas -> 10 neuronios ocultos -> 1 saida), treinada do '
                      'zero em Python puro apenas com os 2 padroes de referencia abaixo — o mesmo '
                      'principio do exemplo "Galinha vs Homem" do slide, agora com uma imagem inteira.',
                 bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_BODY_SM,
                 wraplength=760, justify='left', anchor='w'
                ).pack(fill='x', pady=(0, 12))

        corpo = tk.Frame(outer, bg=T.BG_CARD)
        corpo.pack(fill='both', expand=True)

        # ---- Coluna esquerda: canvas de pintura ----
        esq = tk.Frame(corpo, bg=T.BG_CARD)
        esq.pack(side='left', padx=(0, 28))

        self._img_cell = 26
        cell = self._img_cell
        self._img_canvas = tk.Canvas(esq, width=cell * 8, height=cell * 8,
                                     bg='white', highlightthickness=1,
                                     highlightbackground=T.BORDER_HARD, cursor='pencil')
        self._img_canvas.pack()
        self._img_rects = [[None] * 8 for _ in range(8)]
        for r in range(8):
            for c in range(8):
                rect = self._img_canvas.create_rectangle(
                    c * cell, r * cell, (c + 1) * cell, (r + 1) * cell,
                    fill='#ffffff', outline='#e2e8f0')
                self._img_rects[r][c] = rect

        self.var_intensidade = tk.DoubleVar(value=0.85)

        self._img_canvas.bind('<Button-1>', lambda e: self._pintar_pixel(e, self.var_intensidade.get()))
        self._img_canvas.bind('<B1-Motion>', lambda e: self._pintar_pixel(e, self.var_intensidade.get()))
        self._img_canvas.bind('<Button-3>', lambda e: self._pintar_pixel(e, 0.0))
        self._img_canvas.bind('<B3-Motion>', lambda e: self._pintar_pixel(e, 0.0))

        tk.Label(esq, text='botao esquerdo pinta  ·  botao direito apaga',
                 bg=T.BG_CARD, fg=T.FG_DIM, font=T.FONT_LABEL
                ).pack(pady=(6, 6))

        # ---- Intensidade (densidade) do pixel a ser pintado ----
        intens = tk.Frame(esq, bg=T.BG_CARD)
        intens.pack(fill='x', pady=(0, 10))
        tk.Label(intens, text='INTENSIDADE', bg=T.BG_CARD, fg=T.FG_MUTED,
                 font=T.FONT_KICKER, anchor='w').pack(side='left')
        self._img_intens_swatch = tk.Frame(intens, bg=self._cor_cinza(0.85), width=18, height=18,
                                           highlightthickness=1, highlightbackground=T.BORDER)
        self._img_intens_swatch.pack(side='right')
        self._img_intens_swatch.pack_propagate(False)
        self.lbl_intensidade = tk.Label(intens, text='85%', bg=T.BG_CARD, fg=T.FG,
                                        font=T.FONT_MONO_SM, width=5, anchor='e')
        self.lbl_intensidade.pack(side='right', padx=(0, 8))

        ttk.Scale(esq, from_=0.05, to=1.0, orient='horizontal',
                  variable=self.var_intensidade, command=self._ao_mudar_intensidade
                 ).pack(fill='x', pady=(0, 10))

        botoes = tk.Frame(esq, bg=T.BG_CARD)
        botoes.pack(fill='x')
        ttk.Button(botoes, text='Homem',
                   command=lambda: self._carregar_padrao_imagem(HOMEM_PIXELS)
                  ).pack(side='left', expand=True, fill='x', padx=(0, 4))
        ttk.Button(botoes, text='Galinha',
                   command=lambda: self._carregar_padrao_imagem(GALINHA_PIXELS)
                  ).pack(side='left', expand=True, fill='x', padx=4)
        ttk.Button(botoes, text='Limpar',
                   command=lambda: self._carregar_padrao_imagem(None)
                  ).pack(side='left', expand=True, fill='x', padx=(4, 0))

        # ---- Coluna direita: predicao ao vivo ----
        dir_ = tk.Frame(corpo, bg=T.BG_CARD)
        dir_.pack(side='left', fill='both', expand=True)

        tk.Label(dir_, text='SAIDA DA REDE   (0 = homem  ·  1 = galinha)',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(0, 4))

        self._img_barra_bg = tk.Frame(dir_, bg=T.BG_PANEL, height=26,
                                      highlightthickness=1, highlightbackground=T.BORDER)
        self._img_barra_bg.pack(fill='x', pady=(0, 6))
        self._img_barra_bg.pack_propagate(False)
        self._img_barra_fill = tk.Frame(self._img_barra_bg, bg=T.BORDER)
        self._img_barra_fill.place(relx=0, rely=0, relheight=1, relwidth=0.0)

        self.lbl_img_pred = tk.Label(dir_, text='out = —',
                                     bg=T.BG_CARD, fg=T.FG, font=T.FONT_VALUE_BIG, anchor='w')
        self.lbl_img_pred.pack(fill='x')
        self.lbl_img_classe = tk.Label(dir_, text='Tela em branco — desenhe algo',
                                       bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_TITLE, anchor='w')
        self.lbl_img_classe.pack(fill='x', pady=(0, 16))

        tk.Label(dir_, text='CAMADA OCULTA   (10 neuronios, ativacao sigmoide)',
                 bg=T.BG_CARD, fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', pady=(0, 4))
        hidden_wrap = tk.Frame(dir_, bg=T.BG_CARD)
        hidden_wrap.pack(fill='x')
        self._img_hidden_rects = []
        for i in range(10):
            f = tk.Frame(hidden_wrap, bg='#ffffff', width=30, height=30,
                         highlightthickness=1, highlightbackground=T.BORDER)
            f.grid(row=0, column=i, padx=2)
            f.grid_propagate(False)
            self._img_hidden_rects.append(f)

        separador(dir_, pady=16)
        tk.Label(dir_,
                 text='Quanto mais parecido o desenho for de um dos 2 padroes de referencia '
                      '(em pixels, nao em forma), mais a saida se aproxima de 0 ou de 1. A '
                      'camada oculta acima mostra o quanto cada um dos 10 neuronios "acende" '
                      'para o desenho atual.',
                 bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_LABEL, wraplength=360, justify='left'
                ).pack(fill='x')

        self._atualizar_predicao_imagem()

    @staticmethod
    def _cor_cinza(v):
        """Converte uma intensidade 0.0-1.0 em um tom de cinza hexadecimal (0=branco, 1=escuro)."""
        tom = int(255 - max(0.0, min(1.0, v)) * 220)
        return f'#{tom:02x}{tom:02x}{tom:02x}'

    def _ao_mudar_intensidade(self, _valor_str=None):
        v = self.var_intensidade.get()
        self.lbl_intensidade.configure(text=f'{v * 100:.0f}%')
        self._img_intens_swatch.configure(bg=self._cor_cinza(v))

    def _pintar_pixel(self, event, valor):
        cell = self._img_cell
        col = event.x // cell
        row = event.y // cell
        if 0 <= row < 8 and 0 <= col < 8 and self.pixels[row][col] != valor:
            self.pixels[row][col] = valor
            self._redesenhar_pixel(row, col)
            self._atualizar_predicao_imagem()

    def _redesenhar_pixel(self, row, col):
        cor = self._cor_cinza(self.pixels[row][col])
        self._img_canvas.itemconfig(self._img_rects[row][col], fill=cor)

    def _carregar_padrao_imagem(self, padrao):
        self.pixels = [row[:] for row in padrao] if padrao else [[0.0] * 8 for _ in range(8)]
        for r in range(8):
            for c in range(8):
                self._redesenhar_pixel(r, c)
        self._atualizar_predicao_imagem()

    def _atualizar_predicao_imagem(self):
        entrada = _achatar(self.pixels)
        saida_oculta, saida_rede = self.rede_imagem.forward(entrada)
        out = saida_rede[0]

        if all(v == 0.0 for v in entrada):
            self._img_barra_fill.place(relx=0, rely=0, relheight=1, relwidth=0.0)
            self.lbl_img_pred.configure(text='out = —')
            self.lbl_img_classe.configure(text='Tela em branco — desenhe algo', fg=T.FG_MUTED)
        else:
            self._img_barra_fill.place(relx=0, rely=0, relheight=1, relwidth=max(0.01, out))
            self.lbl_img_pred.configure(text=f'out = {out:.3f}')
            if out < 0.35:
                texto, cor = 'Mais parecido com HOMEM', COR_HOMEM
            elif out > 0.65:
                texto, cor = 'Mais parecido com GALINHA', COR_GALINHA
            else:
                texto, cor = 'Ambiguo entre os dois padroes', T.FG_MUTED
            self.lbl_img_classe.configure(text=texto, fg=cor)
            self._img_barra_fill.configure(bg=cor)

        for i, frame in enumerate(self._img_hidden_rects):
            frame.configure(bg=self._cor_cinza(saida_oculta[i]))

    # ------------------------------------------------------------------
    # Memoria de Calculo (janela LaTeX) — Item (i) Galinha vs Homem
    # ------------------------------------------------------------------
    def _abrir_memoria_galinha_homem(self):
        JanelaMemoriaCalculoMLP(
            self,
            entradas=GH_ENTRADAS, alvo=GH_ALVO, taxa_aprendizado=GH_TAXA_APRENDIZADO,
            pesos_oculta=GH_PESOS_OCULTA, bias_oculta=GH_BIAS_OCULTA,
            pesos_saida=GH_PESOS_SAIDA, bias_saida=GH_BIAS_SAIDA,
            rotulos_ocultos=['b1', 'b2'],
            rotulos_saida=['c1 (homem)', 'c2 (galinha)'],
        )
