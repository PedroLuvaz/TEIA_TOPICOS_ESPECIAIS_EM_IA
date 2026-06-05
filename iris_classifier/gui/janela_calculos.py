"""
Janelas de Memoria de Calculo.

JanelaMemoriaCalculo           — Classificador de Distancia Minima (Aba 1)
JanelaMemoriaCalculoPD         — Perceptron / Regra Delta            (Aba 2)
JanelaMemoriaCalculoMetricas   — Metricas Avancadas (Ag, K, tau, ...)(Aba 3)

Cada janela exibe formulas LaTeX (matplotlib mathtext + PIL),
substituicao numerica passo a passo e referencia arquivo:linha
de cada funcao matematica usada.
"""
import inspect
import io
import os
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from math_utils import (produto_escalar, distancia_euclidiana,
                        coeficientes_superficie_decisao,
                        discriminante, calcular_media)
from classifier import treinar, predizer_todas_classes
from perceptron import treinar_perceptron, predizer_perceptron, _sgn
from delta_rule import treinar_delta_iris, predizer_delta, _treinar_delta
from metricas_avancadas import (
    acerto_global, acuracia_produtor, acuracia_usuario,
    _acerto_casual, kappa, variancia_kappa, tau, variancia_tau,
    z_kappa, z_tau, p_valor_z,
    sensibilidade, especificidade, precisao_binaria,
    mcc, fb_score, _extrair_binario, relatorio_completo,
)

from . import theme as T
from .widgets import Card


CORES_CLASSE = {
    'setosa':     T.DATA_BLUE,
    'versicolor': T.DATA_MINT,
    'virginica':  T.DATA_CORAL,
}

PARES = [('setosa', 'versicolor'),
         ('setosa', 'virginica'),
         ('versicolor', 'virginica')]


def _ref_funcao(func):
    """Retorna (nome_arquivo, linha_inicial) de uma funcao via inspect."""
    try:
        _, linha = inspect.getsourcelines(func)
        nome = os.path.basename(inspect.getfile(func))
        return nome, linha
    except Exception:
        return '?', 0


# ===========================================================================
# Aba 1 — Classificador de Distancia Minima
# ===========================================================================
class JanelaMemoriaCalculo(tk.Toplevel):
    def __init__(self, parent, prototipos, eixos, n_treino_por_classe=35,
                 amostra=None):
        super().__init__(parent)
        self.title('Memoria de Calculo  ·  Distancia Minima')
        self.geometry('980x800')
        self.minsize(820, 600)
        self.configure(bg=T.BG)
        self.transient(parent)

        self.prototipos = prototipos
        self.eixos = eixos
        self.n_treino = n_treino_por_classe
        self.amostra = list(amostra) if amostra else [4.5, 1.5]
        self._imagens_ref = []

        self._construir()

    # ------------------------------------------------------------------
    def _construir(self):
        tk.Frame(self, bg=T.ACCENT, height=2).pack(fill='x', side='top')

        head = tk.Frame(self, bg=T.BG, height=78)
        head.pack(fill='x', side='top')
        head.pack_propagate(False)
        tk.Label(head, text='MEMORIA DE CALCULO',
                 bg=T.BG, fg=T.ACCENT, font=T.FONT_KICKER
                ).pack(anchor='w', padx=24, pady=(16, 0))
        tk.Label(head, text='Substituicao numerica das formulas com os '
                            'valores atuais do modelo treinado',
                 bg=T.BG, fg=T.FG, font=T.FONT_TITLE
                ).pack(anchor='w', padx=24)
        tk.Label(head,
                 text=f'Atributos: {self.eixos[0]}  ·  {self.eixos[1]}',
                 bg=T.BG, fg=T.FG_MUTED, font=T.FONT_SUBTITLE
                ).pack(anchor='w', padx=24, pady=(2, 0))

        tk.Frame(self, bg=T.BORDER, height=1).pack(fill='x', side='top')

        rod = tk.Frame(self, bg=T.BG_PANEL, height=44)
        rod.pack(fill='x', side='bottom')
        rod.pack_propagate(False)
        tk.Frame(self, bg=T.BORDER, height=1).pack(fill='x', side='bottom')
        ttk.Button(rod, text='Fechar', command=self.destroy
                  ).pack(side='right', padx=20, pady=8)
        tk.Label(rod, text='Formulas via matplotlib mathtext  ·  calculos em Python puro',
                 bg=T.BG_PANEL, fg=T.FG_DIM, font=T.FONT_SUBTITLE
                ).pack(side='left', padx=20, pady=11)

        canvas = tk.Canvas(self, bg=T.BG, highlightthickness=0, borderwidth=0)
        scroll = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        wrap = tk.Frame(canvas, bg=T.BG)
        win_id = canvas.create_window((0, 0), window=wrap, anchor='nw')
        wrap.bind('<Configure>',
                  lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',
                    lambda e: canvas.itemconfigure(win_id, width=e.width))

        def _wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        canvas.bind_all('<MouseWheel>', _wheel)
        self.protocol('WM_DELETE_WINDOW',
                      lambda: (canvas.unbind_all('<MouseWheel>'), self.destroy()))

        self._secao_prototipos(wrap)
        self._secao_discriminante(wrap)
        self._secao_distancia(wrap)
        self._secao_fronteiras(wrap)
        tk.Frame(wrap, bg=T.BG, height=24).pack()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _formula(self, latex, fontsize=16, bg=T.BG_CARD):
        fig = Figure(figsize=(0.1, 0.1), dpi=120, facecolor=bg)
        fig.text(0, 0, latex, fontsize=fontsize, color=T.FG,
                 va='bottom', ha='left')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight',
                    pad_inches=0.18, facecolor=bg)
        buf.seek(0)
        photo = ImageTk.PhotoImage(Image.open(buf))
        self._imagens_ref.append(photo)
        return photo

    def _add_formula(self, parent, latex, fontsize=16, bg=T.BG_CARD, pady=(6, 6)):
        photo = self._formula(latex, fontsize=fontsize, bg=bg)
        tk.Label(parent, image=photo, bg=bg).pack(anchor='w', padx=18, pady=pady)

    def _add_step(self, parent, texto):
        tk.Label(parent, text=texto, bg=T.BG_CARD, fg=T.FG,
                 font=T.FONT_MONO_SM, anchor='w', justify='left'
                ).pack(anchor='w', padx=18, pady=1)

    def _add_subkicker(self, parent, texto):
        tk.Label(parent, text=texto.upper(), bg=T.BG_CARD, fg=T.ACCENT,
                 font=T.FONT_KICKER, anchor='w'
                ).pack(anchor='w', padx=18, pady=(10, 4))

    def _add_explain(self, parent, texto, pady=(2, 4)):
        tk.Label(parent, text=texto, bg=T.BG_CARD, fg=T.FG_MUTED,
                 font=T.FONT_LABEL, wraplength=900, justify='left'
                ).pack(anchor='w', padx=18, pady=pady)

    def _add_resultado(self, parent, texto, cor):
        tk.Label(parent, text=texto, bg=T.BG_CARD, fg=cor,
                 font=T.FONT_TITLE, anchor='w'
                ).pack(anchor='w', padx=18, pady=(10, 14))

    def _add_ref(self, parent, func):
        """Exibe referencia arquivo:linha de uma funcao."""
        nome, linha = _ref_funcao(func)
        tk.Label(parent,
                 text=f'  →  {nome}  :  linha {linha}',
                 bg=T.BG_CARD, fg=T.FG_DIM,
                 font=('Consolas', 8, 'normal'), anchor='w'
                ).pack(anchor='w', padx=18, pady=(0, 4))

    def _respiro(self, parent):
        tk.Frame(parent, bg=T.BG_CARD, height=10).pack()

    # ==================================================================
    # SECAO 1 — Prototipos
    # ==================================================================
    def _secao_prototipos(self, parent):
        card = Card(parent, titulo='1. Prototipos  ·  Vetores Medios')
        card.pack(fill='x', padx=22, pady=(20, 0))

        self._add_explain(card,
            'Cada classe e representada pelo seu centroide — a media de todos '
            'os vetores de treino daquela classe.')
        self._add_formula(card,
            r'$m_j \;=\; \dfrac{1}{N_j}\, \sum_{x\, \in\, \omega_j}\, x$',
            fontsize=20)
        self._add_ref(card, treinar)
        self._add_ref(card, calcular_media)

        for classe in ['setosa', 'versicolor', 'virginica']:
            mj = self.prototipos[classe]
            self._add_subkicker(card, f'classe   {classe}')
            self._add_step(card,
                f'  N_{classe[:3]:<3} = {self.n_treino}   (amostras de treino)')
            self._add_step(card,
                f'  m_{classe[:3]:<3} = (1/{self.n_treino})  ·  '
                f'[ Σ {self.eixos[0]} ,  Σ {self.eixos[1]} ]')
            self._add_step(card,
                f'         = [ {mj[0]:.4f} ,  {mj[1]:.4f} ]')
        self._respiro(card)

    # ==================================================================
    # SECAO 2 — Funcao Discriminante
    # ==================================================================
    def _secao_discriminante(self, parent):
        card = Card(parent, titulo='2. Funcao Discriminante  ·  Regra de Decisao')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'A funcao discriminante calcula um "score" para cada classe. '
            'A classe vencedora e a de maior score.')
        self._add_formula(card,
            r'$d_j(x) \;=\; x^{T}\, m_j \;-\; \dfrac{1}{2}\,'
            r'm_j^{T}\, m_j$', fontsize=20)
        self._add_formula(card,
            r'$j^{*} \;=\; \arg\max_{j}\; d_j(x)$', fontsize=18)
        self._add_ref(card, predizer_todas_classes)
        self._add_ref(card, discriminante)
        self._add_ref(card, produto_escalar)

        x = self.amostra
        self._add_explain(card,
            f'Substituicao com x = [{x[0]:.2f},  {x[1]:.2f}]:',
            pady=(8, 4))

        scores = {}
        for classe in ['setosa', 'versicolor', 'virginica']:
            mj = self.prototipos[classe]
            xtmj = produto_escalar(x, mj)
            mjmj = produto_escalar(mj, mj)
            d = xtmj - 0.5 * mjmj
            scores[classe] = d

            self._add_subkicker(card, f'classe   {classe}')
            self._add_step(card,
                f'  m_{classe[:3]} = [{mj[0]:.4f}, {mj[1]:.4f}]')
            self._add_step(card,
                f'  x^T · m_{classe[:3]:<3}  =  {x[0]:.2f}·{mj[0]:.4f}  +  '
                f'{x[1]:.2f}·{mj[1]:.4f}  =  {xtmj:.4f}')
            self._add_step(card,
                f'  m_{classe[:3]}^T · m_{classe[:3]}  =  {mj[0]:.4f}²  +  '
                f'{mj[1]:.4f}²  =  {mjmj:.4f}')
            self._add_step(card,
                f'  d_{classe[:3]:<3}(x)  =  {xtmj:.4f}  -  '
                f'½·{mjmj:.4f}  =  {d:+.4f}')

        vencedor = max(scores, key=scores.get)
        self._add_resultado(card,
            f'  argmax  →   {vencedor.upper()}    (d = {scores[vencedor]:+.4f})',
            cor=CORES_CLASSE[vencedor])

    # ==================================================================
    # SECAO 3 — Distancia Euclidiana e Equivalencia
    # ==================================================================
    def _secao_distancia(self, parent):
        card = Card(parent, titulo='3. Distancia Euclidiana  ·  Equivalencia')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'A distancia euclidiana e a metrica intuitiva: a classe cujo '
            'prototipo esta mais proximo da amostra ganha.')
        self._add_formula(card,
            r'$\|x - m_j\| \;=\; \sqrt{\,\sum_{k}\,(x_k - m_{jk})^{2}\,}$',
            fontsize=20)
        self._add_formula(card,
            r'$\arg\max_{j}\; d_j(x) \;\;\equiv\;\; \arg\min_{j}\; \|x - m_j\|$',
            fontsize=18)
        self._add_ref(card, distancia_euclidiana)

        self._add_explain(card,
            'Por que sao equivalentes? Expandindo o quadrado da distancia:')
        self._add_formula(card,
            r'$\|x - m_j\|^{2} \;=\; x^{T} x \;-\; 2\, x^{T} m_j '
            r'\;+\; m_j^{T} m_j$', fontsize=17)
        self._add_explain(card,
            'O termo  x^T·x  e constante em j. Logo, minimizar ||x - m_j||² '
            'equivale a maximizar  x^T·m_j - ½·m_j^T·m_j  =  d_j(x).')

        self._add_subkicker(card,
            f'validacao numerica  —  x = [{self.amostra[0]:.2f},  {self.amostra[1]:.2f}]')
        x = self.amostra
        for classe in ['setosa', 'versicolor', 'virginica']:
            mj = self.prototipos[classe]
            d_disc = produto_escalar(x, mj) - 0.5 * produto_escalar(mj, mj)
            d_eucl = distancia_euclidiana(x, mj)
            self._add_step(card,
                f'  {classe[:3]}  ·   d_j(x) = {d_disc:+9.4f}     '
                f'||x - m_j|| = {d_eucl:.4f}')
        self._add_explain(card,
            'Maior d_j(x)  ↔  menor ||x - m_j||  —  equivalencia confirmada.',
            pady=(4, 0))
        self._respiro(card)

    # ==================================================================
    # SECAO 4 — Fronteiras de Decisao
    # ==================================================================
    def _secao_fronteiras(self, parent):
        card = Card(parent, titulo='4. Fronteiras de Decisao  ·  Pares de Classes')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'A fronteira entre i e j e onde d_i(x) = d_j(x). '
            'Simplifica para w^T·x + b = 0 no plano:')
        self._add_formula(card,
            r'$w \;=\; m_i - m_j \qquad\quad '
            r'b \;=\; -\dfrac{1}{2}\,\left(\|m_i\|^{2} - \|m_j\|^{2}\right)$',
            fontsize=18)
        self._add_formula(card,
            r'$w_{1}\,x_{1} + w_{2}\,x_{2} + b \;=\; 0 \;\;\Longrightarrow'
            r'\;\; x_{2} \;=\; \dfrac{-\,w_{1}\,x_{1} - b}{w_{2}}$',
            fontsize=17)
        self._add_ref(card, coeficientes_superficie_decisao)
        self._add_ref(card, produto_escalar)

        self._add_explain(card,
            'Aplicando aos prototipos atuais — uma equacao por par:',
            pady=(8, 4))

        for classe_i, classe_j in PARES:
            mi = self.prototipos[classe_i]
            mj = self.prototipos[classe_j]
            w, b = coeficientes_superficie_decisao(mi, mj)
            mi_norm2 = produto_escalar(mi, mi)
            mj_norm2 = produto_escalar(mj, mj)

            self._add_subkicker(card, f'par   {classe_i}   x   {classe_j}')
            self._add_step(card,
                f'  m_{classe_i[:3]} = [{mi[0]:.4f}, {mi[1]:.4f}]    '
                f'm_{classe_j[:3]} = [{mj[0]:.4f}, {mj[1]:.4f}]')
            self._add_step(card,
                f'  w   = m_{classe_i[:3]} - m_{classe_j[:3]}  =  '
                f'[{w[0]:+.4f},  {w[1]:+.4f}]')
            self._add_step(card,
                f'  ||m_{classe_i[:3]}||² = {mi_norm2:.4f}      '
                f'||m_{classe_j[:3]}||² = {mj_norm2:.4f}')
            self._add_step(card,
                f'  b   = -½·({mi_norm2:.4f} - {mj_norm2:.4f})  =  {b:+.4f}')
            self._add_step(card,
                f'  Fronteira:   {w[0]:+.4f}·x₁   {w[1]:+.4f}·x₂   {b:+.4f}  =  0')
            if abs(w[1]) > 1e-9:
                a1 = -w[0] / w[1]
                a0 = -b / w[1]
                self._add_step(card,
                    f'  Reta:    x₂  =  {a0:+.4f}   {a1:+.4f}·x₁')
        self._respiro(card)


# ===========================================================================
# Aba 2 — Perceptron / Regra Delta
# ===========================================================================
class JanelaMemoriaCalculoPD(tk.Toplevel):
    def __init__(self, parent, algo, w, classe_pos, classe_neg,
                 eixos, taxa, epocas_treinadas, historico, amostra=None):
        super().__init__(parent)
        nome_algo = ('Perceptron de Rosenblatt' if algo == 'perceptron'
                     else 'Regra Delta  (Adaline / Widrow-Hoff)')
        self.title(f'Memoria de Calculo  ·  {nome_algo}')
        self.geometry('980x860')
        self.minsize(820, 600)
        self.configure(bg=T.BG)
        self.transient(parent)

        self.algo = algo
        self.w = list(w)
        self.classe_pos = classe_pos
        self.classe_neg = classe_neg
        self.eixos = eixos
        self.taxa = taxa
        self.epocas_treinadas = epocas_treinadas
        self.historico = list(historico)
        self.amostra = list(amostra) if amostra else [4.5, 1.5]
        self._imagens_ref = []

        self._construir()

    # ------------------------------------------------------------------
    def _construir(self):
        nome_algo = ('Perceptron de Rosenblatt' if self.algo == 'perceptron'
                     else 'Regra Delta  (Adaline / Widrow-Hoff)')

        tk.Frame(self, bg=T.ACCENT, height=2).pack(fill='x', side='top')

        head = tk.Frame(self, bg=T.BG, height=90)
        head.pack(fill='x', side='top')
        head.pack_propagate(False)
        tk.Label(head, text='MEMORIA DE CALCULO',
                 bg=T.BG, fg=T.ACCENT, font=T.FONT_KICKER
                ).pack(anchor='w', padx=24, pady=(16, 0))
        tk.Label(head, text=nome_algo,
                 bg=T.BG, fg=T.FG, font=T.FONT_TITLE
                ).pack(anchor='w', padx=24)
        tk.Label(head,
                 text=(f'Par: {self.classe_pos}  x  {self.classe_neg}   '
                       f'|   {self.eixos[0]}  ·  {self.eixos[1]}   '
                       f'|   ρ = {self.taxa}   |   {self.epocas_treinadas} epoca(s)'),
                 bg=T.BG, fg=T.FG_MUTED, font=T.FONT_SUBTITLE
                ).pack(anchor='w', padx=24, pady=(2, 0))

        tk.Frame(self, bg=T.BORDER, height=1).pack(fill='x', side='top')

        rod = tk.Frame(self, bg=T.BG_PANEL, height=44)
        rod.pack(fill='x', side='bottom')
        rod.pack_propagate(False)
        tk.Frame(self, bg=T.BORDER, height=1).pack(fill='x', side='bottom')
        ttk.Button(rod, text='Fechar', command=self.destroy
                  ).pack(side='right', padx=20, pady=8)
        tk.Label(rod, text='Formulas via matplotlib mathtext  ·  calculos em Python puro',
                 bg=T.BG_PANEL, fg=T.FG_DIM, font=T.FONT_SUBTITLE
                ).pack(side='left', padx=20, pady=11)

        canvas = tk.Canvas(self, bg=T.BG, highlightthickness=0, borderwidth=0)
        scroll = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        wrap = tk.Frame(canvas, bg=T.BG)
        win_id = canvas.create_window((0, 0), window=wrap, anchor='nw')
        wrap.bind('<Configure>',
                  lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',
                    lambda e: canvas.itemconfigure(win_id, width=e.width))

        def _wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        canvas.bind_all('<MouseWheel>', _wheel)
        self.protocol('WM_DELETE_WINDOW',
                      lambda: (canvas.unbind_all('<MouseWheel>'), self.destroy()))

        self._secao_vetor_aumentado(wrap)

        if self.algo == 'perceptron':
            self._secao_ativacao_limiar(wrap)
            self._secao_regra_perceptron(wrap)
        else:
            self._secao_saida_linear(wrap)
            self._secao_erro_mse(wrap)
            self._secao_regra_delta(wrap)

        self._secao_pesos_fronteira(wrap)
        self._secao_classificacao(wrap)
        tk.Frame(wrap, bg=T.BG, height=24).pack()

    # ------------------------------------------------------------------
    # Helpers (mesmos de JanelaMemoriaCalculo)
    # ------------------------------------------------------------------
    def _formula(self, latex, fontsize=16, bg=T.BG_CARD):
        fig = Figure(figsize=(0.1, 0.1), dpi=120, facecolor=bg)
        fig.text(0, 0, latex, fontsize=fontsize, color=T.FG,
                 va='bottom', ha='left')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight',
                    pad_inches=0.18, facecolor=bg)
        buf.seek(0)
        photo = ImageTk.PhotoImage(Image.open(buf))
        self._imagens_ref.append(photo)
        return photo

    def _add_formula(self, parent, latex, fontsize=16, bg=T.BG_CARD, pady=(6, 6)):
        photo = self._formula(latex, fontsize=fontsize, bg=bg)
        tk.Label(parent, image=photo, bg=bg).pack(anchor='w', padx=18, pady=pady)

    def _add_step(self, parent, texto):
        tk.Label(parent, text=texto, bg=T.BG_CARD, fg=T.FG,
                 font=T.FONT_MONO_SM, anchor='w', justify='left'
                ).pack(anchor='w', padx=18, pady=1)

    def _add_subkicker(self, parent, texto):
        tk.Label(parent, text=texto.upper(), bg=T.BG_CARD, fg=T.ACCENT,
                 font=T.FONT_KICKER, anchor='w'
                ).pack(anchor='w', padx=18, pady=(10, 4))

    def _add_explain(self, parent, texto, pady=(2, 4)):
        tk.Label(parent, text=texto, bg=T.BG_CARD, fg=T.FG_MUTED,
                 font=T.FONT_LABEL, wraplength=900, justify='left'
                ).pack(anchor='w', padx=18, pady=pady)

    def _add_resultado(self, parent, texto, cor):
        tk.Label(parent, text=texto, bg=T.BG_CARD, fg=cor,
                 font=T.FONT_TITLE, anchor='w'
                ).pack(anchor='w', padx=18, pady=(10, 14))

    def _add_ref(self, parent, func):
        nome, linha = _ref_funcao(func)
        tk.Label(parent,
                 text=f'  →  {nome}  :  linha {linha}',
                 bg=T.BG_CARD, fg=T.FG_DIM,
                 font=('Consolas', 8, 'normal'), anchor='w'
                ).pack(anchor='w', padx=18, pady=(0, 4))

    def _respiro(self, parent):
        tk.Frame(parent, bg=T.BG_CARD, height=10).pack()

    # ==================================================================
    # SECAO 1 — Vetor Aumentado (bias trick)
    # ==================================================================
    def _secao_vetor_aumentado(self, parent):
        card = Card(parent, titulo='1. Vetor Aumentado  ·  Bias Trick')
        card.pack(fill='x', padx=22, pady=(20, 0))

        self._add_explain(card,
            'O bias w₀ e tratado como um peso extra, ligado a uma entrada '
            'fixa x₀ = 1. Isso permite escrever o produto interno de forma '
            'matricial uniforme com transposta:')
        self._add_formula(card,
            r'$\mathbf{x}_{aug} = [1,\; x_1,\; x_2]^{T} \qquad'
            r'\mathbf{w} = [w_0,\; w_1,\; w_2]^{T}$',
            fontsize=17)
        self._add_formula(card,
            r'$net = \mathbf{w}^{T} \cdot \mathbf{x}_{aug} = '
            r'w_0 \cdot 1 + w_1\,x_1 + w_2\,x_2$',
            fontsize=18)
        self._add_ref(card, treinar_perceptron if self.algo == 'perceptron'
                     else treinar_delta_iris)

        v1, v2 = self.amostra
        self._add_subkicker(card, 'substituicao  —  amostra de teste')
        self._add_step(card,
            f'  x       = [{v1:.4f},  {v2:.4f}]')
        self._add_step(card,
            f'  x_aug   = [1,  {v1:.4f},  {v2:.4f}]^T   (bias concatenado)')
        self._add_step(card,
            f'  w^T     = [{self.w[0]:+.6f},  {self.w[1]:+.6f},  {self.w[2]:+.6f}]')
        self._respiro(card)

    # ==================================================================
    # SECAO 2a — Ativacao Limiar (Perceptron)
    # ==================================================================
    def _secao_ativacao_limiar(self, parent):
        card = Card(parent, titulo='2. Ativacao Limiar  ·  sgn(net)')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'O Perceptron usa ativacao de limiar duro. '
            'O sinal de net = w^T·x_aug determina a classe predita.')
        self._add_formula(card,
            r'$\hat{y} = \mathrm{sgn}(net) \;=\; '
            r'+1\;\;\text{se}\;\; net \geq 0,\quad '
            r'-1\;\;\text{caso contrario}$',
            fontsize=16)
        self._add_ref(card, _sgn)

        self._add_subkicker(card, 'mapeamento de classes')
        self._add_step(card, f'  d = +1   →   {self.classe_pos}  (classe positiva)')
        self._add_step(card, f'  d = -1   →   {self.classe_neg}  (classe negativa)')
        self._respiro(card)

    # ==================================================================
    # SECAO 2b — Saida Linear (Delta)
    # ==================================================================
    def _secao_saida_linear(self, parent):
        card = Card(parent, titulo='2. Saida Linear  ·  net sem limiar')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'Diferenca fundamental em relacao ao Perceptron: a Regra Delta '
            'usa a saida LINEAR net = w^T·x_aug — sem aplicar sgn(). '
            'O erro e calculado entre d e o valor continuo net.')
        self._add_formula(card,
            r'$net = \mathbf{w}^{T} \cdot \mathbf{x}_{aug} = '
            r'w_0 + w_1\,x_1 + w_2\,x_2$',
            fontsize=18)
        self._add_ref(card, _treinar_delta)

        self._add_subkicker(card, 'mapeamento de classes')
        self._add_step(card, f'  d = +1.0   →   {self.classe_pos}  (classe positiva)')
        self._add_step(card, f'  d = -1.0   →   {self.classe_neg}  (classe negativa)')
        self._add_explain(card,
            'Na classificacao final: sgn(net) >= 0  →  classe positiva.',
            pady=(4, 0))
        self._respiro(card)

    # ==================================================================
    # SECAO 3b — Erro e MSE
    # ==================================================================
    def _secao_erro_mse(self, parent):
        card = Card(parent, titulo='3. Erro e MSE  ·  Criterio de Convergencia')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'O erro e calculado entre o valor desejado d e a saida linear net. '
            'O MSE medio por epoca mede a qualidade do ajuste.')
        self._add_formula(card,
            r'$e = d - net = d - \mathbf{w}^{T} \cdot \mathbf{x}_{aug}$',
            fontsize=18)
        self._add_formula(card,
            r'$MSE = \dfrac{1}{N}\,\sum_{k=1}^{N} e_k^{2} = '
            r'\dfrac{1}{N}\,\sum_{k=1}^{N} '
            r'\left(d_k - \mathbf{w}^{T} \cdot \mathbf{x}_{k}\right)^{2}$',
            fontsize=16)
        self._add_ref(card, _treinar_delta)

        if self.historico:
            self._add_subkicker(card, 'historico de mse')
            mse_ini = self.historico[0]
            mse_fin = self.historico[-1]
            reducao = (1 - mse_fin / mse_ini) * 100 if mse_ini > 1e-12 else 0
            self._add_step(card, f'  MSE inicial  (epoca   1):  {mse_ini:.6f}')
            self._add_step(card,
                f'  MSE final    (epoca {self.epocas_treinadas:>3}):  {mse_fin:.6f}')
            self._add_step(card, f'  Reducao:                     {reducao:.1f}%')
        self._respiro(card)

    # ==================================================================
    # SECAO 3a — Regra de Atualizacao Perceptron
    # ==================================================================
    def _secao_regra_perceptron(self, parent):
        card = Card(parent, titulo='3. Regra de Atualizacao  ·  Perceptron de Rosenblatt')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'O vetor de pesos e atualizado SOMENTE quando a predicao erra '
            '(d ≠ ŷ). Convergencia garantida pelo Teorema de Rosenblatt '
            'para dados linearmente separaveis.')
        self._add_formula(card,
            r'$\mathbf{w}_{t+1} = \mathbf{w}_{t} + '
            r'\rho \cdot (d - \hat{y}) \cdot \mathbf{x}_{aug}$',
            fontsize=19)
        self._add_formula(card,
            r'$w_{i} \;\leftarrow\; w_{i} + \rho\,(d - \hat{y})\,x_{aug,i}'
            r'\qquad \text{(componente a componente)}$',
            fontsize=14)
        self._add_ref(card, treinar_perceptron)

        taxa = float(self.taxa)
        v1, v2 = self.amostra
        self._add_subkicker(card, 'exemplo de atualizacao com erro')
        self._add_explain(card,
            f'Supondo amostra positiva (d=+1) classificada errada como negativa '
            f'(ŷ=−1).  ρ = {taxa}',
            pady=(4, 4))
        delta_escalar = taxa * (1 - (-1))
        self._add_step(card, f'  d - ŷ      = +1 - (-1) = +2')
        self._add_step(card,
            f'  Δw = ρ·(d-ŷ)·x_aug = {taxa}·2·[1, {v1:.4f}, {v2:.4f}]^T')
        self._add_step(card,
            f'     = [{delta_escalar:.4f},  '
            f'{delta_escalar*v1:.4f},  '
            f'{delta_escalar*v2:.4f}]^T')

        if self.historico:
            self._add_subkicker(card, 'historico de erros  (ultimas epocas)')
            n_mostrar = min(5, len(self.historico))
            for i, erros in enumerate(self.historico[-n_mostrar:]):
                ep = self.epocas_treinadas - n_mostrar + i + 1
                self._add_step(card, f'  Epoca {ep:>4}:  {int(erros)} erro(s)')
            if self.historico[-1] == 0:
                self._add_resultado(card,
                    f'  Convergiu em {self.epocas_treinadas} epocas.',
                    cor=T.SUCCESS)
            else:
                self._add_resultado(card,
                    f'  Nao convergiu — {int(self.historico[-1])} erro(s) final.',
                    cor=T.DANGER)
        self._respiro(card)

    # ==================================================================
    # SECAO 4b — Regra de Atualizacao Delta
    # ==================================================================
    def _secao_regra_delta(self, parent):
        card = Card(parent, titulo='4. Regra de Atualizacao  ·  Widrow-Hoff')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'A atualizacao ocorre em TODA amostra, nao so nos erros. '
            'Minimiza o MSE independentemente de separabilidade linear.')
        self._add_formula(card,
            r'$\mathbf{w}_{t+1} = \mathbf{w}_{t} + \rho \cdot e \cdot \mathbf{x}_{aug}$',
            fontsize=19)
        self._add_formula(card,
            r'$= \mathbf{w}_{t} + \rho\,'
            r'(d - \mathbf{w}_{t}^{T}\,\mathbf{x}_{aug})\,'
            r'\mathbf{x}_{aug}$',
            fontsize=17)
        self._add_ref(card, treinar_delta_iris)
        self._add_ref(card, _treinar_delta)

        v1, v2 = self.amostra
        w0, w1, w2 = self.w
        net = w0 + w1 * v1 + w2 * v2
        d_ex = 1.0
        e_ex = d_ex - net
        taxa = float(self.taxa)
        dw = [taxa * e_ex * xi for xi in [1.0, v1, v2]]

        self._add_subkicker(card, 'exemplo com amostra de teste  (d=+1)')
        self._add_step(card,
            f'  x_aug  = [1,  {v1:.4f},  {v2:.4f}]^T')
        self._add_step(card,
            f'  w^T    = [{w0:+.6f},  {w1:+.6f},  {w2:+.6f}]')
        self._add_step(card,
            f'  net    = w^T·x_aug')
        self._add_step(card,
            f'         = ({w0:+.4f})·1  +  ({w1:+.4f})·{v1:.4f}  '
            f'+  ({w2:+.4f})·{v2:.4f}')
        self._add_step(card, f'         = {net:+.6f}')
        self._add_step(card, f'  e      = d - net = {d_ex:.1f} - ({net:+.4f}) = {e_ex:+.6f}')
        self._add_step(card,
            f'  Δw     = ρ·e·x_aug = {taxa}·({e_ex:.4f})·[1, {v1:.4f}, {v2:.4f}]^T')
        self._add_step(card,
            f'         = [{dw[0]:+.6f},  {dw[1]:+.6f},  {dw[2]:+.6f}]^T')
        self._respiro(card)

    # ==================================================================
    # SECAO N-1 — Pesos Treinados e Fronteira
    # ==================================================================
    def _secao_pesos_fronteira(self, parent):
        n = '4' if self.algo == 'perceptron' else '5'
        card = Card(parent, titulo=f'{n}. Pesos Treinados  ·  Fronteira de Decisao')
        card.pack(fill='x', padx=22, pady=(22, 0))

        w0, w1, w2 = self.w
        self._add_explain(card, 'Vetor de pesos resultante do treinamento:')
        self._add_formula(card,
            r'$\mathbf{w} = [w_0,\; w_1,\; w_2]^{T}$', fontsize=18)
        self._add_ref(card, treinar_perceptron if self.algo == 'perceptron'
                     else treinar_delta_iris)

        self._add_subkicker(card, 'valores treinados')
        self._add_step(card, f'  w₀ (bias) = {w0:+.6f}')
        self._add_step(card, f'  w₁        = {w1:+.6f}')
        self._add_step(card, f'  w₂        = {w2:+.6f}')

        self._add_subkicker(card, 'fronteira de decisao  —  net = 0')
        self._add_formula(card,
            r'$w_0 + w_1\,x_1 + w_2\,x_2 = 0$', fontsize=17)
        self._add_step(card,
            f'  ({w0:+.4f})  +  ({w1:+.4f})·x₁  +  ({w2:+.4f})·x₂  =  0')
        if abs(w2) > 1e-9:
            a1 = -w1 / w2
            a0 = -w0 / w2
            self._add_formula(card,
                r'$x_2 = \dfrac{-\,w_0 - w_1\,x_1}{w_2}$', fontsize=17)
            self._add_step(card, f'  x₂  =  {a0:+.4f}  {a1:+.4f}·x₁')
        self._respiro(card)

    # ==================================================================
    # SECAO N — Classificacao da Amostra
    # ==================================================================
    def _secao_classificacao(self, parent):
        n = '5' if self.algo == 'perceptron' else '6'
        card = Card(parent, titulo=f'{n}. Classificacao  ·  Substituicao Numerica')
        card.pack(fill='x', padx=22, pady=(22, 0))

        v1, v2 = self.amostra
        self._add_explain(card,
            f'Classificando x = [{v1:.4f},  {v2:.4f}] com os pesos treinados:')
        self._add_ref(card, predizer_perceptron if self.algo == 'perceptron'
                     else predizer_delta)

        w0, w1, w2 = self.w
        net = w0 + w1 * v1 + w2 * v2

        self._add_subkicker(card, 'passo a passo')
        self._add_step(card, f'  x_aug        = [1,  {v1:.4f},  {v2:.4f}]^T')
        self._add_step(card, f'  w^T · x_aug  = w₀·1 + w₁·x₁ + w₂·x₂')
        self._add_step(card,
            f'               = ({w0:+.6f})·1'
            f'  +  ({w1:+.6f})·{v1:.4f}'
            f'  +  ({w2:+.6f})·{v2:.4f}')
        self._add_step(card, f'  net          = {net:+.6f}')

        y_hat = 1 if net >= 0 else -1
        pred = self.classe_pos if net >= 0 else self.classe_neg
        self._add_step(card, f'  sgn(net)     = sgn({net:+.4f}) = {y_hat:+d}')

        cor = CORES_CLASSE.get(pred, T.FG)
        self._add_resultado(card, f'  Predicao  →   {pred.upper()}', cor=cor)
        self._respiro(card)


# ===========================================================================
# Aba 3 — Metricas Avancadas (Acerto Global, Kappa, Tau, MCC, F1, F2, etc.)
# ===========================================================================
class JanelaMemoriaCalculoMetricas(tk.Toplevel):
    """
    Janela de memoria de calculo da aba "Metricas Avancadas".

    Recebe o relatorio de um modelo selecionado (via relatorio_completo)
    e exibe formulas + substituicao numerica para cada metrica, com
    referencia arquivo:linha apontando para o codigo correspondente.
    """

    def __init__(self, parent, nome_modelo, relatorio, classes,
                 classe_foco='setosa', perc_vs_delta=None):
        super().__init__(parent)
        self.title(f'Memoria de Calculo  ·  Metricas  ·  {nome_modelo}')
        self.geometry('1020x860')
        self.minsize(860, 600)
        self.configure(bg=T.BG)
        self.transient(parent)

        self.nome_modelo = nome_modelo
        self.rel = relatorio
        self.classes = list(classes)
        self.classe_foco = classe_foco if classe_foco in classes else classes[0]
        # Tupla (rel_perceptron, rel_delta) — None se nao disponivel
        self.perc_vs_delta = perc_vs_delta
        self._imagens_ref = []

        self._construir()

    # ------------------------------------------------------------------
    def _construir(self):
        tk.Frame(self, bg=T.ACCENT, height=2).pack(fill='x', side='top')

        head = tk.Frame(self, bg=T.BG, height=90)
        head.pack(fill='x', side='top')
        head.pack_propagate(False)
        tk.Label(head, text='MEMORIA DE CALCULO  ·  METRICAS DE QUALIDADE',
                 bg=T.BG, fg=T.ACCENT, font=T.FONT_KICKER
                ).pack(anchor='w', padx=24, pady=(16, 0))
        tk.Label(head, text=f'Modelo: {self.nome_modelo}',
                 bg=T.BG, fg=T.FG, font=T.FONT_TITLE
                ).pack(anchor='w', padx=24)
        m_total = sum(self.rel['matriz'][r][p]
                      for r in self.classes for p in self.classes)
        tk.Label(head,
                 text=(f'Classes: {", ".join(self.classes)}   |   '
                       f'm = {m_total} amostras teste   |   '
                       f'Classe foco (OvR): {self.classe_foco}'),
                 bg=T.BG, fg=T.FG_MUTED, font=T.FONT_SUBTITLE
                ).pack(anchor='w', padx=24, pady=(2, 0))

        tk.Frame(self, bg=T.BORDER, height=1).pack(fill='x', side='top')

        rod = tk.Frame(self, bg=T.BG_PANEL, height=44)
        rod.pack(fill='x', side='bottom')
        rod.pack_propagate(False)
        tk.Frame(self, bg=T.BORDER, height=1).pack(fill='x', side='bottom')
        ttk.Button(rod, text='Fechar', command=self.destroy
                  ).pack(side='right', padx=20, pady=8)
        tk.Label(rod,
                 text='Formulas via matplotlib mathtext  ·  calculos em Python puro',
                 bg=T.BG_PANEL, fg=T.FG_DIM, font=T.FONT_SUBTITLE
                ).pack(side='left', padx=20, pady=11)

        canvas = tk.Canvas(self, bg=T.BG, highlightthickness=0, borderwidth=0)
        scroll = ttk.Scrollbar(self, orient='vertical', command=canvas.yview)
        canvas.configure(yscrollcommand=scroll.set)
        scroll.pack(side='right', fill='y')
        canvas.pack(side='left', fill='both', expand=True)

        wrap = tk.Frame(canvas, bg=T.BG)
        win_id = canvas.create_window((0, 0), window=wrap, anchor='nw')
        wrap.bind('<Configure>',
                  lambda e: canvas.configure(scrollregion=canvas.bbox('all')))
        canvas.bind('<Configure>',
                    lambda e: canvas.itemconfigure(win_id, width=e.width))

        def _wheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')
        canvas.bind_all('<MouseWheel>', _wheel)
        self.protocol('WM_DELETE_WINDOW',
                      lambda: (canvas.unbind_all('<MouseWheel>'), self.destroy()))

        # Secoes
        self._secao_matriz(wrap)
        self._secao_acerto_global(wrap)
        self._secao_kappa(wrap)
        self._secao_tau(wrap)
        self._secao_variancias(wrap)
        if self.perc_vs_delta is not None:
            self._secao_teste_z(wrap)
        self._secao_metricas_ovr(wrap)
        self._secao_binarias(wrap)
        self._secao_mcc_fb(wrap)
        tk.Frame(wrap, bg=T.BG, height=24).pack()

    # ------------------------------------------------------------------
    # Helpers (mesmo padrao das outras janelas)
    # ------------------------------------------------------------------
    def _formula(self, latex, fontsize=16, bg=T.BG_CARD):
        fig = Figure(figsize=(0.1, 0.1), dpi=120, facecolor=bg)
        fig.text(0, 0, latex, fontsize=fontsize, color=T.FG,
                 va='bottom', ha='left')
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight',
                    pad_inches=0.18, facecolor=bg)
        buf.seek(0)
        photo = ImageTk.PhotoImage(Image.open(buf))
        self._imagens_ref.append(photo)
        return photo

    def _add_formula(self, parent, latex, fontsize=16, bg=T.BG_CARD,
                     pady=(6, 6)):
        photo = self._formula(latex, fontsize=fontsize, bg=bg)
        tk.Label(parent, image=photo, bg=bg
                ).pack(anchor='w', padx=18, pady=pady)

    def _add_step(self, parent, texto):
        tk.Label(parent, text=texto, bg=T.BG_CARD, fg=T.FG,
                 font=T.FONT_MONO_SM, anchor='w', justify='left'
                ).pack(anchor='w', padx=18, pady=1)

    def _add_subkicker(self, parent, texto):
        tk.Label(parent, text=texto.upper(), bg=T.BG_CARD, fg=T.ACCENT,
                 font=T.FONT_KICKER, anchor='w'
                ).pack(anchor='w', padx=18, pady=(10, 4))

    def _add_explain(self, parent, texto, pady=(2, 4)):
        tk.Label(parent, text=texto, bg=T.BG_CARD, fg=T.FG_MUTED,
                 font=T.FONT_LABEL, wraplength=940, justify='left'
                ).pack(anchor='w', padx=18, pady=pady)

    def _add_resultado(self, parent, texto, cor):
        tk.Label(parent, text=texto, bg=T.BG_CARD, fg=cor,
                 font=T.FONT_TITLE, anchor='w'
                ).pack(anchor='w', padx=18, pady=(10, 14))

    def _add_ref(self, parent, func):
        nome, linha = _ref_funcao(func)
        tk.Label(parent,
                 text=f'  →  {nome}  :  linha {linha}',
                 bg=T.BG_CARD, fg=T.FG_DIM,
                 font=('Consolas', 8, 'normal'), anchor='w'
                ).pack(anchor='w', padx=18, pady=(0, 4))

    def _respiro(self, parent):
        tk.Frame(parent, bg=T.BG_CARD, height=10).pack()

    # ==================================================================
    # SECAO 1 — Matriz de confusao
    # ==================================================================
    def _secao_matriz(self, parent):
        card = Card(parent, titulo='1. Matriz de Confusao  ·  Base de Todos os Calculos')
        card.pack(fill='x', padx=22, pady=(20, 0))

        self._add_explain(card,
            'A matriz de confusao A = [a_ij] cruza classe REAL (linhas) e '
            'classe PREDITA (colunas). Todas as metricas que seguem sao '
            'derivadas dela.')

        matriz = self.rel['matriz']
        # Linha-cabecalho
        grid = tk.Frame(card, bg=T.BG_CARD)
        grid.pack(anchor='w', padx=18, pady=(6, 6))

        def cel(r, c, texto, bg=T.BG_CARD, fg=T.FG, bold=False):
            f = ('Consolas', 9, 'bold') if bold else T.FONT_MONO_SM
            tk.Label(grid, text=texto, bg=bg, fg=fg, font=f,
                     width=14, anchor='center',
                     highlightthickness=1, highlightbackground=T.BORDER
                    ).grid(row=r, column=c, padx=1, pady=1, sticky='nsew')

        cel(0, 0, 'Real \\ Pred', bg=T.BG_PANEL, fg=T.FG_MUTED, bold=True)
        for j, c in enumerate(self.classes):
            cel(0, j + 1, c.capitalize(),
                bg=CORES_CLASSE.get(c, T.BG_PANEL), fg='white', bold=True)
        cel(0, len(self.classes) + 1, 'a_{i+}',
            bg=T.BG_PANEL, fg=T.ACCENT, bold=True)

        for i, real in enumerate(self.classes):
            cel(i + 1, 0, real.capitalize(),
                bg=CORES_CLASSE.get(real, T.BG_PANEL), fg='white', bold=True)
            linha_total = sum(matriz[real][p] for p in self.classes)
            for j, pred in enumerate(self.classes):
                v = matriz[real][pred]
                bg = T.BG_HOVER if i == j else T.BG_CARD
                cel(i + 1, j + 1, str(v), bg=bg)
            cel(i + 1, len(self.classes) + 1, str(linha_total),
                bg=T.BG_PANEL, fg=T.ACCENT, bold=True)

        cel(len(self.classes) + 1, 0, 'a_{+j}',
            bg=T.BG_PANEL, fg=T.ACCENT, bold=True)
        m = 0
        for j, pred in enumerate(self.classes):
            col_total = sum(matriz[r][pred] for r in self.classes)
            cel(len(self.classes) + 1, j + 1, str(col_total),
                bg=T.BG_PANEL, fg=T.ACCENT, bold=True)
            m += col_total
        cel(len(self.classes) + 1, len(self.classes) + 1, f'm={m}',
            bg=T.BG_PANEL, fg=T.SUCCESS, bold=True)

        self._add_step(card,
            f'  a_ii (diagonal):  '
            + '  +  '.join(str(matriz[c][c]) for c in self.classes)
            + f'  =  {sum(matriz[c][c] for c in self.classes)} acertos')
        self._respiro(card)

    # ==================================================================
    # SECAO 2 — Acerto Global
    # ==================================================================
    def _secao_acerto_global(self, parent):
        card = Card(parent, titulo='2. Acerto Global  (Ag)')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'Razao entre acertos (diagonal) e total de amostras. '
            'A medida mais simples de qualidade.')
        self._add_formula(card,
            r'$A_g \;=\; \dfrac{1}{m}\, \sum_{i=1}^{C}\, a_{ii}$',
            fontsize=20)
        self._add_ref(card, acerto_global)

        matriz = self.rel['matriz']
        m = sum(matriz[r][p] for r in self.classes for p in self.classes)
        diag = sum(matriz[c][c] for c in self.classes)
        ag = self.rel['acerto_global']

        self._add_subkicker(card, 'substituicao numerica')
        self._add_step(card,
            f'  Somatorio diagonal:  '
            + '  +  '.join(f'{matriz[c][c]}' for c in self.classes)
            + f'  =  {diag}')
        self._add_step(card, f'  Total m:             {m}')
        self._add_step(card, f'  Ag  =  {diag} / {m}  =  {ag:.6f}')

        cor = T.SUCCESS if ag >= 0.9 else T.ACCENT if ag >= 0.7 else T.DANGER
        self._add_resultado(card,
            f'  Ag  =  {ag:.6f}    ({ag*100:.2f}%)', cor=cor)

    # ==================================================================
    # SECAO 3 — Acerto Casual e Kappa
    # ==================================================================
    def _secao_kappa(self, parent):
        card = Card(parent, titulo='3. Acerto Casual  e  Coeficiente Kappa')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'O Kappa corrige o Ag pelo acerto que ocorreria por acaso. '
            'Primeiro calculamos o acerto casual (Aa) e depois o Kappa.')

        self._add_formula(card,
            r'$A_a \;=\; \dfrac{1}{m^{2}}\, \sum_{i=1}^{C}\, '
            r'a_{i+}\cdot a_{+i}$', fontsize=18)
        self._add_formula(card,
            r'$K \;=\; \dfrac{A_g - A_a}{1 - A_a}$', fontsize=20)
        self._add_ref(card, _acerto_casual)
        self._add_ref(card, kappa)

        matriz = self.rel['matriz']
        m = sum(matriz[r][p] for r in self.classes for p in self.classes)
        aa = _acerto_casual(matriz, self.classes)
        k = self.rel['kappa']
        ag = self.rel['acerto_global']

        self._add_subkicker(card, 'somas marginais  (a_{i+}, a_{+i})')
        for c in self.classes:
            linha = sum(matriz[c][p] for p in self.classes)
            coluna = sum(matriz[r][c] for r in self.classes)
            self._add_step(card,
                f'  {c[:3]}:  a_{c[:3]}+ = {linha}   a_+{c[:3]} = {coluna}   '
                f'produto = {linha*coluna}')

        soma = sum(sum(matriz[c][p] for p in self.classes)
                   * sum(matriz[r][c] for r in self.classes)
                   for c in self.classes)
        self._add_step(card, f'  soma dos produtos  =  {soma}')
        self._add_step(card, f'  m²                 =  {m*m}')
        self._add_step(card, f'  Aa  =  {soma} / {m*m}  =  {aa:.6f}')

        self._add_subkicker(card, 'kappa')
        self._add_step(card, f'  Ag  =  {ag:.6f}')
        self._add_step(card, f'  Aa  =  {aa:.6f}')
        self._add_step(card,
            f'  K   =  ({ag:.4f} - {aa:.4f}) / (1 - {aa:.4f})  =  {k:.6f}')

        cor = T.SUCCESS if k > 0.80 else T.ACCENT if k > 0.40 else T.DANGER
        interp = ('Quase Perfeito' if k > 0.80 else
                  'Substancial'    if k > 0.60 else
                  'Moderado'       if k > 0.40 else
                  'Razoavel'       if k > 0.20 else
                  'Fraco / Nenhum')
        self._add_resultado(card,
            f'  K  =  {k:.6f}   ({interp})', cor=cor)

    # ==================================================================
    # SECAO 4 — Tau
    # ==================================================================
    def _secao_tau(self, parent):
        card = Card(parent, titulo='4. Coeficiente Tau')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'Alternativa ao Kappa que assume distribuicao uniforme entre '
            'classes — assume que o acerto por acaso e exatamente 1/C.')
        self._add_formula(card,
            r'$\tau \;=\; \dfrac{A_g - \frac{1}{C}}{1 - \frac{1}{C}}$',
            fontsize=20)
        self._add_ref(card, tau)

        c = len(self.classes)
        ag = self.rel['acerto_global']
        t = self.rel['tau']

        self._add_subkicker(card, 'substituicao numerica')
        self._add_step(card, f'  C  (numero de classes)  =  {c}')
        self._add_step(card, f'  1/C                     =  {1/c:.6f}')
        self._add_step(card, f'  Ag                      =  {ag:.6f}')
        self._add_step(card,
            f'  tau  =  ({ag:.4f} - {1/c:.4f}) / (1 - {1/c:.4f})  =  {t:.6f}')

        cor = T.SUCCESS if t > 0.80 else T.ACCENT if t > 0.40 else T.DANGER
        self._add_resultado(card, f'  tau  =  {t:.6f}', cor=cor)

    # ==================================================================
    # SECAO 5 — Variancias
    # ==================================================================
    def _secao_variancias(self, parent):
        card = Card(parent, titulo='5. Variancias  (necessarias para o teste Z)')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'A variancia de Kappa (Congalton & Green, 2009) usa 4 termos '
            'auxiliares phi_1..phi_4. A variancia de Tau e mais simples.')

        self._add_formula(card,
            r'$\sigma^{2}_{\tau} \;=\; \dfrac{1}{m}\cdot'
            r'\dfrac{A_g\,(1-A_g)}{\left(1 - \frac{1}{C}\right)^{2}}$',
            fontsize=17)
        self._add_ref(card, variancia_tau)

        self._add_formula(card,
            r'$\sigma^{2}_{K} \;=\; \dfrac{1}{m}\left[ '
            r'\dfrac{\phi_1(1-\phi_1)}{(1-\phi_2)^{2}} + \cdots \right]$',
            fontsize=15)
        self._add_ref(card, variancia_kappa)

        vk = self.rel['variancia_kappa']
        vt = self.rel['variancia_tau']

        self._add_subkicker(card, 'valores calculados')
        self._add_step(card, f'  Var(Kappa)  =  {vk:.8f}')
        self._add_step(card, f'  Var(Tau)    =  {vt:.8f}')
        self._add_step(card, f'  σ(Kappa)    =  {vk**0.5:.6f}')
        self._add_step(card, f'  σ(Tau)      =  {vt**0.5:.6f}')
        self._respiro(card)

    # ==================================================================
    # SECAO 6 — Teste Z (Perceptron vs Delta)
    # ==================================================================
    def _secao_teste_z(self, parent):
        card = Card(parent, titulo='6. Teste Z  ·  Significancia entre 2 Classificadores')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'Testa se a diferenca entre dois Kappas (ou Taus) e '
            'estatisticamente significativa ao nivel de 5%.')
        self._add_formula(card,
            r'$Z_{K} \;=\; \dfrac{K_{1} - K_{2}}'
            r'{\sqrt{\sigma^{2}_{K_{1}} + \sigma^{2}_{K_{2}}}}$',
            fontsize=20)
        self._add_ref(card, z_kappa)
        self._add_ref(card, z_tau)
        self._add_ref(card, p_valor_z)

        perc, delt = self.perc_vs_delta
        k1, vk1 = perc['kappa'], perc['variancia_kappa']
        k2, vk2 = delt['kappa'], delt['variancia_kappa']
        t1, vt1 = perc['tau'],   perc['variancia_tau']
        t2, vt2 = delt['tau'],   delt['variancia_tau']

        zk = z_kappa(k1, vk1, k2, vk2)
        zt = z_tau(t1, vt1, t2, vt2)
        pk = p_valor_z(zk)
        pt = p_valor_z(zt)

        self._add_subkicker(card, 'kappa  —  perceptron ova vs delta ova')
        self._add_step(card,
            f'  K1 = {k1:.6f}   Var(K1) = {vk1:.8f}')
        self._add_step(card,
            f'  K2 = {k2:.6f}   Var(K2) = {vk2:.8f}')
        self._add_step(card,
            f'  Zk =  ({k1:.4f} - {k2:.4f}) / sqrt({vk1:.6f} + {vk2:.6f})')
        self._add_step(card,
            f'     =  {k1-k2:+.6f} / {(vk1+vk2)**0.5:.6f}  =  {zk:+.6f}')
        self._add_step(card, f'  p-valor = {pk:.6f}')

        self._add_subkicker(card, 'tau  —  perceptron ova vs delta ova')
        self._add_step(card,
            f'  T1 = {t1:.6f}   Var(T1) = {vt1:.8f}')
        self._add_step(card,
            f'  T2 = {t2:.6f}   Var(T2) = {vt2:.8f}')
        self._add_step(card,
            f'  Zt =  {zt:+.6f}    p-valor = {pt:.6f}')

        sig_k = pk < 0.05
        sig_t = pt < 0.05
        if sig_k or sig_t:
            self._add_resultado(card,
                f'  Diferenca SIGNIFICATIVA  (rejeita H0)',
                cor=T.DANGER)
        else:
            self._add_resultado(card,
                f'  Diferenca NAO significativa  (mantem H0)',
                cor=T.SUCCESS)

    # ==================================================================
    # SECAO 7 — Acuracia Produtor / Usuario (OvR)
    # ==================================================================
    def _secao_metricas_ovr(self, parent):
        card = Card(parent,
                    titulo='7. Acuracia do Produtor e do Usuario  (por classe)')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'Acuracia do Produtor (sensibilidade / recall): de tudo que era '
            'da classe i no mundo real, quanto foi corretamente capturado.\n'
            'Acuracia do Usuario (precisao / VPP): de tudo que foi predito '
            'como classe i, quanto era realmente i.')

        self._add_formula(card,
            r'$A_{p_i} \;=\; \dfrac{a_{ii}}{a_{+i}}$',
            fontsize=18)
        self._add_ref(card, acuracia_produtor)
        self._add_formula(card,
            r'$A_{u_i} \;=\; \dfrac{a_{ii}}{a_{i+}}$',
            fontsize=18)
        self._add_ref(card, acuracia_usuario)

        matriz = self.rel['matriz']
        for c in self.classes:
            aii = matriz[c][c]
            linha  = sum(matriz[c][p] for p in self.classes)
            coluna = sum(matriz[r][c] for r in self.classes)
            ap = aii / coluna if coluna else 0.0
            au = aii / linha  if linha  else 0.0

            self._add_subkicker(card, f'classe  {c}')
            self._add_step(card,
                f'  a_ii = {aii}    a_+i = {coluna}    a_i+ = {linha}')
            self._add_step(card,
                f'  Ap_{c[:3]}  =  {aii} / {coluna}  =  {ap:.6f}   '
                f'({ap*100:.2f}%)')
            self._add_step(card,
                f'  Au_{c[:3]}  =  {aii} / {linha}   =  {au:.6f}   '
                f'({au*100:.2f}%)')
        self._respiro(card)

    # ==================================================================
    # SECAO 8 — Extracao binaria OvR (VP, FP, FN, VN)
    # ==================================================================
    def _secao_binarias(self, parent):
        card = Card(parent,
                    titulo=f'8. Decomposicao Binaria OvR  ·  foco: {self.classe_foco}')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'Reduz o problema multiclasse a um problema binario '
            f'"{self.classe_foco}" vs resto, extraindo VP, FP, FN, VN da '
            'matriz multiclasse.')
        self._add_formula(card,
            r'$\mathrm{VP} = a_{ii} \quad '
            r'\mathrm{FP} = \sum_{r \neq i} a_{ri} \quad '
            r'\mathrm{FN} = \sum_{p \neq i} a_{ip} \quad '
            r'\mathrm{VN} = \sum_{r \neq i, p \neq i} a_{rp}$',
            fontsize=14)
        self._add_ref(card, _extrair_binario)

        matriz = self.rel['matriz']
        c = self.classe_foco
        vp, fp, fn, vn = _extrair_binario(matriz, c, self.classes)

        self._add_subkicker(card, f'extracao para classe = {c}')
        self._add_step(card, f'  VP  =  a[{c[:3]}][{c[:3]}]  =  {vp}')

        fp_partes = [f'a[{r[:3]}][{c[:3]}]={matriz[r][c]}'
                     for r in self.classes if r != c]
        self._add_step(card,
            f'  FP  =  ' + '  +  '.join(fp_partes) + f'  =  {fp}')

        fn_partes = [f'a[{c[:3]}][{p[:3]}]={matriz[c][p]}'
                     for p in self.classes if p != c]
        self._add_step(card,
            f'  FN  =  ' + '  +  '.join(fn_partes) + f'  =  {fn}')

        self._add_step(card,
            f'  VN  =  soma das celulas que nao envolvem {c}  =  {vn}')

        # Metricas binarias
        sens = sensibilidade(vp, fn)
        espec = especificidade(vn, fp)
        prec = precisao_binaria(vp, fp)

        self._add_subkicker(card, 'metricas binarias derivadas')
        self._add_formula(card,
            r'$\mathrm{Sens} = \dfrac{VP}{VP+FN}  \quad  '
            r'\mathrm{Espec} = \dfrac{VN}{VN+FP}  \quad  '
            r'\mathrm{Prec} = \dfrac{VP}{VP+FP}$',
            fontsize=15)
        self._add_ref(card, sensibilidade)
        self._add_ref(card, especificidade)
        self._add_ref(card, precisao_binaria)

        self._add_step(card,
            f'  Sens   =  {vp} / ({vp}+{fn})  =  {vp}/{vp+fn}  =  '
            f'{sens:.6f}   ({sens*100:.2f}%)')
        self._add_step(card,
            f'  Espec  =  {vn} / ({vn}+{fp})  =  {vn}/{vn+fp}  =  '
            f'{espec:.6f}   ({espec*100:.2f}%)')
        self._add_step(card,
            f'  Prec   =  {vp} / ({vp}+{fp})  =  {vp}/{vp+fp}  =  '
            f'{prec:.6f}   ({prec*100:.2f}%)')
        self._respiro(card)

    # ==================================================================
    # SECAO 9 — MCC e Fb Score
    # ==================================================================
    def _secao_mcc_fb(self, parent):
        card = Card(parent,
                    titulo=f'9. Coeficiente de Matthews  e  Fb Score  ·  {self.classe_foco}')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'MCC e robusto para classes desbalanceadas — usa as 4 celulas '
            'da matriz binaria. Fb e a media harmonica ponderada de '
            'precisao e revocacao (b=1 → F1 equilibrado, b=2 → favorece recall).')

        self._add_formula(card,
            r'$\mathrm{MCC} \;=\; \dfrac{VP\cdot VN \,-\, FP\cdot FN}'
            r'{\sqrt{(VP+FP)(VP+FN)(VN+FP)(VN+FN)}}$',
            fontsize=14)
        self._add_ref(card, mcc)

        self._add_formula(card,
            r'$F_{\beta} \;=\; (1+\beta^{2})\,'
            r'\dfrac{Pr \cdot Re}{\beta^{2}\, Pr + Re}$',
            fontsize=17)
        self._add_ref(card, fb_score)

        matriz = self.rel['matriz']
        c = self.classe_foco
        vp, fp, fn, vn = _extrair_binario(matriz, c, self.classes)
        mv = mcc(vp, vn, fp, fn)
        f1 = fb_score(vp, fp, fn, b=1)
        f2 = fb_score(vp, fp, fn, b=2)

        self._add_subkicker(card, 'mcc  —  substituicao')
        num = vp * vn - fp * fn
        den = ((vp + fp) * (vp + fn) * (vn + fp) * (vn + fn)) ** 0.5
        self._add_step(card,
            f'  numerador    =  {vp}·{vn} - {fp}·{fn}  =  {num}')
        self._add_step(card,
            f'  denominador  =  sqrt(({vp+fp})·({vp+fn})·({vn+fp})·({vn+fn}))  '
            f'=  {den:.4f}')
        self._add_step(card, f'  MCC  =  {num} / {den:.4f}  =  {mv:.6f}')

        self._add_subkicker(card, 'f1 e f2')
        pr = precisao_binaria(vp, fp)
        re = sensibilidade(vp, fn)
        self._add_step(card, f'  Pr = {pr:.6f}    Re = {re:.6f}')
        if pr + re > 0:
            self._add_step(card,
                f'  F1  =  2·({pr:.4f}·{re:.4f}) / ({pr:.4f} + {re:.4f})  '
                f'=  {f1:.6f}')
        denf2 = 4 * pr + re
        if denf2 > 0:
            self._add_step(card,
                f'  F2  =  5·({pr:.4f}·{re:.4f}) / (4·{pr:.4f} + {re:.4f})  '
                f'=  {f2:.6f}')

        cor_mcc = T.SUCCESS if mv > 0.8 else T.ACCENT if mv > 0.4 else T.DANGER
        self._add_resultado(card,
            f'  MCC = {mv:.4f}    F1 = {f1:.4f}    F2 = {f2:.4f}',
            cor=cor_mcc)
