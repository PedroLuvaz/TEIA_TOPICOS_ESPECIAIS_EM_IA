"""
Janelas de Memoria de Calculo.

JanelaMemoriaCalculo    — Classificador de Distancia Minima (Aba 1)
JanelaMemoriaCalculoPD  — Perceptron / Regra Delta (Aba 2)

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
