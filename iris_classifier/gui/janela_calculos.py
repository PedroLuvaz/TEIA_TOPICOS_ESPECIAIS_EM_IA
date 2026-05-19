"""
Janela de Memoria de Calculo - exibe formulas matematicas (LaTeX
renderizadas via matplotlib mathtext) com substituicao numerica
passo a passo usando os dados atuais do modelo.

Permite ao professor (e ao aluno) ver exatamente:
  i.   Como cada prototipo foi calculado
  ii.  Como a funcao discriminante e avaliada para uma amostra
  iii. Equivalencia entre  argmax d_j(x)  e  argmin ||x - m_j||
  iv.  Como w e b da fronteira saem das medias
  v.   Como a reta x2 = (-w1 x1 - b)/w2 vira o que aparece no grafico
"""
import io
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use('TkAgg')
matplotlib.rcParams['mathtext.fontset'] = 'stix'
from matplotlib.figure import Figure
from PIL import Image, ImageTk

from math_utils import (produto_escalar, distancia_euclidiana,
                        coeficientes_superficie_decisao)

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
        self.eixos = eixos                   # ('Comp. Petala', 'Larg. Petala')
        self.n_treino = n_treino_por_classe
        self.amostra = list(amostra) if amostra else [4.5, 1.5]
        self._imagens_ref = []               # mantem ref para PhotoImage

        self._construir()

    # ------------------------------------------------------------------
    # Construcao da janela
    # ------------------------------------------------------------------
    def _construir(self):
        # ---- Faixa visual de topo ----
        tk.Frame(self, bg=T.ACCENT, height=2).pack(fill='x', side='top')

        # ---- Header ----
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

        # ---- Footer ----
        rod = tk.Frame(self, bg=T.BG_PANEL, height=44)
        rod.pack(fill='x', side='bottom')
        rod.pack_propagate(False)
        tk.Frame(self, bg=T.BORDER, height=1).pack(fill='x', side='bottom')
        ttk.Button(rod, text='Fechar', command=self.destroy
                  ).pack(side='right', padx=20, pady=8)
        tk.Label(rod, text='Formulas renderizadas via matplotlib mathtext  ·  '
                           'calculos em Python puro',
                 bg=T.BG_PANEL, fg=T.FG_DIM, font=T.FONT_SUBTITLE
                ).pack(side='left', padx=20, pady=11)

        # ---- Conteudo scrollavel ----
        canvas = tk.Canvas(self, bg=T.BG, highlightthickness=0,
                           borderwidth=0)
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
        # Desligar bind global ao destruir a janela
        self.protocol('WM_DELETE_WINDOW',
                      lambda: (canvas.unbind_all('<MouseWheel>'), self.destroy()))

        # ---- Secoes ----
        self._secao_prototipos(wrap)
        self._secao_discriminante(wrap)
        self._secao_distancia(wrap)
        self._secao_fronteiras(wrap)
        tk.Frame(wrap, bg=T.BG, height=24).pack()  # respiro final

    # ------------------------------------------------------------------
    # Helpers de renderizacao
    # ------------------------------------------------------------------
    def _formula(self, latex, fontsize=16, bg=T.BG_CARD):
        """Renderiza formula LaTeX para tk.PhotoImage.

        Usa matplotlib mathtext com fonte STIX. Mantem ref do PhotoImage
        em self._imagens_ref para o GC nao engolir.
        """
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

    def _add_formula(self, parent, latex, fontsize=16,
                     bg=T.BG_CARD, pady=(6, 6)):
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
                 font=T.FONT_LABEL, wraplength=900, justify='left'
                ).pack(anchor='w', padx=18, pady=pady)

    def _add_resultado(self, parent, texto, cor):
        tk.Label(parent, text=texto, bg=T.BG_CARD, fg=cor,
                 font=T.FONT_TITLE, anchor='w'
                ).pack(anchor='w', padx=18, pady=(10, 14))

    def _respiro(self, parent):
        tk.Frame(parent, bg=T.BG_CARD, height=10).pack()

    # ==================================================================
    # SECAO 1 - Prototipos
    # ==================================================================
    def _secao_prototipos(self, parent):
        card = Card(parent, titulo='1. Prototipos  ·  Vetores Medios')
        card.pack(fill='x', padx=22, pady=(20, 0))

        self._add_explain(card,
            'Cada classe e representada pelo seu centroide — a media de todos '
            'os vetores de treino daquela classe. Esse centroide vira o '
            '"prototipo" usado pelo classificador.')
        self._add_formula(card,
            r'$m_j \;=\; \dfrac{1}{N_j}\, \sum_{x\, \in\, \omega_j}\, x$',
            fontsize=20)

        for classe in ['setosa', 'versicolor', 'virginica']:
            mj = self.prototipos[classe]
            self._add_subkicker(card, f'classe   {classe}')
            self._add_step(card,
                f'  N_{classe[:3]:<3} = {self.n_treino}   (amostras de treino)')
            self._add_step(card,
                f'  m_{classe[:3]:<3} = (1/{self.n_treino})  ·  [ Σ {self.eixos[0]} ,'
                f'  Σ {self.eixos[1]} ]')
            self._add_step(card,
                f'         = [ {mj[0]:.4f} ,  {mj[1]:.4f} ]')
        self._respiro(card)

    # ==================================================================
    # SECAO 2 - Funcao Discriminante
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
                f'  x · m_{classe[:3]:<3}  =  {x[0]:.2f}·{mj[0]:.4f}  +  '
                f'{x[1]:.2f}·{mj[1]:.4f}  =  {xtmj:.4f}')
            self._add_step(card,
                f'  m_{classe[:3]} · m_{classe[:3]}  =  {mj[0]:.4f}²  +  '
                f'{mj[1]:.4f}²  =  {mjmj:.4f}')
            self._add_step(card,
                f'  d_{classe[:3]:<3}(x)  =  {xtmj:.4f}   -   '
                f'½·{mjmj:.4f}   =   {d:+.4f}')

        vencedor = max(scores, key=scores.get)
        self._add_resultado(card,
            f'  argmax  →   {vencedor.upper()}    '
            f'(d = {scores[vencedor]:+.4f})',
            cor=CORES_CLASSE[vencedor])

    # ==================================================================
    # SECAO 3 - Distancia Euclidiana e Equivalencia
    # ==================================================================
    def _secao_distancia(self, parent):
        card = Card(parent, titulo='3. Distancia Euclidiana  ·  Equivalencia')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'A distancia euclidiana e a metrica intuitiva: a classe '
            'cujo prototipo esta mais proximo da amostra ganha.')
        self._add_formula(card,
            r'$\|x - m_j\| \;=\; \sqrt{\,\sum_{k}\,'
            r'(x_k - m_{jk})^{2}\,}$', fontsize=20)
        self._add_formula(card,
            r'$\arg\max_{j}\; d_j(x) \;\;\equiv\;\; '
            r'\arg\min_{j}\; \|x - m_j\|$', fontsize=18)

        self._add_explain(card,
            'Por que sao equivalentes? Expandindo o quadrado da distancia:')
        self._add_formula(card,
            r'$\|x - m_j\|^{2} \;=\; x^{T} x \;-\; 2\, x^{T} m_j '
            r'\;+\; m_j^{T} m_j$', fontsize=17)
        self._add_explain(card,
            'O termo  x·x  e constante em j (nao depende da classe). '
            'Logo, minimizar  ||x - m_j||²  equivale a maximizar  '
            'x·m_j - ½·m_j·m_j  =  d_j(x).')

        # Validacao numerica
        self._add_subkicker(card,
            f'validacao numerica com x = [{self.amostra[0]:.2f},  '
            f'{self.amostra[1]:.2f}]')
        x = self.amostra
        for classe in ['setosa', 'versicolor', 'virginica']:
            mj = self.prototipos[classe]
            d_disc = produto_escalar(x, mj) - 0.5 * produto_escalar(mj, mj)
            d_eucl = distancia_euclidiana(x, mj)
            self._add_step(card,
                f'  {classe[:3]}  ·   d_j(x) = {d_disc:+9.4f}     '
                f'||x - m_j|| = {d_eucl:.4f}')
        self._add_explain(card,
            'Note: o maior d_j(x) corresponde a menor ||x - m_j||, '
            'confirmando a equivalencia.', pady=(4, 0))
        self._respiro(card)

    # ==================================================================
    # SECAO 4 - Fronteiras de Decisao
    # ==================================================================
    def _secao_fronteiras(self, parent):
        card = Card(parent, titulo='4. Fronteiras de Decisao  ·  Pares de Classes')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'A fronteira entre duas classes i e j e o conjunto de pontos '
            'onde os dois discriminantes sao iguais: d_i(x) = d_j(x). '
            'Isso simplifica para uma reta no plano:')
        self._add_formula(card,
            r'$w \;=\; m_i - m_j \qquad\quad b \;=\; -\dfrac{1}{2}\,'
            r'\left(\|m_i\|^{2} - \|m_j\|^{2}\right)$',
            fontsize=18)
        self._add_formula(card,
            r'$w_{1}\,x_{1} + w_{2}\,x_{2} + b \;=\; 0 \;\;\Longrightarrow'
            r'\;\; x_{2} \;=\; \dfrac{-\,w_{1}\,x_{1} - b}{w_{2}}$',
            fontsize=17)

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
                f'  b   = -½·({mi_norm2:.4f} - {mj_norm2:.4f})  =  '
                f'{b:+.4f}')
            self._add_step(card,
                f'  Equacao da fronteira:   '
                f'{w[0]:+.4f} x1   {w[1]:+.4f} x2   {b:+.4f}  =  0')
            if abs(w[1]) > 1e-9:
                a1 = -w[0] / w[1]
                a0 = -b / w[1]
                self._add_step(card,
                    f'  Reta no plano:    '
                    f'x2  =  {a1:+.4f} · x1   {a0:+.4f}')
        self._respiro(card)
