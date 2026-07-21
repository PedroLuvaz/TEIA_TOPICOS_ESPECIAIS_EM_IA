"""
Janelas de Memoria de Calculo.

JanelaMemoriaCalculo           — Classificador de Distancia Minima (Aba 1)
JanelaMemoriaCalculoPD         — Perceptron / Regra Delta            (Aba 2)
JanelaMemoriaCalculoMetricas   — Metricas Avancadas (Ag, K, tau, ...)(Aba 3)
JanelaMemoriaCalculoBayes      — Bayes Otimo / Naive Bayes           (Aba 4)
JanelaMemoriaCalculoMLP        — Feedforward / Backpropagation       (Aba 5)

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
import matplotlib.patches as mpatches
from PIL import Image, ImageTk

from core.math_utils import (produto_escalar, distancia_euclidiana,
                             coeficientes_superficie_decisao,
                             discriminante, calcular_media,
                             det_matriz, inv_matriz, calcular_covariancia,
                             calcular_covariancia_diagonal, distancia_mahalanobis_quad)
from models.classifier import treinar, predizer_todas_classes
from models.bayes_classifier import treinar_bayes, predizer_todas_classes_bayes, predizer_binario_bayes
from models.perceptron import treinar_perceptron, predizer_perceptron, _sgn
from models.delta_rule import treinar_delta_iris, predizer_delta, _treinar_delta
from models.mlp_backprop import RedeFeedforward, sigmoide
from evaluation.metricas_avancadas import (
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


# ---------------------------------------------------------------------------
# Diagrama de arquitetura de rede (MLP) — reutilizado por JanelaMemoriaCalculoMLP
# e JanelaMemoriaCalculoXOR, para desenhar circulos/conexoes/pesos/bias.
# ---------------------------------------------------------------------------
def _posicoes_coluna(n, espaco=1.3):
    """Posicoes Y centradas em 0 para n neuronios de uma coluna."""
    return [(n - 1) / 2 * espaco - i * espaco for i in range(n)]


def _desenhar_arquitetura_rede(rotulos_entrada, rotulos_ocultos, rotulos_saida,
                               pesos_oculta, bias_oculta, pesos_saida, bias_saida,
                               bias_compartilhado=False, bg=T.BG_CARD):
    """
    Desenha o diagrama da rede (circulos = neuronios, linhas = pesos, quadrado
    tracejado "+1" = bias), no mesmo estilo dos diagramas "Entrada / Oculta /
    Saida" dos slides da Aula PR_711. Retorna um io.BytesIO com o PNG.
    """
    n_in, n_hid, n_out = len(rotulos_entrada), len(rotulos_ocultos), len(rotulos_saida)
    espaco = 1.3
    n_max = max(n_in, n_hid, n_out)
    meia_altura = (n_max - 1) / 2 * espaco + 1.35
    largura_fig = 10.4
    altura_fig = max(3.6, meia_altura * 1.75)

    fig = Figure(figsize=(largura_fig, altura_fig), dpi=130, facecolor=bg)
    ax = fig.add_axes([0.01, 0.01, 0.98, 0.98])
    x_in, x_hid, x_out = 0.7, 5.2, 9.7
    ax.set_xlim(-0.5, 10.5)
    ax.set_ylim(-meia_altura, meia_altura)
    ax.set_aspect('equal')
    ax.axis('off')

    y_in = _posicoes_coluna(n_in, espaco)
    y_hid = _posicoes_coluna(n_hid, espaco)
    y_out = _posicoes_coluna(n_out, espaco)
    raio = 0.30

    topo = meia_altura - 0.35
    ax.text(x_in, topo, 'ENTRADA', ha='center', va='top', fontsize=10,
            fontweight='bold', color=T.FG_MUTED)
    ax.text(x_hid, topo, 'OCULTA  (σ)', ha='center', va='top', fontsize=10,
            fontweight='bold', color=T.ACCENT_DEEP)
    ax.text(x_out, topo, 'SAIDA  (σ)', ha='center', va='top', fontsize=10,
            fontweight='bold', color=T.DATA_BLUE)

    def _ligar(x0, ys0, x1, ys1, pesos):
        for i, y1 in enumerate(ys1):
            for j, y0 in enumerate(ys0):
                ax.plot([x0 + raio, x1 - raio], [y0, y1], color=T.BORDER_HARD,
                        lw=0.8, zorder=1, alpha=0.85)
                frac = 0.24 + 0.10 * (j % 3)
                xm = x0 + (x1 - x0) * frac
                ym = y0 + (y1 - y0) * frac
                ax.text(xm, ym, f'{pesos[i][j]:.3g}', fontsize=6.3,
                        color=T.FG_MUTED, ha='center', va='center', zorder=2,
                        bbox=dict(boxstyle='round,pad=0.06', fc=bg, ec='none', alpha=0.9))

    _ligar(x_in, y_in, x_hid, y_hid, pesos_oculta)
    _ligar(x_hid, y_hid, x_out, y_out, pesos_saida)

    def _bias_individual(x, ys, bias):
        for y, b in zip(ys, bias):
            ax.text(x, y - raio - 0.16, f'b={b:.4g}', ha='center', va='top',
                    fontsize=6.3, color=T.ACCENT_DEEP, zorder=2)

    def _bias_unico(x, ys, b):
        topo_circulo = max(ys) + raio
        yb = (topo_circulo + topo) / 2
        ax.scatter([x], [yb], s=190, marker='s', color=T.ACCENT_SOFT,
                   edgecolors=T.ACCENT_DEEP, linewidths=1.3, zorder=3)
        ax.text(x, yb, '+1', ha='center', va='center', fontsize=7.5,
                color=T.ACCENT_DEEP, zorder=4)
        for y in ys:
            ax.plot([x, x], [yb - 0.20, y + raio * 0.3], color=T.ACCENT,
                    lw=0.9, ls=':', zorder=1, alpha=0.85)
        ax.text(x + 0.42, yb, f'b={b:.4g}', ha='left', va='center',
                fontsize=6.5, color=T.ACCENT_DEEP, zorder=2)

    if bias_compartilhado:
        if n_hid:
            _bias_unico(x_hid, y_hid, bias_oculta[0])
        if n_out:
            _bias_unico(x_out, y_out, bias_saida[0])
    else:
        _bias_individual(x_hid, y_hid, bias_oculta)
        _bias_individual(x_out, y_out, bias_saida)

    colunas = [
        (x_in, y_in, rotulos_entrada, T.BG_PANEL, T.BORDER_HARD, T.FG_MUTED),
        (x_hid, y_hid, rotulos_ocultos, T.ACCENT_SOFT, T.ACCENT_DEEP, T.ACCENT_DEEP),
        (x_out, y_out, rotulos_saida, '#DBEAFE', T.DATA_BLUE, T.DATA_BLUE),
    ]
    for x, ys, labels, fc, ec, tc in colunas:
        for y, lbl in zip(ys, labels):
            ax.add_patch(mpatches.Circle((x, y), raio, facecolor=fc,
                                         edgecolor=ec, linewidth=1.6, zorder=4))
            ax.text(x, y, lbl, ha='center', va='center', fontsize=8.5,
                    fontweight='bold', color=tc, zorder=5)

    buf = io.BytesIO()
    fig.savefig(buf, format='png', facecolor=bg, bbox_inches='tight', pad_inches=0.12)
    buf.seek(0)
    return buf


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
        tk.Frame(self, bg=T.ACCENT, height=3).pack(fill='x', side='top')

        head = tk.Frame(self, bg=T.BG, height=78)
        head.pack(fill='x', side='top')
        head.pack_propagate(False)
        tk.Label(head, text='MEMORIA DE CALCULO',
                 bg=T.BG, fg=T.ACCENT_DEEP, font=T.FONT_KICKER
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
        tk.Label(parent, text=texto.upper(), bg=T.BG_CARD, fg=T.ACCENT_DEEP,
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
                 font=T.FONT_REF, anchor='w'
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

        tk.Frame(self, bg=T.ACCENT, height=3).pack(fill='x', side='top')

        head = tk.Frame(self, bg=T.BG, height=90)
        head.pack(fill='x', side='top')
        head.pack_propagate(False)
        tk.Label(head, text='MEMORIA DE CALCULO',
                 bg=T.BG, fg=T.ACCENT_DEEP, font=T.FONT_KICKER
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
        tk.Label(parent, text=texto.upper(), bg=T.BG_CARD, fg=T.ACCENT_DEEP,
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
                 font=T.FONT_REF, anchor='w'
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
        tk.Frame(self, bg=T.ACCENT, height=3).pack(fill='x', side='top')

        head = tk.Frame(self, bg=T.BG, height=90)
        head.pack(fill='x', side='top')
        head.pack_propagate(False)
        tk.Label(head, text='MEMORIA DE CALCULO  ·  METRICAS DE QUALIDADE',
                 bg=T.BG, fg=T.ACCENT_DEEP, font=T.FONT_KICKER
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
        tk.Label(parent, text=texto.upper(), bg=T.BG_CARD, fg=T.ACCENT_DEEP,
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
                 font=T.FONT_REF, anchor='w'
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
            'A matriz de confusao A = [a_ij] cruza classe PREDITA (linhas) e '
            'classe REAL (colunas). Todas as metricas que seguem sao '
            'derivadas dela.')

        matriz = self.rel['matriz']
        # Linha-cabecalho
        grid = tk.Frame(card, bg=T.BG_CARD)
        grid.pack(anchor='w', padx=18, pady=(6, 6))

        def cel(r, c, texto, bg=T.BG_CARD, fg=T.FG, bold=False):
            f = T.FONT_CELL_BOLD if bold else T.FONT_MONO_SM
            tk.Label(grid, text=texto, bg=bg, fg=fg, font=f,
                     width=14, anchor='center',
                     highlightthickness=1, highlightbackground=T.BORDER
                    ).grid(row=r, column=c, padx=1, pady=1, sticky='nsew')

        cel(0, 0, 'Pred \\ Real', bg=T.BG_PANEL, fg=T.FG_MUTED, bold=True)
        for j, c in enumerate(self.classes):
            cel(0, j + 1, c.capitalize(),
                bg=CORES_CLASSE.get(c, T.BG_PANEL), fg='white', bold=True)
        cel(0, len(self.classes) + 1, 'a_{i+} (Pred)',
            bg=T.BG_PANEL, fg=T.ACCENT_DEEP, bold=True)

        for i, pred in enumerate(self.classes):
            cel(i + 1, 0, pred.capitalize(),
                bg=CORES_CLASSE.get(pred, T.BG_PANEL), fg='white', bold=True)
            linha_total = sum(matriz[pred][r] for r in self.classes)
            for j, real in enumerate(self.classes):
                v = matriz[pred][real]
                bg = T.BG_HOVER if i == j else T.BG_CARD
                cel(i + 1, j + 1, str(v), bg=bg)
            cel(i + 1, len(self.classes) + 1, str(linha_total),
                bg=T.BG_PANEL, fg=T.ACCENT_DEEP, bold=True)

        cel(len(self.classes) + 1, 0, 'a_{+j} (Real)',
            bg=T.BG_PANEL, fg=T.ACCENT_DEEP, bold=True)
        m = 0
        for j, real in enumerate(self.classes):
            col_total = sum(matriz[p][real] for p in self.classes)
            cel(len(self.classes) + 1, j + 1, str(col_total),
                bg=T.BG_PANEL, fg=T.ACCENT_DEEP, bold=True)
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
            r'\mathrm{FP} = \sum_{r \neq i} a_{ir} \quad '
            r'\mathrm{FN} = \sum_{p \neq i} a_{pi} \quad '
            r'\mathrm{VN} = \sum_{r \neq i, p \neq i} a_{pr}$',
            fontsize=14)
        self._add_ref(card, _extrair_binario)

        matriz = self.rel['matriz']
        c = self.classe_foco
        vp, fp, fn, vn = _extrair_binario(matriz, c, self.classes)

        self._add_subkicker(card, f'extracao para classe = {c}')
        self._add_step(card, f'  VP  =  a[{c[:3]}][{c[:3]}]  =  {vp}')

        fp_partes = [f'a[{c[:3]}][{r[:3]}]={matriz[c][r]}'
                     for r in self.classes if r != c]
        self._add_step(card,
            f'  FP  =  ' + '  +  '.join(fp_partes) + f'  =  {fp}')

        fn_partes = [f'a[{p[:3]}][{c[:3]}]={matriz[p][c]}'
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


# ===========================================================================
# Aba 4 — Classificadores Bayes Ótimo e Naive Bayes (Nova Feature)
# ===========================================================================
class JanelaMemoriaCalculoBayes(tk.Toplevel):
    def __init__(self, parent, model_bayes, model_naive, active_attr_name, indices_sel,
                 amostra=None, rel_bayes=None, rel_naive=None):
        super().__init__(parent)
        self.title('Memoria de Calculo  ·  Bayes & Naive Bayes')
        self.geometry('980x820')
        self.minsize(820, 600)
        self.configure(bg=T.BG)
        self.transient(parent)

        self.model_bayes = model_bayes
        self.model_naive = model_naive
        self.attr_name = active_attr_name
        self.indices = indices_sel
        # amostra completa de 4 atributos
        self.amostra = list(amostra) if amostra else [5.8, 3.0, 4.5, 1.5]
        self.rel_bayes = rel_bayes
        self.rel_naive = rel_naive
        self._imagens_ref = []

        self._construir()

    def _construir(self):
        tk.Frame(self, bg=T.ACCENT, height=3).pack(fill='x', side='top')

        head = tk.Frame(self, bg=T.BG, height=78)
        head.pack(fill='x', side='top')
        head.pack_propagate(False)
        tk.Label(head, text='MEMORIA DE CALCULO  ·  BAYES & NAIVE BAYES',
                 bg=T.BG, fg=T.ACCENT_DEEP, font=T.FONT_KICKER
                ).pack(anchor='w', padx=24, pady=(16, 0))
        tk.Label(head, text='Substituicao numerica das formulas com os valores atuais dos classificadores',
                 bg=T.BG, fg=T.FG, font=T.FONT_TITLE
                ).pack(anchor='w', padx=24)
        tk.Label(head,
                 text=f'Atributos: {self.attr_name}  ·  Indices selecionados: {self.indices}',
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

        self._secao_parametros_estimados(wrap)
        self._secao_bayes_otimo(wrap)
        self._secao_naive_bayes(wrap)
        self._secao_significancia_kappa(wrap)
        tk.Frame(wrap, bg=T.BG, height=24).pack()

    # Helpers de formatacao e visualizacao
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
        tk.Label(parent, text=texto.upper(), bg=T.BG_CARD, fg=T.ACCENT_DEEP,
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
                 font=T.FONT_REF, anchor='w'
                ).pack(anchor='w', padx=18, pady=(0, 4))

    def _respiro(self, parent):
        tk.Frame(parent, bg=T.BG_CARD, height=10).pack()

    def _format_vetor(self, v):
        return "[" + ", ".join(f"{x:.4f}" for x in v) + "]"

    def _format_matriz(self, M):
        lines = []
        for r in M:
            lines.append("    [ " + "  ".join(f"{x:8.4f}" for x in r) + " ]")
        return "\n".join(lines)

    # Secoes
    def _secao_parametros_estimados(self, parent):
        card = Card(parent, titulo='1. Parametros Estimados (Media e Covariancia)')
        card.pack(fill='x', padx=22, pady=(20, 0))

        self._add_explain(card,
            'Os parametros da densidade condicional P(x|C_j) sao estimados a partir do treino:\n'
            ' - Vetor medio (m_j): a media amostral das amostras de cada classe.\n'
            ' - Matriz de covariancia (Sigma_j): dispersao conjunta das caracteristicas.')
        
        self._add_formula(card,
            r'$m_j = \dfrac{1}{N_j}\sum_{x \in \omega_j} x \quad\quad \Sigma_j = \dfrac{1}{N_j - 1}\sum_{x \in \omega_j} (x - m_j)(x - m_j)^T$',
            fontsize=15)
        self._add_ref(card, treinar_bayes)
        self._add_ref(card, calcular_covariancia)

        for c in CLASSES:
            self._add_subkicker(card, f'classe: {c}')
            params = self.model_bayes[c]
            self._add_step(card, f'  Vetor de Medias m_j: {self._format_vetor(params["media"])}')
            self._add_step(card, f'  Matriz de Covariancia Sigma_j:\n{self._format_matriz(params["cov"])}')
            self._add_step(card, f'  Determinante |Sigma_j|: {params["det"]:.8e}')
            self._add_step(card, f'  Covariancia Inversa Sigma_j^-1:\n{self._format_matriz(params["inv_cov"])}')
            self._respiro(card)

    def _secao_bayes_otimo(self, parent):
        card = Card(parent, titulo='2. Classificador Bayes Otimo (QDA)')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'Sob priori iguais P(C_j), o classificador MAP maximiza o log-discriminante quadratico:\n'
            'd_j(x) = -0.5 * ln|Sigma_j| - 0.5 * d_M^2(x, m_j)  onde  d_M^2 e a distancia de Mahalanobis.')
        
        self._add_formula(card,
            r'$d_j(x) = -\frac{1}{2} \ln |\Sigma_j| - \frac{1}{2} (x - m_j)^T \Sigma_j^{-1} (x - m_j)$',
            fontsize=16)
        self._add_ref(card, predizer_todas_classes_bayes)
        self._add_ref(card, distancia_mahalanobis_quad)

        # Amostra reduzida/selecionada
        x_sel = [self.amostra[i] for i in self.indices]
        self._add_subkicker(card, f'amostra avaliada: x = {self._format_vetor(x_sel)}')

        scores = {}
        for c in CLASSES:
            params = self.model_bayes[c]
            diff = [x_sel[i] - params['media'][i] for i in range(len(x_sel))]
            dm_sq = distancia_mahalanobis_quad(x_sel, params['media'], params['inv_cov'])
            score = -0.5 * math.log(params['det']) - 0.5 * dm_sq
            scores[c] = score
            
            self._add_subkicker(card, f'calculo para classe: {c}')
            self._add_step(card, f'  Diferenca (x - m_j) = {self._format_vetor(diff)}')
            self._add_step(card, f'  Dist. Mahalanobis d_M^2 = {dm_sq:.6f}')
            self._add_step(card, f'  d_{c[:3]}(x) = -0.5 * ln({params["det"]:.3e}) - 0.5 * {dm_sq:.4f}')
            self._add_step(card, f'             = -0.5 * ({math.log(params["det"]):.4f}) - {0.5*dm_sq:.4f}')
            self._add_step(card, f'             = {(-0.5*math.log(params["det"])):.4f} - {0.5*dm_sq:.4f} = {score:.6f}')
            self._respiro(card)

        vencedor = max(scores, key=scores.get)
        self._add_resultado(card, f'Predicao final Bayes Otimo: {vencedor.upper()} (score maximo = {scores[vencedor]:.4f})', CORES_CLASSE[vencedor])

    def _secao_naive_bayes(self, parent):
        card = Card(parent, titulo='3. Classificador Naive Bayes')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'Naive Bayes assume independencia de atributos. A covariancia e forcada a ser diagonal.\n'
            'Isso simplifica a distancia de Mahalanobis para a soma das diferencas quadraticas normalizadas pelas variancias.')
        
        self._add_formula(card,
            r'$d_j(x) = -\frac{1}{2} \sum_{i=1}^d \ln(\sigma_{ji}^2) - \frac{1}{2} \sum_{i=1}^d \frac{(x_i - m_{ji})^2}{\sigma_{ji}^2}$',
            fontsize=16)
        
        x_sel = [self.amostra[i] for i in self.indices]
        self._add_subkicker(card, f'amostra avaliada: x = {self._format_vetor(x_sel)}')

        scores = {}
        for c in CLASSES:
            params = self.model_naive[c]
            dm_sq = distancia_mahalanobis_quad(x_sel, params['media'], params['inv_cov'])
            score = -0.5 * math.log(params['det']) - 0.5 * dm_sq
            scores[c] = score
            
            # Pegar as variâncias (diagonal de cov)
            vars_c = [params['cov'][i][i] for i in range(len(x_sel))]
            self._add_subkicker(card, f'calculo para classe: {c}')
            self._add_step(card, f'  Variancias estimadas: {self._format_vetor(vars_c)}')
            self._add_step(card, f'  Soma termo logaritmico  = {sum(math.log(v) for v in vars_c):.4f}')
            self._add_step(card, f'  Soma termo quadratico   = {dm_sq:.4f}')
            self._add_step(card, f'  d_{c[:3]}(x) = {score:.6f}')
            self._respiro(card)

        vencedor = max(scores, key=scores.get)
        self._add_resultado(card, f'Predicao final Naive Bayes: {vencedor.upper()} (score maximo = {scores[vencedor]:.4f})', CORES_CLASSE[vencedor])

    def _secao_significancia_kappa(self, parent):
        if not self.rel_bayes or not self.rel_naive:
            return
            
        card = Card(parent, titulo='4. Teste de Significancia de Kappa (Comparativo)')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'O teste Z compara se a diferenca entre o Kappa do Bayes Otimo (K1) e do Naive Bayes (K2) '
            'e estatisticamente significativa no conjunto de teste.')
        
        self._add_formula(card,
            r'$Z = \dfrac{K_1 - K_2}{\sqrt{\mathrm{Var}(K_1) + \mathrm{Var}(K_2)}}$',
            fontsize=17)
        self._add_ref(card, z_kappa)
        self._add_ref(card, p_valor_z)

        k1 = self.rel_bayes['kappa']
        var1 = self.rel_bayes['variancia_kappa']
        k2 = self.rel_naive['kappa']
        var2 = self.rel_naive['variancia_kappa']
        
        z_stat = z_kappa(k1, var1, k2, var2)
        p_val = p_valor_z(z_stat)

        self._add_subkicker(card, 'estatisticas obtidas')
        self._add_step(card, f'  Bayes Otimo:  K1 = {k1:.6f}   Var(K1) = {var1:.8f}')
        self._add_step(card, f'  Naive Bayes:  K2 = {k2:.6f}   Var(K2) = {var2:.8f}')
        self._add_step(card, f'  Z = ({k1:.4f} - {k2:.4f}) / sqrt({var1:.6f} + {var2:.6f})')
        self._add_step(card, f'    = {k1 - k2:.6f} / {math.sqrt(var1 + var2):.6f} = {z_stat:.4f}')
        self._add_step(card, f'  p-valor bilateral = {p_val:.6f}')
        
        sig = "E SIGNIFICATIVA" if p_val < 0.05 else "NAO E SIGNIFICATIVA"
        cor_sig = T.SUCCESS if p_val >= 0.05 else T.DANGER
        self._add_resultado(card, f'Diferenca de desempenho {sig} no nivel de 5% (p = {p_val:.4f})', cor_sig)


# ===========================================================================
# Aba 5 — Feedforward (MLP) / Backpropagation — Lab 5, item (i)
# ===========================================================================
class JanelaMemoriaCalculoMLP(tk.Toplevel):
    """
    Janela de memoria de calculo do exemplo "Galinha vs Homem" (Lab 5).

    Recebe a arquitetura e os pesos iniciais da rede (dados no slide da
    Aula PR_711), recalcula a alimentacao adiante e um passo completo de
    retropropagacao, e exibe formulas LaTeX + substituicao numerica para
    cada etapa: net/ativacao, erro total, deltas e atualizacao dos pesos.
    """

    def __init__(self, parent, entradas, alvo, taxa_aprendizado,
                 pesos_oculta, bias_oculta, pesos_saida, bias_saida,
                 rotulos_entrada=None, rotulos_ocultos=None, rotulos_saida=None,
                 titulo_janela='Feedforward (MLP)  ·  Galinha vs Homem',
                 subtitulo='Rede 2-2-2  ·  Exemplo "Galinha vs Homem"  ·  Aula PR_711',
                 bias_compartilhado=False):
        super().__init__(parent)
        self.title(f'Memoria de Calculo  ·  {titulo_janela}')
        self.geometry('1000x980')
        self.minsize(860, 620)
        self.configure(bg=T.BG)
        self.transient(parent)

        self.entradas = list(entradas)
        self.alvo = list(alvo)
        self.taxa = taxa_aprendizado
        self.rotulos_entrada = rotulos_entrada or [f'a{i+1}' for i in range(len(entradas))]
        self.rotulos_ocultos = rotulos_ocultos or [f'b{i+1}' for i in range(len(bias_oculta))]
        self.rotulos_saida = rotulos_saida or [f'c{i+1}' for i in range(len(bias_saida))]
        self.subtitulo = subtitulo
        self.bias_compartilhado = bias_compartilhado
        self._imagens_ref = []

        # Copias "antes" (a rede interna sera mutada pelo passo de treinamento)
        self.w_oculta_antes = [row[:] for row in pesos_oculta]
        self.b_oculta_antes = list(bias_oculta)
        self.w_saida_antes = [row[:] for row in pesos_saida]
        self.b_saida_antes = list(bias_saida)

        self.rede = RedeFeedforward(
            n_entradas=len(self.entradas), n_ocultos=len(bias_oculta), n_saidas=len(bias_saida),
            pesos_oculta=[row[:] for row in pesos_oculta], bias_oculta=list(bias_oculta),
            pesos_saida=[row[:] for row in pesos_saida], bias_saida=list(bias_saida),
        )
        self.r = self.rede.passo_treinamento(self.entradas, self.alvo, self.taxa)

        if self.bias_compartilhado:
            # Convencao do exemplo didatico do slide (Aula PR_711): b1 e b2 sao
            # UM UNICO bias por camada, compartilhado por todos os neuronios
            # dela (nao um bias independente por neuronio). O gradiente desse
            # bias unico soma as contribuicoes de todos os deltas da camada:
            #   dE/db = sum(delta_da_camada)  ->  b_novo = b - eta * dE/db
            # (mesmo valor novo aplicado a todos os neuronios da camada).
            b_oculta_novo = self.b_oculta_antes[0] - self.taxa * sum(self.r['delta_oculta'])
            b_saida_novo = self.b_saida_antes[0] - self.taxa * sum(self.r['delta_saida'])
            self.r['b_oculta_depois'] = [b_oculta_novo] * len(bias_oculta)
            self.r['b_saida_depois'] = [b_saida_novo] * len(bias_saida)
            self.rede.b_oculta = list(self.r['b_oculta_depois'])
            self.rede.b_saida = list(self.r['b_saida_depois'])

        self.nova_saida = self.rede.prever(self.entradas)
        self.novo_erro = self.rede.erro_total(self.nova_saida, self.alvo)

        self._construir()

    # ------------------------------------------------------------------
    def _construir(self):
        tk.Frame(self, bg=T.ACCENT, height=3).pack(fill='x', side='top')

        head = tk.Frame(self, bg=T.BG, height=90)
        head.pack(fill='x', side='top')
        head.pack_propagate(False)
        tk.Label(head, text='MEMORIA DE CALCULO  ·  FEEDFORWARD (MLP)',
                 bg=T.BG, fg=T.ACCENT_DEEP, font=T.FONT_KICKER
                ).pack(anchor='w', padx=24, pady=(16, 0))
        tk.Label(head, text=self.subtitulo,
                 bg=T.BG, fg=T.FG, font=T.FONT_TITLE
                ).pack(anchor='w', padx=24)
        tk.Label(head,
                 text=(f'Entradas: {self._format_vetor(self.entradas)}   |   '
                       f'Alvo: {self._format_vetor(self.alvo)}   |   '
                       f'eta = {self.taxa}'),
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

        self._secao_arquitetura_rede(wrap)
        self._secao_modelo_neuronio(wrap)
        self._secao_forward_oculta(wrap)
        self._secao_forward_saida(wrap)
        self._secao_erro(wrap)
        self._secao_deltas_saida(wrap)
        self._secao_deltas_oculta(wrap)
        self._secao_atualizacao_pesos(wrap)
        self._secao_nova_predicao(wrap)
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

    def _add_formula(self, parent, latex, fontsize=16, bg=T.BG_CARD, pady=(6, 6)):
        photo = self._formula(latex, fontsize=fontsize, bg=bg)
        tk.Label(parent, image=photo, bg=bg).pack(anchor='w', padx=18, pady=pady)

    def _add_step(self, parent, texto):
        tk.Label(parent, text=texto, bg=T.BG_CARD, fg=T.FG,
                 font=T.FONT_MONO_SM, anchor='w', justify='left'
                ).pack(anchor='w', padx=18, pady=1)

    def _add_subkicker(self, parent, texto):
        tk.Label(parent, text=texto.upper(), bg=T.BG_CARD, fg=T.ACCENT_DEEP,
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
                 font=T.FONT_REF, anchor='w'
                ).pack(anchor='w', padx=18, pady=(0, 4))

    def _respiro(self, parent):
        tk.Frame(parent, bg=T.BG_CARD, height=10).pack()

    @staticmethod
    def _format_vetor(v):
        return '[' + ', '.join(f'{x:.4f}' for x in v) + ']'

    # ==================================================================
    # SECAO 1 — Arquitetura da Rede (diagrama)
    # ==================================================================
    def _secao_arquitetura_rede(self, parent):
        card = Card(parent, titulo='1. Arquitetura da Rede  ·  Diagrama')
        card.pack(fill='x', padx=22, pady=(20, 0))

        self._add_explain(card,
            'Diagrama da rede totalmente conectada usada nesta memoria de '
            'calculo, com os pesos e bias iniciais rotulados em cada conexao '
            '(mesmo estilo dos diagramas "Entrada / Oculta / Saida" da Aula '
            'PR_711).')

        buf = _desenhar_arquitetura_rede(
            self.rotulos_entrada, self.rotulos_ocultos, self.rotulos_saida,
            self.w_oculta_antes, self.b_oculta_antes,
            self.w_saida_antes, self.b_saida_antes,
            bias_compartilhado=self.bias_compartilhado, bg=T.BG_CARD)
        photo = ImageTk.PhotoImage(Image.open(buf))
        self._imagens_ref.append(photo)
        tk.Label(card, image=photo, bg=T.BG_CARD).pack(padx=18, pady=(4, 8))

        legenda = tk.Frame(card, bg=T.BG_CARD)
        legenda.pack(fill='x', padx=18, pady=(0, 10))

        def _item(cor_fundo, cor_borda, texto):
            linha = tk.Frame(legenda, bg=T.BG_CARD)
            linha.pack(fill='x', pady=1)
            tk.Frame(linha, bg=cor_fundo, width=14, height=14,
                     highlightthickness=1.5, highlightbackground=cor_borda
                    ).pack(side='left', padx=(0, 8), pady=2)
            tk.Label(linha, text=texto, bg=T.BG_CARD, fg=T.FG_MUTED,
                     font=T.FONT_LABEL, anchor='w', justify='left',
                     wraplength=880
                    ).pack(side='left', fill='x')

        _item(T.BG_PANEL, T.BORDER_HARD,
              'Circulo cinza = neuronio de ENTRADA (recebe o valor do atributo diretamente, sem ativacao).')
        _item(T.ACCENT_SOFT, T.ACCENT_DEEP,
              'Circulo ambar = neuronio da camada OCULTA (aplica a sigmoide sobre o net calculado).')
        _item('#DBEAFE', T.DATA_BLUE,
              'Circulo azul = neuronio de SAIDA (aplica a sigmoide, ativacao final da rede).')
        _item(T.BG_CARD, T.BORDER_HARD,
              'Linhas cinzas = conexoes entre neuronios, rotuladas com o valor do peso correspondente.')
        if self.bias_compartilhado:
            _item(T.ACCENT_SOFT, T.ACCENT_DEEP,
                  'Quadrado ambar "+1" tracejado = bias UNICO da camada, compartilhado por todos os '
                  'neuronios dela (convencao deste exemplo especifico — ver secao 8).')
        else:
            _item(T.BG_CARD, T.BG_CARD,
                  'Rotulo "b=valor" abaixo de cada neuronio = bias individual daquele neuronio '
                  '(um valor independente por neuronio, convencao padrao deste laboratorio).')
        self._respiro(card)

    # ==================================================================
    # SECAO 2 — Modelo do Neuronio (net + ativacao sigmoide)
    # ==================================================================
    def _secao_modelo_neuronio(self, parent):
        card = Card(parent, titulo='2. Modelo do Neuronio  ·  Net + Ativacao Sigmoide')
        card.pack(fill='x', padx=22, pady=(20, 0))

        self._add_explain(card,
            'Cada neuronio soma as entradas ponderadas pelos pesos, adiciona o '
            'bias, e aplica a funcao de ativacao sigmoide.')
        self._add_formula(card,
            r'$z_i(l) \;=\; \sum_{j} w_{ij}(l)\, a_j(l-1) \;+\; b_i(l)'
            r'\qquad\quad a_i(l) \;=\; \sigma\!\left(z_i(l)\right)$',
            fontsize=17)
        self._add_formula(card,
            r'$\sigma(z) \;=\; \dfrac{1}{1 + e^{-z}}$', fontsize=19)
        self._add_ref(card, RedeFeedforward.forward)
        self._add_ref(card, sigmoide)
        self._respiro(card)

    # ==================================================================
    # SECAO 3 — Alimentacao Adiante: Camada Oculta
    # ==================================================================
    def _secao_forward_oculta(self, parent):
        card = Card(parent, titulo='3. Alimentacao Adiante  ·  Camada Oculta')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'Substituindo as entradas e os pesos iniciais na formula do neuronio, '
            'para cada neuronio da camada oculta:')

        for i, nome in enumerate(self.rotulos_ocultos):
            w = self.w_oculta_antes[i]
            b = self.b_oculta_antes[i]
            net = b + sum(w[j] * self.entradas[j] for j in range(len(self.entradas)))
            out = self.r['saida_oculta'][i]

            termos = '  +  '.join(
                f'{self.entradas[j]:.2f}·{w[j]:.4f}' for j in range(len(self.entradas)))
            self._add_subkicker(card, f'neuronio  {nome}')
            self._add_step(card, f'  net_{nome} = {termos}  +  {b:.4f}  =  {net:.4f}')
            self._add_step(card, f'  out_{nome} = sigma({net:.4f}) = {out:.4f}')
        self._respiro(card)

    # ==================================================================
    # SECAO 4 — Alimentacao Adiante: Camada de Saida
    # ==================================================================
    def _secao_forward_saida(self, parent):
        card = Card(parent, titulo='4. Alimentacao Adiante  ·  Camada de Saida')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'As saidas da camada oculta alimentam a camada de saida, com o '
            'mesmo calculo de net + sigmoide.')

        for i, nome in enumerate(self.rotulos_saida):
            w = self.w_saida_antes[i]
            b = self.b_saida_antes[i]
            net = b + sum(w[j] * self.r['saida_oculta'][j] for j in range(len(self.r['saida_oculta'])))
            out = self.r['saida_rede'][i]

            termos = '  +  '.join(
                f'{self.r["saida_oculta"][j]:.4f}·{w[j]:.4f}'
                for j in range(len(self.r['saida_oculta'])))
            self._add_subkicker(card, f'neuronio  {nome}')
            self._add_step(card, f'  net_{nome} = {termos}  +  {b:.4f}  =  {net:.4f}')
            self._add_step(card, f'  out_{nome} = sigma({net:.4f}) = {out:.4f}')
        self._respiro(card)

    # ==================================================================
    # SECAO 5 — Erro Total
    # ==================================================================
    def _secao_erro(self, parent):
        card = Card(parent, titulo='5. Erro Total  ·  Funcao de Custo')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'O erro quadratico total mede a distancia entre a saida desejada '
            '(alvo) e a saida real da rede.')
        self._add_formula(card,
            r'$E \;=\; \dfrac{1}{2}\, \sum_{i}\, (t_i - z_i)^{2}$', fontsize=19)
        self._add_ref(card, RedeFeedforward.erro_total)

        self._add_subkicker(card, 'substituicao numerica')
        for nome, t, z in zip(self.rotulos_saida, self.alvo, self.r['saida_rede']):
            self._add_step(card, f'  (t_{nome} - out_{nome})^2 = ({t:.4f} - {z:.4f})^2 = {(t - z) ** 2:.6f}')
        self._add_resultado(card, f'  E = {self.r["erro_total"]:.5f}', cor=T.ACCENT_DEEP)

    # ==================================================================
    # SECAO 6 — Retropropagacao: Deltas da Camada de Saida
    # ==================================================================
    def _secao_deltas_saida(self, parent):
        card = Card(parent, titulo='6. Retropropagacao  ·  Deltas da Camada de Saida')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'O termo de erro (delta) de cada neuronio de saida combina o erro '
            'bruto com a derivada da sigmoide (que "trava" o ajuste quando o '
            'neuronio ja esta saturado).')
        self._add_formula(card,
            r'$\delta_o \;=\; (z_o - t_o)\; z_o\,(1 - z_o)$', fontsize=19)
        self._add_ref(card, RedeFeedforward.passo_treinamento)

        for nome, t, z, d in zip(self.rotulos_saida, self.alvo,
                                  self.r['saida_rede'], self.r['delta_saida']):
            self._add_subkicker(card, f'delta_{nome}')
            self._add_step(card,
                f'  ({z:.4f} - {t:.4f}) · {z:.4f} · (1 - {z:.4f})  =  {d:.6f}')
        self._respiro(card)

    # ==================================================================
    # SECAO 7 — Retropropagacao: Deltas da Camada Oculta
    # ==================================================================
    def _secao_deltas_oculta(self, parent):
        card = Card(parent, titulo='7. Retropropagacao  ·  Deltas da Camada Oculta')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'O erro de cada neuronio oculto e a soma ponderada dos deltas de '
            'TODOS os neuronios de saida aos quais ele se conecta.')
        self._add_formula(card,
            r'$\delta_h \;=\; \left(\sum_{o}\, \delta_o\, w_{ho}\right)'
            r'\; \text{out}_h\,(1 - \text{out}_h)$', fontsize=18)
        self._add_ref(card, RedeFeedforward.passo_treinamento)

        for i, nome in enumerate(self.rotulos_ocultos):
            out_h = self.r['saida_oculta'][i]
            termos = '  +  '.join(
                f'{self.r["delta_saida"][o]:.6f}·{self.w_saida_antes[o][i]:.4f}'
                for o in range(len(self.rotulos_saida)))
            soma = sum(self.r['delta_saida'][o] * self.w_saida_antes[o][i]
                       for o in range(len(self.rotulos_saida)))
            self._add_subkicker(card, f'delta_{nome}')
            self._add_step(card, f'  soma ponderada = {termos} = {soma:.6f}')
            self._add_step(card,
                f'  delta_{nome} = {soma:.6f} · {out_h:.4f} · (1 - {out_h:.4f}) '
                f'= {self.r["delta_oculta"][i]:.6f}')
        self._respiro(card)

    # ==================================================================
    # SECAO 8 — Atualizacao dos Pesos (Gradiente Descendente)
    # ==================================================================
    def _secao_atualizacao_pesos(self, parent):
        card = Card(parent, titulo='8. Atualizacao dos Pesos  ·  Gradiente Descendente')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'Cada peso e ajustado na direcao que reduz o erro, proporcional ao '
            'delta do neuronio de destino e a ativacao de entrada da conexao.')
        self._add_formula(card,
            r'$w_{\text{novo}} \;=\; w \;-\; \eta \cdot \delta \cdot \text{entrada}'
            r'\qquad\quad b_{\text{novo}} \;=\; b \;-\; \eta \cdot \delta$',
            fontsize=17)
        self._add_ref(card, RedeFeedforward.passo_treinamento)

        if self.bias_compartilhado:
            self._add_explain(card,
                'Este exemplo usa um UNICO bias por camada, compartilhado por '
                'todos os neuronios dela (convencao classica do slide didatico) '
                '— por isso o gradiente do bias soma os deltas de todos os '
                'neuronios da camada, em vez de usar apenas o delta de um.',
                pady=(0, 8))

        self._add_subkicker(card, f'camada de saida  (eta = {self.taxa})')
        for i, nome_o in enumerate(self.rotulos_saida):
            for j, nome_h in enumerate(self.rotulos_ocultos):
                antes = self.w_saida_antes[i][j]
                depois = self.r['w_saida_depois'][i][j]
                self._add_step(card,
                    f'  w({nome_h}->{nome_o}) = {antes:.5f} - {self.taxa}·{self.r["delta_saida"][i]:.6f}'
                    f'·{self.r["saida_oculta"][j]:.4f} = {depois:.5f}')
        if self.bias_compartilhado:
            antes_b = self.b_saida_antes[0]
            depois_b = self.r['b_saida_depois'][0]
            soma_d = sum(self.r['delta_saida'])
            self._add_step(card,
                f'  bias(saida) = {antes_b:.5f} - {self.taxa}·({"+".join(f"{d:.6f}" for d in self.r["delta_saida"])})'
                f' = {antes_b:.5f} - {self.taxa}·{soma_d:.6f} = {depois_b:.5f}')
        else:
            for i, nome_o in enumerate(self.rotulos_saida):
                antes_b = self.b_saida_antes[i]
                depois_b = self.r['b_saida_depois'][i]
                self._add_step(card,
                    f'  bias({nome_o}) = {antes_b:.5f} - {self.taxa}·{self.r["delta_saida"][i]:.6f} = {depois_b:.5f}')

        self._add_subkicker(card, f'camada oculta  (eta = {self.taxa})')
        for i, nome_h in enumerate(self.rotulos_ocultos):
            for j in range(len(self.entradas)):
                antes = self.w_oculta_antes[i][j]
                depois = self.r['w_oculta_depois'][i][j]
                self._add_step(card,
                    f'  w(a{j+1}->{nome_h}) = {antes:.5f} - {self.taxa}·{self.r["delta_oculta"][i]:.6f}'
                    f'·{self.entradas[j]:.4f} = {depois:.5f}')
        if self.bias_compartilhado:
            antes_b = self.b_oculta_antes[0]
            depois_b = self.r['b_oculta_depois'][0]
            soma_d = sum(self.r['delta_oculta'])
            self._add_step(card,
                f'  bias(oculta) = {antes_b:.5f} - {self.taxa}·({"+".join(f"{d:.6f}" for d in self.r["delta_oculta"])})'
                f' = {antes_b:.5f} - {self.taxa}·{soma_d:.6f} = {depois_b:.5f}')
        else:
            for i, nome_h in enumerate(self.rotulos_ocultos):
                antes_b = self.b_oculta_antes[i]
                depois_b = self.r['b_oculta_depois'][i]
                self._add_step(card,
                    f'  bias({nome_h}) = {antes_b:.5f} - {self.taxa}·{self.r["delta_oculta"][i]:.6f} = {depois_b:.5f}')
        self._respiro(card)

    # ==================================================================
    # SECAO 9 — Nova Predicao (apos 1 atualizacao)
    # ==================================================================
    def _secao_nova_predicao(self, parent):
        card = Card(parent, titulo='9. Nova Predicao  ·  Apos 1 Atualizacao de Pesos')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'Recalculando a alimentacao adiante com os pesos ja atualizados, '
            'usando a mesma amostra de entrada:')

        for nome, z in zip(self.rotulos_saida, self.nova_saida):
            self._add_step(card, f'  out_{nome} (novo) = {z:.4f}')

        reduziu = self.novo_erro < self.r['erro_total']
        cor = T.SUCCESS if reduziu else T.DANGER
        variacao = self.r['erro_total'] - self.novo_erro
        self._add_resultado(card,
            f'  E (novo) = {self.novo_erro:.5f}   '
            f'(era {self.r["erro_total"]:.5f}  ·  variacao = {variacao:+.5f})',
            cor=cor)


# ===========================================================================
# Lab 5 — Exercicio B: Problema XOR com MLP (1 epoca, 4 padroes online)
# ===========================================================================
class JanelaMemoriaCalculoXOR(tk.Toplevel):
    """
    Janela de memoria de calculo do Exercicio B do Lab 5: resolver o XOR com
    uma MLP (arquitetura da Fig. 12.28b — 2 entradas, 2 ocultos, 1 saida),
    demonstrando 1 epoca completa (os 4 padroes da tabela-verdade, em modo
    online/estocastico) de retropropagacao.
    """

    def __init__(self, parent, padroes, taxa_aprendizado,
                 pesos_oculta, bias_oculta, pesos_saida, bias_saida):
        super().__init__(parent)
        self.title('Memoria de Calculo  ·  XOR com MLP  ·  1 Epoca')
        self.geometry('1040x900')
        self.minsize(880, 620)
        self.configure(bg=T.BG)
        self.transient(parent)

        self.padroes = [(list(x), list(t)) for x, t in padroes]
        self.taxa = taxa_aprendizado
        self.pesos_oculta_iniciais = [row[:] for row in pesos_oculta]
        self.bias_oculta_iniciais = list(bias_oculta)
        self.pesos_saida_iniciais = [row[:] for row in pesos_saida]
        self.bias_saida_iniciais = list(bias_saida)
        self._imagens_ref = []

        self.rede = RedeFeedforward(
            n_entradas=2, n_ocultos=len(bias_oculta), n_saidas=len(bias_saida),
            pesos_oculta=[row[:] for row in pesos_oculta], bias_oculta=list(bias_oculta),
            pesos_saida=[row[:] for row in pesos_saida], bias_saida=list(bias_saida),
        )

        self.previsoes_antes = [self.rede.prever(x)[0] for x, _ in self.padroes]

        # Processa a epoca inteira (1 passagem online pelos 4 padroes) e
        # guarda o resultado de cada passo, na ordem em que ocorreram.
        self.resultados = [self.rede.passo_treinamento(x, t, self.taxa)
                            for x, t in self.padroes]

        self.previsoes_depois = [self.rede.prever(x)[0] for x, _ in self.padroes]
        self.erro_medio_epoca = sum(r['erro_total'] for r in self.resultados) / len(self.resultados)

        self._construir()

    # ------------------------------------------------------------------
    def _construir(self):
        tk.Frame(self, bg=T.ACCENT, height=3).pack(fill='x', side='top')

        head = tk.Frame(self, bg=T.BG, height=90)
        head.pack(fill='x', side='top')
        head.pack_propagate(False)
        tk.Label(head, text='MEMORIA DE CALCULO  ·  EXERCICIO B (XOR)',
                 bg=T.BG, fg=T.ACCENT_DEEP, font=T.FONT_KICKER
                ).pack(anchor='w', padx=24, pady=(16, 0))
        tk.Label(head, text='XOR com MLP (arquitetura Fig. 12.28b)  ·  1 Epoca  ·  Aula PR_711',
                 bg=T.BG, fg=T.FG, font=T.FONT_TITLE
                ).pack(anchor='w', padx=24)
        tk.Label(head,
                 text=(f'Arquitetura: 2 entradas -> 2 ocultos -> 1 saida   |   '
                       f'4 padroes (modo online)   |   eta = {self.taxa}'),
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

        self._secao_arquitetura(wrap)
        self._secao_formulas(wrap)
        self._secao_epoca(wrap)
        self._secao_resultado_final(wrap)
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

    def _add_formula(self, parent, latex, fontsize=16, bg=T.BG_CARD, pady=(6, 6)):
        photo = self._formula(latex, fontsize=fontsize, bg=bg)
        tk.Label(parent, image=photo, bg=bg).pack(anchor='w', padx=18, pady=pady)

    def _add_step(self, parent, texto):
        tk.Label(parent, text=texto, bg=T.BG_CARD, fg=T.FG,
                 font=T.FONT_MONO_SM, anchor='w', justify='left'
                ).pack(anchor='w', padx=18, pady=1)

    def _add_subkicker(self, parent, texto):
        tk.Label(parent, text=texto.upper(), bg=T.BG_CARD, fg=T.ACCENT_DEEP,
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
                 font=T.FONT_REF, anchor='w'
                ).pack(anchor='w', padx=18, pady=(0, 4))

    def _respiro(self, parent):
        tk.Frame(parent, bg=T.BG_CARD, height=10).pack()

    # ==================================================================
    # SECAO 1 — Arquitetura e Pesos Iniciais
    # ==================================================================
    def _secao_arquitetura(self, parent):
        card = Card(parent, titulo='1. Arquitetura e Pesos Iniciais  ·  Fig. 12.28(b)')
        card.pack(fill='x', padx=22, pady=(20, 0))

        self._add_explain(card,
            'A Figura 12.28(b) do slide mostra a topologia minima que resolve o '
            'XOR (2 entradas -> 2 ocultos -> 1 saida), com pesos rotulados '
            'genericamente w1...w9 — sem valores numericos dados. Os pesos '
            'abaixo foram escolhidos para esta demonstracao (valores pequenos, '
            'tipicos de uma inicializacao aleatoria).')

        self._add_subkicker(card, 'tabela-verdade do xor (4 padroes de treino)')
        for x, t in self.padroes:
            self._add_step(card, f'  ({x[0]:.0f}, {x[1]:.0f})  ->  {t[0]:.0f}')

        self._add_subkicker(card, 'pesos iniciais')
        self._add_step(card, f'  h1: w = {self.pesos_oculta_iniciais[0]}   bias = {self.bias_oculta_iniciais[0]:+.2f}')
        self._add_step(card, f'  h2: w = {self.pesos_oculta_iniciais[1]}   bias = {self.bias_oculta_iniciais[1]:+.2f}')
        self._add_step(card, f'  saida: w = {self.pesos_saida_iniciais[0]}   bias = {self.bias_saida_iniciais[0]:+.2f}')

        self._add_subkicker(card, 'diagrama da rede')
        buf = _desenhar_arquitetura_rede(
            ['x1', 'x2'], ['h1', 'h2'], ['saida'],
            self.pesos_oculta_iniciais, self.bias_oculta_iniciais,
            self.pesos_saida_iniciais, self.bias_saida_iniciais,
            bias_compartilhado=False, bg=T.BG_CARD)
        photo = ImageTk.PhotoImage(Image.open(buf))
        self._imagens_ref.append(photo)
        tk.Label(card, image=photo, bg=T.BG_CARD).pack(padx=18, pady=(4, 8))

        legenda = tk.Frame(card, bg=T.BG_CARD)
        legenda.pack(fill='x', padx=18, pady=(0, 10))

        def _item(cor_fundo, cor_borda, texto):
            linha = tk.Frame(legenda, bg=T.BG_CARD)
            linha.pack(fill='x', pady=1)
            tk.Frame(linha, bg=cor_fundo, width=14, height=14,
                     highlightthickness=1.5, highlightbackground=cor_borda
                    ).pack(side='left', padx=(0, 8), pady=2)
            tk.Label(linha, text=texto, bg=T.BG_CARD, fg=T.FG_MUTED,
                     font=T.FONT_LABEL, anchor='w', justify='left',
                     wraplength=880
                    ).pack(side='left', fill='x')

        _item(T.BG_PANEL, T.BORDER_HARD,
              'Circulo cinza = neuronio de ENTRADA (x1, x2 — valores booleanos 0 ou 1).')
        _item(T.ACCENT_SOFT, T.ACCENT_DEEP,
              'Circulo ambar = neuronio da camada OCULTA (h1, h2 — aplica a sigmoide).')
        _item('#DBEAFE', T.DATA_BLUE,
              'Circulo azul = neuronio de SAIDA (aplica a sigmoide, saida final da rede).')
        _item(T.BG_CARD, T.BORDER_HARD,
              'Linhas cinzas = conexoes entre neuronios, rotuladas com o valor do peso correspondente.')
        _item(T.BG_CARD, T.BG_CARD,
              'Rotulo "b=valor" abaixo de cada neuronio = bias individual daquele neuronio.')
        self._respiro(card)

    # ==================================================================
    # SECAO 2 — Formulas (recapitulacao: net, sigmoide, deltas, atualizacao)
    # ==================================================================
    def _secao_formulas(self, parent):
        card = Card(parent, titulo='2. Formulas Aplicadas a Cada Padrao')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'As mesmas formulas de alimentacao adiante e retropropagacao do '
            'item (i) sao aplicadas aqui, uma vez para cada um dos 4 padroes '
            '(modo online: os pesos sao atualizados apos cada padrao, antes '
            'de processar o proximo).')
        self._add_formula(card,
            r'$z = \sum_j w_j\, a_j + b \qquad a = \sigma(z) = \dfrac{1}{1+e^{-z}}$',
            fontsize=16)
        self._add_formula(card,
            r'$\delta_o = (z_o - t_o)\, z_o(1-z_o) \qquad '
            r'\delta_h = \left(\sum_o \delta_o w_{ho}\right) \text{out}_h(1-\text{out}_h)$',
            fontsize=15)
        self._add_formula(card,
            r'$w_{\text{novo}} = w - \eta \cdot \delta \cdot \text{entrada}$',
            fontsize=16)
        self._add_ref(card, RedeFeedforward.passo_treinamento)
        self._respiro(card)

    # ==================================================================
    # SECAO 3 — A Epoca: os 4 padroes processados em sequencia
    # ==================================================================
    def _secao_epoca(self, parent):
        card = Card(parent, titulo='3. A Epoca  ·  4 Padroes Processados em Sequencia')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'Cada padrao e apresentado uma vez, nesta ordem. O erro de cada '
            'padrao e calculado ANTES da atualizacao de pesos daquele padrao '
            '(mesma convencao usada no Perceptron e na Regra Delta deste '
            'projeto) — por isso os pesos "antes" de um padrao sao os '
            '"depois" do padrao anterior.')

        for i, ((x, t), r) in enumerate(zip(self.padroes, self.resultados)):
            self._add_subkicker(card, f'padrao {i + 1}  ·  x=({x[0]:.0f}, {x[1]:.0f})  alvo={t[0]:.0f}')
            self._add_step(card,
                f'  out_h1={r["saida_oculta"][0]:.4f}   out_h2={r["saida_oculta"][1]:.4f}   '
                f'out={r["saida_rede"][0]:.4f}   erro={r["erro_total"]:.5f}')
            self._add_step(card,
                f'  delta_saida={r["delta_saida"][0]:+.6f}   '
                f'delta_h1={r["delta_oculta"][0]:+.6f}   delta_h2={r["delta_oculta"][1]:+.6f}')
            self._add_step(card,
                f'  pesos oculta -> {[[round(v, 4) for v in row] for row in r["w_oculta_depois"]]}   '
                f'bias -> {[round(v, 4) for v in r["b_oculta_depois"]]}')
            self._add_step(card,
                f'  pesos saida  -> {[round(v, 4) for v in r["w_saida_depois"][0]]}   '
                f'bias -> {r["b_saida_depois"][0]:.4f}')

        self._add_resultado(card,
            f'  Erro medio da epoca = {self.erro_medio_epoca:.5f}', cor=T.ACCENT_DEEP)

    # ==================================================================
    # SECAO 4 — Resultado Final: previsoes antes/depois da epoca
    # ==================================================================
    def _secao_resultado_final(self, parent):
        card = Card(parent, titulo='4. Resultado Final  ·  Previsoes Antes e Depois da Epoca')
        card.pack(fill='x', padx=22, pady=(22, 0))

        self._add_explain(card,
            'Recalculando a saida da rede para os 4 padroes com os pesos ja '
            'atualizados apos a epoca completa:')

        acertos = 0
        for (x, t), antes, depois in zip(self.padroes, self.previsoes_antes, self.previsoes_depois):
            classe = 1 if depois >= 0.5 else 0
            certo = classe == int(t[0])
            acertos += int(certo)
            marcador = 'OK' if certo else 'ainda errado'
            self._add_step(card,
                f'  x=({x[0]:.0f},{x[1]:.0f})  alvo={t[0]:.0f}   '
                f'antes={antes:.4f}  ->  depois={depois:.4f}   ({marcador})')

        cor = T.SUCCESS if acertos == 4 else (T.ACCENT_DEEP if acertos > 0 else T.DANGER)
        self._add_resultado(card,
            f'  {acertos}/4 padroes corretos apos 1 epoca', cor=cor)
        self._add_explain(card,
            'O XOR nao e linearmente separavel: as saidas permanecem proximas '
            'de 0.5 (regiao de maxima incerteza da sigmoide) apos apenas 1 '
            'epoca. Sao necessarias muitas epocas de gradiente descendente '
            'para a rede efetivamente separar os 4 padroes — isso confirma, '
            'na pratica, por que uma camada oculta nao-linear e indispensavel '
            'para o XOR (o mesmo limite ja demonstrado com a Regra Delta '
            'linear na Aba 2, onde o MSE estaciona em 0.25 sem nunca zerar).',
            pady=(8, 0))

