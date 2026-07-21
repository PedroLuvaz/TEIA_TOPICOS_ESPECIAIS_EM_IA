"""
Aba 5.0 do Lab 5 — XOR com MLP (slides 36-37, Aula PR_711)
=============================================================
Exercicio do slide 36: "Resolva o problema XOR utilizando uma MLP de acordo
com a arquitetura da rede fig 12.28(b) ... Exercicte com uma epoca apenas."
A Fig. 12.28(b) mostra a topologia MINIMA que resolve o XOR (2 entradas ->
2 ocultos -> 1 saida, pesos rotulados genericamente w1..w9, sem valores
numericos no slide) — os pesos iniciais desta aba foram escolhidos pelo
grupo para a demonstracao.

O slide 37 ("Exemplo didatico: treinando uma rede de 3 camadas") NAO resolve
o XOR — e um exemplo generico e completo (rede 2-2-2, entradas i1=0.05 /
i2=0.10) de como a conta do backprop e feita passo a passo, incluindo a
2a iteracao e a curva de convergencia (slides 38-43). Ele e reproduzido aqui
como demonstracao do algoritmo, servindo de base antes de aplica-lo ao XOR.

Particularidade deste exemplo (diferente do restante do laboratorio): o
bias b1/b2 e UNICO por camada, compartilhado por todos os neuronios dela —
por isso o gradiente do bias soma os deltas de todos os neuronios da camada
(ver JanelaMemoriaCalculoMLP com bias_compartilhado=True).

Esta aba antecede a aba "Feedforward (MLP)" (itens i/ii do enunciado, que
usam o Iris) e por isso e chamada de "Lab 5.0" nesta entrega — o XOR tem
arquitetura e visualizacao proprias (fronteira de decisao 2D interativa +
curva de convergencia), distintas do restante do laboratorio.
"""
import os
import sys
import tkinter as tk
from tkinter import ttk

import matplotlib
matplotlib.use('TkAgg')
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
                                               NavigationToolbar2Tk)
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap

PROJETO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))))
IRIS_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (PROJETO_ROOT, IRIS_ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from models.mlp_backprop import RedeFeedforward

from . import theme as T
from .widgets import Card, MetricBlock, separador
from .janela_calculos import JanelaMemoriaCalculoMLP, JanelaMemoriaCalculoXOR

# ---------------------------------------------------------------------------
# Exemplo didatico (slide 37): rede 2-2-2 generica i1/i2, bias compartilhado
# por camada (b1 alimenta h1 e h2; b2 alimenta o1 e o2).
# ---------------------------------------------------------------------------
EX_ENTRADAS = [0.05, 0.10]
EX_ALVO = [0.01, 0.99]
EX_TAXA_APRENDIZADO = 0.5
EX_PESOS_OCULTA = [[0.15, 0.20], [0.25, 0.30]]   # h1: w1,w2  ·  h2: w3,w4
EX_BIAS_OCULTA = [0.35, 0.35]                     # b1 (compartilhado)
EX_PESOS_SAIDA = [[0.40, 0.45], [0.50, 0.55]]     # o1: w5,w6  ·  o2: w7,w8
EX_BIAS_SAIDA = [0.60, 0.60]                      # b2 (compartilhado)

# ---------------------------------------------------------------------------
# Exercicio XOR (slide 36): arquitetura Fig. 12.28(b), pesos escolhidos pelo
# grupo (o slide so da a topologia, sem valores numericos), 1 epoca.
# ---------------------------------------------------------------------------
XOR_PADROES = [
    ([0.0, 0.0], [0.0]),
    ([0.0, 1.0], [1.0]),
    ([1.0, 0.0], [1.0]),
    ([1.0, 1.0], [0.0]),
]
XOR_TAXA_APRENDIZADO = 0.5
XOR_PESOS_OCULTA_INICIAIS = [[0.50, 0.50], [-0.50, -0.50]]
XOR_BIAS_OCULTA_INICIAIS = [-0.20, 0.30]
XOR_PESOS_SAIDA_INICIAIS = [[0.60, -0.60]]
XOR_BIAS_SAIDA_INICIAIS = [-0.10]

_CMAP_XOR = LinearSegmentedColormap.from_list(
    'xor_saida', [T.DATA_BLUE, '#FFFFFF', T.DATA_CORAL])


# ===========================================================================
class TabXOR(tk.Frame):
    def __init__(self, master, **kw):
        super().__init__(master, bg=T.BG, **kw)

        self.epoca_atual = 0
        self.historico_erro = []
        self._nova_rede_xor()

        self._construir_layout()
        self._atualizar_tudo()

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

        card_info = Card(wrap, titulo='sobre esta aba  ·  lab 5.0')
        card_info.grid(row=0, column=0, sticky='ew')
        tk.Label(card_info,
                 text='Exercicio do slide 36: resolver o XOR com uma MLP '
                      '(arquitetura minima da Fig. 12.28b), treinada por '
                      'apenas 1 epoca.\n\n'
                      'O slide 37 nao resolve o XOR — e um exemplo didatico '
                      'generico e completo, reproduzido aqui para demonstrar '
                      'o algoritmo passo a passo antes de aplica-lo ao XOR.\n\n'
                      'Os itens (i) e (ii) do enunciado (Iris) estao na '
                      'proxima aba, "Lab 5.1".',
                 bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_BODY,
                 justify='left', anchor='w', wraplength=260
                ).pack(fill='x', padx=T.CARD_PADX, pady=(0, 10))

        card_ex = Card(wrap, titulo='exemplo didatico  ·  slide 37')
        card_ex.grid(row=1, column=0, sticky='ew', pady=(T.GAP_SM, 0))
        tk.Label(card_ex,
                 text=f'Rede 2-2-2 generica  ·  i1={EX_ENTRADAS[0]}  i2={EX_ENTRADAS[1]}\n'
                      f'Alvo: o1={EX_ALVO[0]}  o2={EX_ALVO[1]}\n'
                      f'Taxa de aprendizagem: eta={EX_TAXA_APRENDIZADO}\n'
                      f'Bias unico por camada (b1, b2 compartilhados)',
                 bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_MONO_SM,
                 justify='left', anchor='w'
                ).pack(fill='x', padx=T.CARD_PADX, pady=(0, 6))
        ttk.Button(card_ex, text='Abrir memoria de calculo  >',
                   style='Primary.TButton',
                   command=self._abrir_memoria_exemplo
                  ).pack(fill='x', padx=T.CARD_PADX, pady=(0, 10))

        card_xor = Card(wrap, titulo='exercicio xor  ·  slide 36')
        card_xor.grid(row=2, column=0, sticky='ew', pady=(T.GAP_SM, 0))
        tk.Label(card_xor,
                 text='Arquitetura Fig. 12.28(b): 2 entradas -> 2 ocultos -> '
                      '1 saida.\nPesos iniciais escolhidos pelo grupo (o '
                      'slide so da a topologia).\n'
                      f'Taxa de aprendizagem: eta={XOR_TAXA_APRENDIZADO}',
                 bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_MONO_SM,
                 justify='left', anchor='w'
                ).pack(fill='x', padx=T.CARD_PADX, pady=(0, 6))
        ttk.Button(card_xor, text='Abrir memoria de calculo (1 epoca)  >',
                   style='Primary.TButton',
                   command=self._abrir_memoria_xor
                  ).pack(fill='x', padx=T.CARD_PADX, pady=(0, 10))

        separador(card_xor)
        tk.Label(card_xor, text='TREINAMENTO INTERATIVO', bg=T.BG_CARD,
                 fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w'
                ).pack(fill='x', padx=T.CARD_PADX, pady=(10, 4))
        tk.Label(card_xor,
                 text='A fronteira de decisao ao lado mostra a saida da '
                      'rede em tempo real. O exercicio pede so 1 epoca — '
                      'use "+epocas" para ver o XOR sendo resolvido aos '
                      'poucos (com estes pesos, o erro fica quase parado '
                      'ate ~epoca 500 e so converge de fato perto de '
                      '2000-5000 — algo que a Regra Delta linear, na '
                      'Aba 2, nunca consegue fazer).',
                 bg=T.BG_CARD, fg=T.FG_MUTED, font=T.FONT_BODY_SM,
                 justify='left', anchor='w', wraplength=260
                ).pack(fill='x', padx=T.CARD_PADX, pady=(0, 8))
        ttk.Button(card_xor, text='Rodar exatamente 1 epoca (exercicio)',
                   command=self._rodar_1_epoca_exercicio
                  ).pack(fill='x', padx=T.CARD_PADX, pady=(0, 4))
        linha_extra = tk.Frame(card_xor, bg=T.BG_CARD)
        linha_extra.pack(fill='x', padx=T.CARD_PADX, pady=(0, 4))
        ttk.Button(linha_extra, text='+500 epocas',
                   command=lambda: self._treinar_mais(500)
                  ).pack(side='left', expand=True, fill='x', padx=(0, 4))
        ttk.Button(linha_extra, text='+2000 epocas',
                   command=lambda: self._treinar_mais(2000)
                  ).pack(side='left', expand=True, fill='x', padx=(4, 0))
        ttk.Button(card_xor, text='Reiniciar rede',
                   command=self._reiniciar_rede
                  ).pack(fill='x', padx=T.CARD_PADX, pady=(0, 10))

        wrap.rowconfigure(3, weight=1)

    def _coluna_resultados(self):
        wrap = tk.Frame(self, bg=T.BG)
        wrap.grid(row=0, column=1, sticky='nsew', padx=(T.GAP, T.PAD_PAGE), pady=T.PAD_PAGE)
        wrap.columnconfigure(0, weight=1)
        wrap.rowconfigure(0, weight=0)
        wrap.rowconfigure(1, weight=1)
        wrap.rowconfigure(2, weight=0)

        faixa = tk.Frame(wrap, bg=T.BG)
        faixa.grid(row=0, column=0, sticky='ew')
        faixa.columnconfigure(list(range(3)), weight=1)
        self.mb_epoca = MetricBlock(faixa, 'Epoca atual', '0')
        self.mb_erro = MetricBlock(faixa, 'Erro medio (epoca)', '—')
        self.mb_acertos = MetricBlock(faixa, 'Acertos (4 padroes)', '—')
        for i, mb in enumerate([self.mb_epoca, self.mb_erro, self.mb_acertos]):
            mb.grid(row=0, column=i, sticky='ew', padx=(0 if i == 0 else T.GAP_SM, 0))

        painel = tk.Frame(wrap, bg=T.BG_PANEL,
                          highlightthickness=1,
                          highlightbackground=T.BORDER,
                          highlightcolor=T.BORDER)
        painel.grid(row=1, column=0, sticky='nsew', pady=(10, 0))
        painel.columnconfigure(0, weight=1)
        painel.rowconfigure(0, weight=1)

        self.figura = Figure(figsize=(9.5, 4.0), dpi=100, facecolor=T.BG_CARD)
        gs = self.figura.add_gridspec(
            1, 2, width_ratios=[5, 3],
            left=0.08, right=0.97, bottom=0.14, top=0.90, wspace=0.32,
        )
        self.ax_sc = self.figura.add_subplot(gs[0])
        self.ax_cv = self.figura.add_subplot(gs[1])

        self.canvas = FigureCanvasTkAgg(self.figura, master=painel)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky='nsew',
                                         padx=8, pady=(8, 2))

        self.toolbar = NavigationToolbar2Tk(self.canvas, painel, pack_toolbar=False)
        self.toolbar.update()
        self._estilizar_toolbar(self.toolbar)
        self.toolbar.grid(row=1, column=0, sticky='ew', padx=6, pady=(0, 6))

        self._tabela_holder = tk.Frame(wrap, bg=T.BG)
        self._tabela_holder.grid(row=2, column=0, sticky='ew', pady=(10, 0))

    # ------------------------------------------------------------------
    # Rede XOR — estado interativo
    # ------------------------------------------------------------------
    def _nova_rede_xor(self):
        self.rede_xor = RedeFeedforward(
            n_entradas=2, n_ocultos=2, n_saidas=1,
            pesos_oculta=[row[:] for row in XOR_PESOS_OCULTA_INICIAIS],
            bias_oculta=XOR_BIAS_OCULTA_INICIAIS[:],
            pesos_saida=[row[:] for row in XOR_PESOS_SAIDA_INICIAIS],
            bias_saida=XOR_BIAS_SAIDA_INICIAIS[:],
        )
        self.epoca_atual = 0
        self.historico_erro = []

    def _reiniciar_rede(self):
        self._nova_rede_xor()
        self._atualizar_tudo()

    def _rodar_1_epoca_exercicio(self):
        self._nova_rede_xor()
        self._treinar_n_epocas(1)

    def _treinar_mais(self, n):
        self._treinar_n_epocas(n)

    def _treinar_n_epocas(self, n):
        for _ in range(n):
            soma_erro = 0.0
            for x, t in XOR_PADROES:
                r = self.rede_xor.passo_treinamento(x, t, XOR_TAXA_APRENDIZADO)
                soma_erro += r['erro_total']
            self.epoca_atual += 1
            self.historico_erro.append(soma_erro / len(XOR_PADROES))
        self._atualizar_tudo()

    # ------------------------------------------------------------------
    # Redesenho
    # ------------------------------------------------------------------
    def _atualizar_tudo(self):
        self._desenhar_fronteira()
        self._desenhar_convergencia()
        self.canvas.draw()
        self._atualizar_metricas()
        self._atualizar_tabela()

    def _calcular_grade_saida(self, resolucao=70, margem=0.35):
        lo, hi = -margem, 1.0 + margem
        passo = (hi - lo) / (resolucao - 1)
        eixo = [lo + k * passo for k in range(resolucao)]
        X = [[x for x in eixo] for _ in eixo]
        Y = [[y for _ in eixo] for y in eixo]
        Z = [[self.rede_xor.forward([x, y])[1][0] for x in eixo] for y in eixo]
        return X, Y, Z

    def _desenhar_fronteira(self):
        ax = self.ax_sc
        ax.cla()
        self._estilizar_ax(ax)

        X, Y, Z = self._calcular_grade_saida()
        ax.contourf(X, Y, Z, levels=20, cmap=_CMAP_XOR, vmin=0, vmax=1,
                   alpha=0.85, zorder=1)
        ax.contour(X, Y, Z, levels=[0.5], colors=[T.FG], linestyles='--',
                  linewidths=1.6, zorder=2)

        cores = {0: T.DATA_BLUE, 1: T.DATA_CORAL}
        marcadores = {0: 'o', 1: '^'}
        for x, t in XOR_PADROES:
            classe = int(t[0])
            saida = self.rede_xor.prever(x)[0]
            pred = 1 if saida >= 0.5 else 0
            acertou = pred == classe
            ax.scatter(x[0], x[1], color=cores[classe], marker=marcadores[classe],
                      s=170, zorder=5, linewidths=1.8,
                      edgecolors=(T.SUCCESS if acertou else T.DANGER))
            ax.annotate(f'({int(x[0])},{int(x[1])})  alvo={classe}\nout={saida:.3f}',
                       (x[0], x[1]), textcoords='offset points', xytext=(9, 8),
                       fontsize=7, color=T.FG_MUTED)

        ax.set_xlim(-0.35, 1.35)
        ax.set_ylim(-0.35, 1.35)
        ax.set_xlabel('x₁', fontsize=9)
        ax.set_ylabel('x₂', fontsize=9)
        ax.set_title(f'Saida da rede (fundo)  ·  epoca {self.epoca_atual}',
                    fontsize=9, pad=6)

    def _desenhar_convergencia(self):
        ax = self.ax_cv
        ax.cla()
        self._estilizar_ax(ax)
        ax.set_title('Convergencia', fontsize=9, pad=6)

        if not self.historico_erro:
            ax.text(0.5, 0.5, 'Treine a rede\npara ver a curva',
                   transform=ax.transAxes, ha='center', va='center',
                   fontsize=8, color=T.FG_DIM)
            return

        epocas = list(range(1, len(self.historico_erro) + 1))
        ax.plot(epocas, self.historico_erro, color=T.ACCENT, lw=1.6, zorder=3)
        if len(epocas) > 30:
            ax.set_xscale('log')
        ax.set_ylim(bottom=0)
        ax.set_xlabel('epoca', fontsize=8)
        ax.set_ylabel('erro medio', fontsize=8)

    @staticmethod
    def _estilizar_ax(ax):
        ax.set_facecolor(T.BG_PANEL)
        ax.tick_params(colors=T.FG_MUTED, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(T.BORDER)
        ax.grid(color=T.BORDER, linewidth=0.5, alpha=0.5)
        ax.title.set_color(T.FG)
        ax.xaxis.label.set_color(T.FG_MUTED)
        ax.yaxis.label.set_color(T.FG_MUTED)

    @staticmethod
    def _estilizar_toolbar(toolbar):
        toolbar.configure(background=T.BG_PANEL)
        for child in toolbar.winfo_children():
            try:
                child.configure(background=T.BG_PANEL,
                                activebackground=T.BG_HOVER,
                                relief='flat', borderwidth=0)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Metricas + tabela de predicoes
    # ------------------------------------------------------------------
    def _atualizar_metricas(self):
        self.mb_epoca.set(str(self.epoca_atual))
        if self.historico_erro:
            self.mb_erro.set(f'{self.historico_erro[-1]:.5f}', T.ACCENT)
        else:
            self.mb_erro.set('—')

        acertos = 0
        for x, t in XOR_PADROES:
            saida = self.rede_xor.prever(x)[0]
            pred = 1 if saida >= 0.5 else 0
            if pred == int(t[0]):
                acertos += 1
        cor = T.SUCCESS if acertos == 4 else (T.ACCENT if acertos >= 2 else T.DANGER)
        self.mb_acertos.set(f'{acertos}/4', cor)

    def _atualizar_tabela(self):
        for w in self._tabela_holder.winfo_children():
            w.destroy()

        def cel(row, col, texto, cor=T.FG, bg_c=T.BG_PANEL, larg=14, bold=False):
            f = T.FONT_CELL_BOLD if bold else T.FONT_CELL
            tk.Label(self._tabela_holder, text=texto, bg=bg_c, fg=cor, font=f,
                    width=larg, anchor='center',
                    highlightthickness=1, highlightbackground=T.BORDER
                   ).grid(row=row, column=col, sticky='nsew', padx=1, pady=1)

        for j, h in enumerate(['Padrao (x1,x2)', 'Alvo', 'Saida da rede', 'Classe prevista', 'Status']):
            cel(0, j, h, cor=T.ACCENT_DEEP, bold=True)

        for i, (x, t) in enumerate(XOR_PADROES):
            saida = self.rede_xor.prever(x)[0]
            pred = 1 if saida >= 0.5 else 0
            acertou = pred == int(t[0])
            bg_r = T.BG_CARD if i % 2 == 0 else T.BG_PANEL
            cel(i + 1, 0, f'({int(x[0])}, {int(x[1])})', bg_c=bg_r)
            cel(i + 1, 1, f'{int(t[0])}', bg_c=bg_r)
            cel(i + 1, 2, f'{saida:.4f}', bg_c=bg_r)
            cel(i + 1, 3, f'{pred}', bg_c=bg_r)
            cel(i + 1, 4, 'correto' if acertou else 'incorreto',
               cor=(T.SUCCESS if acertou else T.DANGER), bg_c=bg_r)

    # ------------------------------------------------------------------
    # Memoria de Calculo (janelas LaTeX)
    # ------------------------------------------------------------------
    def _abrir_memoria_exemplo(self):
        JanelaMemoriaCalculoMLP(
            self,
            entradas=EX_ENTRADAS, alvo=EX_ALVO, taxa_aprendizado=EX_TAXA_APRENDIZADO,
            pesos_oculta=EX_PESOS_OCULTA, bias_oculta=EX_BIAS_OCULTA,
            pesos_saida=EX_PESOS_SAIDA, bias_saida=EX_BIAS_SAIDA,
            rotulos_ocultos=['h1', 'h2'],
            rotulos_saida=['o1', 'o2'],
            titulo_janela='Exemplo Didatico (slide 37)  ·  Rede 2-2-2',
            subtitulo='Rede 2-2-2 generica (i1/i2)  ·  Exemplo Didatico  ·  Aula PR_711',
            bias_compartilhado=True,
        )

    def _abrir_memoria_xor(self):
        JanelaMemoriaCalculoXOR(
            self,
            padroes=XOR_PADROES, taxa_aprendizado=XOR_TAXA_APRENDIZADO,
            pesos_oculta=XOR_PESOS_OCULTA_INICIAIS, bias_oculta=XOR_BIAS_OCULTA_INICIAIS,
            pesos_saida=XOR_PESOS_SAIDA_INICIAIS, bias_saida=XOR_BIAS_SAIDA_INICIAIS,
        )
