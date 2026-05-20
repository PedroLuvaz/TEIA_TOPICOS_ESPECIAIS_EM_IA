"""
Interface grafica interativa — Classificador de Distancia Minima.

Aba 'Dataset':  tabela matricial com todas as 150 amostras (linhas x colunas).
Aba 'Grafico':  dispersao interativa — ao passar o mouse sobre um ponto o tooltip
                exibe a qual linha do dataset ele corresponde, e a linha e
                destacada automaticamente na tabela.
"""

import tkinter as tk
from tkinter import ttk
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Mesmas cores/marcadores usados em visualizer.py
CORES_CLASSE = {
    'setosa':     '#2196F3',
    'versicolor': '#4CAF50',
    'virginica':  '#F44336',
}
MARCADORES_CLASSE = {
    'setosa':     'o',
    'versicolor': 's',
    'virginica':  '^',
}

INDICES_PETALA = [2, 3]
RAIO_HOVER = 0.14   # unidades do grafico para ativar tooltip


class JanelaInterativa:
    def __init__(self, dados, prototipos, dados_treino, dados_teste, master=None):
        self.dados        = dados
        self.prototipos   = prototipos
        self.dados_treino = dados_treino
        self.dados_teste  = dados_teste
        self._pontos      = []   # (x, y, dado_dict, split_str)

        if master is not None:
            # Embutido num Toplevel fornecido pelo caller
            self.root = master
            self.root.configure(bg='#FAFAFA')
        else:
            # Janela propria (uso standalone)
            self.root = tk.Tk()
            self.root.title("TEIA — Iris Dataset Interativo")
            self.root.geometry("1150x700")
            self.root.configure(bg='#FAFAFA')
            self.root.resizable(True, True)

        self._construir_interface()

    # ---------------------------------------------------------------------- UI

    def _construir_interface(self):
        nb = ttk.Notebook(self.root)
        nb.pack(fill='both', expand=True, padx=8, pady=8)

        aba_tab = ttk.Frame(nb)
        nb.add(aba_tab, text='  Dataset (Matriz)  ')
        self._construir_aba_tabela(aba_tab)

        aba_graf = ttk.Frame(nb)
        nb.add(aba_graf, text='  Grafico Interativo  ')
        self._construir_aba_grafico(aba_graf)

    # ------------------------------------------------------------------ tabela

    def _construir_aba_tabela(self, frame):
        # cabecalho informativo
        ttk.Label(
            frame,
            text=f'Iris Dataset  —  {len(self.dados)} amostras  |  4 atributos  |  3 classes',
            font=('Segoe UI', 11, 'bold'),
        ).pack(pady=(8, 3))

        # legenda de cores
        leg = ttk.Frame(frame)
        leg.pack(pady=(0, 5))
        for cls, bg in [('setosa', '#BBDEFB'), ('versicolor', '#C8E6C9'), ('virginica', '#FFCDD2')]:
            tk.Frame(leg, bg=bg, width=14, height=14, bd=1, relief='solid').pack(side='left', padx=(10, 3))
            ttk.Label(leg, text=cls).pack(side='left', padx=(0, 10))

        # frame da treeview com scrollbar
        frame_tv = ttk.Frame(frame)
        frame_tv.pack(fill='both', expand=True, padx=8, pady=(0, 8))

        sv = ttk.Scrollbar(frame_tv, orient='vertical')
        sv.pack(side='right', fill='y')

        colunas = ('num', 'cs', 'ls', 'cp', 'lp', 'classe')
        self.treeview = ttk.Treeview(
            frame_tv, columns=colunas, show='headings',
            yscrollcommand=sv.set, selectmode='browse',
        )
        sv.config(command=self.treeview.yview)

        cabecalhos = [
            ('num',    'Nº',                   48),
            ('cs',     'Comp. Sepala (cm)',    148),
            ('ls',     'Larg. Sepala (cm)',    148),
            ('cp',     'Comp. Petala (cm)',    148),
            ('lp',     'Larg. Petala (cm)',    148),
            ('classe', 'Classe',               110),
        ]
        for col, cab, larg in cabecalhos:
            self.treeview.heading(col, text=cab)
            self.treeview.column(col, width=larg, anchor='center', minwidth=40)

        # cores de fundo por classe
        self.treeview.tag_configure('setosa',     background='#E3F2FD')
        self.treeview.tag_configure('versicolor', background='#E8F5E9')
        self.treeview.tag_configure('virginica',  background='#FFEBEE')
        # destaque amarelo quando o mouse passa sobre o ponto correspondente
        self.treeview.tag_configure('hover', background='#FFF59D', font=('Segoe UI', 9, 'bold'))

        for d in self.dados:
            idx = d.get('indice', 0)
            a = d['atributos']
            self.treeview.insert('', 'end', iid=str(idx), values=(
                idx + 1,
                f'{a[0]:.1f}', f'{a[1]:.1f}', f'{a[2]:.1f}', f'{a[3]:.1f}',
                d['classe'],
            ), tags=(d['classe'],))

        self.treeview.pack(fill='both', expand=True)
        self._ultimo_destaque = None

    # ----------------------------------------------------------------- grafico

    def _construir_aba_grafico(self, frame):
        # barra de informacao no topo
        barra = ttk.Frame(frame)
        barra.pack(fill='x', padx=8, pady=(6, 0))

        ttk.Label(barra,
                  text='Passe o mouse sobre um ponto para ver os dados:',
                  font=('Segoe UI', 10)).pack(side='left')

        self.label_hover = ttk.Label(
            barra, text='—',
            font=('Segoe UI', 10, 'bold'), foreground='#1565C0',
        )
        self.label_hover.pack(side='left', padx=14)

        # area do matplotlib
        frame_fig = ttk.Frame(frame)
        frame_fig.pack(fill='both', expand=True, padx=8, pady=(4, 8))

        fig = Figure(figsize=(10, 5.4), dpi=100)
        self._ax  = fig.add_subplot(111)
        self._fig = fig

        self._plotar_pontos()

        # anotacao de tooltip (invisivel por padrao)
        self._annot = self._ax.annotate(
            '', xy=(0, 0), xytext=(18, 18), textcoords='offset points',
            bbox=dict(boxstyle='round,pad=0.45', fc='#FFFDE7', ec='#9E9E9E', alpha=0.96),
            arrowprops=dict(arrowstyle='->', color='#555555', lw=1.2),
            fontsize=8.8, visible=False, zorder=10,
        )

        self._canvas = FigureCanvasTkAgg(fig, master=frame_fig)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill='both', expand=True)

        toolbar = NavigationToolbar2Tk(self._canvas, frame_fig)
        toolbar.update()

        self._canvas.mpl_connect('motion_notify_event', self._on_hover)

    def _plotar_pontos(self):
        ax = self._ax
        ids_treino = set(id(d) for d in self.dados_treino)
        ids_teste  = set(id(d) for d in self.dados_teste)

        ordem = ['virginica', 'versicolor', 'setosa']
        classes = set(d['classe'] for d in self.dados)
        for classe in [c for c in ordem if c in classes]:
            cor = CORES_CLASSE[classe]
            mrc = MARCADORES_CLASSE[classe]

            grupo_treino = [d for d in self.dados if d['classe'] == classe and id(d) in ids_treino]
            if grupo_treino:
                xs = [d['atributos'][INDICES_PETALA[0]] for d in grupo_treino]
                ys = [d['atributos'][INDICES_PETALA[1]] for d in grupo_treino]
                ax.scatter(xs, ys, color=cor, marker=mrc, label=f'{classe} (treino)',
                           edgecolors='white', linewidths=0.6, s=62, alpha=0.85)
                for d, x, y in zip(grupo_treino, xs, ys):
                    self._pontos.append((x, y, d, 'treino'))

            grupo_teste = [d for d in self.dados if d['classe'] == classe and id(d) in ids_teste]
            if grupo_teste:
                xs = [d['atributos'][INDICES_PETALA[0]] for d in grupo_teste]
                ys = [d['atributos'][INDICES_PETALA[1]] for d in grupo_teste]
                ax.scatter(xs, ys, color=cor, marker=mrc, label=f'{classe} (teste)',
                           edgecolors='black', linewidths=1.0, s=82, alpha=0.95)
                for d, x, y in zip(grupo_teste, xs, ys):
                    self._pontos.append((x, y, d, 'teste'))

        if self.prototipos:
            for classe, p in self.prototipos.items():
                cor = CORES_CLASSE.get(classe, 'black')
                ax.scatter(p[0], p[1], color=cor, marker='X',
                           s=230, edgecolors='black', linewidths=1.5,
                           zorder=6, label=f'Media {classe}')
                ax.text(p[0] + 0.05, p[1] + 0.04, classe,
                        fontweight='bold', fontsize=9, color='black')

        ax.set_xlabel('Comp. Petala (cm)', fontsize=10)
        ax.set_ylabel('Larg. Petala (cm)', fontsize=10)
        ax.set_title('Iris Dataset — Distribuicao das Classes  (interativo)', fontsize=11)
        ax.legend(loc='best', fontsize=8.5, framealpha=0.92)
        ax.grid(True, linestyle=':', alpha=0.5)
        self._fig.tight_layout()

    # ------------------------------------------------------- eventos de hover

    def _on_hover(self, event):
        if event.inaxes != self._ax or event.xdata is None:
            self._esconder_tooltip()
            return

        melhor, dist_min = None, float('inf')
        for x, y, dado, split in self._pontos:
            dist = ((event.xdata - x) ** 2 + (event.ydata - y) ** 2) ** 0.5
            if dist < dist_min:
                dist_min = dist
                melhor = (x, y, dado, split)

        if dist_min < RAIO_HOVER and melhor:
            x, y, dado, split = melhor
            idx    = dado.get('indice', '?')
            a      = dado['atributos']
            classe = dado['classe']

            texto = (
                f'Linha {idx + 1}  ({split})\n'
                f'Classe:       {classe}\n'
                f'Comp. Sepala: {a[0]:.1f} cm\n'
                f'Larg. Sepala: {a[1]:.1f} cm\n'
                f'Comp. Petala: {a[2]:.1f} cm\n'
                f'Larg. Petala: {a[3]:.1f} cm'
            )
            self._annot.set_text(texto)
            self._annot.xy = (x, y)
            self._annot.set_visible(True)
            self._canvas.draw_idle()

            self.label_hover.config(
                text=(f'Linha {idx + 1}  |  {classe} ({split})  |  '
                      f'Petala: {a[2]:.1f} x {a[3]:.1f} cm  |  '
                      f'Sepala: {a[0]:.1f} x {a[1]:.1f} cm')
            )
            self._destacar_tabela(idx)
        else:
            self._esconder_tooltip()

    def _esconder_tooltip(self):
        if self._annot.get_visible():
            self._annot.set_visible(False)
            self._canvas.draw_idle()
        self.label_hover.config(text='—')
        self._limpar_destaque_tabela()

    def _destacar_tabela(self, indice):
        iid = str(indice)
        if not self.treeview.exists(iid):
            return
        # remove destaque anterior se for linha diferente
        if self._ultimo_destaque is not None and self._ultimo_destaque != iid:
            antigo = self._ultimo_destaque
            if self.treeview.exists(antigo):
                classe_antiga = self.treeview.item(antigo, 'values')[5]   # coluna 'classe'
                self.treeview.item(antigo, tags=(classe_antiga,))
        # aplica destaque na nova linha
        self.treeview.item(iid, tags=('hover',))
        self.treeview.selection_set(iid)
        self.treeview.see(iid)
        self._ultimo_destaque = iid

    def _limpar_destaque_tabela(self):
        if self._ultimo_destaque is not None:
            iid = self._ultimo_destaque
            if self.treeview.exists(iid):
                classe = self.treeview.item(iid, 'values')[5]
                self.treeview.item(iid, tags=(classe,))
            self._ultimo_destaque = None

    # ------------------------------------------------------------------- main

    def executar(self):
        """Inicia o mainloop (uso standalone sem master externo)."""
        self.root.mainloop()


def abrir_interface(dados, prototipos, dados_treino, dados_teste):
    """Abre a janela interativa standalone. Bloqueia ate o usuario fechar."""
    app = JanelaInterativa(dados, prototipos, dados_treino, dados_teste)
    app.executar()
