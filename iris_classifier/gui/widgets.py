"""
Widgets compostos reutilizaveis - cartoes, blocos de metrica, separadores
e secoes com kicker em ambar. Mantidos genericos para que todas as abas
do projeto compartilhem a mesma linguagem visual.
"""
import tkinter as tk

from . import theme as T


class Card(tk.Frame):
    """Cartao branco elevado com kicker em ambar e regua divisoria.

    Uso:
        card = Card(parent, titulo='atributos')
        ttk.Label(card, ...).pack(padx=card.padx, pady=card.pady)

    O Card e apenas um Frame com cor de fundo e um titulo opcional;
    o conteudo e adicionado como filho normal.
    """
    PADX = T.CARD_PADX
    PADY_TOP = 12
    PADY_BOTTOM = 14

    def __init__(self, master, titulo=None, **kw):
        super().__init__(master, bg=T.BG_CARD,
                         highlightthickness=1,
                         highlightbackground=T.BORDER,
                         highlightcolor=T.BORDER, **kw)
        self.padx = self.PADX
        self.pady = (4, self.PADY_BOTTOM)
        if titulo:
            tk.Label(self, text=titulo.upper(),
                     bg=T.BG_CARD, fg=T.ACCENT_DEEP,
                     font=T.FONT_KICKER, anchor='w'
                    ).pack(fill='x', padx=self.PADX, pady=(self.PADY_TOP, 4))
            tk.Frame(self, bg=T.BORDER, height=1
                    ).pack(fill='x', padx=self.PADX, pady=(0, 6))


class MetricBlock(tk.Frame):
    """Bloco de metrica (KPI): barra de acento lateral + rotulo + valor."""
    def __init__(self, master, rotulo, valor='-', cor_valor=None, **kw):
        super().__init__(master, bg=T.BG_CARD,
                         highlightthickness=1,
                         highlightbackground=T.BORDER,
                         highlightcolor=T.BORDER, **kw)
        self._barra = tk.Frame(self, bg=cor_valor or T.ACCENT, width=3)
        self._barra.pack(side='left', fill='y')
        corpo = tk.Frame(self, bg=T.BG_CARD)
        corpo.pack(side='left', fill='both', expand=True)
        tk.Label(corpo, text=rotulo.upper(),
                 bg=T.BG_CARD, fg=T.FG_DIM,
                 font=T.FONT_KICKER, anchor='w').pack(
            fill='x', padx=12, pady=(10, 0))
        self.var_valor = tk.StringVar(value=valor)
        self.lbl_valor = tk.Label(corpo, textvariable=self.var_valor,
                                  bg=T.BG_CARD, fg=cor_valor or T.FG,
                                  font=T.FONT_HEADLINE, anchor='w')
        self.lbl_valor.pack(fill='x', padx=12, pady=(2, 10))

    def set(self, valor, cor=None):
        self.var_valor.set(valor)
        if cor is not None:
            self.lbl_valor.configure(fg=cor)
            self._barra.configure(bg=cor)


def separador(master, padx=T.CARD_PADX, pady=4):
    """Divisor horizontal sutil de 1px, ja empacotado."""
    bg = master['bg'] if 'bg' in master.keys() else T.BG
    f = tk.Frame(master, bg=T.BORDER, height=1)
    f.pack(fill='x', padx=padx, pady=pady)
    return f


def secao_kicker(master, texto, **kw):
    """Rotulo de secao em CAPS, ambar, peso bold. Anchor west."""
    return tk.Label(master, text=texto.upper(),
                    bg=master['bg'] if 'bg' in master.keys() else T.BG,
                    fg=T.ACCENT_DEEP, font=T.FONT_KICKER, anchor='w', **kw)
