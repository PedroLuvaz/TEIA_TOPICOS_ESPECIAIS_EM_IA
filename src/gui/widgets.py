"""
Widgets compostos reutilizaveis - cartoes, blocos de metrica e
secoes com kicker em ambar. Mantidos genericos para que outras abas
do projeto reaproveitem.
"""
import tkinter as tk

from . import theme as T


class Card(tk.Frame):
    """Cartao com kicker em ambar e conteudo abaixo.

    Uso:
        card = Card(parent, titulo='atributos')
        ttk.Label(card, ...).pack(padx=card.padx, pady=card.pady)

    O Card e apenas um Frame com cor de fundo e um titulo opcional;
    o conteudo e adicionado como filho normal.
    """
    PADX = 14
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
                     bg=T.BG_CARD, fg=T.ACCENT,
                     font=T.FONT_KICKER, anchor='w'
                    ).pack(fill='x', padx=self.PADX, pady=(self.PADY_TOP, 6))


class MetricBlock(tk.Frame):
    """Bloco de metrica: rotulo em caps + valor grande embaixo."""
    def __init__(self, master, rotulo, valor='-', cor_valor=None, **kw):
        super().__init__(master, bg=T.BG_CARD,
                         highlightthickness=1,
                         highlightbackground=T.BORDER,
                         highlightcolor=T.BORDER, **kw)
        tk.Label(self, text=rotulo.upper(),
                 bg=T.BG_CARD, fg=T.ACCENT,
                 font=T.FONT_KICKER, anchor='w').pack(
            fill='x', padx=14, pady=(10, 0))
        self.var_valor = tk.StringVar(value=valor)
        self.lbl_valor = tk.Label(self, textvariable=self.var_valor,
                                  bg=T.BG_CARD, fg=cor_valor or T.FG,
                                  font=T.FONT_HEADLINE, anchor='w')
        self.lbl_valor.pack(fill='x', padx=14, pady=(2, 10))

    def set(self, valor, cor=None):
        self.var_valor.set(valor)
        if cor is not None:
            self.lbl_valor.configure(fg=cor)


def secao_kicker(master, texto, **kw):
    """Rotulo de secao em CAPS, ambar, peso bold. Anchor west."""
    return tk.Label(master, text=texto.upper(),
                    bg=master['bg'] if 'bg' in master.keys() else T.BG,
                    fg=T.ACCENT, font=T.FONT_KICKER, anchor='w', **kw)
