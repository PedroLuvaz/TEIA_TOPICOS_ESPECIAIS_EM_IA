"""
Tema visual do aplicativo - paleta, tipografia e estilos ttk.

Direcao estetica: editorial cientifica em tema claro.
Fundo branco + ambar academico + dados em azul/verde/vermelho.
Tipografia mista: serifa para titulos, sans para corpo, monospace
para numeros. O objetivo e parecer com publicacao academica seria,
nao com painel administrativo generico.
"""
from tkinter import ttk

# ---------------------------------------------------------------------------
# Paleta
# ---------------------------------------------------------------------------
BG          = '#FFFFFF'   # fundo principal (branco)
BG_PANEL    = '#F6F8FA'   # paineis (cinza muito claro)
BG_CARD     = '#EAEEF2'   # cartoes (cinza claro)
BG_HOVER    = '#DDE3EA'   # estado hover
BORDER      = '#D0D7DE'   # bordas suaves
BORDER_HARD = '#ADBAC7'   # bordas com mais contraste

FG          = '#1F2328'   # quase preto (texto principal)
FG_MUTED    = '#57606A'   # cinza intermediario
FG_DIM      = '#8C959F'   # cinza apagado

ACCENT      = '#BF7500'   # ambar escuro para contraste no branco
ACCENT_DEEP = '#8A5500'   # ambar mais escuro (estados ativos)

# Cores das classes - ajustadas para alto contraste no fundo claro
DATA_BLUE   = '#0969DA'   # setosa
DATA_MINT   = '#1A7F37'   # versicolor
DATA_CORAL  = '#CF222E'   # virginica

SUCCESS     = '#1A7F37'
DANGER      = '#CF222E'

# ---------------------------------------------------------------------------
# Tipografia (fontes nativas do Windows 11)
# ---------------------------------------------------------------------------
FONT_DISPLAY    = ('Cambria', 22, 'normal')
FONT_HEADLINE   = ('Cambria', 16, 'bold')
FONT_TITLE      = ('Cambria', 14, 'bold')
FONT_VALUE_BIG  = ('Cambria', 20, 'bold')
FONT_VALUE_HUGE = ('Cambria', 28, 'bold')

FONT_BODY       = ('Segoe UI', 10, 'normal')
FONT_LABEL      = ('Segoe UI', 9, 'normal')
FONT_SUBTITLE   = ('Segoe UI', 9, 'normal')
FONT_BUTTON     = ('Segoe UI Semibold', 10, 'normal')
FONT_TAB        = ('Segoe UI Semibold', 10, 'normal')
FONT_KICKER     = ('Segoe UI', 8, 'bold')   # rotulos em CAPS

FONT_MONO       = ('Consolas', 10, 'normal')
FONT_MONO_SM    = ('Consolas', 9, 'normal')


# ---------------------------------------------------------------------------
# Aplicacao do tema ttk
# ---------------------------------------------------------------------------
def aplicar_tema(root):
    """Configura ttk.Style com a paleta editorial clara.

    Apenas widgets ttk especificos sao estilizados aqui (Notebook, Entry,
    Primary.TButton). Para widgets dentro de cartoes onde a cor de fundo
    importa, preferimos tk.* nativos com configuracao direta - mais simples
    do que mapear estilos ttk hibridos.
    """
    style = ttk.Style(root)
    style.theme_use('clam')

    root.configure(bg=BG)
    root.option_add('*Font', FONT_BODY)
    root.option_add('*tearOff', False)

    # ---- Frame ----
    style.configure('TFrame', background=BG)
    style.configure('Panel.TFrame', background=BG_PANEL)
    style.configure('Card.TFrame', background=BG_CARD)

    # ---- Notebook (assinatura visual: tab ativo em ambar, sem bordas) ----
    style.configure('TNotebook',
        background=BG, borderwidth=0, tabmargins=(0, 8, 0, 0))
    style.configure('TNotebook.Tab',
        background=BG, foreground=FG_MUTED, font=FONT_TAB,
        padding=(22, 11), borderwidth=0)
    style.map('TNotebook.Tab',
        background=[('selected', BG_PANEL), ('active', BG_PANEL),
                    ('disabled', BG)],
        foreground=[('selected', ACCENT), ('active', FG),
                    ('disabled', FG_DIM)])

    # ---- Combobox ----
    style.configure('TCombobox',
        fieldbackground=BG_CARD, foreground=FG,
        insertcolor=ACCENT,
        bordercolor=BORDER_HARD, lightcolor=BORDER_HARD,
        darkcolor=BORDER_HARD,
        arrowcolor=FG_MUTED,
        borderwidth=1, padding=(6, 4))
    style.map('TCombobox',
        fieldbackground=[('readonly', BG_CARD)],
        bordercolor=[('focus', ACCENT)],
        lightcolor=[('focus', ACCENT)],
        darkcolor=[('focus', ACCENT)],
        arrowcolor=[('active', ACCENT)])

    # ---- Scrollbar ----
    style.configure('TScrollbar',
        background=BG_CARD, troughcolor=BG_PANEL,
        bordercolor=BORDER, arrowcolor=FG_MUTED,
        borderwidth=0)
    style.map('TScrollbar',
        background=[('active', BG_HOVER)])

    # ---- Entry ----
    style.configure('TEntry',
        fieldbackground=BG_CARD, foreground=FG,
        insertcolor=ACCENT,
        bordercolor=BORDER_HARD, lightcolor=BORDER_HARD,
        darkcolor=BORDER_HARD,
        borderwidth=1, padding=8)
    style.map('TEntry',
        bordercolor=[('focus', ACCENT)],
        lightcolor=[('focus', ACCENT)],
        darkcolor=[('focus', ACCENT)])

    # ---- Primary button (ambar solido) ----
    style.configure('Primary.TButton',
        background=ACCENT, foreground='#FFFFFF', font=FONT_BUTTON,
        borderwidth=0, focuscolor=ACCENT,
        padding=(16, 10))
    style.map('Primary.TButton',
        background=[('active', ACCENT_DEEP), ('pressed', ACCENT_DEEP)],
        foreground=[('active', '#FFFFFF'), ('pressed', '#FFFFFF')])

    # ---- Treeview (tabela de amostras do dataset) ----
    style.configure('Treeview',
        background=BG_CARD, foreground=FG,
        fieldbackground=BG_CARD,
        bordercolor=BORDER, borderwidth=0,
        rowheight=20, font=FONT_MONO_SM)
    style.configure('Treeview.Heading',
        background=BG_PANEL, foreground=FG_MUTED,
        font=FONT_KICKER, relief='flat',
        bordercolor=BORDER, borderwidth=0, padding=(4, 6))
    style.map('Treeview',
        background=[('selected', ACCENT)],
        foreground=[('selected', '#FFFFFF')])
    style.map('Treeview.Heading',
        background=[('active', BG_HOVER)])
