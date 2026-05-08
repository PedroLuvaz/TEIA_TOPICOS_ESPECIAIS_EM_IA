"""
Tema visual do aplicativo - paleta, tipografia e estilos ttk.

Direcao estetica: editorial cientifica em tema escuro.
Slate profundo + ambar academico + dados em mint/coral/azul.
Tipografia mista: serifa para titulos, sans para corpo, monospace
para numeros. O objetivo e parecer com publicacao academica seria,
nao com painel administrativo generico.
"""
from tkinter import ttk

# ---------------------------------------------------------------------------
# Paleta
# ---------------------------------------------------------------------------
BG          = '#0E1117'   # fundo principal (slate quase preto)
BG_PANEL    = '#161B22'   # paineis (1 nivel acima)
BG_CARD     = '#1C2230'   # cartoes (2 niveis acima)
BG_HOVER    = '#222B3D'   # estado hover
BORDER      = '#2A3142'   # bordas suaves
BORDER_HARD = '#3E4860'   # bordas com mais contraste

FG          = '#E6EDF3'   # branco quente (texto principal)
FG_MUTED    = '#8B949E'   # cinza intermediario
FG_DIM      = '#6E7681'   # cinza apagado

ACCENT      = '#E8A33D'   # ambar - assinatura visual do app
ACCENT_DEEP = '#B07A20'   # ambar escuro (estados ativos)

# Cores das classes - escolhidas para alto contraste no fundo escuro
DATA_BLUE   = '#58A6FF'   # setosa
DATA_MINT   = '#3FB950'   # versicolor
DATA_CORAL  = '#F85149'   # virginica

SUCCESS     = '#3FB950'
DANGER      = '#F85149'

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
    """Configura ttk.Style com a paleta editorial escura.

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
        background=ACCENT, foreground=BG, font=FONT_BUTTON,
        borderwidth=0, focuscolor=ACCENT,
        padding=(16, 10))
    style.map('Primary.TButton',
        background=[('active', ACCENT_DEEP), ('pressed', ACCENT_DEEP)])
