"""
Tema visual do aplicativo - paleta, tipografia e estilos ttk.

Direcao estetica: dashboard interativo moderno em tema claro (slate light).
Fundo branco limpo + realce em indigo eletrico + dados em alta legibilidade.
Tipografia: Segoe UI para toda a interface (sem serifas academicas),
Consolas para numeros e formulas.
"""
from tkinter import ttk

# ---------------------------------------------------------------------------
# Paleta Light Mode Premium (Modern, Clean & Slate Light)
# ---------------------------------------------------------------------------
BG          = '#FFFFFF'   # fundo principal (branco limpo)
BG_PANEL    = '#F8FAFC'   # Slate 50 - painéis de controle e áreas secundárias
BG_CARD     = '#F1F5F9'   # Slate 100 - cartões de conteúdo
BG_HOVER    = '#E2E8F0'   # Slate 200 - estado de hover
BORDER      = '#E2E8F0'   # Slate 200 - bordas sutis e limpas
BORDER_HARD = '#CBD5E1'   # Slate 300 - bordas com mais definição

FG          = '#0F172A'   # Slate 900 - texto principal quase preto para excelente legibilidade
FG_MUTED    = '#475569'   # Slate 600 - texto intermediário
FG_DIM      = '#94A3B8'   # Slate 400 - texto apagado/secundário

ACCENT      = '#E8A33D'   # ambar/laranja - assinatura visual do app
ACCENT_DEEP = '#B07A20'   # ambar escuro (estados ativos)

DATA_BLUE   = '#0284C7'   # Cyan/Sky 600 - Setosa
DATA_MINT   = '#059669'   # Emerald/Green 600 - Versicolor
DATA_CORAL  = '#E11D48'   # Rose/Red 600 - Virginica

SUCCESS     = '#059669'   # Verde sucesso / acerto
DANGER      = '#E11D48'   # Vermelho erro / perigo

# ---------------------------------------------------------------------------
# Tipografia (fontes modernas nativas do Windows 11/10)
# ---------------------------------------------------------------------------
FONT_FAMILY = 'Segoe UI'
FONT_FAMILY_NAME = FONT_FAMILY
FONT_FAMILY_TITLE = 'Segoe UI Semibold'

FONT_DISPLAY    = (FONT_FAMILY_TITLE, 22, 'normal')
FONT_HEADLINE   = (FONT_FAMILY_TITLE, 15, 'bold')
FONT_TITLE      = (FONT_FAMILY_TITLE, 12, 'bold')
FONT_VALUE_BIG  = (FONT_FAMILY_TITLE, 20, 'bold')
FONT_VALUE_HUGE = (FONT_FAMILY_TITLE, 28, 'bold')

FONT_BODY       = (FONT_FAMILY, 10, 'normal')
FONT_LABEL      = (FONT_FAMILY, 9, 'normal')
FONT_SUBTITLE   = (FONT_FAMILY, 9, 'normal')
FONT_BUTTON     = (FONT_FAMILY_TITLE, 10, 'normal')
FONT_TAB        = (FONT_FAMILY_TITLE, 10, 'normal')
FONT_KICKER     = (FONT_FAMILY, 8, 'bold')   # rotulos em CAPS

FONT_MONO       = ('Consolas', 10, 'normal')
FONT_VALUE_HUGE = (FONT_FAMILY_TITLE, 28, 'bold')

FONT_BODY       = (FONT_FAMILY, 10, 'normal')
FONT_LABEL      = (FONT_FAMILY, 9, 'normal')
FONT_SUBTITLE   = (FONT_FAMILY, 9, 'normal')
FONT_BUTTON     = (FONT_FAMILY_TITLE, 10, 'normal')
FONT_TAB        = (FONT_FAMILY_TITLE, 10, 'normal')
FONT_KICKER     = (FONT_FAMILY, 8, 'bold')   # rotulos em CAPS

FONT_MONO       = ('Consolas', 10, 'normal')
FONT_MONO_SM    = ('Consolas', 9, 'normal')


# ---------------------------------------------------------------------------
# Aplicacao do tema ttk
# ---------------------------------------------------------------------------
def aplicar_tema(root):
    """Configura ttk.Style com a paleta editorial escura e define defaults.

    Apenas widgets ttk especificos sao estilizados aqui (Notebook, Entry,
    Primary.TButton). Para widgets dentro de cartoes onde a cor de fundo
    importa, preferimos tk.* nativos com configuracao direta.
    """
    style = ttk.Style(root)
    style.theme_use('clam')

    root.configure(bg=BG)
    root.option_add('*Font', FONT_BODY)
    root.option_add('*Background', BG)
    root.option_add('*Foreground', FG)
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
        background=ACCENT, foreground='#0F172A', font=FONT_BUTTON,
        borderwidth=0, focuscolor=ACCENT,
        padding=(16, 10))
    style.map('Primary.TButton',
        background=[('active', ACCENT_DEEP), ('pressed', ACCENT_DEEP)],
        foreground=[('active', '#0F172A'), ('pressed', '#0F172A')])

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
        foreground=[('selected', '#0F172A')])
    style.map('Treeview.Heading',
        background=[('active', BG_HOVER)])

    # ---- Matplotlib Global Dark Styling ----
    try:
        import matplotlib
        matplotlib.rcParams['text.color'] = FG
        matplotlib.rcParams['axes.labelcolor'] = FG_MUTED
        matplotlib.rcParams['xtick.color'] = FG_MUTED
        matplotlib.rcParams['ytick.color'] = FG_MUTED
        matplotlib.rcParams['axes.edgecolor'] = BORDER_HARD
        matplotlib.rcParams['axes.facecolor'] = BG_PANEL
        matplotlib.rcParams['figure.facecolor'] = BG_PANEL
        matplotlib.rcParams['grid.color'] = BORDER
        matplotlib.rcParams['grid.alpha'] = 0.5
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = [FONT_FAMILY_NAME, 'DejaVu Sans', 'Arial', 'Helvetica']
    except ImportError:
        pass
