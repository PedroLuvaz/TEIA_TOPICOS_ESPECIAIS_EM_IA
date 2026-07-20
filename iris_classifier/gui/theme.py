"""
Tema visual do aplicativo - paleta, tipografia, espacamento e estilos ttk.

Direcao estetica: dashboard cientifico em tema claro (slate light).
Fundo da aplicacao em cinza-ardosia suave + cartoes brancos elevados +
acento ambar profundo como assinatura visual + dados em alta legibilidade.
Tipografia: Segoe UI para toda a interface, Consolas para numeros e formulas.
"""
from tkinter import ttk

# ---------------------------------------------------------------------------
# Paleta Light Mode (dashboard cientifico - cartoes brancos sobre ardosia)
# ---------------------------------------------------------------------------
BG          = '#EEF2F6'   # fundo da aplicacao (ardosia suave - destaca os cartoes)
BG_PANEL    = '#F8FAFC'   # Slate 50 - paineis intermediarios e toolbars
BG_CARD     = '#FFFFFF'   # branco - cartoes de conteudo (elevados sobre o BG)
BG_HOVER    = '#E2E8F0'   # Slate 200 - estado de hover
BORDER      = '#DDE3EA'   # bordas sutis e limpas
BORDER_HARD = '#C5CEDA'   # bordas com mais definicao

FG          = '#0F172A'   # Slate 900 - texto principal quase preto
FG_MUTED    = '#475569'   # Slate 600 - texto intermediario
FG_DIM      = '#94A3B8'   # Slate 400 - texto apagado/secundario

ACCENT      = '#D97706'   # ambar 600 - assinatura visual do app (contraste AA)
ACCENT_DEEP = '#B45309'   # ambar 700 (estados ativos/pressionados)
ACCENT_SOFT = '#FEF3C7'   # ambar 100 - fundos de selecao e realces suaves

DATA_BLUE   = '#0284C7'   # Sky 600 - Setosa
DATA_MINT   = '#059669'   # Emerald 600 - Versicolor
DATA_CORAL  = '#E11D48'   # Rose 600 - Virginica

SUCCESS     = '#15803D'   # verde sucesso / acerto
SUCCESS_BG  = '#DCFCE7'   # fundo suave para celulas de acerto
DANGER      = '#DC2626'   # vermelho erro / perigo
DANGER_BG   = '#FEE2E2'   # fundo suave para celulas de erro

# ---------------------------------------------------------------------------
# Espacamento (constantes unicas para todas as abas)
# ---------------------------------------------------------------------------
PAD_PAGE  = 20    # respiro externo das paginas/abas
GAP       = 10    # espaco padrao entre cartoes e colunas
GAP_SM    = 6     # espaco entre elementos proximos
CARD_PADX = 14    # respiro interno horizontal dos cartoes

# ---------------------------------------------------------------------------
# Tipografia (fontes modernas nativas do Windows 11/10)
# ---------------------------------------------------------------------------
FONT_FAMILY = 'Segoe UI'
FONT_FAMILY_NAME = FONT_FAMILY
FONT_FAMILY_TITLE = 'Segoe UI Semibold'

FONT_DISPLAY    = (FONT_FAMILY_TITLE, 21, 'normal')
FONT_HEADLINE   = (FONT_FAMILY_TITLE, 15, 'bold')
FONT_TITLE      = (FONT_FAMILY_TITLE, 12, 'bold')
FONT_VALUE_BIG  = (FONT_FAMILY_TITLE, 20, 'bold')
FONT_VALUE_HUGE = (FONT_FAMILY_TITLE, 28, 'bold')

FONT_BODY       = (FONT_FAMILY, 10, 'normal')
FONT_BODY_SM    = (FONT_FAMILY, 9, 'normal')
FONT_BODY_XS    = (FONT_FAMILY, 8, 'normal')
FONT_LABEL      = (FONT_FAMILY, 9, 'normal')
FONT_SUBTITLE   = (FONT_FAMILY, 9, 'normal')
FONT_BUTTON     = (FONT_FAMILY_TITLE, 10, 'normal')
FONT_TAB        = (FONT_FAMILY_TITLE, 10, 'normal')
FONT_KICKER     = (FONT_FAMILY, 8, 'bold')   # rotulos em CAPS

FONT_MONO       = ('Consolas', 10, 'normal')
FONT_MONO_SM    = ('Consolas', 9, 'normal')
FONT_MONO_XS    = ('Consolas', 8, 'normal')

# Tokens para usos que antes ficavam hardcoded nas abas
FONT_TEXT_HL    = (FONT_FAMILY_TITLE, 10)            # destaque em widgets Text
FONT_CELL       = ('Consolas', 9, 'normal')          # celula de grade/tabela
FONT_CELL_BOLD  = ('Consolas', 9, 'bold')            # celula de grade enfatizada
FONT_CELL_LG    = ('Consolas', 11, 'bold')           # celula grande (matriz)
FONT_REF        = ('Consolas', 8, 'normal')          # referencias arquivo:linha


# ---------------------------------------------------------------------------
# Aplicacao do tema ttk
# ---------------------------------------------------------------------------
def aplicar_tema(root):
    """Configura ttk.Style com a paleta clara e define defaults globais.

    Apenas widgets ttk especificos sao estilizados aqui (Notebook, Entry,
    Combobox, botoes, Treeview). Para widgets dentro de cartoes onde a cor
    de fundo importa, preferimos tk.* nativos com configuracao direta.
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

    # ---- Notebook (tab ativo: cartao branco com texto ambar) ----
    style.configure('TNotebook',
        background=BG, borderwidth=0, tabmargins=(0, 6, 0, 0))
    style.configure('TNotebook.Tab',
        background=BG, foreground=FG_MUTED, font=FONT_TAB,
        padding=(22, 11), borderwidth=0)
    style.map('TNotebook.Tab',
        background=[('selected', BG_CARD), ('active', BG_PANEL),
                    ('disabled', BG)],
        foreground=[('selected', ACCENT_DEEP), ('active', FG),
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
        background=BG_HOVER, troughcolor=BG_PANEL,
        bordercolor=BORDER, arrowcolor=FG_MUTED,
        borderwidth=0)
    style.map('TScrollbar',
        background=[('active', BORDER_HARD)])

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

    # ---- Primary button (ambar solido, texto branco) ----
    style.configure('Primary.TButton',
        background=ACCENT, foreground='#FFFFFF', font=FONT_BUTTON,
        borderwidth=0, focuscolor=ACCENT,
        padding=(16, 10))
    style.map('Primary.TButton',
        background=[('active', ACCENT_DEEP), ('pressed', ACCENT_DEEP)],
        foreground=[('active', '#FFFFFF'), ('pressed', '#FFFFFF')])

    # ---- Ghost button (secundario, contorno sutil) ----
    style.configure('Ghost.TButton',
        background=BG_CARD, foreground=FG_MUTED, font=FONT_BUTTON,
        borderwidth=1, bordercolor=BORDER_HARD,
        lightcolor=BORDER_HARD, darkcolor=BORDER_HARD,
        focuscolor=BG_CARD, padding=(14, 8))
    style.map('Ghost.TButton',
        background=[('active', BG_PANEL), ('pressed', BG_HOVER)],
        foreground=[('active', ACCENT_DEEP)],
        bordercolor=[('active', ACCENT)],
        lightcolor=[('active', ACCENT)],
        darkcolor=[('active', ACCENT)])

    # ---- Treeview (tabela de amostras do dataset) ----
    style.configure('Treeview',
        background=BG_CARD, foreground=FG,
        fieldbackground=BG_CARD,
        bordercolor=BORDER, borderwidth=0,
        rowheight=22, font=FONT_MONO_SM)
    style.configure('Treeview.Heading',
        background=BG_PANEL, foreground=FG_MUTED,
        font=FONT_KICKER, relief='flat',
        bordercolor=BORDER, borderwidth=0, padding=(4, 6))
    style.map('Treeview',
        background=[('selected', ACCENT_SOFT)],
        foreground=[('selected', ACCENT_DEEP)])
    style.map('Treeview.Heading',
        background=[('active', BG_HOVER)])

    # ---- Matplotlib: estilizacao global coerente com o tema ----
    try:
        import matplotlib
        matplotlib.rcParams['text.color'] = FG
        matplotlib.rcParams['axes.labelcolor'] = FG_MUTED
        matplotlib.rcParams['xtick.color'] = FG_MUTED
        matplotlib.rcParams['ytick.color'] = FG_MUTED
        matplotlib.rcParams['axes.edgecolor'] = BORDER_HARD
        matplotlib.rcParams['axes.facecolor'] = BG_CARD
        matplotlib.rcParams['figure.facecolor'] = BG_CARD
        matplotlib.rcParams['grid.color'] = BORDER
        matplotlib.rcParams['grid.alpha'] = 0.5
        matplotlib.rcParams['font.family'] = 'sans-serif'
        matplotlib.rcParams['font.sans-serif'] = [FONT_FAMILY_NAME, 'DejaVu Sans', 'Arial', 'Helvetica']
    except ImportError:
        pass
