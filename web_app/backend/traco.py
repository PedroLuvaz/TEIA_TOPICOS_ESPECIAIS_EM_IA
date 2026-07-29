"""
Formato generico de "memoria de calculo".

As janelas LaTeX da GUI Tkinter mostram, para cada laboratorio, a teoria e a
substituicao numerica passo a passo. Aqui esse conteudo vira uma estrutura de
dados — secoes numeradas, cada uma com blocos tipados — que um unico
componente React renderiza. Assim, adicionar a memoria de um novo laboratorio
e so montar secoes, sem escrever tela nova.

Blocos disponiveis:
    texto      paragrafo explicativo
    formula    LaTeX (renderizado com KaTeX no cliente)
    passos     linhas monoespacadas com a substituicao numerica
    tabela     colunas + linhas
    resultado  destaque colorido do valor final da secao
    nota       caixa de aviso (info / atencao / ok)
    ref        ponteiro para arquivo:linha da funcao que faz a conta
"""
import inspect
import os


def texto(conteudo):
    return {'tipo': 'texto', 'conteudo': conteudo}


def formula(latex, titulo=None, explicacao=None):
    return {'tipo': 'formula', 'latex': latex,
            'titulo': titulo, 'explicacao': explicacao}


def passos(itens, titulo=None):
    return {'tipo': 'passos', 'itens': [i for i in itens if i is not None],
            'titulo': titulo}


def tabela(colunas, linhas, titulo=None, alinhamento=None):
    """`alinhamento`: lista com 'esq' ou 'dir' por coluna (padrao: 1a esq)."""
    return {'tipo': 'tabela', 'colunas': colunas, 'linhas': linhas,
            'titulo': titulo,
            'alinhamento': alinhamento or
                           ['esq'] + ['dir'] * (len(colunas) - 1)}


def resultado(conteudo, tom='destaque'):
    """tom: 'destaque' | 'bom' | 'medio' | 'ruim'"""
    return {'tipo': 'resultado', 'conteudo': conteudo, 'tom': tom}


def nota(conteudo, tom='info', titulo=None):
    return {'tipo': 'nota', 'conteudo': conteudo, 'tom': tom, 'titulo': titulo}


def ref(func):
    """Ponteiro arquivo:linha da funcao — mesma ideia do `_ref_funcao` da GUI."""
    try:
        _, linha = inspect.getsourcelines(func)
        return {'tipo': 'ref',
                'arquivo': os.path.basename(inspect.getfile(func)),
                'linha': linha,
                'nome': getattr(func, '__name__', '?')}
    except (OSError, TypeError):
        return {'tipo': 'ref', 'arquivo': '?', 'linha': 0, 'nome': '?'}


def secao(titulo, *blocos):
    """Blocos None sao descartados — facilita montar secoes condicionais."""
    return {'titulo': titulo,
            'blocos': [b for b in blocos if b is not None]}


def montar(titulo, subtitulo, *secoes, cabecalho=None):
    """
    Traco completo. `cabecalho` e uma lista de {rotulo, valor} exibida no topo
    (ex.: dataset, atributos, tamanho do split).
    """
    return {
        'formato': 'generico',
        'titulo': titulo,
        'subtitulo': subtitulo,
        'cabecalho': cabecalho or [],
        'secoes': [s for s in secoes if s is not None],
    }


# ---------------------------------------------------------------- formatacao
def n(valor, casas=4):
    """Numero com N casas — string pronta para as linhas de `passos`."""
    if valor is None:
        return '—'
    return f'{valor:.{casas}f}'


def vetor(v, casas=4):
    return '[' + ', '.join(n(x, casas) for x in v) + ']'


def pct(valor, casas=2):
    return f'{valor * 100:.{casas}f}%'
