"""
Lab 5 — Perceptron Multicamadas (MLP) com Backpropagation.

Lab 5.0 · XOR (slides 36-37): exemplo didatico e o exercicio do XOR, com
         treino interativo e fronteira de decisao.
Lab 5.1 · Feedforward (itens i/ii + slide 34): galinha vs homem, reconhecimento
         de imagem 8x8 e o comparativo MLP x Bayes x Naive no Iris.

A memoria de calculo e devolvida como um "traco" estruturado — cada etapa com
os termos da substituicao numerica — para o frontend renderizar em LaTeX.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from evaluation.metricas_avancadas import (p_valor_z, relatorio_completo,
                                           z_kappa)
from models.bayes_classifier import (predizer_todas_classes_bayes,
                                     treinar_bayes)
from models.mlp_backprop import RedeFeedforward
from models.padroes_pixels import GALINHA_PIXELS, HOMEM_PIXELS, achatar

from ..core import classes_de, indices_de, obter_split
from ..lab5_config import EXERCICIOS, PASSO_UNICO

router = APIRouter(prefix='/api/lab5', tags=['lab5'])

# Rede do bonus de imagem: treinada uma unica vez e reutilizada (o treino de
# 4000 epocas leva ~1s; manter em memoria evita repetir a cada predicao).
_rede_imagem = None


class TreinoXORRequest(BaseModel):
    # epocas=0 e valido: devolve o estado inicial da rede, sem treinar
    epocas: int = Field(1, ge=0, le=20000)
    taxa: float = Field(0.5, gt=0, le=5)
    pesos_oculta: list[list[float]] | None = None
    bias_oculta: list[float] | None = None
    pesos_saida: list[list[float]] | None = None
    bias_saida: list[float] | None = None
    resolucao: int = Field(60, ge=20, le=120)


class PixelsRequest(BaseModel):
    pixels: list[list[float]]


class TrajetoriaRequest(BaseModel):
    epocas: int = Field(2000, ge=1, le=50000)
    taxa: float | None = Field(None, gt=0, le=5)
    n_snapshots: int = Field(60, ge=2, le=200)
    pesos_oculta: list[list[float]] | None = None
    bias_oculta: list[float] | None = None
    pesos_saida: list[list[float]] | None = None
    bias_saida: list[float] | None = None


# ---------------------------------------------------------------------------
# Exercicios / memoria de calculo
# ---------------------------------------------------------------------------
@router.get('/exercicios')
def listar_exercicios():
    """Lista os exercicios disponiveis, agrupados por sub-laboratorio."""
    def resumo(e):
        return {
            'id': e['id'], 'titulo': e['titulo'], 'subtitulo': e['subtitulo'],
            'slide': e['slide'], 'taxa': e['taxa'],
            'arquitetura': _arquitetura_texto(e),
            'bias_compartilhado': e['bias_compartilhado'],
        }
    return {
        'lab_5_0': [resumo(EXERCICIOS['didatico']), resumo(EXERCICIOS['xor'])],
        'lab_5_1': [resumo(EXERCICIOS['galinha-homem']), resumo(EXERCICIOS['fig-1232'])],
    }


@router.get('/memoria/{exercicio_id}')
def memoria_calculo(exercicio_id: str):
    """
    Traco completo de um passo de backpropagation (ou de 1 epoca, no XOR),
    com todos os termos intermediarios para renderizacao em LaTeX.
    """
    cfg = EXERCICIOS.get(exercicio_id)
    if cfg is None:
        raise HTTPException(status_code=404,
                            detail=f"Exercicio '{exercicio_id}' nao encontrado.")

    if exercicio_id in PASSO_UNICO:
        return _traco_passo_unico(cfg)
    return _traco_epoca(cfg)


def _nova_rede(cfg):
    return RedeFeedforward(
        n_entradas=len(cfg['rotulos_entrada']),
        n_ocultos=len(cfg['bias_oculta']),
        n_saidas=len(cfg['bias_saida']),
        pesos_oculta=[linha[:] for linha in cfg['pesos_oculta']],
        bias_oculta=list(cfg['bias_oculta']),
        pesos_saida=[linha[:] for linha in cfg['pesos_saida']],
        bias_saida=list(cfg['bias_saida']),
    )


def _arquitetura(cfg, pesos_oculta=None, bias_oculta=None,
                 pesos_saida=None, bias_saida=None):
    """Payload que o frontend usa para desenhar o diagrama da rede."""
    return {
        'rotulos_entrada': cfg['rotulos_entrada'],
        'rotulos_ocultos': cfg['rotulos_ocultos'],
        'rotulos_saida': cfg['rotulos_saida'],
        'pesos_oculta': pesos_oculta or cfg['pesos_oculta'],
        'bias_oculta': bias_oculta or cfg['bias_oculta'],
        'pesos_saida': pesos_saida or cfg['pesos_saida'],
        'bias_saida': bias_saida or cfg['bias_saida'],
        'bias_compartilhado': cfg['bias_compartilhado'],
        'texto': _arquitetura_texto(cfg),
    }


def _arquitetura_texto(cfg):
    return (f"{len(cfg['rotulos_entrada'])}-{len(cfg['rotulos_ocultos'])}"
            f"-{len(cfg['rotulos_saida'])}")


def _traco_passo_unico(cfg):
    """Forward + backward + atualizacao para uma unica amostra."""
    rede = _nova_rede(cfg)
    entradas, alvo, taxa = cfg['entradas'], cfg['alvo'], cfg['taxa']
    r = rede.passo_treinamento(entradas, alvo, taxa)

    # Convencao de bias unico por camada (exemplo didatico do slide 37)
    if cfg['bias_compartilhado']:
        b_oc = cfg['bias_oculta'][0] - taxa * sum(r['delta_oculta'])
        b_sa = cfg['bias_saida'][0] - taxa * sum(r['delta_saida'])
        r['b_oculta_depois'] = [b_oc] * len(cfg['bias_oculta'])
        r['b_saida_depois'] = [b_sa] * len(cfg['bias_saida'])
        rede.b_oculta = list(r['b_oculta_depois'])
        rede.b_saida = list(r['b_saida_depois'])

    nova_saida = rede.prever(entradas)
    novo_erro = rede.erro_total(nova_saida, alvo)

    # --- Etapa: forward da camada oculta ---
    forward_oculta = []
    for i, nome in enumerate(cfg['rotulos_ocultos']):
        w, b = cfg['pesos_oculta'][i], cfg['bias_oculta'][i]
        termos = [{'entrada': entradas[j], 'peso': w[j], 'produto': entradas[j] * w[j]}
                  for j in range(len(entradas))]
        forward_oculta.append({
            'nome': nome, 'termos': termos, 'bias': b,
            'net': b + sum(t['produto'] for t in termos),
            'out': r['saida_oculta'][i],
        })

    # --- Etapa: forward da camada de saida ---
    forward_saida = []
    for i, nome in enumerate(cfg['rotulos_saida']):
        w, b = cfg['pesos_saida'][i], cfg['bias_saida'][i]
        termos = [{'entrada': r['saida_oculta'][j], 'peso': w[j],
                   'produto': r['saida_oculta'][j] * w[j],
                   'origem': cfg['rotulos_ocultos'][j]}
                  for j in range(len(r['saida_oculta']))]
        forward_saida.append({
            'nome': nome, 'termos': termos, 'bias': b,
            'net': b + sum(t['produto'] for t in termos),
            'out': r['saida_rede'][i],
        })

    # --- Etapa: erro ---
    erro_por_saida = [
        {'nome': nome, 'alvo': alvo[i], 'saida': r['saida_rede'][i],
         'erro': 0.5 * (alvo[i] - r['saida_rede'][i]) ** 2}
        for i, nome in enumerate(cfg['rotulos_saida'])
    ]

    # --- Etapa: deltas ---
    deltas_saida = [
        {'nome': nome, 'saida': r['saida_rede'][i], 'alvo': alvo[i],
         'delta': r['delta_saida'][i]}
        for i, nome in enumerate(cfg['rotulos_saida'])
    ]
    deltas_oculta = [
        {'nome': nome, 'out': r['saida_oculta'][i], 'delta': r['delta_oculta'][i],
         'contribuicoes': [
             {'origem': cfg['rotulos_saida'][o],
              'delta': r['delta_saida'][o],
              'peso': cfg['pesos_saida'][o][i],
              'produto': r['delta_saida'][o] * cfg['pesos_saida'][o][i]}
             for o in range(len(cfg['rotulos_saida']))
         ]}
        for i, nome in enumerate(cfg['rotulos_ocultos'])
    ]

    # --- Etapa: atualizacao dos pesos ---
    atualizacao_saida = [
        {'destino': cfg['rotulos_saida'][i], 'origem': cfg['rotulos_ocultos'][j],
         'antes': cfg['pesos_saida'][i][j], 'depois': r['w_saida_depois'][i][j],
         'delta': r['delta_saida'][i], 'entrada': r['saida_oculta'][j]}
        for i in range(len(cfg['rotulos_saida']))
        for j in range(len(cfg['rotulos_ocultos']))
    ]
    atualizacao_oculta = [
        {'destino': cfg['rotulos_ocultos'][i], 'origem': cfg['rotulos_entrada'][j],
         'antes': cfg['pesos_oculta'][i][j], 'depois': r['w_oculta_depois'][i][j],
         'delta': r['delta_oculta'][i], 'entrada': entradas[j]}
        for i in range(len(cfg['rotulos_ocultos']))
        for j in range(len(entradas))
    ]

    if cfg['bias_compartilhado']:
        bias_atualizados = [
            {'camada': 'oculta', 'nome': 'b1 (compartilhado)',
             'antes': cfg['bias_oculta'][0], 'depois': r['b_oculta_depois'][0],
             'deltas': r['delta_oculta'], 'soma_deltas': sum(r['delta_oculta'])},
            {'camada': 'saida', 'nome': 'b2 (compartilhado)',
             'antes': cfg['bias_saida'][0], 'depois': r['b_saida_depois'][0],
             'deltas': r['delta_saida'], 'soma_deltas': sum(r['delta_saida'])},
        ]
    else:
        bias_atualizados = [
            {'camada': 'oculta', 'nome': nome,
             'antes': cfg['bias_oculta'][i], 'depois': r['b_oculta_depois'][i],
             'deltas': [r['delta_oculta'][i]], 'soma_deltas': r['delta_oculta'][i]}
            for i, nome in enumerate(cfg['rotulos_ocultos'])
        ] + [
            {'camada': 'saida', 'nome': nome,
             'antes': cfg['bias_saida'][i], 'depois': r['b_saida_depois'][i],
             'deltas': [r['delta_saida'][i]], 'soma_deltas': r['delta_saida'][i]}
            for i, nome in enumerate(cfg['rotulos_saida'])
        ]

    return {
        'tipo': 'passo-unico',
        'config': {k: cfg[k] for k in
                   ('id', 'titulo', 'subtitulo', 'slide', 'taxa', 'nota',
                    'bias_compartilhado')},
        'arquitetura': _arquitetura(cfg),
        'arquitetura_depois': _arquitetura(
            cfg, r['w_oculta_depois'], r['b_oculta_depois'],
            r['w_saida_depois'], r['b_saida_depois']),
        'entradas': entradas,
        'alvo': alvo,
        'forward_oculta': forward_oculta,
        'forward_saida': forward_saida,
        'erro': {'por_saida': erro_por_saida, 'total': r['erro_total']},
        'deltas_saida': deltas_saida,
        'deltas_oculta': deltas_oculta,
        'atualizacao': {'saida': atualizacao_saida, 'oculta': atualizacao_oculta,
                        'bias': bias_atualizados},
        'nova_predicao': {
            'saidas': [{'nome': nome, 'antes': r['saida_rede'][i],
                        'depois': nova_saida[i], 'alvo': alvo[i]}
                       for i, nome in enumerate(cfg['rotulos_saida'])],
            'erro_antes': r['erro_total'],
            'erro_depois': novo_erro,
            'reduziu': novo_erro < r['erro_total'],
        },
    }


def _traco_epoca(cfg):
    """Uma epoca completa: os 4 padroes do XOR em modo online."""
    rede = _nova_rede(cfg)
    padroes = cfg['padroes']
    taxa = cfg['taxa']

    antes = [rede.prever(p['entrada'])[0] for p in padroes]

    passos = []
    for i, p in enumerate(padroes):
        r = rede.passo_treinamento(p['entrada'], p['alvo'], taxa)
        passos.append({
            'indice': i + 1,
            'entrada': p['entrada'],
            'alvo': p['alvo'][0],
            'saida_oculta': r['saida_oculta'],
            'saida': r['saida_rede'][0],
            'erro': r['erro_total'],
            'delta_saida': r['delta_saida'],
            'delta_oculta': r['delta_oculta'],
            'pesos_oculta': r['w_oculta_depois'],
            'bias_oculta': r['b_oculta_depois'],
            'pesos_saida': r['w_saida_depois'],
            'bias_saida': r['b_saida_depois'],
        })

    depois = [rede.prever(p['entrada'])[0] for p in padroes]
    erro_medio = sum(p['erro'] for p in passos) / len(passos)

    resultados = []
    acertos = 0
    for i, p in enumerate(padroes):
        alvo = int(p['alvo'][0])
        previsto = 1 if depois[i] >= 0.5 else 0
        correto = previsto == alvo
        acertos += int(correto)
        resultados.append({
            'entrada': p['entrada'], 'alvo': alvo,
            'antes': antes[i], 'depois': depois[i],
            'previsto': previsto, 'correto': correto,
        })

    return {
        'tipo': 'epoca',
        'config': {k: cfg[k] for k in
                   ('id', 'titulo', 'subtitulo', 'slide', 'taxa', 'nota',
                    'bias_compartilhado')},
        'arquitetura': _arquitetura(cfg),
        'arquitetura_depois': _arquitetura(
            cfg, rede.w_oculta, rede.b_oculta, rede.w_saida, rede.b_saida),
        'padroes': [{'entrada': p['entrada'], 'alvo': p['alvo'][0]} for p in padroes],
        'passos': passos,
        'erro_medio': erro_medio,
        'resultados': resultados,
        'acertos': acertos,
        'total': len(padroes),
    }


# ---------------------------------------------------------------------------
# Lab 5.0 — treino interativo do XOR
# ---------------------------------------------------------------------------
@router.post('/xor/treinar')
def treinar_xor(req: TreinoXORRequest):
    """
    Treina o XOR por N epocas a partir dos pesos informados (ou dos iniciais)
    e devolve a fronteira de decisao, a curva de erro e o estado dos pesos —
    permitindo continuar o treino incrementalmente no frontend.
    """
    cfg = EXERCICIOS['xor']
    rede = RedeFeedforward(
        n_entradas=2, n_ocultos=2, n_saidas=1,
        pesos_oculta=[linha[:] for linha in (req.pesos_oculta or cfg['pesos_oculta'])],
        bias_oculta=list(req.bias_oculta or cfg['bias_oculta']),
        pesos_saida=[linha[:] for linha in (req.pesos_saida or cfg['pesos_saida'])],
        bias_saida=list(req.bias_saida or cfg['bias_saida']),
    )

    padroes = cfg['padroes']
    historico = []
    for _ in range(req.epocas):
        soma = 0.0
        for p in padroes:
            soma += rede.passo_treinamento(p['entrada'], p['alvo'], req.taxa)['erro_total']
        historico.append(soma / len(padroes))

    resultados, acertos = [], 0
    for p in padroes:
        saida = rede.prever(p['entrada'])[0]
        alvo = int(p['alvo'][0])
        previsto = 1 if saida >= 0.5 else 0
        acertos += int(previsto == alvo)
        resultados.append({'entrada': p['entrada'], 'alvo': alvo,
                           'saida': saida, 'previsto': previsto,
                           'correto': previsto == alvo})

    # Superficie de saida da rede sobre o plano x1 x x2
    n = req.resolucao
    lo, hi = -0.35, 1.35
    passo = (hi - lo) / (n - 1)
    eixo = [lo + k * passo for k in range(n)]
    superficie = [[rede.forward([x, y])[1][0] for x in eixo] for y in eixo]

    return {
        'historico': historico,
        'erro_medio': historico[-1] if historico else None,
        'resultados': resultados,
        'acertos': acertos,
        'superficie': superficie,
        'eixo': eixo,
        'limites': {'min': lo, 'max': hi},
        'pesos': {
            'oculta': rede.w_oculta, 'bias_oculta': rede.b_oculta,
            'saida': rede.w_saida, 'bias_saida': rede.b_saida,
        },
        'arquitetura': _arquitetura(cfg, rede.w_oculta, rede.b_oculta,
                                    rede.w_saida, rede.b_saida),
    }


@router.get('/xor/inicial')
def xor_inicial(resolucao: int = Query(60, ge=20, le=120)):
    """Estado inicial do XOR (epoca 0), antes de qualquer treino."""
    return treinar_xor(TreinoXORRequest(epocas=0, resolucao=resolucao))


# ---------------------------------------------------------------------------
# Trajetoria de treino — alimenta o slider de epocas do frontend
# ---------------------------------------------------------------------------
def _indices_snapshots(epocas: int, quantidade: int) -> list[int]:
    """
    Epocas em que gravar um snapshot, espacadas logaritmicamente.

    O aprendizado muda muito rapido no inicio e quase nada no fim, entao o
    espacamento log da uma resolucao util nas primeiras epocas sem gerar
    milhares de snapshots.
    """
    if epocas <= quantidade:
        return list(range(epocas + 1))
    indices = {0, epocas}
    for k in range(quantidade):
        t = k / (quantidade - 1)
        indices.add(int(round((epocas + 1) ** t)) - 1 + 1)
    return sorted(i for i in indices if 0 <= i <= epocas)


@router.post('/trajetoria/{exercicio_id}')
def trajetoria(exercicio_id: str, req: TrajetoriaRequest):
    """
    Treina o exercicio por N epocas de uma vez e devolve o historico completo
    de erro mais snapshots dos pesos ao longo do caminho.

    O frontend usa isso para o slider de epocas: como os pesos de cada
    snapshot vem juntos, a superficie de decisao e recalculada localmente
    enquanto o usuario arrasta — sem uma chamada de rede por quadro.
    """
    cfg = EXERCICIOS.get(exercicio_id)
    if cfg is None:
        raise HTTPException(status_code=404,
                            detail=f"Exercicio '{exercicio_id}' nao encontrado.")

    taxa = req.taxa if req.taxa is not None else cfg['taxa']
    rede = RedeFeedforward(
        n_entradas=len(cfg['rotulos_entrada']),
        n_ocultos=len(cfg['bias_oculta']),
        n_saidas=len(cfg['bias_saida']),
        pesos_oculta=[l[:] for l in (req.pesos_oculta or cfg['pesos_oculta'])],
        bias_oculta=list(req.bias_oculta or cfg['bias_oculta']),
        pesos_saida=[l[:] for l in (req.pesos_saida or cfg['pesos_saida'])],
        bias_saida=list(req.bias_saida or cfg['bias_saida']),
    )

    # Um exercicio de passo unico e tratado como um "padrao so" repetido:
    # cada epoca aplica um passo de backprop sobre a mesma amostra.
    if exercicio_id in PASSO_UNICO:
        padroes = [(cfg['entradas'], cfg['alvo'])]
    else:
        padroes = [(p['entrada'], p['alvo']) for p in cfg['padroes']]

    marcos = set(_indices_snapshots(req.epocas, req.n_snapshots))
    historico: list[float] = []
    snapshots: list[dict] = []

    def _snapshot(epoca: int, erro: float | None):
        snapshots.append({
            'epoca': epoca,
            'erro': erro,
            'pesos_oculta': [l[:] for l in rede.w_oculta],
            'bias_oculta': list(rede.b_oculta),
            'pesos_saida': [l[:] for l in rede.w_saida],
            'bias_saida': list(rede.b_saida),
            'saidas': [rede.prever(x) for x, _ in padroes],
        })

    if 0 in marcos:
        _snapshot(0, None)

    for epoca in range(1, req.epocas + 1):
        soma = 0.0
        for x, alvo in padroes:
            soma += rede.passo_treinamento(x, alvo, taxa)['erro_total']
        erro_medio = soma / len(padroes)
        historico.append(erro_medio)
        if epoca in marcos:
            _snapshot(epoca, erro_medio)

    alvos = [alvo for _, alvo in padroes]
    return {
        'exercicio': exercicio_id,
        'tipo': 'passo-unico' if exercicio_id in PASSO_UNICO else 'epoca',
        'taxa': taxa,
        'epocas': req.epocas,
        'historico': historico,
        'snapshots': snapshots,
        'padroes': [{'entrada': x, 'alvo': a} for x, a in padroes],
        'alvos': alvos,
        'arquitetura': _arquitetura(cfg),
        'config': {k: cfg[k] for k in
                   ('id', 'titulo', 'subtitulo', 'slide', 'nota',
                    'bias_compartilhado')},
    }


# ---------------------------------------------------------------------------
# Construtor de rede — arquitetura e pesos definidos pelo usuario
# ---------------------------------------------------------------------------
class PadraoCustom(BaseModel):
    entrada: list[float]
    alvo: list[float]


class RedeCustomRequest(BaseModel):
    """Rede montada na interface: arquitetura, pesos e padroes arbitrarios."""
    pesos_oculta: list[list[float]]
    bias_oculta: list[float]
    pesos_saida: list[list[float]]
    bias_saida: list[float]
    padroes: list[PadraoCustom] = Field(min_length=1)
    taxa: float = Field(0.5, gt=0, le=5)
    epocas: int = Field(1000, ge=0, le=50000)
    n_snapshots: int = Field(60, ge=2, le=200)
    rotulos_entrada: list[str] | None = None
    rotulos_ocultos: list[str] | None = None
    rotulos_saida: list[str] | None = None


def _validar_rede(req: RedeCustomRequest):
    """Confere a coerencia das dimensoes antes de instanciar a rede."""
    n_oc = len(req.bias_oculta)
    n_sa = len(req.bias_saida)
    if n_oc == 0 or n_sa == 0:
        raise HTTPException(status_code=400,
                            detail='A rede precisa de ao menos 1 neuronio oculto e 1 de saida.')
    if len(req.pesos_oculta) != n_oc:
        raise HTTPException(
            status_code=400,
            detail=f'pesos_oculta tem {len(req.pesos_oculta)} linhas, '
                   f'mas ha {n_oc} bias na camada oculta.')
    if len(req.pesos_saida) != n_sa:
        raise HTTPException(
            status_code=400,
            detail=f'pesos_saida tem {len(req.pesos_saida)} linhas, '
                   f'mas ha {n_sa} bias na camada de saida.')

    n_ent = len(req.pesos_oculta[0])
    if any(len(linha) != n_ent for linha in req.pesos_oculta):
        raise HTTPException(status_code=400,
                            detail='Todas as linhas de pesos_oculta devem ter o mesmo tamanho.')
    if any(len(linha) != n_oc for linha in req.pesos_saida):
        raise HTTPException(
            status_code=400,
            detail=f'Cada linha de pesos_saida deve ter {n_oc} pesos '
                   '(um por neuronio oculto).')

    for i, p in enumerate(req.padroes):
        if len(p.entrada) != n_ent:
            raise HTTPException(
                status_code=400,
                detail=f'Padrao {i + 1}: esperadas {n_ent} entradas, '
                       f'recebidas {len(p.entrada)}.')
        if len(p.alvo) != n_sa:
            raise HTTPException(
                status_code=400,
                detail=f'Padrao {i + 1}: esperados {n_sa} alvos, '
                       f'recebidos {len(p.alvo)}.')
    return n_ent, n_oc, n_sa


def _cfg_custom(req: RedeCustomRequest, n_ent, n_oc, n_sa):
    """Monta um dicionario no mesmo formato dos exercicios pre-definidos."""
    return {
        'id': 'custom',
        'titulo': 'Rede montada por voce',
        'subtitulo': f'Arquitetura {n_ent}-{n_oc}-{n_sa}',
        'slide': None,
        'taxa': req.taxa,
        'nota': '',
        'bias_compartilhado': False,
        'rotulos_entrada': req.rotulos_entrada or [f'x{i + 1}' for i in range(n_ent)],
        'rotulos_ocultos': req.rotulos_ocultos or [f'h{i + 1}' for i in range(n_oc)],
        'rotulos_saida': req.rotulos_saida or [f'y{i + 1}' for i in range(n_sa)],
        'pesos_oculta': req.pesos_oculta,
        'bias_oculta': req.bias_oculta,
        'pesos_saida': req.pesos_saida,
        'bias_saida': req.bias_saida,
    }


@router.post('/rede/trajetoria')
def rede_trajetoria(req: RedeCustomRequest):
    """
    Treina uma rede montada pelo usuario e devolve o mesmo formato de
    trajetoria dos exercicios pre-definidos — reaproveitando a linha do
    tempo do frontend sem nenhuma adaptacao.
    """
    n_ent, n_oc, n_sa = _validar_rede(req)
    cfg = _cfg_custom(req, n_ent, n_oc, n_sa)

    rede = RedeFeedforward(
        n_entradas=n_ent, n_ocultos=n_oc, n_saidas=n_sa,
        pesos_oculta=[l[:] for l in req.pesos_oculta],
        bias_oculta=list(req.bias_oculta),
        pesos_saida=[l[:] for l in req.pesos_saida],
        bias_saida=list(req.bias_saida),
    )
    padroes = [(p.entrada, p.alvo) for p in req.padroes]

    marcos = set(_indices_snapshots(req.epocas, req.n_snapshots))
    historico: list[float] = []
    snapshots: list[dict] = []

    def _snapshot(epoca: int, erro: float | None):
        snapshots.append({
            'epoca': epoca,
            'erro': erro,
            'pesos_oculta': [l[:] for l in rede.w_oculta],
            'bias_oculta': list(rede.b_oculta),
            'pesos_saida': [l[:] for l in rede.w_saida],
            'bias_saida': list(rede.b_saida),
            'saidas': [rede.prever(x) for x, _ in padroes],
        })

    if 0 in marcos:
        _snapshot(0, None)

    for epoca in range(1, req.epocas + 1):
        soma = 0.0
        for x, alvo in padroes:
            soma += rede.passo_treinamento(x, alvo, req.taxa)['erro_total']
        erro_medio = soma / len(padroes)
        historico.append(erro_medio)
        if epoca in marcos:
            _snapshot(epoca, erro_medio)

    return {
        'exercicio': 'custom',
        'tipo': 'epoca' if len(padroes) > 1 else 'passo-unico',
        'taxa': req.taxa,
        'epocas': req.epocas,
        'historico': historico,
        'snapshots': snapshots,
        'padroes': [{'entrada': x, 'alvo': a} for x, a in padroes],
        'alvos': [a for _, a in padroes],
        'arquitetura': _arquitetura(cfg),
        'config': {k: cfg[k] for k in
                   ('id', 'titulo', 'subtitulo', 'slide', 'nota',
                    'bias_compartilhado')},
    }


@router.post('/rede/memoria')
def rede_memoria(req: RedeCustomRequest):
    """
    Memoria de calculo de um passo de backprop da rede montada pelo usuario,
    usando o primeiro padrao — mesmo formato dos exercicios da aula.
    """
    n_ent, n_oc, n_sa = _validar_rede(req)
    cfg = _cfg_custom(req, n_ent, n_oc, n_sa)
    cfg['entradas'] = req.padroes[0].entrada
    cfg['alvo'] = req.padroes[0].alvo
    cfg['nota'] = ('Rede montada na interface — os pesos e a arquitetura vieram '
                   'dos controles do construtor, nao de um slide da aula.')
    return _traco_passo_unico(cfg)


# ---------------------------------------------------------------------------
# Lab 5.1 — reconhecimento de imagem 8x8
# ---------------------------------------------------------------------------
def _obter_rede_imagem():
    """Treina (uma unica vez) a rede 64-10-1 nos dois padroes de referencia."""
    global _rede_imagem
    if _rede_imagem is None:
        rede = RedeFeedforward(n_entradas=64, n_ocultos=10, n_saidas=1, semente=42)
        rede.treinar([achatar(HOMEM_PIXELS), achatar(GALINHA_PIXELS)],
                     [[0.0], [1.0]], taxa_aprendizado=0.5, epocas=4000)
        _rede_imagem = rede
    return _rede_imagem


@router.get('/imagem/padroes')
def padroes_imagem():
    """Padroes de referencia 8x8 e a saida da rede para cada um."""
    rede = _obter_rede_imagem()
    return {
        'homem': {'pixels': HOMEM_PIXELS, 'saida': rede.prever(achatar(HOMEM_PIXELS))[0]},
        'galinha': {'pixels': GALINHA_PIXELS, 'saida': rede.prever(achatar(GALINHA_PIXELS))[0]},
        'arquitetura': {'entradas': 64, 'ocultos': 10, 'saidas': 1},
    }


@router.post('/imagem/prever')
def prever_imagem(req: PixelsRequest):
    """Classifica um desenho 8x8, devolvendo a saida e as ativacoes ocultas."""
    if len(req.pixels) != 8 or any(len(linha) != 8 for linha in req.pixels):
        raise HTTPException(status_code=400,
                            detail='A grade deve ter exatamente 8x8 valores.')

    rede = _obter_rede_imagem()
    entrada = achatar(req.pixels)
    saida_oculta, saida_rede = rede.forward(entrada)
    out = saida_rede[0]

    if all(v == 0.0 for v in entrada):
        rotulo, classe = 'Tela em branco — desenhe algo', 'vazio'
    elif out < 0.35:
        rotulo, classe = 'Mais parecido com HOMEM', 'homem'
    elif out > 0.65:
        rotulo, classe = 'Mais parecido com GALINHA', 'galinha'
    else:
        rotulo, classe = 'Ambiguo entre os dois padroes', 'ambiguo'

    return {'saida': out, 'ativacoes_ocultas': saida_oculta,
            'rotulo': rotulo, 'classe': classe,
            'vazio': all(v == 0.0 for v in entrada)}


# ---------------------------------------------------------------------------
# Lab 5.1 — item (ii): comparativo no Iris
# ---------------------------------------------------------------------------
@router.get('/iris/comparar')
def comparar_iris(dataset: str = 'v1', atributos: str = 'todas',
                  proporcao: float = Query(0.7, ge=0.1, le=0.9),
                  camada_oculta: int = Query(8, ge=1, le=64),
                  max_iter: int = Query(3000, ge=100, le=20000)):
    """
    Item (ii): rede feedforward (scikit-learn — unico ponto do projeto com
    biblioteca de ML, permitido pelo enunciado) x Bayes Otimo x Naive Bayes,
    com todas as metricas e teste Z entre cada par.
    """
    try:
        from models.mlp_sklearn import prever_mlp_iris, treinar_mlp_iris
    except ImportError:
        raise HTTPException(
            status_code=503,
            detail='scikit-learn nao esta instalado. Rode: pip install scikit-learn')

    try:
        _, treino, teste = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(atributos, dataset)
    CLASSES = classes_de(dataset)
    gabarito = [d['classe'] for d in teste]

    modelo_mlp = treinar_mlp_iris(treino, idx, semente=42,
                                  camadas_ocultas=(camada_oculta,),
                                  max_iter=max_iter)
    relatorios = {
        'mlp': relatorio_completo(prever_mlp_iris(modelo_mlp, teste, idx),
                                  gabarito, CLASSES, 'Feedforward (MLP)'),
    }
    for chave, naive, nome in (('bayes', False, 'Bayes Otimo (QDA)'),
                               ('naive', True, 'Naive Bayes')):
        modelo = treinar_bayes(treino, idx, naive=naive)
        relatorios[chave] = relatorio_completo(
            [predizer_todas_classes_bayes(d['atributos'], modelo, idx)[1] for d in teste],
            gabarito, CLASSES, nome)

    chaves = list(relatorios.keys())
    comparacoes = []
    for i in range(len(chaves)):
        for j in range(i + 1, len(chaves)):
            ra, rb = relatorios[chaves[i]], relatorios[chaves[j]]
            z = z_kappa(ra['kappa'], ra['variancia_kappa'],
                        rb['kappa'], rb['variancia_kappa'])
            p = p_valor_z(z)
            comparacoes.append({'a': chaves[i], 'b': chaves[j],
                                'nome_a': ra['nome'], 'nome_b': rb['nome'],
                                'z': z, 'p': p, 'significativo': p < 0.05})

    return {'relatorios': relatorios, 'comparacoes': comparacoes,
            'n_treino': len(treino), 'n_teste': len(teste),
            'config': {'camada_oculta': camada_oculta, 'max_iter': max_iter,
                       'atributos': atributos}}
