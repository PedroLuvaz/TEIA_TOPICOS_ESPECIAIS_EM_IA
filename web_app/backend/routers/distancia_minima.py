"""
Lab 1 — Classificador de Distancia Minima.

Delega toda a matematica para `models/classifier.py` e `core/math_utils.py`
(Python puro, sem numpy), expondo protótipos, fronteiras lineares, regioes
de decisao e predicao de amostras arbitrarias.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.math_utils import (calcular_media, coeficientes_superficie_decisao,
                             distancia_euclidiana, discriminante,
                             produto_escalar)
from evaluation.metricas_avancadas import relatorio_completo
from models.classifier import predizer_todas_classes, treinar

from .. import traco as T
from ..core import (CLASSES, CONFIG_ATRIBUTOS, PARES_CLASSES, indices_de,
                    indices_plot, limites_com_margem, malha, obter_split,
                    serializar_amostras)

router = APIRouter(prefix='/api/distancia-minima', tags=['distancia-minima'])


class PredicaoRequest(BaseModel):
    dataset: str = 'v1'
    atributos: str = 'petalas'
    proporcao: float = 0.7
    valores: list[float]


def _contexto(dataset, atributos, proporcao):
    try:
        dados, treino, teste = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    idx = indices_de(atributos)
    prototipos = treinar(treino, idx)
    return dados, treino, teste, idx, prototipos


@router.get('/treinar')
def treinar_modelo(dataset: str = 'v1', atributos: str = 'petalas',
                   proporcao: float = Query(0.7, ge=0.1, le=0.9)):
    """
    Treina o classificador e devolve protótipos, metricas completas,
    equacoes das fronteiras e as amostras para plotagem.
    """
    dados, treino, teste, idx, prototipos = _contexto(dataset, atributos, proporcao)

    preds = [predizer_todas_classes(d['atributos'], prototipos, idx)[1] for d in teste]
    gabarito = [d['classe'] for d in teste]
    relatorio = relatorio_completo(preds, gabarito, CLASSES, 'Distancia Minima')

    # Equacoes das fronteiras para cada par de classes
    fronteiras = []
    for ci, cj in PARES_CLASSES:
        w, b = coeficientes_superficie_decisao(prototipos[ci], prototipos[cj])
        fronteiras.append({
            'classe_i': ci, 'classe_j': cj,
            'w': w, 'b': b,
            'equacao': _formatar_equacao(w, b, atributos),
        })

    idx_plot = indices_plot(atributos)
    cfg = CONFIG_ATRIBUTOS[atributos]
    return {
        'prototipos': {c: prototipos[c] for c in CLASSES},
        'prototipos_plot': {
            c: {'x': prototipos[c][idx.index(idx_plot[0])] if idx_plot[0] in idx else prototipos[c][0],
                'y': prototipos[c][idx.index(idx_plot[1])] if idx_plot[1] in idx else prototipos[c][1]}
            for c in CLASSES
        },
        'relatorio': relatorio,
        'fronteiras': fronteiras,
        'amostras': serializar_amostras(dados, idx_plot, treino),
        'n_treino': len(treino),
        'n_teste': len(teste),
        'eixo_x': cfg['eixo_x'],
        'eixo_y': cfg['eixo_y'],
        'dimensoes': len(idx),
    }


@router.get('/regioes')
def regioes(dataset: str = 'v1', atributos: str = 'petalas',
            proporcao: float = Query(0.7, ge=0.1, le=0.9),
            resolucao: int = Query(90, ge=20, le=200)):
    """
    Grade de regioes de decisao: para cada ponto da malha, o indice da classe
    vencedora. O frontend renderiza isso como um heatmap suave.
    """
    dados, treino, _, idx, prototipos = _contexto(dataset, atributos, proporcao)
    idx_plot = indices_plot(atributos)
    lim = limites_com_margem(dados, idx_plot)
    eixo_x, eixo_y = malha(lim, resolucao)

    # Para o modo 4D, as features nao plotadas ficam fixas na media global
    fixos = _valores_fixos(dados, idx, idx_plot)

    grade = []
    for y in eixo_y:
        linha = []
        for x in eixo_x:
            ponto = _montar_ponto(x, y, idx, idx_plot, fixos)
            scores = {c: discriminante(ponto, prototipos[c]) for c in CLASSES}
            linha.append(CLASSES.index(max(scores, key=scores.get)))
        grade.append(linha)

    return {'grade': grade, 'eixo_x': eixo_x, 'eixo_y': eixo_y,
            'limites': lim, 'classes': CLASSES}


@router.post('/predizer')
def predizer(req: PredicaoRequest):
    """Classifica um vetor arbitrario, devolvendo scores e distancias."""
    _, treino, _, idx, prototipos = _contexto(req.dataset, req.atributos, req.proporcao)

    if len(req.valores) != len(idx):
        raise HTTPException(
            status_code=400,
            detail=f'Esperados {len(idx)} valores para "{req.atributos}", '
                   f'recebidos {len(req.valores)}.')

    scores = {c: discriminante(req.valores, prototipos[c]) for c in CLASSES}
    distancias = {c: distancia_euclidiana(req.valores, prototipos[c]) for c in CLASSES}
    vencedor = max(scores, key=scores.get)

    return {
        'classe': vencedor,
        'scores': scores,
        'distancias': distancias,
        'prototipos': {c: prototipos[c] for c in CLASSES},
        'valores': req.valores,
    }


@router.get('/memoria')
def memoria(dataset: str = 'v1', atributos: str = 'petalas',
            proporcao: float = Query(0.7, ge=0.1, le=0.9),
            x1: float | None = None, x2: float | None = None):
    """
    Memoria de calculo do Lab 1 — equivalente web da janela LaTeX da GUI.
    Se x1/x2 nao forem informados, usa uma amostra de teste como exemplo.
    """
    dados, treino, teste, idx, prototipos = _contexto(dataset, atributos, proporcao)
    cfg = CONFIG_ATRIBUTOS[atributos]
    idx_plot = indices_plot(atributos)

    # Amostra de exemplo: a informada pelo usuario ou a primeira do teste
    if x1 is not None and x2 is not None:
        amostra = [x1, x2] if len(idx) == 2 else None
        origem = f'ponto informado ({T.n(x1, 2)}, {T.n(x2, 2)})'
        if amostra is None:
            base = teste[0]['atributos']
            amostra = [base[i] for i in idx]
            amostra[idx.index(idx_plot[0])] = x1
            amostra[idx.index(idx_plot[1])] = x2
            origem = f'ponto informado no plano ({T.n(x1, 2)}, {T.n(x2, 2)})'
        classe_real = None
    else:
        base = teste[0]
        amostra = [base['atributos'][i] for i in idx]
        classe_real = base['classe']
        origem = f'primeira amostra de teste (classe real: {classe_real})'

    n_treino = len(treino)
    n_por_classe = {c: sum(1 for d in treino if d['classe'] == c) for c in CLASSES}

    # --- Secao 1: prototipos ---
    linhas_prot = []
    for c in CLASSES:
        mj = prototipos[c]
        linhas_prot.append(f'N_{c[:3]} = {n_por_classe[c]} amostras de treino')
        linhas_prot.append(f'm_{c[:3]} = (1/{n_por_classe[c]}) · Σ x  =  {T.vetor(mj)}')
        linhas_prot.append('')

    # --- Secao 2: discriminante ---
    scores, linhas_disc = {}, []
    for c in CLASSES:
        mj = prototipos[c]
        xtmj = produto_escalar(amostra, mj)
        mjmj = produto_escalar(mj, mj)
        d = xtmj - 0.5 * mjmj
        scores[c] = d
        termos_x = '  +  '.join(
            f'{T.n(amostra[k], 2)}·{T.n(mj[k])}' for k in range(len(mj)))
        termos_m = '  +  '.join(f'{T.n(mj[k])}²' for k in range(len(mj)))
        linhas_disc += [
            f'classe {c}',
            f'  xᵀ·m_{c[:3]}   =  {termos_x}  =  {T.n(xtmj)}',
            f'  m_{c[:3]}ᵀ·m_{c[:3]} =  {termos_m}  =  {T.n(mjmj)}',
            f'  d_{c[:3]}(x)   =  {T.n(xtmj)} − ½·{T.n(mjmj)}  =  {d:+.4f}',
            '',
        ]
    vencedor = max(scores, key=scores.get)

    # --- Secao 3: equivalencia com a distancia euclidiana ---
    distancias = {c: distancia_euclidiana(amostra, prototipos[c]) for c in CLASSES}
    mais_perto = min(distancias, key=distancias.get)

    # --- Secao 4: fronteiras ---
    linhas_front = []
    for ci, cj in PARES_CLASSES:
        w, b = coeficientes_superficie_decisao(prototipos[ci], prototipos[cj])
        linhas_front += [
            f'{ci} × {cj}',
            f'  w = m_{ci[:3]} − m_{cj[:3]} = {T.vetor(w)}',
            f'  b = −½(‖m_{ci[:3]}‖² − ‖m_{cj[:3]}‖²) = {b:+.4f}',
            f'  {_formatar_equacao(w, b, atributos)}',
            '',
        ]

    return T.montar(
        'Distância Mínima', 'Protótipos, discriminante linear e fronteiras',
        T.secao(
            'Protótipos · vetores médios',
            T.texto('Cada classe é representada pelo seu centroide — a média '
                    'de todos os vetores de treino daquela classe.'),
            T.formula(r'm_j \;=\; \frac{1}{N_j}\sum_{x \in \omega_j} x'),
            T.ref(treinar),
            T.ref(calcular_media),
            T.passos(linhas_prot, titulo='substituição numérica'),
        ),
        T.secao(
            'Função discriminante · regra de decisão',
            T.texto('A função discriminante calcula um "score" para cada '
                    'classe. Vence a de maior score.'),
            T.formula(r'd_j(x) \;=\; x^{T} m_j \;-\; \tfrac{1}{2}\, m_j^{T} m_j'),
            T.formula(r'j^{*} \;=\; \arg\max_j \; d_j(x)'),
            T.ref(predizer_todas_classes),
            T.ref(discriminante),
            T.texto(f'Substituindo com x = {T.vetor(amostra, 2)} — {origem}:'),
            T.passos(linhas_disc),
            T.resultado(
                f'argmax  →  {vencedor.upper()}   (d = {scores[vencedor]:+.4f})',
                tom='bom' if (classe_real is None or vencedor == classe_real) else 'ruim'),
        ),
        T.secao(
            'Distância euclidiana · equivalência',
            T.texto('Maximizar o discriminante equivale a minimizar a '
                    'distância ao protótipo — o discriminante evita calcular '
                    'a raiz quadrada.'),
            T.formula(r'\arg\max_j \; d_j(x) \;\equiv\; \arg\min_j \; \lVert x - m_j \rVert'),
            T.ref(distancia_euclidiana),
            T.tabela(
                ['Classe', 'd_j(x)', '‖x − m_j‖'],
                [[c, f'{scores[c]:+.4f}', T.n(distancias[c])] for c in CLASSES],
            ),
            T.resultado(
                f'argmin da distância  →  {mais_perto.upper()}   '
                f'({"coincide" if mais_perto == vencedor else "diverge"} com o argmax)',
                tom='bom' if mais_perto == vencedor else 'ruim'),
        ),
        T.secao(
            'Fronteiras de decisão · pares de classes',
            T.texto('A fronteira entre duas classes é o lugar onde os dois '
                    'discriminantes se igualam — uma reta (2D) ou hiperplano.'),
            T.formula(r'd_i(x) = d_j(x) \;\Longrightarrow\; w^{T}x + b = 0,'
                      r'\quad w = m_i - m_j,\;\; b = -\tfrac{1}{2}\left('
                      r'\lVert m_i \rVert^2 - \lVert m_j \rVert^2\right)'),
            T.ref(coeficientes_superficie_decisao),
            T.passos(linhas_front),
            T.nota('Como todos os protótipos usam a mesma métrica, as '
                   'fronteiras são sempre lineares — é exatamente esse limite '
                   'que os laboratórios seguintes atacam.', tom='info'),
        ),
        cabecalho=[
            {'rotulo': 'Base', 'valor': dataset},
            {'rotulo': 'Atributos', 'valor': cfg['nome']},
            {'rotulo': 'Treino / teste', 'valor': f'{n_treino} / {len(teste)}'},
            {'rotulo': 'Dimensões', 'valor': str(len(idx))},
        ],
    )


# ---------------------------------------------------------------------------
def _valores_fixos(dados, idx, idx_plot):
    """Media global das features que nao estao no plano de plotagem."""
    fixos = {}
    for j in idx:
        if j not in idx_plot:
            valores = [d['atributos'][j] for d in dados]
            fixos[j] = sum(valores) / len(valores)
    return fixos


def _montar_ponto(x, y, idx, idx_plot, fixos):
    """Monta o vetor de features na ordem de `idx`, variando so o plano 2D."""
    ponto = []
    for j in idx:
        if j == idx_plot[0]:
            ponto.append(x)
        elif j == idx_plot[1]:
            ponto.append(y)
        else:
            ponto.append(fixos[j])
    return ponto


def _formatar_equacao(w, b, atributos):
    """Monta a string da equacao da fronteira: w1*x1 + w2*x2 + b = 0."""
    termos = [f'{wi:+.4f}·x{i + 1}' for i, wi in enumerate(w)]
    return f'{" ".join(termos)} {b:+.4f} = 0'
