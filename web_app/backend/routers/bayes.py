"""
Lab 4 — Classificador Otimo de Bayes (QDA) e Naive Bayes.

Alem do treino e das regioes de decisao quadraticas, expoe o teste de
normalidade multivariada (Henze-Zirkler / Mardia) calculado em Python puro.
"""
import math

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from core.math_utils import distancia_mahalanobis_quad
from evaluation.metricas_avancadas import (p_valor_z, relatorio_completo,
                                           z_kappa)
from evaluation.mvn_tester import calcular_mvn_python
from models.bayes_classifier import (predizer_todas_classes_bayes,
                                     treinar_bayes)

from .. import traco as T
from ..core import (CLASSES, CONFIG_ATRIBUTOS, indices_de, indices_plot,
                    limites_com_margem, malha, obter_split,
                    serializar_amostras)

router = APIRouter(prefix='/api/bayes', tags=['bayes'])


class PredicaoRequest(BaseModel):
    dataset: str = 'v1'
    atributos: str = 'petalas'
    naive: bool = False
    valores: list[float]


def _score(x_sel, params):
    """d_j(x) = -0.5·ln|Sigma_j| - 0.5·(x-m)^T Sigma^-1 (x-m)"""
    d2 = distancia_mahalanobis_quad(x_sel, params['media'], params['inv_cov'])
    return -0.5 * math.log(params['det']) - 0.5 * d2


@router.get('/treinar')
def treinar(dataset: str = 'v1', atributos: str = 'petalas',
            proporcao: float = Query(0.7, ge=0.1, le=0.9)):
    """Treina Bayes Otimo e Naive Bayes lado a lado, com teste Z entre eles."""
    try:
        dados, treino, teste = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(atributos)
    gabarito = [d['classe'] for d in teste]

    resultados = {}
    for chave, naive in (('bayes', False), ('naive', True)):
        modelo = treinar_bayes(treino, idx, naive=naive)
        preds = [predizer_todas_classes_bayes(d['atributos'], modelo, idx)[1]
                 for d in teste]
        nome = 'Bayes Otimo (QDA)' if not naive else 'Naive Bayes'
        resultados[chave] = {
            'relatorio': relatorio_completo(preds, gabarito, CLASSES, nome),
            'parametros': {
                c: {
                    'media': modelo[c]['media'],
                    'cov': modelo[c]['cov'],
                    'det': modelo[c]['det'],
                    'inv_cov': modelo[c]['inv_cov'],
                }
                for c in CLASSES
            },
        }

    ra, rb = resultados['bayes']['relatorio'], resultados['naive']['relatorio']
    z = z_kappa(ra['kappa'], ra['variancia_kappa'], rb['kappa'], rb['variancia_kappa'])
    p = p_valor_z(z)

    idx_plot = indices_plot(atributos)
    cfg = CONFIG_ATRIBUTOS[atributos]
    return {
        **resultados,
        'teste_z': {'z': z, 'p': p, 'significativo': p < 0.05},
        'amostras': serializar_amostras(dados, idx_plot, treino),
        'n_treino': len(treino),
        'n_teste': len(teste),
        'eixo_x': cfg['eixo_x'],
        'eixo_y': cfg['eixo_y'],
        'dimensoes': len(idx),
    }


@router.get('/regioes')
def regioes(dataset: str = 'v1', atributos: str = 'petalas',
            classificador: str = Query('bayes', pattern='^(bayes|naive)$'),
            proporcao: float = Query(0.7, ge=0.1, le=0.9),
            resolucao: int = Query(90, ge=20, le=200)):
    """
    Regioes de decisao quadraticas. Devolve tambem a superficie de diferenca
    de scores por par de classes, permitindo tracar a fronteira exata (nivel 0)
    com marching squares no frontend — sem "escadinhas".
    """
    try:
        dados, treino, _ = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(atributos)
    idx_plot = indices_plot(atributos)
    modelo = treinar_bayes(treino, idx, naive=(classificador == 'naive'))

    lim = limites_com_margem(dados, idx_plot)
    eixo_x, eixo_y = malha(lim, resolucao)

    fixos = {}
    for j in idx:
        if j not in idx_plot:
            valores = [d['atributos'][j] for d in dados]
            fixos[j] = sum(valores) / len(valores)

    grade = []
    superficies = {f'{a}|{b}': [] for a, b in
                   [('setosa', 'versicolor'), ('setosa', 'virginica'),
                    ('versicolor', 'virginica')]}

    for y in eixo_y:
        linha_classe = []
        linhas_par = {k: [] for k in superficies}
        for x in eixo_x:
            ponto = []
            for j in idx:
                if j == idx_plot[0]:
                    ponto.append(x)
                elif j == idx_plot[1]:
                    ponto.append(y)
                else:
                    ponto.append(fixos[j])

            scores = {c: _score(ponto, modelo[c]) for c in CLASSES}
            linha_classe.append(CLASSES.index(max(scores, key=scores.get)))
            for chave in superficies:
                a, b = chave.split('|')
                linhas_par[chave].append(scores[a] - scores[b])

        grade.append(linha_classe)
        for chave in superficies:
            superficies[chave].append(linhas_par[chave])

    return {'grade': grade, 'superficies': superficies,
            'eixo_x': eixo_x, 'eixo_y': eixo_y,
            'limites': lim, 'classes': CLASSES}


@router.get('/normalidade')
def normalidade(dataset: str = 'v1', atributos: str = 'petalas'):
    """
    Teste de aderencia a normalidade multivariada por classe
    (Henze-Zirkler e Mardia), implementado em Python puro.
    """
    try:
        dados, _, _ = obter_split(dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(atributos)
    try:
        resultado = calcular_mvn_python(dados, idx)
    except Exception as e:  # pragma: no cover — protege a UI de erro numerico
        raise HTTPException(status_code=500,
                            detail=f'Falha ao calcular MVN: {e}')

    return {'resultado': resultado, 'atributos': atributos,
            'indices': idx, 'n_features': len(idx)}


@router.get('/memoria')
def memoria(dataset: str = 'v1', atributos: str = 'petalas',
            classificador: str = Query('bayes', pattern='^(bayes|naive)$'),
            proporcao: float = Query(0.7, ge=0.1, le=0.9)):
    """
    Memoria de calculo do Lab 4 — equivalente web da janela LaTeX da GUI.
    Parametros estimados, discriminante quadratico e comparacao Bayes x Naive.
    """
    try:
        _, treino, teste = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(atributos)
    cfg = CONFIG_ATRIBUTOS[atributos]
    naive = classificador == 'naive'
    modelo = treinar_bayes(treino, idx, naive=naive)

    gabarito = [d['classe'] for d in teste]
    relatorios = {}
    for chave, nv, nome in (('bayes', False, 'Bayes Otimo (QDA)'),
                            ('naive', True, 'Naive Bayes')):
        mdl = treinar_bayes(treino, idx, naive=nv)
        preds = [predizer_todas_classes_bayes(d['atributos'], mdl, idx)[1]
                 for d in teste]
        relatorios[chave] = relatorio_completo(preds, gabarito, CLASSES, nome)

    ra, rb = relatorios['bayes'], relatorios['naive']
    z = z_kappa(ra['kappa'], ra['variancia_kappa'], rb['kappa'], rb['variancia_kappa'])
    p = p_valor_z(z)

    # Amostra de exemplo
    exemplo = teste[0]
    x_sel = [exemplo['atributos'][i] for i in idx]

    # --- Parametros estimados ---
    linhas_param = []
    for c in CLASSES:
        pm = modelo[c]
        linhas_param.append(f'classe {c}')
        linhas_param.append(f'  m  = {T.vetor(pm["media"])}')
        for i, linha in enumerate(pm['cov']):
            rotulo = '  Σ  = ' if i == 0 else '       '
            linhas_param.append(rotulo + T.vetor(linha, 6))
        linhas_param.append(f'  |Σ| = {pm["det"]:.6e}')
        linhas_param.append('')

    # --- Discriminante para a amostra ---
    linhas_disc, scores = [], {}
    for c in CLASSES:
        pm = modelo[c]
        d2 = distancia_mahalanobis_quad(x_sel, pm['media'], pm['inv_cov'])
        s = -0.5 * math.log(pm['det']) - 0.5 * d2
        scores[c] = s
        linhas_disc += [
            f'classe {c}',
            f'  (x − m) = {T.vetor([x_sel[k] - pm["media"][k] for k in range(len(x_sel))])}',
            f'  d²_M    = (x−m)ᵀ Σ⁻¹ (x−m) = {T.n(d2, 6)}',
            f'  ln|Σ|   = {T.n(math.log(pm["det"]), 6)}',
            f'  d_{c[:3]}(x) = −½·{T.n(math.log(pm["det"]), 4)} − ½·{T.n(d2, 4)} = {s:+.4f}',
            '',
        ]
    vencedor = max(scores, key=scores.get)

    return T.montar(
        'Bayes Ótimo (QDA)' if not naive else 'Naive Bayes',
        'Parâmetros estimados, discriminante quadrático e teste Z',
        T.secao(
            'Parâmetros estimados por classe',
            T.texto('Cada classe é modelada por uma normal multivariada: um '
                    'vetor médio e uma matriz de covariância estimados do '
                    'conjunto de treino.'),
            T.formula(r'm_j = \frac{1}{N_j}\sum_{x \in \omega_j} x'),
            T.formula(r'\Sigma_j = \frac{1}{N_j - 1}\sum_{x \in \omega_j}'
                      r'(x - m_j)(x - m_j)^{T}'),
            T.ref(treinar_bayes),
            T.passos(linhas_param),
            T.nota(
                'No Naive Bayes os termos fora da diagonal são forçados a '
                'zero — é a suposição de independência entre os atributos, '
                'que reduz o número de parâmetros a estimar.'
                if naive else
                'O Bayes Ótimo usa a matriz cheia, capturando a correlação '
                'entre os atributos. É isso que produz fronteiras quadráticas '
                'em vez de lineares.',
                tom='atencao' if naive else 'info'),
        ),
        T.secao(
            'Função discriminante · máximo a posteriori',
            T.texto('Com prioris iguais, o discriminante log-verossimilhança '
                    'reduz-se a dois termos: o volume da classe (ln|Σ|) e a '
                    'distância de Mahalanobis ao centro.'),
            T.formula(r'd_j(x) = -\tfrac{1}{2}\ln|\Sigma_j| '
                      r'- \tfrac{1}{2}(x - m_j)^{T}\Sigma_j^{-1}(x - m_j)'),
            T.formula(r'j^{*} = \arg\max_j \; d_j(x)'),
            T.ref(predizer_todas_classes_bayes),
            T.ref(distancia_mahalanobis_quad),
            T.texto(f'Substituindo com x = {T.vetor(x_sel, 2)} — primeira '
                    f'amostra de teste, classe real {exemplo["classe"]}:'),
            T.passos(linhas_disc),
            T.resultado(
                f'argmax  →  {vencedor.upper()}   ·   real: '
                f'{exemplo["classe"].upper()}   →  '
                f'{"acerto" if vencedor == exemplo["classe"] else "erro"}',
                tom='bom' if vencedor == exemplo['classe'] else 'ruim'),
        ),
        T.secao(
            'Por que as fronteiras são quadráticas',
            T.texto('Igualando dois discriminantes, os termos quadráticos só '
                    'se cancelam quando as covariâncias são iguais. Como cada '
                    'classe tem a sua, sobra um termo em x² — daí parábolas, '
                    'elipses e hipérboles.'),
            T.formula(r'd_i(x) - d_j(x) = -\tfrac{1}{2}x^{T}'
                      r'\left(\Sigma_i^{-1} - \Sigma_j^{-1}\right)x + \dots = 0'),
            T.nota('Se todas as classes compartilhassem a mesma covariância, o '
                   'termo quadrático sumiria e o classificador viraria linear '
                   '(LDA) — reencontrando as fronteiras retas do Lab 1.',
                   tom='info'),
        ),
        T.secao(
            'Bayes Ótimo × Naive Bayes · teste Z de Kappa',
            T.texto('Os dois classificadores são avaliados no mesmo conjunto '
                    'de teste; o teste Z diz se a diferença observada é '
                    'estatisticamente significativa.'),
            T.formula(r'Z = \frac{\kappa_1 - \kappa_2}'
                      r'{\sqrt{\operatorname{Var}(\kappa_1) + \operatorname{Var}(\kappa_2)}}'),
            T.ref(z_kappa),
            T.tabela(
                ['Classificador', 'Acerto Global', 'Kappa', 'Var(Kappa)'],
                [['Bayes Ótimo (QDA)', T.pct(ra['acerto_global']),
                  T.n(ra['kappa'], 6), f'{ra["variancia_kappa"]:.8f}'],
                 ['Naive Bayes', T.pct(rb['acerto_global']),
                  T.n(rb['kappa'], 6), f'{rb["variancia_kappa"]:.8f}']],
            ),
            T.passos([
                f'Z = ({T.n(ra["kappa"], 6)} − {T.n(rb["kappa"], 6)}) / '
                f'√({ra["variancia_kappa"]:.8f} + {rb["variancia_kappa"]:.8f})',
                f'  = {T.n(z, 6)}',
                f'p = {T.n(p, 6)}',
            ]),
            T.resultado(
                f'Z = {T.n(z, 4)}  ·  p = {T.n(p, 4)}  →  '
                + ('diferença significativa a 5%' if p < 0.05
                   else 'sem diferença significativa a 5%'),
                tom='ruim' if p < 0.05 else 'bom'),
        ),
        cabecalho=[
            {'rotulo': 'Classificador',
             'valor': 'Naive Bayes' if naive else 'Bayes Ótimo (QDA)'},
            {'rotulo': 'Base', 'valor': dataset},
            {'rotulo': 'Atributos', 'valor': cfg['nome']},
            {'rotulo': 'Treino / teste', 'valor': f'{len(treino)} / {len(teste)}'},
        ],
    )


@router.post('/predizer')
def predizer(req: PredicaoRequest):
    """Classifica um vetor arbitrario com Bayes ou Naive Bayes."""
    try:
        _, treino, _ = obter_split(req.dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(req.atributos)
    if len(req.valores) != len(idx):
        raise HTTPException(
            status_code=400,
            detail=f'Esperados {len(idx)} valores, recebidos {len(req.valores)}.')

    modelo = treinar_bayes(treino, idx, naive=req.naive)
    scores = {c: _score(req.valores, modelo[c]) for c in CLASSES}
    mahalanobis = {
        c: distancia_mahalanobis_quad(req.valores, modelo[c]['media'],
                                      modelo[c]['inv_cov'])
        for c in CLASSES
    }
    return {
        'classe': max(scores, key=scores.get),
        'scores': scores,
        'mahalanobis': mahalanobis,
        'valores': req.valores,
    }
