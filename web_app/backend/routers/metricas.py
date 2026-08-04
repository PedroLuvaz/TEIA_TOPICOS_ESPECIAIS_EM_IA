"""
Lab 3 — Metricas Avancadas de Qualidade.

Duas frentes:
  · avaliar uma matriz de confusao arbitraria (editavel na interface)
  · comparar todos os classificadores do projeto no mesmo split, com
    teste Z de significancia de Kappa entre cada par
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from evaluation.metricas_avancadas import (_acerto_casual, _extrair_binario,
                                           acerto_global, fb_score, kappa,
                                           mcc, p_valor_z, relatorio_completo,
                                           tau, variancia_kappa,
                                           variancia_tau, z_kappa, z_tau)
from evaluation.validacao_cruzada import intervalo_confianca, validar_cruzado
from models.bayes_classifier import (predizer_todas_classes_bayes,
                                     treinar_bayes)
from models.classifier import predizer_todas_classes, treinar
from models.delta_rule import predizer_delta_ova, treinar_delta_ova

from .. import traco as T
from ..core import CLASSES, indices_de, obter_split

router = APIRouter(prefix='/api/metricas', tags=['metricas'])


class MatrizRequest(BaseModel):
    """Matriz de confusao: matriz[predito][real] = contagem."""
    matriz: dict[str, dict[str, int]]
    classes: list[str] | None = None
    nome: str = 'Matriz personalizada'


class ComparacaoMatrizesRequest(BaseModel):
    matriz_a: dict[str, dict[str, int]]
    matriz_b: dict[str, dict[str, int]]
    classes: list[str] | None = None
    nome_a: str = 'Classificador A'
    nome_b: str = 'Classificador B'


def _metricas_da_matriz(matriz, classes, nome):
    """Reconstroi o relatorio completo a partir de uma matriz ja pronta."""
    predicoes, gabarito = [], []
    for pred in classes:
        for real in classes:
            n = int(matriz.get(pred, {}).get(real, 0))
            predicoes.extend([pred] * n)
            gabarito.extend([real] * n)
    if not predicoes:
        raise HTTPException(status_code=400,
                            detail='A matriz de confusao esta vazia.')
    return relatorio_completo(predicoes, gabarito, classes, nome)


@router.post('/avaliar')
def avaliar(req: MatrizRequest):
    """Calcula todas as metricas de uma matriz de confusao informada."""
    classes = req.classes or CLASSES
    relatorio = _metricas_da_matriz(req.matriz, classes, req.nome)
    total = sum(sum(linha.values()) for linha in relatorio['matriz'].values())
    return {'relatorio': relatorio, 'total_amostras': total, 'classes': classes}


@router.post('/comparar-matrizes')
def comparar_matrizes(req: ComparacaoMatrizesRequest):
    """Teste Z de Kappa e de Tau entre duas matrizes de confusao."""
    classes = req.classes or CLASSES
    ra = _metricas_da_matriz(req.matriz_a, classes, req.nome_a)
    rb = _metricas_da_matriz(req.matriz_b, classes, req.nome_b)

    zk = z_kappa(ra['kappa'], ra['variancia_kappa'], rb['kappa'], rb['variancia_kappa'])
    zt = z_tau(ra['tau'], ra['variancia_tau'], rb['tau'], rb['variancia_tau'])

    return {
        'a': ra, 'b': rb,
        'kappa': {'z': zk, 'p': p_valor_z(zk), 'significativo': p_valor_z(zk) < 0.05},
        'tau': {'z': zt, 'p': p_valor_z(zt), 'significativo': p_valor_z(zt) < 0.05},
    }


@router.get('/comparar-modelos')
def comparar_modelos(dataset: str = 'v1', atributos: str = 'petalas',
                     proporcao: float = Query(0.7, ge=0.1, le=0.9)):
    """
    Avalia todos os classificadores multiclasse do projeto no mesmo split
    e aplica o teste Z de Kappa em cada par — a comparacao central do Lab 3.
    """
    try:
        _, treino, teste = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(atributos)
    gabarito = [d['classe'] for d in teste]
    relatorios = {}

    prototipos = treinar(treino, idx)
    relatorios['distancia_minima'] = relatorio_completo(
        [predizer_todas_classes(d['atributos'], prototipos, idx)[1] for d in teste],
        gabarito, CLASSES, 'Distancia Minima')

    pesos_ova, _, _ = treinar_delta_ova(treino, idx)
    relatorios['delta_ova'] = relatorio_completo(
        [predizer_delta_ova([d['atributos'][i] for i in idx], pesos_ova)[0] for d in teste],
        gabarito, CLASSES, 'Regra Delta OvA')

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
            comparacoes.append({
                'a': chaves[i], 'b': chaves[j],
                'nome_a': ra['nome'], 'nome_b': rb['nome'],
                'z': z, 'p': p, 'significativo': p < 0.05,
            })

    return {'relatorios': relatorios, 'comparacoes': comparacoes,
            'n_teste': len(teste), 'classes': CLASSES}


@router.get('/simular')
def simular(acerto: float = Query(0.9, ge=0.0, le=1.0),
            n_por_classe: int = Query(15, ge=1, le=500)):
    """
    Gera uma matriz de confusao sintetica com o acerto global desejado,
    distribuindo os erros uniformemente — util para explorar como Kappa e
    Tau reagem a diferentes niveis de acerto.
    """
    n_classes = len(CLASSES)
    acertos_por_classe = round(n_por_classe * acerto)
    erros = n_por_classe - acertos_por_classe

    matriz = {p: {r: 0 for r in CLASSES} for p in CLASSES}
    for i, real in enumerate(CLASSES):
        matriz[real][real] = acertos_por_classe
        if erros > 0:
            outras = [c for c in CLASSES if c != real]
            base, resto = divmod(erros, len(outras))
            for k, pred in enumerate(outras):
                matriz[pred][real] = base + (1 if k < resto else 0)

    relatorio = _metricas_da_matriz(matriz, CLASSES, f'Simulacao ({acerto:.0%})')
    return {'relatorio': relatorio, 'acerto_alvo': acerto,
            'n_por_classe': n_por_classe}


@router.post('/memoria')
def memoria(req: MatrizRequest):
    """
    Memoria de calculo do Lab 3 — equivalente web da janela LaTeX da GUI.
    Recebe a matriz de confusao e detalha cada metrica passo a passo.
    """
    classes = req.classes or CLASSES
    rel = _metricas_da_matriz(req.matriz, classes, req.nome)
    m = rel['matriz']
    n_total = sum(sum(linha.values()) for linha in m.values())
    diagonal = sum(m[c][c] for c in classes)
    ac = _acerto_casual(m, classes)

    # --- Acerto global ---
    termos_diag = ' + '.join(str(m[c][c]) for c in classes)

    # --- Acerto casual: produto dos marginais ---
    linhas_casual = []
    for c in classes:
        linha = sum(m[c][r] for r in classes)      # total predito como c
        coluna = sum(m[p][c] for p in classes)     # total real de c
        linhas_casual.append(
            f'{c:<12} predito={linha:<4} real={coluna:<4} '
            f'produto = {linha}·{coluna} = {linha * coluna}')
    soma_produtos = sum(
        sum(m[c][r] for r in classes) * sum(m[p][c] for p in classes)
        for c in classes)

    # --- OvR de uma classe de referencia ---
    foco = classes[0]
    vp, fp, fn, vn = _extrair_binario(m, foco, classes)
    mf = rel['por_classe'][foco]

    # --- Tau ---
    n_classes = len(classes)

    return T.montar(
        'Métricas Avançadas',
        f'{req.nome} · {n_total} amostras · {n_classes} classes',
        T.secao(
            'Matriz de confusão · base de todos os cálculos',
            T.texto('Linha = classe predita, coluna = classe real. A diagonal '
                    'concentra os acertos; tudo fora dela é erro.'),
            T.tabela(
                ['Predito \\ Real'] + [c for c in classes] + ['Total'],
                [[p] + [m[p][r] for r in classes]
                 + [sum(m[p][r] for r in classes)] for p in classes]
                + [['Total'] + [sum(m[p][r] for p in classes) for r in classes]
                   + [n_total]],
            ),
        ),
        T.secao(
            'Acerto Global (Ag)',
            T.texto('A proporção bruta de acertos — a métrica mais simples, e '
                    'a que mais engana quando as classes são desbalanceadas.'),
            T.formula(r'A_g \;=\; \frac{\sum_i x_{ii}}{N}'),
            T.ref(acerto_global),
            T.passos([
                f'Σ diagonal = {termos_diag} = {diagonal}',
                f'N          = {n_total}',
                f'Ag         = {diagonal} / {n_total} = {T.n(rel["acerto_global"], 6)}',
            ]),
            T.resultado(f'Ag = {T.pct(rel["acerto_global"])}',
                        tom='bom' if rel['acerto_global'] >= 0.9 else 'medio'),
        ),
        T.secao(
            'Acerto casual (Ac) e Coeficiente Kappa',
            T.texto('O acerto casual estima quanto um classificador acertaria '
                    'só por sorte, dados os totais marginais. O Kappa desconta '
                    'esse valor do acerto global.'),
            T.formula(r'A_c \;=\; \frac{1}{N^2}\sum_i x_{i+}\, x_{+i}'),
            T.formula(r'\kappa \;=\; \frac{A_g - A_c}{1 - A_c}'),
            T.ref(_acerto_casual),
            T.ref(kappa),
            T.passos(linhas_casual, titulo='marginais por classe'),
            T.passos([
                f'Σ produtos = {soma_produtos}',
                f'Ac  = {soma_produtos} / {n_total}² = {T.n(ac, 6)}',
                f'κ   = ({T.n(rel["acerto_global"], 6)} − {T.n(ac, 6)}) / '
                f'(1 − {T.n(ac, 6)})',
                f'    = {T.n(rel["kappa"], 6)}',
            ]),
            T.resultado(f'κ = {T.n(rel["kappa"], 6)}',
                        tom='bom' if rel['kappa'] > 0.8 else
                            'medio' if rel['kappa'] > 0.4 else 'ruim'),
            T.nota('Escala de Landis & Koch: κ > 0,8 excelente · 0,6–0,8 '
                   'substancial · 0,4–0,6 moderada · < 0,4 fraca. Note que o '
                   'Kappa fica sempre abaixo do acerto global.', tom='info'),
        ),
        T.secao(
            'Coeficiente Tau',
            T.texto('Alternativa ao Kappa que assume classes equiprováveis — '
                    'em vez dos marginais observados, usa 1/M como acerto '
                    'esperado ao acaso.'),
            T.formula(r'\tau \;=\; \frac{A_g - 1/M}{1 - 1/M}'),
            T.ref(tau),
            T.passos([
                f'M     = {n_classes} classes   →   1/M = {T.n(1 / n_classes, 6)}',
                f'τ     = ({T.n(rel["acerto_global"], 6)} − {T.n(1 / n_classes, 6)}) / '
                f'(1 − {T.n(1 / n_classes, 6)})',
                f'      = {T.n(rel["tau"], 6)}',
            ]),
            T.resultado(f'τ = {T.n(rel["tau"], 6)}'),
        ),
        T.secao(
            'Variâncias · necessárias para o teste Z',
            T.texto('Sem a variância não dá para dizer se a diferença entre '
                    'dois classificadores é real ou ruído amostral.'),
            T.ref(variancia_kappa),
            T.ref(variancia_tau),
            T.passos([
                f'Var(κ) = {rel["variancia_kappa"]:.8f}',
                f'Var(τ) = {rel["variancia_tau"]:.8f}',
                f'σ(κ)   = {rel["variancia_kappa"] ** 0.5:.6f}',
            ]),
        ),
        T.secao(
            'Teste Z de significância',
            T.texto('Compara dois classificadores treinados no mesmo conjunto. '
                    'Rejeita-se a equivalência quando |Z| > 1,96 (α = 5%).'),
            T.formula(r'Z \;=\; \frac{\kappa_1 - \kappa_2}'
                      r'{\sqrt{\operatorname{Var}(\kappa_1) + \operatorname{Var}(\kappa_2)}}'),
            T.ref(z_kappa),
            T.ref(p_valor_z),
            T.nota('Na aba "Comparar modelos" este teste é aplicado a cada par '
                   'de classificadores do projeto — é lá que se vê que '
                   'acurácias diferentes nem sempre significam modelos '
                   'diferentes.', tom='info'),
        ),
        T.secao(
            f'Extração binária One-vs-Rest · classe {foco}',
            T.texto('Cada métrica por classe nasce de uma matriz 2×2: a classe '
                    'de interesse contra todas as outras.'),
            T.ref(_extrair_binario),
            T.tabela(
                ['', f'Real = {foco}', 'Real = resto'],
                [[f'Predito = {foco}', vp, fp],
                 ['Predito = resto', fn, vn]],
                alinhamento=['esq', 'dir', 'dir'],
            ),
            T.passos([
                f'VP = {vp}   FP = {fp}   FN = {fn}   VN = {vn}',
                f'Acurácia do produtor (revocação) = VP/(VP+FN) = '
                f'{vp}/{vp + fn} = {T.n(mf["acuracia_produtor"], 6)}',
                f'Acurácia do usuário (precisão)   = VP/(VP+FP) = '
                f'{vp}/{vp + fp} = {T.n(mf["acuracia_usuario"], 6)}',
                f'Especificidade = VN/(VN+FP) = {vn}/{vn + fp} = '
                f'{T.n(mf["especificidade"], 6)}',
            ]),
        ),
        T.secao(
            'MCC e F-beta',
            T.texto('O F-beta pondera precisão e revocação; o MCC leva em '
                    'conta as quatro células da matriz 2×2, sendo robusto ao '
                    'desbalanceamento.'),
            T.formula(r'F_\beta = (1+\beta^2)\,\frac{P \cdot R}'
                      r'{\beta^2 P + R}'),
            T.formula(r'\text{MCC} = \frac{VP \cdot VN - FP \cdot FN}'
                      r'{\sqrt{(VP+FP)(VP+FN)(VN+FP)(VN+FN)}}'),
            T.ref(fb_score),
            T.ref(mcc),
            T.passos([
                f'F1 (β=1) = {T.n(mf["f1"], 6)}',
                f'F2 (β=2) = {T.n(mf["f2"], 6)}   (mais peso à revocação)',
                f'MCC      = {T.n(mf["mcc"], 6)}',
            ]),
            T.resultado(f'Classe {foco}: F1 = {T.n(mf["f1"], 4)}  ·  '
                        f'MCC = {T.n(mf["mcc"], 4)}'),
        ),
        cabecalho=[
            {'rotulo': 'Matriz', 'valor': req.nome},
            {'rotulo': 'Amostras', 'valor': str(n_total)},
            {'rotulo': 'Ag', 'valor': T.pct(rel['acerto_global'])},
            {'rotulo': 'κ', 'valor': T.n(rel['kappa'], 4)},
        ],
    )


@router.get('/validacao-cruzada')
def validacao_cruzada(dataset: str = 'v1', atributos: str = 'petalas',
                      k: int = Query(5, ge=2, le=10),
                      repeticoes: int = Query(5, ge=1, le=20)):
    """
    Avalia todos os classificadores por validacao cruzada k-fold estratificada,
    reportando media, desvio e intervalo de confianca.

    Um unico split de 70/30 deixa so 45 amostras de teste — o resultado varia
    muito com a semente sorteada. Aqui toda amostra e testada, e o desvio
    entre as dobras mostra o quanto o numero e confiavel.
    """
    try:
        dados, _, _ = obter_split(dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(atributos)

    def _dist_min(treino):
        return treinar(treino, idx)

    def _dist_min_pred(modelo, amostra):
        return predizer_todas_classes(amostra['atributos'], modelo, idx)[1]

    def _ova(treino):
        return treinar_delta_ova(treino, idx)[0]

    def _ova_pred(modelo, amostra):
        return predizer_delta_ova(
            [amostra['atributos'][i] for i in idx], modelo)[0]

    def _bayes(naive):
        return lambda treino: treinar_bayes(treino, idx, naive=naive)

    def _bayes_pred(modelo, amostra):
        return predizer_todas_classes_bayes(amostra['atributos'], modelo, idx)[1]

    modelos = {
        'distancia_minima': ('Distancia Minima', _dist_min, _dist_min_pred),
        'delta_ova': ('Regra Delta OvA', _ova, _ova_pred),
        'bayes': ('Bayes Otimo (QDA)', _bayes(False), _bayes_pred),
        'naive': ('Naive Bayes', _bayes(True), _bayes_pred),
    }

    resultados = {}
    for chave, (nome, treinar_fn, predizer_fn) in modelos.items():
        r = validar_cruzado(dados, treinar_fn, predizer_fn, CLASSES,
                            k=k, repeticoes=repeticoes)
        baixo, alto = intervalo_confianca(r['media'], r['desvio'],
                                          r['n_avaliacoes'])
        relatorio = relatorio_completo([], [], CLASSES, nome)
        relatorio['matriz'] = r['matriz']
        # Recalcula as metricas a partir da matriz acumulada de todas as dobras
        relatorio = _metricas_da_matriz(r['matriz'], CLASSES, nome)

        resultados[chave] = {
            'nome': nome,
            'media': r['media'],
            'desvio': r['desvio'],
            'minimo': r['minimo'],
            'maximo': r['maximo'],
            'ic_baixo': baixo,
            'ic_alto': alto,
            'acuracias': r['acuracias'],
            'n_avaliacoes': r['n_avaliacoes'],
            'relatorio': relatorio,
        }

    # Ordena para o ranking e monta comparacoes par a par pelo Kappa acumulado
    chaves = list(resultados.keys())
    comparacoes = []
    for i in range(len(chaves)):
        for j in range(i + 1, len(chaves)):
            ra = resultados[chaves[i]]['relatorio']
            rb = resultados[chaves[j]]['relatorio']
            z = z_kappa(ra['kappa'], ra['variancia_kappa'],
                        rb['kappa'], rb['variancia_kappa'])
            p = p_valor_z(z)
            comparacoes.append({
                'a': chaves[i], 'b': chaves[j],
                'nome_a': ra['nome'], 'nome_b': rb['nome'],
                'z': z, 'p': p, 'significativo': p < 0.05,
            })

    return {
        'resultados': resultados,
        'comparacoes': comparacoes,
        'config': {'k': k, 'repeticoes': repeticoes,
                   'atributos': atributos, 'dataset': dataset,
                   'n_amostras': len(dados),
                   'n_avaliacoes': k * repeticoes},
        'classes': CLASSES,
    }


@router.get('/curva-kappa')
def curva_kappa(n_por_classe: int = Query(15, ge=1, le=500),
                passos: int = Query(21, ge=3, le=101)):
    """
    Curva de Acerto Global x Kappa x Tau, varrendo o acerto de 0% a 100%.
    Mostra visualmente por que Kappa e mais rigoroso que o acerto bruto.
    """
    pontos = []
    for k in range(passos):
        alvo = k / (passos - 1)
        acertos = round(n_por_classe * alvo)
        erros = n_por_classe - acertos
        matriz = {p: {r: 0 for r in CLASSES} for p in CLASSES}
        for real in CLASSES:
            matriz[real][real] = acertos
            if erros > 0:
                outras = [c for c in CLASSES if c != real]
                base, resto = divmod(erros, len(outras))
                for i, pred in enumerate(outras):
                    matriz[pred][real] = base + (1 if i < resto else 0)
        pontos.append({
            'acerto_alvo': alvo,
            'acerto_global': acerto_global(matriz, CLASSES),
            'kappa': kappa(matriz, CLASSES),
            'tau': tau(matriz, CLASSES),
            'var_kappa': variancia_kappa(matriz, CLASSES),
            'var_tau': variancia_tau(matriz, CLASSES),
        })
    return {'pontos': pontos}
