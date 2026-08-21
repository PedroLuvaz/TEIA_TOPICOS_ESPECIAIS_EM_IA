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
from evaluation.testes_significancia import METRICAS, comparar, mcc_multiclasse
from evaluation.validacao_cruzada import intervalo_confianca, validar_cruzado

from .. import modelos as M
from .. import traco as T
from ..core import classes_de, indices_de, obter_split

router = APIRouter(prefix='/api/metricas', tags=['metricas'])

ROTULO_ATRIBUTOS = {
    'petalas': 'pétalas (comprimento × largura)',
    'sepalas': 'sépalas (comprimento × largura)',
    'todas': 'as 4 features',
}


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
    classes = req.classes or classes_de()
    relatorio = _metricas_da_matriz(req.matriz, classes, req.nome)
    total = sum(sum(linha.values()) for linha in relatorio['matriz'].values())
    return {'relatorio': relatorio, 'total_amostras': total, 'classes': classes}


@router.post('/comparar-matrizes')
def comparar_matrizes(req: ComparacaoMatrizesRequest):
    """Teste Z de Kappa e de Tau entre duas matrizes de confusao."""
    classes = req.classes or classes_de()
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
    Avalia TODOS os classificadores do catalogo no mesmo split e aplica o
    teste Z de Kappa em cada par — a comparacao central do Lab 3, agora
    incluindo o Perceptron OvA, a rede feedforward e a floresta do seminario.
    """
    try:
        _, _, teste = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    CLASSES = classes_de(dataset)
    gabarito = [d['classe'] for d in teste]
    preds = _predicoes_dos_classificadores(dataset, atributos, proporcao)

    relatorios = {chave: relatorio_completo(predicoes, gabarito, CLASSES, nome)
                  for chave, (nome, predicoes) in preds.items()}

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
    CLASSES = classes_de()
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
    classes = req.classes or classes_de()
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


def _predicoes_dos_classificadores(dataset, atributos, proporcao):
    """
    Predicoes de TODOS os modelos do catalogo no MESMO conjunto de teste.

    O pareamento e o que permite aplicar McNemar e o bootstrap pareado — os
    modelos acertam e erram as mesmas amostras dificeis. O resultado vem
    cacheado de `modelos.predicoes_de_todos`, senao a matriz de significancia
    retreinaria os sete classificadores para cada par testado.
    """
    try:
        return M.predicoes_de_todos(dataset, atributos, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get('/classificadores')
def listar_classificadores():
    """
    Classificadores e metricas disponiveis para os testes de significancia.

    A lista vem do mesmo catalogo da tela de classificacao — incluir um
    modelo la o torna comparavel aqui automaticamente.
    """
    return {
        'classificadores': [{'id': m['id'], 'nome': m['nome'],
                             'grupo': m['grupo']} for m in M.catalogo()],
        'metricas': [{'id': k, 'nome': v[0]} for k, v in METRICAS.items()],
    }


@router.get('/significancia')
def significancia(dataset: str = 'v1', atributos: str = 'petalas',
                  proporcao: float = Query(0.7, ge=0.1, le=0.9),
                  modelo_a: str = 'bayes', modelo_b: str = 'naive',
                  metrica: str = 'mcc',
                  n_reamostragens: int = Query(2000, ge=200, le=10000),
                  n_permutacoes: int = Query(2000, ge=200, le=10000)):
    """
    Testa se a diferenca entre dois classificadores e estatisticamente
    significativa — para o MCC e as demais metricas, nao so o Kappa.

    Roda tres testes complementares:
      · McNemar (pareado, sobre os acertos)
      · Bootstrap pareado (IC da diferenca da metrica escolhida)
      · Teste de permutacao (nao parametrico)
    """
    try:
        _, treino, teste = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if metrica not in METRICAS:
        raise HTTPException(
            status_code=400,
            detail=f'Métrica inválida. Use uma de: {", ".join(sorted(METRICAS))}.')

    CLASSES = classes_de(dataset)
    preds = _predicoes_dos_classificadores(dataset, atributos, proporcao)

    if modelo_a not in preds or modelo_b not in preds:
        raise HTTPException(
            status_code=400,
            detail=f'Classificador inválido. Use um de: {", ".join(sorted(preds))}.')
    if modelo_a == modelo_b:
        raise HTTPException(status_code=400,
                            detail='Escolha dois classificadores diferentes.')

    nome_a, pa = preds[modelo_a]
    nome_b, pb = preds[modelo_b]
    gabarito = [d['classe'] for d in teste]

    resultado = comparar(pa, pb, gabarito, CLASSES, metrica,
                         n_reamostragens, n_permutacoes)

    # Todas as metricas dos dois, para a tabela de contexto
    todas = {}
    for chave, fn in ((k, v[1]) for k, v in METRICAS.items()):
        todas[chave] = {
            'nome': METRICAS[chave][0],
            'a': fn(pa, gabarito, CLASSES),
            'b': fn(pb, gabarito, CLASSES),
        }

    # Teste Z classico do Kappa, para contrastar com o McNemar
    ra = relatorio_completo(pa, gabarito, CLASSES, nome_a)
    rb = relatorio_completo(pb, gabarito, CLASSES, nome_b)
    z = z_kappa(ra['kappa'], ra['variancia_kappa'], rb['kappa'], rb['variancia_kappa'])
    p_z = p_valor_z(z)

    return {
        **resultado,
        'modelo_a': {'id': modelo_a, 'nome': nome_a},
        'modelo_b': {'id': modelo_b, 'nome': nome_b},
        'metricas': todas,
        'teste_z_kappa': {'z': z, 'p': p_z, 'significativo': p_z < 0.05},
        'mcc_multiclasse': {
            'a': mcc_multiclasse(ra['matriz'], CLASSES),
            'b': mcc_multiclasse(rb['matriz'], CLASSES),
        },
        'config': {'dataset': dataset, 'atributos': atributos,
                   'proporcao': proporcao, 'n_teste': len(teste)},
    }


@router.get('/significancia/matriz')
def significancia_matriz(dataset: str = 'v1', atributos: str = 'petalas',
                         proporcao: float = Query(0.7, ge=0.1, le=0.9),
                         metrica: str = 'mcc',
                         n_reamostragens: int = Query(600, ge=200, le=3000)):
    """
    Matriz de significancia: testa TODOS os pares de classificadores de uma
    vez, para o comparativo geral pedido pelo professor.
    """
    try:
        _, treino, teste = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    if metrica not in METRICAS:
        raise HTTPException(status_code=400, detail='Métrica inválida.')

    CLASSES = classes_de(dataset)
    preds = _predicoes_dos_classificadores(dataset, atributos, proporcao)
    gabarito = [d['classe'] for d in teste]
    fn = METRICAS[metrica][1]

    modelos = list(preds)
    valores = {m: fn(preds[m][1], gabarito, CLASSES) for m in modelos}

    pares = []
    for i in range(len(modelos)):
        for j in range(i + 1, len(modelos)):
            a, b = modelos[i], modelos[j]
            r = comparar(preds[a][1], preds[b][1], gabarito, CLASSES,
                         metrica, n_reamostragens, n_reamostragens)
            pares.append({
                'a': a, 'b': b,
                'nome_a': preds[a][0], 'nome_b': preds[b][0],
                'valor_a': valores[a], 'valor_b': valores[b],
                'diferenca': r['bootstrap']['diferenca'],
                'ic_baixo': r['bootstrap']['ic_baixo'],
                'ic_alto': r['bootstrap']['ic_alto'],
                'p_mcnemar': r['mcnemar']['p_valor'],
                'p_permutacao': r['permutacao']['p_valor'],
                'significativo_mcnemar': r['mcnemar']['significativo'],
                'significativo_bootstrap': r['bootstrap']['significativo'],
                'significativo_permutacao': r['permutacao']['significativo'],
                'discordantes': r['mcnemar']['discordantes'],
            })

    return {
        'metrica': metrica,
        'nome_metrica': METRICAS[metrica][0],
        'modelos': [{'id': m, 'nome': preds[m][0], 'valor': valores[m]}
                    for m in sorted(modelos, key=lambda x: -valores[x])],
        'pares': pares,
        'config': {'dataset': dataset, 'atributos': atributos,
                   'n_teste': len(teste)},
    }


@router.get('/significancia/memoria')
def significancia_memoria(dataset: str = 'v1', atributos: str = 'petalas',
                          proporcao: float = Query(0.7, ge=0.1, le=0.9),
                          modelo_a: str = 'bayes', modelo_b: str = 'naive',
                          metrica: str = 'mcc',
                          n_reamostragens: int = Query(2000, ge=200, le=10000),
                          n_permutacoes: int = Query(2000, ge=200, le=10000)):
    """Memoria de calculo completa dos tres testes de significancia."""
    r = significancia(dataset, atributos, proporcao, modelo_a, modelo_b,
                      metrica, n_reamostragens, n_permutacoes)

    mn, bs, pm = r['mcnemar'], r['bootstrap'], r['permutacao']
    na, nb = r['modelo_a']['nome'], r['modelo_b']['nome']
    nome_m = r['nome_metrica']
    n_disc = mn['discordantes']

    # No regime exato a "estatistica" e min(b, c) — um inteiro, nao um qui2
    if mn['estatistica'] is None:
        est_mcnemar = '—'
    elif mn['metodo'] == 'binomial exato':
        est_mcnemar = f'min(b, c) = {int(mn["estatistica"])}'
    else:
        est_mcnemar = f'χ² = {T.n(mn["estatistica"], 4)}'

    return T.montar(
        'Testes de Significância Estatística',
        f'{na} × {nb} — métrica: {nome_m}',

        T.secao(
            'O problema: por que o teste Z do Kappa não basta',
            T.texto(
                'O teste Z clássico dos laboratórios compara dois Kappas '
                'somando as variâncias, o que só é válido se as duas '
                'avaliações forem INDEPENDENTES. Aqui os dois '
                'classificadores foram avaliados no MESMO conjunto de teste: '
                'eles erram as mesmas amostras difíceis, então as estimativas '
                'são pareadas e correlacionadas.'),
            T.formula(
                r'Z = \frac{\kappa_A - \kappa_B}'
                r'{\sqrt{\hat{\sigma}^2_{\kappa_A} + \hat{\sigma}^2_{\kappa_B}}}',
                titulo='Teste Z de Kappa (assume independência)',
                explicacao='Somar variâncias ignora a covariância positiva '
                           'entre os dois — o teste fica conservador demais.'),
            T.passos([
                f'Z            = {T.n(r["teste_z_kappa"]["z"], 4)}',
                f'p-valor      = {T.n(r["teste_z_kappa"]["p"], 6)}',
                f'significativo= '
                f'{"sim" if r["teste_z_kappa"]["significativo"] else "não"}',
            ], titulo='Resultado do teste Z clássico (para contraste)'),
            T.nota(
                'Os três testes abaixo são pareados e, por isso, corretos '
                'para este cenário. Compare o p-valor de cada um com o do '
                'teste Z acima.', tom='atencao',
                titulo='Independente × pareado'),
        ),

        T.secao(
            'Métrica escolhida — Coeficiente de Matthews (MCC)',
            T.texto(
                'O MCC é o coeficiente de correlação de Pearson entre a '
                'predição e o gabarito. Vale +1 na predição perfeita, 0 no '
                'acaso e −1 na inversão total. Por usar as quatro células da '
                'matriz, é robusto a classes desbalanceadas — mais do que a '
                'acurácia ou o F1.'),
            T.formula(
                r'\mathrm{MCC} = \frac{VP \cdot VN - FP \cdot FN}'
                r'{\sqrt{(VP+FP)(VP+FN)(VN+FP)(VN+FN)}}',
                titulo='MCC binário'),
            T.formula(
                r'\mathrm{MCC}_K = \frac{c\,s - \sum_k p_k t_k}'
                r'{\sqrt{(s^2 - \sum_k p_k^2)\,(s^2 - \sum_k t_k^2)}}',
                titulo='MCC multiclasse (Gorodkin, 2004)',
                explicacao='c = acertos totais, s = amostras, '
                           'p_k = preditos da classe k, t_k = reais da classe k.'),
            T.passos([
                f'MCC multiclasse  {na:<24} = '
                f'{T.n(r["mcc_multiclasse"]["a"], 4)}',
                f'MCC multiclasse  {nb:<24} = '
                f'{T.n(r["mcc_multiclasse"]["b"], 4)}',
            ]),
            T.tabela(
                ['Métrica', na, nb, 'Diferença'],
                [[m['nome'], T.n(m['a'], 4), T.n(m['b'], 4),
                  f'{m["a"] - m["b"]:+.4f}']
                 for m in r['metricas'].values()],
                titulo='Todas as métricas no mesmo conjunto de teste'),
            T.ref(mcc_multiclasse),
        ),

        T.secao(
            'Teste 1 — McNemar (pareado, exato)',
            T.texto(
                'Monta a tabela 2×2 dos acertos e olha SÓ para as amostras '
                'em que os dois discordam. As que ambos acertaram (a) e as '
                'que ambos erraram (d) não carregam informação sobre qual é '
                'o melhor, e por isso são descartadas.'),
            T.tabela(
                ['', f'{nb} acertou', f'{nb} errou'],
                [[f'{na} acertou', str(mn['a']), str(mn['b'])],
                 [f'{na} errou', str(mn['c']), str(mn['d'])]],
                titulo='Tabela de contingência dos acertos'),
            T.formula(
                r'\chi^2 = \frac{(|b - c| - 1)^2}{b + c}',
                titulo='Estatística de McNemar com correção de continuidade',
                explicacao='1 grau de liberdade. Com poucos discordantes '
                           '(b+c < 25) usa-se o binomial exato.'),
            T.formula(
                r'p = 2 \sum_{i=0}^{\min(b,c)} \binom{b+c}{i} (0{,}5)^{b+c}',
                titulo='Versão exata (binomial bilateral)',
                explicacao='Sob H₀ cada discordância é um cara-ou-coroa justo.'),
            T.passos([
                f'b (só {na} acertou)  = {mn["b"]}',
                f'c (só {nb} acertou)  = {mn["c"]}',
                f'discordantes b + c    = {n_disc}',
                f'método                = {mn["metodo"]}',
                f'estatística           = {est_mcnemar}',
                f'p-valor               = {T.n(mn["p_valor"], 6)}',
            ], titulo='Substituição numérica'),
            T.resultado(
                f'p = {T.n(mn["p_valor"], 6)} — '
                + ('diferença SIGNIFICATIVA (p < 0,05)'
                   if mn['significativo'] else
                   'não há evidência de diferença (p ≥ 0,05)'),
                tom='bom' if mn['significativo'] else 'medio'),
            T.nota(mn['observacao'], tom='info'),
        ),

        T.secao(
            'Teste 2 — Bootstrap pareado da diferença',
            T.texto(
                f'Reamostra {bs["n_reamostragens"]} vezes o conjunto de teste '
                'COM reposição, sempre levando o par (predição de A, predição '
                'de B) da mesma amostra. Em cada reamostragem recalcula a '
                f'{nome_m} dos dois e guarda a diferença. Os percentis 2,5 e '
                '97,5 dessa distribuição formam o intervalo de confiança.'),
            T.formula(
                r'\Delta = M(A) - M(B), \qquad '
                r'IC_{95\%} = [\Delta_{(2{,}5\%)},\ \Delta_{(97{,}5\%)}]',
                explicacao='Se o IC não contém zero, a diferença é '
                           'significativa a 5%.'),
            T.passos([
                f'{nome_m} de {na:<22} = {T.n(bs["metrica_a"], 4)}',
                f'{nome_m} de {nb:<22} = {T.n(bs["metrica_b"], 4)}',
                f'diferença observada Δ        = {bs["diferenca"]:+.4f}',
                f'erro padrão bootstrap        = {T.n(bs["erro_padrao"], 4)}',
                f'IC 95%                       = '
                f'[{bs["ic_baixo"]:+.4f}, {bs["ic_alto"]:+.4f}]',
                f'contém zero?                 = '
                f'{"sim" if bs["contem_zero"] else "não"}',
            ], titulo='Substituição numérica'),
            T.resultado(
                f'IC 95% = [{bs["ic_baixo"]:+.4f}, {bs["ic_alto"]:+.4f}] — '
                + ('não contém zero: diferença SIGNIFICATIVA'
                   if bs['significativo'] else
                   'contém zero: diferença compatível com o acaso'),
                tom='bom' if bs['significativo'] else 'medio'),
        ),

        T.secao(
            'Teste 3 — Permutação (não paramétrico)',
            T.texto(
                'Sob a hipótese nula os dois classificadores são '
                'intercambiáveis: trocar aleatoriamente as predições de A e B '
                'em cada amostra não deveria mudar nada. Fazendo essa troca '
                f'{pm["n_permutacoes"]} vezes constrói-se a distribuição da '
                'diferença sob H₀; o p-valor é a fração de permutações tão '
                'extremas quanto a observada.'),
            T.formula(
                r'p = \frac{1 + \#\{|\Delta^{*}| \geq |\Delta_{obs}|\}}{1 + B}',
                explicacao='O +1 no numerador e denominador evita p = 0 e '
                           'mantém o teste válido.'),
            T.passos([
                f'diferença observada  = {pm["diferenca_observada"]:+.4f}',
                f'permutações extremas = {pm["extremos"]} de '
                f'{pm["n_permutacoes"]}',
                f'p-valor              = {T.n(pm["p_valor"], 6)}',
            ], titulo='Substituição numérica'),
            T.resultado(
                f'p = {T.n(pm["p_valor"], 6)} — '
                + ('diferença SIGNIFICATIVA'
                   if pm['significativo'] else
                   'diferença não significativa'),
                tom='bom' if pm['significativo'] else 'medio'),
            T.ref(comparar),
        ),

        T.secao(
            'Veredito',
            T.tabela(
                ['Teste', 'Estatística', 'p-valor', 'Conclusão (α = 5%)'],
                [
                    ['Z de Kappa (independente)',
                     T.n(r['teste_z_kappa']['z'], 4),
                     T.n(r['teste_z_kappa']['p'], 6),
                     'significativo' if r['teste_z_kappa']['significativo']
                     else 'não significativo'],
                    [f'McNemar ({mn["metodo"]})',
                     est_mcnemar,
                     T.n(mn['p_valor'], 6),
                     'significativo' if mn['significativo']
                     else 'não significativo'],
                    ['Bootstrap pareado',
                     f'{bs["diferenca"]:+.4f}',
                     f'IC [{bs["ic_baixo"]:+.3f}, {bs["ic_alto"]:+.3f}]',
                     'significativo' if bs['significativo']
                     else 'não significativo'],
                    ['Permutação',
                     f'{pm["diferenca_observada"]:+.4f}',
                     T.n(pm['p_valor'], 6),
                     'significativo' if pm['significativo']
                     else 'não significativo'],
                ]),
            T.nota(
                'Os três testes pareados devem concordar na maioria dos '
                'casos. Quando discordam, o McNemar é o mais conservador '
                '(usa só os acertos, ignorando qual classe foi predita) e o '
                'bootstrap é o mais informativo, pois entrega o tamanho do '
                'efeito com incerteza, não apenas um sim/não.',
                tom='info', titulo='Como conciliar os três'),
        ),

        cabecalho=[
            {'rotulo': 'Métrica', 'valor': nome_m},
            {'rotulo': 'Amostras de teste', 'valor': str(r['n_amostras'])},
            {'rotulo': 'Atributos', 'valor': ROTULO_ATRIBUTOS.get(
                atributos, atributos)},
        ],
    )


# Modelos incluidos por padrao na validacao cruzada. A rede fica de fora
# porque treina k x repeticoes vezes: com 5 dobras e 5 repeticoes seriam 25
# treinamentos de backpropagation, dezenas de segundos numa tela interativa.
# Quem quiser inclui-la passa `modelos=...,mlp`.
CV_PADRAO = [m for m in M.ORDEM if m != 'mlp']


@router.get('/validacao-cruzada')
def validacao_cruzada(dataset: str = 'v1', atributos: str = 'petalas',
                      k: int = Query(5, ge=2, le=10),
                      repeticoes: int = Query(5, ge=1, le=20),
                      modelos: str | None = None):
    """
    Avalia os classificadores por validacao cruzada k-fold estratificada,
    reportando media, desvio e intervalo de confianca.

    Um unico split de 70/30 deixa so 45 amostras de teste — o resultado varia
    muito com a semente sorteada. Aqui toda amostra e testada, e o desvio
    entre as dobras mostra o quanto o numero e confiavel.

    `modelos`: lista separada por virgula (ex.: `bayes,floresta,mlp`). Sem o
    parametro, roda todos os do catalogo menos a rede feedforward.
    """
    try:
        dados, _, _ = obter_split(dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    try:
        idx = indices_de(atributos, dataset)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    CLASSES = classes_de(dataset)

    escolhidos = ([m.strip() for m in modelos.split(',') if m.strip()]
                  if modelos else list(CV_PADRAO))
    invalidos = [m for m in escolhidos if m not in M.MODELOS]
    if invalidos:
        raise HTTPException(
            status_code=400,
            detail=f'Modelo(s) inválido(s): {", ".join(invalidos)}. '
                   f'Use: {", ".join(M.ORDEM)}.')

    def _funcoes(mid):
        """Adapta o modelo do catalogo a assinatura de `validar_cruzado`."""
        cfg = M.info(mid)
        params = M.normalizar_parametros(mid, None)
        return (lambda treino: cfg['treinar'](treino, idx, **params),
                lambda modelo, amostra: cfg['predizer'](modelo,
                                                        amostra['atributos']))

    modelos_cv = {}
    for mid in escolhidos:
        treinar_fn, predizer_fn = _funcoes(mid)
        modelos_cv[mid] = (M.info(mid)['nome'], treinar_fn, predizer_fn)

    resultados = {}
    for chave, (nome, treinar_fn, predizer_fn) in modelos_cv.items():
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
    CLASSES = classes_de()
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
