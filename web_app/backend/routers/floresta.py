"""
Seminario — Florestas Aleatorias (Random Forests).

Implementacao propria em Python puro (`models/random_forest.py`), sem
scikit-learn: arvore CART com Gini/entropia, bagging, subespaco aleatorio de
atributos, erro out-of-bag e importancia dos atributos.
"""
from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from evaluation.metricas_avancadas import (p_valor_z, relatorio_completo,
                                           z_kappa)
from evaluation.validacao_cruzada import (intervalo_confianca,
                                          validar_cruzado)
from models.bayes_classifier import (predizer_todas_classes_bayes,
                                     treinar_bayes)
from models.classifier import predizer_todas_classes, treinar
from models.random_forest import (FlorestaAleatoria, caminho_decisao,
                                  construir_arvore, contar_classes,
                                  contar_nos, entropia, gini,
                                  limiares_candidatos, melhor_divisao,
                                  predizer_arvore, profundidade_arvore,
                                  treinar_floresta)

from .. import traco as T
from ..core import (CLASSES, CONFIG_ATRIBUTOS, NOMES_FEATURES, indices_de,
                    indices_plot, limites_com_margem, malha, obter_split,
                    serializar_amostras)

router = APIRouter(prefix='/api/floresta', tags=['floresta'])


class PredicaoRequest(BaseModel):
    dataset: str = 'v1'
    atributos: str = 'petalas'
    proporcao: float = 0.7
    n_arvores: int = 50
    criterio: str = 'gini'
    profundidade_max: int | None = None
    max_atributos: str | None = 'sqrt'
    valores: list[float]


def _config(n_arvores, criterio, profundidade_max, max_atributos,
            min_amostras_folha=1, semente=42):
    """Normaliza os parametros vindos da query em kwargs do modelo."""
    if criterio not in ('gini', 'entropia'):
        raise HTTPException(status_code=400,
                            detail="criterio deve ser 'gini' ou 'entropia'.")
    max_attr = max_atributos
    if max_attr in ('todos', 'none', '', None):
        max_attr = None
    elif max_attr not in ('sqrt', 'log2'):
        try:
            max_attr = int(max_attr)
        except (TypeError, ValueError):
            raise HTTPException(
                status_code=400,
                detail="max_atributos deve ser 'sqrt', 'log2', 'todos' ou um inteiro.")
    return {
        'n_arvores': n_arvores,
        'criterio': criterio,
        'profundidade_max': profundidade_max,
        'min_amostras_folha': min_amostras_folha,
        'max_atributos': max_attr,
        'semente': semente,
    }


@router.get('/treinar')
def treinar_endpoint(dataset: str = 'v1', atributos: str = 'petalas',
                     proporcao: float = Query(0.7, ge=0.1, le=0.9),
                     n_arvores: int = Query(50, ge=1, le=300),
                     criterio: str = 'gini',
                     profundidade_max: int | None = Query(None, ge=1, le=20),
                     max_atributos: str = 'sqrt',
                     min_amostras_folha: int = Query(1, ge=1, le=20)):
    """Treina a floresta e devolve metricas, OOB, importancias e as arvores."""
    try:
        dados, treino, teste = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(atributos)
    cfg = _config(n_arvores, criterio, profundidade_max, max_atributos,
                  min_amostras_folha)
    floresta = treinar_floresta(treino, idx, **cfg)

    gabarito = [d['classe'] for d in teste]
    preds = [floresta.predizer(d['atributos']) for d in teste]
    relatorio = relatorio_completo(preds, gabarito, CLASSES,
                                   'Floresta Aleatoria')

    # Uma arvore isolada, para comparar com o ensemble
    arvore_unica = construir_arvore(
        treino, idx, criterio, profundidade_max,
        min_amostras_folha=min_amostras_folha)
    preds_arvore = [predizer_arvore(arvore_unica, d['atributos']) for d in teste]
    relatorio_arvore = relatorio_completo(preds_arvore, gabarito, CLASSES,
                                          'Arvore unica')

    idx_plot = indices_plot(atributos)
    cfg_attr = CONFIG_ATRIBUTOS[atributos]

    return {
        'relatorio': relatorio,
        'relatorio_arvore_unica': relatorio_arvore,
        'oob': {
            'acuracia': floresta.acuracia_oob,
            'erro': floresta.erro_oob,
        },
        'importancias': [
            {'indice': a, 'nome': NOMES_FEATURES[a],
             'importancia': floresta.importancias.get(a, 0.0)}
            for a in sorted(floresta.importancias,
                            key=lambda x: -floresta.importancias[x])
        ],
        'arvores': floresta.resumo_arvores(),
        'config': {**cfg, 'atributos': atributos,
                   'n_atributos_por_no': floresta._n_atributos_sorteados(len(idx))},
        'amostras': serializar_amostras(dados, idx_plot, treino),
        'n_treino': len(treino),
        'n_teste': len(teste),
        'eixo_x': cfg_attr['eixo_x'],
        'eixo_y': cfg_attr['eixo_y'],
        'dimensoes': len(idx),
    }


@router.get('/arvore/{indice}')
def arvore(indice: int, dataset: str = 'v1', atributos: str = 'petalas',
           proporcao: float = Query(0.7, ge=0.1, le=0.9),
           n_arvores: int = Query(50, ge=1, le=300),
           criterio: str = 'gini',
           profundidade_max: int | None = Query(None, ge=1, le=20),
           max_atributos: str = 'sqrt',
           min_amostras_folha: int = Query(1, ge=1, le=20)):
    """Estrutura completa de uma arvore da floresta, para desenhar o diagrama."""
    try:
        _, treino, _ = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(atributos)
    cfg = _config(n_arvores, criterio, profundidade_max, max_atributos,
                  min_amostras_folha)
    floresta = treinar_floresta(treino, idx, **cfg)

    if not 0 <= indice < len(floresta.arvores):
        raise HTTPException(
            status_code=404,
            detail=f'Árvore {indice} não existe — a floresta tem '
                   f'{len(floresta.arvores)}.')

    arv = floresta.arvores[indice]
    nos, folhas = contar_nos(arv)
    return {
        'indice': indice,
        'arvore': arv.para_dict(),
        'profundidade': profundidade_arvore(arv),
        'nos': nos,
        'folhas': folhas,
        'amostras_oob': len(floresta.indices_oob[indice]),
        'amostras_unicas_bag': len(set(floresta.indices_bootstrap[indice])),
        'nomes_features': NOMES_FEATURES,
        'total_arvores': len(floresta.arvores),
    }


@router.get('/regioes')
def regioes(dataset: str = 'v1', atributos: str = 'petalas',
            proporcao: float = Query(0.7, ge=0.1, le=0.9),
            n_arvores: int = Query(50, ge=1, le=300),
            criterio: str = 'gini',
            profundidade_max: int | None = Query(None, ge=1, le=20),
            max_atributos: str = 'sqrt',
            min_amostras_folha: int = Query(1, ge=1, le=20),
            resolucao: int = Query(90, ge=20, le=160)):
    """
    Regioes de decisao da floresta. Ao contrario dos classificadores
    anteriores, as fronteiras aqui sao escadas alinhadas aos eixos — cada
    divisao da arvore e um corte do tipo `atributo <= limiar`.
    """
    try:
        dados, treino, _ = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(atributos)
    idx_plot = indices_plot(atributos)
    cfg = _config(n_arvores, criterio, profundidade_max, max_atributos,
                  min_amostras_folha)
    floresta = treinar_floresta(treino, idx, **cfg)

    lim = limites_com_margem(dados, idx_plot)
    eixo_x, eixo_y = malha(lim, resolucao)

    fixos = {}
    for j in range(len(NOMES_FEATURES)):
        if j not in idx_plot:
            valores = [d['atributos'][j] for d in dados]
            fixos[j] = sum(valores) / len(valores)

    grade, confianca = [], []
    for y in eixo_y:
        linha_c, linha_conf = [], []
        for x in eixo_x:
            ponto = []
            for j in range(len(NOMES_FEATURES)):
                if j == idx_plot[0]:
                    ponto.append(x)
                elif j == idx_plot[1]:
                    ponto.append(y)
                else:
                    ponto.append(fixos.get(j, 0.0))
            probs = floresta.probabilidades(ponto)
            vencedor = max(probs, key=probs.get)
            linha_c.append(CLASSES.index(vencedor))
            linha_conf.append(probs[vencedor])
        grade.append(linha_c)
        confianca.append(linha_conf)

    return {'grade': grade, 'confianca': confianca,
            'eixo_x': eixo_x, 'eixo_y': eixo_y,
            'limites': lim, 'classes': CLASSES}


@router.post('/predizer')
def predizer(req: PredicaoRequest):
    """Classifica um vetor e devolve a votacao arvore a arvore."""
    try:
        _, treino, _ = obter_split(req.dataset, req.proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(req.atributos)
    if len(req.valores) != len(idx):
        raise HTTPException(
            status_code=400,
            detail=f'Esperados {len(idx)} valores, recebidos {len(req.valores)}.')

    cfg = _config(req.n_arvores, req.criterio, req.profundidade_max,
                  req.max_atributos)
    floresta = treinar_floresta(treino, idx, **cfg)

    # Monta o vetor completo (as features fora do plano ficam na media)
    completo = [0.0] * len(NOMES_FEATURES)
    for pos, j in enumerate(idx):
        completo[j] = req.valores[pos]

    votos = floresta.votos(completo)
    probs = floresta.probabilidades(completo)
    vencedor = max(probs, key=probs.get)

    return {
        'classe': vencedor,
        'votos': votos,
        'probabilidades': probs,
        'total_arvores': len(floresta.arvores),
        'valores': req.valores,
    }


@router.get('/validacao-cruzada')
def validacao(dataset: str = 'v1', atributos: str = 'petalas',
              n_arvores: int = Query(50, ge=1, le=200),
              criterio: str = 'gini',
              profundidade_max: int | None = Query(None, ge=1, le=20),
              max_atributos: str = 'sqrt',
              k: int = Query(5, ge=2, le=10),
              repeticoes: int = Query(3, ge=1, le=10)):
    """
    Compara a floresta com os demais classificadores por validacao cruzada —
    a avaliacao honesta pedida no feedback sobre as metricas.
    """
    try:
        dados, _, _ = obter_split(dataset)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(atributos)
    cfg = _config(n_arvores, criterio, profundidade_max, max_atributos)

    def _floresta_treinar(treino):
        return treinar_floresta(treino, idx, **cfg)

    def _floresta_pred(modelo, amostra):
        return modelo.predizer(amostra['atributos'])

    def _arvore_treinar(treino):
        return construir_arvore(treino, idx, criterio, profundidade_max)

    def _arvore_pred(modelo, amostra):
        return predizer_arvore(modelo, amostra['atributos'])

    modelos = {
        'floresta': ('Floresta Aleatoria', _floresta_treinar, _floresta_pred),
        'arvore': ('Arvore de Decisao', _arvore_treinar, _arvore_pred),
        'distancia_minima': (
            'Distancia Minima',
            lambda tr: treinar(tr, idx),
            lambda m, a: predizer_todas_classes(a['atributos'], m, idx)[1]),
        'bayes': (
            'Bayes Otimo (QDA)',
            lambda tr: treinar_bayes(tr, idx, naive=False),
            lambda m, a: predizer_todas_classes_bayes(a['atributos'], m, idx)[1]),
    }

    resultados = {}
    for chave, (nome, treinar_fn, predizer_fn) in modelos.items():
        r = validar_cruzado(dados, treinar_fn, predizer_fn, CLASSES,
                            k=k, repeticoes=repeticoes)
        baixo, alto = intervalo_confianca(r['media'], r['desvio'],
                                          r['n_avaliacoes'])
        resultados[chave] = {
            'nome': nome, 'media': r['media'], 'desvio': r['desvio'],
            'minimo': r['minimo'], 'maximo': r['maximo'],
            'ic_baixo': baixo, 'ic_alto': alto,
            'n_avaliacoes': r['n_avaliacoes'],
            'matriz': r['matriz'],
        }

    chaves = list(resultados)
    comparacoes = []
    for i in range(len(chaves)):
        for j in range(i + 1, len(chaves)):
            ra = _relatorio_da_matriz(resultados[chaves[i]])
            rb = _relatorio_da_matriz(resultados[chaves[j]])
            z = z_kappa(ra['kappa'], ra['variancia_kappa'],
                        rb['kappa'], rb['variancia_kappa'])
            p = p_valor_z(z)
            comparacoes.append({
                'a': chaves[i], 'b': chaves[j],
                'nome_a': ra['nome'], 'nome_b': rb['nome'],
                'z': z, 'p': p, 'significativo': p < 0.05,
            })

    for chave in resultados:
        resultados[chave]['relatorio'] = _relatorio_da_matriz(resultados[chave])

    return {'resultados': resultados, 'comparacoes': comparacoes,
            'config': {'k': k, 'repeticoes': repeticoes,
                       'n_avaliacoes': k * repeticoes,
                       'n_amostras': len(dados), 'atributos': atributos,
                       'n_arvores': n_arvores, 'criterio': criterio},
            'classes': CLASSES}


def _relatorio_da_matriz(resultado):
    """Reconstroi o relatorio de metricas a partir da matriz acumulada."""
    matriz = resultado['matriz']
    predicoes, gabarito = [], []
    for pred in CLASSES:
        for real in CLASSES:
            n = int(matriz.get(pred, {}).get(real, 0))
            predicoes.extend([pred] * n)
            gabarito.extend([real] * n)
    return relatorio_completo(predicoes, gabarito, CLASSES, resultado['nome'])


# ---------------------------------------------------------------------------
# Memoria de calculo
# ---------------------------------------------------------------------------
@router.get('/memoria')
def memoria(dataset: str = 'v1', atributos: str = 'petalas',
            proporcao: float = Query(0.7, ge=0.1, le=0.9),
            n_arvores: int = Query(50, ge=1, le=300),
            criterio: str = 'gini',
            profundidade_max: int | None = Query(None, ge=1, le=20),
            max_atributos: str = 'sqrt'):
    """Teoria e substituicao numerica das Florestas Aleatorias."""
    try:
        dados, treino, teste = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(atributos)
    cfg_attr = CONFIG_ATRIBUTOS[atributos]
    cfg = _config(n_arvores, criterio, profundidade_max, max_atributos)
    floresta = treinar_floresta(treino, idx, **cfg)
    fn_imp = gini if criterio == 'gini' else entropia
    nome_imp = 'Gini' if criterio == 'gini' else 'Entropia'

    # --- Impureza do no raiz ---
    contagem = contar_classes(treino)
    n = len(treino)
    termos = '  +  '.join(f'({c}/{n})²' if criterio == 'gini'
                          else f'({c}/{n})·log₂({c}/{n})'
                          for c in contagem.values())
    imp_raiz = fn_imp(treino)

    # --- Melhor divisao na raiz (busca exaustiva, todos os atributos) ---
    divisao = melhor_divisao(treino, idx, criterio)
    linhas_divisao = []
    if divisao:
        atr, limiar, ganho, esq, dir_ = divisao
        linhas_divisao = [
            f'melhor divisão: {NOMES_FEATURES[atr]} <= {limiar:.4f}',
            f'  esquerda: {len(esq):>3} amostras   {nome_imp} = {fn_imp(esq):.6f}',
            f'  direita : {len(dir_):>3} amostras   {nome_imp} = {fn_imp(dir_):.6f}',
            '',
            f'  ganho = {imp_raiz:.6f} − [({len(esq)}/{n})·{fn_imp(esq):.6f} '
            f'+ ({len(dir_)}/{n})·{fn_imp(dir_):.6f}]',
            f'        = {ganho:.6f}',
        ]
        n_candidatos = sum(len(limiares_candidatos(treino, a)) for a in idx)
        linhas_divisao.append('')
        linhas_divisao.append(
            f'  (foram testados {n_candidatos} limiares candidatos ao todo)')

    # --- Bootstrap ---
    unicas = [len(set(b)) for b in floresta.indices_bootstrap]
    media_unicas = sum(unicas) / len(unicas)
    oob = [len(o) for o in floresta.indices_oob]
    media_oob = sum(oob) / len(oob)

    # --- Amostra de exemplo, com a votacao ---
    exemplo = teste[0]
    votos = floresta.votos(exemplo['atributos'])
    probs = floresta.probabilidades(exemplo['atributos'])
    vencedor = max(probs, key=probs.get)

    # --- Caminho numa arvore ---
    caminho = caminho_decisao(floresta.arvores[0], exemplo['atributos'])
    linhas_caminho = []
    for p in caminho:
        if p.get('folha'):
            linhas_caminho.append(
                f'  → folha: {p["classe"]}  (n={p["n_amostras"]}, '
                f'{nome_imp.lower()}={p["impureza"]:.4f})')
        else:
            linhas_caminho.append(
                f'  {NOMES_FEATURES[p["atributo"]]} = {p["valor"]:.2f} <= '
                f'{p["limiar"]:.4f} ?  {"sim → esquerda" if p["resultado"] else "não → direita"}')

    gabarito = [d['classe'] for d in teste]
    rel = relatorio_completo(
        [floresta.predizer(d['atributos']) for d in teste],
        gabarito, CLASSES, 'Floresta Aleatoria')

    imps = sorted(floresta.importancias.items(), key=lambda x: -x[1])

    return T.montar(
        'Florestas Aleatórias',
        f'{n_arvores} árvores · critério {nome_imp} · '
        f'{floresta._n_atributos_sorteados(len(idx))} atributo(s) por nó',
        T.secao(
            'Impureza · o que uma divisão tenta reduzir',
            T.texto('Uma divisão é boa quando separa as classes: os nós '
                    'filhos ficam mais "puros" que o pai. A impureza mede o '
                    'quanto as classes estão misturadas num nó.'),
            T.formula(r'G = 1 - \sum_k p_k^2' if criterio == 'gini'
                      else r'H = -\sum_k p_k \log_2 p_k'),
            T.ref(gini if criterio == 'gini' else entropia),
            T.passos([
                f'nó raiz: {n} amostras de treino',
                '  ' + '  ·  '.join(f'{c}: {v}' for c, v in sorted(contagem.items())),
                f'  {nome_imp} = ' + ('1 − [' if criterio == 'gini' else '−[')
                + termos + ']',
                f'         = {imp_raiz:.6f}',
            ]),
            T.nota('Impureza 0 significa nó puro (uma classe só). O máximo '
                   f'com 3 classes é {1 - 1/3:.4f} para Gini e '
                   f'{__import__("math").log2(3):.4f} para entropia.',
                   tom='info'),
        ),
        T.secao(
            'Ganho de impureza · escolhendo a divisão',
            T.texto('Para cada atributo, testam-se os pontos médios entre '
                    'valores consecutivos. Vence a divisão de maior ganho.'),
            T.formula(r'\text{ganho} = I(\text{pai}) - \left['
                      r'\frac{n_{esq}}{n} I(\text{esq}) + '
                      r'\frac{n_{dir}}{n} I(\text{dir})\right]'),
            T.ref(melhor_divisao),
            T.passos(linhas_divisao) if linhas_divisao else None,
        ),
        T.secao(
            'Bagging · bootstrap agregado',
            T.texto('Cada árvore treina numa amostra sorteada COM reposição, '
                    'do mesmo tamanho do conjunto original. Assim nenhuma '
                    'árvore vê exatamente os mesmos dados.'),
            T.formula(r'P(\text{amostra ficar de fora}) = '
                      r'\left(1 - \tfrac{1}{n}\right)^{n} '
                      r'\xrightarrow[n \to \infty]{} e^{-1} \approx 0{,}368'),
            T.passos([
                f'n = {n} amostras de treino, {n_arvores} árvores',
                f'  amostras únicas por bag: {media_unicas:.1f} '
                f'({media_unicas / n * 100:.1f}%)   — teórico: 63,2%',
                f'  amostras out-of-bag:     {media_oob:.1f} '
                f'({media_oob / n * 100:.1f}%)   — teórico: 36,8%',
            ]),
            T.nota('As amostras que ficam de fora servem para estimar o erro '
                   'de generalização sem separar um conjunto de validação — '
                   'é o erro out-of-bag.', tom='ok'),
        ),
        T.secao(
            'Subespaço aleatório de atributos',
            T.texto('Em cada nó, a busca considera apenas um subconjunto '
                    'sorteado de atributos. Sem isso, um atributo dominante '
                    'apareceria no topo de quase todas as árvores e elas '
                    'ficariam parecidas demais — o ensemble perderia força.'),
            T.formula(r'm = \lfloor\sqrt{p}\rfloor'
                      if max_atributos == 'sqrt' else r'm = p'),
            T.passos([
                f'p = {len(idx)} atributos disponíveis',
                f'm = {floresta._n_atributos_sorteados(len(idx))} '
                f'sorteados em cada nó  (regra: {max_atributos})',
            ]),
        ),
        T.secao(
            'Erro out-of-bag',
            T.texto('Cada amostra é votada apenas pelas árvores que NÃO a '
                    'viram no treino. É uma estimativa de generalização que '
                    'sai de graça do próprio processo de bagging.'),
            T.ref(FlorestaAleatoria._calcular_oob),
            T.passos([
                f'acurácia OOB = {floresta.acuracia_oob:.6f}  '
                f'({floresta.acuracia_oob * 100:.2f}%)',
                f'erro OOB     = {floresta.erro_oob:.6f}  '
                f'({floresta.erro_oob * 100:.2f}%)',
                '',
                f'acurácia no conjunto de teste = {rel["acerto_global"] * 100:.2f}%',
            ]),
            T.nota(
                'O OOB costuma ser mais conservador que a acurácia no '
                'conjunto de teste — e mais confiável, porque usa todas as '
                'amostras de treino em vez de apenas as 45 do teste.',
                tom='atencao'),
        ),
        T.secao(
            'Importância dos atributos',
            T.texto('Soma, sobre todos os nós de todas as árvores, do ganho '
                    'de impureza que cada atributo proporcionou — ponderado '
                    'pelo número de amostras que passaram pelo nó. '
                    'Normalizada para somar 1.'),
            T.formula(r'\text{imp}(j) = \frac{\sum_{\text{nós de } j} '
                      r'n_{\text{nó}} \cdot \text{ganho}}'
                      r'{\sum_{j\'} \sum_{\text{nós de } j\'} '
                      r'n_{\text{nó}} \cdot \text{ganho}}'),
            T.tabela(
                ['Atributo', 'Importância'],
                [[NOMES_FEATURES[a], f'{v * 100:.2f}%'] for a, v in imps],
            ),
        ),
        T.secao(
            'Votação · classificando uma amostra',
            T.texto(f'Amostra de teste: {T.vetor([exemplo["atributos"][i] for i in idx], 2)} '
                    f'— classe real: {exemplo["classe"]}.'),
            T.passos(linhas_caminho, titulo='caminho na árvore nº 1'),
            T.tabela(
                ['Classe', 'Votos', 'Proporção'],
                [[c, votos.get(c, 0), f'{probs[c] * 100:.1f}%'] for c in CLASSES],
            ),
            T.resultado(
                f'Voto da maioria → {vencedor.upper()}  '
                f'({votos.get(vencedor, 0)}/{n_arvores} árvores)  ·  '
                f'real: {exemplo["classe"].upper()}  →  '
                f'{"acerto" if vencedor == exemplo["classe"] else "erro"}',
                tom='bom' if vencedor == exemplo['classe'] else 'ruim'),
        ),
        cabecalho=[
            {'rotulo': 'Árvores', 'valor': str(n_arvores)},
            {'rotulo': 'Critério', 'valor': nome_imp},
            {'rotulo': 'Base', 'valor': dataset},
            {'rotulo': 'Atributos', 'valor': cfg_attr['nome']},
            {'rotulo': 'OOB', 'valor': f'{floresta.acuracia_oob * 100:.2f}%'},
        ],
    )
