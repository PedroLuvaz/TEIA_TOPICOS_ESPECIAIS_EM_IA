"""
Lab 2 — Perceptron de Rosenblatt e Regra Delta (Widrow-Hoff / Adaline).

Cobre os quatro experimentos da aba desktop equivalente:
  · perceptron binario (par de classes)
  · regra delta binaria (par de classes)
  · regra delta One-vs-All (multiclasse)
  · XOR com regra delta — limite dos classificadores lineares
"""
from fastapi import APIRouter, HTTPException, Query

from evaluation.metricas_avancadas import relatorio_completo
from models.delta_rule import (acuracia_binaria_delta, predizer_delta,
                               predizer_delta_ova, treinar_delta_iris,
                               treinar_delta_ova, treinar_delta_xor)
from models.perceptron import (acuracia_binaria_perceptron,
                               predizer_perceptron, treinar_perceptron)

from .. import traco as T
from ..core import (classes_de, config_de, indices_de, indices_plot,
                    jitter_de, limites_com_margem, obter_split,
                    serializar_amostras)

router = APIRouter(prefix='/api/perceptron-delta', tags=['perceptron-delta'])


def _par_valido(dataset, classe_pos, classe_neg):
    """
    Ajusta o par de classes ao dataset escolhido.

    Ao trocar de dataset na interface, o seletor de classes pode chegar aqui
    ainda com o par do dataset anterior. Em vez de devolver 400 e piscar um
    erro na tela, caimos no primeiro par valido do dataset — a resposta ecoa
    `classe_pos`/`classe_neg`, entao a interface mostra o que foi usado.
    """
    classes = classes_de(dataset)
    if classe_pos in classes and classe_neg in classes and classe_pos != classe_neg:
        return classe_pos, classe_neg
    return classes[0], classes[1]


@router.get('/binario')
def binario(algoritmo: str = Query('perceptron', pattern='^(perceptron|delta)$'),
            dataset: str = 'v1', atributos: str = 'petalas',
            classe_pos: str = 'setosa', classe_neg: str = 'versicolor',
            taxa: float = Query(0.03, gt=0, le=1),
            max_epocas: int = Query(100, ge=1, le=2000),
            proporcao: float = Query(0.7, ge=0.1, le=0.9)):
    """Treina Perceptron ou Regra Delta para um par de classes."""
    try:
        dados, treino, teste = obter_split(dataset, proporcao)
        idx = indices_de(atributos, dataset)
        classe_pos, classe_neg = _par_valido(dataset, classe_pos, classe_neg)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if algoritmo == 'perceptron':
        w, historico, epocas = treinar_perceptron(
            treino, classe_pos, classe_neg, idx, taxa, max_epocas)
        rotulo_historico = 'erros'
        convergiu = historico[-1] == 0 if historico else False
        acc_treino = acuracia_binaria_perceptron(treino, w, classe_pos, classe_neg, idx)
        acc_teste = acuracia_binaria_perceptron(teste, w, classe_pos, classe_neg, idx)
        preditor = lambda x: predizer_perceptron(x, w)  # noqa: E731
        mapear = lambda y: classe_pos if y == 1 else classe_neg  # noqa: E731
    else:
        w, historico, epocas = treinar_delta_iris(
            treino, classe_pos, classe_neg, idx, taxa, max_epocas)
        rotulo_historico = 'mse'
        convergiu = bool(historico) and historico[-1] < 0.01
        acc_treino = acuracia_binaria_delta(treino, w, classe_pos, classe_neg, idx)
        acc_teste = acuracia_binaria_delta(teste, w, classe_pos, classe_neg, idx)
        preditor = lambda x: predizer_delta(x, w, classe_pos, classe_neg)  # noqa: E731
        mapear = lambda y: y  # noqa: E731

    # Matriz de confusao binaria no conjunto de teste
    teste_par = [d for d in teste if d['classe'] in (classe_pos, classe_neg)]
    preds, gabarito = [], []
    for d in teste_par:
        x_sel = [d['atributos'][i] for i in idx]
        preds.append(mapear(preditor(x_sel)))
        gabarito.append(d['classe'])
    relatorio = relatorio_completo(preds, gabarito, [classe_pos, classe_neg],
                                  'Perceptron' if algoritmo == 'perceptron' else 'Regra Delta')

    idx_plot = indices_plot(atributos, dataset)
    cfg = config_de(atributos, dataset)
    dados_par = [d for d in dados if d['classe'] in (classe_pos, classe_neg)]

    return {
        'algoritmo': algoritmo,
        'pesos': w,
        'historico': historico,
        'rotulo_historico': rotulo_historico,
        'epocas': epocas,
        'convergiu': convergiu,
        'acuracia_treino': acc_treino,
        'acuracia_teste': acc_teste,
        'relatorio': relatorio,
        'amostras': serializar_amostras(dados_par, idx_plot, treino,
                                        jitter=jitter_de(dataset)),
        'limites': limites_com_margem(dados_par, idx_plot),
        'classe_pos': classe_pos,
        'classe_neg': classe_neg,
        'eixo_x': cfg['eixo_x'],
        'eixo_y': cfg['eixo_y'],
        'bidimensional': len(idx) == 2,
    }


@router.get('/ova')
def ova(dataset: str = 'v1', atributos: str = 'petalas',
        taxa: float = Query(0.02, gt=0, le=1),
        max_epocas: int = Query(200, ge=1, le=2000),
        proporcao: float = Query(0.7, ge=0.1, le=0.9)):
    """Regra Delta multiclasse no esquema Um-Contra-Todos."""
    try:
        dados, treino, teste = obter_split(dataset, proporcao)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))

    idx = indices_de(atributos, dataset)
    pesos, historico, epocas = treinar_delta_ova(treino, idx, taxa, max_epocas)

    preds, gabarito = [], []
    for d in teste:
        x_sel = [d['atributos'][i] for i in idx]
        preds.append(predizer_delta_ova(x_sel, pesos)[0])
        gabarito.append(d['classe'])
    relatorio = relatorio_completo(preds, gabarito, classes_de(dataset),
                                   'Regra Delta OvA')

    idx_plot = indices_plot(atributos, dataset)
    cfg = config_de(atributos, dataset)
    return {
        'pesos': pesos,
        'historico': historico,
        'epocas': epocas,
        'relatorio': relatorio,
        'amostras': serializar_amostras(dados, idx_plot, treino,
                                        jitter=jitter_de(dataset)),
        'limites': limites_com_margem(dados, idx_plot),
        'eixo_x': cfg['eixo_x'],
        'eixo_y': cfg['eixo_y'],
        'bidimensional': len(idx) == 2,
    }


@router.get('/xor')
def xor(taxa: float = Query(0.02, gt=0, le=1),
        max_epocas: int = Query(300, ge=1, le=5000)):
    """
    XOR com Regra Delta — demonstra o limite teorico dos classificadores
    lineares: o MSE estaciona proximo de 0,25 e nunca zera.
    """
    w, historico = treinar_delta_xor(max_epocas=max_epocas, taxa_aprendizado=taxa)

    padroes = []
    for x1, x2, alvo in [(0.0, 0.0, 0), (0.0, 1.0, 1), (1.0, 0.0, 1), (1.0, 1.0, 0)]:
        net = w[0] + w[1] * x1 + w[2] * x2
        padroes.append({
            'x1': x1, 'x2': x2, 'alvo': alvo,
            'net': net,
            'previsto': 1 if net >= 0.5 else 0,
            'correto': (1 if net >= 0.5 else 0) == alvo,
        })

    return {
        'pesos': w,
        'historico': historico,
        'mse_final': historico[-1] if historico else None,
        'mse_teorico': 0.25,
        'padroes': padroes,
        'acertos': sum(1 for p in padroes if p['correto']),
    }


@router.get('/memoria')
def memoria(algoritmo: str = Query('perceptron', pattern='^(perceptron|delta)$'),
            dataset: str = 'v1', atributos: str = 'petalas',
            classe_pos: str = 'setosa', classe_neg: str = 'versicolor',
            taxa: float = Query(0.03, gt=0, le=1),
            max_epocas: int = Query(100, ge=1, le=2000),
            proporcao: float = Query(0.7, ge=0.1, le=0.9)):
    """
    Memoria de calculo do Lab 2 — equivalente web da janela LaTeX da GUI.
    Cobre o bias trick, a regra de atualizacao e a classificacao numerica.
    """
    try:
        _, treino, teste = obter_split(dataset, proporcao)
        idx = indices_de(atributos, dataset)
        cfg = config_de(atributos, dataset)
        classe_pos, classe_neg = _par_valido(dataset, classe_pos, classe_neg)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    ehp = algoritmo == 'perceptron'

    if ehp:
        w, historico, epocas = treinar_perceptron(
            treino, classe_pos, classe_neg, idx, taxa, max_epocas)
        convergiu = bool(historico) and historico[-1] == 0
        rotulo_hist = 'erros'
    else:
        w, historico, epocas = treinar_delta_iris(
            treino, classe_pos, classe_neg, idx, taxa, max_epocas)
        convergiu = bool(historico) and historico[-1] < 0.01
        rotulo_hist = 'MSE'

    # Amostra de exemplo: a primeira do teste que pertenca ao par
    exemplo = next((d for d in teste if d['classe'] in (classe_pos, classe_neg)),
                   teste[0])
    x_sel = [exemplo['atributos'][i] for i in idx]
    x_aug = [1.0] + x_sel
    net = sum(wi * xi for wi, xi in zip(w, x_aug))
    previsto = classe_pos if net >= 0 else classe_neg
    acertou = previsto == exemplo['classe']

    termos = '  +  '.join(f'{T.n(w[k], 4)}·{T.n(x_aug[k], 2)}'
                          for k in range(len(w)))

    marcos = sorted({0, epocas // 4, epocas // 2, len(historico) - 1})
    linhas_hist = [f'época {m + 1:>4}   {rotulo_hist} = {T.n(historico[m], 6)}'
                   for m in marcos if 0 <= m < len(historico)]

    return T.montar(
        'Perceptron de Rosenblatt' if ehp else 'Regra Delta (Adaline)',
        f'{classe_pos} (+1)  ×  {classe_neg} (−1)',
        T.secao(
            'Vetor aumentado · bias trick',
            T.texto('O bias vira um peso comum acrescentando uma entrada fixa '
                    'em 1 — assim a atualização trata bias e pesos de forma '
                    'idêntica.'),
            T.formula(r'x_{\text{aug}} = [\,1,\; x_1,\; \dots,\; x_n\,]'
                      r'\qquad w = [\,w_0,\; w_1,\; \dots,\; w_n\,]'),
            T.formula(r'\text{net} \;=\; w^{T} x_{\text{aug}} \;=\; '
                      r'w_0 + \sum_i w_i x_i'),
            T.passos([
                f'x       = {T.vetor(x_sel, 2)}',
                f'x_aug   = {T.vetor(x_aug, 2)}',
                f'mapeamento: {classe_pos} → d = +1    ·    {classe_neg} → d = −1',
            ]),
        ),
        T.secao(
            'Regra de atualização · ' + ('Rosenblatt' if ehp else 'Widrow-Hoff'),
            T.texto(
                'Os pesos só mudam quando a amostra é classificada errado — e a '
                'convergência é garantida apenas se as classes forem linearmente '
                'separáveis (Teorema de Rosenblatt).' if ehp else
                'A saída é linear (sem limiar) e o ajuste é proporcional ao erro, '
                'aplicado em todas as amostras a cada época — gradiente '
                'descendente sobre o erro quadrático médio.'),
            T.formula(
                r'y = \text{sgn}(w^{T}x_{\text{aug}}) = \begin{cases}'
                r'+1 & w^{T}x \ge 0\\ -1 & \text{caso contrário}\end{cases}'
                if ehp else
                r'y = w^{T} x_{\text{aug}} \qquad '
                r'\text{MSE} = \frac{1}{N}\sum_k (d_k - y_k)^2'),
            T.formula(r'w \;\leftarrow\; w \;+\; \eta\,(d - y)\, x_{\text{aug}}',
                      titulo='atualização'),
            T.ref(treinar_perceptron if ehp else treinar_delta_iris),
            T.passos(linhas_hist, titulo=f'{rotulo_hist} ao longo do treino'),
            T.resultado(
                f'Convergiu em {epocas} épocas — todas as amostras corretas.'
                if convergiu else
                f'Não convergiu em {epocas} épocas — {rotulo_hist} final = '
                f'{T.n(historico[-1], 6)}.',
                tom='bom' if convergiu else 'medio'),
            T.nota(
                'O Perceptron para assim que zera os erros; se as classes não '
                'forem separáveis por uma reta, ele oscila indefinidamente.'
                if ehp else
                'A Regra Delta nunca zera o MSE em dados não separáveis — ela '
                'converge para o menor erro possível, não para a separação '
                'perfeita. É esse o piso de 0,25 que aparece no XOR.',
                tom='atencao'),
        ),
        T.secao(
            'Pesos treinados · fronteira de decisão',
            T.passos(
                [f'w{k}{" (bias)" if k == 0 else ""} = {T.n(w[k], 6)}'
                 for k in range(len(w))],
                titulo=f'após {epocas} épocas (η = {taxa})'),
            T.formula(
                rf'{w[0]:+.4f} \;{w[1]:+.4f}\,x_1 \;{w[2]:+.4f}\,x_2 \;=\; 0',
                titulo='equação da reta') if len(w) == 3 else None,
            T.texto('A fronteira é o lugar onde net = 0: de um lado a rede '
                    'responde +1, do outro −1.'),
        ),
        T.secao(
            'Classificação · substituição numérica',
            T.texto(f'Aplicando os pesos treinados na amostra de teste '
                    f'{T.vetor(x_sel, 2)} — classe real: {exemplo["classe"]}.'),
            T.ref(predizer_perceptron if ehp else predizer_delta),
            T.passos([
                f'net = {termos}',
                f'    = {T.n(net, 6)}',
                f'sgn({T.n(net, 4)}) = {"+1" if net >= 0 else "−1"}  →  {previsto}',
            ]),
            T.resultado(
                f'Previsto: {previsto.upper()}   ·   Real: '
                f'{exemplo["classe"].upper()}   →  '
                f'{"acerto" if acertou else "erro"}',
                tom='bom' if acertou else 'ruim'),
        ),
        cabecalho=[
            {'rotulo': 'Algoritmo', 'valor': 'Perceptron' if ehp else 'Regra Delta'},
            {'rotulo': 'Base', 'valor': dataset},
            {'rotulo': 'Atributos', 'valor': cfg['nome']},
            {'rotulo': 'η', 'valor': str(taxa)},
            {'rotulo': 'Épocas', 'valor': f'{epocas} / {max_epocas}'},
        ],
    )
