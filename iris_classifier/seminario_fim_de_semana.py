"""
Seminario de Florestas Aleatorias — o dataset do fim de semana em escala.

Roda tres blocos:

  BLOCO A — validacao contra os slides
      Refaz, sobre as 10 instancias originais, as contas apresentadas:
      entropia da raiz, ganhos de Clima/Pais/Dinheiro e o erro OOB da
      floresta de B=3 arvores com mtry=2 e as amostras bootstrap fixadas.

  BLOCO B — a floresta em 1000 instancias
      Treina a FlorestaCategorica sobre `data/fim_de_semana_1000.csv`,
      com split estratificado 70/30, e reporta acerto, Kappa, MCC, erro OOB,
      importancia dos atributos e a distribuicao das raizes.

  BLOCO C — o mesmo CSV nos outros modulos
      Carrega o arquivo com a codificacao numerica e roda Distancia Minima,
      Regra Delta OvA, Bayes Otimo, Naive Bayes e a floresta CART do Lab —
      provando que o dataset serve ao projeto inteiro, nao so ao seminario.

Uso:
    python seminario_fim_de_semana.py
    python seminario_fim_de_semana.py --n-arvores 200
"""
import argparse
import os
import sys

BASE = os.path.dirname(os.path.abspath(__file__))
if BASE not in sys.path:
    sys.path.insert(0, BASE)

from data.data_loader import (FDS_CLASSES, FDS_NOMES_ATRIBUTOS,
                              carregar_fim_de_semana, split_estratificado)
from data.gerar_fim_de_semana import ORIGINAIS, validar_conceito
from evaluation.metricas_avancadas import relatorio_completo
from evaluation.testes_significancia import mcc_multiclasse
from models.bayes_classifier import predizer_todas_classes_bayes, treinar_bayes
from models.classifier import predizer_todas_classes, treinar
from models.delta_rule import predizer_delta_ova, treinar_delta_ova
from models.floresta_categorica import (FlorestaCategorica, entropia, ganho,
                                        predizer_arvore, construir_arvore)
from models.random_forest import treinar_floresta

CSV = os.path.join(os.path.dirname(BASE), 'data', 'fim_de_semana_1000.csv')
IDX = [0, 1, 2]

# Amostras bootstrap do slide 18 (1-indexadas, como na apresentacao)
BOOTSTRAPS_SLIDE = [
    [1, 2, 2, 2, 3, 4, 4, 5, 9, 10],
    [1, 1, 1, 2, 4, 4, 7, 9, 9, 10],
    [1, 3, 4, 4, 5, 7, 7, 8, 9, 10],
]


def _linha(titulo):
    print('\n' + '=' * 74)
    print(titulo)
    print('=' * 74)


def _amostras_originais():
    return [{'atributos': [c, p, d], 'classe': dec}
            for c, p, d, dec in ORIGINAIS]


# ===========================================================================
# Bloco A — validacao contra os slides
# ===========================================================================
def _conferir(rotulo, obtido, esperado, tolerancia=5e-3):
    """Compara um valor calculado com o que a apresentacao afirma."""
    ok = abs(obtido - esperado) <= tolerancia
    marca = 'OK  ' if ok else 'DIV '
    print(f'  [{marca}] {rotulo:<28} calculado {obtido:8.4f}   '
          f'slide {esperado:8.4f}')
    return ok


def bloco_a():
    _linha('BLOCO A — validacao contra os slides (10 instancias originais)')

    n_combinacoes = validar_conceito()
    print(f'Conceito extraido das 10 linhas do slide 9: reproduz todas elas '
          f'e cobre as {n_combinacoes} combinacoes possiveis.')

    dados = _amostras_originais()
    divergencias = []

    # --- slide 10: entropia da raiz e ganhos da arvore unica ---
    print('\nSlide 10 — arvore unica (ID3 sobre as 10 instancias):')
    _conferir('H(S)', entropia(dados), 1.5710)
    for i, esperado in ((0, 0.70), (1, 0.61), (2, 0.2816)):
        g, _ = ganho(dados, i, 'entropia')
        _conferir(f'Gain(S, {FDS_NOMES_ATRIBUTOS[i]})', g, esperado)

    vencedor = max((ganho(dados, i, 'entropia')[0], FDS_NOMES_ATRIBUTOS[i])
                   for i in IDX)[1]
    print(f'  raiz escolhida = {vencedor}   (slide 10: Clima)')

    # --- slides 19 a 21: as 3 arvores da floresta ---
    # Os pares sorteados por mtry=2 estao explicitos em cada slide.
    esperado_arvores = {
        1: {'h': 1.3610, 'oob': [6, 7, 8], 'pares': [1, 0],
            'ganhos': {1: 1.0000, 0: 0.7245}, 'raiz': 'Pais visitam?'},
        2: {'h': 0.4690, 'oob': [3, 5, 6, 8], 'pares': [1, 2],
            'ganhos': {1: 0.2690, 2: 0.1080}, 'raiz': 'Pais visitam?'},
        3: {'h': 1.2955, 'oob': [2, 6], 'pares': [1, 2],
            'ganhos': {1: 0.4200, 2: 0.6100}, 'raiz': 'Dinheiro'},
    }

    print('\nSlides 19-21 — as 3 arvores, com os bootstraps fixados no slide 18:')
    arvores = []
    for b, indices in enumerate(BOOTSTRAPS_SLIDE, start=1):
        amostra = [dados[i - 1] for i in indices]
        oob = sorted(set(range(1, 11)) - set(indices))
        esp = esperado_arvores[b]

        dist = {}
        for a in amostra:
            dist[a['classe']] = dist.get(a['classe'], 0) + 1

        print(f'\n  Arvore {b} — bootstrap {indices}')
        print(f'    distribuicao calculada: '
              + '  '.join(f'{c}={n}' for c, n in sorted(dist.items())))
        if not _conferir('H(S)', entropia(amostra), esp['h']):
            divergencias.append(f'Arvore {b}: H(S)')

        for i in esp['pares']:
            g, _ = ganho(amostra, i, 'entropia')
            if not _conferir(f'Gain({FDS_NOMES_ATRIBUTOS[i]})', g,
                             esp['ganhos'][i]):
                divergencias.append(f'Arvore {b}: Gain({FDS_NOMES_ATRIBUTOS[i]})')

        raiz = max((ganho(amostra, i, 'entropia')[0], FDS_NOMES_ATRIBUTOS[i])
                   for i in esp['pares'])[1]
        marca = 'OK  ' if raiz == esp['raiz'] else 'DIV '
        print(f'  [{marca}] {"raiz pelos 2 sorteados":<28} calculado '
              f'{raiz:>8}   slide {esp["raiz"]:>8}')
        if raiz != esp['raiz']:
            divergencias.append(f'Arvore {b}: atributo da raiz')

        marca = 'OK  ' if oob == esp['oob'] else 'DIV '
        print(f'  [{marca}] {"OOB":<28} calculado {str(oob):>14}   '
              f'slide {str(esp["oob"]):>14}')

        arvores.append(construir_arvore(amostra, list(IDX), 'entropia'))

    # --- slide 23: votacao de um padrao novo ---
    novo = ['Sol', 'Sim', 'Pobre']
    votos = {}
    for arvore in arvores:
        c = predizer_arvore(arvore, novo)
        votos[c] = votos.get(c, 0) + 1
    print(f'\nSlide 23 — padrao novo {novo}: votos {votos} '
          f'(slide: 3 votos em Cinema)')

    if divergencias:
        print('\n' + '-' * 74)
        print('ATENCAO — divergencias encontradas em relacao a apresentacao:')
        for d in divergencias:
            print(f'  · {d}')
        print(
            '\n  Causa: a contagem de classes das Arvores 2 e 3 no slide nao\n'
            '  corresponde as amostras bootstrap listadas no slide 18.\n'
            '    Bootstrap 2 = {1,1,1,2,4,4,7,9,9,10}: as instancias 2 e 10\n'
            '      sao Tenis, logo Cinema=8 e Tenis=2 (o slide diz 9 e 1).\n'
            '    Bootstrap 3 = {1,3,4,4,5,7,7,8,9,10}: a instancia 8 e\n'
            '      Compras, logo Cinema=7, Tenis=1, FicarEmCasa=1, Compras=1\n'
            '      (o slide diz Cinema=6, Tenis=3, FicarEmCasa=1, sem Compras).\n'
            '  A Arvore 1 e a arvore unica do slide 10 conferem exatamente.')
    else:
        print('\nTodas as contas conferem com a apresentacao.')

    return divergencias


# ===========================================================================
# Bloco B — a floresta em 1000 instancias
# ===========================================================================
def bloco_b(n_arvores):
    _linha(f'BLOCO B — FlorestaCategorica (ID3 multi-way) em 1000 instancias')

    dados = carregar_fim_de_semana(CSV)
    ruidosas = sum(d['ruido'] for d in dados)
    print(f'{len(dados)} instancias · {ruidosas} com rotulo trocado '
          f'({ruidosas / len(dados):.1%}) '
          f'-> teto teorico de acerto {1 - ruidosas / len(dados):.2%}')

    treino, teste = split_estratificado(dados, 0.7, semente=42)
    print(f'Split estratificado: {len(treino)} treino · {len(teste)} teste')

    floresta = FlorestaCategorica(
        n_arvores=n_arvores, criterio='entropia', semente=42,
        nomes_atributos=FDS_NOMES_ATRIBUTOS).treinar(treino, IDX)

    print(f'\nB = {n_arvores} arvores · mtry = {floresta._mtry(3)} de p = 3 '
          f'(regra sqrt(p) do slide 16)')

    gabarito = [d['classe'] for d in teste]
    predicoes = [floresta.predizer(d['atributos']) for d in teste]
    rel = relatorio_completo(predicoes, gabarito, FDS_CLASSES,
                             'Floresta Categorica')

    print(f'\nDesempenho no conjunto de teste:')
    print(f'  Acerto Global = {rel["acerto_global"]:.2%}')
    print(f'  Kappa         = {rel["kappa"]:.4f}')
    print(f'  Tau           = {rel["tau"]:.4f}')
    print(f'  MCC           = {mcc_multiclasse(rel["matriz"], FDS_CLASSES):.4f}')
    print(f'  Erro OOB      = {floresta.erro_oob:.2%} '
          f'({floresta.n_oob_avaliadas} instancias com pelo menos 1 voto OOB)')

    print('\nImportancia dos atributos (MDI, slide 26):')
    ordenadas = sorted(floresta.importancias.items(),
                       key=lambda kv: -kv[1]['normalizada'])
    for i, imp in ordenadas:
        barra = '#' * int(imp['normalizada'] * 40)
        print(f'  {FDS_NOMES_ATRIBUTOS[i]:<14} {imp["normalizada"]:6.1%}  '
              f'{barra}')

    print('\nAtributo usado na raiz de cada arvore (efeito do mtry, slide 22):')
    for nome, c in sorted(floresta.distribuicao_das_raizes().items(),
                          key=lambda kv: -kv[1]):
        print(f'  {nome:<14} {c:4d} arvores ({c / n_arvores:.1%})')

    print('\nMatriz de confusao (linha = predito, coluna = real):')
    largura = max(len(c) for c in FDS_CLASSES) + 2
    print(' ' * largura + ''.join(f'{c[:9]:>11}' for c in FDS_CLASSES))
    for pred in FDS_CLASSES:
        linha = f'{pred:<{largura}}'
        for real in FDS_CLASSES:
            linha += f'{rel["matriz"][pred][real]:>11}'
        print(linha)

    return rel


# ===========================================================================
# Bloco C — o mesmo CSV nos outros modulos
# ===========================================================================
def bloco_c():
    _linha('BLOCO C — o mesmo CSV nos demais modulos (codificacao numerica)')

    dados = carregar_fim_de_semana(CSV, numerico=True)
    treino, teste = split_estratificado(dados, 0.7, semente=42)
    gabarito = [d['classe'] for d in teste]
    print(f'Atributos numericos: Clima(0=Sol,1=Vento,2=Chuva) · '
          f'Pais(0=Nao,1=Sim) · Dinheiro(0=Pobre,1=Rico)')
    print(f'{len(treino)} treino · {len(teste)} teste\n')

    resultados = {}

    prototipos = treinar(treino, IDX)
    resultados['Distancia Minima'] = [
        predizer_todas_classes(d['atributos'], prototipos, IDX)[1] for d in teste]

    pesos, _, _ = treinar_delta_ova(treino, IDX)
    resultados['Regra Delta OvA'] = [
        predizer_delta_ova([d['atributos'][i] for i in IDX], pesos)[0]
        for d in teste]

    for nome, naive in (('Bayes Otimo (QDA)', False), ('Naive Bayes', True)):
        try:
            modelo = treinar_bayes(treino, IDX, naive=naive)
            resultados[nome] = [
                predizer_todas_classes_bayes(d['atributos'], modelo, IDX)[1]
                for d in teste]
        except Exception as e:
            print(f'  {nome}: falhou ({type(e).__name__}: {e})')

    cart = treinar_floresta(treino, IDX, n_arvores=100, semente=42)
    resultados['Floresta CART (Lab)'] = [
        cart.predizer(d['atributos']) for d in teste]

    print(f'{"Classificador":<22}{"Acerto":>10}{"Kappa":>10}{"MCC":>10}')
    print('-' * 52)
    linhas = []
    for nome, preds in resultados.items():
        rel = relatorio_completo(preds, gabarito, FDS_CLASSES, nome)
        linhas.append((rel['acerto_global'], nome, rel))
    for acerto, nome, rel in sorted(linhas, reverse=True):
        mcc = mcc_multiclasse(rel['matriz'], FDS_CLASSES)
        print(f'{nome:<22}{acerto:>9.2%}{rel["kappa"]:>10.4f}{mcc:>10.4f}')

    return resultados


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--n-arvores', type=int, default=100)
    p.add_argument('--bloco', choices=['a', 'b', 'c', 'todos'], default='todos')
    args = p.parse_args()

    if not os.path.exists(CSV):
        print(f'CSV nao encontrado em {CSV}.\n'
              f'Gere com: python -m data.gerar_fim_de_semana')
        return 1

    if args.bloco in ('a', 'todos'):
        bloco_a()
    if args.bloco in ('b', 'todos'):
        bloco_b(args.n_arvores)
    if args.bloco in ('c', 'todos'):
        bloco_c()

    print()
    return 0


if __name__ == '__main__':
    sys.exit(main())
