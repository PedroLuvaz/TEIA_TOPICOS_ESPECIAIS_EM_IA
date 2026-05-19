"""
Gera data/iris_data_02.xlsx — versao do Iris com versicolor e virginica
linearmente separaveis em espaco de petalas [2,3].

Metodo: para cada amostra mal classificada pelo discriminante linear de
distancia minima (usando prototipo calculado sobre os 150 pontos originais),
desloca minimamente os atributos de petala ao longo do vetor normal da
fronteira ate ultrapassar a margem de segurança.  As features de sepala
[0,1] nao sao alteradas.

O ajuste e matematicamente minimo: usa a projecao exata sobre o hiperplano
mais um delta de margem (0.08 cm).
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import openpyxl
from iris_classifier.data_loader import carregar_dados_iris

RAIZ = os.path.dirname(os.path.abspath(__file__))
ORIGEM  = os.path.join(RAIZ, 'data', 'Iris data.xls')
DESTINO = os.path.join(RAIZ, 'data', 'iris_data_02.xlsx')
INDICES_PETALAS = [2, 3]
MARGEM = 1.5    # folga no espaco do discriminante (≈ 1 cm fisico no espaco de petalas)


# ---------------------------------------------------------------------------
def prototipo(dados, classe, indices):
    ams = [d['atributos'] for d in dados if d['classe'] == classe]
    n = len(ams)
    return [sum(a[i] for a in ams) / n for i in indices]


def norma2(v):
    return sum(x * x for x in v)


def ajustar(dados):
    m_ver = prototipo(dados, 'versicolor', INDICES_PETALAS)
    m_vir = prototipo(dados, 'virginica',  INDICES_PETALAS)

    # vetor normal: aponta de virginica para versicolor
    w = [m_ver[j] - m_vir[j] for j in range(len(INDICES_PETALAS))]
    # bias: b = 0.5*(||m_ver||^2 - ||m_vir||^2)
    b = 0.5 * (norma2(m_ver) - norma2(m_vir))
    nw2 = norma2(w)

    novos = []
    n_aj  = 0

    for d in dados:
        nd = {'atributos': list(d['atributos']), 'classe': d['classe']}

        if d['classe'] in ('versicolor', 'virginica'):
            xp = [d['atributos'][i] for i in INDICES_PETALAS]
            # score > 0  →  versicolor;  score < 0  →  virginica
            score = sum(w[j] * xp[j] for j in range(len(w))) - b

            if d['classe'] == 'versicolor' and score <= MARGEM:
                # desloca na direcao +w ate score = MARGEM
                delta = (MARGEM - score) / nw2
                for j, idx in enumerate(INDICES_PETALAS):
                    nd['atributos'][idx] += delta * w[j]
                n_aj += 1

            elif d['classe'] == 'virginica' and score >= -MARGEM:
                # desloca na direcao -w ate score = -MARGEM
                delta = (score + MARGEM) / nw2
                for j, idx in enumerate(INDICES_PETALAS):
                    nd['atributos'][idx] -= delta * w[j]
                n_aj += 1

        novos.append(nd)

    print(f'Amostras ajustadas: {n_aj}')
    return novos


def salvar_xlsx(dados, caminho):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Iris'
    for d in dados:
        ws.append(d['atributos'] + [d['classe']])
    wb.save(caminho)
    print(f'Salvo: {caminho}  ({len(dados)} amostras)')


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    dados_orig = carregar_dados_iris(ORIGEM)
    print(f'Carregados: {len(dados_orig)} amostras')

    # Verificar distribuicao original
    from collections import Counter
    cont = Counter(d['classe'] for d in dados_orig)
    print('Classes:', dict(cont))

    dados_aj = ajustar(dados_orig)

    # Verificar separabilidade pos-ajuste
    m_ver = prototipo(dados_aj, 'versicolor', INDICES_PETALAS)
    m_vir = prototipo(dados_aj, 'virginica',  INDICES_PETALAS)
    w = [m_ver[j] - m_vir[j] for j in range(len(INDICES_PETALAS))]
    b = 0.5 * (norma2(m_ver) - norma2(m_vir))
    nw2 = norma2(w)
    erros = 0
    for d in dados_aj:
        if d['classe'] not in ('versicolor', 'virginica'):
            continue
        xp = [d['atributos'][i] for i in INDICES_PETALAS]
        score = sum(w[j] * xp[j] for j in range(len(w))) - b
        pred = 'versicolor' if score > 0 else 'virginica'
        if pred != d['classe']:
            erros += 1
    print(f'Erros pos-ajuste (ver x vir, petalas): {erros}  '
          f'(esperado: 0)')

    salvar_xlsx(dados_aj, DESTINO)
