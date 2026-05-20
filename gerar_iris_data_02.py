"""
Gera data/iris_data_02.xlsx — versao do Iris com versicolor e virginica
linearmente separaveis em espaco de petalas [2,3].

Estrategia minimalista: apenas as amostras que cruzam (ou ficam muito
perto da) fronteira de decisao sao deslocadas, pelo menor valor necessario
para ficarem do lado correto com uma pequena margem de seguranca.
Todas as demais amostras ficam identicas ao dataset original.

Setosa e features de sepala [0,1] nao sao alteradas.
"""
import os, sys
RAIZ_ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(RAIZ_, 'src'))

import openpyxl
from core.data_loader import carregar_dados_iris

RAIZ = os.path.dirname(os.path.abspath(__file__))
ORIGEM  = os.path.join(RAIZ, 'data', 'Iris data.xls')
DESTINO = os.path.join(RAIZ, 'data', 'iris_data_02.xlsx')

INDICES_PETALAS = [2, 3]
MARGEM = 0.12   # folga no espaco discriminante — deslocamento fisico ~0.08 cm


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

    # w aponta de virginica para versicolor; score > 0 => versicolor
    w  = [m_ver[j] - m_vir[j] for j in range(len(INDICES_PETALAS))]
    b  = 0.5 * (norma2(m_ver) - norma2(m_vir))
    nw2 = norma2(w)

    novos = []
    n_aj  = 0

    for s in dados:
        ns = {'atributos': list(s['atributos']), 'classe': s['classe']}

        if s['classe'] in ('versicolor', 'virginica'):
            xp    = [s['atributos'][i] for i in INDICES_PETALAS]
            score = sum(w[j] * xp[j] for j in range(len(w))) - b

            if s['classe'] == 'versicolor' and score <= MARGEM:
                # desloca minimamente para o lado versicolor
                delta = (MARGEM - score) / nw2
                for j, idx in enumerate(INDICES_PETALAS):
                    ns['atributos'][idx] += delta * w[j]
                n_aj += 1

            elif s['classe'] == 'virginica' and score >= -MARGEM:
                # desloca minimamente para o lado virginica
                delta = (score + MARGEM) / nw2
                for j, idx in enumerate(INDICES_PETALAS):
                    ns['atributos'][idx] -= delta * w[j]
                n_aj += 1

        novos.append(ns)

    print(f'Amostras ajustadas: {n_aj}')
    return novos


def salvar_xlsx(dados, caminho):
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = 'Iris'
    for d in dados:
        row = [round(v, 4) for v in d['atributos']] + [d['classe']]
        ws.append(row)
    wb.save(caminho)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    dados_orig = carregar_dados_iris(ORIGEM)
    print(f'Original: {len(dados_orig)} amostras')

    dados_aj = ajustar(dados_orig)

    # Verificar separabilidade final
    m_ver = prototipo(dados_aj, 'versicolor', INDICES_PETALAS)
    m_vir = prototipo(dados_aj, 'virginica',  INDICES_PETALAS)
    w = [m_ver[j] - m_vir[j] for j in range(len(INDICES_PETALAS))]
    b = 0.5 * (norma2(m_ver) - norma2(m_vir))
    erros = sum(
        1 for d in dados_aj
        if d['classe'] in ('versicolor', 'virginica')
        and (
            (d['classe'] == 'versicolor' and
             sum(w[j]*d['atributos'][INDICES_PETALAS[j]] for j in range(2)) - b <= 0)
            or
            (d['classe'] == 'virginica' and
             sum(w[j]*d['atributos'][INDICES_PETALAS[j]] for j in range(2)) - b >= 0)
        )
    )

    # Mostrar range das features ajustadas
    for cl in ('versicolor', 'virginica'):
        pl = [d['atributos'][2] for d in dados_aj if d['classe'] == cl]
        pw = [d['atributos'][3] for d in dados_aj if d['classe'] == cl]
        print(f'{cl}: petal_len=[{min(pl):.2f}, {max(pl):.2f}]  '
              f'petal_wid=[{min(pw):.2f}, {max(pw):.2f}]')

    print(f'Erros pos-ajuste: {erros}  (esperado: 0)')
    salvar_xlsx(dados_aj, DESTINO)
    print(f'Salvo: {DESTINO}  ({len(dados_aj)} amostras)')
