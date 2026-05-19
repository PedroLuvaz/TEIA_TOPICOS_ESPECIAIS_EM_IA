"""
Gera data/iris_data_02.xlsx — versao do Iris com versicolor e virginica
linearmente separaveis em espaco de petalas [2,3] e visualmente mais dispersas.

Estrategia (nao cria padrao artificial):
  1. Para versicolor E virginica em petalas:
       x_novo = centroide_classe + ESCALA * (x - centroide_classe)
     Isso aumenta a dispersao interna preservando a forma natural do cluster.
  2. Translacao rigida de cada classe em sentidos opostos (d_unit):
       versicolor  +=  -DESLOCAMENTO * d_unit
       virginica   +=  +DESLOCAMENTO * d_unit
     Isso cria o gap visual sem criar padrao diagonal artificial.
  3. Clamping para valores de petala realistas.
  4. Verificacao final: qualquer amostra que ainda cruze a fronteira e
     empurrada minimamente para o lado correto (margem 0.05).

Setosa e features de sepala [0,1] nao sao alteradas.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import openpyxl
from iris_classifier.data_loader import carregar_dados_iris

RAIZ = os.path.dirname(os.path.abspath(__file__))
ORIGEM  = os.path.join(RAIZ, 'data', 'Iris data.xls')
DESTINO = os.path.join(RAIZ, 'data', 'iris_data_02.xlsx')

INDICES_PETALAS = [2, 3]
ESCALA          = 1.25   # fator de expansao da dispersao interna de cada classe
DESLOCAMENTO    = 1.10   # cm no sentido de d_unit — separa as duas classes

# Limites realistas para features de petala
PL_MIN, PL_MAX = 2.0, 7.5   # comprimento da petala (cm)
PW_MIN, PW_MAX = 0.4, 3.0   # largura da petala (cm)
CLAMP = [
    (INDICES_PETALAS[0], PL_MIN, PL_MAX),
    (INDICES_PETALAS[1], PW_MIN, PW_MAX),
]

MARGEM_FINAL = 0.05   # margem de seguranca no espaco discriminante


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

    # Vetor unitario de separacao (de versicolor para virginica)
    d = [m_vir[j] - m_ver[j] for j in range(len(INDICES_PETALAS))]
    norm_d = norma2(d) ** 0.5
    d_unit = [x / norm_d for x in d]

    novos = []
    for s in dados:
        ns = {'atributos': list(s['atributos']), 'classe': s['classe']}

        if s['classe'] in ('versicolor', 'virginica'):
            is_ver = (s['classe'] == 'versicolor')
            m      = m_ver if is_ver else m_vir
            sinal  = -1    if is_ver else +1

            xp = [s['atributos'][i] for i in INDICES_PETALAS]
            for j, idx in enumerate(INDICES_PETALAS):
                # 1. Escala em torno do centroide da classe
                scaled = m[j] + ESCALA * (xp[j] - m[j])
                # 2. Translacao rigida
                shifted = scaled + sinal * DESLOCAMENTO * d_unit[j]
                ns['atributos'][idx] = shifted

        novos.append(ns)

    # 3. Clamping
    for ns in novos:
        for idx, vmin, vmax in CLAMP:
            ns['atributos'][idx] = max(vmin, min(vmax, ns['atributos'][idx]))

    # 4. Verificacao final — recalcula prototipos com dados ja transformados
    m_ver2 = prototipo(novos, 'versicolor', INDICES_PETALAS)
    m_vir2 = prototipo(novos, 'virginica',  INDICES_PETALAS)
    w  = [m_ver2[j] - m_vir2[j] for j in range(len(INDICES_PETALAS))]
    b  = 0.5 * (norma2(m_ver2) - norma2(m_vir2))
    nw2 = norma2(w)

    n_fix = 0
    for ns in novos:
        if ns['classe'] not in ('versicolor', 'virginica'):
            continue
        xp = [ns['atributos'][i] for i in INDICES_PETALAS]
        score = sum(w[j] * xp[j] for j in range(len(w))) - b

        if ns['classe'] == 'versicolor' and score <= MARGEM_FINAL:
            delta = (MARGEM_FINAL - score) / nw2
            for j, idx in enumerate(INDICES_PETALAS):
                ns['atributos'][idx] += delta * w[j]
            n_fix += 1
        elif ns['classe'] == 'virginica' and score >= -MARGEM_FINAL:
            delta = (score + MARGEM_FINAL) / nw2
            for j, idx in enumerate(INDICES_PETALAS):
                ns['atributos'][idx] -= delta * w[j]
            n_fix += 1

    if n_fix:
        print(f'  Correcoes pos-clamp: {n_fix}')
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
