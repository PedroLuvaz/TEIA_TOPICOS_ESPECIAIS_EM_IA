import matplotlib.pyplot as plt
from math_utils import coeficientes_superficie_decisao

# Cor + marcador por classe — pares distintos garantem que pontos sobrepostos
# permanecam identificaveis mesmo onde as nuvens se cruzam (overlap real
# entre versicolor e virginica nas petalas, e quase total nas sepalas).
CORES_CLASSE = {
    'setosa':     '#2196F3',
    'versicolor': '#4CAF50',
    'virginica':  '#F44336',
}

MARCADORES_CLASSE = {
    'setosa':     'o',   # circulo
    'versicolor': 's',   # quadrado
    'virginica':  '^',   # triangulo
}


def _rotulo_eixo(indice, nomes_atributos):
    if nomes_atributos and indice in nomes_atributos:
        return nomes_atributos[indice]
    return f'Atributo {indice}'


def plotar_superficie_decisao(pi, pj, dados_c1, dados_c2, classe_i, classe_j,
                              indices_atributos, dados_treino=None,
                              dados_teste=None, nomes_atributos=None,
                              titulo="Superficie de Decisao",
                              caminho_salvar=None):
    """
    Plota a fronteira de decisao binaria entre duas classes.
    Fundo colorido = regiao de decisao do classificador.
    Marcadores distintos por classe + bordas distintas por split (treino/teste).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    x1_c1 = [d['atributos'][indices_atributos[0]] for d in dados_c1]
    x2_c1 = [d['atributos'][indices_atributos[1]] for d in dados_c1]
    x1_c2 = [d['atributos'][indices_atributos[0]] for d in dados_c2]
    x2_c2 = [d['atributos'][indices_atributos[1]] for d in dados_c2]

    todos_x1 = x1_c1 + x1_c2
    todos_x2 = x2_c1 + x2_c2
    margem = 0.4
    x1_min = min(todos_x1) - margem
    x1_max = max(todos_x1) + margem
    x2_min = min(todos_x2) - margem
    x2_max = max(todos_x2) + margem

    w, b = coeficientes_superficie_decisao(pi, pj)

    cor_i = CORES_CLASSE.get(classe_i, '#2196F3')
    cor_j = CORES_CLASSE.get(classe_j, '#F44336')
    marcador_i = MARCADORES_CLASSE.get(classe_i, 'o')
    marcador_j = MARCADORES_CLASSE.get(classe_j, 's')

    # Colorir regioes de decisao — lado de pi satisfaz w*x + b > 0
    if abs(w[1]) > 1e-9:
        x1_fill = [x1_min, x1_max]
        y_fronteira = [(-w[0] * x - b) / w[1] for x in x1_fill]

        if w[1] > 0:
            ax.fill_between(x1_fill, y_fronteira, [x2_max, x2_max], alpha=0.13, color=cor_i)
            ax.fill_between(x1_fill, [x2_min, x2_min], y_fronteira, alpha=0.13, color=cor_j)
        else:
            ax.fill_between(x1_fill, [x2_min, x2_min], y_fronteira, alpha=0.13, color=cor_i)
            ax.fill_between(x1_fill, y_fronteira, [x2_max, x2_max], alpha=0.13, color=cor_j)

    ids_treino = set(id(d) for d in dados_treino) if dados_treino is not None else set()
    ids_teste = set(id(d) for d in dados_teste) if dados_teste is not None else set()
    destacar_split = bool(ids_treino or ids_teste)

    if destacar_split:
        for classe, dados_classe, cor, marcador in [
            (classe_i, dados_c1, cor_i, marcador_i),
            (classe_j, dados_c2, cor_j, marcador_j),
        ]:
            dados_classe_treino = [d for d in dados_classe if id(d) in ids_treino]
            x1_treino = [d['atributos'][indices_atributos[0]] for d in dados_classe_treino]
            x2_treino = [d['atributos'][indices_atributos[1]] for d in dados_classe_treino]
            ax.scatter(x1_treino, x2_treino, color=cor, marker=marcador,
                       label=f'{classe} (treino)',
                       edgecolors='white', linewidths=0.6,
                       s=60, alpha=0.85, zorder=3)

            dados_classe_teste = [d for d in dados_classe if id(d) in ids_teste]
            x1_teste = [d['atributos'][indices_atributos[0]] for d in dados_classe_teste]
            x2_teste = [d['atributos'][indices_atributos[1]] for d in dados_classe_teste]
            ax.scatter(x1_teste, x2_teste, label=f'{classe} (teste)',
                       color=cor, marker=marcador, edgecolors='black',
                       linewidths=1.0, s=80, alpha=0.95, zorder=4)
    else:
        ax.scatter(x1_c1, x2_c1, label=classe_i,
                   color=cor_i, marker=marcador_i,
                   edgecolors='white', linewidths=0.6, s=60, alpha=0.85, zorder=3)
        ax.scatter(x1_c2, x2_c2, label=classe_j,
                   color=cor_j, marker=marcador_j,
                   edgecolors='white', linewidths=0.6, s=60, alpha=0.85, zorder=3)

    # Prototipos — X grande com borda escura
    ax.scatter(pi[0], pi[1], color=cor_i, marker='X', s=250,
               edgecolors='black', linewidths=1.5, label=f'Media {classe_i}', zorder=5)
    ax.scatter(pj[0], pj[1], color=cor_j, marker='X', s=250,
               edgecolors='black', linewidths=1.5, label=f'Media {classe_j}', zorder=5)

    # Linha da fronteira d_ij(x) = 0
    if abs(w[1]) > 1e-9:
        coordenadas_x = [x1_min, x1_max]
        coordenadas_y = [(-w[0] * x - b) / w[1] for x in coordenadas_x]
        ax.plot(coordenadas_x, coordenadas_y, color='black', linestyle='--', linewidth=2,
                label='Fronteira dij(x)=0', zorder=5)
    else:
        ax.axvline(x=-b / w[0], color='black', linestyle='--', linewidth=2,
                   label='Fronteira dij(x)=0')

    ax.set_xlim(x1_min, x1_max)
    ax.set_ylim(x2_min, x2_max)
    ax.set_xlabel(_rotulo_eixo(indices_atributos[0], nomes_atributos))
    ax.set_ylabel(_rotulo_eixo(indices_atributos[1], nomes_atributos))
    ax.set_title(titulo)
    ax.legend(loc='best', fontsize=9, framealpha=0.92)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    if caminho_salvar:
        plt.savefig(caminho_salvar, dpi=110)
        print(f"Grafico salvo em: {caminho_salvar}")
    plt.close()


def plotar_dispersao_todas_classes(dados, indices_atributos, prototipos=None,
                                   dados_treino=None, dados_teste=None,
                                   nomes_atributos=None,
                                   titulo="Iris Dataset — Distribuicao das Classes",
                                   caminho_salvar=None):
    """
    Dispersao das 3 classes. Cada classe usa COR + MARCADOR distintos para que
    pontos sobrepostos (overlap real versicolor/virginica) sejam identificaveis
    mesmo onde se cruzam.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    classes = sorted(list(set(d['classe'] for d in dados)))

    # Ordem de desenho: setosa por ultimo (cluster mais isolado fica por cima)
    ordem = ['virginica', 'versicolor', 'setosa']
    classes_ordenadas = [c for c in ordem if c in classes] + [c for c in classes if c not in ordem]

    ids_treino = set(id(d) for d in dados_treino) if dados_treino is not None else set()
    ids_teste = set(id(d) for d in dados_teste) if dados_teste is not None else set()
    destacar_split = bool(ids_treino or ids_teste)

    for classe in classes_ordenadas:
        cor = CORES_CLASSE.get(classe, 'gray')
        marcador = MARCADORES_CLASSE.get(classe, 'o')

        if destacar_split:
            dados_classe = [d for d in dados if d['classe'] == classe and id(d) in ids_treino]
            x1 = [d['atributos'][indices_atributos[0]] for d in dados_classe]
            x2 = [d['atributos'][indices_atributos[1]] for d in dados_classe]
            ax.scatter(x1, x2, color=cor, marker=marcador,
                       label=f'{classe} (treino)',
                       edgecolors='white', linewidths=0.6,
                       s=60, alpha=0.85)

            dados_classe = [d for d in dados if d['classe'] == classe and id(d) in ids_teste]
            x1 = [d['atributos'][indices_atributos[0]] for d in dados_classe]
            x2 = [d['atributos'][indices_atributos[1]] for d in dados_classe]
            ax.scatter(x1, x2, label=f'{classe} (teste)', color=cor, marker=marcador,
                       edgecolors='black', linewidths=1.0, s=80, alpha=0.95)
            continue

        dados_classe = [d for d in dados if d['classe'] == classe]
        x1 = [d['atributos'][indices_atributos[0]] for d in dados_classe]
        x2 = [d['atributos'][indices_atributos[1]] for d in dados_classe]
        ax.scatter(x1, x2, label=classe, color=cor, marker=marcador,
                   edgecolors='white', linewidths=0.6, s=60, alpha=0.85)

    if prototipos:
        for classe, p in prototipos.items():
            cor = CORES_CLASSE.get(classe, 'black')
            ax.scatter(p[0], p[1], color=cor, marker='X',
                       s=220, edgecolors='black', linewidths=1.5,
                       zorder=6, label=f'Media {classe}')
            ax.text(p[0] + 0.05, p[1] + 0.04, classe,
                    fontweight='bold', fontsize=9, color='black')

    ax.set_xlabel(_rotulo_eixo(indices_atributos[0], nomes_atributos))
    ax.set_ylabel(_rotulo_eixo(indices_atributos[1], nomes_atributos))
    ax.set_title(titulo)
    ax.legend(loc='best', fontsize=9, framealpha=0.92)
    ax.grid(True, linestyle=':', alpha=0.6)

    plt.tight_layout()
    if caminho_salvar:
        plt.savefig(caminho_salvar, dpi=110)
        print(f"Grafico salvo em: {caminho_salvar}")
    plt.close()


def plotar_matriz_confusao(matriz, classes,
                           titulo="Matriz de Confusao",
                           caminho_salvar=None):
    """
    Heatmap da matriz de confusao. Linhas = classe real, colunas = predita.
    Diagonal = acertos. Fora da diagonal = erros (e onde o classificador
    confunde).
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    n = len(classes)
    valores = [[matriz[real][pred] for pred in classes] for real in classes]
    valor_max = max(max(linha) for linha in valores) if valores else 1
    if valor_max == 0:
        valor_max = 1

    im = ax.imshow(valores, cmap='Blues', vmin=0, vmax=valor_max)

    # Anotar cada celula com a contagem
    for i in range(n):
        for j in range(n):
            v = valores[i][j]
            cor_texto = 'white' if v > valor_max * 0.55 else 'black'
            peso = 'bold' if i == j else 'normal'
            ax.text(j, i, str(v), ha='center', va='center',
                    color=cor_texto, fontweight=peso, fontsize=12)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(classes, rotation=20, ha='right')
    ax.set_yticklabels(classes)
    ax.set_xlabel('Classe Predita')
    ax.set_ylabel('Classe Real')
    ax.set_title(titulo)

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Contagem')

    plt.tight_layout()
    if caminho_salvar:
        plt.savefig(caminho_salvar, dpi=110)
        print(f"Grafico salvo em: {caminho_salvar}")
    plt.close()
