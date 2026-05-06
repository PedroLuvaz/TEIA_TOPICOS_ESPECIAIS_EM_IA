import matplotlib.pyplot as plt
from math_utils import coeficientes_superficie_decisao

def plotar_superficie_decisao(pi, pj, dados_c1, dados_c2, classe_i, classe_j, indices_atributos, titulo="Superfície de Decisão", caminho_salvar=None):
    """
    Plota um gráfico de dispersão 2D de duas classes e a fronteira de decisão entre elas.
    """
    plt.figure(figsize=(10, 6))
    
    # Extrair atributos para plotagem
    x1_c1 = [d['atributos'][indices_atributos[0]] for d in dados_c1]
    x2_c1 = [d['atributos'][indices_atributos[1]] for d in dados_c1]
    
    x1_c2 = [d['atributos'][indices_atributos[0]] for d in dados_c2]
    x2_c2 = [d['atributos'][indices_atributos[1]] for d in dados_c2]
    
    plt.scatter(x1_c1, x2_c1, label=classe_i, color='blue', alpha=0.7)
    plt.scatter(x1_c2, x2_c2, label=classe_j, color='red', alpha=0.7)
    
    # Plotar protótipos
    plt.scatter(pi[0], pi[1], color='darkblue', marker='X', s=200, label=f'Média {classe_i}')
    plt.scatter(pj[0], pj[1], color='darkred', marker='X', s=200, label=f'Média {classe_j}')
    
    # Superfície de decisão: w * x + b = 0
    # w[0]*x1 + w[1]*x2 + b = 0  => x2 = (-w[0]*x1 - b) / w[1]
    w, b = coeficientes_superficie_decisao(pi, pj)
    
    todos_x1 = x1_c1 + x1_c2
    min_x1, max_x1 = min(todos_x1), max(todos_x1)
    
    # Estender a fronteira um pouco
    margem = (max_x1 - min_x1) * 0.1
    coordenadas_x = [min_x1 - margem, max_x1 + margem]
    
    if abs(w[1]) > 1e-9:
        coordenadas_y = [(-w[0]*x - b) / w[1] for x in coordenadas_x]
        plt.plot(coordenadas_x, coordenadas_y, color='black', linestyle='--', linewidth=2, label='Fronteira dij(x)=0')
    else:
        # Caso de linha vertical
        val_x = -b / w[0]
        plt.axvline(x=val_x, color='black', linestyle='--', linewidth=2, label='Fronteira dij(x)=0')
        
    plt.xlabel(f'Atributo {indices_atributos[0]}')
    plt.ylabel(f'Atributo {indices_atributos[1]}')
    plt.title(titulo)
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    
    if caminho_salvar:
        plt.savefig(caminho_salvar)
        print(f"Gráfico salvo em: {caminho_salvar}")
    plt.close()

def plotar_dispersao_todas_classes(dados, indices_atributos, prototipos=None, caminho_salvar=None):
    """Plota a dispersão de todas as classes."""
    plt.figure(figsize=(10, 6))
    classes = sorted(list(set(d['classe'] for d in dados)))
    cores = {'setosa': 'blue', 'versicolor': 'green', 'virginica': 'red'}
    
    for classe in classes:
        dados_classe = [d for d in dados if d['classe'] == classe]
        x1 = [d['atributos'][indices_atributos[0]] for d in dados_classe]
        x2 = [d['atributos'][indices_atributos[1]] for d in dados_classe]
        plt.scatter(x1, x2, label=classe, color=cores.get(classe, 'black'), alpha=0.6)
        
    if prototipos:
        for classe, p in prototipos.items():
            plt.scatter(p[0], p[1], color='black', marker='X', s=150)
            plt.text(p[0], p[1], f"  {classe}", fontweight='bold')
            
    plt.xlabel(f'Atributo {indices_atributos[0]}')
    plt.ylabel(f'Atributo {indices_atributos[1]}')
    plt.title("Iris Dataset - Distribuição das Classes")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)

    if caminho_salvar:
        plt.savefig(caminho_salvar)
        print(f"Gráfico salvo em: {caminho_salvar}")
    plt.close()

def plotar_matriz_confusao(mc, classes, caminho_salvar=None):
    """
    Plota um heatmap da matriz de confusão usando matplotlib puro.
    Linhas = Classe Real, Colunas = Classe Predita.
    """
    n = len(classes)
    valores = [[mc[real][pred] for pred in classes] for real in classes]
    valor_max = max(v for linha in valores for v in linha) or 1

    fig, ax = plt.subplots(figsize=(7, 6))

    for i in range(n):
        for j in range(n):
            intensidade = valores[i][j] / valor_max
            if i == j:
                cor_fundo = (1 - intensidade * 0.8, 1 - intensidade * 0.8, 1.0)
            else:
                cor_fundo = (1.0, 1 - intensidade * 0.8, 1 - intensidade * 0.8)
            ax.add_patch(plt.Rectangle((j, n - 1 - i), 1, 1, color=cor_fundo))
            cor_texto = 'black' if intensidade < 0.6 else 'white'
            ax.text(j + 0.5, n - 1 - i + 0.5, str(valores[i][j]),
                    ha='center', va='center', fontsize=14, fontweight='bold', color=cor_texto)

    ax.set_xlim(0, n)
    ax.set_ylim(0, n)
    ax.set_xticks([i + 0.5 for i in range(n)])
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_yticks([i + 0.5 for i in range(n)])
    ax.set_yticklabels(list(reversed(classes)), fontsize=11)
    ax.set_xlabel('Predito', fontsize=12)
    ax.set_ylabel('Real', fontsize=12)
    ax.set_title('Matriz de Confusão', fontsize=14, fontweight='bold')
    ax.grid(False)

    for i in range(n + 1):
        ax.axhline(i, color='gray', linewidth=0.5)
        ax.axvline(i, color='gray', linewidth=0.5)

    plt.tight_layout()
    if caminho_salvar:
        plt.savefig(caminho_salvar)
        print(f"Gráfico salvo em: {caminho_salvar}")
    plt.close()
