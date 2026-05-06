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
