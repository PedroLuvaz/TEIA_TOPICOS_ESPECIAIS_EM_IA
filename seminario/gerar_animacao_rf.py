"""
Animação: Fronteira de decisão -- Árvore única vs. Random Forest
=================================================================
Gera um GIF mostrando visualmente por que o Random Forest generaliza melhor
que uma única árvore de decisão: conforme aumentamos o número de árvores (B),
a fronteira de decisão "irregular" de uma árvore profunda vai suavizando.

Esse exemplo usa um dataset sintético 2D (contínuo) só para fins de
visualização -- é um complemento intuitivo ao exemplo categórico (Clima,
Pais, Dinheiro) usado no resto da apresentação, que é o exemplo "de baixo
nível, com as contas" cobrado pelo professor.

Requisitos: matplotlib, scikit-learn, numpy, pillow (todos padrão em
qualquer instalação científica de Python).

Uso:
    python3 gerar_animacao_rf.py

Saídas (na pasta ./saida/):
    - rf_fronteira_decisao.gif   (animação completa)
    - frame_arvore_unica.png     (frame estático: árvore única)
    - frame_B1.png, frame_B10.png, frame_B100.png (frames estáticos da floresta)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter, FuncAnimation
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------
# 0. Configuração e paleta (mesmas cores do tema da apresentação)
# ---------------------------------------------------------------
SEED = 42
OUT_DIR = os.path.join(os.path.dirname(__file__), "saida")
os.makedirs(OUT_DIR, exist_ok=True)

FOREST_GREEN = "#1B5E20"
FOREST_MID = "#388E3C"
FOREST_LIGHT = "#C8E6C9"
FOREST_ALERT = "#BF360C"
CLASS_COLORS = [FOREST_GREEN, FOREST_ALERT]
CMAP_POINTS = ListedColormap(CLASS_COLORS)
CMAP_BG = ListedColormap([FOREST_LIGHT, "#FFE0D2"])

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "figure.facecolor": "white",
})

# ---------------------------------------------------------------
# 1. Dataset sintético 2D (não-linearmente separável, tipo "luas")
# ---------------------------------------------------------------
X, y = make_moons(n_samples=240, noise=0.28, random_state=SEED)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED, stratify=y
)

x_min, x_max = X[:, 0].min() - 0.6, X[:, 0].max() + 0.6
y_min, y_max = X[:, 1].min() - 0.6, X[:, 1].max() + 0.6
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 350), np.linspace(y_min, y_max, 350))
grid = np.c_[xx.ravel(), yy.ravel()]

# ---------------------------------------------------------------
# 2. Modelos: árvore única profunda + florestas com B crescente
# ---------------------------------------------------------------
tree_model = DecisionTreeClassifier(random_state=SEED)  # sem limite de profundidade -> overfit proposital
tree_model.fit(X_train, y_train)

B_VALUES = [1, 3, 5, 10, 20, 50, 100]
forest_models = []
for B in B_VALUES:
    rf = RandomForestClassifier(
        n_estimators=B, random_state=SEED, bootstrap=True, max_features="sqrt"
    )
    rf.fit(X_train, y_train)
    forest_models.append(rf)

# ---------------------------------------------------------------
# 3. Função utilitária para desenhar um frame (modelo -> eixo)
# ---------------------------------------------------------------
def desenhar_frame(ax, model, titulo, subtitulo):
    ax.clear()
    Z = model.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z, cmap=CMAP_BG, alpha=0.9, levels=[-0.5, 0.5, 1.5])
    ax.scatter(
        X_train[:, 0], X_train[:, 1], c=y_train, cmap=CMAP_POINTS,
        edgecolor="white", linewidth=0.6, s=45, zorder=3, label="treino"
    )
    ax.scatter(
        X_test[:, 0], X_test[:, 1], c=y_test, cmap=CMAP_POINTS,
        edgecolor="black", linewidth=0.9, s=70, marker="^", zorder=4, label="teste"
    )
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)
    ax.set_title(titulo, color=FOREST_GREEN)
    ax.set_xlabel(subtitulo, fontsize=11)
    ax.text(
        0.02, 0.02,
        f"Acurácia treino: {train_acc*100:.1f}%   |   Acurácia teste: {test_acc*100:.1f}%",
        transform=ax.transAxes, fontsize=10, color="white",
        bbox=dict(boxstyle="round,pad=0.35", facecolor=FOREST_GREEN, alpha=0.92),
        verticalalignment="bottom",
    )
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

# ---------------------------------------------------------------
# 4. Sequência de frames: árvore única -> floresta crescendo
# ---------------------------------------------------------------
sequence = [("tree", None)] + [("forest", B) for B in B_VALUES]
# Segura no primeiro e no último frame por mais tempo (efeito "pausa")
HOLD_FIRST = 6
HOLD_LAST = 8
frame_list = [sequence[0]] * HOLD_FIRST + sequence[1:-1] + [sequence[-1]] * HOLD_LAST

fig, ax = plt.subplots(figsize=(7.2, 6.4), dpi=130)
fig.subplots_adjust(top=0.90, bottom=0.10, left=0.04, right=0.98)

def update(i):
    kind, B = frame_list[i]
    if kind == "tree":
        desenhar_frame(
            ax, tree_model,
            "Árvore única (sem limite de profundidade)",
            "Fronteira \"quadriculada\" -- decorou o ruído do treino (overfitting)"
        )
    else:
        idx = B_VALUES.index(B)
        desenhar_frame(
            ax, forest_models[idx],
            f"Random Forest com B = {B} árvore{'s' if B > 1 else ''}",
            "Fronteira ficando mais suave e estável conforme B cresce"
        )
    return []

anim = FuncAnimation(fig, update, frames=len(frame_list), blit=False)
gif_path = os.path.join(OUT_DIR, "rf_fronteira_decisao.gif")
anim.save(gif_path, writer=PillowWriter(fps=1.4))
print(f"GIF salvo em: {gif_path}")

# ---------------------------------------------------------------
# 5. Frames estáticos de alta qualidade (para inserir em slides)
# ---------------------------------------------------------------
def salvar_frame_estatico(nome_arquivo, kind, B=None):
    fig2, ax2 = plt.subplots(figsize=(7.2, 6.4), dpi=200)
    fig2.subplots_adjust(top=0.90, bottom=0.10, left=0.04, right=0.98)
    if kind == "tree":
        desenhar_frame(
            ax2, tree_model,
            "Árvore única (sem limite de profundidade)",
            "Fronteira \"quadriculada\" -- decorou o ruído do treino (overfitting)"
        )
    else:
        idx = B_VALUES.index(B)
        desenhar_frame(
            ax2, forest_models[idx],
            f"Random Forest com B = {B} árvore{'s' if B > 1 else ''}",
            "Fronteira ficando mais suave e estável conforme B cresce"
        )
    path = os.path.join(OUT_DIR, nome_arquivo)
    fig2.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig2)
    print(f"Frame estático salvo em: {path}")

salvar_frame_estatico("frame_arvore_unica.png", "tree")
salvar_frame_estatico("frame_B1.png", "forest", B=1)
salvar_frame_estatico("frame_B10.png", "forest", B=10)
salvar_frame_estatico("frame_B100.png", "forest", B=100)

print("\nConcluído. Resumo de acurácia (teste, um único split):")
print(f"  Árvore única : {tree_model.score(X_test, y_test)*100:.1f}%")
for B, rf in zip(B_VALUES, forest_models):
    print(f"  Forest B={B:<3}: {rf.score(X_test, y_test)*100:.1f}%")

# ---------------------------------------------------------------
# 6. Gráfico complementar: acurácia MÉDIA sobre várias repetições
#    (um único split pode "sortear" um resultado de sorte; para ser
#    rigoroso, repetimos o experimento com várias sementes e tiramos
#    a média -- isso é o que de fato mostra a redução de variância)
# ---------------------------------------------------------------
N_REPEATS = 30
tree_accs = []
forest_accs = {B: [] for B in B_VALUES}

for rep in range(N_REPEATS):
    Xr, yr = make_moons(n_samples=240, noise=0.28, random_state=rep)
    Xr_tr, Xr_te, yr_tr, yr_te = train_test_split(
        Xr, yr, test_size=0.3, random_state=rep, stratify=yr
    )
    t = DecisionTreeClassifier(random_state=rep).fit(Xr_tr, yr_tr)
    tree_accs.append(t.score(Xr_te, yr_te))
    for B in B_VALUES:
        rf_r = RandomForestClassifier(
            n_estimators=B, random_state=rep, bootstrap=True, max_features="sqrt"
        ).fit(Xr_tr, yr_tr)
        forest_accs[B].append(rf_r.score(Xr_te, yr_te))

tree_mean, tree_std = np.mean(tree_accs), np.std(tree_accs)
forest_means = [np.mean(forest_accs[B]) for B in B_VALUES]
forest_stds = [np.std(forest_accs[B]) for B in B_VALUES]

print(f"\nMédia sobre {N_REPEATS} repetições (seeds diferentes):")
print(f"  Árvore única : {tree_mean*100:.1f}% (+/- {tree_std*100:.1f})")
for B, m, s in zip(B_VALUES, forest_means, forest_stds):
    print(f"  Forest B={B:<3}: {m*100:.1f}% (+/- {s*100:.1f})")

fig3, ax3 = plt.subplots(figsize=(7.5, 5.2), dpi=200)
ax3.axhline(tree_mean*100, color=FOREST_ALERT, linestyle="--", linewidth=1.8,
            label=f"Árvore única (média = {tree_mean*100:.1f}%)")
ax3.fill_between([1, max(B_VALUES)], (tree_mean - tree_std)*100, (tree_mean + tree_std)*100,
                  color=FOREST_ALERT, alpha=0.08)
ax3.errorbar(B_VALUES, [m*100 for m in forest_means], yerr=[s*100 for s in forest_stds],
             color=FOREST_GREEN, marker="o", linewidth=2.2, capsize=4,
             label="Random Forest (média +/- desvio padrão)")
ax3.set_xscale("log")
ax3.set_xticks(B_VALUES)
ax3.set_xticklabels([str(b) for b in B_VALUES])
ax3.set_ylim(75, 98)
ax3.set_xlabel("Número de árvores (B)")
ax3.set_ylabel("Acurácia no teste (%)")
ax3.set_title(f"Acurácia média sobre {N_REPEATS} repetições (seeds diferentes)", color=FOREST_GREEN)
ax3.legend(loc="lower right", fontsize=10)
ax3.grid(alpha=0.25)
curve_path = os.path.join(OUT_DIR, "curva_acuracia_media.png")
fig3.savefig(curve_path, bbox_inches="tight", facecolor="white")
plt.close(fig3)
print(f"\nGráfico de acurácia média salvo em: {curve_path}")