"""
Configuracoes dos exercicios do Lab 5 (Aula PR_711).

Espelha exatamente os valores dos slides e dos scripts demonstrativos
`iris_classifier/lab05_*.py`, que continuam sendo a referencia pedagogica
lida pelo aluno. Aqui eles ficam num formato unico de dicionario para que
a API sirva qualquer exercicio pelo mesmo endpoint.
"""

# --- Lab 5.0 · Exemplo didatico (slide 37, resolvido nos slides 38-42) ------
# Particularidade: b1 e b2 sao UM UNICO bias por camada, compartilhado por
# todos os neuronios dela (o gradiente do bias soma os deltas da camada).
EXEMPLO_DIDATICO = {
    'id': 'didatico',
    'titulo': 'Exemplo Didatico — Rede 2-2-2',
    'subtitulo': 'Slide 37 · resolvido passo a passo nos slides 38-42',
    'slide': 37,
    'entradas': [0.05, 0.10],
    'alvo': [0.01, 0.99],
    'taxa': 0.5,
    'pesos_oculta': [[0.15, 0.20], [0.25, 0.30]],
    'bias_oculta': [0.35, 0.35],
    'pesos_saida': [[0.40, 0.45], [0.50, 0.55]],
    'bias_saida': [0.60, 0.60],
    'rotulos_entrada': ['i1', 'i2'],
    'rotulos_ocultos': ['h1', 'h2'],
    'rotulos_saida': ['o1', 'o2'],
    'bias_compartilhado': True,
    'nota': ('Este exemplo usa um unico bias por camada, compartilhado por '
             'todos os neuronios dela — por isso o gradiente do bias soma os '
             'deltas de toda a camada, em vez de usar apenas o delta de um '
             'neuronio. E a unica parte do laboratorio com essa convencao.'),
}

# --- Lab 5.0 · Exercicio XOR (slide 36, arquitetura da Fig. 12.28b) ---------
# O slide da apenas a topologia (w1..w9 genericos), sem valores numericos:
# os pesos iniciais abaixo foram escolhidos pelo grupo para a demonstracao.
XOR = {
    'id': 'xor',
    'titulo': 'XOR com MLP — Fig. 12.28(b)',
    'subtitulo': 'Slide 36 · 1 epoca (4 padroes, modo online)',
    'slide': 36,
    'padroes': [
        {'entrada': [0.0, 0.0], 'alvo': [0.0]},
        {'entrada': [0.0, 1.0], 'alvo': [1.0]},
        {'entrada': [1.0, 0.0], 'alvo': [1.0]},
        {'entrada': [1.0, 1.0], 'alvo': [0.0]},
    ],
    'taxa': 0.5,
    'pesos_oculta': [[0.50, 0.50], [-0.50, -0.50]],
    'bias_oculta': [-0.20, 0.30],
    'pesos_saida': [[0.60, -0.60]],
    'bias_saida': [-0.10],
    'rotulos_entrada': ['x1', 'x2'],
    'rotulos_ocultos': ['h1', 'h2'],
    'rotulos_saida': ['saida'],
    'bias_compartilhado': False,
    'nota': ('A Fig. 12.28(b) mostra apenas a topologia minima que resolve o '
             'XOR (pesos rotulados w1...w9, sem valores numericos). Os pesos '
             'iniciais desta demonstracao foram escolhidos pelo grupo.'),
}

# --- Lab 5.1 · Item (i): rede "Galinha vs Homem" ---------------------------
GALINHA_HOMEM = {
    'id': 'galinha-homem',
    'titulo': 'Galinha vs Homem — Rede 2-2-2',
    'subtitulo': 'Item (i) do enunciado · pesos iniciais do slide',
    'slide': None,
    'entradas': [0.15, 0.35],
    'alvo': [0.0, 1.0],          # c1 = homem (0), c2 = galinha (1)
    'taxa': 0.05,
    'pesos_oculta': [[0.10, 0.12], [0.20, 0.17]],
    'bias_oculta': [0.80, 0.25],
    'pesos_saida': [[0.05, 0.33], [0.40, 0.07]],
    'bias_saida': [0.15, 0.70],
    'rotulos_entrada': ['a1', 'a2'],
    'rotulos_ocultos': ['b1', 'b2'],
    'rotulos_saida': ['c1 (homem)', 'c2 (galinha)'],
    'bias_compartilhado': False,
    'nota': ('Valores conferidos contra o slide: out_b1=0,7020  out_b2=0,5841  '
             'out_c1=0,5934  out_c2=0,7353  E=0,21108.'),
}

# --- Lab 5.1 · Exercicio extra (slide 34, rede da Fig. 12.32) --------------
FIG_1232 = {
    'id': 'fig-1232',
    'titulo': 'Rede da Fig. 12.32 — 1 Iteracao',
    'subtitulo': 'Slide 34 · saida desejada C1=1, C2=0',
    'slide': 34,
    'entradas': [3.0, 0.0, 1.0],
    'alvo': [1.0, 0.0],
    'taxa': 0.5,
    'pesos_oculta': [[0.1, 0.2, 0.6], [0.4, 0.3, 0.1]],
    'bias_oculta': [0.4, 0.2],
    'pesos_saida': [[0.2, 0.1], [0.1, 0.4]],
    'bias_saida': [0.6, 0.3],
    'rotulos_entrada': ['x1', 'x2', 'x3'],
    'rotulos_ocultos': ['b1', 'b2'],
    'rotulos_saida': ['c1', 'c2'],
    'bias_compartilhado': False,
    'nota': ('O slide nao especifica a taxa de aprendizagem para este '
             'exercicio; adotamos eta=0,5 (mesma ordem de grandeza do exemplo '
             'didatico da aula). Valores de forward conferidos: '
             'out_b1=0,7858  out_b2=0,8176  out_c1=0,6982  out_c2=0,6694.'),
}

EXERCICIOS = {e['id']: e for e in
              (EXEMPLO_DIDATICO, XOR, GALINHA_HOMEM, FIG_1232)}

# Exercicios de passo unico (uma amostra, um passo de backprop)
PASSO_UNICO = {'didatico', 'galinha-homem', 'fig-1232'}
