"""
Florestas Aleatorias (Random Forests) — Python puro.

Seminario da disciplina. Implementado do zero, sem scikit-learn, seguindo a
mesma restricao dos demais laboratorios: listas nativas, lacos `for` e a
matematica escrita explicitamente.

Ideia central
-------------
Uma arvore de decisao sozinha e instavel: pequenas mudancas nos dados de
treino produzem arvores bem diferentes (variancia alta). Breiman (2001)
propos combinar muitas arvores propositalmente descorrelacionadas, e decidir
por voto da maioria. Duas fontes de aleatoriedade fazem essa
descorrelacao:

1. **Bagging** (bootstrap aggregating) — cada arvore treina numa amostra
   sorteada COM reposicao do conjunto original, do mesmo tamanho. Em media,
   ~63,2% das amostras entram; as ~36,8% que ficam de fora sao as amostras
   *out-of-bag* (OOB) daquela arvore.

2. **Subespaco aleatorio** — em cada no, a busca pela melhor divisao
   considera apenas um subconjunto sorteado de atributos (tipicamente
   raiz(n_atributos)). Sem isso, um atributo muito dominante apareceria no
   topo de quase todas as arvores, e elas ficariam parecidas demais.

O erro OOB e uma estimativa de generalizacao "de graca": cada amostra e
avaliada apenas pelas arvores que nao a viram no treino, dispensando um
conjunto de validacao separado.

Criterios de impureza
---------------------
    Gini:     G = 1 - sum_k p_k^2
    Entropia: H = - sum_k p_k * log2(p_k)

O ganho de uma divisao e a reducao ponderada da impureza:

    ganho = I(pai) - [ (n_esq/n) * I(esq) + (n_dir/n) * I(dir) ]
"""

import math
import random


# ===========================================================================
# Impureza
# ===========================================================================
def contar_classes(amostras):
    """Frequencia de cada classe numa lista de amostras."""
    contagem = {}
    for a in amostras:
        contagem[a['classe']] = contagem.get(a['classe'], 0) + 1
    return contagem


def gini(amostras):
    """G = 1 - sum p_k^2  — zero quando o no e puro."""
    n = len(amostras)
    if n == 0:
        return 0.0
    contagem = contar_classes(amostras)
    return 1.0 - sum((c / n) ** 2 for c in contagem.values())


def entropia(amostras):
    """H = -sum p_k log2(p_k)  — zero quando o no e puro."""
    n = len(amostras)
    if n == 0:
        return 0.0
    contagem = contar_classes(amostras)
    total = 0.0
    for c in contagem.values():
        p = c / n
        if p > 0:
            total -= p * math.log2(p)
    return total


CRITERIOS = {'gini': gini, 'entropia': entropia}


# ===========================================================================
# Arvore de decisao (CART binaria)
# ===========================================================================
class No:
    """
    No da arvore.

    Folha           -> `classe` preenchida, `atributo` None.
    No de decisao   -> divide por `atributo <= limiar`.
    """

    def __init__(self, atributo=None, limiar=None, esquerda=None, direita=None,
                 classe=None, n_amostras=0, impureza=0.0, ganho=0.0,
                 distribuicao=None, profundidade=0):
        self.atributo = atributo
        self.limiar = limiar
        self.esquerda = esquerda
        self.direita = direita
        self.classe = classe
        self.n_amostras = n_amostras
        self.impureza = impureza
        self.ganho = ganho
        self.distribuicao = distribuicao or {}
        self.profundidade = profundidade

    @property
    def eh_folha(self):
        return self.atributo is None

    def para_dict(self):
        """Serializa a arvore — usado pela API para desenhar o diagrama."""
        base = {
            'folha': self.eh_folha,
            'n_amostras': self.n_amostras,
            'impureza': self.impureza,
            'distribuicao': self.distribuicao,
            'profundidade': self.profundidade,
        }
        if self.eh_folha:
            base['classe'] = self.classe
        else:
            base.update({
                'atributo': self.atributo,
                'limiar': self.limiar,
                'ganho': self.ganho,
                'esquerda': self.esquerda.para_dict(),
                'direita': self.direita.para_dict(),
            })
        return base


def classe_majoritaria(amostras):
    """Classe mais frequente — desempate alfabetico, para ser deterministico."""
    contagem = contar_classes(amostras)
    if not contagem:
        return None
    maior = max(contagem.values())
    return sorted(c for c, n in contagem.items() if n == maior)[0]


def limiares_candidatos(amostras, atributo):
    """
    Pontos medios entre valores consecutivos distintos do atributo.

    Testar apenas os pontos medios e suficiente: qualquer limiar entre dois
    valores consecutivos produz exatamente a mesma particao.
    """
    valores = sorted({a['atributos'][atributo] for a in amostras})
    return [(valores[i] + valores[i + 1]) / 2 for i in range(len(valores) - 1)]


def melhor_divisao(amostras, indices_atributos, criterio='gini',
                   min_amostras_folha=1):
    """
    Procura a divisao (atributo, limiar) de maior ganho de impureza.

    Retorna (atributo, limiar, ganho, esquerda, direita) ou None se nenhuma
    divisao valida existir.
    """
    fn_impureza = CRITERIOS[criterio]
    impureza_pai = fn_impureza(amostras)
    n = len(amostras)

    melhor = None
    melhor_ganho = 0.0

    for atributo in indices_atributos:
        for limiar in limiares_candidatos(amostras, atributo):
            esquerda, direita = [], []
            for a in amostras:
                (esquerda if a['atributos'][atributo] <= limiar
                 else direita).append(a)

            if (len(esquerda) < min_amostras_folha
                    or len(direita) < min_amostras_folha):
                continue

            impureza_filhos = (
                len(esquerda) / n * fn_impureza(esquerda)
                + len(direita) / n * fn_impureza(direita))
            ganho = impureza_pai - impureza_filhos

            if ganho > melhor_ganho + 1e-12:
                melhor_ganho = ganho
                melhor = (atributo, limiar, ganho, esquerda, direita)

    return melhor


def construir_arvore(amostras, indices_atributos, criterio='gini',
                     profundidade_max=None, min_amostras_divisao=2,
                     min_amostras_folha=1, n_atributos_sorteados=None,
                     rng=None, profundidade=0, importancias=None):
    """
    Constroi a arvore recursivamente.

    `n_atributos_sorteados` ativa o subespaco aleatorio: em cada no, sorteia
    esse numero de atributos entre os disponiveis. E o ingrediente que
    descorrelaciona as arvores da floresta.

    `importancias` acumula a reducao de impureza ponderada por atributo —
    a importancia de Gini (mean decrease in impurity).
    """
    rng = rng or random.Random()
    fn_impureza = CRITERIOS[criterio]
    impureza = fn_impureza(amostras)
    distribuicao = contar_classes(amostras)

    def folha():
        return No(classe=classe_majoritaria(amostras),
                  n_amostras=len(amostras), impureza=impureza,
                  distribuicao=distribuicao, profundidade=profundidade)

    # Criterios de parada
    if (impureza == 0.0
            or len(amostras) < min_amostras_divisao
            or (profundidade_max is not None and profundidade >= profundidade_max)):
        return folha()

    # Subespaco aleatorio de atributos
    if n_atributos_sorteados and n_atributos_sorteados < len(indices_atributos):
        candidatos = rng.sample(list(indices_atributos), n_atributos_sorteados)
    else:
        candidatos = list(indices_atributos)

    divisao = melhor_divisao(amostras, candidatos, criterio, min_amostras_folha)
    if divisao is None:
        return folha()

    atributo, limiar, ganho, esquerda, direita = divisao

    if importancias is not None:
        # Pondera pelo numero de amostras que passam pelo no
        importancias[atributo] = (importancias.get(atributo, 0.0)
                                  + ganho * len(amostras))

    return No(
        atributo=atributo, limiar=limiar, ganho=ganho,
        n_amostras=len(amostras), impureza=impureza,
        distribuicao=distribuicao, profundidade=profundidade,
        esquerda=construir_arvore(
            esquerda, indices_atributos, criterio, profundidade_max,
            min_amostras_divisao, min_amostras_folha, n_atributos_sorteados,
            rng, profundidade + 1, importancias),
        direita=construir_arvore(
            direita, indices_atributos, criterio, profundidade_max,
            min_amostras_divisao, min_amostras_folha, n_atributos_sorteados,
            rng, profundidade + 1, importancias),
    )


def predizer_arvore(no, atributos):
    """Percorre a arvore ate uma folha e devolve a classe."""
    while not no.eh_folha:
        no = (no.esquerda if atributos[no.atributo] <= no.limiar
              else no.direita)
    return no.classe


def caminho_decisao(no, atributos):
    """
    Sequencia de decisoes ate a folha — usada na memoria de calculo para
    mostrar exatamente por que a amostra caiu naquela classe.
    """
    passos = []
    while not no.eh_folha:
        valor = atributos[no.atributo]
        vai_esquerda = valor <= no.limiar
        passos.append({
            'atributo': no.atributo,
            'limiar': no.limiar,
            'valor': valor,
            'condicao': f'x[{no.atributo}] <= {no.limiar:.4f}',
            'resultado': vai_esquerda,
            'n_amostras': no.n_amostras,
            'impureza': no.impureza,
        })
        no = no.esquerda if vai_esquerda else no.direita
    passos.append({
        'folha': True,
        'classe': no.classe,
        'n_amostras': no.n_amostras,
        'impureza': no.impureza,
        'distribuicao': no.distribuicao,
    })
    return passos


def profundidade_arvore(no):
    if no.eh_folha:
        return no.profundidade
    return max(profundidade_arvore(no.esquerda), profundidade_arvore(no.direita))


def contar_nos(no):
    """(total de nos, numero de folhas)"""
    if no.eh_folha:
        return 1, 1
    te, fe = contar_nos(no.esquerda)
    td, fd = contar_nos(no.direita)
    return 1 + te + td, fe + fd


# ===========================================================================
# Floresta Aleatoria
# ===========================================================================
class FlorestaAleatoria:
    """
    Ensemble de arvores treinadas com bagging + subespaco aleatorio.

    Parametros
    ----------
    n_arvores            : quantas arvores compoem a floresta
    criterio             : 'gini' ou 'entropia'
    profundidade_max     : limite de profundidade (None = sem limite)
    min_amostras_divisao : minimo de amostras para tentar dividir um no
    min_amostras_folha   : minimo de amostras em cada folha resultante
    max_atributos        : 'sqrt', 'log2', um inteiro, ou None (todos)
    semente              : reprodutibilidade
    """

    def __init__(self, n_arvores=50, criterio='gini', profundidade_max=None,
                 min_amostras_divisao=2, min_amostras_folha=1,
                 max_atributos='sqrt', semente=42):
        if criterio not in CRITERIOS:
            raise ValueError(f"criterio deve ser 'gini' ou 'entropia', recebido '{criterio}'")
        self.n_arvores = n_arvores
        self.criterio = criterio
        self.profundidade_max = profundidade_max
        self.min_amostras_divisao = min_amostras_divisao
        self.min_amostras_folha = min_amostras_folha
        self.max_atributos = max_atributos
        self.semente = semente

        self.arvores = []
        self.indices_bootstrap = []   # indices sorteados por arvore
        self.indices_oob = []         # indices fora do bag, por arvore
        self.classes = []
        self.indices_atributos = []
        self.importancias = {}
        self.erro_oob = None
        self.acuracia_oob = None
        self.votos_oob = {}

    # ------------------------------------------------------------------
    def _n_atributos_sorteados(self, total):
        """Traduz `max_atributos` no numero de atributos por no."""
        if self.max_atributos is None:
            return total
        if isinstance(self.max_atributos, int):
            return max(1, min(total, self.max_atributos))
        if self.max_atributos == 'sqrt':
            return max(1, int(math.sqrt(total)))
        if self.max_atributos == 'log2':
            return max(1, int(math.log2(total))) if total > 1 else 1
        return total

    def treinar(self, dados_treino, indices_atributos):
        """Treina a floresta e calcula o erro out-of-bag."""
        rng = random.Random(self.semente)
        n = len(dados_treino)
        self.classes = sorted({d['classe'] for d in dados_treino})
        self.indices_atributos = list(indices_atributos)
        self.arvores, self.indices_bootstrap, self.indices_oob = [], [], []

        n_sorteados = self._n_atributos_sorteados(len(indices_atributos))
        importancias = {}

        for _ in range(self.n_arvores):
            # --- Bootstrap: sorteio COM reposicao, mesmo tamanho ---
            idx_bag = [rng.randrange(n) for _ in range(n)]
            amostras_bag = [dados_treino[i] for i in idx_bag]
            idx_oob = sorted(set(range(n)) - set(idx_bag))

            arvore = construir_arvore(
                amostras_bag, self.indices_atributos, self.criterio,
                self.profundidade_max, self.min_amostras_divisao,
                self.min_amostras_folha, n_sorteados, rng,
                importancias=importancias)

            self.arvores.append(arvore)
            self.indices_bootstrap.append(idx_bag)
            self.indices_oob.append(idx_oob)

        # --- Importancia dos atributos (normalizada) ---
        total = sum(importancias.values())
        self.importancias = (
            {a: v / total for a, v in importancias.items()} if total > 0
            else {a: 0.0 for a in self.indices_atributos})
        for a in self.indices_atributos:
            self.importancias.setdefault(a, 0.0)

        self._calcular_oob(dados_treino)
        return self

    # ------------------------------------------------------------------
    def _calcular_oob(self, dados_treino):
        """
        Erro out-of-bag: cada amostra e votada apenas pelas arvores que NAO
        a viram no treino. Estimativa de generalizacao sem separar validacao.
        """
        votos = {i: {} for i in range(len(dados_treino))}
        for arvore, idx_oob in zip(self.arvores, self.indices_oob):
            for i in idx_oob:
                pred = predizer_arvore(arvore, dados_treino[i]['atributos'])
                votos[i][pred] = votos[i].get(pred, 0) + 1

        corretos = avaliadas = 0
        for i, contagem in votos.items():
            if not contagem:
                continue  # amostra entrou em todos os bags — raro
            avaliadas += 1
            maior = max(contagem.values())
            vencedor = sorted(c for c, v in contagem.items() if v == maior)[0]
            if vencedor == dados_treino[i]['classe']:
                corretos += 1

        self.votos_oob = votos
        if avaliadas:
            self.acuracia_oob = corretos / avaliadas
            self.erro_oob = 1.0 - self.acuracia_oob
        else:  # pragma: no cover — so ocorreria com n muito pequeno
            self.acuracia_oob = self.erro_oob = None

    # ------------------------------------------------------------------
    def votos(self, atributos):
        """Voto de cada arvore para uma amostra: {classe: quantidade}."""
        contagem = {}
        for arvore in self.arvores:
            c = predizer_arvore(arvore, atributos)
            contagem[c] = contagem.get(c, 0) + 1
        return contagem

    def predizer(self, atributos):
        """Classe vencedora por voto da maioria."""
        contagem = self.votos(atributos)
        if not contagem:
            return None
        maior = max(contagem.values())
        return sorted(c for c, v in contagem.items() if v == maior)[0]

    def probabilidades(self, atributos):
        """Proporcao de votos por classe — serve de confianca da predicao."""
        contagem = self.votos(atributos)
        total = sum(contagem.values()) or 1
        return {c: contagem.get(c, 0) / total for c in self.classes}

    # ------------------------------------------------------------------
    def resumo_arvores(self):
        """Estatisticas por arvore — profundidade, nos, folhas e OOB."""
        resumo = []
        for i, (arvore, idx_bag, idx_oob) in enumerate(
                zip(self.arvores, self.indices_bootstrap, self.indices_oob)):
            nos, folhas = contar_nos(arvore)
            resumo.append({
                'indice': i,
                'profundidade': profundidade_arvore(arvore),
                'nos': nos,
                'folhas': folhas,
                'amostras_unicas_bag': len(set(idx_bag)),
                'amostras_oob': len(idx_oob),
            })
        return resumo


def treinar_floresta(dados_treino, indices_atributos, **kwargs):
    """Atalho no estilo dos demais modulos do projeto."""
    return FlorestaAleatoria(**kwargs).treinar(dados_treino, indices_atributos)
