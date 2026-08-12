"""
Floresta Aleatoria sobre atributos CATEGORICOS (ID3 multi-way).

Por que um modulo separado de `random_forest.py`
------------------------------------------------
`models/random_forest.py` implementa a CART binaria: cada no divide por
`atributo <= limiar`, que e o certo para atributos continuos como os do Iris.

O seminario, porem, faz as contas sobre atributos CATEGORICOS (Clima, Pais,
Dinheiro) usando ID3: entropia, ganho de informacao e divisao multi-way — um
ramo por valor do atributo, exatamente como nos slides 8 a 21. Codificar
"Sol/Vento/Chuva" como 0/1/2 e deixar a CART cortar por limiar funcionaria,
mas inventaria uma ordem que nao existe (Sol < Vento < Chuva) e nao
reproduziria as contas apresentadas.

Este modulo implementa o algoritmo do slide 17 ao pe da letra:

    Para b = 1..B:
      1. sorteie uma amostra bootstrap T_b (tamanho n, com reposicao)
      2. construa a arvore h_b sobre T_b; em cada no,
         a. sorteie m atributos dentre os p disponiveis (sem repor os ja
            usados no caminho da raiz ate o no)
         b. calcule o ganho de informacao de cada um desses m
         c. divida pelo atributo de maior ganho
         d. repita ate o no ficar puro ou os atributos se esgotarem
      3. registre as instancias fora de T_b como o OOB de h_b
    Previsao: votacao majoritaria entre as B arvores.

Tudo em Python puro, com `for` e listas nativas (regra do CLAUDE.md).

Formato das amostras — o mesmo do resto do projeto, porem com valores de
atributo em texto:

    {'atributos': ['Sol', 'Sim', 'Rico'], 'classe': 'Cinema'}
"""
import math
import random


# ===========================================================================
# Impureza e ganho (slides 7 e 8)
# ===========================================================================
def contar_classes(amostras):
    """Frequencia de cada classe numa lista de amostras."""
    contagem = {}
    for a in amostras:
        contagem[a['classe']] = contagem.get(a['classe'], 0) + 1
    return contagem


def entropia(amostras):
    """H(S) = -sum p_i log2(p_i) — zero quando o no e puro."""
    n = len(amostras)
    if n == 0:
        return 0.0
    total = 0.0
    for c in contar_classes(amostras).values():
        p = c / n
        if p > 0:
            total -= p * math.log2(p)
    return total


def gini(amostras):
    """Gini(S) = 1 - sum p_i^2."""
    n = len(amostras)
    if n == 0:
        return 0.0
    return 1.0 - sum((c / n) ** 2 for c in contar_classes(amostras).values())


CRITERIOS = {'entropia': entropia, 'gini': gini}


def particionar(amostras, atributo):
    """
    Divide as amostras por valor do atributo — um subconjunto S_v por valor v.

    E a divisao multi-way do ID3: se Clima tem 3 valores, o no tem 3 ramos.
    """
    ramos = {}
    for a in amostras:
        ramos.setdefault(a['atributos'][atributo], []).append(a)
    return ramos


def ganho(amostras, atributo, criterio='entropia'):
    """
    Gain(S, A) = H(S) - sum_v (|S_v| / |S|) H(S_v)   (slide 8)

    Devolve (ganho, ramos) — os ramos ja particionados, para nao recalcular.
    """
    fn = CRITERIOS[criterio]
    n = len(amostras)
    if n == 0:
        return 0.0, {}

    ramos = particionar(amostras, atributo)
    ponderada = sum(len(sv) / n * fn(sv) for sv in ramos.values())
    return fn(amostras) - ponderada, ramos


# ===========================================================================
# Arvore ID3
# ===========================================================================
class NoCategorico:
    """
    No da arvore.

    Folha         -> `classe` preenchida, `atributo` None.
    No de decisao -> `atributo` escolhido e um filho por valor em `ramos`.
    """

    def __init__(self, atributo=None, ramos=None, classe=None, n_amostras=0,
                 impureza=0.0, ganho=0.0, distribuicao=None, profundidade=0):
        self.atributo = atributo
        self.ramos = ramos or {}
        self.classe = classe
        self.n_amostras = n_amostras
        self.impureza = impureza
        self.ganho = ganho
        self.distribuicao = distribuicao or {}
        self.profundidade = profundidade

    @property
    def eh_folha(self):
        return self.atributo is None

    def para_dict(self, nomes_atributos=None):
        """Serializa a arvore — para a API e para os diagramas."""
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
            nome = (nomes_atributos[self.atributo] if nomes_atributos
                    else self.atributo)
            base.update({
                'atributo': self.atributo,
                'nome_atributo': nome,
                'ganho': self.ganho,
                'ramos': {v: no.para_dict(nomes_atributos)
                          for v, no in sorted(self.ramos.items())},
            })
        return base


def classe_majoritaria(amostras):
    """Classe mais frequente — desempate alfabetico, para ser deterministico."""
    contagem = contar_classes(amostras)
    if not contagem:
        return None
    maior = max(contagem.values())
    return sorted(c for c, n in contagem.items() if n == maior)[0]


def construir_arvore(amostras, atributos_disponiveis, criterio='entropia',
                     n_atributos_sorteados=None, rng=None, profundidade=0,
                     importancias=None, n_treino=None, classe_pai=None):
    """
    Constroi a arvore ID3 recursivamente.

    `atributos_disponiveis` sao os indices ainda nao usados no caminho da raiz
    ate aqui — o slide 17 diz explicitamente "sem repor atributos ja usados no
    caminho", o que e o comportamento classico do ID3 com divisao multi-way.

    `n_atributos_sorteados` e o mtry: quantos atributos sortear em cada no.

    `importancias` acumula (n_t / n) * Gain_t(A) por atributo — o Mean
    Decrease Impurity do slide 26.
    """
    rng = rng or random.Random()
    fn = CRITERIOS[criterio]
    impureza = fn(amostras)
    distribuicao = contar_classes(amostras)
    n_treino = n_treino if n_treino is not None else len(amostras)

    def folha():
        return NoCategorico(
            classe=classe_majoritaria(amostras) or classe_pai,
            n_amostras=len(amostras), impureza=impureza,
            distribuicao=distribuicao, profundidade=profundidade)

    # Parada: no puro, sem amostras ou sem atributos restantes
    if impureza == 0.0 or not amostras or not atributos_disponiveis:
        return folha()

    # mtry — subconjunto aleatorio de atributos (slide 16)
    disponiveis = list(atributos_disponiveis)
    if n_atributos_sorteados and n_atributos_sorteados < len(disponiveis):
        candidatos = rng.sample(disponiveis, n_atributos_sorteados)
    else:
        candidatos = disponiveis

    melhor_atributo, melhor_ganho, melhores_ramos = None, 0.0, None
    for atributo in candidatos:
        g, ramos = ganho(amostras, atributo, criterio)
        if len(ramos) < 2:
            continue                      # atributo constante: nao divide nada
        if g > melhor_ganho + 1e-12:
            melhor_atributo, melhor_ganho, melhores_ramos = atributo, g, ramos

    if melhor_atributo is None:
        return folha()

    if importancias is not None:
        importancias[melhor_atributo] = (
            importancias.get(melhor_atributo, 0.0)
            + (len(amostras) / n_treino) * melhor_ganho)

    restantes = [a for a in atributos_disponiveis if a != melhor_atributo]
    maioria = classe_majoritaria(amostras)

    return NoCategorico(
        atributo=melhor_atributo, ganho=melhor_ganho,
        n_amostras=len(amostras), impureza=impureza,
        distribuicao=distribuicao, profundidade=profundidade,
        ramos={
            valor: construir_arvore(
                sub, restantes, criterio, n_atributos_sorteados, rng,
                profundidade + 1, importancias, n_treino, maioria)
            for valor, sub in melhores_ramos.items()
        },
    )


def predizer_arvore(no, atributos):
    """
    Percorre a arvore ate uma folha.

    Se o padrao trouxer um valor que nao apareceu no treino daquele ramo,
    devolve a classe majoritaria do no atual — o fallback usual do ID3.
    """
    while not no.eh_folha:
        valor = atributos[no.atributo]
        proximo = no.ramos.get(valor)
        if proximo is None:
            return classe_majoritaria_de_distribuicao(no.distribuicao)
        no = proximo
    return no.classe


def classe_majoritaria_de_distribuicao(distribuicao):
    if not distribuicao:
        return None
    maior = max(distribuicao.values())
    return sorted(c for c, n in distribuicao.items() if n == maior)[0]


def caminho_decisao(no, atributos, nomes_atributos=None):
    """Sequencia de decisoes ate a folha — para a memoria de calculo."""
    passos = []
    while not no.eh_folha:
        valor = atributos[no.atributo]
        nome = (nomes_atributos[no.atributo] if nomes_atributos
                else f'x[{no.atributo}]')
        proximo = no.ramos.get(valor)
        passos.append({
            'atributo': no.atributo,
            'nome_atributo': nome,
            'valor': valor,
            'condicao': f'{nome} = {valor}',
            'n_amostras': no.n_amostras,
            'impureza': no.impureza,
            'ganho': no.ganho,
            'ramo_conhecido': proximo is not None,
        })
        if proximo is None:
            passos.append({
                'folha': True,
                'classe': classe_majoritaria_de_distribuicao(no.distribuicao),
                'n_amostras': no.n_amostras,
                'impureza': no.impureza,
                'distribuicao': no.distribuicao,
                'observacao': 'valor nao visto no treino — usa a majoritaria',
            })
            return passos
        no = proximo
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
    return max(profundidade_arvore(f) for f in no.ramos.values())


def contar_nos(no):
    """(total de nos, numero de folhas)"""
    if no.eh_folha:
        return 1, 1
    total, folhas = 1, 0
    for filho in no.ramos.values():
        t, f = contar_nos(filho)
        total += t
        folhas += f
    return total, folhas


# ===========================================================================
# Floresta
# ===========================================================================
class FlorestaCategorica:
    """
    Ensemble de arvores ID3 com bagging + mtry (slides 13 a 17).

    Parametros
    ----------
    n_arvores : B, numero de arvores
    criterio  : 'entropia' (ID3, o dos slides) ou 'gini' (Breiman classico)
    mtry      : quantos atributos sortear por no. None -> round(sqrt(p)),
                a regra do slide 16 para classificacao
    semente   : reprodutibilidade
    """

    def __init__(self, n_arvores=100, criterio='entropia', mtry=None,
                 semente=42, nomes_atributos=None):
        self.n_arvores = n_arvores
        self.criterio = criterio
        self.mtry = mtry
        self.semente = semente
        self.nomes_atributos = nomes_atributos
        self.arvores = []
        self.bootstraps = []          # indices sorteados por arvore
        self.oob_indices = []         # indices fora do saco por arvore
        self.importancias = {}
        self.erro_oob = None
        self.votos_oob = {}
        self.indices_atributos = []

    def _mtry(self, p):
        """m ~= sqrt(p) para classificacao (slide 16)."""
        if self.mtry:
            return max(1, min(self.mtry, p))
        return max(1, min(p, int(round(math.sqrt(p)))))

    def treinar(self, dados_treino, indices_atributos):
        rng = random.Random(self.semente)
        n = len(dados_treino)
        p = len(indices_atributos)
        m = self._mtry(p)

        self.indices_atributos = list(indices_atributos)
        self.arvores, self.bootstraps, self.oob_indices = [], [], []
        importancias = {}

        for _ in range(self.n_arvores):
            # 1. bootstrap: n sorteios com reposicao (slide 14)
            indices = [rng.randrange(n) for _ in range(n)]
            amostra = [dados_treino[i] for i in indices]
            usados = set(indices)

            # 2. arvore sobre a amostra
            arvore = construir_arvore(
                amostra, list(indices_atributos), self.criterio, m, rng,
                importancias=importancias, n_treino=n)

            # 3. OOB: o que ficou de fora
            self.arvores.append(arvore)
            self.bootstraps.append(indices)
            self.oob_indices.append([i for i in range(n) if i not in usados])

        # Media por arvore e normalizacao (slide 26)
        total = sum(importancias.values()) or 1.0
        self.importancias = {
            a: {
                'soma_ponderada': importancias.get(a, 0.0),
                'media_por_arvore': importancias.get(a, 0.0) / self.n_arvores,
                'normalizada': importancias.get(a, 0.0) / total,
            }
            for a in indices_atributos
        }

        self._calcular_oob(dados_treino)
        return self

    def _calcular_oob(self, dados_treino):
        """
        Erro OOB (slides 24 e 25): para cada instancia, so votam as arvores
        que NAO a viram no treino.
        """
        votos = {i: {} for i in range(len(dados_treino))}
        for arvore, oob in zip(self.arvores, self.oob_indices):
            for i in oob:
                classe = predizer_arvore(arvore, dados_treino[i]['atributos'])
                votos[i][classe] = votos[i].get(classe, 0) + 1

        avaliadas = erros = 0
        detalhe = {}
        for i, contagem in votos.items():
            if not contagem:
                continue                    # em todos os bootstraps: ignorada
            avaliadas += 1
            maior = max(contagem.values())
            predita = sorted(c for c, v in contagem.items() if v == maior)[0]
            acertou = predita == dados_treino[i]['classe']
            if not acertou:
                erros += 1
            detalhe[i] = {'votos': contagem, 'predita': predita,
                          'real': dados_treino[i]['classe'],
                          'acertou': acertou}

        self.votos_oob = detalhe
        self.n_oob_avaliadas = avaliadas
        self.erro_oob = erros / avaliadas if avaliadas else None
        return self.erro_oob

    def votos(self, atributos):
        """Contagem de votos das B arvores para um padrao."""
        contagem = {}
        for arvore in self.arvores:
            classe = predizer_arvore(arvore, atributos)
            contagem[classe] = contagem.get(classe, 0) + 1
        return contagem

    def predizer(self, atributos):
        """Votacao majoritaria (slide 23)."""
        contagem = self.votos(atributos)
        if not contagem:
            return None
        maior = max(contagem.values())
        return sorted(c for c, v in contagem.items() if v == maior)[0]

    def probabilidades(self, atributos):
        contagem = self.votos(atributos)
        total = sum(contagem.values()) or 1
        return {c: v / total for c, v in contagem.items()}

    def resumo_arvores(self):
        """Estatisticas por arvore — profundidade, nos, raiz, tamanho do OOB."""
        resumo = []
        for i, (arvore, oob) in enumerate(zip(self.arvores, self.oob_indices)):
            total, folhas = contar_nos(arvore)
            raiz = (self.nomes_atributos[arvore.atributo]
                    if self.nomes_atributos and not arvore.eh_folha
                    else arvore.atributo)
            resumo.append({
                'indice': i,
                'raiz': raiz,
                'profundidade': profundidade_arvore(arvore),
                'n_nos': total,
                'n_folhas': folhas,
                'n_oob': len(oob),
            })
        return resumo

    def distribuicao_das_raizes(self):
        """
        Quantas arvores usaram cada atributo na raiz.

        E a evidencia numerica do slide 22: com mtry, a raiz deixa de ser
        sempre o mesmo atributo.
        """
        contagem = {}
        for arvore in self.arvores:
            if arvore.eh_folha:
                continue
            nome = (self.nomes_atributos[arvore.atributo]
                    if self.nomes_atributos else arvore.atributo)
            contagem[nome] = contagem.get(nome, 0) + 1
        return contagem


def treinar_floresta_categorica(dados_treino, indices_atributos, **kwargs):
    """Atalho: instancia e treina numa chamada."""
    return FlorestaCategorica(**kwargs).treinar(dados_treino, indices_atributos)
