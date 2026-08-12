"""
Gerador do dataset "fim de semana" em escala.

O seminario de Florestas Aleatorias usa, do inicio ao fim, o mesmo conjunto
de 10 padroes do material da disciplina (slide 9): decidir o programa do fim
de semana a partir de tres atributos categoricos.

    #   Clima   Pais visitam?  Dinheiro   Decisao
    1   Sol     Sim            Rico       Cinema
    2   Sol     Nao            Rico       Tenis
    3   Vento   Sim            Rico       Cinema
    4   Chuva   Sim            Pobre      Cinema
    5   Chuva   Nao            Rico       Ficar em casa
    6   Chuva   Sim            Pobre      Cinema
    7   Vento   Nao            Pobre      Cinema
    8   Vento   Nao            Rico       Compras
    9   Vento   Sim            Rico       Cinema
    10  Sol     Nao            Rico       Tenis

Dez instancias sao suficientes para fazer as contas a mao, mas pequenas
demais para avaliar um classificador: no proprio seminario o erro OOB deu
44,4% justamente por isso (B=3 arvores, 10 instancias, 4 classes). Este
modulo gera uma versao com N instancias do MESMO problema.

Como as 1000 instancias sao geradas
-----------------------------------
1. CONCEITO. As 10 linhas originais definem, sem ambiguidade, uma funcao
   completa sobre as 3x2x2 = 12 combinacoes possiveis de atributos:

       Pais = Sim                          -> Cinema
       Pais = Nao e Dinheiro = Pobre       -> Cinema
       Pais = Nao e Dinheiro = Rico:
           Clima = Sol                     -> Tenis
           Clima = Vento                   -> Compras
           Clima = Chuva                   -> Ficar em casa

   Todas as 10 linhas do slide obedecem a essa regra, e ela cobre as 12
   combinacoes — e por isso o conceito "verdadeiro" que replicamos.

2. ATRIBUTOS. Sorteados de forma independente, com as frequencias marginais
   observadas nas 10 instancias originais:

       Clima     Sol 30%   Vento 40%   Chuva 30%
       Pais      Sim 50%   Nao 50%
       Dinheiro  Rico 70%  Pobre 30%

3. RUIDO DE ROTULO. Uma fracao `taxa_ruido` das instancias tem o rotulo
   trocado por outra classe, sorteada uniformemente entre as 3 restantes.

   Sem ruido o problema seria uma funcao deterministica dos atributos e
   qualquer arvore razoavel acertaria 100% — exatamente a critica que o
   professor fez as metricas do projeto. Com ruido existe um teto teorico
   de acerto (o classificador de Bayes acerta 1 - taxa_ruido), o erro OOB
   passa a medir algo real e os testes de significancia tem o que comparar.

   A coluna `ruido` do CSV marca quais linhas foram alteradas, para que o
   experimento continue auditavel.

Saida
-----
`data/fim_de_semana_1000.csv`, com as colunas:

    id             1..N
    clima          Sol | Vento | Chuva
    pais           Sim | Nao
    dinheiro       Rico | Pobre
    decisao        Cinema | Tenis | Compras | Ficar em casa
    clima_cod      Sol=0  Vento=1  Chuva=2
    pais_cod       Nao=0  Sim=1
    dinheiro_cod   Pobre=0  Rico=1
    ruido          1 se o rotulo foi trocado, 0 caso contrario

As colunas `_cod` existem para que o mesmo arquivo sirva aos demais modulos
do projeto (Distancia Minima, Bayes, Regra Delta, metricas), que trabalham
com vetores numericos. Ver a ressalva sobre codificacao ordinal em
`docs/seminario_dataset_fim_de_semana.md`.

Uso:
    python -m data.gerar_fim_de_semana                 # 1000 instancias
    python -m data.gerar_fim_de_semana --n 5000
"""
import argparse
import os
import random

# ---------------------------------------------------------------------------
# Dominio (slide 9)
# ---------------------------------------------------------------------------
CLIMAS = ['Sol', 'Vento', 'Chuva']
PAIS = ['Sim', 'Nao']
DINHEIRO = ['Rico', 'Pobre']
DECISOES = ['Cinema', 'Tenis', 'Compras', 'Ficar em casa']

# Frequencias marginais observadas nas 10 instancias originais
MARGINAIS = {
    'clima': [('Sol', 0.3), ('Vento', 0.4), ('Chuva', 0.3)],
    'pais': [('Sim', 0.5), ('Nao', 0.5)],
    'dinheiro': [('Rico', 0.7), ('Pobre', 0.3)],
}

# Codificacao ordinal usada nas colunas `_cod`
CODIGOS = {
    'clima': {'Sol': 0, 'Vento': 1, 'Chuva': 2},
    'pais': {'Nao': 0, 'Sim': 1},
    'dinheiro': {'Pobre': 0, 'Rico': 1},
}

# As 10 instancias originais do slide 9 — usadas para validar o conceito
ORIGINAIS = [
    ('Sol', 'Sim', 'Rico', 'Cinema'),
    ('Sol', 'Nao', 'Rico', 'Tenis'),
    ('Vento', 'Sim', 'Rico', 'Cinema'),
    ('Chuva', 'Sim', 'Pobre', 'Cinema'),
    ('Chuva', 'Nao', 'Rico', 'Ficar em casa'),
    ('Chuva', 'Sim', 'Pobre', 'Cinema'),
    ('Vento', 'Nao', 'Pobre', 'Cinema'),
    ('Vento', 'Nao', 'Rico', 'Compras'),
    ('Vento', 'Sim', 'Rico', 'Cinema'),
    ('Sol', 'Nao', 'Rico', 'Tenis'),
]

CAMINHO_PADRAO = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    'data', 'fim_de_semana_1000.csv')


def conceito(clima, pais, dinheiro):
    """
    Rotulo verdadeiro (sem ruido) de uma combinacao de atributos.

    E a funcao que as 10 instancias do slide 9 definem — ver `validar_conceito`.
    """
    if pais == 'Sim':
        return 'Cinema'
    if dinheiro == 'Pobre':
        return 'Cinema'
    return {'Sol': 'Tenis', 'Vento': 'Compras', 'Chuva': 'Ficar em casa'}[clima]


def validar_conceito():
    """
    Confere que a regra reproduz as 10 instancias originais e cobre as 12
    combinacoes possiveis. Levanta AssertionError se algo divergir.
    """
    for clima, pais, dinheiro, esperado in ORIGINAIS:
        obtido = conceito(clima, pais, dinheiro)
        assert obtido == esperado, (
            f'({clima}, {pais}, {dinheiro}): conceito devolve {obtido!r}, '
            f'mas o slide 9 diz {esperado!r}')

    combinacoes = [(c, p, d) for c in CLIMAS for p in PAIS for d in DINHEIRO]
    assert len(combinacoes) == 12
    for c, p, d in combinacoes:
        assert conceito(c, p, d) in DECISOES
    return len(combinacoes)


def _sortear(rng, distribuicao):
    """Sorteio categorico a partir de uma lista [(valor, probabilidade), ...]."""
    u = rng.random()
    acumulado = 0.0
    for valor, prob in distribuicao:
        acumulado += prob
        if u < acumulado:
            return valor
    return distribuicao[-1][0]


def gerar(n=1000, taxa_ruido=0.08, semente=42):
    """
    Gera `n` instancias do problema do fim de semana.

    Devolve uma lista de dicts com as chaves do CSV. Determinístico para uma
    mesma semente — o arquivo versionado no repositorio e reprodutivel.
    """
    validar_conceito()
    rng = random.Random(semente)
    linhas = []

    for i in range(1, n + 1):
        clima = _sortear(rng, MARGINAIS['clima'])
        pais = _sortear(rng, MARGINAIS['pais'])
        dinheiro = _sortear(rng, MARGINAIS['dinheiro'])

        verdadeiro = conceito(clima, pais, dinheiro)
        if rng.random() < taxa_ruido:
            decisao = rng.choice([d for d in DECISOES if d != verdadeiro])
            ruido = 1
        else:
            decisao = verdadeiro
            ruido = 0

        linhas.append({
            'id': i,
            'clima': clima,
            'pais': pais,
            'dinheiro': dinheiro,
            'decisao': decisao,
            'clima_cod': CODIGOS['clima'][clima],
            'pais_cod': CODIGOS['pais'][pais],
            'dinheiro_cod': CODIGOS['dinheiro'][dinheiro],
            'ruido': ruido,
        })

    return linhas


COLUNAS = ['id', 'clima', 'pais', 'dinheiro', 'decisao',
           'clima_cod', 'pais_cod', 'dinheiro_cod', 'ruido']


def escrever_csv(linhas, caminho=CAMINHO_PADRAO):
    """Grava o CSV em UTF-8, sem depender do modulo csv nem do pandas."""
    os.makedirs(os.path.dirname(caminho), exist_ok=True)
    with open(caminho, 'w', encoding='utf-8', newline='') as f:
        f.write(','.join(COLUNAS) + '\n')
        for linha in linhas:
            f.write(','.join(str(linha[c]) for c in COLUNAS) + '\n')
    return caminho


def resumo(linhas):
    """Contagens uteis para conferir o arquivo gerado."""
    def contar(chave):
        c = {}
        for linha in linhas:
            c[linha[chave]] = c.get(linha[chave], 0) + 1
        return c

    n = len(linhas)
    ruidosas = sum(linha['ruido'] for linha in linhas)
    return {
        'n': n,
        'clima': contar('clima'),
        'pais': contar('pais'),
        'dinheiro': contar('dinheiro'),
        'decisao': contar('decisao'),
        'ruidosas': ruidosas,
        'taxa_ruido_efetiva': ruidosas / n if n else 0.0,
        'teto_bayes': 1 - ruidosas / n if n else 0.0,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    p.add_argument('--n', type=int, default=1000,
                   help='numero de instancias (padrao: 1000)')
    p.add_argument('--ruido', type=float, default=0.08,
                   help='taxa de ruido de rotulo (padrao: 0.08)')
    p.add_argument('--semente', type=int, default=42)
    p.add_argument('--saida', default=CAMINHO_PADRAO)
    args = p.parse_args()

    n_combinacoes = validar_conceito()
    print(f'Conceito validado: reproduz as 10 instancias do slide 9 e cobre '
          f'as {n_combinacoes} combinacoes possiveis.')

    linhas = gerar(args.n, args.ruido, args.semente)
    caminho = escrever_csv(linhas, args.saida)
    r = resumo(linhas)

    print(f'\nArquivo gerado: {caminho}')
    print(f'Instancias: {r["n"]}')
    print(f'Ruido: {r["ruidosas"]} rotulos trocados '
          f'({r["taxa_ruido_efetiva"]:.2%}) '
          f'-> teto teorico de acerto: {r["teto_bayes"]:.2%}')
    for chave in ('clima', 'pais', 'dinheiro', 'decisao'):
        itens = sorted(r[chave].items(), key=lambda kv: -kv[1])
        print(f'  {chave:<9} ' + '  '.join(
            f'{v}={c} ({c / r["n"]:.1%})' for v, c in itens))


if __name__ == '__main__':
    main()
