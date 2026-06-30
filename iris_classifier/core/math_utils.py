"""
Algebra linear em Python puro para o Classificador de Distancia Minima.

Todas as operacoes sao implementadas com listas nativas e lacos for —
sem numpy, scipy ou qualquer biblioteca de algebra.

Formulas centrais:
  Prototipo:      m_j = (1/N_j) * sum(x  para x em omega_j)
  Discriminante:  d_j(x) = x^T * m_j  -  (1/2) * m_j^T * m_j
  Fronteira:      w = m_i - m_j,   b = -(1/2)*(||m_i||^2 - ||m_j||^2)
"""

import math


def produto_escalar(a, b):
    """
    a^T * b = sum( a_i * b_i )
    Operacao base para o discriminante e para o calculo do bias da fronteira.
    """
    return sum(x * y for x, y in zip(a, b))


def subtrair_vetores(a, b):
    """
    a - b  (componente a componente)
    Usado para calcular w = m_i - m_j (vetor normal a fronteira de decisao).
    """
    return [x - y for x, y in zip(a, b)]


def multiplicar_escalar(s, v):
    """s * v  (escalar por vetor)"""
    return [s * x for x in v]


def distancia_euclidiana(a, b):
    """
    ||a - b|| = sqrt( sum( (a_i - b_i)^2 ) )
    Regra de decisao: argmin_j ||x - m_j||  (menor distancia = classe predita).
    Matematicamente equivalente a argmax_j d_j(x).
    """
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def calcular_media(vetores):
    """
    m = (1/N) * sum(v  para v em vetores)
    Calcula o prototipo (centroide) de uma classe a partir de suas amostras de treino.
    """
    if not vetores:
        return []
    n = len(vetores)
    dim = len(vetores[0])
    media = [0.0] * dim
    for v in vetores:
        for i in range(dim):
            media[i] += v[i]
    return [x / n for x in media]


def discriminante(x, mj):
    """
    Funcao Discriminante Linear:
        d_j(x) = x^T * m_j  -  (1/2) * m_j^T * m_j

    Derivacao: minimizar ||x - m_j||^2 expande para x^Tx - 2*x^T*m_j + m_j^T*m_j.
    Como x^Tx e constante (nao depende de j), minimizar a distancia equivale a
    maximizar  x^T*m_j - (1/2)*m_j^T*m_j  =  d_j(x).

    Regra de decisao: argmax_j d_j(x)  <==>  argmin_j ||x - m_j||
    """
    return produto_escalar(x, mj) - 0.5 * produto_escalar(mj, mj)


def coeficientes_superficie_decisao(mi, mj):
    """
    Fronteira entre classes i e j: d_i(x) = d_j(x)  =>  w^T * x + b = 0

    Derivacao:
        d_i(x) - d_j(x) = 0
        (m_i - m_j)^T * x  -  (1/2)*(||m_i||^2 - ||m_j||^2)  =  0

    Coeficientes:
        w = m_i - m_j
        b = -(1/2) * (||m_i||^2 - ||m_j||^2)

    Para plotar a reta em 2D: x2 = (-w1*x1 - b) / w2

    Retorna: (w, b)
    """
    w = subtrair_vetores(mi, mj)
    b = -0.5 * (produto_escalar(mi, mi) - produto_escalar(mj, mj))
    return w, b


# ===========================================================================
# Operacoes de Matriz para Bayes e Naive Bayes (Python Puro)
# ===========================================================================

def calcular_covariancia(amostras, media):
    """
    Calcula a matriz de covariancia de uma lista de vetores amostras de dimensao d.
    Formula: Sigma = (1 / (N - 1)) * sum( (x - m)(x - m)^T )
    """
    n = len(amostras)
    d = len(media)
    cov = [[0.0 for _ in range(d)] for _ in range(d)]
    if n <= 1:
        return cov
    for x in amostras:
        diff = [x[i] - media[i] for i in range(d)]
        for i in range(d):
            for j in range(d):
                cov[i][j] += diff[i] * diff[j]
    for i in range(d):
        for j in range(d):
            cov[i][j] /= (n - 1)
    return cov


def calcular_covariancia_diagonal(amostras, media):
    """
    Calcula a matriz de covariancia diagonal (utilizada no Naive Bayes).
    Garante que os elementos fora da diagonal principal sejam zero.
    """
    cov = calcular_covariancia(amostras, media)
    d = len(media)
    for i in range(d):
        for j in range(d):
            if i != j:
                cov[i][j] = 0.0
    return cov


def regularizar_covariancia(cov, eps=1e-9):
    """
    Adiciona um pequeno valor regularizador a diagonal principal para
    garantir que a matriz seja estritamente positiva-definida e inversivel.
    """
    d = len(cov)
    cov_reg = [row[:] for row in cov]
    for i in range(d):
        cov_reg[i][i] += eps
    return cov_reg


def det_matriz(M):
    """
    Calcula o determinante de uma matriz quadrada M (lista de listas) recursivamente
    usando a expansao de cofatores de Laplace. Funciona para qualquer dimensao d.
    """
    n = len(M)
    if n == 1:
        return M[0][0]
    if n == 2:
        return M[0][0] * M[1][1] - M[0][1] * M[1][0]
    det = 0.0
    for c in range(n):
        # Submatriz excluindo linha 0 e coluna c
        sub_M = [row[:c] + row[c+1:] for row in M[1:]]
        det += ((-1) ** c) * M[0][c] * det_matriz(sub_M)
    return det


def inv_matriz(M):
    """
    Calcula a inversa de uma matriz quadrada M (lista de listas)
    usando o algoritmo de eliminacao de Gauss-Jordan com pivotamento parcial.
    Retorna a matriz inversa (lista de listas).
    """
    n = len(M)
    # Criar matriz identidade I de mesma dimensao n x n
    I = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    # Fazer copia de trabalho da matriz M
    A = [row[:] for row in M]
    
    for i in range(n):
        # Pivotamento parcial: encontrar linha com maior valor absoluto na coluna i
        max_row = i
        for r in range(i + 1, n):
            if abs(A[r][i]) > abs(A[max_row][i]):
                max_row = r
        # Trocar linhas em A e I
        A[i], A[max_row] = A[max_row], A[i]
        I[i], I[max_row] = I[max_row], I[i]
        
        pivot = A[i][i]
        if abs(pivot) < 1e-15:
            raise ValueError("Matriz singular ou quase-singular, nao pode ser invertida.")
            
        # Normalizar a linha do pivo (dividir todos os elementos pelo pivo)
        for j in range(i, n):
            A[i][j] /= pivot
        for j in range(n):
            I[i][j] /= pivot
            
        # Eliminar os elementos nas outras linhas para a coluna i
        for r in range(n):
            if r == i:
                continue
            factor = A[r][i]
            for j in range(i, n):
                A[r][j] -= factor * A[i][j]
            for j in range(n):
                I[r][j] -= factor * I[i][j]
    return I


def distancia_mahalanobis_quad(x, media, inv_cov):
    """
    Calcula o quadrado da distancia de Mahalanobis:
    d_M^2(x, m) = (x - m)^T * Sigma^-1 * (x - m)
    """
    d = len(media)
    diff = [x[i] - media[i] for i in range(d)]
    
    # Calcular diff^T * Sigma^-1
    temp = [0.0] * d
    for j in range(d):
        temp[j] = sum(diff[i] * inv_cov[i][j] for i in range(d))
        
    # Calcular (diff^T * Sigma^-1) * diff
    return sum(temp[j] * diff[j] for j in range(d))


def inv_chi2_4df(p):
    """
    Calcula o quantil da distribuicao Chi-Quadrado com 4 graus de liberdade.
    Usa o metodo de Newton-Raphson para resolver:
        1 - exp(-x/2) * (1 + x/2) - p = 0
    """
    if p <= 0:
        return 0.0
    if p >= 1:
        return 40.0
    # Chute inicial (media da chi2 com 4 gl e 4, variancia e 8)
    x = 4.0
    for _ in range(20):
        exp_term = math.exp(-x / 2.0)
        f_x = 1.0 - exp_term * (1.0 + x / 2.0) - p
        df_x = 0.25 * x * exp_term  # derivada da CDF (PDF da Chi-Quadrado com 4 gl)
        if abs(df_x) < 1e-15:
            break
        dx = f_x / df_x
        x -= dx
        if x < 0:
            x = 1e-9
        if abs(dx) < 1e-7:
            break
    return x


def inv_chi2(p, df):
    """
    Calcula o quantil da distribuicao Chi-Quadrado para probabilidade p e df graus de liberdade.
    Suporta df = 2 e df = 4.
    """
    if df == 2:
        # CDF para df=2 e F(x) = 1 - exp(-x/2)
        # Inversa analitica: x = -2 * ln(1 - p)
        return -2.0 * math.log(1.0 - p)
    elif df == 4:
        return inv_chi2_4df(p)
    else:
        # Fallback razoavel para outros graus de liberdade
        return -2.0 * math.log(1.0 - p)


