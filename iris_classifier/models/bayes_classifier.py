"""
Logica de treinamento e predicao dos classificadores Bayes Otimo (QDA) e Naive Bayes.
Implementado do zero em Python puro, sem bibliotecas de ML.
"""
import math
from core.math_utils import (
    calcular_media,
    calcular_covariancia,
    calcular_covariancia_diagonal,
    regularizar_covariancia,
    det_matriz,
    inv_matriz,
    distancia_mahalanobis_quad
)

def treinar_bayes(dados_treino, indices_atributos, naive=False):
    """
    Treina o classificador Bayes: calcula medias e matrizes de covariancia para cada classe.
    Se naive=True, forca as covariancias fora da diagonal a zero (Naive Bayes).
    Garante regularizacao para evitar matrizes singulares.
    Retorna: dict {classe: {media, cov, cov_reg, det, inv_cov}}
    """
    classes = sorted(list(set(d['classe'] for d in dados_treino)))
    model_params = {}
    
    for classe in classes:
        # Filtrar amostras da classe e selecionar os atributos
        amostras = [d['atributos'] for d in dados_treino if d['classe'] == classe]
        amostras_sel = [[s[i] for i in indices_atributos] for s in amostras]
        
        # 1. Vetor medio (m_j)
        media = calcular_media(amostras_sel)
        
        # 2. Matriz de covariancia (Sigma_j)
        if naive:
            cov = calcular_covariancia_diagonal(amostras_sel, media)
        else:
            cov = calcular_covariancia(amostras_sel, media)
            
        # Regularizar a matriz de covariancia para garantir estabilidade numerica (eps = 1e-9)
        cov_reg = regularizar_covariancia(cov, eps=1e-9)
        
        # 3. Determinante da covariancia (|Sigma_j|)
        det = det_matriz(cov_reg)
        if det <= 0:
            det = 1e-15
            
        # 4. Inversa da covariancia (Sigma_j^-1)
        inv_cov = inv_matriz(cov_reg)
        
        model_params[classe] = {
            'media': media,
            'cov': cov,
            'cov_reg': cov_reg,
            'det': det,
            'inv_cov': inv_cov
        }
        
    return model_params

def predizer_todas_classes_bayes(x, model_params, indices_atributos):
    """
    Calcula as densidades de probabilidade discriminantes logaritmicas (MAP com priori iguais):
        d_j(x) = -0.5 * ln|Sigma_j| - 0.5 * (x - m_j)^T * Sigma_j^-1 * (x - m_j)
    
    Retorna: (dict {classe: score}, classe_vencedora)
    """
    x_sel = [x[i] for i in indices_atributos]
    scores = {}
    
    for classe, params in model_params.items():
        media = params['media']
        inv_cov = params['inv_cov']
        det = params['det']
        
        # d_M^2(x, m_j)
        d_mahalanobis_sq = distancia_mahalanobis_quad(x_sel, media, inv_cov)
        
        # d_j(x)
        score = -0.5 * math.log(det) - 0.5 * d_mahalanobis_sq
        scores[classe] = score
        
    vencedor = max(scores, key=scores.get)
    return scores, vencedor

def predizer_binario_bayes(x, model_params, classe_i, classe_j, indices_atributos):
    """
    Classifica uma amostra x entre duas classes especificas (classe_i e classe_j).
    Retorna a classe com maior score discriminante.
    """
    x_sel = [x[i] for i in indices_atributos]
    
    scores = {}
    for c in [classe_i, classe_j]:
        params = model_params[c]
        media = params['media']
        inv_cov = params['inv_cov']
        det = params['det']
        d_mahalanobis_sq = distancia_mahalanobis_quad(x_sel, media, inv_cov)
        scores[c] = -0.5 * math.log(det) - 0.5 * d_mahalanobis_sq
        
    return max(scores, key=scores.get)
