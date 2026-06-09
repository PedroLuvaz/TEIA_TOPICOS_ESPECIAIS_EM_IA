/**
 * Cliente da API para o Classificador Iris.
 * Abstrai as chamadas HTTP (fetch) com tratamento de erros integrado.
 */

const API_BASE = '/api';

/**
 * Função utilitária para chamadas GET.
 */
async function fetchGet(url, params = {}) {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
        if (value !== undefined && value !== null) {
            query.append(key, value);
        }
    }
    const queryString = query.toString();
    const endpoint = queryString ? `${API_BASE}${url}?${queryString}` : `${API_BASE}${url}`;
    
    const resposta = await fetch(endpoint);
    if (!resposta.ok) {
        const errData = await resposta.json().catch(() => ({}));
        throw new Error(errData.erro || `Erro na requisição: ${resposta.status}`);
    }
    return resposta.json();
}

/**
 * Função utilitária para chamadas POST.
 */
async function fetchPost(url, body = {}) {
    const resposta = await fetch(`${API_BASE}${url}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(body)
    });
    if (!resposta.ok) {
        const errData = await resposta.json().catch(() => ({}));
        throw new Error(errData.erro || `Erro na requisição: ${resposta.status}`);
    }
    return resposta.json();
}

// ---------------------------------------------------------------------------
// Chamadas de Dados
// ---------------------------------------------------------------------------
export async function getInfoDataset(dataset = 'v1') {
    return fetchGet('/data', { dataset });
}

export async function getAmostras(dataset = 'v1', split = 'todos', classe = null, limite = null, offset = 0) {
    return fetchGet('/data/samples', { dataset, split, classe, limite, offset });
}

// ---------------------------------------------------------------------------
// Chamadas do Classificador de Distância Mínima (Aba 1)
// ---------------------------------------------------------------------------
export async function getPrototipos(dataset = 'v1', features = '2,3') {
    return fetchGet('/prototypes', { dataset, features });
}

export async function classificarAmostra(featuresInput, indicesAtributos, dataset = 'v1') {
    return fetchPost('/classify', {
        features: featuresInput,
        indices_atributos: indicesAtributos,
        dataset: dataset
    });
}

export async function getFronteiras(features = '2,3', dataset = 'v1') {
    return fetchGet('/boundaries', { features, dataset });
}

export async function getDistanciaMetrics(features = '2,3', dataset = 'v1') {
    return fetchGet('/distancia/metrics', { features, dataset });
}

// ---------------------------------------------------------------------------
// Chamadas do Perceptron (Aba 2)
// ---------------------------------------------------------------------------
export async function treinarPerceptron(config) {
    // config: { dataset, indices_atributos, taxa_aprendizado, max_epocas, classe_pos, classe_neg, proporcao_treino, semente }
    return fetchPost('/perceptron/train', config);
}

export async function predizerPerceptron(featuresInput, pesos) {
    return fetchPost('/perceptron/predict', { features: featuresInput, pesos });
}

// ---------------------------------------------------------------------------
// Chamadas da Regra Delta e Delta OvA (Aba 2)
// ---------------------------------------------------------------------------
export async function treinarDelta(config) {
    return fetchPost('/delta/train', config);
}

export async function treinarDeltaOva(config) {
    return fetchPost('/delta-ova/train', config);
}

export async function predizerDelta(featuresInput, pesos, classePos, classeNeg) {
    return fetchPost('/delta/predict', { features: featuresInput, pesos, classe_pos: classePos, classe_neg: classeNeg });
}

export async function predizerDeltaOva(featuresInput, pesos) {
    return fetchPost('/delta-ova/predict', { features: featuresInput, pesos });
}

// ---------------------------------------------------------------------------
// Chamadas de Métricas Avançadas (Aba 3)
// ---------------------------------------------------------------------------
export async function treinarTodosModelos(config) {
    // config: { dataset, atributos, proporcao_treino, semente, comparacao }
    return fetchPost('/metricas/train-all', config);
}

export async function zTestComparacao(modeloA, modeloB) {
    return fetchPost('/metricas/z-test', { modelo_a: modeloA, modelo_b: modeloB });
}

// ---------------------------------------------------------------------------
// Chamadas de Gráficos (Base64)
// ---------------------------------------------------------------------------
export async function getPlotScatter(dataset = 'v1', atributos = 'petalas') {
    return fetchGet('/plots/scatter', { dataset, atributos });
}

export async function getPlotBoundary(dataset = 'v1', atributos = 'petalas', classe1 = 'setosa', classe2 = 'versicolor') {
    return fetchGet('/plots/boundary', { dataset, atributos, classe1, classe2 });
}

export async function getPlotConfusion(modelo = 'Dist. Minima') {
    return fetchGet('/plots/confusion', { modelo });
}

export async function getPlotConvergence(historico, metricaNome, modeloTitulo) {
    return fetchPost('/plots/convergence', {
        historico: historico,
        metrica_nome: metricaNome,
        modelo_titulo: modeloTitulo
    });
}
