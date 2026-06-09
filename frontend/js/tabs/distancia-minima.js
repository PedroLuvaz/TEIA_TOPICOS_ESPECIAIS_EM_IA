/**
 * Aba 1 — Classificador de Distância Mínima
 */

import * as api from '../api.js';
import * as ui from '../components.js';
import * as charts from '../charts.js';

let activeContainer = null;
let currentDataset = 'v1';
let currentAttributes = 'petalas';
let currentGraphMode = 'dispersao'; // 'dispersao' ou 'setosa_versicolor', etc.

/**
 * Inicializa a aba.
 */
export async function init(container) {
    activeContainer = container;
    
    // Constrói o layout em três colunas (Grid flexível)
    container.innerHTML = `
        <div class="tab-grid">
            <!-- Coluna 1: Controles -->
            <div class="col-controls">
                <div class="card">
                    <div class="card-header">
                        <h4>Dataset</h4>
                    </div>
                    <div class="form-group" style="padding: 10px 0;">
                        <label class="radio-label">
                            <input type="radio" name="dataset-dm" value="v1" checked>
                            <span>Base Iris Original (v1)</span>
                        </label>
                        <label class="radio-label">
                            <input type="radio" name="dataset-dm" value="v2">
                            <span>Base Iris Estendida (v2)</span>
                        </label>
                    </div>
                </div>
                
                <div class="card mt-16">
                    <div class="card-header">
                        <h4>Atributos do Modelo</h4>
                    </div>
                    <div class="form-group" style="padding: 10px 0;">
                        <label class="radio-label">
                            <input type="radio" name="attr-dm" value="petalas" checked>
                            <span>Pétalas (índices [2, 3])</span>
                        </label>
                        <label class="radio-label">
                            <input type="radio" name="attr-dm" value="sepalas">
                            <span>Sépalas (índices [0, 1])</span>
                        </label>
                    </div>
                </div>
                
                <div class="card mt-16">
                    <div class="card-header">
                        <h4>Visualização</h4>
                    </div>
                    <div class="form-group" style="padding: 10px 0;" id="visualizacao-grupo">
                        <label class="radio-label">
                            <input type="radio" name="graph-dm" value="dispersao" checked>
                            <span>Dispersão Geral</span>
                        </label>
                        <label class="radio-label">
                            <input type="radio" name="graph-dm" value="setosa,versicolor">
                            <span>Fronteira: Setosa × Versicolor</span>
                        </label>
                        <label class="radio-label">
                            <input type="radio" name="graph-dm" value="versicolor,virginica">
                            <span>Fronteira: Versicolor × Virginica</span>
                        </label>
                        <label class="radio-label">
                            <input type="radio" name="graph-dm" value="setosa,virginica">
                            <span>Fronteira: Setosa × Virginica</span>
                        </label>
                    </div>
                </div>
                
                <button class="btn btn-secondary w-100 mt-16" id="btn-memoria-calculo">
                    📂 Memória de Cálculo
                </button>
            </div>
            
            <!-- Coluna 2: Gráfico Matplotlib -->
            <div class="col-chart">
                <div class="card h-100 flex-column">
                    <div class="card-header flex-row justify-between align-center">
                        <h4>Gráfico do Espaço de Atributos</h4>
                        <span class="badge badge-info" id="graph-tag">Dispersão</span>
                    </div>
                    <div id="plot-dm-container" class="flex-grow flex-center" style="min-height: 400px;">
                        <!-- Gráfico injetado pelo JS -->
                    </div>
                </div>
            </div>
            
            <!-- Coluna 3: Métricas e Classificação Manual -->
            <div class="col-metrics">
                <div class="card">
                    <div class="card-header">
                        <h4>Desempenho no Teste (30%)</h4>
                    </div>
                    <div class="metrics-grid mt-10" id="metrics-panel">
                        <!-- Injetado por JS -->
                    </div>
                </div>
                
                <div class="card mt-16">
                    <div class="card-header">
                        <h4>Protótipos Calculados</h4>
                    </div>
                    <div id="prototypes-panel" class="mt-10">
                        <!-- Injetado por JS -->
                    </div>
                </div>
                
                <div class="card mt-16">
                    <div class="card-header">
                        <h4>Classificação de Amostra</h4>
                    </div>
                    <form id="form-classify" class="mt-10">
                        <div class="flex-row gap-10">
                            <div class="flex-grow" id="input-x1-container"></div>
                            <div class="flex-grow" id="input-x2-container"></div>
                        </div>
                        <button type="submit" class="btn btn-primary w-100 mt-10">Classificar</button>
                    </form>
                    
                    <div id="classify-result" class="mt-10" style="display: none;">
                        <!-- Injetado por JS -->
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Bind eventos
    const setupEvents = () => {
        container.querySelectorAll('input[name="dataset-dm"]').forEach(el => {
            el.addEventListener('change', (e) => {
                currentDataset = e.target.value;
                atualizarTudo();
            });
        });
        
        container.querySelectorAll('input[name="attr-dm"]').forEach(el => {
            el.addEventListener('change', (e) => {
                currentAttributes = e.target.value;
                atualizarLabelsInput();
                atualizarTudo();
            });
        });
        
        container.querySelectorAll('input[name="graph-dm"]').forEach(el => {
            el.addEventListener('change', (e) => {
                currentGraphMode = e.target.value;
                atualizarGrafico();
            });
        });
        
        container.querySelector('#form-classify').addEventListener('submit', executarClassificacaoManual);
        container.querySelector('#btn-memoria-calculo').addEventListener('click', abrirModalMemoriaCalculo);
    };
    
    setupEvents();
    atualizarLabelsInput();
    await atualizarTudo();
}

/**
 * Atualiza os labels dos inputs de classificação manual baseados nas features ativas.
 */
function atualizarLabelsInput() {
    const x1Container = activeContainer.querySelector('#input-x1-container');
    const x2Container = activeContainer.querySelector('#input-x2-container');
    if (!x1Container || !x2Container) return;
    
    const labelX1 = currentAttributes === 'petalas' ? 'C. Pétala (x₁)' : 'C. Sépala (x₁)';
    const labelX2 = currentAttributes === 'petalas' ? 'L. Pétala (x₂)' : 'L. Sépala (x₂)';
    
    x1Container.innerHTML = '';
    x2Container.innerHTML = '';
    
    x1Container.appendChild(ui.createInputGroup(labelX1, 'number', 'input-x1', { step: 0.1, required: true, value: 4.5 }));
    x2Container.appendChild(ui.createInputGroup(labelX2, 'number', 'input-x2', { step: 0.1, required: true, value: 1.5 }));
}

/**
 * Atualiza o gráfico Matplotlib no painel central.
 */
async function atualizarGrafico() {
    const plotContainer = activeContainer.querySelector('#plot-dm-container');
    const graphTag = activeContainer.querySelector('#graph-tag');
    if (!plotContainer) return;
    
    charts.showPlotLoading(plotContainer);
    
    try {
        let base64Img = '';
        if (currentGraphMode === 'dispersao') {
            graphTag.innerText = 'Dispersão';
            const res = await api.getPlotScatter(currentDataset, currentAttributes);
            base64Img = res.image;
        } else {
            graphTag.innerText = 'Fronteira';
            const [c1, c2] = currentGraphMode.split(',');
            const res = await api.getPlotBoundary(currentDataset, currentAttributes, c1, c2);
            base64Img = res.image;
        }
        charts.renderPlot(plotContainer, base64Img, 'Gráfico do classificador');
    } catch (e) {
        charts.renderPlotError(plotContainer, `Erro ao carregar o gráfico: ${e.message}`);
    }
}

/**
 * Atualiza métricas, protótipos e o gráfico.
 */
async function atualizarTudo() {
    await atualizarGrafico();
    await atualizarMetricasEPrototipos();
}

/**
 * Busca e renderiza métricas e protótipos da API.
 */
async function atualizarMetricasEPrototipos() {
    const metricsPanel = activeContainer.querySelector('#metrics-panel');
    const prototypesPanel = activeContainer.querySelector('#prototypes-panel');
    if (!metricsPanel || !prototypesPanel) return;
    
    metricsPanel.innerHTML = '';
    prototypesPanel.innerHTML = '';
    
    metricsPanel.appendChild(ui.createSkeleton(2));
    prototypesPanel.appendChild(ui.createSkeleton(2));
    
    try {
        const featuresParam = currentAttributes === 'petalas' ? '2,3' : '0,1';
        
        // Chamada paralela das métricas e dos protótipos
        const [metricasData, prototiposData] = await Promise.all([
            api.getDistanciaMetrics(featuresParam, currentDataset),
            api.getPrototipos(currentDataset, featuresParam)
        ]);
        
        // Renderiza métricas
        metricsPanel.innerHTML = '';
        const accPerc = (metricasData.acuracia_teste * 100).toFixed(2) + '%';
        
        metricsPanel.appendChild(ui.createMetricCard('Acurácia Teste', accPerc, { variant: 'success' }));
        metricsPanel.appendChild(ui.createMetricCard('Erros Teste', metricasData.erros_teste, { variant: metricasData.erros_teste > 0 ? 'warning' : 'default' }));
        metricsPanel.appendChild(ui.createMetricCard('Amostras Treino', metricasData.total_treino));
        metricsPanel.appendChild(ui.createMetricCard('Amostras Teste', metricasData.total_teste));
        
        // Renderiza tabela de protótipos
        prototypesPanel.innerHTML = '';
        const cols = ['Classe', 'Vetor Médio [x1, x2]'];
        const rows = Object.entries(prototiposData.prototipos).map(([classe, vetor]) => {
            const tag = ui.createSpeciesTag(classe);
            const vetorTexto = `[${vetor[0].toFixed(3)}, ${vetor[1].toFixed(3)}]`;
            return [tag, vetorTexto];
        });
        
        prototypesPanel.appendChild(ui.createDataTable(cols, rows));
        
    } catch (e) {
        metricsPanel.innerHTML = `<span class="text-danger">Falha ao carregar métricas: ${e.message}</span>`;
        prototypesPanel.innerHTML = `<span class="text-danger">Falha ao carregar protótipos.</span>`;
    }
}

/**
 * Classifica a amostra do input manual.
 */
async function executarClassificacaoManual(e) {
    e.preventDefault();
    const resultPanel = activeContainer.querySelector('#classify-result');
    if (!resultPanel) return;
    
    const x1 = parseFloat(activeContainer.querySelector('#input-x1').value);
    const x2 = parseFloat(activeContainer.querySelector('#input-x2').value);
    
    if (isNaN(x1) || isNaN(x2)) {
        ui.showToast('Insira coordenadas válidas', 'warning');
        return;
    }
    
    resultPanel.style.display = 'block';
    resultPanel.innerHTML = '';
    resultPanel.appendChild(ui.createSkeleton(2));
    
    try {
        const featuresParam = currentAttributes === 'petalas' ? [2, 3] : [0, 1];
        
        // Monta o vetor de 4 features para a chamada (injetando zero nos atributos não utilizados)
        const fullVector = [0, 0, 0, 0];
        fullVector[featuresParam[0]] = x1;
        fullVector[featuresParam[1]] = x2;
        
        const res = await api.classificarAmostra(fullVector, featuresParam, currentDataset);
        
        resultPanel.innerHTML = `
            <div class="result-box mt-10" style="padding: 12px; background: rgba(255,255,255,0.02); border: 1px solid var(--border-medium); border-radius: var(--radius-md);">
                <div class="flex-row align-center gap-10">
                    <span class="text-secondary">Resultado:</span>
                    <span class="tag tag-${res.classe}" style="font-size: 15px; font-weight: 700;">
                        ${res.classe.toUpperCase()}
                    </span>
                </div>
                <div class="mt-10">
                    <span class="text-secondary" style="font-size: 12px; display: block; margin-bottom: 4px;">Funções Discriminantes d_j(x):</span>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px;">
                        ${Object.entries(res.scores).map(([classe, score]) => `
                            <div style="background: var(--bg-input); padding: 6px; border-radius: var(--radius-sm); border: 1px solid var(--border-subtle); text-align: center;">
                                <span class="tag tag-${classe}" style="font-size: 10px; padding: 2px 4px;">${classe.charAt(0).toUpperCase()}</span>
                                <span style="font-family: 'JetBrains Mono', monospace; font-size: 11px; display: block; margin-top: 4px; color: ${score === Math.max(...Object.values(res.scores)) ? 'var(--accent-green)' : 'var(--text-primary)'};">
                                    ${score.toFixed(4)}
                                </span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            </div>
        `;
    } catch (e) {
        resultPanel.innerHTML = `<span class="text-danger">Erro na classificação: ${e.message}</span>`;
    }
}

/**
 * Abre o modal de Memória de Cálculo.
 */
function abrirModalMemoriaCalculo() {
    const formulasHTML = `
        <div style="display: flex; flex-direction: column; gap: 16px; font-size: 14px; color: var(--text-primary);">
            <div>
                <h5 style="color: var(--accent-blue); margin-bottom: 6px;">1. Vetores Médios (Protótipos)</h5>
                <p style="margin-bottom: 6px;">Calculados como o baricentro (média) dos vetores de cada classe nos dados de treino:</p>
                <code style="display: block; background: var(--bg-input); padding: 10px; border-radius: var(--radius-sm); font-family: monospace; text-align: center; font-size: 15px; border: 1px solid var(--border-subtle);">
                    m_j = (1 / N_j) * &Sigma; x_i
                </code>
            </div>
            
            <div>
                <h5 style="color: var(--accent-blue); margin-bottom: 6px;">2. Função Discriminante</h5>
                <p style="margin-bottom: 6px;">Dada uma amostra <code style="font-family: monospace;">x</code>, ela pertence à classe que maximizar a projeção discriminante linear:</p>
                <code style="display: block; background: var(--bg-input); padding: 10px; border-radius: var(--radius-sm); font-family: monospace; text-align: center; font-size: 15px; border: 1px solid var(--border-subtle);">
                    d_j(x) = x&bull;m_j - 0.5 * ||m_j||&sup2;
                </code>
                <p style="margin-top: 6px; font-size: 12px; color: var(--text-secondary);">
                    Nota: Maximizar <code style="font-family: monospace;">d_j(x)</code> é matematicamente idêntico a minimizar a distância Euclidiana <code style="font-family: monospace;">||x - m_j||</code> se as classes tiverem a mesma covariância.
                </p>
            </div>
            
            <div>
                <h5 style="color: var(--accent-blue); margin-bottom: 6px;">3. Equação da Superfície de Decisão</h5>
                <p style="margin-bottom: 6px;">A fronteira linear entre as classes <code style="font-family: monospace;">i</code> e <code style="font-family: monospace;">j</code> é obtida quando <code style="font-family: monospace;">d_i(x) - d_j(x) = 0</code>:</p>
                <code style="display: block; background: var(--bg-input); padding: 10px; border-radius: var(--radius-sm); font-family: monospace; text-align: center; font-size: 15px; border: 1px solid var(--border-subtle);">
                    w&bull;x + b = 0 &nbsp;&nbsp;&rArr;&nbsp;&nbsp; w = m_i - m_j, &nbsp; b = -0.5 * (||m_i||&sup2; - ||m_j||&sup2;)
                </code>
            </div>
        </div>
    `;
    
    ui.createModal('Memória de Cálculo — Distância Mínima', formulasHTML);
}

/**
 * Finaliza a aba.
 */
export function destroy() {
    activeContainer = null;
}
