/**
 * Aba 2 — Perceptron e Regra Delta
 */

import * as api from '../api.js';
import * as ui from '../components.js';
import * as charts from '../charts.js';

let activeContainer = null;
let currentAlgo = 'perceptron';
let currentPair = 'setosa,versicolor';
let currentAttr = 'petalas';

let trainedWeights = null;
let trainedBias = null;
let trainedWeightsOva = null; // Para OVA {classe: w}
let classPos = '';
let classNeg = '';

/**
 * Inicializa a aba.
 */
export async function init(container) {
    activeContainer = container;
    
    container.innerHTML = `
        <div class="tab-grid">
            <!-- Coluna 1: Controles -->
            <div class="col-controls">
                <div class="card">
                    <div class="card-header">
                        <h4>Algoritmo</h4>
                    </div>
                    <div class="form-group" style="padding: 10px 0;">
                        <label class="radio-label">
                            <input type="radio" name="algo-pd" value="perceptron" checked>
                            <span>Perceptron Binário</span>
                        </label>
                        <label class="radio-label">
                            <input type="radio" name="algo-pd" value="delta">
                            <span>Regra Delta Binário</span>
                        </label>
                        <label class="radio-label">
                            <input type="radio" name="algo-pd" value="delta_ova">
                            <span>Regra Delta OVA (Multiclasse)</span>
                        </label>
                        <label class="radio-label">
                            <input type="radio" name="algo-pd" value="xor">
                            <span>Problema do XOR</span>
                        </label>
                    </div>
                </div>
                
                <div class="card mt-16" id="card-pair-container">
                    <div class="card-header">
                        <h4>Par de Classes (Binário)</h4>
                    </div>
                    <div class="form-group" style="padding: 10px 0;">
                        <label class="radio-label">
                            <input type="radio" name="pair-pd" value="setosa,versicolor" checked>
                            <span>Setosa × Versicolor</span>
                        </label>
                        <label class="radio-label">
                            <input type="radio" name="pair-pd" value="versicolor,virginica">
                            <span>Versicolor × Virginica</span>
                        </label>
                        <label class="radio-label">
                            <input type="radio" name="pair-pd" value="setosa,virginica">
                            <span>Setosa × Virginica</span>
                        </label>
                    </div>
                </div>
                
                <div class="card mt-16" id="card-attr-container">
                    <div class="card-header">
                        <h4>Atributos</h4>
                    </div>
                    <div class="form-group" style="padding: 10px 0;">
                        <label class="radio-label">
                            <input type="radio" name="attr-pd" value="petalas" checked>
                            <span>Pétalas (índices [2, 3])</span>
                        </label>
                        <label class="radio-label">
                            <input type="radio" name="attr-pd" value="sepalas">
                            <span>Sépalas (índices [0, 1])</span>
                        </label>
                        <label class="radio-label" id="radio-todas-container">
                            <input type="radio" name="attr-pd" value="todas">
                            <span>Todas (4 Features)</span>
                        </label>
                    </div>
                </div>
                
                <div class="card mt-16">
                    <div class="card-header">
                        <h4>Hiperparâmetros</h4>
                    </div>
                    <div class="form-grid mt-10">
                        <div id="lr-container"></div>
                        <div id="epochs-container"></div>
                        <div id="split-container"></div>
                        <div id="seed-container"></div>
                    </div>
                </div>
                
                <button class="btn btn-primary w-100 mt-16" id="btn-train-pd">
                    ⚡ Treinar Modelo
                </button>
            </div>
            
            <!-- Coluna 2: Gráfico de Convergência -->
            <div class="col-chart">
                <div class="card h-100 flex-column">
                    <div class="card-header flex-row justify-between align-center">
                        <h4>Histórico de Convergência</h4>
                        <span class="badge badge-info" id="convergence-tag">Erros</span>
                    </div>
                    <div id="plot-pd-container" class="flex-grow flex-center" style="min-height: 400px;">
                        <!-- Gráfico injetado pelo JS -->
                    </div>
                </div>
            </div>
            
            <!-- Coluna 3: Resultados e Predição -->
            <div class="col-metrics">
                <div class="card">
                    <div class="card-header">
                        <h4>Status do Treinamento</h4>
                    </div>
                    <div id="train-status-panel" class="mt-10">
                        <p class="text-secondary">Modelo não treinado. Ajuste os hiperparâmetros e clique em Treinar.</p>
                    </div>
                </div>
                
                <div class="card mt-16" id="card-weights-container" style="display: none;">
                    <div class="card-header">
                        <h4>Pesos do Modelo</h4>
                    </div>
                    <div id="weights-panel" class="mt-10">
                        <!-- Pesos renderizados -->
                    </div>
                </div>
                
                <div class="card mt-16" id="card-manual-classify-container" style="display: none;">
                    <div class="card-header">
                        <h4>Predizer Amostra</h4>
                    </div>
                    <form id="form-classify-pd" class="mt-10">
                        <div id="features-inputs-grid" style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                            <!-- Gerado dinamicamente dependendo dos atributos ativos -->
                        </div>
                        <button type="submit" class="btn btn-secondary w-100 mt-10">Predizer</button>
                    </form>
                    
                    <div id="classify-result-pd" class="mt-10" style="display: none;">
                        <!-- Resultado -->
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Injeta hiperparâmetros iniciais
    injetarHiperparametros();
    
    // Event listeners
    container.querySelectorAll('input[name="algo-pd"]').forEach(el => {
        el.addEventListener('change', (e) => {
            currentAlgo = e.target.value;
            atualizarVisibilidadeControles();
            injetarInputsPredicao();
        });
    });
    
    container.querySelectorAll('input[name="pair-pd"]').forEach(el => {
        el.addEventListener('change', (e) => {
            currentPair = e.target.value;
        });
    });
    
    container.querySelectorAll('input[name="attr-pd"]').forEach(el => {
        el.addEventListener('change', (e) => {
            currentAttr = e.target.value;
            injetarInputsPredicao();
        });
    });
    
    container.querySelector('#btn-train-pd').addEventListener('click', executarTreinamento);
    container.querySelector('#form-classify-pd').addEventListener('submit', executarPredicaoManual);
    
    atualizarVisibilidadeControles();
    injetarInputsPredicao();
    charts.renderPlotError(container.querySelector('#plot-pd-container'), 'Clique em "Treinar Modelo" para visualizar a convergência.');
}

/**
 * Controla quais opções de interface aparecem de acordo com o algoritmo selecionado.
 */
function atualizarVisibilidadeControles() {
    const cardPair = activeContainer.querySelector('#card-pair-container');
    const cardAttr = activeContainer.querySelector('#card-attr-container');
    const radioTodas = activeContainer.querySelector('#radio-todas-container');
    
    if (currentAlgo === 'xor') {
        if (cardPair) cardPair.style.display = 'none';
        if (cardAttr) cardAttr.style.display = 'none';
    } else if (currentAlgo === 'delta_ova') {
        if (cardPair) cardPair.style.display = 'none';
        if (cardAttr) cardAttr.style.display = 'block';
        if (radioTodas) radioTodas.style.display = 'block';
    } else {
        // perceptron ou delta binário
        if (cardPair) cardPair.style.display = 'block';
        if (cardAttr) cardAttr.style.display = 'block';
        // Perceptron binário só suporta 2 features por simplificação da interface no projeto original
        if (currentAlgo === 'perceptron') {
            if (radioTodas) radioTodas.style.display = 'none';
            // Se o atributo ativo era 'todas', volta para 'petalas'
            if (currentAttr === 'todas') {
                activeContainer.querySelector('input[name="attr-pd"][value="petalas"]').checked = true;
                currentAttr = 'petalas';
            }
        } else {
            if (radioTodas) radioTodas.style.display = 'block';
        }
    }
}

/**
 * Renderiza os inputs de parâmetros de treinamento.
 */
function injetarHiperparametros() {
    const lrBox = activeContainer.querySelector('#lr-container');
    const epBox = activeContainer.querySelector('#epochs-container');
    const spBox = activeContainer.querySelector('#split-container');
    const sdBox = activeContainer.querySelector('#seed-container');
    
    lrBox.innerHTML = '';
    epBox.innerHTML = '';
    spBox.innerHTML = '';
    sdBox.innerHTML = '';
    
    lrBox.appendChild(ui.createInputGroup('Neta (&eta;)', 'number', 'input-lr', { step: 0.001, value: 0.03, min: 0.001 }));
    epBox.appendChild(ui.createInputGroup('Épocas Máx', 'number', 'input-epochs', { step: 1, value: 100, min: 1 }));
    spBox.appendChild(ui.createInputGroup('Treino %', 'number', 'input-split', { step: 0.05, value: 0.70, min: 0.1, max: 0.9 }));
    sdBox.appendChild(ui.createInputGroup('Semente', 'number', 'input-seed', { step: 1, value: 42 }));
}

/**
 * Cria os campos numéricos na predição manual baseado no número de features selecionadas.
 */
function injetarInputsPredicao() {
    const box = activeContainer.querySelector('#features-inputs-grid');
    if (!box) return;
    box.innerHTML = '';
    
    if (currentAlgo === 'xor') {
        box.appendChild(ui.createInputGroup('x1', 'number', 'pred-val-0', { step: 1, value: 1 }));
        box.appendChild(ui.createInputGroup('x2', 'number', 'pred-val-1', { step: 1, value: 0 }));
        return;
    }
    
    if (currentAttr === 'petalas') {
        box.appendChild(ui.createInputGroup('C. Pétala (x₁)', 'number', 'pred-val-0', { step: 0.1, value: 4.5 }));
        box.appendChild(ui.createInputGroup('L. Pétala (x₂)', 'number', 'pred-val-1', { step: 0.1, value: 1.5 }));
    } else if (currentAttr === 'sepalas') {
        box.appendChild(ui.createInputGroup('C. Sépala (x₁)', 'number', 'pred-val-0', { step: 0.1, value: 5.8 }));
        box.appendChild(ui.createInputGroup('L. Sépala (x₂)', 'number', 'pred-val-1', { step: 0.1, value: 2.7 }));
    } else {
        // Todas
        box.style.gridTemplateColumns = '1fr 1fr';
        box.appendChild(ui.createInputGroup('C. Sépala (x₁)', 'number', 'pred-val-0', { step: 0.1, value: 5.8 }));
        box.appendChild(ui.createInputGroup('L. Sépala (x₂)', 'number', 'pred-val-1', { step: 0.1, value: 2.7 }));
        box.appendChild(ui.createInputGroup('C. Pétala (x₃)', 'number', 'pred-val-2', { step: 0.1, value: 4.5 }));
        box.appendChild(ui.createInputGroup('L. Pétala (x₄)', 'number', 'pred-val-3', { step: 0.1, value: 1.5 }));
    }
}

/**
 * Executa o fluxo de treinamento via chamada à API Flask.
 */
async function executarTreinamento() {
    const statusPanel = activeContainer.querySelector('#train-status-panel');
    const plotContainer = activeContainer.querySelector('#plot-pd-container');
    const cardWeights = activeContainer.querySelector('#card-weights-container');
    const cardClassify = activeContainer.querySelector('#card-manual-classify-container');
    const labelConvergence = activeContainer.querySelector('#convergence-tag');
    
    if (!statusPanel || !plotContainer) return;
    
    // Esconde caixas antigas
    cardWeights.style.display = 'none';
    cardClassify.style.display = 'none';
    
    statusPanel.innerHTML = '';
    statusPanel.appendChild(ui.createLoader());
    charts.showPlotLoading(plotContainer);
    
    const lr = parseFloat(activeContainer.querySelector('#input-lr').value);
    const epochs = parseInt(activeContainer.querySelector('#input-epochs').value);
    const split = parseFloat(activeContainer.querySelector('#input-split').value);
    const seed = parseInt(activeContainer.querySelector('#input-seed').value);
    
    try {
        let indicesParam = [2, 3];
        if (currentAttr === 'sepalas') indicesParam = [0, 1];
        if (currentAttr === 'todas') indicesParam = [0, 1, 2, 3];
        
        let res;
        let historico = [];
        let metricaNome = 'Erros';
        let modeloTitulo = '';
        
        if (currentAlgo === 'perceptron') {
            const [cp, cn] = currentPair.split(',');
            classPos = cp;
            classNeg = cn;
            metricaNome = 'Erros de Classificação';
            modeloTitulo = 'Perceptron';
            
            res = await api.treinarPerceptron({
                indices_atributos: indicesParam,
                taxa_aprendizado: lr,
                max_epocas: epochs,
                classe_pos: cp,
                classe_neg: cn,
                proporcao_treino: split,
                semente: seed
            });
            
            trainedWeights = res.pesos;
            trainedWeightsOva = null;
            historico = res.historico_erros;

            statusPanel.innerHTML = '';
            const gridPerc = document.createElement('div');
            gridPerc.className = 'metrics-grid';
            gridPerc.appendChild(ui.createMetricCard('Acurácia Teste', (res.acuracia_teste * 100).toFixed(2) + '%', { variant: 'success' }));
            gridPerc.appendChild(ui.createMetricCard('Épocas Reais', res.epocas_reais));
            statusPanel.appendChild(gridPerc);
            
            // Renderiza os pesos
            const wPanel = activeContainer.querySelector('#weights-panel');
            wPanel.innerHTML = `
                <p class="text-secondary" style="font-size: 13px; margin-bottom: 8px;">Vetor de pesos ajustado (incluindo Bias w₀):</p>
                <code style="display: block; background: var(--bg-input); padding: 8px; border-radius: var(--radius-sm); font-family: monospace; font-size: 11px; word-break: break-all; border: 1px solid var(--border-subtle);">
                    [${res.pesos.map(w => w.toFixed(5)).join(', ')}]
                </code>
            `;
            cardWeights.style.display = 'block';
            cardClassify.style.display = 'block';
            
        } else if (currentAlgo === 'delta') {
            const [cp, cn] = currentPair.split(',');
            classPos = cp;
            classNeg = cn;
            metricaNome = 'Erro Quadrático Médio (MSE)';
            modeloTitulo = 'Regra Delta';
            
            res = await api.treinarDelta({
                indices_atributos: indicesParam,
                taxa_aprendizado: lr,
                max_epocas: epochs,
                classe_pos: cp,
                classe_neg: cn,
                proporcao_treino: split,
                semente: seed
            });
            
            trainedWeights = res.pesos;
            trainedWeightsOva = null;
            historico = res.historico_mse;

            statusPanel.innerHTML = '';
            const gridDelta = document.createElement('div');
            gridDelta.className = 'metrics-grid';
            gridDelta.appendChild(ui.createMetricCard('Acurácia Teste', (res.acuracia_teste * 100).toFixed(2) + '%', { variant: 'success' }));
            gridDelta.appendChild(ui.createMetricCard('Épocas Reais', res.epocas_reais));
            statusPanel.appendChild(gridDelta);

            const wPanel = activeContainer.querySelector('#weights-panel');
            wPanel.innerHTML = `
                <p class="text-secondary" style="font-size: 13px; margin-bottom: 8px;">Vetor de pesos ajustado (w₀, w₁, ...):</p>
                <code style="display: block; background: var(--bg-input); padding: 8px; border-radius: var(--radius-sm); font-family: monospace; font-size: 11px; word-break: break-all; border: 1px solid var(--border-subtle);">
                    [${res.pesos.map(w => w.toFixed(5)).join(', ')}]
                </code>
            `;
            cardWeights.style.display = 'block';
            cardClassify.style.display = 'block';
            
        } else if (currentAlgo === 'delta_ova') {
            metricaNome = 'Erro Quadrático Médio (MSE)';
            modeloTitulo = 'Regra Delta OVA';
            
            res = await api.treinarDeltaOva({
                indices_atributos: indicesParam,
                taxa_aprendizado: lr,
                max_epocas: epochs,
                proporcao_treino: split,
                semente: seed
            });
            
            trainedWeights = null;
            trainedWeightsOva = res.pesos;
            // No OVA, nós temos históricos múltiplos (um por classe).
            // Usamos a média do MSE entre as 3 classes para exibir no gráfico de convergência.
            const len = Object.values(res.historicos)[0].length;
            historico = [];
            for (let i = 0; i < len; i++) {
                let soma = 0;
                let count = 0;
                for (const classHist of Object.values(res.historicos)) {
                    soma += classHist[i];
                    count++;
                }
                historico.push(soma / count);
            }
            
            statusPanel.innerHTML = '';
            const gridOva = document.createElement('div');
            gridOva.className = 'metrics-grid';
            gridOva.appendChild(ui.createMetricCard('Acurácia Teste', (res.acuracia_teste * 100).toFixed(2) + '%', { variant: 'success' }));
            gridOva.appendChild(ui.createMetricCard('Épocas Reais', res.epocas_reais));
            statusPanel.appendChild(gridOva);

            const wPanel = activeContainer.querySelector('#weights-panel');
            wPanel.innerHTML = Object.entries(res.pesos).map(([classe, wVec]) => `
                <div class="mt-6">
                    <span class="tag tag-${classe}" style="font-size: 10px; padding: 2px 4px;">${classe.toUpperCase()}</span>
                    <code style="display: block; background: var(--bg-input); padding: 6px; border-radius: var(--radius-sm); font-family: monospace; font-size: 10px; margin-top: 2px; word-break: break-all; border: 1px solid var(--border-subtle);">
                        [${wVec.map(w => w.toFixed(4)).join(', ')}]
                    </code>
                </div>
            `).join('');
            
            cardWeights.style.display = 'block';
            cardClassify.style.display = 'block';
            
        } else if (currentAlgo === 'xor') {
            metricaNome = 'Erro Quadrático Médio (MSE)';
            modeloTitulo = 'XOR (Delta Rule)';
            
            // Chamar treinamento do XOR
            const resXor = await api.treinarDelta({
                xor: true,
                taxa_aprendizado: lr,
                max_epocas: epochs
            });
            
            trainedWeights = resXor.pesos;
            trainedWeightsOva = null;
            historico = resXor.historico_mse;
            
            statusPanel.innerHTML = '';
            const gridXor = document.createElement('div');
            gridXor.className = 'metrics-grid';
            gridXor.appendChild(ui.createMetricCard('MSE Final', resXor.mse_final.toFixed(6), { variant: 'success' }));
            gridXor.appendChild(ui.createMetricCard('Épocas', resXor.epocas_reais));
            statusPanel.appendChild(gridXor);
            
            const wPanel = activeContainer.querySelector('#weights-panel');
            wPanel.innerHTML = `
                <p class="text-secondary" style="font-size: 13px; margin-bottom: 8px;">Pesos (w₀ bias, w₁ x₁, w₂ x₂):</p>
                <code style="display: block; background: var(--bg-input); padding: 8px; border-radius: var(--radius-sm); font-family: monospace; font-size: 11px; border: 1px solid var(--border-subtle);">
                    [${resXor.pesos.map(w => w.toFixed(5)).join(', ')}]
                </code>
            `;
            cardWeights.style.display = 'block';
            cardClassify.style.display = 'block';
        }
        
        labelConvergence.innerText = metricaNome.includes('MSE') ? 'MSE' : 'Erros';
        
        // Solicita o gráfico de convergência estilizado da API
        const plotRes = await api.getPlotConvergence(historico, metricaNome, modeloTitulo);
        charts.renderPlot(plotContainer, plotRes.image, 'Histórico de convergência');
        
        ui.showToast('Treinamento finalizado com sucesso!', 'success');
        
    } catch (e) {
        statusPanel.innerHTML = `<span class="text-danger">Erro no treino: ${e.message}</span>`;
        charts.renderPlotError(plotContainer, `Erro ao carregar convergência: ${e.message}`);
    }
}

/**
 * Prediz classe para a amostra inserida manualmente.
 */
async function executarPredicaoManual(e) {
    e.preventDefault();
    const resultPanel = activeContainer.querySelector('#classify-result-pd');
    if (!resultPanel) return;
    
    resultPanel.style.display = 'block';
    resultPanel.innerHTML = '';
    resultPanel.appendChild(ui.createSkeleton(1));
    
    try {
        const val0 = parseFloat(activeContainer.querySelector('#pred-val-0').value);
        const val1 = parseFloat(activeContainer.querySelector('#pred-val-1').value);
        
        let inputs = [val0, val1];
        if (currentAlgo !== 'xor' && currentAttr === 'todas') {
            const val2 = parseFloat(activeContainer.querySelector('#pred-val-2').value);
            const val3 = parseFloat(activeContainer.querySelector('#pred-val-3').value);
            inputs = [val0, val1, val2, val3];
        }
        
        let classeResult = '';
        let tagVariant = 'default';
        let detailHTML = '';
        
        if (currentAlgo === 'xor') {
            const res = await api.predizerDelta(inputs, trainedWeights, '1', '0');
            classeResult = res.classe === '1' ? '1 (Verdadeiro)' : '0 (Falso)';
            tagVariant = res.classe === '1' ? 'versicolor' : 'virginica';
            detailHTML = `Saída contínua do Adaline: <code style="font-family: monospace;">${res.saida_continua.toFixed(4)}</code>`;

        } else if (currentAlgo === 'perceptron') {
            const res = await api.predizerPerceptron(inputs, trainedWeights);
            const classeFinal = res.classe === 1 ? classPos : classNeg;
            classeResult = classeFinal.toUpperCase();
            tagVariant = classeFinal;

        } else if (currentAlgo === 'delta') {
            const res = await api.predizerDelta(inputs, trainedWeights, classPos, classNeg);
            classeResult = res.classe.toUpperCase();
            tagVariant = res.classe;
            detailHTML = `Saída rede linear: <code style="font-family: monospace;">${res.saida_continua.toFixed(4)}</code>`;

        } else if (currentAlgo === 'delta_ova') {
            const res = await api.predizerDeltaOva(inputs, trainedWeightsOva);
            classeResult = res.classe.toUpperCase();
            tagVariant = res.classe;
            
            detailHTML = `
                <div class="mt-6">
                    <span class="text-secondary" style="font-size: 11px;">Ativações por classe (Net):</span>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; margin-top: 4px;">
                        ${Object.entries(res.ativacoes).map(([c, val]) => `
                            <div style="background: var(--bg-input); padding: 4px; border-radius: var(--radius-sm); text-align: center; border: 1px solid var(--border-subtle);">
                                <span class="tag tag-${c}" style="font-size: 9px; padding: 1px 3px;">${c.slice(0,3)}</span>
                                <span style="font-family: monospace; font-size: 10px; display: block; margin-top: 2px;">${val.toFixed(4)}</span>
                            </div>
                        `).join('')}
                    </div>
                </div>
            `;
        }
        
        resultPanel.innerHTML = `
            <div style="padding: 10px; background: rgba(255,255,255,0.02); border: 1px solid var(--border-medium); border-radius: var(--radius-md);">
                <div class="flex-row align-center gap-10">
                    <span class="text-secondary">Predição:</span>
                    <span class="tag tag-${tagVariant}" style="font-size: 13px; font-weight: 700;">
                        ${classeResult}
                    </span>
                </div>
                ${detailHTML ? `<div class="mt-6" style="font-size: 12px; color: var(--text-secondary);">${detailHTML}</div>` : ''}
            </div>
        `;
        
    } catch (e) {
        resultPanel.innerHTML = `<span class="text-danger">Erro na predição: ${e.message}</span>`;
    }
}

/**
 * Destrói a aba.
 */
export function destroy() {
    activeContainer = null;
    trainedWeights = null;
    trainedBias = null;
    trainedWeightsOva = null;
}
