/**
 * Aba 3 — Métricas Avançadas
 */

import * as api from '../api.js';
import * as ui from '../components.js';
import * as charts from '../charts.js';

let activeContainer = null;
let currentDataset = 'v1';
let currentAttributes = 'petalas';
let currentComparison = 'todas'; // 'todas', 'setosa_versicolor', etc.

let resultadosModelos = null; // Guardará o retorno de /api/metricas/train-all
let classesAtivas = [];

// Sub-aba interna ativa: 'comparativo', 'detalhe', 'matriz', 'comparacao_kt'
let activeSubTab = 'comparativo'; 

// Modelos atualmente selecionados no detalhe e no teste Z
let selectedModelDetail = '';
let selectedModelZ1 = '';
let selectedModelZ2 = '';

/**
 * Inicializa a aba.
 */
export async function init(container) {
    activeContainer = container;
    
    container.innerHTML = `
        <div class="tab-grid">
            <!-- Coluna 1: Controles de Treinamento -->
            <div class="col-controls">
                <div class="card">
                    <div class="card-header">
                        <h4>Dataset</h4>
                    </div>
                    <div class="form-group" style="padding: 10px 0;">
                        <label class="radio-label">
                            <input type="radio" name="dataset-ma" value="v1" checked>
                            <span>Base Iris Original (v1)</span>
                        </label>
                        <label class="radio-label">
                            <input type="radio" name="dataset-ma" value="v2">
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
                            <input type="radio" name="attr-ma" value="petalas" checked>
                            <span>Pétalas (índices [2, 3])</span>
                        </label>
                        <label class="radio-label">
                            <input type="radio" name="attr-ma" value="sepalas">
                            <span>Sépalas (índices [0, 1])</span>
                        </label>
                        <label class="radio-label">
                            <input type="radio" name="attr-ma" value="todas">
                            <span>Todas (4 Features)</span>
                        </label>
                    </div>
                </div>
                
                <div class="card mt-16">
                    <div class="card-header">
                        <h4>Comparação (Classes)</h4>
                    </div>
                    <div class="form-group" style="padding: 10px 0;">
                        <label class="radio-label">
                            <input type="radio" name="comp-ma" value="todas" checked>
                            <span>3 Classes (Todas)</span>
                        </label>
                        <label class="radio-label">
                            <input type="radio" name="comp-ma" value="setosa_versicolor">
                            <span>Setosa × Versicolor</span>
                        </label>
                        <label class="radio-label">
                            <input type="radio" name="comp-ma" value="versicolor_virginica">
                            <span>Versicolor × Virginica</span>
                        </label>
                        <label class="radio-label">
                            <input type="radio" name="comp-ma" value="setosa_virginica">
                            <span>Setosa × Virginica</span>
                        </label>
                    </div>
                </div>
                
                <div class="card mt-16">
                    <div class="card-header">
                        <h4>Hiperparâmetros</h4>
                    </div>
                    <div class="form-grid mt-10">
                        <div id="split-ma-container"></div>
                        <div id="seed-ma-container"></div>
                    </div>
                </div>
                
                <button class="btn btn-primary w-100 mt-16" id="btn-train-ma">
                    ⚡ Treinar e Calcular Métricas
                </button>
            </div>
            
            <!-- Coluna 2: Sub-abas internas de Resultados (Ampla) -->
            <div class="col-chart" style="grid-column: span 2;">
                <div class="card h-100 flex-column" id="results-card" style="min-height: 500px;">
                    <!-- Seletor de Sub-abas Internas (Menu Horizontal) -->
                    <div class="subtab-bar" style="display: flex; gap: 10px; border-bottom: 1px solid var(--border-medium); padding-bottom: 10px;">
                        <button class="btn btn-ghost subtab-btn active" data-subtab="comparativo">Comparativo Geral</button>
                        <button class="btn btn-ghost subtab-btn" data-subtab="detalhe" id="btn-subtab-detalhe" style="display:none;">Detalhe por Classe</button>
                        <button class="btn btn-ghost subtab-btn" data-subtab="matriz" id="btn-subtab-matriz" style="display:none;">Matriz de Confusão</button>
                        <button class="btn btn-ghost subtab-btn" data-subtab="comparacao_kt" id="btn-subtab-kt" style="display:none;">Comparação K & T (Z-test)</button>
                    </div>
                    
                    <div id="subtab-content" class="flex-grow mt-16">
                        <!-- Injetado por JS -->
                    </div>
                </div>
            </div>
        </div>
    `;
    
    // Injeta inputs
    injetarHiperparametros();
    
    // Bind eventos
    container.querySelectorAll('input[name="dataset-ma"]').forEach(el => {
        el.addEventListener('change', (e) => currentDataset = e.target.value);
    });
    container.querySelectorAll('input[name="attr-ma"]').forEach(el => {
        el.addEventListener('change', (e) => currentAttributes = e.target.value);
    });
    container.querySelectorAll('input[name="comp-ma"]').forEach(el => {
        el.addEventListener('change', (e) => currentComparison = e.target.value);
    });
    
    container.querySelector('#btn-train-ma').addEventListener('click', executarTreinamentoAvancado);
    
    // Eventos de troca de sub-aba
    container.querySelectorAll('.subtab-btn').forEach(btn => {
        btn.addEventListener('click', (e) => {
            container.querySelectorAll('.subtab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            activeSubTab = btn.dataset.subtab;
            renderizarSubAbaAtiva();
        });
    });
    
    // Renderização do estado vazio inicial
    renderizarEmptyState();
}

function injetarHiperparametros() {
    const spBox = activeContainer.querySelector('#split-ma-container');
    const sdBox = activeContainer.querySelector('#seed-ma-container');
    spBox.innerHTML = '';
    sdBox.innerHTML = '';
    spBox.appendChild(ui.createInputGroup('Treino %', 'number', 'ma-split', { step: 0.05, value: 0.70, min: 0.1, max: 0.9 }));
    sdBox.appendChild(ui.createInputGroup('Semente', 'number', 'ma-seed', { step: 1, value: 42 }));
}

function renderizarEmptyState() {
    const box = activeContainer.querySelector('#subtab-content');
    if (box) {
        box.innerHTML = '';
        box.appendChild(ui.createEmptyState('Clique em "Treinar e Calcular Métricas" para processar todos os 6 modelos de classificação.', '⚡'));
    }
}

/**
 * Dispara o treinamento simultâneo dos 6 modelos na API.
 */
async function executarTreinamentoAvancado() {
    const box = activeContainer.querySelector('#subtab-content');
    if (!box) return;
    
    box.innerHTML = '';
    box.appendChild(ui.createLoader());
    
    const split = parseFloat(activeContainer.querySelector('#ma-split').value);
    const seed = parseInt(activeContainer.querySelector('#ma-seed').value);
    
    try {
        const res = await api.treinarTodosModelos({
            dataset: currentDataset,
            atributos: currentAttributes,
            proporcao_treino: split,
            semente: seed,
            comparacao: currentComparison
        });
        
        resultadosModelos = res.resultados;
        classesAtivas = res.classes;
        
        // Ativa botões das outras sub-abas agora que temos dados
        activeContainer.querySelector('#btn-subtab-detalhe').style.display = 'block';
        activeContainer.querySelector('#btn-subtab-matriz').style.display = 'block';
        activeContainer.querySelector('#btn-subtab-kt').style.display = 'block';
        
        // Define seletores padrão
        const modelosNomes = Object.keys(resultadosModelos);
        selectedModelDetail = modelosNomes[0];
        selectedModelZ1 = modelosNomes.find(n => n.includes('Perceptron')) || modelosNomes[0];
        selectedModelZ2 = modelosNomes.find(n => n.includes('Delta') && !n.includes('Bin.')) || modelosNomes[1] || modelosNomes[0];
        
        ui.showToast('Todos os 6 classificadores foram treinados com sucesso!', 'success');
        
        // Renderiza sub-aba ativa (geralmente comparativo)
        renderizarSubAbaAtiva();
        
    } catch (e) {
        box.innerHTML = `<div class="text-danger" style="padding: 20px;">Falha ao treinar classificadores: ${e.message}</div>`;
        ui.showToast(`Erro no treinamento: ${e.message}`, 'error');
    }
}

/**
 * Renderiza o painel da sub-aba interna ativa.
 */
function renderizarSubAbaAtiva() {
    const box = activeContainer.querySelector('#subtab-content');
    if (!box || !resultadosModelos) return;
    
    box.innerHTML = '';
    
    switch (activeSubTab) {
        case 'comparativo':
            renderizarComparativoGeral(box);
            break;
        case 'detalhe':
            renderizarDetalheClasse(box);
            break;
        case 'matriz':
            renderizarMatrizConfusao(box);
            break;
        case 'comparacao_kt':
            renderizarZTest(box);
            break;
    }
}

/**
 * Sub-aba 1: Tabela comparativa entre todos os classificadores.
 */
function renderizarComparativoGeral(container) {
    const cols = ['Classificador', 'Acerto Global', 'Coeficiente Kappa', 'Classificação Kappa', 'Coeficiente Tau'];
    const rows = Object.entries(resultadosModelos).map(([nome, report]) => {
        const kappaVal = report.kappa;
        
        // Interpretação qualitativa do Kappa
        let classif = 'Fraco';
        if (kappaVal > 0.81) classif = 'Quase Perfeito';
        else if (kappaVal > 0.61) classif = 'Substancial';
        else if (kappaVal > 0.41) classif = 'Moderado';
        else if (kappaVal > 0.21) classif = 'Razoável';
        
        return [
            nome,
            `${(report.acerto_global * 100).toFixed(2)}%`,
            kappaVal.toFixed(4),
            ui.createBadge(classif, kappaVal > 0.61 ? 'success' : 'warning'),
            report.tau.toFixed(4)
        ];
    });
    
    container.innerHTML = `
        <h4 style="color: var(--accent-purple); margin-bottom: 12px;">Comparativo Geral dos Modelos</h4>
        <p class="text-secondary" style="font-size: 13px; margin-bottom: 16px;">
            Métricas de validação global calculadas no conjunto de teste (30% dos dados).
        </p>
    `;
    container.appendChild(ui.createDataTable(cols, rows));
}

/**
 * Sub-aba 2: Métricas por classe para um modelo selecionado.
 */
function renderizarDetalheClasse(container) {
    const modelosNomes = Object.keys(resultadosModelos);
    
    const divHeader = document.createElement('div');
    divHeader.className = 'flex-row align-center justify-between mt-10 mb-16';
    divHeader.innerHTML = `
        <div>
            <h4 style="color: var(--accent-purple);">Métricas Detalhadas por Classe</h4>
            <p class="text-secondary" style="font-size: 13px;">Selecione um modelo para visualizar precisão, revocação e pontuações F-score.</p>
        </div>
    `;
    
    // Dropdown de seleção de modelo
    const selectGroup = ui.createSelectGroup(
        'Classificador:', 
        modelosNomes.map(n => ({ value: n, label: n })), 
        'select-model-detail', 
        { value: selectedModelDetail }
    );
    selectGroup.style.width = '240px';
    selectGroup.querySelector('select').addEventListener('change', (e) => {
        selectedModelDetail = e.target.value;
        renderizarSubAbaAtiva();
    });
    
    divHeader.appendChild(selectGroup);
    container.appendChild(divHeader);
    
    const report = resultadosModelos[selectedModelDetail];
    if (!report || !report.por_classe) return;
    
    const cols = ['Classe', 'Acurácia Produtor', 'Acurácia Usuário', 'Sensibilidade', 'Especificidade', 'Precisão', 'F1-Score', 'F2-Score', 'MCC'];
    
    const rows = Object.entries(report.por_classe).map(([classe, m]) => {
        const tag = ui.createSpeciesTag(classe);
        return [
            tag,
            `${(m.acuracia_produtor * 100).toFixed(1)}%`,
            `${(m.acuracia_usuario * 100).toFixed(1)}%`,
            m.sensibilidade.toFixed(4),
            m.especificidade.toFixed(4),
            m.precisao.toFixed(4),
            m.f1.toFixed(4),
            m.f2.toFixed(4),
            m.mcc.toFixed(4)
        ];
    });
    
    container.appendChild(ui.createDataTable(cols, rows));
}

/**
 * Sub-aba 3: Matriz de Confusão em Heatmap.
 */
function renderizarMatrizConfusao(container) {
    const modelosNomes = Object.keys(resultadosModelos);
    
    const divHeader = document.createElement('div');
    divHeader.className = 'flex-row align-center justify-between mt-10 mb-16';
    divHeader.innerHTML = `
        <div>
            <h4 style="color: var(--accent-purple);">Matriz de Confusão</h4>
            <p class="text-secondary" style="font-size: 13px;">Distribuição de acertos e falsos positivos gerados pelo visualizador gráfico.</p>
        </div>
    `;
    
    const selectGroup = ui.createSelectGroup(
        'Classificador:', 
        modelosNomes.map(n => ({ value: n, label: n })), 
        'select-model-confusion', 
        { value: selectedModelDetail }
    );
    selectGroup.style.width = '240px';
    selectGroup.querySelector('select').addEventListener('change', (e) => {
        selectedModelDetail = e.target.value;
        carregarPlotConfusao(selectedModelDetail);
    });
    
    divHeader.appendChild(selectGroup);
    container.appendChild(divHeader);
    
    // Espaço para o gráfico
    const plotBox = charts.createPlotContainer('plot-confusion-ma-container');
    container.appendChild(plotBox);
    
    // Carrega o gráfico
    carregarPlotConfusao(selectedModelDetail);
}

async function carregarPlotConfusao(modelo) {
    const plotBox = activeContainer.querySelector('#plot-confusion-ma-container');
    if (!plotBox) return;
    
    charts.showPlotLoading(plotBox);
    try {
        const res = await api.getPlotConfusion(modelo);
        charts.renderPlot(plotBox, res.image, `Matriz de confusão ${modelo}`);
    } catch (e) {
        charts.renderPlotError(plotBox, `Erro ao carregar matriz de confusão: ${e.message}`);
    }
}

/**
 * Sub-aba 4: Teste Z de significância de diferença de Kappa e Tau (Z-test).
 */
async function renderizarZTest(container) {
    const modelosNomes = Object.keys(resultadosModelos);
    
    container.innerHTML = `
        <h4 style="color: var(--accent-purple); margin-bottom: 4px;">Comparação de Significância (Teste Z)</h4>
        <p class="text-secondary" style="font-size: 13px; margin-bottom: 16px;">
            Verifica se a diferença de Kappa e Tau entre dois modelos é estatisticamente significativa ao nível de significância de 5% (p-valor < 0.05).
        </p>
        <div class="flex-row gap-16 align-center mt-10 mb-16" id="ztest-selectors"></div>
        <div id="ztest-results-panel"></div>
    `;
    
    const selectorsBox = container.querySelector('#ztest-selectors');
    
    const selA = ui.createSelectGroup('Classificador A:', modelosNomes.map(n => ({ value: n, label: n })), 'z-sel-a', { value: selectedModelZ1 });
    const selB = ui.createSelectGroup('Classificador B:', modelosNomes.map(n => ({ value: n, label: n })), 'z-sel-b', { value: selectedModelZ2 });
    
    selectorsBox.appendChild(selA);
    selectorsBox.appendChild(selB);
    
    const triggerTest = async () => {
        selectedModelZ1 = activeContainer.querySelector('#z-sel-a').value;
        selectedModelZ2 = activeContainer.querySelector('#z-sel-b').value;
        await calcularZTest();
    };
    
    selA.querySelector('select').addEventListener('change', triggerTest);
    selB.querySelector('select').addEventListener('change', triggerTest);
    
    await calcularZTest();
}

async function calcularZTest() {
    const resultsPanel = activeContainer.querySelector('#ztest-results-panel');
    if (!resultsPanel) return;
    
    resultsPanel.innerHTML = '';
    resultsPanel.appendChild(ui.createSkeleton(3));
    
    try {
        const res = await api.zTestComparacao(selectedModelZ1, selectedModelZ2);
        
        const cols = ['Métrica', selectedModelZ1, selectedModelZ2, 'Z Calculado', 'p-valor', 'Conclusão (Alfa=5%)'];
        const rows = [
            [
                'Coeficiente Kappa',
                res.kappa.val_a.toFixed(4),
                res.kappa.val_b.toFixed(4),
                res.kappa.z_calculado.toFixed(4),
                res.kappa.p_valor.toFixed(6),
                res.kappa.significativo ? ui.createBadge('Diferença Significativa', 'success') : ui.createBadge('Sem Diferença', 'default')
            ],
            [
                'Coeficiente Tau',
                res.tau.val_a.toFixed(4),
                res.tau.val_b.toFixed(4),
                res.tau.z_calculado.toFixed(4),
                res.tau.p_valor.toFixed(6),
                res.tau.significativo ? ui.createBadge('Diferença Significativa', 'success') : ui.createBadge('Sem Diferença', 'default')
            ]
        ];
        
        resultsPanel.innerHTML = `
            <div style="background: var(--bg-input); padding: 12px; border-radius: var(--radius-md); border: 1px solid var(--border-subtle); margin-bottom: 16px; font-size: 13px;">
                <span style="color: var(--accent-amber); font-weight: 600; display: block; margin-bottom: 6px;">💡 Hipóteses do Teste de Hipótese:</span>
                <span style="display: block; color: var(--text-secondary);">• <strong>H₀ (Hipótese Nula)</strong>: Os dois classificadores possuem desempenhos equivalentes no conjunto de teste.</span>
                <span style="display: block; color: var(--text-secondary);">• <strong>H₁ (Hipótese Alternativa)</strong>: Há uma diferença estatisticamente significativa nos acertos dos classificadores.</span>
            </div>
        `;
        resultsPanel.appendChild(ui.createDataTable(cols, rows));
        
    } catch (e) {
        resultsPanel.innerHTML = `<span class="text-danger">Erro ao calcular Teste Z: ${e.message}</span>`;
    }
}

/**
 * Destrói a aba.
 */
export function destroy() {
    activeContainer = null;
    resultadosModelos = null;
}
