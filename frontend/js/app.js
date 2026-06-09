/**
 * Controlador da SPA (Single Page Application).
 * Gerencia o sistema de abas e a navegação da aplicação.
 */

import { showToast } from './components.js';

// Dicionário de abas registradas
const abas = {};
let abaAtivaId = null;

/**
 * Registra uma aba no sistema.
 * @param {string} id - Identificador exclusivo da aba.
 * @param {string} label - Rótulo exibido no botão.
 * @param {function} initFn - Função de inicialização chamada ao abrir.
 * @param {function} destroyFn - Função de limpeza chamada ao fechar.
 * @param {string} icon - Ícone (emoji) opcional.
 */
export function registrarAba(id, label, initFn, destroyFn, icon = '') {
    abas[id] = { id, label, init: initFn, destroy: destroyFn, icon };
}

/**
 * Alterna para uma aba específica.
 */
export async function alternarAba(tabId) {
    if (tabId === abaAtivaId || !abas[tabId]) return;
    
    const container = document.getElementById('tab-container');
    const nav = document.getElementById('tab-nav');
    if (!container || !nav) return;
    
    // 1. Destrói a aba ativa anterior se existir
    if (abaAtivaId && abas[abaAtivaId].destroy) {
        try {
            abas[abaAtivaId].destroy();
        } catch (e) {
            console.error(`Erro ao destruir aba ${abaAtivaId}:`, e);
        }
    }
    
    // 2. Atualiza a classe ativa nos botões
    const botoes = nav.querySelectorAll('.tab-btn');
    botoes.forEach(btn => {
        if (btn.dataset.tab === tabId) {
            btn.classList.add('active');
            // Move o indicador físico se houver
            atualizarIndicador(btn);
        } else {
            btn.classList.remove('active');
        }
    });
    
    // 3. Efeito de fade out/in no container
    container.style.opacity = '0';
    container.style.transform = 'translateY(8px)';
    
    setTimeout(async () => {
        container.innerHTML = '';
        abaAtivaId = tabId;
        
        try {
            // Inicializa a nova aba
            await abas[tabId].init(container);
            container.style.opacity = '1';
            container.style.transform = 'translateY(0)';
        } catch (e) {
            console.error(`Erro ao inicializar aba ${tabId}:`, e);
            container.innerHTML = `
                <div class="card error-card" style="margin: 20px; border-color: var(--accent-red);">
                    <h3 style="color: var(--accent-red); margin-bottom: 10px;">Erro ao carregar aba</h3>
                    <p style="color: var(--text-secondary);">${e.message || e}</p>
                </div>
            `;
            container.style.opacity = '1';
            container.style.transform = 'translateY(0)';
            showToast(`Falha ao abrir aba: ${e.message}`, 'error');
        }
    }, 150);
}

/**
 * Atualiza o indicador deslizante sob a aba ativa.
 */
function atualizarIndicador(botaoAtivo) {
    let indicador = document.getElementById('tab-indicator');
    if (!indicador) {
        indicador = document.createElement('div');
        indicador.id = 'tab-indicator';
        botaoAtivo.parentElement.appendChild(indicador);
    }
    
    const rectBotao = botaoAtivo.getBoundingClientRect();
    const rectParent = botaoAtivo.parentElement.getBoundingClientRect();
    
    // Calcula posição relativa
    const left = rectBotao.left - rectParent.left + botaoAtivo.parentElement.scrollLeft;
    
    indicador.style.width = `${rectBotao.width}px`;
    indicador.style.left = `${left}px`;
}

/**
 * Constrói a barra de navegação no DOM.
 */
function construirNavegacao() {
    const nav = document.getElementById('tab-nav');
    if (!nav) return;
    
    nav.innerHTML = '';
    
    // Cria os botões
    Object.values(abas).forEach(aba => {
        const btn = document.createElement('button');
        btn.className = 'tab-btn';
        btn.dataset.tab = aba.id;
        
        const emoji = aba.icon ? `<span class="tab-btn-icon">${aba.icon}</span> ` : '';
        btn.innerHTML = `${emoji}${aba.label}`;
        
        btn.addEventListener('click', () => alternarAba(aba.id));
        nav.appendChild(btn);
    });
    
    // Cria o indicador físico
    const indicador = document.createElement('div');
    indicador.id = 'tab-indicator';
    nav.appendChild(indicador);
    
    // Reposiciona indicador caso o tamanho da tela mude
    window.addEventListener('resize', () => {
        const botaoAtivo = nav.querySelector('.tab-btn.active');
        if (botaoAtivo) atualizarIndicador(botaoAtivo);
    });
}

// ---------------------------------------------------------------------------
// Inicialização principal da aplicação
// ---------------------------------------------------------------------------
document.addEventListener('DOMContentLoaded', async () => {
    // Importa as abas de forma dinâmica para evitar dependência estática
    const distMinima = await import('./tabs/distancia-minima.js').catch(e => console.error(e));
    const perceptronDelta = await import('./tabs/perceptron-delta.js').catch(e => console.error(e));
    const metricasAvancadas = await import('./tabs/metricas-avancadas.js').catch(e => console.error(e));
    
    // Registra as 3 abas principais
    if (distMinima) {
        registrarAba('distancia-minima', 'Distância Mínima', distMinima.init, distMinima.destroy, '📏');
    }
    if (perceptronDelta) {
        registrarAba('perceptron-delta', 'Perceptron & Delta', perceptronDelta.init, perceptronDelta.destroy, '🧠');
    }
    if (metricasAvancadas) {
        registrarAba('metricas-avancadas', 'Métricas Avançadas', metricasAvancadas.init, metricasAvancadas.destroy, '📊');
    }
    
    // Constrói navegação
    construirNavegacao();
    
    // Ativa a primeira aba por padrão
    const abasDisponiveis = Object.keys(abas);
    if (abasDisponiveis.length > 0) {
        alternarAba(abasDisponiveis[0]);
    }
});
