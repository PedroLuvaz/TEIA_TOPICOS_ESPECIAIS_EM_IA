/**
 * Utilitários para renderização de gráficos Matplotlib via API.
 */

import { createSkeleton } from './components.js';

/**
 * Cria o contêiner básico para um gráfico.
 */
export function createPlotContainer(id, titulo = '') {
    const container = document.createElement('div');
    container.className = 'plot-container';
    container.id = id;
    
    if (titulo) {
        const titleEl = document.createElement('div');
        titleEl.className = 'plot-title';
        titleEl.innerText = titulo;
        container.appendChild(titleEl);
    }
    
    const wrapper = document.createElement('div');
    wrapper.className = 'plot-wrapper';
    
    // Skeleton inicial
    wrapper.appendChild(createSkeleton(5));
    
    container.appendChild(wrapper);
    return container;
}

/**
 * Exibe o estado de carregamento no contêiner de gráfico.
 */
export function showPlotLoading(container) {
    if (!container) return;
    const wrapper = container.querySelector('.plot-wrapper') || container;
    wrapper.innerHTML = '';
    wrapper.appendChild(createSkeleton(5));
}

/**
 * Renderiza a imagem em base64 recebida da API no contêiner.
 */
export function renderPlot(container, base64Image, alt = 'Gráfico estatístico') {
    if (!container) return;
    const wrapper = container.querySelector('.plot-wrapper') || container;
    wrapper.innerHTML = '';
    
    const img = document.createElement('img');
    img.src = base64Image;
    img.alt = alt;
    img.className = 'plot-image';
    
    // Suaviza a aparição da imagem
    img.style.opacity = '0';
    img.style.transition = 'opacity 300ms ease';
    
    img.onload = () => {
        img.style.opacity = '1';
    };
    
    wrapper.appendChild(img);
}

/**
 * Exibe estado de erro caso falhe ao buscar o gráfico.
 */
export function renderPlotError(container, mensagem) {
    if (!container) return;
    const wrapper = container.querySelector('.plot-wrapper') || container;
    wrapper.innerHTML = `
        <div class="plot-error">
            <span class="plot-error-icon">⚠️</span>
            <p class="plot-error-msg">${mensagem}</p>
        </div>
    `;
}
