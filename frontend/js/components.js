/**
 * Componentes visuais reutilizáveis para o frontend do Classificador Iris.
 * Abstrai a criação de elementos DOM com estilos consistentes.
 */

/**
 * Cria um Toast temporário para exibir notificações de erro, sucesso ou info.
 */
export function showToast(mensagem, tipo = 'info') {
    const container = document.getElementById('toast-container');
    if (!container) return;
    
    const toast = document.createElement('div');
    toast.className = `toast toast-${tipo}`;
    
    let icone = 'ℹ️';
    if (tipo === 'success') icone = '✅';
    if (tipo === 'error') icone = '❌';
    if (tipo === 'warning') icone = '⚠️';
    
    toast.innerHTML = `
        <span class="toast-icon">${icone}</span>
        <span class="toast-message">${mensagem}</span>
    `;
    
    container.appendChild(toast);
    
    // Remove após 4 segundos
    setTimeout(() => {
        toast.style.animation = 'toast-slide-out 300ms forwards';
        toast.addEventListener('animationend', () => {
            toast.remove();
        });
    }, 4000);
}

/**
 * Cria um Card de métrica (número gigante + rótulo).
 */
export function createMetricCard(label, value, options = {}) {
    const card = document.createElement('div');
    card.className = 'metric-card';
    if (options.variant) {
        card.classList.add(`metric-${options.variant}`);
    }
    
    card.innerHTML = `
        <span class="metric-label">${label}</span>
        <span class="metric-value">${value}</span>
    `;
    return card;
}

/**
 * Cria um container com efeito skeleton para simular carregamento.
 */
export function createSkeleton(linhas = 3) {
    const skeleton = document.createElement('div');
    skeleton.className = 'skeleton-container';
    for (let i = 0; i < linhas; i++) {
        const item = document.createElement('div');
        item.className = 'skeleton-line';
        if (i === 0) item.style.width = '40%';
        if (i === linhas - 1) item.style.width = '60%';
        skeleton.appendChild(item);
    }
    return skeleton;
}

/**
 * Cria um spinner de loading.
 */
export function createLoader() {
    const loader = document.createElement('div');
    loader.className = 'loader-container';
    loader.innerHTML = `
        <div class="loader"></div>
        <span class="loader-text">Processando dados...</span>
    `;
    return loader;
}

/**
 * Cria uma badge / etiqueta colorida.
 */
export function createBadge(texto, tipo = 'default') {
    const badge = document.createElement('span');
    badge.className = `badge badge-${tipo}`;
    badge.innerText = texto;
    return badge;
}

/**
 * Cria uma tag de espécie com cor dedicada.
 */
export function createSpeciesTag(species) {
    const tag = document.createElement('span');
    const cleanSpec = species.toLowerCase().trim();
    tag.className = `tag tag-${cleanSpec}`;
    tag.innerText = species.charAt(0).toUpperCase() + species.slice(1);
    return tag;
}

/**
 * Cria uma tabela de dados responsiva.
 */
export function createDataTable(colunas, linhas, options = {}) {
    const wrapper = document.createElement('div');
    wrapper.className = 'table-wrapper';
    
    const table = document.createElement('table');
    table.className = 'data-table';
    
    // Cabeçalho
    const thead = document.createElement('thead');
    const trHead = document.createElement('tr');
    colunas.forEach(col => {
        const th = document.createElement('th');
        th.innerText = col;
        trHead.appendChild(th);
    });
    thead.appendChild(trHead);
    table.appendChild(thead);
    
    // Corpo
    const tbody = document.createElement('tbody');
    if (linhas.length === 0) {
        const trEmpty = document.createElement('tr');
        const tdEmpty = document.createElement('td');
        tdEmpty.colSpan = colunas.length;
        tdEmpty.className = 'text-center text-muted';
        tdEmpty.innerText = options.emptyMessage || 'Nenhum dado disponível.';
        trEmpty.appendChild(tdEmpty);
        tbody.appendChild(trEmpty);
    } else {
        linhas.forEach(row => {
            const tr = document.createElement('tr');
            row.forEach(cell => {
                const td = document.createElement('td');
                if (cell instanceof HTMLElement) {
                    td.appendChild(cell);
                } else {
                    td.innerText = cell !== null && cell !== undefined ? cell : '-';
                }
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
    }
    table.appendChild(tbody);
    wrapper.appendChild(table);
    return wrapper;
}

/**
 * Cria um input group estilizado (label + input).
 */
export function createInputGroup(label, type, id, options = {}) {
    const group = document.createElement('div');
    group.className = 'input-group';
    
    const labelEl = document.createElement('label');
    labelEl.htmlFor = id;
    labelEl.innerText = label;
    group.appendChild(labelEl);
    
    const input = document.createElement('input');
    input.type = type;
    input.id = id;
    input.className = 'input';
    input.value = options.value || '';
    if (options.min !== undefined) input.min = options.min;
    if (options.max !== undefined) input.max = options.max;
    if (options.step !== undefined) input.step = options.step;
    if (options.placeholder) input.placeholder = options.placeholder;
    if (options.required) input.required = true;
    
    group.appendChild(input);
    return group;
}

/**
 * Cria um select group estilizado (label + select).
 */
export function createSelectGroup(label, opcoes, id, options = {}) {
    const group = document.createElement('div');
    group.className = 'input-group';
    
    const labelEl = document.createElement('label');
    labelEl.htmlFor = id;
    labelEl.innerText = label;
    group.appendChild(labelEl);
    
    const select = document.createElement('select');
    select.id = id;
    select.className = 'select';

    opcoes.forEach(opt => {
        const option = document.createElement('option');
        option.value = opt.value;
        option.innerText = opt.label;
        if (opt.value === options.value) {
            option.selected = true;
        }
        select.appendChild(option);
    });
    
    group.appendChild(select);
    return group;
}

/**
 * Cria um estado vazio (Empty State).
 */
export function createEmptyState(mensagem, icone = '📊') {
    const el = document.createElement('div');
    el.className = 'empty-state';
    el.innerHTML = `
        <div class="empty-icon">${icone}</div>
        <p class="empty-message">${mensagem}</p>
    `;
    return el;
}

/**
 * Cria um Modal genérico.
 */
export function createModal(titulo, htmlConteudo, onClose) {
    const overlay = document.createElement('div');
    overlay.className = 'modal-overlay';
    
    const modal = document.createElement('div');
    modal.className = 'modal-container';
    
    modal.innerHTML = `
        <div class="modal-header">
            <h3>${titulo}</h3>
            <button class="modal-close">&times;</button>
        </div>
        <div class="modal-body">
            ${htmlConteudo}
        </div>
    `;
    
    overlay.appendChild(modal);
    document.body.appendChild(overlay);
    
    // Trava rolagem do body
    document.body.style.overflow = 'hidden';
    
    const fechar = () => {
        overlay.classList.add('modal-fade-out');
        overlay.addEventListener('animationend', () => {
            overlay.remove();
            document.body.style.overflow = '';
            if (onClose) onClose();
        });
    };
    
    modal.querySelector('.modal-close').addEventListener('click', fechar);
    overlay.addEventListener('click', (e) => {
        if (e.target === overlay) fechar();
    });
    
    return {
        close: fechar,
        element: modal
    };
}
