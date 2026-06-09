/**
 * Template para Criação de Novas Abas
 * ------------------------------------
 * Como adicionar uma nova aba ao classificador Iris:
 * 
 * 1. Copie este arquivo para `frontend/js/tabs/nome-da-sua-aba.js` e renomeie.
 * 2. Implemente a lógica nos métodos init() e destroy().
 * 3. Importe e registre sua aba em `frontend/js/app.js`:
 *      const suaAba = await import('./tabs/nome-da-sua-aba.js');
 *      registrarAba('sua-aba-id', 'Título da Aba', suaAba.init, suaAba.destroy, '⚙️');
 * 
 * E a aba aparecerá automaticamente na barra de navegação superior, integrada e responsiva!
 */

import * as api from '../api.js';
import * as ui from '../components.js';
import * as charts from '../charts.js';

let activeContainer = null;

/**
 * Método de inicialização da aba.
 * Recebe o container HTML e renderiza os elementos e ativa ouvintes.
 * @param {HTMLElement} container - O elemento principal onde a aba deve ser injetada.
 */
export async function init(container) {
    activeContainer = container;
    
    container.innerHTML = `
        <div class="card" style="padding: 20px;">
            <h3 class="text-gradient">Nova Feature do Classificador</h3>
            <p class="text-secondary mt-10">
                Esta aba foi criada a partir do template de extensão do frontend do Classificador Iris.
            </p>
            <div class="mt-16" id="conteudo-demo">
                <!-- Conteúdo de exemplo -->
            </div>
            <button class="btn btn-primary mt-16" id="btn-demo">Executar Lógica</button>
        </div>
    `;
    
    // Bind eventos
    container.querySelector('#btn-demo').addEventListener('click', executarAcaoExemplo);
}

/**
 * Função de exemplo
 */
function executarAcaoExemplo() {
    const conteudo = activeContainer.querySelector('#conteudo-demo');
    if (conteudo) {
        conteudo.innerHTML = `
            <div class="result-box" style="padding: 10px; background: rgba(52,211,153,0.1); border: 1px solid var(--accent-green); border-radius: var(--radius-sm);">
                ✅ Lógica executada!
            </div>
        `;
    }
    ui.showToast('Lógica do template executada com sucesso!', 'success');
}

/**
 * Método de cleanup / destruição da aba.
 * Limpa timeouts, referências circulares ou eventos globais ao trocar de aba.
 */
export function destroy() {
    activeContainer = null;
    console.log('Cleanup da nova aba efetuado.');
}
