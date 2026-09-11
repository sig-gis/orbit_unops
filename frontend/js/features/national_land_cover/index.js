import { UI } from './ui.js';

const NLCPlugin = {
    init() {
        UI.init();
        
        const btnNlc = document.getElementById('btn-nlc');
        if (btnNlc) {
            btnNlc.addEventListener('click', () => {
                this.onPanelOpened();
            });
        }
    },

    onPanelOpened() {
        // Hide SDG panel if it's open
        const sdgPanel = document.getElementById('sdg-panel');
        if (sdgPanel) sdgPanel.style.display = 'none';

        UI.openPanel();
    },

    onPanelClosed() {
        UI.closePanel();
    }
};

export default NLCPlugin;
