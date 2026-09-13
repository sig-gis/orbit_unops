import { UI } from './ui.js';

const NLCPlugin = {
    init() {
        UI.init();
        
        const btnNlc = document.getElementById('btn-nlc');
        if (btnNlc) {
            btnNlc.addEventListener('click', () => {
                if (btnNlc.classList.contains('active')) {
                    this.onPanelClosed();
                } else {
                    this.onPanelOpened();
                }
            });
        }
    },

    onPanelOpened() {
        const btnNlc = document.getElementById('btn-nlc');
        if (btnNlc) btnNlc.classList.add('active');
        // Close SDG panels cleanly via the SDG manager
        if (typeof SDG !== 'undefined' && SDG.closeCurrentIndicator) {
            SDG.closeCurrentIndicator();
            
            // Also unhighlight all SDG sidebar buttons
            document.querySelectorAll('.sdg-toggle-btn').forEach(b => b.classList.remove('active'));
        }

        // Navigate to map view if we are on a different page (like jobs)
        if (typeof App !== 'undefined' && App.navigate) {
            App.navigate('map');
        }
        
        // Hide analytics panel if open
        const nlcAnalyticsPanel = document.getElementById('nlc-analytics-panel');
        if (nlcAnalyticsPanel) nlcAnalyticsPanel.style.display = 'none';
        const nlcAnalyticsRestore = document.getElementById('nlc-analytics-panel-restore');
        if (nlcAnalyticsRestore) nlcAnalyticsRestore.style.display = 'none';

        UI.openPanel();
    },

    onPanelClosed() {
        const btnNlc = document.getElementById('btn-nlc');
        if (btnNlc) btnNlc.classList.remove('active');
        UI.closePanel();
    }
};

export default NLCPlugin;
