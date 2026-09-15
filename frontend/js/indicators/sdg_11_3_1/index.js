import { UI } from './ui.js';
import { Data } from './data.js';

const SDG1131Plugin = {
    init() {
        UI.init();
    },

    onPanelOpened() {
        UI.openPanel();
    },

    onPanelClosed() {
        UI.closePanel();
    },

    onCountrySelected(countryName) {
        UI.openPanel();
        Data.fetchAndVisualizeData(countryName);
    },

    onCountryDeselected() {
        // Clear panel data or show empty state
        UI.setEmpty("Select a country to view analysis.");
    }
};

export default SDG1131Plugin;
