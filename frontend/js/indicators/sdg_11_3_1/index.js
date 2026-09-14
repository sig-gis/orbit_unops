import { UI } from './ui.js?v=14Sep26-1';
import { Data } from './data.js?v=14Sep26-1';

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
