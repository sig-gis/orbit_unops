/* ═══════════════════════════════════════════════════════
   COMPASS - ORBIT Operations Center — Indicator Manager
   ═══════════════════════════════════════════════════════ */

const SDG = {
    plugins: {}, // Holds registered SDG plugins
    activeIndicator: null,

    registerPlugin(indicatorId, plugin) {
        this.plugins[indicatorId] = plugin;
        console.log(`[IndicatorManager] Registered plugin for SDG ${indicatorId}`);
        plugin.init(); // Initialize the plugin's UI/Events
    },

    init() {
        // Bind sidebar toggle buttons
        document.querySelectorAll('.sdg-toggle-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                const indicatorId = btn.dataset.indicator;
                this.toggleIndicator(indicatorId, btn);
            });
        });
    },

    toggleIndicator(indicatorId, btnElement) {
        if (this.activeIndicator === indicatorId) {
            // Close it
            this.closeCurrentIndicator();
            btnElement.classList.remove('active');
        } else {
            // Switch to new indicator
            this.closeCurrentIndicator();
            
            this.activeIndicator = indicatorId;
            btnElement.classList.add('active');
            
            const plugin = this.plugins[this.activeIndicator];
            if (plugin) {
                plugin.onPanelOpened();
                
                // If a country is already selected, trigger the data load for this new SDG
                if (typeof App !== 'undefined' && App.currentCountry) {
                    plugin.onCountrySelected(App.currentCountry);
                }
            }
        }
    },

    closeCurrentIndicator() {
        if (!this.activeIndicator) return;

        const plugin = this.plugins[this.activeIndicator];
        if (plugin && plugin.onPanelClosed) {
            plugin.onPanelClosed();
        }

        // Remove active class from buttons
        document.querySelectorAll('.sdg-toggle-btn').forEach(btn => btn.classList.remove('active'));
        
        this.activeIndicator = null;
    },

    // ═══ Interface for App.js ═══

    onCountrySelected(countryName) {
        if (!this.activeIndicator) return;
        const plugin = this.plugins[this.activeIndicator];
        if (plugin && plugin.onCountrySelected) {
            plugin.onCountrySelected(countryName);
        }
    },

    onCountryDeselected() {
        if (!this.activeIndicator) return;
        const plugin = this.plugins[this.activeIndicator];
        if (plugin && plugin.onCountryDeselected) {
            plugin.onCountryDeselected();
        }
    }
};

document.addEventListener('DOMContentLoaded', () => SDG.init());
