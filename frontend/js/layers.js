/* ═══════════════════════════════════════════════════════
   COMPASS - ORBIT Operations Center — Layers Module
   Handles GEE tile layer loading with cache support
   ═══════════════════════════════════════════════════════ */

const Layers = {
    layers: [],
    _loading: false,
    _cached: false,

    async load(aoiId) {
        if (this._loading) return;
        this._loading = true;

        try {
            const response = await API.getTileUrls(aoiId);

            // Handle new response format { layers: [...], cached: bool }
            if (response.layers) {
                this.layers = response.layers;
                this._cached = response.cached === true;
            } else if (Array.isArray(response)) {
                // Fallback for old response format
                this.layers = response;
                this._cached = false;
            }

            this._renderSidebarToggles();
            this._addLayersToMap();
        } catch (err) {
            console.warn('Failed to load layers:', err);
            Toast.show('Loading GEE layers... this may take a moment.', 'info');
        } finally {
            this._loading = false;
        }
    },

    _renderSidebarToggles() {
        const container = document.getElementById('layer-toggles');
        if (!container) return;

        const colors = {
            'sentinel2': '#0092D1',
        };

        const allowedLayers = ['sentinel2_rgb', 'sentinel2'];
        const layersToRender = this.layers.filter(l => allowedLayers.includes(l.type) || allowedLayers.includes(l.id));

        container.innerHTML = layersToRender.map(layer => `
            <div class="layer-toggle-item ${layer.visible ? 'active' : ''}"
                 data-layer-id="${layer.id}"
                 title="${layer.description}">
                <span class="layer-swatch" style="background:${colors[layer.type] || '#0092D1'}"></span>
                <span class="nav-label">${layer.name}</span>
                <span class="layer-toggle-switch"></span>
            </div>
        `).join('');

        // Bind toggle events
        container.querySelectorAll('.layer-toggle-item').forEach(item => {
            item.addEventListener('click', () => {
                const id = item.dataset.layerId;
                const isActive = item.classList.toggle('active');
                MapModule.toggleLayer(id, isActive);
            });
        });
    },

    _addLayersToMap() {
        this.layers.forEach(layer => {
            if (layer.tile_url) {
                MapModule.addTileLayer(layer.id, layer.tile_url, {
                    opacity: layer.opacity,
                    visible: layer.visible,
                });
            }
        });
    },

    setOpacity(layerId, opacity) {
        MapModule.setLayerOpacity(layerId, opacity);
        const layer = this.layers.find(l => l.id === layerId);
        if (layer) layer.opacity = opacity;
    }
};
