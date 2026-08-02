/* ═══════════════════════════════════════════════════════
   COMPASS - ORBIT Operations Center — Map Module
   ═══════════════════════════════════════════════════════ */

const MapModule = {
    map: null,
    baseLayers: {},
    dataLayers: {},
    drawnItems: null,
    aoiLayer: null,
    countryLayer: null,
    _initialized: false,

    init() {
        if (this._initialized) return;

        // Initialize map centered on world view
        this.map = L.map('map', {
            center: [20, 0],
            zoom: 2,
            zoomControl: true,
            attributionControl: true,
        });

        // Base layers (theme-aware)
        this.baseLayers.light = L.tileLayer(
            'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
            { attribution: '&copy; <a href="https://carto.com/">CARTO</a>', maxZoom: 19, subdomains: 'abcd' }
        );
        this.baseLayers.dark = L.tileLayer(
            'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
            { attribution: '&copy; <a href="https://carto.com/">CARTO</a>', maxZoom: 19, subdomains: 'abcd' }
        );
        this.baseLayers.satellite = L.tileLayer(
            'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            { attribution: 'Tiles &copy; Esri', maxZoom: 19 }
        );

        // Add the correct base layer based on theme
        const theme = document.documentElement.dataset.theme || 'light';
        this.baseLayers[theme].addTo(this.map);

        // Add basemap switcher control
        const baseMaps = {
            "Light Map": this.baseLayers.light,
            "Dark Map": this.baseLayers.dark,
            "Satellite": this.baseLayers.satellite
        };
        L.control.layers(baseMaps, null, { position: 'bottomleft' }).addTo(this.map);

        // Drawing layer
        this.drawnItems = new L.FeatureGroup();
        this.map.addLayer(this.drawnItems);

        // AOI display layer
        this.aoiLayer = L.geoJSON(null, {
            style: {
                color: '#0092D1',
                weight: 2,
                fillColor: '#00A997',
                fillOpacity: 0.08,
                dashArray: '6,4',
            }
        }).addTo(this.map);

        // Country boundaries layer
        this.countryLayer = L.geoJSON(null, {
            style: {
                color: '#4A90E2',
                weight: 1,
                fillColor: 'transparent',
                fillOpacity: 0
            },
            onEachFeature: (feature, layer) => {
                layer.on({
                    mouseover: (e) => {
                        const l = e.target;
                        // Only highlight if not currently selected
                        if (l.options.color !== '#E85C0E') {
                            l.setStyle({ weight: 2, fillOpacity: 0.1 });
                            l.bringToFront();
                        }
                    },
                    mouseout: (e) => {
                        const l = e.target;
                        // Reset if not selected
                        if (l.options.color !== '#E85C0E') {
                            this.countryLayer.resetStyle(l);
                        }
                    },
                    click: (e) => this._onCountryClick(e)
                });
            }
        }).addTo(this.map);

        // Load country boundaries
        fetch('/assets/countries.geojson')
            .then(res => res.json())
            .then(data => {
                this.countryLayer.addData(data);
                if (typeof App !== 'undefined' && App.onCountriesLoaded) {
                    App.onCountriesLoaded(data);
                }
            })
            .catch(err => console.error('Failed to load countries.geojson:', err));



        this._initialized = true;
    },

    // ── Custom 2-click rectangle ──
    switchTheme(theme) {
        const otherTheme = theme === 'dark' ? 'light' : 'dark';
        if (this.map.hasLayer(this.baseLayers[otherTheme])) {
            this.map.removeLayer(this.baseLayers[otherTheme]);
        }
        if (!this.map.hasLayer(this.baseLayers[theme])) {
            this.baseLayers[theme].addTo(this.map);
            this.baseLayers[theme].bringToBack();
        }
    },

    _onCountryClick(e) {
        const layer = e.target;
        const feature = layer.feature;
        
        // Highlight clicked country
        this.highlightCountry(feature, layer);

        // Notify App to sync top bar
        if (typeof App !== 'undefined' && App.selectCountry) {
            App.selectCountry(feature.properties.name, true); 
        }
    },

    highlightCountry(feature, layer = null) {
        // Reset all styles
        this.countryLayer.setStyle({
            color: '#4A90E2',
            weight: 1,
            fillColor: 'transparent',
            fillOpacity: 0.1
        });

        if (this.maskLayer) {
            this.map.removeLayer(this.maskLayer);
            this.maskLayer = null;
        }

        if (!layer && feature) {
            this.countryLayer.eachLayer(l => {
                if (l.feature.properties.name === feature.properties.name) layer = l;
            });
        }

        if (layer && feature) {
            layer.setStyle({
                weight: 2,
                color: '#E85C0E', // highlight color
                fillColor: 'transparent', // Don't obscure raster
                fillOpacity: 0
            });
            layer.bringToFront();
            this.map.fitBounds(layer.getBounds(), { padding: [50, 50] });

            // Create inverted mask polygon
            const outerBounds = [
                [-90, -360],
                [90, -360],
                [90, 360],
                [-90, 360]
            ];
            let holes = [];
            if (feature.geometry.type === 'Polygon') {
                holes.push(feature.geometry.coordinates[0].map(c => [c[1], c[0]]));
            } else if (feature.geometry.type === 'MultiPolygon') {
                feature.geometry.coordinates.forEach(poly => {
                    holes.push(poly[0].map(c => [c[1], c[0]]));
                });
            }
            if (holes.length > 0) {
                this.maskLayer = L.polygon([outerBounds, ...holes], {
                    stroke: false,
                    fillColor: '#000',
                    fillOpacity: 0.7,
                    interactive: false
                }).addTo(this.map);
            }
        }
    },

    addLegend(title, desc, color) {
        this.removeLegend();
        const legend = L.control({ position: 'bottomright' });
        legend.onAdd = function (map) {
            const div = L.DomUtil.create('div', 'info legend');
            div.style.backgroundColor = 'var(--bg-panel)';
            div.style.padding = '12px';
            div.style.borderRadius = '8px';
            div.style.boxShadow = '0 4px 15px rgba(0,0,0,0.1)';
            div.style.border = '1px solid var(--border-color)';
            div.style.color = 'var(--text-primary)';
            div.style.minWidth = '200px';
            div.innerHTML = `
                <div style="font-weight:bold;margin-bottom:8px;font-size:0.9rem;">${title}</div>
                <div style="display:flex;align-items:center;gap:8px;font-size:0.8rem;color:var(--text-muted)">
                    <div style="width:16px;height:16px;background:${color};border-radius:2px;"></div>
                    ${desc}
                </div>
            `;
            return div;
        };
        legend.addTo(this.map);
        this.activeLegend = legend;
    },

    removeLegend() {
        if (this.activeLegend) {
            this.map.removeControl(this.activeLegend);
            this.activeLegend = null;
        }
    },

    // ── Custom 2-click rectangle ──
    _rectFirstClick: null,
    _rectPreview: null,
    _drawingRect: false,

    enableDrawing() {
        this._drawingRect = true;
        this._rectFirstClick = null;
        this.map.getContainer().style.cursor = 'crosshair';
        Toast.show('Click the first corner of your AOI rectangle', 'info');

        // Bind click handler
        this._rectClickHandler = (e) => this._handleRectClick(e);
        this._rectMoveHandler = (e) => this._handleRectMove(e);
        this.map.on('click', this._rectClickHandler);
    },

    _handleRectClick(e) {
        if (!this._drawingRect) return;

        if (!this._rectFirstClick) {
            // First click — set corner 1
            this._rectFirstClick = e.latlng;
            this.map.on('mousemove', this._rectMoveHandler);
            Toast.show('Now click the opposite corner to complete the rectangle', 'info');
        } else {
            // Second click — complete rectangle
            const bounds = L.latLngBounds(this._rectFirstClick, e.latlng);
            const rect = L.rectangle(bounds, {
                color: '#0092D1',
                weight: 2,
                fillColor: '#00A997',
                fillOpacity: 0.15,
            });

            // Clean up preview
            if (this._rectPreview) {
                this.map.removeLayer(this._rectPreview);
                this._rectPreview = null;
            }

            // Add to drawn items and process
            this.drawnItems.addLayer(rect);
            this._finishDrawing();

            // Convert to GeoJSON
            const geojson = rect.toGeoJSON();
            this._processDrawnAOI(geojson.geometry);
        }
    },

    _handleRectMove(e) {
        if (!this._rectFirstClick) return;
        const bounds = L.latLngBounds(this._rectFirstClick, e.latlng);

        if (this._rectPreview) {
            this._rectPreview.setBounds(bounds);
        } else {
            this._rectPreview = L.rectangle(bounds, {
                color: '#0092D1',
                weight: 1,
                fillColor: '#00A997',
                fillOpacity: 0.1,
                dashArray: '4,4',
            }).addTo(this.map);
        }
    },

    _finishDrawing() {
        this._drawingRect = false;
        this._rectFirstClick = null;
        this.map.getContainer().style.cursor = '';
        this.map.off('click', this._rectClickHandler);
        this.map.off('mousemove', this._rectMoveHandler);
    },

    disableDrawing() {
        this._finishDrawing();
        if (this._rectPreview) {
            this.map.removeLayer(this._rectPreview);
            this._rectPreview = null;
        }
    },

    _wizardCallback: null,

    async _processDrawnAOI(geometry) {
        const name = prompt('Name this AOI:', `AOI ${new Date().toLocaleDateString()}`);
        if (!name) {
            // Remove last drawn item
            const layers = this.drawnItems.getLayers();
            if (layers.length) this.drawnItems.removeLayer(layers[layers.length - 1]);
            return;
        }

        try {
            const aoi = await API.createAOI({ name, geometry });
            Toast.show(`AOI "${name}" saved (${aoi.area_km2} km²)`, 'success');
            App.loadAOIs();

            if (this._wizardCallback) {
                this._wizardCallback(aoi);
            }
        } catch (err) {
            Toast.show(`Failed to save AOI: ${err.message}`, 'error');
        }
    },



    showAOI(geometry) {
        this.aoiLayer.clearLayers();
        if (geometry) {
            this.aoiLayer.addData({
                type: 'Feature',
                geometry: geometry,
                properties: {}
            });
            // Fit bounds
            const bounds = this.aoiLayer.getBounds();
            if (bounds.isValid()) {
                this.map.fitBounds(bounds, { padding: [50, 50], maxZoom: 13 });
            }
        }
    },

    addTileLayer(id, url, options = {}) {
        // Remove existing layer with same id
        if (this.dataLayers[id]) {
            this.map.removeLayer(this.dataLayers[id]);
        }

        if (!url) return;

        const layer = L.tileLayer(url, {
            opacity: options.opacity || 0.7,
            maxZoom: 18,
            attribution: 'Google Earth Engine',
        });

        if (options.visible !== false) {
            layer.addTo(this.map);
        }

        this.dataLayers[id] = layer;
        return layer;
    },

    toggleLayer(id, visible) {
        const layer = this.dataLayers[id];
        if (!layer) return;

        if (visible) {
            if (!this.map.hasLayer(layer)) {
                layer.addTo(this.map);
            }
        } else {
            if (this.map.hasLayer(layer)) {
                this.map.removeLayer(layer);
            }
        }
    },

    setLayerOpacity(id, opacity) {
        const layer = this.dataLayers[id];
        if (layer) layer.setOpacity(opacity);
    },

    invalidateSize() {
        if (this.map) {
            setTimeout(() => this.map.invalidateSize(), 100);
        }
    },
};
