import { State } from './state.js';
import { parseCSVHeaders, parseCSVData } from './utils/csv.js';
import { submitLandCoverJob, fetchGeeColumns, fetchGcsColumns } from './api.js';

export const UI = {
    panel: null,
    
    init() {
        this.panel = document.getElementById('nlc-panel');

        const closeBtn = document.getElementById('nlc-panel-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                this.closePanel();
            });
        }

        const minBtn = document.getElementById('nlc-panel-minimize');
        if (minBtn) {
            minBtn.addEventListener('click', () => {
                if (this.panel) {
                    this.panel.style.display = 'none';
                    const restoreBtn = document.getElementById('nlc-panel-restore');
                    if (restoreBtn) restoreBtn.style.display = 'flex';
                }
            });
        }
        
        const restoreBtn = document.getElementById('nlc-panel-restore');
        if (restoreBtn) {
            restoreBtn.addEventListener('click', () => {
                if (this.panel) {
                    this.panel.style.display = 'flex';
                    restoreBtn.style.display = 'none';
                }
            });
        }

        const fileInput = document.getElementById('nlc-file-upload');
        if (fileInput) {
            fileInput.addEventListener('change', (e) => this.handleFileUpload(e));
        }

        const submitBtn = document.getElementById('btn-submit-nlc-job');
        if (submitBtn) {
            submitBtn.addEventListener('click', () => this.submitJob());
        }

        const loadBtn = document.getElementById('btn-load-nlc-data');
        if (loadBtn) {
            loadBtn.addEventListener('click', () => this.loadData());
        }

        // Hook into App to auto-populate Country Name
        if (typeof App !== 'undefined') {
            const originalSelect = App.selectCountry;
            App.selectCountry = function(countryName, updateDropdown) {
                if (originalSelect) originalSelect.call(App, countryName, updateDropdown);
                const nlcCountry = document.getElementById('nlc-country-name');
                if (nlcCountry) nlcCountry.textContent = countryName;
            };
            const originalDeselect = App.deselectCountry;
            App.deselectCountry = function() {
                if (originalDeselect) originalDeselect.call(App);
                const nlcCountry = document.getElementById('nlc-country-name');
                if (nlcCountry) nlcCountry.textContent = 'None (Select on map)';
            };
            
            // Set initial if already selected
            if (App.currentCountry) {
                const nlcCountry = document.getElementById('nlc-country-name');
                if (nlcCountry) nlcCountry.textContent = App.currentCountry;
            }
        }

        // Close NLC panel if an SDG indicator is clicked
        document.querySelectorAll('.sdg-toggle-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this.closePanel();
            });
        });

        // Setup Data Source tabs
        document.querySelectorAll('button[data-nlc-source]').forEach(btn => {
            btn.addEventListener('click', (e) => this.handleSourceTabClick(e));
        });

        // Setup Blur Listeners for Auto-Fetching Columns
        const geeInput = document.getElementById('nlc-gee-asset');
        if (geeInput) {
            geeInput.addEventListener('blur', (e) => this.handleGeeBlur(e));
        }
        
        const gcsInput = document.getElementById('nlc-gcs-uri');
        if (gcsInput) {
            gcsInput.addEventListener('blur', (e) => this.handleGcsBlur(e));
        }

        // Initialize default tab state
        const activeTabBtn = document.querySelector('button[data-nlc-source].active');
        if (activeTabBtn) {
            this.handleSourceTabClick({ target: activeTabBtn });
        }
    },

    openPanel() {
        if (this.panel) this.panel.style.display = 'flex';
        const restoreBtn = document.getElementById('nlc-panel-restore');
        if (restoreBtn) restoreBtn.style.display = 'none';
    },

    closePanel() {
        if (this.panel) this.panel.style.display = 'none';
        const btnNlc = document.getElementById('btn-nlc');
        if (btnNlc) btnNlc.classList.remove('active');
        const restoreBtn = document.getElementById('nlc-panel-restore');
        if (restoreBtn) restoreBtn.style.display = 'none';
    },

    handleSourceTabClick(event) {
        const btn = event.target;
        const source = btn.dataset.nlcSource;
        State.customSourceType = source;
        
        // Update active tab styling
        document.querySelectorAll('button[data-nlc-source]').forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        // Hide all source containers
        document.getElementById('nlc-source-csv-container').style.display = 'none';
        document.getElementById('nlc-source-gee-container').style.display = 'none';
        document.getElementById('nlc-source-gcs-container').style.display = 'none';
        
        // Show correct container
        document.getElementById(`nlc-source-${source}-container`).style.display = 'block';
        
        // Only show mapping if we have headers
        const mappingSection = document.getElementById('nlc-mapping-section');
        const mappingDropdowns = document.getElementById('nlc-mapping-dropdowns');
        
        if (State.headers && State.headers.length > 0) {
            mappingSection.style.display = 'block';
            mappingDropdowns.style.display = 'flex';
        } else {
            mappingSection.style.display = 'none';
        }
    },

    async handleFileUpload(event) {
        const file = event.target.files[0];
        if (!file) return;

        State.file = file;
        try {
            const headers = await parseCSVHeaders(file);
            State.headers = headers;
            this.populateMappingDropdowns(headers);
            document.getElementById('nlc-mapping-section').style.display = 'block';
            document.getElementById('nlc-mapping-dropdowns').style.display = 'flex';
        } catch (error) {
            if (typeof Toast !== 'undefined') Toast.show('Failed to parse CSV headers', 'error');
            console.error(error);
        }
    },

    async handleGeeBlur(event) {
        const assetId = event.target.value.trim();
        if (!assetId) return;
        
        try {
            this.setMappingLoading(true);
            const data = await fetchGeeColumns(assetId);
            State.headers = data.columns;
            this.populateMappingDropdowns(data.columns);
            document.getElementById('nlc-mapping-section').style.display = 'block';
            document.getElementById('nlc-mapping-dropdowns').style.display = 'flex';
        } catch (error) {
            if (typeof Toast !== 'undefined') Toast.show('Failed to load GEE columns', 'error');
            console.error(error);
        } finally {
            this.setMappingLoading(false);
        }
    },

    async handleGcsBlur(event) {
        const gcsUri = event.target.value.trim();
        if (!gcsUri) return;
        
        try {
            this.setMappingLoading(true);
            const data = await fetchGcsColumns(gcsUri);
            State.headers = data.columns;
            this.populateMappingDropdowns(data.columns);
            document.getElementById('nlc-mapping-section').style.display = 'block';
            document.getElementById('nlc-mapping-dropdowns').style.display = 'flex';
        } catch (error) {
            if (typeof Toast !== 'undefined') Toast.show('Failed to load GCS columns', 'error');
            console.error(error);
        } finally {
            this.setMappingLoading(false);
        }
    },

    setMappingLoading(isLoading) {
        const mapLat = document.getElementById('nlc-map-lat');
        const mapLon = document.getElementById('nlc-map-lon');
        const mapTarget = document.getElementById('nlc-map-target');
        
        if (isLoading) {
            document.getElementById('nlc-mapping-section').style.display = 'block';
            document.getElementById('nlc-mapping-dropdowns').style.display = 'flex';
            const html = '<option>Loading...</option>';
            mapLat.innerHTML = html; mapLat.disabled = true;
            mapLon.innerHTML = html; mapLon.disabled = true;
            mapTarget.innerHTML = html; mapTarget.disabled = true;
        } else {
            mapLat.disabled = false;
            mapLon.disabled = false;
            mapTarget.disabled = false;
        }
    },

    populateMappingDropdowns(headers) {
        const createOptions = () => headers.map(h => `<option value="${h}">${h}</option>`).join('');
        const optionsHtml = createOptions();

        const mapLat = document.getElementById('nlc-map-lat');
        const mapLon = document.getElementById('nlc-map-lon');
        const mapTarget = document.getElementById('nlc-map-target');

        mapLat.innerHTML = optionsHtml;
        mapLon.innerHTML = optionsHtml;
        mapTarget.innerHTML = optionsHtml;

        const findMatch = (terms) => headers.find(h => terms.includes(h.toLowerCase()));
        
        const latMatch = findMatch(['lat', 'latitude', 'y']);
        if (latMatch) mapLat.value = latMatch;

        const lonMatch = findMatch(['lon', 'longitude', 'x', 'lng']);
        if (lonMatch) mapLon.value = lonMatch;

        const targetMatch = findMatch(['target', 'class', 'label', 'loi_pct', 'y']);
        if (targetMatch) mapTarget.value = targetMatch;
    },

    async loadData() {
        if (!State.file) {
            if (typeof Toast !== 'undefined') Toast.show('Please upload a CSV file first', 'error');
            return;
        }

        const btn = document.getElementById('btn-load-nlc-data');
        if (btn) btn.innerHTML = '<span class="processing-dots"><span></span><span></span><span></span></span> Loading...';

        try {
            const data = await parseCSVData(State.file);
            
            // Plot points on MapModule.map
            if (typeof MapModule !== 'undefined' && MapModule.map) {
                if (window.nlcGroundTruthLayer) {
                    MapModule.map.removeLayer(window.nlcGroundTruthLayer);
                }
                window.nlcGroundTruthLayer = L.layerGroup().addTo(MapModule.map);
                
                let numBlocks = new Set();
                
                data.forEach((row, i) => {
                    if (row.lat && row.lon) {
                        const isTrain = (i % 5 !== 0); // Mock 80/20 split
                        const color = isTrain ? '#4A90E2' : '#E85C0E'; // Blue/Orange
                        
                        L.circleMarker([row.lat, row.lon], {
                            radius: 4,
                            fillColor: color,
                            color: '#fff',
                            weight: 1,
                            opacity: 1,
                            fillOpacity: 0.8
                        }).addTo(window.nlcGroundTruthLayer);
                        
                        if (row.block_id) numBlocks.add(row.block_id);
                    }
                });
                
                // Zoom to points
                if (data.length > 0) {
                    const bounds = L.latLngBounds(data.map(r => [r.lat, r.lon]).filter(c => c[0] && c[1]));
                    MapModule.map.fitBounds(bounds, { padding: [20, 20] });
                }

                // Add legend
                const legendControl = L.control({ position: 'bottomright' });
                legendControl.onAdd = function () {
                    const div = L.DomUtil.create('div', 'info legend');
                    div.style.backgroundColor = 'var(--bg-secondary)';
                    div.style.padding = '12px';
                    div.style.borderRadius = '8px';
                    div.style.border = '1px solid var(--border-color)';
                    div.style.boxShadow = '0 4px 15px rgba(0,0,0,0.1)';
                    div.innerHTML = `
                        <div style="font-weight:bold; margin-bottom:8px; font-size:0.85rem; color: var(--text-primary);">Ground Truth Data</div>
                        <div style="display:flex; align-items:center; margin-bottom:4px; font-size:0.75rem; color: var(--text-secondary);">
                            <span style="display:inline-block; width:12px; height:12px; background:#4A90E2; border-radius:50%; margin-right:8px; border:1px solid #fff;"></span> Training Points
                        </div>
                        <div style="display:flex; align-items:center; font-size:0.75rem; color: var(--text-secondary);">
                            <span style="display:inline-block; width:12px; height:12px; background:#E85C0E; border-radius:50%; margin-right:8px; border:1px solid #fff;"></span> Validation Points
                        </div>
                    `;
                    return div;
                };
                if (window.nlcGroundTruthLegend) MapModule.map.removeControl(window.nlcGroundTruthLegend);
                legendControl.addTo(MapModule.map);
                window.nlcGroundTruthLegend = legendControl;

                // Show recommendations
                document.getElementById('nlc-rec-ppb').textContent = "5";
                document.getElementById('nlc-rec-mbf').textContent = "0.5";
                document.getElementById('nlc-rec-points').textContent = "7,775";
                document.getElementById('nlc-rec-blocks').textContent = "1,601";
                
                document.getElementById('nlc-recommendations-section').style.display = 'block';
                document.getElementById('btn-submit-nlc-job').style.display = 'flex';
                if (btn) btn.style.display = 'none'; // Hide Load Data button
                
                if (typeof Toast !== 'undefined') Toast.show(`Loaded ${data.length} points for tasking`, 'success');
            }
        } catch (error) {
            console.error(error);
            if (typeof Toast !== 'undefined') Toast.show('Failed to load data points', 'error');
            if (btn) {
                btn.innerHTML = '<i data-lucide="upload-cloud" class="icon"></i> Load Data';
                if (typeof lucide !== 'undefined') lucide.createIcons();
            }
        }
    },

    async submitJob() {
        const btn = document.getElementById('btn-submit-nlc-job');
        const countryNameText = document.getElementById('nlc-country-name').textContent;
        
        if (!countryNameText || countryNameText === 'None (Select on map)') {
            if (typeof Toast !== 'undefined') Toast.show('Please select a country on the map first', 'error');
            return;
        }

        btn.innerHTML = '<span class="processing-dots"><span></span><span></span><span></span></span> Submitting...';

        try {
            let payload = {
                ...State.taskingDefaults,
                aoi_name: countryNameText,
                run_name: "custom_run_" + Date.now(),
                latitude_column: document.getElementById('nlc-map-lat').value,
                longitude_column: document.getElementById('nlc-map-lon').value,
                target_column: document.getElementById('nlc-map-target').value,
                target_threshold: parseFloat(document.getElementById('nlc-target-threshold')?.value) || 0.5
            };

            if (State.customSourceType === 'csv') {
                if (!State.file) throw new Error("Please load data first.");
                // Overriding upload for demo: Use hardcoded EE asset ID instead of uploading CSV
                payload.input_asset_id = "projects/damage-control-403117/assets/peatlands-aboettcher-test-20260826_points";
            } else if (State.customSourceType === 'gee') {
                payload.input_asset_id = document.getElementById('nlc-gee-asset').value.trim();
                if (!payload.input_asset_id) throw new Error("Please enter a GEE Asset ID.");
            } else if (State.customSourceType === 'gcs') {
                const gcsUri = document.getElementById('nlc-gcs-uri').value.trim();
                if (!gcsUri) throw new Error("Please enter a GCS URI.");
                payload.csv_url = gcsUri;
            }
            
            await submitLandCoverJob(payload);
            
            if (typeof Toast !== 'undefined') {
                Toast.show('Land Map Tasking job submitted successfully', 'success');
            }
            
            this.closePanel();
            
            if (typeof App !== 'undefined' && App.navigate) {
                App.navigate('jobs');
            }
            if (typeof Jobs !== 'undefined' && Jobs.loadJobs) {
                Jobs.loadJobs();
            }
        } catch (error) {
            console.error('Job submission error:', error);
            if (typeof Toast !== 'undefined') Toast.show(error.message, 'error');
        } finally {
            btn.innerHTML = '<i data-lucide="cpu" class="icon"></i> Run Classification';
            if (typeof lucide !== 'undefined') lucide.createIcons();
        }
    }
};
