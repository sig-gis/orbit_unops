import { State } from './state.js';
import { readCSV, validateCSVRows, assessCountryOrientation } from './utils/validation.js';
import { submitLandCoverJob, fetchGeeColumns, fetchGcsColumns, uploadCSV } from './api.js';

export const UI = {
    panel: null,
    validationVersion: 0,
    validated: false,
    pendingPoints: null,

    showValidation(message = '', warning = false, upload = false) {
        const el = document.getElementById(upload ? 'nlc-upload-error' : 'nlc-validation-message');
        el.textContent = message;
        el.style.color = warning ? '#d4a017' : '#dc2626';
        el.style.display = message ? 'block' : 'none';
    },

    resetValidation() {
        this.validationVersion++;
        this.validated = false;
        this.pendingPoints = null;
        this.showValidation();
        document.getElementById('nlc-warning-actions').hidden = true;
        document.getElementById('btn-submit-nlc-job').style.display = 'none';
        document.getElementById('nlc-recommendations-section').style.display = 'none';
        const btn = document.getElementById('btn-load-nlc-data');
        btn.style.display = '';
        btn.disabled = false;
        btn.textContent = 'Load Data';
        if (window.nlcGroundTruthLayer && typeof MapModule !== 'undefined') MapModule.map.removeLayer(window.nlcGroundTruthLayer);
        if (window.nlcGroundTruthLegend && typeof MapModule !== 'undefined') MapModule.map.removeControl(window.nlcGroundTruthLegend);
        window.nlcGroundTruthLayer = null;
        window.nlcGroundTruthLegend = null;
    },

    init() {
        this.panel = document.getElementById('nlc-panel');
        ['nlc-map-lat', 'nlc-map-lon', 'nlc-map-target'].forEach(id => {
            document.getElementById(id).addEventListener('change', () => this.resetValidation());
        });
        document.getElementById('nlc-warning-continue').addEventListener('click', () => {
            const points = this.pendingPoints;
            this.pendingPoints = null;
            document.getElementById('nlc-warning-actions').hidden = true;
            if (points) this.loadData(points);
        });
        document.getElementById('nlc-warning-cancel').addEventListener('click', () => this.resetValidation());

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
            App.selectCountry = function (countryName, updateDropdown) {
                UI.resetValidation();
                if (originalSelect) originalSelect.call(App, countryName, updateDropdown);
                const nlcCountry = document.getElementById('nlc-country-name');
                if (nlcCountry) nlcCountry.textContent = countryName;
            };
            const originalDeselect = App.deselectCountry;
            App.deselectCountry = function () {
                UI.resetValidation();
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

        // Keep selected country in sync with App state
        const nlcCountry = document.getElementById('nlc-country-name');
        if (nlcCountry) {
            if (typeof App !== 'undefined' && App.currentCountry) {
                nlcCountry.textContent = App.currentCountry;
                nlcCountry.style.color = 'var(--brand-secondary)';
            } else {
                nlcCountry.textContent = 'None (Select on map or search)';
                nlcCountry.style.color = 'var(--text-muted)';
            }
        }
    },

    closePanel() {
        if (this.panel) this.panel.style.display = 'none';
        const btnNlc = document.getElementById('btn-nlc');
        if (btnNlc) btnNlc.classList.remove('active');
        const restoreBtn = document.getElementById('nlc-panel-restore');
        if (restoreBtn) restoreBtn.style.display = 'none';
    },

    handleSourceTabClick(event) {
        this.resetValidation();
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
        this.resetValidation();
        this.showValidation('', false, true);
        State.file = null;
        State.headers = [];
        document.getElementById('nlc-mapping-section').style.display = 'none';
        const file = event.target.files[0];
        if (!file) return;
        const version = this.validationVersion;
        try {
            const { headers } = await readCSV(file);
            if (version !== this.validationVersion) return;
            State.file = file;
            State.headers = headers;
            this.populateMappingDropdowns(headers);
            document.getElementById('nlc-mapping-section').style.display = 'block';
            document.getElementById('nlc-mapping-dropdowns').style.display = 'flex';
        } catch (error) {
            if (version !== this.validationVersion) return;
            this.showValidation(error.message, false, true);
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
        const createOptions = () => headers.map(h => {
            const option = document.createElement('option');
            option.value = h;
            option.textContent = h;
            return option.outerHTML;
        }).join('');
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

    async loadData(approvedPoints = null) {
        if (!State.file) {
            this.showValidation('Please upload a CSV file first.');
            return;
        }
        const version = this.validationVersion;
        const btn = document.getElementById('btn-load-nlc-data');
        btn.disabled = true;
        if (btn) btn.innerHTML = '<span class="processing-dots"><span></span><span></span><span></span></span> Loading...';

        try {
            if (!App.currentCountry) throw new Error('Please select a country before loading data.');
            let data = approvedPoints;
            if (!Array.isArray(data)) {
                this.showValidation();
                this.pendingPoints = null;
                document.getElementById('nlc-warning-actions').hidden = true;
                const { rows } = await readCSV(State.file);
                if (version !== this.validationVersion) return;
                data = validateCSVRows(rows, {
                    lat: document.getElementById('nlc-map-lat').value,
                    lon: document.getElementById('nlc-map-lon').value,
                    target: document.getElementById('nlc-map-target').value
                });
                const { outside, hint } = assessCountryOrientation(data, App.currentCountry, window.countriesGeoJSON, window.turf);
                const message = `${outside} of ${data.length} points do not intersect the selected country, ${App.currentCountry}.${hint}`;
                if (outside === data.length) throw new Error(`${message} Check the country and coordinate columns.`);
                if (outside) {
                    this.showValidation(`${message} Continue with all points?`, true);
                    this.pendingPoints = data;
                    document.getElementById('nlc-warning-actions').hidden = false;
                    return;
                }
            }

            // Plot points on MapModule.map
            if (typeof MapModule !== 'undefined' && MapModule.map) {
                if (window.nlcGroundTruthLayer) {
                    MapModule.map.removeLayer(window.nlcGroundTruthLayer);
                }
                window.nlcGroundTruthLayer = L.layerGroup().addTo(MapModule.map);

                let blocks = {};
                let validPointsCount = 0;

                // Render up to 5000 points to avoid browser lag
                const maxPoints = 5000;
                const sampleRate = Math.max(1, Math.ceil(data.length / maxPoints));

                data.forEach((row, i) => {
                    if (Number.isFinite(row.lat) && Number.isFinite(row.lon)) {
                        validPointsCount++;
                        if (row.block_id) {
                            blocks[row.block_id] = true;
                        }

                        // Plot a sample of points
                        if (i % sampleRate === 0) {
                            // Assume target >= 0.5 is one class, else another
                            const targetVal = row.target;
                            const isPeat = !isNaN(targetVal) && targetVal >= 0.5;
                            const color = isPeat ? '#E85C0E' : '#4A90E2';
                            
                            L.circleMarker([parseFloat(row.lat), parseFloat(row.lon)], {
                                radius: 4,
                                color: color,
                                fillColor: color,
                                fillOpacity: 0.8,
                                weight: 1
                            }).addTo(window.nlcGroundTruthLayer);
                        }
                    }
                });

                // Zoom to points
                if (data.length > 0) {
                    const bounds = L.latLngBounds(data.map(r => [r.lat, r.lon]));
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
                            <span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#E85C0E; margin-right:8px;"></span> Positive (Target)
                        </div>
                        <div style="display:flex; align-items:center; font-size:0.75rem; color: var(--text-secondary);">
                            <span style="display:inline-block; width:12px; height:12px; border-radius:50%; background:#4A90E2; margin-right:8px;"></span> Negative (Non-Target)
                        </div>
                    `;
                    return div;
                };
                if (window.nlcGroundTruthLegend) MapModule.map.removeControl(window.nlcGroundTruthLegend);
                legendControl.addTo(MapModule.map);
                window.nlcGroundTruthLegend = legendControl;

                // Show dynamic recommendations
                const totalBlocks = Object.keys(blocks).length;
                document.getElementById('nlc-rec-ppb').textContent = "5"; // Default configurable recommendation
                document.getElementById('nlc-rec-mbf').textContent = "0.5";
                document.getElementById('nlc-rec-points').textContent = validPointsCount.toLocaleString();
                document.getElementById('nlc-rec-blocks').textContent = totalBlocks > 0 ? totalBlocks.toLocaleString() : "N/A";

                document.getElementById('nlc-recommendations-section').style.display = 'block';
                document.getElementById('btn-submit-nlc-job').style.display = 'flex';
                this.validated = true;
                if (btn) btn.style.display = 'none'; // Hide Load Data button

                if (typeof Toast !== 'undefined') Toast.show(`Loaded ${data.length} points for tasking`, 'success');
            }
        } catch (error) {
            if (version !== this.validationVersion) return;
            console.error(error);
            this.showValidation(error.message);
            if (btn) {
                btn.innerHTML = '<i data-lucide="upload-cloud" class="icon"></i> Load Data';
                if (typeof lucide !== 'undefined') lucide.createIcons();
            }
        } finally {
            if (version === this.validationVersion) {
                btn.disabled = false;
                btn.textContent = 'Load Data';
            }
        }
    },

    async submitJob() {
        const version = this.validationVersion;
        if (State.customSourceType === 'csv' && !this.validated) {
            this.showValidation('Please load and validate the CSV before running classification.');
            return;
        }
        const btn = document.getElementById('btn-submit-nlc-job');
        let selectedCountry = (typeof App !== 'undefined' && App.currentCountry) ? App.currentCountry.trim() : '';
        if (!selectedCountry) {
            const countryEl = document.getElementById('nlc-country-name');
            const txt = countryEl ? countryEl.textContent.trim() : '';
            if (txt && !txt.toLowerCase().includes('select') && txt.toLowerCase() !== 'none') {
                selectedCountry = txt;
            }
        }

        if (!selectedCountry) {
            if (typeof Toast !== 'undefined') {
                Toast.show('Please select a country on the map or search bar before running the job.', 'error');
            }
            const countryEl = document.getElementById('nlc-country-name');
            if (countryEl) {
                countryEl.style.color = '#FF5722';
                countryEl.textContent = 'Country Required (Select on map or search)';
            }
            return;
        }

        btn.innerHTML = '<span class="processing-dots"><span></span><span></span><span></span></span> Submitting...';

        try {
            let payload = {
                ...State.taskingDefaults,
                aoi_name: selectedCountry,
                country: selectedCountry,
                run_name: "custom_run_" + Date.now(),
                latitude_column: document.getElementById('nlc-map-lat').value,
                longitude_column: document.getElementById('nlc-map-lon').value,
                target_column: document.getElementById('nlc-map-target').value,
                target_threshold: parseFloat(document.getElementById('nlc-target-threshold')?.value) || 0.5
            };

            if (State.customSourceType === 'csv') {
                if (!State.file) throw new Error("Please load data first.");
                // Upload the CSV to the backend and get the GCS URI
                const uploadResponse = await uploadCSV(State.file, payload);
                if (version !== this.validationVersion || !this.validated) throw new Error('Inputs changed during upload. Load and validate the CSV again.');
                payload.target_threshold = 0.5;
                payload.csv_url = uploadResponse.gcs_uri;
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
