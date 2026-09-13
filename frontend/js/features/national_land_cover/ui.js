import { State } from './state.js';
import { parseCSVHeaders } from './utils/csv.js';
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
                if (this.panel) this.panel.style.display = 'none';
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
    },

    closePanel() {
        if (this.panel) this.panel.style.display = 'none';
        const btnNlc = document.getElementById('btn-nlc');
        if (btnNlc) btnNlc.classList.remove('active');
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

        const targetMatch = findMatch(['target', 'class', 'label', 'loi_pct']);
        if (targetMatch) mapTarget.value = targetMatch;
    },

    async submitJob() {
        const btn = document.getElementById('btn-submit-nlc-job');
        const countryNameInput = document.getElementById('nlc-country-name');
        
        if (!countryNameInput.value.trim()) {
            if (typeof Toast !== 'undefined') Toast.show('Please enter a Country Name', 'error');
            return;
        }

        btn.innerHTML = '<span class="processing-dots"><span></span><span></span><span></span></span> Submitting...';

        try {
            let payload = {
                ...State.taskingDefaults,
                aoi_name: countryNameInput.value.trim(),
                run_name: "custom_run_" + Date.now(),
                latitude_column: document.getElementById('nlc-map-lat').value,
                longitude_column: document.getElementById('nlc-map-lon').value,
                target_column: document.getElementById('nlc-map-target').value
            };

            if (State.customSourceType === 'csv') {
                if (!State.file) throw new Error("Please upload a CSV file first.");
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
