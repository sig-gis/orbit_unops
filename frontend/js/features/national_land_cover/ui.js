import { State } from './state.js';
import { parseCSVHeaders } from './utils/csv.js';
import { submitLandCoverJob } from './api.js';

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

        const btnDemo = document.getElementById('nlc-btn-demo');
        const btnCustom = document.getElementById('nlc-btn-custom');
        
        if (btnDemo) {
            btnDemo.addEventListener('click', () => this.handleDemoClick());
        }
        
        if (btnCustom) {
            btnCustom.addEventListener('click', () => this.handleCustomClick());
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
    },

    openPanel() {
        if (this.panel) this.panel.style.display = 'flex';
    },

    closePanel() {
        if (this.panel) this.panel.style.display = 'none';
    },

    handleDemoClick() {
        State.isDemoMode = true;
        document.getElementById('nlc-form-container').style.display = 'block';
        document.getElementById('nlc-demo-info').style.display = 'block';
        document.getElementById('nlc-upload-section').style.display = 'none';
        document.getElementById('nlc-mapping-section').style.display = 'none';
        
        // Auto-fill parameters and disable inputs for Demo mode
        const thresholdInput = document.getElementById('nlc-target-threshold');
        const treesInput = document.getElementById('nlc-trees');
        
        thresholdInput.value = State.demoParams.target_threshold;
        thresholdInput.disabled = true;
        
        treesInput.value = State.demoParams.number_of_trees;
        treesInput.disabled = true;
        
        // Visual feedback
        document.getElementById('nlc-btn-demo').style.background = 'var(--bg-tertiary)';
        document.getElementById('nlc-btn-demo').style.border = '1px solid var(--brand-secondary)';
        document.getElementById('nlc-btn-custom').style.background = '';
        document.getElementById('nlc-btn-custom').style.border = '';
    },

    handleCustomClick() {
        State.isDemoMode = false;
        document.getElementById('nlc-form-container').style.display = 'block';
        document.getElementById('nlc-demo-info').style.display = 'none';
        document.getElementById('nlc-upload-section').style.display = 'block';
        
        // Enable inputs for Custom mode
        document.getElementById('nlc-target-threshold').disabled = false;
        document.getElementById('nlc-trees').disabled = false;

        // Visual feedback
        document.getElementById('nlc-btn-custom').style.background = 'var(--bg-tertiary)';
        document.getElementById('nlc-btn-custom').style.border = '1px solid var(--brand-secondary)';
        document.getElementById('nlc-btn-demo').style.background = '';
        document.getElementById('nlc-btn-demo').style.border = '';
        
        // Initialize default tab state
        const activeTabBtn = document.querySelector('button[data-nlc-source].active');
        if (activeTabBtn) {
            this.handleSourceTabClick({ target: activeTabBtn });
        }
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
        
        // Handle mapping section
        const mappingSection = document.getElementById('nlc-mapping-section');
        const mappingDropdowns = document.getElementById('nlc-mapping-dropdowns');
        const mappingInputs = document.getElementById('nlc-mapping-inputs');
        
        if (source === 'csv') {
            if (State.file && State.headers.length > 0) {
                mappingSection.style.display = 'block';
                mappingDropdowns.style.display = 'flex';
                mappingInputs.style.display = 'none';
            } else {
                mappingSection.style.display = 'none';
            }
        } else {
            // For GEE/GCS, always show manual inputs
            mappingSection.style.display = 'block';
            mappingDropdowns.style.display = 'none';
            mappingInputs.style.display = 'flex';
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
            document.getElementById('nlc-mapping-inputs').style.display = 'none';
        } catch (error) {
            if (typeof Toast !== 'undefined') Toast.show('Failed to parse CSV headers', 'error');
            console.error(error);
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

        // Try to auto-select if headers match common names
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
        btn.innerHTML = '<span class="processing-dots"><span></span><span></span><span></span></span> Submitting...';

        try {
            let payload = {};

            if (State.isDemoMode) {
                payload = { ...State.demoParams };
                // Override with user edits if they changed the input fields
                payload.target_threshold = parseFloat(document.getElementById('nlc-target-threshold').value);
                payload.number_of_trees = parseInt(document.getElementById('nlc-trees').value);
            } else {
                // Extract parameters based on source type
                let latCol, lonCol, targetCol, inputAssetId;

                if (State.customSourceType === 'csv') {
                    if (!State.file) throw new Error("Please upload a CSV file first.");
                    latCol = document.getElementById('nlc-map-lat').value;
                    lonCol = document.getElementById('nlc-map-lon').value;
                    targetCol = document.getElementById('nlc-map-target').value;
                    inputAssetId = "projects/pc655-gcpa-unops-geo-is/assets/custom_upload";
                } else if (State.customSourceType === 'gee') {
                    inputAssetId = document.getElementById('nlc-gee-asset').value.trim();
                    if (!inputAssetId) throw new Error("Please enter a GEE Asset ID.");
                    latCol = document.getElementById('nlc-input-lat').value.trim() || "lat";
                    lonCol = document.getElementById('nlc-input-lon').value.trim() || "lon";
                    targetCol = document.getElementById('nlc-input-target').value.trim() || "target";
                } else if (State.customSourceType === 'gcs') {
                    const gcsUri = document.getElementById('nlc-gcs-uri').value.trim();
                    if (!gcsUri) throw new Error("Please enter a GCS URI.");
                    inputAssetId = gcsUri; // Store GCS URI in the asset ID field for now
                    latCol = document.getElementById('nlc-input-lat').value.trim() || "lat";
                    lonCol = document.getElementById('nlc-input-lon').value.trim() || "lon";
                    targetCol = document.getElementById('nlc-input-target').value.trim() || "target";
                }
                
                payload = {
                    ...State.demoParams, // Base defaults
                    run_name: "custom_run_" + Date.now(),
                    longitude_column: lonCol,
                    latitude_column: latCol,
                    target_column: targetCol,
                    target_threshold: parseFloat(document.getElementById('nlc-target-threshold').value),
                    number_of_trees: parseInt(document.getElementById('nlc-trees').value),
                    input_asset_id: inputAssetId
                };
            }

            const response = await submitLandCoverJob(payload);
            
            if (typeof Toast !== 'undefined') {
                Toast.show('Land Cover job submitted successfully', 'success');
            }
            
            this.closePanel();
            
            if (typeof App !== 'undefined' && App.navigate) {
                App.navigate('jobs');
            }
            if (typeof Jobs !== 'undefined' && Jobs.loadJobs) {
                Jobs.loadJobs();
            }

        } catch (error) {
            if (typeof Toast !== 'undefined') {
                Toast.show(error.message, 'error');
            }
        } finally {
            btn.innerHTML = '<i data-lucide="cpu" class="icon"></i> Run Classification';
            if (typeof lucide !== 'undefined') lucide.createIcons();
        }
    }
};
