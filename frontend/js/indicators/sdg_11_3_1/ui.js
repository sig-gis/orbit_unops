import * as Data from './data.js';

export const UI = {
    panel: null,
    heroData: null,
    jobSelectorContainer: null,

    init() {
        this.panel = document.getElementById('sdg-panel');
        this.heroData = document.getElementById('sdg-hero-data');

        const closeBtn = document.getElementById('sdg-panel-close');
        if (closeBtn) {
            closeBtn.addEventListener('click', () => {
                if (typeof SDG !== 'undefined') SDG.closeCurrentIndicator();
            });
        }

        const minBtn = document.getElementById('sdg-panel-minimize');
        const resBtn = document.getElementById('sdg-panel-restore');
        if (minBtn) {
            minBtn.addEventListener('click', () => {
                if (this.panel) this.panel.style.display = 'none';
                if (resBtn) resBtn.style.display = 'flex';
            });
        }
        if (resBtn) {
            resBtn.addEventListener('click', () => {
                if (this.panel) this.panel.style.display = 'flex';
                resBtn.style.display = 'none';
            });
        }

        document.getElementById('btn-submit-sdg-job')?.addEventListener('click', () => {
            this.submitJob();
        });

        this._populateYears();
    },

    _populateYears() {
        const startSelect = document.getElementById('sdg-start-year');
        const endSelect = document.getElementById('sdg-end-year');
        if (!startSelect || !endSelect) return;

        const currentYear = new Date().getFullYear();
        const baseYear = 2015;

        let html = '';
        for (let y = currentYear; y >= baseYear; y--) {
            html += `<option value="${y}">${y}</option>`;
        }

        startSelect.innerHTML = html;
        endSelect.innerHTML = html;

        startSelect.value = "2015";
        endSelect.value = currentYear.toString();
    },

    openPanel() {
        if (this.panel) this.panel.style.display = 'block';
    },

    closePanel() {
        if (this.panel) this.panel.style.display = 'none';
        const resBtn = document.getElementById('sdg-panel-restore');
        if (resBtn) resBtn.style.display = 'none';
        
        // Clean up the map when the panel closes
        if (window.MapModule) {
            MapModule.clearAllDataLayers();
        }
    },

    setLoading(country) {
        if (!this.heroData) return;
        this.heroData.innerHTML = `
            <div style="text-align: center; padding: 20px;">
                <span class="processing-dots"><span></span><span></span><span></span></span>
                <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 10px;">Fetching Pre-computed Analytics for ${country}...</p>
            </div>
        `;
    },

    setProcessing(country) {
        if (!this.heroData) return;
        this.heroData.innerHTML = `
            <div style="text-align: center; padding: 30px 20px;">
                <span class="processing-dots"><span></span><span></span><span></span></span>
                <div style="margin-top: 15px; font-weight: 500; color: var(--brand-secondary);">Analysis in Progress</div>
                <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 5px;">A job is currently running for ${country}. Check the Jobs panel for status.</p>
            </div>`;
    },

    setEmpty(msg) {
        if (!this.heroData) return;
        this.heroData.innerHTML = `
            <div style="background: var(--bg-primary); padding: 20px; border-radius: 8px; border: 1px solid var(--border-color); margin-top: 10px; text-align: center;">
                <i data-lucide="database" style="color: var(--text-muted); opacity: 0.5; width: 32px; height: 32px; margin-bottom: 10px;"></i>
                <p style="color: var(--text-muted); font-size: 0.9rem; margin: 0;">${msg}</p>
            </div>
        `;
        if (typeof lucide !== 'undefined') lucide.createIcons();
    },
    
    renderJobSelector(jobs, currentJob, onSelect) {
        if (jobs.length <= 1) return ''; // No dropdown if 1 or 0 jobs
        
        let options = jobs.map(j => {
            const isSelected = j.id === currentJob.id ? 'selected' : '';
            return `<option value="${j.id}" ${isSelected}>Analysis: ${j.date_range_start} - ${j.date_range_end}</option>`;
        }).join('');

        return `
            <div style="margin-bottom: 15px;">
                <label for="sdg-job-selector" style="font-size: 0.75rem; color: var(--text-muted); text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 4px; display: block;">Select Run</label>
                <select id="sdg-job-selector" class="form-input" style="padding: 6px; font-size: 0.9rem;">
                    ${options}
                </select>
            </div>
        `;
    },

    bindJobSelector(jobs, onSelect) {
        const select = document.getElementById('sdg-job-selector');
        if (select) {
            select.addEventListener('change', (e) => {
                const selectedJob = jobs.find(j => j.id === e.target.value);
                if (selectedJob) {
                    onSelect(selectedJob);
                }
            });
        }
    },

    async submitJob() {
        const activeCountry = (typeof App !== 'undefined') ? App.currentCountry : null;
        if (!activeCountry) {
            Toast.show('Please select a country first.', 'error');
            return;
        }

        const startYear = document.getElementById('sdg-start-year').value;
        const endYear = document.getElementById('sdg-end-year').value;
        const start = parseInt(startYear);
        const end = parseInt(endYear);

        if (!start || !end || start >= end) {
            Toast.show('End year must be greater than start year.', 'error');
            return;
        }
        
        if (end - start < 3) {
            Toast.show('SDG 11.3.1 requires at least a 4-year span (e.g. 2019 to 2022) to measure growth.', 'error');
            return;
        }

        const params = {
            indicator_id: '11.3.1',
            country: activeCountry,
            map_year: 2020, // Default map year
            year_start: parseInt(startYear),
            year_end: parseInt(endYear),
            span_target: parseInt(endYear) - parseInt(startYear),
            export_formats: ['csv', 'geotiff']
        };

        const btn = document.getElementById('btn-submit-sdg-job');
        if (btn) btn.innerHTML = '<span class="processing-dots"><span></span><span></span><span></span></span> Submitting...';

        try {
            if (typeof API !== 'undefined') {
                await API.createJob(params);
            }
            Toast.show(`Submitted job for ${activeCountry} (${startYear}-${endYear})`, 'success');
            
            this.closePanel();
            if (typeof SDG !== 'undefined') SDG.closeCurrentIndicator();
            
            if (typeof App !== 'undefined' && App.navigate) {
                App.navigate('jobs');
                setTimeout(() => {
                    document.querySelector('.tab-btn[data-tab="active"]')?.click();
                }, 50);
            }
            if (typeof Jobs !== 'undefined' && Jobs.loadJobs) {
                Jobs.loadJobs();
            }
        } catch (err) {
            Toast.show(`Job creation failed: ${err.message}`, 'error');
        } finally {
            if (btn) btn.innerHTML = '<i data-lucide="image" class="icon"></i> Generate Raster Export';
            if (typeof lucide !== 'undefined') lucide.createIcons();
        }
    }
};
