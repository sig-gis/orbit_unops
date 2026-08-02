const SDG = {
    panel: null,
    btn: null,
    closeBtn: null,
    heroData: null,
    activeCountry: null,
    currentJob: null,
    charts: { cumulative: null, growth: null },

    init() {
        this.panel = document.getElementById('sdg-panel');
        this.btn = document.getElementById('btn-sdg-11');
        this.closeBtn = document.getElementById('sdg-panel-close');
        this.heroData = document.getElementById('sdg-hero-data');

        if (this.btn) {
            this.btn.addEventListener('click', () => this.togglePanel());
        }

        if (this.closeBtn) {
            this.closeBtn.addEventListener('click', () => this.closePanel());
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

    togglePanel() {
        if (this.panel.style.display === 'none') {
            this.openPanel();
        } else {
            this.closePanel();
        }
    },

    openPanel() {
        this.panel.style.display = 'block';
        this.btn?.classList.add('active');
        if (!this.currentJob && this.activeCountry) {
             this._fetchAndVisualizeData(this.activeCountry);
        }
    },

    closePanel() {
        this.panel.style.display = 'none';
        this.btn?.classList.remove('active');
    },

    onCountrySelected(countryName) {
        this.activeCountry = countryName;
        this.currentJob = null; // Reset
        this.openPanel();
        this._fetchAndVisualizeData(countryName);
    },

    async loadJobDashboard(job, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;

        container.innerHTML = `
            <div style="text-align: center; padding: 20px;">
                <span class="processing-dots"><span></span><span></span><span></span></span>
                <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 10px;">Fetching Pre-computed Analytics for ${job.aoi_name}...</p>
            </div>
        `;

        try {
            const countryFmt = job.aoi_name.toLowerCase().replace(/ /g, "_").replace(/-/g, "_");
            const dates = `${job.date_range_start}-${job.date_range_end}`;
            const baseUrl = `https://storage.googleapis.com/unops/exports/unops/${job.fileId}/urban_extent_${countryFmt}_${dates}`;
            
            const parseCSV = (url) => new Promise((resolve, reject) => {
                const proxyUrl = `http://localhost:8000/proxy-csv?url=${encodeURIComponent(url)}`;
                Papa.parse(proxyUrl, {
                    download: true, header: true, dynamicTyping: true, skipEmptyLines: true,
                    transformHeader: h => h.trim(),
                    complete: (results) => resolve(results.data),
                    error: (err) => reject(err)
                });
            });

            const [annualData, areaData, spanData, statsData] = await Promise.all([
                parseCSV(`${baseUrl}_LCRPGR_annual_10m.csv`),
                parseCSV(`${baseUrl}_area10m_smoothed.csv`),
                parseCSV(`${baseUrl}_LCRPGR_5yr_10m.csv`),
                parseCSV(`${baseUrl}_prediction_stats.csv`)
            ]);

            // Create a wrapper for hero data so _renderDashboard can use it
            container.innerHTML = '<div id="sdg-hero-data-injected"></div><div id="sdg-charts-injected"></div>';
            this.heroData = document.getElementById('sdg-hero-data-injected');
            this._renderDashboard(annualData, areaData, spanData, statsData, job.aoi_name, 'sdg-charts-injected', job);
        } catch (error) {
            let msg = error.message || String(error);
            if (msg.includes('404')) {
                msg = "Data not found. The CSV results may not have exported correctly.";
            }
            container.innerHTML = `<div class="error-inline">Failed to load analytics: ${msg}</div>`;
        }
    },

    async _fetchAndVisualizeData(country) {
        const heroData = document.getElementById('sdg-hero-data');
        if (!heroData) return;

        heroData.innerHTML = `
            <div style="text-align: center; padding: 20px;">
                <span class="processing-dots"><span></span><span></span><span></span></span>
                <p style="color: var(--text-muted); font-size: 0.85rem; margin-top: 10px;">Fetching Pre-computed Analytics for ${country}...</p>
            </div>
        `;

        try {
            const countryFmt = country.toLowerCase().replace(/ /g, "_").replace(/-/g, "_");
            
            // Find latest completed job for this country
            const jobs = await API.listJobs();
            const latestJob = jobs.find(j => j.aoi_name.toLowerCase() === country.toLowerCase() && j.state === 'COMPLETED');
            
            let fileId, dates;
            if (latestJob) {
                fileId = latestJob.fileId;
                dates = `${latestJob.date_range_start}-${latestJob.date_range_end}`;
            } else {
                heroData.innerHTML = `<div class="empty-state-sm" style="padding:20px;text-align:center;">No completed analytics found for ${country}. Please run an export job first.</div>`;
                return;
            }

            const baseUrl = `https://storage.googleapis.com/unops/exports/unops/${fileId}/urban_extent_${countryFmt}_${dates}`;
            
            const parseCSV = (url) => new Promise((resolve, reject) => {
                const proxyUrl = `http://localhost:8000/proxy-csv?url=${encodeURIComponent(url)}`;
                Papa.parse(proxyUrl, {
                    download: true, header: true, dynamicTyping: true, skipEmptyLines: true,
                    transformHeader: h => h.trim(),
                    complete: (results) => resolve(results.data),
                    error: (err) => reject(err)
                });
            });

            const [annualData, areaData, spanData, statsData] = await Promise.all([
                parseCSV(`${baseUrl}_LCRPGR_annual_10m.csv`),
                parseCSV(`${baseUrl}_area10m_smoothed.csv`),
                parseCSV(`${baseUrl}_LCRPGR_5yr_10m.csv`),
                parseCSV(`${baseUrl}_prediction_stats.csv`)
            ]);

            this.heroData = heroData;
            this._renderDashboard(annualData, areaData, spanData, statsData, country, null, latestJob);
        } catch (error) {
            let msg = error.message || String(error);
            if (msg.includes("404") || msg.includes("ProgressEvent")) {
                msg = `No pre-computed data available for ${country}. Submit a raster export job below to generate it.`;
            }
            heroData.innerHTML = `
                <div style="background: var(--bg-primary); padding: 20px; border-radius: 8px; border: 1px solid var(--border-color); margin-top: 10px; text-align: center;">
                    <i data-lucide="database" style="color: var(--text-muted); opacity: 0.5; width: 32px; height: 32px; margin-bottom: 10px;"></i>
                    <p style="color: var(--text-muted); font-size: 0.9rem; margin: 0;">${msg}</p>
                </div>
            `;
            if (typeof lucide !== 'undefined') lucide.createIcons();
        }
    },

    _renderDashboard(annualData, areaData, spanData, statsData, country, chartContainerId = null, job = null) {
        if (!this.heroData) return;

        const validAnnual = annualData.filter(r => r.mid_year && r.urban_method === 'RF' && r.pop_source === 'GHS_POP');
        const validArea = areaData.filter(r => r.year);
        const validSpan = spanData.filter(r => r.span && r.urban_method === 'RF' && r.pop_source === 'GHS_POP');

        if (validAnnual.length === 0 || validArea.length === 0) {
            this.heroData.innerHTML = `<p style="color: var(--text-muted);">Tabular data is incomplete for ${country}.</p>`;
            return;
        }

        let overallRatio = "N/A";
        let verdict = "";
        let verdictColor = "var(--text-muted)";
        let bupc0 = "N/A", bupc1 = "N/A";
        
        if (validSpan.length > 0) {
            const row = validSpan[0];
            const lcrpgr = row.LCRPGR;
            overallRatio = lcrpgr.toFixed(2);
            if (lcrpgr > 1) { verdict = "SPRAWLING"; verdictColor = "#FF5722"; }
            else if (lcrpgr > 0 && lcrpgr <= 1) { verdict = "DENSIFYING"; verdictColor = "#4CAF50"; }
            else { verdict = "DENSIFYING"; verdictColor = "#4CAF50"; }
            
            bupc0 = row.BUpc_t0_m2 ? Math.round(row.BUpc_t0_m2) : "N/A";
            bupc1 = row.BUpc_t1_m2 ? Math.round(row.BUpc_t1_m2) : "N/A";
        }

        let aiAccuracy = "N/A";
        if (statsData && statsData.length > 0 && statsData[0].validation_accuracy) {
            aiAccuracy = (statsData[0].validation_accuracy * 100).toFixed(1) + "%";
        }

        // Inject Job specific actions (Toggle Raster and Downloads)
        let jobActionsHtml = '';
        if (job) {
            const datesStr = `${job.date_range_start}-${job.date_range_end}`;
            jobActionsHtml = `
                <div style="margin-top: 15px; border-top: 1px solid var(--border-color); padding-top: 15px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4 style="margin:0; font-size: 0.85rem;">Job Actions (${datesStr})</h4>
                        <button id="sdg-toggle-raster-btn" class="action-btn" style="padding: 4px 8px;">
                            <i data-lucide="eye" class="icon sm"></i> Show Raster
                        </button>
                    </div>
                    <div id="sdg-downloads-container" style="margin-top: 10px;"></div>
                </div>
            `;
        }

        let dateStr = "Analysis";
        if (job) {
             dateStr = `${job.date_range_start}-${job.date_range_end} Analysis`;
        } else if (validSpan.length > 0 && validSpan[0].window) {
             dateStr = `${validSpan[0].window} Analysis`;
        }

        this.heroData.innerHTML = `
            <div style="background: var(--bg-primary); padding: 15px; border-radius: 8px; border: 1px solid var(--border-color); margin-top: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                <div style="margin-bottom: 15px; padding-bottom: 15px; border-bottom: 1px solid var(--border-color);">
                    <h4 style="margin: 0 0 5px 0; font-size: 1rem;"><i data-lucide="bar-chart-2" class="icon sm" style="color: var(--brand-secondary);"></i> ${country} SDG 11.3.1</h4>
                    <p style="margin: 0; font-size: 0.8rem; color: var(--text-muted);">${dateStr} (RF + GHS_POP)</p>
                </div>
                
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
                    <div style="background: rgba(0,0,0,0.02); padding: 10px; border-radius: 6px; text-align: center; grid-column: span 2;">
                        <div style="font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); margin-bottom: 5px;">LCRPGR Verdict</div>
                        <div style="font-size: 1.2rem; font-weight: bold; color: ${verdictColor};">${overallRatio}</div>
                        <div style="font-size: 0.7rem; color: ${verdictColor}; opacity: 0.8;">${verdict}</div>
                    </div>
                    <div style="background: rgba(0,0,0,0.02); padding: 10px; border-radius: 6px; text-align: center;">
                        <div style="font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); margin-bottom: 5px;">Area Per Person</div>
                        <div style="font-size: 0.95rem; font-weight: bold; color: var(--text-primary);">${bupc0} ➔ ${bupc1}</div>
                        <div style="font-size: 0.7rem; color: var(--text-muted);">m² / capita</div>
                    </div>
                    <div style="background: rgba(0,0,0,0.02); padding: 10px; border-radius: 6px; text-align: center;">
                        <div style="font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); margin-bottom: 5px;">AI Confidence</div>
                        <div style="font-size: 0.95rem; font-weight: bold; color: var(--text-primary);">${aiAccuracy}</div>
                        <div style="font-size: 0.7rem; color: var(--text-muted);">Validation Acc.</div>
                    </div>
                </div>
                
                ${!chartContainerId ? `
                <div style="display: flex; flex-direction: column; gap: 15px; margin-top: 15px;">
                    <div style="height: 180px;"><canvas id="hero-area-chart"></canvas></div>
                    <div style="height: 180px;"><canvas id="hero-rate-chart"></canvas></div>
                </div>
                ` : `<div id="${chartContainerId}"></div>`}
                
                ${jobActionsHtml}
            </div>
        `;
        
        if (job && !chartContainerId) {
            // Bind toggle button ONLY if it's the floating panel
            let layersVisible = false;
            const toggleBtn = document.getElementById('sdg-toggle-raster-btn');
            if (toggleBtn) {
                toggleBtn.onclick = () => {
                    layersVisible = !layersVisible;
                    toggleBtn.innerHTML = layersVisible ? '<i data-lucide="eye-off" class="icon sm"></i> Hide Raster' : '<i data-lucide="eye" class="icon sm"></i> Show Raster';
                    if (typeof lucide !== 'undefined') lucide.createIcons();
                    if (job.layers) {
                        job.layers.forEach(l => {
                            if (layersVisible) {
                                // Add layer on demand if it wasn't there
                                if (typeof MapModule !== 'undefined') {
                                    MapModule.addTileLayer(`job_${l.id}`, l.tile_url, {
                                        opacity: 0.7,
                                        visible: true,
                                    });
                                }
                            } else {
                                if (typeof MapModule !== 'undefined') {
                                    MapModule.toggleLayer(`job_${l.id}`, false);
                                }
                            }
                        });
                        if (layersVisible && typeof Toast !== 'undefined') {
                            Toast.show(`Loaded ${job.layers.length} high-res layers on map`, 'success');
                        }
                    }
                };
            }
            // Render downloads
            this._renderJobDownloads(job.id, 'sdg-downloads-container');
        }

        this._renderCharts(validAnnual, validArea, validSpan, chartContainerId);
    },

    async _renderJobDownloads(jobId, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '<div style="font-size:0.8rem; color:var(--text-muted);">Loading downloads...</div>';
        try {
            const downloads = await API.getJobDownloads(jobId);
            if (downloads && downloads.length > 0) {
                container.innerHTML = downloads.map(dl => `
                    <div class="download-item" style="padding: 6px; margin-bottom:4px; font-size:0.8rem;">
                        <i data-lucide="file-output" class="icon sm"></i>
                        <div class="download-info">
                            <span class="download-name">${dl.name}</span>
                        </div>
                        <a href="${dl.url}" target="_blank" class="download-btn" title="Download File">
                            <i data-lucide="download" class="icon sm"></i>
                        </a>
                    </div>
                `).join('');
                lucide.createIcons();
            } else {
                container.innerHTML = '<div class="empty-state-sm" style="font-size:0.8rem;">No downloads found.</div>';
            }
        } catch(e) {
            container.innerHTML = `<div class="error-inline" style="font-size:0.8rem;">Error loading downloads</div>`;
        }
    },

    _renderCharts(annualData, areaData, spanData, chartContainerId = null) {
        const areaLabels = areaData.map(r => r.year.toString());
        const rfArea = areaData.map(r => r.RF_sm_km2);
        const annualLabels = annualData.map(r => r.window);
        const lcrData = annualData.map(r => r.LCR * 100);
        const pgrData = annualData.map(r => r.PGR * 100);

        if (chartContainerId) {
            const container = document.getElementById(chartContainerId);
            container.innerHTML = `
                <div style="height: 180px;"><canvas id="hero-area-chart"></canvas></div>
                <div style="height: 180px; margin-top: 15px;"><canvas id="hero-rate-chart"></canvas></div>
            `;
        }

        // Destroy old instances
        if (this.charts.cumulative) this.charts.cumulative.destroy();
        if (this.charts.growth) this.charts.growth.destroy();

        this.charts.cumulative = new Chart(document.getElementById('hero-area-chart').getContext('2d'), {
            type: 'bar',
            data: {
                labels: areaLabels,
                datasets: [{
                    label: 'Total Urban Area (km²)',
                    data: rfArea,
                    backgroundColor: 'rgba(74, 144, 226, 0.7)',
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: true, position: 'top', labels: { boxWidth: 10, font: {size: 10} } } },
                scales: { x: { grid: {display: false} }, y: { title: {display: true, text: 'km²', font: {size: 10}} } }
            }
        });

        this.charts.growth = new Chart(document.getElementById('hero-rate-chart').getContext('2d'), {
            type: 'bar',
            data: {
                labels: annualLabels,
                datasets: [
                    { label: 'Land Area Growth (LCR)', data: lcrData, backgroundColor: 'rgba(255, 87, 34, 0.8)', borderRadius: 4 },
                    { label: 'Population Growth (PGR)', data: pgrData, backgroundColor: 'rgba(74, 144, 226, 0.8)', borderRadius: 4 }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { 
                    legend: { display: true, position: 'bottom', labels: { usePointStyle: true, boxWidth: 8, font: {size: 10} } },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + context.parsed.y.toFixed(2) + '%';
                            }
                        }
                    }
                },
                scales: { 
                    x: { grid: {display: false} }, 
                    y: { 
                        title: {display: true, text: 'Annual Growth (%)', font: {size: 10}},
                        ticks: {
                            callback: function(value) {
                                return value > 0 ? '+' + value + '%' : value + '%';
                            }
                        }
                    } 
                }
            }
        });
    },


    async submitJob() {
        if (!this.activeCountry) {
            Toast.show('Please select a country first.', 'error');
            return;
        }

        const startYear = document.getElementById('sdg-start-year').value;
        const endYear = document.getElementById('sdg-end-year').value;

        if (!startYear || !endYear || parseInt(startYear) >= parseInt(endYear)) {
            Toast.show('End year must be greater than start year.', 'error');
            return;
        }

        const params = {
            indicator_id: '11.3.1',
            country: this.activeCountry,
            map_year: 2020,
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
            Toast.show(`Submitted job for ${this.activeCountry} (${startYear}-${endYear})`, 'success');
            
            this.closePanel();
            
            if (typeof App !== 'undefined' && App.navigate) {
                App.navigate('jobs');
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

document.addEventListener('DOMContentLoaded', () => SDG.init());
