import { UI } from './ui.js';
import { Charts } from './charts.js';

const Data = {
    async fetchAndVisualizeData(countryName) {
        if (!countryName) return;

        UI.setLoading(countryName);
        try {
            const jobs = await API.listJobs();
            const countryJobs = jobs.filter(j => 
                j.aoi_name === countryName && 
                j.indicator_id === "11.3.1" && 
                j.state === "COMPLETED"
            );
            
            if (countryJobs.length > 0) {
                // Pick the most recent one (API.listJobs returns sorted by submitted_at DESC)
                this.loadJobData(countryJobs[0], countryName, countryJobs);
            } else {
                UI.setEmpty(`No completed analysis found for ${countryName}. Use the "Run New Analysis" button above to generate one.`);
            }
        } catch (error) {
            UI.setEmpty(`Failed to fetch jobs: ${error.message || error}`);
        }
    },

    async pollJobStatus(jobId, country) {
        const interval = setInterval(async () => {
            try {
                const job = await API.get(`/exports/${jobId}`);
                if (job.status === "completed") {
                    clearInterval(interval);
                    this.loadJobData(job, country);
                } else if (job.status === "failed") {
                    clearInterval(interval);
                    UI.setEmpty(`Analysis failed: ${job.error}`);
                }
            } catch (error) {
                console.error("Polling error:", error);
            }
        }, 3000);
    },

    async loadJobData(job, country, countryJobs = null) {
        try {
            const fileId = job.fileId || job.id;
            const downloads = await API.getJobDownloads(fileId);
            
            const getCsvUrl = (suffix) => {
                const match = downloads.find(d => d.name.endsWith(suffix));
                return match ? match.url : null;
            };

            const parseCSV = (suffix) => new Promise((resolve, reject) => {
                const url = getCsvUrl(suffix);
                if (!url) return resolve([]);
                const proxyUrl = `${API.baseUrl}/proxy-csv?url=${encodeURIComponent(url)}`;
                Papa.parse(proxyUrl, {
                    download: true, header: true, dynamicTyping: true, skipEmptyLines: true,
                    transformHeader: h => h.trim(),
                    complete: (results) => resolve(results.data),
                    error: (err) => reject(err)
                });
            });

            const [annualData, areaData, spanData, statsData] = await Promise.all([
                parseCSV("_LCRPGR_annual_10m.csv"),
                parseCSV("_area10m_smoothed.csv"),
                parseCSV("_LCRPGR_5yr_10m.csv"),
                parseCSV("_prediction_stats.csv")
            ]);

            this._renderDashboard(annualData, areaData, spanData, statsData, country, job, countryJobs);
        } catch (error) {
            console.error("loadJobData Error:", error);
            UI.setEmpty(`Failed to load data for ${job.date_range_start}-${job.date_range_end}. Error: ${error.message || error}`);
        }
    },

    _renderDashboard(annualData, areaData, spanData, statsData, country, job, countryJobs) {
        // Filter and deduplicate
        const validAnnual = Array.from(
            new Map(annualData.filter(d => d.window !== undefined && d.LCR !== undefined && d.PGR !== undefined)
            .map(d => [d.window, d])).values()
        );
        
        const validArea = Array.from(
            new Map(areaData.filter(d => d.year !== undefined && d.RF_sm_km2 !== undefined)
            .map(d => [d.year, d])).values()
        );
        
        const validSpans = spanData.filter(d => d.LCRPGR !== undefined);
        const validSpan = validSpans.length > 0 ? validSpans[0] : null;

        let overallRatio = "N/A";
        let verdict = "";
        let verdictColor = "var(--text-muted)";
        let bupc0 = "N/A", bupc1 = "N/A";
        
        if (validSpan && validSpan.LCRPGR !== undefined) {
            const lcrpgr = validSpan.LCRPGR;
            overallRatio = lcrpgr.toFixed(2);
            if (lcrpgr > 1) { verdict = "SPRAWLING"; verdictColor = "#FF5722"; }
            else if (lcrpgr > 0 && lcrpgr <= 1) { verdict = "DENSIFYING"; verdictColor = "#4CAF50"; }
            else { verdict = "DENSIFYING"; verdictColor = "#4CAF50"; }
            
            bupc0 = validSpan.BUpc_t0_m2 ? Math.round(validSpan.BUpc_t0_m2) : "N/A";
            bupc1 = validSpan.BUpc_t1_m2 ? Math.round(validSpan.BUpc_t1_m2) : "N/A";
        }

        let aiAccuracy = "N/A";
        if (statsData && statsData.length > 0 && statsData[0].validation_accuracy) {
            aiAccuracy = (statsData[0].validation_accuracy * 100).toFixed(1) + "%";
        }

        let initialRasterBtnHtml = `<i data-lucide="eye" class="icon sm"></i> Show Raster`;
        let layersVisible = false;
        
        if (job && job.layers && job.layers.length > 0 && window.MapModule && MapModule.dataLayers) {
            const firstLayerId = `job_${job.layers[0].id}`;
            if (MapModule.dataLayers[firstLayerId] && MapModule.map && MapModule.map.hasLayer(MapModule.dataLayers[firstLayerId])) {
                layersVisible = true;
                initialRasterBtnHtml = `<i data-lucide="eye-off" class="icon sm"></i> Hide Raster`;
            }
        }

        let jobActionsHtml = '';
        if (job) {
            const datesStr = `${job.date_range_start}-${job.date_range_end}`;
            jobActionsHtml = `
                <div style="margin-top: 15px; border-top: 1px solid var(--border-color); padding-top: 15px;">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <h4 style="margin:0; font-size: 0.85rem;">Job Actions (${datesStr})</h4>
                        <button id="sdg-toggle-raster-btn" class="action-btn" style="padding: 4px 8px;">
                            ${initialRasterBtnHtml}
                        </button>
                    </div>
                    <div id="sdg-downloads-container" style="margin-top: 10px;"></div>
                </div>
            `;
        }

        let dateStr = "Analysis";
        if (job) {
             dateStr = `${job.date_range_start}-${job.date_range_end} Analysis`;
        } else if (validSpan && validSpan.window) {
             dateStr = `${validSpan.window} Analysis`;
        }

        let jobSelectorHtml = '';
        if (countryJobs && countryJobs.length > 1) {
            jobSelectorHtml = UI.renderJobSelector(countryJobs, job);
        }

        if (UI.heroData) {
            UI.heroData.innerHTML = `
                <div style="background: var(--bg-primary); padding: 15px; border-radius: 8px; border: 1px solid var(--border-color); margin-top: 10px; box-shadow: 0 4px 12px rgba(0,0,0,0.05);">
                    ${jobSelectorHtml}
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
                            <div style="font-size: 0.95rem; font-weight: bold; color: var(--text-primary);">${bupc1} m²</div>
                            <div style="font-size: 0.7rem; color: var(--text-muted);">End Year (${bupc0} Start Year)</div>
                        </div>
                        <div style="background: rgba(0,0,0,0.02); padding: 10px; border-radius: 6px; text-align: center;">
                            <div style="font-size: 0.65rem; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); margin-bottom: 5px;">AI Confidence</div>
                            <div style="font-size: 0.95rem; font-weight: bold; color: var(--text-primary);">${aiAccuracy}</div>
                            <div style="font-size: 0.7rem; color: var(--text-muted);">Validation Acc.</div>
                        </div>
                    </div>
                    
                    <div style="display: flex; flex-direction: column; gap: 15px; margin-top: 15px;">
                        <div style="height: 180px;"><canvas id="hero-area-chart"></canvas></div>
                        <div style="height: 180px;"><canvas id="hero-rate-chart"></canvas></div>
                    </div>
                    
                    ${jobActionsHtml}
                </div>
            `;
            
            // Bind toggle raster button
            const toggleBtn = document.getElementById('sdg-toggle-raster-btn');
            if (toggleBtn) {
                toggleBtn.onclick = () => {
                    layersVisible = !layersVisible;
                    toggleBtn.innerHTML = layersVisible ? '<i data-lucide="eye-off" class="icon sm"></i> Hide Raster' : '<i data-lucide="eye" class="icon sm"></i> Show Raster';
                    if (typeof lucide !== 'undefined') lucide.createIcons();
                    if (job.layers) {
                        job.layers.forEach(l => {
                            if (layersVisible) {
                                if (window.MapModule) {
                                    MapModule.addTileLayer(`job_${l.id}`, l.tile_url, { opacity: 0.7, visible: true });
                                }
                            } else {
                                if (window.MapModule) {
                                    MapModule.toggleLayer(`job_${l.id}`, false);
                                }
                            }
                        });
                        
                        if (window.MapModule && MapModule.addLegend) {
                            if (layersVisible) {
                                MapModule.addLegend('Urban Extent Raster', `Red areas represent classified built-up surfaces for ${dateStr}.`, '#E85C0E');
                            } else {
                                MapModule.removeLegend();
                            }
                        }
                    }
                };
            }
            
            if (countryJobs && countryJobs.length > 1) {
                UI.bindJobSelector(countryJobs, (selectedJob) => {
                    this.loadJobData(selectedJob, country, countryJobs);
                });
            }
            
            // Render downloads
            this._renderJobDownloads(job.fileId || job.id, 'sdg-downloads-container');
            
            if (typeof lucide !== 'undefined') lucide.createIcons();
        }

        Charts.renderCharts(validAnnual, validArea, validSpan);
    },

    async _renderJobDownloads(jobId, containerId) {
        const container = document.getElementById(containerId);
        if (!container) return;
        
        container.innerHTML = '<div style="font-size:0.8rem; color:var(--text-muted);">Loading downloads...</div>';
        try {
            const downloads = await API.getJobDownloads(jobId);
            if (downloads && downloads.length > 0) {
                container.innerHTML = downloads.map(dl => `
                    <div class="download-item" style="padding: 6px; margin-bottom:4px; font-size:0.8rem; background: var(--bg-secondary); border-radius: 4px; display: flex; justify-content: space-between; align-items: center;">
                        <div class="download-info" style="display: flex; align-items: center; gap: 8px; overflow: hidden;">
                            <i data-lucide="file-output" class="icon sm"></i>
                            <span class="download-name" style="text-overflow: ellipsis; white-space: nowrap; overflow: hidden;" title="${dl.name.split('/').pop()}">${dl.name.split('/').pop()}</span>
                        </div>
                        <a href="${dl.url}" target="_blank" class="download-btn" title="Download File" style="padding: 4px;">
                            <i data-lucide="download" class="icon sm"></i>
                        </a>
                    </div>
                `).join('');
                if (typeof lucide !== 'undefined') lucide.createIcons();
            } else {
                container.innerHTML = '<div class="empty-state-sm" style="font-size:0.8rem;">No downloads found.</div>';
            }
        } catch(e) {
            container.innerHTML = `<div class="error-inline" style="font-size:0.8rem;">Error loading downloads</div>`;
        }
    }
};

export { Data };
