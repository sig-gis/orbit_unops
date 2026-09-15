/* ═══════════════════════════════════════════════════════
   COMPASS - ORBIT Operations Center — Jobs Module
   ═══════════════════════════════════════════════════════ */

const Jobs = {
    currentStep: 1,
    selectedAOI: null,
    selectedIndicator: 'SDG_11_3_1',
    _pollInterval: null,
    _jobs: [],
    _indicatorSchemas: {},

    init() {
        this._bindEvents();
        this.startPolling();
    },

    startPolling() {
        if (this._pollInterval) clearInterval(this._pollInterval);
        this._pollInterval = setInterval(async () => {
            const hasActive = this._jobs.some(j => !['COMPLETED', 'FAILED', 'CANCELLED'].includes(j.state));
            if (hasActive) {
                // Background refresh without throwing errors on UI
                try {
                    const latestJobs = await API.listJobs();
                    this._jobs = latestJobs;
                    this._renderJobsTable();
                    this._renderHistoryTable();
                    this._updateBadge();
                } catch (e) {
                    console.warn("Polling failed:", e);
                }
            }
        }, 5000);
    },

    stopPolling() {
        if (this._pollInterval) {
            clearInterval(this._pollInterval);
            this._pollInterval = null;
        }
    },

    _bindEvents() {
        // New Job button
        document.getElementById('btn-new-job')?.addEventListener('click', () => this.openWizard());

        // Modal close
        document.getElementById('job-modal-close')?.addEventListener('click', () => this.closeWizard());

        // Wizard navigation
        document.getElementById('wizard-next')?.addEventListener('click', () => this.nextStep());
        document.getElementById('wizard-prev')?.addEventListener('click', () => this.prevStep());
        document.getElementById('wizard-submit')?.addEventListener('click', () => this.submitJob());

        // Draw AOI from wizard — closes wizard, enables drawing, reopens on completion
        document.getElementById('wizard-draw-aoi')?.addEventListener('click', () => {
            this.closeWizard();
            MapModule.enableDrawing();
            // Set a one-time callback for when drawing completes
            MapModule._wizardCallback = async (aoi) => {
                MapModule._wizardCallback = null;
                // Reopen wizard with the new AOI pre-selected
                this.selectedAOI = aoi;
                this.openWizard();
                Toast.show(`AOI "${aoi.name}" created — now continue your job`, 'success');
            };
        });
        // Job filter
        document.getElementById('jobs-filter-state')?.addEventListener('change', (e) => {
            this.loadJobs({ state: e.target.value });
        });

        // Refresh
        document.getElementById('btn-refresh-jobs')?.addEventListener('click', () => this.loadJobs());

        // Tab Switching
        document.querySelectorAll('.tab-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
                document.querySelectorAll('.tab-content').forEach(c => {
                    c.classList.remove('active');
                    c.style.display = 'none';
                });

                btn.classList.add('active');
                const content = document.getElementById(`tab-${btn.dataset.tab}`);
                if (content) {
                    content.classList.add('active');
                    content.style.display = 'block';
                }

                if (btn.dataset.tab === 'history') this.loadHistory();
            });
        });

        // History Search
        document.getElementById('history-search')?.addEventListener('input', (e) => {
            this._renderHistoryTable(e.target.value);
        });

        // Wizard AOI Search Dropdown Behavior
        const aoiSearch = document.getElementById('wizard-aoi-search');
        const aoiList = document.getElementById('wizard-aoi-list');
        const container = document.getElementById('aoi-search-container');

        aoiSearch?.addEventListener('click', (e) => {
            e.stopPropagation();
            this._filterWizardAOIs(''); // Show all
            if (aoiList) aoiList.style.display = 'block';
            container?.classList.add('open');
            aoiSearch.readOnly = false;
        });

        aoiSearch?.addEventListener('input', (e) => {
            const q = e.target.value.toLowerCase();
            this._filterWizardAOIs(q);
        });

        // Close search results when clicking outside
        document.addEventListener('click', (e) => {
            if (!e.target.closest('#aoi-search-container')) {
                if (aoiList) aoiList.style.display = 'none';
                container?.classList.remove('open');
                if (aoiSearch) aoiSearch.readOnly = true;
            }
        });

        // Set default dates
        const today = new Date();
        const sixMonthsAgo = new Date(today);
        sixMonthsAgo.setMonth(sixMonthsAgo.getMonth() - 6);
        const paramStart = document.getElementById('param-date-start');
        const paramEnd = document.getElementById('param-date-end');
        if (paramStart) paramStart.value = sixMonthsAgo.toISOString().split('T')[0];
        if (paramEnd) paramEnd.value = today.toISOString().split('T')[0];
    },
    async openWizard() {
        this.currentStep = 1;
        this._updateStepUI();
        this._loadAOIOptions();
        await this._loadIndicators();
        document.getElementById('job-modal').style.display = 'flex';
        document.getElementById('job-modal').classList.add('active');
    },

    closeWizard() {
        document.getElementById('job-modal').style.display = 'none';
        document.getElementById('job-modal').classList.remove('active');
    },

    async _loadIndicators() {
        try {
            this._indicatorSchemas = await API.listIndicators();
            this._renderIndicatorCards();
        } catch (err) {
            console.error('Failed to load indicators', err);
            document.getElementById('wizard-indicator-cards').innerHTML = '<p style="color:var(--error)">Failed to load SDK modules.</p>';
        }
    },

    _renderIndicatorCards() {
        const container = document.getElementById('wizard-indicator-cards');
        if (!container) return;

        let html = '';
        for (const [id, schema] of Object.entries(this._indicatorSchemas)) {
            html += `
                <div class="indicator-card ${this.selectedIndicator === id ? 'selected' : ''}" data-indicator="${id}">
                    <div class="indicator-icon"><i data-lucide="${schema.icon || 'activity'}" class="icon"></i></div>
                    <div class="indicator-info">
                        <h4>${schema.name}</h4>
                        <p>${schema.description}</p>
                    </div>
                    <div class="indicator-check"><i data-lucide="check-circle" class="icon"></i></div>
                </div>
            `;
        }
        container.innerHTML = html;
        if (typeof lucide !== 'undefined') lucide.createIcons();

        // Bind clicks dynamically
        container.querySelectorAll('.indicator-card').forEach(card => {
            card.addEventListener('click', () => {
                container.querySelectorAll('.indicator-card').forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                this.selectedIndicator = card.dataset.indicator;
                this._renderDynamicParameters(this.selectedIndicator);
            });
        });

        // Trigger render for the default selection
        if (this.selectedIndicator && this._indicatorSchemas[this.selectedIndicator]) {
            this._renderDynamicParameters(this.selectedIndicator);
        } else {
            // Select first one if default doesn't exist
            const keys = Object.keys(this._indicatorSchemas);
            if (keys.length > 0) {
                this.selectedIndicator = keys[0];
                container.querySelector(`[data-indicator="${this.selectedIndicator}"]`)?.classList.add('selected');
                this._renderDynamicParameters(this.selectedIndicator);
            }
        }
    },

    _renderDynamicParameters(indicatorId) {
        const container = document.getElementById('dynamic-parameters-grid');
        if (!container) return;

        const schema = this._indicatorSchemas[indicatorId];
        if (!schema || !schema.parameters) {
            container.innerHTML = '<p class="loading-inline">No specific parameters required.</p>';
            return;
        }

        let html = '';
        schema.parameters.forEach(param => {
            html += `
                <div class="form-group">
                    <label for="dyn-${param.name}">
                        ${param.label} ${param.required ? '<span style="color:var(--error)">*</span>' : ''}
                    </label>
                    <input type="${param.type === 'number' ? 'number' : 'text'}" 
                           id="dyn-${param.name}" 
                           class="form-input" 
                           value="${param.default !== undefined ? param.default : ''}">
                </div>
            `;
        });
        container.innerHTML = html;
    },

    async _loadAOIOptions() {
        const container = document.getElementById('wizard-aoi-list');
        const searchInput = document.getElementById('wizard-aoi-search');
        try {
            this._cachedAOIs = await API.listAOIs();
            this._renderWizardAOIs(this._cachedAOIs);

            // Auto-select first/current
            if (this._cachedAOIs.length > 0) {
                if (!this.selectedAOI) {
                    this.selectedAOI = this._cachedAOIs[0];
                }
                if (searchInput) searchInput.value = this.selectedAOI.name;
            }
        } catch (err) {
            container.innerHTML = '<p style="color:var(--error)">Failed to load AOIs</p>';
        }
    },

    _renderWizardAOIs(aois) {
        const container = document.getElementById('wizard-aoi-list');
        if (!container) return;

        container.innerHTML = aois.map(aoi => `
            <div class="aoi-search-item" data-aoi-id="${aoi.id}">
                <div class="aoi-info">
                    <h5>${aoi.name}</h5>
                    <span>${aoi.description || 'Custom AOI'}</span>
                </div>
                <div class="area-tag">${aoi.area_km2?.toFixed(0) || '—'} km²</div>
            </div>
        `).join('');

        container.querySelectorAll('.aoi-search-item').forEach(item => {
            item.addEventListener('click', () => {
                const aoi = aois.find(a => a.id === item.dataset.aoiId);
                this.selectedAOI = aoi;
                const searchInput = document.getElementById('wizard-aoi-search');
                if (searchInput) searchInput.value = aoi.name;
                container.style.display = 'none';
                Toast.show(`Selected AOI: ${aoi.name}`, 'info');
            });
        });
    },

    _filterWizardAOIs(query) {
        const filtered = this._cachedAOIs.filter(a =>
            a.name.toLowerCase().includes(query) ||
            (a.description && a.description.toLowerCase().includes(query))
        );
        this._renderWizardAOIs(filtered);
        const container = document.getElementById('wizard-aoi-list');
        if (container) container.style.display = 'block';
    },

    nextStep() {
        if (this.currentStep === 1 && !this.selectedAOI) {
            Toast.show('Please select an Area of Interest', 'warning');
            return;
        }

        if (this.currentStep < 4) {
            this.currentStep++;
            this._updateStepUI();

            if (this.currentStep === 4) {
                this._buildReview();
            }
        }
    },

    prevStep() {
        if (this.currentStep > 1) {
            this.currentStep--;
            this._updateStepUI();
        }
    },

    _updateStepUI() {
        // Update step indicators
        document.querySelectorAll('.step').forEach((step, i) => {
            step.classList.remove('active', 'completed');
            if (i + 1 === this.currentStep) step.classList.add('active');
            if (i + 1 < this.currentStep) step.classList.add('completed');
        });

        // Show/hide wizard steps
        for (let i = 1; i <= 4; i++) {
            const el = document.getElementById(`wizard-step-${i}`);
            if (el) {
                el.style.display = i === this.currentStep ? 'block' : 'none';
                if (i === this.currentStep) el.classList.add('fade-in');
            }
        }

        // Navigation buttons
        document.getElementById('wizard-prev').style.display = this.currentStep > 1 ? 'flex' : 'none';
        document.getElementById('wizard-next').style.display = this.currentStep < 4 ? 'flex' : 'none';
        document.getElementById('wizard-submit').style.display = this.currentStep === 4 ? 'flex' : 'none';
    },

    _getFormParams() {
        const sources = [];
        document.querySelectorAll('#wizard-step-3 input[type="checkbox"]:checked').forEach(cb => {
            if (['SENTINEL_2', 'SENTINEL_1', 'LANDSAT'].includes(cb.value)) sources.push(cb.value);
        });

        const formats = [];
        document.querySelectorAll('#wizard-step-3 input[type="checkbox"]:checked').forEach(cb => {
            if (['GEOTIFF', 'GEOJSON', 'CSV'].includes(cb.value)) formats.push(cb.value);
        });

        const params = {
            aoi_id: this.selectedAOI?.id,
            indicator_id: this.selectedIndicator,
            date_range_start: document.getElementById('param-date-start').value,
            date_range_end: document.getElementById('param-date-end').value,
            resolution_m: parseInt(document.getElementById('param-resolution').value),
            priority: document.getElementById('param-priority').value,
            data_sources: sources.length ? sources : ['SENTINEL_2'],
            export_formats: formats.length ? formats : ['GEOTIFF'],
            client_ref: document.getElementById('param-ref').value || null,
        };

        // Gather dynamic parameters
        const schema = this._indicatorSchemas[this.selectedIndicator];
        if (schema && schema.parameters) {
            schema.parameters.forEach(param => {
                const el = document.getElementById(`dyn-${param.name}`);
                if (el) {
                    let val = el.value;
                    if (param.type === 'number') val = parseFloat(val);
                    if (param.type === 'text' && val) {
                        // Support comma separated strings into arrays if needed by backend
                        if (val.includes(',')) val = val.split(',').map(s => s.trim());
                        else val = [val.trim()]; // API expects list[str] for things like population_sources
                    }
                    params[param.name] = val;
                }
            });
        }

        return params;
    },

    _buildReview() {
        const params = this._getFormParams();

        let reviewHTML = `
            <div class="review-item"><span class="review-item-label">AOI</span><span class="review-item-value">${this.selectedAOI?.name || '—'}</span></div>
            <div class="review-item"><span class="review-item-label">Area</span><span class="review-item-value">${this.selectedAOI?.area_km2?.toFixed(0) || '—'} km²</span></div>
            <div class="review-item"><span class="review-item-label">Indicator</span><span class="review-item-value">${this._indicatorSchemas[params.indicator_id]?.name || params.indicator_id}</span></div>
            <div class="review-item"><span class="review-item-label">Resolution</span><span class="review-item-value">${params.resolution_m}m</span></div>
            <div class="review-item"><span class="review-item-label">Date Range</span><span class="review-item-value">${params.date_range_start} → ${params.date_range_end}</span></div>
            <div class="review-item"><span class="review-item-label">Priority</span><span class="review-item-value">${params.priority}</span></div>
            <div class="review-item"><span class="review-item-label">Sources</span><span class="review-item-value">${params.data_sources.join(', ')}</span></div>
            <div class="review-item"><span class="review-item-label">Formats</span><span class="review-item-value">${params.export_formats.join(', ')}</span></div>
        `;

        // Append dynamic params to review
        const schema = this._indicatorSchemas[params.indicator_id];
        if (schema && schema.parameters) {
            schema.parameters.forEach(param => {
                const val = params[param.name] !== undefined ? params[param.name] : '—';
                reviewHTML += `<div class="review-item"><span class="review-item-label">${param.label}</span><span class="review-item-value" style="color:var(--brand-secondary)">${val}</span></div>`;
            });
        }

        document.getElementById('review-grid').innerHTML = reviewHTML;

        // Estimate cost (rough client-side for display — real estimate comes from server)
        const area = this.selectedAOI?.area_km2 || 100;
        const coeff = { SENTINEL_2: 0.12, SENTINEL_1: 0.15, LANDSAT: 0.08 };
        const resMult = { 10: 1.5, 30: 1.0, 100: 0.5 };
        const priMult = params.priority === 'URGENT' ? 1.5 : 1.0;

        let totalEecu = 0;
        const rows = params.data_sources.map(src => {
            const c = coeff[src] || 0.1;
            const eecu = area * c * (resMult[params.resolution_m] || 1) * priMult;
            totalEecu += eecu;
            return { source: src, eecu, cost: eecu * 0.025 };
        });

        const confidence = area < 1000 ? 'HIGH' : area < 10000 ? 'MEDIUM' : 'LOW';

        document.getElementById('cost-confidence').textContent = confidence;
        document.getElementById('cost-confidence').className = `cost-confidence ${confidence}`;
        document.querySelector('.cost-amount').textContent = `$${(totalEecu * 0.025).toFixed(2)}`;
        document.querySelector('.cost-eecu').textContent = `${totalEecu.toFixed(1)} EECU`;

        document.getElementById('cost-breakdown').innerHTML = rows.map(r => `
            <div class="cost-row">
                <span class="cost-row-source">${r.source}</span>
                <span class="cost-row-value">$${r.cost.toFixed(2)} (${r.eecu.toFixed(1)} EECU)</span>
            </div>
        `).join('');
    },

    async submitJob() {
        const params = this._getFormParams();
        const btn = document.getElementById('wizard-submit');
        btn.disabled = true;
        btn.innerHTML = '<span class="processing-dots"><span></span><span></span><span></span></span> Submitting...';

        try {
            const job = await API.createJob(params);
            Toast.show(`Job submitted — awaiting approval`, 'success');
            this.closeWizard();
            this.loadJobs();

            // Auto-navigate to jobs view
            App.navigate('jobs');
        } catch (err) {
            Toast.show(`Job creation failed: ${err.message}`, 'error');
        } finally {
            btn.disabled = false;
            btn.innerHTML = '<i data-lucide="rocket" class="icon"></i> Submit Job';
            lucide.createIcons();
        }
    },

    async loadJobs(filters = {}) {
        try {
            this._jobs = await API.listJobs(filters);
            this._renderJobsTable();
            this._renderHistoryTable();
            this._updateBadge();
            this._updateAnalytics();
        } catch (err) {
            console.warn('Failed to load jobs:', err);
        }
    },

    async loadHistory() {
        // History is populated from this._jobs which includes all states
        // But we might want to refresh if we haven't loaded yet
        if (this._jobs.length === 0) await this.loadJobs();
        this._renderHistoryTable();
    },

    _renderJobsTable() {
        const tbody = document.getElementById('jobs-table-body');
        const empty = document.getElementById('jobs-empty');
        if (!tbody) return;

        const activeJobs = this._jobs.filter(j =>
            !['COMPLETED', 'FAILED', 'CANCELLED'].includes(j.state)
        );

        if (activeJobs.length === 0) {
            tbody.innerHTML = '';
            if (empty) empty.style.display = 'flex';
            return;
        }
        if (empty) empty.style.display = 'none';

        tbody.innerHTML = activeJobs.map(job => `
            <tr class="fade-in">
                <td class="job-id-cell" title="${job.id}">${job.id.substring(0, 8)}…</td>
                <td>${this._formatCountryName(job)}</td>
                <td><span class="badge" style="background:var(--overlay-bg); color:var(--brand-secondary)">${this._formatIndicatorName(job)}</span></td>
                <td>${this._stateBadge(job)}</td>
                <td>${this._timeAgo(job.submitted_at)}</td>
                <td class="job-actions">${this._actionButtons(job)}</td>
            </tr>
        `).join('');

        // Bind action buttons
        tbody.querySelectorAll('[data-action]').forEach(btn => {
            btn.addEventListener('click', () => this._handleAction(btn.dataset.action, btn.dataset.jobId));
        });

        // Bind rename buttons
        tbody.querySelectorAll('.job-rename-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                this._renameJob(btn.dataset.jobId, btn.dataset.currentLabel);
            });
        });
    },

    _renderHistoryTable(search = '') {
        const tbody = document.getElementById('history-table-body');
        if (!tbody) return;

        let filtered = this._jobs.filter(j =>
            ['COMPLETED', 'FAILED', 'CANCELLED'].includes(j.state)
        );

        if (search) {
            const q = search.toLowerCase();
            filtered = filtered.filter(j =>
                j.id.toLowerCase().includes(q) ||
                (j.client_ref && j.client_ref.toLowerCase().includes(q)) ||
                (j.aoi_name && j.aoi_name.toLowerCase().includes(q)) ||
                j.indicator_id.toLowerCase().includes(q)
            );
        }

        tbody.innerHTML = filtered.map(job => `
            <tr>
                <td class="job-id-cell" title="${job.id}">${job.id.substring(0, 8)}…</td>
                <td>${this._formatCountryName(job)}</td>
                <td><span class="badge" style="background:var(--overlay-bg); color:var(--brand-secondary)">${this._formatIndicatorName(job)}</span></td>
                <td>${this._stateBadge(job)}</td>
                <td>${this._timeAgo(job.submitted_at)}</td>
                <td>
                    ${job.state === 'FAILED' && job.error ?
                `<div style="color:var(--error); font-size: 0.75rem; max-width: 250px; white-space: normal; line-height: 1.2;" title="${this._translateError(job.error)}">${this._translateError(job.error)}</div>`
                : (job.completed_at ? this._timeAgo(job.completed_at) : '—')}
                </td>
                <td class="job-actions">
                    ${job.state === 'COMPLETED' ? ((job.result?.html_chart || job.indicator_id === 'TASKING' || job.result?.viewer_url) ? `
                        <button class="action-btn view nlc-view-btn" data-action="view-nlc" data-job-id="${job.id}" ${job.result?.html_chart ? `data-chart="${job.result.html_chart}"` : ''} ${job.result?.json_report ? `data-json="${job.result.json_report}"` : ''} title="View Results"><i data-lucide="bar-chart-2" class="icon sm"></i> View Results</button>
                    ` : `<button class="action-btn view" data-action="view" data-job-id="${job.id}"><i data-lucide="eye" class="icon sm"></i> View</button>`) : ''}
                    ${job.state === 'FAILED' ? `<button class="action-btn approve" data-action="retry" data-job-id="${job.id}"><i data-lucide="refresh-cw" class="icon sm"></i> Retry</button>` : ''}
                    <button class="action-btn cancel" data-action="delete" data-job-id="${job.id}" title="Delete Job Record"><i data-lucide="trash-2" class="icon sm"></i></button>
                </td>
            </tr>
        `).join('');

        tbody.querySelectorAll('[data-action]').forEach(btn => {
            btn.addEventListener('click', () => this._handleAction(btn.dataset.action, btn.dataset.jobId));
        });

        tbody.querySelectorAll('.job-rename-btn').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                this._renameJob(btn.dataset.jobId, btn.dataset.currentLabel);
            });
        });

        lucide.createIcons();
    },

    _updateBadge() {
        const badge = document.getElementById('active-jobs-badge');
        if (!badge) return;
        const active = this._jobs.filter(j =>
            ['PROCESSING', 'AWAITING_APPROVAL', 'ESTIMATING'].includes(j.state)
        ).length;

        if (active > 0) {
            badge.textContent = active;
            badge.style.display = 'inline';
        } else {
            badge.style.display = 'none';
        }
    },

    _updateAnalytics() {
        const el = (id) => document.getElementById(id);
        const set = (id, value) => { const e = el(id); if (e) e.textContent = value; };
        set('analytics-total-jobs', this._jobs.length);
        set('analytics-completed', this._jobs.filter(j => j.state === 'COMPLETED').length);
        set('analytics-processing', this._jobs.filter(j => j.state === 'PROCESSING').length);
        set('analytics-failed', this._jobs.filter(j => j.state === 'FAILED').length);

        const totalCost = this._jobs.reduce((s, j) => s + (j.estimated_cost_usd || 0), 0);
        set('analytics-cost', `$${totalCost.toFixed(2)}`);
    },

    _stateBadge(job) {
        const state = job.state;
        const cls = {
            PENDING: 'badge-pending',
            ESTIMATING: 'badge-estimating',
            AWAITING_APPROVAL: 'badge-awaiting',
            PROCESSING: 'badge-processing',
            COMPLETED: 'badge-completed',
            FAILED: 'badge-failed',
            CANCELLED: 'badge-cancelled',
        }[state] || '';

        const label = state.replace(/_/g, ' ');

        if (state === 'PROCESSING') {
            const pct = job.progress_pct || 0;
            const pctText = pct > 0 ? `${pct.toFixed(0)}% ` : '';
            return `
                <div class="progress-badge-container">
                    <span class="badge ${cls}">${pctText}Processing</span>
                    <div class="progress-bar-mini">
                        <div class="progress-fill" style="width: ${pct}%"></div>
                    </div>
                </div>
            `;
        }

        return `<span class="badge ${cls}">${label}</span>`;
    },

    _priorityBadge(priority) {
        const cls = priority === 'URGENT' ? 'badge-urgent' : 'badge-standard';
        return `<span class="badge ${cls}">${priority}</span>`;
    },

    _indicatorLabel(id) {
        const names = {
            '11.3.1': 'SDG 11.3.1',
            '15.1.1': 'SDG 15.1.1',
            '6.6.1': 'SDG 6.6.1',
            '15.4.2': 'SDG 15.4.2',
            '15.3.1': 'SDG 15.3.1',
            '11.1.1': 'SDG 11.1.1',
            'SDG_11_3_1': 'SDG 11.3.1'
        };
        return names[id] || `SDG ${id}`;
    },

    _isAdmin() {
        try {
            const user = JSON.parse(localStorage.getItem('orbit_user') || '{}');
            return user.role === 'ADMIN';
        } catch { return false; }
    },

    _labelCell(job) {
        const label = job.client_ref || '';
        const escaped = label.replace(/"/g, '&quot;');
        const display = label || '<span style="color:var(--text-tertiary);font-style:italic">No label</span>';
        const editBtn = this._isAdmin()
            ? `<button class="job-rename-btn" data-job-id="${job.id}" data-current-label="${escaped}" title="Rename"><i data-lucide="pencil" class="icon sm"></i></button>`
            : '';
        return `<span class="job-label-cell">${display} ${editBtn}</span>`;
    },

    async _renameJob(jobId, currentLabel) {
        const newLabel = prompt('Enter a label for this job:', currentLabel || '');
        if (newLabel === null) return; // cancelled
        try {
            await API.renameJob(jobId, newLabel);
            Toast.show(newLabel ? `Job labelled "${newLabel}"` : 'Job label cleared', 'success');
            this.loadJobs();
        } catch (err) {
            Toast.show(`Rename failed: ${err.message}`, 'error');
        }
    },

    _actionButtons(job) {
        const btns = [];
        if (job.state === 'AWAITING_APPROVAL') {
            btns.push(`<button class="action-btn approve" data-action="approve" data-job-id="${job.id}"><i data-lucide="check" class="icon sm"></i> Approve</button>`);
            btns.push(`<button class="action-btn cancel" data-action="cancel" data-job-id="${job.id}"><i data-lucide="x" class="icon sm"></i></button>`);
        }
        if (job.state === 'PROCESSING') {
            btns.push(`<button class="action-btn view" data-action="view" data-job-id="${job.id}"><i data-lucide="eye" class="icon sm"></i> Preview</button>`);
            btns.push(`<button class="action-btn cancel" data-action="cancel" data-job-id="${job.id}"><i data-lucide="x" class="icon sm"></i> Cancel</button>`);
        }
        if (job.state === 'COMPLETED') {
            btns.push(`<button class="action-btn view" data-action="view" data-job-id="${job.id}"><i data-lucide="eye" class="icon sm"></i> View</button>`);
        }
        if (job.state === 'FAILED') {
            btns.push(`<button class="action-btn approve" data-action="retry" data-job-id="${job.id}"><i data-lucide="refresh-cw" class="icon sm"></i></button>`);
        }
        btns.push(`<button class="action-btn cancel" data-action="delete" data-job-id="${job.id}" title="Delete Job Record"><i data-lucide="trash-2" class="icon sm"></i></button>`);
        return btns.join('');
    },

    async _handleAction(action, jobId) {
        try {
            if (action === 'approve') {
                await API.approveJob(jobId);
                Toast.show('Job approved — processing started', 'success');
            } else if (action === 'cancel') {
                await API.cancelJob(jobId);
                Toast.show('Job cancelled', 'warning');
            } else if (action === 'retry') {
                await API.retryJob(jobId);
                Toast.show('Job retry initiated', 'info');
            } else if (action === 'view') {
                const jobSummary = this._jobs.find(j => j.id === jobId);
                if (jobSummary && jobSummary.aoi_name) {
                    const countryName = jobSummary.aoi_name;

                    // Automatically open the SDG panel for this job's indicator
                    if (typeof SDG !== 'undefined' && jobSummary.indicator_id) {
                        if (SDG.activeIndicator !== jobSummary.indicator_id) {
                            const btn = document.querySelector(`.sdg-toggle-btn[data-indicator="${jobSummary.indicator_id}"]`);
                            if (btn) {
                                SDG.toggleIndicator(jobSummary.indicator_id, btn);
                            }
                        }
                    }

                    App.selectCountry(countryName, true);
                    if (typeof MapModule !== 'undefined' && MapModule.countryLayer) {
                        let feature = null;
                        let targetLayer = null;
                        MapModule.countryLayer.eachLayer(l => {
                            if (l.feature.properties.name === countryName) {
                                feature = l.feature;
                                targetLayer = l;
                            }
                        });
                        if (feature) {
                            MapModule.highlightCountry(feature, targetLayer);
                        }
                    }
                }

                App.navigate('map');
                App.navigate('map');
                Toast.show('Loading raster layers...', 'info');
                this._viewJobLayers(jobSummary);
            } else if (action === 'view-nlc') {
                const jobSummary = this._jobs.find(j => j.id === jobId);
                if (jobSummary && jobSummary.result) {
                    this._viewNLCResults(jobSummary);
                }
            } else if (action === 'delete') {
                if (confirm('Are you sure you want to delete this job record? This cannot be undone.')) {
                    await API.deleteJob(jobId);
                    await this.loadJobs();
                    Toast.show('Job record deleted', 'success');
                }
            }
            this.loadJobs();
        } catch (err) {
            Toast.show(`Action failed: ${err.message}`, 'error');
        }
    },



    async _viewNLCResults(job) {
        // Switch to map view
        if (typeof App !== 'undefined' && App.navigate) {
            App.navigate('map');
        }

        // Ensure panels are managed correctly
        const nlcAnalyticsPanel = document.getElementById('nlc-analytics-panel');
        const sdgPanel = document.getElementById('sdg-panel');
        if (sdgPanel) sdgPanel.style.display = 'none';
        const nlcTaskingPanel = document.getElementById('nlc-panel');
        if (nlcTaskingPanel) nlcTaskingPanel.style.display = 'none';

        if (!nlcAnalyticsPanel) return;

        // Clear any previous NLC layers/legends from the map to prevent overlap confusion
        if (typeof MapModule !== 'undefined' && MapModule.map) {
            if (window.nlcRasterLayer) {
                MapModule.map.removeLayer(window.nlcRasterLayer);
                window.nlcRasterLayer = null;
            }
            if (window.nlcLegend) {
                MapModule.map.removeControl(window.nlcLegend);
                window.nlcLegend = null;
            }
        }

        // Zoom to Ireland (or standard AOI)
        if (typeof MapModule !== 'undefined' && MapModule.map) {
            MapModule.map.setView([53.4, -8.0], 7);
        }

        const metricsContainer = document.getElementById('nlc-analytics-metrics');

        // Update titles to include both the specific AOI name AND the feature name
        const aoiName = job.aoi_id || 'Job';
        const titleEl = document.getElementById('nlc-analytics-title');
        const restoreTextEl = document.getElementById('nlc-analytics-restore-text');
        if (titleEl) titleEl.textContent = `${aoiName} - National Land Cover Tasking`;
        if (restoreTextEl) restoreTextEl.textContent = `View ${aoiName} Tasking`;

        nlcAnalyticsPanel.style.display = 'flex';

        try {
            // Check if results are nested
            const results = job.result?.results || job.result;
            if (!results) throw new Error("No results found in job data");

            // Populate recommendations
            const summary = job.result?.summary || {};
            const selection = results.selection || {};

            // Populate Accuracy metrics
            const eeMetrics = results.earth_engine || {};
            const acc = eeMetrics.accuracy ? (eeMetrics.accuracy * 100).toFixed(1) : '0';
            const auc = eeMetrics.auc ? (eeMetrics.auc).toFixed(3) : '0';
            
            document.getElementById('nlc-analytics-acc').textContent = `${acc}%`;
            document.getElementById('nlc-analytics-kappa').textContent = auc; // Re-using Kappa box for AUC for now

            // Render ROC Chart
            const rocCtx = document.getElementById('nlc-roc-chart');
            if (rocCtx && eeMetrics.roc) {
                // Destroy old chart if exists
                if (window.nlcRocChart) {
                    window.nlcRocChart.destroy();
                }
                
                const fpr = eeMetrics.roc.false_positive_rate || [];
                const tpr = eeMetrics.roc.true_positive_rate || [];
                
                // Construct points
                const points = fpr.map((x, i) => ({ x: x, y: tpr[i] }));
                
                // Find the closest threshold index to plot a point
                const thresholds = eeMetrics.roc.threshold || eeMetrics.roc.thresholds || [];
                const targetThreshold = eeMetrics.threshold || 0.5;
                let closestIdx = 0;
                let minDiff = Infinity;
                thresholds.forEach((t, i) => {
                    const diff = Math.abs(t - targetThreshold);
                    if (diff < minDiff) { minDiff = diff; closestIdx = i; }
                });
                
                const thresholdPoint = (fpr.length > 0 && tpr.length > 0) 
                    ? { x: fpr[closestIdx], y: tpr[closestIdx] } 
                    : null;
                    
                const datasets = [];
                
                if (thresholdPoint) {
                    datasets.push({
                        type: 'scatter',
                        label: 'Selected Threshold',
                        data: [thresholdPoint],
                        backgroundColor: '#E85C0E',
                        pointRadius: 6,
                        pointHoverRadius: 8,
                        borderColor: '#fff',
                        borderWidth: 2,
                        z: 10
                    });
                }
                
                datasets.push(
                    {
                        label: 'Earth Engine ROC',
                        data: points,
                        borderColor: '#0284c7', // brand-primary
                        backgroundColor: 'rgba(2, 132, 199, 0.1)',
                        fill: true,
                        tension: 0.4,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        order: 2
                    },
                    {
                        label: 'Random Classifier',
                        data: [{x: 0, y: 0}, {x: 1, y: 1}],
                        borderColor: '#94a3b8',
                        borderDash: [5, 5],
                        fill: false,
                        pointRadius: 0,
                        borderWidth: 1,
                        order: 3
                    }
                );

                window.nlcRocChart = new Chart(rocCtx, {
                    type: 'line',
                    data: { datasets: datasets },
                    options: {
                        responsive: true,
                        maintainAspectRatio: false,
                        interaction: {
                            intersect: false,
                            mode: 'index',
                        },
                        plugins: {
                            legend: {
                                display: false
                            },
                            tooltip: {
                                callbacks: {
                                    label: (ctx) => `TPR: ${ctx.parsed.y.toFixed(2)}, FPR: ${ctx.parsed.x.toFixed(2)}`
                                }
                            }
                        },
                        scales: {
                            x: {
                                type: 'linear',
                                title: { display: true, text: 'False Positive Rate' },
                                min: 0, max: 1
                            },
                            y: {
                                title: { display: true, text: 'True Positive Rate' },
                                min: 0, max: 1
                            }
                        }
                    }
                });
            }

            // Load GeoTIFF and calculate Total Area using GeoRaster
            document.getElementById('nlc-analytics-area').textContent = 'Loading...';
            
            try {
                if (typeof parseGeoraster !== 'undefined' && typeof GeoRasterLayer !== 'undefined') {
                    // Using the backend proxy to bypass CORS
                    const rawUrl = "https://storage.googleapis.com/unops/orbit-lc/_200m/ireland_national_pred_200m.tif";
                    const url_to_geotiff_file = `${API.baseUrl}/proxy-csv?url=${encodeURIComponent(rawUrl)}`;
                    const response = await fetch(url_to_geotiff_file);
                    const arrayBuffer = await response.arrayBuffer();
                    const georaster = await parseGeoraster(arrayBuffer);
                    
                    // Add dynamic legend
                    if (window.nlcLegend) {
                        MapModule.map.removeControl(window.nlcLegend);
                    }
                    const legend = L.control({ position: 'bottomright' });
                    legend.onAdd = function () {
                        const div = L.DomUtil.create('div', 'info legend');
                        div.style.background = 'var(--bg-primary)';
                        div.style.padding = '10px 15px';
                        div.style.borderRadius = '8px';
                        div.style.boxShadow = '0 4px 12px rgba(0,0,0,0.15)';
                        div.style.border = '1px solid var(--border-color)';
                        div.style.color = 'var(--text-main)';
                        div.style.fontFamily = 'var(--font-family)';
                        div.innerHTML = `
                            <div style="font-weight: 600; font-size: 0.85rem; margin-bottom: 8px;">Classification</div>
                            <div style="display: flex; align-items: center; gap: 8px;">
                                <div style="width: 16px; height: 16px; background: #8B4513; border-radius: 4px; border: 1px solid rgba(0,0,0,0.2);"></div>
                                <span style="font-size: 0.85rem; color: var(--text-main);">Peat Soil</span>
                            </div>
                        `;
                        return div;
                    };
                    legend.addTo(MapModule.map);
                    window.nlcLegend = legend;

                    const layer = new GeoRasterLayer({
                        georaster: georaster,
                        opacity: 0.85,
                        pixelValuesToColorFn: values => {
                            const loi = values[0];
                            if (loi >= 30.0) {
                                return '#8B4513';
                            }
                            return null;
                        },
                        resolution: 128
                    });
                    
                    if (window.nlcRasterLayer) {
                        MapModule.map.removeLayer(window.nlcRasterLayer);
                    }
                    window.nlcRasterLayer = layer;
                    layer.addTo(MapModule.map);
                    
                    // Iterate the raw data array once to calculate total area
                    let totalPeatPixels = 0;
                    const data = georaster.values[0]; // Band 1
                    for (let y = 0; y < georaster.height; y++) {
                        for (let x = 0; x < georaster.width; x++) {
                            const val = data[y][x];
                            // Check for valid data and threshold (>= 30.0)
                            if (val !== georaster.noDataValue && val >= 30.0) {
                                totalPeatPixels++;
                            }
                        }
                    }
                    
                    // 200m x 200m pixels = 40,000 sq meters = 0.04 sq km per pixel
                    const areaSqKm = totalPeatPixels * 0.04; 
                    document.getElementById('nlc-analytics-area').textContent = areaSqKm.toLocaleString(undefined, { maximumFractionDigits: 0 });
                    
                    MapModule.map.fitBounds(layer.getBounds());
                } else {
                    document.getElementById('nlc-analytics-area').textContent = 'Ext missing';
                }
            } catch (err) {
                console.error("GeoRaster error:", err);
                document.getElementById('nlc-analytics-area').textContent = 'Error';
            }

        } catch (error) {
            console.error("Error rendering NLC results:", error);
            if (typeof Toast !== 'undefined') Toast.show('Error parsing classification results', 'error');
        }


        // Hook up panel buttons if not already hooked
        if (!nlcAnalyticsPanel.dataset.hooked) {
            document.getElementById('nlc-analytics-panel-close').onclick = () => {
                nlcAnalyticsPanel.style.display = 'none';
                if (window.nlcRasterLayer && typeof MapModule !== 'undefined') {
                    MapModule.map.removeLayer(window.nlcRasterLayer);
                    window.nlcRasterLayer = null;
                }
                if (window.nlcLegend && typeof MapModule !== 'undefined') {
                    MapModule.map.removeControl(window.nlcLegend);
                    window.nlcLegend = null;
                }
            };
            document.getElementById('nlc-analytics-panel-minimize').onclick = () => {
                nlcAnalyticsPanel.style.display = 'none';
                document.getElementById('nlc-analytics-panel-restore').style.display = 'flex';
            };
            document.getElementById('nlc-analytics-panel-restore').onclick = () => {
                document.getElementById('nlc-analytics-panel-restore').style.display = 'none';
                nlcAnalyticsPanel.style.display = 'flex';
            };
            
            const btnDownloadRaster = document.getElementById('btn-download-nlc-raster');
            if (btnDownloadRaster) {
                btnDownloadRaster.onclick = () => {
                    const rawUrl = "https://storage.googleapis.com/unops/orbit-lc/_200m/ireland_national_pred_200m.tif";
                    // Attempt to download the file directly
                    const link = document.createElement('a');
                    link.href = rawUrl;
                    link.target = '_blank';
                    link.download = 'ireland_national_pred_200m.tif';
                    document.body.appendChild(link);
                    link.click();
                    document.body.removeChild(link);
                };
            }
            
            // Ground Truth Toggle
            document.getElementById('nlc-toggle-ground-truth').onchange = (e) => {
                if (typeof MapModule !== 'undefined' && window.nlcGroundTruthLayer) {
                    if (e.target.checked) {
                        window.nlcGroundTruthLayer.addTo(MapModule.map);
                        if (window.nlcGroundTruthLegend) window.nlcGroundTruthLegend.addTo(MapModule.map);
                    } else {
                        MapModule.map.removeLayer(window.nlcGroundTruthLayer);
                        if (window.nlcGroundTruthLegend) MapModule.map.removeControl(window.nlcGroundTruthLegend);
                    }
                }
            };
            
            nlcAnalyticsPanel.dataset.hooked = 'true';
        }
        
        // Reset and manage Ground Truth toggle state for this viewing session
        const toggleContainer = document.getElementById('nlc-ground-truth-toggle-container');
        const toggleInput = document.getElementById('nlc-toggle-ground-truth');
        if (window.nlcGroundTruthLayer && typeof MapModule !== 'undefined') {
            toggleContainer.style.display = 'flex';
            toggleInput.checked = false; // Hide by default when raster is viewed
            MapModule.map.removeLayer(window.nlcGroundTruthLayer);
            if (window.nlcGroundTruthLegend) MapModule.map.removeControl(window.nlcGroundTruthLegend);
        } else {
            toggleContainer.style.display = 'none';
        }
    },


    async _viewJobLayers(jobRef) {
        // Switch to map view
        App.navigate('map');

        // Auto-select the AOI in the header to update stats and zoom
        if (jobRef.aoi_id) {
            App.selectCountry(jobRef.aoi_id);
        }
        
        let job = jobRef;
        try {
            job = await API.getJob(jobRef.id);
        } catch(e) {
            console.error("Failed to fetch full job info", e);
        }

        if (job.layers?.length) {
            job.layers.forEach(layer => {
                if (layer.tile_url) {
                    MapModule.addTileLayer(`job_${layer.id}`, layer.tile_url, {
                        opacity: 0.7,
                        visible: true,
                    });
                }
            });
            Toast.show(`Loaded ${job.layers.length} layers from job`, 'success');

            if (typeof MapModule !== 'undefined' && MapModule.addLegend) {
                const dateStr = (job.date_range_start && job.date_range_end)
                    ? `${job.date_range_start}-${job.date_range_end} Analysis`
                    : 'Analysis';
                MapModule.addLegend('Urban Extent Raster', `Red areas represent classified built-up surfaces for ${dateStr}.`, '#E85C0E');
            }
        }
    },

    _formatIndicatorName(job) {
        if (job.indicator_id === 'NLC') return 'National Land Cover';
        if (job.indicator_id === 'TASKING') return 'Space for Time Tasking';
        // Fallback for older jobs before the fix
        if (!job.indicator_id && (job.result?.country === 'pc655-gcpa-unops-geo-is' || job.request?.classifier_type)) return 'National Land Cover';
        if (!job.indicator_id) return 'Unknown Indicator';
        const config = window.ORBIT_CONFIG?.INDICATORS?.[job.indicator_id];
        return config ? config.name : `SDG ${job.indicator_id}`;
    },

    _formatCountryName(job) {
        let name = job.aoi_name || job.country || job.aoi_id;
        // Fix for old NLC jobs that used the google cloud project id as the country name
        if (name === 'pc655-gcpa-unops-geo-is' && (!job.indicator_id || job.indicator_id === 'NLC')) {
            return 'Demo (Ireland)';
        }
        return name || 'Global';
    },

    _timeAgo(dateStr) {
        if (!dateStr) return '—';
        const date = new Date(dateStr);
        const now = new Date();
        const diff = Math.floor((now - date) / 1000);

        if (diff < 60) return 'Just now';
        if (diff < 3600) return `${Math.floor(diff / 60)}m ago`;
        if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`;
        return date.toLocaleDateString();
    },

    startPolling() {
        this._pollInterval = setInterval(() => {
            if (Auth.isAuthenticated()) this.loadJobs();
        }, 5000);
    },

    stopPolling() {
        if (this._pollInterval) clearInterval(this._pollInterval);
    },

    _translateError(errorMsg) {
        if (!errorMsg) return "Unknown error occurred";
        const msg = errorMsg.toLowerCase();

        if (msg.includes("empty") || msg.includes("no data") || msg.includes("does not contain all the bands")) {
            return "Data is not available for this country during the selected time period. Please try a different year span.";
        }
        if (msg.includes("memory limit") || msg.includes("maxpixels") || msg.includes("user memory limit exceeded") || msg.includes("computation timed out")) {
            return "This country is too large to process over this many years. Please try selecting a shorter year span (e.g., 4 years).";
        }
        return errorMsg;
    },

    _fmtSize(bytes) {
        if (!bytes) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
};
