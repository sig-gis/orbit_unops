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
                <td>${job.aoi_name || job.country || '—'}</td>
                <td>${this._indicatorLabel(job.indicator_id)}</td>
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
                <td>${job.aoi_name || job.country || '—'}</td>
                <td>${this._indicatorLabel(job.indicator_id)}</td>
                <td>${this._stateBadge(job)}</td>
                <td>${this._timeAgo(job.submitted_at)}</td>
                <td>${job.completed_at ? this._timeAgo(job.completed_at) : '—'}</td>
                <td class="job-actions">
                    ${job.state === 'COMPLETED' ? `<button class="action-btn view" data-action="view" data-job-id="${job.id}"><i data-lucide="eye" class="icon sm"></i> View</button>` : ''}
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
            SDG_11_3_1: 'Urban Expansion',
            DAMAGE_MAP_SAR: 'Damage Map',
            FLOOD_EXTENT: 'Flood Extent',
            VEGETATION_HEALTH: 'Vegetation',
            PROTOTYPE_MODEL: 'Prototype',
        };
        return names[id] || id;
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
                
                // Show loading overlay
                const loader = document.createElement('div');
                loader.id = 'map-job-loader';
                loader.innerHTML = `
                    <div style="position:absolute; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.6); z-index:9999; display:flex; flex-direction:column; align-items:center; justify-content:center; color:white;">
                        <span class="processing-dots" style="transform:scale(1.5);"><span></span><span></span><span></span></span>
                        <div style="margin-top:20px; font-weight:bold; font-size:1.1rem; letter-spacing:1px;">Loading High-Resolution Raster Data...</div>
                    </div>
                `;
                document.getElementById('view-map').appendChild(loader);
                
                const job = await API.getJob(jobId);
                this._viewJobLayers(job);
                

                // Remove loading overlay
                if (document.getElementById('map-job-loader')) {
                    document.getElementById('map-job-loader').remove();
                }
            } else if (action === 'delete') {
                if (confirm('Are you sure you want to delete this job record? This cannot be undone.')) {
                    await API.deleteJob(jobId);
                    Toast.show('Job record deleted', 'success');
                }
            }
            this.loadJobs();
        } catch (err) {
            Toast.show(`Action failed: ${err.message}`, 'error');
        }
    },



    _viewJobLayers(job) {
        // Switch to map view
        App.navigate('map');

        // Auto-select the AOI in the header to update stats and zoom
        if (job.aoi_id) {
            App.selectCountry(job.aoi_id);
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
                MapModule.addLegend('Urban Extent Raster', 'Red areas represent classified built-up surfaces.', '#E85C0E');
            }
        }
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

    _fmtSize(bytes) {
        if (!bytes) return '0 B';
        const k = 1024;
        const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
    }
};
