/* ═══════════════════════════════════════════════════════
   COMPASS - ORBIT Operations Center — Application Controller
   ═══════════════════════════════════════════════════════ */

const Toast = {
    show(message, type = 'info', duration = 4000) {
        const container = document.getElementById('toast-container');
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        toast.innerHTML = `<span>${message}</span>`;
        container.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = '0';
            toast.style.transform = 'translateX(20px)';
            setTimeout(() => toast.remove(), 300);
        }, duration);
    }
};

const App = {
    currentView: 'map',
    currentCountry: null,
    initialized: false,

    async init() {
        // Theme
        this._initTheme();

        // Sidebar toggle
        document.getElementById('sidebar-toggle')?.addEventListener('click', () => {
            document.getElementById('sidebar').classList.toggle('collapsed');
            setTimeout(() => MapModule.invalidateSize(), 350);
        });

        // Mobile menu
        document.getElementById('mobile-menu-btn')?.addEventListener('click', () => {
            document.getElementById('sidebar').classList.toggle('open');
        });

        // Navigation
        document.querySelectorAll('.nav-item[data-view]').forEach(item => {
            item.addEventListener('click', (e) => {
                e.preventDefault();
                this.navigate(item.dataset.view);
            });
        });

        // Country selector
        document.getElementById('country-select')?.addEventListener('change', (e) => {
            const countryName = e.target.value;
            if (countryName) {
                this.selectCountry(countryName, false);
                // Highlight on map
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
        });

        // System Menu Toggle
        const systemMenuBtn = document.getElementById('system-menu-btn');
        const systemMenuContainer = systemMenuBtn?.parentElement;
        systemMenuBtn?.addEventListener('click', (e) => {
            e.stopPropagation();
            systemMenuContainer.classList.toggle('open');
        });

        // Close dropdown on outside click
        document.addEventListener('click', () => {
            systemMenuContainer?.classList.remove('open');
        });

        // Dropdown Items
        document.querySelectorAll('.dropdown-item').forEach(item => {
            item.addEventListener('click', (e) => {
                const view = item.dataset.view || item.id?.replace('nav-', '');
                if (view && ['map', 'jobs', 'analytics', 'logs'].includes(view)) {
                    e.preventDefault();
                    this.navigate(view);
                }
                systemMenuContainer?.classList.remove('open');
            });
        });

        // Logout
        document.getElementById('btn-logout')?.addEventListener('click', () => Auth.logout());

        // Initialize Lucide icons
        lucide.createIcons();

        // Initialize map
        MapModule.init();

        // Initialize jobs module
        Jobs.init();


        // Initialize tooltips
        this.initTooltips();

        // Auth check - Wait for backend verification
        const isAuth = await Auth.init();
        if (isAuth) {
            await this._postAuth();
        } else {
            // Show login modal
            const modal = document.getElementById('login-modal');
            if (modal) {
                modal.classList.add('active');
                modal.style.display = 'flex';
            }
        }

        // Check health
        this._checkHealth();
    },

    async _postAuth() {
        if (this.initialized) return;
        this.initialized = true;

        console.log('🏁 Initializing application data...');
        try {
            // AOI loading removed for Phase 2
            Jobs.loadJobs();
            Jobs.startPolling();

            lucide.createIcons();
        } catch (err) {
            console.error('Initialization error:', err);
        }
    },

    /**
     * Tooltip Engine: Injects tooltips at body level to prevent clipping
     */
    initTooltips() {
        const tooltipEl = document.createElement('div');
        tooltipEl.id = 'global-tooltip';
        tooltipEl.className = 'global-tooltip';
        document.body.appendChild(tooltipEl);

        document.addEventListener('mouseover', (e) => {
            const target = e.target.closest('.info-tooltip');
            if (target) {
                const text = target.getAttribute('title') || target.dataset.title;
                if (!text) return;
                
                // Store original title to prevent native tooltip
                if (target.hasAttribute('title')) {
                    target.dataset.title = target.getAttribute('title');
                    target.removeAttribute('title');
                }

                tooltipEl.textContent = target.dataset.title;
                tooltipEl.classList.add('visible');

                const rect = target.getBoundingClientRect();
                tooltipEl.style.left = `${rect.left + rect.width / 2}px`;
                tooltipEl.style.top = `${rect.top - 10}px`;
            }
        });

        document.addEventListener('mouseout', (e) => {
            if (e.target.closest('.info-tooltip')) {
                tooltipEl.classList.remove('visible');
            }
        });
    },

    navigate(view) {
        this.currentView = view;

        // Update nav (sidebar)
        document.querySelectorAll('.nav-item[data-view]').forEach(item => {
            item.classList.toggle('active', item.dataset.view === view);
        });
        
        // Update nav (dropdown)
        document.querySelectorAll('.dropdown-item[data-view]').forEach(item => {
            item.classList.toggle('active', item.dataset.view === view);
        });

        // Show/hide views
        document.querySelectorAll('.view').forEach(v => v.style.display = 'none');
        const viewEl = document.getElementById(`view-${view}`);
        if (viewEl) {
            viewEl.style.display = 'block';
            viewEl.classList.add('active');
        }

        // Title update
        const titles = {
            map: 'Operations Center',
            jobs: 'Job Management',
            analytics: 'Admin Panel',
            logs: 'System Logs'
        };
        const icons = {
            map: 'satellite',
            jobs: 'activity',
            analytics: 'shield',
            logs: 'terminal'
        };
        document.getElementById('header-title').innerHTML = `
            <i data-lucide="${icons[view] || 'satellite'}" class="icon header-icon"></i>
            <span>${titles[view] || 'Operations Center'}</span>
        `;
        lucide.createIcons();

        // Map needs resize
        if (view === 'map') MapModule.invalidateSize();

        // Load data per view
        if (view === 'jobs') {
            Jobs.loadJobs();
            Jobs.loadHistory();
        }
        if (view === 'analytics' && typeof Admin !== 'undefined') {
            Admin.load();
        }
    },

    onCountriesLoaded(data) {
        const datalist = document.getElementById('country-list');
        if (!datalist) return;

        // Sort countries alphabetically
        const features = data.features.sort((a, b) => {
            const nameA = a.properties.name || '';
            const nameB = b.properties.name || '';
            return nameA.localeCompare(nameB);
        });

        datalist.innerHTML = '';
        features.forEach(f => {
            if (f.properties.name) {
                datalist.innerHTML += `<option value="${f.properties.name}"></option>`;
            }
        });
    },

    selectCountry(countryName, updateDropdown = true) {
        this.currentCountry = countryName;
        
        // Update selector
        if (updateDropdown) {
            const select = document.getElementById('country-select');
            if (select) select.value = countryName;
        }

        if (typeof SDG !== 'undefined' && SDG.onCountrySelected) {
            SDG.onCountrySelected(countryName);
        }
    },

    _initTheme() {
        const saved = localStorage.getItem('orbit_theme') || 'light';
        document.documentElement.dataset.theme = saved;

        document.getElementById('theme-toggle')?.addEventListener('click', () => {
            const current = document.documentElement.dataset.theme;
            const next = current === 'dark' ? 'light' : 'dark';
            document.documentElement.dataset.theme = next;
            localStorage.setItem('orbit_theme', next);
            MapModule.switchTheme(next);
        });
    },

    async _checkHealth() {
        try {
            await API.health();
            const dot = document.querySelector('#gee-status-footer .status-dot');
            const label = document.querySelector('#gee-status-footer .status-label');
            if (dot) {
                dot.classList.remove('offline');
                dot.classList.add('online');
            }
            if (label) label.textContent = 'Google Earth Engine';
        } catch (err) {
            const dot = document.querySelector('#gee-status-footer .status-dot');
            const label = document.querySelector('#gee-status-footer .status-label');
            if (dot) {
                dot.classList.remove('online');
                dot.classList.add('offline');
            }
            if (label) label.textContent = 'GEE Connection Error';
        }
    }
};

// ── Boot ──
document.addEventListener('DOMContentLoaded', () => App.init());
