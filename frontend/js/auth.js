/* ═══════════════════════════════════════════════════════
   COMPASS - ORBIT Operations Center — Auth Module
   ═══════════════════════════════════════════════════════ */

const Auth = {
    user: { id: "mock-admin", name: "UNOPS Admin", role: "ADMIN", email: "admin@unops.org" },
    token: "mock-token-for-poc",

    async init() {
        this._bindEvents();
        
        // Simulating immediate successful auth for POC
        localStorage.setItem('orbit_token', this.token);
        localStorage.setItem('orbit_user', JSON.stringify(this.user));
        this._onLoginSuccess();
        return true;
    },

    isAuthenticated() {
        return true; // Always authenticated for POC
    },

    getRole() {
        return this.user.role;
    },

    _bindEvents() {
        document.getElementById('login-submit')?.addEventListener('click', () => this._handleLogin());
        document.getElementById('login-password')?.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') this._handleLogin();
        });
        document.querySelectorAll('.role-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                this._handleLogin();
            });
        });
    },

    async _handleLogin() {
        // UI simulation of login process
        const btn = document.getElementById('login-submit');
        if (btn) {
            btn.disabled = true;
            btn.innerHTML = '<span class="processing-dots"><span></span><span></span><span></span></span> Signing in...';
        }

        setTimeout(() => {
            this._onLoginSuccess();
            if (typeof App !== 'undefined' && !App.initialized) {
                App._postAuth();
            }
            Toast.show(`Welcome, ${this.user.name}`, 'success');
            
            if (btn) {
                btn.disabled = false;
                btn.innerHTML = '<i data-lucide="log-in" class="icon"></i> Sign In';
                if (typeof lucide !== 'undefined') lucide.createIcons();
            }
        }, 500);
    },

    _onLoginSuccess() {
        const loginModal = document.getElementById('login-modal');
        if (loginModal) {
            loginModal.classList.remove('active');
            loginModal.style.display = 'none';
        }

        const nameEl = document.getElementById('user-name');
        const roleEl = document.getElementById('user-role');
        if (nameEl) nameEl.textContent = this.user.name;
        if (roleEl) roleEl.textContent = this.user.role;
    },

    logout(reload = true) {
        // Logout is disabled in POC
        Toast.show("Logout is disabled in the Proof of Concept.", "info");
    }
};
