/* ═══════════════════════════════════════════════════════
   COMPASS - ORBIT Operations Center — Admin Module
   User Management · AOI Management · System Stats
   ═══════════════════════════════════════════════════════ */

const Admin = {
    _loaded: false,

    async load() {
        try {
            await Promise.all([
                this._loadUsers(),
                this._loadAOIs(),
                this._loadJobStats(),
            ]);
            this._loaded = true;
            lucide.createIcons();
        } catch (err) {
            console.warn('Admin load failed:', err);
        }
    },

    async _loadJobStats() {
        try {
            const jobs = await API.listJobs();
            const total = jobs.length;
            const completed = jobs.filter(j => j.state === 'COMPLETED').length;
            const el1 = document.getElementById('analytics-total-jobs');
            const el2 = document.getElementById('analytics-completed');
            if (el1) el1.textContent = total;
            if (el2) el2.textContent = completed;
        } catch (e) { /* ignore */ }
    },

    _getInitials(name) {
        if (!name) return '?';
        return name.split(' ').map(w => w[0]).join('').toUpperCase().slice(0, 2);
    },

    _timeAgo(dateStr) {
        if (!dateStr) return '—';
        const d = new Date(dateStr);
        const now = new Date();
        const diffMs = now - d;
        const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));
        if (diffDays === 0) return 'Today';
        if (diffDays === 1) return 'Yesterday';
        if (diffDays < 30) return `${diffDays}d ago`;
        if (diffDays < 365) return `${Math.floor(diffDays / 30)}mo ago`;
        return d.toLocaleDateString();
    },

    async _loadUsers() {
        try {
            const users = await API.adminListUsers();
            const el = document.getElementById('analytics-users');
            if (el) el.textContent = users.length;

            const tbody = document.getElementById('admin-users-body');
            if (!tbody) return;

            tbody.innerHTML = users.map(u => {
                const initials = this._getInitials(u.name);
                const roleBadge = u.role === 'ADMIN'
                    ? '<span class="badge badge-admin">Admin</span>'
                    : '<span class="badge badge-analyst">Analyst</span>';
                const lastLogin = u.last_login
                    ? this._timeAgo(u.last_login)
                    : '<span class="text-muted">Never</span>';
                const created = u.created_at
                    ? this._timeAgo(u.created_at)
                    : '—';
                const jobClass = u.job_count > 0 ? 'stat-num' : 'stat-num-zero';

                return `<tr>
                    <td>
                        <div class="user-cell">
                            <div class="user-avatar-sm">${initials}</div>
                            <div>
                                <div class="user-name-primary">${u.name}</div>
                                <div class="user-email-sub">${u.email}</div>
                            </div>
                        </div>
                    </td>
                    <td>${roleBadge}</td>
                    <td><span class="${jobClass}">${u.job_count}</span></td>
                    <td>${lastLogin}</td>
                    <td>${created}</td>
                </tr>`;
            }).join('');
        } catch (err) {
            console.warn('Failed to load users:', err);
        }
    },

    async _loadAOIs() {
        try {
            const aois = await API.adminListAOIs();

            const activeCount = aois.filter(a => !a.archived).length;
            const el = document.getElementById('analytics-aois');
            if (el) el.textContent = activeCount;

            const tbody = document.getElementById('admin-aois-body');
            if (!tbody) return;

            tbody.innerHTML = aois.map(a => {
                const isArchived = a.archived;
                const dotClass = isArchived ? 'aoi-dot-archived' : 'aoi-dot-active';
                const statusBadge = isArchived
                    ? '<span class="badge badge-archived">Archived</span>'
                    : '<span class="badge badge-active">Active</span>';
                const actionBtn = isArchived
                    ? `<button class="btn-sm btn-restore" onclick="Admin.restoreAOI('${a.id}')"><i data-lucide="rotate-ccw" class="icon"></i> Restore</button>`
                    : `<button class="btn-sm btn-archive" onclick="Admin.archiveAOI('${a.id}')"><i data-lucide="archive" class="icon"></i> Archive</button>`;
                const created = a.created_at
                    ? this._timeAgo(a.created_at)
                    : '—';
                const area = a.area_km2 ? `<span class="area-value">${a.area_km2.toFixed(0)}</span><span class="area-unit">km²</span>` : '—';
                const jobClass = a.job_count > 0 ? 'stat-num' : 'stat-num-zero';

                const escapedName = a.name.replace(/"/g, '&quot;');
                const renameBtn = `<button class="btn-sm btn-rename" onclick="Admin.renameAOI('${a.id}', '${escapedName}')"><i data-lucide="pencil" class="icon"></i> Rename</button>`;

                return `<tr class="${isArchived ? 'row-archived' : ''}">
                    <td>
                        <div class="aoi-name-cell">
                            <span class="aoi-dot ${dotClass}"></span>
                            <strong>${a.name}</strong>
                        </div>
                    </td>
                    <td>${area}</td>
                    <td><span class="${jobClass}">${a.job_count}</span></td>
                    <td>${statusBadge}</td>
                    <td>${created}</td>
                    <td class="aoi-actions">${renameBtn} ${actionBtn}</td>
                </tr>`;
            }).join('');
        } catch (err) {
            console.warn('Failed to load AOIs:', err);
        }
    },

    async archiveAOI(aoiId) {
        try {
            const res = await API.adminArchiveAOI(aoiId);
            console.log('Archive result:', res);
            if (typeof Toast !== 'undefined') Toast.show(`AOI "${res.name}" archived`, 'success');
            await this.load();
        } catch (err) {
            console.error('Archive failed:', err);
            if (typeof Toast !== 'undefined') Toast.show(`Archive failed: ${err.message}`, 'error');
        }
    },

    async restoreAOI(aoiId) {
        try {
            const res = await API.adminRestoreAOI(aoiId);
            console.log('Restore result:', res);
            if (typeof Toast !== 'undefined') Toast.show(`AOI "${res.name}" restored`, 'success');
            await this.load();
        } catch (err) {
            console.error('Restore failed:', err);
            if (typeof Toast !== 'undefined') Toast.show(`Restore failed: ${err.message}`, 'error');
        }
    },

    async renameAOI(aoiId, currentName) {
        const newName = prompt('Rename this Area of Interest:', currentName || '');
        if (newName === null || !newName.trim()) return;
        try {
            const res = await API.adminRenameAOI(aoiId, newName.trim());
            Toast.show(`AOI renamed: "${res.old_name}" → "${res.name}"`, 'success');
            await this.load();
            // Refresh the AOI selector dropdown so the new name appears everywhere
            await App.loadAOIs();
        } catch (err) {
            Toast.show(`Rename failed: ${err.message}`, 'error');
        }
    },
};
