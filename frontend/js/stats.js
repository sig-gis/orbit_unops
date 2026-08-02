/* ═══════════════════════════════════════════════════════
   COMPASS - ORBIT Operations Center — Stats Module
   ═══════════════════════════════════════════════════════ */

const Stats = {
    _data: null,

    async load(aoiId) {
        this.reset();
        try {
            this._data = await API.getStats(aoiId);
            this._render();
        } catch (err) {
            console.warn('Stats load failed:', err);
        }
    },

    _render() {
        if (!this._data) return;
        const d = this._data;

        this._animateValue('stat-area-value', d.area_km2?.toFixed(0) || '—');
        this._animateValue('stat-ndvi-value', d.mean_ndvi?.toFixed(3) || '—');
        this._animateValue('stat-urban-value', d.urban_pct ? d.urban_pct + '%' : '—');
        this._animateValue('stat-veg-value', d.vegetation_pct ? d.vegetation_pct + '%' : '—');
        this._animateValue('stat-source-value', d.source || 'Sentinel-2');
    },

    _animateValue(elementId, value) {
        const el = document.getElementById(elementId);
        if (!el) return;

        el.style.opacity = '0';
        el.style.transform = 'translateY(4px)';

        setTimeout(() => {
            el.textContent = value;
            el.style.transition = 'opacity 0.3s, transform 0.3s';
            el.style.opacity = '1';
            el.style.transform = 'translateY(0)';
        }, 100);
    },
    reset() {
        this._data = null;
        ['stat-area-value', 'stat-ndvi-value', 'stat-urban-value', 'stat-veg-value', 'stat-source-value'].forEach(id => {
            const el = document.getElementById(id);
            if (el) el.textContent = '—';
        });
    }
};
