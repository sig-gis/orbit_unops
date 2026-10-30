// Frontend visibility only. Set sdg to true to restore the preserved SDG UI.
window.ORBIT_FEATURES = { sdg: false, ...(window.ORBIT_FEATURES || {}) };
window.OrbitFeatures = {
    get sdgEnabled() { return window.ORBIT_FEATURES.sdg === true; },
    isSDGJob(job) {
        const id = String(job.indicator_id || job.request?.indicator_id || '').trim();
        const module = String(job.module_name || '').toLowerCase();
        if (/^(NLC|TASKING)$/i.test(id) || /national land\s*cover/.test(module)) return false;
        return /^sdg/i.test(id) || /^\d+\.\d+\.\d+/.test(id) || /sdg|urban.expansion/.test(module);
    },
    isJobVisible(job) { return this.sdgEnabled || !this.isSDGJob(job); }
};
document.documentElement.classList.toggle('sdg-disabled', !window.OrbitFeatures.sdgEnabled);