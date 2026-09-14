import { State } from './state.js';

export const uploadCSV = async (file) => {
    const baseUrl = (window.ORBIT_CONFIG && window.ORBIT_CONFIG.API_BASE_URL) || 'http://localhost:8000';
    const formData = new FormData();
    formData.append('file', file);
    
    const response = await fetch(`${baseUrl}/api/tasking/upload`, {
        method: 'POST',
        body: formData
    });
    
    if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to upload CSV file');
    }
    
    return await response.json();
};

export const fetchGeeColumns = async (assetId) => {
    const baseUrl = (window.ORBIT_CONFIG && window.ORBIT_CONFIG.API_BASE_URL) || 'http://localhost:8000';
    const response = await fetch(`${baseUrl}/api/tasking/columns/gee?asset_id=${encodeURIComponent(assetId)}`);
    if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to fetch GEE Asset columns');
    }
    return await response.json();
};

export const fetchGcsColumns = async (gcsUri) => {
    const baseUrl = (window.ORBIT_CONFIG && window.ORBIT_CONFIG.API_BASE_URL) || 'http://localhost:8000';
    const response = await fetch(`${baseUrl}/api/tasking/columns/gcs?gcs_uri=${encodeURIComponent(gcsUri)}`);
    if (!response.ok) {
        const errData = await response.json().catch(() => ({}));
        throw new Error(errData.detail || 'Failed to fetch GCS columns');
    }
    return await response.json();
};

export const submitLandCoverJob = async (payload) => {
    try {
        const baseUrl = (window.ORBIT_CONFIG && window.ORBIT_CONFIG.API_BASE_URL) || 'http://localhost:8000';
        
        // For the demo, we are skipping the actual CSV upload to GCS 
        // because we injected the hardcoded input_asset_id in ui.js instead.
        if (State.customSourceType === 'csv' && State.file) {
            delete payload.csv_url; // Ensure csv_url is not sent
        }

        const response = await fetch(`${baseUrl}/api/tasking/run`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(payload)
        });
        
        if (!response.ok) {
            const errData = await response.json().catch(() => ({}));
            throw new Error(errData.detail || 'Failed to submit National Land Cover job');
        }
        
        return await response.json();
    } catch (error) {
        console.error("Land Cover Job Submission Error:", error);
        throw error;
    }
};
