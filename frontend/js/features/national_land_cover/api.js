export const submitLandCoverJob = async (payload) => {
    try {
        const baseUrl = (window.ORBIT_CONFIG && window.ORBIT_CONFIG.API_BASE_URL) || 'http://localhost:8000';
        const response = await fetch(`${baseUrl}/api/land-cover/run`, {
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
