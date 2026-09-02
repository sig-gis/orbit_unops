export const Charts = {
    instances: { cumulative: null, growth: null },

    renderCharts(annualData, areaData, spanData) {
        const areaLabels = areaData.map(r => r.year.toString());
        const rfArea = areaData.map(r => r.RF_sm_km2);
        const annualLabels = annualData.map(r => r.window);
        const lcrData = annualData.map(r => r.LCR * 100);
        const pgrData = annualData.map(r => r.PGR * 100);

        // Destroy old instances
        if (this.instances.cumulative) this.instances.cumulative.destroy();
        if (this.instances.growth) this.instances.growth.destroy();

        const areaChartCtx = document.getElementById('hero-area-chart');
        const rateChartCtx = document.getElementById('hero-rate-chart');
        
        if (!areaChartCtx || !rateChartCtx) return;

        this.instances.cumulative = new Chart(areaChartCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: areaLabels,
                datasets: [{
                    label: 'Total Urban Area (km²)',
                    data: rfArea,
                    backgroundColor: 'rgba(74, 144, 226, 0.7)',
                    borderRadius: 4
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: true, position: 'top', labels: { boxWidth: 10, font: {size: 10} } } },
                scales: { x: { grid: {display: false} }, y: { title: {display: true, text: 'km²', font: {size: 10}} } }
            }
        });

        this.instances.growth = new Chart(rateChartCtx.getContext('2d'), {
            type: 'bar',
            data: {
                labels: annualLabels,
                datasets: [
                    { label: 'Land Area Growth (LCR)', data: lcrData, backgroundColor: 'rgba(255, 87, 34, 0.8)', borderRadius: 4 },
                    { label: 'Population Growth (PGR)', data: pgrData, backgroundColor: 'rgba(74, 144, 226, 0.8)', borderRadius: 4 }
                ]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { 
                    legend: { display: true, position: 'bottom', labels: { usePointStyle: true, boxWidth: 8, font: {size: 10} } },
                    tooltip: {
                        callbacks: {
                            label: function(context) {
                                return context.dataset.label + ': ' + context.parsed.y.toFixed(2) + '%';
                            }
                        }
                    }
                },
                scales: { 
                    x: { grid: {display: false} }, 
                    y: { 
                        title: {display: true, text: 'Annual Growth (%)', font: {size: 10}},
                        ticks: {
                            callback: function(value) {
                                return value > 0 ? '+' + value + '%' : value + '%';
                            }
                        }
                    } 
                }
            }
        });
    }
};
