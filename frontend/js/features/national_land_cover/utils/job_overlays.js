// Restore input overlays, not the post-filtering/capping training sample.
export function overlayConfig(job) {
    const results = job.result?.results || job.result || {};
    const run = results.run || {};
    const request = job.request || job.parameters || {};
    return {
        url: run.csv_url || request.csv_url,
        size: Number(run.block_size_m ?? request.block_size_m),
        crs: run.block_crs || request.block_crs,
        lat: run.columns?.latitude || request.latitude_column || 'lat',
        lon: run.columns?.longitude || request.longitude_column || 'lon',
        target: run.columns?.target || request.target_column || 'target',
        x: run.columns?.block_x || request.block_x_column,
        y: run.columns?.block_y || request.block_y_column,
        threshold: Number(run.target_threshold ?? request.target_threshold ?? 0.5)
    };
}

export function restoreOverlayData(rows, config) {
    const number = value => value == null || String(value).trim() === '' ? NaN : Number(value);
    const points = rows.map(row => ({
        lat: number(row[config.lat]), lon: number(row[config.lon]),
        target: number(row[config.target]), x: number(row[config.x]), y: number(row[config.y])
    }));
    if (!points.length || points.some(p => !Number.isFinite(p.lat) || !Number.isFinite(p.lon) ||
        !Number.isFinite(p.target) || Math.abs(p.lat) > 90 || Math.abs(p.lon) > 180)) {
        throw new Error('Saved input contains missing or invalid point coordinates/classes.');
    }
    const cells = new Map();
    let gridError = '';
    if (config.crs !== 'EPSG:3857' || !Number.isSafeInteger(config.size) || config.size < 1) {
        gridError = 'Sampling grid unavailable: missing size or unsupported saved projection.';
    } else {
        const R = 6378137;
        const inverse = (x, y) => [180 / Math.PI * (2 * Math.atan(Math.exp(y / R)) - Math.PI / 2), x / R * 180 / Math.PI];
        for (const point of points) {
            const x = config.x ? point.x : R * point.lon * Math.PI / 180;
            const y = config.y ? point.y : R * Math.log(Math.tan(Math.PI / 4 + point.lat * Math.PI / 360));
            if (!Number.isFinite(x) || !Number.isFinite(y) || Math.abs(point.lat) >= 85.05112878) {
                gridError = 'Sampling grid unavailable: invalid saved projected coordinates.';
                break;
            }
            const col = Math.floor(x / config.size), row = Math.floor(y / config.size);
            const id = `${col}_${row}`;
            if (!cells.has(id)) cells.set(id, { count: 0, bounds: [
                inverse(col * config.size, row * config.size),
                inverse((col + 1) * config.size, (row + 1) * config.size)
            ] });
            cells.get(id).count++;
            if (cells.size > 1000) {
                gridError = 'Sampling grid unavailable: exceeds the 1,000-block display limit.';
                break;
            }
        }
    }
    return { points, cells: gridError ? [] : [...cells.values()], gridError };
}

export async function loadJobOverlays(job, baseUrl, signal) {
    const config = overlayConfig(job);
    if (!config.url) throw new Error('Input overlays unavailable: this job has no saved CSV source.');
    const url = config.url.replace(/^gs:\/\/([^/]+)\/(.+)$/, 'https://storage.googleapis.com/$1/$2');
    if (!url.startsWith('https://storage.googleapis.com/')) throw new Error('Input overlays unavailable: unsupported CSV source.');
    const response = await fetch(`${baseUrl}/proxy-csv?url=${encodeURIComponent(url)}`, { signal });
    if (!response.ok) throw new Error('Input overlays unavailable: the saved CSV could not be loaded.');
    const parsed = Papa.parse(await response.text(), { header: true, skipEmptyLines: true });
    if (parsed.errors.length) throw new Error('Input overlays unavailable: the saved CSV is malformed.');
    return { ...restoreOverlayData(parsed.data, config), threshold: config.threshold };
}