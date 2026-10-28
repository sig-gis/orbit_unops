// Match the backend's origin-aligned floor(x / size), floor(y / size) grid.
const R = 6378137;
export function projectPoint(point) {
    if (!Number.isFinite(point.lat) || !Number.isFinite(point.lon) || Math.abs(point.lat) >= 85.05112878) {
        throw new Error('Grid preview supports latitudes between -85.05112878 and 85.05112878 degrees.');
    }
    return { ...point, x: R * point.lon * Math.PI / 180,
        y: R * Math.log(Math.tan(Math.PI / 4 + point.lat * Math.PI / 360)) };
}

export function buildGrid(points, target, cellSize = null) {
    if (cellSize !== null && (!Number.isSafeInteger(cellSize) || cellSize < 1)) {
        throw new Error('Cell size must be a positive whole number of projected meters.');
    }
    if (cellSize === null && (!Number.isInteger(target) || target < 2 || target > 1000)) {
        throw new Error('Target blocks must be a whole number between 2 and 1000.');
    }
    if (!points.length) throw new Error('No in-country points are available for the grid.');
    const projected = points.map(projectPoint);
    const group = size => {
        const cells = new Map();
        for (const point of projected) {
            const col = Math.floor(point.x / size), row = Math.floor(point.y / size);
            const id = `${col}_${row}`;
            if (!cells.has(id)) cells.set(id, { id, col, row, count: 0, positive: 0 });
            const cell = cells.get(id);
            cell.count++;
            cell.positive += point.target === 1 ? 1 : 0;
        }
        return cells;
    };
    // Occupancy is not strictly monotonic when an origin-aligned grid changes size.
    // Search a bounded logarithmic range, then refine around the best candidate.
    let best;
    const consider = size => {
        size = Math.max(1, Math.round(size));
        const cells = group(size);
        const error = Math.abs(cells.size - target);
        if (!best || error < best.error || (error === best.error && size > best.size)) best = { size, cells, error };
    };
    if (cellSize !== null) {
        best = { size: cellSize, cells: group(cellSize) };
    } else {
        for (let i = 0; i <= 100; i++) consider(Math.exp(Math.log(40075017) * i / 100));
        const center = best.size;
        for (let i = -20; i <= 20; i++) consider(center * Math.exp(i / 100));
    }
    const inverse = (x, y) => [180 / Math.PI * (2 * Math.atan(Math.exp(y / R)) - Math.PI / 2), x / R * 180 / Math.PI];
    const cells = [...best.cells.values()].map(cell => ({ ...cell, bounds: [
        inverse(cell.col * best.size, cell.row * best.size),
        inverse((cell.col + 1) * best.size, (cell.row + 1) * best.size)
    ] }));
    return { size: best.size, cells, points: projected, average: points.length / cells.length };
}

export function gridCSV(grid) {
    return 'lat,lon,target,block_x_m,block_y_m\n' + grid.points.map(p =>
        [p.lat, p.lon, p.target, p.x, p.y].join(',')).join('\n');
}