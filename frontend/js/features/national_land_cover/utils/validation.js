const unsafeText = /[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd]/;
const invisible = /[\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff]/;

export const readCSV = async file => {
    if (!file || !/\.csv$/i.test(file.name)) throw new Error('Unsupported file type. Only CSV (.csv) files are supported.');
    if (!file.size) throw new Error('The CSV file is empty.');
    let text;
    try {
        text = new TextDecoder('utf-8', { fatal: true }).decode(await file.arrayBuffer());
    } catch {
        throw new Error('Cannot read CSV. Upload a valid UTF-8 CSV file.');
    }
    if (unsafeText.test(text)) throw new Error('CSV contains unsupported control or replacement characters. Correct the file and upload again.');
    return parseCSVText(text);
};

const parseCSVText = text => new Promise((resolve, reject) => {
    Papa.parse(text, {
        skipEmptyLines: true,
        complete(results) {
            try {
                if (results.errors.length) throw new Error(`Malformed CSV: ${results.errors[0].message}`);
                const [headers, ...records] = results.data;
                if (!headers || !records.length) throw new Error('The CSV file contains no data records.');
                if (headers.some(h => !h.trim()) || new Set(headers.map(h => h.trim())).size !== headers.length) throw new Error('CSV column names must be nonempty and unique.');
                if (headers.some(h => invisible.test(h) || /[\t\r\n]/.test(h))) throw new Error('CSV column names contain invisible or control characters.');
                if (records.some(row => row.length !== headers.length)) throw new Error('Malformed CSV: records must have the same number of fields as the header.');
                resolve({ headers, rows: records.map(row => Object.fromEntries(headers.map((h, i) => [h, row[i]]))) });
            } catch (error) { reject(error); }
        },
        error: reject
    });
});

const numeric = value => typeof value === 'string' && !invisible.test(value) && /^[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?$/i.test(value.trim()) && Number.isFinite(Number(value));

export function validateCSVRows(rows, columns) {
    const names = [columns.lat, columns.lon, columns.target];
    if (new Set(names).size !== 3 || names.some(name => !name || !Object.prototype.hasOwnProperty.call(rows[0] || {}, name))) throw new Error('Select three distinct latitude, longitude, and class columns.');
    const points = rows.map((row, index) => {
        const lat = row[columns.lat], lon = row[columns.lon], target = row[columns.target];
        if (!numeric(lat)) throw new Error(`Record ${index + 1}: latitude must be a finite number, without missing values or unexpected characters.`);
        if (!numeric(lon)) throw new Error(`Record ${index + 1}: longitude must be a finite number, without missing values or unexpected characters.`);
        const y = Number(lat), x = Number(lon);
        if (Math.abs(y) > 90 || Math.abs(x) > 180) {
            const hint = Math.abs(x) <= 90 && Math.abs(y) <= 180 ? ' Latitude and longitude may be reversed; check your column selections.' : '';
            const errors = [];
            if (Math.abs(y) > 90) errors.push(`latitude ${y} is outside the allowed range -90 to 90`);
            if (Math.abs(x) > 180) errors.push(`longitude ${x} is outside the allowed range -180 to 180`);
            throw new Error(`Record ${index + 1}: ${errors.join('; ')}.${hint}`);
        }
        if (!numeric(target) || ![0, 1].includes(Number(target))) throw new Error(`Record ${index + 1}: the class value must be 0 or 1.`);
        return { ...row, lat: y, lon: x, target: Number(target) };
    });
    const locations = new Map();
    for (const [index, point] of points.entries()) {
        const key = `${point.lat},${point.lon}`;
        const previous = locations.get(key);
        if (previous) {
            const reason = previous.target !== point.target ? 'Conflicting class labels at the same point location' : 'Duplicate point location';
            throw new Error(`${reason}: records ${previous.record} and ${index + 1}. Correct the CSV and upload again.`);
        }
        locations.set(key, { target: point.target, record: index + 1 });
    }
    if (new Set(points.map(p => p.target)).size !== 2) throw new Error('The class column must contain both classes: 0 and 1.');
    return points;
}

export function assessCountryOrientation(points, country, countries, spatial) {
    const outside = countOutsideCountry(points, country, countries, spatial);
    if (!outside) return { outside, hint: '' };
    // Only test swapped positions whose new latitude is in range.
    const swappable = points.filter(p => Math.abs(p.lon) <= 90);
    const swappedInside = swappable.length - countOutsideCountry(swappable.map(p => ({ lat: p.lon, lon: p.lat })), country, countries, spatial);
    const inside = points.length - outside;
    const counts = `Current mapping: ${inside} of ${points.length} points intersect ${country}; swapped mapping: ${swappedInside} of ${points.length}.`;
    if (!inside && swappedInside === points.length) throw new Error(`Latitude and longitude appear to be reversed. ${counts} Swap the selected columns and load again.`);
    let hint = '';
    if (swappedInside > inside) hint = ` Latitude and longitude may be reversed. ${counts} Check your column selections.`;
    else {
        const rescued = points.filter(p => Math.abs(p.lon) <= 90 && countOutsideCountry([p], country, countries, spatial) && !countOutsideCountry([{ lat: p.lon, lon: p.lat }], country, countries, spatial)).length;
        if (rescued) hint = ` ${rescued} outside point(s) would intersect the country if swapped; some records may have reversed coordinates. Check the CSV; do not swap all rows automatically.`;
    }
    return { outside, hint };
}

export function countOutsideCountry(points, country, countries, spatial) {
    const features = countries?.features.filter(f => f.properties.nam_en === country);
    if (!features?.length || !spatial) throw new Error('Country boundary is unavailable. Reload the page and try again.');
    return points.filter(p => !features.some(f => spatial.booleanPointInPolygon(spatial.point([p.lon, p.lat]), f, { ignoreBoundary: false }))).length;
}