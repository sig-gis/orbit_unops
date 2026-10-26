"""Strict validation for local binary-classification CSV uploads."""
import csv
import io
import math
import re


def validate_csv(contents, latitude_column, longitude_column, target_column):
    try:
        text = contents.decode('utf-8-sig')
        if re.search(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f\ufffd]', text):
            raise ValueError('CSV contains unsupported control or replacement characters.')
        records = [row for row in csv.reader(io.StringIO(text), strict=True) if row and row != ['']]
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ValueError('Malformed CSV. Upload a valid UTF-8 CSV file.') from exc
    if len(records) < 2:
        raise ValueError('The CSV file contains no data records.')
    headers, *rows = records
    invisible = r'[\u200b-\u200f\u202a-\u202e\u2060-\u2069\ufeff]'
    if any(re.search(invisible, h) or re.search(r'[\t\r\n]', h) for h in headers):
        raise ValueError('CSV column names contain invisible or control characters.')
    if any(not h.strip() for h in headers) or len(set(h.strip() for h in headers)) != len(headers):
        raise ValueError('CSV column names must be nonempty and unique.')
    columns = [latitude_column, longitude_column, target_column]
    if len(set(columns)) != 3 or any(c not in headers for c in columns):
        raise ValueError('Select three distinct latitude, longitude, and class columns.')
    indices = [headers.index(c) for c in columns]
    classes = set()
    locations = {}
    for number, row in enumerate(rows, 1):
        if len(row) != len(headers):
            raise ValueError(f'Record {number}: field count does not match the header.')
        raw_values = [row[i] for i in indices]
        values = [v.strip() for v in raw_values]
        for name, raw, value in zip(('latitude', 'longitude', 'class'), raw_values, values):
            if re.search(invisible, raw) or not re.fullmatch(r'[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?', value, re.ASCII):
                raise ValueError(f'Record {number}: {name} must be numeric and nonempty, without unexpected characters.')
        lat, lon, target = map(float, values)
        if not all(math.isfinite(v) for v in (lat, lon, target)):
            raise ValueError(f'Record {number}: values must be finite numbers.')
        if not -90 <= lat <= 90 or not -180 <= lon <= 180:
            hint = ' Latitude and longitude may be reversed.' if -90 <= lon <= 90 and -180 <= lat <= 180 else ''
            errors = []
            if not -90 <= lat <= 90:
                errors.append(f'latitude {lat} is outside the allowed range -90 to 90')
            if not -180 <= lon <= 180:
                errors.append(f'longitude {lon} is outside the allowed range -180 to 180')
            raise ValueError(f'Record {number}: {"; ".join(errors)}.{hint}')
        if target not in (0, 1):
            raise ValueError(f'Record {number}: class must be 0 or 1.')
        key = (lat, lon)
        if key in locations:
            previous_number, previous_target = locations[key]
            reason = 'Conflicting class labels at the same point location' if previous_target != target else 'Duplicate point location'
            raise ValueError(f'{reason}: records {previous_number} and {number}. Correct the CSV and upload again.')
        locations[key] = (number, target)
        classes.add(target)
    if classes != {0, 1}:
        raise ValueError('The class column must contain both classes: 0 and 1.')