"""Load frozen, backbone-specific H2SVC head classifications."""
import json
import argparse
import math
from pathlib import Path

BUDGET_FRAMES = {'instantaneous': 4, 'short_term': 16, 'episodic': 64}


def classify_heads(tad_v2v, tad_t2v):
    """Freeze one route per (layer, KV head) from offline TAD measurements."""
    if set(tad_v2v) != set(tad_t2v) or not tad_v2v:
        raise ValueError('TAD maps must contain the same nonempty head set')
    keys = sorted(tad_v2v)
    def normalized(values):
        nums = [float(values[key]) for key in keys]
        if any(not math.isfinite(x) for x in nums):
            raise ValueError('TAD measurements must be finite')
        low, high = min(nums), max(nums)
        if low == high:
            raise ValueError('TAD measurements cannot all be identical')
        return {key: (float(values[key]) - low)/(high - low) for key in keys}
    visual, text = normalized(tad_v2v), normalized(tad_t2v)
    ranked = sorted(keys, key=lambda key: ((visual[key]+text[key])/2, key))
    instant = round(len(keys)*0.10)
    episodic = round(len(keys)*0.50)
    short_end = len(keys) - episodic
    return {key: ('instantaneous' if index < instant else
                  'short_term' if index < short_end else 'episodic')
            for index, key in enumerate(ranked)}


def read_head_map(path, layers=28, kv_heads=4):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f'H2SVC needs a measured, frozen head map: {path}')
    data = json.loads(path.read_text())
    if data.get('schema_version') != 1 or not data.get('calibration_source_sha256'):
        raise ValueError('Head map needs calibration provenance')
    videos = data.get('calibration_videos')
    if not isinstance(videos, list) or len(videos) != 50 or len(set(map(str, videos))) != 50:
        raise ValueError('H2SVC head map must identify 50 distinct calibration videos')
    rows = data.get('heads')
    if not isinstance(rows, list):
        raise ValueError('Head map must contain a heads list')
    routes = {}
    for row in rows:
        key = (row['layer'], row['head'])
        if key in routes or row['route'] not in BUDGET_FRAMES:
            raise ValueError('Duplicate or invalid head classification')
        routes[key] = row['route']
    expected = {(layer, head) for layer in range(layers) for head in range(kv_heads)}
    if set(routes) != expected:
        raise ValueError('Head map does not cover every KV head')
    if (sum(route == 'instantaneous' for route in routes.values()) != round(len(expected)*0.10) or
        sum(route == 'episodic' for route in routes.values()) != round(len(expected)*0.50)):
        raise ValueError('Head map does not match the paper 10/40/50 routing')
    return routes, data


def map_from_tad(data):
    """Convert measured TAD values into the paper's static 10/40/50 routing."""
    if data.get('schema_version') != 1 or not data.get('source_sha256'):
        raise ValueError('TAD measurements need source provenance')
    visual, text = {}, {}
    for row in data['heads']:
        key = (row['layer'], row['head'])
        if key in visual:
            raise ValueError('Duplicate TAD head')
        visual[key], text[key] = row['tad_v2v'], row['tad_t2v']
    routes = classify_heads(visual, text)
    return {'schema_version': 1, 'model': data['model'],
            'calibration_source_sha256': data['source_sha256'],
            'calibration_videos': data['calibration_videos'],
            'heads': [{'layer': layer, 'head': head, 'route': routes[layer, head]}
                      for layer, head in sorted(routes)]}


def main():
    parser = argparse.ArgumentParser(description='Freeze a backbone-specific H2SVC route map from measured TAD values')
    parser.add_argument('measurements', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    result = map_from_tad(json.loads(args.measurements.read_text()))
    if args.output.exists():
        raise FileExistsError('Never overwrite a frozen H2SVC head map')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n')


if __name__ == '__main__':
    main()
