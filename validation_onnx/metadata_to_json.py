#!/usr/bin/env python3
"""Extract ONNX metadata into one readable JSON file per model.

Usage:
    python metadata_to_json.py *.onnx
    python metadata_to_json.py model1.onnx model2.onnx

Requires: pip install onnx
"""

import argparse
import json
import sys
from pathlib import Path

# Fields that OLLL v0.1 stores as JSON-encoded strings in ONNX metadata.
JSON_FIELDS = {
    'source', 'channels', 'obs_yields', 'bkg_yields', 'bkg_unc',
    'removeCRsVRs', 'remove_channels', 'x_min', 'x_max', 'y_min',
    'y_max', 'preprocessing', 'standardization', 'model_parameters',
    'contact', 'persistent_id',
    'nLL_exp_max', 'nLL_obs_max', 'nLLA_exp_max', 'nLLA_obs_max',
}
FLOAT_FIELDS = {
    'nLL_exp_mu0', 'nLL_obs_mu0', 'nLLA_exp_mu0', 'nLLA_obs_mu0',
}


def extract(path):
    import onnx

    model = onnx.load(str(path), load_external_data=False)
    metadata = {}
    for entry in model.metadata_props:
        if entry.key in metadata:
            raise ValueError(f'duplicate metadata key: {entry.key!r}')
        value = entry.value
        if entry.key in JSON_FIELDS:
            value = json.loads(value)
        elif entry.key in FLOAT_FIELDS:
            value = float(value)
        elif entry.key == 'run_config':
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                try:
                    import yaml
                except ImportError:
                    # Keep original YAML text if PyYAML is unavailable.
                    pass
                else:
                    value = yaml.safe_load(value)
        metadata[entry.key] = value
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs='+', type=Path, help='ONNX model files')
    args = parser.parse_args()
    try:
        import onnx  # noqa: F401
    except ImportError:
        parser.exit(2, 'Missing dependency: pip install onnx\n')

    failures = 0
    for path in args.models:
        if path.suffix.lower() != '.onnx':
            print(f'ERROR {path}: expected .onnx file', file=sys.stderr)
            failures += 1
            continue
        destination = path.with_suffix('.json')
        try:
            metadata = extract(path)
            destination.write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
            print(f'{path} -> {destination}')
        except (OSError, ValueError, TypeError, RuntimeError) as exc:
            print(f'ERROR {path}: {exc}', file=sys.stderr)
            failures += 1
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
