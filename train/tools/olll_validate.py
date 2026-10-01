#!/usr/bin/env python3
"""OLLL v0.1 metadata validator.

Usage:
    python olll_validate.py model.onnx [other.onnx ...]
    python olll_validate.py metadata.json
    python olll_validate.py model.onnx --json-report report.json
    python olll_validate.py model.onnx --inspection

Install ONNX support: pip install onnx
JSON/YAML authoring templates can be checked without ONNX; YAML and YAML-encoded run_config require PyYAML.

Exit codes: 0 valid, including warnings; 1 invalid; 2 dependency/input failure.
This program never changes the supplied model or its metadata.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import date
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import urlparse

VERSION = '0.1'
REQUIRED = (
    'olll_schema_version', 'model_type', 'model_name', 'model_author',
    'model_license', 'source', 'input_type', 'channels', 'obs_yields',
    'bkg_yields', 'bkg_unc', 'removeCRsVRs', 'remove_channels',
    'x_min', 'x_max', 'y_min', 'y_max', 'preprocessing', 'standardization',
    'nLL_exp_mu0', 'nLL_obs_mu0', 'nLLA_exp_mu0', 'nLLA_obs_mu0',
    'nLL_exp_max', 'nLL_obs_max', 'nLLA_exp_max', 'nLLA_obs_max',
)
LIKELIHOOD_MU0_KEYS = REQUIRED[-8:-4]
LIKELIHOOD_MAX_KEYS = REQUIRED[-4:]
LIKELIHOOD_KEYS = LIKELIHOOD_MU0_KEYS + LIKELIHOOD_MAX_KEYS
JSON_FIELDS = {
    'source', 'channels', 'obs_yields', 'bkg_yields', 'bkg_unc',
    'removeCRsVRs', 'remove_channels', 'x_min', 'x_max', 'y_min',
    'y_max', 'preprocessing', 'standardization', 'model_parameters',
    'contact', 'persistent_id',
}
SUPPORTED_OPS = {'log_w_negatives', 'standardization', 'log', 'asinh'}


class Report:
    def __init__(self, path):
        self.path = str(path)
        self.issues = []

    def add(self, severity, code, message):
        self.issues.append({'severity': severity, 'code': code, 'message': message})

    def error(self, code, message): self.add('ERROR', code, message)
    def warn(self, code, message): self.add('WARNING', code, message)
    def info(self, code, message): self.add('INFO', code, message)

    def as_dict(self):
        errors = sum(x['severity'] == 'ERROR' for x in self.issues)
        warnings = sum(x['severity'] == 'WARNING' for x in self.issues)
        return {'file': self.path,
                'status': 'INVALID' if errors else ('VALID WITH WARNINGS' if warnings else 'VALID'),
                'errors': errors, 'warnings': warnings, 'issues': self.issues}


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def decode_metadata(entries, report):
    """Read raw ONNX metadata entries without losing duplicate keys."""
    # Inspect every key, including optional and unknown metadata fields.
    by_key = {}
    for entry in entries:
        by_key.setdefault(entry.key, []).append(entry.value)
    for key, values in by_key.items():
        if len(values) <= 1:
            continue
        if len(set(values)) == 1:
            report.warn('DUPLICATE_KEY',
                        f'{key}: occurs {len(values)} times with identical values; remove duplicates')
        else:
            report.error('DUPLICATE_KEY',
                         f'{key}: occurs {len(values)} times with conflicting values; resolve explicitly')
    result = {}
    for entry in entries:
        if entry.key in result:
            continue  # report the error; never silently select the last value
        if entry.key == 'run_config':
            # Legacy models store this optional configuration as either JSON or YAML.
            try:
                result[entry.key] = json.loads(entry.value)
            except (ValueError, TypeError):
                try:
                    import yaml
                except ImportError:
                    report.error('YAML_DEPENDENCY',
                                 'run_config is not JSON; install PyYAML to parse YAML (pip install pyyaml)')
                    result[entry.key] = None
                else:
                    try:
                        result[entry.key] = yaml.safe_load(entry.value)
                    except yaml.YAMLError as exc:
                        report.error('INVALID_RUN_CONFIG', f'run_config is neither valid JSON nor YAML ({exc})')
                        result[entry.key] = None
            if result[entry.key] is not None and not isinstance(result[entry.key], dict):
                report.error('INVALID_RUN_CONFIG', 'run_config must decode to a mapping/object')
        elif entry.key in JSON_FIELDS:
            try:
                result[entry.key] = json.loads(entry.value)
            except (ValueError, TypeError) as exc:
                report.error('INVALID_JSON', f'{entry.key}: invalid JSON ({exc})')
                result[entry.key] = None
        elif entry.key in LIKELIHOOD_MU0_KEYS:
            try:
                result[entry.key] = float(entry.value)
            except (ValueError, TypeError):
                result[entry.key] = entry.value
        elif entry.key in LIKELIHOOD_MAX_KEYS:
            try:
                result[entry.key] = json.loads(entry.value)
            except (ValueError, TypeError):
                result[entry.key] = entry.value
        else:
            result[entry.key] = entry.value
    return result


def load_input(path, report, check_graph=True):
    suffix = path.suffix.lower()
    if suffix == '.json':
        # object_pairs_hook catches duplicate JSON keys at all levels
        def unique_pairs(pairs):
            d = {}
            for k, v in pairs:
                if k in d:
                    report.error('DUPLICATE_KEY', f'Duplicate JSON key: {k}')
                d[k] = v
            return d
        return json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=unique_pairs), None
    if suffix in ('.yaml', '.yml'):
        try:
            import yaml
        except ImportError as exc:
            raise RuntimeError('PyYAML is required for YAML files: pip install pyyaml') from exc
        # Standard safe_load does not reject duplicate mapping keys; enforce uniqueness.
        class UniqueLoader(yaml.SafeLoader): pass
        def unique_mapping(loader, node):
            mapping = {}
            for key_node, value_node in node.value:
                key = loader.construct_object(key_node)
                if key in mapping:
                    report.error('DUPLICATE_KEY', f'Duplicate YAML key: {key}')
                mapping[key] = loader.construct_object(value_node)
            return mapping
        UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)
        return yaml.load(path.read_text(encoding='utf-8'), Loader=UniqueLoader), None
    if suffix != '.onnx':
        raise RuntimeError('Expected .onnx, .json, .yaml or .yml')
    try:
        import onnx
    except ImportError as exc:
        raise RuntimeError('onnx is required to inspect .onnx files: pip install onnx') from exc
    model = onnx.load(str(path), load_external_data=False)
    if check_graph:
        try:
            onnx.checker.check_model(model)
        except Exception as exc:
            report.error('ONNX_CHECKER', str(exc))
    return decode_metadata(model.metadata_props, report), model


def check_source(source, report):
    if not isinstance(source, dict):
        report.error('SOURCE', 'source must be a JSON object')
        return
    for key in ('analysis_id', 'arxiv', 'experiment'):
        if not isinstance(source.get(key), str) or not source[key].strip():
            report.error('SOURCE', f'source.{key} must be a nonempty string')
    if 'arxiv' in source and isinstance(source['arxiv'], str) and not re.fullmatch(r'(?:\d{4}\.\d{4,5}|[\w.-]+/\d{7})(?:v\d+)?', source['arxiv']):
        report.warn('ARXIV_FORMAT', 'source.arxiv does not look like an arXiv identifier')
    if not finite_number(source.get('sqrt_s')) or source['sqrt_s'] <= 0:
        report.error('SOURCE', 'source.sqrt_s must be positive (TeV)')
    if 'luminosity' in source and (not finite_number(source['luminosity']) or source['luminosity'] <= 0):
        report.error('SOURCE', 'source.luminosity must be positive (fb^-1)')
    if 'inspire_id' in source and not str(source['inspire_id']).strip():
        report.error('SOURCE', 'source.inspire_id cannot be empty')
    stat = source.get('statistical_model')
    if not isinstance(stat, dict):
        report.error('SOURCE', 'source.statistical_model must be an object')
        return
    if stat.get('type') not in ('histfactory', 'hs3', 'combine'):
        report.error('SOURCE', 'source.statistical_model.type must be histfactory, hs3 or combine')
    for key in ('reference', 'filename'):
        if not isinstance(stat.get(key), str) or not stat[key].strip():
            report.error('SOURCE', f'source.statistical_model.{key} must be nonempty')
    ref = stat.get('reference')
    if isinstance(ref, str) and ref and not (urlparse(ref).scheme in ('http', 'https') and urlparse(ref).netloc or ref.startswith('10.')):
        report.warn('SOURCE_REFERENCE', 'source.statistical_model.reference should be a URL or DOI')


def check_yields(data, report):
    names_by_key = {}
    for key in ('obs_yields', 'bkg_yields', 'bkg_unc'):
        value = data.get(key)
        if not isinstance(value, list):
            report.error('YIELDS', f'{key} must be an ordered list of [bin_name, value] pairs')
            continue
        names = []
        for i, item in enumerate(value):
            if not isinstance(item, list) or len(item) != 2 or not isinstance(item[0], str) or not item[0]:
                report.error('YIELDS', f'{key}[{i}] must be [nonempty bin_name, value]')
                continue
            name, number = item
            names.append(name)
            if not finite_number(number) or number < 0:
                report.error('YIELDS', f'{key}[{i}]: value must be finite and nonnegative')
            if key == 'obs_yields' and finite_number(number) and number != int(number):
                report.warn('OBS_COUNT', f'{key}[{i}]: observed count is not integer-valued')
        if len(names) != len(set(names)):
            report.error('YIELDS', f'{key} has duplicate bin names')
        names_by_key[key] = names
    base = names_by_key.get('obs_yields')
    for key in ('bkg_yields', 'bkg_unc'):
        if base is not None and key in names_by_key and base != names_by_key[key]:
            report.error('YIELD_ORDER', f'{key} bin names/order differ from obs_yields')
    channels = data.get('channels')
    if isinstance(channels, dict) and base is not None:
        # A bin can be named exactly as a channel or channel + '-' + bin label.
        # Use longest channel prefix to avoid ambiguity for similarly named channels.
        for name in base:
            matches = [c for c in channels if name == c or name.startswith(c + '-')]
            if not matches:
                report.error('BIN_CHANNEL', f'Bin {name!r} has no identifiable channel in channels')
    return len(base) if base is not None else None


def check_bounds(data, report, model):
    dims = {}
    for prefix in ('x', 'y'):
        lo, hi = data.get(prefix + '_min'), data.get(prefix + '_max')
        if not isinstance(lo, list) or not isinstance(hi, list):
            report.error('BOUNDS', f'{prefix}_min and {prefix}_max must be arrays')
            continue
        if len(lo) != len(hi) or not lo:
            report.error('BOUNDS', f'{prefix} bound arrays must have equal nonzero length')
            continue
        dims[prefix] = len(lo)
        for i, (a, b) in enumerate(zip(lo, hi)):
            if not finite_number(a) or not finite_number(b):
                report.error('BOUNDS', f'{prefix} bounds at {i} must be finite numbers')
            elif a > b:
                report.error('BOUNDS', f'{prefix}_min[{i}] exceeds {prefix}_max[{i}]')
            elif a == b:
                report.warn('FIXED_BOUND', f'{prefix} dimension {i} has equal bounds')
    # OLLL v0.1 has exactly four likelihood targets and eight raw outputs.
    # The first four are preprocessed nLL differences; the last four are
    # their preprocessed uncertainties (not log-variances).
    if 'y' in dims and dims['y'] != 4:
        report.error('OUTPUT_CONVENTION',
                     f'OLLL v0.1 requires exactly four y-bound likelihood targets; found {dims["y"]}')
    params = data.get('model_parameters')
    if isinstance(params, dict) and 'out_shape' in params and params['out_shape'] != 4:
        report.error('OUTPUT_CONVENTION',
                     f'model_parameters.out_shape must be 4 likelihood targets; found {params["out_shape"]!r}')
    if model is not None:
        # Exclude graph initializers from actual runtime inputs.
        initializers = {v.name for v in model.graph.initializer}
        graph_inputs = [v for v in model.graph.input if v.name not in initializers]
        graph_outputs = list(model.graph.output)
        for kind, values, key in [('input', graph_inputs, 'x'), ('output', graph_outputs, 'y')]:
            if len(values) != 1:
                if kind == 'output':
                    report.error('OUTPUT_CONVENTION',
                                 f'OLLL v0.1 requires exactly one eight-component output tensor; found {len(values)}')
                else:
                    report.warn('GRAPH_IO', f'Expected one tensor for graph {kind}; found {len(values)}; dimension check skipped')
                continue
            tensor = values[0].type.tensor_type
            shape = tensor.shape.dim
            if not shape:
                report.warn('GRAPH_SHAPE', f'Graph {kind} shape unknown')
                continue
            last = shape[-1]
            if last.HasField('dim_value') and key in dims:
                expected = 8 if kind == 'output' else dims[key]
                if last.dim_value != expected:
                    report.error('GRAPH_DIM', f'Graph {kind} last dimension {last.dim_value} != expected {expected} (OLLL v0.1: four nLL differences + four uncertainties)' if kind == 'output' else f'Graph input last dimension {last.dim_value} != expected {expected} (x bounds length)')
            elif not last.HasField('dim_value'):
                if kind == 'output':
                    report.error('OUTPUT_CONVENTION', 'Graph output last dimension must be statically known to equal 8')
                else:
                    report.warn('GRAPH_SHAPE', 'Graph input last dimension symbolic or unknown')
    return dims


def flattened_vector(value):
    if not isinstance(value, list): return None
    if len(value) == 1 and isinstance(value[0], list): return value[0]
    return value


def check_pipeline(value, label, report):
    if isinstance(value, list):
        for op in value:
            if isinstance(op, str):
                if op not in SUPPORTED_OPS:
                    report.warn('PIPELINE_OP', f'{label}: unknown operation {op!r}; producer-specific operation may require manual review')
            elif isinstance(op, dict):
                name = op.get('name', op.get('type'))
                if name not in SUPPORTED_OPS:
                    report.warn('PIPELINE_OP', f'{label}: unrecognized operation object {op!r}')
            else:
                report.error('PIPELINE', f'{label}: pipeline entries must be strings or objects')
    elif isinstance(value, dict):
        # Preserve legacy per-output pipeline representations; recursively inspect
        # recognizable pipeline values, warn instead of inventing a new format.
        for key, item in value.items():
            if isinstance(item, (list, dict)):
                check_pipeline(item, f'{label}.{key}', report)
    else:
        report.error('PIPELINE', f'{label} must be an array or object')


def check_transforms(data, report, dims):
    preprocessing = data.get('preprocessing')
    standardization = data.get('standardization')
    if not isinstance(preprocessing, dict):
        report.error('PREPROCESSING', 'preprocessing must be an object')
        return
    for key in ('features_pipeline', 'nLLs_pipeline'):
        if key not in preprocessing:
            report.error('PREPROCESSING', f'preprocessing.{key} is required')
        else:
            check_pipeline(preprocessing[key], f'preprocessing.{key}', report)
    if not isinstance(standardization, dict):
        report.error('STANDARDIZATION', 'standardization must be an object')
        return
    for kind, dimkey in [('features', 'x'), ('nLLs', 'y')]:
        mean, std = standardization.get(kind + '_mean'), standardization.get(kind + '_std')
        # If absent, check whether the corresponding pipeline uses standardization.
        pipe = preprocessing.get(kind + '_pipeline')
        uses_standardization = 'standardization' in json.dumps(pipe)
        if mean is None and std is None and not uses_standardization:
            continue
        meanvec, stdvec = flattened_vector(mean), flattened_vector(std)
        if not isinstance(meanvec, list) or not isinstance(stdvec, list):
            report.error('STANDARDIZATION', f'{kind}_mean and {kind}_std must be numeric arrays')
            continue
        if len(meanvec) != len(stdvec):
            report.error('STANDARDIZATION', f'{kind}: mean and std lengths differ')
        if dimkey in dims and len(meanvec) != dims[dimkey]:
            report.error('STANDARDIZATION', f'{kind}: parameter length {len(meanvec)} != {dimkey} dimension {dims[dimkey]}')
        for i, value in enumerate(meanvec):
            if not finite_number(value): report.error('STANDARDIZATION', f'{kind}_mean[{i}] must be finite')
        for i, value in enumerate(stdvec):
            if not finite_number(value) or value <= 0:
                report.error('STANDARDIZATION', f'{kind}_std[{i}] must be finite and > 0')


def validate(data, report, model=None, inspection=False):
    if not isinstance(data, dict):
        report.error('METADATA', 'Top-level metadata must be a mapping')
        return
    for key in REQUIRED:
        if key not in data:
            (report.warn if inspection else report.error)('MISSING_REQUIRED', f'Missing required field: {key}')
    if data.get('olll_schema_version') not in (VERSION, None):
        report.error('VERSION', f'Expected olll_schema_version = {VERSION!r}')
    for key in ('model_type', 'model_name', 'model_author', 'model_license'):
        if key in data and (not isinstance(data[key], str) or not data[key].strip()):
            report.error('MODEL', f'{key} must be a nonempty string')
    if 'training_date' in data:
        try: date.fromisoformat(data['training_date'])
        except (ValueError, TypeError): report.warn('TRAINING_DATE', 'training_date should be YYYY-MM-DD')
    if 'source' in data: check_source(data['source'], report)
    if 'input_type' in data and data['input_type'] != 'total_yields':
        report.error('INPUT_TYPE', 'v0.1 requires input_type = total_yields')
    channels = data.get('channels')
    if 'channels' in data:
        if not isinstance(channels, dict) or not channels:
            report.error('CHANNELS', 'channels must be a nonempty mapping')
        else:
            for name, cls in channels.items():
                if not isinstance(name, str) or not name or cls not in ('SR', 'CR', 'VR'):
                    report.error('CHANNELS', f'Invalid channel/classification: {name!r}: {cls!r}')
    if any(k in data for k in ('obs_yields', 'bkg_yields', 'bkg_unc')):
        check_yields(data, report)
    flag, removed = data.get('removeCRsVRs'), data.get('remove_channels')
    if 'removeCRsVRs' in data and not isinstance(flag, bool):
        report.error('REMOVAL', 'removeCRsVRs must be a JSON Boolean')
    if 'remove_channels' in data:
        if not isinstance(removed, list) or any(not isinstance(x, str) or not x for x in removed):
            report.error('REMOVAL', 'remove_channels must be an array of nonempty channel names')
        else:
            if len(removed) != len(set(removed)):
                report.error('REMOVAL', 'remove_channels contains duplicate names')
            if isinstance(channels, dict):
                for name in removed:
                    if name not in channels:
                        report.error('REMOVAL', f'Removed channel {name!r} not found in channels')
            if isinstance(flag, bool):
                if flag and not removed:
                    report.warn('REMOVAL_MISMATCH', 'removeCRsVRs is true, but remove_channels is empty')
                if not flag and removed:
                    report.warn('REMOVAL_MISMATCH', 'removeCRsVRs is false, but remove_channels is not empty')
    dims = check_bounds(data, report, model)
    if 'preprocessing' in data and 'standardization' in data:
        check_transforms(data, report, dims)
    for key in LIKELIHOOD_MU0_KEYS:
        if key in data and not finite_number(data[key]):
            report.error('LIKELIHOOD', f'{key} must be a finite scalar')
    for key in LIKELIHOOD_MAX_KEYS:
        if key in data:
            value = data[key]
            if not isinstance(value, list) or len(value) != 2 or not all(finite_number(x) for x in value):
                report.error('LIKELIHOOD',
                             f'{key} must be a two-element array [mu_at_max, negative_log_max_likelihood] of finite numbers')
    if 'contact' in data and not isinstance(data['contact'], dict):
        report.error('CONTACT', 'contact must be a JSON object')
    if 'persistent_id' in data and not isinstance(data['persistent_id'], dict):
        report.error('PERSISTENT_ID', 'persistent_id must be a JSON object')
    if 'signal_unc' in data:
        report.warn('V1_FIELD', 'signal_unc is deferred to OLLL v1.0; v0.1 does not standardize it')
    if 'start method' in data:
        report.warn('LEGACY_KEY', 'Use start_method rather than start method in new models')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('files', nargs='+', type=Path)
    parser.add_argument('--inspection', action='store_true', help='Legacy inspection: missing v0.1 fields are warnings')
    parser.add_argument('--skip-onnx-checker', action='store_true', help='Skip ONNX graph checker (still inspect graph dimensions)')
    parser.add_argument('--json-report', type=Path, help='Write machine-readable validation report')
    args = parser.parse_args(argv)
    reports = []
    for path in args.files:
        report = Report(path)
        try:
            data, model = load_input(path, report, check_graph=not args.skip_onnx_checker)
            validate(data, report, model=model, inspection=args.inspection)
        except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
            report.error('INPUT_FAILURE', str(exc))
        result = report.as_dict()
        reports.append(result)
        print(f'{path}: {result["status"]} ({result["errors"]} errors, {result["warnings"]} warnings)')
        for issue in result['issues']:
            print(f'  [{issue["severity"]}] {issue["code"]}: {issue["message"]}')
    if args.json_report:
        args.json_report.write_text(json.dumps(reports, indent=2) + '\n', encoding='utf-8')
    return 2 if any(any(x['code'] == 'INPUT_FAILURE' for x in r['issues']) for r in reports) else (1 if any(r['errors'] for r in reports) else 0)


if __name__ == '__main__':
    sys.exit(main())
