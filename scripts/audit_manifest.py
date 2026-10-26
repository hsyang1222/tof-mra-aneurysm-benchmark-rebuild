#!/usr/bin/env python3
"""Read-only frozen-manifest audit; optionally check current author TeX sources.

Standard library only. This tool never splits cases, recomputes scanner metadata,
deduplicates source references, or reads image data. JSON reports are aggregate.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from collections import Counter
from pathlib import Path, PurePosixPath


SOURCE_FIELDS = (
    'case_id', 'dataset', 'source_kind', 'source_identity', 'source_subject_id',
    'source_session_id', 'source_acquisition_key', 'source_locator',
    'reference_filename', 'is_positive', 'label_source_kind', 'label_source_locator',
)
PUBLIC_FIELDS = ('manufacturer', 'tesla', 'primary_cohort', 'scanner_id') + tuple(
    f'setting_{setting}_{field}' for setting in 'ABC' for field in ('role', 'split', 'domain_group')
)
ROLE_SPLITS = {'source_train': 'train', 'source_val': 'val', 'id_test': 'test',
               'cs_id_primary': 'cs-test', 'cs_id_etc': 'cs-test'}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        if not fields or len(set(fields)) != len(fields):
            raise ValueError('CSV has missing or duplicate column names')
        rows = list(reader)
    if any(None in row or None in row.values() for row in rows):
        raise ValueError('CSV row width differs from header')
    return fields, rows


def source_digest(rows):
    """Order-sensitive digest of the original source columns, before metadata."""
    values = [[row[field] for field in SOURCE_FIELDS] for row in rows]
    return hashlib.sha256(json.dumps(values, ensure_ascii=True, separators=(',', ':')).encode()).hexdigest()


def subset(rows, setting, role=None, domain=None):
    return [row for row in rows if (role is None or row[f'setting_{setting}_role'] == role)
            and (domain is None or row[f'setting_{setting}_domain_group'] == domain)]


def counts(rows):
    positive = sum(int(row['is_positive']) for row in rows)
    return {'cases': len(rows), 'positive': positive, 'negative': len(rows) - positive}


def aggregate(rows):
    primary = [r for r in rows if r['primary_cohort'] == '1']
    return {
        'counting_unit': 'fixed_manifest_case_row', 'candidate': counts(rows), 'primary': counts(primary),
        'datasets': {dataset: counts([r for r in rows if r['dataset'] == dataset])
                     for dataset in sorted({r['dataset'] for r in rows})},
        'settings': {setting: {
            'roles': {role: counts(subset(rows, setting, role))
                      for role in sorted({r[f'setting_{setting}_role'] for r in rows})},
            'source_train_plus_val': counts([r for r in rows if r[f'setting_{setting}_role'] in ('source_train', 'source_val')]),
            'primary_evaluation_domains': {domain: counts(subset(rows, setting, 'cs_id_primary', domain))
                for domain in sorted({r[f'setting_{setting}_domain_group'] for r in rows if r[f'setting_{setting}_role'] == 'cs_id_primary'})},
        } for setting in 'ABC'},
    }


def validate_rows(fields, rows):
    if tuple(fields[:len(SOURCE_FIELDS)]) != SOURCE_FIELDS or tuple(fields[len(SOURCE_FIELDS):]) != PUBLIC_FIELDS:
        raise ValueError('Unexpected public schema/order')
    ids = [r['case_id'] for r in rows]
    if len(set(ids)) != len(ids):
        raise ValueError('Duplicate manifest case_id')
    for r in rows:
        if r['is_positive'] not in ('0', '1') or r['primary_cohort'] not in ('0', '1'):
            raise ValueError('Invalid binary label or cohort flag')
        if r['dataset'] not in ('ds003949', 'ds005096', 'ixi', 'rsna'):
            raise ValueError('Unknown dataset')
        if r['reference_filename'] != r['case_id'] + '.nii.gz':
            raise ValueError('Case and reference filename differ')
        for field in ('source_locator', 'label_source_locator', 'reference_filename'):
            value = r[field].split('#', 1)[0]
            if value and (PurePosixPath(value).is_absolute() or '..' in PurePosixPath(value).parts
                          or '\\' in value or '://' in value):
                raise ValueError('Locators must be source-root-relative POSIX paths')
        for setting in 'ABC':
            role, split = r[f'setting_{setting}_role'], r[f'setting_{setting}_split']
            if role not in (*ROLE_SPLITS, 'excluded'):
                raise ValueError('Unknown setting role')
            if role in ROLE_SPLITS and split != ROLE_SPLITS[role]:
                raise ValueError('Role/split disagreement')
            if role == 'excluded' and split not in ('excluded', 'cs-test'):
                raise ValueError('Unexpected excluded legacy split')
            if role == 'cs_id_primary' and (r['primary_cohort'] != '1' or r['manufacturer'] not in ('GE', 'Philips', 'Siemens')):
                raise ValueError('Non-primary case in primary evaluation')
            if role == 'cs_id_etc' and (r['manufacturer'] != 'Toshiba' or r['tesla'] not in ('1.5T', '3T')):
                raise ValueError('Unexpected ETC membership')
            if r['tesla'] == 'Unknown' and role != 'excluded':
                if setting != 'C' or r['manufacturer'] != 'Siemens' or role not in ('source_train', 'source_val', 'id_test'):
                    raise ValueError('Unknown field strength leaked outside Setting-C source')
            if role in ('source_train', 'source_val', 'id_test'):
                if setting == 'A' and r['tesla'] != '1.5T':
                    raise ValueError('Setting-A source Tesla mismatch')
                if setting == 'B' and r['tesla'] != '3T':
                    raise ValueError('Setting-B source Tesla mismatch')
                if setting == 'C' and r['manufacturer'] != 'Siemens':
                    raise ValueError('Setting-C source manufacturer mismatch')


def tex_table(tex, label):
    marker = '\\label{' + label + '}'
    if tex.count(marker) != 1:
        raise ValueError('Expected one manuscript table label: ' + label)
    return tex.split(marker, 1)[1].split('\\end{tabular', 1)[0]


def tex_rows(tex, label):
    for line in tex_table(tex, label).splitlines():
        if '&' in line and line.strip().endswith('\\\\'):
            yield [c.strip() for c in line.strip()[:-2].split('&')]


def integer(value):
    return int(value.replace(',', ''))


def unwrap(cell):
    return re.sub(r'^\\multirow\{\d+\}\{\*\}\{(.*?)\}$', r'\1', cell)


def manuscript_checks(rows, paper, supplement):
    """Parse current TeX; no dependence on an earlier check CSV or ledger."""
    checks = []
    extra_checks = []

    def check(location, field, expected, actual):
        checks.append({'location': location, 'field': field, 'manuscript': expected,
                       'manifest': actual, 'match': str(expected) == str(actual)})

    dataset = None
    table1_rows = 0
    for cells in tex_rows(paper, 'tab:detailed_stats'):
        if len(cells) != 10 or cells[1] not in ('1.5T', '3.0T'):
            continue
        if cells[0]:
            dataset = unwrap(cells[0]).lower()
        tesla = cells[1].replace('3.0T', '3T')
        group = [r for r in rows if r['dataset'] == dataset and r['tesla'] == tesla and r['primary_cohort'] == '1']
        actual = counts(group)
        for name, value in zip(('positive', 'negative', 'cases'), cells[2:5]):
            check(f'Main Table1 {dataset} {tesla}', name, integer(value), actual[name])
        for name, value in zip(('GE', 'Philips', 'Siemens', 'Toshiba'), cells[5:9]):
            check(f'Main Table1 {dataset} {tesla}', name, 0 if value == '---' else integer(value), sum(r['manufacturer'] == name for r in group))
        check(f'Main Table1 {dataset} {tesla}', 'subtotal', integer(cells[9]), len(group))
        table1_rows += 1
    if table1_rows != 8:
        raise ValueError('Incomplete main dataset table parse')

    cohort_rows = 0
    for cells in tex_rows(supplement, 'tab:supp_cohort_accounting'):
        if len(cells) == 4 and cells[0].lower() in ('ds003949', 'ds005096', 'ixi', 'rsna'):
            group = [r for r in rows if r['dataset'] == cells[0].lower()]
            check('Supp Table2 ' + cells[0], 'candidate', integer(cells[1]), len(group))
            check('Supp Table2 ' + cells[0], 'primary cohort', integer(cells[2]), sum(r['primary_cohort'] == '1' for r in group))
            cohort_rows += 1
    if cohort_rows != 4:
        raise ValueError('Incomplete cohort accounting parse')
    match = re.search(r'primary known-scanner cohort contains ([\d,]+) cases, including ([\d,]+) positives and ([\d,]+) negatives', paper)
    if not match:
        raise ValueError('Primary cohort narrative not found')
    primary = counts([r for r in rows if r['primary_cohort'] == '1'])
    for name, value in zip(('cases', 'positive', 'negative'), match.groups()):
        check('Main primary cohort narrative', name, integer(value), primary[name])

    setting = None
    split_entries = 0
    for cells in tex_rows(supplement, 'tab:supp_split_definition'):
        if len(cells) != 9 or cells[1] not in ('Source train', 'Source val', 'ID-test', 'cs-ID'):
            continue
        if cells[0]:
            setting = unwrap(cells[0])
        role = {'Source train': 'source_train', 'Source val': 'source_val', 'ID-test': 'id_test'}.get(cells[1], 'cs_id_etc' if cells[2].startswith('ETC') else 'cs_id_primary')
        group = subset(rows, setting, role, cells[2])
        actual = counts(group)
        for name, value in zip(('cases', 'positive', 'negative'), cells[4:7]):
            check(f'Supp Table3 {setting} {role} {cells[2]}', name, integer(value), actual[name])
        extra_checks.append({'location': f'Supp Table3 {setting} {role} {cells[2]}',
                             'field': 'split', 'manuscript': cells[3],
                             'manifest': '|'.join(sorted({r[f'setting_{setting}_split'] for r in group})),
                             'match': {r[f'setting_{setting}_split'] for r in group} == {cells[3]}})
        split_entries += 1
    if split_entries != 23:
        raise ValueError('Incomplete detailed split table parse')

    composition_entries = 0
    for cells in tex_rows(supplement, 'tab:supp_csid_composition_audit'):
        if len(cells) != 7 or cells[0] not in ('A', 'B', 'C'):
            continue
        group = subset(rows, cells[0], 'cs_id_primary', cells[1])
        location = f'Supp Table4 {cells[0]} {cells[1]}'
        check(location, 'cases', integer(cells[3]), len(group))
        check(location, 'positive', integer(cells[4]), counts(group)['positive'])
        check(location, 'scanner IDs', integer(cells[5]), len({r['scanner_id'] for r in group}))
        for component in cells[6].split(';'):
            dataset, value = component.strip().rsplit(' ', 1)
            check(location, 'dataset ' + dataset, integer(value), sum(r['dataset'] == dataset.lower() for r in group))
        composition_entries += 1
    if composition_entries != 10:
        raise ValueError('Incomplete composition table parse')

    fingerprints = 0
    for cells in tex_rows(supplement, 'tab:supp_episode_fingerprint'):
        if cells[0] in ('A', 'B', 'C'):
            n = sum(r[f'setting_{cells[0]}_role'] in ('source_train', 'source_val') for r in rows)
            check('Supp source fingerprint ' + cells[0], 'full-source train+val', integer(cells[1]), n)
            fingerprints += 1
    if fingerprints != 3:
        raise ValueError('Incomplete source count parse')

    match = re.search(r'Group sizes \(subjects/positive subjects\):(.*?)ID-test sizes \(A/B/C\): ([^\n]+)', paper, re.S)
    if not match:
        raise ValueError('Main group denominator footnote not found')
    for domain, n, positive in re.findall(r'(GE 3T|Philips 3T|Siemens 3T|GE 1\.5T|Philips 1\.5T|Siemens 1\.5T), (\d+)/(\d+)', match[1]):
        setting = 'A' if domain.endswith(' 3T') else 'B'
        group = subset(rows, setting, 'cs_id_primary', domain)
        check('Main result footnote ' + domain, 'cases', int(n), len(group))
        check('Main result footnote ' + domain, 'positive', int(positive), counts(group)['positive'])
    id_values = re.findall(r'(\d+)/(\d+)', match[2])
    if len(id_values) != 3:
        raise ValueError('Main ID-test denominator footnote incomplete')
    for setting, (n, positive) in zip('ABC', id_values):
        group = subset(rows, setting, 'id_test')
        check('Main ID-test footnote ' + setting, 'cases', int(n), len(group))
        check('Main ID-test footnote ' + setting, 'positive', int(positive), counts(group)['positive'])

    delta_entries = 0
    for cells in tex_rows(supplement, 'tab:supp_main_delta_counts'):
        if len(cells) != 7:
            continue
        setting, domain, pair = map(unwrap, cells[:3])
        if setting not in ('A', 'B', 'C'):
            continue
        if domain == 'ID-test':
            group = subset(rows, setting, 'id_test')
        else:
            domain += ' 3T' if setting == 'A' else ' 1.5T' if setting == 'B' else ''
            group = subset(rows, setting, 'cs_id_primary', domain)
        check(f'Supp comparison denominator {setting} {domain}', 'N only (not lesions)', int(pair.split('/')[0]), len(group))
        delta_entries += 1
    if delta_entries != 13:
        raise ValueError('Incomplete comparison denominator table parse')

    unknown = [r for r in rows if r['manufacturer'] == 'Siemens' and r['tesla'] == 'Unknown']
    match = re.search(r'Setting C retains (\d+) Siemens cases with unknown field strength.*?\((\d+) train, (\d+) validation, (\d+) ID-test\)', supplement, re.S)
    if not match:
        raise ValueError('Siemens unknown-field narrative not found')
    for role, value in zip(('source_train', 'source_val', 'id_test'), match.groups()[1:]):
        check('Supp unknown-field source split', role, int(value), len(subset(unknown, 'C', role)))
    vendor = '|'.join(sorted({r['manufacturer'] for r in rows if r['tesla'] == '1.2T'}))
    narrative = re.search(r'The one (\w+) 1\.2T case is excluded', supplement)
    table = re.search(r'; 1 (\w+) 1\.2T case is excluded', tex_table(supplement, 'tab:supp_cohort_accounting'))
    if not narrative or not table:
        raise ValueError('Excluded manufacturer narrative not found')
    check('Supp excluded 1.2T narrative', 'manufacturer', narrative[1], vendor)
    check('Supp excluded 1.2T table', 'manufacturer', table[1], vendor)
    # Above coverage equals the previous 241-item scope, freshly parsed here.
    if len(checks) != 241:
        raise ValueError(f'Expected 241 subject/metadata checks, parsed {len(checks)}')
    checks.extend(extra_checks)

    # Extra assertions for totals, exclusions, legacy split semantics and source roles.
    check('Supp unknown-field source total', 'cases', int(match[1]), len(unknown))
    total_row = next(c for c in tex_rows(supplement, 'tab:supp_cohort_accounting') if c[0] == 'Total')
    check('Supp cohort accounting total', 'candidate', integer(total_row[1]), len(rows))
    check('Supp cohort accounting total', 'primary', integer(total_row[2]), primary['cases'])
    excluded_match = re.search(r'(\d+) cases are excluded from all study settings', total_row[3])
    if not excluded_match:
        raise ValueError('All-settings excluded count not found')
    excluded_all = int(excluded_match[1])
    for setting in 'ABC':
        expected = excluded_all + (len(unknown) if setting != 'C' else 0)
        check('Frozen excluded role ' + setting, 'cases', expected, len(subset(rows, setting, 'excluded')))
    check('Never enters any setting', 'excluded all settings', excluded_all,
          sum(all(r[f'setting_{s}_role'] == 'excluded' for s in 'ABC') for r in rows))
    total_main = next(c for c in tex_rows(paper, 'tab:detailed_stats') if c[0] == '\\textbf{Total}')
    number = lambda cell: integer(re.sub(r'\\textbf\{([^}]+)\}', r'\1', cell))
    for name, value in zip(('positive', 'negative', 'cases'), total_main[2:5]):
        check('Main Table1 total', name, number(value), primary[name])
    for vendor, cell in zip(('GE', 'Philips', 'Siemens', 'Toshiba'), total_main[5:9]):
        check('Main Table1 total', vendor, number(cell), sum(r['manufacturer'] == vendor and r['primary_cohort'] == '1' for r in rows))
    check('Main Table1 total', 'subtotal', number(total_main[9]), primary['cases'])
    rsna_narrative = next(c[3] for c in tex_rows(supplement, 'tab:supp_cohort_accounting') if c[0] == 'RSNA')
    toshiba = re.search(r'(\d+) Toshiba 1\.5T/3T cases', rsna_narrative)
    if not toshiba:
        raise ValueError('Toshiba role narrative not found')
    check('Supp Toshiba source/ETC narrative', 'known-field cases', int(toshiba[1]), sum(r['manufacturer'] == 'Toshiba' and r['tesla'] in ('1.5T', '3T') for r in rows))
    check('Supp excluded 1.2T narrative', 'one case', 1, sum(r['tesla'] == '1.2T' for r in rows))
    return checks


def audit(manifest_dir, paper=None, supplement=None):
    directory = Path(manifest_dir)
    version = json.loads((directory / 'manifest_version.json').read_text())
    for name, digest in version['files_sha256'].items():
        if name not in ('authoritative_source_manifest.csv', 'smoke16_seed20260810.csv', 'mra_validation_contract.csv') or sha256(directory / name) != digest:
            raise ValueError('Frozen file hash mismatch: ' + name)
    fields, rows = read_csv(directory / 'authoritative_source_manifest.csv')
    validate_rows(fields, rows)
    if source_digest(rows) != version['source_columns_sha256']:
        raise ValueError('Original source columns were modified')
    cfields, contract = read_csv(directory / 'mra_validation_contract.csv')
    if len({r['reference_filename'] for r in contract}) != len(rows) or {r['reference_filename'] for r in contract} != {r['reference_filename'] for r in rows}:
        raise ValueError('Validation contract identities differ')
    sfields, smoke = read_csv(directory / 'smoke16_seed20260810.csv')
    if sfields != list(SOURCE_FIELDS) + ['selection_rank', 'selection_seed'] + list(PUBLIC_FIELDS):
        raise ValueError('Unexpected smoke schema')
    keyed = {r['case_id']: r for r in rows}
    if len(smoke) != 16 or len({r['case_id'] for r in smoke}) != 16 or Counter(r['dataset'] for r in smoke) != dict.fromkeys(('rsna', 'ixi', 'ds003949', 'ds005096'), 4):
        raise ValueError('Smoke selection membership changed')
    for row in smoke:
        if any(row[field] != keyed[row['case_id']][field] for field in (*SOURCE_FIELDS, *PUBLIC_FIELDS)):
            raise ValueError('Smoke source or metadata differs from full manifest')
    selection = [[r['case_id'], r['selection_rank'], r['selection_seed']] for r in smoke]
    selection_hash = hashlib.sha256(json.dumps(selection, separators=(',', ':')).encode()).hexdigest()
    if selection_hash != version['smoke_selection_sha256']:
        raise ValueError('Smoke selection/rank/seed modified')
    summary = aggregate(rows)
    if summary != version['aggregate_counts']:
        raise ValueError('Frozen aggregate count mismatch')
    checks = []
    if (paper is None) != (supplement is None):
        raise ValueError('Supply both manuscript sources or neither')
    if paper is not None:
        checks = manuscript_checks(rows, Path(paper).read_text(), Path(supplement).read_text())
    return {'status': 'pass' if all(c['match'] for c in checks) else 'fail',
            'version': version['version'], 'aggregate_counts': summary,
            'frozen_files_verified': True, 'original_columns_verified': True,
            'smoke_cases_verified': len(smoke), 'validation_contract_cases': len(contract),
            'manuscript_checks': len(checks), 'manuscript_matches': sum(c['match'] for c in checks),
            'manuscript_differences': [c for c in checks if not c['match']], 'checks': checks,
            'not_asserted': ['unique patients', 'source asset deduplication', 'lesion counts', 'performance metrics']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest-dir', type=Path, default=Path(__file__).resolve().parents[1] / 'manifests')
    parser.add_argument('--paper', type=Path)
    parser.add_argument('--supplement', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    try:
        report = audit(args.manifest_dir, args.paper, args.supplement)
    except (ValueError, KeyError) as error:
        report = {'status': 'fail', 'error': str(error)}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({key: value for key, value in report.items() if key not in ('checks', 'aggregate_counts')}, indent=2))
    return 0 if report['status'] == 'pass' else 1


if __name__ == '__main__':
    raise SystemExit(main())
