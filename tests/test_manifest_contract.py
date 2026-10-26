"""Standard-library regression tests for the public frozen-manifest contract."""
import csv
import importlib.util
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import audit_manifest as audit


class FrozenManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name) / 'manifests'
        shutil.copytree(ROOT / 'manifests', self.directory)

    def tearDown(self):
        self.temp.cleanup()

    def modify(self, name, mutate, refresh_hash=False):
        path = self.directory / name
        fields, rows = audit.read_csv(path)
        mutate(rows)
        with path.open('w', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, lineterminator='\n')
            writer.writeheader()
            writer.writerows(rows)
        if refresh_hash:
            metadata = json.loads((self.directory / 'manifest_version.json').read_text())
            metadata['files_sha256'][name] = audit.sha256(path)
            (self.directory / 'manifest_version.json').write_text(json.dumps(metadata))

    def test_public_audit_needs_no_private_ledger_or_source_images(self):
        result = audit.audit(self.directory)
        self.assertEqual(result['status'], 'pass')
        self.assertEqual(result['validation_contract_cases'], 2085)
        self.assertEqual(result['aggregate_counts']['primary'], {'cases': 2057, 'positive': 660, 'negative': 1397})

    def test_original_source_columns_cannot_change_even_after_file_hash_refresh(self):
        self.modify('authoritative_source_manifest.csv', lambda rows: rows[0].update(source_locator='other/scan.nii.gz'), True)
        with self.assertRaisesRegex(ValueError, 'Original source columns'):
            audit.audit(self.directory)

    def test_any_unfrozen_csv_edit_fails_hash(self):
        self.modify('authoritative_source_manifest.csv', lambda rows: rows[0].update(manufacturer='GE'))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            audit.audit(self.directory)

    def test_unknown_field_cases_cannot_enter_primary_evaluation(self):
        def leak(rows):
            row = next(r for r in rows if r['tesla'] == 'Unknown' and r['manufacturer'] == 'Siemens')
            row.update(setting_C_role='cs_id_primary', setting_C_split='cs-test')
        self.modify('authoritative_source_manifest.csv', leak, True)
        with self.assertRaisesRegex(ValueError, 'Non-primary case|Unknown field'):
            audit.audit(self.directory)

    def test_legacy_cs_test_value_does_not_make_excluded_cases_eligible(self):
        _, rows = audit.read_csv(self.directory / 'authoritative_source_manifest.csv')
        legacy = [r for r in rows if r['setting_C_role'] == 'excluded' and r['setting_C_split'] == 'cs-test']
        self.assertEqual(len(legacy), 8)
        self.assertTrue(all(r not in audit.subset(rows, 'C', 'cs_id_primary') for r in legacy))

    def test_same_smoke_ids_but_changed_seed_is_rejected(self):
        self.modify('smoke16_seed20260810.csv', lambda rows: rows[0].update(selection_seed='1'), True)
        with self.assertRaisesRegex(ValueError, 'rank/seed'):
            audit.audit(self.directory)

    def test_absolute_locator_is_rejected_without_touching_raw_metadata(self):
        fields, rows = audit.read_csv(self.directory / 'authoritative_source_manifest.csv')
        rows[0]['source_locator'] = '/private/scanner/file.nii.gz'
        with self.assertRaisesRegex(ValueError, 'relative POSIX'):
            audit.validate_rows(fields, rows)

    def test_smoke_metadata_must_equal_full_manifest(self):
        self.modify('smoke16_seed20260810.csv', lambda rows: rows[0].update(tesla='1.2T'), True)
        with self.assertRaisesRegex(ValueError, 'Smoke source or metadata'):
            audit.audit(self.directory)

    def test_source_train_plus_val_counts_use_roles_not_all_source_cases(self):
        result = audit.audit(self.directory)
        actual = [result['aggregate_counts']['settings'][s]['source_train_plus_val']['cases'] for s in 'ABC']
        self.assertEqual(actual, [975, 873, 822])


if __name__ == '__main__':
    unittest.main()
