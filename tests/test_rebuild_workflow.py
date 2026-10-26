"""CPU-only integration checks using synthetic images; no benchmark data required.

Run with: python -m unittest discover -s tests -p 'test_rebuild_workflow.py' -v
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import nibabel as nib
import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "rebuild_full_manifest.py"


def write_csv(path, rows, fields=None):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class WorkflowTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="rebuild_fixture_")
        self.root = Path(self.tmp.name)
        self.sources = self.root / "source"
        self.sources.mkdir()
        self.manifest = self.root / "manifest.csv"
        self.contract = self.root / "contract.csv"
        self.rows, self.contract_rows = [], []

    def tearDown(self):
        self.tmp.cleanup()

    def case(self, name, *, dataset="ixi", positive=False, shape=(24, 24, 24), expected_shape=None,
             expected_spacing="1x1x1", missing=False, corrupt=False):
        locator = name + ".nii.gz"
        source = self.sources / locator
        label_locator = name + "_label.nii.gz" if positive else ""
        if corrupt:
            source.write_text("not a NIfTI image")
        elif not missing:
            data = np.arange(np.prod(shape), dtype=np.float32).reshape(shape)
            nib.save(nib.Nifti1Image(data, np.eye(4)), source)
            if positive:
                mask = np.zeros(shape, dtype=np.int16)
                mask[2:5, 2:5, 2:5] = 1
                nib.save(nib.Nifti1Image(mask, np.eye(4)), self.sources / label_locator)
        self.rows.append({"case_id": name, "dataset": dataset, "source_locator": locator,
                          "label_source_locator": label_locator, "is_positive": str(int(positive)),
                          "label_source_kind": "bids_label_nifti" if positive else "zero_label",
                          "source_identity": name, "reference_filename": name + ".nii.gz"})
        expected_shape = expected_shape or shape
        self.contract_rows.append({"reference_filename": name + ".nii.gz",
                                   "shape_xyz": "x".join(map(str, expected_shape)),
                                   "spacing_xyz": expected_spacing,
                                   "voxel_count": str(np.prod(expected_shape))})

    def save_inputs(self):
        write_csv(self.manifest, self.rows)
        write_csv(self.contract, self.contract_rows)

    def run_cli(self, output="output", extra=(), workers=1, expected=0):
        self.save_inputs()
        output_path = self.root / output
        command = [sys.executable, str(SCRIPT), "--manifest-csv", str(self.manifest),
                   "--mra-contract-csv", str(self.contract), "--output-root", str(output_path),
                   "--work-root", str(self.root / "work"), "--num-workers", str(workers)]
        for dataset in ("ixi", "ds003949", "ds005096", "rsna"):
            command += ["--" + dataset + "-root", str(self.sources)]
        command += list(extra)
        environment = dict(os.environ, CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
        completed = subprocess.run(command, text=True, capture_output=True, env=environment)
        self.assertEqual(completed.returncode, expected, completed.stdout + completed.stderr)
        report = output_path / "rebuild_report.json"
        return (json.loads(report.read_text()) if report.exists() else None), completed

    def test_serial_parallel_equivalence_and_real_process_overlap(self):
        for index in range(8):
            self.case("case_" + str(index), positive=index % 2 == 1, shape=(48, 48, 48))
        serial, _ = self.run_cli(output="serial")
        parallel, _ = self.run_cli(output="parallel", workers=2)
        self.assertTrue(serial["input_manifest_verified"])
        self.assertTrue(parallel["input_manifest_verified"])
        self.assertFalse(serial["full_benchmark_verified"])
        self.assertFalse(parallel["full_benchmark_verified"])
        self.assertEqual(parallel["manifest_scope"], "custom_or_subset")
        run = self.root / "parallel" / "runs" / parallel["run_id"]
        with (run / "cases.csv").open() as handle:
            records = list(csv.DictReader(handle))
        self.assertGreaterEqual(len({record["pid"] for record in records}), 2)
        overlaps = [(left, right) for left in records for right in records
                    if left["pid"] != right["pid"] and float(left["started"]) < float(right["finished"])
                    and float(right["started"]) < float(left["finished"])]
        self.assertTrue(overlaps, "Separate worker PIDs must actually overlap")
        print("PARALLEL_EVIDENCE", json.dumps({"pids": sorted({r["pid"] for r in records}),
                                              "overlapping_pairs": len(overlaps), "cases": len(records)}))
        for row in self.rows:
            for kind in ("MRA", "AN"):
                left = nib.load(self.root / "serial" / kind / (row["case_id"] + ".nii.gz"))
                right = nib.load(self.root / "parallel" / kind / (row["case_id"] + ".nii.gz"))
                np.testing.assert_array_equal(np.asarray(left.dataobj), np.asarray(right.dataobj))
                np.testing.assert_array_equal(left.affine, right.affine)

    def test_failures_warnings_and_other_datasets_continue(self):
        self.case("normal")
        self.case("missing", missing=True)
        self.case("corrupt", corrupt=True)
        self.case("warning", expected_spacing="2x2x2")
        self.case("bad_geometry", expected_shape=(30, 30, 30))
        self.case("rsna_missing_localizers", dataset="rsna", missing=True)
        report, completed = self.run_cli(expected=1, workers=2)
        self.assertIn("todo=6, fail=0, done=0, warning=0", completed.stderr)
        self.assertIn("todo=0, fail=5, done=1, warning=1", completed.stderr)
        self.assertEqual(report["selected_validation"], {"pass": 1, "warning": 1, "fail": 4})
        self.assertTrue(report["execution_completed"])
        self.assertFalse(report["full_benchmark_verified"])
        run = self.root / "output" / "runs" / report["run_id"]
        failures = [json.loads(line) for line in (run / "failures.jsonl").read_text().splitlines()]
        rsna = next(row for row in failures if row["case_id"] == "rsna_missing_localizers")
        self.assertIn("localizer CSV", rsna["error"])
        self.assertIn("Traceback", rsna["traceback"])
        with (run / "retry_cases.csv").open() as handle:
            self.assertEqual(len(list(csv.DictReader(handle))), 5)
        self.assertFalse((self.root / "output" / "MRA" / "bad_geometry.nii.gz").exists())
        self.assertFalse((self.root / "output" / "AN" / "bad_geometry.nii.gz").exists())

    def test_manual_override_subset_resume_and_manifest_change(self):
        self.case("normal")
        self.case("pending", missing=True)
        subset = self.root / "selected.csv"
        write_csv(subset, [{"case_id": "normal"}])
        report, _ = self.run_cli(extra=["--case-list-csv", str(subset)])
        self.assertEqual(report["pending_count"], 1)
        self.assertFalse(report["full_benchmark_verified"])
        report, _ = self.run_cli(extra=["--case-list-csv", str(subset)])
        state = json.loads((self.root / "output" / "cumulative_state.json").read_text())
        self.assertTrue(state["normal"]["resumed"])
        override = self.root / "override.csv"
        write_csv(override, [{"case_id": "pending", "source_locator": "normal.nii.gz"}])
        write_csv(subset, [{"case_id": "pending"}])
        report, _ = self.run_cli(extra=["--case-list-csv", str(subset), "--overrides-csv", str(override)])
        self.assertEqual(report["selected_validation"], {"pass": 1})
        self.assertFalse(report["full_benchmark_verified"], "Subset cannot claim a full benchmark pass")
        # Manifest identity change makes existing outputs non-resumable and clears verification.
        self.rows[0]["source_identity"] = "updated_source_release"
        write_csv(subset, [{"case_id": "normal"}])
        report, _ = self.run_cli(extra=["--case-list-csv", str(subset)], expected=1)
        state = json.loads((self.root / "output" / "cumulative_state.json").read_text())
        self.assertFalse(state["normal"]["verified"])
        report, _ = self.run_cli(extra=["--case-list-csv", str(subset), "--retry"])
        self.assertEqual(report["selected_validation"], {"pass": 1})
        # A failed explicit retry may leave historical bytes but must never leave a verified marker.
        (self.sources / "normal.nii.gz").unlink()
        self.run_cli(extra=["--case-list-csv", str(subset), "--retry"], expected=1)
        state = json.loads((self.root / "output" / "cumulative_state.json").read_text())
        self.assertFalse(state["normal"]["verified"])

    def test_saved_pair_tampering_requires_explicit_retry(self):
        self.case("normal")
        self.run_cli()
        path = self.root / "output" / "MRA" / "normal.nii.gz"
        image = nib.load(path)
        nib.save(nib.Nifti1Image(np.asarray(image.dataobj) + 1, image.affine), path)
        self.run_cli(expected=1)
        state = json.loads((self.root / "output" / "cumulative_state.json").read_text())
        self.assertFalse(state["normal"]["verified"])
        self.assertIn("hashes changed", state["normal"]["error"])
        report, _ = self.run_cli(extra=["--retry"])
        self.assertTrue(report["input_manifest_verified"])
        self.assertFalse(report["full_benchmark_verified"])

    def test_immutable_override_and_traversal_rejected(self):
        self.case("normal")
        override = self.root / "override.csv"
        write_csv(override, [{"case_id": "normal", "is_positive": "1"}])
        _, completed = self.run_cli(extra=["--overrides-csv", str(override)], expected=1)
        self.assertIn("Overrides may contain only", completed.stderr)
        write_csv(override, [{"case_id": "normal", "source_locator": "../secret.nii.gz"}])
        _, completed = self.run_cli(extra=["--overrides-csv", str(override)], expected=1)
        self.assertIn("cannot contain", completed.stderr)

    def test_positive_empty_label_and_affine_mismatch_fail(self):
        self.case("empty_positive", positive=True)
        self.case("different_grid", positive=True)
        image = nib.load(self.sources / "empty_positive_label.nii.gz")
        nib.save(nib.Nifti1Image(np.zeros(image.shape, dtype=np.int16), image.affine), self.sources / "empty_positive_label.nii.gz")
        image = nib.load(self.sources / "different_grid_label.nii.gz")
        affine = image.affine.copy()
        affine[0, 3] = 10
        nib.save(nib.Nifti1Image(np.asarray(image.dataobj), affine), self.sources / "different_grid_label.nii.gz")
        report, _ = self.run_cli(expected=1)
        self.assertEqual(report["selected_validation"], {"fail": 2})
        state = json.loads((self.root / "output" / "cumulative_state.json").read_text())
        self.assertIn("positivity", state["empty_positive"]["error"])
        self.assertIn("affine mismatch", state["different_grid"]["error"])


if __name__ == "__main__":
    unittest.main()
