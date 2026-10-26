#!/usr/bin/env python3
"""Rebuild all benchmark pairs from one fixed full-source manifest.

Each upstream source is resolved locally through its own acquisition and label
route.  All resolved pairs then share one RAS NIfTI writer: MRA is float32 and
AN is binary int16.  MRA geometry is checked against the committed contract; label checks cover
pair geometry, binary values and manifest positivity, not annotation accuracy.
Failures and warnings require local researcher review; datasets are not silently
pruned and benchmark metadata is never rewritten.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import hashlib
import os
import re
import time
import traceback
from datetime import datetime, timezone
from concurrent.futures import ProcessPoolExecutor, as_completed
from collections import Counter
from pathlib import Path

import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
import torchio as tio

from pair_io import save_canonical_pair
from tqdm import tqdm


def rows_by_dataset(path: Path) -> dict[str, list[dict[str, str]]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {"rsna", "ixi", "ds003949", "ds005096"}
    unexpected = {row["dataset"] for row in rows} - expected
    if unexpected:
        raise SystemExit(f"Unsupported dataset values: {sorted(unexpected)}")
    grouped = {dataset: [] for dataset in expected}
    for row in rows:
        grouped[row["dataset"]].append(row)
    return grouped


def nifti_subject(row: dict[str, str], source_root: Path) -> tio.Subject:
    image = tio.ScalarImage(source_root / row["source_locator"])
    route = row["label_source_kind"]
    if route == "zero_label":
        label = tio.LabelMap(tensor=torch.zeros_like(image.data, dtype=torch.int16), affine=image.affine)
        return tio.Subject(case_id=row["case_id"], route=route, mra=image, label=label)
    if route == "bids_label_nifti":
        label = tio.LabelMap(source_root / row["label_source_locator"])
        return tio.Subject(case_id=row["case_id"], route=route, mra=image, label=label)
    if route == "burn_in_difference":
        burn_in = tio.ScalarImage(source_root / row["label_source_locator"])
        return tio.Subject(case_id=row["case_id"], route=route, mra=image, burn_in=burn_in)
    raise RuntimeError(f"{row['case_id']}: unsupported NIfTI route {route}")


def localizers_by_series(rsna_root: Path) -> dict[str, list[dict[str, str]]]:
    path = rsna_root / "train_localizers.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing RSNA localizer CSV: {path}")
    result: dict[str, list[dict[str, str]]] = {}
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            result.setdefault(row["SeriesInstanceUID"], []).append(row)
    return result


def rsna_sop_index(filenames: list[str]) -> dict[str, int]:
    reader = sitk.ImageFileReader()
    reader.SetGlobalWarningDisplay(False)
    result: dict[str, int] = {}
    for index, filename in enumerate(filenames):
        reader.SetFileName(filename)
        reader.ReadImageInformation()
        result[reader.GetMetaData("0008|0018")] = index
    return result


def draw_sphere(mask: np.ndarray, x: float, y: float, z: int, radius: int = 5) -> None:
    z0, z1 = max(0, z - radius), min(mask.shape[0], z + radius + 1)
    y0, y1 = max(0, int(y) - radius), min(mask.shape[1], int(y) + radius + 1)
    x0, x1 = max(0, int(x) - radius), min(mask.shape[2], int(x) + radius + 1)
    zz, yy, xx = np.ogrid[z0:z1, y0:y1, x0:x1]
    mask[z0:z1, y0:y1, x0:x1] |= (xx - x) ** 2 + (yy - y) ** 2 + (zz - z) ** 2 <= radius**2


def rsna_subject(row: dict[str, str], rsna_root: Path, localizers: dict[str, list[dict[str, str]]], work_root: Path) -> tio.Subject:
    series_dir = rsna_root / row["source_locator"]
    if not series_dir.is_dir():
        raise FileNotFoundError(f"Missing RSNA series for {row['case_id']}: {series_dir}")
    reader = sitk.ImageSeriesReader()
    reader.SetGlobalWarningDisplay(False)
    filenames = list(reader.GetGDCMSeriesFileNames(str(series_dir)))
    if not filenames:
        raise RuntimeError(f"{row['case_id']}: no DICOM files")
    reader.SetFileNames(filenames)
    native_image = reader.Execute()
    route = row["label_source_kind"]
    annotations = localizers.get(row["source_identity"], [])
    if route == "roi_sphere_localizer" and not annotations:
        raise RuntimeError(f"{row['case_id']}: positive RSNA case has no localizer")
    if route == "zero_label" and annotations:
        raise RuntimeError(f"{row['case_id']}: zero-label RSNA case has localizers")
    mask = np.zeros(sitk.GetArrayViewFromImage(native_image).shape, dtype=np.uint8)
    if route == "roi_sphere_localizer":
        sop_index = rsna_sop_index(filenames)
        for annotation in annotations:
            if annotation["SOPInstanceUID"] not in sop_index:
                raise RuntimeError(f"{row['case_id']}: localizer SOP is absent from series")
            point = ast.literal_eval(annotation["coordinates"])
            draw_sphere(mask, float(point["x"]), float(point["y"]), sop_index[annotation["SOPInstanceUID"]])
    elif route != "zero_label":
        raise RuntimeError(f"{row['case_id']}: unsupported RSNA route {route}")
    label_image = sitk.GetImageFromArray(mask)
    label_image.CopyInformation(native_image)
    label_path = work_root / "rsna_native_labels" / f"{row['case_id']}.nii.gz"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    sitk.WriteImage(label_image, str(label_path))
    return tio.Subject(case_id=row["case_id"], route=route, mra=tio.ScalarImage(series_dir), label=tio.LabelMap(label_path))


def add_burn_in_label(subject: tio.Subject) -> None:
    if subject["mra"].spatial_shape != subject["burn_in"].spatial_shape:
        raise RuntimeError(f"{subject['case_id']}: MRA and burn-in shape mismatch")
    lesion = ((subject["mra"].data != 0) & (subject["burn_in"].data == 0)).to(dtype=torch.int16)
    subject.add_image(tio.LabelMap(tensor=lesion, affine=subject["mra"].affine), "label")


def read_contract(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    names = [row["reference_filename"] for row in rows]
    if not rows or len(set(names)) != len(names):
        raise ValueError("MRA contract must contain unique reference filenames and cannot be empty")
    return dict(zip(names, rows))


def validate_mra(path: Path, contract: dict[str, dict[str, str]]) -> tuple[str, list[str]]:
    expected = contract.get(path.name)
    if expected is None:
        return "fail", ["missing_contract_row"]
    image = nib.load(str(path))
    actual_shape = tuple(image.shape[:3])
    actual_spacing = tuple(float(value) for value in image.header.get_zooms()[:3])
    actual_voxel_count = int(np.prod(actual_shape))
    expected_shape = tuple(int(value) for value in expected["shape_xyz"].split("x"))
    expected_spacing = tuple(float(value) for value in expected["spacing_xyz"].split("x"))
    expected_voxel_count = int(expected["voxel_count"])
    mismatches: list[str] = []
    if actual_shape != expected_shape:
        mismatches.append("shape")
    if not np.allclose(actual_spacing, expected_spacing, rtol=0, atol=1e-3):
        mismatches.append("spacing")
    if actual_voxel_count != expected_voxel_count:
        mismatches.append("voxel_count")
    return ("pass" if not mismatches else "warning" if len(mismatches) == 1 else "fail"), mismatches


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_locator(root: Path, locator: str) -> str:
    relative = Path(locator)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Source locator must be relative and cannot contain '..'")
    resolved = (root / relative).resolve()
    if not resolved.is_relative_to(root.resolve()):
        raise ValueError("Source locator resolves outside its original dataset root")
    return locator


def validate_pair(image_path: Path, label_path: Path, positive: str) -> None:
    image, label = nib.load(str(image_path)), nib.load(str(label_path))
    if image.shape != label.shape or not np.allclose(image.affine, label.affine):
        raise RuntimeError("output image--label geometry mismatch")
    label_values = np.asarray(label.dataobj)
    if not np.isin(label_values, [0, 1]).all():
        raise RuntimeError("output label is not binary")
    if bool(np.any(label_values)) != (positive == "1"):
        raise RuntimeError("output label presence disagrees with fixed manifest positivity")
    if nib.aff2axcodes(image.affine) != ("R", "A", "S"):
        raise RuntimeError("output pair is not canonical RAS")


def case_worker(task: dict) -> dict:
    # Each worker owns loading, canonicalization, writing and validation for one case.
    torch.set_num_threads(1)
    started = time.time()
    row, root = task["row"], Path(task["source_root"])
    case_id = row["case_id"]
    output = Path(task["output_root"])
    image_path, label_path = output / "MRA" / f"{case_id}.nii.gz", output / "AN" / f"{case_id}.nii.gz"
    result = {"case_id": case_id, "dataset": row["dataset"], "status": "fail",
              "fields": "", "error": "", "pid": os.getpid(), "started": started,
              "fingerprint": task["fingerprint"], "verified": False, "resumed": False}
    staging = output / ".staging" / task["run_id"] / case_id
    try:
        previous = task.get("previous", {})
        if not task["retry"] and (image_path.exists() or label_path.exists()):
            if not previous.get("verified") or previous.get("fingerprint") != task["fingerprint"]:
                raise RuntimeError("existing outputs are unverified or belong to different inputs/options; inspect and use --retry")
            if previous.get("output_sha256") != {"mra": sha256(image_path), "an": sha256(label_path)}:
                raise RuntimeError("existing output hashes changed; inspect and use --retry")
            validate_pair(image_path, label_path, row["is_positive"])
            status, fields = validate_mra(image_path, task["contract"])
            result.update(status=status, fields=",".join(fields), verified=status != "fail", resumed=True)
        else:
            # Validate input geometry before the shared writer assigns the MRA affine.
            if row["dataset"] == "rsna":
                subject = rsna_subject(row, root, localizers_by_series(root), Path(task["work_root"]))
            else:
                subject = nifti_subject(row, root)
            subject = tio.ToCanonical()(subject)
            if row["label_source_kind"] == "burn_in_difference":
                if not np.allclose(subject["mra"].affine, subject["burn_in"].affine):
                    raise RuntimeError("MRA and burn-in affine mismatch")
                add_burn_in_label(subject)
            if not np.allclose(subject["mra"].affine, subject["label"].affine):
                raise RuntimeError("source image--label affine mismatch")
            staging.mkdir(parents=True, exist_ok=True)
            staged_image, staged_label = staging / image_path.name, staging / ("label_" + label_path.name)
            save_canonical_pair(subject, staged_image, staged_label)
            validate_pair(staged_image, staged_label, row["is_positive"])
            status, fields = validate_mra(staged_image, task["contract"])
            result.update(status=status, fields=",".join(fields), verified=status != "fail")
            if status != "fail":
                staged_image.replace(image_path)
                staged_label.replace(label_path)
        if result["verified"]:
            result["output_sha256"] = {"mra": sha256(image_path), "an": sha256(label_path)}
        if result["status"] == "fail":
            result["error"] = "MRA contract validation failed: " + result["fields"]
    except Exception as error:
        result.update(status="fail", error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc(), verified=False)
    finally:
        # Only explicitly owned temporary pair files are removed; failures remain in reports.
        if staging.exists():
            for path in staging.iterdir():
                if path.is_file():
                    path.unlink()
            staging.rmdir()
        result["finished"] = time.time()
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest-csv", type=Path, required=True)
    for dataset in ("ixi", "ds003949", "ds005096", "rsna"):
        parser.add_argument(f"--{dataset}-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True, help="Local directory outside this repository for temporary labels")
    parser.add_argument("--mra-contract-csv", type=Path, required=True)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--case-list-csv", type=Path, help="CSV containing only case_id; select a manual retry subset")
    parser.add_argument("--overrides-csv", type=Path, help="Local CSV: case_id,source_locator,label_source_locator; only paths may change")
    parser.add_argument("--retry", action="store_true", help="Explicitly rebuild and replace selected output pairs")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_workers < 0:
        raise SystemExit("--num-workers must be non-negative")
    repo = Path(__file__).resolve().parents[1]
    for destination in (args.output_root, args.work_root):
        if destination.resolve().is_relative_to(repo):
            raise SystemExit("Output and work directories must be outside this repository")
    grouped = rows_by_dataset(args.manifest_csv)
    rows = [row for values in grouped.values() for row in values]
    identifiers = [row["case_id"] for row in rows]
    if not rows:
        raise SystemExit("Manifest cannot be empty")
    if len(set(identifiers)) != len(identifiers) or any(not re.fullmatch(r"[A-Za-z0-9_.-]+", value) or value in (".", "..") for value in identifiers):
        raise SystemExit("Manifest case IDs must be unique and safe filenames")
    contract = read_contract(args.mra_contract_csv)
    roots = {dataset: getattr(args, dataset + "_root").resolve() for dataset in grouped}
    overrides = {}
    if args.overrides_csv:
        with args.overrides_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if set(reader.fieldnames or []) - {"case_id", "source_locator", "label_source_locator"} or "case_id" not in (reader.fieldnames or []):
                raise SystemExit("Overrides may contain only case_id,source_locator,label_source_locator")
            for override in reader:
                key = override["case_id"]
                if key not in identifiers or key in overrides:
                    raise SystemExit("Unknown or repeated override case_id: " + key)
                overrides[key] = {field: value for field, value in override.items() if field != "case_id" and value}
    for row in rows:
        row.update(overrides.get(row["case_id"], {}))
        for field in ("source_locator", "label_source_locator"):
            if row.get(field):
                safe_locator(roots[row["dataset"]], row[field])
        if row["is_positive"] not in ("0", "1"):
            raise SystemExit("Manifest positivity must be 0 or 1")
        if row["is_positive"] == "1" and row["label_source_kind"] == "zero_label":
            raise SystemExit("A positive case cannot use zero_label")
    selected_ids = set(identifiers)
    if args.case_list_csv:
        with args.case_list_csv.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["case_id"]:
                raise SystemExit("Case-list CSV must contain only the case_id column")
            selected_ids = {row["case_id"] for row in reader}
        if not selected_ids or not selected_ids.issubset(identifiers):
            raise SystemExit("Case list must contain known case IDs and cannot be empty")
    args.output_root.mkdir(parents=True, exist_ok=True)
    for folder in ("MRA", "AN"):
        (args.output_root / folder).mkdir(exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    run_root = args.output_root / "runs" / run_id
    run_root.mkdir(parents=True)
    (run_root / "failures.jsonl").touch()
    state_path = args.output_root / "cumulative_state.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {}
    full_study_manifest = repo / "manifests" / "authoritative_source_manifest.csv"
    manifest_scope = "full_study" if full_study_manifest.is_file() and sha256(args.manifest_csv) == sha256(full_study_manifest) else "custom_or_subset"
    context = {"manifest_sha256": sha256(args.manifest_csv), "contract_sha256": sha256(args.mra_contract_csv),
               "source_roots": {key: str(value) for key, value in roots.items()}, "workflow_version": 1}
    tasks = []
    fingerprints = {}
    for row in rows:
        case_hash = fingerprint({"context": context, "row": row})
        fingerprints[row["case_id"]] = case_hash
        if row["case_id"] in selected_ids:
            tasks.append({"row": row, "source_root": str(roots[row["dataset"]]), "output_root": str(args.output_root),
                          "work_root": str(args.work_root), "run_id": run_id,
                          "contract": {row["case_id"] + ".nii.gz": contract[row["case_id"] + ".nii.gz"]}
                          if row["case_id"] + ".nii.gz" in contract else {},
                          "fingerprint": case_hash, "previous": state.get(row["case_id"], {}), "retry": args.retry})
    # Invalidate selected verification BEFORE attempting work, including interrupted runs.
    for task in tasks:
        state[task["row"]["case_id"]] = {"status": "pending", "verified": False, "fingerprint": task["fingerprint"]}
    atomic_json(state_path, state)
    results = []
    progress = tqdm(total=len(tasks), desc="Verified pass", unit="case",
                    postfix=f"todo={len(tasks)}, fail=0, done=0, warning=0")
    def consume(result):
        results.append(result)
        state[result["case_id"]] = result
        atomic_json(state_path, state)
        if result["status"] != "pass":
            with (run_root / "failures.jsonl").open("a") as handle:
                handle.write(json.dumps(result, sort_keys=True) + "\n")
        fields = ["case_id", "dataset", "status", "fields", "error", "resumed", "verified", "pid", "started", "finished", "fingerprint"]
        write_csv(run_root / "cases.csv", results, fields)
        write_csv(run_root / "retry_cases.csv", [r for r in results if r["status"] != "pass"], ["case_id"])
        counts = Counter(r["status"] for r in results)
        progress.set_postfix({"todo": len(tasks) - len(results),
                              "fail": counts["fail"] + counts["warning"],
                              "done": counts["pass"], "warning": counts["warning"]}, refresh=False)
        if result["status"] == "pass":
            progress.update()
        progress.refresh()
    try:
        if args.num_workers <= 1:
            for task in tasks:
                consume(case_worker(task))
        else:
            with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
                futures = {executor.submit(case_worker, task): task for task in tasks}
                for future in as_completed(futures):
                    try:
                        result = future.result()
                    except Exception as error:
                        task = futures[future]
                        result = {"case_id": task["row"]["case_id"], "dataset": task["row"]["dataset"],
                                  "status": "fail", "verified": False, "fingerprint": task["fingerprint"],
                                  "error": str(error), "traceback": traceback.format_exc()}
                    consume(result)
    finally:
        progress.close()
        counts = Counter(result["status"] for result in results)
        statuses = Counter()
        for row in rows:
            prior = state.get(row["case_id"], {})
            current = prior.get("fingerprint") == fingerprints[row["case_id"]]
            statuses[prior.get("status", "pending") if current else "pending"] += 1
        report = {"run_id": run_id, **context, "fixed_case_count": len(rows), "selected_case_count": len(tasks),
                  "processed_case_count": len(results), "selected_validation": dict(counts),
                  "cumulative_status": dict(statuses), "pending_count": statuses["pending"],
                  "execution_completed": len(results) == len(tasks),
                  "manifest_scope": manifest_scope,
                  "input_manifest_verified": len(selected_ids) == len(rows) and statuses["pass"] == len(rows),
                  "full_benchmark_verified": manifest_scope == "full_study" and len(selected_ids) == len(rows) and statuses["pass"] == len(rows),
                  "warnings_require_review": bool(statuses["warning"]),
                  "note": "Warnings are completed pairs requiring manual review; they are not a clean benchmark pass. "
                          "Progress done counts verified passes only; fail includes warnings requiring review; todo+fail+done=selected cases."}
        atomic_json(run_root / "summary.json", report)
        atomic_json(args.output_root / "rebuild_report.json", report)
    print(json.dumps(report, sort_keys=True))
    if counts["fail"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
