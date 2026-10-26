# TOF-MRA Aneurysm Benchmark Reconstruction

Reconstruction specification and utilities for **Clinically Constrained Augmentation for Worst-Case Domain-Generalized TOF-MRA Aneurysm Detection** (WACV 2027).

This package provides a **fixed case-level manifest**, scanner/split metadata, a geometry validation contract, and reconstruction code. It contains no source images, voxel-level labels, reconstructed volumes, or model weights. The manifest intentionally includes upstream case identifiers, relative source locators, positive/negative status and study membership. It is not an anonymous aggregate-only dataset.

## Fixed study specification

The candidate manifest contains **2,085 cases**. The paper's primary known-scanner cohort contains **2,057 cases (660 positive / 1,397 negative)**. These are different scopes: 20 additional Siemens cases with unknown field strength enter the Setting-C source split only, and eight cases are excluded from all study settings. Pooled Toshiba source cases and non-primary ETC target groups follow the paper's setting-specific rules.

- Setting A: source 1.5T; held-out 3T acquisition groups.
- Setting B: source 3T; held-out 1.5T acquisition groups.
- Setting C: Siemens source (including the source-only unknown-field cases); held-out GE/Philips groups.

Source train+validation counts are **975 / 873 / 822** for A/B/C. Use the fixed role and split columns rather than splitting again. `role` determines inclusion: legacy split values on an excluded row do not make it a study evaluation case. Stable case IDs are the counting unit; the package does not claim to verify unique patient identity across upstream records.

[Manifest documentation](manifests/README.md) defines all fields and eligibility rules. Version, SHA256 hashes and aggregate count tables are provided in `manifests/`. The manifest audit checks these frozen files before reconstruction:

```bash
python scripts/audit_manifest.py
```

## Obtain the source datasets

Obtain every source independently through its original access route and comply with its access and reuse terms. Keep downloads and generated files outside this repository. The package does not download or sublicense source datasets.

| Source | Access route | Local input |
| --- | --- | --- |
| ds003949 | [OpenNeuro](https://openneuro.org/datasets/ds003949) | Extracted dataset root |
| ds005096 | [OpenNeuro](https://openneuro.org/datasets/ds005096) | Extracted dataset root |
| IXI | [IXI data portal](https://brain-development.org/ixi-dataset/) | Extracted MRA archive |
| RSNA | [RSNA Imaging Registry dataset 7](https://imaging.rsna.org/dataset/7) | Authorized extracted release, series directories and `train_localizers.csv` |

RSNA access requires a request and acceptance of its applicable terms. Do not interpret this code release as permission to access or redistribute any dataset.

## Environment

```bash
conda env create -f environment.yml
conda activate tof_mra_benchmark_rebuild
```

This is a CPU reconstruction workflow; no GPU is required.

## Reconstruct, inspect, then selectively retry

Upstream layouts and releases can change. A single invocation is **not a guarantee of a complete benchmark**. Start with the fixed 16-case smoke manifest (four cases per source):

```bash
python scripts/rebuild_full_manifest.py \
  --manifest-csv manifests/smoke16_seed20260810.csv \
  --rsna-root /path/to/authorized_data/rsna \
  --ixi-root /path/to/authorized_data/IXI-MRA \
  --ds003949-root /path/to/authorized_data/ds003949 \
  --ds005096-root /path/to/authorized_data/ds005096 \
  --output-root /path/to/benchmark_smoke \
  --work-root /path/to/rebuild_work \
  --mra-contract-csv manifests/mra_validation_contract.csv \
  --num-workers 8
```

Then use the full fixed manifest and a separate output directory:

```bash
python scripts/rebuild_full_manifest.py \
  --manifest-csv manifests/authoritative_source_manifest.csv \
  --rsna-root /path/to/authorized_data/rsna \
  --ixi-root /path/to/authorized_data/IXI-MRA \
  --ds003949-root /path/to/authorized_data/ds003949 \
  --ds005096-root /path/to/authorized_data/ds005096 \
  --output-root /path/to/benchmark_full \
  --work-root /path/to/rebuild_work \
  --mra-contract-csv manifests/mra_validation_contract.csv \
  --num-workers 8
```

Review `rebuild_report.json`, `cumulative_state.json` and the latest `runs/<run_id>/` reports in the output directory. `input_manifest_verified` refers to the supplied manifest; `full_benchmark_verified` additionally requires the frozen 2,085-case manifest. A smoke or custom subset cannot establish full benchmark verification. An individual missing/corrupt case is reported while unrelated cases continue. A finished pass, a zero exit code, and successful reconstruction of the selected subset do **not** imply that all study cases have passed validation. Warnings require manual inspection; failures are not silently excluded.

The local run directory contains:

- `cases.csv`: per-case status for this attempt.
- `failures.jsonl`: detailed failed-case diagnostics and traceback.
- `retry_cases.csv`: case IDs requiring further attention.
- `summary.json`: selected/cumulative status and whether the full input manifest is verified.

Create or edit a local selection CSV containing only the `case_id` column. Select IDs from the frozen manifest or the retry report; do not renumber or remove them from the released manifest. Re-run the full command above with the same full manifest/output directory and add:

```text
--case-list-csv /path/to/local/retry_cases.csv --retry
```

If a source file moved, create a local override CSV with `case_id,source_locator,label_source_locator`. Locators are relative to the corresponding dataset root; blank values retain the fixed locator. Add `--overrides-csv /path/to/local/overrides.csv` to the retry command. Overrides cannot change positivity, membership, labels' reconstruction rule, or splits. Do not use an override to substitute an unrelated case.

After repairing selected cases, run the full-manifest command again without `--case-list-csv` or `--retry` (retain any needed `--overrides-csv`) to revalidate all outputs and inspect the final full-benchmark status. Verified outputs are revalidated and reused by default. Use `--retry` for deliberate replacement after investigating a failed or stale case. Keep all report directories for provenance; retries do not erase earlier attempt reports. A source mapping correction does not prove that an updated upstream volume is identical to the historical study volume. If correspondence remains unresolved, report the discrepancy; do not manufacture a replacement label or silently change the benchmark denominator.

## Output and validation boundaries

Generated pairs are `MRA/<case_id>.nii.gz` (canonical RAS, float32) and `AN/<case_id>.nii.gz` (binary int16). Positive-label routes include supplied masks, burn-in differences and RSNA localizer-derived spheres; these are not uniformly clinician-measured lesion boundaries.

MRA geometry is checked against the fixed contract using shape, spacing and voxel count. Zero discrepant fields passes; one produces a warning; two or more fail. Pair geometry and binary-label checks catch malformed outputs. Geometry agreement does not prove voxel identity, annotation correctness, or clinical validity. Never regenerate the released contract merely to make a changed source pass.

The public package reconstructs the data layout only. It does not implement the paper's augmentation policy, train the study models, reproduce all metrics, or convert generated pairs into every downstream nnU-Net training artifact. Use the published method description for that separate research workflow.

## Validation and licensing

```bash
python -m unittest discover -s tests -v
```

Synthetic tests exercise reconstruction failure isolation and selective retry; they do not certify every case in a newer upstream release. Fixed case metadata and aggregate checks support comparison with the paper, while source access and case review remain the researcher's responsibility.

The original code and documentation are licensed under [MIT](LICENSE). This license does not grant rights to upstream images, annotations or source-controlled metadata. Access and reuse of upstream materials remain subject to their original terms.
