# Frozen manifests

Version `2026-10-26.1` fixes **2,085 manifest case rows** and their existing source
routes and Setting A/B/C assignments. It does not create new splits, infer
scanner metadata from downloads, or deduplicate source references. Case IDs are
benchmark row identifiers; they do not establish unique patients or acquisitions.

Files:

- `authoritative_source_manifest.csv`: all 2,085 fixed cases; original 12 source
  columns preserved in order and value, followed by 13 frozen metadata columns.
- `mra_validation_contract.csv`: expected MRA shape, spacing and voxel count for
  each reference filename. Preserved without modification; it does not validate
  lesion labels or clinical lesion diameters.
- `smoke16_seed20260810.csv`: the existing 16 cases, four per dataset. Case order,
  selection rank and seed are preserved. Two historical identical duplicate
  rank/seed columns were collapsed after checking their values agree. The same
  frozen metadata as the full manifest is appended; this is not a new selection.
- `VERSION` and `manifest_version.json`: schema version,
  original-column digest, file SHA-256 hashes, provenance and aggregate counts.

The reconstruction runner accepts the complete manifest or a fixed subset such
as smoke16. All locators are relative to the corresponding authorized source
root; they are not paths on the authors' machine. Empty label locators are
intentional for documented negative cases.

## Original source columns

| Field | Meaning |
|---|---|
| `case_id` | Stable benchmark case-row ID; do not renumber or deduplicate. |
| `dataset` | `ds003949`, `ds005096`, `ixi` or `rsna`. |
| `source_kind` | Source acquisition organization used by the resolver. |
| `source_identity` | Existing source acquisition identity used for exact lookup. |
| `source_subject_id`, `source_session_id` | Existing source labels where present; not newly derived patient identities. |
| `source_acquisition_key` | Existing acquisition lookup key, where applicable. |
| `source_locator` | Source-root-relative image/acquisition locator. |
| `reference_filename` | Fixed reconstructed filename, `<case_id>.nii.gz`. |
| `is_positive` | Frozen source case label: `1` positive, `0` documented negative. |
| `label_source_kind` | Existing label-construction route. |
| `label_source_locator` | Source-root-relative label locator; an optional `#key` selects a localizer record. |

## Frozen metadata and split columns

| Field | Meaning |
|---|---|
| `manufacturer` | Author-harmonized source manufacturer: GE, Philips, Siemens, Toshiba or Hitachi. |
| `tesla` | Frozen field-strength category: `1.5T`, `3T`, `1.2T` or `Unknown`; no missing-value imputation. |
| `primary_cohort` | `1` for the 2,057 known-scanner cohort rows in the main composition table; `0` otherwise. This flag alone does **not** determine a setting's target eligibility. |
| `scanner_id` | A normalized manufacturer/model/protocol **category**, e.g. `GE_DISCOVERY MR750`. It is not a patient identifier, physical scanner serial number, or unique-device ID. Raw model/protocol fields are not included. |
| `setting_A_role`, `setting_B_role`, `setting_C_role` | Authoritative use of the case in each setting; see roles below. |
| `setting_A_split`, `setting_B_split`, `setting_C_split` | Original experiment split values; do not regenerate them. |
| `setting_A_domain_group`, `setting_B_domain_group`, `setting_C_domain_group` | Exact source or target domain name used by the corresponding manuscript table. |

Roles take precedence over the legacy split field when deciding eligibility:

- `source_train` / `train`: model training and source-side policy construction.
- `source_val` / `val`: source monitoring, not target-based model selection.
- `id_test` / `test`: held-out source-domain reference, not primary cs-ID.
- `cs_id_primary` / `cs-test`: primary held-out acquisition-stratified evaluation.
- `cs_id_etc` / `cs-test`: retained ETC target group, excluded from primary
  subgroup/worst-case reporting.
- `excluded`: unused in that setting. In A/B its legacy split is `excluded`;
  the eight excluded C rows retain the historical `cs-test` split value. They
  remain excluded: **never select evaluation cases from split alone**.

The historical source split used seed 42 and an approximately 8:1:1 stratified
train/validation/ID-test assignment. This release distributes those assignments
directly and never recomputes a split from the metadata or seed.

## Cohort and setting counts

The primary known-scanner cohort has **2,057 cases: 660 positive and 1,397
negative**. Its manufacturer/field-strength composition matches main Table 1.
All 70 Toshiba known-field cases belong to that global cohort; they participate
in pooled source domains when appropriate but are ETC when held out as targets.
Twenty Siemens unknown-field cases are outside that global cohort and are used
only within Setting C's source split (14 train, 4 validation, 2 ID-test). Seven
GE/Philips unknown-field cases and one Hitachi 1.2T case are excluded from all
three settings. Candidate count 2,085 is therefore different from both primary
cohort size and individual setting eligibility.

| Setting | Source | Train | Validation | ID-test | Primary cs-ID | ETC targets | Excluded | Source train+val |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | 1.5T pooled | 867 | 108 | 110 | 943 | 29 | 28 | 975 |
| B | 3T pooled | 777 | 96 | 99 | 1,044 | 41 | 28 | 873 |
| C | Siemens, including 20 unknown-field source cases | 731 | 91 | 93 | 1,092 | 70 | 8 | 822 |

Primary target domains are A: GE/Philips/Siemens 3T; B:
GE/Philips/Siemens 1.5T; C: GE/Philips at 1.5T and 3T. All are preassigned in the
manifest. The source train+val column describes the full source cohort rather
than run-specific eligibility for auxiliary vessel supervision.

## Read-only validation

Run from the repository root using Python 3 (standard library only):

```bash
python scripts/audit_manifest.py
python -m unittest discover -s tests -p test_manifest_contract.py -v
```

The audit checks frozen hashes, schema, source-column identity, relative
locators, role/split constraints, smoke metadata/selection and validation
contract membership. It does not read images or contact an external service.

Authors who have the manuscript TeX sources can independently check the current
tables and subject-count narrative (no private ledger or previous count report
is needed):

```bash
python scripts/audit_manifest.py --paper /path/to/paper.tex --supplement /path/to/supplement.tex --output /path/to/aggregate-audit.json
```

The checked scope is subjects, positivity, dataset/vendor/Tesla composition,
scanner-category counts and fixed split/domain membership. It includes the
previous 241-item audit scope plus total and split-semantic checks. Lesion
component counts, performance results, patient uniqueness and source-asset
deduplication are outside this manifest audit. Reports contain aggregate values;
the CSV itself is a case-level reconstruction contract under the original source
access requirements. No images or segmentation arrays are redistributed.
