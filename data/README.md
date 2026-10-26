# Local source data and generated outputs

This directory contains documentation only. Download each source through its original access route and keep it outside the repository. See the root README for source routes and the reconstruction/review workflow.

The versioned files in `manifests/` define the study candidate cohort and frozen study splits. They contain case-level reconstruction metadata, including upstream identifiers, relative locators, positivity and scanner categories; they contain no voxel arrays or segmentation masks.

Local source downloads, reconstructed MRA/AN pairs, temporary RSNA masks, case selections, path overrides and run reports belong outside this repository. New or changed upstream releases may require manual source mapping; this must not silently change study membership or splits. A repaired source mapping is not proof of voxel-level equivalence.

Code licensing does not redistribute or sublicense any upstream dataset. The original access and reuse terms continue to apply.
