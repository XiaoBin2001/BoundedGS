# BoundedGS implementation conventions

- The supported reproduction entry is `run.py` with the root `training_config.json`; keep `total_config.json` and `total_config.schema.json` as parameter references. The paper preset is `configs/boundedgs.json`; ablation configs are under `configs/ablations/` and their shared implementations are under `boundedgs/`. Use the active, verified Conda environment and the isolated `diff_gaussian_rasterization_boundedgs` extension.
- Preserve rendering mathematics, gradient accumulation, one update per view, data splits, random sampling and structure-update schedules when doing engineering cleanup. The paper's final training iteration retains its existing no-optimizer-step behavior.
- Keep stage order: render/backward → optimizer → evaluation/PLY save → densification/pruning → checkpoint. Do not change the default in-training evaluation schedule without an explicit request.
- Group/Leaf cache lifetimes and count-scan buffer ordering are part of the memory guarantee. Keep CPU JPEG supply and Leaf prefetch bounded.
- Name projection partitions and candidate tables `group`, adaptive raster subdivisions `leaf`, shared rectangular raster operations `rect`, and image crops `crop`. Keep configuration keys, statistics, Python calls, and native bindings consistent with these meanings.
- Report allocated/reserved memory and temporary working sets separately. Exclude evaluation, PLY saving and accounted monitor overhead from pure training time.
- Validate changes with relevant small tests. Full training, batch evaluation, commits and publication require the user's instruction. Do not replace a failed numerical check with a weaker quality-only check.
- Keep optional paper component variants in `boundedgs/`; do not enable them in the default BoundedGS path. Development tests do not need to be included in the release repository.
