<h1 align="center">
  BoundedGS<br>
  <sub>Memory-Efficient High-Resolution 3DGS Training</sub>
</h1>

<p align="center">English · <a href="README.zh-CN.md">简体中文</a></p>

## 1. About BoundedGS

**BoundedGS: Memory-Efficient High-Resolution Training for 3D Gaussian Splatting**

Xiaobin Deng · Changyu Diao<sup>*</sup> · Min Li · Duanqing Xu<sup>*</sup><br>
Zhejiang University · <sup>*</sup>Corresponding authors

[Project page](https://xiaobin2001.github.io/BoundedGS-Web/) · [Paper (PDF)](https://xiaobin2001.github.io/BoundedGS-Web/assets/boundedgs-paper.pdf) · arXiv: coming soon

BoundedGS makes high-resolution 3D Gaussian Splatting (3DGS) training more manageable by controlling temporary rendering memory. It processes an image in smaller regions, accumulates their gradients, and updates the model once for the complete view.

Three ideas work together: separate rasterization and projection backward passes so local buffers can be released earlier; adapt raster regions to their estimated workload; and keep projection state and parameter gradients compact. CPU-assisted structure updates and bounded image prefetch carry the same approach through training.

The official implementation and paper ablations live here. Ready to try it? Keep reading. For method definitions and experimental details, head to the paper; visit the project page for more visual examples.

**Keywords:** 3D Gaussian Splatting (3DGS) · novel view synthesis · memory-efficient training · high-resolution training · GPU memory optimization

## 2. Install and run

### Set up the environment

We tested this release on **Ubuntu 22.04 LTS, PyTorch 2.5.1 (CUDA 12.1), and an NVIDIA GeForce RTX 4090 with 24 GB VRAM**, using Python 3.10.19 and CUDA Toolkit 12.1.

You do not need to copy these versions exactly. You can use your own CUDA-enabled PyTorch environment with a matching torchvision, CUDA toolkit (`nvcc`), and C++ compiler; make sure the three bundled CUDA extensions below can compile and import in that environment. If you need to install PyTorch, see the [PyTorch installation page](https://pytorch.org/get-started/locally/). For a new environment, Python 3.10–3.12 fits the repository's Python syntax and NumPy 1.x dependency.

> **Windows mode:** the scripts automatically detect native Windows and disable Linux-only `resource_limits` (CPU affinity and both `guard`/`strict` RAM limits), with a startup notice. Tasks then run without these CPU/RAM limits. BoundedGS also disables Linux CPU memory statistics, while keeping CUDA memory statistics and training timing. Process startup/cleanup and GPU lock files use Windows-compatible interfaces. CUDA compilation and training in this mode have not yet been tested on a Windows machine. Building extensions on Windows also requires MSVC C++ Build Tools and `nvcc` in your build environment.

The remaining Python dependencies are `numpy<2` (NumPy 1.x), `Pillow`, `opencv-python`, `plyfile`, `tqdm`, and `psutil`; extension builds also use `setuptools` and `wheel`. We keep the NumPy 1.x compatibility constraint without pinning exact package versions. [Ninja is optional](https://docs.pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.BuildExtension): it speeds up compilation, and PyTorch's extension builder falls back to the standard backend when it is absent.

With your environment active and PyTorch installed, you can choose and install these packages yourself, or use the list in [requirements.txt](requirements.txt) from the repository root:

```bash
python -m pip install -r requirements.txt
```

Then install the three bundled extensions in the same environment:

```bash
python -m pip install ./submodules/diff-gaussian-rasterization --no-build-isolation --no-deps
python -m pip install ./submodules/simple-knn --no-build-isolation --no-deps
python -m pip install ./submodules/fused-ssim --no-build-isolation --no-deps
```

If compatible `simple_knn` and `fused_ssim` packages are already installed, you can skip the last two commands. LPIPS downloads pretrained weights on first use; prepare the Torch cache beforehand for offline evaluation.

### Prepare your data

Use undistorted COLMAP scenes with this layout:

```text
HR-Data/
  bicycle/
    images/
    sparse/0/
  garden/
    images/
    sparse/0/
  ...
```

The paper preset contains 13 scenes: `bicycle`, `flowers`, `garden`, `stump`, `treehill`, `bonsai`, `counter`, `kitchen`, `room`, `Barn`, `Family`, `Train`, and `Truck`. Names are case-sensitive. BoundedGS uses original-resolution images (`resolution: 1`) and holds out every eighth sorted image for testing.

### Run the main pipeline

For a first run, [training_config.json](training_config.json) has the main-table pipeline ready for you. Point it to your data and choose a GPU:

```bash
# Preview the queue without starting training.
python run.py --data_root /path/to/HR-Data --gpu_id 0 --dry-run

# Train, save, render held-out views, and evaluate them.
python run.py --data_root /path/to/HR-Data --gpu_id 0
```

For a first run on just Bicycle, add `--only-scene bicycle`. Keep the full queue to reproduce all scenes. The default runs 30K iterations, with training-time evaluation every 3K iterations.

Results appear under `output/boundedgs/<scene>/`. Look for `point_cloud/iteration_30000/point_cloud.ply` for the model, `test/ours_30000/renders/` for rendered images, and `result_test.json` for PSNR, SSIM, LPIPS, and rendering speed. `resource_summary.json` records resource use; batch CSV files collect the scene results.

> Tip: take a peek with `--dry-run` first, then start when the paths and scene list look right.

## 3. Your options

### Choose the memory/compute trade-off

The defaults in [configs/boundedgs.json](configs/boundedgs.json) match the paper's main setup. Edit `training_config.json` for your own runs, or pass a config with `python run.py -c <config>`.

A **Group** selects the Gaussians needed for one projection region. Within it, **Leaves** are smaller rectangles processed one at a time for rasterization and backward computation. More Groups and a smaller Leaf budget can reduce temporary memory, at the cost of more execution overhead.

| Option in `train_args` | Default | What it controls |
| :---: | :---: | :---: |
| `group_grid` | `1` | Groups per image side; `2` gives a 2×2 grid |
| `leaf_raster_budget_mib` | `384` | Target raster payload per Leaf, in MiB; `0` keeps fixed rectangles |
| `projection_mode` | `group_compact` | Compact candidate projection; `full` and `simple_split` support comparisons |
| `use_compact_parameter_gradients` | `true` | Store gradients for active candidates compactly |
| `structure_rebuild_backend` | `cpu` | Stage structure rebuilding on the CPU to reduce GPU peaks |
| `training_gt_mode` | `leaf_prefetch` | Bounded Leaf prefetch; `leaf` uses synchronous image supply |
| `use_gns_pruning` | `true` | Enable GNS pruning alongside fixed ABS-threshold + LAS densification |

The Leaf budget covers local raster work. Model parameters, optimizer state, and other buffers also need GPU memory, so this is not a total-memory cap. Leaf layouts are cached per camera and their current workload is checked before reuse.

[total_config.json](total_config.json) groups the parameters; [total_config.schema.json](total_config.schema.json) lists string-valued choices. Copy the fields you need into a runnable config. Shared settings go in `train_args`; scene-specific changes go in `scenes[].train_args`. On Linux, the shipped `resource_limits` sets CPU and host RAM budgets to 45%; adjust or remove it for your machine. The default `guard` mode stops a task when monitored RSS exceeds its budget; it is not a hard memory cap.

### Try the paper ablations

These experiments use the same `run.py` entry. Some start from a saved model, so check the input column first.

| Experiment | Config | Start from |
| :---: | :---: | :---: |
| Full-frame Original reference | [original.json](configs/original.json) | Dataset |
| Six component variants | [components.json](configs/ablations/components.json) | 15K models from [component_source.json](configs/ablations/component_source.json) |
| 1K–16K resolution sweep | [resolution.json](configs/ablations/resolution.json) | 30K Original Bicycle |
| Fixed 2048-pixel blocks, Appendix C.1 | [pixel_blocks.json](configs/ablations/pixel_blocks.json) | 30K Original Bicycle |
| Independent block updates, Appendix C.2 | [independent_blocks.json](configs/ablations/independent_blocks.json) | 15K component-source Bicycle |
| CPU rebuilding and JPEG supply | [resources/](configs/ablations/resources/) | Dataset |

For example, prepare the shared models and then run the component comparison:

```bash
python run.py -c configs/ablations/component_source.json --data_root /path/to/HR-Data --gpu_id 0
python run.py -c configs/ablations/components.json --data_root /path/to/HR-Data --gpu_id 0
```

Fixed-model comparisons keep Gaussian count fixed and use the paper's L1 protocol. Their configs select variants and input models; update `source_root` or `model_path` if you saved the inputs elsewhere. Each experiment writes per-view records and a summary. Use a new output directory for a new measurement.

### Reuse a run

For long high-resolution runs, set `train_args.checkpoint_iterations`, for example `[15000, 20000]`. An empty list saves no restart checkpoints. Later, set top-level `resume_training: true` to continue from the latest saved node with its training arguments and state. PLY exports are controlled separately by `save_iterations` and `save_final_model`.

You can also render an existing model again or collect its saved metrics:

```bash
python run.py --postprocess-only
python run.py --rebuild-summary
```

Use the same config and output root as the original run. `repeat` defaults to `1`; when enabled, `cleanup_repeats` keeps the best test-PSNR run and recoverably archives other full outputs while retaining every repeat's records.

## 4. Code origins and acknowledgements

BoundedGS is developed from the [ImprovedGS open-source repository](https://github.com/XiaoBin2001/Improved-GS). It builds on that training framework and adds the BoundedGS execution scheme described above.

For this release, we streamlined the shared configuration workflow for training, rendering, evaluation, and paper experiments. The CUDA extension has its own import name (`diff_gaussian_rasterization_boundedgs`), the bundled GLM dependency is replaced by the small set of math operations used by the renderer, and RGB-only rendering can skip unused depth computation. Saved settings, metrics, and summaries help you inspect and reuse a run.

We thank the ImprovedGS authors and the contributors to 3D Gaussian Splatting, AbsGS, Speedy-Splat, simple-knn, fused-ssim, and LPIPS for the implementations and ideas this work builds on. Original copyright and dependency notices are retained in the source.

## 5. Citation and license

BibTeX: coming soon.

See [ImprovedGS](https://github.com/XiaoBin2001/Improved-GS) for the upstream paper and citation.

- Original code and algorithm implementations introduced by BoundedGS are licensed under [MIT](LICENSE.md).
- Inherited code follows its original licenses, primarily those of [3DGS](LICENSE.md#3dgs-license) and [ImprovedGS](LICENSE.md#improvedgs-license).
