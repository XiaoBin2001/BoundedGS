<h1 align="center">
  BoundedGS<br>
  <sub>节省显存的高分辨率 3DGS 训练</sub>
</h1>

<p align="center"><a href="README.md">English</a> · 简体中文</p>

## 1. 认识 BoundedGS

**BoundedGS: Memory-Efficient High-Resolution Training for 3D Gaussian Splatting**

Xiaobin Deng · Changyu Diao<sup>*</sup> · Min Li · Duanqing Xu<sup>*</sup><br>
浙江大学 · <sup>*</sup>通讯作者

[项目页](https://xiaobin2001.github.io/BoundedGS-Web/zh.html) · [论文 PDF](https://xiaobin2001.github.io/BoundedGS-Web/assets/boundedgs-paper.pdf) · [arXiv](https://arxiv.org/abs/2610.34233)

BoundedGS 关注高分辨率 3D Gaussian Splatting（3DGS）训练中的临时显存开销。它把一张图像划分成较小的区域，依次完成计算并累积梯度，最后对整张视图执行一次模型更新，让高分辨率训练更容易落地。

这里有三个相互配合的设计：分开光栅化与投影的反向传播，让局部缓存更早释放；根据预计的计算负载自适应划分区域；紧凑地保存投影状态与参数梯度。CPU 辅助结构更新和有界图像预取，则把这套控制显存的思路延伸到训练流程中。

这里放的是官方完整实现和论文消融。想先跑起来，往下看就好；想弄清方法定义和实验细节，可以翻论文；更多直观案例可以去项目页看看。

**关键词：** 3D 高斯泼溅（3D Gaussian Splatting / 3DGS）· 新视角合成 · 显存高效训练 · 高分辨率训练 · GPU 显存优化

## 2. 安装与运行

### 准备环境

本开源版本的实际测试环境为 **Ubuntu 22.04 LTS、PyTorch 2.5.1（CUDA 12.1）、NVIDIA GeForce RTX 4090 24 GB**，使用 Python 3.10.19 和 CUDA Toolkit 12.1。

不必照搬这些版本。你也可以沿用自己的 CUDA 版 PyTorch 环境，搭配兼容的 torchvision、CUDA Toolkit（含 `nvcc`）和 C++ 编译器，确认下方三个随库 CUDA 扩展可以在该环境中编译、导入即可。需要安装 PyTorch 时，可以参考 [PyTorch 安装页](https://pytorch.org/get-started/locally/)。如果新建环境，建议选择 Python 3.10–3.12，以兼顾本库的 Python 语法和 NumPy 1.x 依赖。

> **Windows 模式：**脚本会自动识别原生 Windows，关闭依赖 Linux 的 `resource_limits`（CPU 亲和性及 `guard`/`strict` 内存限制），并在启动时提示；此时任务不受这些 CPU/内存限制。BoundedGS 另关闭 Linux CPU 内存统计，保留 CUDA 显存统计和训练计时。进程启动/清理和 GPU 锁文件使用适用于 Windows 的接口。本模式尚未在 Windows 实机上验证 CUDA 编译与训练。Windows 下编译扩展还需准备 MSVC C++ Build Tools，并确保构建环境可以找到 `nvcc`。

其余 Python 依赖为 `numpy<2`（NumPy 1.x）、`Pillow`、`opencv-python`、`plyfile`、`tqdm`、`psutil`；编译扩展还需要 `setuptools` 和 `wheel`。保留 NumPy 1.x 的兼容约束，其余不固定具体版本。[Ninja 是可选的](https://docs.pytorch.org/docs/stable/cpp_extension.html#torch.utils.cpp_extension.BuildExtension)，用于加快编译；未安装时，PyTorch 的扩展构建工具会使用标准构建后端。

激活环境、装好 PyTorch 后，如果希望自己选择上述 Python 包的版本，可以自行安装；也可以在仓库根目录使用 [requirements.txt](requirements.txt) 直接安装：

```bash
python -m pip install -r requirements.txt
```

然后在同一环境中安装三个随库扩展：

```bash
python -m pip install ./submodules/diff-gaussian-rasterization --no-build-isolation --no-deps
python -m pip install ./submodules/simple-knn --no-build-isolation --no-deps
python -m pip install ./submodules/fused-ssim --no-build-isolation --no-deps
```

如果已有环境中安装了兼容的 `simple_knn` 和 `fused_ssim`，可以跳过最后两条命令。LPIPS 首次使用时会下载预训练权重；离线评测前，请先准备好对应的 Torch 缓存。

### 准备数据

使用完成去畸变的 COLMAP 场景，按下面的方式摆放：

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

论文预设包含 13 个场景：`bicycle`、`flowers`、`garden`、`stump`、`treehill`、`bonsai`、`counter`、`kitchen`、`room`、`Barn`、`Family`、`Train` 和 `Truck`。场景名区分大小写。BoundedGS 使用原始图像分辨率（`resolution: 1`），按排序后的图像每八张留出一张作为测试视图。

### 跑通默认流程

想先复现主方法？[training_config.json](training_config.json) 已经备好了论文主表流程，填上数据路径、选好 GPU 就能开跑：

```bash
# 先预览运行队列，此命令不会开始训练。
python run.py --data_root /path/to/HR-Data --gpu_id 0 --dry-run

# 训练、保存模型、渲染测试视图并完成评测。
python run.py --data_root /path/to/HR-Data --gpu_id 0
```

第一次尝试时，可以加上 `--only-scene bicycle`，先跑一个场景；复现全部场景时保留完整队列。默认训练 30K 次迭代，每 3K 次进行一次训练中评测。

结果保存在 `output/boundedgs/<scene>/`。模型在 `point_cloud/iteration_30000/point_cloud.ply`，渲染图像在 `test/ours_30000/renders/`，PSNR、SSIM、LPIPS 和渲染速度在 `result_test.json`。`resource_summary.json` 记录资源使用情况，批量 CSV 汇总各场景结果。

> 小提示：先用 `--dry-run` 看一眼队列，路径和场景都对了，再开跑。

## 3. 试试这些选项

### 调整显存与计算开销的取舍

[configs/boundedgs.json](configs/boundedgs.json) 保留了论文主方法的默认设置。日常使用可以直接编辑 `training_config.json`，也可以通过 `python run.py -c <config>` 选择另一份配置。

**Group** 是投影分组，负责筛选当前区域可能用到的高斯；一个 Group 内又可以划分成多个 **Leaf**，逐个完成光栅化与反向传播。增加 Group 数量或减小 Leaf 预算，通常可以降低临时显存，但会增加执行开销。

| `train_args` 中的选项 | 默认值 | 可以调整什么 |
| :---: | :---: | :---: |
| `group_grid` | `1` | 每个图像方向上的 Group 数；`2` 表示 2×2 分组 |
| `leaf_raster_budget_mib` | `384` | 每个 Leaf 的目标光栅载荷预算，单位 MiB；`0` 表示固定矩形划分 |
| `projection_mode` | `group_compact` | 紧凑候选投影；`full` 和 `simple_split` 用于对照 |
| `use_compact_parameter_gradients` | `true` | 紧凑地保存当前候选高斯的参数梯度 |
| `structure_rebuild_backend` | `cpu` | 在 CPU 侧暂存结构重建数据，降低 GPU 峰值 |
| `training_gt_mode` | `leaf_prefetch` | 有界 Leaf 预取；`leaf` 使用同步供图 |
| `use_gns_pruning` | `true` | 在固定 ABS 阈值＋LAS 致密化路径上启用 GNS 剪枝 |

Leaf 预算针对局部光栅计算。模型参数、优化器状态和其他缓存也会占用显存，因此这个值不是整次训练的总显存上限。Leaf 布局会按相机缓存，复用前仍会检查当前负载。

[total_config.json](total_config.json) 按组列出参数，[total_config.schema.json](total_config.schema.json) 列出字符串选项的可选值。把需要的字段复制到运行配置中即可：公共训练设置放在 `train_args`，逐场景覆盖放在 `scenes[].train_args`。在 Linux 上，随附配置中的 `resource_limits` 将 CPU 和主机内存预算设为 45%，可以按自己的机器调整或移除。默认 `guard` 模式在监测到 RSS 超限时停止任务，并非硬性内存上限。

### 运行论文消融

这些实验同样通过 `run.py` 启动。部分实验需要已有模型，运行前先看看表格中的输入要求。

| 实验 | 配置 | 所需输入 |
| :---: | :---: | :---: |
| 整图执行的 Original 对照 | [original.json](configs/original.json) | 数据集 |
| 六种组件变体 | [components.json](configs/ablations/components.json) | [component_source.json](configs/ablations/component_source.json) 产生的 15K 模型 |
| 1K–16K 分辨率扫描 | [resolution.json](configs/ablations/resolution.json) | 30K Original Bicycle 模型 |
| 固定 2048 像素分块，附录 C.1 | [pixel_blocks.json](configs/ablations/pixel_blocks.json) | 30K Original Bicycle 模型 |
| 独立分块更新，附录 C.2 | [independent_blocks.json](configs/ablations/independent_blocks.json) | 15K component-source Bicycle 模型 |
| CPU 重建与 JPEG 供图 | [resources/](configs/ablations/resources/) | 数据集 |

例如，先准备共用模型，再比较各组件：

```bash
python run.py -c configs/ablations/component_source.json --data_root /path/to/HR-Data --gpu_id 0
python run.py -c configs/ablations/components.json --data_root /path/to/HR-Data --gpu_id 0
```

固定模型对照保持高斯数量不变，采用论文中的 L1 协议。配置负责选择变体和输入模型；如果模型保存在其他位置，修改 `source_root` 或 `model_path` 即可。每项实验会输出逐视图记录和汇总。重新测量时请使用新的输出目录。

### 接着训练或重新评测

高分辨率训练耗时较长时，可以设置 `train_args.checkpoint_iterations`，例如 `[15000, 20000]`。空列表表示不保存续训检查点。之后把顶层 `resume_training` 设为 `true`，就会从最近保存的节点继续，并恢复当时的训练参数与状态。PLY 导出由 `save_iterations` 和 `save_final_model` 单独控制。

如果只想重新渲染已有模型，或重新汇总已有指标：

```bash
python run.py --postprocess-only
python run.py --rebuild-summary
```

请沿用原来的配置和输出目录。`repeat` 默认为 `1`；启用 `cleanup_repeats` 后，会保留测试 PSNR 最佳的一次完整输出，将其他完整输出放入可恢复归档，同时保留所有重复的记录。

## 4. 代码来源与致谢

BoundedGS 从 [ImprovedGS 开源代码库](https://github.com/XiaoBin2001/Improved-GS) 发展而来，在其训练框架上实现了前面介绍的 BoundedGS 执行方式。

这次开源整理也简化了训练、渲染、评测和论文实验共用的配置流程。CUDA 扩展使用独立导入名 `diff_gaussian_rasterization_boundedgs`；原来随库携带的 GLM 被替换为渲染器实际需要的一小组数学运算；纯 RGB 渲染可以跳过不需要的深度计算。保存的配置、指标和汇总，也方便大家查看和复用一次运行的结果。

感谢 ImprovedGS 作者，以及 3D Gaussian Splatting、AbsGS、Speedy-Splat、simple-knn、fused-ssim 和 LPIPS 的贡献者提供的实现与研究思路。源码中保留了原有版权声明和依赖说明。

## 5. 引用与许可

如果本工作对你的研究有帮助，欢迎引用：

```bibtex
@misc{deng2026boundedgsmemoryefficienthighresolutiontraining,
      title={BoundedGS: Memory-Efficient High-Resolution Training for 3D Gaussian Splatting},
      author={Xiaobin Deng and Changyu Diao and Min Li and Duanqing Xu},
      year={2026},
      eprint={2610.34233},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2610.34233},
}
```

上游论文及引用信息见 [ImprovedGS](https://github.com/XiaoBin2001/Improved-GS)。

- BoundedGS 本工作新增的代码与算法实现采用 [MIT 许可](LICENSE.md)。
- 继承部分遵循原有工作的许可，主要为 [3DGS](LICENSE.md#3dgs-license) 和 [ImprovedGS](LICENSE.md#improvedgs-license)。
