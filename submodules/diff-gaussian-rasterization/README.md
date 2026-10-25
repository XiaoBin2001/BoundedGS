# Differential Gaussian Rasterization

Used as the rasterization engine for the paper "3D Gaussian Splatting for Real-Time Rendering of Radiance Fields". If you can make use of it in your own research, please be so kind to cite us.

<section class="section" id="BibTeX">
  <div class="container is-max-desktop content">
    <h2 class="title">BibTeX</h2>
    <pre><code>@Article{kerbl3Dgaussians,
      author       = {Kerbl, Bernhard and Kopanas, Georgios and Leimk{\"u}hler, Thomas and Drettakis, George},
      title        = {3D Gaussian Splatting for Real-Time Radiance Field Rendering},
      journal      = {ACM Transactions on Graphics},
      number       = {4},
      volume       = {42},
      month        = {July},
      year         = {2023},
      url          = {https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/}
}</code></pre>
  </div>
</section>

## BoundedGS package

From the project root, install the isolated extension into the active environment:

```bash
MAX_JOBS=2 python -m pip install ./submodules/diff-gaussian-rasterization --no-build-isolation --no-deps
```

The distribution is `diff-gaussian-rasterization-boundedgs`; Python imports
`diff_gaussian_rasterization_boundedgs`, with its own `_C` extension. The original
source directory is retained through `package_dir`. Existing KNN and fused SSIM
installations are reused. Set `TORCH_CUDA_ARCH_LIST` for the target GPU when needed.

`cuda_rasterizer/raster_math.h` provides only the packed float vectors, column-major
3x3 matrices and operations used here. No GLM checkout is required.

Native bindings distinguish Group candidate selection
(`build_gaussian_group_candidates_raw`) from shared rectangular rasterization
(`rasterize_gaussians_rect`, including precounted and backward variants). Python
Leaf subdivision uses these rectangle primitives. Rebuild the local extension
after updating its source or bindings.
