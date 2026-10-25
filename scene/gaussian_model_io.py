"""
GaussianModel point-cloud initialization and PLY I/O.

The methods here convert between the in-memory Gaussian tensors and the PLY
format used for checkpoints and final outputs.
"""
from __future__ import annotations

import os

import numpy as np
import torch
from plyfile import PlyData, PlyElement
from torch import nn

from simple_knn._C import distCUDA2
from utils.general_utils import RGB2SH, mkdir_p
from utils.graphics_utils import BasicPointCloud


class GaussianModelIOMixin:
    """
        Mixin that creates Gaussian tensors from point clouds and reads/writes PLY files.
    """

    def create_from_pcd(self, pcd: BasicPointCloud, spatial_lr_scale: float) -> None:
        """
            Initialize Gaussian tensors from an input point cloud.

            Positions and colors come from the point cloud, scales are estimated
            from nearest-neighbor distance, rotations start as identity, and
            opacity starts from a simple default.
        """
        self.spatial_lr_scale = spatial_lr_scale
        fused_point_cloud = torch.tensor(np.asarray(pcd.points)).float().cuda()
        fused_color = RGB2SH(torch.tensor(np.asarray(pcd.colors)).float().cuda())
        features = torch.zeros((fused_color.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0 ] = fused_color
        features[:, 3:, 1:] = 0.0

        print("Number of points at initialisation : ", fused_point_cloud.shape[0])

        dist2 = torch.clamp_min(distCUDA2(torch.from_numpy(np.asarray(pcd.points)).float().cuda()), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[..., None].repeat(1, 3)
        rots = torch.zeros((fused_point_cloud.shape[0], 4), device="cuda")
        rots[:, 0] = 1

        opacities = self.inverse_opacity_activation(
            float(self.initial_opacity) * torch.ones((fused_point_cloud.shape[0], 1), dtype=torch.float, device="cuda")
        )

        self._xyz = nn.Parameter(fused_point_cloud.requires_grad_(True))
        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(True))
        self._opacity = nn.Parameter(opacities.requires_grad_(True))

    def construct_list_of_attributes(self) -> list[str]:
        """
            Build the ordered PLY attribute list for all saved Gaussian fields.
        """
        l = ['x', 'y', 'z', 'nx', 'ny', 'nz']
        # Save DC and higher-order SH coefficients as flat PLY columns.
        for i in range(self._features_dc.shape[1]*self._features_dc.shape[2]):
            l.append('f_dc_{}'.format(i))
        for i in range(self._features_rest.shape[1]*self._features_rest.shape[2]):
            l.append('f_rest_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        return l

    def save_ply(self, path: str) -> None:
        """
            Write current raw Gaussian tensors to a PLY point-cloud file.
        """
        mkdir_p(os.path.dirname(path))

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        count = int(self._xyz.shape[0])
        elements = np.empty(count, dtype=dtype_full)
        # All fields are packed float32. Fill the same PLY layout directly,
        # without constructing one Python tuple/scalar object per row/field.
        attributes = elements.view(np.float32).reshape(count, len(dtype_full))
        sources = (self._xyz, self._features_dc, self._features_rest,
                   self._opacity, self._scaling, self._rotation)
        for start in range(0, count, 65536):
            end = min(start + 65536, count)
            attributes[start:end, 3:6] = 0  # PLY normals are placeholders.
            column = 0
            for index, source in enumerate(sources):
                chunk = source[start:end].detach()
                if index in (1, 2):
                    chunk = chunk.transpose(1, 2).flatten(start_dim=1)
                values = chunk.contiguous().cpu().numpy()
                width = values.shape[1]
                attributes[start:end, column:column + width] = values
                column += width
                if index == 0:
                    column += 3
                del values, chunk
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply(self, path: str) -> None:
        """
            Load raw Gaussian tensors from a saved PLY file.
        """
        plydata = PlyData.read(path)
        xyz = np.stack((np.asarray(plydata.elements[0]["x"]),
                        np.asarray(plydata.elements[0]["y"]),
                        np.asarray(plydata.elements[0]["z"])),  axis=1)
        opacities = np.asarray(plydata.elements[0]["opacity"])[..., np.newaxis]

        features_dc = np.zeros((xyz.shape[0], 3, 1))
        features_dc[:, 0, 0] = np.asarray(plydata.elements[0]["f_dc_0"])
        features_dc[:, 1, 0] = np.asarray(plydata.elements[0]["f_dc_1"])
        features_dc[:, 2, 0] = np.asarray(plydata.elements[0]["f_dc_2"])

        extra_f_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("f_rest_")]
        extra_f_names = sorted(extra_f_names, key = lambda x: int(x.split('_')[-1]))
        assert len(extra_f_names)==3*(self.max_sh_degree + 1) ** 2 - 3
        features_extra = np.zeros((xyz.shape[0], len(extra_f_names)))
        for idx, attr_name in enumerate(extra_f_names):
            features_extra[:, idx] = np.asarray(plydata.elements[0][attr_name])
        # Reshape flat SH columns to [points, channels, coefficients except DC].
        features_extra = features_extra.reshape((features_extra.shape[0], 3, (self.max_sh_degree + 1) ** 2 - 1))

        scale_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata.elements[0][attr_name])

        rot_names = [p.name for p in plydata.elements[0].properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata.elements[0][attr_name])

        self._xyz = nn.Parameter(torch.tensor(xyz, dtype=torch.float, device="cuda").requires_grad_(True))
        self._features_dc = nn.Parameter(torch.tensor(features_dc, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(torch.tensor(features_extra, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(True))

        self.active_sh_degree = self.max_sh_degree
