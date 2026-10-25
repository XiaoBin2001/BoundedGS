"""
Scene management for training and evaluation.

The Scene class hides dataset-format differences. It loads Colmap or Blender
metadata, builds train/test Camera objects, writes reproducibility files, and
either initializes a Gaussian model from the input point cloud or loads a saved
iteration.
"""
from __future__ import annotations

import os
import random
import json
from typing import Any

import numpy as np

from utils.general_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from scene.gaussian_model import GaussianModel as GaussianModel3DGS
from arguments import GroupParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON


def _load_scene_info(args: GroupParams) -> Any:
    """
        Detect the dataset type from files on disk and load matching scene metadata.
    """
    if os.path.exists(os.path.join(args.source_path, "sparse")):
        return sceneLoadTypeCallbacks["Colmap"](
            args.source_path,
            args.images,
            args.depths,
            args.eval,
            getattr(args, "init_type", "sfm"),
            getattr(args, "max_initial_points", -1),
        )
    if os.path.exists(os.path.join(args.source_path, "transforms_train.json")):
        print("Found transforms_train.json file, assuming Blender data set!")
        return sceneLoadTypeCallbacks["Blender"](args.source_path, args.white_background, args.depths, args.eval)
    assert False, "Could not recognize scene type!"


def _write_initial_scene_files(model_path: str, scene_info: Any) -> None:
    """
        Copy the starting point cloud and camera metadata into the experiment folder.
    """
    input_ply_path = os.path.join(model_path, "input.ply")
    point_cloud = getattr(scene_info, "point_cloud", None)
    if point_cloud is not None:
        storePly(
            input_ply_path,
            np.asarray(point_cloud.points),
            np.clip(np.asarray(point_cloud.colors) * 255.0, 0.0, 255.0),
            np.asarray(point_cloud.normals),
        )
    else:
        with open(scene_info.ply_path, 'rb') as src_file, open(input_ply_path, 'wb') as dest_file:
            dest_file.write(src_file.read())
    json_cams = []
    camlist = []
    if scene_info.test_cameras:
        camlist.extend(scene_info.test_cameras)
    if scene_info.train_cameras:
        camlist.extend(scene_info.train_cameras)
    for id, cam in enumerate(camlist):
        json_cams.append(camera_to_JSON(id, cam))
    with open(os.path.join(model_path, "cameras.json"), 'w') as file:
        json.dump(json_cams, file)


def _load_or_create_gaussians(
    gaussians: GaussianModel3DGS,
    scene_info: Any,
    model_path: str,
    loaded_iter: int | None,
    cameras_extent: float,
) -> None:
    """
        Fill the Gaussian model either from a saved iteration or the initial point cloud.

        Saved iterations load from PLY; new runs initialize from the scene point cloud.
    """
    if loaded_iter:
        ply_path = os.path.join(
            model_path,
            "point_cloud",
            "iteration_" + str(loaded_iter),
            "point_cloud.ply",
        )
        gaussians.load_ply(ply_path)
        return

    gaussians.create_from_pcd(scene_info.point_cloud, cameras_extent)


class Scene:
    """
        Loaded dataset plus the Gaussian model associated with it.
    """

    gaussians : GaussianModel3DGS

    def __init__(
        self,
        args: GroupParams,
        gaussians: GaussianModel3DGS,
        load_iteration: int | None = None,
        shuffle: bool = True,
        resolution_scales: list[float] | None = None,
    ) -> None:
        """
            Load cameras, scene bounds, and Gaussian state for one experiment.
        """
        if resolution_scales is None:
            resolution_scales = [1.0]
        self.model_path = args.model_path
        self.loaded_iter = None
        self.gaussians = gaussians

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        scene_info = _load_scene_info(args)

        if not self.loaded_iter:
            _write_initial_scene_files(self.model_path, scene_info)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args, scene_info.is_nerf_synthetic)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args, scene_info.is_nerf_synthetic)

        _load_or_create_gaussians(
            self.gaussians,
            scene_info,
            self.model_path,
            self.loaded_iter,
            self.cameras_extent,
        )

    def save(self, iteration: int) -> None:
        """
            Save the current point cloud for one iteration.
        """
        point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))

    def getTrainCameras(self, scale: float = 1.0) -> list[Any]:
        """
            Return training cameras for the requested resolution scale.
        """
        return self.train_cameras[scale]

    def getTestCameras(self, scale: float = 1.0) -> list[Any]:
        """
            Return test cameras for the requested resolution scale.
        """
        return self.test_cameras[scale]
