"""
Dataset readers for Colmap and Blender/NeRF synthetic scenes.

Each reader returns a SceneInfo object containing the initial point cloud,
train/test camera metadata, normalization values, and dataset type. Scene then
turns this metadata into Camera objects and initializes the Gaussian model.
"""
from __future__ import annotations

import os
import sys
from PIL import Image
from typing import Any, NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import BasicPointCloud, getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from numpy.typing import NDArray
from plyfile import PlyData, PlyElement
from utils.general_utils import SH2RGB

class CameraInfo(NamedTuple):
    """
        Lightweight camera metadata produced by dataset readers.
    """
    uid: int
    R: NDArray[np.floating]
    T: NDArray[np.floating]
    FovY: float
    FovX: float
    depth_params: dict[str, Any] | None
    image_path: str
    image_name: str
    depth_path: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    """
        Complete dataset description consumed by Scene.
    """
    point_cloud: BasicPointCloud
    train_cameras: list[CameraInfo]
    test_cameras: list[CameraInfo]
    nerf_normalization: dict[str, NDArray[np.floating] | float]
    ply_path: str
    is_nerf_synthetic: bool


def resolveDepthPath(depths_folder: str, image_stem: str) -> str:
    """
        Resolve an optional depth map path, preferring array formats over legacy PNG.
    """
    if depths_folder == "":
        return ""
    candidates = [
        os.path.join(depths_folder, f"{image_stem}.npz"),
        os.path.join(depths_folder, f"{image_stem}.npy"),
        os.path.join(depths_folder, f"{image_stem}.png"),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def getNerfppNorm(cam_info: list[CameraInfo]) -> dict[str, NDArray[np.floating] | float]:
    """
        Compute scene translation and radius from training camera centers.
    """

    def get_center_and_diag(cam_centers: list[NDArray[np.floating]]) -> tuple[NDArray[np.floating], float]:
        """
            Return the average camera center and the largest distance from it.
        """
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(
    cam_extrinsics: dict[int, Any],
    cam_intrinsics: dict[int, Any],
    depths_params: dict[str, Any] | None,
    images_folder: str,
    depths_folder: str,
    test_cam_names_list: list[str],
) -> list[CameraInfo]:
    """
        Read Blender/NeRF transform JSON and convert frames into CameraInfo records.
    """
    """
        Convert parsed Colmap camera records into project CameraInfo records.
    """
    cam_infos: list[CameraInfo] = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # Keep progress output on one terminal line while cameras are parsed.
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        depth_path = resolveDepthPath(depths_folder, extr.name[:-n_remove])

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, depth_params=depth_params,
                              image_path=image_path, image_name=image_name, depth_path=depth_path,
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)

    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path: str) -> BasicPointCloud:
    """
        Read a PLY point cloud into the BasicPointCloud container.
    """
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(
    path: str,
    xyz: NDArray[np.floating],
    rgb: NDArray[np.floating],
    normals: NDArray[np.floating] | None = None,
) -> None:
    """
        Write xyz, RGB, and optional normals as a PLY point cloud.
    """
    # PLY stores each vertex as a structured row with position, normal, and color fields.
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    if normals is None:
        normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Convert the structured rows into a PlyData object and write it to disk.
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def limit_point_cloud_size(pcd: BasicPointCloud | None, max_points: int) -> BasicPointCloud | None:
    """
        Randomly subsample an initial point cloud when a positive point limit is set.
    """
    if pcd is None or int(max_points) < 0 or pcd.points.shape[0] <= int(max_points):
        return pcd
    sampled_indices = np.random.choice(pcd.points.shape[0], size=int(max_points), replace=False)
    print("Randomly sampled initial point cloud from {} to {} points.".format(pcd.points.shape[0], int(max_points)))
    return BasicPointCloud(
        points=pcd.points[sampled_indices],
        colors=pcd.colors[sampled_indices],
        normals=pcd.normals[sampled_indices],
    )


def build_random_colmap_point_cloud(
    path: str,
    nerf_normalization: dict[str, NDArray[np.floating] | float],
    num_pts: int = 100_000,
) -> tuple[BasicPointCloud, str]:
    """
        Build a random initial point cloud for Colmap scenes.

        The random points are sampled inside a cube scaled by the scene radius.
    """
    ply_path = os.path.join(path, "random.ply")
    if not os.path.exists(ply_path):
        print(f"Generating random point cloud ({num_pts})...")
        scene_radius = float(nerf_normalization["radius"])
        xyz = np.random.random((num_pts, 3)) * scene_radius * 6.0 - (scene_radius * 3.0)
        shs = np.random.random((num_pts, 3)) / 255.0
        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    point_cloud = fetchPly(ply_path)
    return point_cloud, ply_path


def _read_colmap_camera_metadata(path: str) -> tuple[dict[int, Any], dict[int, Any]]:
    """
        Read Colmap camera extrinsics and intrinsics.

        Binary files are preferred because they are the normal COLMAP output;
        text files are used as a fallback for exported reconstructions.
    """
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        return read_extrinsics_binary(cameras_extrinsic_file), read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        return read_extrinsics_text(cameras_extrinsic_file), read_intrinsics_text(cameras_intrinsic_file)


def _read_depth_params(path: str, depths: str) -> dict[str, Any] | None:
    """
        Read depth-scale metadata and attach the median valid scale to each entry.
    """
    if depths == "":
        return None

    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    try:
        with open(depth_params_file, "r") as f:
            depths_params = json.load(f)
        all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
        if (all_scales > 0).sum():
            med_scale = np.median(all_scales[all_scales > 0])
        else:
            med_scale = 0
        for key in depths_params:
            depths_params[key]["med_scale"] = med_scale
        return depths_params

    except FileNotFoundError:
        print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
        sys.exit(1)


def _resolve_colmap_test_camera_names(path: str, cam_extrinsics: dict[int, Any], eval: bool, llffhold: int) -> list[str]:
    """
        Build the Colmap test-view name list from evaluation settings.

        LLFF-style scenes use every `llffhold`-th sorted image as test. Other
        scenes can provide an explicit `test.txt`.
    """
    if not eval:
        return []
    if "360" in path:
        llffhold = 8
    if llffhold:
        print("------------LLFF HOLD-------------")
        cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
        cam_names = sorted(cam_names)
        return [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
    with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
        return [line.strip() for line in file]


def _load_colmap_initial_point_cloud(
    path: str,
    init_type: str,
    nerf_normalization: dict[str, NDArray[np.floating] | float],
) -> tuple[BasicPointCloud | None, str]:
    """
        Read or generate the initial point cloud for a Colmap scene.

        `sfm` converts COLMAP points to PLY on first use. `random` creates a
        synthetic scene-radius-scaled initialization.
    """
    normalized_init_type = str(init_type).lower()
    if normalized_init_type == "sfm":
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        try:
            return fetchPly(ply_path), ply_path
        except:
            return None, ply_path
    if normalized_init_type == "random":
        return build_random_colmap_point_cloud(path, nerf_normalization)
    raise ValueError("Unsupported init_type for Colmap scene: {}".format(init_type))


def readColmapSceneInfo(
    path: str,
    images: str | None,
    depths: str,
    eval: bool,
    init_type: str = "sfm",
    max_initial_points: int = -1,
    llffhold: int = 8,
) -> SceneInfo:
    """
        Load a full Colmap scene description.

        The function reads camera metadata, creates the train/test split, computes
        scene normalization, loads the initial point cloud, and returns SceneInfo.
    """
    cam_extrinsics, cam_intrinsics = _read_colmap_camera_metadata(path)
    depths_params = _read_depth_params(path, depths)
    test_cam_names_list = _resolve_colmap_test_camera_names(path, cam_extrinsics, eval, llffhold)

    reading_dir = "images" if images == None else images
    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params,
        images_folder=os.path.join(path, reading_dir),
        depths_folder=os.path.join(path, depths) if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]

    nerf_normalization = getNerfppNorm(train_cam_infos)
    pcd, ply_path = _load_colmap_initial_point_cloud(path, init_type, nerf_normalization)
    pcd = limit_point_cloud_size(pcd, max_initial_points)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=False)
    return scene_info

def readCamerasFromTransforms(
    path: str,
    transformsfile: str,
    depths_folder: str,
    white_background: bool,
    is_test: bool,
    extension: str = ".png",
) -> list[CameraInfo]:
    """
        Read Blender/NeRF transform JSON and convert frames into CameraInfo records.
    """
    cam_infos: list[CameraInfo] = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF stores camera-to-world transforms; invert them for renderer-ready camera poses.
            c2w = np.array(frame["transform_matrix"])
            # Convert OpenGL/Blender camera axes (Y up, Z back) to COLMAP axes (Y down, Z forward).
            c2w[:3, 1:3] *= -1

            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            with Image.open(image_path) as image:
                image_width, image_height = image.size

            fovy = focal2fov(fov2focal(fovx, image_width), image_height)
            FovY = fovy 
            FovX = fovx

            depth_path = resolveDepthPath(depths_folder, image_name)

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX,
                            image_path=image_path, image_name=image_name,
                            width=image_width, height=image_height, depth_path=depth_path, depth_params=None, is_test=is_test))
            
    return cam_infos

def readNerfSyntheticInfo(
    path: str,
    white_background: bool,
    depths: str,
    eval: bool,
    extension: str = ".png",
) -> SceneInfo:
    """
        Load a Blender/NeRF synthetic scene description.

        The reader loads train/test transform files, optionally merges test views
        into training when evaluation is disabled, and creates a random point
        cloud because synthetic scenes do not include COLMAP points.
    """

    depths_folder=os.path.join(path, depths) if depths != "" else ""
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", depths_folder, white_background, False, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", depths_folder, white_background, True, extension)
    
    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Synthetic scenes have no COLMAP points, so create a reusable random initialization.
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")
        
        # Sample inside the standard synthetic-scene bounds used by the original 3DGS code.
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           is_nerf_synthetic=True)
    return scene_info

sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
