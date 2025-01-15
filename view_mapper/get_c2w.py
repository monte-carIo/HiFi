import os
import numpy as np
from PIL import Image

def read_cameras_txt(filepath):
    """Reads the cameras.txt file from the sparse folder."""
    cameras = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            elements = line.split()
            camera_id = int(elements[0])
            params = list(map(float, elements[4:]))
            cameras[camera_id] = {
                "model": elements[1],
                "width": int(elements[2]),
                "height": int(elements[3]),
                "params": params
            }
    return cameras

def read_images_txt(filepath):
    """Reads the images.txt file from the sparse folder."""
    images = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            if line.startswith('#') or not line.strip():
                continue
            if idx % 2 == 0:
                elements = line.split()
                image_id = int(elements[0])
                qw, qx, qy, qz = map(float, elements[1:5])  # Quaternion
                tx, ty, tz = map(float, elements[5:8])      # Translation
                camera_id = int(elements[8])
                name = elements[9]
                images[image_id] = {
                    "qvec": np.array([qw, qx, qy, qz]),
                    "tvec": np.array([tx, ty, tz]),
                    "camera_id": camera_id,
                    "name": name
                }
    return images

def read_points3d_txt(filepath):
    """Reads the points3D.txt file from the sparse folder."""
    points3d = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.startswith('#') or not line.strip():
                continue
            elements = line.split()
            point_id = int(elements[0])
            xyz = np.array(list(map(float, elements[1:4])))
            rgb = list(map(int, elements[4:7]))
            error = float(elements[7])
            track = list(map(int, elements[8:]))
            points3d[point_id] = {
                "xyz": xyz,
                "rgb": rgb,
                "error": error,
                "track": track
            }
    return points3d

def compute_c2w(qvec, tvec):
    """Converts COLMAP camera pose (qvec, tvec) to camera-to-world matrix."""
    # Quaternion to rotation matrix
    q0, q1, q2, q3 = qvec
    R = np.array([
        [1 - 2*q2**2 - 2*q3**2, 2*q1*q2 - 2*q0*q3, 2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3, 1 - 2*q1**2 - 2*q3**2, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2, 2*q2*q3 + 2*q0*q1, 1 - 2*q1**2 - 2*q2**2]
    ])
    tvec = tvec.reshape((3, 1))
    c2w = np.hstack((R.T, -R.T @ tvec))  # Convert to camera-to-world
    c2w = np.vstack((c2w, [0, 0, 0, 1]))  # Homogeneous transformation
    return c2w

def extract_colmap_data(sparse_folder):
    """Extracts all information from COLMAP sparse folder."""
    cameras = read_cameras_txt(os.path.join(sparse_folder, "cameras.txt"))
    images = read_images_txt(os.path.join(sparse_folder, "images.txt"))
    points3d = read_points3d_txt(os.path.join(sparse_folder, "points3D.txt"))

    # Combine and print information in a dictionary with image_name as key
    scene_data = {}
    for image_id, image_data in images.items():
        c2w = compute_c2w(image_data["qvec"], image_data["tvec"])
        camera_params = cameras[image_data["camera_id"]]
        scene_data[image_data["name"]] = {
            "c2w": c2w,
            "intrinsics": camera_params,
            "camera_id": image_data["camera_id"]
        }

    return scene_data


def project_points(K, c2w, points3d):
    """Project 3D points into the image plane to compute sparse depth."""
    # Compute world-to-camera transformation
    w2c = np.linalg.inv(c2w)
    R, t = w2c[:3, :3], w2c[:3, 3]
    
    # Transform points to the camera coordinate frame
    points_cam = R @ points3d.T + t[:, np.newaxis]
    
    # Filter out points behind the camera
    valid_mask = points_cam[2, :] > 0
    points_cam = points_cam[:, valid_mask]
    
    # Project points onto the image plane
    points_proj = K @ points_cam[:3, :]
    points_proj /= points_proj[2, :]  # Normalize by depth (z)
    
    # Return 2D pixel coordinates and corresponding depths
    return points_proj[:2, :].T, points_cam[2, valid_mask]

def create_sparse_depth_map(image_width, image_height, pixel_coords, depths):
    """Create a sparse depth map with projected points."""
    sparse_depth = np.zeros((image_height, image_width), dtype=np.float32)
    for (x, y), depth in zip(pixel_coords, depths):
        px, py = int(round(x)), int(round(y))
        if 0 <= px < image_width and 0 <= py < image_height:
            sparse_depth[py, px] = depth  # Assign depth to pixel location
    return sparse_depth

def save_scene_and_depth(sparse_folder):
    # Load COLMAP data
    cameras = read_cameras_txt(os.path.join(sparse_folder, "cameras.txt"))
    images = read_images_txt(os.path.join(sparse_folder, "images.txt"))
    points3d = read_points3d_txt(os.path.join(sparse_folder, "points3D.txt"))

    scene_data = {}
    depth_maps = {}

    # For each image, compute the sparse depth map
    for image_data in images.values():
        # Extract intrinsics and extrinsics
        camera_params = cameras[image_data["camera_id"]]
        width, height = camera_params["width"], camera_params["height"]
        K = np.array([
            [camera_params["params"][0], 0, camera_params["params"][2]],
            [0, camera_params["params"][1], camera_params["params"][3]],
            [0, 0, 1]
        ])
        c2w = compute_c2w(image_data["qvec"], image_data["tvec"])

        # Extract 3D points
        points_xyz = np.array([points3d[pid]["xyz"] for pid in points3d if pid in points3d])
        
        # Project 3D points into the image
        pixel_coords, depths = project_points(K, c2w, points_xyz)
        
        # Create sparse depth map
        sparse_depth = create_sparse_depth_map(width, height, pixel_coords, depths)
        
        # Append scene data and depth map to dictionaries
        scene_data[image_data["name"].split('.')[0]] = {
            "c2w": c2w,
            "intrinsics": camera_params,
            "camera_id": image_data["camera_id"]
        }
        depth_maps[image_data["name"].split('.')[0]] = sparse_depth

    return scene_data, depth_maps
