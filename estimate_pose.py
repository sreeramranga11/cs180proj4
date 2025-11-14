from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import cv2
except ImportError as exc:
    raise SystemExit(
        "OpenCV (cv2) is required. Install it with 'pip install opencv-python'."
    ) from exc

import numpy as np


ARUCO_DICT_OPTIONS = [
    name
    for name in dir(cv2.aruco)
    if name.startswith("DICT_") and isinstance(getattr(cv2.aruco, name), int)
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("camera"),
        help="Directory containing the object-scan photos.",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("camera_calibration.json"),
        help="JSON file produced by calibrate_camera.py.",
    )
    parser.add_argument(
        "--tag-size-meters",
        type=float,
        default=None,
        help="Physical tag size. Defaults to the value stored in the calibration JSON.",
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default="DICT_4X4_50",
        choices=ARUCO_DICT_OPTIONS,
        help="Which predefined ArUco dictionary to use.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("camera_poses.json"),
        help="Where to save the pose results.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image pose information.",
    )
    return parser.parse_args()


def load_calibration(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Calibration file '{path}' not found.")
    return json.loads(path.read_text())


def find_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory '{image_dir}' does not exist.")
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    images = sorted(p for p in image_dir.iterdir() if p.suffix.lower() in exts)
    if not images:
        raise FileNotFoundError(
            f"No image files found in '{image_dir}'. Supported extensions: {sorted(exts)}"
        )
    return images


def build_object_corners(tag_size: float) -> np.ndarray:
    return np.array(
        [
            [0.0, 0.0, 0.0],
            [tag_size, 0.0, 0.0],
            [tag_size, tag_size, 0.0],
            [0.0, tag_size, 0.0],
        ],
        dtype=np.float32,
    )


def detect_markers(image: np.ndarray, dictionary_name: str) -> Tuple[List[np.ndarray], np.ndarray]:
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(image)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            image, aruco_dict, parameters=parameters
        )
    return corners or [], ids


def compute_reprojection_error(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, distortion)
    diff = projected.reshape(-1, 2) - image_points
    return float(np.mean(np.linalg.norm(diff, axis=1)))


def world_to_camera_matrix(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(rvec)
    extrinsics = np.eye(4, dtype=np.float64)
    extrinsics[:3, :3] = rotation
    extrinsics[:3, 3] = tvec.reshape(-1)
    return extrinsics


def camera_to_world_matrix(w2c: np.ndarray) -> np.ndarray:
    rotation = w2c[:3, :3]
    translation = w2c[:3, 3]
    c2w = np.eye(4, dtype=np.float64)
    c2w[:3, :3] = rotation.T
    c2w[:3, 3] = -rotation.T @ translation
    return c2w


def main() -> int:
    args = parse_args()

    calibration = load_calibration(args.calibration)
    camera_matrix = np.array(calibration["camera_matrix"], dtype=np.float64)
    distortion = np.array(calibration["distortion_coefficients"], dtype=np.float64)
    tag_size = args.tag_size_meters or calibration.get("tag_size_meters")
    image_size = calibration.get("image_size")
    if tag_size is None:
        raise ValueError(
            "Tag size is required (supply --tag-size-meters or include it in the calibration JSON)."
        )

    object_points = build_object_corners(tag_size)
    image_paths = find_images(args.image_dir)

    pose_results = []
    successes = 0

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Unable to read image '{image_path}'. Skipping.")
            continue

        corners, ids = detect_markers(image, args.dictionary)
        if ids is None or len(corners) == 0:
            if args.verbose:
                print(f"[INFO] No tags detected in {image_path.name}")
            continue

        image_points = corners[0].reshape(-1, 2).astype(np.float32)
        success, rvec, tvec = cv2.solvePnP(
            object_points,
            image_points,
            camera_matrix,
            distortion,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

        if not success:
            print(f"[WARN] solvePnP failed for '{image_path.name}'.")
            continue

        w2c = world_to_camera_matrix(rvec, tvec)
        c2w = camera_to_world_matrix(w2c)
        reproj_error = compute_reprojection_error(
            object_points, image_points, rvec, tvec, camera_matrix, distortion
        )

        successes += 1
        if args.verbose:
            tx, ty, tz = c2w[:3, 3]
            print(
                f"[INFO] {image_path.name}: tag {int(ids[0][0])}, "
                f"position = ({tx:.3f}, {ty:.3f}, {tz:.3f}) m, "
                f"reproj err = {reproj_error:.4f}px"
            )

        pose_results.append(
            {
                "image": image_path.name,
                "tag_id": int(ids[0][0]),
                "rvec": rvec.reshape(-1).tolist(),
                "tvec": tvec.reshape(-1).tolist(),
                "world_to_camera": w2c.tolist(),
                "camera_to_world": c2w.tolist(),
                "reprojection_error_px": reproj_error,
            }
        )

    print(f"Processed {len(image_paths)} images. Poses estimated for {successes} of them.")

    output = {
        "calibration_file": str(args.calibration),
        "image_dir": str(args.image_dir),
        "tag_size_meters": tag_size,
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.tolist(),
        "poses": pose_results,
    }
    if image_size is not None:
        output["image_size"] = image_size
    args.output_json.write_text(json.dumps(output, indent=2))
    print(f"Saved pose data to {args.output_json}")

    if successes == 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
