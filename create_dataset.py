from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--poses",
        type=Path,
        default=Path("camera_poses.json"),
        help="JSON output by estimate_pose.py",
    )
    parser.add_argument(
        "--calibration",
        type=Path,
        default=Path("camera_calibration.json"),
        help="Calibration JSON that stores the raw intrinsics and distortion coeffs.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help=(
            "Alpha passed to cv2.getOptimalNewCameraMatrix (0=crop aggressively,"
            " 1=keep all pixels, more black borders)."
        ),
    )
    parser.add_argument(
        "--dataset-out",
        type=Path,
        default=Path("nerf_dataset.npz"),
        help="Where to save the packaged dataset.",
    )
    parser.add_argument(
        "--undistorted-dir",
        type=Path,
        default=Path("undistorted"),
        help="Directory to write undistorted images (set to '' to skip).",
    )
    parser.add_argument(
        "--no-image-export",
        action="store_true",
        help="Skip writing undistorted image files to disk.",
    )
    parser.add_argument(
        "--val-count",
        type=int,
        default=3,
        help="Number of validation images to sample.",
    )
    parser.add_argument(
        "--test-count",
        type=int,
        default=3,
        help="Number of test images to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the train/val/test split.",
    )
    return parser.parse_args()


def load_json(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Required file '{path}' not found.")
    return json.loads(path.read_text())


def compute_new_camera_matrix(
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    image_size: Tuple[int, int],
    alpha: float,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    w, h = image_size
    new_K, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        distortion,
        (w, h),
        alpha,
        (w, h),
    )
    x, y, w_roi, h_roi = roi
    if w_roi == 0 or h_roi == 0:
        x, y, w_roi, h_roi = 0, 0, w, h
    adjusted_K = new_K.copy()
    adjusted_K[0, 2] -= x
    adjusted_K[1, 2] -= y
    return adjusted_K, (x, y, w_roi, h_roi)


def build_remap(
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
    new_camera_matrix: np.ndarray,
    image_size: Tuple[int, int],
) -> Tuple[np.ndarray, np.ndarray]:
    w, h = image_size
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        distortion,
        None,
        new_camera_matrix,
        (w, h),
        cv2.CV_32FC1,
    )
    return map1, map2


def split_indices(
    total: int, val_count: int, test_count: int, seed: int
) -> Tuple[List[int], List[int], List[int]]:
    if val_count < 0 or test_count < 0:
        raise ValueError("val-count and test-count must be >= 0.")
    if val_count + test_count >= total:
        raise ValueError("Need at least one training frame; reduce val/test counts.")
    rng = random.Random(seed)
    indices = list(range(total))
    rng.shuffle(indices)
    test_idx = sorted(indices[:test_count])
    val_idx = sorted(indices[test_count : test_count + val_count])
    test_set = set(test_idx)
    val_set = set(val_idx)
    train_idx = sorted(i for i in range(total) if i not in test_set and i not in val_set)
    return train_idx, val_idx, test_idx


def gather(array: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    if len(indices) == 0:
        return np.empty((0, *array.shape[1:]), dtype=array.dtype)
    return array[indices]


def main() -> int:
    args = parse_args()

    poses_data = load_json(args.poses)
    calibration = load_json(args.calibration)

    image_dir = Path(poses_data["image_dir"])
    camera_matrix = np.array(calibration["camera_matrix"], dtype=np.float64)
    distortion = np.array(calibration["distortion_coefficients"], dtype=np.float64)

    sample_image_path = image_dir / poses_data["poses"][0]["image"]
    sample_image = cv2.imread(str(sample_image_path))
    if sample_image is None:
        raise FileNotFoundError(f"Couldn't read sample image '{sample_image_path}'.")
    h, w = sample_image.shape[:2]
    new_camera_matrix, roi = compute_new_camera_matrix(
        camera_matrix, distortion, (w, h), args.alpha
    )
    map1, map2 = build_remap(camera_matrix, distortion, new_camera_matrix, (w, h))
    x, y, w_roi, h_roi = roi

    export_images = not args.no_image_export
    export_dir = None
    if export_images:
        export_dir = args.undistorted_dir
        export_dir.mkdir(parents=True, exist_ok=True)

    undistorted_images: List[np.ndarray] = []
    c2ws: List[np.ndarray] = []
    used_images: List[str] = []

    for pose in poses_data["poses"]:
        image_path = image_dir / pose["image"]
        bgr = cv2.imread(str(image_path))
        if bgr is None:
            print(f"[WARN] Unable to read '{image_path}'. Skipping frame.")
            continue
        undistorted = cv2.remap(bgr, map1, map2, interpolation=cv2.INTER_LINEAR)
        cropped = undistorted[y : y + h_roi, x : x + w_roi]
        if export_dir is not None:
            out_path = export_dir / pose["image"]
            cv2.imwrite(str(out_path), cropped)

        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        undistorted_images.append(rgb.astype(np.uint8))
        c2ws.append(np.array(pose["camera_to_world"], dtype=np.float32))
        used_images.append(pose["image"])

    if not undistorted_images:
        raise RuntimeError("No images were processed; check your camera folder paths.")

    images_np = np.stack(undistorted_images, axis=0)
    c2ws_np = np.stack(c2ws, axis=0)
    train_idx, val_idx, test_idx = split_indices(
        images_np.shape[0], args.val_count, args.test_count, args.seed
    )

    dataset = {
        "images_train": gather(images_np, train_idx),
        "c2ws_train": gather(c2ws_np, train_idx),
        "images_val": gather(images_np, val_idx),
        "c2ws_val": gather(c2ws_np, val_idx),
        "c2ws_test": gather(c2ws_np, test_idx),
        "focal": float((new_camera_matrix[0, 0] + new_camera_matrix[1, 1]) / 2.0),
        "intrinsics": new_camera_matrix,
        "principal_point": new_camera_matrix[:2, 2],
        "roi": np.array([x, y, w_roi, h_roi], dtype=np.int32),
        "image_names": np.array(used_images),
        "train_indices": np.array(train_idx),
        "val_indices": np.array(val_idx),
        "test_indices": np.array(test_idx),
    }

    np.savez(args.dataset_out, **dataset)

    print(f"Saved undistorted dataset to {args.dataset_out}")
    print(f"Resolution: {h_roi} x {w_roi}")
    print(f"Train/Val/Test counts: {len(train_idx)}/{len(val_idx)}/{len(test_idx)}")
    if export_dir:
        print(f"Undistorted images written to {export_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
