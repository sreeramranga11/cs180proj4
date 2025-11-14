from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

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
        default=Path("tags"),
        help="Directory that holds the calibration photos",
    )
    parser.add_argument(
        "--tag-size-meters",
        type=float,
        default=0.02,
        help="Edge length of each printed ArUco tag (meters)",
    )
    parser.add_argument(
        "--dictionary",
        type=str,
        default="DICT_4X4_50",
        choices=ARUCO_DICT_OPTIONS,
        help="Which predefined ArUco dictionary to use",
    )
    parser.add_argument(
        "--min-detections",
        type=int,
        default=5,
        help="Minimum number of usable images needed before running calibration",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=Path("camera_calibration.json"),
        help="Where to store the computed intrinsics/distortion",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-image diagnostics while processing",
    )
    return parser.parse_args()


def find_images(image_dir: Path) -> List[Path]:
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory '{image_dir}' does not exist")
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    image_paths = sorted(
        [p for p in image_dir.iterdir() if p.suffix.lower() in exts]
    )
    if not image_paths:
        raise FileNotFoundError(
            f"No image files found in '{image_dir}'. Supported extensions: {sorted(exts)}"
        )
    return image_paths


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


def detect_markers(
    image: np.ndarray,
    dictionary_name: str,
) -> Tuple[List[np.ndarray], np.ndarray]:
    aruco_dict = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, dictionary_name))
    parameters = cv2.aruco.DetectorParameters()

    if hasattr(cv2.aruco, "ArucoDetector"):
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        corners, ids, _ = detector.detectMarkers(image)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(
            image,
            aruco_dict,
            parameters=parameters,
        )
    return corners or [], ids


def calibrate(
    object_points: List[np.ndarray],
    image_points: List[np.ndarray],
    image_size: Tuple[int, int],
) -> Tuple[float, np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray]]:
    return cv2.calibrateCamera(
        objectPoints=object_points,
        imagePoints=image_points,
        imageSize=image_size,
        cameraMatrix=None,
        distCoeffs=None,
    )


def compute_reprojection_error(
    object_points: List[np.ndarray],
    image_points: List[np.ndarray],
    rvecs: List[np.ndarray],
    tvecs: List[np.ndarray],
    camera_matrix: np.ndarray,
    distortion: np.ndarray,
) -> float:
    total_err = 0.0
    total_points = 0
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, distortion)
        error = cv2.norm(img, projected.reshape(-1, 2), cv2.NORM_L2)
        total_err += error * error
        total_points += len(obj)
    return np.sqrt(total_err / max(total_points, 1))


def main() -> int:
    args = parse_args()

    try:
        image_paths = find_images(args.image_dir)
    except FileNotFoundError as exc:
        print(exc, file=sys.stderr)
        return 1

    object_corners = build_object_corners(args.tag_size_meters)

    collected_object_points: List[np.ndarray] = []
    collected_image_points: List[np.ndarray] = []
    image_size: Tuple[int, int] | None = None

    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"[WARN] Unable to read image '{image_path}'. Skipping.")
            continue

        if image_size is None:
            image_size = (image.shape[1], image.shape[0])

        corners, ids = detect_markers(image, args.dictionary)
        if ids is None or len(corners) == 0:
            if args.verbose:
                print(f"[INFO] No tags detected in {image_path.name}")
            continue

        selected_corners = corners[0].reshape(-1, 2).astype(np.float32)
        collected_object_points.append(object_corners)
        collected_image_points.append(selected_corners)

        if args.verbose:
            tag_id = int(ids[0][0])
            print(f"[INFO] {image_path.name}: tag {tag_id} detected")

    valid_images = len(collected_object_points)
    if valid_images < args.min_detections:
        print(
            f"Only {valid_images} usable images detected (min required {args.min_detections})."
        )
        return 1

    assert image_size is not None, "image_size should be set when detections exist"

    rms, camera_matrix, distortion, rvecs, tvecs = calibrate(
        collected_object_points,
        collected_image_points,
        image_size,
    )

    reproj_error = compute_reprojection_error(
        collected_object_points,
        collected_image_points,
        rvecs,
        tvecs,
        camera_matrix,
        distortion,
    )

    print("Calibration succeeded!")
    print(f" - Images used: {valid_images}")
    print(f" - Image size: {image_size[0]} x {image_size[1]} px")
    print(f" - RMS error reported by OpenCV: {rms:.6f}")
    print(f" - Mean reprojection error: {reproj_error:.6f} px")
    print(" - Camera matrix (K):")
    print(camera_matrix)
    print(" - Distortion coefficients (k1, k2, p1, p2, k3, ...):")
    print(distortion.ravel())

    calibration = {
        "image_count": valid_images,
        "image_size": {"width": image_size[0], "height": image_size[1]},
        "tag_size_meters": args.tag_size_meters,
        "camera_matrix": camera_matrix.tolist(),
        "distortion_coefficients": distortion.tolist(),
        "rms": float(rms),
        "mean_reprojection_error": float(reproj_error),
    }

    if args.output_json:
        args.output_json.write_text(json.dumps(calibration, indent=2))
        print(f"Saved calibration to {args.output_json}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
