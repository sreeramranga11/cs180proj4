from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

try:
    import cv2
except ImportError as exc: 
    raise SystemExit("Install OpenCV first: pip install opencv-python") from exc

import numpy as np

try:
    import viser
except ImportError as exc: 
    raise SystemExit("Install Viser first: pip install viser") from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--poses",
        type=Path,
        default=Path("camera_poses.json"),
        help="JSON file output by estimate_pose.py",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Override frustum scale in world units. Default auto-scales to ~30% of the camera radius.",
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Request a public share URL from Viser (expires in ~24h).",
    )
    parser.add_argument(
        "--no-images",
        action="store_true",
        help="Only show frustum wireframes (omit textured planes).",
    )
    return parser.parse_args()


def infer_scale(positions: np.ndarray) -> float:
    if len(positions) < 2:
        return 0.02
    center = positions.mean(axis=0)
    radii = np.linalg.norm(positions - center, axis=1)
    radii = radii[radii > 1e-6]
    if radii.size == 0:
        diffs = np.linalg.norm(np.diff(positions, axis=0), axis=1)
        radii = diffs[diffs > 1e-6]
    typical_radius = np.median(radii) if radii.size else 0.02
    return max(typical_radius * 0.3, 0.002)


def main() -> int:
    args = parse_args()

    data = json.loads(args.poses.read_text())
    camera_matrix = np.array(data["camera_matrix"])
    image_size = data.get("image_size")
    if image_size is None:
        raise ValueError("camera_poses.json lacks image_size; re-run estimate_pose.py.")

    height = image_size["height"]
    width = image_size["width"]
    frustum_fov = 2 * np.arctan2(height / 2, camera_matrix[0, 0])

    positions = []
    for pose in data["poses"]:
        c2w = np.array(pose["camera_to_world"])
        positions.append(c2w[:3, 3])
    positions = np.array(positions) if positions else np.zeros((0, 3))
    frustum_scale = args.scale or infer_scale(positions)

    server = viser.ViserServer()
    host = server.get_host()
    port = server.get_port()
    print(f"Viser running locally at http://{host}:{port}")
    print(f"Frustum scale: {frustum_scale:.4f} (override with --scale)")
    share_url = None
    if args.share:
        print("Requesting share URL...")
        share_url = server.request_share_url()
        if share_url:
            print("Share URL:", share_url)

    image_dir = Path(data["image_dir"])
    for i, pose in enumerate(data["poses"]):
        c2w = np.array(pose["camera_to_world"])
        image_path = image_dir / pose["image"]
        rgb = None
        if not args.no_images:
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"[WARN] Missing image {image_path}, skipping texture.")
            else:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        server.scene.add_camera_frustum(
            f"/cameras/{i}",
            fov=frustum_fov,
            aspect=width / height,
            scale=frustum_scale,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=rgb,
        )

    if share_url:
        print("Open the share URL above in your browser and take screenshots.")
    else:
        print(f"Open http://{host}:{port} in your browser and take screenshots.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nExiting visualization.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
