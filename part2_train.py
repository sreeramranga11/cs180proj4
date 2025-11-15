from __future__ import annotations

import argparse
import json
import logging
import math
import shutil
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

try:
    import viser  
except ImportError:
    viser = None  


# ---------------------------------------------------------------------------
# Logging / configuration helpers
# ---------------------------------------------------------------------------


def setup_logger(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def get_device(name: Optional[str] = None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# Camera math (Part 2.1)


def build_intrinsics(focal: float, width: int, height: int) -> torch.Tensor:
    cx = width / 2.0
    cy = height / 2.0
    K = torch.tensor(
        [
            [focal, 0.0, cx],
            [0.0, focal, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    return K


def dataset_intrinsics(dataset: SceneDataset, width: int, height: int) -> torch.Tensor:
    if dataset.intrinsics is not None:
        return torch.from_numpy(dataset.intrinsics).float()
    return build_intrinsics(dataset.focal, width, height)


def mse_to_psnr(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(torch.clamp(mse, min=1e-10))


def recommend_bounds(c2ws: np.ndarray, margin: float = 0.1) -> Tuple[float, float]:
    dists = np.linalg.norm(c2ws[:, :3, 3], axis=1)
    near = max(0.01, dists.min() - margin)
    far = dists.max() + margin
    return near, far


def transform_points(c2w: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Transform points from camera to world coordinates using a c2w matrix."""
    R = c2w[:3, :3]
    t = c2w[:3, 3]
    world = (points @ R.T) + t
    logging.debug("transform_points: %s -> %s", points.shape, world.shape)
    return world


def pixel_to_camera(
    K: torch.Tensor,
    uv: torch.Tensor,
    depth: torch.Tensor,
) -> torch.Tensor:
    """Convert pixel coordinates + depth to camera coordinates."""
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (uv[..., 0] - cx) / fx * depth
    y = (uv[..., 1] - cy) / fy * depth
    z = depth
    cam = torch.stack((x, y, z), dim=-1)
    logging.debug("pixel_to_camera: uv=%s depth=%s -> %s", uv.shape, depth.shape, cam.shape)
    return cam


def pixel_to_ray(K: torch.Tensor, c2w: torch.Tensor, uv: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (origin, direction) for each pixel coordinate (x,y)."""
    device = c2w.device
    depth = torch.ones(uv.shape[:-1], device=device)
    cam_pts = pixel_to_camera(K, uv, depth)
    world_pts = transform_points(c2w, cam_pts)
    origin = c2w[:3, 3].unsqueeze(0).expand(world_pts.shape[0], 3)
    dirs = world_pts - origin
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    logging.debug("pixel_to_ray: uvs=%s -> origins=%s dirs=%s", uv.shape, origin.shape, dirs.shape)
    return origin, dirs


# Dataset + sampling (Parts 2.2+2.3)


@dataclass
class SceneDataset:
    images_train: np.ndarray
    c2ws_train: np.ndarray
    images_val: np.ndarray
    c2ws_val: np.ndarray
    c2ws_test: np.ndarray
    focal: float
    intrinsics: Optional[np.ndarray] = None


def load_lego_dataset(path: Path) -> SceneDataset:
    data = np.load(path)
    dataset = SceneDataset(
        images_train=(data["images_train"] / 255.0).astype(np.float32),
        c2ws_train=data["c2ws_train"],
        images_val=(data["images_val"] / 255.0).astype(np.float32),
        c2ws_val=data["c2ws_val"],
        c2ws_test=data["c2ws_test"],
        focal=float(data["focal"]),
        intrinsics=None,
    )
    logging.debug(
        "Loaded lego dataset: train=%s val=%s test=%s",
        dataset.images_train.shape,
        dataset.images_val.shape,
        dataset.c2ws_test.shape,
    )
    return dataset


def resize_images(images: np.ndarray, max_dim: Optional[int]) -> Tuple[np.ndarray, float]:
    if max_dim is None or max_dim <= 0:
        return images.astype(np.float32) / 255.0, 1.0
    h, w = images.shape[1:3]
    max_current = max(h, w)
    if max_current <= max_dim:
        return images.astype(np.float32) / 255.0, 1.0
    scale = max_dim / max_current
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = []
    for img in images:
        pil = Image.fromarray(img)
        pil = pil.resize((new_w, new_h), Image.Resampling.LANCZOS)
        resized.append(np.asarray(pil))
    resized = np.stack(resized).astype(np.float32) / 255.0
    return resized, scale


def load_custom_dataset(path: Path, max_dim: Optional[int]) -> SceneDataset:
    data = np.load(path)
    images_train, scale = resize_images(data["images_train"], max_dim)
    images_val, _ = resize_images(data["images_val"], max_dim)
    c2ws_train = data["c2ws_train"]
    c2ws_val = data["c2ws_val"]
    c2ws_test = data["c2ws_test"]
    intrinsics = data["intrinsics"].astype(np.float32)
    intrinsics[:2, :] *= scale
    focal = float((intrinsics[0, 0] + intrinsics[1, 1]) / 2.0)
    logging.info(
        "Custom dataset loaded: train=%s val=%s scale=%.3f",
        images_train.shape,
        images_val.shape,
        scale,
    )
    return SceneDataset(
        images_train=images_train,
        c2ws_train=c2ws_train,
        images_val=images_val,
        c2ws_val=c2ws_val,
        c2ws_test=c2ws_test,
        focal=focal,
        intrinsics=intrinsics,
    )


class RaysData:
    """Flattened ray dataset that supports random sampling."""

    def __init__(
        self,
        images: np.ndarray,
        c2ws: np.ndarray,
        K: torch.Tensor,
        device: torch.device,
    ) -> None:
        self.device = device
        self.images = images
        self.c2ws = c2ws
        self.height, self.width = images.shape[1:3]
        self.K = K.to(device)

        logging.info(
            "Precomputing rays for %d images at %dx%d",
            images.shape[0],
            self.width,
            self.height,
        )
        rays_o, rays_d, pixels, uvs, img_ids = self._precompute_all_rays()
        self.rays_o = rays_o
        self.rays_d = rays_d
        self.pixels = pixels
        self.uvs = uvs
        self.image_ids = img_ids

    def _precompute_all_rays(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        H, W = self.height, self.width
        base_uvs = np.stack(np.meshgrid(np.arange(W), np.arange(H), indexing="xy"), axis=-1).reshape(-1, 2)
        base_uvs = base_uvs.astype(np.float32) + 0.5  # pixel center offset

        rays_o_list = []
        rays_d_list = []
        pixels_list = []
        uvs_list = []
        ids_list = []

        for img_id, (image, c2w_np) in enumerate(zip(self.images, self.c2ws)):
            uv = torch.from_numpy(base_uvs).to(self.device)
            c2w = torch.from_numpy(c2w_np.astype(np.float32)).to(self.device)
            origins, dirs = pixel_to_ray(self.K, c2w, uv)
            colors = torch.from_numpy(image.reshape(-1, 3).astype(np.float32)).to(self.device)

            rays_o_list.append(origins)
            rays_d_list.append(dirs)
            pixels_list.append(colors)
            uvs_list.append(uv)
            ids_list.append(torch.full((H * W,), img_id, device=self.device, dtype=torch.long))

        rays_o = torch.cat(rays_o_list, dim=0)
        rays_d = torch.cat(rays_d_list, dim=0)
        pixels = torch.cat(pixels_list, dim=0)
        uvs = torch.cat(uvs_list, dim=0)
        ids = torch.cat(ids_list, dim=0)

        logging.info(
            "Total rays stored: %d (%.2f MB)",
            rays_o.shape[0],
            (rays_o.numel() * 4 * 2) / (1024 ** 2),
        )
        return rays_o, rays_d, pixels, uvs, ids

    def sample_rays(self, count: int, restrict_to: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """Return random rays, optionally restricting to a given image id."""
        if restrict_to is not None:
            eligible = torch.where(self.image_ids == restrict_to)[0]
            if eligible.numel() == 0:
                raise ValueError(f"No rays found for image id {restrict_to}.")
            indices = eligible[torch.randint(0, eligible.shape[0], (count,), device=self.device)]
        else:
            indices = torch.randint(0, self.rays_o.shape[0], (count,), device=self.device)

        batch = {
            "rays_o": self.rays_o[indices],
            "rays_d": self.rays_d[indices],
            "pixels": self.pixels[indices],
            "uvs": self.uvs[indices],
            "image_ids": self.image_ids[indices],
        }
        logging.debug(
            "sample_rays: count=%d restrict=%s -> indices shape=%s",
            count,
            restrict_to,
            batch["rays_o"].shape,
        )
        return batch


# Positional Encoding & Neural Radiance Field (Part 2.4)


class PositionalEncoding(nn.Module):
    def __init__(self, num_dims: int, max_freq: int):
        super().__init__()
        self.num_dims = num_dims
        self.max_freq = max_freq
        frequencies = 2.0 ** torch.arange(max_freq)
        self.register_buffer("frequencies", frequencies, persistent=False)

    @property
    def out_dim(self) -> int:
        return self.num_dims * (2 * self.max_freq + 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        encodings = [x]
        for freq in self.frequencies:
            for fn in (torch.sin, torch.cos):
                encodings.append(fn(x * freq * math.pi))
        encoded = torch.cat(encodings, dim=-1)
        logging.debug("PositionalEncoding: input=%s -> %s", x.shape, encoded.shape)
        return encoded


class NeuralRadianceField(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 256,
        depth: int = 8,
        xyz_freq: int = 10,
        dir_freq: int = 4,
    ):
        super().__init__()
        self.xyz_encoder = PositionalEncoding(3, xyz_freq)
        self.dir_encoder = PositionalEncoding(3, dir_freq)

        xyz_in_dim = self.xyz_encoder.out_dim
        dir_in_dim = self.dir_encoder.out_dim

        self.skip_layer = depth // 2
        layers = []
        in_dim = xyz_in_dim
        for i in range(depth):
            layers.append(nn.Linear(in_dim, hidden_dim))
            if i == self.skip_layer - 1:
                in_dim = hidden_dim + xyz_in_dim
            else:
                in_dim = hidden_dim
        self.xyz_layers = nn.ModuleList(layers)
        self.activation = nn.ReLU(inplace=True)

        self.sigma_head = nn.Sequential(nn.Linear(hidden_dim, 1), nn.ReLU(inplace=True))
        self.feature_head = nn.Linear(hidden_dim, hidden_dim)
        self.color_layers = nn.Sequential(
            nn.Linear(hidden_dim + dir_in_dim, hidden_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid(),
        )

    def forward(self, points: torch.Tensor, directions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logging.debug("NRF forward: points=%s directions=%s", points.shape, directions.shape)
        xyz_enc = self.xyz_encoder(points)
        h = xyz_enc
        for i, layer in enumerate(self.xyz_layers):
            h = self.activation(layer(h))
            if i == self.skip_layer - 1:
                h = torch.cat([h, xyz_enc], dim=-1)

        sigma = self.sigma_head(h)
        features = self.feature_head(h)
        dir_enc = self.dir_encoder(directions)
        color_input = torch.cat([features, dir_enc], dim=-1)
        rgb = self.color_layers(color_input)
        logging.debug("NRF output: sigma=%s rgb=%s", sigma.shape, rgb.shape)
        return sigma, rgb


def sample_along_rays(
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    n_samples: int,
    near: float,
    far: float,
    perturb: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Uniform samples along each ray between near/far bounds."""
    device = rays_o.device
    num_rays = rays_o.shape[0]
    bin_edges = torch.linspace(near, far, steps=n_samples + 1, device=device)
    bin_edges = bin_edges.unsqueeze(0).expand(num_rays, -1)
    lower = bin_edges[:, :-1]
    upper = bin_edges[:, 1:]
    if perturb:
        t_rand = torch.rand_like(lower)
        z_vals = lower + (upper - lower) * t_rand
    else:
        z_vals = 0.5 * (lower + upper)

    deltas = upper - lower
    points = rays_o.unsqueeze(1) + rays_d.unsqueeze(1) * z_vals.unsqueeze(-1)
    logging.debug(
        "sample_along_rays: rays=%s samples=%d -> z_vals=%s points=%s deltas=%s",
        rays_o.shape[0],
        n_samples,
        z_vals.shape,
        points.shape,
        deltas.shape,
    )
    return z_vals, points, deltas


def volume_render(
    sigmas: torch.Tensor,
    rgbs: torch.Tensor,
    deltas: torch.Tensor,
    white_bg: bool = False,
) -> torch.Tensor:
    """Compute expected color for each ray using discrete volume rendering."""
    sigmas = sigmas.squeeze(-1)
    sigma_delta = sigmas * deltas
    alphas = 1.0 - torch.exp(-sigma_delta)
    accumulated = torch.cumsum(sigma_delta, dim=-1)
    transmittance = torch.exp(
        -torch.cat(
            [torch.zeros((sigmas.shape[0], 1), device=sigmas.device), accumulated[:, :-1]],
            dim=-1,
        )
    )
    weights = transmittance * alphas
    rendered = (weights.unsqueeze(-1) * rgbs).sum(dim=1)
    if white_bg:
        rendered = rendered + (1.0 - weights.sum(dim=1, keepdim=True))
    logging.debug(
        "volume_render: sigmas=%s rgbs=%s deltas=%s white_bg=%s -> rendered=%s",
        sigmas.shape,
        rgbs.shape,
        deltas.shape,
        white_bg,
        rendered.shape,
    )
    return rendered


def render_image(
    model: NeuralRadianceField,
    K: torch.Tensor,
    c2w: torch.Tensor,
    height: int,
    width: int,
    near: float,
    far: float,
    n_samples: int,
    device: torch.device,
    chunk: int = 8192,
    white_bg: bool = False,
) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        xs, ys = torch.meshgrid(
            torch.arange(width, device=device),
            torch.arange(height, device=device),
            indexing="xy",
        )
        uv = torch.stack([xs.reshape(-1), ys.reshape(-1)], dim=-1).float() + 0.5
        c2w = c2w.to(device)
        rays_o, rays_d = pixel_to_ray(K.to(device), c2w, uv)
        colors = []
        for start in range(0, rays_o.shape[0], chunk):
            end = start + chunk
            ro = rays_o[start:end]
            rd = rays_d[start:end]
            _, pts, deltas = sample_along_rays(ro, rd, n_samples, near, far, perturb=False)
            dirs = rd.unsqueeze(1).expand(-1, n_samples, -1)
            sigma, rgb = model(pts.reshape(-1, 3), dirs.reshape(-1, 3))
            sigma = sigma.view(-1, n_samples, 1)
            rgb = rgb.view(-1, n_samples, 3)
            chunk_colors = volume_render(sigma, rgb, deltas, white_bg=white_bg).cpu()
            colors.append(chunk_colors)
        image = torch.cat(colors, dim=0).view(height, width, 3)
        return image


def look_at_origin(pos: np.ndarray) -> np.ndarray:
    forward = -pos / np.linalg.norm(pos)
    up = np.array([0.0, 1.0, 0.0])
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    up = np.cross(forward, right)
    c2w = np.eye(4)
    c2w[:3, 0] = right
    c2w[:3, 1] = up
    c2w[:3, 2] = forward
    c2w[:3, 3] = pos
    return c2w


def rot_x(phi: float) -> np.ndarray:
    return np.array(
        [
            [math.cos(phi), -math.sin(phi), 0, 0],
            [math.sin(phi), math.cos(phi), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )


def render_orbit_gif(
    model: NeuralRadianceField,
    K: torch.Tensor,
    height: int,
    width: int,
    near: float,
    far: float,
    n_samples: int,
    device: torch.device,
    output_path: Path,
    start_pos: np.ndarray,
    num_frames: int,
    fps: int,
    chunk: int,
    white_bg: bool,
) -> None:
    frames: List[np.ndarray] = []
    for phi in np.linspace(0.0, 2 * math.pi, num_frames, endpoint=False):
        c2w = look_at_origin(start_pos)
        extrinsic = rot_x(phi) @ c2w
        image = render_image(
            model,
            K,
            torch.from_numpy(extrinsic).float(),
            height,
            width,
            near,
            far,
            n_samples,
            device,
            chunk=chunk,
            white_bg=white_bg,
        )
        frames.append((image.clamp(0, 1).numpy() * 255).astype(np.uint8))
    imageio.mimsave(output_path, frames, fps=fps, loop=0, duration=1.0 / fps)
    logging.info("Saved orbit gif to %s", output_path)


def train_lego_nerf(args: argparse.Namespace, device: torch.device) -> None:
    dataset = load_lego_dataset(Path(args.dataset))
    H, W = dataset.images_train.shape[1:3]
    K = build_intrinsics(dataset.focal, W, H).to(device)
    train_rays = RaysData(dataset.images_train, dataset.c2ws_train, K, device)
    val_rays = RaysData(dataset.images_val, dataset.c2ws_val, K, device) if len(dataset.images_val) else None

    nerf = NeuralRadianceField(
        hidden_dim=args.lego_hidden_dim,
        depth=args.lego_depth,
        xyz_freq=args.lego_xyz_freq,
        dir_freq=args.lego_dir_freq,
    ).to(device)
    optimizer = torch.optim.Adam(nerf.parameters(), lr=args.lego_learning_rate)
    output_dir = args.lego_output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, float]] = []
    eval_samples = args.lego_eval_samples or args.lego_n_samples

    for step in range(1, args.lego_train_steps + 1):
        batch = train_rays.sample_rays(args.lego_batch_size)
        z_vals, pts, deltas = sample_along_rays(
            batch["rays_o"],
            batch["rays_d"],
            args.lego_n_samples,
            args.lego_near,
            args.lego_far,
            perturb=not args.no_perturb,
        )
        dirs = batch["rays_d"].unsqueeze(1).expand(-1, args.lego_n_samples, -1)
        sigma, rgb = nerf(pts.reshape(-1, 3), dirs.reshape(-1, 3))
        sigma = sigma.view(-1, args.lego_n_samples, 1)
        rgb = rgb.view(-1, args.lego_n_samples, 3)
        pred = volume_render(sigma, rgb, deltas)
        loss = F.mse_loss(pred, batch["pixels"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        record: Dict[str, float] = {"step": step, "loss": float(loss.item())}
        if step % args.lego_eval_every == 0 and val_rays is not None:
            with torch.no_grad():
                val_batch = val_rays.sample_rays(min(args.lego_batch_size, val_rays.rays_o.shape[0]))
                z_vals_v, pts_v, deltas_v = sample_along_rays(
                    val_batch["rays_o"],
                    val_batch["rays_d"],
                    eval_samples,
                    args.lego_near,
                    args.lego_far,
                    perturb=False,
                )
                dirs_v = val_batch["rays_d"].unsqueeze(1).expand(-1, eval_samples, -1)
                sigma_v, rgb_v = nerf(pts_v.reshape(-1, 3), dirs_v.reshape(-1, 3))
                sigma_v = sigma_v.view(-1, eval_samples, 1)
                rgb_v = rgb_v.view(-1, eval_samples, 3)
                pred_v = volume_render(sigma_v, rgb_v, deltas_v)
                val_loss = F.mse_loss(pred_v, val_batch["pixels"])
                record["val_loss"] = float(val_loss.item())
                record["val_psnr"] = float(mse_to_psnr(val_loss).item())

        history.append(record)

        if step % args.lego_log_every == 0:
            msg = f"Lego step {step}/{args.lego_train_steps} loss={record['loss']:.4e}"
            if "val_loss" in record:
                msg += f" val={record['val_loss']:.4e} psnr={record['val_psnr']:.2f}"
            logging.info(msg)

        if step % args.lego_render_every == 0:
            img = render_image(
                nerf,
                K,
                torch.from_numpy(dataset.c2ws_train[0]).float(),
                H,
                W,
                args.lego_near,
                args.lego_far,
                eval_samples,
                device,
                chunk=args.lego_chunk_size,
                white_bg=False,
            )
            Image.fromarray((img.clamp(0, 1).numpy() * 255).astype(np.uint8)).save(
                output_dir / f"render_{step:05d}.png"
            )

    (output_dir / "loss_history.json").write_text(json.dumps(history, indent=2))
    torch.save(nerf.state_dict(), output_dir / "model.pt")
    logging.info("Saved lego training artifacts to %s", output_dir)

    if args.lego_render_gif:
        frames = []
        for pose in dataset.c2ws_test[: args.lego_gif_frames]:
            img = render_image(
                nerf,
                K,
                torch.from_numpy(pose).float(),
                H,
                W,
                args.lego_near,
                args.lego_far,
                eval_samples,
                device,
                chunk=args.lego_chunk_size,
                white_bg=False,
            )
            frames.append((img.clamp(0, 1).cpu().numpy() * 255).astype("uint8"))

        imageio.mimsave(
            output_dir / "lego_spherical.gif",
            frames,
            fps=args.lego_gif_fps,
            loop=0,
        )
        logging.info("Saved lego spherical gif to %s", output_dir / "lego_spherical.gif")


def train_custom_nerf(args: argparse.Namespace, device: torch.device) -> None:
    dataset = load_custom_dataset(args.custom_dataset, args.custom_max_dim)
    H, W = dataset.images_train.shape[1:3]
    K = dataset_intrinsics(dataset, W, H).to(device)

    recommended_near, recommended_far = recommend_bounds(dataset.c2ws_train)
    auto_near = max(0.02, recommended_near - 0.15)
    auto_far = recommended_far + 0.05
    if args.custom_near > 0:
        near = args.custom_near
    else:
        near = auto_near
        logging.info("Using auto near bound %.4f (recommended %.4f)", near, recommended_near)
    if args.custom_far > 0:
        far = args.custom_far
    else:
        far = max(auto_far, recommended_far)
        logging.info("Using auto far bound %.4f (recommended %.4f)", far, recommended_far)
    if far <= near:
        far = near + 0.1
    if far < recommended_far:
        logging.warning(
            "custom_far=%.3f is smaller than recommended far=%.3f; clamping to recommended.",
            args.custom_far,
            recommended_far,
        )
        far = recommended_far
    logging.info("Custom training bounds: near=%.4f far=%.4f (recommended %.4f/%.4f)", near, far, recommended_near, recommended_far)

    train_rays = RaysData(dataset.images_train, dataset.c2ws_train, K, device)
    val_rays = None
    if dataset.images_val.size > 0:
        val_rays = RaysData(dataset.images_val, dataset.c2ws_val, K, device)

    nerf = NeuralRadianceField(
        hidden_dim=args.hidden_dim,
        depth=args.depth,
        xyz_freq=args.xyz_freq,
        dir_freq=args.dir_freq,
    ).to(device)
    optimizer = torch.optim.Adam(nerf.parameters(), lr=args.learning_rate)
    output_dir = args.output_dir / "custom_train"
    loss_file = output_dir / "loss_history.json"
    if output_dir.exists() and not args.resume:
        logging.info("Clearing previous training directory %s", output_dir)
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history: List[Dict[str, float]] = []
    start_step = 0
    if args.resume and (output_dir / "model.pt").exists():
        logging.info("Resuming training from %s", output_dir / "model.pt")
        nerf.load_state_dict(torch.load(output_dir / "model.pt", map_location=device))
        if loss_file.exists():
            history = json.loads(loss_file.read_text())
            if history:
                start_step = history[-1]["step"]
                logging.info("Last recorded step %d", start_step)
    eval_samples = args.eval_samples or args.n_samples
    gif_samples = args.gif_samples or eval_samples
    target_steps = args.train_steps
    if start_step >= target_steps:
        logging.info("Model already trained to step %d >= target %d", start_step, target_steps)
        return

    for step in range(start_step + 1, target_steps + 1):
        batch = train_rays.sample_rays(args.batch_size)
        z_vals, pts, deltas = sample_along_rays(
            batch["rays_o"],
            batch["rays_d"],
            args.n_samples,
            near,
            far,
            perturb=not args.no_perturb,
        )
        dirs = batch["rays_d"].unsqueeze(1).expand(-1, args.n_samples, -1)
        sigma, rgb = nerf(pts.reshape(-1, 3), dirs.reshape(-1, 3))
        sigma = sigma.view(-1, args.n_samples, 1)
        rgb = rgb.view(-1, args.n_samples, 3)
        pred = volume_render(sigma, rgb, deltas, white_bg=args.white_bg)
        loss = F.mse_loss(pred, batch["pixels"])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        record = {"step": step, "loss": float(loss.item())}
        if step % args.eval_every == 0 and val_rays is not None:
            with torch.no_grad():
                val_batch = val_rays.sample_rays(min(args.batch_size, val_rays.rays_o.shape[0]))
                z_vals_v, pts_v, deltas_v = sample_along_rays(
                    val_batch["rays_o"],
                    val_batch["rays_d"],
                    eval_samples,
                    near,
                    far,
                    perturb=False,
                )
                dirs_v = val_batch["rays_d"].unsqueeze(1).expand(-1, eval_samples, -1)
                sigma_v, rgb_v = nerf(pts_v.reshape(-1, 3), dirs_v.reshape(-1, 3))
                sigma_v = sigma_v.view(-1, eval_samples, 1)
                rgb_v = rgb_v.view(-1, eval_samples, 3)
                pred_v = volume_render(sigma_v, rgb_v, deltas_v, white_bg=args.white_bg)
                val_loss = F.mse_loss(pred_v, val_batch["pixels"])
                record["val_loss"] = float(val_loss.item())
                record["val_psnr"] = float(mse_to_psnr(val_loss).item())

        history.append(record)

        if step % args.log_every == 0:
            sigma_min, sigma_max = float(sigma.min().item()), float(sigma.max().item())
            rgb_min, rgb_max = float(rgb.min().item()), float(rgb.max().item())
            msg = f"Step {step}/{args.train_steps} loss={record['loss']:.4e} sigma[{sigma_min:.2e},{sigma_max:.2e}] rgb[{rgb_min:.3f},{rgb_max:.3f}]"
            if "val_loss" in record:
                msg += f" val={record['val_loss']:.4e} psnr={record['val_psnr']:.2f}"
            logging.info(msg)

        if step % args.render_every == 0:
            img = render_image(
                nerf,
                K,
                torch.from_numpy(dataset.c2ws_train[0]).float(),
                H,
                W,
                near,
                far,
                eval_samples,
                device,
                chunk=args.chunk_size,
                white_bg=args.white_bg,
            )
            Image.fromarray((img.clamp(0, 1).numpy() * 255).astype(np.uint8)).save(
                output_dir / f"render_{step:05d}.png"
            )

    loss_file.write_text(json.dumps(history, indent=2))
    torch.save(nerf.state_dict(), output_dir / "model.pt")
    logging.info("Saved training artifacts to %s", output_dir)

    if args.render_gif:
        start_pos = dataset.c2ws_train[0][:3, 3]
        gif_path = output_dir / "orbit.gif"
        render_orbit_gif(
            nerf,
            K,
            H,
            W,
            near,
            far,
            gif_samples,
            device,
            gif_path,
            start_pos=start_pos,
            num_frames=args.gif_frames,
            fps=args.gif_fps,
            chunk=args.chunk_size,
            white_bg=args.white_bg,
        )


# Visualization helpers


def maybe_visualize(
    dataset: RaysData,
    rays_batch: Dict[str, torch.Tensor],
    points: torch.Tensor,
    share: bool,
) -> None:
    if viser is None:
        logging.warning("Viser not installed; skipping visualization.")
        return

    H, W = dataset.height, dataset.width
    server = viser.ViserServer(share=share)
    if hasattr(server, "get_url"):
        logging.info("Viser running at %s", server.get_url())
    else:
        logging.info("Viser running (check console for URL).")

    for i, (image_np, c2w_np) in enumerate(zip(dataset.images, dataset.c2ws)):
        c2w = torch.from_numpy(c2w_np)
        server.scene.add_camera_frustum(
            f"/cameras/{i}",
            fov=2 * np.arctan2(H / 2, float(dataset.K[0, 0].detach().cpu().item())),
            aspect=W / H,
            scale=0.15,
            wxyz=viser.transforms.SO3.from_matrix(c2w[:3, :3]).wxyz,
            position=c2w[:3, 3],
            image=image_np,
        )

    rays_o = rays_batch["rays_o"].cpu().numpy()
    rays_d = rays_batch["rays_d"].cpu().numpy()
    for i, (origin, direction) in enumerate(zip(rays_o, rays_d)):
        positions = np.stack((origin, origin + direction * 6.0))
        server.scene.add_spline_catmull_rom(f"/rays/{i}", positions=positions)

    server.scene.add_point_cloud(
        "/samples",
        colors=np.zeros_like(points.cpu().numpy()).reshape(-1, 3),
        points=points.cpu().numpy().reshape(-1, 3),
        point_size=0.02,
    )

    logging.info("Press Ctrl+C to terminate the visualization server.")
    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        logging.info("Shutting down Viser.")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Part 2 data prep: rays + samples.")
    parser.add_argument("--dataset", type=Path, default=Path("lego_200x200.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("part2_runs"))
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-lego", action="store_true", help="Skip lego sampling/tests to save time.")
    parser.add_argument("--train-lego", action="store_true", help="Train NeRF on the lego dataset.")
    parser.add_argument("--lego-output-dir", type=Path, default=Path("part2_runs_lego"))
    parser.add_argument("--lego-train-steps", type=int, default=1000)
    parser.add_argument("--lego-batch-size", type=int, default=10000)
    parser.add_argument("--lego-learning-rate", type=float, default=5e-4)
    parser.add_argument("--lego-n-samples", type=int, default=64)
    parser.add_argument("--lego-eval-samples", type=int, default=None)
    parser.add_argument("--lego-hidden-dim", type=int, default=256)
    parser.add_argument("--lego-depth", type=int, default=8)
    parser.add_argument("--lego-xyz-freq", type=int, default=10)
    parser.add_argument("--lego-dir-freq", type=int, default=4)
    parser.add_argument("--lego-near", type=float, default=2.0)
    parser.add_argument("--lego-far", type=float, default=6.0)
    parser.add_argument("--lego-render-every", type=int, default=200)
    parser.add_argument("--lego-eval-every", type=int, default=200)
    parser.add_argument("--lego-log-every", type=int, default=50)
    parser.add_argument("--lego-chunk-size", type=int, default=4096)
    parser.add_argument("--lego-render-gif", action="store_true")
    parser.add_argument("--lego-gif-frames", type=int, default=60)
    parser.add_argument("--lego-gif-fps", type=int, default=15)
    parser.add_argument("--lego-white-bg", action="store_true", help="Composite lego renders over white background.")
    parser.add_argument("--n-rays", dest="n_rays", type=int, default=2048)
    parser.add_argument("--n-samples", dest="n_samples", type=int, default=64)
    parser.add_argument(
        "--eval-samples",
        type=int,
        default=None,
        help="Samples per ray for evaluation/renders (default to n-samples).",
    )
    parser.add_argument(
        "--gif-samples",
        type=int,
        default=None,
        help="Samples per ray for GIF rendering (default to eval-samples).",
    )
    parser.add_argument("--near", type=float, default=2.0)
    parser.add_argument("--far", type=float, default=6.0)
    parser.add_argument("--restrict-image", type=int, default=None)
    parser.add_argument("--no-perturb", action="store_true")
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--share", action="store_true", help="Share Viser session if visualize.")
    parser.add_argument("--xyz-freq", type=int, default=10)
    parser.add_argument("--dir-freq", type=int, default=4)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=8)
    parser.add_argument("--network-test", action="store_true", help="Run a forward pass through the NeRF for debugging.")
    parser.add_argument("--render-test", action="store_true", help="Run the volume rendering unit test.")
    parser.add_argument("--train-custom", action="store_true", help="Train NeRF on the custom dataset from Part 0.")
    parser.add_argument("--custom-dataset", type=Path, default=Path("nerf_dataset.npz"))
    parser.add_argument("--custom-max-dim", type=int, default=512)
    parser.add_argument("--custom-near", type=float, default=-1.0, help="Set <=0 to auto infer from camera positions.")
    parser.add_argument("--custom-far", type=float, default=-1.0, help="Set <=0 to auto infer from camera positions.")
    parser.add_argument("--train-steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--render-every", type=int, default=200)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--log-every", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=8192)
    parser.add_argument("--render-gif", action="store_true")
    parser.add_argument("--gif-frames", type=int, default=60)
    parser.add_argument("--gif-fps", type=int, default=10)
    parser.add_argument("--white-bg", action="store_true", help="Composite renders over a white background.")
    parser.add_argument("--resume", action="store_true", help="Resume custom training from existing checkpoints.")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    device = get_device(args.device)
    dataset_np = None

    if not args.skip_lego:
        dataset_np = load_lego_dataset(args.dataset)
        H, W = dataset_np.images_train.shape[1:3]
        logging.info(
            "Loaded dataset from %s (train=%d, val=%d)",
            args.dataset,
            len(dataset_np.images_train),
            len(dataset_np.images_val),
        )
        K = build_intrinsics(dataset_np.focal, W, H).to(device)
        rays_dataset = RaysData(dataset_np.images_train, dataset_np.c2ws_train, K, device)
        rays_batch = rays_dataset.sample_rays(args.n_rays, restrict_to=args.restrict_image)
        z_vals, points, deltas = sample_along_rays(
            rays_batch["rays_o"],
            rays_batch["rays_d"],
            args.n_samples,
            args.near,
            args.far,
            perturb=not args.no_perturb,
        )
        args.output_dir.mkdir(parents=True, exist_ok=True)
        batch_path = args.output_dir / "sample_batch.npz"
        torch.save(
            {
                "rays_o": rays_batch["rays_o"].cpu(),
                "rays_d": rays_batch["rays_d"].cpu(),
                "pixels": rays_batch["pixels"].cpu(),
                "uvs": rays_batch["uvs"].cpu(),
                "image_ids": rays_batch["image_ids"].cpu(),
                "z_vals": z_vals.cpu(),
                "points": points.cpu(),
                "deltas": deltas.cpu(),
                "config": {
                    "n_rays": args.n_rays,
                    "n_samples": args.n_samples,
                    "near": args.near,
                    "far": args.far,
                    "restrict_image": args.restrict_image,
                    "device": str(device),
                },
            },
            batch_path,
        )
        logging.info("Saved sampled batch to %s", batch_path)
        meta = {
            "dataset": str(args.dataset),
            "train_images": int(dataset_np.images_train.shape[0]),
            "resolution": [H, W],
            "focal": dataset_np.focal,
            "n_rays": args.n_rays,
            "n_samples": args.n_samples,
        }
        (args.output_dir / "summary.json").write_text(json.dumps(meta, indent=2))
    else:
        logging.info("Skipping lego sampling/tests.")

    if args.network_test:
        if dataset_np is None:
            logging.warning("Network test requested but lego dataset skipped.")
        else:
            nerf = NeuralRadianceField(
                hidden_dim=args.hidden_dim,
                depth=args.depth,
                xyz_freq=args.xyz_freq,
                dir_freq=args.dir_freq,
            ).to(device)
            points_flat = points.reshape(-1, 3)
            dirs = rays_batch["rays_d"].unsqueeze(1).expand(-1, args.n_samples, -1).reshape(-1, 3)
            sigma, rgb = nerf(points_flat, dirs)
            stats = {
                "sigma_min": float(sigma.min().item()),
                "sigma_max": float(sigma.max().item()),
                "sigma_mean": float(sigma.mean().item()),
                "rgb_min": float(rgb.min().item()),
                "rgb_max": float(rgb.max().item()),
            }
            (args.output_dir / "network_stats.json").write_text(json.dumps(stats, indent=2))
            logging.info("NeRF forward test complete. Stats: %s", stats)

    if args.render_test:
        torch.manual_seed(42)
        cpu = torch.device("cpu")
        sigmas_cpu = torch.rand((10, 64, 1), device=cpu)
        rgbs_cpu = torch.rand((10, 64, 3), device=cpu)
        deltas_test_cpu = torch.full((10, 64), (6.0 - 2.0) / 64, device=cpu)
        colors = volume_render(
            sigmas_cpu.to(device),
            rgbs_cpu.to(device),
            deltas_test_cpu.to(device),
        ).cpu()
        expected = torch.tensor(
            [
                [0.5006, 0.3728, 0.4728],
                [0.4322, 0.3559, 0.4134],
                [0.4027, 0.4394, 0.4610],
                [0.4514, 0.3829, 0.4196],
                [0.4002, 0.4599, 0.4103],
                [0.4471, 0.4044, 0.4069],
                [0.4285, 0.4072, 0.3777],
                [0.4152, 0.4190, 0.4361],
                [0.4051, 0.3651, 0.3969],
                [0.3253, 0.3587, 0.4215],
            ],
            device=cpu,
        )
        torch.testing.assert_close(colors, expected, rtol=1e-4, atol=1e-4)
        logging.info("Volume rendering test passed.")

    if args.visualize:
        if dataset_np is None:
            logging.warning("Visualization requested but lego dataset skipped.")
        else:
            maybe_visualize(rays_dataset, rays_batch, points, share=args.share)

    if args.train_lego:
        train_lego_nerf(args, device)

    if args.train_custom:
        train_custom_nerf(args, device)

if __name__ == "__main__":
    main()
