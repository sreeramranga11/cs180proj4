from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
from PIL import Image
import torch
from torch import nn


# ---------------------------------------------------------------------------
# Utilities and configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    image_path: Path
    width: int
    max_freq: int
    iterations: int
    batch_size: int
    lr: float
    eval_every: int
    hidden_layers: int
    seed: int
    max_dim: int | None
    chunk_size: int


def setup_logger(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="[%(asctime)s][%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )


def get_device(preferred: str | None = None) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Positional encoding + model definitions
# ---------------------------------------------------------------------------


class PositionalEncoder:
    """Sinusoidal positional encoding that retains the original input (per NeRF)."""

    def __init__(self, max_freq: int):
        self.max_freq = max_freq

    @property
    def out_dim(self) -> int:
        # For 2D coords: base 2 dims + (sin+cos) * 2 dims per frequency.
        return 2 + 4 * self.max_freq

    def __call__(self, coords: torch.Tensor) -> torch.Tensor:
        encodings = [coords]
        for i in range(self.max_freq):
            freq = (2.0 ** i) * math.pi
            encodings.append(torch.sin(freq * coords))
            encodings.append(torch.cos(freq * coords))
        return torch.cat(encodings, dim=-1)


class PixelMLP(nn.Module):
    """Simple ReLU MLP with Sigmoid output."""

    def __init__(self, in_dim: int, hidden_dim: int, hidden_layers: int):
        super().__init__()
        layers: List[nn.Module] = []
        last_dim = in_dim
        for _ in range(hidden_layers):
            layers.append(nn.Linear(last_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            last_dim = hidden_dim
        layers.append(nn.Linear(last_dim, 3))
        self.net = nn.Sequential(*layers)
        self.output_act = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_act(self.net(x))


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------


RESAMPLE_LANCZOS = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


def load_image(path: Path, max_dim: int | None = None) -> Tuple[np.ndarray, Tuple[int, int], Dict[str, int]]:
    """Load an image, optionally resize, return numpy array (H, W, 3) in [0, 1]."""
    with Image.open(path) as img:
        img = img.convert("RGB")
        original_size = img.size  # (W, H)
        if max_dim is not None and max(img.size) > max_dim:
            scale = max_dim / max(img.size)
            new_size = (int(round(img.size[0] * scale)), int(round(img.size[1] * scale)))
            img = img.resize(new_size, RESAMPLE_LANCZOS)
            logging.info(
                "Resized %s from %s to %s to respect max_dim=%d",
                path.name,
                original_size,
                img.size,
                max_dim,
            )
        np_img = np.asarray(img).astype(np.float32) / 255.0
    height, width, _ = np_img.shape
    meta = {"original_width": original_size[0], "original_height": original_size[1], "width": width, "height": height}
    return np_img, (height, width), meta


def prepare_pixels(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return flattened normalized coordinates (N,2) and colors (N,3)."""
    height, width, _ = image.shape
    ys, xs = np.meshgrid(np.arange(height, dtype=np.float32), np.arange(width, dtype=np.float32), indexing="ij")
    xs = xs / max(width - 1, 1)
    ys = ys / max(height - 1, 1)
    coords = np.stack((xs, ys), axis=-1).reshape(-1, 2)
    colors = image.reshape(-1, 3)
    return coords, colors


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------


def mse_to_psnr(mse: torch.Tensor) -> torch.Tensor:
    return -10.0 * torch.log10(torch.clamp(mse, min=1e-10))


def render_model(
    model: nn.Module,
    encoder: PositionalEncoder,
    coords: torch.Tensor,
    height: int,
    width: int,
    device: torch.device,
    chunk_size: int,
) -> torch.Tensor:
    """Render the full image by chunking through the coordinate list."""
    model.eval()
    outputs: List[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, coords.shape[0], chunk_size):
            end = start + chunk_size
            batch = coords[start:end].to(device)
            encoded = encoder(batch)
            preds = model(encoded).detach().cpu()
            outputs.append(preds)
    recon = torch.cat(outputs, dim=0)
    return recon.view(height, width, 3)


def save_image(array: torch.Tensor, path: Path) -> None:
    arr = torch.clamp(array, 0.0, 1.0).mul(255).byte().cpu().numpy()
    Image.fromarray(arr).save(path)


def make_result_grids(
    images: Iterable[Path],
    widths: List[int],
    freqs: List[int],
    results: Dict[Tuple[Path, int, int], Dict],
    output_root: Path,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        logging.warning("matplotlib unavailable (%s). Skipping grid generation.", exc)
        return

    widths_sorted = sorted(set(widths))
    freqs_sorted = sorted(set(freqs))

    for image_path in images:
        fig, axes = plt.subplots(
            len(freqs_sorted),
            len(widths_sorted),
            figsize=(4 * len(widths_sorted), 3 * len(freqs_sorted)),
        )
        axes = np.array(axes, dtype=object)
        if axes.ndim == 0:
            axes = axes[np.newaxis, np.newaxis]
        elif axes.ndim == 1:
            if len(freqs_sorted) == 1:
                axes = axes[np.newaxis, :]
            else:
                axes = axes[:, np.newaxis]

        any_found = False
        for r, freq in enumerate(freqs_sorted):
            for c, width in enumerate(widths_sorted):
                ax = axes[r][c]
                key = (image_path.resolve(), width, freq)
                entry = results.get(key)
                if entry and entry["final_image"].exists():
                    img = Image.open(entry["final_image"])
                    ax.imshow(img)
                    any_found = True
                else:
                    ax.text(0.5, 0.5, "missing", ha="center", va="center", fontsize=12)
                    ax.set_facecolor("#f8f8f8")
                ax.set_title(f"L={freq}, width={width}")
                ax.axis("off")

        if any_found:
            fig.suptitle(f"{image_path.stem} positional encoding sweep", fontsize=14)
            fig.tight_layout()
            grid_path = output_root / f"{image_path.stem}_grid.png"
            fig.savefig(grid_path, dpi=200)
            logging.info("Saved comparison grid to %s", grid_path)
        plt.close(fig)

def train_single(config: TrainingConfig, device: torch.device, output_root: Path, log_metrics: bool = True) -> Dict:
    image_array, (height, width), meta = load_image(config.image_path, config.max_dim)
    coords_np, colors_np = prepare_pixels(image_array)
    total_pixels = coords_np.shape[0]

    coords = torch.from_numpy(coords_np)
    colors = torch.from_numpy(colors_np)

    encoder = PositionalEncoder(config.max_freq)
    model = PixelMLP(encoder.out_dim, config.width, config.hidden_layers).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    criterion = nn.MSELoss()

    experiment_name = (
        f"{config.image_path.stem}_w{config.width}_L{config.max_freq}_iter{config.iterations}_bs{config.batch_size}"
    )
    out_dir = output_root / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)

    config_dict = asdict(config)
    config_dict["image_path"] = str(config.image_path)
    (out_dir / "config.json").write_text(
        json.dumps({**config_dict, "device": str(device), **meta, "total_pixels": total_pixels}, indent=2)
    )

    logging.info(
        "Starting training for %s (pixels=%d, width=%d, L=%d) on %s",
        config.image_path.name,
        total_pixels,
        config.width,
        config.max_freq,
        device,
    )

    rng = torch.Generator(device="cpu").manual_seed(config.seed)
    metrics: List[Dict[str, float]] = []
    start_time = time.time()

    for step in range(1, config.iterations + 1):
        batch_indices = torch.randint(
            0,
            total_pixels,
            (config.batch_size,),
            generator=rng,
        )
        batch_coords = coords[batch_indices].to(device)
        batch_colors = colors[batch_indices].to(device)

        encoded = encoder(batch_coords)
        preds = model(encoded)
        loss = criterion(preds, batch_colors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if log_metrics and (step % config.eval_every == 0 or step == 1 or step == config.iterations):
            recon = render_model(model, encoder, coords, height, width, device, config.chunk_size)
            mse = torch.mean((recon.view(-1, 3) - colors) ** 2)
            psnr = mse_to_psnr(mse)
            elapsed = time.time() - start_time
            save_path = out_dir / f"iter_{step:05d}.png"
            save_image(recon, save_path)
            record = {
                "step": step,
                "loss": float(loss.item()),
                "eval_mse": float(mse.item()),
                "psnr": float(psnr.item()),
                "elapsed_sec": elapsed,
                "image_path": config.image_path.name,
            }
            metrics.append(record)
            logging.info(
                "[%s] step=%d loss=%.4e mse=%.4e psnr=%.2f dB saved=%s",
                experiment_name,
                step,
                loss.item(),
                mse.item(),
                psnr.item(),
                save_path.name,
            )

    metrics_path = out_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2))
    final_image = out_dir / f"iter_{config.iterations:05d}.png"
    logging.info("Finished %s; metrics stored at %s", experiment_name, metrics_path)
    return {
        "out_dir": out_dir,
        "metrics": metrics_path,
        "final_image": final_image,
        "config": config,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train NeRF-style neural fields on 2D images.")
    parser.add_argument(
        "--images",
        nargs="+",
        type=Path,
        default=[Path("part1_files/fox.jpg"), Path("part1_files/goat.jpg")],
        help="List of image paths to train on.",
    )
    parser.add_argument(
        "--widths",
        nargs="+",
        type=int,
        default=[32, 256],
        help="Hidden layer widths to sweep.",
    )
    parser.add_argument(
        "--freqs",
        nargs="+",
        type=int,
        default=[2, 10],
        help="Positional encoding max frequency levels L to sweep.",
    )
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--hidden-layers", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--max-dim", type=int, default=512, help="Resize longer image side to this (<=0 disables).")
    parser.add_argument("--chunk-size", type=int, default=131072, help="Pixels per chunk when rendering full image.")
    parser.add_argument("--output-dir", type=Path, default=Path("part1_runs"))
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--log-level", type=str, default="INFO")
    parser.add_argument("--device", type=str, default=None, help="Optional device override (cpu/cuda/mps).")
    parser.add_argument(
        "--no-grid",
        action="store_true",
        help="Disable generation of the width/frequency result grid per image.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logger(args.log_level)

    if args.max_dim is not None and args.max_dim <= 0:
        args.max_dim = None

    device = get_device(args.device)
    logging.info("Using device: %s", device)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)

    configs: List[TrainingConfig] = []
    for image_path in args.images:
        if not image_path.exists():
            logging.error("Image %s not found; skipping.", image_path)
            continue
        for width in args.widths:
            for freq in args.freqs:
                configs.append(
                    TrainingConfig(
                        image_path=image_path,
                        width=width,
                        max_freq=freq,
                        iterations=args.iterations,
                        batch_size=args.batch_size,
                        lr=args.lr,
                        eval_every=args.eval_every,
                        hidden_layers=args.hidden_layers,
                        seed=args.seed,
                        max_dim=args.max_dim,
                        chunk_size=args.chunk_size,
                    )
                )

    if not configs:
        logging.error("No valid configurations to run. Check image paths.")
        return

    results: Dict[Tuple[Path, int, int], Dict] = {}
    for cfg in configs:
        result = train_single(cfg, device, args.output_dir, log_metrics=True)
        results[(cfg.image_path.resolve(), cfg.width, cfg.max_freq)] = result

    if not args.no_grid:
        unique_images = sorted({cfg.image_path.resolve() for cfg in configs})
        make_result_grids(unique_images, args.widths, args.freqs, results, args.output_dir)


if __name__ == "__main__":
    main()
