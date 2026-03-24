"""
Spy Game Reward Utilities for Flow-GRPO Bagel Training.

Provides voting grid construction and Bagel understanding-mode voting,
adapted from SPY-UMM/models/showo2_spy_wrapper.py:judge_vote().
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from typing import List


# ─── Tensor-native voting grid (avoids GPU→CPU→PIL→CPU round-trip) ───────────

def build_voting_grid_tensor(image_tensors: List[torch.Tensor],
                              cell_size: int = 256) -> torch.Tensor:
    """Build a voting grid directly from GPU tensors. No PIL needed.

    Args:
        image_tensors: List of [3,H,W] tensors on GPU, values in [0,1].
        cell_size: Target size per cell.

    Returns:
        Single [3, grid_H, grid_W] tensor on same device.
    """
    N = len(image_tensors)
    cols = min(N, 2)
    rows = (N + cols - 1) // cols
    device = image_tensors[0].device

    # Resize all images to cell_size in one batch (GPU-accelerated)
    stacked = torch.stack(image_tensors)  # [N, 3, H, W]
    resized = F.interpolate(stacked, size=(cell_size, cell_size),
                            mode='bilinear', align_corners=False)  # [N, 3, cs, cs]

    # Build grid on GPU
    grid = torch.full((3, rows * cell_size, cols * cell_size), 0.78,
                       device=device, dtype=resized.dtype)  # gray bg
    for idx in range(N):
        r, c = divmod(idx, cols)
        y, x = r * cell_size, c * cell_size
        grid[:, y:y + cell_size, x:x + cell_size] = resized[idx]

    return grid


def grid_tensor_to_pil(grid_tensor: torch.Tensor,
                       num_players: int = 0,
                       cell_size: int = 256) -> Image.Image:
    """Convert grid tensor to PIL and add player labels.

    Args:
        grid_tensor: [3, H, W] tensor from build_voting_grid_tensor.
        num_players: If > 0, draw "Player N" labels on each cell.
        cell_size: Cell size used in grid construction (for label placement).
    """
    img = (grid_tensor * 255).round().clamp(0, 255).to(torch.uint8).cpu()
    pil = Image.fromarray(img.permute(1, 2, 0).numpy())

    if num_players > 0:
        draw = ImageDraw.Draw(pil)
        cols = min(num_players, 2)
        for idx in range(num_players):
            r, c = divmod(idx, cols)
            x = c * cell_size + 5
            y = r * cell_size + 2
            draw.text((x, y), f"Player {idx + 1}", fill=(255, 0, 0))

    return pil


# ─── Legacy PIL-based grid (for logging only) ────────────────────────────────

def build_voting_grid(pil_images: List[Image.Image], cell_size: int = 256) -> Image.Image:
    """Build a labeled grid from PIL images. Used for wandb logging."""
    N = len(pil_images)
    cols = min(N, 2)
    rows = (N + cols - 1) // cols
    grid_w = cols * cell_size
    grid_h = rows * (cell_size + 20)
    grid = Image.new('RGB', (grid_w, grid_h), (200, 200, 200))
    draw = ImageDraw.Draw(grid)
    for idx, img in enumerate(pil_images):
        r, c = divmod(idx, cols)
        x = c * cell_size
        y = r * (cell_size + 20)
        draw.text((x + 5, y + 2), f"Player {idx + 1}", fill=(0, 0, 0))
        grid.paste(img.resize((cell_size, cell_size)), (x, y + 20))
    return grid


def tensor_images_to_pil(images: torch.Tensor) -> List[Image.Image]:
    """Convert batch [N,3,H,W] in [0,1] to PIL. Only for logging."""
    if images.dim() == 3:
        images = images.unsqueeze(0)
    imgs = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    imgs = imgs.transpose(0, 2, 3, 1)
    return [Image.fromarray(img) for img in imgs]


# ─── Voting ──────────────────────────────────────────────────────────────────

def run_bagel_vote(inferencer, grid_image: Image.Image, vote_prompt: str,
                   max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Run Bagel understanding mode to vote on a grid image."""
    output = inferencer.interleave_inference(
        input_lists=[grid_image, vote_prompt],
        understanding_output=True,
        do_sample=True,
        text_temperature=temperature,
        max_think_token_n=max_tokens,
    )
    if isinstance(output, list) and len(output) > 0:
        return output[0]
    return str(output)


def run_bagel_votes_cached(inferencer, grid_image: Image.Image,
                           vote_prompts: List[str],
                           max_tokens: int = 512,
                           temperature: float = 0.7) -> List[str]:
    """Run multiple votes on the same grid image, caching ViT encoding.

    Encodes the grid image once with ViT, then reuses the image KV cache
    for each vote prompt. ~N× faster than calling run_bagel_vote N times.

    Args:
        inferencer: Bagel InterleaveInferencer.
        grid_image: Grid PIL image (same for all voters).
        vote_prompts: List of vote prompts (one per voter).
        max_tokens: Max tokens for each vote response.
        temperature: Sampling temperature.

    Returns:
        List of vote text strings.
    """
    from copy import deepcopy
    from flow_grpo.bagel.data.data_utils import pil_img2rgb

    results = []

    with torch.autocast(device_type="cuda", enabled=True, dtype=torch.bfloat16):
        # Step 1: Encode the grid image ONCE into a base context
        base_context = inferencer.init_gen_context()
        transformed_img = inferencer.vae_transform.resize_transform(pil_img2rgb(grid_image))
        base_context = inferencer.update_context_image(
            transformed_img, base_context, vae=False)  # understanding mode: vit only

        # Step 2: For each voter, clone base context + encode text + generate
        for prompt in vote_prompts:
            # Clone the image-encoded context (avoids re-encoding image)
            vote_context = deepcopy(base_context)
            vote_context = inferencer.update_context_text(prompt, vote_context)
            gen_text = inferencer.gen_text(
                vote_context, do_sample=True,
                temperature=temperature, max_length=max_tokens)
            results.append(gen_text)

    return results


# ─── Advantages ──────────────────────────────────────────────────────────────

def compute_group_advantages(flat_rewards: List[float], eps: float = 1e-4) -> torch.Tensor:
    """Compute group-relative advantages from flat reward list."""
    rewards = torch.tensor(flat_rewards, dtype=torch.float32)
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)
