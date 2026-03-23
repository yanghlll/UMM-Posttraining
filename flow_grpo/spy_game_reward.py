"""
Spy Game Reward Utilities for Flow-GRPO Bagel Training.

Provides voting grid construction and Bagel understanding-mode voting,
adapated from SPY-UMM/models/showo2_spy_wrapper.py:judge_vote().
"""

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from typing import List


def build_voting_grid(pil_images: List[Image.Image], cell_size: int = 256) -> Image.Image:
    """Build a labeled grid of player images for voting.

    Creates a 2×2 (or appropriate) grid with player labels,
    similar to showo2_spy_wrapper.py:486-515.

    Args:
        pil_images: List of PIL images, one per player.
        cell_size: Size of each cell in the grid.

    Returns:
        Single PIL image containing all players' images in a labeled grid.
    """
    N = len(pil_images)
    cols = min(N, 2)
    rows = (N + cols - 1) // cols
    grid_w = cols * cell_size
    grid_h = rows * (cell_size + 20)  # 20px for label

    grid = Image.new('RGB', (grid_w, grid_h), (200, 200, 200))
    draw = ImageDraw.Draw(grid)

    for idx, img in enumerate(pil_images):
        r, c = divmod(idx, cols)
        x = c * cell_size
        y = r * (cell_size + 20)

        # Draw label
        label = f"Player {idx + 1}"
        draw.text((x + 5, y + 2), label, fill=(0, 0, 0))

        # Paste resized image
        resized = img.resize((cell_size, cell_size))
        grid.paste(resized, (x, y + 20))

    return grid


def tensor_images_to_pil(images: torch.Tensor) -> List[Image.Image]:
    """Convert batch of tensor images [N,3,H,W] in [0,1] to PIL images."""
    if images.dim() == 3:
        images = images.unsqueeze(0)
    imgs = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
    imgs = imgs.transpose(0, 2, 3, 1)  # NCHW -> NHWC
    return [Image.fromarray(img) for img in imgs]


def run_bagel_vote(inferencer, grid_image: Image.Image, vote_prompt: str,
                   max_tokens: int = 512, temperature: float = 0.7) -> str:
    """Run Bagel in understanding mode to vote on a grid image.

    Args:
        inferencer: Bagel InterleaveInferencer instance.
        grid_image: Grid image showing all players' generated images.
        vote_prompt: Voting prompt text.
        max_tokens: Max tokens for vote response.
        temperature: Sampling temperature.

    Returns:
        Generated vote text.
    """
    output = inferencer.interleave_inference(
        input_lists=[grid_image, vote_prompt],
        understanding_output=True,
        do_sample=True,
        text_temperature=temperature,
        max_think_token_n=max_tokens,
    )
    # interleave_inference returns a list; take the text output
    if isinstance(output, list) and len(output) > 0:
        return output[0]
    return str(output)


def compute_group_advantages(flat_rewards: List[float], eps: float = 1e-4) -> torch.Tensor:
    """Compute group-relative advantages from flat reward list."""
    rewards = torch.tensor(flat_rewards, dtype=torch.float32)
    mean = rewards.mean()
    std = rewards.std()
    return (rewards - mean) / (std + eps)
