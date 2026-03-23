"""
Spy-Civ Game Reward for Flow-GRPO Bagel Training.

Executes a single-round "Who's the Odd One Out?" game via an external VLM
(sglang endpoint) and returns per-image rewards based on Vision-Zero's
verifiable reward formulas.

Game flow per image:
  1. Clue phase: N players sequentially describe the image (spy has modified context)
  2. Vote phase: N players vote for who they think is the spy
  3. Reward: computed from vote correctness, stealth, and suspicion metrics
"""

import asyncio
import base64
import re
import random
import threading
from io import BytesIO
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image


def _pil_to_base64(image: Image.Image) -> str:
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    encoded = base64.b64encode(buffered.getvalue()).decode("utf-8")
    return f"data:image;base64,{encoded}"


# ─── Vote extraction (adapted from Vision-Zero clevr_spotdiff_generator.py:384-475)
def _extract_vote(response: str) -> Optional[Any]:
    """Extract vote from response. Returns int, 'N/A', or None."""
    # Try boxed format
    boxed_match = re.search(r'\\\\?boxed\{(.*?)\}', response, re.DOTALL)
    if boxed_match:
        content = boxed_match.group(1).strip()
    else:
        # Try <answer> tags
        answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if answer_match:
            content = answer_match.group(1).strip()
        else:
            return None

    if content.upper() in ("N/A", "NA"):
        return "N/A"

    numbers = re.findall(r'\b([1-9])\b', content)
    if numbers:
        return int(numbers[0])
    return None


def _extract_clue(response: str) -> str:
    """Extract clue from boxed format."""
    boxed_match = re.search(r'\\\\?boxed\{(.*?)\}', response, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()
    # Fallback: use last sentence
    sentences = response.strip().split('.')
    return sentences[-1].strip() if sentences else response.strip()


# ─── Reward calculation (adapted from Vision-Zero clevr_spotdiff_generator.py:477-606)
def _calculate_game_rewards(num_players: int, spy_player: int,
                            player_votes: List[Optional[Dict[str, Any]]],
                            alpha: float = 0.6,
                            beta: float = 0.5) -> List[float]:
    """Calculate per-player rewards from voting results.

    Returns list of rewards, one per player (0-indexed).
    """
    vote_counts = {i: 0 for i in range(1, num_players + 1)}
    valid_votes = []

    for player_id, vote_info in enumerate(player_votes, 1):
        if vote_info is not None:
            voted = vote_info.get("voted_spy")
            if voted == "N/A":
                valid_votes.append("N/A")
            elif isinstance(voted, int) and 1 <= voted <= num_players:
                vote_counts[voted] += 1
                valid_votes.append(voted)
            else:
                valid_votes.append(None)
        else:
            valid_votes.append(None)

    rewards = [0.0] * num_players

    for player_id in range(1, num_players + 1):
        idx = player_id - 1

        if player_id == spy_player:
            # Stealth score
            votes_received = vote_counts[spy_player]
            stealth = 1.0 - votes_received / (num_players - 1)

            # Misleading bonus
            mislead_count = 0.0
            for v in valid_votes:
                if v is not None and v != spy_player:
                    mislead_count += 0.5 if v == "N/A" else 1.0
            misleading = beta * mislead_count / (num_players - 1)

            # Self-reveal penalty
            own_vote = player_votes[idx]
            self_reveal = 0.0
            if own_vote and own_vote.get("voted_spy") == spy_player:
                self_reveal = -1.0

            rewards[idx] = stealth + misleading + self_reveal
        else:
            # Civilian
            vote_info = player_votes[idx]
            if vote_info is None:
                rewards[idx] = -1.0
                continue

            voted = vote_info.get("voted_spy")
            case_solving = 1.0 if voted == spy_player else -1.0

            votes_received = vote_counts[player_id]
            suspicion = -alpha * votes_received / (num_players - 1)

            rewards[idx] = case_solving + suspicion

        # Clip to [-1, 1]
        rewards[idx] = max(-1.0, min(1.0, rewards[idx]))

    return rewards


# ─── Format reward check ─────────────────────────────────────────────────────
def _format_reward(response: str) -> float:
    """Score response format compliance (0-1)."""
    score = 0.0
    # Think tags or reasoning before boxed
    has_think = bool(re.search(r'<think>.*?</think>', response, re.DOTALL))
    boxed_match = re.search(r'\\\\?boxed\{.*?\}', response, re.DOTALL)
    if has_think:
        score += 0.3
    elif boxed_match:
        # Check if there's reasoning before boxed
        pre_boxed = response[:boxed_match.start()].strip()
        if len(pre_boxed) > 20:
            score += 0.2

    # Answer format
    if boxed_match:
        score += 0.4
    elif re.search(r'<answer>.*?</answer>', response, re.DOTALL):
        score += 0.3

    return score


# ─── Main reward class ───────────────────────────────────────────────────────
class SpyCivGameReward:
    """Runs a single-round spy-civ game via sglang VLM and computes rewards."""

    def __init__(self, endpoint: str = "http://127.0.0.1:17140/v1",
                 model_name: str = "Qwen2-VL-7B",
                 num_players: int = 4,
                 clue_max_tokens: int = 256,
                 vote_max_tokens: int = 512,
                 format_weight: float = 0.1):
        from openai import AsyncOpenAI
        self.client = AsyncOpenAI(base_url=endpoint, api_key="flowgrpo")
        self.model_name = model_name
        self.num_players = num_players
        self.clue_max_tokens = clue_max_tokens
        self.vote_max_tokens = vote_max_tokens
        self.format_weight = format_weight

        # EMA baselines for role-based advantage
        self.b_spy = 0.0
        self.b_civ = 0.0
        self.ema_alpha = 0.9
        self._lock = threading.Lock()

    # ── VLM calls ────────────────────────────────────────────────────────────
    async def _vlm_call(self, image_b64: str, text: str,
                        max_tokens: int) -> str:
        response = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_b64}},
                    {"type": "text", "text": text},
                ],
            }],
            max_tokens=max_tokens,
            temperature=0.7,
        )
        return response.choices[0].message.content

    # ── Prompt construction ──────────────────────────────────────────────────
    def _clue_prompt(self, player_id: int, spy_player: int,
                     num_players: int, previous_clues: str) -> str:
        """Build clue-phase prompt (adapted from clevr_spotdiff_generator.py:125-191)."""
        if player_id == spy_player:
            role_info = "Your role: SPY (you have the different picture)"
            role_instruction = ("You are the spy with the different picture. "
                                "Two objects in your picture are different from what others see, "
                                "but they remain in the same positions as in the original image.")
        else:
            role_info = "Your role: NOT SPY (you have the same picture as most others)"
            role_instruction = ("You are not the spy. Your picture is the same as "
                                "most other players' pictures.")

        prompt = f"""You are PLAYER {player_id} of the game "Who's the Odd One Out?". This is ROUND 1. Your task is to describe your picture in one sentence.

{role_info}

Game Instructions:
- There are {num_players} players in total. Only one player sees a different picture.
- Player who are voted as the odd player will be severely punished.
- In the odd player's picture, two objects are replaced with different ones, but they remain in the same positions.
- {role_instruction}

What you should do:
1. Compare your picture with all previous clues to understand what others are seeing.
2. If you are NOT the spy: Describe a clear, specific feature of your picture.
3. If you are the spy: Be strategic! Describe features that seem consistent with others.

Required output format:
First, conduct your private step-by-step reasoning.
Then, put your one-sentence description inside \\boxed{{}}.
Example: \\boxed{{The cylinder is to the left of the cube}}"""

        if previous_clues:
            prompt += f"\n\nPrevious clues:\n{previous_clues}"
        else:
            prompt += f"\n\n(You are the first to speak, no previous clues.)"

        return prompt

    def _vote_prompt(self, player_id: int, spy_player: int,
                     num_players: int, all_clues: str) -> str:
        """Build vote-phase prompt (adapted from clevr_spotdiff_generator.py:193-261)."""
        if player_id == spy_player:
            role_info = "Your role: SPY (you have the different picture)"
            role_instruction = ("You are the spy. Vote for someone else to avoid suspicion. "
                                "DON'T vote for yourself.")
        else:
            role_info = "Your role: NOT SPY (you have the same picture as most others)"
            role_instruction = ("Look for clues that seem inconsistent with your picture. "
                                "Vote for the player you believe is the spy.")

        prompt = f"""You are PLAYER {player_id} of the game "Who's the Odd One Out?". Now this is the decision-making stage. Your task is to vote on who is the Odd Player.

{role_info}

Game Instructions:
- There are {num_players} players in total. Only one player sees a different picture.
- {role_instruction}

Required output format:
<think>
your private step-by-step reasoning
</think>
<answer>
PLAYER_NUMBER or N/A
</answer>

All Clues from the Clue-giving Stage:
{all_clues}"""
        return prompt

    # ── Game execution ───────────────────────────────────────────────────────
    async def _run_game(self, image: Image.Image, metadata: Dict[str, Any]) -> float:
        """Run one game for one image, return scalar reward."""
        num_players = metadata.get('num_players', self.num_players)
        # Randomly assign spy
        rng = random.Random()
        spy_player = rng.randint(1, num_players)

        image_b64 = _pil_to_base64(image)

        # ── Clue phase (sequential) ─────────────────────────────────────
        clues = {}
        previous_clues_text = ""
        for pid in range(1, num_players + 1):
            prompt = self._clue_prompt(pid, spy_player, num_players, previous_clues_text)
            response = await self._vlm_call(image_b64, prompt, self.clue_max_tokens)
            clue = _extract_clue(response)
            clues[pid] = clue
            previous_clues_text += f"PLAYER {pid}: {clue}\n"

        # ── Vote phase (parallel) ───────────────────────────────────────
        all_clues_text = "\n".join(f"PLAYER {pid}: {clue}" for pid, clue in clues.items())

        async def _vote(pid):
            prompt = self._vote_prompt(pid, spy_player, num_players, all_clues_text)
            response = await self._vlm_call(image_b64, prompt, self.vote_max_tokens)
            vote = _extract_vote(response)
            fmt_score = _format_reward(response)
            vote_info = {"voted_spy": vote, "format_score": fmt_score} if vote is not None else None
            return vote_info

        vote_tasks = [_vote(pid) for pid in range(1, num_players + 1)]
        player_votes = await asyncio.gather(*vote_tasks)

        # ── Reward computation ──────────────────────────────────────────
        rewards = _calculate_game_rewards(num_players, spy_player, player_votes)

        # Add format bonus
        for i, vote_info in enumerate(player_votes):
            if vote_info and "format_score" in vote_info:
                rewards[i] += self.format_weight * vote_info["format_score"]
                rewards[i] = max(-1.0, min(1.0, rewards[i]))

        # Aggregate: average of all player rewards
        image_reward = sum(rewards) / len(rewards)

        # Update EMA baselines
        spy_r = rewards[spy_player - 1]
        civ_rs = [rewards[i] for i in range(num_players) if i != spy_player - 1]
        civ_avg = sum(civ_rs) / len(civ_rs) if civ_rs else 0.0

        with self._lock:
            self.b_spy = self.ema_alpha * self.b_spy + (1 - self.ema_alpha) * spy_r
            self.b_civ = self.ema_alpha * self.b_civ + (1 - self.ema_alpha) * civ_avg

        return image_reward

    # ── Batch scoring ────────────────────────────────────────────────────
    def score_batch(self, images: list, prompts: list,
                    metadatas: list) -> np.ndarray:
        """Score a batch of images. Runs games concurrently."""

        async def _run_all():
            tasks = [self._run_game(img, meta) for img, meta in zip(images, metadatas)]
            return await asyncio.gather(*tasks)

        # Use a new event loop to avoid conflicts with existing loops
        loop = asyncio.new_event_loop()
        try:
            results = loop.run_until_complete(_run_all())
        finally:
            loop.close()

        return np.array(results, dtype=np.float32)


# ─── Factory function (follows flow_grpo/rewards.py convention) ──────────────
def spy_game_score(device, config=None):
    """Create spy game reward function.

    Args:
        device: Unused (kept for interface compatibility with multi_score).
        config: Optional ConfigDict with spy_game sub-config.
    """
    kwargs = {}
    if config is not None:
        spy_cfg = getattr(config, 'spy_game', None)
        if spy_cfg is not None:
            kwargs = {
                'endpoint': getattr(spy_cfg, 'vlm_endpoint', "http://127.0.0.1:17140/v1"),
                'model_name': getattr(spy_cfg, 'vlm_model', "Qwen2-VL-7B"),
                'num_players': getattr(spy_cfg, 'num_players', 4),
                'clue_max_tokens': getattr(spy_cfg, 'clue_max_tokens', 256),
                'vote_max_tokens': getattr(spy_cfg, 'vote_max_tokens', 512),
            }

    scorer = SpyCivGameReward(**kwargs)

    def _fn(images, prompts, metadata, only_strict=False):
        # Convert tensor images to PIL
        if isinstance(images, torch.Tensor):
            images_np = (images * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
            images_np = images_np.transpose(0, 2, 3, 1)  # NCHW -> NHWC
            pil_images = [Image.fromarray(img).resize((512, 512)) for img in images_np]
        else:
            pil_images = [Image.fromarray(img).resize((512, 512)) for img in images]

        scores = scorer.score_batch(pil_images, prompts, metadata)

        return {"spy_game": scores, "avg": scores.tolist()}, {}

    return _fn
