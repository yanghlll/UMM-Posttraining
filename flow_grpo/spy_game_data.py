"""
Spy Game Data Generator for Flow-GRPO Bagel Training.

Generates CLEVR-style scene description pairs and orchestrates the
"Who's the Odd One Out?" spy-civ game. Adapted from:
  - SPY-UMM/data/game_data_generator.py (game logic)
  - SPY-UMM/data/scene_description_generator.py (scene generation)
"""

import json
import os
import random
import re
from typing import Dict, Any, List, Tuple, Optional


# ─── CLEVR-style object properties ───────────────────────────────────────────
COLORS = ['red', 'blue', 'green', 'yellow', 'purple', 'cyan', 'brown', 'gray']
SHAPES = ['cube', 'sphere', 'cylinder']
SIZES = ['small', 'large']
MATERIALS = ['metallic', 'rubber']
ABSOLUTE_POSITIONS = [
    'on the left side', 'on the right side', 'in the center',
    'in the front', 'in the back', 'in the far left corner',
    'in the far right corner', 'near the center'
]


# ─── Scene generation ────────────────────────────────────────────────────────

def _generate_object(rng, exclude=None):
    exclude = exclude or []
    for _ in range(100):
        obj = {
            'color': rng.choice(COLORS), 'shape': rng.choice(SHAPES),
            'size': rng.choice(SIZES), 'material': rng.choice(MATERIALS),
        }
        key = (obj['color'], obj['shape'], obj['size'], obj['material'])
        if key not in exclude:
            return obj
    return obj


def _generate_scene(rng, num_min=3, num_max=6):
    n = rng.randint(num_min, num_max)
    objects, used = [], []
    for i in range(n):
        obj = _generate_object(rng, exclude=used)
        obj['position'] = rng.choice(ABSOLUTE_POSITIONS)
        obj['index'] = i
        objects.append(obj)
        used.append((obj['color'], obj['shape'], obj['size'], obj['material']))
    return objects


def _modify_scene(rng, objects, num_modify=2, max_props_per_obj=1):
    """Create a subtly modified version of the scene.

    Args:
        rng: Random instance.
        objects: List of object dicts.
        num_modify: How many objects to modify.
        max_props_per_obj: Max properties to change per object (1=subtle, 4=full replace).
            1 = only change color OR shape OR size OR material (very subtle)
            2 = change up to 2 properties (moderate)
    """
    modified = [dict(o) for o in objects]
    num_modify = min(num_modify, len(objects))
    indices = rng.sample(range(len(objects)), num_modify)

    prop_pools = {
        'color': [c for c in COLORS],
        'shape': [s for s in SHAPES],
        'size': [s for s in SIZES],
        'material': [m for m in MATERIALS],
    }

    for idx in indices:
        old_obj = modified[idx]
        new_obj = dict(old_obj)

        # Pick which properties to change (1 to max_props_per_obj)
        changeable = ['color', 'shape', 'size', 'material']
        n_change = rng.randint(1, min(max_props_per_obj, len(changeable)))
        props_to_change = rng.sample(changeable, n_change)

        for prop in props_to_change:
            old_val = old_obj[prop]
            candidates = [v for v in prop_pools[prop] if v != old_val]
            if candidates:
                new_obj[prop] = rng.choice(candidates)

        modified[idx] = new_obj

    return modified, indices


def _describe_scene(objects, style='list'):
    if not objects:
        return "An empty scene."
    descs = [f"a {o['size']} {o['color']} {o['material']} {o['shape']} {o['position']}"
             for o in objects]
    if len(descs) == 1:
        return f"A scene with {descs[0]}."
    if style == 'list':
        return f"A scene containing {', '.join(descs[:-1])}, and {descs[-1]}."
    elif style == 'narrative':
        parts = [f"There is {descs[0]}"] + descs[1:-1] + [f"and {descs[-1]}"]
        return '. '.join(parts) + '.'
    else:
        lines = ["Scene description:"] + [f"- Object {i+1}: {d}" for i, d in enumerate(descs)]
        return ' '.join(lines)


def generate_scene_pair(seed, num_to_modify=2, max_props_per_obj=1):
    """Generate (original_desc, modified_desc, metadata) pair.

    Args:
        seed: Random seed.
        num_to_modify: Number of objects to modify.
        max_props_per_obj: Max properties changed per object (1=subtle, 4=full).
    """
    rng = random.Random(seed)
    objects = _generate_scene(rng)
    modified, indices = _modify_scene(rng, objects, num_modify=num_to_modify,
                                       max_props_per_obj=max_props_per_obj)
    style = rng.choice(['list', 'narrative', 'structured'])
    orig = _describe_scene(objects, style=style)
    mod = _describe_scene(modified, style=style)
    diffs = [{
        'position_index': idx,
        'original': {k: objects[idx][k] for k in ('color', 'shape', 'size', 'material')},
        'modified': {k: modified[idx][k] for k in ('color', 'shape', 'size', 'material')},
    } for idx in indices]
    return orig, mod, {'num_objects': len(objects), 'differences': diffs}


# ─── Game Data Generator (adapted from SPY-UMM/data/game_data_generator.py) ─

class SpyGameDataGenerator:
    """Generates spy game instances for Bagel text-to-image training.

    Game flow:
      1. Generate text description pair (original vs modified)
      2. Assign spy player (gets modified description)
      3. Each player generates an image from their description
      4. All players see all images and vote on who is the spy
    """

    def __init__(self, num_players=4, num_objects_min=3, num_objects_max=6,
                 num_to_modify=2, max_props_per_obj=1):
        self.num_players = num_players
        self.num_objects_min = num_objects_min
        self.num_objects_max = num_objects_max
        self.num_to_modify = num_to_modify
        self.max_props_per_obj = max_props_per_obj  # 1=subtle, 2=moderate, 4=full replace
        # EMA baselines
        self.b_spy = 0.0
        self.b_civ = 0.0
        self.ema_alpha = 0.9
        self.update_count = 0

    def generate_game(self, epoch, sample_idx):
        """Generate a complete game instance."""
        seed = epoch * 10000 + sample_idx
        orig, mod, diff_meta = generate_scene_pair(
            seed, num_to_modify=self.num_to_modify,
            max_props_per_obj=self.max_props_per_obj)
        rng = random.Random(seed + 1)
        spy_player = rng.randint(1, self.num_players)

        player_descriptions = []
        for pid in range(1, self.num_players + 1):
            player_descriptions.append(mod if pid == spy_player else orig)

        return {
            "game_id": f"spy_e{epoch}_s{sample_idx}",
            "epoch": epoch,
            "sample_idx": sample_idx,
            "num_players": self.num_players,
            "spy_player": spy_player,
            "player_descriptions": player_descriptions,
            "original_description": orig,
            "modified_description": mod,
            "diff_metadata": diff_meta,
        }

    def format_generation_prompt(self, game_data, player_id):
        """Create image generation prompt for a player.

        Each player knows their player number, total players, and role.
        Spy gets modified description with stealth instruction.
        Civilian gets original description with faithful render instruction.
        """
        desc = game_data["player_descriptions"][player_id - 1]
        spy_player = game_data["spy_player"]
        N = game_data["num_players"]

        if player_id == spy_player:
            role_hint = (
                f"You are PLAYER {player_id} of {N} in a game of \"Who's the Odd One Out?\". "
                f"Your role: SPY. Your scene description is slightly different "
                f"from the other players'. Generate an image that looks as similar "
                f"as possible to what the original scene might look like, to avoid detection."
            )
        else:
            role_hint = (
                f"You are PLAYER {player_id} of {N} in a game of \"Who's the Odd One Out?\". "
                f"Your role: CIVILIAN. Generate an image that faithfully represents "
                f"the scene description. Your image should be consistent with "
                f"other civilian players' images."
            )

        return (
            f"{role_hint}\n\n"
            f"Generate a high-quality image of the following 3D rendered scene. "
            f"The scene contains geometric objects (cubes, spheres, cylinders) "
            f"with different colors, sizes, and materials on a flat surface.\n\n"
            f"Scene: {desc}"
        )

    def format_generation_prompt_simple(self, game_data, player_id):
        """Simple generation prompt with player identity."""
        desc = game_data["player_descriptions"][player_id - 1]
        N = game_data["num_players"]
        return (
            f"You are Player {player_id} of {N}. "
            f"A 3D rendered scene with geometric objects on a flat surface: {desc}"
        )

    def format_voting_prompt(self, game_data, player_id=None):
        """Create voting prompt for identifying the spy.

        The voter sees all generated images as a labeled grid.
        """
        N = game_data["num_players"]

        if player_id is None:
            # God judge perspective
            return (
                f'You are analyzing a game of "Who\'s the Odd One Out?" where '
                f'{N} players each generated an image from a scene description. '
                f'One player (the spy) received a slightly different description, '
                f'so their generated image should look subtly different.\n\n'
                f'All {N} generated images are shown (labeled Player 1-{N}). '
                f'Compare carefully and identify which player generated the odd image.\n\n'
                f'Required output format:\n'
                f'<think>\nYour step-by-step reasoning.\n</think>\n'
                f'<answer>\nPLAYER_NUMBER (just the number)\n</answer>'
            )

        spy_player = game_data["spy_player"]
        if player_id == spy_player:
            role_info = "Your role: SPY (your description was different)"
            instruction = (
                "You know your image might look different. "
                "Try to vote for someone else to avoid suspicion."
            )
        else:
            role_info = "Your role: CIVILIAN (your description was the same as most)"
            instruction = (
                "Look for the image that seems inconsistent with the others. "
                "Vote for the player you think is the spy."
            )

        return (
            f'You are PLAYER {player_id} in a game of "Who\'s the Odd One Out?". '
            f'Each player generated an image from a scene description. '
            f'One player (the spy) had a different description.\n\n'
            f'{role_info}\n\n'
            f'All {N} generated images are shown. {instruction}\n\n'
            f'Required output format:\n'
            f'<think>\nYour step-by-step reasoning.\n</think>\n'
            f'<answer>\nPLAYER_NUMBER or N/A\n</answer>'
        )

    def extract_vote(self, response):
        """Extract vote from response text. Returns dict or None."""
        # Try <answer> tags
        m = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
        if m:
            content = m.group(1).strip()
        else:
            # Try boxed format
            m = re.search(r'\\\\?boxed\{(.*?)\}', response, re.DOTALL)
            if m:
                content = m.group(1).strip()
            else:
                return None

        if content.upper() in ("N/A", "NA"):
            return {"voted_spy": "N/A"}
        nums = re.findall(r'\b([1-9])\b', content)
        if nums:
            return {"voted_spy": int(nums[0])}
        return None

    def calculate_game_rewards(self, game_data, player_votes):
        """Calculate per-player rewards from voting. Vision-Zero formula."""
        N = game_data["num_players"]
        spy = game_data["spy_player"]
        alpha, beta = 0.6, 0.5

        vote_counts = {i: 0 for i in range(1, N + 1)}
        valid_votes = []
        for pid, vote in enumerate(player_votes, 1):
            if vote and "voted_spy" in vote:
                v = vote["voted_spy"]
                if v == "N/A":
                    valid_votes.append("N/A")
                elif isinstance(v, int) and 1 <= v <= N:
                    vote_counts[v] += 1
                    valid_votes.append(v)
                else:
                    valid_votes.append(None)
            else:
                valid_votes.append(None)

        rewards = {}
        for pid in range(1, N + 1):
            if pid == spy:
                stealth = 1.0 - vote_counts[pid] / max(N - 1, 1)
                mislead = sum(
                    1.0 if v is not None and v != spy and v != "N/A"
                    else 0.5 if v == "N/A" else 0.0
                    for v in valid_votes
                )
                rewards[pid] = max(-1.0, min(1.0, stealth + beta * mislead / max(N - 1, 1)))
            else:
                if not player_votes[pid - 1] or "voted_spy" not in player_votes[pid - 1]:
                    rewards[pid] = -1.0
                    continue
                voted = player_votes[pid - 1]["voted_spy"]
                case = 1.0 if voted == spy else -1.0
                suspicion = -alpha * vote_counts[pid] / max(N - 1, 1)
                rewards[pid] = max(-1.0, min(1.0, case + suspicion))

        spy_caught = vote_counts[spy] > N // 2
        return {
            "player_rewards": rewards,
            "spy_caught": spy_caught,
            "vote_counts": dict(vote_counts),
            "spy_player": spy,
        }

    def compute_generation_rewards(self, game_outcome):
        """Convert game outcome to per-player generation rewards for Flow-GRPO."""
        N = len(game_outcome["player_rewards"])
        spy = game_outcome["spy_player"]
        caught = game_outcome["spy_caught"]

        gen_rewards = []
        for pid in range(1, N + 1):
            if pid == spy:
                gen_rewards.append(1.0 if not caught else -1.0)
            else:
                gen_rewards.append(1.0 if caught else -0.5)
        return gen_rewards

    def update_baselines(self, spy_reward, civ_avg_reward):
        """Update EMA role baselines."""
        self.b_spy = self.ema_alpha * self.b_spy + (1 - self.ema_alpha) * spy_reward
        self.b_civ = self.ema_alpha * self.b_civ + (1 - self.ema_alpha) * civ_avg_reward
        self.update_count += 1

    def apply_role_advantage(self, rewards, spy_player):
        """Apply role-based advantage adjustment (Vision-Zero style).

        Subtracts per-role EMA baselines from rewards before group normalization.
        This prevents the spy/civilian reward imbalance from dominating the
        advantage signal. Without it, the model may learn that "being civilian
        always gives higher reward" instead of learning to generate better.

        Formula:
            adjusted_reward[spy] = reward[spy] - b_spy
            adjusted_reward[civ] = reward[civ] - b_civ

        Args:
            rewards: List of per-player rewards (0-indexed).
            spy_player: 1-indexed spy player ID.

        Returns:
            List of adjusted rewards.
        """
        adjusted = []
        for i, r in enumerate(rewards):
            pid = i + 1
            if pid == spy_player:
                adjusted.append(r - self.b_spy)
            else:
                adjusted.append(r - self.b_civ)
        return adjusted


# ─── Text-file based game data generators ────────────────────────────────────

# Prompt modification utilities for real-world prompts

_COLOR_MAP = {
    'red': 'blue', 'blue': 'red', 'green': 'yellow', 'yellow': 'green',
    'white': 'black', 'black': 'white', 'pink': 'orange', 'orange': 'pink',
    'purple': 'brown', 'brown': 'purple', 'gray': 'golden', 'golden': 'gray',
}

_SIZE_MAP = {
    'large': 'small', 'small': 'large', 'big': 'tiny', 'tiny': 'big',
    'tall': 'short', 'short': 'tall',
}

_COUNT_MAP = {
    'one': 'two', 'two': 'three', 'three': 'four', 'four': 'five',
    'five': 'six', 'six': 'seven',
}

_WORD_SWAPS = {
    'First': 'Last', 'Last': 'First', 'Start': 'Stop', 'Stop': 'Start',
    'Open': 'Close', 'Close': 'Open', 'Yes': 'No', 'No': 'Yes',
    'Hot': 'Cold', 'Cold': 'Hot', 'Up': 'Down', 'Down': 'Up',
    'Left': 'Right', 'Right': 'Left', 'New': 'Old', 'Old': 'New',
    'Day': 'Night', 'Night': 'Day', 'Spring': 'Autumn',
    'Active': 'Inactive', 'On': 'Off', 'Off': 'On',
    'Food': 'Water', 'Water': 'Food',
}

_POS_MAP = {
    'below': 'above', 'above': 'below', 'left': 'right', 'right': 'left',
    'behind': 'in front of', 'in front of': 'behind',
}


def modify_ocr_prompt(prompt, rng):
    """Modify 1-2 words inside quoted text of an OCR prompt."""
    quoted = re.findall(r'"([^"]+)"', prompt)
    if not quoted:
        return modify_natural_prompt(prompt, rng)

    target = rng.choice(quoted)
    words = target.split()
    if len(words) < 1:
        return modify_natural_prompt(prompt, rng)

    # Try semantic word swap first
    for i, w in enumerate(words):
        if w in _WORD_SWAPS:
            new_words = words.copy()
            new_words[i] = _WORD_SWAPS[w]
            new_text = ' '.join(new_words)
            modified = prompt.replace(f'"{target}"', f'"{new_text}"', 1)
            return modified, f'text: "{target}"→"{new_text}"'

    # Fallback: replace one random word
    idx = rng.randint(0, len(words) - 1)
    replacements = ['Special', 'Ultimate', 'Secret', 'Final', 'Golden', 'Ancient']
    old_word = words[idx]
    new_words = words.copy()
    new_words[idx] = rng.choice(replacements)
    new_text = ' '.join(new_words)
    modified = prompt.replace(f'"{target}"', f'"{new_text}"', 1)
    return modified, f'text word: "{old_word}"→"{new_words[idx]}"'


def modify_natural_prompt(prompt, rng):
    """Modify a natural-language prompt: swap color, size, count, or add detail."""
    # Try color swap
    colors_found = [c for c in _COLOR_MAP if re.search(r'\b' + c + r'\b', prompt, re.I)]
    if colors_found:
        old = rng.choice(colors_found)
        new = _COLOR_MAP[old]
        modified = re.sub(r'\b' + old + r'\b', new, prompt, count=1, flags=re.I)
        return modified, f"color: {old}→{new}"

    # Try size swap
    sizes_found = [s for s in _SIZE_MAP if re.search(r'\b' + s + r'\b', prompt, re.I)]
    if sizes_found:
        old = rng.choice(sizes_found)
        new = _SIZE_MAP[old]
        modified = re.sub(r'\b' + old + r'\b', new, prompt, count=1, flags=re.I)
        return modified, f"size: {old}→{new}"

    # Try count swap
    for old, new in _COUNT_MAP.items():
        if re.search(r'\b' + old + r'\b', prompt, re.I):
            modified = re.sub(r'\b' + old + r'\b', new, prompt, count=1, flags=re.I)
            return modified, f"count: {old}→{new}"

    # Fallback: append detail
    extras = ["with a small red ball", "near a wooden fence", "under a cloudy sky",
              "next to a blue car", "beside a green tree"]
    extra = rng.choice(extras)
    modified = prompt.rstrip('.') + f", {extra}."
    return modified, f"added: {extra}"


def modify_geneval_prompt(prompt, metadata, rng):
    """Modify a geneval prompt based on its tag (color/count/position/two_object)."""
    tag = metadata.get('tag', '')

    if tag in ('color_attr', 'colors'):
        colors = [c for c in _COLOR_MAP if c in prompt.lower()]
        if colors:
            old = rng.choice(colors)
            new = _COLOR_MAP[old]
            return prompt.replace(old, new, 1), f"color: {old}→{new}"

    if tag == 'counting':
        for old, new in _COUNT_MAP.items():
            if re.search(r'\b' + old + r'\b', prompt, re.I):
                modified = re.sub(r'\b' + old + r'\b', new, prompt, count=1, flags=re.I)
                return modified, f"count: {old}→{new}"

    if tag == 'position':
        for old, new in _POS_MAP.items():
            if old in prompt.lower():
                modified = prompt.lower().replace(old, new, 1)
                modified = prompt[0] + modified[1:]
                return modified, f"position: {old}→{new}"

    if tag == 'two_object':
        includes = metadata.get('include', [])
        if len(includes) >= 2:
            a, b = includes[0]['class'], includes[1]['class']
            if a in prompt and b in prompt:
                modified = prompt.replace(a, '__T__', 1).replace(b, a, 1).replace('__T__', b, 1)
                return modified, f"swap: {a}↔{b}"

    return modify_natural_prompt(prompt, rng)


class TextFileGameDataGenerator(SpyGameDataGenerator):
    """Game data generator that loads prompts from a text file.

    Each line in the file is a prompt. The spy's prompt is created by
    automatically modifying the original prompt (swap color/size/count/text).

    Supports:
      - OCR prompts (modifies quoted text)
      - Geneval prompts (modifies based on tag metadata)
      - Natural prompts (modifies colors/sizes/counts)
    """

    def __init__(self, dataset_path, split='train', prompt_type='ocr',
                 num_players=4):
        super().__init__(num_players=num_players)
        self.prompt_type = prompt_type
        self.prompts = []
        self.metadatas = []

        if prompt_type == 'ocr':
            txt_path = os.path.join(dataset_path, f'{split}.txt')
            with open(txt_path, 'r') as f:
                self.prompts = [line.strip() for line in f if len(line.strip()) > 20]
        elif prompt_type == 'geneval':
            jsonl_path = os.path.join(dataset_path, f'{split}_metadata.jsonl')
            import json
            with open(jsonl_path, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    self.prompts.append(data['prompt'])
                    self.metadatas.append(data)
        elif prompt_type == 'pickscore':
            txt_path = os.path.join(dataset_path, f'{split}.txt')
            with open(txt_path, 'r') as f:
                self.prompts = [line.strip() for line in f if len(line.strip()) > 20]
        else:
            raise ValueError(f"Unknown prompt_type: {prompt_type}")

    def generate_game(self, epoch, sample_idx):
        """Generate a game from the loaded prompt file."""
        rng = random.Random(epoch * 10000 + sample_idx)

        # Pick a prompt
        idx = rng.randint(0, len(self.prompts) - 1)
        orig = self.prompts[idx]

        # Create modified version
        if self.prompt_type == 'ocr':
            mod, change_desc = modify_ocr_prompt(orig, rng)
        elif self.prompt_type == 'geneval':
            meta = self.metadatas[idx] if idx < len(self.metadatas) else {}
            mod, change_desc = modify_geneval_prompt(orig, meta, rng)
        else:
            mod, change_desc = modify_natural_prompt(orig, rng)

        # Assign spy
        spy_player = rng.randint(1, self.num_players)

        player_descriptions = []
        for pid in range(1, self.num_players + 1):
            player_descriptions.append(mod if pid == spy_player else orig)

        return {
            "game_id": f"spy_e{epoch}_s{sample_idx}",
            "epoch": epoch,
            "sample_idx": sample_idx,
            "num_players": self.num_players,
            "spy_player": spy_player,
            "player_descriptions": player_descriptions,
            "original_description": orig,
            "modified_description": mod,
            "diff_metadata": {"change": change_desc, "prompt_type": self.prompt_type},
        }

    def format_generation_prompt(self, game_data, player_id):
        """Generation prompt for text-file based games (no CLEVR-specific wording)."""
        desc = game_data["player_descriptions"][player_id - 1]
        spy_player = game_data["spy_player"]
        N = game_data["num_players"]

        if player_id == spy_player:
            role_hint = (
                f"You are PLAYER {player_id} of {N} in a game of \"Who's the Odd One Out?\". "
                f"Your role: SPY. Your prompt is slightly different from the other players'. "
                f"Generate an image that looks as similar as possible to what the others "
                f"might generate, to avoid detection."
            )
        else:
            role_hint = (
                f"You are PLAYER {player_id} of {N} in a game of \"Who's the Odd One Out?\". "
                f"Your role: CIVILIAN. Generate an image that faithfully represents "
                f"the description. Your image should be consistent with other players'."
            )

        return f"{role_hint}\n\n{desc}"

    def format_generation_prompt_simple(self, game_data, player_id):
        """Simple prompt with player identity."""
        desc = game_data["player_descriptions"][player_id - 1]
        N = game_data["num_players"]
        return f"You are Player {player_id} of {N}. {desc}"
