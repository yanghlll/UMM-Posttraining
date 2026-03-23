"""
Spy Game Dataset for Flow-GRPO Bagel Training.

Generates CLEVR-style scene description pairs (original/modified) for the
"Who's the Odd One Out?" spy-civ game. Each sample provides a prompt for
Bagel image generation and metadata for the spy game reward.
"""

import json
import os
import random
from typing import Dict, Any, List, Tuple

from torch.utils.data import Dataset


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


# ─── Scene generation (adapted from SPY-UMM/data/scene_description_generator.py)
def _generate_object(rng: random.Random,
                     exclude: List[Tuple[str, str, str, str]] = None) -> Dict[str, str]:
    exclude = exclude or []
    for _ in range(100):
        obj = {
            'color': rng.choice(COLORS),
            'shape': rng.choice(SHAPES),
            'size': rng.choice(SIZES),
            'material': rng.choice(MATERIALS),
        }
        key = (obj['color'], obj['shape'], obj['size'], obj['material'])
        if key not in exclude:
            return obj
    return obj


def _generate_scene(rng: random.Random, num_min: int = 3,
                    num_max: int = 6) -> List[Dict[str, str]]:
    num_objects = rng.randint(num_min, num_max)
    objects = []
    used_keys = []
    for i in range(num_objects):
        obj = _generate_object(rng, exclude=used_keys)
        obj['position'] = rng.choice(ABSOLUTE_POSITIONS)
        obj['index'] = i
        objects.append(obj)
        used_keys.append((obj['color'], obj['shape'], obj['size'], obj['material']))
    return objects


def _modify_scene(rng: random.Random, objects: List[Dict[str, str]],
                  num_modify: int = 2) -> Tuple[List[Dict[str, str]], List[int]]:
    modified = [dict(obj) for obj in objects]
    num_modify = min(num_modify, len(objects))
    modify_indices = rng.sample(range(len(objects)), num_modify)
    existing_keys = [
        (o['color'], o['shape'], o['size'], o['material']) for o in objects
    ]
    for idx in modify_indices:
        old_obj = modified[idx]
        new_obj = _generate_object(rng, exclude=existing_keys)
        new_obj['position'] = old_obj['position']
        new_obj['index'] = old_obj['index']
        modified[idx] = new_obj
        existing_keys[idx] = (new_obj['color'], new_obj['shape'],
                              new_obj['size'], new_obj['material'])
    return modified, modify_indices


def _describe_scene(objects: List[Dict[str, str]], style: str = 'list') -> str:
    if not objects:
        return "An empty scene."

    descriptions = []
    for obj in objects:
        desc = f"a {obj['size']} {obj['color']} {obj['material']} {obj['shape']}"
        descriptions.append(f"{desc} {obj['position']}")

    if len(descriptions) == 1:
        return f"A scene with {descriptions[0]}."

    if style == 'list':
        items = ', '.join(descriptions[:-1]) + f', and {descriptions[-1]}'
        return f"A scene containing {items}."
    elif style == 'narrative':
        parts = []
        for i, desc in enumerate(descriptions):
            if i == 0:
                parts.append(f"There is {desc}")
            elif i == len(descriptions) - 1:
                parts.append(f"and {desc}")
            else:
                parts.append(desc)
        return '. '.join(parts) + '.'
    else:  # structured
        lines = ["Scene description:"]
        for i, desc in enumerate(descriptions):
            lines.append(f"- Object {i+1}: {desc}")
        return ' '.join(lines)


def generate_scene_pair(seed: int) -> Tuple[str, str, Dict[str, Any]]:
    """Generate (original_desc, modified_desc, metadata) pair."""
    rng = random.Random(seed)
    objects = _generate_scene(rng)
    modified_objects, modify_indices = _modify_scene(rng, objects)
    style = rng.choice(['list', 'narrative', 'structured'])
    original_desc = _describe_scene(objects, style=style)
    modified_desc = _describe_scene(modified_objects, style=style)

    differences = []
    for idx in modify_indices:
        differences.append({
            'position_index': idx,
            'original': {k: objects[idx][k] for k in ('color', 'shape', 'size', 'material')},
            'modified': {k: modified_objects[idx][k] for k in ('color', 'shape', 'size', 'material')},
        })

    metadata = {
        'num_objects': len(objects),
        'num_modified': len(modify_indices),
        'modify_indices': modify_indices,
        'differences': differences,
    }
    return original_desc, modified_desc, metadata


# ─── Dataset classes ─────────────────────────────────────────────────────────

class SpyGamePromptDataset(Dataset):
    """Dataset that yields scene descriptions for spy-civ game training.

    Supports two modes:
    - 'jsonl': Load pre-generated pairs from a JSONL file.
    - 'procedural': Generate pairs on-the-fly from seeds.

    Each item returns a prompt (the scene description Bagel will generate an
    image for) and metadata needed by the spy game reward function.
    """

    def __init__(self, dataset_path: str, split: str = 'train',
                 num_players: int = 4, num_procedural: int = 10000):
        self.num_players = num_players
        self.items: List[Dict[str, Any]] = []

        jsonl_path = os.path.join(dataset_path, f'{split}.jsonl')
        if os.path.exists(jsonl_path):
            self._load_jsonl(jsonl_path)
        else:
            self._generate_procedural(num_procedural, split)

    def _load_jsonl(self, path: str):
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line.strip())
                self.items.append({
                    'original_desc': data['original_desc'],
                    'modified_desc': data['modified_desc'],
                    'diff_metadata': data.get('diff_metadata', {}),
                })

    def _generate_procedural(self, n: int, split: str):
        base_seed = 0 if split == 'train' else 1_000_000
        for i in range(n):
            orig, mod, meta = generate_scene_pair(seed=base_seed + i)
            self.items.append({
                'original_desc': orig,
                'modified_desc': mod,
                'diff_metadata': meta,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        # The spy player gets modified_desc as prompt; civilians get original.
        # For GRPO, we use the *original* description as the prompt so that
        # K images in a group are from the same prompt (standard GRPO).
        # The modified description is passed in metadata for the game reward.
        return {
            "prompt": item['original_desc'],
            "metadata": {
                "original_desc": item['original_desc'],
                "modified_desc": item['modified_desc'],
                "diff_metadata": item.get('diff_metadata', {}),
                "num_players": self.num_players,
            }
        }

    @staticmethod
    def collate_fn(examples):
        prompts = [ex["prompt"] for ex in examples]
        metadatas = [ex["metadata"] for ex in examples]
        return prompts, metadatas
