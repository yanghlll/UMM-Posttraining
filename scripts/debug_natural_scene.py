"""
Test spy game inference with real-world prompts from:
  1. Vision-Zero / flow_grpo pickscore dataset (natural scene captions)
  2. flow_grpo geneval dataset (structured object descriptions)

For each prompt source, run 2 games × 4 players.
The "spy" gets a slightly modified prompt (object swap / attribute change).
Saves all images + prompts to debug_natural_scene/.
"""
import os
import json
import random
import torch
from PIL import Image
from accelerate import load_checkpoint_and_dispatch, init_empty_weights
from flow_grpo.bagel.data.data_utils import add_special_tokens
from flow_grpo.bagel.data.transforms import ImageTransform
from flow_grpo.bagel.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from flow_grpo.bagel.modeling.qwen2 import Qwen2Tokenizer
from flow_grpo.bagel.modeling.autoencoder import load_ae
from flow_grpo.bagel.inferencer import InterleaveInferencer
from huggingface_hub import snapshot_download
import ml_collections
import re

OUTPUT_DIR = "debug_natural_scene"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda:0"
dtype = torch.bfloat16
NUM_PLAYERS = 4

# ─── Prompt modification for natural scenes ──────────────────────────────────

def modify_natural_prompt(prompt, seed=42):
    """Create a subtle modification of a natural scene prompt.

    Strategies:
      - Swap a color word (red→blue, green→yellow, etc.)
      - Swap a size word (large→small, big→tiny)
      - Swap a count word (two→three, one→two)
      - Add/remove an object
    """
    rng = random.Random(seed)
    modified = prompt

    # Try color swap first
    color_map = {
        'red': 'blue', 'blue': 'red', 'green': 'yellow', 'yellow': 'green',
        'white': 'black', 'black': 'white', 'pink': 'orange', 'orange': 'pink',
        'purple': 'brown', 'brown': 'purple', 'gray': 'golden', 'golden': 'gray',
    }
    colors_in_prompt = [c for c in color_map if re.search(r'\b' + c + r'\b', prompt, re.I)]
    if colors_in_prompt:
        old_color = rng.choice(colors_in_prompt)
        new_color = color_map[old_color]
        modified = re.sub(r'\b' + old_color + r'\b', new_color, prompt, count=1, flags=re.I)
        return modified, f"color: {old_color}→{new_color}"

    # Try size swap
    size_map = {'large': 'small', 'small': 'large', 'big': 'tiny', 'tiny': 'big',
                'tall': 'short', 'short': 'tall'}
    sizes_in_prompt = [s for s in size_map if re.search(r'\b' + s + r'\b', prompt, re.I)]
    if sizes_in_prompt:
        old_size = rng.choice(sizes_in_prompt)
        new_size = size_map[old_size]
        modified = re.sub(r'\b' + old_size + r'\b', new_size, prompt, count=1, flags=re.I)
        return modified, f"size: {old_size}→{new_size}"

    # Try count swap
    count_map = {'one': 'two', 'two': 'three', 'three': 'four', 'four': 'five',
                 'a ': 'two ', 'an ': 'two '}
    for old, new in count_map.items():
        if old in prompt.lower():
            modified = prompt.replace(old, new, 1)
            if modified != prompt:
                return modified, f"count: {old.strip()}→{new.strip()}"

    # Fallback: append a small detail
    extras = ["with a small red ball", "near a wooden fence", "under a cloudy sky",
              "next to a blue car", "beside a green tree"]
    extra = rng.choice(extras)
    modified = prompt.rstrip('.') + f", {extra}."
    return modified, f"added: {extra}"


def modify_geneval_prompt(prompt, metadata, seed=42):
    """Create a subtle modification of a geneval prompt using its metadata."""
    rng = random.Random(seed)

    color_map = {
        'red': 'blue', 'blue': 'red', 'green': 'yellow', 'yellow': 'green',
        'white': 'black', 'black': 'white', 'pink': 'orange', 'orange': 'pink',
        'purple': 'brown', 'brown': 'purple',
    }

    tag = metadata.get('tag', '')

    # Color attribute: swap one color
    if tag in ('color_attr', 'colors'):
        colors_in_prompt = [c for c in color_map if c in prompt.lower()]
        if colors_in_prompt:
            old = rng.choice(colors_in_prompt)
            new = color_map[old]
            modified = prompt.replace(old, new, 1)
            return modified, f"color: {old}→{new}"

    # Counting: change the count by ±1
    if tag == 'counting':
        count_words = {'one': 'two', 'two': 'three', 'three': 'four', 'four': 'five',
                       'five': 'six', 'six': 'seven'}
        for old, new in count_words.items():
            if old in prompt.lower():
                modified = re.sub(r'\b' + old + r'\b', new, prompt, count=1, flags=re.I)
                return modified, f"count: {old}→{new}"

    # Position: swap the spatial relation
    if tag == 'position':
        pos_map = {'below': 'above', 'above': 'below', 'left': 'right', 'right': 'left',
                   'behind': 'in front of', 'in front of': 'behind'}
        for old, new in pos_map.items():
            if old in prompt.lower():
                modified = prompt.lower().replace(old, new, 1)
                # Restore original casing of first char
                modified = prompt[0] + modified[1:]
                return modified, f"position: {old}→{new}"

    # two_object: swap the two objects
    if tag == 'two_object':
        includes = metadata.get('include', [])
        if len(includes) >= 2:
            obj_a = includes[0]['class']
            obj_b = includes[1]['class']
            if obj_a in prompt and obj_b in prompt:
                modified = prompt.replace(obj_a, '__TMP__', 1).replace(obj_b, obj_a, 1).replace('__TMP__', obj_b, 1)
                return modified, f"swap objects: {obj_a}↔{obj_b}"

    # Fallback: use natural prompt modifier
    return modify_natural_prompt(prompt, seed)


def modify_ocr_prompt(prompt, seed=42):
    """Create a subtle modification of an OCR/text-rendering prompt.

    Strategies: change 1-2 words in the quoted text, swap a number, or change a color.
    """
    rng = random.Random(seed)

    # Find all quoted strings in the prompt
    quoted = re.findall(r'"([^"]+)"', prompt)

    if quoted:
        target = rng.choice(quoted)
        words = target.split()

        if len(words) >= 2:
            # Strategy 1: swap one word in the quoted text
            word_swaps = {
                'First': 'Last', 'Last': 'First', 'Start': 'Stop', 'Stop': 'Start',
                'Open': 'Close', 'Close': 'Open', 'Yes': 'No', 'No': 'Yes',
                'Hot': 'Cold', 'Cold': 'Hot', 'Up': 'Down', 'Down': 'Up',
                'Left': 'Right', 'Right': 'Left', 'New': 'Old', 'Old': 'New',
                'Day': 'Night', 'Night': 'Day', 'Love': 'Hope', 'Spring': 'Autumn',
                'Active': 'Inactive', 'On': 'Off', 'Off': 'On',
                'Food': 'Water', 'Water': 'Food',
            }
            for i, w in enumerate(words):
                if w in word_swaps:
                    new_words = words.copy()
                    new_words[i] = word_swaps[w]
                    new_text = ' '.join(new_words)
                    modified = prompt.replace(f'"{target}"', f'"{new_text}"', 1)
                    return modified, f'text: "{target}"→"{new_text}"'

            # Strategy 2: change one word to a similar word
            idx = rng.randint(0, len(words) - 1)
            replacements = ['Amazing', 'Special', 'Ultimate', 'Secret', 'Final', 'Golden']
            new_word = rng.choice(replacements)
            new_words = words.copy()
            old_word = new_words[idx]
            new_words[idx] = new_word
            new_text = ' '.join(new_words)
            modified = prompt.replace(f'"{target}"', f'"{new_text}"', 1)
            return modified, f'text word: "{old_word}"→"{new_word}"'

    # No quoted text: try color/number swap
    return modify_natural_prompt(prompt, seed)


# ─── Model loading (same as debug_spy_inference.py) ──────────────────────────

model_path = "ByteDance-Seed/BAGEL-7B-MoT"
model_local_dir = snapshot_download(repo_id=model_path) if not os.path.exists(model_path) else model_path

llm_config = Qwen2Config.from_json_file(os.path.join(model_local_dir, "llm_config.json"))
llm_config.qk_norm = True
llm_config.tie_word_embeddings = False
llm_config.layer_module = "Qwen2MoTDecoderLayer"

vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_local_dir, "vit_config.json"))
vit_config.rope = False
vit_config.num_hidden_layers = vit_config.num_hidden_layers - 1

vae_model, vae_config = load_ae(local_path=os.path.join(model_local_dir, "ae.safetensors"))

bagel_config = BagelConfig(
    visual_gen=True, visual_und=True,
    llm_config=llm_config, vit_config=vit_config, vae_config=vae_config,
    vit_max_num_patch_per_side=70, connector_act='gelu_pytorch_tanh',
    latent_patch_size=2, max_latent_size=64,
)

with init_empty_weights():
    language_model = Qwen2ForCausalLM(llm_config)
    vit_model = SiglipVisionModel(vit_config)
    model = Bagel(language_model, vit_model, bagel_config)
    model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

tokenizer = Qwen2Tokenizer.from_pretrained(model_local_dir)
tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)
vae_transform = ImageTransform(512, 256, 8)
vit_transform = ImageTransform(490, 112, 7)

print("Loading checkpoint...")
model = load_checkpoint_and_dispatch(
    model, checkpoint=os.path.join(model_local_dir, "ema.safetensors"),
    device_map={"": device}, offload_buffers=False, dtype=dtype,
    force_hooks=True, offload_folder="/tmp/offload"
)
model.eval()
vae_model.to(device, dtype=dtype)
model.to(device, dtype=dtype)

inferencer = InterleaveInferencer(
    model=model, vae_model=vae_model, tokenizer=tokenizer,
    vae_transform=vae_transform, vit_transform=vit_transform,
    new_token_ids=new_token_ids,
)

inference_hyper = dict(
    cfg_img_scale=1.0, cfg_interval=[0, 1.0], timestep_shift=3.0,
    cfg_renorm_min=0.0, cfg_renorm_type="global", image_shapes=(512, 512),
)

grpo_config = ml_collections.ConfigDict()
grpo_config.sample = ml_collections.ConfigDict()
grpo_config.sample.sde_window_size = 3
grpo_config.sample.sde_window_range = (0, 7)
grpo_config.train = ml_collections.ConfigDict()
grpo_config.train.clip_range = 1e-4
grpo_config.train.clip_range_lt = 1e-5
grpo_config.train.clip_range_gt = 1e-5
grpo_config.train.beta = 0.0
grpo_config.train.adv_clip_max = 5.0
grpo_config.train.timestep_fraction = 1.0


def generate_image(prompt):
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        output = inferencer(
            text=prompt, noise_level=1.3,
            num_timesteps=15, cfg_text_scale=4.0,
            grpo_config=grpo_config, **inference_hyper,
        )
    img_t = output['image']
    img_np = (img_t * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy().transpose(1, 2, 0)
    return Image.fromarray(img_np)


def run_game(game_name, original_prompt, modified_prompt, change_desc, spy_player=2):
    """Run one game: 4 players, spy gets modified prompt."""
    log_lines = []
    log_lines.append(f"\n{'='*80}")
    log_lines.append(f"GAME: {game_name}")
    log_lines.append(f"  Original:  {original_prompt}")
    log_lines.append(f"  Modified:  {modified_prompt}")
    log_lines.append(f"  Change:    {change_desc}")
    log_lines.append(f"  Spy:       Player {spy_player}")
    log_lines.append(f"{'='*80}")

    for pid in range(1, NUM_PLAYERS + 1):
        role = "SPY" if pid == spy_player else "CIV"
        if pid == spy_player:
            prompt = f"You are Player {pid} of {NUM_PLAYERS}. {modified_prompt}"
        else:
            prompt = f"You are Player {pid} of {NUM_PLAYERS}. {original_prompt}"

        log_lines.append(f"\n  Player {pid} ({role})")
        log_lines.append(f"  Prompt: {prompt}")

        print(f"  Generating {game_name} Player {pid} ({role})...", end=" ", flush=True)
        pil = generate_image(prompt)
        fname = f"{game_name}_player{pid}_{role}.jpg"
        pil.save(os.path.join(OUTPUT_DIR, fname))
        print(f"→ {fname}")

        log_lines.append(f"  Image:  {fname}")

    return log_lines


# ─── Load datasets ───────────────────────────────────────────────────────────

# 1. Pickscore (natural captions)
with open("dataset/pickscore/train.txt") as f:
    pickscore_prompts = [line.strip() for line in f if len(line.strip()) > 20]

# 2. Geneval (structured)
with open("dataset/geneval/train_metadata.jsonl") as f:
    geneval_data = [json.loads(line) for line in f]

rng = random.Random(42)
all_log_lines = []

# ─── Pickscore games (natural scene) ─────────────────────────────────────────

print("\n" + "#" * 80)
print("# PICKSCORE (Natural Scene Captions)")
print("#" * 80)

# Pick 2 good prompts (short enough, descriptive)
good_pickscore = [p for p in pickscore_prompts if 30 < len(p) < 200]
rng.shuffle(good_pickscore)

for i in range(2):
    orig = good_pickscore[i]
    mod, change = modify_natural_prompt(orig, seed=i * 100)
    spy_player = rng.randint(1, NUM_PLAYERS)
    lines = run_game(f"pickscore_{i+1}", orig, mod, change, spy_player)
    all_log_lines.extend(lines)

# ─── OCR games (text rendering) ───────────────────────────────────────────────

print("\n" + "#" * 80)
print("# OCR (Visual Text Rendering)")
print("#" * 80)

# Load OCR prompts
with open("dataset/ocr/train.txt") as f:
    ocr_prompts = [line.strip() for line in f if len(line.strip()) > 30 and '"' in line]

rng.shuffle(ocr_prompts)

for i in range(2):
    orig = ocr_prompts[i]
    mod, change = modify_ocr_prompt(orig, seed=i * 300)
    spy_player = rng.randint(1, NUM_PLAYERS)
    lines = run_game(f"ocr_{i+1}", orig, mod, change, spy_player)
    all_log_lines.extend(lines)

# ─── Geneval games: color_attr ────────────────────────────────────────────────

print("\n" + "#" * 80)
print("# GENEVAL - Color Attribute")
print("#" * 80)

geneval_color = [d for d in geneval_data if d.get('tag') in ('color_attr', 'colors')]
rng.shuffle(geneval_color)

for i in range(2):
    data = geneval_color[i]
    orig = data['prompt']
    mod, change = modify_geneval_prompt(orig, data, seed=i * 400)
    spy_player = rng.randint(1, NUM_PLAYERS)
    lines = run_game(f"geneval_color_{i+1}", orig, mod, change, spy_player)
    all_log_lines.extend(lines)

# ─── Geneval games: counting ──────────────────────────────────────────────────

print("\n" + "#" * 80)
print("# GENEVAL - Counting")
print("#" * 80)

geneval_count = [d for d in geneval_data if d.get('tag') == 'counting']
rng.shuffle(geneval_count)

for i in range(2):
    data = geneval_count[i]
    orig = data['prompt']
    mod, change = modify_geneval_prompt(orig, data, seed=i * 500)
    spy_player = rng.randint(1, NUM_PLAYERS)
    lines = run_game(f"geneval_count_{i+1}", orig, mod, change, spy_player)
    all_log_lines.extend(lines)

# ─── Geneval games: position ──────────────────────────────────────────────────

print("\n" + "#" * 80)
print("# GENEVAL - Spatial Position")
print("#" * 80)

geneval_pos = [d for d in geneval_data if d.get('tag') == 'position']
rng.shuffle(geneval_pos)

for i in range(2):
    data = geneval_pos[i]
    orig = data['prompt']
    mod, change = modify_geneval_prompt(orig, data, seed=i * 600)
    spy_player = rng.randint(1, NUM_PLAYERS)
    lines = run_game(f"geneval_pos_{i+1}", orig, mod, change, spy_player)
    all_log_lines.extend(lines)

# ─── Save all prompts ────────────────────────────────────────────────────────

with open(os.path.join(OUTPUT_DIR, "prompts.txt"), "w") as f:
    f.write("\n".join(all_log_lines))

total_games = 2 + 2 + 2 + 2 + 2  # pickscore + ocr + color + count + position
total_images = total_games * NUM_PLAYERS
print(f"\nAll images and prompts saved to {OUTPUT_DIR}/")
print(f"Total: {total_images} images ({total_games} games × {NUM_PLAYERS} players)")
