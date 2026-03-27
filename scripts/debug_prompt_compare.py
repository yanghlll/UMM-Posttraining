"""
Compare image generation with two prompt styles:
  A) Old: format_generation_prompt_simple (no player ID) — same as previous test
  B) New: format_generation_prompt (with player ID + role)

Only Game 1 (seed=0), 4 players. Saves 8 images total.
"""
import os
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
from flow_grpo.spy_game_data import SpyGameDataGenerator
from huggingface_hub import snapshot_download
import ml_collections

OUTPUT_DIR = "debug_prompt_compare"
os.makedirs(OUTPUT_DIR, exist_ok=True)

device = "cuda:0"
dtype = torch.bfloat16

# ── Load model (same as before) ──
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
    force_hooks=True, offload_folder="/adialab/usr/shadabk/MedUMM/.offload"
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

# ── Generate Game 1 with both prompt styles ──
game_gen = SpyGameDataGenerator(num_players=4)
game_data = game_gen.generate_game(epoch=0, sample_idx=0)
spy = game_data['spy_player']

print(f"\n{'='*80}")
print(f"Game 1: spy=Player {spy}")
print(f"Original: {game_data['original_description']}")
print(f"Modified: {game_data['modified_description']}")
print(f"{'='*80}")

def generate_and_save(prompt, fname, label):
    print(f"\n--- {label} ---")
    print(f"Prompt: {prompt}")
    print(f"Generating...", end=" ", flush=True)
    with torch.no_grad(), torch.amp.autocast('cuda', dtype=dtype):
        output = inferencer(
            text=prompt, noise_level=1.3,
            num_timesteps=15, cfg_text_scale=4.0,
            grpo_config=grpo_config, **inference_hyper,
        )
    img_t = output['image']
    img_np = (img_t * 255).round().clamp(0, 255).to(torch.uint8).cpu().numpy().transpose(1, 2, 0)
    pil = Image.fromarray(img_np)
    pil.save(os.path.join(OUTPUT_DIR, fname))
    print(f"saved → {fname}")

# ── Style A: Old prompt (no player ID) — recreate what previous test used ──
print(f"\n{'#'*80}")
print(f"# STYLE A: Old prompt (no player ID)")
print(f"{'#'*80}")

for pid in range(1, 5):
    desc = game_data["player_descriptions"][pid - 1]
    # This is what the OLD format_generation_prompt_simple looked like (before player ID was added)
    old_prompt = f"A 3D rendered scene with geometric objects on a flat surface: {desc}"
    role = "SPY" if pid == spy else "CIV"
    generate_and_save(old_prompt, f"old_player{pid}_{role}.jpg", f"Old P{pid}({role})")

# ── Style B: New prompt (with player ID + role) ──
print(f"\n{'#'*80}")
print(f"# STYLE B: New format_generation_prompt (player ID + role)")
print(f"{'#'*80}")

for pid in range(1, 5):
    new_prompt = game_gen.format_generation_prompt(game_data, pid)
    role = "SPY" if pid == spy else "CIV"
    generate_and_save(new_prompt, f"new_player{pid}_{role}.jpg", f"New P{pid}({role})")

# ── Style C: New format_generation_prompt_simple (player ID only) ──
print(f"\n{'#'*80}")
print(f"# STYLE C: New format_generation_prompt_simple (player ID, no role)")
print(f"{'#'*80}")

for pid in range(1, 5):
    simple_prompt = game_gen.format_generation_prompt_simple(game_data, pid)
    role = "SPY" if pid == spy else "CIV"
    generate_and_save(simple_prompt, f"simple_player{pid}_{role}.jpg", f"Simple P{pid}({role})")

print(f"\n{'='*80}")
print(f"All 12 images saved to {OUTPUT_DIR}/")
print(f"Compare: old_* (no player ID) vs new_* (full prompt) vs simple_* (player ID only)")
