"""
Use the EXACT original Bagel inference code to:
1. Generate 4 player images
2. Run VLM understanding (voting) with interleave_inference

Everything matches /adialab/usr/shadabk/MedUMM/Bagel/inference.ipynb EXACTLY:
  vae_transform = ImageTransform(1024, 512, 16)
  vit_transform = ImageTransform(980, 224, 14)

CUDA_VISIBLE_DEVICES=2 conda run -n spy_bagel python scripts/debug_original_bagel.py
"""

import os, sys, torch, time
from PIL import Image
from copy import deepcopy

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Use the SAME imports as original Bagel
from flow_grpo.bagel.data.data_utils import add_special_tokens, pil_img2rgb
from flow_grpo.bagel.data.transforms import ImageTransform
from flow_grpo.bagel.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from flow_grpo.bagel.modeling.qwen2 import Qwen2Tokenizer
from flow_grpo.bagel.modeling.autoencoder import load_ae
from flow_grpo.bagel.inferencer import InterleaveInferencer
from flow_grpo.spy_game_data import SpyGameDataGenerator

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.hooks import remove_hook_from_module
from huggingface_hub import snapshot_download

OUT = "/adialab/usr/shadabk/MedUMM/flow_grpo/test_samples/debug_original_bagel"


def main():
    device = "cuda:0"
    os.makedirs(OUT, exist_ok=True)
    dtype = torch.bfloat16

    print("=" * 70)
    print("Original Bagel Inference (exact notebook params)")
    print("=" * 70)

    # ─── Load model (exact same as inference.ipynb) ───
    model_path = snapshot_download(repo_id="ByteDance-Seed/BAGEL-7B-MoT")

    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

    config = BagelConfig(
        visual_gen=True, visual_und=True,
        llm_config=llm_config, vit_config=vit_config, vae_config=vae_config,
        vit_max_num_patch_per_side=70, connector_act='gelu_pytorch_tanh',
        latent_patch_size=2, max_latent_size=64,
    )

    with init_empty_weights():
        language_model = Qwen2ForCausalLM(llm_config)
        vit_model = SiglipVisionModel(vit_config)
        model = Bagel(language_model, vit_model, config)
        model.vit_model.vision_model.embeddings.convert_conv2d_to_linear(vit_config, meta=True)

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # ─── EXACT transforms from inference.ipynb ───
    vae_transform = ImageTransform(1024, 512, 16)
    vit_transform = ImageTransform(980, 224, 14)

    model = load_checkpoint_and_dispatch(
        model, checkpoint=os.path.join(model_path, "ema.safetensors"),
        device_map={"": device}, offload_buffers=False,
        dtype=dtype, force_hooks=True, offload_folder="/adialab/usr/shadabk/MedUMM/.offload"
    )
    for m in model.modules():
        remove_hook_from_module(m)
    model = model.eval()
    torch.cuda.empty_cache()
    vae_model.requires_grad_(False)
    model.requires_grad_(False)
    vae_model.to(device, dtype=dtype)
    model.to(device, dtype=dtype)

    inferencer = InterleaveInferencer(
        model=model, vae_model=vae_model, tokenizer=tokenizer,
        vae_transform=vae_transform, vit_transform=vit_transform,
        new_token_ids=new_token_ids,
    )
    print(f"Model loaded. GPU: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    print(f"vae_transform: ImageTransform(1024, 512, 16)")
    print(f"vit_transform: ImageTransform(980, 224, 14)")

    # ─── Generate game ───
    game_gen = SpyGameDataGenerator(num_players=4)
    game_data = game_gen.generate_game(epoch=200, sample_idx=7)
    spy_pid = game_data["spy_player"]
    print(f"\nSpy: P{spy_pid}")
    for pid in range(1, 5):
        role = "SPY" if pid == spy_pid else "CIV"
        print(f"  P{pid}[{role}]: {game_data['player_descriptions'][pid-1][:80]}...")

    # ─── Load pre-generated images (from debug_baseline run) ───
    prev_dir = "/adialab/usr/shadabk/MedUMM/flow_grpo/test_samples/debug_baseline"
    print(f"\nLoading pre-generated images from {prev_dir}...")
    pil_images = []
    for pid in range(1, 5):
        src = os.path.join(prev_dir, f"player_{pid}.png")
        img = Image.open(src).convert("RGB")
        img.save(os.path.join(OUT, f"player_{pid}.png"))
        pil_images.append(img)
        print(f"  P{pid}: loaded {img.size}")

    # ─── VLM Understanding: Original interleave_inference ───
    # This is EXACTLY how Bagel's inference.ipynb does VLM understanding
    print("\n" + "=" * 70)
    print("VLM Understanding via interleave_inference (original Bagel)")
    print("=" * 70)

    f = open(os.path.join(OUT, "results.txt"), "w")
    f.write(f"Spy: P{spy_pid}\n\n")
    for pid in range(1, 5):
        role = "SPY" if pid == spy_pid else "CIV"
        f.write(f"P{pid}[{role}]: {game_data['player_descriptions'][pid-1]}\n")
    f.write("\n")

    # Test each player voting
    for voter_pid in range(1, 5):
        role = "SPY" if voter_pid == spy_pid else "CIV"
        vote_prompt = game_gen.format_voting_prompt(game_data, player_id=voter_pid)

        f.write(f"\n{'='*60}\n")
        f.write(f"Voter: P{voter_pid} [{role}]\n")
        f.write(f"{'='*60}\n")

        # Build input_lists exactly like original Bagel: [text, image, text, image, ..., text]
        input_lists = []
        for i, img in enumerate(pil_images):
            input_lists.append(f"Player {i+1}'s generated image:")
            input_lists.append(img)
        input_lists.append(vote_prompt)

        # Test multiple temperatures
        for temp in [0.3, 0.7, 1.0]:
            for do_sample in ([False] if temp == 0.3 else [True]):
                label = f"temp={temp}" + ("" if do_sample else " greedy")

                t0 = time.time()
                output = inferencer.interleave_inference(
                    input_lists=input_lists,
                    understanding_output=True,
                    do_sample=do_sample,
                    text_temperature=temp,
                    max_think_token_n=512,
                )
                dt = time.time() - t0
                text = output[0] if isinstance(output, list) else str(output)
                extracted = game_gen.extract_vote(text)
                voted = extracted.get("voted_spy", "?") if extracted else "FAIL"

                print(f"  P{voter_pid}[{role}] {label}: → P{voted} ({dt:.1f}s) | {text[:100].replace(chr(10), ' | ')}")

                f.write(f"\n--- {label} ({dt:.1f}s) → P{voted} ---\n")
                f.write(f"{text}\n")

    # Also test with think mode
    print("\n" + "=" * 70)
    print("VLM Understanding with think=True")
    print("=" * 70)

    voter_pid = [p for p in range(1, 5) if p != spy_pid][0]
    role = "CIV"
    vote_prompt = game_gen.format_voting_prompt(game_data, player_id=voter_pid)

    input_lists = []
    for i, img in enumerate(pil_images):
        input_lists.append(f"Player {i+1}'s generated image:")
        input_lists.append(img)
    input_lists.append(vote_prompt)

    f.write(f"\n\n{'='*60}\n")
    f.write(f"Think mode: P{voter_pid} [{role}]\n")
    f.write(f"{'='*60}\n")

    t0 = time.time()
    output = inferencer.interleave_inference(
        input_lists=input_lists,
        understanding_output=True,
        think=True,
        do_sample=False,
        max_think_token_n=1024,
    )
    dt = time.time() - t0
    text = output[0] if isinstance(output, list) else str(output)
    extracted = game_gen.extract_vote(text)
    voted = extracted.get("voted_spy", "?") if extracted else "FAIL"

    print(f"  P{voter_pid}[{role}] think greedy: → P{voted} ({dt:.1f}s)")
    print(f"  {text[:200].replace(chr(10), ' | ')}")

    f.write(f"\n--- think greedy ({dt:.1f}s) → P{voted} ---\n")
    f.write(f"{text}\n")

    f.close()
    print(f"\nResults saved: {OUT}/results.txt")
    print(f"Images: {OUT}/player_*.png")
    print(f"GPU peak: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")


if __name__ == "__main__":
    main()
