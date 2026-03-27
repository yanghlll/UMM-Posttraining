"""
Baseline test: Use ORIGINAL Bagel code paths to vote, then compare with ours.

Method A: Original Bagel interleave_inference (understanding_output=True)
         — This is the ORIGINAL code from /adialab/usr/shadabk/MedUMM/Bagel/inferencer.py
Method B: Original Bagel model.chat()
         — This is what Bagel eval uses
Method C: Our run_bagel_vote_multi_image_repeated()
         — This is what our training uses

All use the SAME images and SAME prompt.

CUDA_VISIBLE_DEVICES=2 conda run -n spy_bagel python scripts/debug_baseline_vote.py 2>&1 | tee test_samples/debug_baseline/run.log
"""

import os, sys, torch, time, json
from copy import deepcopy
from collections import Counter
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from flow_grpo.bagel.data.data_utils import add_special_tokens, pil_img2rgb
from flow_grpo.bagel.data.transforms import ImageTransform
from flow_grpo.bagel.modeling.bagel import (
    BagelConfig, Bagel, Qwen2Config, Qwen2ForCausalLM,
    SiglipVisionConfig, SiglipVisionModel
)
from flow_grpo.bagel.modeling.qwen2 import Qwen2Tokenizer
from flow_grpo.bagel.modeling.autoencoder import load_ae
from flow_grpo.bagel.modeling.bagel.qwen2_navit import NaiveCache
from flow_grpo.bagel.inferencer import InterleaveInferencer
from flow_grpo.spy_game_data import SpyGameDataGenerator
from flow_grpo.spy_game_reward import (
    run_bagel_vote_multi_image_repeated,
    build_vlm_image_transform,
    tensor_images_to_pil,
)

from accelerate import init_empty_weights, load_checkpoint_and_dispatch
from accelerate.hooks import remove_hook_from_module
from huggingface_hub import snapshot_download

OUT = "/adialab/usr/shadabk/MedUMM/flow_grpo/test_samples/debug_baseline"
TEMPS = [0.3, 0.7, 1.0]
K = 4


def load_model(device="cuda:0"):
    dtype = torch.bfloat16
    model_path = snapshot_download(repo_id="ByteDance-Seed/BAGEL-7B-MoT")

    llm_config = Qwen2Config.from_json_file(os.path.join(model_path, "llm_config.json"))
    llm_config.qk_norm = True
    llm_config.tie_word_embeddings = False
    llm_config.layer_module = "Qwen2MoTDecoderLayer"

    vit_config = SiglipVisionConfig.from_json_file(os.path.join(model_path, "vit_config.json"))
    vit_config.rope = False
    vit_config.num_hidden_layers -= 1

    vae_model, vae_config = load_ae(local_path=os.path.join(model_path, "ae.safetensors"))

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

    tokenizer = Qwen2Tokenizer.from_pretrained(model_path)
    tokenizer, new_token_ids, _ = add_special_tokens(tokenizer)

    # Original Bagel uses these transforms:
    # vae_transform: ImageTransform(512, 256, 8) for gen mode
    # vit_transform: ImageTransform(490, 112, 7) for the inferencer
    # eval vlm uses: ImageTransform(980, 378, 14, max_pixels=2_007_040)
    vae_transform = ImageTransform(512, 256, 8)
    vit_transform = ImageTransform(490, 112, 7)

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
    return inferencer


def generate_image(inferencer, prompt):
    import ml_collections
    mc = ml_collections.ConfigDict()
    mc.sample = ml_collections.ConfigDict()
    mc.sample.sde_window_size = 0
    mc.sample.sde_window_range = (0, 0)
    mc.train = ml_collections.ConfigDict()
    mc.train.clip_range_lt = 0.2
    mc.train.clip_range_gt = 0.2
    mc.train.beta = 0

    with torch.no_grad():
        out = inferencer.interleave_inference(
            input_lists=[prompt], understanding_output=False,
            do_sample=True, text_temperature=1.0,
            image_shapes=(512, 512), cfg_text_scale=4.0,
            cfg_img_scale=1.0, cfg_interval=[0, 1.0],
            num_timesteps=50, timestep_shift=3.0,
            cfg_renorm_min=0.0, cfg_renorm_type="global",
            grpo_config=mc,
        )
    r = out[0] if isinstance(out, list) else out
    if isinstance(r, dict):
        r = r['image']
    if isinstance(r, torch.Tensor):
        r = tensor_images_to_pil(r.unsqueeze(0) if r.dim() == 3 else r)[0]
    return r


# ──────────── Method A: Original interleave_inference ────────────
def method_A_interleave(inferencer, pil_images, vote_prompt, temp, k):
    """Exact original Bagel interleave_inference path for understanding."""
    results = []
    for _ in range(k):
        input_lists = []
        for i, img in enumerate(pil_images):
            input_lists.append(f"Player {i+1}'s generated image:")
            input_lists.append(img)
        input_lists.append(vote_prompt)

        out = inferencer.interleave_inference(
            input_lists=input_lists,
            understanding_output=True,
            do_sample=True,
            text_temperature=temp,
            max_think_token_n=512,
        )
        text = out[0] if isinstance(out, list) else str(out)
        results.append(text)
    return results


# ──────────── Method B: model.chat() with eval vlm_transform ────────────
def method_B_chat(inferencer, pil_images, vote_prompt, temp, k):
    """Original Bagel eval model.chat() — correct VLM transform."""
    vlm_transform = build_vlm_image_transform()
    model = inferencer.model
    tokenizer = inferencer.tokenizer
    new_token_ids = inferencer.new_token_ids

    images_rgb = [pil_img2rgb(img) for img in pil_images]

    # model.chat() takes images + single text prompt
    # Images are passed separately, text includes player labels
    full_prompt = ""
    for i in range(len(pil_images)):
        full_prompt += f"Player {i+1}'s generated image:\n"
    full_prompt += "\n" + vote_prompt

    results = []
    for _ in range(k):
        with torch.no_grad():
            out = model.chat(
                tokenizer=tokenizer,
                new_token_ids=new_token_ids,
                image_transform=vlm_transform,
                images=images_rgb,
                prompt=full_prompt,
                max_length=512,
                do_sample=True,
                temperature=temp,
            )
        results.append(out)
    return results


# ──────────── Method C: Our training function ────────────
def method_C_ours(inferencer, pil_images, vote_prompt, temp, k):
    """Our run_bagel_vote_multi_image_repeated — what training uses."""
    with torch.no_grad():
        results = run_bagel_vote_multi_image_repeated(
            inferencer, pil_images, vote_prompt,
            num_generations=k, max_tokens=512, temperature=temp,
        )
    return results


def main():
    device = "cuda:0"
    os.makedirs(OUT, exist_ok=True)

    print("=" * 70)
    print("Baseline Vote Test: Original Bagel vs Our Implementation")
    print("=" * 70)

    inferencer = load_model(device)
    game_gen = SpyGameDataGenerator(num_players=4)
    game_data = game_gen.generate_game(epoch=200, sample_idx=7)
    spy_pid = game_data["spy_player"]
    print(f"Spy: P{spy_pid}")

    # Generate real images
    print("Generating 4 images...")
    pil_images = []
    for pid in range(1, 5):
        prompt = game_gen.format_generation_prompt_simple(game_data, pid)
        img = generate_image(inferencer, prompt)
        img.save(os.path.join(OUT, f"player_{pid}.png"))
        pil_images.append(img)
        print(f"  P{pid} done")

    # Test with Player 1's vote prompt (civilian perspective)
    civ_pid = [p for p in range(1, 5) if p != spy_pid][0]
    vote_prompt = game_gen.format_voting_prompt(game_data, player_id=civ_pid)

    with open(os.path.join(OUT, "setup.txt"), "w") as f:
        f.write(f"Spy: P{spy_pid}\n")
        f.write(f"Voter: P{civ_pid} [CIV]\n\n")
        for pid in range(1, 5):
            role = "SPY" if pid == spy_pid else "CIV"
            f.write(f"P{pid} [{role}]: {game_data['player_descriptions'][pid-1]}\n")
        f.write(f"\nVote prompt:\n{vote_prompt}\n")

    methods = {
        "A_interleave": lambda t, k: method_A_interleave(inferencer, pil_images, vote_prompt, t, k),
        "B_chat": lambda t, k: method_B_chat(inferencer, pil_images, vote_prompt, t, k),
        "C_ours": lambda t, k: method_C_ours(inferencer, pil_images, vote_prompt, t, k),
    }

    all_results = {}

    for temp in TEMPS:
        print(f"\n{'='*70}")
        print(f"Temperature = {temp}")
        print(f"{'='*70}")
        all_results[temp] = {}

        for name, fn in methods.items():
            print(f"\n  [{name}] running K={K}...")
            t0 = time.time()
            texts = fn(temp, K)
            dt = time.time() - t0

            votes = []
            for i, text in enumerate(texts):
                v = game_gen.extract_vote(text)
                voted = v.get("voted_spy", "?") if v else "FAIL"
                votes.append(voted)
                short = text[:150].replace('\n', '\\n')
                print(f"    {i+1}. P{voted} | {short}")

            dist = dict(Counter(votes))
            print(f"    Time: {dt:.1f}s | Distribution: {dist}")
            all_results[temp][name] = {
                'texts': texts, 'votes': votes, 'dist': dist, 'time': dt,
            }

    # Save full output
    with open(os.path.join(OUT, "results.txt"), "w") as f:
        f.write(f"Spy: P{spy_pid}, Voter: P{civ_pid} [CIV]\n")
        f.write(f"Temps: {TEMPS}, K={K}\n\n")

        for temp in TEMPS:
            f.write(f"\n{'='*70}\n")
            f.write(f"Temperature = {temp}\n")
            f.write(f"{'='*70}\n\n")

            for name in methods:
                r = all_results[temp][name]
                f.write(f"--- [{name}] Time={r['time']:.1f}s Dist={r['dist']} ---\n\n")
                for i, text in enumerate(r['texts']):
                    f.write(f"  Vote {i+1} (→ P{r['votes'][i]}):\n")
                    f.write(f"  {text}\n\n")

    print(f"\nResults: {OUT}/results.txt")
    print(f"GPU peak: {torch.cuda.max_memory_allocated()/1e9:.1f}GB")


if __name__ == "__main__":
    main()
