# WAN 2.2 LoRA training — step-by-step (Musubi-Tuner, VastAI, H100)

> core idea: the pipeline is still Musubi-Tuner + cached latents + cached text encoder outputs. for WAN 2.2 you train **two LoRAs** on the **same dataset**: one on the **low-noise** model (`min_timestep=0, max_timestep=875`) and one on the **high-noise** model (`min_timestep=875, max_timestep=1000`). keep everything else identical between the two runs.

---

## 0) Quick spec (so you can sanity-check later)

* Trainer: **Musubi-Tuner** (Kohya-style).
* Branch/commit for WAN 2.2: `feature-wan-2-2` pinned to `d0a1930`.
* PyTorch/Xformers (CUDA 12.8 wheels): `torch==2.7.0`, `torchvision==0.22.0`, `xformers==0.0.30`.
* cuDNN packages: `libcudnn8=8.9.7.29-1+cuda12.2` and `libcudnn8-dev=8.9.7.29-1+cuda12.2`.
* Text encoder: `Wan-AI/Wan2.1-I2V-14B-720P` → `models_t5_umt5-xxl-enc-bf16.pth`.
* VAE: `Comfy-Org/Wan_2.1_ComfyUI_repackaged` → `wan_2.1_vae.safetensors`.
* Diffusion models (2.2):

  * Low-noise: `wan2.2_t2v_low_noise_14B_fp16.safetensors`
  * High-noise: `wan2.2_t2v_high_noise_14B_fp16.safetensors`
* LoRA shape: `network_dim=16`, `network_alpha=16`.
* LR schedule: polynomial with `--lr_scheduler_power 8` and `--lr_scheduler_min_lr_ratio "5e-5"`.
* Optim: AdamW, `lr=3e-4`, `weight_decay=0.1`, `max_grad_norm=0`, `fp8_base`, mixed precision `fp16`.
* Epochs: 100, save on epoch 100 (one final).
* Batch size: 1 (dataset is small and diversified).
* Dataset size: **18 images** (the author’s “always 18” workflow).
* Timestep windows:

  * **Low-noise**: `--min_timestep 0 --max_timestep 875`
  * **High-noise**: `--min_timestep 875 --max_timestep 1000`

---

## 1) Prepare your dataset & captions (unchanged)

* **18 images**. prioritize diversity (poses, compositions, backgrounds, lighting, clothing, shot types).
* put **one `.txt` caption per image** with the same filename. keep captions concrete, neutral, short, and purely descriptive. (your 2.1 captioning prompt still applies; just double-check filenames are aligned.)
* folder structure you’ll use later:

```
/workspace/musubi-tuner/
  └── dataset/
      ├── img_01.png
      ├── img_01.txt
      ├── ...
      ├── img_18.png
      ├── img_18.txt
      └── dataset.toml
```

### Minimal `dataset.toml` (same as 2.1)

```toml
[general]
resolution = [960, 960]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "/workspace/musubi-tuner/dataset"
cache_directory = "/workspace/musubi-tuner/dataset/cache"
num_repeats = 1
```

> tip: keep the same dataset.toml for both low-noise and high-noise runs; only change the model path and min/max timesteps at the command line.

---

## 2) Spin up VastAI machine (unchanged core, new repo branch)

* Template: **PyTorch (Vast)**
* Version Tag: **`2.7.0-cuda-12.8.1-py310-22.04`**
* Storage: **200 GB**
* GPU: H100 (as before)

Open a terminal and run:

```bash
# Get Musubi-Tuner and pin to WAN 2.2 branch + commit
git clone --recursive https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner
git checkout feature-wan-2-2
git checkout d0a193061a23a51c90664282205d753605a641c1

# cuDNN (allow change held packages)
apt update
apt install -y libcudnn8=8.9.7.29-1+cuda12.2 libcudnn8-dev=8.9.7.29-1+cuda12.2 --allow-change-held-packages

# Python env + deps
python3 -m venv venv
source venv/bin/activate

# Pinned CUDA wheels (CUDA 12.8 index)
pip install -e .
pip install protobuf six
pip install torch==2.7.0 torchvision==0.22.0 xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128
```

---

## 3) Download models (new for 2.2)

Log in to HF if needed:

```bash
huggingface-cli login
# paste your token
```

Download the exact files:

```bash
# Text encoder
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P \
  models_t5_umt5-xxl-enc-bf16.pth \
  --local-dir models/text_encoders

# VAE
huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
  split_files/vae/wan_2.1_vae.safetensors \
  --local-dir models/vae

# WAN 2.2 diffusion models
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
  split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors \
  --local-dir models/diffusion_models

huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
  split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors \
  --local-dir models/diffusion_models
```

Put your images + captions in:

```
/workspace/musubi-tuner/dataset/
```

Create `dataset.toml` there (as above).

---

## 4) One-time accelerate setup (unchanged)

Anytime you open a new shell:

```bash
cd /workspace/musubi-tuner
source venv/bin/activate
```

Run this **once** to init Accelerate (answer “no” to all):

```bash
accelerate config
```

---

## 5) Cache latents & text-encoder outputs (unchanged)

Re-run these two **whenever you change** the dataset or captions.

```bash
# Cache latents (VAE path is the same for both low/high runs)
python src/musubi_tuner/wan_cache_latents.py \
  --dataset_config /workspace/musubi-tuner/dataset/dataset.toml \
  --vae /workspace/musubi-tuner/models/vae/split_files/vae/wan_2.1_vae.safetensors

# Cache text encoder outputs (T5 path same for both runs)
python src/musubi_tuner/wan_cache_text_encoder_outputs.py \
  --dataset_config /workspace/musubi-tuner/dataset/dataset.toml \
  --t5 /workspace/musubi-tuner/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth
```

---

## 6) Train BOTH LoRAs (the key WAN 2.2 change)

> You’ll do **two** almost-identical `accelerate launch` runs:
>
> 1. **High-noise** (timesteps 875–1000)
> 2. **Low-noise** (timesteps 0–875)

**Shared flags** (copied from your 2.2 notes, including new dim/alpha/power):

* `--task t2v-A14B`
* `--xformers --mixed_precision fp16 --fp8_base`
* `--optimizer_type adamw --learning_rate 3e-4`
* `--gradient_checkpointing --gradient_accumulation_steps 1`
* `--max_data_loader_n_workers 2`
* `--network_module networks.lora_wan --network_dim 16 --network_alpha 16`
* `--timestep_sampling shift --discrete_flow_shift 1.0`
* `--max_train_epochs 100 --save_every_n_epochs 100`
* `--seed 5`
* `--optimizer_args weight_decay=0.1 --max_grad_norm 0`
* `--lr_scheduler polynomial --lr_scheduler_power 8 --lr_scheduler_min_lr_ratio "5e-5"`
* `--preserve_distribution_shape`
* consistent `--output_dir` and distinct `--output_name`/metadata for high vs low

### 6a) High-noise LoRA

```bash
accelerate launch --num_cpu_threads_per_process 1 \
  src/musubi_tuner/wan_train_network.py \
  --task t2v-A14B \
  --dit /workspace/musubi-tuner/models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors \
  --vae /workspace/musubi-tuner/models/vae/split_files/vae/wan_2.1_vae.safetensors \
  --t5 /workspace/musubi-tuner/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth \
  --dataset_config /workspace/musubi-tuner/dataset/dataset.toml \
  --xformers --mixed_precision fp16 --fp8_base \
  --optimizer_type adamw --learning_rate 3e-4 \
  --gradient_checkpointing --gradient_accumulation_steps 1 \
  --max_data_loader_n_workers 2 \
  --network_module networks.lora_wan --network_dim 16 --network_alpha 16 \
  --timestep_sampling shift --discrete_flow_shift 1.0 \
  --max_train_epochs 100 --save_every_n_epochs 100 \
  --seed 5 \
  --optimizer_args weight_decay=0.1 --max_grad_norm 0 \
  --lr_scheduler polynomial --lr_scheduler_power 8 --lr_scheduler_min_lr_ratio "5e-5" \
  --output_dir /workspace/musubi-tuner/output \
  --output_name WAN2.2-HighNoise_YourModel_v1 \
  --metadata_title WAN2.2-HighNoise_YourModel_v1 \
  --metadata_author YourName \
  --preserve_distribution_shape \
  --min_timestep 875 --max_timestep 1000
```

### 6b) Low-noise LoRA

```bash
accelerate launch --num_cpu_threads_per_process 1 \
  src/musubi_tuner/wan_train_network.py \
  --task t2v-A14B \
  --dit /workspace/musubi-tuner/models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors \
  --vae /workspace/musubi-tuner/models/vae/split_files/vae/wan_2.1_vae.safetensors \
  --t5 /workspace/musubi-tuner/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth \
  --dataset_config /workspace/musubi-tuner/dataset/dataset.toml \
  --xformers --mixed_precision fp16 --fp8_base \
  --optimizer_type adamw --learning_rate 3e-4 \
  --gradient_checkpointing --gradient_accumulation_steps 1 \
  --max_data_loader_n_workers 2 \
  --network_module networks.lora_wan --network_dim 16 --network_alpha 16 \
  --timestep_sampling shift --discrete_flow_shift 1.0 \
  --max_train_epochs 100 --save_every_n_epochs 100 \
  --seed 5 \
  --optimizer_args weight_decay=0.1 --max_grad_norm 0 \
  --lr_scheduler polynomial --lr_scheduler_power 8 --lr_scheduler_min_lr_ratio "5e-5" \
  --output_dir /workspace/musubi-tuner/output \
  --output_name WAN2.2-LowNoise_YourModel_v1 \
  --metadata_title WAN2.2-LowNoise_YourModel_v1 \
  --metadata_author YourName \
  --preserve_distribution_shape \
  --min_timestep 0 --max_timestep 875
```

> naming: change `YourModel_v1` to your project name. keep `HighNoise` and `LowNoise` in the filename so you never mix them up.

---

## 7) What you’ll have at the end

* Two LoRA files under `/workspace/musubi-tuner/output`:

  * `WAN2.2-LowNoise_YourModel_v1.safetensors`
  * `WAN2.2-HighNoise_YourModel_v1.safetensors`
* Both trained on the **same dataset** and settings; only the base diffusion checkpoint and timestep window differ.

---

## 8) Inference notes (how to use the pair)

* In ComfyUI / your WAN 2.2 pipeline, **load the matching LoRA onto the matching base**:

  * When using **wan2.2\_t2v\_low\_noise\_14B\_fp16**, apply the **LowNoise LoRA**.
  * When using **wan2.2\_t2v\_high\_noise\_14B\_fp16**, apply the **HighNoise LoRA**.
* Strengths: start modest (e.g., 0.6–0.8) and adjust. Because dim/alpha=16, these LoRAs are smaller and often a bit “tighter”; you may find 0.5–0.7 feels right.
* If your runtime swaps between low/high noise models for the same prompt, keep two separate presets so the correct LoRA is always applied with the correct base.

---

## 9) Optional: one-shot setup script

Paste into a fresh Vast terminal for a new run (edit HF token usage and project names as needed):

```bash
set -e

# --- paths ---
ROOT=/workspace/musubi-tuner
DATA=$ROOT/dataset
MODELS=$ROOT/models
OUT=$ROOT/output

# --- repo + deps ---
git clone --recursive https://github.com/kohya-ss/musubi-tuner.git "$ROOT"
cd "$ROOT"
git checkout feature-wan-2-2
git checkout d0a193061a23a51c90664282205d753605a641c1

apt update
apt install -y libcudnn8=8.9.7.29-1+cuda12.2 libcudnn8-dev=8.9.7.29-1+cuda12.2 --allow-change-held-packages

python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install protobuf six
pip install torch==2.7.0 torchvision==0.22.0 xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128

# --- models ---
mkdir -p "$MODELS"/{vae,text_encoders,diffusion_models}
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P \
  models_t5_umt5-xxl-enc-bf16.pth --local-dir "$MODELS/text_encoders"
huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged \
  split_files/vae/wan_2.1_vae.safetensors --local-dir "$MODELS/vae"
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
  split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors --local-dir "$MODELS/diffusion_models"
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged \
  split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors --local-dir "$MODELS/diffusion_models"

# --- cache (rerun if dataset changes) ---
python src/musubi_tuner/wan_cache_latents.py \
  --dataset_config "$DATA/dataset.toml" \
  --vae "$MODELS/vae/split_files/vae/wan_2.1_vae.safetensors"

python src/musubi_tuner/wan_cache_text_encoder_outputs.py \
  --dataset_config "$DATA/dataset.toml" \
  --t5 "$MODELS/text_encoders/models_t5_umt5-xxl-enc-bf16.pth"

# --- train high-noise ---
accelerate launch --num_cpu_threads_per_process 1 \
  src/musubi_tuner/wan_train_network.py \
  --task t2v-A14B \
  --dit "$MODELS/diffusion_models/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors" \
  --vae "$MODELS/vae/split_files/vae/wan_2.1_vae.safetensors" \
  --t5 "$MODELS/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
  --dataset_config "$DATA/dataset.toml" \
  --xformers --mixed_precision fp16 --fp8_base \
  --optimizer_type adamw --learning_rate 3e-4 \
  --gradient_checkpointing --gradient_accumulation_steps 1 \
  --max_data_loader_n_workers 2 \
  --network_module networks.lora_wan --network_dim 16 --network_alpha 16 \
  --timestep_sampling shift --discrete_flow_shift 1.0 \
  --max_train_epochs 100 --save_every_n_epochs 100 \
  --seed 5 \
  --optimizer_args weight_decay=0.1 --max_grad_norm 0 \
  --lr_scheduler polynomial --lr_scheduler_power 8 --lr_scheduler_min_lr_ratio "5e-5" \
  --output_dir "$OUT" \
  --output_name WAN2.2-HighNoise_YourModel_v1 \
  --metadata_title WAN2.2-HighNoise_YourModel_v1 \
  --metadata_author YourName \
  --preserve_distribution_shape \
  --min_timestep 875 --max_timestep 1000

# --- train low-noise ---
accelerate launch --num_cpu_threads_per_process 1 \
  src/musubi_tuner/wan_train_network.py \
  --task t2v-A14B \
  --dit "$MODELS/diffusion_models/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors" \
  --vae "$MODELS/vae/split_files/vae/wan_2.1_vae.safetensors" \
  --t5 "$MODELS/text_encoders/models_t5_umt5-xxl-enc-bf16.pth" \
  --dataset_config "$DATA/dataset.toml" \
  --xformers --mixed_precision fp16 --fp8_base \
  --optimizer_type adamw --learning_rate 3e-4 \
  --gradient_checkpointing --gradient_accumulation_steps 1 \
  --max_data_loader_n_workers 2 \
  --network_module networks.lora_wan --network_dim 16 --network_alpha 16 \
  --timestep_sampling shift --discrete_flow_shift 1.0 \
  --max_train_epochs 100 --save_every_n_epochs 100 \
  --seed 5 \
  --optimizer_args weight_decay=0.1 --max_grad_norm 0 \
  --lr_scheduler polynomial --lr_scheduler_power 8 --lr_scheduler_min_lr_ratio "5e-5" \
  --output_dir "$OUT" \
  --output_name WAN2.2-LowNoise_YourModel_v1 \
  --metadata_title WAN2.2-LowNoise_YourModel_v1 \
  --metadata_author YourName \
  --preserve_distribution_shape \
  --min_timestep 0 --max_timestep 875
```

---

## 10) Troubleshooting & gotchas

* **Wrong branch/commit**: if you forget `feature-wan-2-2` at `d0a1930`, training can fail or silently mismatch options.
* **cuDNN versions**: stick to the pinned versions; mixing CUDA 12.8 wheels with other cuDNNs can cause runtime errors.
* **Forgetting to recache**: if you edit captions or swap images, re-run both cache scripts before re-training.
* **Filename mismatches**: double-check each `image.ext` has a matching `image.txt`.
* **Overfitting symptoms** (repeated faces/poses, background “tattoos”): increase dataset diversity; keep batch size=1; don’t crank epochs above 100 in this 18-image workflow.
* **LoRA size**: dim/alpha 16 roughly halves file size vs 32; it also often generalizes better per the author’s notes. If you need more “force,” try a slightly higher strength at inference before increasing dim.

---

## 11) Minimal checklist (print-worthy)

* [ ] 18 diverse images + 18 concise captions
* [ ] `dataset.toml` in `/workspace/musubi-tuner/dataset/`
* [ ] Musubi-Tuner `feature-wan-2-2` @ `d0a1930`
* [ ] cuDNN pinned; Torch 2.7.0 / TV 0.22.0 / Xformers 0.0.30 (CUDA 12.8)
* [ ] Downloaded: WAN 2.2 low/high fp16, WAN 2.1 VAE, T5 bf16
* [ ] Cached latents + cached T5 outputs
* [ ] Trained **High-noise** (`875–1000`) and **Low-noise** (`0–875`) with identical settings otherwise
* [ ] Saved two LoRAs with clear names
* [ ] Inference: match LoRA to base (low↔low, high↔high)

---

if you want, I can wrap this into a printable PDF or a one-pager “cheat sheet” for your students/assistants—just say the word and I’ll format it.
