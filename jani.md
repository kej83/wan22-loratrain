perfect — here are the WAN-2.2 cache + train commands rewritten for your **Ubuntu VM** layout shown in your screenshot (repo at `~/Downloads/musubi-tuner`). I’m using **relative paths from the repo root**, so you can just `cd` there and paste.

> LoRA name set to **`jani-wan22-1`** (saved as two files: `jani-wan22-1-HighNoise.safetensors` and `jani-wan22-1-LowNoise.safetensors`).

---

## 0) Go to repo + venv

```bash
cd ~/Downloads/musubi-tuner
source venv/bin/activate
```

If you haven’t run `accelerate` here yet:

```bash
accelerate config   # answer "no" to all
```

---

## 1) Cache (uses the top-level scripts you have)

```bash
# Latents
python ./cache_latents.py \
  --dataset_config ./dataset/dataset.toml \
  --vae ./models/vae/split_files/vae/wan_2.1_vae.safetensors

# Text-encoder outputs
python ./cache_text_encoder_outputs.py \
  --dataset_config ./dataset/dataset.toml \
  --t5 ./models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth
```

---

## 2) Train — HIGH-NOISE (875–1000)

```bash
accelerate launch --num_cpu_threads_per_process 1 \
  src/musubi_tuner/wan_train_network.py \
  --task t2v-A14B \
  --dit ./models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors \
  --vae ./models/vae/split_files/vae/wan_2.1_vae.safetensors \
  --t5  ./models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth \
  --dataset_config ./dataset/dataset.toml \
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
  --output_dir ./output \
  --output_name jani-wan22-1-HighNoise \
  --metadata_title jani-wan22-1-HighNoise \
  --metadata_author massedcompute \
  --preserve_distribution_shape \
  --min_timestep 875 --max_timestep 1000
```

---

## 3) Train — LOW-NOISE (0–875)

```bash
accelerate launch --num_cpu_threads_per_process 1 \
  src/musubi_tuner/wan_train_network.py \
  --task t2v-A14B \
  --dit ./models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors \
  --vae ./models/vae/split_files/vae/wan_2.1_vae.safetensors \
  --t5  ./models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth \
  --dataset_config ./dataset/dataset.toml \
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
  --output_dir ./output \
  --output_name jani-wan22-1-LowNoise \
  --metadata_title jani-wan22-1-LowNoise \
  --metadata_author massedcompute \
  --preserve_distribution_shape \
  --min_timestep 0 --max_timestep 875
```

---

### Quick sanity check (optional)

Before launching, confirm the model files exist at these exact paths:

```bash
ls ./models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth
ls ./models/vae/split_files/vae/wan_2.1_vae.safetensors
ls ./models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors
ls ./models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors
```

If any `ls` fails, tell me what your actual filenames/paths are and I’ll tweak the commands.
