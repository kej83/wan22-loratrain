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
## 0.5) Make sure images and txt are in dataset folder
pic1.png, pic1.txt, pic2.png, pic2.txt etc.
yep — don’t run the cache steps until your `dataset/` actually has images (and their `.txt` captions). Those cache scripts read the dataset; if it’s empty, they error or do nothing.

Here’s the quick, safe order:

## 1) Put images + captions in place

* On Windows, finish renaming (`pic1.png` …) and captioning (`pic1.txt` …).
* Copy the whole set to your VM into:

```
~/Downloads/musubi-tuner/dataset/
```

Ways to copy:

* File transfer in ThinLinc, or
* From Windows PowerShell:

  ```powershell
  scp -r "C:\Users\you\path\to\dataset" ubuntu@<vm-ip>:/home/ubuntu/Downloads/musubi-tuner/
  ```

  (Accept the host key the first time; enter your VM password.)

## 2) Sanity-check the dataset on the VM

```bash
cd ~/Downloads/musubi-tuner

# Count images and captions
ls dataset/*.{png,jpg,jpeg,webp} 2>/dev/null | wc -l
ls dataset/*.txt 2>/dev/null | wc -l

# Show any image missing a matching .txt
for p in dataset/*.{png,jpg,jpeg,webp}; do
  [ -e "$p" ] || continue
  b="${p%.*}"
  [ -f "$b.txt" ] || echo "Missing caption: $(basename "$p")"
done

# Show any .txt that has no image
for t in dataset/*.txt; do
  [ -e "$t" ] || continue
  b="${t%.txt}"
  comp=$(ls "$b".{png,jpg,jpeg,webp} 2>/dev/null | head -n1)
  [ -n "$comp" ] || echo "Orphan caption: $(basename "$t")"
done
```

Make sure `~/Downloads/musubi-tuner/dataset/dataset.toml` exists and points to the same folder:

```toml
[general]
resolution = [960, 960]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "./dataset"
cache_directory = "./dataset/cache"
num_repeats = 1
```

## 3) Now run the cache steps (only after images exist)

```bash
cd ~/Downloads/musubi-tuner
source venv/bin/activate

# Latents
python src/musubi_tuner/wan_cache_latents.py --dataset_config ./dataset/dataset.toml --vae ./models/vae/split_files/vae/wan_2.1_vae.safetensors

# Text-encoder outputs (WAN-specific script)
python src/musubi_tuner/wan_cache_text_encoder_outputs.py --dataset_config ./dataset/dataset.toml --t5 ./models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth
```

> Any time you change images or captions, re-run both cache commands.

## 4) Train (your names/paths)

High-noise:

```bash
accelerate launch --num_cpu_threads_per_process 1 src/musubi_tuner/wan_train_network.py --task t2v-A14B --dit ./models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors --vae ./models/vae/split_files/vae/wan_2.1_vae.safetensors --t5 ./models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth --dataset_config ./dataset/dataset.toml --xformers --mixed_precision fp16 --fp8_base --optimizer_type adamw --learning_rate 3e-4 --gradient_checkpointing --gradient_accumulation_steps 1 --max_data_loader_n_workers 2 --network_module networks.lora_wan --network_dim 16 --network_alpha 16 --timestep_sampling shift --discrete_flow_shift 1.0 --max_train_epochs 100 --save_every_n_epochs 100 --seed 5 --optimizer_args weight_decay=0.1 --max_grad_norm 0 --lr_scheduler polynomial --lr_scheduler_power 8 --lr_scheduler_min_lr_ratio "5e-5" --output_dir ./output --output_name jani-wan22-1-HighNoise --metadata_title jani-wan22-1-HighNoise --metadata_author massedcompute --preserve_distribution_shape --min_timestep 875 --max_timestep 1000
```

Low-noise:

```bash
accelerate launch --num_cpu_threads_per_process 1 src/musubi_tuner/wan_train_network.py --task t2v-A14B --dit ./models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors --vae ./models/vae/split_files/vae/wan_2.1_vae.safetensors --t5 ./models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth --dataset_config ./dataset/dataset.toml --xformers --mixed_precision fp16 --fp8_base --optimizer_type adamw --learning_rate 3e-4 --gradient_checkpointing --gradient_accumulation_steps 1 --max_data_loader_n_workers 2 --network_module networks.lora_wan --network_dim 16 --network_alpha 16 --timestep_sampling shift --discrete_flow_shift 1.0 --max_train_epochs 100 --save_every_n_epochs 100 --seed 5 --optimizer_args weight_decay=0.1 --max_grad_norm 0 --lr_scheduler polynomial --lr_scheduler_power 8 --lr_scheduler_min_lr_ratio "5e-5" --output_dir ./output --output_name jani-wan22-1-LowNoise --metadata_title jani-wan22-1-LowNoise --metadata_author massedcompute --preserve_distribution_shape --min_timestep 0 --max_timestep 875
```

If you want, I can give you a tiny bash checker that enforces exactly 18 pairs and warns on any mismatch before you cache/train.
