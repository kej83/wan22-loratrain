# wan22-loratrain
Make a better structured step by step for training a wan 2.2 lora. He had a article about wan 2.1 but now he updated it for 2.2. I'm gonna send u both and you make a complete tutorial for wan 2.2 for me based on his notes. First 2.2 update article:
My WAN2.2 LoRa training workflow TLDR

180

AI_Characters's Avatar
AI_Characters

Aug 4, 2025

training guide
wan2.2
wan2.1
workflow
guide
tldr

+ 1
My WAN2.2 LoRa training workflow TLDR
The basis for this workflow is my original WAN2.1 training guide: My WAN2.1 LoRa training workflow TLDR | Civitai

In this new article I will explain only the necessary differences between WAN2.2 and WAN2.1 training!

For everything else consult the old guide.

1. Dataset and Captions

No differences.

2. VastAI

New command:

git clone --recursive https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner
git checkout feature-wan-2-2
git checkout d0a193061a23a51c90664282205d753605a641c1
apt install -y libcudnn8=8.9.7.29-1+cuda12.2 libcudnn8-dev=8.9.7.29-1+cuda12.2 --allow-change-held-packages
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install protobuf
pip install six
pip install torch==2.7.0 torchvision==0.22.0 xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128
Downloading the necessary models:

huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P models_t5_umt5-xxl-enc-bf16.pth --local-dir models/text_encoders
huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/vae/wan_2.1_vae.safetensors --local-dir models/vae
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors --local-dir models/diffusion_models
huggingface-cli download Comfy-Org/Wan_2.2_ComfyUI_Repackaged split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors --local-dir models/diffusion_models
Everything else is again the same.

Training

Everything the same except for the training command(s). You obviously need to train the LoRa on both the Low-noise and High-noise models separately (using the same dataset etc).

High-noise training command:

accelerate launch --num_cpu_threads_per_process 1 src/musubi_tuner/wan_train_network.py --task t2v-A14B --dit /workspace/musubi-tuner/models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp16.safetensors --vae /workspace/musubi-tuner/models/vae/split_files/vae/wan_2.1_vae.safetensors --t5 /workspace/musubi-tuner/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth --dataset_config /workspace/musubi-tuner/dataset/dataset.toml --xformers --mixed_precision fp16 --fp8_base --optimizer_type adamw --learning_rate 3e-4 --gradient_checkpointing --gradient_accumulation_steps 1 --max_data_loader_n_workers 2 --network_module networks.lora_wan --network_dim 16 --network_alpha 16 --timestep_sampling shift --discrete_flow_shift 1.0 --max_train_epochs 100 --save_every_n_epochs 100 --seed 5 --optimizer_args weight_decay=0.1 --max_grad_norm 0 --lr_scheduler polynomial --lr_scheduler_power 8 --lr_scheduler_min_lr_ratio="5e-5" --output_dir /workspace/musubi-tuner/output --output_name WAN2.2-HighNoise_SmartphoneSnapshotPhotoReality_v3_by-AI_Characters --metadata_title WAN2.2-HighNoise_SmartphoneSnapshotPhotoReality_v3_by-AI_Characters --metadata_author AI_Characters --preserve_distribution_shape --min_timestep 875 --max_timestep 1000
Low-noise training command:

accelerate launch --num_cpu_threads_per_process 1 src/musubi_tuner/wan_train_network.py --task t2v-A14B --dit /workspace/musubi-tuner/models/diffusion_models/split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp16.safetensors --vae /workspace/musubi-tuner/models/vae/split_files/vae/wan_2.1_vae.safetensors --t5 /workspace/musubi-tuner/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth --dataset_config /workspace/musubi-tuner/dataset/dataset.toml --xformers --mixed_precision fp16 --fp8_base --optimizer_type adamw --learning_rate 3e-4 --gradient_checkpointing --gradient_accumulation_steps 1 --max_data_loader_n_workers 2 --network_module networks.lora_wan --network_dim 16 --network_alpha 16 --timestep_sampling shift --discrete_flow_shift 1.0 --max_train_epochs 100 --save_every_n_epochs 100 --seed 5 --optimizer_args weight_decay=0.1 --max_grad_norm 0 --lr_scheduler polynomial --lr_scheduler_power 8 --lr_scheduler_min_lr_ratio="5e-5" --output_dir /workspace/musubi-tuner/output --output_name WAN2.2-LowNoise_SmartphoneSnapshotPhotoReality_v3_by-AI_Characters --metadata_title WAN2.2-LowNoise_SmartphoneSnapshotPhotoReality_v3_by-AI_Characters --metadata_author AI_Characters --preserve_distribution_shape --min_timestep 0 --max_timestep 875
Only differences between the two are the model trained on and the timestep settings at the end. Everything else is the same.

Thats it. Nothing else needs to be changed. Works very well and gives much better results than WAn2.1.

Note that I also changed the training command(s) to do dim and alpha at 16 and learning rate power at 8 as opposed to my original guides recommendation of 32 and 4 respectively, as I find these values to work better, while also giving a model half the size.

Here is the previous wan 2.1 article:
I keep getting asked how I train my WAN2.1 text2image LoRa's and I am kinda burned out right now so I'll just post this TLDR of my workflow here. I won't explain anything more than what I write here. And I wont explain why I do what I do. The answer is always the same: I tested a lot and that is what I found to be most optimal. Perhaps there is a more optimal way to do it, I dont care right now. Feel free to experiment on your own.

I use Musubi-Tuner in stead of AI-toolkit or something else because I am used to training using Kohyas SD-scripts and it usually has the most customization options.

Also this aint perfect. I find that it works very well in 99% of cases, but there are still the 1% that dont work well or sometimes most things in a model will work well except for a few prompts for some reason. E.g. I have a Rick and Morty style model on the backburner for a week now because while it generates perfect representations of the style in most cases, in a few cases it for whatever reasons does not get the style through and I have yet to figure out how after 4 different retrains.

1. Dataset

18 images. Always. No exceptions.

Styles are by far the easiest. Followed by concepts and characters.

Diversity is important to avoid overtraining on a specific thing. That includes both what is depicted and the style it is depicted in (does not apply to style LoRa's obviously).

With 3d rendered characters or concepts I find it very hard to force through a real photographic style. For some reason datasets that are majorly 3d renders struggle with that a lot. But only photos, anime and other things seem to usually work fine. So make sure to include many cosplay photos (ones that look very close) or img2img/kontext/chatgpt photo versions of the character in question. Same issue but to a lesser extent exists with anime/cartoon characters. Photo characters (e.g. celebrities) seem to work just fine though.

2. Captions

I use ChatGPT generated captions. I find that they work very well enough. I use the following prompt for them:

please individually analyse each of the images that i just uploaded for their visual contents and pair each of them with a corresponding caption that perfectly describes that image to a blind person. use objective, neutral, and natural language. do not use purple prose such as unnecessary or overly abstract verbiage. when describing something more extensively, favour concrete details that standout and can be visualised. conceptual or mood-like terms should be avoided at all costs.
some things that you can describe are:
- the style of the image (e.g. photo, artwork, anime screencap, etc)
- the subjects appearance (hair style, hair length, hair colour, eye colour, skin color, etc)
- the clothing worn by the subject
- the actions done by the subject
- the framing/shot types (e.g. full-body view, close-up portrait, etc...)
- the background/surroundings
- the lighting/time of day
- etc…
write the captions as short sentences.
three example captions:
1. "early 2010s snapshot photo captured with a phone and uploaded to facebook. three men in formal attire stand indoors on a wooden floor under a curved glass ceiling. the man on the left wears a burgundy suit with a tie, the middle man wears a black suit with a red tie, and the man on the right wears a gray tweed jacket with a patterned tie. other people are seen in the background."
2. "early 2010s snapshot photo captured with a phone and uploaded to facebook. a snowy city sidewalk is seen at night. tire tracks and footprints cover the snow. cars are parked along the street to the left, with red brake lights visible. a bus stop shelter with illuminated advertisements stands on the right side, and several streetlights illuminate the scene."
3. "early 2010s snapshot photo captured with a phone and uploaded to facebook. a young man with short brown hair, light skin, and glasses stands in an office full of shelves with files and paperwork. he wears a light brown jacket, white t-shirt, beige pants, white sneakers with black stripes, and a black smartwatch. he smiles with his hands clasped in front of him."
consistently caption the artstyle depicted in the images as “cartoon screencap in rm artstyle” and always put it at the front as the first tag in the caption. also caption the cartoonish bodily proportions as well as the simplified, exaggerated facial features with the big, round eyes with small pupils, expressive mouths, and often simplified nose shapes. caption also the clean bold black outlines, flat shading, and vibrant and saturated colors.
put the captions inside .txt files that have the same filename as the images they belong to. once youre finished, bundle them all up together into a zip archive for me to download.
Keep in mind that for some reason it often fails to number the .txt files correctly, so you will likely need to correct that or else you have the wrong captions assigned to the wrong images.

3. VastAI

I use VastAI for training. I rent H100s.

I use the following template:

Template Name: PyTorch (Vast)

Version Tag: 2.7.0-cuda-12.8.1-py310-22.04

I use 200gb storage space.

I run the following terminal command to install Musubi-Tuner and the necessary dependencies:

git clone --recursive https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner
git checkout 9c6c3ca172f41f0b4a0c255340a0f3d33468a52b
apt install -y libcudnn8=8.9.7.29-1+cuda12.2 libcudnn8-dev=8.9.7.29-1+cuda12.2 --allow-change-held-packages
python3 -m venv venv
source venv/bin/activate
pip install torch==2.7.0 torchvision==0.22.0 xformers==0.0.30 --index-url https://download.pytorch.org/whl/cu128
pip install -e .
pip install protobuf
pip install six
Use the following command to download the necessary models:

huggingface-cli login
<your HF token>
huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/diffusion_models/wan2.1_t2v_14B_fp8_e4m3fn.safetensors --local-dir models/diffusion_models
huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P models_t5_umt5-xxl-enc-bf16.pth --local-dir models/text_encoders
huggingface-cli download Comfy-Org/Wan_2.1_ComfyUI_repackaged split_files/vae/wan_2.1_vae.safetensors --local-dir models/vae
Put your images and captions into /workspace/musubi-tuner/dataset/

Create the following dataset.toml and put it into /workspace/musubi-tuner/dataset/

# resolution, caption_extension, batch_size, num_repeats, enable_bucket, bucket_no_upscale should be set in either general or datasets
# otherwise, the default values will be used for each item
# general configurations
[general]
resolution = [960 , 960]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false
[[datasets]]
image_directory = "/workspace/musubi-tuner/dataset"
cache_directory = "/workspace/musubi-tuner/dataset/cache"
num_repeats = 1 # optional, default is 1. Number of times to repeat the dataset. Useful to balance the multiple datasets with different sizes.
# other datasets can be added here. each dataset can have different configurations
4. Training

Use the following command whenever you open a new terminal window and need to do something (in order to activate the venv and be in the correct folder, usually):

cd /workspace/musubi-tuner
source venv/bin/activate
Run the following command to create the necessary latents for the training (need to rerun this everytime you change the dataset/captions):

python src/musubi_tuner/wan_cache_latents.py --dataset_config /workspace/musubi-tuner/dataset/dataset.toml --vae /workspace/musubi-tuner/models/vae/split_files/vae/wan_2.1_vae.safetensors
Run the following command to create the necessary text encoder cache for the training (need to rerun this everytime you change the dataset/captions):

python src/musubi_tuner/wan_cache_text_encoder_outputs.py --dataset_config /workspace/musubi-tuner/dataset/dataset.toml --t5 /workspace/musubi-tuner/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth
Run accelerate config once before training (everything no).

Final training command (aka my training config):

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/wan_train_network.py --task t2v-14B --dit /workspace/musubi-tuner/models/diffusion_models/split_files/diffusion_models/wan2.1_t2v_14B_fp8_e4m3fn.safetensors --vae /workspace/musubi-tuner/models/vae/split_files/vae/wan_2.1_vae.safetensors --t5 /workspace/musubi-tuner/models/text_encoders/models_t5_umt5-xxl-enc-bf16.pth --dataset_config /workspace/musubi-tuner/dataset/dataset.toml --xformers --mixed_precision bf16 --fp8_base --optimizer_type adamw --learning_rate 3e-4 --gradient_checkpointing --gradient_accumulation_steps 1 --max_data_loader_n_workers 2 --network_module networks.lora_wan --network_dim 32 --network_alpha 32 --timestep_sampling shift --discrete_flow_shift 1.0 --max_train_epochs 100 --save_every_n_epochs 100 --seed 5 --optimizer_args weight_decay=0.1 --max_grad_norm 0 --lr_scheduler polynomial --lr_scheduler_power 4 --lr_scheduler_min_lr_ratio="5e-5" --output_dir /workspace/musubi-tuner/output --output_name WAN2.1_RickAndMortyStyle_v1_by-AI_Characters --metadata_title WAN2.1_RickAndMortyStyle_v1_by-AI_Characters --metadata_author AI_Characters
I always use this same config everytime for everything. But its well tuned for my specific workflow with the 18 images and captions and everything so if you change something it will probably not work well.
