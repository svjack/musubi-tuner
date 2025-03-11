```bash
huggingface-cli upload svjack/CLANNAD_Season_1_Videos ./CLANNAD_Season_1_Videos --repo-type dataset

huggingface-cli upload svjack/FateZero_Videos ./FateZero_Videos --repo-type dataset

huggingface-cli upload svjack/Tamako_Market_Videos ./Tamako_Market_Videos --repo-type dataset

huggingface-cli upload svjack/Beyond_the_Boundary_Videos ./Beyond_the_Boundary_Videos --repo-type dataset

huggingface-cli upload svjack/Nagi_no_Asukara_Videos_Captioned ./Nagi_no_Asukara_Videos_Captioned --repo-type dataset

python wan_generate_video.py --fp8 --task t2v-1.3B --video_size 768 1024 --video_length 81 --infer_steps 20 \
--save_path save --output_type both \
--dit wan2.1_t2v_1.3B_bf16.safetensors --vae Wan2.1_VAE.pth \
--t5 models_t5_umt5-xxl-enc-bf16.pth \
--attn_mode torch \
--lora_weight GOW_outputs/GOW_w1_3_lora-000006.safetensors \
--lora_multiplier 2.0 \
--prompt "In the style of Garden Of Words , The video opens with a view of a park entrance on a rainy day. The scene is dominated by the lush greenery of trees and the grey, overcast sky. Raindrops are visible in the air, creating a misty atmosphere. The park entrance features a stone gate with a sign that reads "SOS" in large letters, indicating an emergency call point. A person wearing a dark jacket and carrying a bag is seen walking towards the gate, holding an umbrella to shield themselves from the rain. The ground appears wet, reflecting the light from the surroundings. The overall color palette of the video is muted, with greens, greys, and browns being the most prominent."

python llava_qwen_video_caption.py --input_path "Beyond_the_Boundary_Videos" --output_path "Beyond_the_Boundary_Videos_Captioned" --max_frames 19 --fps 1 --force_sample

```

```bash
#### For I2V

vim pixel_video_config.toml

[general]
resolution = [960, 544]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_directory = "test-HunyuanVideo-pixelart-videos_960x544x6"
cache_directory = "test-HunyuanVideo-pixelart-videos_960x544x6_cache"
target_frames = [25, 45]
frame_extraction = "head"

#### Pre Compute
python wan_cache_latents.py --dataset_config pixel_video_config.toml --vae Wan2.1_VAE.pth --clip models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
python wan_cache_text_encoder_outputs.py --dataset_config pixel_video_config.toml --t5 models_t5_umt5-xxl-enc-bf16.pth --batch_size 16

wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors
wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_i2v_480p_14B_bf16.safetensors

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 wan_train_network.py \
    --task i2v-14B --t5 models_t5_umt5-xxl-enc-bf16.pth --clip models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
    --dit wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors \
    --dataset_config pixel_video_config.toml --sdpa --mixed_precision bf16 --fp8_base \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_wan --network_dim 32 \
    --timestep_sampling shift --discrete_flow_shift 3.0 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir pixel_outputs --output_name pixel_w14_lora
```

```
##### 应该先聚类 再精准

#### 第一帧 图片 聚类 或者 分多帧聚类

https://github.com/deepghs/imgutils

from imgutils.metrics import lpips_clustering

images = [f'lpips/{i}.jpg' for i in range(1, 10)]
print(images)
# ['lpips/1.jpg', 'lpips/2.jpg', 'lpips/3.jpg', 'lpips/4.jpg', 'lpips/5.jpg', 'lpips/6.jpg', 'lpips/7.jpg', 'lpips/8.jpg', 'lpips/9.jpg']
print(lpips_clustering(images))  # -1 means noises, the same as that in sklearn
# [0, 0, 0, 1, 1, -1, -1, -1, -1]

##### 可以进行尝试
https://huggingface.co/datasets/svjack/Beyond_the_Boundary_Videos_Captioned
```
