```bash

sudo apt-get update && sudo apt-get install git-lfs ffmpeg cbm 

git clone -b frame-pack-inference https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner

pip install torch torchvision
pip install -r requirements.txt
pip install ascii-magic matplotlib tensorboard huggingface_hub datasets
pip install moviepy==1.0.3
pip install sageattention==1.0.6

git clone https://huggingface.co/lllyasviel/FramePackI2V_HY
->
--dit FramePackI2V_HY/diffusion_pytorch_model-00001-of-00003.safetensors

huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts
->
ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt
OR 
git clone https://huggingface.co/hunyuanvideo-community/HunyuanVideo
->
--vae HunyuanVideo/vae/diffusion_pytorch_model.safetensors

git clone https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged
->
--text_encoder1 HunyuanVideo_repackaged/split_files/text_encoders/llava_llama3_fp16.safetensors
--text_encoder2 HunyuanVideo_repackaged/split_files/text_encoders/clip_l.safetensors 

git clone https://huggingface.co/Comfy-Org/sigclip_vision_384
->
--image_encoder sigclip_vision_384/sigclip_vision_patch14_384.safetensors


```

```python
#### bigger than 37
import os
import shutil
from moviepy.editor import VideoFileClip

# 定义路径
src_dir = "Yi_Chen_Dancing_Animation_Videos_White_Background_Splited_Captioned_960x544x6"
dst_dir = "Yi_Chen_Dancing_Animation_Videos_White_Background_Splited_Captioned_960x544x6_upper_60fm"

# 创建目标目录（如果不存在）
os.makedirs(dst_dir, exist_ok=True)

# 遍历源目录中的所有文件
for filename in os.listdir(src_dir):
    if filename.endswith(".mp4"):
        mp4_path = os.path.join(src_dir, filename)
        txt_path = os.path.join(src_dir, filename.replace(".mp4", ".txt"))

        # 检查是否存在对应的.txt文件
        if not os.path.exists(txt_path):
            print(f"警告: 未找到 {filename} 对应的 .txt 文件，跳过。")
            continue

        # 使用MoviePy获取视频帧数
        try:
            with VideoFileClip(mp4_path) as video:
                frame_count = int(video.duration * video.fps)
                print(f"处理: {filename} | 帧数: {frame_count}")

                # 如果帧数>60，拷贝文件对
                if frame_count > 60:
                    shutil.copy2(mp4_path, dst_dir)
                    shutil.copy2(txt_path, dst_dir)
                    print(f"已拷贝: {filename} 和同名 .txt 文件到 {dst_dir}")
        except Exception as e:
            print(f"错误: 处理 {filename} 时出错 - {str(e)}")

print("处理完成！")
```

```toml
[general]
resolution = [960, 544]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_directory = "Yi_Chen_Dancing_Animation_Videos_White_Background_Splited_Captioned_960x544x6_upper_60fm"
cache_directory = "Yi_Chen_Dancing_Animation_Videos_White_Background_Splited_Captioned_960x544x6_upper_60fm_cache"
target_frames = [1, 25, 79]
max_frames = 129
frame_extraction = "full"
```

```toml
[general]
resolution = [960, 544]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_directory = "Yi_Chen_Dancing_Animation_Videos_White_Background_Splited_Captioned_960x544x6_upper_60fm"
cache_directory = "Yi_Chen_Dancing_Animation_Videos_White_Background_Splited_Captioned_960x544x6_upper_60fm_45_cache"
target_frames = [1, 25, 45]
max_frames = 45
frame_extraction = "full"
```

```bash
python fpack_cache_latents.py \
    --dataset_config video_config.toml \
    --vae HunyuanVideo/vae/diffusion_pytorch_model.safetensors \
    --image_encoder sigclip_vision_384/sigclip_vision_patch14_384.safetensors \
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128

python fpack_cache_text_encoder_outputs.py \
    --dataset_config video_config.toml \
    --text_encoder1 HunyuanVideo_repackaged/split_files/text_encoders/llava_llama3_fp16.safetensors \
    --text_encoder2 HunyuanVideo_repackaged/split_files/text_encoders/clip_l.safetensors \
    --batch_size 16

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 fpack_train_network.py \
    --dit FramePackI2V_HY/diffusion_pytorch_model-00001-of-00003.safetensors \
    --vae HunyuanVideo/vae/diffusion_pytorch_model.safetensors \
    --text_encoder1 HunyuanVideo_repackaged/split_files/text_encoders/llava_llama3_fp16.safetensors \
    --text_encoder2 HunyuanVideo_repackaged/split_files/text_encoders/clip_l.safetensors \
    --image_encoder sigclip_vision_384/sigclip_vision_patch14_384.safetensors \
    --dataset_config video_config.toml \
    --sdpa --mixed_precision bf16 \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 3.0 \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_framepack --network_dim 32 \
    --max_train_epochs 500 --save_every_n_epochs 1 --seed 42 \
    --output_dir framepack_yichen_output --output_name framepack-yichen-lora
```


