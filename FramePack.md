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

```bash
python fpack_generate_video.py \
    --dit FramePackI2V_HY/diffusion_pytorch_model-00001-of-00003.safetensors \
    --vae HunyuanVideo/vae/diffusion_pytorch_model.safetensors \
    --text_encoder1 HunyuanVideo_repackaged/split_files/text_encoders/llava_llama3_fp16.safetensors \
    --text_encoder2 HunyuanVideo_repackaged/split_files/text_encoders/clip_l.safetensors \
    --image_encoder sigclip_vision_384/sigclip_vision_patch14_384.safetensors \
    --image_path fln.png \
    --prompt "In the style of Yi Chen Dancing White Background , The character's movements shift dynamically throughout the video, transitioning from poised stillness to lively dance steps. Her expressions evolve seamlessly—starting with focused determination, then flashing surprise as she executes a quick spin, before breaking into a joyful smile mid-leap. Her hands flow through choreographed positions, sometimes extending gracefully like unfolding wings, other times clapping rhythmically against her wrists. During a dramatic hip sway, her fingers fan open near her cheek, then sweep downward as her whole body dips into a playful crouch, the sequins on her costume catching the light with every motion." \
    --video_size 960 544 --video_seconds 3 --fps 30 --infer_steps 25 \
    --attn_mode sdpa --fp8_scaled \
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 \
    --save_path save --output_type both \
    --seed 1234 --lora_multiplier 1.0 --lora_weight framepack_yichen_output/framepack-yichen-lora-000002.safetensors
```




https://github.com/user-attachments/assets/e695d2a7-4145-4cf2-b9b5-2f4ca764ec02

```bash
wget https://huggingface.co/tori29umai/FramePack_LoRA/resolve/main/rotate_landscape_V4.safetensors

huggingface-cli download --repo-type dataset \
    --resume-download \
    svjack/Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-SP \
    --include "genshin_impact_NILOU_images_and_texts/*" \
    --local-dir .
```bash

```python
import os
import glob
import shutil
from tqdm import tqdm
import subprocess
import time

def get_latest_mp4(save_dir):
    """获取save目录下最新创建的mp4文件"""
    mp4_files = glob.glob(os.path.join(save_dir, '*.mp4'))
    if not mp4_files:
        return None
    return max(mp4_files, key=os.path.getctime)

def process_images_and_texts():
    # 输入和输出目录
    input_dir = "genshin_impact_NILOU_images_and_texts"
    output_dir = "genshin_impact_NILOU_FramePack_Rotate_Dancing_Captioned"
    save_dir = "save"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # 获取所有.png文件
    png_files = glob.glob(os.path.join(input_dir, '*.png'))

    # 固定prompt
    prompt = """In the style of Yi Chen Dancing White Background , The character's movements shift dynamically throughout the video, transitioning from poised stillness to lively dance steps. Her expressions evolve seamlessly—starting with focused determination, then flashing surprise as she executes a quick spin, before breaking into a joyful smile mid-leap. Her hands flow through choreographed positions, sometimes extending gracefully like unfolding wings, other times clapping rhythmically against her wrists. During a dramatic hip sway, her fingers fan open near her cheek, then sweep downward as her whole body dips into a playful crouch, the sequins on her costume catching the light with every motion."""

    # 使用tqdm显示进度
    for png_file in tqdm(png_files, desc="Processing images"):
        # 获取对应的txt文件
        base_name = os.path.splitext(os.path.basename(png_file))[0]
        txt_file = os.path.join(input_dir, f"{base_name}.txt")

        if not os.path.exists(txt_file):
            print(f"Warning: No corresponding .txt file found for {png_file}")
            continue

        # 构建命令
        cmd = [
            "python", "fpack_generate_video.py",
            "--dit", "FramePackI2V_HY/diffusion_pytorch_model-00001-of-00003.safetensors",
            "--vae", "HunyuanVideo/vae/diffusion_pytorch_model.safetensors",
            "--text_encoder1", "HunyuanVideo_repackaged/split_files/text_encoders/llava_llama3_fp16.safetensors",
            "--text_encoder2", "HunyuanVideo_repackaged/split_files/text_encoders/clip_l.safetensors",
            "--image_encoder", "sigclip_vision_384/sigclip_vision_patch14_384.safetensors",
            "--image_path", png_file,
            "--prompt", prompt,
            "--video_size", "960", "544",
            "--video_seconds", "3",
            "--fps", "30",
            "--infer_steps", "25",
            "--attn_mode", "sdpa",
            "--fp8_scaled",
            "--vae_chunk_size", "32",
            "--vae_spatial_tile_sample_min_size", "128",
            "--save_path", save_dir,
            "--output_type", "both",
            "--seed", "1234",
            "--lora_multiplier", "1.0",
            "--lora_weight", "rotate_landscape_V4.safetensors"
        ]

        # 运行命令
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {png_file}: {e}")
            continue

        # 获取最新生成的mp4文件
        latest_mp4 = get_latest_mp4(save_dir)
        if latest_mp4 is None:
            print(f"Warning: No .mp4 file generated for {png_file}")
            continue

        # 构建输出文件名
        output_mp4 = os.path.join(output_dir, f"{base_name}.mp4")
        output_txt = os.path.join(output_dir, f"{base_name}.txt")

        # 拷贝文件
        shutil.move(latest_mp4, output_mp4)
        shutil.copy2(txt_file, output_txt)

        '''
        # 清理save目录
        for f in glob.glob(os.path.join(save_dir, '*')):
            try:
                os.remove(f)
            except:
                pass
        '''

if __name__ == "__main__":
    process_images_and_texts()
    print("All processing completed!")
```
