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

#### 示例训练
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


#### 尼露例子
<br/>

```bash
wget https://huggingface.co/tori29umai/FramePack_LoRA/resolve/main/rotate_landscape_V4.safetensors

huggingface-cli download --repo-type dataset \
    --resume-download \
    svjack/Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-SP \
    --include "genshin_impact_NILOU_images_and_texts/*" \
    --local-dir .
```

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

#### 原神生贺例子

```python
#### git clone https://huggingface.co/datasets/svjack/Genshin_Impact_Birthday_Art_Images

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
    input_dir = "Genshin_Impact_Birthday_Art_Images"
    output_dir = "Genshin_Impact_Birthday_Art_FramePack_Rotate_Dancing_Captioned_New"
    save_dir = "save"

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # 获取所有.png文件
    png_files_0 = glob.glob(os.path.join(input_dir, '*.png'))
    png_files_1 = glob.glob(os.path.join(input_dir, '*.jpg'))
    png_files_2 = glob.glob(os.path.join(input_dir, '*.jpeg'))
    png_files_3 = glob.glob(os.path.join(input_dir, '*.webp'))
    png_files = list(png_files_0) + list(png_files_1) + list(png_files_2) + list(png_files_3)


    # 固定prompt
    prompt = """The camera smoothly orbits around the center of the scene, keeping the center point fixed and always in view."""

    # 使用tqdm显示进度
    for png_file in tqdm(png_files, desc="Processing images"):
        # 获取对应的txt文件
        base_name = os.path.splitext(os.path.basename(png_file))[0]
        txt_file = os.path.join(input_dir, f"{base_name}.txt")

        '''
        if not os.path.exists(txt_file):
            print(f"Warning: No corresponding .txt file found for {png_file}")
            continue
        '''
        
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
            "--video_size", "768", "768",
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
        #shutil.copy2(txt_file, output_txt)

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

#### 原神生贺 比起XXX我更喜欢你 二创

##### 音频srt分割
```python
# git clone https://huggingface.co/datasets/svjack/I_prefer_you_over_something_TWO_VIDEOS


import os
import re
from pydub import AudioSegment
from moviepy.editor import VideoFileClip
from datetime import datetime, timedelta

def parse_srt(srt_content):
    """Parse SRT content into a list of subtitle segments with duration validation"""
    segments = []
    pattern = re.compile(r'(\d+)\n(\d{2}:\d{2}:\d{2},\d{3}) --> (\d{2}:\d{2}:\d{2},\d{3})\n(.+?)(?:\n\n|\n$)', re.DOTALL)

    for match in pattern.finditer(srt_content):
        index = int(match.group(1))
        start_time = parse_srt_time(match.group(2))
        end_time = parse_srt_time(match.group(3))
        text = match.group(4).strip()

        # Validate time range
        if end_time <= start_time:
            print(f"Warning: Skipping segment {index} - End time {end_time} <= Start time {start_time}")
            continue

        segments.append((index, start_time, end_time, text))

    return segments

def parse_srt_time(time_str):
    """Convert SRT time format to seconds with validation"""
    try:
        hours, mins, secs = time_str.split(':')
        secs, millis = secs.split(',')
        return timedelta(
            hours=int(hours),
            minutes=int(mins),
            seconds=int(secs),
            milliseconds=int(millis)
        ).total_seconds()
    except Exception as e:
        raise ValueError(f"Invalid time format '{time_str}': {str(e)}")

def sanitize_filename(filename):
    """Sanitize filenames with special characters"""
    # Keep Chinese/Japanese characters and basic symbols
    return re.sub(r'[<>:"/\\|?*\x00-\x1f]', '', filename).strip()

def get_video_duration(video_path):
    """Get accurate video duration in seconds"""
    try:
        with VideoFileClip(video_path) as video:
            return video.duration
    except Exception as e:
        print(f"Warning: Could not get duration for {video_path}: {str(e)}")
        return None

def adjust_segment_times(segments, max_duration):
    """Adjust segment times to fit within video duration"""
    adjusted_segments = []
    for idx, start, end, text in segments:
        # Cap end time at video duration
        if max_duration and end > max_duration:
            print(f"Adjusting segment {idx} end time from {end}s to {max_duration}s")
            end = max_duration

        # Skip if start time is beyond video duration
        if max_duration and start >= max_duration:
            print(f"Skipping segment {idx} - Start time {start}s >= video duration {max_duration}s")
            continue

        # Skip if duration becomes zero after adjustment
        if end <= start:
            print(f"Skipping segment {idx} - Invalid duration after adjustment (start: {start}s, end: {end}s)")
            continue

        adjusted_segments.append((idx, start, end, text))
    return adjusted_segments

def split_media(input_path, srt_content, output_path, output_type='wav'):
    """Improved media splitting with duration validation"""
    os.makedirs(output_path, exist_ok=True)

    try:
        # Get video duration first
        video_duration = None
        if output_type == 'mp4':
            video_duration = get_video_duration(input_path)

        # Parse and validate segments
        segments = parse_srt(srt_content)
        if video_duration:
            segments = adjust_segment_times(segments, video_duration)

        if not segments:
            print("No valid segments to process")
            return False

        input_name = os.path.splitext(os.path.basename(input_path))[0]
        clean_name = sanitize_filename(input_name)

        if output_type == 'wav':
            audio = AudioSegment.from_file(input_path)
            audio_duration = len(audio) / 1000  # pydub uses milliseconds

            for idx, start_time, end_time, text in segments:
                # Convert to milliseconds and validate against audio duration
                start_ms = int(start_time * 1000)
                end_ms = int(end_time * 1000)

                if end_ms > len(audio):
                    end_ms = len(audio)
                    print(f"Adjusting audio segment {idx} end time to {end_ms/1000}s")

                segment = audio[start_ms:end_ms]

                base_name = f"{idx:04d}_{clean_name}"
                audio_file = os.path.join(output_path, f"{base_name}.wav")
                text_file = os.path.join(output_path, f"{base_name}.txt")

                segment.export(audio_file, format='wav')
                with open(text_file, 'w', encoding='utf-8') as f:
                    f.write(text)

        elif output_type == 'mp4':
            with VideoFileClip(input_path) as video:
                for idx, start_time, end_time, text in segments:
                    # Ensure times are within video duration
                    if end_time > video.duration:
                        end_time = video.duration
                        print(f"Adjusting video segment {idx} end time to {end_time}s")

                    base_name = f"{idx:04d}_{clean_name}"
                    video_file = os.path.join(output_path, f"{base_name}.mp4")
                    text_file = os.path.join(output_path, f"{base_name}.txt")

                    segment = video.subclip(start_time, end_time)
                    segment.write_videofile(
                        video_file,
                        codec='libx264',
                        audio_codec='aac',
                        verbose=False,
                        threads=4,
                        preset='fast',
                        ffmpeg_params=['-movflags', '+faststart']
                    )
                    with open(text_file, 'w', encoding='utf-8') as f:
                        f.write(text)

        return True

    except Exception as e:
        print(f"Error processing {input_path}: {str(e)}")
        return False

# Main execution with enhanced error handling
if __name__ == "__main__":
    video_files = [
        "trimmed_🙋比起🍦我更喜欢国男v.mp4",
        "trimmed_[原神动画]我更喜欢你！比心.mp4"
    ]

    # SRT contents
    srt_contents = {
    "ja": """
1
00:00:00,000 --> 00:00:02,050
どうぞ

2
00:00:02,050 --> 00:00:04,050
何が好き？

3
00:00:04,050 --> 00:00:07,100
ミントよりもあなたが好き

4
00:00:07,100 --> 00:00:11,100
カタトちゃん、何が好き？

5
00:00:11,100 --> 00:00:14,300
ストロベリーよりもあなたが好き

6
00:00:14,300 --> 00:00:18,150
赤ちゃん、何が好き？

7
00:00:18,150 --> 00:00:23,000
クッキークリームよりもあなたが好き
""",
    "zh": """
1
00:00:00,000 --> 00:00:02,050
请说吧

2
00:00:02,050 --> 00:00:04,050
你喜欢什么？

3
00:00:04,050 --> 00:00:07,100
比起薄荷味我更喜欢你

4
00:00:07,100 --> 00:00:11,100
卡塔托酱，你喜欢什么？

5
00:00:11,100 --> 00:00:14,300
比起草莓味我更喜欢你

6
00:00:14,300 --> 00:00:18,150
宝宝喜欢什么？

7
00:00:18,150 --> 00:00:23,000
比起饼干奶油味我更喜欢你
""",
    "en": """
1
00:00:00,000 --> 00:00:02,050
Please tell me

2
00:00:02,050 --> 00:00:04,050
What do you like?

3
00:00:04,050 --> 00:00:07,100
I love you more than mint

4
00:00:07,100 --> 00:00:11,100
Katato-chan, what do you like?

5
00:00:11,100 --> 00:00:14,300
I love you more than strawberry

6
00:00:14,300 --> 00:00:18,150
Baby, what do you like?

7
00:00:18,150 --> 00:00:23,000
I love you more than cookies and cream
"""
}
    #output_types = ['mp4', 'wav']
    output_types = ['wav']

    for video in video_files:
        if not os.path.exists(video):
            print(f"Error: Input file not found - {video}")
            continue

        for lang, srt in srt_contents.items():
            for out_type in output_types:
                base_name = os.path.splitext(video)[0]
                output_dir = f"output_{lang}_{out_type}_{sanitize_filename(base_name)}"

                print(f"\nProcessing: {video} | {lang} | {out_type}")
                print(f"Output to: {output_dir}")

                success = split_media(
                    input_path=video,
                    srt_content=srt,
                    output_path=output_dir,
                    output_type=out_type
                )

                if success:
                    print("✓ Successfully processed")
                else:
                    print("✗ Processing failed")

```

##### 初级音视频拼接
```python
#git clone https://huggingface.co/datasets/svjack/I_prefer_you_over_something_GIRL_AUDIOS_SPLITED
#git clone https://huggingface.co/datasets/svjack/I_prefer_you_over_something_BOY_AUDIOS_SPLITED
#git clone https://huggingface.co/datasets/svjack/Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-SP
#git clone https://huggingface.co/datasets/svjack/Genshin_Impact_Birthday_Art_FramePack_Rotate_Named


import os
import random
import pandas as pd
from itertools import combinations
import subprocess
from moviepy.editor import concatenate_videoclips, AudioFileClip, ImageClip, CompositeAudioClip, VideoFileClip
from moviepy.video.fx import all as vfx
from moviepy.video.fx.all import crop
from moviepy.audio.AudioClip import AudioClip
from PIL import Image
import numpy as np

# 配置参数
config = {
    'num_images_per_char': 3,  # 每个角色使用的图片数量
    'output_resolution': (1024, 1024),  # 输出视频分辨率
    'image_duration': 3,  # 每张图片显示时长(秒)
    'video_duration': 5,  # 每个视频片段时长(秒)
    'audio_fade_duration': 0.5,  # 音频淡入淡出时长(秒)
}

# 角色名字映射和性别映射
name_mapping = {
    '芭芭拉': 'BARBARA', '柯莱': 'COLLEI', '雷电将军': 'RAIDEN SHOGUN', '云堇': 'YUN JIN',
    '八重神子': 'YAE MIKO', '妮露': 'NILOU', '绮良良': 'KIRARA', '砂糖': 'SUCROSE',
    '珐露珊': 'FARUZAN', '琳妮特': 'LYNETTE', '纳西妲': 'NAHIDA', '诺艾尔': 'NOELLE',
    '凝光': 'NINGGUANG', '鹿野院平藏': 'HEIZOU', '琴': 'JEAN', '枫原万叶': 'KAEDEHARA KAZUHA',
    '芙宁娜': 'FURINA', '艾尔海森': 'ALHAITHAM', '甘雨': 'GANYU', '凯亚': 'KAEYA',
    '荒泷一斗': 'ARATAKI ITTO', '优菈': 'EULA', '迪奥娜': 'DIONA', '温迪': 'VENTI',
    '神里绫人': 'KAMISATO AYATO', '阿贝多': 'ALBEDO', '重云': 'CHONGYUN', '钟离': 'ZHONGLI',
    '行秋': 'XINGQIU', '胡桃': 'HU TAO', '魈': 'XIAO', '赛诺': 'CYNO',
    '神里绫华': 'KAMISATO AYAKA', '五郎': 'GOROU', '林尼': 'LYNEY', '迪卢克': 'DILUC',
    '安柏': 'AMBER', '烟绯': 'YANFEI', '宵宫': 'YOIMIYA', '珊瑚宫心海': 'SANGONOMIYA KOKOMI',
    '罗莎莉亚': 'ROSARIA', '七七': 'QIQI', '久岐忍': 'KUKI SHINOBU', '申鹤': 'SHENHE',
    '托马': 'THOMA', '芙寧娜': 'FURINA', '雷泽': 'RAZOR'
}

gender_mapping = {
    '久岐忍': '女', '云堇': '女', '五郎': '男', '优菈': '女', '凝光': '女', '凯亚': '男',
    '安柏': '女', '宵宫': '女', '温迪': '男', '烟绯': '女', '珊瑚宫心海': '女', '琴': '女',
    '甘雨': '女', '申鹤': '女', '砂糖': '女', '神里绫人': '男', '神里绫华': '女', '绮良良': '女',
    '罗莎莉亚': '女', '胡桃': '女', '艾尔海森': '男', '荒泷一斗': '男', '行秋': '男', '诺艾尔': '女',
    '迪卢克': '男', '迪奥娜': '女', '重云': '男', '钟离': '男', '阿贝多': '男', '雷泽': '男',
    '雷电将军': '女', '魈': '男', '鹿野院平藏': '男'
}

# 路径配置
paths = {
    'female_audios': 'I_prefer_you_over_something_GIRL_AUDIOS_SPLITED',
    'male_audios': 'I_prefer_you_over_something_BOY_AUDIOS_SPLITED',
    'images': 'Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-SP',
    'videos': 'Genshin_Impact_Birthday_Art_FramePack_Rotate_Named'
}

# 加载视频元数据
video_metadata = pd.read_csv(os.path.join(paths['videos'], 'metadata.csv'))

def get_character_resources(gender, num_groups):
    """获取指定性别的角色组合和资源"""
    # 按性别筛选角色
    chars = [name for name, g in gender_mapping.items() if g == gender]

    # 生成所有3角色组合
    all_combinations = list(combinations(chars, 3))
    random.shuffle(all_combinations)

    # 只取需要的数量
    selected_combinations = all_combinations[:num_groups]

    results = []
    for combo in selected_combinations:
        group_data = []
        for char in combo:
            # 获取角色英文名
            en_name = name_mapping.get(char, char).replace(' ', '_').upper()

            # 获取音频文件
            audio_dir = paths['female_audios'] if gender == '女' else paths['male_audios']
            audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav') or f.endswith('.mp3')])

            # 获取图片文件
            image_dir = os.path.join(paths['images'], f'genshin_impact_{en_name}_images_and_texts')
            image_files = []
            if os.path.exists(image_dir):
                image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]
                random.shuffle(image_files)
                image_files = image_files[:config['num_images_per_char']]

            # 获取视频文件
            video_files = video_metadata[video_metadata['prompt'] == char]['file_name'].tolist()

            group_data.append({
                'name': char,
                'en_name': en_name,
                'audio_files': audio_files,
                'image_files': [os.path.join(image_dir, f) for f in image_files],
                'video_files': [os.path.join(paths['videos'], f) for f in video_files]
            })

        results.append(group_data)

    return results

def create_video_clip(group_data, output_path):
    """为单个角色组合创建视频（根据音频时间分配图片显示时间）"""
    clips = []
    audio_clips = []
    current_time = 0  # 跟踪当前时间位置

    # 获取音频文件
    audio_dir = paths['female_audios'] if gender_mapping[group_data[0]['name']] == '女' else paths['male_audios']
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3'))])

    # 确保有足够的音频文件（7个）
    if len(audio_files) < 7:
        raise ValueError("音频文件不足7个，无法生成视频")

    # 加载所有音频并记录时长
    audio_durations = []
    for i in range(7):  # 加载7个音频
        audio_path = os.path.join(audio_dir, audio_files[i])
        audio_clip = AudioFileClip(audio_path)
        audio_durations.append(audio_clip.duration)
        audio_clips.append(audio_clip)

    def apply_effects(img_clip):
        """对图片剪辑应用特效"""
        # 轻微旋转和缩放
        img_clip = img_clip.fx(vfx.rotate, angle=lambda t: 5 * np.sin(2 * np.pi * t / img_clip.duration), expand=False)
        img_clip = img_clip.fx(vfx.resize, lambda t: 1 + 0.1 * np.sin(2 * np.pi * t / img_clip.duration))

        # 淡入淡出效果
        img_clip = img_clip.fx(vfx.fadein, 0.1).fx(vfx.fadeout, 0.1)

        return img_clip

    def apply_video_effects(video_clip):
        """对视频剪辑应用特效，分三次应用"""
        duration = video_clip.duration
        third_duration = duration / 3

        def apply_single_effect(clip):
            # 轻微旋转和缩放
            clip = clip.fx(vfx.rotate, angle=lambda t: 5 * np.sin(2 * np.pi * t / clip.duration), expand=False)
            clip = clip.fx(vfx.resize, lambda t: 1 + 0.1 * np.sin(2 * np.pi * t / clip.duration))
            return clip

        # 分割视频剪辑为三个部分
        part1 = video_clip.subclip(0, third_duration)
        part2 = video_clip.subclip(third_duration, 2 * third_duration)
        part3 = video_clip.subclip(2 * third_duration, duration)

        # 对每个部分应用特效
        part1 = apply_single_effect(part1)
        part2 = apply_single_effect(part2)
        part3 = apply_single_effect(part3)

        # 合并三个部分
        final_video_clip = concatenate_videoclips([part1, part2, part3])
        return final_video_clip

    # 角色1: 3图片共享音频1+2时长，视频使用音频3
    char1 = group_data[0]
    char1_audio_duration = audio_durations[0] + audio_durations[1]

    # 计算每张图片的显示时间（平均分配）
    if len(char1['image_files']) >= 3:
        per_image_duration = char1_audio_duration / 3
        for i in range(3):
            img_clip = ImageClip(char1['image_files'][i], duration=per_image_duration)
            img_clip = img_clip.resize(config['output_resolution'])
            img_clip = apply_effects(img_clip)
            clips.append(img_clip)
            current_time += per_image_duration

    # 视频1使用音频3
    if len(char1['video_files']) > 0:
        video_clip = VideoFileClip(char1['video_files'][0])
        original_duration = video_clip.duration
        target_duration = audio_durations[2]
        speed_factor = original_duration / target_duration
        video_clip = video_clip.fx(vfx.speedx, speed_factor)
        video_clip = video_clip.resize(config['output_resolution'])
        video_clip = apply_video_effects(video_clip)
        clips.append(video_clip)
        current_time += target_duration

    # 角色2: 3图片共享音频4时长，视频使用音频5
    char2 = group_data[1]
    char2_audio_duration = audio_durations[3]

    if len(char2['image_files']) >= 3:
        per_image_duration = char2_audio_duration / 3
        for i in range(3):
            img_clip = ImageClip(char2['image_files'][i], duration=per_image_duration)
            img_clip = img_clip.resize(config['output_resolution'])
            img_clip = apply_effects(img_clip)
            clips.append(img_clip)
            current_time += per_image_duration

    # 视频2使用音频5
    if len(char2['video_files']) > 0:
        video_clip = VideoFileClip(char2['video_files'][0])
        original_duration = video_clip.duration
        target_duration = audio_durations[4]
        speed_factor = original_duration / target_duration
        video_clip = video_clip.fx(vfx.speedx, speed_factor)
        video_clip = video_clip.resize(config['output_resolution'])
        video_clip = apply_video_effects(video_clip)
        clips.append(video_clip)
        current_time += target_duration

    # 角色3: 3图片共享音频6时长，视频使用音频7
    char3 = group_data[2]
    char3_audio_duration = audio_durations[5]

    if len(char3['image_files']) >= 3:
        per_image_duration = char3_audio_duration / 3
        for i in range(3):
            img_clip = ImageClip(char3['image_files'][i], duration=per_image_duration)
            img_clip = img_clip.resize(config['output_resolution'])
            img_clip = apply_effects(img_clip)
            clips.append(img_clip)
            current_time += per_image_duration

    # 视频3使用音频7
    if len(char3['video_files']) > 0:
        video_clip = VideoFileClip(char3['video_files'][0])
        original_duration = video_clip.duration
        target_duration = audio_durations[6]
        speed_factor = original_duration / target_duration
        video_clip = video_clip.fx(vfx.speedx, speed_factor)
        video_clip = video_clip.resize(config['output_resolution'])
        video_clip = apply_video_effects(video_clip)
        clips.append(video_clip)
        current_time += target_duration

    # 合并视频片段
    final_video = concatenate_videoclips(clips, method="compose")

    # 合并音频（严格对齐）
    aligned_audio_clips = []
    current_audio_time = 0

    # 角色1音频（音频1+2用于图片，音频3用于视频）
    aligned_audio_clips.append(audio_clips[0].set_start(current_audio_time))
    current_audio_time += audio_durations[0]
    aligned_audio_clips.append(audio_clips[1].set_start(current_audio_time))
    current_audio_time += audio_durations[1]
    aligned_audio_clips.append(audio_clips[2].set_start(current_audio_time))
    current_audio_time += audio_durations[2]

    # 角色2音频（音频4用于图片，音频5用于视频）
    aligned_audio_clips.append(audio_clips[3].set_start(current_audio_time))
    current_audio_time += audio_durations[3]
    aligned_audio_clips.append(audio_clips[4].set_start(current_audio_time))
    current_audio_time += audio_durations[4]

    # 角色3音频（音频6用于图片，音频7用于视频）
    aligned_audio_clips.append(audio_clips[5].set_start(current_audio_time))
    current_audio_time += audio_durations[5]
    aligned_audio_clips.append(audio_clips[6].set_start(current_audio_time))

    final_audio = CompositeAudioClip(aligned_audio_clips)
    final_video.audio = final_audio

    # 写入输出文件
    final_video.write_videofile(output_path, fps=24, codec='libx264', audio_codec='aac')



def generate_videos(gender, num_groups, output_dir='output'):
    """生成指定数量的视频"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取角色组合和资源
    character_groups = get_character_resources(gender, num_groups)

    #print(character_groups)

    # 为每个组合创建视频
    for i, group in enumerate(character_groups):
        output_path = os.path.join(output_dir, f'{gender}_group_{i+1}.mp4')
        print(f"正在生成视频: {output_path}")
        print(f"角色组合: {[char['name'] for char in group]}")

        try:
            create_video_clip(group, output_path)
            print(f"成功生成: {output_path}")
        except Exception as e:
            print(f"生成视频失败: {e}")
            continue

generate_videos('女', 2)

generate_videos('男', 2)
```

#### 加入视频特效和字幕拼接 
```python
import os
import random
import pandas as pd
from itertools import combinations
import subprocess
from moviepy.editor import concatenate_videoclips, AudioFileClip, ImageClip, CompositeAudioClip, VideoFileClip, TextClip, ColorClip, CompositeVideoClip
from moviepy.video.fx import all as vfx
from moviepy.video.fx.all import crop
from moviepy.audio.AudioClip import AudioClip
from PIL import Image
import numpy as np

# 配置参数
config = {
    'num_images_per_char': 3,  # 每个角色使用的图片数量
    'output_resolution': (1024, 1024),  # 输出视频分辨率
    'image_duration': 3,  # 每张图片显示时长(秒)
    'video_duration': 5,  # 每个视频片段时长(秒)
    'audio_fade_duration': 0.5,  # 音频淡入淡出时长(秒)
}

# 角色名字映射和性别映射
name_mapping = {
    '芭芭拉': 'BARBARA', '柯莱': 'COLLEI', '雷电将军': 'RAIDEN SHOGUN', '云堇': 'YUN JIN',
    '八重神子': 'YAE MIKO', '妮露': 'NILOU', '绮良良': 'KIRARA', '砂糖': 'SUCROSE',
    '珐露珊': 'FARUZAN', '琳妮特': 'LYNETTE', '纳西妲': 'NAHIDA', '诺艾尔': 'NOELLE',
    '凝光': 'NINGGUANG', '鹿野院平藏': 'HEIZOU', '琴': 'JEAN', '枫原万叶': 'KAEDEHARA KAZUHA',
    '芙宁娜': 'FURINA', '艾尔海森': 'ALHAITHAM', '甘雨': 'GANYU', '凯亚': 'KAEYA',
    '荒泷一斗': 'ARATAKI ITTO', '优菈': 'EULA', '迪奥娜': 'DIONA', '温迪': 'VENTI',
    '神里绫人': 'KAMISATO AYATO', '阿贝多': 'ALBEDO', '重云': 'CHONGYUN', '钟离': 'ZHONGLI',
    '行秋': 'XINGQIU', '胡桃': 'HU TAO', '魈': 'XIAO', '赛诺': 'CYNO',
    '神里绫华': 'KAMISATO AYAKA', '五郎': 'GOROU', '林尼': 'LYNEY', '迪卢克': 'DILUC',
    '安柏': 'AMBER', '烟绯': 'YANFEI', '宵宫': 'YOIMIYA', '珊瑚宫心海': 'SANGONOMIYA KOKOMI',
    '罗莎莉亚': 'ROSARIA', '七七': 'QIQI', '久岐忍': 'KUKI SHINOBU', '申鹤': 'SHENHE',
    '托马': 'THOMA', '芙寧娜': 'FURINA', '雷泽': 'RAZOR'
}

gender_mapping = {
    '久岐忍': '女', '云堇': '女', '五郎': '男', '优菈': '女', '凝光': '女', '凯亚': '男',
    '安柏': '女', '宵宫': '女', '温迪': '男', '烟绯': '女', '珊瑚宫心海': '女', '琴': '女',
    '甘雨': '女', '申鹤': '女', '砂糖': '女', '神里绫人': '男', '神里绫华': '女', '绮良良': '女',
    '罗莎莉亚': '女', '胡桃': '女', '艾尔海森': '男', '荒泷一斗': '男', '行秋': '男', '诺艾尔': '女',
    '迪卢克': '男', '迪奥娜': '女', '重云': '男', '钟离': '男', '阿贝多': '男', '雷泽': '男',
    '雷电将军': '女', '魈': '男', '鹿野院平藏': '男'
}

character_info = {
    '久岐忍': {
        '[对角色的称呼]': '雷元素奶妈（但奶量全靠队友自己努力）',
        '[比起XXX我更喜欢你]': '比起奶队友，我更喜欢你（反正他们也死不了）'
    },
    '云堇': {
        '[对角色的称呼]': '璃月戏曲名角（但观众主要是为了看脸）',
        '[比起XXX我更喜欢你]': '比起唱戏，我更喜欢你（反正你也听不懂戏词）'
    },
    '五郎': {
        '[对角色的称呼]': '海祇岛大将（但打架全靠狗狗帮忙）',
        '[比起XXX我更喜欢你]': '比起打仗，我更喜欢你（反正你也打不过我）'
    },
    '优菈': {
        '[对角色的称呼]': '浪花骑士（但浪花主要是用来逃跑的）',
        '[比起XXX我更喜欢你]': '比起复仇之舞，我更喜欢你（反正你也记不住仇）'
    },
    '凝光': {
        '[对角色的称呼]': '天权星（但钱都用来买新衣服了）',
        '[比起XXX我更喜欢你]': '比起赚钱，我更喜欢你（反正你也赚不到我的钱）'
    },
    '凯亚': {
        '[对角色的称呼]': '渡海真君（但渡海主要靠冰面滑行）',
        '[比起XXX我更喜欢你]': '比起冰面滑行，我更喜欢你（反正你也滑不过我）'
    },
    '安柏': {
        '[对角色的称呼]': '侦察骑士（但侦察主要靠兔兔伯爵）',
        '[比起XXX我更喜欢你]': '比起飞行冠军，我更喜欢你（反正你也飞不过我）'
    },
    '宵宫': {
        '[对角色的称呼]': '烟花大师（但烟花主要是用来炸鱼的）',
        '[比起XXX我更喜欢你]': '比起放烟花，我更喜欢你（反正你也躲不开我的烟花）'
    },
    '温迪': {
        '[对角色的称呼]': '吟游诗人（但主要收入来源是蹭酒）',
        '[比起XXX我更喜欢你]': '比起喝酒，我更喜欢你（反正你也喝不过我）'
    },
    '烟绯': {
        '[对角色的称呼]': '律法专家（但打官司主要靠嘴炮）',
        '[比起XXX我更喜欢你]': '比起打官司，我更喜欢你（反正你也说不过我）'
    },
    '珊瑚宫心海': {
        '[对角色的称呼]': '现人神巫女（但军事策略全靠锦囊）',
        '[比起XXX我更喜欢你]': '比起军事策略，我更喜欢你（反正你也看不懂锦囊）'
    },
    '琴': {
        '[对角色的称呼]': '蒲公英骑士（但主要工作是批文件）',
        '[比起XXX我更喜欢你]': '比起批文件，我更喜欢你（反正你也批不完）'
    },
    '甘雨': {
        '[对角色的称呼]': '麒麟少女（但加班加到忘记自己是麒麟）',
        '[比起XXX我更喜欢你]': '比起加班，我更喜欢你（反正你也加不完）'
    },
    '申鹤': {
        '[对角色的称呼]': '驱邪方士（但驱邪主要靠物理超度）',
        '[比起XXX我更喜欢你]': '比起除妖，我更喜欢你（反正你也打不过我）'
    },
    '砂糖': {
        '[对角色的称呼]': '炼金术士（但实验主要靠运气）',
        '[比起XXX我更喜欢你]': '比起做实验，我更喜欢你（反正你也看不懂配方）'
    },
    '神里绫人': {
        '[对角色的称呼]': '社奉行家主（但工作主要靠妹妹帮忙）',
        '[比起XXX我更喜欢你]': '比起处理政务，我更喜欢你（反正你也处理不完）'
    },
    '神里绫华': {
        '[对角色的称呼]': '白鹭公主（但剑术表演主要为了好看）',
        '[比起XXX我更喜欢你]': '比起剑术表演，我更喜欢你（反正你也学不会）'
    },
    '绮良良': {
        '[对角色的称呼]': '快递员（但送货主要靠滚来滚去）',
        '[比起XXX我更喜欢你]': '比起送快递，我更喜欢你（反正你也追不上我）'
    },
    '罗莎莉亚': {
        '[对角色的称呼]': '修女（但祷告时间主要用来睡觉）',
        '[比起XXX我更喜欢你]': '比起夜间巡逻，我更喜欢你（反正你也找不到我）'
    },
    '胡桃': {
        '[对角色的称呼]': '往生堂堂主（但推销棺材主要靠押韵）',
        '[比起XXX我更喜欢你]': '比起推销棺材，我更喜欢你（反正你也逃不掉）'
    },
    '艾尔海森': {
        '[对角色的称呼]': '书记官（但看书主要为了抬杠）',
        '[比起XXX我更喜欢你]': '比起看书，我更喜欢你（反正你也说不过我）'
    },
    '荒泷一斗': {
        '[对角色的称呼]': '鬼族豪杰（但打架主要靠嗓门大）',
        '[比起XXX我更喜欢你]': '比起相扑比赛，我更喜欢你（反正你也赢不了）'
    },
    '行秋': {
        '[对角色的称呼]': '飞云商会二小姐（但写小说主要靠脑补）',
        '[比起XXX我更喜欢你]': '比起看武侠小说，我更喜欢你（反正你也写不出来）'
    },
    '诺艾尔': {
        '[对角色的称呼]': '女仆骑士（但打扫范围包括整个蒙德）',
        '[比起XXX我更喜欢你]': '比起打扫卫生，我更喜欢你（反正你也拦不住我）'
    },
    '迪卢克': {
        '[对角色的称呼]': '暗夜英雄（但行侠主要靠钞能力）',
        '[比起XXX我更喜欢你]': '比起打击犯罪，我更喜欢你（反正你也买不起酒庄）'
    },
    '迪奥娜': {
        '[对角色的称呼]': '猫尾酒保（但调酒主要为了难喝）',
        '[比起XXX我更喜欢你]': '比起调酒，我更喜欢你（反正你也不敢喝）'
    },
    '重云': {
        '[对角色的称呼]': '驱邪世家传人（但最怕吃辣）',
        '[比起XXX我更喜欢你]': '比起吃冰棍，我更喜欢你（反正你也忍不住）'
    },
    '钟离': {
        '[对角色的称呼]': '往生堂客卿（但记账主要靠公子）',
        '[比起XXX我更喜欢你]': '比起听戏，我更喜欢你（反正你也付不起钱）'
    },
    '阿贝多': {
        '[对角色的称呼]': '白垩之子（但画画主要靠炼金术）',
        '[比起XXX我更喜欢你]': '比起画画，我更喜欢你（反正你也看不懂）'
    },
    '雷泽': {
        '[对角色的称呼]': '狼少年（但说话主要靠卢皮卡）',
        '[比起XXX我更喜欢你]': '比起和狼群玩耍，我更喜欢你（反正你也听不懂）'
    },
    '雷电将军': {
        '[对角色的称呼]': '御建鸣神主尊大御所大人（但做饭会引发核爆）',
        '[比起XXX我更喜欢你]': '比起追求永恒，我更喜欢你（反正你也逃不掉）'
    },
    '魈': {
        '[对角色的称呼]': '护法夜叉（但总在屋顶看风景）',
        '[比起XXX我更喜欢你]': '比起除魔，我更喜欢你（反正你也找不到我）'
    },
    '鹿野院平藏': {
        '[对角色的称呼]': '天领奉行侦探（但破案主要靠直觉）',
        '[比起XXX我更喜欢你]': '比起破案，我更喜欢你（反正你也猜不透）'
    }
}


# 路径配置
paths = {
    'female_audios': 'I_prefer_you_over_something_GIRL_AUDIOS_SPLITED',
    'male_audios': 'I_prefer_you_over_something_BOY_AUDIOS_SPLITED',
    'images': 'Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-SP',
    'videos': 'Genshin_Impact_Birthday_Art_FramePack_Rotate_Named'
}

# 加载视频元数据
video_metadata = pd.read_csv(os.path.join(paths['videos'], 'metadata.csv'))

def get_character_resources(gender, num_groups):
    """获取指定性别的角色组合和资源"""
    # 按性别筛选角色
    chars = [name for name, g in gender_mapping.items() if g == gender]

    # 生成所有3角色组合
    all_combinations = list(combinations(chars, 3))
    random.shuffle(all_combinations)

    # 只取需要的数量
    selected_combinations = all_combinations[:num_groups]

    results = []
    for combo in selected_combinations:
        group_data = []
        for char in combo:
            # 获取角色英文名
            en_name = name_mapping.get(char, char).replace(' ', '_').upper()

            # 获取音频文件
            audio_dir = paths['female_audios'] if gender == '女' else paths['male_audios']
            audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith('.wav') or f.endswith('.mp3')])

            # 获取图片文件
            image_dir = os.path.join(paths['images'], f'genshin_impact_{en_name}_images_and_texts')
            image_files = []
            if os.path.exists(image_dir):
                image_files = [f for f in os.listdir(image_dir) if f.lower().endswith('.png')]
                random.shuffle(image_files)
                image_files = image_files[:config['num_images_per_char']]

            # 获取视频文件
            video_files = video_metadata[video_metadata['prompt'] == char]['file_name'].tolist()

            group_data.append({
                'name': char,
                'en_name': en_name,
                'audio_files': audio_files,
                'image_files': [os.path.join(image_dir, f) for f in image_files],
                'video_files': [os.path.join(paths['videos'], f) for f in video_files]
            })

        results.append(group_data)

    return results

def create_video_clip(group_data, output_path):
    """为单个角色组合创建视频（根据音频时间分配图片显示时间），并生成独立的SRT字幕文件"""
    clips = []
    audio_clips = []
    current_time = 0  # 跟踪当前时间位置

    # 获取角色名字用于输出文件名
    char_names = "_".join([char['name'] for char in group_data])
    base_output_path = output_path.replace('.mp4', f'_{char_names}.mp4')
    srt_output_path = base_output_path.replace('.mp4', '.srt')  # 字幕文件路径

    # 获取音频文件
    audio_dir = paths['female_audios'] if gender_mapping[group_data[0]['name']] == '女' else paths['male_audios']
    audio_files = sorted([f for f in os.listdir(audio_dir) if f.endswith(('.wav', '.mp3'))])

    # 确保有足够的音频文件（7个）
    if len(audio_files) < 7:
        raise ValueError("音频文件不足7个，无法生成视频")

    # 加载所有音频并记录时长
    audio_durations = []
    for i in range(7):  # 加载7个音频
        audio_path = os.path.join(audio_dir, audio_files[i])
        audio_clip = AudioFileClip(audio_path)
        audio_durations.append(audio_clip.duration)
        audio_clips.append(audio_clip)

    # 生成字幕内容
    srt_template = """
1
00:00:00,000 --> 00:00:02,050
{char1_title}

2
00:00:02,050 --> 00:00:04,050
你喜欢什么？

3
00:00:04,050 --> 00:00:07,100
{char1_preference}

4
00:00:07,100 --> 00:00:11,100
{char2_title}，你喜欢什么？

5
00:00:11,100 --> 00:00:14,300
{char2_preference}

6
00:00:14,300 --> 00:00:18,150
{char3_title}，你喜欢什么？

7
00:00:18,150 --> 00:00:23,000
{char3_preference}

8
00:00:23,000 --> 00:00:23,000
{char3_preference}
"""

    # 填充字幕模板
    char1_info = character_info.get(group_data[0]['name'], {'[对角色的称呼]': group_data[0]['name'], '[比起XXX我更喜欢你]': f'比起XXX，我更喜欢你'})
    char2_info = character_info.get(group_data[1]['name'], {'[对角色的称呼]': group_data[1]['name'], '[比起XXX我更喜欢你]': f'比起XXX，我更喜欢你'})
    char3_info = character_info.get(group_data[2]['name'], {'[对角色的称呼]': group_data[2]['name'], '[比起XXX我更喜欢你]': f'比起XXX，我更喜欢你'})

    srt_content = srt_template.format(
        char1_title=char1_info['[对角色的称呼]'],
        char1_preference=char1_info['[比起XXX我更喜欢你]'],
        char2_title=char2_info['[对角色的称呼]'],
        char2_preference=char2_info['[比起XXX我更喜欢你]'],
        char3_title=char3_info['[对角色的称呼]'],
        char3_preference=char3_info['[比起XXX我更喜欢你]']
    ).strip()

    # 将字幕内容写入SRT文件
    with open(srt_output_path, 'w', encoding='utf-8') as f:
        f.write(srt_content)

    def apply_effects(img_clip):
        """对图片剪辑应用特效"""
        # 轻微旋转和缩放
        img_clip = img_clip.fx(vfx.rotate, angle=lambda t: 5 * np.sin(2 * np.pi * t / img_clip.duration), expand=False)
        img_clip = img_clip.fx(vfx.resize, lambda t: 1 + 0.1 * np.sin(2 * np.pi * t / img_clip.duration))

        # 淡入淡出效果
        img_clip = img_clip.fx(vfx.fadein, 0.1).fx(vfx.fadeout, 0.1)

        return img_clip

    def apply_video_effects(video_clip):
        """对视频剪辑应用特效，分三次应用"""
        duration = video_clip.duration
        third_duration = duration / 3

        def apply_single_effect(clip):
            # 轻微旋转和缩放
            clip = clip.fx(vfx.rotate, angle=lambda t: 5 * np.sin(2 * np.pi * t / clip.duration), expand=False)
            clip = clip.fx(vfx.resize, lambda t: 1 + 0.1 * np.sin(2 * np.pi * t / clip.duration))
            return clip

        # 分割视频剪辑为三个部分
        part1 = video_clip.subclip(0, third_duration)
        part2 = video_clip.subclip(third_duration, 2 * third_duration)
        part3 = video_clip.subclip(2 * third_duration, duration)

        # 对每个部分应用特效
        part1 = apply_single_effect(part1)
        part2 = apply_single_effect(part2)
        part3 = apply_single_effect(part3)

        # 合并三个部分
        final_video_clip = concatenate_videoclips([part1, part2, part3])
        return final_video_clip

    # 角色1: 3图片共享音频1+2时长，视频使用音频3
    char1 = group_data[0]
    char1_audio_duration = audio_durations[0] + audio_durations[1]

    # 计算每张图片的显示时间（平均分配）
    if len(char1['image_files']) >= 3:
        per_image_duration = char1_audio_duration / 3
        for i in range(3):
            img_clip = ImageClip(char1['image_files'][i], duration=per_image_duration)
            img_clip = img_clip.resize(config['output_resolution'])
            img_clip = apply_effects(img_clip)
            clips.append(img_clip)
            current_time += per_image_duration

    # 视频1使用音频3
    if len(char1['video_files']) > 0:
        video_clip = VideoFileClip(char1['video_files'][0])
        original_duration = video_clip.duration
        target_duration = audio_durations[2]
        speed_factor = original_duration / target_duration
        video_clip = video_clip.fx(vfx.speedx, speed_factor)
        video_clip = video_clip.resize(config['output_resolution'])
        video_clip = apply_video_effects(video_clip)
        clips.append(video_clip)
        current_time += target_duration

    # 角色2: 3图片共享音频4时长，视频使用音频5
    char2 = group_data[1]
    char2_audio_duration = audio_durations[3]

    if len(char2['image_files']) >= 3:
        per_image_duration = char2_audio_duration / 3
        for i in range(3):
            img_clip = ImageClip(char2['image_files'][i], duration=per_image_duration)
            img_clip = img_clip.resize(config['output_resolution'])
            img_clip = apply_effects(img_clip)
            clips.append(img_clip)
            current_time += per_image_duration

    # 视频2使用音频5
    if len(char2['video_files']) > 0:
        video_clip = VideoFileClip(char2['video_files'][0])
        original_duration = video_clip.duration
        target_duration = audio_durations[4]
        speed_factor = original_duration / target_duration
        video_clip = video_clip.fx(vfx.speedx, speed_factor)
        video_clip = video_clip.resize(config['output_resolution'])
        video_clip = apply_video_effects(video_clip)
        clips.append(video_clip)
        current_time += target_duration

    # 角色3: 3图片共享音频6时长，视频使用音频7
    char3 = group_data[2]
    char3_audio_duration = audio_durations[5]

    if len(char3['image_files']) >= 3:
        per_image_duration = char3_audio_duration / 3
        for i in range(3):
            img_clip = ImageClip(char3['image_files'][i], duration=per_image_duration)
            img_clip = img_clip.resize(config['output_resolution'])
            img_clip = apply_effects(img_clip)
            clips.append(img_clip)
            current_time += per_image_duration

    # 视频3使用音频7
    if len(char3['video_files']) > 0:
        video_clip = VideoFileClip(char3['video_files'][0])
        original_duration = video_clip.duration
        target_duration = audio_durations[6]
        speed_factor = original_duration / target_duration
        video_clip = video_clip.fx(vfx.speedx, speed_factor)
        video_clip = video_clip.resize(config['output_resolution'])
        video_clip = apply_video_effects(video_clip)
        clips.append(video_clip)
        current_time += target_duration

    # 合并视频片段
    final_video = concatenate_videoclips(clips, method="compose")

    # 合并音频（严格对齐）
    aligned_audio_clips = []
    current_audio_time = 0

    # 角色1音频（音频1+2用于图片，音频3用于视频）
    aligned_audio_clips.append(audio_clips[0].set_start(current_audio_time))
    current_audio_time += audio_durations[0]
    aligned_audio_clips.append(audio_clips[1].set_start(current_audio_time))
    current_audio_time += audio_durations[1]
    aligned_audio_clips.append(audio_clips[2].set_start(current_audio_time))
    current_audio_time += audio_durations[2]

    # 角色2音频（音频4用于图片，音频5用于视频）
    aligned_audio_clips.append(audio_clips[3].set_start(current_audio_time))
    current_audio_time += audio_durations[3]
    aligned_audio_clips.append(audio_clips[4].set_start(current_audio_time))
    current_audio_time += audio_durations[4]

    # 角色3音频（音频6用于图片，音频7用于视频）
    aligned_audio_clips.append(audio_clips[5].set_start(current_audio_time))
    current_audio_time += audio_durations[5]
    aligned_audio_clips.append(audio_clips[6].set_start(current_audio_time))

    final_audio = CompositeAudioClip(aligned_audio_clips)
    final_video.audio = final_audio

    # 写入输出文件（不带字幕）
    final_video.write_videofile(base_output_path, fps=24, codec='libx264', audio_codec='aac')

def convert_srt_time_to_seconds(time_str):
    """将SRT时间格式转换为秒"""
    h, m, s = time_str.split(':')
    s, ms = s.split(',')
    return int(h)*3600 + int(m)*60 + int(s) + int(ms)/1000

def generate_videos(gender, num_groups, output_dir='output'):
    """生成指定数量的视频"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取角色组合和资源
    character_groups = get_character_resources(gender, num_groups)

    #print(character_groups)

    # 为每个组合创建视频
    for i, group in enumerate(character_groups):
        output_path = os.path.join(output_dir, f'{gender}_group_{i+1}.mp4')
        print(f"正在生成视频: {output_path}")
        print(f"角色组合: {[char['name'] for char in group]}")

        try:
            create_video_clip(group, output_path)
            print(f"成功生成: {output_path}")
        except Exception as e:
            print(f"生成视频失败: {e}")
            continue

generate_videos('女', 2, output_dir = "Genshin_Impact_Girls_prefer_you_over_OTHERS")

generate_videos('男', 2, output_dir = "Genshin_Impact_Boys_prefer_you_over_OTHERS")


import os
from pathlib import Path
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
from moviepy.video.tools.subtitles import SubtitlesClip

def add_subtitles_with_moviepy(input_path, output_path, font="simhei.ttf", font_size=54, color="white", bg_color="black", stroke_color="black", stroke_width=1):
    """
    使用 moviepy 将 SRT 字幕添加到 MP4 视频底部，并输出到指定文件夹。

    参数:
        input_path (str): 输入文件夹路径（包含 .mp4 和 .srt 文件）
        output_path (str): 输出文件夹路径
        font (str): 字体（默认 "Arial"）
        font_size (int): 字体大小（默认 24）
        color (str): 字体颜色（默认 "white"）
        bg_color (str): 背景颜色（默认 "black"）
        stroke_color (str): 描边颜色（默认 "black"）
        stroke_width (int): 描边宽度（默认 1）
    """
    # 确保输出目录存在
    Path(output_path).mkdir(parents=True, exist_ok=True)

    # 扫描输入目录下的 .mp4 和 .srt 文件
    for root, _, files in os.walk(input_path):
        for file in files:
            if file.lower().endswith(".mp4"):
                video_path = os.path.join(root, file)
                base_name = os.path.splitext(file)[0]
                srt_path = os.path.join(root, f"{base_name}.srt")

                # 检查是否存在对应的 .srt 文件
                if not os.path.exists(srt_path):
                    print(f"⚠️ 未找到 {base_name}.srt 字幕文件，跳过 {file}")
                    continue

                # 加载视频
                video = VideoFileClip(video_path)

                '''
                # 检查分辨率是否为 1024x1024
                if video.size != (1024, 1024):
                    print(f"⚠️ {file} 不是 1024x1024 分辨率，跳过")
                    video.close()
                    continue
                '''

                # 加载字幕（使用 moviepy 的 SubtitlesClip）
                def make_text(txt):
                    """生成字幕样式"""

                    text_clip = TextClip(
                        txt,
                        font=font,
                        fontsize=font_size,
                        color=color,
                        bg_color=bg_color,
                        stroke_color=stroke_color,
                        stroke_width=stroke_width,
                        size=(video.w, None),  # 宽度与视频相同，高度自动调整
                        method="caption",  # 自动换行
                        align="center",  # 水平居中
                    )

                    # 计算字幕位置（距离底部 margin 像素）
                    margin = 130  # 调整这个值来控制与底部的距离
                    text_position = ("center", video.h - text_clip.h - margin)

                    # 设置位置
                    text_clip = text_clip.set_position(text_position)
                    return text_clip

                subtitles = SubtitlesClip(srt_path, make_text)

                # 合成视频+字幕
                final_video = CompositeVideoClip([video, subtitles.set_position(("center", "bottom"))])

                # 输出文件路径
                output_file = os.path.join(output_path, f"{base_name}_subtitled.mp4")

                # 写入输出文件
                final_video.write_videofile(
                    output_file,
                    codec="libx264",  # H.264 编码
                    audio_codec="aac",  # 保持原音频
                    fps=video.fps,  # 保持原帧率
                    threads=4,  # 多线程加速
                )

                print(f"✅ 字幕已添加：{output_file}")

                # 释放资源
                video.close()
                final_video.close()

# 示例调用
if __name__ == "__main__":
    input_dir = "Genshin_Impact_Girls_prefer_you_over_OTHERS"  # 替换为你的输入路径
    output_dir = "Genshin_Impact_Girls_prefer_you_over_OTHERS_Subtitled"  # 替换为你的输出路径
    add_subtitles_with_moviepy(input_dir, output_dir)


# 示例调用
if __name__ == "__main__":
    input_dir = "Genshin_Impact_Boys_prefer_you_over_OTHERS"  # 替换为你的输入路径
    output_dir = "Genshin_Impact_Boys_prefer_you_over_OTHERS_Subtitled"  # 替换为你的输出路径
    add_subtitles_with_moviepy(input_dir, output_dir)

```
