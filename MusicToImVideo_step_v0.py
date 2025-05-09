#### sdxl inpainting

sudo apt-get update && sudo apt-get install git-lfs ffmpeg cbm

pip install diffusers transformers peft torch torchvision datasets opencv-python controlnet_aux

import torch
from diffusers import StableDiffusionXLInpaintPipeline
from diffusers.utils import load_image

# Initialize the pipeline
pipe = StableDiffusionXLInpaintPipeline.from_pretrained(
    "cagliostrolab/animagine-xl-4.0",
    torch_dtype=torch.float16,
    #variant="fp16",
    use_safetensors=True,
)
pipe.to("cuda")

# Load images
img_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo.png"
mask_url = "https://raw.githubusercontent.com/CompVis/latent-diffusion/main/data/inpainting_examples/overture-creations-5sI6fQgYIuo_mask.png"

init_image = load_image(img_url).convert("RGB")
mask_image = load_image(mask_url).convert("RGB")

# Generate image
prompt = "A majestic tiger sitting on a bench"
image = pipe(
    prompt=prompt,
    image=init_image,
    mask_image=mask_image,
    num_inference_steps=50,
    strength=0.80
).images[0]

from datasets import load_dataset
ds = load_dataset("svjack/Day_if_sentient_beings_SPLITED_BY_XIAO_IM_SIGN_DEPTH_TEXT_47")

ds["train"][0]["mask_image"]

# Generate image
prompt = "Xiao hold a blank sign. genshin impact."
image = pipe(
    prompt=prompt,
    image=ds["train"][0]["original_image"].convert("RGB"),
    mask_image=ds["train"][0]["mask_image"],
    num_inference_steps=50,
    strength=0.5
).images[0]

image.resize((512, 512))

#################################################################################
#### depth controlnet

git clone https://github.com/huggingface/diffusers

vim run_two_character.py

import os
import time
from datetime import datetime

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import sys
sys.path.insert(0, "diffusers")

import cv2
import numpy as np
import torch
from controlnet_aux.midas import MidasDetector
from PIL import Image

from diffusers import AutoencoderKL, ControlNetModel, MultiAdapter, T2IAdapter
from diffusers.pipelines.controlnet.multicontrolnet import MultiControlNetModel
from diffusers.utils import load_image
from examples.community.pipeline_stable_diffusion_xl_controlnet_adapter import (
    StableDiffusionXLControlNetAdapterPipeline,
)

from datasets import load_dataset

# Create output directories if they don't exist
os.makedirs("KAEDEHARA_KAZUHA", exist_ok=True)
os.makedirs("Scaramouche", exist_ok=True)

# Load dataset
ds = load_dataset("svjack/Day_if_sentient_beings_SPLITED_BY_XIAO_IM_SIGN_DEPTH_TEXT_47")

# Initialize models
controlnet_depth = ControlNetModel.from_pretrained(
    "diffusers/controlnet-depth-sdxl-1.0",
    torch_dtype=torch.float16,
    variant="fp16",
    use_safetensors=True
)
adapter_depth = T2IAdapter.from_pretrained(
    "TencentARC/t2i-adapter-depth-midas-sdxl-1.0", torch_dtype=torch.float16, variant="fp16", use_safetensors=True
)
vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16, use_safetensors=True)

pipe = StableDiffusionXLControlNetAdapterPipeline.from_pretrained(
    "cagliostrolab/animagine-xl-4.0",
    controlnet=controlnet_depth,
    adapter=adapter_depth,
    vae=vae,
    use_safetensors=True,
    torch_dtype=torch.float16,
)
pipe = pipe.to("cuda")

midas_depth = MidasDetector.from_pretrained(
    "valhalla/t2iadapter-aux-models", filename="dpt_large_384.pt", model_type="dpt_large"
).to("cuda")

def gen_one_person_prompt(name, action):
    return f"SOLO, {name}, {action}, masterpiece, genshin impact style"

# Counter for unique filenames across loops
loop_counter = 0

# Infinite loop
while True:
    loop_counter += 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    for idx, item in enumerate(ds["train"]):
        try:
            # Process for KAEDEHARA KAZUHA
            prompt = gen_one_person_prompt("KAEDEHARA KAZUHA", "hold a blank sign")
            image = item["original_image"]

            depth_image = midas_depth(
                image, detect_resolution=512, image_resolution=1024
            )

            images = pipe(
                prompt,
                control_image=depth_image,
                adapter_image=depth_image,
                num_inference_steps=50,
                controlnet_conditioning_scale=0.5,
                adapter_conditioning_scale=0.1,
            ).images

            # Save with loop counter and timestamp to prevent overwrites
            output_path = os.path.join("KAEDEHARA_KAZUHA", f"kazuha_{loop_counter}_{idx}_{timestamp}.png")
            images[0].save(output_path)

            # Process for Scaramouche
            prompt = gen_one_person_prompt("Scaramouche", "hold a blank sign")
            image = item["original_image"]

            depth_image = midas_depth(
                image, detect_resolution=512, image_resolution=1024
            )

            images = pipe(
                prompt,
                control_image=depth_image,
                adapter_image=depth_image,
                num_inference_steps=50,
                controlnet_conditioning_scale=0.5,
                adapter_conditioning_scale=0.1,
            ).images

            # Save with loop counter and timestamp to prevent overwrites
            output_path = os.path.join("Scaramouche", f"scaramouche_{loop_counter}_{idx}_{timestamp}.png")
            images[0].save(output_path)

            print(f"Processed and saved images for loop {loop_counter}, index {idx}")

        except Exception as e:
            print(f"Error processing loop {loop_counter}, index {idx}: {str(e)}")
            continue


1、音乐源

先使用歌词搜索功能 搜索歌曲 和对应的 srt 文件
https://github.com/jitwxs/163MusicLyrics

查看歌曲是否可下载
https://github.com/gengark/netease-cloud-music-download

wyy dl "https://music.163.com/song?id=1941990933"

wyy dl "https://music.163.com/song?id=1465225525"

从而得到 对应的
.mp3 文件和 对应的 srt 文件

2、分割

位于某文件夹下 的同名 (.mp3, .srt) 文件对儿

分割单个 对儿 数据集格式的代码

'''
python run_srt_split.py "天若有情 - 杜宣达.mp3" "天若有情 - 杜宣达.srt" "天若有情"

python run_srt_split.py "明天过后 - 张杰.mp3" "明天过后 - 张杰.srt" "明天过后"
'''

import os
import re
import sys
from pydub import AudioSegment
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

def split_audio(audio_path, srt_path, output_path=None):
    """Split audio file into segments based on SRT timings"""
    try:
        # Read SRT file
        with open(srt_path, 'r', encoding='utf-8') as f:
            srt_content = f.read()

        # Parse SRT segments
        segments = parse_srt(srt_content)
        if not segments:
            print("No valid segments to process")
            return False

        # Load audio file
        audio = AudioSegment.from_file(audio_path)
        audio_duration = len(audio) / 1000  # pydub uses milliseconds

        # Create output directory if not specified
        if output_path is None:
            base_name = os.path.splitext(os.path.basename(audio_path))[0]
            output_path = f"output_{sanitize_filename(base_name)}"

        os.makedirs(output_path, exist_ok=True)

        # Process each segment
        for idx, start_time, end_time, text in segments:
            # Convert to milliseconds and validate against audio duration
            start_ms = int(start_time * 1000)
            end_ms = int(end_time * 1000)

            if start_ms >= len(audio):
                print(f"Skipping segment {idx} - Start time beyond audio duration")
                continue

            if end_ms > len(audio):
                end_ms = len(audio)
                print(f"Adjusting segment {idx} end time to {end_ms/1000}s")

            segment = audio[start_ms:end_ms]

            # Create output filenames
            base_name = f"{idx:04d}_{sanitize_filename(os.path.splitext(os.path.basename(audio_path))[0])}"
            audio_file = os.path.join(output_path, f"{base_name}.mp3")
            text_file = os.path.join(output_path, f"{base_name}.txt")

            # Export segment as MP3
            segment.export(audio_file, format='mp3', bitrate="192k")

            # Save corresponding text
            with open(text_file, 'w', encoding='utf-8') as f:
                f.write(text)

        return True

    except Exception as e:
        print(f"Error processing {audio_path}: {str(e)}")
        return False

def main():
    if len(sys.argv) < 3:
        print("Usage: python script.py <audio_file.mp3|.wav> <srt_file.srt> [output_directory]")
        sys.exit(1)

    audio_path = sys.argv[1]
    srt_path = sys.argv[2]
    output_path = sys.argv[3] if len(sys.argv) > 3 else None

    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found - {audio_path}")
        sys.exit(1)

    if not os.path.exists(srt_path):
        print(f"Error: SRT file not found - {srt_path}")
        sys.exit(1)

    print(f"\nProcessing audio: {audio_path}")
    print(f"Using SRT file: {srt_path}")
    if output_path:
        print(f"Output directory: {output_path}")
    else:
        print("Output directory: auto-generated")

    success = split_audio(
        audio_path=audio_path,
        srt_path=srt_path,
        output_path=output_path
    )

    if success:
        print("✓ Successfully processed")
    else:
        print("✗ Processing failed")

if __name__ == "__main__":
    main()

huggingface-cli download svjack/Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-SP --include="genshin_impact_KAEDEHARA_KAZUHA_images_and_texts/*" --local-dir . --repo-type dataset

git clone https://huggingface.co/datasets/svjack/Genshin_Impact_Scaramouche_Images

git clone https://huggingface.co/spaces/svjack/ReSize-Image-Outpainting && cd ReSize-Image-Outpainting
pip uninstall fastapi -y
pip install -r requirements.txt
python app.py

vim run_1024_1024.py

from datasets import load_dataset
from gradio_client import Client, handle_file
import os
from PIL import Image
import tempfile
from tqdm import tqdm

# Load the dataset
ds = load_dataset("svjack/Genshin_Impact_Scaramouche_Images")

# Initialize Gradio client
client = Client("http://localhost:7860")

# Create output directory if it doesn't exist
output_dir = "Genshin_Impact_Scaramouche_Images_1024x1024"
os.makedirs(output_dir, exist_ok=True)

# Determine the number of digits needed for padding
total_items = len(ds["train"])
padding_length = len(str(total_items))  # This ensures all filenames have the same length

# Iterate through all items in the training set
for idx, item in tqdm(enumerate(ds["train"])):
    try:
        image = item["image"]
        #joy_caption = item["joy-caption"]

        # Create a temporary file for the input image
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as temp_file:
            temp_image_path = temp_file.name
            image.save(temp_image_path)

        # Process the image through the API
        result = client.predict(
            image=handle_file(temp_image_path),
            width=1024,
            height=1024,
            overlap_percentage=10,
            num_inference_steps=8,
            resize_option="Full",
            custom_resize_percentage=50,
            prompt_input="",
            alignment="Middle",
            overlap_left=True,
            overlap_right=True,
            overlap_top=True,
            overlap_bottom=True,
            api_name="/infer"
        )

        # Get the processed image path from the result
        processed_image_path = result[1]

        # Define output paths with zero-padded index
        padded_idx = str(idx).zfill(padding_length)
        base_filename = f"processed_{padded_idx}"
        output_image_path = os.path.join(output_dir, f"{base_filename}.png")
        #output_text_path = os.path.join(output_dir, f"{base_filename}.txt")

        # Ensure the output is saved as PNG
        if processed_image_path.lower().endswith('.png'):
            # If already PNG, just copy
            with Image.open(processed_image_path) as img:
                img.save(output_image_path, 'PNG')
        else:
            # If not PNG, open and convert to PNG
            with Image.open(processed_image_path) as img:
                img.save(output_image_path, 'PNG')

        '''
        # Save the joy-caption as a text file
        with open(output_text_path, 'w', encoding='utf-8') as f:
            f.write(joy_caption)
        '''

        print(f"Processed item {idx}: Image saved to {output_image_path}")

    except Exception as e:
        print(f"Error processing item {idx}: {str(e)}")
    finally:
        # Clean up temporary files
        if 'temp_image_path' in locals() and os.path.exists(temp_image_path):
            os.remove(temp_image_path)

print("Processing complete!")


import os
import shutil
from tqdm import tqdm
import soundfile as sf
import torch
from moviepy.editor import VideoFileClip
from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor
from qwen_omni_utils import process_mm_info

# Load model and processor
print("Loading model and processor...")
model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
    "Qwen/Qwen2.5-Omni-3B",
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2",
)
processor = Qwen2_5OmniProcessor.from_pretrained("Qwen/Qwen2.5-Omni-3B")
USE_AUDIO_IN_VIDEO = False

# System prompt
'''
system_prompt = {
    "role": "system",
    "content": [
        {"type": "text", "text": "你是一个Video Captioner,根据我给你的视频生成对应的中文 Caption。不要回复其他内容，也不要进行其他询问。"}
    ],
}
'''

system_text = "你是一个专注于日本动漫的智能Caption生成器，请按以下要求制作中文视频描述：\n" + \
                    "1. 【人物特征】精确描述：发色渐变/瞳孔纹样/服装细节（如『左肩破损的黑色学生制服』『闪烁星芒的碧绿蛇瞳』）\n" + \
                    "2. 【景物特征】动态捕捉：天气变化（『雨滴在刀锋上碎裂』）、光影效果（『夕阳将和室拉出三道渐变阴影』）\n" + \
                    "3. 【动作事件】逐帧解析：战斗动作（『太刀反手居合时刀鞘迸出火星』）、微表情变化（『说话时右眼不自然地抽搐』）\n" + \
                    "4. 【分镜语言】技术标注：推镜头（『0.5秒内从全景急推到角色颤抖的指尖』）、鱼眼变形（『背景扭曲表现精神冲击』）\n" + \
                    "5. 【美术风格】特征识别：" + \
                    "- 赛璐璐：『边缘锐利的色块与高光』" + \
                    "- 数字绘景：『多层景深合成的蒸汽都市』" + \
                    "- 特殊效果：『爆衣时飞散的晶体化布料』\n" + \
                    "6. 【单句描写】在300字以上完成包含3个以上动态细节的复合描写（如『紫绀色马尾辫随后空翻甩出虹彩残影，染血木屐踏碎水面时惊起十七枚银针状雨滴』）"

system_prompt = {
    "role": "system",
    "content": [
        {"type": "text", "text": system_text}
    ],
}

# Setup directories
input_dir = "genshin_impact_KAEDEHARA_KAZUHA_images_and_texts"
output_dir = "genshin_impact_KAEDEHARA_KAZUHA_images_and_texts_PreProcess"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

def process_video(video_path, output_dir):
    """Process a single video file and generate caption"""
    try:
        # Prepare conversation
        conversation = [
            system_prompt,
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": video_path},
                    {"type": "text", "text": "使用中文描述这个图片。"}
                ],
            },
        ]

        # Prepare inputs
        text = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=False)
        audios, images, videos = process_mm_info(conversation, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = processor(text=text, audio=audios, images=images, videos=videos,
                          return_tensors="pt", padding=True, use_audio_in_video=USE_AUDIO_IN_VIDEO)
        inputs = inputs.to(model.device).to(model.dtype)

        # Generate caption
        text_ids = model.generate(**inputs, use_audio_in_video=USE_AUDIO_IN_VIDEO, return_audio=False)
        text = processor.batch_decode(text_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return text[0]
    except Exception as e:
        print(f"Error processing {video_path}: {str(e)}")
        return None

# Get all MP4 files
print("Finding video files...")
video_files = []
for filename in tqdm(os.listdir(input_dir)):
    if filename.lower().endswith('.png'):
        filepath = os.path.join(input_dir, filename)
        video_files.append((filename, filepath))

print(f"Found {len(video_files)} potential videos")

# Process each video with progress bar
processed_count = 0
for filename, filepath in tqdm(video_files, desc="Processing videos"):
    '''
    # Check video duration first
    duration = get_video_duration(filepath)
    if duration > 30:
        continue  # Skip videos longer than 30 seconds
    '''
    # Generate caption
    caption = process_video(filepath, output_dir)

    if caption is not None:
        # Copy video file to output directory
        output_video_path = os.path.join(output_dir, filename)
        shutil.copy2(filepath, output_video_path)

        # Save caption as text file
        txt_filename = os.path.splitext(filename)[0] + '.txt'
        txt_path = os.path.join(output_dir, txt_filename)
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(caption)

        processed_count += 1

print(f"Processing complete! Processed {processed_count} videos (≤30s) out of {len(video_files)} total files.")
