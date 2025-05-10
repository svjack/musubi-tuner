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

import os
import shutil

# Define paths
source_dir = "genshin_impact_KAEDEHARA_KAZUHA_images_and_texts_PreProcess"
target_dir = "genshin_impact_KAEDEHARA_KAZUHA_Omni_Captioned"

# Create target directory if it doesn't exist
os.makedirs(target_dir, exist_ok=True)

# Iterate through files in source directory
for filename in os.listdir(source_dir):
    base_name, ext = os.path.splitext(filename)

    # Process .mp4 files (copy directly)
    if ext.lower() == '.png':
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(target_dir, filename)
        shutil.copy2(src_path, dst_path)
        print(f"Copied: {filename}")

    # Process .txt files (modify content)
    elif ext.lower() == '.txt':
        src_path = os.path.join(source_dir, filename)
        dst_path = os.path.join(target_dir, filename)

        # Read and process the text file
        with open(src_path, 'r', encoding='utf-8') as f:
            content = f.read()

        # Apply the text processing
        processed_content = content.split("assistant")[-1].strip().replace("女", "男").replace("她", "他")

        # Write the processed content to new file
        with open(dst_path, 'w', encoding='utf-8') as f:
            f.write(processed_content)

        print(f"Processed: {filename}")

print("Operation completed successfully!")

#### 散兵 1024

from datasets import load_dataset
from PIL import Image, ImageOps

def resize_and_pad(image, target_size=(1024, 1024)):
    # 计算原始图像的宽高比
    width, height = image.size
    target_width, target_height = target_size
    ratio = min(target_width / width, target_height / height)

    # 等比例缩放图像
    new_size = (int(width * ratio), int(height * ratio))
    resized_image = image.resize(new_size)

    # 创建一个新的黑色背景图像
    new_image = Image.new("RGB", target_size, (0, 0, 0))

    # 将缩放后的图像粘贴到新图像的中心
    new_image.paste(resized_image, ((target_width - new_size[0]) // 2, (target_height - new_size[1]) // 2))

    return new_image

# 加载数据集
ds = load_dataset("svjack/Genshin_Impact_Scaramouche_Images_Captioned")

# 对数据集中的 image 列进行处理
def process_example(example):
    example['image'] = resize_and_pad(example['image'])
    return example

# 应用处理函数到整个数据集
ds = ds.map(process_example)

ds = ds["train"]
import os
from uuid import uuid1
os.makedirs("Genshin_Impact_Scaramouche_Images_Captioned_Local")

for ele in ds:
  uuid_ = uuid1()
  im_name = os.path.join("Genshin_Impact_Scaramouche_Images_Captioned_Local", "{}.png".format(uuid_))
  txt_name = os.path.join("Genshin_Impact_Scaramouche_Images_Captioned_Local", "{}.txt".format(uuid_))
  ele["image"].save(im_name)
  with open(txt_name, "w") as f:
    f.write(ele["joy-caption"])

#### 万叶 1024
huggingface-cli download svjack/Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-SP --include="genshin_impact_KAEDEHARA_KAZUHA_images_and_texts/*" --local-dir . --repo-type dataset

#### 散兵x万叶 1024

vim run_couple.py

import torch
import os
from diffusers import StableDiffusionXLPipeline
from diffusers.utils import load_image
import random

# Function to sanitize filenames
def sanitize_filename(text):
    keepcharacters = (' ','.','_')
    return "".join(c for c in text if c.isalnum() or c in keepcharacters).rstrip().replace(" ", "_")

# Initialize the pipeline
pipe = StableDiffusionXLPipeline.from_pretrained(
    "cagliostrolab/animagine-xl-4.0",
    torch_dtype=torch.float16,
    use_safetensors=True,
).to("cuda")

# Define function to generate prompt
def gen_two_person_prompt(name1, name2, action=""):
    return f"COUPLE, {name1}, {name2} (genshin impact) highres, masterpiece, {action}"

# Define negative prompt
negative_prompt = "nsfw,lowres,(bad),text,error,fewer,extra,missing,worst quality,jpeg artifacts,low quality,watermark,unfinished,displeasing,oldest,early,chromatic aberration,signature,extra digits,artistic error,username,scan,[abstract],"

def generate_and_save_image(pipeline, prompt, negative_prompt, seed, save_dir="KAEDEHARA_KAZUHA_X_Scaramouche_Images_Captioned", index=None):
    os.makedirs(save_dir, exist_ok=True)

    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        generator=torch.manual_seed(seed),
    ).images[0]

    # Create filename with index prefix for ordering
    filename = f"{index:03d}_{sanitize_filename(prompt)}_seed_{seed}.png"
    save_path = os.path.join(save_dir, filename)

    image.save(save_path)
    print(f"Generated and saved: {save_path}")

# Set names
name1 = "KAEDEHARA_KAZUHA"
name2 = "Scaramouche"

# Infinite loop to generate images
index = 0
while True:
    # Generate random seed
    seed = random.randint(0, 2**32 - 1)

    # Generate prompt without action
    prompt = gen_two_person_prompt(name1, name2)

    # Generate and save image
    generate_and_save_image(pipe, prompt, negative_prompt, seed, index=index)

    index += 1


from datasets import Dataset, DatasetDict, Image, load_dataset
import os
from PIL import Image as PILImage

# 1. Load the existing dataset
ds = load_dataset("svjack/Day_if_sentient_beings_SPLITED_BY_XIAO_IM_SIGN_DEPTH_TEXT_47")
existing_train = ds["train"]

# 2. Create a new dataset from the images
image_dir = "KAEDEHARA_KAZUHA"
image_files = [f for f in os.listdir(image_dir) if f.startswith("kazuha_") and f.endswith(".png")]

def create_image_dataset(image_files, image_dir):
    data = []
    for img_file in image_files:
        # Extract index from filename (format: kazuha_{group}_{index}_date_time.png)
        parts = img_file.split("_")
        group = parts[1]  # e.g., "1", "2", "3"
        index = int(parts[2])

        '''
        # Only use files from group "1" to match existing dataset indices
        if group != "1":
            continue
        '''

        # Get corresponding features from existing dataset
        if index < len(existing_train):
            features = {
                "sign_mask": existing_train[index]["sign_mask"],
                "depth": existing_train[index]["depth"],
                "mask_image": existing_train[index]["mask_image"]
            }
        else:
            # Handle case where index is out of bounds
            features = {
                "sign_mask": None,
                "depth": None,
                "mask_image": None
            }

        # Add image path and features to dataset
        data.append({
            "image": os.path.join(image_dir, img_file),
            **features
        })

    return Dataset.from_list(data)

# Create the dataset
image_dataset = create_image_dataset(image_files, image_dir)

# Cast the image column to Image type
image_dataset = image_dataset.cast_column("image", Image())

# You can now use image_dataset as your new dataset
# Optionally save it to the hub:
# image_dataset.push_to_hub("your-username/your-dataset-name")

image_dataset.push_to_hub("svjack/KAEDEHARA_KAZUHA_IM_SIGN_DEPTH_TEXT")

#### card caption

cd joy-caption-alpha-two
python caption_generator_name_ds_save_interval.py "svjack/KAEDEHARA_KAZUHA_IM_SIGN_DEPTH_TEXT" \
    --caption_column="joy-caption" \
    --output_path="KAEDEHARA_KAZUHA_IM_SIGN_DEPTH_TEXT" \
    --caption_type="Descriptive" \
    --caption_length="long" \
    --extra_options 0 1 8 \
    --save_interval 3000

python caption_generator_name_ds_save_interval.py "svjack/Scaramouche_IM_SIGN_DEPTH_TEXT" \
    --caption_column="joy-caption" \
    --output_path="Scaramouche_IM_SIGN_DEPTH_TEXT" \
    --caption_type="Descriptive" \
    --caption_length="long" \
    --extra_options 0 1 8 \
    --save_interval 3000

#### couple caption

python caption_generator_name_ds_save_interval.py "svjack/Genshin_Impact_KAEDEHARA_KAZUHA_X_Scaramouche" \
    --caption_column="joy-caption" \
    --output_path="Genshin_Impact_KAEDEHARA_KAZUHA_X_Scaramouche" \
    --caption_type="Descriptive" \
    --caption_length="long" \
    --extra_options 0 1 8 \
    --save_interval 3000

#### merge different Source Data

genshin_impact_KAEDEHARA_KAZUHA_images_and_texts

from datasets import load_dataset
from PIL import Image
import os
from uuid import uuid1

def resize_and_pad(image, target_size=(1024, 1024)):
    # 计算原始图像的宽高比
    width, height = image.size
    target_width, target_height = target_size
    ratio = min(target_width / width, target_height / height)

    # 等比例缩放图像
    new_size = (int(width * ratio), int(height * ratio))
    resized_image = image.resize(new_size)

    # 创建一个新的黑色背景图像
    new_image = Image.new("RGB", target_size, (0, 0, 0))

    # 将缩放后的图像粘贴到新图像的中心
    new_image.paste(resized_image, ((target_width - new_size[0]) // 2, (target_height - new_size[1]) // 2))

    return new_image

def process_dataset(dataset_name, output_dir, caption_key="joy-caption"):
    # 加载数据集
    ds = load_dataset(dataset_name)

    # 对数据集中的 image 列进行处理
    def process_example(example):
        example['image'] = resize_and_pad(example['image'])
        # 移除特定字符串
        if caption_key in example:
            example[caption_key] = example[caption_key].replace("In the style of Scaramouche ,", "").strip()
        return example

    # 应用处理函数到整个数据集
    ds = ds.map(process_example)

    ds = ds["train"]

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 保存处理后的图像和文本
    for ele in ds:
        uuid_ = uuid1()
        im_name = os.path.join(output_dir, f"{uuid_}.png")
        txt_name = os.path.join(output_dir, f"{uuid_}.txt")

        # 保存图像
        ele["image"].save(im_name)

        # 保存文本
        if caption_key in ele:
            with open(txt_name, "w") as f:
                f.write(ele[caption_key])

# 定义要处理的数据集列表
datasets = [
    ("svjack/Genshin_Impact_Scaramouche_Images_Captioned", "Genshin_Impact_Scaramouche_Images_Captioned_Local"),
    ("svjack/Genshin_Impact_KAEDEHARA_KAZUHA_X_Scaramouche_CAPTION", "Genshin_Impact_KAEDEHARA_KAZUHA_X_Scaramouche_CAPTION_Local"),
    ("svjack/Scaramouche_IM_SIGN_DEPTH_TEXT_CAPTION", "Scaramouche_IM_SIGN_DEPTH_TEXT_CAPTION_Local"),
    ("svjack/KAEDEHARA_KAZUHA_IM_SIGN_DEPTH_TEXT_CAPTION", "KAEDEHARA_KAZUHA_IM_SIGN_DEPTH_TEXT_CAPTION_Local")
]

# 处理所有数据集
for dataset_name, output_dir in datasets:
    print(f"Processing dataset: {dataset_name}")
    process_dataset(dataset_name, output_dir)
    print(f"Finished processing {dataset_name}. Output saved to {output_dir}")

cp -r ../genshin_impact_KAEDEHARA_KAZUHA_images_and_texts genshin_impact_KAEDEHARA_KAZUHA_images_and_texts_Local

import os
import shutil
import uuid
from pathlib import Path

# 输入结构和路径映射
structure_mapping = {
    "In the style of SCARAMOUCHE ,": "Genshin_Impact_Scaramouche_Images_Captioned_Local",
    "In the style of KAEDEHARA_KAZUHA ,": "genshin_impact_KAEDEHARA_KAZUHA_images_and_texts_Local",
    "In the style of SCARAMOUCHE KAEDEHARA_KAZUHA ,": "Genshin_Impact_KAEDEHARA_KAZUHA_X_Scaramouche_CAPTION_Local",
    "In the style of SCARAMOUCHE SIGN ,": "Scaramouche_IM_SIGN_DEPTH_TEXT_CAPTION_Local",
    "In the style of KAEDEHARA_KAZUHA SIGN ,": "KAEDEHARA_KAZUHA_IM_SIGN_DEPTH_TEXT_CAPTION_Local"
}

# 目标目录
target_dir = "Genshin_Impact_KAEDEHARA_KAZUHA_Scaramouche_SIGN_CAPTION"

def process_files():
    # 创建目标目录
    os.makedirs(target_dir, exist_ok=True)

    for prefix, source_dir in structure_mapping.items():
        # 确保源目录存在
        if not os.path.exists(source_dir):
            print(f"Warning: Source directory {source_dir} does not exist, skipping...")
            continue

        # 遍历源目录中的文件
        for filename in os.listdir(source_dir):
            base_name, ext = os.path.splitext(filename)

            # 只处理成对的文件（.png 和对应的 .txt）
            if ext.lower() == '.png':
                txt_file = f"{base_name}.txt"
                png_path = os.path.join(source_dir, filename)
                txt_path = os.path.join(source_dir, txt_file)

                # 检查对应的txt文件是否存在
                if not os.path.exists(txt_path):
                    print(f"Warning: Missing text file for {filename}, skipping...")
                    continue

                # 生成新的UUID文件名
                new_uuid = str(uuid.uuid4())
                new_png_name = f"{new_uuid}.png"
                new_txt_name = f"{new_uuid}.txt"

                # 创建目标路径
                new_png_path = os.path.join(target_dir, new_png_name)
                new_txt_path = os.path.join(target_dir, new_txt_name)

                # 拷贝png文件
                shutil.copy2(png_path, new_png_path)

                # 处理txt文件：添加前缀并拷贝
                with open(txt_path, 'r', encoding='utf-8') as f:
                    original_content = f.read()

                new_content = f"{prefix}\n{original_content}"

                with open(new_txt_path, 'w', encoding='utf-8') as f:
                    f.write(new_content)

                print(f"Copied and renamed: {filename} -> {new_png_name}")
                print(f"Processed text: {txt_file} -> {new_txt_name}")

if __name__ == "__main__":
    process_files()
    print("File processing completed!")



python wan_cache_latents.py --dataset_config image.toml --vae Wan2.1_VAE.pth

python wan_cache_text_encoder_outputs.py --dataset_config image.toml --t5 models_t5_umt5-xxl-enc-bf16.pth --batch_size 16

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 wan_train_network.py \
    --task t2v-1.3B --t5 models_t5_umt5-xxl-enc-bf16.pth \
    --dit wan2.1_t2v_1.3B_bf16.safetensors \
    --dataset_config image.toml --sdpa --mixed_precision bf16 --fp8_base \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_wan --network_dim 32 \
    --timestep_sampling shift --discrete_flow_shift 3.0 \
    --max_train_epochs 500 --save_every_n_epochs 1 --seed 42 \
    --output_dir KAEDEHARA_KAZUHA_X_Scaramouche_outputs --output_name KAEDEHARA_KAZUHA_X_Scaramouche_w1_3_lora

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 wan_train_network.py \
    --task t2v-14B --t5 models_t5_umt5-xxl-enc-bf16.pth \
    --dit wan2.1_t2v_14B_bf16.safetensors \
    --dataset_config image.toml --sdpa --mixed_precision bf16 --fp8_base \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_wan --network_dim 32 \
    --timestep_sampling shift --discrete_flow_shift 3.0 \
    --max_train_epochs 500 --save_every_n_epochs 1 --seed 42 \
    --output_dir KAEDEHARA_KAZUHA_X_Scaramouche_w14_outputs --output_name KAEDEHARA_KAZUHA_X_Scaramouche_w14_lora


git clone https://github.com/ostris/ai-toolkit.git
cd ai-toolkit
git submodule update --init --recursive
# install torch first
pip install --no-cache-dir torch==2.6.0 torchvision==0.21.0
pip install -r requirements.txt
pip install datasets
pip install hf_xet

edit os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0" in run.py

cp config/examples/train_lora_chroma_24gb.yaml config

python run.py config/train_lora_chroma_24gb.yaml
