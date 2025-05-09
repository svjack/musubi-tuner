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

vim run_9_16.py

from datasets import load_dataset
from gradio_client import Client, handle_file
import os
from PIL import Image
import tempfile
from tqdm import tqdm

# Load the dataset
ds = load_dataset("svjack/Genshin_Impact_XIAO_VENTI_Images")

# Initialize Gradio client
client = Client("http://localhost:7860")

# Create output directory if it doesn't exist
output_dir = "Genshin_Impact_XIAO_VENTI_Images_9_16"
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
            width=720,
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
