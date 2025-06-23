# README for Musubi-Tuner Training and Inference

This guide provides step-by-step instructions for setting up, training, and generating videos using the Musubi-Tuner framework. The process involves installing dependencies, downloading datasets and models, training the model, and generating videos.

---

## 1. Prerequisites

Ensure you have the following installed on your system:

• **Python 3.8 or higher**
• **Git**
• **CUDA** (if using GPU)
• **pip**

---

## 2. Installation

### 2.1 Install System Dependencies

Run the following commands to install necessary system packages:

```bash
sudo apt-get update && sudo apt-get install cbm git-lfs ffmpeg
```

### 2.2 Clone the Repository

Clone the Musubi-Tuner repository:

```bash
git clone https://github.com/kohya-ss/musubi-tuner && cd musubi-tuner
```

### 2.3 Install Python Dependencies

Install the required Python packages:

```bash
pip install torch torchvision
#pip install -r requirements.txt
pip install -e .
pip install ascii-magic matplotlib tensorboard huggingface_hub datasets
pip install moviepy==1.0.3
pip install sageattention==1.0.6
```

---

## 3. Dataset Preparation

### 3.1 Download Dataset

Download the dataset from Hugging Face:

```bash
huggingface-cli download --repo-type dataset svjack/Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-SP --include "genshin_impact_RAIDEN_SHOGUN_images_and_texts/*" --local-dir .
wget https://huggingface.co/datasets/svjack/Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-SP/resolve/main/genshin_impact_RAIDEN_SHOGUN_image_config.toml
```

### 3.2 genshin_impact_RAIDEN_SHOGUN_image_config.toml
```toml
[[datasets]]
image_directory = "genshin_impact_RAIDEN_SHOGUN_images_and_texts"

[general]
resolution = [ 1024, 1024,]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false
```

---

## 4. Model Preparation

### 4.1 Download Model Files

Download the required model files:

```bash
wget https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/models_t5_umt5-xxl-enc-bf16.pth
wget https://huggingface.co/DeepBeepMeep/Wan2.1/resolve/main/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth
wget https://huggingface.co/Wan-AI/Wan2.1-T2V-14B/resolve/main/Wan2.1_VAE.pth
wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_1.3B_bf16.safetensors
wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/diffusion_models/wan2.1_t2v_14B_bf16.safetensors
```

---

## 5. Pre-Computation

### 5.1 Cache Latents

Cache the latents for the dataset:

```bash
python wan_cache_latents.py --dataset_config genshin_impact_RAIDEN_SHOGUN_image_config.toml --vae Wan2.1_VAE.pth
```

### 5.2 Cache Text Encoder Outputs

Cache the text encoder outputs:

```bash
python wan_cache_text_encoder_outputs.py --dataset_config genshin_impact_RAIDEN_SHOGUN_image_config.toml --t5 models_t5_umt5-xxl-enc-bf16.pth --batch_size 16
```

---

## 6. Training

### 6.1 Train for 1.3B Model

Train the 1.3B model:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 wan_train_network.py \
    --task t2v-1.3B --t5 models_t5_umt5-xxl-enc-bf16.pth \
    --dit wan2.1_t2v_1.3B_bf16.safetensors \
    --dataset_config genshin_impact_RAIDEN_SHOGUN_image_config.toml --sdpa --mixed_precision bf16 --fp8_base \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_wan --network_dim 32 \
    --timestep_sampling shift --discrete_flow_shift 3.0 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir RAIDEN_SHOGUN_outputs --output_name RAIDEN_SHOGUN_w1_3_lora
```

#### batch inference
```bash
import pandas as pd
text = "\n".join(pd.read_csv("Origin_Images_Captioned/metadata.csv")["prompt"].map(
    lambda x: "A dynamic anime landscape Video ," + x
).map(
    lambda x: x.replace("\n" , " ")
).head(2).values.tolist())
with open("Origin_prompt_2.txt", "w") as f:
    f.write(text)

text = "\n".join(pd.read_csv("Origin_Images_Captioned/metadata.csv")["prompt"].map(
    lambda x: "A dynamic anime landscape Video ," + x
).map(
    lambda x: x.replace("\n" , " ")
).values.tolist())
with open("Origin_prompt.txt", "w") as f:
    f.write(text)

python wan_generate_video.py --fp8 --task t2v-1.3B --video_size 480 832 --video_length 81 --infer_steps 35 \
--save_path Origin_save --output_type video \
--dit wan2.1_t2v_1.3B_bf16.safetensors --vae Wan2.1_VAE.pth \
--t5 models_t5_umt5-xxl-enc-bf16.pth \
--attn_mode torch \
--lora_weight Origin_outputs/Origin_w1_3_lora-000008.safetensors \
--lora_multiplier 1.0 \
--from_file Origin_prompt.txt


import pandas as pd
text = "\n".join(pd.read_csv("Dont_be_your_lover_Images_Captioned/metadata.csv")["prompt"].map(
    lambda x: "A dynamic anime landscape Video ," + x
).map(
    lambda x: x.replace("\n" , " ")
).head(2).values.tolist())
with open("Dont_be_your_lover_prompt_2.txt", "w") as f:
    f.write(text)

text = "\n".join(pd.read_csv("Dont_be_your_lover_Images_Captioned/metadata.csv")["prompt"].map(
    lambda x: "A dynamic anime landscape Video ," + x
).map(
    lambda x: x.replace("\n" , " ")
).values.tolist())
with open("Dont_be_your_lover_prompt.txt", "w") as f:
    f.write(text)

python wan_generate_video.py --fp8 --task t2v-1.3B --video_size 480 832 --video_length 81 --infer_steps 35 \
--save_path Dont_be_your_lover_save --output_type video \
--dit wan2.1_t2v_1.3B_bf16.safetensors --vae Wan2.1_VAE.pth \
--t5 models_t5_umt5-xxl-enc-bf16.pth \
--attn_mode torch \
--lora_weight Dont_be_your_lover_outputs/Dont_be_your_lover_w1_3_lora-000034.safetensors \
--lora_multiplier 1.0 \
--from_file Dont_be_your_lover_prompt.txt

```

### 6.2 Train for 14B Model

Train the 14B model:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 wan_train_network.py \
    --task t2v-14B --t5 models_t5_umt5-xxl-enc-bf16.pth \
    --dit wan2.1_t2v_14B_bf16.safetensors \
    --dataset_config genshin_impact_RAIDEN_SHOGUN_image_config.toml --sdpa --mixed_precision bf16 --fp8_base \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_wan --network_dim 32 \
    --timestep_sampling shift --discrete_flow_shift 3.0 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir RAIDEN_SHOGUN_w14_outputs --output_name RAIDEN_SHOGUN_w14_lora
```

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 wan_train_network.py \
    --task t2v-14B --t5 models_t5_umt5-xxl-enc-bf16.pth \
    --dit wan2.1_t2v_14B_bf16.safetensors \
    --dataset_config video_config.toml --sdpa --mixed_precision bf16 --fp8_base \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_wan --network_dim 32 \
    --timestep_sampling shift --discrete_flow_shift 3.0 \
    --max_train_epochs 5000 --save_every_n_steps 500 --seed 42 \
    --output_dir ani_landscape_w14_outputs --output_name ani_landscape_w14_lora \
    --sample_prompts landscape.txt --sample_every_n_steps 100 --sample_at_first --vae Wan2.1_VAE.pth
```

---

## 7. Inference

### 7.1 Generate Video with 1.3B Model

Generate a video using the 1.3B model:

```bash
python wan_generate_video.py --fp8 --task t2v-1.3B --video_size 832 480 --video_length 81 --infer_steps 20 \
--save_path save.mp4 --output_type both \
--dit wan2.1_t2v_1.3B_bf16.safetensors --vae Wan2.1_VAE.pth \
--t5 models_t5_umt5-xxl-enc-bf16.pth \
--attn_mode torch \
--lora_weight RAIDEN_SHOGUN_outputs/RAIDEN_SHOGUN_w1_3_lora-000002.safetensors \
--lora_multiplier 1.0 \
--prompt "In this vibrant anime-style digital artwork, RAIDEN SHOGUN, a character with long, flowing purple hair adorned with a blue flower, sits at a table in a sunlit room. She wears an elaborate, revealing outfit with a deep neckline, showing ample cleavage. In front of her is a plate of strawberries and cream, and she holds a strawberry delicately. The background features a window with a scenic view, a bottle of wine, and a cake. The atmosphere is warm and inviting, with soft lighting enhancing the cozy setting."
```

### 7.2 Generate Video with 14B Model

Generate a video using the 14B model:

```bash
python wan_generate_video.py --fp8 --task t2v-14B --video_size 832 480 --video_length 81 --infer_steps 20 \
--save_path save --output_type both \
--dit wan2.1_t2v_14B_bf16.safetensors --vae Wan2.1_VAE.pth \
--t5 models_t5_umt5-xxl-enc-bf16.pth \
--attn_mode torch \
--lora_weight RAIDEN_SHOGUN_outputs/RAIDEN_SHOGUN_w1_3_lora-000002.safetensors \
--lora_multiplier 1.0 \
--prompt "In this vibrant anime-style digital artwork, RAIDEN SHOGUN, with long, flowing purple hair adorned by a delicate blue flower, exudes serene elegance. She sits gracefully at a wooden table, her slender fingers gently stroking a pristine white cat, its emerald eyes gleaming with contentment. Warm golden light bathes the scene, highlighting her porcelain skin and intricate attire. Cherry blossom petals drift in the air, adding whimsy to the tranquil Japanese-inspired setting. Her calm yet commanding expression blends strength and tenderness, capturing a harmonious moment of quiet sophistication and emotional depth."
```

---

## 8. Additional Processing

### 8.1 Clone Pixel Art Dataset

Clone the pixel art dataset:

```bash
git clone https://huggingface.co/datasets/svjack/test-HunyuanVideo-pixelart-videos
```

### 8.2 Resize and Process Videos

Run the Python script to resize and process videos:

```python
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import shutil

def change_resolution_and_save(input_path, output_path, target_width=1024, target_height=768, max_duration=4):
    """Process images and videos to target resolution and split videos into segments."""
    os.makedirs(output_path, exist_ok=True)

    for root, dirs, files in os.walk(input_path):
        for file in tqdm(files, desc="Processing files"):
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, input_path)
            output_dir = os.path.dirname(os.path.join(output_path, relative_path))

            # Process images
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = cv2.imread(file_path)
                    h, w = img.shape[:2]
                    scale = min(target_width / w, target_height / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                    x_offset = (target_width - new_w) // 2
                    y_offset = (target_height - new_h) // 2
                    background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
                    output_file_path = os.path.join(output_path, relative_path)
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    cv2.imwrite(output_file_path, background)

                    # Copy corresponding txt file
                    base_name = os.path.splitext(file)[0]
                    txt_source = os.path.join(root, f"{base_name}.txt")
                    if os.path.exists(txt_source):
                        txt_target = os.path.join(output_dir, f"{base_name}.txt")
                        shutil.copy2(txt_source, txt_target)
                except Exception as e:
                    print(f"Failed to process image {file_path}: {e}")

            # Process videos
            elif file.lower().endswith('.mp4'):
                try:
                    clip = VideoFileClip(file_path)
                    total_duration = clip.duration
                    base_name = os.path.splitext(file)[0]

                    if total_duration <= max_duration:
                        # Process the entire video
                        output_filename = f"{base_name}.mp4"
                        output_file_path = os.path.join(output_dir, output_filename)
                        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                        def process_frame(frame):
                            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            h, w = img.shape[:2]
                            scale = min(target_width / w, target_height / h)
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                            background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                            x_offset = (target_width - new_w) // 2
                            y_offset = (target_height - new_h) // 2
                            background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
                            return cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

                        processed_clip = clip.fl_image(process_frame)
                        fps = processed_clip.fps if processed_clip.fps else 24
                        processed_clip.write_videofile(
                            output_file_path,
                            codec='libx264',
                            fps=fps,
                            preset='slow',
                            threads=4,
                            audio=False
                        )
                        processed_clip.close()

                        # Copy corresponding txt file
                        txt_source = os.path.join(root, f"{base_name}.txt")
                        if os.path.exists(txt_source):
                            txt_target = os.path.join(output_dir, f"{base_name}.txt")
                            shutil.copy2(txt_source, txt_target)
                    else:
                        # Split and process the video
                        num_segments = int(total_duration // max_duration)
                        for i in range(num_segments):
                            start_time = i * max_duration
                            end_time = min((i+1) * max_duration, total_duration)
                            sub_clip = clip.subclip(start_time, end_time)

                            output_filename = f"{base_name}_{i}.mp4"
                            output_file_path = os.path.join(output_dir, output_filename)
                            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                            def process_frame(frame):
                                img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                h, w = img.shape[:2]
                                scale = min(target_width / w, target_height / h)
                                new_w = int(w * scale)
                                new_h = int(h * scale)
                                resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                                background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                                x_offset = (target_width - new_w) // 2
                                y_offset = (target_height - new_h) // 2
                                background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
                                return cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

                            processed_clip = sub_clip.fl_image(process_frame)
                            fps = processed_clip.fps if processed_clip.fps else 24
                            processed_clip.write_videofile(
                                output_file_path,
                                codec='libx264',
                                fps=fps,
                                preset='slow',
                                threads=4,
                                audio=False
                            )
                            processed_clip.close()

                            # Copy corresponding txt file
                            txt_source = os.path.join(root, f"{base_name}.txt")
                            if os.path.exists(txt_source):
                                txt_target = os.path.join(output_dir, f"{base_name}_{i}.txt")
                                shutil.copy2(txt_source, txt_target)

                    clip.close()
                except Exception as e:
                    print(f"Failed to process video {file_path}: {e}")

# Example usage
change_resolution_and_save(
    input_path="test-HunyuanVideo-pixelart-videos",
    output_path="test-HunyuanVideo-pixelart-videos_960x544x6",
    target_width=960,
    target_height=544,
    max_duration=6
)

change_resolution_and_save(
    input_path="Sebastian_Michaelis_Videos_Captioned",
    output_path="Sebastian_Michaelis_Videos_Captioned_512x384x2",
    target_width=512,
    target_height=384,
    max_duration=2
)
```

### 8.3 Remove Training Directory

Remove the training directory:

```bash
rm -rf test-HunyuanVideo-pixelart-videos_960x544x6/train_ds
```

### 8.4 Edit Configuration File

Edit the `pixel_video_config.toml` file:

```toml
[general]
resolution = [960, 544]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_directory = "test-HunyuanVideo-pixelart-videos_960x544x6"
cache_directory = "test-HunyuanVideo-pixelart-videos_960x544x6_cache"
target_frames = [1, 25, 45]
frame_extraction = "head"
```

```toml
[general]
resolution = [512, 384]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_directory = "Sebastian_Michaelis_Videos_Captioned_512x384x2"
cache_directory = "Sebastian_Michaelis_Videos_Captioned_512x384x2_cache"
target_frames = [1, 25]
frame_extraction = "head"
```

### 8.5 Train Pixel Art Model

Train the pixel art model:

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 wan_train_network.py \
    --task t2v-1.3B --t5 models_t5_umt5-xxl-enc-bf16.pth \
    --dit wan2.1_t2v_1.3B_bf16.safetensors \
    --dataset_config pixel_video_config.toml --sdpa --mixed_precision bf16 --fp8_base \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_wan --network_dim 32 \
    --timestep_sampling shift --discrete_flow_shift 3.0 \
    --max_train_epochs 16 --save_every_n_epochs 1 --seed 42 \
    --output_dir pixel_outputs --output_name pixel_w1_3_lora
```

---

### 8.6 I2V 
```python
# Example usage
change_resolution_and_save(
    input_path="test-HunyuanVideo-pixelart-videos",
    output_path="test-HunyuanVideo-pixelart-videos_512x384x3",
    target_width=512,
    target_height=384,
    max_duration=3
)
```

```toml
# general configurations
[general]
resolution = [512, 384]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
video_directory = "test-HunyuanVideo-pixelart-videos_512x384x3"
cache_directory = "test-HunyuanVideo-pixelart-videos_512x384x3_cache" # recommended to set cache directory
target_frames = [25, 45]
frame_extraction = "head"
```

```bash
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

```bash
python wan_generate_video.py --fp8 --video_size 832 480 --video_length 45 --infer_steps 20 \
--save_path save --output_type both \
--task i2v-14B --t5 models_t5_umt5-xxl-enc-bf16.pth --clip models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth \
--dit wan2.1_i2v_480p_14B_fp8_e4m3fn.safetensors --vae Wan2.1_VAE.pth \
--t5 models_t5_umt5-xxl-enc-bf16.pth \
--attn_mode torch \
--lora_weight pixel_outputs/pixel_w14_lora-000008.safetensors \
--lora_multiplier 1.0 \
--image_path "pixel_im1.png" \
--prompt "The video showcases a young girl with orange hair and blue eyes, sitting on the ground. She's wearing a colorful dress with a brown skirt and a yellow top, along with red shoes. The girl is holding a red cup with a straw and has a green hat with a red band. The background features a pink sky with hearts and a yellow plant."

```

```python
import os
import cv2
import numpy as np
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import shutil

def change_resolution_and_save(input_path, output_path, target_width=1024, target_height=768, max_duration=4, default_txt_content=""):
    """Process images and videos to target resolution and split videos into segments.
    
    Args:
        input_path: Path to input directory
        output_path: Path to output directory
        target_width: Target width for resizing
        target_height: Target height for resizing
        max_duration: Maximum duration for video segments (in seconds)
        default_txt_content: Default text content to use when txt file doesn't exist
    """
    os.makedirs(output_path, exist_ok=True)

    for root, dirs, files in os.walk(input_path):
        for file in tqdm(files, desc="Processing files"):
            file_path = os.path.join(root, file)
            relative_path = os.path.relpath(file_path, input_path)
            output_dir = os.path.dirname(os.path.join(output_path, relative_path))

            # Process images
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                try:
                    img = cv2.imread(file_path)
                    h, w = img.shape[:2]
                    scale = min(target_width / w, target_height / h)
                    new_w = int(w * scale)
                    new_h = int(h * scale)
                    resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                    x_offset = (target_width - new_w) // 2
                    y_offset = (target_height - new_h) // 2
                    background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
                    output_file_path = os.path.join(output_path, relative_path)
                    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                    cv2.imwrite(output_file_path, background)

                    # Handle corresponding txt file
                    base_name = os.path.splitext(file)[0]
                    txt_source = os.path.join(root, f"{base_name}.txt")
                    txt_target = os.path.join(output_dir, f"{base_name}.txt")
                    
                    if os.path.exists(txt_source):
                        shutil.copy2(txt_source, txt_target)
                    else:
                        # Create txt file with default content if it doesn't exist
                        with open(txt_target, 'w') as f:
                            f.write(default_txt_content)
                except Exception as e:
                    print(f"Failed to process image {file_path}: {e}")

            # Process videos
            elif file.lower().endswith('.mp4'):
                try:
                    clip = VideoFileClip(file_path)
                    total_duration = clip.duration
                    base_name = os.path.splitext(file)[0]

                    if total_duration <= max_duration:
                        # Process the entire video
                        output_filename = f"{base_name}.mp4"
                        output_file_path = os.path.join(output_dir, output_filename)
                        os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                        def process_frame(frame):
                            img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                            h, w = img.shape[:2]
                            scale = min(target_width / w, target_height / h)
                            new_w = int(w * scale)
                            new_h = int(h * scale)
                            resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                            background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                            x_offset = (target_width - new_w) // 2
                            y_offset = (target_height - new_h) // 2
                            background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
                            return cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

                        processed_clip = clip.fl_image(process_frame)
                        fps = processed_clip.fps if processed_clip.fps else 24
                        processed_clip.write_videofile(
                            output_file_path,
                            codec='libx264',
                            fps=fps,
                            preset='slow',
                            threads=4,
                            audio=False
                        )
                        processed_clip.close()

                        # Handle corresponding txt file
                        txt_source = os.path.join(root, f"{base_name}.txt")
                        txt_target = os.path.join(output_dir, f"{base_name}.txt")
                        
                        if os.path.exists(txt_source):
                            shutil.copy2(txt_source, txt_target)
                        else:
                            # Create txt file with default content if it doesn't exist
                            with open(txt_target, 'w') as f:
                                f.write(default_txt_content)
                    else:
                        # Split and process the video
                        num_segments = int(total_duration // max_duration)
                        for i in range(num_segments):
                            start_time = i * max_duration
                            end_time = min((i+1) * max_duration, total_duration)
                            sub_clip = clip.subclip(start_time, end_time)

                            output_filename = f"{base_name}_{i}.mp4"
                            output_file_path = os.path.join(output_dir, output_filename)
                            os.makedirs(os.path.dirname(output_file_path), exist_ok=True)

                            def process_frame(frame):
                                img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                                h, w = img.shape[:2]
                                scale = min(target_width / w, target_height / h)
                                new_w = int(w * scale)
                                new_h = int(h * scale)
                                resized_img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                                background = np.zeros((target_height, target_width, 3), dtype=np.uint8)
                                x_offset = (target_width - new_w) // 2
                                y_offset = (target_height - new_h) // 2
                                background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
                                return cv2.cvtColor(background, cv2.COLOR_BGR2RGB)

                            processed_clip = sub_clip.fl_image(process_frame)
                            fps = processed_clip.fps if processed_clip.fps else 24
                            processed_clip.write_videofile(
                                output_file_path,
                                codec='libx264',
                                fps=fps,
                                preset='slow',
                                threads=4,
                                audio=False
                            )
                            processed_clip.close()

                            # Handle corresponding txt file
                            txt_source = os.path.join(root, f"{base_name}.txt")
                            txt_target = os.path.join(output_dir, f"{base_name}_{i}.txt")
                            
                            if os.path.exists(txt_source):
                                shutil.copy2(txt_source, txt_target)
                            else:
                                # Create txt file with default content if it doesn't exist
                                with open(txt_target, 'w') as f:
                                    f.write(default_txt_content)

                    clip.close()
                except Exception as e:
                    print(f"Failed to process video {file_path}: {e}")

# Example usage with default text content
# Example usage
change_resolution_and_save(
    input_path="Genshin_StarRail_Longshu_Sketch_Tail_Videos_Reversed_1_5_seconds",
    output_path="Genshin_StarRail_Longshu_Sketch_Tail_Videos_Reversed_1_5_seconds_512x384x3",
    target_width=512,
    target_height=384,
    max_duration=3,
    default_txt_content = "In the style of Long shu Reverse Sketch, This video demonstrates the step-by-step process of converting an anime-style image into a prototype sketch."
)
```

## 9. Conclusion

You have successfully set up, trained, and generated videos using the Musubi-Tuner framework. For further customization, refer to the configuration files and experiment with different parameters.
