# Musubi Tuner

[English](./README.md) | [日本語](./README.ja.md)

This repository provides scripts for training LoRA (Low-Rank Adaptation) models with HunyuanVideo.

__This repository is under development. Only image training has been verified.__

### Hardware Requirements

- VRAM: 24GB or more (May work with 12GB+ but this is unverified)
- Main Memory: 64GB or more recommended

### Features

- Memory-efficient implementation
- Windows compatible (Linux compatibility not yet verified)
- Multi-GPU support not implemented

## Installation

Create a virtual environment and install PyTorch and torchvision matching your CUDA version. Verified to work with version 2.5.1.

```bash
sudo apt-get update && sudo apt-get install git-lfs ffmpeg cbm

conda create -n musubi-tuner python=3.10
conda activate musubi-tuner
pip install ipykernel
python -m ipykernel install --user --name musubi-tuner --display-name "musubi-tuner"

pip install torch==2.5.0 torchvision

#pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

Install the required dependencies using the following command:

```bash
git clone https://github.com/svjack/musubi-tuner && cd musubi-tuner
pip install -r requirements.txt
```

Optionally, you can use FlashAttention and SageAttention (see [SageAttention Installation](#sageattention-installation) for installation instructions).

Additionally, install `ascii-magic` (used for dataset verification), `matplotlib` (used for timestep visualization), and `tensorboard` (used for logging training progress) as needed:

```bash
pip install ascii-magic matplotlib tensorboard huggingface_hub datasets
pip install moviepy==1.0.3
pip install sageattention==1.0.6
```

### Model Download

Download the model following the [official README](https://github.com/Tencent/HunyuanVideo/blob/main/ckpts/README.md) and place it in your chosen directory with the following structure:

```bash
huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts
cd ckpts
huggingface-cli download xtuner/llava-llama-3-8b-v1_1-transformers --local-dir ./llava-llama-3-8b-v1_1-transformers
wget https://raw.githubusercontent.com/Tencent/HunyuanVideo/refs/heads/main/hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py
python preprocess_text_encoder_tokenizer_utils.py --input_dir llava-llama-3-8b-v1_1-transformers --output_dir text_encoder
huggingface-cli download openai/clip-vit-large-patch14 --local-dir ./text_encoder_2
```

```
  ckpts
    ├──hunyuan-video-t2v-720p
    │  ├──transformers
    │  ├──vae
    ├──text_encoder
    ├──text_encoder_2
    ├──...
```

## Usage

### Dataset Configuration

Please refer to [dataset configuration guide](./dataset/dataset_config.md).

### Latent Pre-caching

Latent pre-caching is required. Create the cache using the following command:
```bash
git clone https://huggingface.co/datasets/svjack/Genshin-Impact-XiangLing-animatediff-with-score-organized
```

- Video Dataset
```python
from moviepy.editor import VideoFileClip
import os
import toml  # 需要安装 toml 库

def generate_video_config(video_path, save_path=None):
    # 加载视频
    clip = VideoFileClip(video_path)

    # 计算视频的分辨率（长宽）和帧数
    width, height = clip.size
    frame_count = int(clip.fps * clip.duration)

    # 初始的 target_frames
    target_frames = [1, 25, 45]

    # 去掉大于 frame_count 的元素
    target_frames = [frame for frame in target_frames if frame <= frame_count]

    # 确保 target_frames 严格递增
    target_frames = sorted(set(target_frames))  # 去重并排序

    # 确保最后一个元素是 frame_count
    if frame_count not in target_frames:
        target_frames.append(frame_count)
    target_frames = sorted(set(target_frames))  # 再次确保严格递增

    # 构建 TOML 格式的配置字典
    config = {
        "general": {
            "resolution": [width, height],
            "caption_extension": ".txt",
            "batch_size": 1,
            "enable_bucket": True,
            "bucket_no_upscale": False,
        },
        "datasets": [
            {
                "video_directory": "/path/to/video_dir",
                "target_frames": target_frames,
                "frame_extraction": "head",
            }
        ],
    }

    # 将配置字典转换为 TOML 格式字符串
    config_str = toml.dumps(config)

    # 打印生成的配置
    print("Generated Configuration (TOML):")
    print(config_str)

    # 如果提供了保存路径，将配置保存到本地文件
    if save_path:
        with open(save_path, 'w') as f:
            toml.dump(config, f)
        print(f"Configuration saved to {save_path}")

    # 关闭视频剪辑
    clip.close()

    return config_str

# 示例使用
import pathlib
video_path = str(list(pathlib.Path("Genshin-Impact-XiangLing-animatediff-with-score-organized").rglob("*.mp4"))[0])
save_path = "video_config.toml"  # 配置保存路径，可选
# 生成并保存配置
config = generate_video_config(video_path, save_path)
```

- add Genshin-Impact-XiangLing-animatediff-with-score-organized as video_directory

- Image Dataset
```python
import os
from moviepy.editor import VideoFileClip  # 使用 moviepy.editor 中的 VideoFileClip
from tqdm import tqdm
import shutil

def extract_first_frame_and_copy_txt(input_path, output_path):
    """
    遍历输入路径中的所有视频文件，当有对应的 .txt 文件时，
    将视频的第一帧保存为 .png 文件，并将对应的 .txt 文件拷贝到输出路径。

    :param input_path: 输入路径，包含视频和 .txt 文件
    :param output_path: 输出路径，保存 .png 和 .txt 文件
    """
    # 确保输出路径存在
    os.makedirs(output_path, exist_ok=True)

    # 获取输入路径中的所有文件
    files = os.listdir(input_path)
    video_files = [f for f in files if f.endswith(('.mp4', '.avi', '.mov', '.mkv'))]

    # 使用 tqdm 显示进度
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_name = os.path.splitext(video_file)[0]
        txt_file = f"{video_name}.txt"

        # 检查是否存在对应的 .txt 文件
        if txt_file in files:
            # 视频文件路径
            video_path = os.path.join(input_path, video_file)
            # 输出 .png 文件路径
            png_output_path = os.path.join(output_path, f"{video_name}.png")
            # 输出 .txt 文件路径
            txt_output_path = os.path.join(output_path, txt_file)

            try:
                # 使用 moviepy 提取视频的第一帧
                with VideoFileClip(video_path) as clip:
                    frame = clip.get_frame(0)  # 获取第一帧
                    clip.save_frame(png_output_path, t=0)  # 保存为 .png 文件

                # 拷贝对应的 .txt 文件
                shutil.copy2(os.path.join(input_path, txt_file), txt_output_path)

                print(f"Processed: {video_file} -> {png_output_path}, {txt_file} -> {txt_output_path}")
            except Exception as e:
                print(f"Error processing {video_file}: {e}")
        else:
            print(f"Skipped: {video_file} (no corresponding .txt file)")

input_path = "Genshin-Impact-XiangLing-animatediff-with-score-organized/"
output_path = "Genshin-Impact-XiangLing-animatediff-with-score-organized-Image"
extract_first_frame_and_copy_txt(input_path, output_path)

import os
from PIL import Image
import toml  # 需要安装 toml 库

def generate_image_config(image_path, save_path=None):
    # 加载图片
    img = Image.open(image_path)

    # 获取图片的分辨率（长宽）
    width, height = img.size

    # 构建 TOML 格式的配置字典
    config = {
        "general": {
            "resolution": [width, height],
            "caption_extension": ".txt",
            "batch_size": 1,
            "enable_bucket": True,
            "bucket_no_upscale": False,
        },
        "datasets": [
            {
                "image_directory": os.path.dirname(image_path),
                # 移除了 image_files 字段
            }
        ],
    }

    # 将配置字典转换为 TOML 格式字符串
    config_str = toml.dumps(config)

    # 打印生成的配置
    print("Generated Configuration (TOML):")
    print(config_str)

    # 如果提供了保存路径，将配置保存到本地文件
    if save_path:
        with open(save_path, 'w') as f:
            toml.dump(config, f)
        print(f"Configuration saved to {save_path}")

    # 关闭图片
    img.close()

    return config_str

# 示例使用
import pathlib
image_path = str(list(pathlib.Path("Genshin-Impact-XiangLing-animatediff-with-score-organized-Image").rglob("*.png"))[0])
save_path = "image_config.toml"  # 配置保存路径，可选
# 生成并保存配置
config = generate_image_config(image_path, save_path)
```
- Image Dataset
```python
from datasets import load_dataset
image_ds = load_dataset("svjack/Genshin-Impact-ZhongLi-NingGuang-Couple-Image")
image_df = image_ds["train"].to_pandas()
man_prompt = "ZHONGLI\\(genshin impact\\)."
woman_prompt = "NING GUANG\\(genshin impact\\) in red cheongsam."
couple_prompt = "ZHONGLI\\(genshin impact\\) with NING GUANG\\(genshin impact\\) in red cheongsam."
image_df["typed_prompt"] = image_df.apply(
    lambda x: couple_prompt + " " + x["action"] if x["image_type"] == "couple" else (
      man_prompt + " " + x["action"]  if x["image_type"] == "man" else woman_prompt + " " + x["action"]
    ), axis = 1
)
image_df["typed_prompt"].value_counts()

from PIL import Image
import io

def bytes_to_pil_image(image_bytes):
    """
    将 bytes 字符串转换为 PIL Image 对象。

    参数:
        image_bytes (bytes): 图像的 bytes 数据。

    返回:
        PIL.Image.Image: 转换后的 PIL Image 对象。
    """
    # 使用 io.BytesIO 将 bytes 数据转换为文件类对象
    image_file = io.BytesIO(image_bytes)

    # 使用 PIL.Image.open 打开图像
    pil_image = Image.open(image_file)

    return pil_image

image_df["image_obj"] = image_df["image"].map(lambda x: bytes_to_pil_image(x["bytes"]))

image_df[["image_obj", "typed_prompt"]]


import os
import uuid
from PIL import Image

def save_image_and_text(image_df, output_dir):
    """
    将 DataFrame 中的 PIL 对象和文本保存为 PNG 和 TXT 文件。

    参数:
        image_df (pd.DataFrame): 包含 "image_obj" 和 "typed_prompt" 列的 DataFrame。
        output_dir (str): 输出文件的目录路径。

    返回:
        None
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 按行迭代 DataFrame
    for _, row in image_df.iterrows():
        # 生成唯一的文件名（基于 UUID）
        file_name = str(uuid.uuid4())

        # 保存 PIL 对象为 PNG 文件
        image_path = os.path.join(output_dir, f"{file_name}.png")
        row["image_obj"].save(image_path)

        # 保存文本为 TXT 文件
        text_path = os.path.join(output_dir, f"{file_name}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(row["typed_prompt"])

        print(f"Saved: {file_name}.png and {file_name}.txt")

# 示例调用
save_image_and_text(image_df[["image_obj", "typed_prompt"]], "zhongli_ningguang_couple_img")


import os
from PIL import Image
import toml  # 需要安装 toml 库

def generate_image_config(image_path, save_path=None):
    # 加载图片
    img = Image.open(image_path)

    # 获取图片的分辨率（长宽）
    width, height = img.size

    # 构建 TOML 格式的配置字典
    config = {
        "general": {
            "resolution": [width, height],
            "caption_extension": ".txt",
            "batch_size": 1,
            "enable_bucket": True,
            "bucket_no_upscale": False,
        },
        "datasets": [
            {
                "image_directory": os.path.dirname(image_path),
                # 移除了 image_files 字段
            }
        ],
    }

    # 将配置字典转换为 TOML 格式字符串
    config_str = toml.dumps(config)

    # 打印生成的配置
    print("Generated Configuration (TOML):")
    print(config_str)

    # 如果提供了保存路径，将配置保存到本地文件
    if save_path:
        with open(save_path, 'w') as f:
            toml.dump(config, f)
        print(f"Configuration saved to {save_path}")

    # 关闭图片
    img.close()

    return config_str

# 示例使用
import pathlib
image_path = str(list(pathlib.Path("zhongli_ningguang_couple_img").rglob("*.png"))[0])
save_path = "zhongning_image_config.toml"  # 配置保存路径，可选
# 生成并保存配置
config = generate_image_config(image_path, save_path)
```

- Video
```bash
python cache_latents.py --dataset_config video_config.toml --vae ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_tiling
```
- Image
```bash
python cache_latents.py --dataset_config image_config.toml --vae ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_tiling
```
- Image
```bash
python cache_latents.py --dataset_config zhongning_image_config.toml --vae ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_tiling
```

```bash
python cache_latents.py --dataset_config path/to/toml --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_tiling
```

For additional options, use `python cache_latents.py --help`.

If you're running low on VRAM, reduce `--vae_spatial_tile_sample_min_size` to around 128 and lower the `--batch_size`.

Use `--debug_mode image` to display dataset images and captions in a new window, or `--debug_mode console` to display them in the console (requires `ascii-magic`).

### Text Encoder Output Pre-caching

Text Encoder output pre-caching is required. Create the cache using the following command:

- Video
```bash
python cache_text_encoder_outputs.py --dataset_config video_config.toml  --text_encoder1 ckpts/text_encoder --text_encoder2 ckpts/text_encoder_2 --batch_size 16
```

- Image
```bash
python cache_text_encoder_outputs.py --dataset_config image_config.toml  --text_encoder1 ckpts/text_encoder --text_encoder2 ckpts/text_encoder_2 --batch_size 16
```

- Image
```bash
python cache_text_encoder_outputs.py --dataset_config zhongning_image_config.toml  --text_encoder1 ckpts/text_encoder --text_encoder2 ckpts/text_encoder_2 --batch_size 16
```

```bash
python cache_text_encoder_outputs.py --dataset_config path/to/toml  --text_encoder1 path/to/ckpts/text_encoder --text_encoder2 path/to/ckpts/text_encoder_2 --batch_size 16
```

For additional options, use `python cache_text_encoder_outputs.py --help`.

Adjust `--batch_size` according to your available VRAM.

For systems with limited VRAM (less than ~16GB), use `--fp8_llm` to run the LLM in fp8 mode.

### Training

Start training using the following command (input as a single line):

- Video In A6000
```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 hv_train_network.py \
    --dit ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --dataset_config video_config.toml \
    --sdpa \
    --mixed_precision bf16 \
    --fp8_base \
    --optimizer_type adamw8bit \
    --learning_rate 1e-3 \
    --gradient_checkpointing \
    --max_data_loader_n_workers 2 \
    --persistent_data_loader_workers \
    --network_module networks.lora \
    --network_dim 32 \
    --timestep_sampling sigmoid \
    --discrete_flow_shift 1.0 \
    --max_train_epochs 16 \
    --save_every_n_epochs 1 \
    --seed 42 \
    --output_dir xiangling_lora_dir \
    --output_name xiangling_lora
```

- Image in RTX 4090
```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 hv_train_network.py \
    --dit ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --dataset_config image_config.toml \
    --sdpa \
    --mixed_precision bf16 \
    --fp8_base \
    --optimizer_type adamw8bit \
    --learning_rate 1e-3 \
    --gradient_checkpointing \
    --max_data_loader_n_workers 2 \
    --persistent_data_loader_workers \
    --network_module networks.lora \
    --network_dim 32 \
    --timestep_sampling sigmoid \
    --discrete_flow_shift 1.0 \
    --max_train_epochs 16 \
    --save_every_n_epochs 1 \
    --seed 42 \
    --output_dir xiangling_im_lora_dir \
    --output_name xiangling_im_lora
```

- Image in RTX 4090
```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 hv_train_network.py \
    --dit ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --dataset_config zhongning_image_config.toml \
    --sdpa \
    --mixed_precision bf16 \
    --fp8_base \
    --optimizer_type adamw8bit \
    --learning_rate 1e-3 \
    --gradient_checkpointing \
    --max_data_loader_n_workers 2 \
    --persistent_data_loader_workers \
    --network_module networks.lora \
    --network_dim 32 \
    --timestep_sampling sigmoid \
    --discrete_flow_shift 1.0 \
    --max_train_epochs 16 \
    --save_every_n_epochs 1 \
    --seed 42 \
    --output_dir zhongli_ningguang_couple_im_lora_dir \
    --output_name zhongli_ningguang_couple_im_lora
```

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 hv_train_network.py 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt 
    --dataset_config path/to/toml --sdpa --mixed_precision bf16 --fp8_base 
    --optimizer_type adamw8bit --learning_rate 1e-3 --gradient_checkpointing 
     --max_data_loader_n_workers 2 --persistent_data_loader_workers 
    --network_module=networks.lora --network_dim=32 
    --timestep_sampling sigmoid --discrete_flow_shift 1.0 
    --max_train_epochs 16 --save_every_n_epochs=1 --seed 42
    --output_dir path/to/output_dir --output_name name-of-lora
```

For additional options, use `python hv_train_network.py --help` (note that many options are unverified).

Specifying `--fp8_base` runs DiT in fp8 mode. Without this flag, mixed precision data type will be used. fp8 can significantly reduce memory consumption but may impact output quality.

If you're running low on VRAM, use `--blocks_to_swap` to offload some blocks to CPU. Maximum value is 36.

(The idea of block swap is based on the implementation by 2kpr. Thanks again to 2kpr.)

Use `--sdpa` for PyTorch's scaled dot product attention, `--sage_attn` for SageAttention (note: training issues observed in testing, trained model may not work as expected), or `--flash_attn` for FlashAttention (untested).

Sample video generation is not yet implemented.

The format of LoRA trained is the same as `sd-scripts`.

`--show_timesteps` can be set to `image` (requires `matplotlib`) or `console` to display timestep distribution and loss weighting during training.

Appropriate learning rates, training steps, timestep distribution, loss weighting, etc. are not yet known. Feedback is welcome.

### Inference

Generate videos using the following command:

```bash
python hv_generate_video.py \
    --fp8 \
    --video_size 544 960 \
    --video_length 60 \
    --infer_steps 30 \
    --prompt "solo,Xiangling, cook rice in a pot ,genshin impact ,1girl,highres," \
    --save_path . \
    --output_type both \
    --dit ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --attn_mode sdpa \
    --vae ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt \
    --vae_chunk_size 32 \
    --vae_spatial_tile_sample_min_size 128 \
    --text_encoder1 ckpts/text_encoder \
    --text_encoder2 ckpts/text_encoder_2 \
    --seed 1234 \
    --lora_multiplier 1.0 \
    --lora_weight xiangling_im_lora_dir/xiangling_im_lora-000003.safetensors
```


https://github.com/user-attachments/assets/fe15b82c-ffa3-4820-afe5-1454762c7ec6


```bash
python hv_generate_video.py \
    --fp8 \
    --video_size 544 960 \
    --video_length 60 \
    --infer_steps 30 \
    --prompt "solo,Xiangling, drink water, (genshin impact) ,1girl,highres, dynamic" \
    --save_path . \
    --output_type both \
    --dit ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --attn_mode sdpa \
    --vae ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt \
    --vae_chunk_size 32 \
    --vae_spatial_tile_sample_min_size 128 \
    --text_encoder1 ckpts/text_encoder \
    --text_encoder2 ckpts/text_encoder_2 \
    --seed 1234 \
    --lora_multiplier 1.0 \
    --lora_weight xiangling_im_lora_dir/xiangling_im_lora-000003.safetensors
```




https://github.com/user-attachments/assets/5e8bdcf0-f590-4441-b47b-55e6af8f4cf3


```baah
python hv_generate_video.py \
    --fp8 \
    --video_size 544 960 \
    --video_length 60 \
    --infer_steps 30 \
    --prompt "ZHONGLI\\(genshin impact\\) with NING GUANG\\(genshin impact\\) in red cheongsam. cook rice in a pot" \
    --save_path . \
    --output_type both \
    --dit ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --attn_mode sdpa \
    --vae ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt \
    --vae_chunk_size 32 \
    --vae_spatial_tile_sample_min_size 128 \
    --text_encoder1 ckpts/text_encoder \
    --text_encoder2 ckpts/text_encoder_2 \
    --seed 1234 \
    --lora_multiplier 1.0 \
    --lora_weight zhongli_ningguang_couple_im_lora_dir/zhongli_ningguang_couple_im_lora-000012.safetensors
```




https://github.com/user-attachments/assets/b56540c0-0b29-409d-853e-9aaed2a25a28


```baah
python hv_generate_video.py \
    --fp8 \
    --video_size 544 960 \
    --video_length 60 \
    --infer_steps 30 \
    --prompt "ZHONGLI\\(genshin impact\\). drink tea" \
    --save_path . \
    --output_type both \
    --dit ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --attn_mode sdpa \
    --vae ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt \
    --vae_chunk_size 32 \
    --vae_spatial_tile_sample_min_size 128 \
    --text_encoder1 ckpts/text_encoder \
    --text_encoder2 ckpts/text_encoder_2 \
    --seed 1234 \
    --lora_multiplier 1.0 \
    --lora_weight zhongli_ningguang_couple_im_lora_dir/zhongli_ningguang_couple_im_lora-000012.safetensors
```



https://github.com/user-attachments/assets/555aa2ea-9a04-4f05-8ab0-6602eeb4c866


# Convert
```bash
python convert_lora.py --input genshin_impact_ep_landscape_lora_dir/genshin_impact_ep_landscape_lora-000004.safetensors --output genshin_impact_ep_landscape_lora-000004-comfy.safetensor --target other
```
- after convert can use in https://github.com/svjack/HunyuanVideoGP by: 
```bash
python gradio_server.py --fastest --lora-weight ../work/pixel_hunyuan_video_lora_test/20250216_11-34-50/epoch2/adapter_model.safetensors --lora-multiplier 1
```


```bash
python hv_generate_video.py --fp8 --video_size 544 960 --video_length 5 --infer_steps 30 
    --prompt "A cat walks on the grass, realistic style."  --save_path path/to/save/dir --output_type both 
    --dit path/to/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt --attn_mode sdpa 
    --vae path/to/ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt 
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 
    --text_encoder1 path/to/ckpts/text_encoder 
    --text_encoder2 path/to/ckpts/text_encoder_2 
    --seed 1234 --lora_multiplier 1.0 --lora_weight path/to/lora.safetensors
```

For additional options, use `python hv_generate_video.py --help`.

Specifying `--fp8` runs DiT in fp8 mode. fp8 can significantly reduce memory consumption but may impact output quality.

If you're running low on VRAM, use `--blocks_to_swap` to offload some blocks to CPU. Maximum value is 38.

For `--attn_mode`, specify either `flash`, `torch`, `sageattn`, or `sdpa` (same as `torch`). These correspond to FlashAttention, scaled dot product attention, and SageAttention respectively. Default is `torch`. SageAttention is effective for VRAM reduction.

For `--output_type`, specify either `both`, `latent`, or `video`. `both` outputs both latents and video. Recommended to use `both` in case of Out of Memory errors during VAE processing. You can specify saved latents with `--latent_path` and use `--output_type video` to only perform VAE decoding.

`--seed` is optional. A random seed will be used if not specified.

`--video_length` should be specified as "a multiple of 4 plus 1".

### Convert LoRA to another format

You can convert LoRA to a format compatible with ComfyUI (presumed to be Diffusion-pipe) using the following command:

```bash
python convert_lora.py --input path/to/musubi_lora.safetensors --output path/to/another_format.safetensors --target other
```

Specify the input and output file paths with `--input` and `--output`, respectively.

Specify `other` for `--target`. Use `default` to convert from another format to the format of this repository.

## Miscellaneous

### SageAttention Installation

For reference, here are basic steps for installing SageAttention. You may need to update Microsoft Visual C++ Redistributable to the latest version.

1. Download and install triton 3.1.0 wheel matching your Python version from [here](https://github.com/woct0rdho/triton-windows/releases/tag/v3.1.0-windows.post5).

2. Install Microsoft Visual Studio 2022 or Build Tools for Visual Studio 2022, configured for C++ builds.

3. Clone the SageAttention repository in your preferred directory:
    ```shell
    git clone https://github.com/thu-ml/SageAttention.git
    ```

4. Open `math.cuh` in the `SageAttention/csrc` folder and change `ushort` to `unsigned short` on lines 71 and 146, then save.

5. Open `x64 Native Tools Command Prompt for VS 2022` from the Start menu under Visual Studio 2022.

6. Activate your venv, navigate to the SageAttention folder, and run the following command. If you get a DISTUTILS not configured error, set `set DISTUTILS_USE_SDK=1` and try again:
    ```shell
    python setup.py install
    ```

This completes the SageAttention installation.


## Disclaimer

This repository is unofficial and not affiliated with the official HunyuanVideo repository. 

This repository is experimental and under active development. While we welcome community usage and feedback, please note:

- This is not intended for production use
- Features and APIs may change without notice
- Some functionalities are still experimental and may not work as expected
- Video training features are still under development

If you encounter any issues or bugs, please create an Issue in this repository with:
- A detailed description of the problem
- Steps to reproduce
- Your environment details (OS, GPU, VRAM, Python version, etc.)
- Any relevant error messages or logs

## Contributing

We welcome contributions! However, please note:

- Due to limited maintainer resources, PR reviews and merges may take some time
- Before starting work on major changes, please open an Issue for discussion
- For PRs:
  - Keep changes focused and reasonably sized
  - Include clear descriptions
  - Follow the existing code style
  - Ensure documentation is updated

## License

Code under the `hunyuan_model` directory is modified from [HunyuanVideo](https://github.com/Tencent/HunyuanVideo) and follows their license.

Other code is under the Apache License 2.0. Some code is copied and modified from Diffusers.
