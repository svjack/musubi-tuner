from datasets import load_dataset, Dataset
import pandas as pd
import re
import numpy as np
import os
import uuid
from PIL import Image
import toml

# 1. 加载数据集
ds = load_dataset("svjack/Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender")
df = load_dataset("svjack/Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-Joy-Caption-only-Caption")["train"].to_pandas()

# 2. 合并数据集
joy_caption_dict = dict(zip(df["im_name"], df["joy-caption"]))

def add_joy_caption(example):
    im_name = example["im_name"]
    example["joy-caption"] = joy_caption_dict.get(im_name, None)
    return example

dss = ds.map(add_joy_caption, num_proc=6)

# 3. 过滤掉 joy-caption 为 None 的行
dss = dss.filter(lambda example: example["joy-caption"] is not None, num_proc=6)

# 4. 抽取中文字符串
def extract_chinese_strings(s):
    pattern = re.compile(r'[\u4e00-\u9fff]+')
    matches = pattern.findall(s)
    return matches

def process_joy_caption(example):
    chinese_strings = extract_chinese_strings(example["joy-caption"])
    example["chinese_strings"] = chinese_strings if len(chinese_strings) == 1 else None
    return example

dss = dss.map(process_joy_caption, num_proc=6)

# 5. 过滤掉 chinese_strings 为 None 的行
dss = dss.filter(lambda example: example["chinese_strings"] is not None, num_proc=6)

# 6. 获取符合条件的中文字符串列表
valid_chinese_strings = pd.Series(
    dss["train"].to_pandas()["chinese_strings"]
    .map(lambda x: x[0])  # 提取中文字符串
    .value_counts()  # 统计频率
    .iloc[:-3]  # 去掉最后3个低频项
    .index.values  # 获取索引（中文字符串）
).tolist()

# 7. 进一步过滤数据集，只保留符合条件的数据
dss = dss.filter(lambda example: example["chinese_strings"][0] in valid_chinese_strings, num_proc=6)

# 8. 中文名到英文名的映射（全部大写）
name_mapping = {
    '芭芭拉': 'BARBARA',
    '柯莱': 'COLLEI',
    '雷电将军': 'RAIDEN SHOGUN',
    '云堇': 'YUN JIN',
    '八重神子': 'YAE MIKO',
    '妮露': 'NILOU',
    '绮良良': 'KIRARA',
    '砂糖': 'SUCROSE',
    '珐露珊': 'FARUZAN',
    '琳妮特': 'LYNETTE',
    '纳西妲': 'NAHIDA',
    '诺艾尔': 'NOELLE',
    '凝光': 'NINGGUANG',
    '鹿野院平藏': 'HEIZOU',
    '琴': 'JEAN',
    '枫原万叶': 'KAEDEHARA KAZUHA',
    '芙宁娜': 'FURINA',
    '艾尔海森': 'ALHAITHAM',
    '甘雨': 'GANYU',
    '凯亚': 'KAEYA',
    '荒泷一斗': 'ARATAKI ITTO',
    '优菈': 'EULA',
    '迪奥娜': 'DIONA',
    '温迪': 'VENTI',
    '神里绫人': 'KAMISATO AYATO',
    '阿贝多': 'ALBEDO',
    '重云': 'CHONGYUN',
    '钟离': 'ZHONGLI',
    '行秋': 'XINGQIU',
    '胡桃': 'HU TAO',
    '魈': 'XIAO',
    '赛诺': 'CYNO',
    '神里绫华': 'KAMISATO AYAKA',
    '五郎': 'GOROU',
    '林尼': 'LYNEY',
    '迪卢克': 'DILUC',
    '安柏': 'AMBER',
    '烟绯': 'YANFEI',
    '宵宫': 'YOIMIYA',
    '珊瑚宫心海': 'SANGONOMIYA KOKOMI',
    '罗莎莉亚': 'ROSARIA',
    '七七': 'QIQI',
    '久岐忍': 'KUKI SHINOBU',
    '申鹤': 'SHENHE',
    '托马': 'THOMA',
    '芙寧娜': 'FURINA',
    '雷泽': 'RAZOR'
}

# 9. 将 joy-caption 中的中文名字替换为英文名字
def replace_chinese_with_english(example):
    joy_caption = example["joy-caption"]
    for chinese_name, english_name in name_mapping.items():
        joy_caption = re.sub(chinese_name, english_name, joy_caption)
    example["joy-caption-english"] = joy_caption
    return example

dss = dss.map(replace_chinese_with_english, num_proc=6)

# 10. 可选：只保留 name_mapping 中某个列表的样本（注释掉默认不做筛选）
selected_names = ['芙宁娜', '芙寧娜']  # 定义要保留的中文名字列表
dss = dss.filter(lambda example: example["chinese_strings"][0] in selected_names, num_proc=6)

# 11. 保存图片和文本文件
def save_image_and_text(dataset: Dataset, output_dir: str):
    """
    将数据集中的图片和文本保存为 PNG 和 TXT 文件。

    参数:
        dataset (Dataset): Hugging Face 数据集，包含 "image" 和 "joy-caption-english" 列。
        output_dir (str): 输出文件的目录路径。
    """
    os.makedirs(output_dir, exist_ok=True)

    for example in dataset:
        file_name = str(uuid.uuid4())
        image_path = os.path.join(output_dir, f"{file_name}.png")
        example["image"].save(image_path)

        text_path = os.path.join(output_dir, f"{file_name}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(example["joy-caption-english"])

        print(f"Saved: {file_name}.png and {file_name}.txt")

# 12. 生成配置文件
def generate_image_config(image_dir: str, save_path: str = None):
    """
    生成图片配置文件的 TOML 格式。

    参数:
        image_dir (str): 图片目录路径。
        save_path (str): 配置文件的保存路径（可选）。
    """
    image_files = list(os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(".png"))
    if not image_files:
        raise ValueError("No PNG files found in the directory.")

    image_path = image_files[0]
    img = Image.open(image_path)
    width, height = img.size

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
                "image_directory": image_dir,
            }
        ],
    }

    config_str = toml.dumps(config)
    print("Generated Configuration (TOML):")
    print(config_str)

    if save_path:
        with open(save_path, "w") as f:
            toml.dump(config, f)
        print(f"Configuration saved to {save_path}")

    img.close()

    return config_str

# 13. 保存图片和文本文件
output_dir = "FURINA_single_images_and_texts"
save_image_and_text(dss["train"], output_dir)

# 14. 生成配置文件
config_save_path = "FURINA_image_config.toml"
generate_image_config(output_dir, config_save_path)

python cache_latents.py --dataset_config FURINA_image_config.toml --vae ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_tiling

python cache_text_encoder_outputs.py --dataset_config FURINA_image_config.toml  --text_encoder1 ckpts/text_encoder --text_encoder2 ckpts/text_encoder_2 --batch_size 16

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 hv_train_network.py \
    --dit ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \
    --dataset_config FURINA_image_config.toml \
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
    --output_dir FURINA_im_lora_dir \
    --output_name FURINA_im_lora

python hv_generate_video.py \
    --fp8 \
    --video_size 544 960 \
    --video_length 60 \
    --infer_steps 30 \
    --prompt "This is a digital anime-style drawing of FURINA, a young woman with shoulder-length, wavy, white hair accented with light blue streaks. She has large, expressive blue eyes and a gentle smile. She is leaning on her elbow on a bed with a white sheet, wearing a loose white t-shirt. The background shows a softly lit room with a wooden bedside table and a lamp emitting a warm glow. The overall atmosphere is cozy and serene." \
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
    --lora_weight FURINA_im_lora_dir/FURINA_im_lora-000010.safetensors
