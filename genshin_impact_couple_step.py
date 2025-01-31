import os
import uuid
import re
import numpy as np
import pandas as pd
from datasets import load_dataset, Dataset
from PIL import Image
import toml

from tqdm import tqdm
from IPython import display

# 1. 加载数据集
ds = load_dataset("svjack/Genshin-Impact-Couple-with-Tags-IID-Gender-Only-Two-Joy-Caption")

# 2. 移除 image 列并转换为 Pandas DataFrame
df = ds["train"].remove_columns(["image"]).to_pandas()

# 3. 定义字典
new_dict = {
    '砂糖': 'SUCROSE', '五郎': 'GOROU', '雷电将军': 'RAIDEN SHOGUN', '七七': 'QIQI', '重云': 'CHONGYUN',
    '荒泷一斗': 'ARATAKI ITTO', '申鹤': 'SHENHE', '赛诺': 'CYNO', '绮良良': 'KIRARA', '优菈': 'EULA',
    '魈': 'XIAO', '行秋': 'XINGQIU', '枫原万叶': 'KAEDEHARA KAZUHA', '凯亚': 'KAEYA', '凝光': 'NING GUANG',
    '安柏': 'AMBER', '柯莱': 'COLLEI', '林尼': 'LYNEY', '胡桃': 'HU TAO', '甘雨': 'GANYU',
    '神里绫华': 'KAMISATO AYAKA', '钟离': 'ZHONGLI', '纳西妲': 'NAHIDA', '云堇': 'YUN JIN',
    '久岐忍': 'KUKI SHINOBU', '迪西娅': 'DEHYA', '珐露珊': 'FARUZAN', '公子 达达利亚': 'TARTAGLIA',
    '琳妮特': 'LYNETTE', '罗莎莉亚': 'ROSARIA', '八重神子': 'YAE MIKO', '迪奥娜': 'DIONA',
    '迪卢克': 'DILUC', '托马': 'THOMA', '神里绫人': 'KAMISATO AYATO', '鹿野院平藏': 'SHIKANOIN HEIZOU',
    '阿贝多': 'ALBEDO', '琴': 'JEAN', '芭芭拉': 'BARBARA', '雷泽': 'RAZOR',
    '珊瑚宫心海': 'SANGONOMIYA KOKOMI', '温迪': 'VENTI', '烟绯': 'YANFEI', '艾尔海森': 'ALHAITHAM',
    '诺艾尔': 'NOELLE', '流浪者 散兵': 'SCARAMOUCHE', '班尼特': 'BENNETT', '芙宁娜': 'FURINA',
    '夏洛蒂': 'CHARLOTTE', '宵宫': 'YOIMIYA', '妮露': 'NILOU', '瑶瑶': 'YAOYAO'
}

# 4. 反转字典，方便通过英文名查找中文名
reverse_dict = {v: k for k, v in new_dict.items()}

# 5. 定义函数处理 im_name
def process_im_name(im_name):
    # 使用正则表达式提取 COUPLE_ 和 _g 之间的部分
    matches = re.findall(r"COUPLE_(.*?)_g", im_name)
    if len(matches) == 1 and matches[0] == matches[0].upper():
        # 将下划线替换为空格
        matched_name = matches[0].replace("_", " ")
        # 检查 matched_name 是否包含两个不同的 reverse_dict 键
        matched_keys = [key for key in reverse_dict if key in matched_name]
        if len(matched_keys) == 2 and matched_keys[0] != matched_keys[1]:
            # 获取两个英文名和中文名
            english_names = " & ".join(matched_keys)
            chinese_names = " & ".join([reverse_dict[key] for key in matched_keys])
            return english_names, chinese_names
    return np.nan, np.nan

# 6. 应用函数处理数据集
df[["english_names", "chinese_names"]] = df["im_name"].apply(
    lambda x: pd.Series(process_im_name(x))
)

# 7. 过滤掉不符合条件的行
df = df.dropna(subset=["english_names", "chinese_names"])

# 8. 获取符合条件的 im_name 列表
valid_im_names = df["im_name"].tolist()

# 9. 从原始 ds 中筛选出符合条件的行
filtered_ds = ds["train"].filter(lambda x: x["im_name"] in valid_im_names)

# 10. 将 english_names 和 chinese_names 添加到 filtered_ds
# 创建一个映射字典：im_name -> (english_names, chinese_names)
name_mapping = df.set_index("im_name")[["english_names", "chinese_names"]].to_dict(orient="index")

# 定义函数为每一行添加 english_names 和 chinese_names
def add_names(row):
    row["english_names"] = name_mapping[row["im_name"]]["english_names"]
    row["chinese_names"] = name_mapping[row["im_name"]]["chinese_names"]
    return row

# 应用函数
updated_ds = filtered_ds.map(add_names)

# 11. 创建新的列 joy-caption-english
def add_joy_caption_english(example):
    """
    将 english_names 和 joy-caption 结合生成新的列 joy-caption-english。
    """
    example["joy-caption-english"] = f"{example['english_names']} : {example['joy-caption']}"
    return example

# 12. 应用函数生成新的列
updated_ds = updated_ds.map(add_joy_caption_english)

# 13. 保存图片和文本文件
def save_image_and_text(dataset: Dataset, output_dir: str):
    """
    将数据集中的图片和文本保存为 PNG 和 TXT 文件。

    参数:
        dataset (Dataset): Hugging Face 数据集，包含 "image" 和 "joy-caption-english" 列。
        output_dir (str): 输出文件的目录路径。
    """
    os.makedirs(output_dir, exist_ok=True)

    for example in tqdm(dataset):
        file_name = str(uuid.uuid4())
        image_path = os.path.join(output_dir, f"{file_name}.png")
        example["image"].save(image_path)

        text_path = os.path.join(output_dir, f"{file_name}.txt")
        with open(text_path, "w", encoding="utf-8") as f:
            f.write(example["joy-caption-english"])

        print(f"Saved: {file_name}.png and {file_name}.txt")
        display.clear_output(wait = True)

# 14. 生成配置文件
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

# 15. 保存图片和文本文件
output_dir = "genshin_impact_couple_images_and_texts"
save_image_and_text(updated_ds, output_dir)

# 16. 生成配置文件
config_save_path = "genshin_impact_image_config.toml"
generate_image_config(output_dir, config_save_path)

# 17. 训练命令
print("""
运行以下命令进行训练：

python cache_latents.py --dataset_config genshin_impact_image_config.toml --vae ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt --vae_chunk_size 32 --vae_tiling

python cache_text_encoder_outputs.py --dataset_config genshin_impact_image_config.toml  --text_encoder1 ckpts/text_encoder --text_encoder2 ckpts/text_encoder_2 --batch_size 16

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 hv_train_network.py \\
    --dit ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt \\
    --dataset_config genshin_impact_image_config.toml \\
    --sdpa \\
    --mixed_precision bf16 \\
    --fp8_base \\
    --optimizer_type adamw8bit \\
    --learning_rate 1e-3 \\
    --gradient_checkpointing \\
    --max_data_loader_n_workers 2 \\
    --persistent_data_loader_workers \\
    --network_module networks.lora \\
    --network_dim 32 \\
    --timestep_sampling sigmoid \\
    --discrete_flow_shift 1.0 \\
    --max_train_epochs 16 \\
    --save_every_n_epochs 1 \\
    --seed 42 \\
    --output_dir genshin_impact_couple_im_lora_dir \\
    --output_name genshin_impact_couple_im_lora
""")
