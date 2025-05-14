from PIL import Image

def overlay_images(background_path, foreground_path, output_path):
    """
    将前景图片（RGBA）四角对齐覆盖到背景图片上

    参数:
        background_path (str): 背景图片路径
        foreground_path (str): 前景图片路径（支持透明通道）
        output_path (str): 输出图片路径
    """
    # 打开背景和前景图片
    background = Image.open(background_path).convert("RGBA")
    foreground = Image.open(foreground_path).convert("RGBA")

    # 确保前景图片不大于背景图片
    if foreground.width > background.width or foreground.height > background.height:
        raise ValueError("前景图片尺寸不能大于背景图片")

    # 合并图层
    combined = Image.alpha_composite(foreground, background)

    # 保存结果
    combined.save(output_path)

# 使用示例
overlay_images(
    background_path="xiang_im_38.png",
    foreground_path="a27a21cb-f66b-4671-9d59-a7e0ae43f635.png",
    output_path="output.png"
)

"sk-edb7312a2abe40f1a84a9b0e8943e4a7"

from datasets import load_dataset
from gradio_client import Client
from tqdm import tqdm

# 1. 加载数据集
print("正在加载数据集...")
ds = load_dataset("svjack/After_Tomorrow_SPLITED")
train_ds = ds["train"]  # 获取训练集
train_df = train_ds.to_pandas()

# 2. 初始化Gradio客户端
print("正在连接Gradio服务...")
client = Client("http://localhost:7860")

# 3. 定义处理函数
def process_sample(sample, idx):
    try:
        # 调用Gradio API生成景观提示
        result = client.predict(
            lyrics=sample["prompt"],
            api_name="/generate_scenery"
        )
        # 替换换行符为空格
        processed_prompt = result.replace("\n", " ")
        return {"landscape_prompt": processed_prompt}
    except Exception as e:
        print(f"\n处理样本 {idx} 时出错: {str(e)}")
        return {"landscape_prompt": ""}  # 出错时返回空字符串

# 4. 处理数据集并显示进度
print("开始处理数据集...")
processed_samples = []

# 使用tqdm显示详细进度
for i in tqdm(range(len(train_ds)), desc="生成景观提示", unit="样本"):
    sample = train_df.iloc[i]
    processed = process_sample(sample, i)
    processed_samples.append(processed)

# 5. 将处理结果添加到数据集
print("正在将结果添加到数据集...")
landscape_prompts = [x["landscape_prompt"] for x in processed_samples]
processed_ds = train_ds.add_column("landscape_prompt", landscape_prompts)

# 6. 打印处理后的数据集信息
print("\n处理完成！")
print(f"原始数据集大小: {len(train_ds)}")
print(f"处理后数据集大小: {len(processed_ds)}")
print("新增列示例:")
for i in range(min(3, len(processed_ds))):  # 打印前3个样本作为示例
    print(f"\n样本 {i}:")
    print(f"原始prompt: {processed_ds[i]['prompt'][:50]}...")
    print(f"景观prompt: {processed_ds[i]['landscape_prompt'][:50]}...")

# 如果需要保存处理后的数据集
# processed_ds.save_to_disk("processed_dataset")

processed_ds.push_to_hub("svjack/After_Tomorrow_SPLITED_Landscape_Captioned")

git clone https://huggingface.co/spaces/KingNish/SDXL-Flash

import os
from PIL import Image
from gradio_client import Client
from datasets import Dataset, Image as HFImage
from tqdm import tqdm

# 1. 初始化Gradio客户端（注意端口改为7861）
client = Client("http://127.0.0.1:7861")

# 2. 创建保存图片的目录
os.makedirs("landscape_images", exist_ok=True)

# 3. 定义图片生成函数
def generate_and_save_image(prompt, idx):
    try:
        # 调用Gradio API生成图片
        result = client.predict(
    		prompt=prompt,
    		negative_prompt="(deformed, distorted, disfigured:1.3), poorly drawn, bad anatomy, wrong anatomy, extra limb, missing limb, floating limbs, (mutated hands and fingers:1.4), disconnected limbs, mutation, mutated, ugly, disgusting, blurry, amputation, NSFW",
    		use_negative_prompt=True,
    		seed=0,
    		width=1024,
    		height=1024,
    		guidance_scale=3,
    		num_inference_steps=8,
    		randomize_seed=True,
    		api_name="/run"
        )
        # 保存图片到本地
        img_path = f"landscape_images/{idx}.jpg"
        Image.open(result[0][0]["image"]).save(img_path)
        return img_path
    except Exception as e:
        print(f"\n生成图片 {idx} 时出错: {str(e)}")
        return None

# 4. 处理数据集并添加图片列
print("开始生成景观图片...")
image_paths = []

processed_df = processed_ds.to_pandas()

for i in tqdm(range(len(processed_ds)), desc="生成图片进度"):
    prompt = processed_df.iloc[i]["landscape_prompt"]
    img_path = generate_and_save_image(prompt, i)
    image_paths.append(img_path if img_path else "")

# 5. 添加图片路径到数据集
processed_ds = processed_ds.add_column("landscape_image", image_paths)

# 6. 转换为图片类型
processed_ds = processed_ds.cast_column("landscape_image", HFImage())

# 7. 验证结果
print("\n处理完成！")
print(f"成功生成 {len([x for x in image_paths if x])}/{len(processed_ds)} 张图片")
print("数据集结构:", processed_ds)

processed_ds.push_to_hub("svjack/After_Tomorrow_SPLITED_Landscape")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import os
from datasets import load_dataset, Dataset

def find_max_black_rectangle(mask_image):
    """
    找到mask图像中黑色区域的最大内接矩形
    返回: (x, y, w, h) 矩形坐标和宽高
    """
    # 转换为numpy数组
    mask_array = np.array(mask_image)

    # 转换为单通道灰度图像
    if len(mask_array.shape) == 3:
        mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)

    # 二值化处理(黑色区域为255，白色为0)
    _, binary = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY_INV)

    # 查找轮廓(注意OpenCV版本差异)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_rect = (0, 0, 0, 0)  # x, y, w, h

    for contour in contours:
        # 获取边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area > max_area:
            max_area = area
            best_rect = (x, y, w, h)

    return best_rect

def add_text_to_image(base_image, rect, text, font_path="华文琥珀.ttf", color=(255, 255, 0)):
    """
    在图像的指定矩形区域添加居中文字
    """
    x, y, w, h = rect
    draw = ImageDraw.Draw(base_image)

    # 尝试加载字体，自动调整大小
    try:
        # 根据矩形高度计算字体大小，保留20%边距
        font_size = int(min(w * 0.8 / max(1, len(text)), h * 0.8))
        font = ImageFont.truetype(font_path, font_size)
    except:
        print(f"字体 {font_path} 加载失败，使用默认字体")
        font = ImageFont.load_default()
        font_size = int(min(w * 0.8 / max(1, len(text)), h * 0.8))
        try:
            font = font.font_variant(size=font_size)
        except:
            pass

    # 计算文字位置(居中)
    try:
        # 新版Pillow
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except:
        # 旧版Pillow
        text_w, text_h = draw.textsize(text, font=font)

    pos_x = x + (w - text_w) // 2
    pos_y = y + (h - text_h) // 2

    # 添加文字
    draw.text((pos_x, pos_y), text, fill=color, font=font)
    return base_image

def process_dataset(image_dataset_name, caption_dataset_name):
    """
    处理整个数据集，为每张图片添加对应的prompt文本

    参数:
        image_dataset_name: 包含图片和mask的数据集名称，如"svjack/Xiang_Card_DreamO_Images_Filtered_CARD_MASK_RMBG"
        caption_dataset_name: 包含prompt文本的数据集名称，如"svjack/After_Tomorrow_SPLITED_Landscape_Captioned"

    返回:
        包含处理后的图像的新数据集
    """
    # 加载数据集
    image_ds = load_dataset(image_dataset_name)["train"]
    caption_ds = load_dataset(caption_dataset_name)["train"]

    # 确保我们只处理两个数据集中较小的那个大小
    min_len = min(len(image_ds), len(caption_ds))
    image_ds = image_ds.select(range(min_len))
    caption_ds = caption_ds.select(range(min_len))
    caption_df = caption_ds.to_pandas()

    # 处理每张图片
    processed_images = []
    for i in range(min_len):
        im = image_ds[i]["transparent_image"].copy()
        mask_im = image_ds[i]["sign_mask_image"].copy()
        text = caption_df.iloc[i]["prompt"]

        # 找到最大黑色矩形区域
        rect = find_max_black_rectangle(mask_im)

        # 添加文字
        result = add_text_to_image(
            base_image=im,
            rect=rect,
            text=text,
            font_path="华文琥珀.ttf",
            color=(255, 255, 0)  # 黄色
        )

        processed_images.append(result)

    # 创建新数据集
    new_ds = Dataset.from_dict({
        "original_image": [image_ds[i]["transparent_image"] for i in range(min_len)],
        "sign_mask_image": [image_ds[i]["sign_mask_image"] for i in range(min_len)],
        "removed_mask": [image_ds[i]["removed_mask"] for i in range(min_len)],
        "processed_image": processed_images,
        "prompt": [caption_df.iloc[i]["prompt"] for i in range(min_len)]
    })

    return new_ds

if __name__ == "__main__":
    # 设置数据集名称
    image_dataset_name = "svjack/Xiang_Card_DreamO_Images_Filtered_CARD_MASK_RMBG"
    caption_dataset_name = "svjack/After_Tomorrow_SPLITED_Landscape_Captioned"

    # 处理数据集
    processed_dataset = process_dataset(image_dataset_name, caption_dataset_name)

    # 显示第一个结果
    processed_dataset[0]["processed_image"].show()

    # 可以保存处理后的数据集
    # processed_dataset.save_to_disk("processed_dataset")
    #processed_dataset.push_to_hub("svjack/Xiang_Card_DreamO_Images_Filtered_CARD_MASK_RMBG_TEXT")

processed_dataset.push_to_hub("svjack/Xiang_Card_DreamO_After_Tomorrow_SPLITED")

from datasets import load_dataset
from PIL import Image
import os
from datasets import Image as HfImage
import tempfile

# 加载两个数据集
xiang_dataset = load_dataset("svjack/Xiang_Card_DreamO_After_Tomorrow_SPLITED", split="train")
landscape_dataset = load_dataset("svjack/After_Tomorrow_SPLITED_Landscape", split="train")

# 创建临时目录保存合成图像
output_dir = tempfile.mkdtemp()

def process_and_overlay(index):
    # 获取当前样本
    xiang_sample = xiang_dataset[index]
    landscape_sample = landscape_dataset[index]

    # 获取图像路径
    processed_image_path = xiang_sample["processed_image"]
    landscape_image_path = landscape_sample["landscape_image"]

    # 合成图像
    output_path = os.path.join(output_dir, f"composite_{index}.png")

    # 使用PIL打开图像并合成
    background = landscape_image_path.convert("RGBA")
    foreground = processed_image_path.convert("RGBA")

    # 确保前景不大于背景
    if foreground.width > background.width or foreground.height > background.height:
        # 如果需要调整大小，可以取消下面这行注释
        # foreground = foreground.resize((background.width, background.height))
        raise ValueError(f"前景图片尺寸({foreground.size})大于背景图片({background.size})")

    # 合并图层
    combined = Image.alpha_composite(background, foreground)
    combined.save(output_path)

    return output_path

# 为Xiang数据集添加新列
composite_images = []
for i in range(len(xiang_dataset)):
    composite_path = process_and_overlay(i)
    composite_images.append(composite_path)
    '''
    except Exception as e:
        print(f"处理第{i}个样本时出错: {str(e)}")
        composite_images.append(None)
    '''

# 先添加列（原始路径字符串）
xiang_dataset = xiang_dataset.add_column("composite_image_path", composite_images)

# 然后转换列类型
xiang_dataset = xiang_dataset.cast_column("composite_image_path", HfImage())

# 可以选择重命名列（如果需要）
xiang_dataset = xiang_dataset.rename_column("composite_image_path", "composite_image")

print("处理完成！合成图像保存在:", output_dir)
print("Xiang数据集现在包含列:", xiang_dataset.column_names)
print("composite_image列的类型:", type(xiang_dataset[0]["composite_image"]))

xiang_dataset.push_to_hub("svjack/Xiang_Card_DreamO_After_Tomorrow_SPLITED_composite")

from datasets import load_dataset
from PIL import Image
import os
from datasets import Image as HfImage
import tempfile
from gradio_client import Client, handle_file

# 加载两个数据集
xiang_dataset = load_dataset("svjack/Xiang_Card_DreamO_After_Tomorrow_SPLITED", split="train")
landscape_dataset = load_dataset("svjack/After_Tomorrow_SPLITED_Landscape", split="train")

# 创建临时目录保存合成图像
output_dir = tempfile.mkdtemp()

# Initialize Gradio client
client = Client("http://localhost:7861/")

def process_and_overlay(index):
    # 获取当前样本
    xiang_sample = xiang_dataset[index]
    landscape_sample = landscape_dataset[index]

    # 创建临时文件保存图像
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as fg_temp:
        xiang_sample["original_image"].save(fg_temp.name)
        fg_path = fg_temp.name

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as bg_temp:
        landscape_sample["landscape_image"].save(bg_temp.name)
        bg_path = bg_temp.name

    # 使用Gradio API合成图像
    try:
        result = client.predict(
            fg_image=handle_file(fg_path),
            bg_image=handle_file(bg_path),
            num_sampling_steps=10,
            api_name="/evaluate"
        )

        # 保存结果图像
        output_path = os.path.join(output_dir, f"composite_{index}.png")
        Image.open(result[1]).save(output_path)

        # 清理临时文件
        os.unlink(fg_path)
        os.unlink(bg_path)

        return output_path
    except Exception as e:
        print(f"处理第{index}个样本时出错: {str(e)}")
        # 清理临时文件
        os.unlink(fg_path)
        os.unlink(bg_path)
        return None

# 为Xiang数据集添加新列
composite_images = []
for i in range(len(xiang_dataset)):
    composite_path = process_and_overlay(i)
    composite_images.append(composite_path)

# 先添加列（原始路径字符串）
xiang_dataset = xiang_dataset.add_column("composite_image_path", composite_images)

# 然后转换列类型
xiang_dataset = xiang_dataset.cast_column("composite_image_path", HfImage())

# 可以选择重命名列（如果需要）
xiang_dataset = xiang_dataset.rename_column("composite_image_path", "relight_image")

print("处理完成！合成图像保存在:", output_dir)
print("Xiang数据集现在包含列:", xiang_dataset.column_names)
print("composite_image列的类型:", type(xiang_dataset[0]["relight_image"]))

xiang_dataset.push_to_hub("svjack/Xiang_Card_DreamO_After_Tomorrow_SPLITED_LBM_Relight")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
from datasets import Dataset

def find_max_black_rectangle(mask_image):
    """
    找到mask图像中黑色区域的最大内接矩形
    返回: (x, y, w, h) 矩形坐标和宽高
    """
    # 转换为numpy数组
    mask_array = np.array(mask_image)

    # 转换为单通道灰度图像
    if len(mask_array.shape) == 3:
        mask_array = cv2.cvtColor(mask_array, cv2.COLOR_RGB2GRAY)

    # 二值化处理(黑色区域为255，白色为0)
    _, binary = cv2.threshold(mask_array, 127, 255, cv2.THRESH_BINARY_INV)

    # 查找轮廓(注意OpenCV版本差异)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    best_rect = (0, 0, 0, 0)  # x, y, w, h

    for contour in contours:
        # 获取边界矩形
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        if area > max_area:
            max_area = area
            best_rect = (x, y, w, h)

    return best_rect

def add_text_to_image(base_image, rect, text, font_path="华文琥珀.ttf", color=(255, 255, 0)):
    """
    在图像的指定矩形区域添加居中文字
    """
    x, y, w, h = rect
    draw = ImageDraw.Draw(base_image)

    # 尝试加载字体，自动调整大小
    try:
        # 根据矩形高度计算字体大小，保留20%边距
        font_size = int(min(w * 0.8 / max(1, len(text)), h * 0.8))
        font = ImageFont.truetype(font_path, font_size)
    except:
        print(f"字体 {font_path} 加载失败，使用默认字体")
        font = ImageFont.load_default()
        font_size = int(min(w * 0.8 / max(1, len(text)), h * 0.8))
        try:
            font = font.font_variant(size=font_size)
        except:
            pass

    # 计算文字位置(居中)
    try:
        # 新版Pillow
        bbox = draw.textbbox((0, 0), text, font=font)
        text_w = bbox[2] - bbox[0]
        text_h = bbox[3] - bbox[1]
    except:
        # 旧版Pillow
        text_w, text_h = draw.textsize(text, font=font)

    pos_x = x + (w - text_w) // 2
    pos_y = y + (h - text_h) // 2

    # 添加文字
    draw.text((pos_x, pos_y), text, fill=color, font=font)
    return base_image

def process_xiang_dataset(xiang_dataset):
    """
    处理xiang_dataset，为每张图片添加对应的prompt文本

    参数:
        xiang_dataset: 包含relight_image, sign_mask_image和prompt的数据集

    返回:
        包含处理后的图像的新数据集
    """
    processed_images = []

    for i in range(len(xiang_dataset)):
        im = xiang_dataset[i]["relight_image"].copy()
        mask_im = xiang_dataset[i]["sign_mask_image"].copy()
        text = xiang_dataset[i]["prompt"]

        # 找到最大黑色矩形区域
        rect = find_max_black_rectangle(mask_im)

        # 添加文字
        result = add_text_to_image(
            base_image=im,
            rect=rect,
            text=text,
            font_path="华文琥珀.ttf",
            color=(255, 255, 0)  # 黄色
        )

        processed_images.append(result)

    # 创建新数据集（保留原有列并添加processed_image列）
    new_dict = {col: xiang_dataset[col] for col in xiang_dataset.column_names}
    new_dict["processed_image"] = processed_images

    new_ds = Dataset.from_dict(new_dict)

    return new_ds

# 使用示例
if __name__ == "__main__":
    from datasets import load_dataset

    # 加载数据集
    xiang_dataset = load_dataset("svjack/Xiang_Card_DreamO_After_Tomorrow_SPLITED_LBM_Relight", split="train")

    # 处理数据集
    processed_dataset = process_xiang_dataset(xiang_dataset)

    # 显示第一个结果
    processed_dataset[0]["processed_image"].show()

    # 可以保存处理后的数据集
    # processed_dataset.push_to_hub("your_hub_repo_name")

processed_dataset.push_to_hub("svjack/Xiang_Card_DreamO_After_Tomorrow_SPLITED_LBM_Relight_SIGN")

sudo apt-get update && sudo apt-get install cbm ffmpeg git-lfs

from datasets import load_dataset
import os
from PIL import Image
import io

# 加载数据集
ds = load_dataset("svjack/Xiang_Card_DreamO_After_Tomorrow_SPLITED_LBM_Relight_SIGN_Adjust")

# 创建保存目录
output_dir = "adjusted_images"
os.makedirs(output_dir, exist_ok=True)

# 保存adjust_image列
for idx, item in enumerate(ds["train"]):
    if "adjust_image" in item:
        image = item["adjust_image"]

        # 如果图像是字节格式
        if isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        # 如果图像已经是PIL.Image格式
        elif hasattr(image, "save"):
            img = image
        else:
            continue

        # 按字典序命名 (使用索引确保唯一性)
        filename = f"adjusted_{idx:05d}.png"
        img.save(os.path.join(output_dir, filename))

print(f"所有adjust_image已保存到 {output_dir} 目录")

from datasets import load_dataset
import os
from PIL import Image
import io

# 加载数据集
ds = load_dataset("svjack/Xiang_Card_DreamO_After_Tomorrow_SPLITED_composite")

# 创建保存目录
output_dir = "composite_images"
os.makedirs(output_dir, exist_ok=True)

# 保存composite_image列
for idx, item in enumerate(ds["train"]):
    if "composite_image" in item:
        image = item["composite_image"]

        # 如果图像是字节格式
        if isinstance(image, bytes):
            img = Image.open(io.BytesIO(image))
        # 如果图像已经是PIL.Image格式
        elif hasattr(image, "save"):
            img = image
        else:
            continue

        # 按字典序命名 (使用索引确保唯一性)
        filename = f"composite_{idx:05d}.png"
        img.save(os.path.join(output_dir, filename))

print(f"所有composite_image已保存到 {output_dir} 目录")

adjusted_images

composite_image

使用下面的api 对 adjusted_images 路径下的所有视频进行遍历

from gradio_client import Client, handle_file

client = Client("http://localhost:7860")
result = client.predict(
		input_image=handle_file('adjusted_images/adjusted_00000.png'),
		input_mask=handle_file('white_background.png'),
		prompt="a man hold a sign in dynamic landscape.",
		t2v=False,
		n_prompt="",
		seed=31337,
		total_second_length=4,
		latent_window_size=9,
		steps=25,
		cfg=1,
		gs=10,
		rs=0,
		gpu_memory_preservation=6,
		use_teacache=True,
		mp4_crf=16,
		api_name="/process"
)
print(result)

from shutil import copy2
copy2(result[0]["video"], result[0]["video"].split("/")[-1])

将结果保存到一个新的输出路径 并保持输出文件名与图片名称相同，但为mp4 文件 tqdm 打印流程

vim run_adj.py

from gradio_client import Client, handle_file
import os
from pathlib import Path
from tqdm import tqdm
import time

# Initialize client
client = Client("http://localhost:7860")

# Directories
input_dir = 'adjusted_images'
output_dir = 'adjusted_images_mp4'
mask_path = 'white_background.png'

# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Get all PNG files in input directory
input_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
input_files = [os.path.join(input_dir, f) for f in input_files]

# Process each file with progress tracking
for input_path in tqdm(input_files, desc="Processing images"):
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.mp4")

    # Skip if output already exists
    if os.path.exists(output_path):
        tqdm.write(f"Skipping {input_path} - output already exists")
        continue

    # Process with individual progress bar
    with tqdm(total=100, desc=f"Generating {base_name}", leave=False) as pbar:
        # Wrap the prediction in a function to update progress
        def progress_callback(progress_data):
            if isinstance(progress_data, dict) and 'progress' in progress_data:
                pbar.update(progress_data['progress'] - pbar.n)

        try:
            result = client.predict(
                input_image=handle_file(input_path),
                input_mask=handle_file(mask_path),
                prompt="a slient quiet smile man hold a sign in dynamic landscape.",
                t2v=False,
                n_prompt="singing, say, open mouth",
                seed=31337,
                total_second_length=4,
                latent_window_size=9,
                steps=25,
                cfg=1,
                gs=10,
                rs=0,
                gpu_memory_preservation=6,
                use_teacache=True,
                mp4_crf=16,
                api_name="/process"
            )

            # Get the generated video path
            generated_video = result[0]["video"]

            # Copy to our output directory with consistent naming
            os.rename(generated_video, output_path)

            tqdm.write(f"Successfully processed {input_path} -> {output_path}")

        except Exception as e:
            tqdm.write(f"Error processing {input_path}: {str(e)}")
            continue

print("All files processed!")

from gradio_client import Client, handle_file
import os
from pathlib import Path
from tqdm import tqdm
import time

# Initialize client
client = Client("http://localhost:7860")

# Directories
input_dir = 'composite_images'
output_dir = 'composite_images_mp4'
mask_path = 'white_background.png'

# Create output directory if it doesn't exist
Path(output_dir).mkdir(parents=True, exist_ok=True)

# Get all PNG files in input directory
input_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.png')]
input_files = [os.path.join(input_dir, f) for f in input_files]

# Process each file with progress tracking
for input_path in tqdm(input_files, desc="Processing images"):
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    output_path = os.path.join(output_dir, f"{base_name}.mp4")

    # Skip if output already exists
    if os.path.exists(output_path):
        tqdm.write(f"Skipping {input_path} - output already exists")
        continue

    # Process with individual progress bar
    with tqdm(total=100, desc=f"Generating {base_name}", leave=False) as pbar:
        # Wrap the prediction in a function to update progress
        def progress_callback(progress_data):
            if isinstance(progress_data, dict) and 'progress' in progress_data:
                pbar.update(progress_data['progress'] - pbar.n)

        try:
            result = client.predict(
                input_image=handle_file(input_path),
                input_mask=handle_file(mask_path),
                prompt="a slient quiet smile man hold a sign in dynamic landscape.",
                t2v=False,
                n_prompt="singing, say, open mouth",
                seed=31337,
                total_second_length=4,
                latent_window_size=9,
                steps=25,
                cfg=1,
                gs=10,
                rs=0,
                gpu_memory_preservation=6,
                use_teacache=True,
                mp4_crf=16,
                api_name="/process"
            )

            # Get the generated video path
            generated_video = result[0]["video"]

            # Copy to our output directory with consistent naming
            os.rename(generated_video, output_path)

            tqdm.write(f"Successfully processed {input_path} -> {output_path}")

        except Exception as e:
            tqdm.write(f"Error processing {input_path}: {str(e)}")
            continue

print("All files processed!")


https://huggingface.co/spaces/fffiloni/LatentSync
