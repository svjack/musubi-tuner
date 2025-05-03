```python
### git clone https://huggingface.co/datasets/svjack/Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-SP

import os
import glob
import shutil
from tqdm import tqdm
import subprocess

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

# 全局变量记录每个角色文件夹已处理的图片
processed_images_per_character = {}

def get_latest_mp4(save_dir):
    """获取save目录下最新创建的mp4文件"""
    mp4_files = glob.glob(os.path.join(save_dir, '*.mp4'))
    if not mp4_files:
        return None
    return max(mp4_files, key=os.path.getctime)

def get_chinese_name_from_folder(folder_name):
    """从文件夹名中提取中文角色名"""
    parts = folder_name.split('_')
    #print(parts)
    if len(parts) >= 3:
        english_parts = parts[2:-2] if parts[-2] == 'images' and parts[-1] == 'and_texts' else parts[2:]
        english_name = ' '.join(english_parts[:-3]).replace('-', ' ').strip()

        #print(english_parts, english_name)
        ### ['genshin', 'impact', 'KAMISATO', 'AYAKA', 'images', 'and', 'texts']
        ### ['KAMISATO', 'AYAKA', 'images', 'and', 'texts'] KAMISATO AYAKA images and texts
        
        for chinese, english in name_mapping.items():
            if english.upper() == english_name.upper():
                return chinese
    return None

def process_images_and_texts():
    global processed_images_per_character
    
    input_root_dir = "Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-SP"
    output_root_dir = "Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-SP_Processed"
    save_dir = "save"

    os.makedirs(output_root_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    # 获取所有角色文件夹并按字母顺序排序
    character_folders = sorted([d for d in os.listdir(input_root_dir) 
                              if os.path.isdir(os.path.join(input_root_dir, d)) 
                              and d.startswith('genshin_impact_')])

    # 初始化已处理图片记录
    if not processed_images_per_character:
        processed_images_per_character = {folder: set() for folder in character_folders}

    # 遍历每个角色文件夹
    for folder in tqdm(character_folders, desc="Processing character folders"):
        chinese_name = get_chinese_name_from_folder(folder)        
        if not chinese_name:
            print(f"Warning: Could not determine Chinese name for folder {folder}")
            continue
            
        input_dir = os.path.join(input_root_dir, folder)
        
        # 获取该角色文件夹下所有未处理的图片
        png_files_0 = glob.glob(os.path.join(input_dir, '*.png'))
        png_files_1 = glob.glob(os.path.join(input_dir, '*.jpg'))
        png_files_2 = glob.glob(os.path.join(input_dir, '*.jpeg'))
        png_files_3 = glob.glob(os.path.join(input_dir, '*.webp'))
        all_images = list(png_files_0) + list(png_files_1) + list(png_files_2) + list(png_files_3)
        
        # 过滤掉已处理的图片
        unprocessed_images = [img for img in all_images if img not in processed_images_per_character[folder]]
        
        if not unprocessed_images:
            print(f"No unprocessed images found in {folder}")
            continue
            
        # 按文件名排序后取第一个未处理的图片
        selected_png = sorted(unprocessed_images)[0]
        
        # 记录已处理的图片
        processed_images_per_character[folder].add(selected_png)
        
        prompt = """The camera smoothly orbits around the center of the scene, keeping the center point fixed and always in view."""

        cmd = [
            "python", "fpack_generate_video.py",
            "--dit", "FramePackI2V_HY/diffusion_pytorch_model-00001-of-00003.safetensors",
            "--vae", "HunyuanVideo/vae/diffusion_pytorch_model.safetensors",
            "--text_encoder1", "HunyuanVideo_repackaged/split_files/text_encoders/llava_llama3_fp16.safetensors",
            "--text_encoder2", "HunyuanVideo_repackaged/split_files/text_encoders/clip_l.safetensors",
            "--image_encoder", "sigclip_vision_384/sigclip_vision_patch14_384.safetensors",
            "--image_path", selected_png,
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

        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error processing {selected_png}: {e}")
            continue

        latest_mp4 = get_latest_mp4(save_dir)
        if latest_mp4 is None:
            print(f"Warning: No .mp4 file generated for {selected_png}")
            continue

        base_name = os.path.splitext(os.path.basename(selected_png))[0]
        output_dir = os.path.join(output_root_dir, folder)
        os.makedirs(output_dir, exist_ok=True)
        
        output_mp4 = os.path.join(output_dir, f"{chinese_name}_{base_name}.mp4")
        output_txt = os.path.join(output_dir, f"{chinese_name}_{base_name}.txt")

        txt_file = os.path.join(input_dir, f"{base_name}.txt")
        if not os.path.exists(txt_file):
            txt_file = os.path.join(input_dir, f"{base_name}.caption")
            if not os.path.exists(txt_file):
                print(f"Warning: No corresponding text file found for {selected_png}")
                continue

        shutil.move(latest_mp4, output_mp4)
        shutil.copy2(txt_file, output_txt)

if __name__ == "__main__":
    # 可以在这里加载之前处理过的图片记录
    # if os.path.exists("processed_images_per_character.pkl"):
    #     with open("processed_images_per_character.pkl", "rb") as f:
    #         processed_images_per_character = pickle.load(f)
    
    process_images_and_texts()
    
    # 可以在这里保存处理过的图片记录
    # with open("processed_images_per_character.pkl", "wb") as f:
    #     pickle.dump(processed_images_per_character, f)
    
    print("All processing completed!")

```
