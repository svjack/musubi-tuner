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
    '托马': 'THOMA', '雷泽': 'RAZOR'
}

gender_mapping = {
    '久岐忍': '女', '云堇': '女', '五郎': '男', '优菈': '女', '凝光': '女', '凯亚': '男',
    '安柏': '女', '宵宫': '女', '温迪': '男', '烟绯': '女', '珊瑚宫心海': '女', '琴': '女',
    '甘雨': '女', '申鹤': '女', '砂糖': '女', '神里绫人': '男', '神里绫华': '女', '绮良良': '女',
    '罗莎莉亚': '女', '胡桃': '女', '艾尔海森': '男', '荒泷一斗': '男', '行秋': '男', '诺艾尔': '女',
    '迪卢克': '男', '迪奥娜': '女', '重云': '男', '钟离': '男', '阿贝多': '男', '雷泽': '男',
    '雷电将军': '女', '魈': '男', '鹿野院平藏': '男', '芭芭拉': '女', '柯莱': '女', '八重神子': '女',
    '妮露': '女', '珐露珊': '女', '琳妮特': '女', '纳西妲': '女', '枫原万叶': '男', '芙宁娜': '女',
    '林尼': '男', '赛诺': '男', '七七': '女', '托马': '男'
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
        '[比起XXX我更喜欢你]': '比起处理政务，我更喜欢你（反正你也处理不完)'
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
    },
    '芭芭拉': {
        '[对角色的称呼]': '祈礼牧师（但治疗主要靠歌声）',
        '[比起XXX我更喜欢你]': '比起治疗，我更喜欢你（反正你也听不够）'
    },
    '柯莱': {
        '[对角色的称呼]': '见习巡林员（但巡林主要靠运气）',
        '[比起XXX我更喜欢你]': '比起巡林，我更喜欢你（反正你也找不到路）'
    },
    '八重神子': {
        '[对角色的称呼]': '宫司大人（但工作主要靠摸鱼）',
        '[比起XXX我更喜欢你]': '比起处理神社事务，我更喜欢你（反正你也管不了我）'
    },
    '妮露': {
        '[对角色的称呼]': '舞者（但跳舞主要靠即兴）',
        '[比起XXX我更喜欢你]': '比起跳舞，我更喜欢你（反正你也跟不上节奏）'
    },
    '珐露珊': {
        '[对角色的称呼]': '学者（但研究主要靠脑补）',
        '[比起XXX我更喜欢你]': '比起研究，我更喜欢你（反正你也看不懂论文）'
    },
    '琳妮特': {
        '[对角色的称呼]': '魔术师助手（但魔术主要靠障眼法）',
        '[比起XXX我更喜欢你]': '比起魔术表演，我更喜欢你（反正你也看不穿）'
    },
    '纳西妲': {
        '[对角色的称呼]': '小吉祥草王（但治国主要靠做梦）',
        '[比起XXX我更喜欢你]': '比起治理须弥，我更喜欢你（反正你也醒不来）'
    },
    '枫原万叶': {
        '[对角色的称呼]': '浪人武士（但战斗主要靠听风）',
        '[比起XXX我更喜欢你]': '比起听风，我更喜欢你（反正你也听不见）'
    },
    '芙宁娜': {
        '[对角色的称呼]': '水神（但审判主要靠演技）',
        '[比起XXX我更喜欢你]': '比起审判，我更喜欢你（反正你也看不穿）'
    },
    '林尼': {
        '[对角色的称呼]': '魔术师（但魔术主要靠机关）',
        '[比起XXX我更喜欢你]': '比起魔术，我更喜欢你（反正你也找不到机关）'
    },
    '赛诺': {
        '[对角色的称呼]': '风纪官（但执法主要靠冷笑话）',
        '[比起XXX我更喜欢你]': '比起执法，我更喜欢你（反正你也笑不出来）'
    },
    '七七': {
        '[对角色的称呼]': '僵尸采药童（但记性不太好）',
        '[比起XXX我更喜欢你]': '比起采药，我更喜欢你（反正你也记不住）'
    },
    '托马': {
        '[对角色的称呼]': '家政官（但家政范围包括外交）',
        '[比起XXX我更喜欢你]': '比起打扫，我更喜欢你（反正你也拦不住我）'
    }
}

# 路径配置
paths = {
    'female_audios': 'I_prefer_you_over_something_GIRL_AUDIOS_SPLITED',
    'male_audios': 'I_prefer_you_over_something_BOY_AUDIOS_SPLITED',
    'images': 'Genshin-Impact-Portrait-with-Tags-Filtered-IID-Gender-SP',
    'videos': 'Genshin_Impact_Animagine_Xl_Portrait_FramePack_Rotate_Named'
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
    
    # 确保角色不重复
    used_chars = set()
    selected_combinations = []
    
    for combo in all_combinations:
        if len(used_chars.intersection(combo)) == 0:
            selected_combinations.append(combo)
            used_chars.update(combo)
            if len(selected_combinations) >= num_groups:
                break
    
    # 如果不够，随机添加剩余角色
    if len(selected_combinations) < num_groups:
        remaining_chars = [c for c in chars if c not in used_chars]
        while len(remaining_chars) >= 3 and len(selected_combinations) < num_groups:
            combo = tuple(random.sample(remaining_chars, 3))
            selected_combinations.append(combo)
            remaining_chars = [c for c in remaining_chars if c not in combo]
    
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

generate_videos('女', 128, output_dir = "Genshin_Impact_Girls_XL_prefer_you_over_OTHERS")

generate_videos('男', 128, output_dir = "Genshin_Impact_Boys_XL_prefer_you_over_OTHERS")
```
