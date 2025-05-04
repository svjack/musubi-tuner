#### MusicToImVideo_step.py

1、音乐源

先使用歌词搜索功能 搜索歌曲 和对应的 srt 文件
https://github.com/jitwxs/163MusicLyrics

查看歌曲是否可下载
https://github.com/gengark/netease-cloud-music-download

wyy dl "https://music.163.com/song?id=1941990933"

从而得到 对应的
.mp3 文件和 对应的 srt 文件

2、分割

位于某文件夹下 的同名 (.mp3, .srt) 文件对儿

分割单个 对儿 数据集格式的代码

'''
python run_srt_split.py "天若有情 - 杜宣达.mp3" "天若有情 - 杜宣达.srt" "天若有情"
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

Day if sentient beings

sudo apt-get update && sudo apt-get install cbm git-lfs ffmpeg

pip install datasets huggingface_hub moviepy==1.0.3 "httpx[socks]" tabulate pydub

huggingface-cli download --repo-type dataset svjack/Day_if_sentient_beings_SPLITED --local-dir ./Day_if_sentient_beings_SPLITED

import pathlib
import pandas as pd

def r_func(txt_path):
    with open(txt_path, "r", encoding="utf-8") as f:
        return f.read().strip()

def generate_metadata(input_dir):
    # 创建Path对象并标准化路径
    input_path = pathlib.Path(input_dir).resolve()

    # 收集所有视频和文本文件
    file_list = []
    for file_path in input_path.rglob("*"):
        if file_path.suffix.lower() in ('.mp3', '.txt'):
            file_list.append({
                "stem": file_path.stem,
                "path": file_path,
                "type": "video" if file_path.suffix.lower() == '.mp3' else "text"
            })

    # 创建DataFrame并分组处理
    df = pd.DataFrame(file_list)
    grouped = df.groupby('stem')

    metadata = []
    for stem, group in grouped:
        # 获取组内文件
        videos = group[group['type'] == 'video']
        texts = group[group['type'] == 'text']

        # 确保每组有且只有一个视频和一个文本文件
        if len(videos) == 1 and len(texts) == 1:
            video_path = videos.iloc[0]['path']
            text_path = texts.iloc[0]['path']

            metadata.append({
                "file_name": video_path.name,  # 自动处理不同系统的文件名
                "prompt": r_func(text_path)
            })

    # 保存结果到CSV
    output_path = input_path.parent / "metadata.csv"
    pd.DataFrame(metadata).to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"Metadata generated at: {output_path}")

'''
---
configs:
- config_name: default
  data_files:
  - split: train
    path:
    - "*.mp3"
    - "metadata.csv"
---
'''

generate_metadata("Day_if_sentient_beings_SPLITED")

huggingface-cli login
huggingface-cli upload svjack/Day_if_sentient_beings_SPLITED Day_if_sentient_beings_SPLITED --repo-type dataset

3、重构场景规则

import pandas as pd
import os
root_path = "Day_if_sentient_beings_SPLITED"
df = pd.read_csv("{}/metadata.csv".format(root_path))
# 添加时长列
df['duration_sec'] = df['file_name'].apply(
    lambda x: len(AudioSegment.from_file(os.path.join(root_path, x))) / 1000.0
)
print(df.to_markdown())

|    | file_name                  | prompt                                   |   duration_sec |
|---:|:---------------------------|:-----------------------------------------|---------------:|
|  0 | 0001_天若有情 - 杜宣达.mp3 | 作词 : 纪如璟                            |           1    |
|  1 | 0002_天若有情 - 杜宣达.mp3 | 作曲 : 黄义达                            |           7.31 |
|  2 | 0003_天若有情 - 杜宣达.mp3 | 出品：网易飓风 X 网易青云                |           4.14 |
|  3 | 0004_天若有情 - 杜宣达.mp3 | 风扬起时繁花落尽                         |           5.88 |
|  4 | 0005_天若有情 - 杜宣达.mp3 | 谁执笔为你绘丹青                         |           6.18 |
|  5 | 0006_天若有情 - 杜宣达.mp3 | 月下独影泪湿青衣                         |           6.03 |
|  6 | 0007_天若有情 - 杜宣达.mp3 | 流水不付一世深情                         |           6.03 |
|  7 | 0008_天若有情 - 杜宣达.mp3 | 只身回望太匆匆                           |           3.12 |
|  8 | 0009_天若有情 - 杜宣达.mp3 | 此生多少情与仇                           |           3.75 |
|  9 | 0010_天若有情 - 杜宣达.mp3 | 只愿与你长相守                           |           5.34 |
| 10 | 0011_天若有情 - 杜宣达.mp3 | 无边丝雨细如愁                           |           3.06 |
| 11 | 0012_天若有情 - 杜宣达.mp3 | 朝来寒雨几回眸                           |           3.84 |
| 12 | 0013_天若有情 - 杜宣达.mp3 | 你在哪一方停留                           |           5.22 |
| 13 | 0014_天若有情 - 杜宣达.mp3 | 天若有情亦无情                           |           5.85 |
| 14 | 0015_天若有情 - 杜宣达.mp3 | 爱到最后要分离                           |           5.97 |
| 15 | 0016_天若有情 - 杜宣达.mp3 | 你轮回的印记落在我眉宇                   |           6.39 |
| 16 | 0017_天若有情 - 杜宣达.mp3 | 直到有一天不能呼吸                       |          24.42 |
| 17 | 0018_天若有情 - 杜宣达.mp3 | 月下独影泪湿青衣                         |           6.06 |
| 18 | 0019_天若有情 - 杜宣达.mp3 | 流水不付一世深情                         |           6    |
| 19 | 0020_天若有情 - 杜宣达.mp3 | 只身回望太匆匆                           |           3.03 |
| 20 | 0021_天若有情 - 杜宣达.mp3 | 此生多少情与仇                           |           3.81 |
| 21 | 0022_天若有情 - 杜宣达.mp3 | 只愿与你长相守                           |           5.43 |
| 22 | 0023_天若有情 - 杜宣达.mp3 | 无边丝雨细如愁                           |           2.97 |
| 23 | 0024_天若有情 - 杜宣达.mp3 | 朝来寒雨几回眸                           |           3.81 |
| 24 | 0025_天若有情 - 杜宣达.mp3 | 你在哪一方停留                           |           5.28 |
| 25 | 0026_天若有情 - 杜宣达.mp3 | 天若有情亦无情                           |           5.76 |
| 26 | 0027_天若有情 - 杜宣达.mp3 | 爱到最后要分离                           |           6.09 |
| 27 | 0028_天若有情 - 杜宣达.mp3 | 你轮回的印记落在我眉宇                   |           6.33 |
| 28 | 0029_天若有情 - 杜宣达.mp3 | 直到有一天不能呼吸                       |           6.09 |
| 29 | 0030_天若有情 - 杜宣达.mp3 | 天若有情亦无情                           |           5.79 |
| 30 | 0031_天若有情 - 杜宣达.mp3 | 万丈红尘我等你                           |           6.45 |
| 31 | 0032_天若有情 - 杜宣达.mp3 | 用你的牵挂染尽我白发                     |           6    |
| 32 | 0033_天若有情 - 杜宣达.mp3 | 咫尺天涯你终未远离                       |           1.95 |
| 33 | 0034_天若有情 - 杜宣达.mp3 | 制作人 Producer：王圆坤                  |           0.66 |
| 34 | 0035_天若有情 - 杜宣达.mp3 | 编曲 Arranger：任斌                      |           0.72 |
| 35 | 0036_天若有情 - 杜宣达.mp3 | 吉他 Guitar：吴家裕                      |           0.6  |
| 36 | 0037_天若有情 - 杜宣达.mp3 | 和声 Backing Vocals：潘斯贝              |           0.69 |
| 37 | 0038_天若有情 - 杜宣达.mp3 | 混音工程师 Mixing：郑昊杰                |           0.78 |
| 38 | 0039_天若有情 - 杜宣达.mp3 | 母带工程师 Master：郑昊杰                |          10.44 |
| 39 | 0040_天若有情 - 杜宣达.mp3 | 企划：王嘉晟                             |           0.36 |
| 40 | 0041_天若有情 - 杜宣达.mp3 | 统筹：陈尚禔/黄路欢/ELANUS               |           0.27 |
| 41 | 0042_天若有情 - 杜宣达.mp3 | 监制：王嘉晟                             |           0.36 |
| 42 | 0043_天若有情 - 杜宣达.mp3 | 营销推广：网易飓风                       |           0.57 |
| 43 | 0044_天若有情 - 杜宣达.mp3 | 出品人：谢奇笛 X 唐晶晶                  |           1.38 |
| 44 | 0045_天若有情 - 杜宣达.mp3 | OP/ SP：索尼音乐版权代理（北京）有限公司 |           1.05 |
| 45 | 0046_天若有情 - 杜宣达.mp3 | 【此版本为正式授权翻唱作品】             |           1    |
| 46 | 0047_天若有情 - 杜宣达.mp3 | 原唱 : 黄丽玲                            |           1.73 |

这是一个对于 歌曲进行分割 的 mp3 文件和对应的字幕 和秒数
要求你 根据对这些信息的理解 将 不同的部分结合为不同场景
每个场景包含若干个 mp3 片段的连接
并给出每个场景对应的主题
给出合适的python 数据结构作为结果解决我这个问题
要求：
1、除了主歌、副歌以外的信息部分各个合并 主歌、副歌部分每行都是独立场景
2、你要注意 segments 的单位必须是相连的 不相连的部分 要分割开
3、按照音乐播放顺序描述

得到的结果仿造下面的格式

scenes = [
    {
        "theme": "歌曲基本信息",
        "segments": [
            {"file_name": "0001_天若有情 - 杜宣达.mp3", "prompt": "作词 : 纪如璟", "duration_sec": 1},
            {"file_name": "0002_天若有情 - 杜宣达.mp3", "prompt": "作曲 : 黄义达", "duration_sec": 7.31},
            {"file_name": "0003_天若有情 - 杜宣达.mp3", "prompt": "出品：网易飓风 X 网易青云", "duration_sec": 4.14}
        ]
    },
    {
        "theme": "主歌1",
        "segments": [
            {"file_name": "0004_天若有情 - 杜宣达.mp3", "prompt": "风扬起时繁花落尽", "duration_sec": 5.88},
            {"file_name": "0005_天若有情 - 杜宣达.mp3", "prompt": "谁执笔为你绘丹青", "duration_sec": 6.18},
            {"file_name": "0006_天若有情 - 杜宣达.mp3", "prompt": "月下独影泪湿青衣", "duration_sec": 6.03},
            {"file_name": "0007_天若有情 - 杜宣达.mp3", "prompt": "流水不付一世深情", "duration_sec": 6.03}
        ]
    },
    {
        "theme": "副歌1",
        "segments": [
            {"file_name": "0008_天若有情 - 杜宣达.mp3", "prompt": "只身回望太匆匆", "duration_sec": 3.12},
            {"file_name": "0009_天若有情 - 杜宣达.mp3", "prompt": "此生多少情与仇", "duration_sec": 3.75},
            {"file_name": "0010_天若有情 - 杜宣达.mp3", "prompt": "只愿与你长相守", "duration_sec": 5.34},
            {"file_name": "0011_天若有情 - 杜宣达.mp3", "prompt": "无边丝雨细如愁", "duration_sec": 3.06},
            {"file_name": "0012_天若有情 - 杜宣达.mp3", "prompt": "朝来寒雨几回眸", "duration_sec": 3.84},
            {"file_name": "0013_天若有情 - 杜宣达.mp3", "prompt": "你在哪一方停留", "duration_sec": 5.22},
            {"file_name": "0014_天若有情 - 杜宣达.mp3", "prompt": "天若有情亦无情", "duration_sec": 5.85},
            {"file_name": "0015_天若有情 - 杜宣达.mp3", "prompt": "爱到最后要分离", "duration_sec": 5.97},
            {"file_name": "0016_天若有情 - 杜宣达.mp3", "prompt": "你轮回的印记落在我眉宇", "duration_sec": 6.39},
            {"file_name": "0017_天若有情 - 杜宣达.mp3", "prompt": "直到有一天不能呼吸", "duration_sec": 24.42}
        ]
    },
    {
        "theme": "主歌2（重复）",
        "segments": [
            {"file_name": "0018_天若有情 - 杜宣达.mp3", "prompt": "月下独影泪湿青衣", "duration_sec": 6.06},
            {"file_name": "0019_天若有情 - 杜宣达.mp3", "prompt": "流水不付一世深情", "duration_sec": 6.00}
        ]
    },
    {
        "theme": "副歌2（重复）",
        "segments": [
            {"file_name": "0020_天若有情 - 杜宣达.mp3", "prompt": "只身回望太匆匆", "duration_sec": 3.03},
            {"file_name": "0021_天若有情 - 杜宣达.mp3", "prompt": "此生多少情与仇", "duration_sec": 3.81},
            {"file_name": "0022_天若有情 - 杜宣达.mp3", "prompt": "只愿与你长相守", "duration_sec": 5.43},
            {"file_name": "0023_天若有情 - 杜宣达.mp3", "prompt": "无边丝雨细如愁", "duration_sec": 2.97},
            {"file_name": "0024_天若有情 - 杜宣达.mp3", "prompt": "朝来寒雨几回眸", "duration_sec": 3.81},
            {"file_name": "0025_天若有情 - 杜宣达.mp3", "prompt": "你在哪一方停留", "duration_sec": 5.28},
            {"file_name": "0026_天若有情 - 杜宣达.mp3", "prompt": "天若有情亦无情", "duration_sec": 5.76},
            {"file_name": "0027_天若有情 - 杜宣达.mp3", "prompt": "爱到最后要分离", "duration_sec": 6.09},
            {"file_name": "0028_天若有情 - 杜宣达.mp3", "prompt": "你轮回的印记落在我眉宇", "duration_sec": 6.33},
            {"file_name": "0029_天若有情 - 杜宣达.mp3", "prompt": "直到有一天不能呼吸", "duration_sec": 6.09}
        ]
    },
    {
        "theme": "结尾升华",
        "segments": [
            {"file_name": "0030_天若有情 - 杜宣达.mp3", "prompt": "天若有情亦无情", "duration_sec": 5.79},
            {"file_name": "0031_天若有情 - 杜宣达.mp3", "prompt": "万丈红尘我等你", "duration_sec": 6.45},
            {"file_name": "0032_天若有情 - 杜宣达.mp3", "prompt": "用你的牵挂染尽我白发", "duration_sec": 6.00},
            {"file_name": "0033_天若有情 - 杜宣达.mp3", "prompt": "咫尺天涯你终未远离", "duration_sec": 1.95}
        ]
    },
    {
        "theme": "制作团队信息",
        "segments": [
            {"file_name": "0034_天若有情 - 杜宣达.mp3", "prompt": "制作人 Producer：王圆坤", "duration_sec": 0.66},
            {"file_name": "0035_天若有情 - 杜宣达.mp3", "prompt": "编曲 Arranger：任斌", "duration_sec": 0.72},
            {"file_name": "0036_天若有情 - 杜宣达.mp3", "prompt": "吉他 Guitar：吴家裕", "duration_sec": 0.60},
            {"file_name": "0037_天若有情 - 杜宣达.mp3", "prompt": "和声 Backing Vocals：潘斯贝", "duration_sec": 0.69},
            {"file_name": "0038_天若有情 - 杜宣达.mp3", "prompt": "混音工程师 Mixing：郑昊杰", "duration_sec": 0.78},
            {"file_name": "0039_天若有情 - 杜宣达.mp3", "prompt": "母带工程师 Master：郑昊杰", "duration_sec": 10.44}
        ]
    },
    {
        "theme": "企划团队信息",
        "segments": [
            {"file_name": "0040_天若有情 - 杜宣达.mp3", "prompt": "企划：王嘉晟", "duration_sec": 0.36},
            {"file_name": "0041_天若有情 - 杜宣达.mp3", "prompt": "统筹：陈尚禔/黄路欢/ELANUS", "duration_sec": 0.27},
            {"file_name": "0042_天若有情 - 杜宣达.mp3", "prompt": "监制：王嘉晟", "duration_sec": 0.36},
            {"file_name": "0043_天若有情 - 杜宣达.mp3", "prompt": "营销推广：网易飓风", "duration_sec": 0.57},
            {"file_name": "0044_天若有情 - 杜宣达.mp3", "prompt": "出品人：谢奇笛 X 唐晶晶", "duration_sec": 1.38},
            {"file_name": "0045_天若有情 - 杜宣达.mp3", "prompt": "OP/ SP：索尼音乐版权代理（北京）有限公司", "duration_sec": 1.05}
        ]
    },
    {
        "theme": "版权声明",
        "segments": [
            {"file_name": "0046_天若有情 - 杜宣达.mp3", "prompt": "【此版本为正式授权翻唱作品】", "duration_sec": 1},
            {"file_name": "0047_天若有情 - 杜宣达.mp3", "prompt": "原唱 : 黄丽玲", "duration_sec": 1.73}
        ]
    }
]

场景是这样的：
1、两个人物 一男一女 character 只能在 男女两个人之间选取
2、随着歌曲在男女之间切换（切换规则根据你对于歌词或元信息的理解进行）

结合上面的数据结构对于每一个 {"file_name": "", "prompt": "", "duration_sec": },
添加两个键值 一个是角色 一个是场景的描述

得到类似下面的数据结构

scenes = [
    {
        "theme": "歌曲基本信息",
        "segments": [
            {
                "file_name": "0001_天若有情 - 杜宣达.mp3",
                "prompt": "作词 : 纪如璟",
                "duration_sec": 1,
                "character": "男",  # 男性旁白
                "scene_desc": "黑色背景浮现白色文字"
            },
            {
                "file_name": "0002_天若有情 - 杜宣达.mp3",
                "prompt": "作曲 : 黄义达",
                "duration_sec": 7.31,
                "character": "女",  # 女性旁白
                "scene_desc": "水墨晕染出作曲者姓名"
            },
            {
                "file_name": "0003_天若有情 - 杜宣达.mp3",
                "prompt": "出品：网易飓风 X 网易青云",
                "duration_sec": 4.14,
                "character": "男",  # 男性旁白
                "scene_desc": "金色logo缓缓浮现"
            }
        ]
    },
    {
        "theme": "主歌1（风花雪月）",
        "segments": [
            {
                "file_name": "0004_天若有情 - 杜宣达.mp3",
                "prompt": "风扬起时繁花落尽",
                "duration_sec": 5.88,
                "character": "男",
                "scene_desc": "男主角在樱花雨中独行，花瓣拂过脸庞"
            },
            {
                "file_name": "0005_天若有情 - 杜宣达.mp3",
                "prompt": "谁执笔为你绘丹青",
                "duration_sec": 6.18,
                "character": "女",
                "scene_desc": "女主角在书房提笔作画，眼泪滴落宣纸"
            },
            {
                "file_name": "0006_天若有情 - 杜宣达.mp3",
                "prompt": "月下独影泪湿青衣",
                "duration_sec": 6.03,
                "character": "男",
                "scene_desc": "月光下男主角抚摸染血的衣襟特写"
            },
            {
                "file_name": "0007_天若有情 - 杜宣达.mp3",
                "prompt": "流水不付一世深情",
                "duration_sec": 6.03,
                "character": "女",
                "scene_desc": "女主角将信笺放入溪流，镜头随水流远去"
            }
        ]
    },
    {
        "theme": "副歌1（爱恨纠缠）",
        "segments": [
            {
                "file_name": "0008_天若有情 - 杜宣达.mp3",
                "prompt": "只身回望太匆匆",
                "duration_sec": 3.12,
                "character": "男",
                "scene_desc": "男主角在战火中回首，慢动作处理"
            },
            {
                "file_name": "0009_天若有情 - 杜宣达.mp3",
                "prompt": "此生多少情与仇",
                "duration_sec": 3.75,
                "character": "女",
                "scene_desc": "女主角撕碎画卷，碎片化作蝴蝶"
            },
            {
                "file_name": "0010_天若有情 - 杜宣达.mp3",
                "prompt": "只愿与你长相守",
                "duration_sec": 5.34,
                "character": "男",
                "scene_desc": "男主角跪地握剑插入地面，眼中含泪"
            },
            {
                "file_name": "0011_天若有情 - 杜宣达.mp3",
                "prompt": "无边丝雨细如愁",
                "duration_sec": 3.06,
                "character": "女",
                "scene_desc": "雨丝穿过女主角撑开的油纸伞"
            },
            {
                "file_name": "0012_天若有情 - 杜宣达.mp3",
                "prompt": "朝来寒雨几回眸",
                "duration_sec": 3.84,
                "character": "男",
                "scene_desc": "男主角在雨中与女主角擦肩而过的慢镜头"
            },
            {
                "file_name": "0013_天若有情 - 杜宣达.mp3",
                "prompt": "你在哪一方停留",
                "duration_sec": 5.22,
                "character": "女",
                "scene_desc": "女主角推开雕花木窗远眺"
            },
            {
                "file_name": "0014_天若有情 - 杜宣达.mp3",
                "prompt": "天若有情亦无情",
                "duration_sec": 5.85,
                "character": "男",
                "scene_desc": "男主角剑指苍穹，雷电交加"
            },
            {
                "file_name": "0015_天若有情 - 杜宣达.mp3",
                "prompt": "爱到最后要分离",
                "duration_sec": 5.97,
                "character": "女",
                "scene_desc": "女主角剪断琴弦特写"
            },
            {
                "file_name": "0016_天若有情 - 杜宣达.mp3",
                "prompt": "你轮回的印记落在我眉宇",
                "duration_sec": 6.39,
                "character": "男",
                "scene_desc": "男主角抚摸眉间疤痕的闪回镜头"
            },
            {
                "file_name": "0017_天若有情 - 杜宣达.mp3",
                "prompt": "直到有一天不能呼吸",
                "duration_sec": 24.42,
                "character": "女",
                "scene_desc": "长镜头：女主角在雪地中怀抱男主角逐渐冰封"
            }
        ]
    },
{
    "theme": "主歌2（重复）",
    "segments": [
        {
            "file_name": "0018_天若有情 - 杜宣达.mp3",
            "prompt": "月下独影泪湿青衣",
            "duration_sec": 6.06,
            "character": "男",
            "scene_desc": "男主角在废墟中擦拭染血的衣袖"
        },
        {
            "file_name": "0019_天若有情 - 杜宣达.mp3",
            "prompt": "流水不付一世深情",
            "duration_sec": 6.00,
            "character": "女",
            "scene_desc": "女主角将定情玉佩沉入湖底"
        }
    ]
},
{
    "theme": "副歌2（重复）",
    "segments": [
        {
            "file_name": "0020_天若有情 - 杜宣达.mp3",
            "prompt": "只身回望太匆匆",
            "duration_sec": 3.03,
            "character": "男",
            "scene_desc": "男主角策马回头时的面部特写"
        },
        {
            "file_name": "0021_天若有情 - 杜宣达.mp3",
            "prompt": "此生多少情与仇",
            "duration_sec": 3.81,
            "character": "女",
            "scene_desc": "女主角烧毁信件的火盆特写"
        },
        {
            "file_name": "0022_天若有情 - 杜宣达.mp3",
            "prompt": "只愿与你长相守",
            "duration_sec": 5.43,
            "character": "男",
            "scene_desc": "男主角折断长剑插入墓碑"
        },
        {
            "file_name": "0023_天若有情 - 杜宣达.mp3",
            "prompt": "无边丝雨细如愁",
            "duration_sec": 2.97,
            "character": "女",
            "scene_desc": "雨滴在女主角的铜镜上晕开"
        },
        {
            "file_name": "0024_天若有情 - 杜宣达.mp3",
            "prompt": "朝来寒雨几回眸",
            "duration_sec": 3.81,
            "character": "男",
            "scene_desc": "男主角隔着雨帘与女主角对视"
        },
        {
            "file_name": "0025_天若有情 - 杜宣达.mp3",
            "prompt": "你在哪一方停留",
            "duration_sec": 5.28,
            "character": "女",
            "scene_desc": "女主角在佛前摇签的特写"
        },
        {
            "file_name": "0026_天若有情 - 杜宣达.mp3",
            "prompt": "天若有情亦无情",
            "duration_sec": 5.76,
            "character": "男",
            "scene_desc": "男主角在悬崖边张开双臂"
        },
        {
            "file_name": "0027_天若有情 - 杜宣达.mp3",
            "prompt": "爱到最后要分离",
            "duration_sec": 6.09,
            "character": "女",
            "scene_desc": "女主角扯断项链珍珠散落"
        },
        {
            "file_name": "0028_天若有情 - 杜宣达.mp3",
            "prompt": "你轮回的印记落在我眉宇",
            "duration_sec": 6.33,
            "character": "男",
            "scene_desc": "男主角对着铜镜描画女主角的眉形"
        },
        {
            "file_name": "0029_天若有情 - 杜宣达.mp3",
            "prompt": "直到有一天不能呼吸",
            "duration_sec": 6.09,
            "character": "女",
            "scene_desc": "女主角在冰棺中闭眼的特写"
        }
    ]
},
{
    "theme": "结尾升华",
    "segments": [
        {
            "file_name": "0030_天若有情 - 杜宣达.mp3",
            "prompt": "天若有情亦无情",
            "duration_sec": 5.79,
            "character": "男",
            "scene_desc": "男主角化为青烟消散"
        },
        {
            "file_name": "0031_天若有情 - 杜宣达.mp3",
            "prompt": "万丈红尘我等你",
            "duration_sec": 6.45,
            "character": "女",
            "scene_desc": "女主角在轮回井边纵身跃下"
        },
        {
            "file_name": "0032_天若有情 - 杜宣达.mp3",
            "prompt": "用你的牵挂染尽我白发",
            "duration_sec": 6.00,
            "character": "男",
            "scene_desc": "转世后的男主角白发抚琴"
        },
        {
            "file_name": "0033_天若有情 - 杜宣达.mp3",
            "prompt": "咫尺天涯你终未远离",
            "duration_sec": 1.95,
            "character": "女",
            "scene_desc": "女主角的魂魄从背后拥抱男主角"
        }
    ]
},
{
    "theme": "制作团队信息",
    "segments": [
        {
            "file_name": "0034_天若有情 - 杜宣达.mp3",
            "prompt": "制作人 Producer：王圆坤",
            "duration_sec": 0.66,
            "character": "男",
            "scene_desc": "毛笔字书写制作人姓名"
        },
        {
            "file_name": "0035_天若有情 - 杜宣达.mp3",
            "prompt": "编曲 Arranger：任斌",
            "duration_sec": 0.72,
            "character": "女",
            "scene_desc": "古琴谱卷轴展开显示名字"
        },
        {
            "file_name": "0036_天若有情 - 杜宣达.mp3",
            "prompt": "吉他 Guitar：吴家裕",
            "duration_sec": 0.60,
            "character": "男",
            "scene_desc": "琵琶拨弦动画带出名字"
        },
        {
            "file_name": "0037_天若有情 - 杜宣达.mp3",
            "prompt": "和声 Backing Vocals：潘斯贝",
            "duration_sec": 0.69,
            "character": "女",
            "scene_desc": "水墨晕染出和声者名字"
        },
        {
            "file_name": "0038_天若有情 - 杜宣达.mp3",
            "prompt": "混音工程师 Mixing：郑昊杰",
            "duration_sec": 0.78,
            "character": "男",
            "scene_desc": "朱砂印章盖出混音师姓名"
        },
        {
            "file_name": "0039_天若有情 - 杜宣达.mp3",
            "prompt": "母带工程师 Master：郑昊杰",
            "duration_sec": 10.44,
            "character": "女",
            "scene_desc": "玉玺盖章效果显示母带处理信息"
        }
    ]
},
{
    "theme": "企划团队信息",
    "segments": [
        {
            "file_name": "0040_天若有情 - 杜宣达.mp3",
            "prompt": "企划：王嘉晟",
            "duration_sec": 0.36,
            "character": "男",
            "scene_desc": "竹简展开显示企划姓名"
        },
        {
            "file_name": "0041_天若有情 - 杜宣达.mp3",
            "prompt": "统筹：陈尚禔/黄路欢/ELANUS",
            "duration_sec": 0.27,
            "character": "女",
            "scene_desc": "灯笼旋转显示统筹名单"
        },
        {
            "file_name": "0042_天若有情 - 杜宣达.mp3",
            "prompt": "监制：王嘉晟",
            "duration_sec": 0.36,
            "character": "男",
            "scene_desc": "铜镜反射出监制名字"
        },
        {
            "file_name": "0043_天若有情 - 杜宣达.mp3",
            "prompt": "营销推广：网易飓风",
            "duration_sec": 0.57,
            "character": "女",
            "scene_desc": "旋风卷出网易飓风logo"
        },
        {
            "file_name": "0044_天若有情 - 杜宣达.mp3",
            "prompt": "出品人：谢奇笛 X 唐晶晶",
            "duration_sec": 1.38,
            "character": "男",
            "scene_desc": "双龙戏珠图案托起出品人姓名"
        },
        {
            "file_name": "0045_天若有情 - 杜宣达.mp3",
            "prompt": "OP/ SP：索尼音乐版权代理（北京）有限公司",
            "duration_sec": 1.05,
            "character": "女",
            "scene_desc": "金色卷轴展开版权信息"
        }
    ]
},
{
    "theme": "版权声明",
    "segments": [
        {
            "file_name": "0046_天若有情 - 杜宣达.mp3",
            "prompt": "【此版本为正式授权翻唱作品】",
            "duration_sec": 1,
            "character": "男",
            "scene_desc": "玉牌刻字显示授权信息"
        },
        {
            "file_name": "0047_天若有情 - 杜宣达.mp3",
            "prompt": "原唱 : 黄丽玲",
            "duration_sec": 1.73,
            "character": "女",
            "scene_desc": "牡丹花绽放呈现原唱者姓名"
        }
    ]
}
]

pd.DataFrame(scenes).explode("segments")["segments"].map(lambda x: x["file_name"])

#### 得到的场景 是这样的

print(pd.DataFrame(scenes).explode("segments")["segments"].map(lambda x: (x["character"], x["scene_desc"])).to_markdown())

得到类似这样的结果

|    | segments                                           |
|---:|:---------------------------------------------------|
|  0 | ('男', '黑色背景浮现白色文字')                     |
|  0 | ('女', '水墨晕染')                       |
|  0 | ('男', '金色缓缓浮现')                         |
|  1 | ('男', '男主角在樱花雨中独行，花瓣拂过脸庞')       |
|  1 | ('女', '女主角在书房提笔作画，眼泪滴落宣纸')       |
|  1 | ('男', '月光下男主角抚摸染血的衣襟特写')           |
|  1 | ('女', '女主角将信笺放入溪流，镜头随水流远去')     |
|  2 | ('男', '男主角在战火中回首，慢动作处理')           |
|  2 | ('女', '女主角撕碎画卷，碎片化作蝴蝶')             |
|  2 | ('男', '男主角跪地握剑插入地面，眼中含泪')         |
|  2 | ('女', '雨丝穿过女主角撑开的油纸伞')               |
|  2 | ('男', '男主角在雨中擦肩而过的慢镜头')     |
|  2 | ('女', '女主角推开雕花木窗远眺')                   |
|  2 | ('男', '男主角剑指苍穹，雷电交加')                 |
|  2 | ('女', '女主角剪断琴弦特写')                       |
|  2 | ('男', '男主角抚摸眉间疤痕的闪回镜头')             |
|  2 | ('女', '长镜头：女主角在雪地中逐渐冰封') |
|  3 | ('男', '男主角在废墟中擦拭染血的衣袖')             |
|  3 | ('女', '女主角将定情玉佩沉入湖底')                 |
|  4 | ('男', '男主角策马回头时的面部特写')               |
|  4 | ('女', '女主角烧毁信件的火盆特写')                 |
|  4 | ('男', '男主角折断长剑插入墓碑')                   |
|  4 | ('女', '雨滴在女主角的铜镜上晕开')                 |
|  4 | ('男', '男主角隔着雨帘与女主角对视')               |
|  4 | ('女', '女主角在佛前摇签的特写')                   |
|  4 | ('男', '男主角在悬崖边张开双臂')                   |
|  4 | ('女', '女主角扯断项链珍珠散落')                   |
|  4 | ('男', '男主角对着铜镜')           |
|  4 | ('女', '女主角在冰棺中闭眼的特写')                 |
|  5 | ('男', '男主角化为青烟消散')                       |
|  5 | ('女', '女主角在轮回井边纵身跃下')                 |
|  5 | ('男', '转世后的男主角白发抚琴')                   |
|  5 | ('女', '女主角的魂魄从背后拥抱男主角')             |
|  6 | ('男', '毛笔字书写')                       |
|  6 | ('女', '古琴谱卷轴展开')                     |
|  6 | ('男', '琵琶拨弦')                       |
|  6 | ('女', '水墨晕染')                       |
|  6 | ('男', '朱砂印章盖')                     |
|  6 | ('女', '玉玺盖章')               |
|  7 | ('男', '竹简展开')                       |
|  7 | ('女', '灯笼旋转')                       |
|  7 | ('男', '铜镜反射')                         |
|  7 | ('女', '鲜花绽放')                     |
|  7 | ('男', '双龙戏珠')                 |
|  7 | ('女', '金色卷轴')                     |
|  8 | ('男', '玉牌刻字')                       |
|  8 | ('女', '牡丹花绽放')                   |

4、实验SD作图或视频生成（这里先简单用SD快速实验结果:包含字幕举牌和非举牌）
https://github.com/svjack/semantic-draw

4.1 原神 SD 版本

pip install diffusers transformers peft torch torchvision

import os
from tqdm import tqdm
from diffusers import StableDiffusionXLPipeline
import torch
import re

# Translation dictionary for Chinese to English
action_translation = {
    '黑色背景浮现白色文字': 'white_text_appearing_on_black_background',
    '水墨晕染': 'ink_wash_diffusion',
    '金色缓缓浮现': 'gold_slowly_emerging',
    '男主角在樱花雨中独行，花瓣拂过脸庞': 'male_lead_walking_alone_in_cherry_blossom_rain',
    '女主角在书房提笔作画，眼泪滴落宣纸': 'female_lead_drawing_in_study_with_tears_on_paper',
    '月光下男主角抚摸染血的衣襟特写': 'closeup_of_male_lead_touching_bloodstained_clothes_in_moonlight',
    '女主角将信笺放入溪流，镜头随水流远去': 'female_lead_placing_letter_in_stream_camera_following_water',
    '男主角在战火中回首，慢动作处理': 'male_lead_turning_back_in_war_slow_motion',
    '女主角撕碎画卷，碎片化作蝴蝶': 'female_lead_tearing_painting_into_butterflies',
    '男主角跪地握剑插入地面，眼中含泪': 'male_lead_kneeling_with_sword_in_ground_teary_eyes',
    '雨丝穿过女主角撑开的油纸伞': 'raindrops_passing_through_female_leads_oil_paper_umbrella',
    '男主角在雨中擦肩而过的慢镜头': 'male_lead_passing_by_in_rain_slow_motion',
    '女主角推开雕花木窗远眺': 'female_lead_opening_carved_window_to_gaze',
    '男主角剑指苍穹，雷电交加': 'male_lead_pointing_sword_at_sky_with_lightning',
    '女主角剪断琴弦特写': 'closeup_of_female_lead_cutting_lute_strings',
    '男主角抚摸眉间疤痕的闪回镜头': 'flashback_of_male_lead_touching_forehead_scar',
    '长镜头：女主角在雪地中逐渐冰封': 'long_shot_female_lead_freezing_in_snow',
    '男主角在废墟中擦拭染血的衣袖': 'male_lead_cleaning_bloodstained_sleeve_in_ruins',
    '女主角将定情玉佩沉入湖底': 'female_lead_sinking_love_token_into_lake',
    '男主角策马回头时的面部特写': 'closeup_of_male_lead_turning_horse_head',
    '女主角烧毁信件的火盆特写': 'closeup_of_female_lead_burning_letters_in_brazier',
    '男主角折断长剑插入墓碑': 'male_lead_breaking_sword_into_gravestone',
    '雨滴在女主角的铜镜上晕开': 'raindrops_blurring_on_female_leads_bronze_mirror',
    '男主角隔着雨帘与女主角对视': 'male_lead_and_female_lead_gazing_through_rain_curtain',
    '女主角在佛前摇签的特写': 'closeup_of_female_lead_shaking_lots_before_buddha',
    '男主角在悬崖边张开双臂': 'male_lead_spreading_arms_on_cliff_edge',
    '女主角扯断项链珍珠散落': 'female_lead_snapping_necklace_pearls_scattering',
    '男主角对着铜镜': 'male_lead_before_bronze_mirror',
    '女主角在冰棺中闭眼的特写': 'closeup_of_female_lead_eyes_closed_in_ice_coffin',
    '男主角化为青烟消散': 'male_lead_dissolving_into_smoke',
    '女主角在轮回井边纵身跃下': 'female_lead_jumping_into_reincarnation_well',
    '转世后的男主角白发抚琴': 'reincarnated_male_lead_white_haired_playing_lute',
    '女主角的魂魄从背后拥抱男主角': 'female_leads_spirit_embracing_male_lead_from_behind',
    '毛笔字书写': 'brush_calligraphy_writing',
    '古琴谱卷轴展开': 'unrolling_lute_music_score_scroll',
    '琵琶拨弦': 'plucking_pipa_strings',
    '朱砂印章盖': 'vermilion_seal_stamping',
    '玉玺盖章': 'imperial_jade_seal_stamping',
    '竹简展开': 'unrolling_bamboo_scrolls',
    '灯笼旋转': 'lanterns_rotating',
    '铜镜反射': 'bronze_mirror_reflection',
    '鲜花绽放': 'flowers_blooming',
    '双龙戏珠': 'twin_dragons_playing_with_pearl',
    '金色卷轴': 'golden_scroll',
    '玉牌刻字': 'engraving_jade_tablet',
    '牡丹花绽放': 'peony_blossoming'
}

action_translation = dict(
map(lambda t2: (t2[0], t2[1].replace("female", "").replace("male", "").replace("lead", "").replace("_", " ")), action_translation.items())
)
print(action_translation)

# Function to sanitize filenames
def sanitize_filename(text):
    text = re.sub(r'[\\/*?:"<>|]', "", text)
    text = text.replace(" ", "_")
    return text

# Define function to generate prompt
def gen_one_person_prompt(name, action):
    return f"SOLO, {name}, {action}, masterpiece, genshin impact style"

# Define name_dict
new_dict = {
    "女": "VENTI",
    "男": "XIAO"
}

# Initialize Stable Diffusion XL Pipeline
pipeline = StableDiffusionXLPipeline.from_pretrained(
    "cagliostrolab/animagine-xl-4.0",
    torch_dtype=torch.float16
).to("cuda")


# Define negative prompt
negative_prompt = "nsfw,lowres,(bad),text,error,fewer,extra,missing,worst quality,jpeg artifacts,low quality,watermark,unfinished,displeasing,oldest,early,chromatic aberration,signature,extra digits,artistic error,username,scan,[abstract],"

def generate_and_save_image(pipeline, prompt, negative_prompt, seed, save_dir="output_images", index=None):
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

# Complete segments data with index, gender, and Chinese action
segments = [
    (0, '男', '黑色背景浮现白色文字'),
    (0, '女', '水墨晕染'),
    (0, '男', '金色缓缓浮现'),
    (1, '男', '男主角在樱花雨中独行，花瓣拂过脸庞'),
    (1, '女', '女主角在书房提笔作画，眼泪滴落宣纸'),
    (1, '男', '月光下男主角抚摸染血的衣襟特写'),
    (1, '女', '女主角将信笺放入溪流，镜头随水流远去'),
    (2, '男', '男主角在战火中回首，慢动作处理'),
    (2, '女', '女主角撕碎画卷，碎片化作蝴蝶'),
    (2, '男', '男主角跪地握剑插入地面，眼中含泪'),
    (2, '女', '雨丝穿过女主角撑开的油纸伞'),
    (2, '男', '男主角在雨中擦肩而过的慢镜头'),
    (2, '女', '女主角推开雕花木窗远眺'),
    (2, '男', '男主角剑指苍穹，雷电交加'),
    (2, '女', '女主角剪断琴弦特写'),
    (2, '男', '男主角抚摸眉间疤痕的闪回镜头'),
    (2, '女', '长镜头：女主角在雪地中逐渐冰封'),
    (3, '男', '男主角在废墟中擦拭染血的衣袖'),
    (3, '女', '女主角将定情玉佩沉入湖底'),
    (4, '男', '男主角策马回头时的面部特写'),
    (4, '女', '女主角烧毁信件的火盆特写'),
    (4, '男', '男主角折断长剑插入墓碑'),
    (4, '女', '雨滴在女主角的铜镜上晕开'),
    (4, '男', '男主角隔着雨帘与女主角对视'),
    (4, '女', '女主角在佛前摇签的特写'),
    (4, '男', '男主角在悬崖边张开双臂'),
    (4, '女', '女主角扯断项链珍珠散落'),
    (4, '男', '男主角对着铜镜'),
    (4, '女', '女主角在冰棺中闭眼的特写'),
    (5, '男', '男主角化为青烟消散'),
    (5, '女', '女主角在轮回井边纵身跃下'),
    (5, '男', '转世后的男主角白发抚琴'),
    (5, '女', '女主角的魂魄从背后拥抱男主角'),
    (6, '男', '毛笔字书写'),
    (6, '女', '古琴谱卷轴展开'),
    (6, '男', '琵琶拨弦'),
    (6, '女', '水墨晕染'),
    (6, '男', '朱砂印章盖'),
    (6, '女', '玉玺盖章'),
    (7, '男', '竹简展开'),
    (7, '女', '灯笼旋转'),
    (7, '男', '铜镜反射'),
    (7, '女', '鲜花绽放'),
    (7, '男', '双龙戏珠'),
    (7, '女', '金色卷轴'),
    (8, '男', '玉牌刻字'),
    (8, '女', '牡丹花绽放')
]

# Generate images for all segments
for idx, (group_idx, gender, chinese_action) in enumerate(segments):
    name = new_dict[gender]
    english_action = action_translation.get(chinese_action, chinese_action)
    prompt = gen_one_person_prompt(name, english_action)
    generate_and_save_image(pipeline, prompt, negative_prompt, seed=-1,
                            save_dir="output_images_v0",
                            index=idx)

4.2 Hi-Dream 版本

conda activate system
pip install torch==2.5.0 torchvision

pip install hdi1 --no-build-isolation

pip uninstall torch torchvision -y
pip install torch==2.5.0 torchvision
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

python -m hdi1 "A cat holding a sign that says 'hello world'"

python -m hdi1 "A cat holding a sign that says 'hello world'" -m fast

edit share = True
vim /environment/miniconda3/envs/system/lib/python*/site-packages/hdi1/web.py

python -m hdi1.web
featurize port export 7860

能否举牌中文歌词

X
python -m hdi1 "A cat holding a sign that says '你轮回的印记落在我眉宇'"
Y
python -m hdi1 "A cat holding a sign that says 'The mark of your reincarnation falls between my brows'"

可ui 调用

古装提示词：
这是一张高质量的艺术照片，一位留着黑色长发的年轻女子，穿着白色的中国传统旗袍，站在模糊的雪地背景下。灯光柔和自然，突显出她平静的表情。该图像使用浅景深，聚焦在背景模糊的物体上。构图遵循三分法，女人的脸偏离中心。这张照片可能是用单反相机拍摄的，可能是佳能EOS 5D Mark IV，光圈设置为f/2.8，快门速度为1/200，ISO 400。其美学品质极高，展现出优雅的简洁与宁静的氛围。

给出其举牌英文歌词的调用

git clone https://huggingface.co/datasets/svjack/Day_if_sentient_beings_SPLITED && cd Day_if_sentient_beings_SPLITED
使用 ui 方式调用

python -m hdi1.web

import pandas as pd
from gradio_client import Client
from PIL import Image
import os
import shutil
from tqdm import tqdm

# Load the DataFrame
df = pd.read_csv("Day_if_sentient_beings_SPLITED/metadata.csv")

# Add English translations
# Create a translation dictionary (you may want to use a proper translation API for better results)
translation_dict = {
    "作词 : 纪如璟": "Lyrics by Ji Rujing",
    "作曲 : 黄义达": "Composed by Huang Yida",
    "出品：网易飓风 X 网易青云": "Produced by NetEase Hurricane X NetEase Qingyun",
    "风扬起时繁花落尽": "When the wind rises, all flowers fall",
    "谁执笔为你绘丹青": "Who holds the brush to paint for you",
    "月下独影泪湿青衣": "Lonely shadow under the moon, tears wet the blue robe",
    "流水不付一世深情": "Flowing water doesn't repay lifelong devotion",
    "只身回望太匆匆": "Looking back alone, too hurried",
    "此生多少情与仇": "How much love and hate in this life",
    "只愿与你长相守": "Only wish to stay with you forever",
    "无边丝雨细如愁": "Boundless drizzle fine as sorrow",
    "朝来寒雨几回眸": "Morning cold rain, how many glances back",
    "你在哪一方停留": "Where do you linger",
    "天若有情亦无情": "If heaven has feelings, it's also heartless",
    "爱到最后要分离": "Love ends in separation",
    "你轮回的印记落在我眉宇": "The mark of your reincarnation rests between my brows",
    "直到有一天不能呼吸": "Until one day I can't breathe",
    "万丈红尘我等你": "In the vast mortal world, I wait for you",
    "用你的牵挂染尽我白发": "Let your concern dye my white hair",
    "咫尺天涯你终未远离": "So near yet so far, you never truly left",
    "制作人 Producer：王圆坤": "Producer: Wang Yuankun",
    "编曲 Arranger：任斌": "Arranger: Ren Bin",
    "吉他 Guitar：吴家裕": "Guitar: Wu Jiayu",
    "和声 Backing Vocals：潘斯贝": "Backing Vocals: Pan Sibei",
    "混音工程师 Mixing：郑昊杰": "Mixing Engineer: Zheng Haojie",
    "母带工程师 Master：郑昊杰": "Mastering Engineer: Zheng Haojie",
    "企划：王嘉晟": "Planning: Wang Jiasheng",
    "统筹：陈尚禔/黄路欢/ELANUS": "Coordination: Chen Shangti/Huang Luhuan/ELANUS",
    "监制：王嘉晟": "Supervisor: Wang Jiasheng",
    "营销推广：网易飓风": "Marketing: NetEase Hurricane",
    "出品人：谢奇笛 X 唐晶晶": "Producers: Xie Qidi X Tang Jingjing",
    "OP/ SP：索尼音乐版权代理（北京）有限公司": "OP/SP: Sony Music Publishing (Beijing) Co., Ltd.",
    "【此版本为正式授权翻唱作品】": "[This is an officially licensed cover version]",
    "原唱 : 黄丽玲": "Original singer: A-Lin"
}
df['en_prompt'] = df['prompt'].map(translation_dict)

# Initialize Gradio client
client = Client("http://localhost:7860/")

# Create output directory if it doesn't exist
output_dir = "Day_if_sentient_beings_SPLITED_BY_CAT"
os.makedirs(output_dir, exist_ok=True)

# Process each row with progress bar
for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating images"):
    try:
        # Generate image with cat holding sign
        prompt = f"A cat holding a sign that says '{row['en_prompt']}'"
        print(prompt)
        result = client.predict(
            model="fast",
            prompt=prompt,
            res="1024 × 1024 (Square)",
            seed=-1,
            api_name="/gen_img_helper"
        )

        # Save image (change extension to .png)
        img_path = os.path.join(output_dir, os.path.splitext(row['file_name'])[0] + ".png")
        Image.open(result[0]).save(img_path)

        # Copy corresponding MP3 file
        mp3_src = os.path.join("Day_if_sentient_beings_SPLITED", row['file_name'])
        mp3_dest = os.path.join(output_dir, row['file_name'])
        shutil.copy2(mp3_src, mp3_dest)

    except Exception as e:
        tqdm.write(f"Error processing {row['file_name']}: {str(e)}")

print("Processing complete!")

from datasets import Dataset, Audio, Image
import os
import pandas as pd

# 假设您的目录路径
data_dir = "Day_if_sentient_beings_SPLITED_BY_CAT"

# 收集所有 (mp3, png) 对
audio_files = [f for f in os.listdir(data_dir) if f.endswith(".mp3")]
image_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]

# 保持字典序
audio_files = sorted(audio_files)

# 确保音频和图像文件一一对应（假设文件名前缀相同）
data = []
for audio in audio_files:
    base_name = os.path.splitext(audio)[0]
    image = f"{base_name}.png"
    if image in image_files:
        data.append({
            "audio": os.path.join(data_dir, audio),
            "image": os.path.join(data_dir, image)
        })

# 创建数据集
dataset = Dataset.from_pandas(pd.DataFrame(data))

# 将类型转换为对应的 Audio 和 Image 类型
dataset = dataset.cast_column("audio", Audio())
dataset = dataset.cast_column("image", Image())

# 打印数据集结构
print(dataset)

# 保存数据集到磁盘
#dataset.save_to_disk("day_if_sentient_beings_dataset")
dataset.push_to_hub("svjack/Day_if_sentient_beings_SPLITED_BY_CAT_CARD")

4.3 CogView4 是否能够举牌（不能 长中文文字表达差）

conda activate base
pip install -U diffusers

from diffusers import CogView4Pipeline
import torch

pipe = CogView4Pipeline.from_pretrained("THUDM/CogView4-6B", torch_dtype=torch.bfloat16)

# Open it for reduce GPU memory usage
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

### "一张红色的海报，中间写有“开门大吉”"
prompt = "A dog holding a sign that says '你轮回的印记落在我眉宇'"
image = pipe(
    prompt=prompt,
    guidance_scale=3.5,
    num_images_per_prompt=1,
    num_inference_steps=50,
    width=1024,
    height=1024,
).images[0]

image.save("cogview4.png")

4.4 使用 Step1X-Edit 将英文改成中文

https://github.com/svjack/Step1X-Edit 是否能取代 存疑

### 具备初级能力
将图片中的文字改为“杨志鹏”

将图片中的英文改为“你轮回的印记落在我眉宇”

将图片中猫咪举起的牌子的内容改为“你轮回的印记落在我眉宇”

在图片中猫咪举的牌子上添加文字“你轮回的印记落在我眉宇”

### 先生成空看板
import pandas as pd
from gradio_client import Client
from PIL import Image
import os
import shutil
from tqdm import tqdm

# Load the DataFrame
df = pd.read_csv("Day_if_sentient_beings_SPLITED/metadata.csv")

# Add English translations
# Create a translation dictionary (you may want to use a proper translation API for better results)
translation_dict = {
    "作词 : 纪如璟": "Lyrics by Ji Rujing",
    "作曲 : 黄义达": "Composed by Huang Yida",
    "出品：网易飓风 X 网易青云": "Produced by NetEase Hurricane X NetEase Qingyun",
    "风扬起时繁花落尽": "When the wind rises, all flowers fall",
    "谁执笔为你绘丹青": "Who holds the brush to paint for you",
    "月下独影泪湿青衣": "Lonely shadow under the moon, tears wet the blue robe",
    "流水不付一世深情": "Flowing water doesn't repay lifelong devotion",
    "只身回望太匆匆": "Looking back alone, too hurried",
    "此生多少情与仇": "How much love and hate in this life",
    "只愿与你长相守": "Only wish to stay with you forever",
    "无边丝雨细如愁": "Boundless drizzle fine as sorrow",
    "朝来寒雨几回眸": "Morning cold rain, how many glances back",
    "你在哪一方停留": "Where do you linger",
    "天若有情亦无情": "If heaven has feelings, it's also heartless",
    "爱到最后要分离": "Love ends in separation",
    "你轮回的印记落在我眉宇": "The mark of your reincarnation rests between my brows",
    "直到有一天不能呼吸": "Until one day I can't breathe",
    "万丈红尘我等你": "In the vast mortal world, I wait for you",
    "用你的牵挂染尽我白发": "Let your concern dye my white hair",
    "咫尺天涯你终未远离": "So near yet so far, you never truly left",
    "制作人 Producer：王圆坤": "Producer: Wang Yuankun",
    "编曲 Arranger：任斌": "Arranger: Ren Bin",
    "吉他 Guitar：吴家裕": "Guitar: Wu Jiayu",
    "和声 Backing Vocals：潘斯贝": "Backing Vocals: Pan Sibei",
    "混音工程师 Mixing：郑昊杰": "Mixing Engineer: Zheng Haojie",
    "母带工程师 Master：郑昊杰": "Mastering Engineer: Zheng Haojie",
    "企划：王嘉晟": "Planning: Wang Jiasheng",
    "统筹：陈尚禔/黄路欢/ELANUS": "Coordination: Chen Shangti/Huang Luhuan/ELANUS",
    "监制：王嘉晟": "Supervisor: Wang Jiasheng",
    "营销推广：网易飓风": "Marketing: NetEase Hurricane",
    "出品人：谢奇笛 X 唐晶晶": "Producers: Xie Qidi X Tang Jingjing",
    "OP/ SP：索尼音乐版权代理（北京）有限公司": "OP/SP: Sony Music Publishing (Beijing) Co., Ltd.",
    "【此版本为正式授权翻唱作品】": "[This is an officially licensed cover version]",
    "原唱 : 黄丽玲": "Original singer: A-Lin"
}
df['en_prompt'] = df['prompt'].map(translation_dict)

# Initialize Gradio client
client = Client("http://localhost:7860/")

# Create output directory if it doesn't exist
output_dir = "Day_if_sentient_beings_SPLITED_BY_CAT_EMPTY"
os.makedirs(output_dir, exist_ok=True)

# Process each row with progress bar
for _, row in tqdm(df.iterrows(), total=len(df), desc="Generating images"):
    try:
        # Generate image with cat holding sign
        #prompt = f"A cat holding a sign that says '{row['en_prompt']}'"
        prompt = f"A cat holding a empty sign."
        print(prompt)
        result = client.predict(
            model="fast",
            prompt=prompt,
            res="1024 × 1024 (Square)",
            seed=-1,
            api_name="/gen_img_helper"
        )

        # Save image (change extension to .png)
        img_path = os.path.join(output_dir, os.path.splitext(row['file_name'])[0] + ".png")
        Image.open(result[0]).save(img_path)

        # Copy corresponding MP3 file
        mp3_src = os.path.join("Day_if_sentient_beings_SPLITED", row['file_name'])
        mp3_dest = os.path.join(output_dir, row['file_name'])
        shutil.copy2(mp3_src, mp3_dest)

    except Exception as e:
        tqdm.write(f"Error processing {row['file_name']}: {str(e)}")

print("Processing complete!")

from datasets import Dataset, Audio, Image
import os
import pandas as pd

# 假设您的目录路径
data_dir = "Day_if_sentient_beings_SPLITED_BY_CAT_EMPTY"

# 收集所有 (mp3, png) 对
audio_files = [f for f in os.listdir(data_dir) if f.endswith(".mp3")]
image_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]

# 保持字典序
audio_files = sorted(audio_files)

# 确保音频和图像文件一一对应（假设文件名前缀相同）
data = []
for audio in audio_files:
    base_name = os.path.splitext(audio)[0]
    image = f"{base_name}.png"
    if image in image_files:
        data.append({
            "audio": os.path.join(data_dir, audio),
            "image": os.path.join(data_dir, image)
        })

# 创建数据集
dataset = Dataset.from_pandas(pd.DataFrame(data))

# 将类型转换为对应的 Audio 和 Image 类型
dataset = dataset.cast_column("audio", Audio())
dataset = dataset.cast_column("image", Image())

# 打印数据集结构
print(dataset)

# 保存数据集到磁盘
#dataset.save_to_disk("day_if_sentient_beings_dataset")
dataset.push_to_hub("svjack/Day_if_sentient_beings_SPLITED_BY_CAT_EMPTY_CARD")

python gradio_demo.py

from gradio_client import Client, handle_file
client = Client("http://localhost:7860")
result = client.predict(
		prompt="Remove the person from the image.",
		ref_images=handle_file('023614aa-5b08-4fbe-8f1a-70e7e5642808 (2).png'),
		seed=-1,
		size_level=1024,
		quantized=True,
		offload=True,
		api_name="/inference"
)
print(result)

from shutil import copy2
copy2(result[0][1], result[0][1].split("/")[-1])

上面是一个 api 调用方法

下面是 Day_if_sentient_beings_SPLITED 下 (.mp3, .txt) 对应的内容

import pandas as pd
print(pd.read_csv("Day_if_sentient_beings_SPLITED/metadata.csv").head(3).to_markdown())

|    | file_name                  | prompt                    |
|---:|:---------------------------|:--------------------------|
|  0 | 0001_天若有情 - 杜宣达.mp3 | 作词 : 纪如璟             |
|  1 | 0002_天若有情 - 杜宣达.mp3 | 作曲 : 黄义达             |
|  2 | 0003_天若有情 - 杜宣达.mp3 | 出品：网易飓风 X 网易青云 |

下面是 Day_if_sentient_beings_SPLITED_BY_CAT 下 (.mp3, .png) 对应的文件名映射

import pandas as pd
print(pd.read_csv("Day_if_sentient_beings_SPLITED_BY_CAT/metadata.csv").head(3).to_markdown())

|    | audio_file_name            | png_file_name              |
|---:|:---------------------------|:---------------------------|
|  0 | 0001_天若有情 - 杜宣达.mp3 | 0001_天若有情 - 杜宣达.png |
|  1 | 0002_天若有情 - 杜宣达.mp3 | 0002_天若有情 - 杜宣达.png |
|  2 | 0003_天若有情 - 杜宣达.mp3 | 0003_天若有情 - 杜宣达.png |

现在要求你遍历 Day_if_sentient_beings_SPLITED_BY_CAT 的所有 (.mp3, .png)
对儿，将其中的 .png 作为 上面api 调用方法中 handle_file 的输入
prompt 改为 将图片中的英文改为“{}” 其中 {} 为 prompt 对应行的文本
将输出的 result[0][1] 使用 Image.open进行读取之后 save 到
路径
Day_if_sentient_beings_SPLITED_BY_CAT_ZH
下 保持相同的文件名 并且 将 对应的.mp3 文件copy2 到对应的路径
使用tqdm 打印流程

标点符号 替换
为图片中的猫咪举的牌子添加下面的文字：“作词  纪如璟”

为图片中的猫咪举的牌子添加下面的文字：“作 词  纪 如 璟”

为图片中的猫咪举的牌子添加下面的文字：“出品 网易飓风 X 网易青云”

import os
from tqdm import tqdm
from PIL import Image
from shutil import copy2
from gradio_client import Client, handle_file
import pandas as pd

# Initialize the client
client = Client("http://localhost:7860")

# Create output directory if it doesn't exist
output_dir = "Day_if_sentient_beings_SPLITED_BY_CAT_ZH"
os.makedirs(output_dir, exist_ok=True)

# Load metadata from both directories
metadata_by_cat = pd.read_csv("Day_if_sentient_beings_SPLITED_BY_CAT/metadata.csv")
metadata_prompts = pd.read_csv("Day_if_sentient_beings_SPLITED/metadata.csv")

# Create a dictionary mapping file names to prompts
prompt_dict = dict(zip(metadata_prompts['file_name'], metadata_prompts['prompt']))

# Process each pair with tqdm progress bar
for _, row in tqdm(metadata_by_cat.iterrows(), total=len(metadata_by_cat), desc="Processing files"):
    audio_file = row['audio_file_name']
    png_file = row['png_file_name']

    # Get the corresponding prompt
    # The audio file names should match between the two directories
    prompt = prompt_dict.get(audio_file, "")
    #prompt = " ".join(map(lambda x: x.strip() ,list(prompt)))
    print("prompt :", f"为图片中的猫咪举的牌子添加下面的文字：“{prompt}”")
    # Process the image through the API
    result = client.predict(
        prompt=f"为图片中的猫咪举的牌子添加下面的文字：“{prompt}”",
        ref_images=handle_file(os.path.join("Day_if_sentient_beings_SPLITED_BY_CAT_EMPTY", png_file)),
        seed=-1,
        size_level=1024,
        quantized=True,
        offload=True,
        api_name="/inference"
    )

    # Save the processed image
    output_image_path = os.path.join(output_dir, png_file)
    with Image.open(result[0][1]) as img:
        img.save(output_image_path)

    # Copy the corresponding audio file
    output_audio_path = os.path.join(output_dir, audio_file)
    copy2(
        os.path.join("Day_if_sentient_beings_SPLITED_BY_CAT_EMPTY", audio_file),
        output_audio_path
    )

print("Processing complete!")


from datasets import Dataset, Audio, Image
import os
import pandas as pd

# 假设您的目录路径
data_dir = "Day_if_sentient_beings_SPLITED_BY_CAT_ZH/"

# 收集所有 (mp3, png) 对
audio_files = [f for f in os.listdir(data_dir) if f.endswith(".mp3")]
image_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]

# 保持字典序
audio_files = sorted(audio_files)

# 确保音频和图像文件一一对应（假设文件名前缀相同）
data = []
for audio in audio_files:
    base_name = os.path.splitext(audio)[0]
    image = f"{base_name}.png"
    if image in image_files:
        data.append({
            "audio": os.path.join(data_dir, audio),
            "image": os.path.join(data_dir, image)
        })

# 创建数据集
dataset = Dataset.from_pandas(pd.DataFrame(data))

# 将类型转换为对应的 Audio 和 Image 类型
dataset = dataset.cast_column("audio", Audio())
dataset = dataset.cast_column("image", Image())

# 打印数据集结构
print(dataset)

# 保存数据集到磁盘
#dataset.save_to_disk("day_if_sentient_beings_dataset")
dataset.push_to_hub("svjack/Day_if_sentient_beings_SPLITED_BY_CAT_ZH_CARD")


### 找个英文歌曲 一举牌

4.5 专注于艺术文字图片生成或相关功能：
例子：
https://github.com/AIGText/Glyph-ByT5

涉及过的项目：
https://github.com/svjack/bizgen

git clone https://github.com/svjack/bizgen
cd bizgen
pip install -r requirements.txt
pip install "numpy<2"

git clone https://huggingface.co/PYY2001/BizGen
mv BizGen checkpoints

git clone https://huggingface.co/madebyollin/sdxl-vae-fp16-fix

huggingface-cli login

python inference.py --sample_list zh_dog.json

[
    {
        "index": "dog_0_1",
        "layers_all": [
            {
                "category": "base",
                "top_left": [0, 0],
                "bottom_right": [1024, 768],
                "caption": "图片展示一只金毛犬前爪举着木质标牌，站在阳光下的草地上。狗狗眼神温柔深邃，毛发在光线中呈现金色渐变。标牌做旧处理带有自然木纹，背景是虚化的春日树林和光斑效果。"
            },
            {
                "category": "element",
                "top_left": [200, 100],
                "bottom_right": [824, 668],
                "caption": "金色圆形背景区块"
            },
            {
                "category": "element",
                "top_left": [300, 300],
                "bottom_right": [724, 500],
                "caption": "白色圆形区块"
            },
            {
                "category": "text",
                "top_left": [320, 350],
                "bottom_right": [704, 450],
                "caption": "手写风格文字\"你轮回的印记落在我眉宇\" <color-89>, <cn-font-205>",
                "text": "你轮回的印记落在我眉宇"
            },
            {
                "category": "element",
                "top_left": [0, 0],
                "bottom_right": [1024, 768],
                "caption": "橙色圆形图标"
            },
            {
                "category": "element",
                "top_left": [900, 650],
                "bottom_right": [1020, 750],
                "caption": "浅棕色矩形区块"
            }
        ],
        "full_image_caption": "画面中心一只金毛犬虔诚地举起做旧木牌，牌上手写诗句'你轮回的印记落在我眉宇'充满哲思。采用1024x768横向构图，阳光透过树叶在狗狗毛发上形成光晕，爪印装饰强化生命轮回的意象，整体营造温暖而深邃的禅意氛围。"
    }
]


4.6 wan 宣称能 动态字幕 艺术字体 进行实验
https://zhuanlan.zhihu.com/p/26436233702

以红色新年宣纸为背景，出现一滴水墨，晕染墨汁缓缓晕染开来。文字的笔画边缘模糊且自然，随着晕染的进行，水墨在纸上呈现“福”字，墨色从深到浅过渡，呈现出独特的东方韵味。背景高级简洁，杂志摄影感。

### 验证地址
https://huggingface.co/spaces/markury/Wan-2.1-T2V-1.3B-LoRA




4.7 最好是 instantcharacter

https://github.com/softicelee2/aishare/tree/main/92

conda activate base

git clone https://github.com/softicelee2/aishare.git
cd aishare/92/

pip install -r requirements.txt
pip install -U gradio

###git clone https://github.com/Tencent/InstantCharacter
huggingface-cli download --resume-download Tencent/InstantCharacter --local-dir checkpoints --local-dir-use-symlinks False

cp app.py InstantCharacter
cp pipeline.py InstantCharacter
cd InstantCharacter

huggingface-cli download black-forest-labs/FLUX.1-dev --local-dir checkpoints/FLUX.1-dev
huggingface-cli download tencent/InstantCharacter --local-dir checkpoints/InstantCharacter
huggingface-cli download facebook/dinov2-giant --local-dir checkpoints/dinov2-giant
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir checkpoints/siglip-so400m-patch14-384
huggingface-cli download InstantX/FLUX.1-dev-LoRA-Ghibli --local-dir checkpoints/FLUX.1-dev-LoRA-Ghibli
huggingface-cli download InstantX/FLUX.1-dev-LoRA-Makoto-Shinkai --local-dir checkpoints/FLUX.1-dev-LoRA-Makoto-Shinkai

share = True

python app.py

5、改变分辨率
conda activate base

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

huggingface-cli upload svjack/Genshin_Impact_XIAO_VENTI_Images_9_16 Genshin_Impact_XIAO_VENTI_Images_9_16 --repo-type dataset

5、对于举牌类数据集 直接合并成视频

from moviepy.editor import *
import os

# 配置路径和参数
#input_dir = "Day_if_sentient_beings_SPLITED_BY_CAT"
#output_file = "Day_if_sentient_beings_SPLITED_BY_CAT.mp4"
#input_dir = "Defying_Gravity_SPLITED_BY_CAT"
#output_file = "Defying_Gravity_SPLITED_BY_CAT.mp4"
input_dir = "Day_if_sentient_beings_SPLITED_BY_CAT_ZH"
output_file = "Day_if_sentient_beings_SPLITED_BY_CAT_ZH.mp4"
font_path = "simhei.ttf"  # 确保字体文件存在
transition_duration = 0.5  # 水滴特效持续时间

# 获取并排序文件
all_files = sorted(os.listdir(input_dir))
audio_files = [f for f in all_files if f.endswith(".mp3")]
image_files = [f for f in all_files if f.endswith(".png")]

# 验证文件配对
if len(audio_files) != len(image_files):
    raise ValueError("音频与图片文件数量不匹配")

# 创建独立的video_clips列表
video_clips = []

for audio_file, image_file in zip(audio_files, image_files):
    # 加载音频（确保无淡入淡出）
    audio = AudioFileClip(os.path.join(input_dir, audio_file))

    # 加载图片并设置持续时间
    img_clip = ImageClip(os.path.join(input_dir, image_file))
    img_clip = img_clip.set_duration(audio.duration)

    # 添加淡入淡出效果（仅对图片）
    img_clip = img_clip.fadein(0.3).fadeout(0.3)  # 0.3秒淡入，0.3秒淡出

    # 创建视频片段（图片+音频）
    video_clip = img_clip.set_audio(audio)
    video_clips.append(video_clip)

# 连接所有视频片段（不添加过渡效果）
final_video = concatenate_videoclips(video_clips, method="compose")

# 输出视频（优化编码参数）
final_video.write_videofile(
    output_file,
    codec="libx264",
    audio_codec="aac",
    fps=24,
    threads=8,
    preset="fast",
    ffmpeg_params=["-crf", "23"]
)

# 释放资源
for clip in video_clips:
    clip.close()
final_video.close()

6、构造 srt 及 合并后的视频

#### 图片
git clone https://huggingface.co/datasets/svjack/Genshin_Impact_XIAO_VENTI_Images_9_16
字典序 png 图片

#### 音频和歌词
git clone https://huggingface.co/datasets/svjack/Day_if_sentient_beings_SPLITED
字典序 (.mp3, .txt) 文件对

sudo apt install imagemagick

<!--
sudo vim /etc/ImageMagick-6/policy.xml
##### change row to
<policy domain="path" rights="read|write" pattern="@*"/>
-->

from moviepy.editor import *
from moviepy.video.VideoClip import TextClip
import os

# 配置路径和参数
image_dir = "Genshin_Impact_XIAO_VENTI_Images_9_16"
audio_text_dir = "Day_if_sentient_beings_SPLITED"
output_file = "XIAO_VENTI_Day_if_sentient_beings.mp4"
font_path = "simhei.ttf"  # 确保字体文件存在
font_size = 44
subtitle_color = 'white'
subtitle_bg_color = 'black'

# 获取并排序文件
png_files = sorted([f for f in os.listdir(image_dir) if f.endswith(".png")])
audio_files = sorted([f for f in os.listdir(audio_text_dir) if f.endswith(".mp3")])
text_files = sorted([f for f in os.listdir(audio_text_dir) if f.endswith(".txt")])

# 验证文件数量匹配
if len(audio_files) != len(text_files) or len(audio_files) != len(png_files):
    raise ValueError("音频、字幕和图片文件数量不匹配")

# 创建video_clips列表
video_clips = []

for audio_file, png_file, text_file in zip(audio_files, png_files, text_files):
    # 加载音频
    audio = AudioFileClip(os.path.join(audio_text_dir, audio_file))

    # 加载图片并设置持续时间
    img_clip = ImageClip(os.path.join(image_dir, png_file))
    img_clip = img_clip.set_duration(audio.duration)

    # 添加淡入淡出效果
    img_clip = img_clip.fadein(0.3).fadeout(0.3)

    # 读取字幕文本
    with open(os.path.join(audio_text_dir, text_file), 'r', encoding='utf-8') as f:
        subtitle_text = f.read()

    # 创建字幕Clip
    txt_clip = TextClip(subtitle_text, fontsize=font_size, color=subtitle_color,
                        font=font_path, bg_color=subtitle_bg_color)
    #txt_clip = txt_clip.set_position(('center', 'bottom')).set_duration(audio.duration)
    txt_clip = txt_clip.set_position(lambda t: ('center', 0.9), relative=True).set_duration(audio.duration)  # 位于画面高度80%处

    # 合并图片和字幕
    video_clip = CompositeVideoClip([img_clip, txt_clip]).set_audio(audio)
    video_clips.append(video_clip)

# 连接所有视频片段
final_video = concatenate_videoclips(video_clips, method="compose")

# 输出视频
final_video.write_videofile(
    output_file,
    codec="libx264",
    audio_codec="aac",
    fps=24,
    threads=8,
    preset="fast",
    ffmpeg_params=["-crf", "23"]
)

# 释放资源
for clip in video_clips:
    clip.close()
final_video.close()

问题：framepack 是否保持文字
