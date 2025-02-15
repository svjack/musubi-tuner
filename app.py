import gradio as gr
import subprocess
import os
import torch
from datetime import datetime
from huggingface_hub import snapshot_download
from pathlib import Path
import traceback

python_cmd = "python"

# 模型下载和预处理
def setup_models():
    os.makedirs("ckpts", exist_ok=True)
    
    try:
        #if torch.cuda.is_available():
        if True:
            # 下载HunyuanVideo模型
            hunyuan_path = Path("ckpts/hunyuan-video-t2v-720p")
            if not hunyuan_path.exists():
                print("Downloading HunyuanVideo model...")
                snapshot_download(
                    repo_id="tencent/HunyuanVideo",
                    local_dir="ckpts",
                    force_download=True
                )

            # 处理LLaVA模型
            llava_path = Path("ckpts/llava-llama-3-8b-v1_1-transformers")
            text_encoder_path = Path("ckpts/text_encoder")
            
            if not llava_path.exists():
                print("Downloading LLaVA model...")
                snapshot_download(
                    repo_id="xtuner/llava-llama-3-8b-v1_1-transformers",
                    local_dir=llava_path,
                    force_download=True
                )

            if not text_encoder_path.exists():
                print("Preprocessing text encoder...")
                preprocess_script = Path("preprocess_text_encoder_tokenizer_utils.py")
                if not preprocess_script.exists():
                    subprocess.run(
                        ["wget", "https://raw.githubusercontent.com/Tencent/HunyuanVideo/refs/heads/main/hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py"],
                        check=True,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT
                    )
                subprocess.run(
                    [
                        python_cmd,
                        "preprocess_text_encoder_tokenizer_utils.py",
                        "--input_dir", str(llava_path),
                        "--output_dir", str(text_encoder_path)
                    ],
                    check=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT
                )

            # 下载CLIP模型
            clip_path = Path("ckpts/text_encoder_2")
            if not clip_path.exists():
                print("Downloading CLIP model...")
                snapshot_download(
                    repo_id="openai/clip-vit-large-patch14",
                    local_dir=clip_path,
                    force_download=True
                )

    except Exception as e:
        print("模型初始化失败:")
        traceback.print_exc()
        raise

# 获取最新生成的视频文件
def get_latest_video(output_dir="outputs"):
    try:
        video_files = list(Path(output_dir).glob("*.mp4"))
        if not video_files:
            return None
        return max(video_files, key=lambda x: x.stat().st_ctime)
    except Exception as e:
        print(f"获取最新视频失败: {str(e)}")
        return None

# 扫描LoRA权重文件
def scan_lora_weights(lora_dir="lora_dir"):
    return [str(f) for f in Path(lora_dir).glob("*.safetensors")]

# 生成视频函数（包含完整异常处理）
def generate_video(
    prompt,
    seed,
    video_width,
    video_height,
    video_length,
    infer_steps,
    lora_multiplier,
    lora_weight
):
    try:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = Path("outputs")
        output_dir.mkdir(exist_ok=True)
        
        cmd = [
            python_cmd, "hv_generate_video.py",
            "--fp8",
            "--video_size", str(video_width), str(video_height),
            "--video_length", str(video_length),
            "--infer_steps", str(infer_steps),
            "--prompt", f'"{prompt}"',
            "--save_path", str(output_dir),
            "--output_type", "both",
            "--dit", "ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states.pt",
            "--attn_mode", "sdpa",
            "--vae", "ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt",
            "--vae_chunk_size", "32",
            "--vae_spatial_tile_sample_min_size", "128",
            "--text_encoder1", "ckpts/text_encoder",
            "--text_encoder2", "ckpts/text_encoder_2",
            "--seed", str(seed),
            "--lora_multiplier", str(lora_multiplier),
        ]
        
        # 如果 lora_weight 不为空，则添加 --lora_weight 参数
        if lora_weight:
            cmd.append(f"--lora_weight={lora_weight}")
        
        # 执行生成命令
        try:
            result = subprocess.run(
                cmd,
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True
            )
            print("生成日志:\n", result.stdout)
        except subprocess.CalledProcessError as e:
            error_msg = f"视频生成失败（代码 {e.returncode}）:\n{e.output}"
            print(error_msg)
            raise RuntimeError(error_msg)
        
        # 获取输出文件
        latest_video = get_latest_video()
        if not latest_video:
            raise FileNotFoundError("未找到生成的视频文件")
            
        return str(latest_video), None
    
    except Exception as e:
        error_info = f"错误发生: {str(e)}\n\n完整追踪:\n{traceback.format_exc()}"
        print(error_info)
        return None, error_info

# 初始化模型（带错误提示）
try:
    setup_models()
except Exception as e:
    print(f"初始化失败，请检查模型配置: {str(e)}")
    # 此处可以添加 Gradio 错误提示组件

# 创建Gradio界面
with gr.Blocks(title="Hunyuan Video Generator") as demo:
    gr.Markdown("# 🎥 Hunyuan Text-to-Video Generator (LORA)")
    gr.Markdown("Generate videos using HunyuanVideo can use LoRA")
    
    with gr.Row():
        with gr.Column():
            # 核心参数
            prompt_input = gr.Textbox(
                label="Prompt",
                placeholder="Enter your video description...",
                lines=3,
                value="Unreal 5 render of a handsome man img. warm atmosphere, at home, bedroom."
            )
            seed_input = gr.Number(
                label="Seed",
                value=1234,
                precision=0
            )
            
            # 视频参数滑块
            with gr.Row():
                video_width = gr.Slider(256, 1920, value=544, step=16, label="Width")
                video_height = gr.Slider(256, 1080, value=960, step=16, label="Height")
            video_length = gr.Slider(1, 120, value=60, step=1, label="Video Length (frames)")
            infer_steps = gr.Slider(1, 100, value=30, step=1, label="Inference Steps")
            
            # LoRA参数
            lora_multiplier = gr.Slider(0.1, 2.0, value=1.0, step=0.1, label="LoRA Multiplier")
            lora_weight = gr.Dropdown(
                choices=[""] + scan_lora_weights(),
                label="LoRA Weight",
                value="",  # 默认为空
                #placeholder="Select LoRA weight or leave empty"
            )
            
            generate_btn = gr.Button("Generate Video", variant="primary")
        
        with gr.Column():
            video_output = gr.Video(label="生成结果")
            param_info = gr.Textbox(label="错误信息", visible=True)
    
    # 交互逻辑
    generate_btn.click(
        fn=generate_video,
        inputs=[
            prompt_input,
            seed_input,
            video_width,
            video_height,
            video_length,
            infer_steps,
            lora_multiplier,
            lora_weight
        ],  
        outputs=[
            video_output,
            param_info
        ]
    )

    # 示例提示词
    gr.Examples(
        examples=[
            ["1girl,[artist:WANKE|artist:free_style |[artist:ningen_mame]|ciloranko],tokoyami towa,mesugaki,devil,sensitive,dark theme,glowing eyes,silhouette,sword", ""],
        ["1boy, solo, hair: long red hair, flowing in wind, partially obscuring face, eyes: hidden by hair, but implied intense gaze, clothing: flowing tattered cloak, crimson and black, simple dark shirt underneath, leather gloves, setting: wide shot of sprawling ruins, fires burning intensely, night, full moon obscured by smoke, wind blowing debris, **scattered playing cards on ground**, **Joker card prominently visible**, pose: standing amidst scattered cards, arms slightly outstretched, as if gesturing to the chaos, mysterious aura, atmosphere: chaotic, uncertain, fateful, ominous, powerful, dramatic, sense of impending doom, quality: masterpiece, extremely aesthetic, epic scale, highly detailed ruins, dynamic composition, rim lighting from fires and moon, vibrant fire colors, vivid colors, saturated, artist style: by frank frazetta, by moebius (for lineart and detail), trending on artstation", ""],
        ["1boy, solo, fiery messy hair, red hair, orange hair, red eyes, glowing eyes, intense gaze, torn dark clothes, tattered robes, worn leather, long sleeves, ruins, burning city, fire, embers, smoke, ash, night, desolate wasteland, collapsed buildings, looking back, determined look, serious expression, masterpiece, extremely aesthetic, dreamy atmosphere, intense warm colors, vivid colors, saturated, detailed background, dramatic lighting, by rella, by konya karasue, vivid colors, saturated", ""],
            ["Unreal 5 render of a handsome man img. warm atmosphere, at home, bedroom. a small fishing village on a pier in the background.", "lora_dir/Xiang_Consis_im_lora-000006.safetensors"],
            ["Unreal 5 render of a handsome man, warm atmosphere, in a lush, vibrant forest. The scene is bathed in golden sunlight filtering through the dense canopy.", "lora_dir/Xiang_Consis_im_lora-000006.safetensors"],
        ],
        inputs=[prompt_input, lora_weight]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", share=True)
