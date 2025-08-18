### Model download

```bash
huggingface-cli download Comfy-Org/Qwen-Image_ComfyUI split_files/text_encoders/qwen_2.5_vl_7b.safetensors --local-dir="."
huggingface-cli download Comfy-Org/Qwen-Image_ComfyUI split_files/diffusion_models/qwen_image_bf16.safetensors --local-dir="."
huggingface-cli download Qwen/Qwen-Image vae/diffusion_pytorch_model.safetensors --local-dir="."
```

### Caption

```bash
git clone https://huggingface.co/datasets/svjack/GPT-4o-Design-Images

edit default caption in src/musubi_tuner/caption_images_by_qwen_vl.py 

DEFAULT_PROMPT = """# Image Annotator
You are a professional image annotator. Please complete the following task based on the input image.
## Create Image Caption
1. Write the caption using natural, descriptive text without structured formats or rich text.
2. Enrich caption details by including: object attributes, vision relations between objects, and environmental details.
3. Identify the text visible in the image, without translation or explanation, and highlight it in the caption with quotation marks.
4. Maintain authenticity and accuracy, avoid generalizations.
5. Every caption starts with : "In the style of GPT-4o-Design-Images, Generate 4 image samples for the current design concept and piece them into 4 square blocks."
"""

python src/musubi_tuner/caption_images_by_qwen_vl.py --image_dir GPT-4o-Design-Images \
 --model_path qwen_2.5_vl_7b.safetensors --output_format text
```

### Qwen Image

vim image_config.toml

```toml
[[datasets]]
image_directory = "Xiang_Images_QwenVL_2_5_Captioned"

[general]
resolution = [ 832, 480,]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false
```

```toml
[[datasets]]
image_directory = "GPT-4o-Design-Images-QwenVL_2_5_Captioned"

[general]
resolution = [ 512, 512,]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false
```

```bash
python src/musubi_tuner/qwen_image_cache_latents.py \
    --dataset_config image_config.toml \
    --vae diffusion_pytorch_model.safetensors


python src/musubi_tuner/qwen_image_cache_text_encoder_outputs.py \
    --dataset_config image_config.toml \
    --text_encoder qwen_2.5_vl_7b.safetensors \
    --batch_size 16
```

```bash
vim Xiang.txt 

​​夏日清凉：​​ 王翔，​​一个戴着眼镜的清爽青年，身穿简约白色T恤和卡其色短裤，站在阳光斑驳的树荫下，笑容灿烂地品尝着一支缀满巧克力碎的香草冰淇淋。​​ 暖色调，生活感镜头。
```

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/qwen_image_train_network.py \
    --dit qwen_image_bf16.safetensors  \
    --vae diffusion_pytorch_model.safetensors \
    --text_encoder qwen_2.5_vl_7b.safetensors \
    --dataset_config image_config.toml \
    --network_module=networks.lora_qwen_image \
    --sdpa --mixed_precision bf16 --fp8_base --fp8_scaled --fp8_vl --blocks_to_swap 16 \
    --timestep_sampling shift \
    --weighting_scheme none --discrete_flow_shift 3.0 \
    --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_dim 32 \
    --max_train_epochs 500 --save_every_n_steps 365 --seed 42 \
    --output_dir xiang_qwen_image_output --output_name xiang_qwen_image_lora \
    --sample_prompts Xiang.txt --sample_every_n_steps 365 --sample_at_first
```

```bash
vim Four.txt 

In the style of GPT-4o-Design-Images, Generate 4 image samples for the current design concept and piece them into 4 square blocks. 1 is a bright yellow gelato ball, topped with white ghost-shaped chocolate chips, with a neutral beige background; Square 2 is a pure white sundae cup with a cream top shaped like the outline of the Apple logo, and a light beige table; Square 3 is a brown chocolate ice cream with a mustache face frosting and a red syrup bow tie on the surface, with a soft beige background; Square 4 is an orange popsicle with a green leaf pattern and the Fanta blue trademark, with a light beige base. All products use a 3D cartoon texture, with soft shadows and a uniform background color. The style is 3D cartoon, and the ratio is 1:1
```

```bash
accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 src/musubi_tuner/qwen_image_train_network.py \
    --dit qwen_image_bf16.safetensors  \
    --vae diffusion_pytorch_model.safetensors \
    --text_encoder qwen_2.5_vl_7b.safetensors \
    --dataset_config image_config.toml \
    --network_module=networks.lora_qwen_image \
    --sdpa --mixed_precision bf16 --fp8_base --fp8_scaled --fp8_vl --blocks_to_swap 16 \
    --timestep_sampling shift \
    --weighting_scheme none --discrete_flow_shift 3.0 \
    --optimizer_type adamw8bit --learning_rate 1e-4 --gradient_checkpointing \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_dim 32 \
    --max_train_epochs 500 --save_every_n_steps 365 --seed 42 \
    --output_dir Four_qwen_image_output --output_name Four_qwen_image_lora \
    --sample_prompts Four.txt --sample_every_n_steps 365 --sample_at_first
```
