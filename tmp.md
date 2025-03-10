```bash
huggingface-cli upload svjack/CLANNAD_Season_1_Videos ./CLANNAD_Season_1_Videos --repo-type dataset

huggingface-cli upload svjack/FateZero_Videos ./FateZero_Videos --repo-type dataset

huggingface-cli upload svjack/Tamako_Market_Videos ./Tamako_Market_Videos --repo-type dataset

huggingface-cli upload svjack/Beyond_the_Boundary_Videos ./Beyond_the_Boundary_Videos --repo-type dataset

huggingface-cli upload svjack/Nagi_no_Asukara_Videos_Captioned ./Nagi_no_Asukara_Videos_Captioned --repo-type dataset

python wan_generate_video.py --fp8 --task t2v-1.3B --video_size 768 1024 --video_length 81 --infer_steps 20 \
--save_path save --output_type both \
--dit wan2.1_t2v_1.3B_bf16.safetensors --vae Wan2.1_VAE.pth \
--t5 models_t5_umt5-xxl-enc-bf16.pth \
--attn_mode torch \
--lora_weight GOW_outputs/GOW_w1_3_lora-000006.safetensors \
--lora_multiplier 2.0 \
--prompt "In the style of Garden Of Words , The video opens with a view of a park entrance on a rainy day. The scene is dominated by the lush greenery of trees and the grey, overcast sky. Raindrops are visible in the air, creating a misty atmosphere. The park entrance features a stone gate with a sign that reads "SOS" in large letters, indicating an emergency call point. A person wearing a dark jacket and carrying a bag is seen walking towards the gate, holding an umbrella to shield themselves from the rain. The ground appears wet, reflecting the light from the surroundings. The overall color palette of the video is muted, with greens, greys, and browns being the most prominent."

python llava_qwen_video_caption.py --input_path "Beyond_the_Boundary_Videos" --output_path "Beyond_the_Boundary_Videos_Captioned" --max_frames 19 --fps 1 --force_sample

```
