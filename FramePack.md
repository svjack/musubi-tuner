```bash

sudo apt-get update && sudo apt-get install git-lfs ffmpeg cbm 

git clone -b frame-pack-inference https://github.com/kohya-ss/musubi-tuner.git
cd musubi-tuner

pip install torch torchvision
pip install -r requirements.txt
pip install ascii-magic matplotlib tensorboard huggingface_hub datasets
pip install moviepy==1.0.3
pip install sageattention==1.0.6

git clone https://huggingface.co/lllyasviel/FramePackI2V_HY
->
--dit FramePackI2V_HY/diffusion_pytorch_model-00001-of-00003.safetensors

huggingface-cli download tencent/HunyuanVideo --local-dir ./ckpts
->
ckpts/hunyuan-video-t2v-720p/vae/pytorch_model.pt
OR 
git clone https://huggingface.co/hunyuanvideo-community/HunyuanVideo
->
--vae HunyuanVideo/vae/diffusion_pytorch_model.safetensors

git clone https://huggingface.co/Comfy-Org/HunyuanVideo_repackaged
->
--text_encoder1 HunyuanVideo_repackaged/split_files/text_encoders/llava_llama3_fp16.safetensors
--text_encoder2 HunyuanVideo_repackaged/split_files/text_encoders/clip_l.safetensors 

git clone https://huggingface.co/Comfy-Org/sigclip_vision_384
->
--image_encoder sigclip_vision_384/sigclip_vision_patch14_384.safetensors


```
