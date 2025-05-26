from datasets import load_dataset
from PIL import Image
import os
from tqdm import tqdm

# 定义数据集配置列表
DATASET_CONFIGS = [
    {
        "name": "svjack/Robot_Holding_A_Sign_Images_MASK_DEPTH",
        "image_col": "original_image",
        "mask_col": "sign_mask",
        "prompt": "a robot hold the empty sign"
    },
    {
        "name": "svjack/XIAO_Holding_A_Sign_Images_MASK_DEPTH",
        "image_col": "original_image",
        "mask_col": "sign_mask",
        "prompt": "a green anime boy hold the empty sign"
    },
    {
        "name": "svjack/ZHONGLI_Holding_A_Sign_Images_MASK_DEPTH",
        "image_col": "original_image",
        "mask_col": "sign_mask",
        "prompt": "a brown anime man hold the empty sign"
    },
    {
        "name": "svjack/KAEDEHARA_KAZUHA_IM_SIGN_DEPTH_TEXT",
        "image_col": "image",
        "mask_col": "sign_mask",
        "prompt": "a red anime boy hold the empty sign"
    },
    {
        "name": "svjack/Scaramouche_IM_SIGN_DEPTH_TEXT",
        "image_col": "image",
        "mask_col": "sign_mask",
        "prompt": "a purple anime boy hold the empty sign"
    },
    {
        "name": "svjack/Xiang_Card_DreamO_Images_Filtered_CARD_MASK_RMBG",
        "image_col": "image",
        "mask_col": "sign_mask_image",
        "prompt": "a handsome glasses man hold the empty sign"
    }
]

# 创建主目录
os.makedirs("combined_output", exist_ok=True)
image_directory = os.path.join("combined_output", "image_directory")
control_directory = os.path.join("combined_output", "control_directory")
os.makedirs(image_directory, exist_ok=True)
os.makedirs(control_directory, exist_ok=True)

# 全局计数器
global_counter = 0

# 处理每个数据集
for config in DATASET_CONFIGS:
    print(f"\n正在处理数据集: {config['name']}")

    try:
        # 加载数据集
        dataset = load_dataset(config["name"])

        # 获取正确的split名称（有些数据集可能使用'train'，有些可能使用'validation'等）
        split_name = 'train' if 'train' in dataset else list(dataset.keys())[0]

        # 处理数据集中的每一行
        for row in tqdm(dataset[split_name], desc=f"处理 {config['name']}"):
            # 格式化文件名
            filename = f"{global_counter:04d}.png"

            # 处理图像
            if config["image_col"] in row and row[config["image_col"]] is not None:
                img = row[config["image_col"]].resize((1024, 1024), Image.BILINEAR)
                img.save(os.path.join(image_directory, filename))

                # 创建对应的txt文件
                with open(os.path.join(image_directory, f"{global_counter:04d}.txt"), "w") as f:
                    f.write(config["prompt"])

            # 处理mask图像
            if config["mask_col"] in row and row[config["mask_col"]] is not None:
                mask = row[config["mask_col"]].resize((1024, 1024), Image.BILINEAR)
                mask.save(os.path.join(control_directory, filename))

            global_counter += 1

    except Exception as e:
        print(f"处理数据集 {config['name']} 时出错: {str(e)}")
        continue

print(f"\n处理完成！共处理了 {global_counter} 个样本。")
print(f"图像保存在: {image_directory}")
print(f"控制图像保存在: {control_directory}")

#### toml
vim fp_single.toml

[general]
resolution = [1024, 1024]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "combined_output/image_directory"
control_directory = "combined_output/control_directory"
cache_directory = "combined_output/cache_directory"

python fpack_cache_latents.py \
    --dataset_config fp_single.toml \
    --vae HunyuanVideo/vae/diffusion_pytorch_model.safetensors \
    --image_encoder sigclip_vision_384/sigclip_vision_patch14_384.safetensors \
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 --one_frame --latent_window_size 9

python fpack_cache_text_encoder_outputs.py \
    --dataset_config fp_single.toml \
    --text_encoder1 HunyuanVideo_repackaged/split_files/text_encoders/llava_llama3_fp16.safetensors \
    --text_encoder2 HunyuanVideo_repackaged/split_files/text_encoders/clip_l.safetensors \
    --batch_size 16

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 fpack_train_network.py \
    --dit FramePackI2V_HY/diffusion_pytorch_model-00001-of-00003.safetensors \
    --vae HunyuanVideo/vae/diffusion_pytorch_model.safetensors \
    --text_encoder1 HunyuanVideo_repackaged/split_files/text_encoders/llava_llama3_fp16.safetensors \
    --text_encoder2 HunyuanVideo_repackaged/split_files/text_encoders/clip_l.safetensors \
    --image_encoder sigclip_vision_384/sigclip_vision_patch14_384.safetensors \
    --dataset_config fp_single.toml \
    --sdpa --mixed_precision bf16 \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 3.0 \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_framepack --network_dim 32 \
    --max_train_epochs 5000 --save_every_n_epochs 1 --seed 42 \
    --output_dir framepack_sign_output --output_name framepack-sign-lora --one_frame

python fpack_generate_video.py \
    --dit FramePackI2V_HY/diffusion_pytorch_model-00001-of-00003.safetensors \
    --vae HunyuanVideo/vae/diffusion_pytorch_model.safetensors \
    --text_encoder1 HunyuanVideo_repackaged/split_files/text_encoders/llava_llama3_fp16.safetensors \
    --text_encoder2 HunyuanVideo_repackaged/split_files/text_encoders/clip_l.safetensors \
    --image_encoder sigclip_vision_384/sigclip_vision_patch14_384.safetensors \
    --image_path mask.jpg \
    --prompt "a robot hold the empty sign" \
    --video_size 1088  1920 --fps 30 --infer_steps 25 \
    --attn_mode sdpa --fp8_scaled \
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 \
    --save_path save --video_sections 1 --output_type latent_images --one_frame_inference zero_post \
    --seed 1234 --lora_multiplier 1.0 --lora_weight framepack_sign_output/framepack-sign-lora-000004.safetensors

vim real_prompt.txt

A silver-haired cyborg with glowing blue neural circuits pulsing beneath synthetic skin holds the empty sign under the flickering neon lights of a rain-slicked cyberpunk alley, steam rising from nearby vents as holographic advertisements in Japanese and Chinese characters flicker erratically overhead.
A freckle-faced girl of about twelve in a bright yellow raincoat several sizes too large holds the empty sign while crouched under a rusting fire escape in the pouring rain, her oversized boots splashing in growing puddles as water drips rhythmically from the broken gutter above her.
A towering war robot standing nearly three meters tall with heavily dented titanium armor plates and exposed hydraulic systems holds the empty sign amidst the smoldering wreckage of a battlefield littered with debris - shattered mech parts, spent shell casings, and the occasional still-sparking wire creating an apocalyptic tableau.
An elderly Asian gentleman with a long white beard that reaches his waist and deep smile lines holds the empty sign outside a traditional wooden tea house, the scent of oolong and jasmine mixing with the morning mist rising from the nearby bamboo forest where songbirds greet the dawn.
A purple-skinned alien with three vertically-stacked glowing yellow eyes and elongated limbs holds the empty sign in the bustling departure terminal of a futuristic spaceport, surrounded by floating luggage drones and holographic departure boards displaying destinations like Proxima Centauri and Europa Colony.
A lanky teenage hacker with augmented reality glasses projecting scrolling code across their lenses holds the empty sign while leaning against a support beam in a crowded Tokyo subway station, the press of commuters creating waves of movement as train announcements echo through the tiled corridors.
A golden-plated android with a severely cracked facial plate revealing glimpses of whirring optical mechanisms beneath holds the empty sign in the ruins of an abandoned laboratory, broken glass crunching underfoot as emergency lighting flickers intermittently across overturned examination tables and shattered specimen jars.
A barrel-chested blacksmith with arms covered in soot and old burn scars holds the empty sign outside his forge at sunset, the glow from the furnace casting long shadows across the dirt yard where various half-finished metalworks await completion.
A petite ballerina in a tattered pink tutu with several missing sequins holds the empty sign center stage in an abandoned theater, dust motes floating through shafts of light that penetrate holes in the collapsed ceiling as the ghostly echoes of applause seem to linger in the air.
A masked vigilante in a long trench coat with the collar turned up against the cold holds the empty sign on the edge of a foggy rooftop, the blinking lights of the city stretching out below as police sirens wail somewhere in the distance.
A red-haired mechanic with grease smeared across her freckled cheeks holds the empty sign in the cluttered garage where she works, surrounded by engine parts and tools hanging neatly on pegboards as the smell of motor oil and gasoline permeates the air.
A translucent holographic woman composed of shimmering blue light particles holds the empty sign in the sterile environment of a high-tech control room, her form occasionally glitching as she stands before a wall of monitors displaying complex schematics and scrolling data streams.
A blind monk with milky white eyes and a gnarled wooden staff holds the empty sign at the gate of a mountain temple, the morning chant of his fellow monks audible from the main hall while incense smoke drifts lazily through the ancient cedar trees surrounding the compound.
A chrome-plated security droid with glowing red ocular sensors holds the empty sign in the ultramodern lobby of a corporate skyscraper, its polished surface reflecting the nervous faces of employees passing through the biometric scanners on their way to the elevators.
A curly-haired child no older than six in footed pajamas holds the empty sign while standing on their bed in a moonlit bedroom, stuffed animals arranged in a semicircle as if holding court, the nightlight casting soft shapes of stars and planets on the walls.
A scarred war veteran with a sophisticated prosthetic arm containing visible hydraulics holds the empty sign at the base of a massive stone memorial engraved with hundreds of names, his eyes fixed on one in particular as autumn leaves drift down around him.
A steampunk inventor with welding goggles pushed up onto his forehead holds the empty sign in his cluttered workshop, surrounded by half-built contraptions of brass and wood, blueprints pinned haphazardly to every available surface while a complicated clockwork device ticks loudly in the corner.
A heavily tattooed biker in weathered leathers holds the empty sign while straddling his motorcycle outside a roadside diner, the smell of frying burgers mixing with exhaust fumes as truckers come and go through the buzzing neon sign that reads "EAT" in bright red letters.
A floating AI core suspended in a spherical containment field with dozens of glowing wires connecting it to the ceiling holds the empty sign in the humming server room, cooling fans creating a constant white noise as status lights blink rhythmically across racks of equipment.
A nun in traditional black-and-white habit holds the empty sign in the quiet solitude of a candlelit chapel, the flickering flames casting dancing shadows across the stone walls where centuries of prayers seem to linger in the very air.
A post-apocalyptic scavenger wearing a patched-together gas mask holds the empty sign in the ruins of what was once a department store, now just crumbling concrete and twisted metal, the distant howl of something unnatural echoing through the skeletal remains of the city.
A Japanese shrine maiden in red and white ceremonial robes holds the empty sign beneath an ancient cherry blossom tree, delicate pink petals drifting down around her as wind chimes tinkle softly from the nearby shrine where visitors come to make offerings.
A cowboy-themed android with a Stetson hat tilted at a rakish angle holds the empty sign in the middle of a desert ghost town, tumbleweeds bouncing past abandoned storefronts where swinging saloon doors creak ominously in the hot, dry wind.
A pregnant woman in stained overalls with a bandana tied around her forehead holds the empty sign in the middle of a thriving community garden, the rich smell of turned earth and growing plants surrounding her as neighbors tend their plots and children chase butterflies between the rows.
A disheveled scientist with wild Einstein-like hair holds the empty sign in a laboratory that shows clear signs of a recent explosion - scorch marks radiate outward from a central point where equipment lies in ruins, papers still fluttering down from where they were blown into the air.
A mime artist in full whiteface makeup with exaggerated black features holds the empty sign on a busy Parisian street corner, frozen in an invisible box as tourists snap photos and children giggle at his antics while the sounds of the city swirl around him.
A futuristic delivery drone with a cracked display screen and one slightly bent rotor holds the empty sign on the railing of a high-rise balcony, its camera lens focusing and refocusing as it waits for repairs while the city sprawls out below in a glittering nighttime panorama.
A grizzled old fisherman with a corncob pipe clenched between his teeth holds the empty sign on the weathered wooden dock where he's moored his boat, the smell of saltwater and fish strong in the air as seagulls cry overhead and waves lap rhythmically against the pilings.
A Victorian-era ghost with translucent skin that reveals glimpses of the wallpaper behind him holds the empty sign in the dusty drawing room of an abandoned mansion, his mournful expression matching the general state of decay around him as moonlight filters through tattered curtains.
A young programmer with dark circles under his eyes from too many sleepless nights holds the empty sign in his messy cubicle, empty energy drink cans and snack wrappers littering the desk where multiple monitors display lines of code, one showing the infamous "blue screen of death."
A medieval knight in armor that's seen better days - dents and scratches telling stories of recent battles - holds the empty sign on a war-torn field where the bodies of fallen comrades and enemies alike lie scattered, crows already gathering in the gray afternoon light.
A retro-futuristic housewife robot with a chrome-plated smile and a polka-dotted apron holds the empty sign in a perfectly preserved 1950s-style kitchen, where all the appliances look simultaneously cutting-edge and hopelessly outdated by modern standards, the smell of apple pie somehow emanating from non-functional ovens.
A homeless veteran with all his worldly possessions in a shopping cart holds the empty sign under a graffiti-covered bridge where the concrete is stained with decades of urban grime, the distant sounds of traffic mixing with the closer dripping of water from pipes overhead.
A cybernetic ninja with glowing red eyes visible through his face mask holds the empty sign on the edge of a rooftop overlooking the neon-drenched city, his black-clad form nearly invisible against the night sky except for those eerie eyes and the occasional glint of metal from his prosthetics.
An elderly black woman with a vibrantly colored headwrap holds the empty sign on her porch swing, the gentle creak of the chains mixing with the sounds of wind chimes and distant children playing as she surveys the quiet street where she's lived for fifty years.
A child-sized helper robot with enormous expressive eyes holds the empty sign in the fluorescent-lit hallway of a busy hospital, its white plastic shell scuffed from years of service as doctors and nurses rush past, some pausing to pat its dome-shaped head affectionately.
A bearded lumberjack in red-and-black checked flannel holds the empty sign at the center of a forest clearing he's just finished chopping, the rich scent of freshly cut pine strong in the air as wood chips litter the ground around his heavy boots and the stump he's using as a makeshift table.
A futuristic soldier encased in powered armor with visible weapon ports holds the empty sign in the middle of a war-torn street where buildings lie in ruins, fires still smoldering in some places as the occasional burst of gunfire can be heard a few blocks away.
A young witch with emerald green hair that seems to move slightly on its own holds the empty sign in a cramped occult bookstore where the shelves are overflowing with ancient-looking tomes, the air thick with the scent of incense and something more mysterious that makes the shadows seem deeper than they should be.
A maintenance robot with multiple articulated arms folded neatly against its torso holds the empty sign in the sterile white corridor of a space station, the curved walls giving a slight sensation of disorientation as it stands near a large observation window showing the blue curve of Earth far below.
A masked plague doctor in a long black coat and beaked mask holds the empty sign in the fog-shrouded streets of a medieval town where the occasional cough or moan can be heard from behind shuttered windows, the smell of vinegar and herbs hanging heavy in the damp air.
A retro gaming robot with pixelated eyes that change expression in blocky increments holds the empty sign in a dimly lit arcade where the sounds of 8-bit music and joystick clicks create a nostalgic cacophony, its metal body reflecting the glow of CRT screens displaying classic games.
A tired nurse in wrinkled blue scrubs holds the empty sign in the hospital break room where a microwave hums and an ancient coffee maker drips its last few drops into a stained carafe, the bulletin board behind her covered in memos and schedules that no one has the energy to read anymore.
A samurai robot with a cracked visor that reveals glimpses of flickering optics beneath holds the empty sign in a peaceful bamboo forest where the tall stalks creak gently in the breeze, its armored feet sinking slightly into the soft earth as birds sing unconcernedly overhead.
A little girl in an oversized dinosaur costume holds the empty sign in the middle of a playground where other children run and scream with laughter, her costume's tail dragging in the wood chips as she watches the action from inside her green felt and foam T-rex outfit.
A postman with a heavy leather satchel slung across his shoulder holds the empty sign on the neatly trimmed lawn of a suburban doorstep where a "Beware of Dog" sign contradicts the tiny yipping coming from inside, his uniform crisp despite the humidity that makes his forehead shine with sweat.
A deep-sea diving suit with a broken faceplate that reveals only darkness within holds the empty sign on the ocean floor where strange fish dart away from its hulking form, streams of bubbles rising from various joints as it stands amidst the wreckage of some long-lost vessel.
A street performer dressed as a living statue with metallic silver paint covering every visible inch of skin holds the empty sign in the bustling town square where tourists drop coins into his hat, his pose so perfect and still that children sometimes dare each other to touch him.
A librarian with half-moon glasses perched on her nose holds the empty sign in the hushed silence of an ancient library where the smell of old paper and leather bindings mixes with wood polish, sunlight filtering through high windows to illuminate the dust motes floating between towering bookshelves.
A broken-down factory robot missing an arm and with exposed wiring in its chest cavity holds the empty sign in the echoing emptiness of an abandoned warehouse where rusted machinery stands silent, the only movement coming from pigeons that have nested in the broken skylights far above.

vim manga_sign_prompt.txt

A shonen manga protagonist with wild, spiky golden hair holds the empty sign while standing amidst swirling dust clouds, his tattered cape billowing dramatically behind him as speed lines radiate outward to emphasize his explosive energy.
A shojo manga heroine with impossibly large, starry violet eyes holds the empty sign delicately between her gloved hands, surrounded by floating cherry blossoms and sparkling screentone effects that create a dreamy atmosphere.
In a gritty seinen manga panel, a grizzled detective holds the empty sign under the flickering neon lights of a rainy alleyway, the heavy crosshatching shadows obscuring half his scarred face while cigarette smoke curls around him.
A chibi manga mascot character with an oversized head and tiny body bounces energetically while holding the empty sign, spring-shaped motion trails emphasizing each exaggerated hop across the page.
A cyberpunk manga android with glowing circuit tattoos holds the empty sign against a dystopian cityscape, digital glitch effects distorting the holographic advertisements that float behind its sleek chrome-plated body.
A sports manga athlete mid-dunk holds the empty sign with one hand while basketball motion lines streak behind him, sweat droplets flying from his determined face as the basket's net stretches dramatically.
A horror manga ghost child floats eerily while holding the empty sign, their translucent form rendered with wispy screentone patterns that fade into the haunted schoolhouse background.
A cooking manga chef dramatically holds the empty sign aloft while flames erupt from the kitchen behind him, ingredients and utensils suspended mid-air with speed lines emphasizing the chaotic energy.
A fantasy manga elf archer gracefully holds the empty sign while standing atop an ancient tree, magical energy circles pulsing beneath their feet as glowing runes float in the air around them.
A historical manga samurai holds the empty sign with quiet dignity amidst a shower of cherry blossoms, his tattered kimono sleeves fluttering in the wind as ink wash-style mountains loom in the background.
A mecha manga pilot holds the empty sign inside their damaged cockpit, the reflection in their cracked visor showing the burning battlefield outside with mechanical parts flying through the air.
A magical girl manga transformation sequence freezes mid-pose as the heroine holds the empty sign, crystalline shard effects radiating outward while her hair and ribbons defy gravity in perfect spirals.
A yakuza manga enforcer holds the empty sign in a smoky backroom, his scarred knuckles gripping it tightly as traditional Japanese patterns swirl in the background, suggesting hidden danger.
A school manga delinquent leans cockily against a wall while holding the empty sign, his oversized uniform jacket flaring dramatically as impact stars emphasize his challenging glare.
A sci-fi manga astronaut floats in zero gravity while holding the empty sign, their helmet visor reflecting the swirling nebula outside the space station's observation window.
A fantasy manga dragon tamer holds the empty sign confidently as a massive serpentine creature coils protectively around them, scale textures meticulously rendered with intricate screentone patterns.
A vampire manga lord elegantly holds the empty sign in a gothic cathedral, blood droplets suspended mid-air around him as candlelight casts dramatic shadows across his pale features.
A ninja manga shinobi holds the empty sign while crouched on a rooftop, swirling autumn leaves obscuring half their form as they prepare to vanish in a puff of smoke.
A pirate manga captain boldly holds the empty sign on the deck of their ship, woodcut-style waves crashing dramatically in the background as the crew cheers behind them.
A zombie manga survivor desperately holds the empty sign in a ruined city, decayed screentone textures emphasizing their deteriorating condition while hordes lurk in the shadows.
A robot manga companion gently holds the empty sign with their mechanical hands, intricate gear and circuit patterns visible through transparent panels in their sleek white armor.
A demon manga overlord ominously holds the empty sign on their hellish throne, flames licking at the edges of the panel as damned souls swirl in the molten background.
An angel manga warrior solemnly holds the empty sign at the gates of paradise, their feathery wings creating delicate screentone patterns against the golden light.
A detective manga assistant curiously holds the empty sign up to a magnifying glass, the reflection showing distorted clues in a complex web of mystery lines.
A monster manga trainer excitedly holds the empty sign as bizarre creatures of various shapes and sizes crowd around them, each rendered with unique texture patterns.
A ronin manga wanderer holds the empty sign at a crossroads, ink wash-style rain falling diagonally across the panel as their straw hat obscures their world-weary eyes.
A knight manga protector stands resolutely holding the empty sign before a castle gate, their chainmail armor rendered with meticulous dot patterns that catch the morning light.
A witch manga apprentice nervously holds the empty sign in a cluttered potion shop, floating spellbook pages and bubbling cauldrons creating chaotic background elements.
A superhero manga sidekick proudly holds the empty sign on a rooftop, their simple costume design contrasting with the detailed cityscape spread out behind them.
A gang manga leader arrogantly holds the empty sign on their motorcycle, graffiti-style tags and spray paint effects covering the brick wall behind them.
A doctor manga surgeon intensely holds the empty sign in an operating room, medical diagrams and floating instrument silhouettes adding tension to the scene.
A chef manga prodigy dramatically holds the empty sign in a kitchen battlefield, exaggerated steam clouds rising from sizzling pans as ingredients fly through the air.
An artist manga creator thoughtfully holds the empty sign in a messy studio, paint splatters and ink washes creating abstract patterns across their smock.
A musician manga performer passionately holds the empty sign on stage, musical notes and sound waves radiating outward in vibrant screentone patterns.
A dancer manga star gracefully holds the empty sign mid-pirouette, motion lines tracing the arc of their movement across the polished stage floor.
A pilot manga ace confidently holds the empty sign in the cockpit, cloud formations streaming past the canopy in streaked, speed-lined patterns.
A soldier manga veteran wearily holds the empty sign in a trench, camouflage patterns blending into the muddy background as smoke drifts across no man's land.
A spy manga agent stealthily holds the empty sign in a ventilation shaft, their silhouette barely visible against the gridded metal background.
A thief manga phantom mysteriously holds the empty sign on a rooftop, their shadow stretching long across the moonlit cityscape below.
A hunter manga tracker carefully holds the empty sign in a dense forest, animal paw prints and broken branches telling a story in the detailed background.
A fisherman manga master patiently holds the empty sign on a weathered dock, water ripple effects spreading outward from the boat's hull below.
A farmer manga grower happily holds the empty sign in a sun-drenched field, rows of crops stretching into the distance with perfect perspective lines.
A blacksmith manga artisan proudly holds the empty sign in their forge, the glowing metal textures contrasting with the dark, soot-stained walls.
A tailor manga designer meticulously holds the empty sign in their workshop, fabric swatches and dress patterns floating in organized chaos around them.
A librarian manga keeper quietly holds the empty sign in a vast book archive, the shelves stretching into infinity with delicate screentone shading.
A scientist manga researcher excitedly holds the empty sign in a lab, complex equations and molecular structures floating in the air around bubbling beakers.
An engineer manga builder confidently holds the empty sign at a construction site, blueprint lines and geometric shapes overlaying the half-built structure.
An astronaut manga explorer awestruck holds the empty sign on an alien world, bizarre flora and strange rock formations filling the panel with wonder.
A diver manga adventurer carefully holds the empty sign near a coral reef, bubble trails and light refraction effects creating an underwater atmosphere.
An explorer manga discoverer triumphantly holds the empty sign atop a newly mapped peak, the vast landscape unfolding below in detailed isometric perspective.

vim manga_sign_prompt_5.txt

A shonen manga protagonist with wild, spiky golden hair holds the empty sign while standing amidst swirling dust clouds, his tattered cape billowing dramatically behind him as speed lines radiate outward to emphasize his explosive energy.
A shojo manga heroine with impossibly large, starry violet eyes holds the empty sign delicately between her gloved hands, surrounded by floating cherry blossoms and sparkling screentone effects that create a dreamy atmosphere.
In a gritty seinen manga panel, a grizzled detective holds the empty sign under the flickering neon lights of a rainy alleyway, the heavy crosshatching shadows obscuring half his scarred face while cigarette smoke curls around him.
A chibi manga mascot character with an oversized head and tiny body bounces energetically while holding the empty sign, spring-shaped motion trails emphasizing each exaggerated hop across the page.
A cyberpunk manga android with glowing circuit tattoos holds the empty sign against a dystopian cityscape, digital glitch effects distorting the holographic advertisements that float behind its sleek chrome-plated body.

combined_output/control_directory

import os

def add_image_paths_to_prompts(input_txt, output_txt, image_dir):
    # 获取所有.png文件并按字母顺序排序
    png_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])

    with open(input_txt, 'r', encoding='utf-8') as infile, \
         open(output_txt, 'w', encoding='utf-8') as outfile:

        for i, line in enumerate(infile):
            prompt = line.strip()
            if i < len(png_files):  # 确保有对应的png文件
                png_path = os.path.join(image_dir, png_files[i])
                new_line = f"{prompt} --i {png_path}\n"
                outfile.write(new_line)
            else:
                # 如果png文件比prompt少，只写入原始prompt
                outfile.write(f"{prompt}\n")

# 使用示例
input_txt = 'manga_sign_prompt_5.txt'
output_txt = 'manga_sign_prompt_5_with_images.txt'
image_dir = 'combined_output/control_directory'

add_image_paths_to_prompts(input_txt, output_txt, image_dir)

# 使用示例
input_txt = 'manga_sign_prompt.txt'
output_txt = 'manga_sign_prompt_with_images.txt'
image_dir = 'combined_output/control_directory'

add_image_paths_to_prompts(input_txt, output_txt, image_dir)

# 使用示例
input_txt = 'real_prompt.txt'
output_txt = 'real_prompt_with_images.txt'
image_dir = 'combined_output/control_directory'

add_image_paths_to_prompts(input_txt, output_txt, image_dir)

python fpack_generate_video.py \
    --dit FramePackI2V_HY/diffusion_pytorch_model-00001-of-00003.safetensors \
    --vae HunyuanVideo/vae/diffusion_pytorch_model.safetensors \
    --text_encoder1 HunyuanVideo_repackaged/split_files/text_encoders/llava_llama3_fp16.safetensors \
    --text_encoder2 HunyuanVideo_repackaged/split_files/text_encoders/clip_l.safetensors \
    --image_encoder sigclip_vision_384/sigclip_vision_patch14_384.safetensors \
    --image_path mask.jpg \
    --from_file manga_sign_prompt_with_images.txt \
    --video_size 512  512 --fps 30 --infer_steps 25 \
    --attn_mode sdpa --fp8_scaled \
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 \
    --save_path manga_save --video_sections 1 --output_type latent_images --one_frame_inference zero_post \
    --seed 1234 --lora_multiplier 1.0 --lora_weight framepack_sign_output/framepack-sign-lora-000004.safetensors

python fpack_generate_video.py \
    --dit FramePackI2V_HY/diffusion_pytorch_model-00001-of-00003.safetensors \
    --vae HunyuanVideo/vae/diffusion_pytorch_model.safetensors \
    --text_encoder1 HunyuanVideo_repackaged/split_files/text_encoders/llava_llama3_fp16.safetensors \
    --text_encoder2 HunyuanVideo_repackaged/split_files/text_encoders/clip_l.safetensors \
    --image_encoder sigclip_vision_384/sigclip_vision_patch14_384.safetensors \
    --image_path mask.jpg \
    --from_file real_prompt_with_images.txt \
    --video_size 512  512 --fps 30 --infer_steps 25 \
    --attn_mode sdpa --fp8_scaled \
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 \
    --save_path real_save --video_sections 1 --output_type latent_images --one_frame_inference zero_post \
    --seed 1234 --lora_multiplier 1.0 --lora_weight framepack_sign_output/framepack-sign-lora-000004.safetensors

import os
from datasets import Dataset, Image

def create_huggingface_dataset(prompt_file, image_dir):
    # 读取prompt文件，解析prompt和mask_image
    prompts = []
    mask_images = []
    with open(prompt_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' --i ')
            prompt = parts[0]
            mask_image = parts[1] if len(parts) > 1 else None
            prompts.append(prompt)
            mask_images.append(mask_image)

    # 获取并排序PNG文件
    png_files = sorted([f for f in os.listdir(image_dir) if f.endswith('.png')])
    image_paths = [os.path.join(image_dir, f) for f in png_files]

    # 确保数据长度一致
    min_length = min(len(prompts), len(image_paths))
    prompts = prompts[:min_length]
    mask_images = mask_images[:min_length]
    image_paths = image_paths[:min_length]

    # 创建数据集字典
    data_dict = {
        "prompt": prompts,
        "mask_image": mask_images,
        "image": image_paths
    }

    # 创建Hugging Face数据集
    dataset = Dataset.from_dict(data_dict)
    dataset = dataset.cast_column("mask_image", Image())
    dataset = dataset.cast_column("image", Image())
    return dataset

# 使用示例
prompt_file = "manga_sign_prompt_with_images.txt"
image_dir = "manga_save"
dataset = create_huggingface_dataset(prompt_file, image_dir)

dataset.push_to_hub("svjack/FramePack_mask_to_sign_dataset_0")

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

!pip install -U datasets

from datasets import Dataset, load_dataset

# 流式加载数据集
ds = load_dataset("UCSC-VLAA/HQ-Edit", streaming=True)

# 提取前 2500 个样本
samples = []
for batch in ds["train"].take(2500):
    samples.append(batch)

# 转换为新的 Dataset 对象
new_dataset = Dataset.from_dict({
    "input": [sample["input"] for sample in samples],
    "input_image": [sample["input_image"] for sample in samples],
    "edit": [sample["edit"] for sample in samples],
    "inverse_edit": [sample["inverse_edit"] for sample in samples],
    "output": [sample["output"] for sample in samples],
    "output_image": [sample["output_image"] for sample in samples],
})

# 保存到磁盘（可选）
#new_dataset.save_to_disk("hq_edit_2500_samples")

new_dataset.push_to_hub("svjack/HQ-Edit-Sample-2500")

from datasets import load_dataset
import cv2
import numpy as np
import os
from tqdm import tqdm

# 目标尺寸
TARGET_WIDTH = 512
TARGET_HEIGHT = 512

# 创建输出目录
os.makedirs("edit_output/control_directory", exist_ok=True)
os.makedirs("edit_output/image_directory", exist_ok=True)

# 流式加载并处理数据集
for i, sample in tqdm(enumerate(load_dataset("UCSC-VLAA/HQ-Edit", streaming=True)["train"].take(2500)),
                      total=2500,
                      desc="Processing samples"):
    # 生成文件名前缀，格式为000i
    filename_prefix = f"{i:04d}"

    # 处理 input_image（保持宽高比 + 居中填充）
    input_img = cv2.cvtColor(np.array(sample["input_image"]), cv2.COLOR_RGB2BGR)
    h, w = input_img.shape[:2]
    scale = min(TARGET_WIDTH / w, TARGET_HEIGHT / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_img = cv2.resize(input_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    background = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    x_offset = (TARGET_WIDTH - new_w) // 2
    y_offset = (TARGET_HEIGHT - new_h) // 2
    background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    cv2.imwrite(f"edit_output/control_directory/{filename_prefix}.png", background)

    # 保存 edit 文本
    with open(f"edit_output/image_directory/{filename_prefix}.txt", "w") as f:
        f.write(sample["edit"])

    # 处理 output_image（保持宽高比 + 居中填充）
    output_img = cv2.cvtColor(np.array(sample["output_image"]), cv2.COLOR_RGB2BGR)
    h, w = output_img.shape[:2]
    scale = min(TARGET_WIDTH / w, TARGET_HEIGHT / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized_img = cv2.resize(output_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    background = np.zeros((TARGET_HEIGHT, TARGET_WIDTH, 3), dtype=np.uint8)
    x_offset = (TARGET_WIDTH - new_w) // 2
    y_offset = (TARGET_HEIGHT - new_h) // 2
    background[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    cv2.imwrite(f"edit_output/image_directory/{filename_prefix}.png", background)


#### toml
vim fp_single.toml

[general]
resolution = [512, 512]
caption_extension = ".txt"
batch_size = 1
enable_bucket = true
bucket_no_upscale = false

[[datasets]]
image_directory = "edit_output/image_directory"
control_directory = "edit_output/control_directory"
cache_directory = "edit_output/cache_directory"

python fpack_cache_latents.py \
    --dataset_config fp_single.toml \
    --vae HunyuanVideo/vae/diffusion_pytorch_model.safetensors \
    --image_encoder sigclip_vision_384/sigclip_vision_patch14_384.safetensors \
    --vae_chunk_size 32 --vae_spatial_tile_sample_min_size 128 --one_frame --latent_window_size 9

python fpack_cache_text_encoder_outputs.py \
    --dataset_config fp_single.toml \
    --text_encoder1 HunyuanVideo_repackaged/split_files/text_encoders/llava_llama3_fp16.safetensors \
    --text_encoder2 HunyuanVideo_repackaged/split_files/text_encoders/clip_l.safetensors \
    --batch_size 16

accelerate launch --num_cpu_threads_per_process 1 --mixed_precision bf16 fpack_train_network.py \
    --dit FramePackI2V_HY/diffusion_pytorch_model-00001-of-00003.safetensors \
    --vae HunyuanVideo/vae/diffusion_pytorch_model.safetensors \
    --text_encoder1 HunyuanVideo_repackaged/split_files/text_encoders/llava_llama3_fp16.safetensors \
    --text_encoder2 HunyuanVideo_repackaged/split_files/text_encoders/clip_l.safetensors \
    --image_encoder sigclip_vision_384/sigclip_vision_patch14_384.safetensors \
    --dataset_config fp_single.toml \
    --sdpa --mixed_precision bf16 \
    --optimizer_type adamw8bit --learning_rate 2e-4 --gradient_checkpointing \
    --timestep_sampling shift --weighting_scheme none --discrete_flow_shift 3.0 \
    --max_data_loader_n_workers 2 --persistent_data_loader_workers \
    --network_module networks.lora_framepack --network_dim 32 \
    --max_train_epochs 5000 --save_every_n_epochs 1 --seed 42 \
    --output_dir framepack_edit_output --output_name framepack-edit-lora --one_frame
