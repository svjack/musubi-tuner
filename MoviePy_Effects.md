以下是 MoviePy 支持的视频特效分类总结，涵盖图片处理、视频处理、动态生成和高级合成四大类，并提供效果解释和示例代码：

---

**一、图片处理特效**
针对静态图像或序列帧的加工处理。

1. **基础滤镜**
• 效果：调整颜色、对比度、模糊等。

• 代码示例：

  ```python
  from moviepy.editor import ImageClip, vfx

  clip = ImageClip("input.jpg")
  clip = clip.fx(vfx.colorx, 0.5)     # 降低饱和度
  clip = clip.fx(vfx.blackwhite)       # 黑白滤镜
  clip = clip.fx(vfx.gaussian_blur, 5) # 高斯模糊
  clip.write_videofile("output.mp4", fps=24)
  ```

2. **几何变换**
• 效果：旋转、裁剪、缩放、镜像。

• 代码示例：

  ```python
  clip = clip.fx(vfx.rotate, 45)      # 旋转45度
  clip = clip.fx(vfx.crop, x1=100, y1=100, x2=540, y2=380)  # 裁剪
  clip = clip.fx(vfx.mirror_x)        # 水平镜像
  ```

3. **遮罩与蒙版**
• 效果：圆形/矩形聚焦、自定义形状遮罩。

• 代码示例：

  ```python
  from moviepy.video.tools.drawing import circle_mask

  mask = circle_mask(clip.size, center=(320, 240), radius=100)  # 圆形遮罩
  clip = clip.set_mask(mask)  # 应用遮罩（仅显示圆形区域）
  ```

---

**二、视频处理特效**
针对动态视频流的处理与合成。

1. **转场效果**
• 效果：淡入淡出、滑动切换、溶解过渡。

• 代码示例：

  ```python
  from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx

  clip1 = VideoFileClip("clip1.mp4").fx(vfx.fadeout, 1)  # 淡出
  clip2 = VideoFileClip("clip2.mp4").fx(vfx.fadein, 1)    # 淡入
  final = concatenate_videoclips([clip1, clip2], method="compose")  # 拼接
  ```

2. **时间操控**
• 效果：加速、减速、倒放、循环播放。

• 代码示例：

  ```python
  clip = clip.fx(vfx.speedx, 2)       # 2倍速
  clip = clip.fx(vfx.time_mirror)     # 倒放
  ```

3. **动态叠加**
• 效果：画中画、滚动字幕、动态水印。

• 代码示例：

  ```python
  from moviepy.editor import TextClip, CompositeVideoClip

  main_clip = VideoFileClip("main.mp4")
  sub_clip = VideoFileClip("sub.mp4").resize(0.3).set_position(("right", "top"))
  text = TextClip("MOVIEPY", fontsize=30, color="white").set_position(("center", "bottom")).set_duration(5)
  final = CompositeVideoClip([main_clip, sub_clip, text])
  ```

---

**三、动态生成特效**
通过代码逐帧生成动画或数据可视化。

1. **矢量图形动画**
• 效果：动态绘制SVG图形（如线条生长、形状变换）。

• 代码示例（需配合 `svgwrite` 和 `cairosvg`）：

  ```python
  import svgwrite
  from moviepy.editor import VideoClip

  def draw_frame(t):
      dwg = svgwrite.Drawing(size=(640, 480))
      dwg.add(dwg.line(start=(0, 240), end=(t*100, 240), stroke="red"))  # 随时间增长的线条
      return dwg.to_string()

  clip = VideoClip(lambda t: ImageClip(draw_frame(t)).get_frame(t), duration=5)
  ```

2. **粒子系统**
• 效果：模拟雨雪、火焰、星空等粒子效果。

• 代码示例（基础版）：

  ```python
  import numpy as np
  from moviepy.editor import VideoClip

  def particle_frame(t):
      frame = np.zeros((480, 640, 3), dtype=np.uint8)
      for _ in range(100):
          x, y = np.random.randint(0, 640), np.random.randint(0, 480)
          cv2.circle(frame, (x, y), 2, (255, 255, 255), -1)  # 随机白点
      return frame

  clip = VideoClip(particle_frame, duration=5)
  ```

3. **数据驱动动画**
• 效果：根据外部数据生成动态图表（如股票走势、实时波形）。

• 代码示例：

  ```python
  import numpy as np
  from moviepy.editor import VideoClip

  def chart_frame(t):
      frame = np.zeros((480, 640, 3), dtype=np.uint8)
      data = np.sin(t * np.pi) * 100  # 正弦波数据
      cv2.line(frame, (0, 240), (640, 240), (255, 255, 255), 1)  # 横轴
      cv2.line(frame, (int(t*64), 240 - int(data)), (int(t*64) + 1, 240 - int(data)), (0, 255, 0), 5)  # 数据点
      return frame

  clip = VideoClip(chart_frame, duration=10)
  ```

---

**四、高级合成特效**
需依赖第三方库或工具链。

1. **绿幕抠像（Chroma Key）**
• 效果：替换视频中的纯色背景。

• 代码示例（结合 OpenCV）：

  ```python
  import cv2
  from moviepy.editor import VideoFileClip

  def chroma_key(frame):
      hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
      mask = cv2.inRange(hsv, (35, 50, 50), (85, 255, 255))  # 绿色范围
      return cv2.bitwise_and(frame, frame, mask=~mask)  # 去绿幕

  clip = VideoFileClip("green_screen.mp4").fl_image(chroma_key)
  ```

2. **3D动画合成**
• 效果：将 Blender/Manim 生成的3D动画序列导入 MoviePy。

• 工作流：

  1. 用 Manim 渲染3D场景：`manim render -o 3d_anim.mp4 my_scene.py`  
  2. 用 MoviePy 合成到视频中：
     ```python
     from moviepy.editor import VideoFileClip, CompositeVideoClip

     3d_clip = VideoFileClip("3d_anim.mp4").resize(0.5).set_position(("left", "top"))
     final = CompositeVideoClip([background_clip, 3d_clip])
     ```

3. **音频可视化**
• 效果：将音频波形或频谱转换为动态图形。

• 代码示例：

  ```python
  from moviepy.editor import AudioFileClip, VideoClip
  import numpy as np

  audio = AudioFileClip("music.mp3")
  def waveform_frame(t):
      samples = audio.get_frame(t)[:, 0]  # 左声道数据
      frame = np.zeros((480, 640, 3))
      for i in range(640):
          height = int(abs(samples[i]) * 100)
          cv2.line(frame, (i, 240 - height), (i, 240 + height), (0, 255, 0), 1)
      return frame

  clip = VideoClip(waveform_frame, duration=audio.duration)
  ```

---

**五、特效扩展能力**
| 特效类型       | 原生支持 | 需结合 OpenCV | 需第三方库       |
|--------------------|--------------|-------------------|---------------------|
| 基础滤镜           | ✔️           | -                 | -                   |
| 动态文本           | ✔️           | -                 | -                   |
| 粒子系统           | ❌           | ✔️                | Taichi（GPU加速）   |
| 绿幕抠像           | ❌           | ✔️                | -                   |
| 3D动画             | ❌           | ❌                | Manim/Blender       |
| 音频可视化         | ❌           | ✔️                | -                   |

---

**总结**
• 核心优势：MoviePy 擅长视频剪辑、基础特效和动态合成，适合快速生成教育、宣传或社交媒体内容。  

• 进阶路线：结合 OpenCV、Taichi 或 Manim 可实现复杂特效（如粒子、3D动画）。  

• 局限：缺乏 React 的组件化开发体验，复杂特效需手动逐帧处理。
