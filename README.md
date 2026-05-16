# image-clean

`image-clean` 是一个面向本地批处理的图片文字清理工具。目标是扫描指定目录下的所有图片，自动判断图片里是否存在文字；有文字则按场景进行去字修复，没有文字则直接跳过。

项目优先适配 Windows 环境，并面向 AMD 显卡做本地推理加速规划。

## 运行环境

当前目标运行环境：

| 项目     | 说明                                            |
| :------- | :---------------------------------------------- |
| 操作系统 | Windows 10 / Windows 11                         |
| 显卡     | AMD GPU                                         |
| 推理加速 | DirectML                                        |
| 主要语言 | Python 3.12+                                    |
| 图片处理 | OpenCV                                          |
| 文字检测 | PaddleOCR                                       |
| 图像修复 | OpenCV Inpaint / LaMa ONNX / 其他 AIGC 修复模型 |

AMD 显卡在 Windows 下使用 DirectML 路线，避免强依赖 CUDA。需要运行 ONNX 模型时，可通过 ONNX Runtime 的 DirectML 后端调用显卡。

## 核心需求

给定一个目录后，程序需要递归扫描目录中的图片文件，并按以下流程处理：

1. 读取目录下所有支持的图片格式。
2. 对每张图片进行文字检测。
3. 如果没有检测到文字，跳过该图片。
4. 如果检测到文字，生成文字区域遮罩。
5. 判断文字是否覆盖在人身上。
6. 根据不同场景选择不同修复策略。
7. 将处理后的图片输出到指定目录，保留原目录结构或文件名映射。

支持的图片格式规划：

- `.jpg`
- `.jpeg`
- `.png`
- `.webp`
- `.bmp`

## 处理策略

### 1. 图片中没有文字

直接跳过，不做任何修复处理。

这类图片不应该被重复编码，避免画质损失。

### 2. 文字没有覆盖在人身上

如果文字位于背景、天空、墙面、海报边缘、水印区域、纯色区域等位置，可以优先使用传统图像修复模型处理。

推荐策略：

- 使用 PaddleOCR 检测文字区域。
- 根据文字框生成 mask。
- 对 mask 做适当膨胀，覆盖文字边缘、阴影和抗锯齿区域。
- 使用 LaMa / OpenCV Inpaint / ONNX 图像修复模型清理文字。

适合场景：

- 水印
- 字幕
- 角标
- 海报文字
- 背景上的小字
- 没有压住人物主体的标题

这类场景优先使用轻量修复，因为速度快、成本低、结果稳定。

### 3. 文字覆盖在人身上

如果文字覆盖在人的脸、身体、皮肤、衣服、头发等关键区域，普通修复模型容易把人物结构修坏。

这类图片应进入 AIGC 修复流程。

推荐策略：

- 检测文字区域并生成 mask。
- 使用人物检测或分割模型判断 mask 是否与人体区域重叠。
- 如果文字覆盖人体区域，则使用 Stable Diffusion Inpaint、BrushNet、ControlNet Inpaint 等生成式修复方案。
- 修复时尽量保留人物轮廓、衣服纹理、皮肤结构和背景一致性。

适合场景：

- 文字压在脸上
- 文字压在身体上
- 文字压在衣服纹理上
- 大标题穿过人物主体
- 字幕遮挡人物关键部位

这类场景不追求最快速度，优先保证修复自然。

## 目录扫描流程

批处理不再设计成单条固定链路，而是多阶段、可迭代的流水线。每个阶段都可以增加新的模型或规则，最终通过统一的 mask、人体保护和质量复检来决定是否继续处理。

```text
输入目录
  |
  v
扫描所有图片
  |
  v
候选检测阶段
  |
  +-- PaddleOCR 检测文字
  +-- 角落 logo / 水印启发式检测
  +-- 贴纸 / 印章 / 海报标题检测（实验）
  +-- 后续可接入其他检测模型
  |
  v
候选融合阶段
  |
  +-- 合并检测框
  +-- 过滤低置信度结果
  +-- 生成初始 mask
  +-- 膨胀 / 闭运算 / 边缘补偿
  |
  v
保护与路由阶段
  |
  +-- 人体 / 人脸 / 皮肤保护检测
  +-- mask 未覆盖人体 -> 轻量修复（OpenCV / LaMa）
  +-- mask 覆盖人体 -> AIGC 修复或人工复核
  +-- 高风险误伤 -> 跳过并输出 debug
  |
  v
质量复检阶段
  |
  +-- 复跑 OCR / 水印检测
  +-- 检查残留文字、残留贴纸、异常涂抹
  +-- 通过 -> 输出结果
  +-- 未通过 -> 迭代补 mask / 换后端 / 标记待处理
```

当前实现只完成了其中一部分：PaddleOCR、角落水印补偿、mask 生成、OpenCV / LaMa 修复、debug 输出。后续会围绕新增样例继续迭代检测模型、人体保护和复检逻辑。

## 安装依赖

建议先创建虚拟环境：

```bash
python -m venv .venv
.\.venv\Scripts\activate
python -m pip install -U pip
python -m pip install -r requirements.txt
```

当前本机默认 Python 版本为 `Python 3.12.10`。

## 使用命令

如果没有手动激活虚拟环境，可以直接使用：

```bash
.\.venv\Scripts\python.exe main.py --input ./images --output ./output
```

先检查目录扫描是否正常：

```bash
python main.py --input ./images --output ./output --recursive --dry-run
```

执行自动处理：

```bash
python main.py --input ./images --output ./output
```

保存调试对比图：

```bash
python main.py --input ./images --output ./output --save-mask --save-debug
```

使用 LaMa ONNX 后端：

```bash
python main.py ^
  --input ./images ^
  --output ./output ^
  --inpaint-backend lama-onnx ^
  --lama-model ./models/Carve/LaMa-ONNX/lama.onnx
```

当前已下载 Hugging Face ONNX 版 LaMa 模型：

```text
models/Carve/LaMa-ONNX/lama.onnx
```

该模型来自 `Carve/LaMa-ONNX`，输入尺寸固定为 `512x512`。代码会在 LaMa 后端内部缩放推理，再按原始 mask 合成回原图，避免整张输出图被重采样覆盖。

常用参数：

```bash
python main.py \
  --input ./images \
  --output ./output \
  --recursive \
  --device directml \
  --dilate 8 \
  --mode auto
```

参数说明：

| 参数                       | 说明                                  |
| :------------------------- | :------------------------------------ |
| `--input`                  | 输入图片目录                          |
| `--output`                 | 输出目录                              |
| `--recursive`              | 是否递归扫描子目录                    |
| `--device`                 | 推理设备，例如 `cpu`、`directml`      |
| `--dilate`                 | mask 膨胀像素，用于覆盖文字边缘       |
| `--mode`                   | 处理模式，默认 `auto`                 |
| `--inpaint-backend`        | 修复后端，`opencv` 或 `lama-onnx`     |
| `--inpaint-radius`         | OpenCV 修复半径，默认 `2.0`           |
| `--inpaint-method`         | OpenCV 修复算法，`telea` 或 `ns`      |
| `--lama-model`             | LaMa ONNX 模型路径                    |
| `--lang`                   | PaddleOCR 语言，默认 `ch`             |
| `--ocr-version`            | PaddleOCR 模型版本，默认 `PP-OCRv4`   |
| `--ocr-upscale-small`      | 小图 OCR 前放大的短边阈值，默认 `640` |
| `--ocr-min-score`          | OCR 最低置信度，默认 `0.55`           |
| `--no-watermark-corners`   | 关闭角落水印启发式检测                |
| `--sticker-watermarks`     | 启用实验性全图贴纸检测，默认关闭      |
| `--watermark-corner-ratio` | 角落扫描范围比例，默认 `0.18`         |
| `--dry-run`                | 只扫描图片，不加载 PaddleOCR          |
| `--save-mask`              | 保存文字区域 mask                     |
| `--save-debug`             | 保存原图 / mask / 结果对比图          |
| `--log`                    | 输出 JSONL 处理日志                   |

当前版本已实现：

- 扫描目录并筛选图片。
- 使用 PaddleOCR 检测文字。
- 对靠近角落文字的彩色 logo / 水印做启发式补充检测。
- 无文字图片跳过。
- 根据文字框生成 mask。
- 对非人体覆盖场景使用 OpenCV inpaint 做基础修复。
- 接入 LaMa ONNX 修复后端，可通过 `--inpaint-backend lama-onnx` 切换。
- 可保存调试对比图，方便检查 mask 和修复效果。
- 可保存 mask 和处理日志。

人体覆盖判断与 AIGC 修复接口已预留，后续接入人体分割和生成式 inpaint。

### Windows 注意事项

当前代码默认使用 `PP-OCRv4`，并关闭 PaddleOCR 的 MKL-DNN 加速，避免 Windows CPU 环境下部分 PaddlePaddle 3.x 推理路径报错。`--device directml` 会优先让 LaMa ONNX 使用 ONNX Runtime DirectML；如果当前模型在 DirectML 上触发 provider 运行错误，会自动回退到 `CPUExecutionProvider` 完成处理。PaddleOCR 本身仍按 PaddlePaddle 支持的后端运行。

### 当前限制

PaddleOCR 主要负责识别文字，不能保证识别纯图形 logo。当前版本额外加入了角落水印启发式检测，用来覆盖类似左上角 logo + 文字的水印组合，但它不是通用 logo 检测模型。

小图、贴边竖排文字容易漏检，当前默认会将短边小于 `640` 的图片临时放大后再跑 OCR，并使用 `--ocr-min-score 0.55` 过滤低置信度碎片。可以根据样例调整这两个参数。

`--sticker-watermarks` 是实验性功能，可能把靠近文字的人体或肤色区域误识别为贴纸。默认关闭，批量处理真人图片时建议先只看 `--save-debug` 输出再决定是否启用。

OpenCV inpaint 是第一版快速修复方案，适合小字、水印、纯背景区域。遇到大面积文字、复杂纹理、皮肤、衣服、头发等区域时，出现涂抹感是预期限制。LaMa ONNX 通常比 OpenCV 更适合复杂背景，但当前 `Carve/LaMa-ONNX` 模型固定 `512x512` 输入，大图细节仍会受缩放影响。人体遮挡场景后续仍需要 AIGC inpaint。

## 开发计划

- [x] 扫描目录并筛选图片文件。
- [x] 接入 PaddleOCR，检测图片文字。
- [x] 根据 PaddleOCR 检测结果生成文字 mask。
- [x] 对无文字图片执行跳过逻辑。
- [x] 接入 OpenCV inpaint 基础修复，处理非人体区域文字。
- [x] 增加角落水印启发式检测，补充 PaddleOCR 漏掉的 logo 区域。
- [ ] 接入人体检测或人体分割模型。
- [ ] 判断文字 mask 是否覆盖人体区域。
- [x] 接入 LaMa ONNX 修复后端入口，降低 OpenCV inpaint 的涂抹感。
- [x] 增加 debug 对比图输出，方便检查 mask 和修复结果。
- [x] 下载 ModelScope 版 LaMa 模型到 `models/damo/cv_fft_inpainting_lama`。
- [ ] 接入 ModelScope LaMa 后端，直接使用已下载的 PyTorch 模型。
- [x] 准备并验证 LaMa ONNX 模型文件。
- [ ] 接入 AIGC inpaint 流程，处理覆盖人体的文字。
- [x] 输出处理日志，记录跳过、成功、失败的图片。
- [x] 适配 Windows + AMD GPU + DirectML 推理入口，并增加 CPU 自动回退。

## 项目目标

最终希望实现一个本地化、可批量运行的图片清理工具：

- 自动扫描目录。
- 自动判断图片是否有文字。
- 自动跳过无文字图片。
- 自动区分普通背景文字和人体遮挡文字。
- 普通场景快速修复。
- 人体遮挡场景使用 AIGC 精修。
- 尽量利用 Windows 下的 AMD GPU 算力。
