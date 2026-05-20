# Models

本项目的模型文件统一放在 `models/` 目录下。该目录已加入 `.gitignore`，不会提交到 Git。

## LaMa ONNX

当前推荐使用 Hugging Face 上的 ONNX 版本：

```text
Carve/LaMa-ONNX
```

下载后的目标路径：

```text
models/Carve/LaMa-ONNX/lama.onnx
```

### Windows PowerShell 初始化命令

在项目根目录执行：

```powershell
.\.venv\Scripts\activate
$env:HF_ENDPOINT = "https://hf-mirror.com"
python -c "from huggingface_hub import hf_hub_download; print(hf_hub_download(repo_id='Carve/LaMa-ONNX', filename='lama.onnx', local_dir='models/Carve/LaMa-ONNX'))"
```

### 验证模型文件

```powershell
Get-ChildItem .\models\Carve\LaMa-ONNX
```

应能看到：

```text
lama.onnx
```

当前模型大小约 `207 MB`。

### 验证 ONNX Runtime / DirectML

```powershell
.\.venv\Scripts\python.exe -c "import onnxruntime as ort; print(ort.__version__); print(ort.get_available_providers())"
```

Windows + AMD GPU 环境下，应至少能看到：

```text
DmlExecutionProvider
CPUExecutionProvider
```

当前 `Carve/LaMa-ONNX` 在本机 DirectML 上可能触发 ONNX Runtime provider 错误。程序会优先尝试 `DmlExecutionProvider`，失败后自动切换到 `CPUExecutionProvider`，不会中断整批处理。

### 运行 LaMa ONNX 后端

```powershell
.\.venv\Scripts\python.exe main.py `
  --input .\.tmp `
  --output .\output\lama `
  --recursive `
  --save-mask `
  --save-debug `
  --inpaint-backend lama-onnx `
  --lama-model .\models\Carve\LaMa-ONNX\lama.onnx
```

成功后会在输出日志里看到类似：

```text
route=lama-onnx:CPUExecutionProvider:512x512
```

如果 DirectML provider 可正常执行，也可能显示 `DmlExecutionProvider`。

### 模型输入限制

该 ONNX 模型输入固定为 `512x512`：

```text
l_image_: [1, 3, 512, 512]
l_mask_: [1, 1, 512, 512]
```

程序会自动把原图和 mask 缩放到模型尺寸，推理完成后再把修复结果按原始 mask 合成回原图。

## 已下载过的 ModelScope 版本

曾下载过 ModelScope 版 LaMa：

```text
models/damo/cv_fft_inpainting_lama
```

该版本主权重是 `pytorch_model.pt`，不是 ONNX 格式，不能直接用于当前 `lama-onnx` 后端。

## Florence-2 Base

Florence-2 用作可选视觉文字检测层，补充 PaddleOCR 对贴边、竖排、低对比海报文字的漏检。

当前推荐模型：

```text
microsoft/Florence-2-base
```

### Windows PowerShell 初始化命令

在项目根目录执行：

```powershell
.\.venv\Scripts\activate
python -m pip install torch transformers pillow
$env:HF_ENDPOINT = "https://hf-mirror.com"
python -c "from huggingface_hub import snapshot_download; print(snapshot_download(repo_id='microsoft/Florence-2-base', local_dir='models/microsoft/Florence-2-base'))"
```

### 运行 Florence-2 视觉检测

```powershell
.\.venv\Scripts\python.exe main.py `
  --input .\.tmp `
  --output .\output `
  --auto-run-dir `
  --recursive `
  --save-mask `
  --save-debug `
  --anonymous-report `
  --vision-text `
  --vision-model .\models\microsoft\Florence-2-base `
  --vision-trigger empty `
  --vision-max-side 768 `
  --edge-text `
  --inpaint-backend opencv `
  --device cpu
```

报告中会出现：

```text
Vision Text
```

该字段表示 Florence-2 额外提供的文字候选数量。它只是检测候选，不默认替代 PaddleOCR。

## ComfyUI API Inpaint

ComfyUI 当前作为高级修复后端使用，不负责识别 mask。项目侧继续通过 PaddleOCR、边缘 OCR、Florence-2 和水印补偿生成 mask，再把原图和 mask 交给 ComfyUI 做局部重绘。

`.env` 可配置：

```text
COMFYUI_URL=https://comfyui.wodcloud.com/shucheng
COMFYUI_WORKFLOW=workflows/sd1.5_inpaint_api.json
COMFYUI_TIMEOUT=900
COMFYUI_POLL_INTERVAL=1.5
```

### 验证 ComfyUI 服务

```powershell
Invoke-WebRequest "$env:COMFYUI_URL/system_stats"
```

如果没有把 `COMFYUI_URL` 写入当前 PowerShell 环境，也可以直接使用 `.env` 里的地址：

```powershell
Invoke-WebRequest "https://comfyui.wodcloud.com/shucheng/system_stats"
```

### 单图探针命令

建议先用少量样例验证链路，不直接全量提交到远端队列：

```powershell
.\.venv\Scripts\python.exe main.py `
  --preset watermark-first `
  --input .\.tmp\comfy_probe `
  --output .\output `
  --auto-run-dir `
  --recursive `
  --save-mask `
  --save-debug `
  --anonymous-report `
  --inpaint-backend comfyui-api `
  --protected-action repair `
  --device cpu
```

2026-05-19 验证记录：

```text
output/20260519_1636.md
status: cleaned_protected
residual_text_count: 0
route: comfyui-api + post dark residual cleanup
```

该轮证明 `Python mask -> ComfyUI inpaint -> 项目侧复检/残留清理` 链路已经打通。肉眼检查 debug 图时，底部大字和右侧竖排标题能被覆盖并清理；人物区域仍有可见涂抹感，后续优化重点应放在 ComfyUI 模型、denoise、prompt 和局部重绘参数，而不是把 mask 识别搬进 ComfyUI。
