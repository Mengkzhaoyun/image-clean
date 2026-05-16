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
